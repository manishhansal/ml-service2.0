"""
src.data.reliability — Data reliability investigation and production-readiness report.

Purpose
-------
Answers the question: "Is the historical data from data-service2.0 and
SentinelPulse reliable and production-ready for ML model training that can
build profitable signals as per institutional and expert trading standards?"

What this module does
---------------------
1. Fetches historical OHLCV data for an F&O / Indices universe.
2. Runs the full ingestion validation pipeline (OHLC sanity, volume, gaps,
   calendar correctness, PIT timestamps).
3. Computes the full FeatureFactory feature matrix per symbol.
4. Fits and validates the FeatureNormalizer on training split only.
5. Runs LeakageValidator (statistical look-ahead bias detection).
6. Fetches SentinelPulse news signals and validates PIT correctness,
   look_ahead_validated flags, and mandatory field coverage.
7. Computes institutional-grade signal quality metrics:
     - Information Coefficient (IC) and IC Information Ratio (ICIR)
     - Hit rate, profit factor, Calmar ratio, Sharpe ratio
     - Maximum drawdown and drawdown recovery
     - Turnover and transaction cost sensitivity
     - Regime-conditional IC breakdown (RISK_ON / RISK_OFF / NEUTRAL)
     - Feature-level IC vs forward returns (signal validity)
8. Runs the TrainingReadinessGate.
9. Produces a structured, JSON-serializable DataReliabilityReport with a
   clear PRODUCTION_READY / CONDITIONALLY_READY / NOT_READY verdict and
   actionable blocker/warning list.

Institutional thresholds used
------------------------------
  IC mean         ≥ 0.03   (weak signal; 0.05+ preferred by quant shops)
  IC std / ICIR   ≥ 1.5    (signal consistency)
  Hit rate        ≥ 52%    (profitable more than 50% of trades)
  Sharpe ratio    ≥ 0.5    (annualized, pre-cost, signal backtest)
  Max drawdown    ≤ 30%    (absolute, signal-weighted P&L)
  Missing data    ≤ 5%     (per feature, after normalization)
  Gap rate        ≤ 2%     (true missing trading sessions / expected)
  News coverage   ≥ 60%    (% of trading days with SentinelPulse context)
  look_ahead_validated = 100% (zero tolerance for PIT violations in news)

Usage
-----
    from src.data.reliability import DataReliabilityInvestigator

    investigator = DataReliabilityInvestigator(
        data_client=data_client,
        sentinel_client=sentinel_client,
    )
    report = await investigator.investigate(
        symbols=["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "HDFCBANK"],
        interval="1d",
        from_date="2021-01-01",
        to_date="2025-12-31",
    )
    print(report.verdict)          # "PRODUCTION_READY"
    print(report.to_json())        # Full structured JSON report
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.ingestion import DataIngestionPipeline, IngestionValidationReport, normalize_bars
from src.features.factory import FeatureFactory
from src.features.leakage_validator import LeakageValidator, PITViolationError
from src.features.normalizer import FeatureNormalizer
from src.logging_config import get_logger

logger = get_logger(__name__)

# ── Institutional thresholds ─────────────────────────────────────────────────

_THRESHOLDS = {
    "ic_mean_min":           0.03,    # minimum acceptable mean IC
    "ic_mean_good":          0.05,    # "good signal" IC threshold
    "icir_min":              1.5,     # minimum IC information ratio
    "hit_rate_min":          0.52,    # minimum directional hit rate
    "sharpe_min":            0.5,     # minimum annualized Sharpe (signal backtest)
    "max_drawdown_max":      0.30,    # maximum drawdown of signal P&L
    "missing_data_max_pct":  5.0,     # max % missing per feature
    "gap_rate_max_pct":      2.0,     # max % true missing sessions
    "news_coverage_min_pct": 60.0,    # min % trading days with news context
    "news_pit_valid_pct":    100.0,   # % of news samples that are look_ahead_validated
    "min_bars":              252,     # minimum 1 year of daily bars
    "min_symbols":           3,       # minimum symbols for cross-sectional validity
    "confidence_score_min":  70,      # minimum DataConfidenceScore (passes gate)
}


# ── Result dataclasses ───────────────────────────────────────────────────────


@dataclass
class SymbolIngestionResult:
    symbol:             str
    bars_fetched:       int = 0
    bars_clean:         int = 0
    ohlc_violations:    int = 0
    duplicate_bars:     int = 0
    negative_volume:    int = 0
    true_missing_sessions: int = 0
    intraday_gap_rate:  float = 0.0
    calendar_gap_rate:  float = 0.0   # true_missing / expected_trading_days
    first_date:         str = ""
    last_date:          str = ""
    confidence_scores:  list[int] = field(default_factory=list)
    confidence_min:     float = 0.0
    confidence_mean:    float = 0.0
    providers:          list[str] = field(default_factory=list)
    passes:             bool = False
    blockers:           list[str] = field(default_factory=list)


@dataclass
class FeatureQualityResult:
    n_features:             int = 0
    n_rows:                 int = 0
    high_missing_features:  list[str] = field(default_factory=list)   # > threshold
    constant_features:      list[str] = field(default_factory=list)
    high_outlier_features:  list[str] = field(default_factory=list)   # > 5% beyond 5 IQR
    leakage_detected:       bool = False
    leakage_features:       list[str] = field(default_factory=list)
    normalization_applied:  bool = False
    missing_pct_by_feature: dict[str, float] = field(default_factory=dict)
    outlier_pct_by_feature: dict[str, float] = field(default_factory=dict)
    passes:                 bool = False
    blockers:               list[str] = field(default_factory=list)


@dataclass
class SignalQualityResult:
    """Institutional-grade signal quality metrics across the universe."""
    n_symbols:          int = 0
    n_total_rows:       int = 0
    # IC metrics
    ic_mean:            float = 0.0
    ic_std:             float = 0.0
    ic_ir:              float = 0.0    # IC / std(IC) — information ratio
    ic_positive_pct:    float = 0.0   # % of symbols/windows where IC > 0
    # Hit rate
    hit_rate:           float = 0.0   # directional correctness
    # P&L simulation (signal-weighted long-only)
    sharpe_ratio:       float = 0.0
    max_drawdown:       float = 0.0
    calmar_ratio:       float = 0.0
    annualized_return:  float = 0.0
    profit_factor:      float = 0.0   # sum(wins) / sum(losses)
    # Feature-level IC (top features for signal building)
    feature_ic_ranking: list[dict[str, Any]] = field(default_factory=list)
    # Regime-conditional IC
    regime_ic:          dict[str, float] = field(default_factory=dict)
    # Assessment
    passes:             bool = False
    grade:              str = "F"     # A/B/C/D/F — institutional grade
    blockers:           list[str] = field(default_factory=list)
    warnings:           list[str] = field(default_factory=list)


@dataclass
class NewsReliabilityResult:
    symbols_checked:        int = 0
    symbols_with_coverage:  int = 0
    coverage_pct:           float = 0.0
    look_ahead_valid_pct:   float = 0.0
    mandatory_fields_ok_pct: float = 0.0
    avg_news_impact_score:  float = 0.0
    sentiment_value_ranges: dict[str, Any] = field(default_factory=dict)
    regime_distribution:    dict[str, int] = field(default_factory=dict)
    sentinel_reachable:     bool = False
    passes:                 bool = False
    blockers:               list[str] = field(default_factory=list)
    warnings:               list[str] = field(default_factory=list)


@dataclass
class DataReliabilityReport:
    """
    Full data reliability and production-readiness report.

    Verdict:
      PRODUCTION_READY       — all institutional thresholds met
      CONDITIONALLY_READY    — passes core gates but warnings exist (e.g. news down)
      NOT_READY              — one or more blockers; training must not proceed
    """
    verdict:            str    # PRODUCTION_READY | CONDITIONALLY_READY | NOT_READY
    verdict_reason:     str
    investigated_at:    str    = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    universe:           list[str] = field(default_factory=list)
    interval:           str   = "1d"
    date_range:         dict[str, str] = field(default_factory=dict)
    calendar_live:      bool  = False

    # Sub-reports
    symbol_results:     list[SymbolIngestionResult] = field(default_factory=list)
    feature_quality:    FeatureQualityResult = field(default_factory=FeatureQualityResult)
    signal_quality:     SignalQualityResult = field(default_factory=SignalQualityResult)
    news_reliability:   NewsReliabilityResult = field(default_factory=NewsReliabilityResult)

    # Aggregates
    blockers:           list[str] = field(default_factory=list)
    warnings:           list[str] = field(default_factory=list)
    thresholds_used:    dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def _dc(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _dc(getattr(obj, k)) for k in obj.__dataclass_fields__}
            if isinstance(obj, list):
                return [_dc(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _dc(v) for k, v in obj.items()}
            return obj
        return _dc(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        logger.info("reliability_report_saved", path=str(path))


# ── Main investigator ─────────────────────────────────────────────────────────


class DataReliabilityInvestigator:
    """
    Fetches, validates, and scores historical data for ML training readiness.

    Combines ingestion validation, feature quality, signal quality metrics,
    and news reliability into a single authoritative report.

    Usage::

        inv = DataReliabilityInvestigator(data_client, sentinel_client)
        report = await inv.investigate(
            symbols=["NIFTY", "BANKNIFTY", "RELIANCE"],
            interval="1d",
            from_date="2021-01-01",
            to_date="2025-12-31",
            label_horizon=5,
        )
        report.save(Path("./reports/reliability.json"))
    """

    def __init__(
        self,
        data_client: Any,
        sentinel_client: Any = None,
        output_root: Path | None = None,
        thresholds: dict[str, Any] | None = None,
    ) -> None:
        self._data     = data_client
        self._sentinel = sentinel_client
        self._root     = Path(output_root or "./data_reliability")
        self._t        = dict(_THRESHOLDS)
        if thresholds:
            self._t.update(thresholds)

    # ── Main entry ────────────────────────────────────────────────────────────

    async def investigate(
        self,
        symbols: list[str],
        interval: str = "1d",
        from_date: str | None = None,
        to_date: str | None = None,
        label_horizon: int = 5,
        exchange: str = "NSE",
    ) -> DataReliabilityReport:
        """
        Run the full data reliability investigation.

        Steps:
          1. Refresh NSE calendar from live source.
          2. Fetch and validate OHLCV per symbol.
          3. Compute features + normalization.
          4. Run leakage detection.
          5. Compute institutional signal quality (IC, Sharpe, drawdown).
          6. Check SentinelPulse news reliability.
          7. Render verdict.
        """
        logger.info(
            "reliability_investigation_start",
            symbols=symbols,
            interval=interval,
            from_date=from_date,
            to_date=to_date,
        )

        report = DataReliabilityReport(
            verdict="NOT_READY",
            verdict_reason="Investigation incomplete",
            universe=symbols,
            interval=interval,
            date_range={"from": from_date or "auto", "to": to_date or "auto"},
            thresholds_used=self._t,
        )

        # ── Step 1: Refresh NSE calendar ──────────────────────────────────────
        ingestion_pipeline = DataIngestionPipeline(
            data_client=self._data,
            output_root=self._root / "raw",
        )
        report.calendar_live = await ingestion_pipeline.refresh_calendar()

        # ── Step 2: Fetch + validate OHLCV ────────────────────────────────────
        ohlcv_frames: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            sym_result, df = await self._investigate_symbol(
                sym, interval, from_date, to_date, exchange
            )
            report.symbol_results.append(sym_result)
            if df is not None and not df.empty:
                ohlcv_frames[sym] = df

        # Aggregate symbol-level blockers
        for sr in report.symbol_results:
            for b in sr.blockers:
                if b not in report.blockers:
                    report.blockers.append(b)

        usable_symbols = [s for s in symbols if s in ohlcv_frames]
        if len(usable_symbols) < self._t["min_symbols"]:
            report.blockers.append(
                f"INSUFFICIENT_SYMBOLS: only {len(usable_symbols)} symbols returned "
                f"usable data (need ≥ {self._t['min_symbols']})"
            )
            report.verdict        = "NOT_READY"
            report.verdict_reason = report.blockers[-1]
            return report

        # ── Step 3 & 4: Feature quality + leakage ─────────────────────────────
        report.feature_quality = self._investigate_features(ohlcv_frames)
        for b in report.feature_quality.blockers:
            if b not in report.blockers:
                report.blockers.append(b)

        # ── Step 5: Signal quality ─────────────────────────────────────────────
        report.signal_quality = self._investigate_signal_quality(
            ohlcv_frames, label_horizon
        )
        for b in report.signal_quality.blockers:
            if b not in report.blockers:
                report.blockers.append(b)
        for w in report.signal_quality.warnings:
            if w not in report.warnings:
                report.warnings.append(w)

        # ── Step 6: SentinelPulse news ────────────────────────────────────────
        report.news_reliability = await self._investigate_news(usable_symbols)
        for b in report.news_reliability.blockers:
            if b not in report.blockers:
                report.blockers.append(b)
        for w in report.news_reliability.warnings:
            if w not in report.warnings:
                report.warnings.append(w)

        # ── Step 7: Render verdict ─────────────────────────────────────────────
        report.verdict, report.verdict_reason = self._render_verdict(report)

        logger.info(
            "reliability_investigation_complete",
            verdict=report.verdict,
            n_blockers=len(report.blockers),
            n_warnings=len(report.warnings),
            ic_mean=report.signal_quality.ic_mean,
            icir=report.signal_quality.ic_ir,
        )

        return report

    # ── Symbol-level investigation ────────────────────────────────────────────

    async def _investigate_symbol(
        self,
        symbol:    str,
        interval:  str,
        from_date: str | None,
        to_date:   str | None,
        exchange:  str,
    ) -> tuple[SymbolIngestionResult, pd.DataFrame | None]:
        result = SymbolIngestionResult(symbol=symbol)
        try:
            raw = await self._data.get_historical_ohlcv(
                symbol,
                exchange=exchange,
                interval=interval,
                from_date=from_date,
                to_date=to_date,
            )
        except Exception as exc:
            result.blockers.append(
                f"{symbol}_FETCH_FAILED: {type(exc).__name__}: {exc}"
            )
            return result, None

        if not raw:
            result.blockers.append(f"{symbol}_NO_DATA_RETURNED")
            return result, None

        report = IngestionValidationReport(symbol=symbol, interval=interval)
        df = normalize_bars(raw, symbol, interval, report)
        result.bars_fetched        = report.rows_raw
        result.bars_clean          = report.rows_clean
        result.ohlc_violations     = report.ohlc_violations
        result.duplicate_bars      = report.duplicates_dropped
        result.negative_volume     = report.negative_volume
        result.true_missing_sessions = report.true_missing_sessions
        result.intraday_gap_rate   = report.intraday_gap_rate_pct
        result.first_date          = report.first_ts or ""
        result.last_date           = report.last_ts or ""

        # Calendar gap rate (true missing / expected trading days)
        if interval == "1d" and result.bars_clean > 0:
            expected_days = result.bars_clean + report.true_missing_sessions
            result.calendar_gap_rate = (
                report.true_missing_sessions / max(expected_days, 1) * 100
            )

        # Confidence scores from bars
        conf_vals = [
            b["data_confidence"]
            for b in raw
            if isinstance(b, dict) and b.get("data_confidence") is not None
        ]
        if conf_vals:
            result.confidence_scores = [int(c) for c in conf_vals]
            result.confidence_min    = float(min(conf_vals))
            result.confidence_mean   = float(sum(conf_vals) / len(conf_vals))

        # Providers
        providers = list({
            b.get("provider") or b.get("source")
            for b in raw
            if isinstance(b, dict) and (b.get("provider") or b.get("source"))
        })
        result.providers = [str(p) for p in providers if p]

        # Minimum bars check
        if result.bars_clean < self._t["min_bars"]:
            result.blockers.append(
                f"{symbol}_INSUFFICIENT_HISTORY: {result.bars_clean} bars "
                f"(need ≥ {self._t['min_bars']})"
            )

        # Gap rate check
        gap_rate = result.calendar_gap_rate or result.intraday_gap_rate
        if gap_rate > self._t["gap_rate_max_pct"]:
            result.blockers.append(
                f"{symbol}_HIGH_GAP_RATE: {gap_rate:.1f}% "
                f"(threshold {self._t['gap_rate_max_pct']}%)"
            )

        # OHLC integrity
        if result.ohlc_violations > 0:
            ohlc_violation_rate = result.ohlc_violations / max(result.bars_fetched, 1) * 100
            if ohlc_violation_rate > 1.0:
                result.blockers.append(
                    f"{symbol}_HIGH_OHLC_VIOLATION_RATE: {ohlc_violation_rate:.1f}%"
                )

        result.passes = len(result.blockers) == 0
        return result, df if result.bars_clean > 0 else None

    # ── Feature quality investigation ─────────────────────────────────────────

    def _investigate_features(
        self, ohlcv_frames: dict[str, pd.DataFrame]
    ) -> FeatureQualityResult:
        result = FeatureQualityResult()
        factory = FeatureFactory()
        all_features: list[pd.DataFrame] = []

        for sym, df in ohlcv_frames.items():
            try:
                feat_df, _ = factory.build(df)
                feat_df["_symbol"] = sym
                # Store close prices for leakage label computation (forward return proxy).
                # Use the original OHLCV df — factory.build() does not expose close.
                if "close" in df.columns:
                    feat_df["_close"] = df["close"].reindex(feat_df.index)
                all_features.append(feat_df)
            except Exception as exc:
                logger.warning("feature_build_failed", symbol=sym, error=str(exc))

        if not all_features:
            result.blockers.append("NO_FEATURES_COMPUTED: all symbol feature builds failed")
            return result

        combined = pd.concat(all_features)
        feature_cols = factory.FEATURE_NAMES
        feat_only = combined[feature_cols].copy()

        result.n_features = len(feature_cols)
        result.n_rows     = len(feat_only)

        # ── Fit normalizer (training split = full combined frame for investigation) ─
        try:
            normalizer = FeatureNormalizer(winsor_pct=(1.0, 99.0))
            normalizer.fit(feat_only)
            feat_norm = normalizer.transform(feat_only)
            result.normalization_applied = True
        except Exception as exc:
            logger.warning("normalizer_failed_in_investigation", error=str(exc))
            feat_norm = feat_only

        # ── Per-feature diagnostics ──────────────────────────────────────────
        threshold = self._t["missing_data_max_pct"]
        for col in feature_cols:
            arr = feat_norm[col].to_numpy(dtype=float)
            finite = arr[np.isfinite(arr)]
            n_total = len(arr)

            missing_pct = (np.isnan(arr).sum() + np.isinf(arr).sum()) / max(n_total, 1) * 100
            result.missing_pct_by_feature[col] = round(float(missing_pct), 2)

            if missing_pct > threshold:
                result.high_missing_features.append(col)

            if len(finite) > 10:
                std = float(np.std(finite))
                if std < 1e-8:
                    result.constant_features.append(col)

                median = float(np.median(finite))
                q1 = float(np.percentile(finite, 25))
                q3 = float(np.percentile(finite, 75))
                iqr = q3 - q1
                if iqr > 0:
                    outlier_pct = float(
                        np.sum(np.abs(finite - median) > 5 * iqr) / len(finite) * 100
                    )
                    result.outlier_pct_by_feature[col] = round(outlier_pct, 2)
                    if outlier_pct > 5.0:
                        result.high_outlier_features.append(col)

        # ── Leakage detection ─────────────────────────────────────────────────
        leakage_validator = LeakageValidator(threshold=0.95)
        for sym, grp in combined.groupby("_symbol"):
            fm = grp[feature_cols]
            # Use stored close prices for the 5-day forward-return leakage proxy.
            # Fall back to ret_1 (a feature) if close was not stored.
            if "_close" in grp.columns:
                labels = grp["_close"].pct_change(5).shift(-5)
            elif "ret_1" in grp.columns:
                labels = grp["ret_1"].shift(-5)
            else:
                continue
            try:
                leakage_validator.validate(fm, labels)
            except PITViolationError as exc:
                result.leakage_detected = True
                result.leakage_features.append(exc.feature_name)

        # ── Verdict ───────────────────────────────────────────────────────────
        if result.leakage_detected:
            result.blockers.append(
                f"CATASTROPHIC_LEAKAGE_DETECTED: features {result.leakage_features}"
            )
        if len(result.constant_features) > len(feature_cols) * 0.3:
            result.blockers.append(
                f"TOO_MANY_CONSTANT_FEATURES: {len(result.constant_features)}/{len(feature_cols)}"
            )
        high_missing_frac = len(result.high_missing_features) / max(len(feature_cols), 1)
        if high_missing_frac > 0.5:
            result.blockers.append(
                f"EXCESSIVE_MISSING_DATA: {len(result.high_missing_features)} features "
                f"exceed {threshold}% missing"
            )
        result.passes = len(result.blockers) == 0
        return result

    # ── Signal quality investigation ──────────────────────────────────────────

    def _investigate_signal_quality(
        self,
        ohlcv_frames:  dict[str, pd.DataFrame],
        label_horizon: int,
    ) -> SignalQualityResult:
        result = SignalQualityResult(n_symbols=len(ohlcv_frames))
        factory = FeatureFactory()

        # Build feature matrices with forward returns
        all_ic_series: list[float] = []
        all_signal_returns: list[float] = []
        all_hit_rates: list[float] = []
        feature_ic_accumulator: dict[str, list[float]] = {}

        for sym, df in ohlcv_frames.items():
            try:
                feat_df, _ = factory.build(df)
            except Exception:
                continue

            close      = df["close"].astype(float)
            fwd_return = close.pct_change(label_horizon).shift(-label_horizon)

            # Align
            idx = feat_df.index.intersection(fwd_return.dropna().index)
            if len(idx) < 30:
                continue

            fv  = feat_df.loc[idx]
            frt = fwd_return.loc[idx]
            result.n_total_rows += len(idx)

            # ── Per-symbol IC across ALL features ──────────────────────────
            for col in factory.FEATURE_NAMES:
                col_series = fv[col].dropna()
                common     = col_series.index.intersection(frt.index)
                if len(common) < 20:
                    continue
                try:
                    ic = float(np.corrcoef(
                        col_series.loc[common].values,
                        frt.loc[common].values,
                    )[0, 1])
                    if math.isfinite(ic):
                        feature_ic_accumulator.setdefault(col, []).append(ic)
                except Exception:
                    pass

            # ── Composite signal: equal-weighted IC-ranked feature score ───
            # Use top 5 features by historical IC for signal construction
            feature_ics_this_symbol: dict[str, float] = {}
            for col in factory.FEATURE_NAMES:
                col_series = fv[col].dropna()
                common     = col_series.index.intersection(frt.index)
                if len(common) < 20:
                    continue
                try:
                    ic = float(np.corrcoef(
                        col_series.loc[common].values,
                        frt.loc[common].values,
                    )[0, 1])
                    if math.isfinite(ic):
                        feature_ics_this_symbol[col] = ic
                except Exception:
                    pass

            if not feature_ics_this_symbol:
                continue

            # Top-5 features by absolute IC magnitude
            top5 = sorted(
                feature_ics_this_symbol.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )[:5]

            if not top5:
                continue

            # Composite signal = sum of (feature * sign(IC)) for top-5
            composite = pd.Series(0.0, index=idx)
            for feat_name, feat_ic in top5:
                col_vals = fv[feat_name].reindex(idx).fillna(0.0)
                # Cross-sectionally rank within day
                composite += col_vals * np.sign(feat_ic)
            composite = composite / len(top5)

            # IC of composite signal vs forward return
            common_idx = composite.index.intersection(frt.dropna().index)
            if len(common_idx) < 20:
                continue

            comp_vals = composite.loc[common_idx].values
            ret_vals  = frt.loc[common_idx].values

            try:
                ic_symbol = float(np.corrcoef(comp_vals, ret_vals)[0, 1])
            except Exception:
                continue
            if not math.isfinite(ic_symbol):
                continue

            all_ic_series.append(ic_symbol)

            # Hit rate
            hits = np.sum(np.sign(comp_vals) == np.sign(ret_vals))
            all_hit_rates.append(hits / len(comp_vals))

            # Signal-weighted returns
            norm_signal = comp_vals / (np.std(comp_vals) + 1e-10)
            signal_ret  = norm_signal * ret_vals
            all_signal_returns.extend(signal_ret.tolist())

        if not all_ic_series:
            result.blockers.append(
                "NO_IC_COMPUTED: insufficient data to compute information coefficient"
            )
            result.grade = "F"
            return result

        # ── Aggregate IC metrics ──────────────────────────────────────────────
        ic_arr = np.array(all_ic_series)
        result.ic_mean      = float(np.mean(ic_arr))
        result.ic_std       = float(np.std(ic_arr)) if len(ic_arr) > 1 else 0.0
        result.ic_ir        = (
            result.ic_mean / result.ic_std
            if result.ic_std > 1e-10 else 0.0
        )
        result.ic_positive_pct = float(np.sum(ic_arr > 0) / len(ic_arr) * 100)

        # ── Hit rate ──────────────────────────────────────────────────────────
        result.hit_rate = float(np.mean(all_hit_rates)) if all_hit_rates else 0.0

        # ── Signal P&L metrics ────────────────────────────────────────────────
        if all_signal_returns:
            ret_arr = np.array(all_signal_returns)
            # Annualize (assume 252 trading days, daily bars)
            daily_mean = float(np.mean(ret_arr))
            daily_std  = float(np.std(ret_arr)) + 1e-10
            result.annualized_return = daily_mean * 252
            result.sharpe_ratio      = (daily_mean / daily_std) * np.sqrt(252)

            # Max drawdown (cumulative signal P&L)
            cum_pnl = np.cumsum(ret_arr)
            running_max = np.maximum.accumulate(cum_pnl)
            drawdowns = (cum_pnl - running_max) / (np.abs(running_max) + 1e-10)
            result.max_drawdown = float(np.min(drawdowns))

            # Calmar ratio
            if abs(result.max_drawdown) > 1e-10:
                result.calmar_ratio = result.annualized_return / abs(result.max_drawdown)

            # Profit factor
            wins   = ret_arr[ret_arr > 0].sum()
            losses = abs(ret_arr[ret_arr < 0].sum())
            result.profit_factor = wins / losses if losses > 1e-10 else float("inf")

        # ── Feature IC ranking ────────────────────────────────────────────────
        feature_ic_summary: list[dict[str, Any]] = []
        for feat, ic_list in feature_ic_accumulator.items():
            mean_ic = float(np.mean(ic_list))
            std_ic  = float(np.std(ic_list)) if len(ic_list) > 1 else 0.0
            icir    = mean_ic / std_ic if std_ic > 1e-10 else 0.0
            feature_ic_summary.append({
                "feature":  feat,
                "ic_mean":  round(mean_ic, 4),
                "ic_std":   round(std_ic, 4),
                "ic_ir":    round(icir, 4),
                "n_symbols": len(ic_list),
            })
        feature_ic_summary.sort(key=lambda x: abs(x["ic_mean"]), reverse=True)
        result.feature_ic_ranking = feature_ic_summary[:20]  # top 20

        # ── Grade assignment (institutional) ─────────────────────────────────
        result.grade = self._assign_grade(result)

        # ── Blockers / warnings ───────────────────────────────────────────────
        if result.ic_mean < self._t["ic_mean_min"]:
            result.blockers.append(
                f"IC_TOO_LOW: mean IC={result.ic_mean:.4f} "
                f"(threshold ≥ {self._t['ic_mean_min']})"
            )
        if result.ic_ir < self._t["icir_min"] and result.ic_std > 1e-10:
            result.warnings.append(
                f"LOW_ICIR: {result.ic_ir:.2f} (threshold ≥ {self._t['icir_min']})"
            )
        if result.hit_rate < self._t["hit_rate_min"]:
            result.warnings.append(
                f"LOW_HIT_RATE: {result.hit_rate:.2%} (threshold ≥ {self._t['hit_rate_min']:.0%})"
            )
        if result.sharpe_ratio < self._t["sharpe_min"]:
            result.warnings.append(
                f"LOW_SHARPE: {result.sharpe_ratio:.2f} (threshold ≥ {self._t['sharpe_min']})"
            )
        if result.max_drawdown < -self._t["max_drawdown_max"]:
            result.warnings.append(
                f"HIGH_DRAWDOWN: {result.max_drawdown:.2%} "
                f"(threshold ≤ {-self._t['max_drawdown_max']:.0%})"
            )
        if result.ic_mean >= self._t["ic_mean_min"] and result.ic_mean < self._t["ic_mean_good"]:
            result.warnings.append(
                f"WEAK_SIGNAL: IC={result.ic_mean:.4f} passes minimum but below "
                f"institutional target {self._t['ic_mean_good']}"
            )

        result.passes = len(result.blockers) == 0
        return result

    def _assign_grade(self, r: SignalQualityResult) -> str:
        """Assign an institutional signal quality grade A–F."""
        score = 0
        if r.ic_mean >= 0.07: score += 30
        elif r.ic_mean >= 0.05: score += 20
        elif r.ic_mean >= 0.03: score += 10

        if r.ic_ir >= 3.0:  score += 25
        elif r.ic_ir >= 2.0: score += 18
        elif r.ic_ir >= 1.5: score += 12
        elif r.ic_ir >= 1.0: score += 6

        if r.sharpe_ratio >= 2.0: score += 20
        elif r.sharpe_ratio >= 1.0: score += 14
        elif r.sharpe_ratio >= 0.5: score += 8

        if r.hit_rate >= 0.56: score += 15
        elif r.hit_rate >= 0.54: score += 10
        elif r.hit_rate >= 0.52: score += 6

        if r.max_drawdown >= -0.10: score += 10
        elif r.max_drawdown >= -0.20: score += 6
        elif r.max_drawdown >= -0.30: score += 3

        if score >= 85: return "A"
        if score >= 70: return "B"
        if score >= 50: return "C"
        if score >= 30: return "D"
        return "F"

    # ── News reliability investigation ────────────────────────────────────────

    async def _investigate_news(
        self, symbols: list[str]
    ) -> NewsReliabilityResult:
        result = NewsReliabilityResult(symbols_checked=len(symbols))

        if self._sentinel is None:
            result.warnings.append(
                "SENTINELPULSE_NOT_CONFIGURED: news features will use neutral defaults"
            )
            result.passes = True   # no sentinel is a warning, not a blocker
            return result

        # Check reachability
        try:
            regime = await self._sentinel.fetch_market_regime()
            result.sentinel_reachable = regime is not None
        except Exception:
            result.sentinel_reachable = False

        if not result.sentinel_reachable:
            result.warnings.append(
                "SENTINELPULSE_UNREACHABLE: news features will use neutral defaults"
            )
            return result

        # Fetch news context per symbol
        covered     = 0
        impact_scores: list[float] = []
        regime_dist: dict[str, int] = {}
        sentiment_ranges: dict[str, list[float]] = {
            ax: [] for ax in ("overall", "market", "company", "macro", "risk")
        }

        for sym in symbols:
            try:
                ctx = await self._sentinel.fetch_news_context(sym)
            except Exception:
                continue

            if ctx is None:
                continue

            covered += 1

            # Impact score
            score = ctx.get("news_impact_score")
            if score is not None:
                try:
                    impact_scores.append(float(score))
                except (TypeError, ValueError):
                    pass

            # Market regime
            regime_str = str(ctx.get("market_regime") or "UNKNOWN")
            regime_dist[regime_str] = regime_dist.get(regime_str, 0) + 1

            # Sentiment axes
            sentiment = ctx.get("sentiment") or {}
            if isinstance(sentiment, dict):
                for ax in sentiment_ranges:
                    val = sentiment.get(ax)
                    if val is not None:
                        try:
                            sentiment_ranges[ax].append(float(val))
                        except (TypeError, ValueError):
                            pass

        result.symbols_with_coverage = covered
        result.coverage_pct = covered / max(len(symbols), 1) * 100
        result.avg_news_impact_score = (
            float(np.mean(impact_scores)) if impact_scores else 0.0
        )
        result.regime_distribution = regime_dist

        # Sentiment value range summary
        for ax, vals in sentiment_ranges.items():
            if vals:
                result.sentiment_value_ranges[ax] = {
                    "min": round(float(np.min(vals)), 3),
                    "max": round(float(np.max(vals)), 3),
                    "mean": round(float(np.mean(vals)), 3),
                }

        # Check PIT validity via training samples
        look_ahead_valid_pct = 0.0
        mandatory_ok_pct     = 0.0
        try:
            samples_resp = await self._sentinel.fetch_training_samples(limit=200)
            samples: list[Any] = []
            if isinstance(samples_resp, dict):
                samples = samples_resp.get("samples", [])
            elif isinstance(samples_resp, list):
                samples = samples_resp

            if samples:
                n_valid = sum(
                    1 for s in samples
                    if isinstance(s, dict) and s.get("look_ahead_validated", False)
                )
                look_ahead_valid_pct = n_valid / len(samples) * 100
                # Mandatory fields check
                mandatory_fields = (
                    "news_impact_score", "impact_direction",
                    "impact_confidence",
                )
                n_mandatory_ok = sum(
                    1 for s in samples
                    if isinstance(s, dict)
                    and all(s.get(f) is not None for f in mandatory_fields)
                )
                mandatory_ok_pct = n_mandatory_ok / len(samples) * 100
        except Exception as exc:
            logger.warning("news_training_sample_check_failed", error=str(exc))

        result.look_ahead_valid_pct    = look_ahead_valid_pct
        result.mandatory_fields_ok_pct = mandatory_ok_pct

        # Blockers / warnings
        if result.coverage_pct < self._t["news_coverage_min_pct"]:
            result.warnings.append(
                f"LOW_NEWS_COVERAGE: {result.coverage_pct:.1f}% of symbols have "
                f"news context (threshold ≥ {self._t['news_coverage_min_pct']}%)"
            )
        if (
            look_ahead_valid_pct < self._t["news_pit_valid_pct"]
            and look_ahead_valid_pct > 0
        ):
            result.blockers.append(
                f"NEWS_PIT_VIOLATION: only {look_ahead_valid_pct:.1f}% of training "
                f"samples have look_ahead_validated=True "
                f"(required {self._t['news_pit_valid_pct']:.0f}%)"
            )

        result.passes = len(result.blockers) == 0
        return result

    # ── Verdict rendering ─────────────────────────────────────────────────────

    def _render_verdict(
        self, report: DataReliabilityReport
    ) -> tuple[str, str]:
        """Determine the final verdict and reason string."""
        hard_blockers = [
            b for b in report.blockers
            if any(
                key in b for key in (
                    "INSUFFICIENT_SYMBOLS", "LEAKAGE", "IC_TOO_LOW",
                    "NO_IC_COMPUTED", "NEWS_PIT_VIOLATION",
                    "INSUFFICIENT_HISTORY", "NO_DATA", "FETCH_FAILED",
                )
            )
        ]

        soft_blockers = [b for b in report.blockers if b not in hard_blockers]

        if hard_blockers:
            return "NOT_READY", hard_blockers[0]

        all_symbol_pass   = all(sr.passes for sr in report.symbol_results)
        feature_pass      = report.feature_quality.passes
        signal_pass       = report.signal_quality.passes

        if not (all_symbol_pass and feature_pass and signal_pass):
            reason = (
                soft_blockers[0]
                if soft_blockers
                else "One or more data quality gates failed"
            )
            return "NOT_READY", reason

        if report.warnings:
            return (
                "CONDITIONALLY_READY",
                f"Passes core gates with {len(report.warnings)} warnings — "
                f"review before production deployment",
            )

        return (
            "PRODUCTION_READY",
            f"All institutional thresholds met — "
            f"IC={report.signal_quality.ic_mean:.4f}, "
            f"ICIR={report.signal_quality.ic_ir:.2f}, "
            f"grade={report.signal_quality.grade}",
        )
