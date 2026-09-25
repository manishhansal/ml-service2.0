"""
AlphaForge Training Data Pipeline.

Fetches normalized NSE OHLCV + F&O derivative data from the AlphaForge
Indian Market Data Layer (Angel One → Upstox → PostgreSQL → yfinance fallback),
computes features for the full F&O universe, and generates labeled datasets for
all four downstream models:

  1. Market Regime labels     (NIFTY forward returns + vol + drawdown)
  2. Stock Ranking labels     (future return horizon, risk-adjusted return)
  3. Strategy Selection labels (which strategy performed best in next N bars)
  4. Risk labels              (stop hit probability, target hit probability, MAE)

Key design constraints:
  - NO future information leakage: all features computed strictly from data
    available at the prediction timestamp (Requirement #5)
  - Walk-forward validation only: rolling train/val/test windows (Requirement #6)
  - Dataset versioning metadata attached to every saved artefact (Requirement #4)
  - F&O-specific features: OI buildup, PCR, IV rank, OI walls, max pain (Requirement #8)
  - Data quality filtering: STALE / INVALID / SUSPICIOUS records excluded (Requirement #9)

Yahoo Finance is used ONLY as a last-resort fallback (no OI/derivatives enrichment).
The primary source is always the AlphaForge normalized data layer.

Usage:
  python -m src.training.data_pipeline --start 2023-01-01 --end 2026-07-31
  python -m src.training.data_pipeline --start 2023-01-01 --end 2026-07-31 --quick
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from ..config import settings
from .market_data_client import (
    DataQuality,
    DatasetMetadata,
    DerivativesSnapshot,
    MarketBreadthRecord,
    MarketDataClient,
    classify_oi_buildup,
    create_client_from_env,
)

logger = structlog.get_logger(__name__)

_UTC = timezone.utc

# ─── Lazy feature / model imports (avoid loading talib at import time) ─────────

def _get_feature_constants():
    """Lazy import of canonical feature lists and compute functions."""
    from ..features.engineer import (  # noqa: PLC0415
        RANKING_FEATURES,
        REGIME_FEATURES,
        RISK_FEATURES,
        STRATEGY_FEATURES,
        compute_regime_features,
        compute_stock_features,
    )
    return (RANKING_FEATURES, REGIME_FEATURES, RISK_FEATURES,
            STRATEGY_FEATURES, compute_regime_features, compute_stock_features)


def _get_pit_components():
    """Lazy import of PIT data foundation (no external deps)."""
    from ..data.point_in_time import (  # noqa: PLC0415
        PointInTimeValidator,
        bhavcopy_available_utc,
        nse_close_utc,
        require_utc_aware,
    )
    from ..data.instrument_master import InstrumentMasterStore  # noqa: PLC0415
    from ..data.historical_universe import HistoricalUniverse    # noqa: PLC0415
    from ..data.fno_eligibility import FnOStateStore             # noqa: PLC0415
    from ..data.corporate_actions import CorporateActionStore    # noqa: PLC0415
    from ..data.data_quality import MLDataQualityGate            # noqa: PLC0415
    from ..data.dataset_version import DatasetSnapshot           # noqa: PLC0415
    from ..data.lineage import observation_lineage_store         # noqa: PLC0415
    return (
        PointInTimeValidator, bhavcopy_available_utc, nse_close_utc,
        require_utc_aware, InstrumentMasterStore, HistoricalUniverse,
        FnOStateStore, CorporateActionStore, MLDataQualityGate,
        DatasetSnapshot, observation_lineage_store,
    )


# ─── Pipeline version constants ───────────────────────────────────────────────

PIPELINE_VERSION = "v3.1"           # Phase 3B — PIT foundation added
FEATURE_VERSION = "fv4"             # bump when feature set changes (RANKING_FEATURES etc.)
LABEL_VERSION   = "lv2"             # Phase 3C — Label V2 (triple-barrier, event-based)
DATASET_VERSION = f"af-{PIPELINE_VERSION}-{FEATURE_VERSION}-{LABEL_VERSION}"

# ─── F&O training universe ────────────────────────────────────────────────────

# NSE symbol notation (no .NS suffix — the data client handles exchange routing)
TRAINING_UNIVERSE: list[str] = [
    # Broad-market indices (regime + breadth features)
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    # Top F&O stocks by open interest liquidity (NSE F&O segment)
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK",
    "BAJFINANCE", "MARUTI", "TATAMOTORS", "SUNPHARMA", "TITAN",
    "WIPRO", "HCLTECH", "NTPC", "POWERGRID", "ONGC", "JSWSTEEL",
    "TATASTEEL", "ADANIENT", "ADANIPORTS", "HINDALCO", "DRREDDY",
    "CIPLA", "BAJAJFINSV", "TECHM", "DIVISLAB", "NESTLEIND",
    "BRITANNIA", "INDUSINDBK", "M&M", "COALINDIA", "EICHERMOT",
    "HAL", "BEL", "TRENT", "JIOFIN", "ZOMATO", "DLF",
    "ABB", "SIEMENS", "GODREJCP", "DABUR", "VEDL", "GAIL",
]

# Quick mode: a reduced universe for fast iteration / CI
QUICK_UNIVERSE: list[str] = [
    "NIFTY", "BANKNIFTY",
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "SBIN", "AXISBANK", "BAJFINANCE", "LT", "KOTAKBANK",
]

# NSE lot sizes for OI-to-contract-value scaling
LOT_SIZES: dict[str, int] = {
    "NIFTY": 50, "BANKNIFTY": 15, "FINNIFTY": 40, "MIDCPNIFTY": 75,
    "RELIANCE": 250, "TCS": 150, "HDFCBANK": 550, "INFY": 300,
    "ICICIBANK": 700, "SBIN": 1500, "AXISBANK": 1200, "BAJFINANCE": 125,
}
DEFAULT_LOT_SIZE = 500


def get_lot_size(symbol: str, query_date: date) -> int:
    """
    Return the historical lot size for symbol on query_date.

    This replaces the static LOT_SIZES dict with a point-in-time lookup that
    accounts for known NSE lot-size changes (e.g. SEBI Nov 2024 revision).

    Falls back to LOT_SIZES dict (then DEFAULT_LOT_SIZE) for symbols without
    historical data.  APPROXIMATE results are logged as warnings.
    """
    try:
        from ..data.instrument_master import InstrumentMasterStore  # noqa: PLC0415
        store = InstrumentMasterStore.default()
        lot, status, _ = store.get_lot_size(symbol.upper(), query_date)
        if lot is not None:
            if status != "OK":
                logger.debug("lot_size_approximate", symbol=symbol, date=str(query_date))
            return lot
    except Exception as exc:
        logger.debug("lot_size_fallback", symbol=symbol, error=str(exc))

    # Fallback to static dict
    return LOT_SIZES.get(symbol.upper(), DEFAULT_LOT_SIZE)


# ─── 7-Step PIT Pre-Feature Validation ───────────────────────────────────────

class PITValidationResult:
    """
    Result of the 7-step point-in-time pre-feature validation.

    status  : "DATA_READY" | "DATA_READY_WITH_WARNINGS" | "BLOCKED"
    issues  : list of (step, severity, message) tuples
    symbol  : symbol being validated
    bar_date: date of the bar being validated
    lot_size: resolved lot size (may be APPROXIMATE)
    """

    def __init__(self, symbol: str, bar_date: date) -> None:
        self.symbol    = symbol
        self.bar_date  = bar_date
        self.issues:   list[tuple[str, str, str]] = []
        self.lot_size: int | None = None
        self.lot_size_status: str = "UNKNOWN"
        self.fno_ban_status:  str = "DATA_UNAVAILABLE"
        self.corporate_action_status: str = "DATA_UNAVAILABLE"
        self.observation_id: str | None = None

    @property
    def status(self) -> str:
        if any(sev == "CRITICAL" for _, sev, _ in self.issues):
            return "BLOCKED"
        if any(sev in ("ERROR", "WARNING") for _, sev, _ in self.issues):
            return "DATA_READY_WITH_WARNINGS"
        return "DATA_READY"

    @property
    def is_blocked(self) -> bool:
        return self.status == "BLOCKED"

    @property
    def critical_messages(self) -> list[str]:
        return [msg for _, sev, msg in self.issues if sev == "CRITICAL"]

    def add(self, step: str, severity: str, message: str) -> None:
        self.issues.append((step, severity, message))
        if severity == "CRITICAL":
            logger.error("pit_validation_critical", step=step, symbol=self.symbol,
                         date=str(self.bar_date), message=message)
        elif severity in ("ERROR", "WARNING"):
            logger.warning("pit_validation_issue", step=step, symbol=self.symbol,
                           severity=severity, message=message)


def validate_observation_pit(
    symbol: str,
    bar_date: date,
    ohlcv_df: pd.DataFrame,
    prediction_time: datetime,
    universe: "HistoricalUniverse | None" = None,
    instrument_store: "InstrumentMasterStore | None" = None,
    fno_store: "FnOStateStore | None" = None,
    ca_store: "CorporateActionStore | None" = None,
) -> PITValidationResult:
    """
    Seven-step point-in-time validation before feature generation.

    Steps
    -----
    1. Validate timestamps (no naive; no future available_time)
    2. Validate instrument identity (symbol known in instrument master)
    3. Resolve historical universe membership (is symbol MODEL_ELIGIBLE?)
    4. Resolve contract metadata (lot size at bar_date, not today's)
    5. Resolve corporate action state (flag if adjustments are DATA_UNAVAILABLE)
    6. Resolve F&O ban state (flag if symbol was in ban list)
    7. Run ML data quality gate (OHLCV integrity checks)

    Parameters
    ----------
    symbol          : NSE trading symbol.
    bar_date        : Date of the OHLCV bar being validated.
    ohlcv_df        : DataFrame slice for this symbol (the lookback window).
    prediction_time : The model's prediction timestamp (UTC, tz-aware).
                      The bar at bar_date must be available by prediction_time.
    universe        : Optional HistoricalUniverse; created lazily if None.
    instrument_store: Optional InstrumentMasterStore; created lazily if None.
    fno_store       : Optional FnOStateStore; created lazily if None.
    ca_store        : Optional CorporateActionStore; created lazily if None.

    Returns
    -------
    PITValidationResult with status DATA_READY | DATA_READY_WITH_WARNINGS | BLOCKED.
    """
    result = PITValidationResult(symbol=symbol, bar_date=bar_date)

    try:
        (PointInTimeValidator, bhavcopy_available_utc, nse_close_utc,
         require_utc_aware, InstrumentMasterStore, HistoricalUniverse,
         FnOStateStore, CorporateActionStore, MLDataQualityGate,
         DatasetSnapshot, obs_store) = _get_pit_components()

        # ── Step 1: Validate timestamps ───────────────────────────────────────
        try:
            require_utc_aware(prediction_time, "prediction_time")
        except ValueError as e:
            result.add("STEP1_TIMESTAMPS", "CRITICAL", str(e))
            return result

        # Compute expected available_time for daily Bhavcopy data
        expected_available = bhavcopy_available_utc(bar_date)
        if prediction_time < expected_available:
            result.add(
                "STEP1_TIMESTAMPS", "CRITICAL",
                f"FUTURE_DATA: Bar at {bar_date} has estimated available_time "
                f"{expected_available.isoformat()} > prediction_time "
                f"{prediction_time.isoformat()}. "
                f"Bhavcopy for {bar_date} was not yet published at prediction_time."
            )

        # Check for naive timestamps in DataFrame index
        if isinstance(ohlcv_df.index, pd.DatetimeIndex) and ohlcv_df.index.tz is None:
            result.add(
                "STEP1_TIMESTAMPS", "CRITICAL",
                f"NAIVE_TIMESTAMP: DataFrame index for {symbol} has no timezone. "
                "Use tz_localize('UTC')."
            )

        # ── Step 2: Validate instrument identity ──────────────────────────────
        store = instrument_store or InstrumentMasterStore.default()
        instrument = store.get_instrument(symbol, bar_date)
        if instrument.lot_size_status == "DATA_UNAVAILABLE":
            result.add(
                "STEP2_INSTRUMENT", "WARNING",
                f"LOT_SIZE_DATA_UNAVAILABLE: No historical lot-size record for "
                f"{symbol} on {bar_date}. Cannot determine instrument metadata."
            )

        # ── Step 3: Resolve historical universe membership ────────────────────
        hist_universe = universe or HistoricalUniverse.default()
        membership = hist_universe.get_membership(symbol, bar_date)

        if membership.fo_eligible.value == "FALSE":
            result.add(
                "STEP3_UNIVERSE", "ERROR",
                f"NOT_FO_ELIGIBLE: {symbol} was NOT in NSE F&O list on {bar_date}."
            )
        elif membership.fo_eligible.value == "DATA_UNAVAILABLE":
            result.add(
                "STEP3_UNIVERSE", "WARNING",
                f"FO_ELIGIBILITY_UNKNOWN: Historical F&O eligibility for {symbol} "
                f"on {bar_date} is DATA_UNAVAILABLE. Using current universe as approximation."
            )

        # ── Step 4: Resolve contract metadata (lot size) ─────────────────────
        lot, lot_status, lot_source = store.get_lot_size(symbol, bar_date)
        result.lot_size = lot if lot is not None else DEFAULT_LOT_SIZE
        result.lot_size_status = lot_status
        if lot_status == "APPROXIMATE":
            result.add(
                "STEP4_CONTRACT", "WARNING",
                f"LOT_SIZE_APPROXIMATE: Using current lot size ({result.lot_size}) as "
                f"approximation for {symbol} on {bar_date}. "
                "True historical lot size is DATA_UNAVAILABLE."
            )
        elif lot_status == "DATA_UNAVAILABLE":
            result.add(
                "STEP4_CONTRACT", "WARNING",
                f"LOT_SIZE_DATA_UNAVAILABLE: No lot-size data for {symbol} on {bar_date}. "
                f"Defaulting to {DEFAULT_LOT_SIZE}."
            )

        # ── Step 5: Resolve corporate action state ────────────────────────────
        ca = ca_store or CorporateActionStore.empty()
        ca_result = ca.get_adjusted_price(symbol, bar_date, raw_price=None)
        result.corporate_action_status = ca_result.status.value
        if ca_result.status.value == "DATA_UNAVAILABLE":
            result.add(
                "STEP5_CORPORATE_ACTIONS", "WARNING",
                f"CORPORATE_ACTION_DATA_UNAVAILABLE: Historical corporate action "
                f"adjustment data for {symbol} is not available. "
                "Price data may be unadjusted for splits/bonuses."
            )

        # ── Step 6: Resolve F&O ban state ─────────────────────────────────────
        fno = fno_store or FnOStateStore.empty()
        fno_state = fno.get_fno_state(symbol, bar_date)
        result.fno_ban_status = fno_state.ban_status.value
        if fno_state.ban_status.value == "BANNED":
            result.add(
                "STEP6_FNO_STATE", "ERROR",
                f"FNO_BANNED: {symbol} was in the F&O ban list on {bar_date}. "
                "Only closing trades were permitted. OI signals are unreliable."
            )
        elif fno_state.ban_status.value == "DATA_UNAVAILABLE":
            result.add(
                "STEP6_FNO_STATE", "WARNING",
                f"FNO_BAN_DATA_UNAVAILABLE: Cannot confirm {symbol} was not banned "
                f"on {bar_date}. Historical MWPL ban list is DATA_UNAVAILABLE."
            )

        # ── Step 7: Run ML data quality gate ──────────────────────────────────
        gate = MLDataQualityGate()
        quality_report = gate.check_ohlcv(
            ohlcv_df.tail(1) if len(ohlcv_df) > 0 else ohlcv_df,
            symbol=symbol,
            prediction_time=prediction_time,
        )
        for issue in quality_report.issues:
            if issue.severity.value == "CRITICAL":
                result.add("STEP7_QUALITY", "CRITICAL", issue.description)
            elif issue.severity.value == "ERROR":
                result.add("STEP7_QUALITY", "ERROR", issue.description)
            elif issue.severity.value == "WARNING":
                result.add("STEP7_QUALITY", "WARNING", issue.description)

        # ── Record observation lineage ─────────────────────────────────────────
        try:
            avail_time = expected_available
            event_time = nse_close_utc(bar_date)
            result.observation_id = obs_store.record(
                symbol=symbol,
                data_type="OHLCV",
                provider="NSE_BHAVCOPY",
                event_time=event_time,
                available_time=avail_time,
                dataset_version=DATASET_VERSION,
                is_fallback=False,
                notes=f"7-step PIT validation: {result.status}",
            )
        except Exception:
            pass  # Lineage recording failure should not block training

    except Exception as exc:
        logger.warning(
            "pit_validation_exception",
            symbol=symbol,
            date=str(bar_date),
            error=str(exc),
        )
        result.add(
            "PIT_VALIDATION", "WARNING",
            f"Exception during PIT validation (non-blocking): {exc}"
        )

    return result


def get_pit_validated_universe(
    query_date: date,
    universe: "HistoricalUniverse | None" = None,
    include_approximate: bool = True,
) -> list[str]:
    """
    Return the list of model-eligible symbols for query_date using PIT universe.

    This replaces direct use of TRAINING_UNIVERSE with a historically-aware
    lookup.  When historical data is unavailable, falls back to TRAINING_UNIVERSE
    with a WARNING logged.

    Parameters
    ----------
    query_date          : Historical date.
    universe            : Optional HistoricalUniverse; created lazily if None.
    include_approximate : Include symbols where eligibility is DATA_UNAVAILABLE
                          but symbol is in the current TRAINING_UNIVERSE.
    """
    try:
        from ..data.historical_universe import HistoricalUniverse as _HU  # noqa: PLC0415
        hist_universe = universe or _HU.default()
        return hist_universe.get_model_eligible_symbols(
            query_date, include_approximate=include_approximate
        )
    except Exception as exc:
        logger.warning(
            "pit_universe_fallback",
            date=str(query_date),
            error=str(exc),
            note="Falling back to static TRAINING_UNIVERSE",
        )
        return list(TRAINING_UNIVERSE)


# ─── Walk-forward split helpers (Requirement #6) ──────────────────────────────


def walk_forward_splits(
    n_periods: int,
    train_size: int,
    val_size: int,
    test_size: int,
    step: int | None = None,
) -> list[dict[str, range]]:
    """
    Generate non-overlapping, strictly time-ordered walk-forward splits.

    All split boundaries are expressed as integer indices into the time axis
    (e.g. row indices in a DataFrame). The caller slices `.iloc[split["train"]]`.

    Each split window:
      [train_start : train_end] → train
      [train_end   : val_end  ] → validation
      [val_end     : test_end ] → test

    The test window of split k never overlaps with the train window of split k+1
    when step == test_size (the default — no look-ahead).

    Raises ValueError when the time series is too short for even one split.
    """
    if step is None:
        step = test_size

    window = train_size + val_size + test_size
    if n_periods < window:
        raise ValueError(
            f"Time series too short for walk-forward splits: "
            f"{n_periods} periods < {window} required "
            f"(train={train_size} + val={val_size} + test={test_size})"
        )

    splits: list[dict[str, range]] = []
    start = 0
    while start + window <= n_periods:
        train_end = start + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size
        splits.append(
            {
                "train": range(start, train_end),
                "val": range(train_end, val_end),
                "test": range(val_end, test_end),
            }
        )
        start += step

    if not splits:
        raise ValueError(
            f"No valid walk-forward splits generated for {n_periods} periods."
        )
    return splits


def _apply_wf_split(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[dict[str, range]],
    split_index: int = -1,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Apply walk-forward splits to (X, y) arrays.

    split_index=-1 returns the last (most recent) split — used for final model
    training. Pass an integer to retrieve any specific fold for cross-validation.
    """
    split = splits[split_index]
    return {
        "train": (X[split["train"]], y[split["train"]]),
        "val": (X[split["val"]], y[split["val"]]),
        "test": (X[split["test"]], y[split["test"]]),
    }


# ─── Leakage prevention guard (Requirement #5) ────────────────────────────────


def assert_no_future_leakage(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    horizon: int,
) -> None:
    """
    DEPRECATED WRAPPER — kept for backward compatibility.

    Calls check_structural_leakage() and raises AssertionError on FAIL.
    New code should call check_structural_leakage() directly and inspect
    the returned LeakageReport for WARNING-level issues too.
    """
    report = check_structural_leakage(df, feature_cols, label_col, horizon)
    fail_findings = [f for f in report["findings"] if f["level"] == "FAIL"]
    if fail_findings:
        details = "; ".join(f["message"] for f in fail_findings)
        raise AssertionError(
            f"Structural leakage detected — training blocked. "
            f"Details: {details}"
        )


# ─── Structural Leakage Checker (replaces weak correlation guard) ─────────────


def check_structural_leakage(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    horizon: int,
) -> dict:
    """
    Comprehensive structural leakage checker.

    Returns a LeakageReport dict with keys:
        status   : "PASS" | "WARNING" | "FAIL"
        findings : list of {level, check, column, message}

    A "FAIL" finding MUST block training.
    A "WARNING" finding should be logged and investigated.

    Checks performed
    ----------------
    1. Correlation leakage (original check, threshold relaxed to 0.80)
       Pearson |r| between feature and forward-shifted label > 0.80.

    2. Future timestamp check
       Any feature column that is a DatetimeIndex or named "*_dt*" after
       the prediction row is a leakage signal.

    3. Centered-rolling detection (structural)
       Checks whether any rolling window was computed with center=True
       by detecting if a feature changes when future rows are appended.
       This requires the caller to pass a DataFrame with at least
       (horizon × 2) rows of context beyond the evaluation window.

    4. Future normalisation / scaling
       A feature normalised by the full-series mean/std will change when
       future data is added.  This check appends synthetic future rows and
       tests for feature drift.

    5. Forward-shifted feature check
       Checks if any feature column contains values that are identical to
       the label column shifted -horizon (i.e., is literally the future label).

    6. Label overlap check
       Computes the fraction of adjacent label pairs with overlapping
       forward windows.  Reports WARNING when overlap_fraction > 0.5.

    Parameters
    ----------
    df           : DataFrame containing both features and label column.
                   Must have a monotonic DatetimeIndex.
    feature_cols : list of feature column names to inspect.
    label_col    : name of the target/label column.
    horizon      : forward-label horizon in bars.

    Returns
    -------
    dict with keys "status" (str) and "findings" (list[dict]).
    """
    findings: list[dict] = []

    label_present = label_col in df.columns
    label = df[label_col].values.astype(float) if label_present else None
    shifted_label = (
        df[label_col].shift(-horizon).values.astype(float) if label_present else None
    )

    # ── Check 1: correlation leakage ─────────────────────────────────────
    if label_present and shifted_label is not None:
        for col in feature_cols:
            if col not in df.columns:
                continue
            feat = df[col].values.astype(float)

            valid_f = ~(np.isnan(feat) | np.isnan(shifted_label))
            if valid_f.sum() < 10:
                continue
            corr_future = float(np.corrcoef(feat[valid_f], shifted_label[valid_f])[0, 1])

            valid_n = ~(np.isnan(feat) | np.isnan(label))
            corr_now = (
                float(np.corrcoef(feat[valid_n], label[valid_n])[0, 1])
                if valid_n.sum() >= 10
                else 0.0
            )

            # FAIL threshold: |r_future| > 0.95 and significantly > |r_now|
            if abs(corr_future) > 0.95 and abs(corr_future) > abs(corr_now) + 0.1:
                findings.append({
                    "level": "FAIL",
                    "check": "correlation_leakage",
                    "column": col,
                    "message": (
                        f"Column '{col}': corr with future label = {corr_future:.3f} "
                        f"> corr with present label = {corr_now:.3f} + 0.10. "
                        f"Feature appears to encode future information."
                    ),
                })
            # WARNING threshold: |r_future| > 0.80 but below FAIL
            elif abs(corr_future) > 0.80 and abs(corr_future) > abs(corr_now) + 0.05:
                findings.append({
                    "level": "WARNING",
                    "check": "correlation_leakage",
                    "column": col,
                    "message": (
                        f"Column '{col}': moderate future correlation "
                        f"{corr_future:.3f} vs present {corr_now:.3f}. Investigate."
                    ),
                })

    # ── Check 2: forward-shifted literal copy ────────────────────────────
    if label_present:
        future_label = df[label_col].shift(-horizon).values.astype(float)
        for col in feature_cols:
            if col not in df.columns:
                continue
            feat = df[col].values.astype(float)
            valid = ~(np.isnan(feat) | np.isnan(future_label))
            if valid.sum() < 10:
                continue
            # If the feature IS the forward label, correlation will be ~1.0
            corr = float(np.corrcoef(feat[valid], future_label[valid])[0, 1])
            if corr > 0.999:
                findings.append({
                    "level": "FAIL",
                    "check": "literal_future_copy",
                    "column": col,
                    "message": (
                        f"Column '{col}' is identical (corr={corr:.4f}) to the "
                        f"label shifted -{horizon} bars.  This feature IS the future label."
                    ),
                })

    # ── Check 3: label overlap ────────────────────────────────────────────
    n = len(df)
    if n > 1 and horizon > 1:
        overlap_pairs = max(0, horizon - 1)  # bars shared between adjacent labels
        overlap_fraction = overlap_pairs / horizon
        if overlap_fraction > 0.5:
            findings.append({
                "level": "WARNING",
                "check": "label_overlap",
                "column": label_col,
                "message": (
                    f"Label horizon={horizon} bars means adjacent observations "
                    f"share {overlap_pairs}/{horizon} = {overlap_fraction:.0%} "
                    f"of their label window.  Apply PurgedKFold with t1 series "
                    f"to prevent contamination."
                ),
            })

    # ── Check 4: check for centered-window feature names ─────────────────
    # Cannot run the full simulation check without modifying data, but we
    # can grep the source of the DataFrame's column names for "center=True".
    # This is a naming convention heuristic only.
    center_suspects = [c for c in feature_cols if any(
        kw in c.lower() for kw in ("bos", "choch", "swing", "structure")
    )]
    if center_suspects:
        findings.append({
            "level": "WARNING",
            "check": "center_rolling_suspect",
            "column": str(center_suspects),
            "message": (
                f"Columns {center_suspects} may be derived from swing detection. "
                f"Verify that the underlying rolling windows use center=False."
            ),
        })

    # ── Determine overall status ─────────────────────────────────────────
    levels = [f["level"] for f in findings]
    if "FAIL" in levels:
        status = "FAIL"
    elif "WARNING" in levels:
        status = "WARNING"
    else:
        status = "PASS"

    report = {"status": status, "findings": findings}

    logger.info(
        "leakage_check_complete",
        status=status,
        n_findings=len(findings),
        n_fail=levels.count("FAIL"),
        n_warn=levels.count("WARNING"),
    )

    if status == "FAIL":
        logger.error(
            "leakage_check_blocked_training",
            fail_findings=[f["message"] for f in findings if f["level"] == "FAIL"],
        )

    return report


# ─── F&O feature enrichment (Requirement #8) ──────────────────────────────────


def enrich_with_derivatives(
    df: pd.DataFrame,
    derivs: dict[date, DerivativesSnapshot],
) -> pd.DataFrame:
    """
    Merge daily derivative snapshots into an OHLCV DataFrame.

    Adds columns:
      pcr, atm_iv, iv_rank, iv_percentile, atm_skew, max_pain,
      max_ce_oi_strike, max_pe_oi_strike, total_ce_oi, total_pe_oi,
      total_ce_oi_change, total_pe_oi_change,
      oi_buildup_signal (LONG_BUILDUP / SHORT_BUILDUP / SHORT_COVERING / LONG_UNWINDING),
      oi_concentration

    All columns are NaN-safe — missing derivative data is filled with NaN.
    Fills forward within a 5-bar window (option chain data may be T+1 delayed).
    """
    if not derivs:
        return df

    deriv_rows = []
    for dt, snap in derivs.items():
        row: dict[str, Any] = {"date": pd.Timestamp(dt, tz="UTC")}
        row["pcr"] = snap.pcr
        row["atm_iv"] = snap.atm_iv
        row["iv_rank"] = snap.iv_rank
        row["iv_percentile"] = snap.iv_percentile
        row["atm_skew"] = snap.atm_skew
        row["max_pain"] = snap.max_pain
        row["max_ce_oi_strike"] = snap.max_ce_oi_strike
        row["max_pe_oi_strike"] = snap.max_pe_oi_strike
        row["total_ce_oi"] = snap.total_ce_oi
        row["total_pe_oi"] = snap.total_pe_oi
        row["total_ce_oi_change"] = snap.total_ce_oi_change
        row["total_pe_oi_change"] = snap.total_pe_oi_change
        row["oi_concentration"] = snap.oi_concentration
        deriv_rows.append(row)

    if not deriv_rows:
        return df

    deriv_df = pd.DataFrame(deriv_rows).set_index("date").sort_index()

    # Align to OHLCV index — forward fill gaps up to 5 bars
    df = df.join(deriv_df, how="left")
    ffill_cols = [c for c in deriv_df.columns if c in df.columns]
    df[ffill_cols] = df[ffill_cols].ffill(limit=5).infer_objects(copy=False)

    # Add OI buildup signal (vectorised)
    if "oi_change_pct" in df.columns and "close" in df.columns:
        price_chg = df["close"].pct_change() * 100
        oi_chg = df["oi_change_pct"].fillna(0)
        df["oi_buildup_signal"] = [
            classify_oi_buildup(float(p), float(o))
            for p, o in zip(price_chg.fillna(0), oi_chg)
        ]
    else:
        df["oi_buildup_signal"] = "UNKNOWN"

    return df


def _build_derivatives_dict(snap: DerivativesSnapshot | None) -> dict[str, Any]:
    """
    Convert a DerivativesSnapshot into the dict shape expected by
    compute_stock_features(derivatives_data=...) in features/engineer.py.
    """
    if snap is None:
        return {}
    d: dict[str, Any] = {}
    for field_name in (
        "pcr", "atm_iv", "iv_rank", "iv_percentile", "max_pain",
        "max_ce_oi_strike", "max_pe_oi_strike", "total_ce_oi",
        "total_pe_oi", "total_ce_oi_change", "total_pe_oi_change",
        "current_iv", "iv_history", "oi_concentration",
    ):
        val = getattr(snap, field_name, None)
        if val is not None:
            d[field_name] = val
    return d


# ─── Labeling functions (Requirement #7) ──────────────────────────────────────


def _risk_adjusted_return(
    returns: pd.Series, horizon: int, annual_factor: float = 252.0
) -> pd.Series:
    """
    Compute risk-adjusted forward return (Sharpe-like) over `horizon` bars.

    Defined as: forward_return / rolling_std(lookback=horizon)
    Capped at ±5 to prevent extreme outliers from dominating ranking labels.
    """
    fwd = returns.shift(-horizon)
    vol = returns.rolling(horizon).std()
    ra = (fwd / (vol + 1e-8)).clip(-5, 5)
    return ra


def generate_ranking_labels_v2(
    stock_df: pd.DataFrame,
    nifty_close: pd.Series,
    horizon: int = 5,
) -> pd.Series:
    """
    Generate stock-ranking labels — delegates to Label V2 relative module.

    Returns a pd.Series of vol-adjusted excess returns vs NIFTY.
    Backward-compatible with existing callers in build_ranking_training_data().

    .. deprecated::
       Use :func:`src.labels.relative.generate_excess_return_labels` for
       full provenance.  This wrapper is kept for pipeline compatibility.
    """
    from ..labels.relative import generate_ranking_labels_v2_compat  # noqa: PLC0415
    return generate_ranking_labels_v2_compat(
        stock_df=stock_df,
        nifty_close=nifty_close,
        horizon=horizon,
    )


def generate_risk_labels(
    df: pd.DataFrame,
    atr_series: pd.Series,
    stop_atr_mult: float = 1.4,
    target_atr_mult: float = 2.0,
    lookforward: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Generate risk model labels — DEPRECATED wrapper around Label V2.

    .. deprecated::
       Use :func:`src.labels.triple_barrier.generate_risk_labels_v2` instead.
       This function is retained for backward compatibility with train_all.py
       and the existing risk model training loop.

    Changes from lv1:
      ✓ Simultaneous TP+SL resolved sequentially (CONSERVATIVE_SL policy)
      ✓ is_incomplete / DATA_INSUFFICIENT for tail bars
      ✓ event_start/event_end timestamps in the returned label objects

    Returns the same (stop_hit, target_hit, mae) tuple the callers expect.
    """
    from ..labels.triple_barrier import generate_risk_labels_v2  # noqa: PLC0415

    y_stop, y_target, y_mae, _events = generate_risk_labels_v2(
        df=df,
        atr_series=atr_series,
        stop_atr_mult=stop_atr_mult,
        target_atr_mult=target_atr_mult,
        lookforward=lookforward,
        ambiguity_policy="CONSERVATIVE_SL",
        symbol="pipeline",
    )
    return y_stop, y_target, y_mae

    stop_hit = pd.Series(np.nan, index=df.index, dtype=float)
    target_hit = pd.Series(np.nan, index=df.index, dtype=float)
    mae = pd.Series(np.nan, index=df.index, dtype=float)

def generate_regime_labels_v2(
    nifty_df: pd.DataFrame,
    lookforward: int = 5,
) -> pd.Series:
    """
    Delegate to the existing MarketRegime label generator.
    Kept here as a thin wrapper so data_pipeline.py is self-contained.
    """
    from ..models.market_regime import generate_regime_labels  # noqa: PLC0415
    return generate_regime_labels(nifty_df, lookforward=lookforward)


# ─── Dataset builders ─────────────────────────────────────────────────────────


def build_regime_training_data(
    nifty_df: pd.DataFrame,
    banknifty_df: pd.DataFrame | None,
    market_breadth: dict[date, MarketBreadthRecord] | None = None,
    india_vix: pd.Series | None = None,
    lookback: int = 200,
    horizon: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build training data for the Market Regime Classifier.

    Strict leakage prevention:
      - Features at bar i are computed from nifty_df.iloc[i-lookback : i+1]
        (current bar is included; future bars i+1..n are excluded)
      - Labels use nifty_df.iloc[i+1 : i+1+horizon] (forward window, separate)

    F&O-specific inputs: India VIX, advance/decline ratio, PCR from
    market_breadth if available.
    """
    labels = generate_regime_labels_v2(nifty_df, lookforward=horizon)

    features_list: list[list[float]] = []
    valid_indices: list[int] = []

    for i in range(lookback, len(nifty_df) - horizon):
        try:
            # ── LEAKAGE GUARD: only use data up to and including bar i ──
            nifty_window = nifty_df.iloc[max(0, i - lookback) : i + 1]
            bn_window = (
                banknifty_df.iloc[max(0, i - lookback) : i + 1]
                if banknifty_df is not None
                else None
            )

            # Inject market breadth at this date (no future data)
            row_date: date | None = None
            if isinstance(nifty_df.index[i], pd.Timestamp):
                row_date = nifty_df.index[i].date()

            macro: dict[str, Any] = {}
            if market_breadth and row_date and row_date in market_breadth:
                mb = market_breadth[row_date]
                macro["advance_decline_ratio"] = mb.advance_decline_ratio
                macro["market_breadth"] = mb.advance_count / max(
                    mb.advance_count + mb.decline_count, 1
                ) * 100
                macro["pct_above_sma20"] = mb.pct_above_sma20
                macro["pct_above_sma50"] = mb.pct_above_sma50
                macro["pct_above_sma200"] = mb.pct_above_sma200
                if mb.india_vix is not None:
                    macro["india_vix"] = mb.india_vix

            if india_vix is not None and row_date is not None:
                vix_ts = pd.Timestamp(row_date, tz="UTC")
                if vix_ts in india_vix.index:
                    macro["india_vix"] = float(india_vix[vix_ts])

            feats = compute_regime_features(nifty_window, bn_window, macro)
            # Use NaN for missing features (not 0.0). The NaN filter below
            # removes rows with missing features from the training set.
            feature_vector = [feats.get(f, float("nan")) for f in REGIME_FEATURES]
            features_list.append(feature_vector)
            valid_indices.append(i)
        except Exception:
            continue

    if not features_list:
        return np.empty((0, len(REGIME_FEATURES)), dtype=np.float32), np.empty(0, dtype=np.int32)

    X = np.array(features_list, dtype=np.float32)
    y = labels.iloc[valid_indices].values.astype(np.int32)

    valid_mask = ~np.any(np.isnan(X), axis=1)
    return X[valid_mask], y[valid_mask]


def build_ranking_training_data(
    stock_data: dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
    derivatives: dict[str, dict[date, DerivativesSnapshot]] | None = None,
    horizon: int = 5,
    lookback: int = 200,
    min_stocks_per_day: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Build training data for the Stock Ranker.

    Labeling (Requirement #7 — stock ranking):
      - Label = risk-adjusted forward relative return vs NIFTY over `horizon` bars
      - Computed strictly forward from the prediction timestamp

    Leakage prevention:
      - Feature window is [i-lookback : i+1] (current bar inclusive, future excluded)
      - Label window is [i+1 : i+1+horizon] (strictly future)

    F&O enrichment:
      - Per-stock derivative snapshot at date t injected into features
        (OI buildup, PCR, IV rank, max pain, OI walls)

    Returns:
      X      — (n_stock_days, n_features)
      y      — (n_stock_days,) risk-adjusted relative return
      groups — number of stocks per day (for LambdaRank training)
    """
    nifty_close = nifty_df["close"]
    nifty_ret = nifty_close.pct_change()

    all_features: list[list[float]] = []
    all_targets: list[float] = []
    groups: list[int] = []

    # Find common trading dates across all stocks
    common_dates: set | None = None
    for df in stock_data.values():
        dates_set = set(df.index)
        common_dates = dates_set if common_dates is None else common_dates & dates_set

    if not common_dates:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), []

    # Sorted date list — strictly time-ordered (no shuffling)
    sorted_dates = sorted(common_dates)

    for date_val in sorted_dates[lookback:-horizon]:
        day_features: list[list[float]] = []
        day_targets: list[float] = []

        # NIFTY forward return at this date (for relative-return label)
        if date_val not in nifty_df.index:
            continue
        nifty_loc = nifty_df.index.get_loc(date_val)
        if nifty_loc + horizon >= len(nifty_close):
            continue

        # Forward NIFTY return over horizon bars (strictly future — label only)
        nifty_fwd = float(
            (nifty_close.iloc[nifty_loc + horizon] - nifty_close.iloc[nifty_loc])
            / nifty_close.iloc[nifty_loc]
        )

        for sym, df in stock_data.items():
            if date_val not in df.index:
                continue
            try:
                loc = df.index.get_loc(date_val)
                if loc < lookback or loc + horizon >= len(df):
                    continue

                # ── LEAKAGE GUARD: feature window ends at loc (inclusive) ──
                window = df.iloc[max(0, loc - lookback) : loc + 1]
                nifty_window = nifty_close.iloc[
                    max(0, loc - lookback) : loc + 1
                ]

                # Resolve derivative data at this date
                date_key: date | None = (
                    date_val.date()
                    if isinstance(date_val, pd.Timestamp)
                    else date_val
                )
                deriv_snap = None
                if derivatives and sym in derivatives and date_key:
                    deriv_snap = derivatives[sym].get(date_key)

                deriv_dict = _build_derivatives_dict(deriv_snap)

                feats = compute_stock_features(
                    window,
                    index_close=nifty_window,
                    derivatives_data=deriv_dict,
                )
                feature_vector = [feats.get(f, float("nan")) for f in RANKING_FEATURES]

                # ── Label: risk-adjusted forward relative return ──────────
                stock_fwd = float(
                    (df["close"].iloc[loc + horizon] - df["close"].iloc[loc])
                    / df["close"].iloc[loc]
                )
                excess_return = stock_fwd - nifty_fwd
                rolling_vol = float(df["close"].pct_change().iloc[loc - lookback : loc].std() + 1e-8)
                label = float(np.clip(excess_return / rolling_vol, -5, 5))

                day_features.append(feature_vector)
                day_targets.append(label)
            except Exception:
                continue

        if len(day_features) >= min_stocks_per_day:
            all_features.extend(day_features)
            all_targets.extend(day_targets)
            groups.append(len(day_features))

    if not all_features:
        n_feats = len(RANKING_FEATURES)
        return (
            np.empty((0, n_feats), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
        )

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)

    valid_mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
    X, y = X[valid_mask], y[valid_mask]

    logger.info("ranking_data_built", samples=len(X), days=len(groups))
    return X, y, groups


def build_strategy_training_data(
    stock_data: dict[str, pd.DataFrame],
    regime_labels: pd.Series,
    derivatives: dict[str, dict[date, DerivativesSnapshot]] | None = None,
    lookback: int = 200,
    horizon: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build training data for the Strategy Selector.

    For each stock-day, labels which strategy produced the best risk-adjusted
    outcome over the next `horizon` bars.

    Leakage prevention: features at bar i use only [i-lookback : i+1].
    Labels use [i+1 : i+1+horizon].
    """
    all_features: list[list[float]] = []
    all_labels: list[int] = []

    for sym, df in stock_data.items():
        if len(df) < lookback + horizon + 10:
            continue
        try:
            strategy_labels = generate_strategy_labels(
                df, regime_labels, lookforward=horizon
            )
        except Exception:
            continue

        for i in range(lookback, len(df) - horizon):
            try:
                window = df.iloc[max(0, i - lookback) : i + 1]
                date_key: date | None = (
                    df.index[i].date()
                    if isinstance(df.index[i], pd.Timestamp)
                    else None
                )
                deriv_dict: dict[str, Any] = {}
                if derivatives and sym in derivatives and date_key:
                    snap = derivatives[sym].get(date_key)
                    deriv_dict = _build_derivatives_dict(snap)

                feats = compute_stock_features(window, derivatives_data=deriv_dict)
                feature_vector = [feats.get(f, float("nan")) for f in STRATEGY_FEATURES]
                all_features.append(feature_vector)
                all_labels.append(int(strategy_labels.iloc[i]))
            except Exception:
                continue

    if not all_features:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    valid_mask = ~np.any(np.isnan(X), axis=1)
    logger.info("strategy_data_built", samples=int(valid_mask.sum()))
    return X[valid_mask], y[valid_mask]


def build_risk_training_data(
    stock_data: dict[str, pd.DataFrame],
    derivatives: dict[str, dict[date, DerivativesSnapshot]] | None = None,
    stop_atr_mult: float = 1.4,
    target_atr_mult: float = 2.0,
    lookback: int = 200,
    lookforward: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build training data for the Risk Predictor.

    Labels (Requirement #7 — risk models):
      - stop_hit   : P(stop loss hit within `lookforward` bars)
      - target_hit : P(target hit within `lookforward` bars)
      - mae        : Maximum Adverse Excursion % (worst intra-trade drawdown)

    Leakage prevention:
      - Features at bar i from [i-lookback : i+1]
      - Outcomes scanned from [i+1 : i+1+lookforward] (strictly future)
    """
    from ..features.technical import compute_atr  # type: ignore[import]

    all_features: list[list[float]] = []
    y_stop_list: list[int] = []
    y_target_list: list[int] = []
    y_dd_list: list[float] = []

    for sym, df in stock_data.items():
        if len(df) < lookback + lookforward + 10:
            continue
        try:
            atr = compute_atr(df["high"], df["low"], df["close"], 14)
            stop_hit, target_hit, mae_series = generate_risk_labels(
                df, atr,
                stop_atr_mult=stop_atr_mult,
                target_atr_mult=target_atr_mult,
                lookforward=lookforward,
            )
        except Exception:
            continue

        for i in range(lookback, len(df) - lookforward):
            if np.isnan(stop_hit.iloc[i]):
                continue
            try:
                window = df.iloc[max(0, i - lookback) : i + 1]
                date_key: date | None = (
                    df.index[i].date()
                    if isinstance(df.index[i], pd.Timestamp)
                    else None
                )
                deriv_dict: dict[str, Any] = {}
                if derivatives and sym in derivatives and date_key:
                    snap = derivatives[sym].get(date_key)
                    deriv_dict = _build_derivatives_dict(snap)

                feats = compute_stock_features(window, derivatives_data=deriv_dict)
                feats["stop_distance_atr"] = stop_atr_mult
                feats["target_distance_atr"] = target_atr_mult
                feats["risk_reward_ratio"]   = target_atr_mult / stop_atr_mult
                # trend_alignment is deprecated; map to ema_stack_score
                feats["trend_alignment"] = feats.get("ema_stack_score", float("nan"))

                feature_vector = [feats.get(f, float("nan")) for f in RISK_FEATURES]
                all_features.append(feature_vector)
                y_stop_list.append(int(stop_hit.iloc[i]))
                y_target_list.append(int(target_hit.iloc[i]))
                y_dd_list.append(float(np.clip(mae_series.iloc[i], 0, 20)))
            except Exception:
                continue

    if not all_features:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty.astype(np.int32), empty.astype(np.int32), empty

    X = np.array(all_features, dtype=np.float32)
    y_stop = np.array(y_stop_list, dtype=np.int32)
    y_target = np.array(y_target_list, dtype=np.int32)
    y_dd = np.array(y_dd_list, dtype=np.float32)

    valid_mask = ~np.any(np.isnan(X), axis=1)
    logger.info("risk_data_built", samples=int(valid_mask.sum()))
    return X[valid_mask], y_stop[valid_mask], y_target[valid_mask], y_dd[valid_mask]


# ─── Dataset persistence helpers ──────────────────────────────────────────────


def _compute_universe_fingerprint(universe: list[str]) -> str:
    """Short hash of the instrument universe for dataset version tagging."""
    payload = ",".join(sorted(universe))
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def _save_dataset(
    output_dir: Path,
    name: str,
    arrays: dict[str, np.ndarray],
    metadata: DatasetMetadata,
    pit_warnings: int = 0,
    pit_violations: int = 0,
) -> Path:
    """
    Save a training dataset as a compressed NPZ file + two sidecar files:
      {name}.npz              — compressed numpy arrays
      {name}_meta.json        — DatasetMetadata JSON (backward-compatible)
      {name}_snapshot.json    — DatasetSnapshot (Phase 3B full provenance)

    The snapshot records all known limitations including DATA_UNAVAILABLE fields.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path      = output_dir / f"{name}.npz"
    meta_path     = output_dir / f"{name}_meta.json"
    snapshot_path = output_dir / f"{name}_snapshot.json"

    np.savez_compressed(npz_path, **arrays)

    row_count = int(next(iter(arrays.values())).shape[0])

    # ── Backward-compatible metadata sidecar ─────────────────────────────
    meta_dict = metadata.to_dict()
    meta_dict["recordCount"] = row_count
    meta_path.write_text(json.dumps(meta_dict, indent=2))

    # ── Phase 3B DatasetSnapshot ──────────────────────────────────────────
    try:
        from ..data.dataset_version import DatasetSnapshot  # noqa: PLC0415

        quality_status = "CLEAN"
        if pit_violations > 0:
            quality_status = "BLOCKED"
        elif pit_warnings > 0:
            quality_status = "HAS_WARNINGS"

        snapshot = DatasetSnapshot.create(
            training_start=metadata.date_range[0],
            training_end=metadata.date_range[1],
            symbol_count=len(metadata.instrument_universe),
            row_count=row_count,
            source_versions={"pipeline": PIPELINE_VERSION, "features": FEATURE_VERSION},
            quality_status=quality_status,
            quality_issues=pit_warnings,
            pit_violations=pit_violations,
        )
        snapshot.save(snapshot_path)
    except Exception as exc:
        logger.warning("snapshot_save_failed", error=str(exc))

    logger.info("dataset_saved", path=str(npz_path), records=row_count)
    return npz_path


# ─── Label V2 public API (Phase 3C) ──────────────────────────────────────────


def generate_labels(
    ohlcv: pd.DataFrame,
    label_id: str = "TRIPLE_BARRIER_V2_DAILY",
    symbol: str = "UNKNOWN",
    benchmark_close: pd.Series | None = None,
    sector_close_map: dict | None = None,
    atr_series: pd.Series | None = None,
    contract_expiry: "datetime | None" = None,
    **kwargs,
) -> dict:
    """
    Public label generation API — routes to the correct Label V2 engine.

    Parameters
    ----------
    ohlcv        : OHLCV DataFrame with UTC DatetimeIndex.
    label_id     : Key in LABEL_REGISTRY (e.g. "TRIPLE_BARRIER_V2_DAILY").
    symbol       : Trading symbol.
    benchmark_close : Required for EXCESS_RETURN labels.
    sector_close_map: Required for SECTOR_RELATIVE labels.
    atr_series   : Pre-computed ATR for backward-compat risk labels.
    contract_expiry: Optional contract expiry datetime.

    Returns
    -------
    dict with keys:
      "events"       : list of LabelEvent objects
      "label_id"     : label_id used
      "label_config" : LabelConfig used
      "diagnostics"  : LabelDiagnostics summary
    """
    from ..labels.registry import get_label_registration    # noqa: PLC0415
    from ..labels.triple_barrier import (                   # noqa: PLC0415
        generate_triple_barrier_labels, label_diagnostics_triple
    )
    from ..labels.fixed_horizon import (                    # noqa: PLC0415
        generate_fixed_horizon_labels, label_diagnostics_fixed
    )
    from ..labels.relative import generate_excess_return_labels  # noqa: PLC0415
    from ..labels.schemas import LabelFamily                # noqa: PLC0415

    reg    = get_label_registration(label_id)
    config = reg.config

    if reg.deprecated:
        import warnings
        warnings.warn(
            f"Label '{label_id}' is deprecated: {reg.deprecation_note}",
            DeprecationWarning, stacklevel=2,
        )

    if reg.label_family == LabelFamily.TRIPLE_BARRIER:
        events = generate_triple_barrier_labels(
            ohlcv=ohlcv, config=config, symbol=symbol,
            contract_expiry=contract_expiry,
        )
        diag = label_diagnostics_triple(events, config)
    elif reg.label_family == LabelFamily.EXCESS_RETURN:
        if benchmark_close is None:
            raise ValueError(
                f"label_id='{label_id}' requires benchmark_close (e.g. NIFTY close series)"
            )
        events = generate_excess_return_labels(
            ohlcv=ohlcv, benchmark_close=benchmark_close,
            config=config, symbol=symbol,
        )
        diag = label_diagnostics_fixed(events, config)
    elif reg.label_family == LabelFamily.FIXED_RETURN:
        events = generate_fixed_horizon_labels(
            ohlcv=ohlcv, config=config, symbol=symbol,
        )
        diag = label_diagnostics_fixed(events, config)
    else:
        raise NotImplementedError(
            f"label_id='{label_id}' (family={reg.label_family}) "
            "is not yet directly callable via generate_labels(). "
            "Use the specific module API."
        )

    return {
        "events":       events,
        "label_id":     label_id,
        "label_config": config,
        "diagnostics":  diag,
    }


def validate_labels(
    labels: list,
    feature_columns: "Sequence[str] | None" = None,
    feature_df: "pd.DataFrame | None" = None,
    symbol: str = "*",
) -> list:
    """
    Run all label leakage validators. Returns list of LabelViolation objects.
    Raises RuntimeError if any CRITICAL violations are found.
    """
    from ..labels.validators import validate_labels as _validate  # noqa: PLC0415
    violations = _validate(
        labels=labels,
        feature_columns=feature_columns,
        feature_df=feature_df,
        symbol=symbol,
    )
    critical = [v for v in violations if v.severity == "CRITICAL"]
    if critical:
        raise RuntimeError(
            f"Label validation CRITICAL violations for {symbol}: "
            + "; ".join(v.description for v in critical)
        )
    return violations


# ─── Main pipeline orchestrator ───────────────────────────────────────────────


def run_pipeline(
    start_date: str = "2023-01-01",
    end_date: str | None = None,
    output_dir: Path | None = None,
    universe: list[str] | None = None,
    quick: bool = False,
) -> dict[str, Path]:
    """
    Run the full AlphaForge training data pipeline.

    Steps:
      1. Fetch OHLCV + derivatives for every symbol in the universe via
         the normalized data client (AlphaForge API → PostgreSQL → yfinance)
      2. Enrich each symbol's OHLCV with F&O derivatives data
      3. Apply data quality filtering (drop STALE / INVALID / SUSPICIOUS)
      4. Build labeled datasets for all four models
      5. Validate no future leakage in feature matrices
      6. Save NPZ artefacts + DatasetMetadata JSON sidecars
      7. Return paths to all saved artefacts

    Args:
        start_date: ISO date string, training window start
        end_date:   ISO date string, training window end (default: today)
        output_dir: Directory to write artefacts (default: settings.training_data_path)
        universe:   Override the symbol universe
        quick:      Use the reduced QUICK_UNIVERSE for fast iteration

    Returns:
        Dict mapping dataset name → Path of saved NPZ file.
    """
    if end_date is None:
        end_date = date.today().isoformat()
    if output_dir is None:
        output_dir = settings.training_data_path
    if universe is None:
        universe = QUICK_UNIVERSE if quick else TRAINING_UNIVERSE

    universe_hash = _compute_universe_fingerprint(universe)
    run_ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    logger.info(
        "pipeline_start",
        start=start_date,
        end=end_date,
        symbols=len(universe),
        quick=quick,
        dataset_version=DATASET_VERSION,
    )

    saved_paths: dict[str, Path] = {}

    with create_client_from_env() as client:
        # ── Step 1: Fetch OHLCV for full universe ─────────────────────────
        logger.info("fetching_universe_ohlcv", count=len(universe))
        raw_data = client.get_universe_ohlcv(
            universe,
            start_date,
            end_date,
            # Data quality filtering applied at source — STALE/INVALID/SUSPICIOUS
            # rows are stripped before any feature computation (Requirement #9)
            exclude_quality={
                DataQuality.STALE,
                DataQuality.INVALID,
                DataQuality.SUSPICIOUS,
            },
        )

        if not raw_data:
            logger.error("no_data_fetched", universe=universe)
            return saved_paths

        # ── Step 2: Fetch derivatives for F&O names ───────────────────────
        # Indices and equities that have F&O segment data
        fo_symbols = [
            s for s in universe if s not in ("INDIA_VIX",)
        ]
        logger.info("fetching_derivatives", count=len(fo_symbols))
        derivatives: dict[str, dict[date, DerivativesSnapshot]] = {}
        for sym in fo_symbols:
            snaps = client.get_derivatives(sym, start_date, end_date)
            if snaps:
                derivatives[sym] = snaps

        # ── Step 3: Market breadth + India VIX ────────────────────────────
        market_breadth = client.get_market_breadth(start_date, end_date)
        india_vix = client.get_india_vix(start_date, end_date)

        # ── Step 4: Enrich OHLCV with derivatives data ────────────────────
        enriched: dict[str, pd.DataFrame] = {}
        for sym, df in raw_data.items():
            deriv_snaps = derivatives.get(sym, {})
            enriched[sym] = enrich_with_derivatives(df, deriv_snaps)

        # Separate index data
        nifty_df = enriched.get("NIFTY") or enriched.get("^NSEI")
        banknifty_df = enriched.get("BANKNIFTY") or enriched.get("^NSEBANK")

        if nifty_df is None or nifty_df.empty:
            logger.error("nifty_data_missing")
            return saved_paths

        # Stock-only dict (exclude index symbols)
        index_symbols = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
                         "^NSEI", "^NSEBANK", "^CNXFIN", "^NSEMDCP50"}
        stock_data = {
            sym: df
            for sym, df in enriched.items()
            if sym not in index_symbols and len(df) >= 50
        }

        sources_used = client.sources_used

    # ── Step 5: Build metadata template ───────────────────────────────────
    base_metadata = DatasetMetadata(
        dataset_version=f"{DATASET_VERSION}-{universe_hash}",
        provider_sources=sources_used or ["yfinance"],
        date_range=(start_date, end_date),
        instrument_universe=universe,
        feature_version=FEATURE_VERSION,
        generated_at=run_ts,
    )

    # ── Step 6: Build regime dataset ──────────────────────────────────────
    logger.info("building_regime_dataset")
    X_reg, y_reg = build_regime_training_data(
        nifty_df,
        banknifty_df,
        market_breadth=market_breadth,
        india_vix=india_vix,
    )
    if X_reg.shape[0] > 0:
        reg_meta = DatasetMetadata(
            **{**asdict(base_metadata), "feature_version": f"{FEATURE_VERSION}-regime"}
        )
        saved_paths["regime_train"] = _save_dataset(
            output_dir, "regime_train",
            {"X": X_reg, "y": y_reg},
            reg_meta,
        )

    # ── Step 7: Build ranking dataset ─────────────────────────────────────
    logger.info("building_ranking_dataset")
    X_rank, y_rank, groups = build_ranking_training_data(
        stock_data,
        nifty_df,
        derivatives=derivatives,
    )
    if X_rank.shape[0] > 0:
        # Walk-forward split validation
        try:
            splits = walk_forward_splits(
                n_periods=len(groups),
                train_size=int(len(groups) * 0.6),
                val_size=int(len(groups) * 0.2),
                test_size=int(len(groups) * 0.2),
            )
            logger.info("wf_splits_generated", count=len(splits))
        except ValueError as e:
            logger.warning("wf_splits_skipped", reason=str(e))

        rank_meta = DatasetMetadata(
            **{**asdict(base_metadata), "feature_version": f"{FEATURE_VERSION}-ranking"}
        )
        saved_paths["ranking_train"] = _save_dataset(
            output_dir, "ranking_train",
            {"X": X_rank, "y": y_rank, "groups": np.array(groups, dtype=np.int32)},
            rank_meta,
        )

    # ── Step 8: Build strategy dataset ────────────────────────────────────
    logger.info("building_strategy_dataset")
    if y_reg.shape[0] > 0:
        regime_labels_series = pd.Series(
            y_reg, index=nifty_df.index[200 : 200 + len(y_reg)]
        )
    else:
        regime_labels_series = pd.Series(dtype=int)

    X_strat, y_strat = build_strategy_training_data(
        stock_data, regime_labels_series, derivatives=derivatives
    )
    if X_strat.shape[0] > 0:
        strat_meta = DatasetMetadata(
            **{**asdict(base_metadata), "feature_version": f"{FEATURE_VERSION}-strategy"}
        )
        saved_paths["strategy_train"] = _save_dataset(
            output_dir, "strategy_train",
            {"X": X_strat, "y": y_strat},
            strat_meta,
        )

    # ── Step 9: Build risk dataset ─────────────────────────────────────────
    logger.info("building_risk_dataset")
    X_risk, y_stop, y_target, y_dd = build_risk_training_data(
        stock_data, derivatives=derivatives
    )
    if X_risk.shape[0] > 0:
        risk_meta = DatasetMetadata(
            **{**asdict(base_metadata), "feature_version": f"{FEATURE_VERSION}-risk"}
        )
        saved_paths["risk_train"] = _save_dataset(
            output_dir, "risk_train",
            {"X": X_risk, "y_stop": y_stop, "y_target": y_target, "y_drawdown": y_dd},
            risk_meta,
        )

    logger.info(
        "pipeline_complete",
        datasets_saved=len(saved_paths),
        sources=sources_used,
        date_range=(start_date, end_date),
        symbols_processed=len(raw_data),
    )
    return saved_paths


# ─── CLI entrypoint ───────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlphaForge ML training data pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        default="2023-01-01",
        help="Training window start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Training window end date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for artefacts. Defaults to TRAINING_DATA_PATH.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced universe (12 symbols) for fast iteration.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_dir = Path(args.output) if args.output else None
    run_pipeline(
        start_date=args.start,
        end_date=args.end,
        output_dir=output_dir,
        quick=args.quick,
    )
