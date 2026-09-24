"""
src.analytics.cross_sectional — genuine cross-sectional alpha research
(mandate §17-§37, §41, §68-§69, §81-§83).

The existing walk-forward / CPCV validators compute a POOLED information
coefficient (predictions vs realized return across all rows, mixing symbols and
dates). That answers a *time-series* question. This module answers the
different, higher-value question the mandate cares about:

    At each timestamp T, rank N symbols by a model score. Do the top-ranked
    names out-perform the bottom-ranked names OUT OF SAMPLE, after realistic
    costs and turnover?

Everything here is PIT-safe and leakage-guarded:

  - Features at T use only bars up to and including T (FeatureFactory is causal).
  - Labels are strictly forward (T -> T+h) and are computed per timestamp.
  - Beta / benchmark estimates use ONLY history before T (rolling, shifted).
  - Walk-forward splits are by DATE with a purge + embargo gap; the model never
    sees the OOS window or the embargo band.
  - Cross-sectional IC is computed WITHIN each timestamp (Spearman rank IC of
    score vs realized forward return across the symbols present at T), then
    averaged over OOS timestamps — this is the true cross-sectional IC.

No synthetic data. No zero-filling of missing bars. Missing symbol/timestamp
cells are dropped, never imputed to zero.

Label families (mandate §18):
    A RAW      forward return r_{i,T->T+h}
    B EXCESS   r_i - r_market            (market = cross-sectional mean return)
    C RESIDUAL r_i - beta_i * r_market   (beta from history strictly before T)
    D RANK     cross-sectional percentile rank of the residual return

The model regresses features -> a training target; the OOS SCORE is then ranked
cross-sectionally and evaluated against the realized forward return. A negative
or absent edge is a valid, reported outcome.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from src.analytics.independent_metrics import pearson_ic, rank_ic
from src.features.factory import FeatureFactory

CrossSectionalLabel = Literal["raw", "excess", "residual", "rank"]

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------


@dataclass
class PanelConfig:
    """Configuration for cross-sectional panel construction."""

    horizon: int = 1                 # forward bars for the label
    beta_window: int = 60            # bars of history for rolling beta (residual label)
    vol_window: int = 20             # bars for realized-vol normalisation
    min_symbols_per_ts: int = 10     # drop timestamps with too few names for a cross-section
    cost_bps: float = 10.0           # round-trip transaction cost (basis points)


def build_cross_sectional_panel(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    config: PanelConfig,
    feature_factory: FeatureFactory | None = None,
) -> pd.DataFrame:
    """Build a long-format cross-sectional panel indexed by (timestamp, symbol).

    Returns a DataFrame with columns:
        <24 base features>, <cross-sectional relative features>,
        fwd_return_raw, fwd_return_excess, fwd_return_residual,
        fwd_return_rank, fwd_return_volnorm, symbol
    indexed by a MultiIndex (ts, symbol). All labels are strictly forward and
    PIT-safe; beta uses only pre-T history.

    Every returned row satisfies: features_as_of == T, label window == (T, T+h].
    """
    ff = feature_factory or FeatureFactory()
    frames: list[pd.DataFrame] = []

    # ── 1. Per-symbol causal features + strictly-forward raw return ────────
    for symbol, ohlcv in ohlcv_by_symbol.items():
        if len(ohlcv) < max(config.beta_window, 60) + config.horizon + 5:
            continue
        df = ohlcv.sort_index()
        close = df["close"].astype(float)
        open_ = df["open"].astype(float) if "open" in df.columns else close

        features, _avail = ff.build(df)  # causal: only uses bars <= each row
        fwd_raw = (close.shift(-config.horizon) - close) / close  # (T -> T+h], forward

        # per-symbol realized vol (causal, for vol-normalised label)
        realized_vol = close.pct_change().rolling(config.vol_window).std()

        panel = features.copy()
        panel["symbol"] = symbol
        panel["_close"] = close
        panel["_open"] = open_  # for next-open-execution backtest (§32)
        panel["fwd_return_raw"] = fwd_raw
        panel["_ret_1"] = close.pct_change()  # contemporaneous 1-bar return (for beta)
        panel["_realized_vol"] = realized_vol
        frames.append(panel)

    if not frames:
        raise ValueError("cross_sectional: no symbols produced usable feature rows.")

    long = pd.concat(frames)
    long.index.name = "ts"
    long = long.reset_index().set_index(["ts", "symbol"]).sort_index()

    # ── 2. Cross-sectional market return per timestamp (equal-weight mean) ──
    # r_market,t is the contemporaneous cross-sectional mean 1-bar return.
    ret_1 = long["_ret_1"]
    market_ret_1 = ret_1.groupby(level="ts").transform("mean")
    long["_market_ret_1"] = market_ret_1

    # Forward market return over the SAME horizon (equal-weight mean of fwd_raw).
    fwd_market = long["fwd_return_raw"].groupby(level="ts").transform("mean")
    long["_fwd_market"] = fwd_market

    # ── 3. Rolling beta per symbol using ONLY history before T (PIT-safe) ──
    # beta_i,T = cov(r_i, r_mkt) / var(r_mkt) over a trailing window, then
    # SHIFTED by 1 so the beta used at T excludes T's own return.
    betas: dict[str, pd.Series] = {}
    for symbol, grp in long.groupby(level="symbol"):
        s = grp.reset_index(level="symbol", drop=True)
        r_i = s["_ret_1"]
        r_m = s["_market_ret_1"]
        cov = r_i.rolling(config.beta_window).cov(r_m)
        var = r_m.rolling(config.beta_window).var()
        beta = (cov / var).shift(1)  # strictly pre-T
        beta = beta.clip(-4.0, 4.0)  # guard against degenerate estimates
        betas[symbol] = beta
    beta_series = pd.concat(
        {sym: b for sym, b in betas.items()}, names=["symbol", "ts"]
    ).reorder_levels(["ts", "symbol"]).sort_index()
    long["_beta"] = beta_series

    # ── 4. Cross-sectional labels (mandate §18) ────────────────────────────
    long["fwd_return_excess"] = long["fwd_return_raw"] - long["_fwd_market"]
    long["fwd_return_residual"] = long["fwd_return_raw"] - long["_beta"] * long["_fwd_market"]
    # Volatility-normalised residual (label E).
    long["fwd_return_volnorm"] = long["fwd_return_residual"] / long["_realized_vol"]
    long["fwd_return_volnorm"] = long["fwd_return_volnorm"].replace([np.inf, -np.inf], np.nan)
    # Cross-sectional percentile rank of the residual (label D), in [0, 1].
    long["fwd_return_rank"] = (
        long["fwd_return_residual"].groupby(level="ts").rank(pct=True)
    )

    # ── 5. Cross-sectional RELATIVE features (mandate §21) ─────────────────
    # These are PIT-safe: computed from contemporaneous (<=T) feature values
    # across the cross-section, not from any forward information.
    long["xs_ret5_demean"] = long["ret_5"] - long["ret_5"].groupby(level="ts").transform("mean")
    long["xs_ret20_demean"] = long["ret_20"] - long["ret_20"].groupby(level="ts").transform("mean")
    long["xs_ret5_rank"] = long["ret_5"].groupby(level="ts").rank(pct=True)
    long["xs_ret20_rank"] = long["ret_20"].groupby(level="ts").rank(pct=True)
    _vol20_med = long["vol_20"].groupby(level="ts").transform("median")
    long["xs_relvol"] = long["vol_20"] / _vol20_med
    _ret5_std = long["ret_5"].groupby(level="ts").transform("std")
    _ret5_mean = long["ret_5"].groupby(level="ts").transform("mean")
    long["xs_ret5_z"] = (long["ret_5"] - _ret5_mean) / _ret5_std
    long = long.replace([np.inf, -np.inf], np.nan)

    return long


# ---------------------------------------------------------------------------
# Cross-sectional walk-forward evaluation
# ---------------------------------------------------------------------------


@dataclass
class WindowIC:
    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train_rows: int
    n_test_ts: int
    mean_xs_ic: float
    mean_xs_rank_ic: float
    top_bottom_spread: float
    long_only_top_mean: float
    turnover: float
    net_spread: float


@dataclass
class CrossSectionalResult:
    label: str
    model: str
    feature_set: str
    horizon: int
    n_windows: int
    n_oos_timestamps: int
    n_oos_predictions: int
    # PRIMARY decision metric: cross-sectional Spearman RANK IC. Rank IC is the
    # honest cross-sectional metric — robust to the fat-tailed daily returns that
    # make Pearson IC explode on a few outliers (mandate §23, §40, §81).
    mean_rank_ic: float = 0.0
    median_rank_ic: float = 0.0
    worst_rank_ic: float = 0.0
    best_rank_ic: float = 0.0
    rank_ic_std: float = 0.0
    positive_window_fraction: float = 0.0
    # SECONDARY / diagnostic: Pearson IC (outlier-sensitive — NOT the decision
    # metric). A large gap vs rank IC flags outlier domination.
    mean_pearson_ic: float = 0.0
    pearson_vs_rank_gap: float = 0.0
    outlier_dominated: bool = False
    # Portfolio diagnostics. Spread uses the MEDIAN daily top-bottom decile
    # spread (robust) alongside the mean.
    mean_top_bottom_spread: float = 0.0
    median_top_bottom_spread: float = 0.0
    mean_long_only_top: float = 0.0
    mean_turnover: float = 0.0
    gross_spread_annual: float = 0.0
    net_spread_annual: float = 0.0
    net_sharpe_ls: float = 0.0
    net_sharpe_long_only: float = 0.0
    cost_breakeven_bps: float = 0.0
    windows: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d


class CrossSectionalValidator:
    """Walk-forward cross-sectional evaluator with per-timestamp IC + portfolios.

    Splits the timeline into ``n_windows`` sequential OOS blocks by DATE. For
    each block: fit on all data strictly before the block (minus an embargo
    gap), predict the training target, then evaluate the OOS SCORE
    cross-sectionally within each timestamp.
    """

    def __init__(
        self,
        n_windows: int = 6,
        embargo_bars: int = 5,
        cost_bps: float = 10.0,
        decile: float = 0.1,
        min_symbols_per_ts: int = 10,
        periods_per_year: int = TRADING_DAYS,
    ) -> None:
        self.n_windows = n_windows
        self.embargo_bars = embargo_bars
        self.cost_bps = cost_bps
        self.decile = decile
        self.min_symbols_per_ts = min_symbols_per_ts
        self.periods_per_year = periods_per_year

    def evaluate(
        self,
        panel: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        realized_col: str,
        model_factory: Callable[[], Any],
        model_name: str,
        feature_set_name: str,
        horizon: int,
    ) -> tuple[CrossSectionalResult, pd.DataFrame]:
        """Run the walk-forward cross-sectional evaluation.

        Args:
            panel: (ts, symbol)-indexed panel from ``build_cross_sectional_panel``.
            feature_cols: feature columns to use as the model input matrix.
            label_col: the TRAINING target column (e.g. fwd_return_residual).
            realized_col: the column used to SCORE OOS performance — always the
                economically meaningful forward return (raw for portfolio P&L,
                residual for pure alpha IC). Kept separate from the training
                target so the evaluation target cannot be gamed.
            model_factory: zero-arg callable returning a fresh estimator.
            model_name / feature_set_name / horizon: bookkeeping.

        Returns:
            (CrossSectionalResult, oos_predictions_df) where the predictions
            frame is the persisted OOS prediction store (mandate §50).
        """
        needed = set(feature_cols) | {label_col, realized_col}
        clean = panel.dropna(subset=list(needed)).copy()

        # Drop thin cross-sections.
        ts_counts = clean.groupby(level="ts").size()
        keep_ts = ts_counts[ts_counts >= self.min_symbols_per_ts].index
        clean = clean[clean.index.get_level_values("ts").isin(keep_ts)]
        if clean.empty:
            raise ValueError("cross_sectional: no usable rows after cleaning.")

        unique_ts = np.array(sorted(clean.index.get_level_values("ts").unique()))
        n_ts = len(unique_ts)
        if n_ts < (self.n_windows + 1) * 3:
            # not enough distinct timestamps for the requested windows
            self.n_windows = max(2, n_ts // 6)

        edges = np.linspace(0, n_ts, self.n_windows + 2, dtype=int)
        cost = self.cost_bps / 10_000.0

        windows: list[WindowIC] = []
        all_preds: list[pd.DataFrame] = []
        # portfolio return series across all OOS timestamps (for Sharpe)
        ls_net_series: list[float] = []
        long_only_net_series: list[float] = []
        prev_long: set = set()
        prev_short: set = set()

        for w in range(1, self.n_windows + 1):
            test_lo, test_hi = edges[w], edges[w + 1]
            train_hi = test_lo
            if train_hi < 3 or test_hi - test_lo < 1:
                continue

            # Embargo: drop the last `embargo_bars` timestamps before the test block.
            train_ts = unique_ts[: max(0, train_hi - self.embargo_bars)]
            test_ts = unique_ts[test_lo:test_hi]
            if len(train_ts) < 3 or len(test_ts) < 1:
                continue

            train_mask = clean.index.get_level_values("ts").isin(train_ts)
            test_mask = clean.index.get_level_values("ts").isin(test_ts)
            train = clean[train_mask]
            test = clean[test_mask]
            if train.empty or test.empty:
                continue

            X_train = train[feature_cols].to_numpy(dtype=float)
            y_train = train[label_col].to_numpy(dtype=float)
            # Binary label for classifiers: above-cross-sectional-median target.
            y_train_bin = (y_train > np.median(y_train)).astype(float)

            model = model_factory()
            try:
                # Classifiers expect a binary target; regressors use the raw target.
                if _is_classifier(model):
                    model.fit(X_train, y_train_bin)
                else:
                    model.fit(X_train, y_train)
            except Exception:
                continue

            X_test = test[feature_cols].to_numpy(dtype=float)
            score = np.asarray(model.predict(X_test), dtype=float)

            test = test.copy()
            test["_score"] = score

            (
                win_ic,
                win_rank_ic,
                win_spread,
                win_long_only,
                win_turnover,
                win_net_spread,
                ts_ls_net,
                ts_lo_net,
                prev_long,
                prev_short,
                preds_df,
            ) = self._score_window_by_ts(
                test, realized_col, cost, prev_long, prev_short, w - 1,
                model_name, label_col, feature_set_name, horizon,
            )

            ls_net_series.extend(ts_ls_net)
            long_only_net_series.extend(ts_lo_net)
            all_preds.append(preds_df)

            windows.append(
                WindowIC(
                    window_index=w - 1,
                    train_start=str(pd.Timestamp(train_ts.min()).date()),
                    train_end=str(pd.Timestamp(train_ts.max()).date()),
                    test_start=str(pd.Timestamp(test_ts.min()).date()),
                    test_end=str(pd.Timestamp(test_ts.max()).date()),
                    n_train_rows=len(train),
                    n_test_ts=len(test_ts),
                    mean_xs_ic=round(win_ic, 6),
                    mean_xs_rank_ic=round(win_rank_ic, 6),
                    top_bottom_spread=round(win_spread, 6),
                    long_only_top_mean=round(win_long_only, 6),
                    turnover=round(win_turnover, 4),
                    net_spread=round(win_net_spread, 6),
                )
            )

        result = self._aggregate(
            windows, ls_net_series, long_only_net_series,
            label_col, model_name, feature_set_name, horizon,
        )
        preds = pd.concat(all_preds) if all_preds else pd.DataFrame()
        return result, preds

    def _score_window_by_ts(
        self, test, realized_col, cost, prev_long, prev_short, widx,
        model_name, label_col, feature_set_name, horizon,
    ):
        """Score one OOS window, computing metrics WITHIN each timestamp."""
        ics, rank_ics, spreads, long_onlys, turnovers, net_spreads = [], [], [], [], [], []
        ts_ls_net: list[float] = []
        ts_lo_net: list[float] = []
        pred_rows: list[dict] = []

        for ts, grp in test.groupby(level="ts"):
            score = grp["_score"].to_numpy(dtype=float)
            realized = grp[realized_col].to_numpy(dtype=float)
            symbols = grp.index.get_level_values("symbol").tolist()
            n = len(score)
            if n < self.min_symbols_per_ts or np.std(score) < 1e-12:
                continue

            # Per-timestamp cross-sectional IC (this is the true xs IC).
            ics.append(pearson_ic(score.tolist(), realized.tolist()))
            rank_ics.append(rank_ic(score.tolist(), realized.tolist()))

            # Decile buckets by score.
            k = max(1, int(round(n * self.decile)))
            order = np.argsort(score)  # ascending
            bottom_idx = order[:k]
            top_idx = order[-k:]
            top_ret = float(np.mean(realized[top_idx]))
            bottom_ret = float(np.mean(realized[bottom_idx]))
            spread = top_ret - bottom_ret
            spreads.append(spread)
            long_onlys.append(top_ret)

            top_syms = {symbols[i] for i in top_idx}
            bottom_syms = {symbols[i] for i in bottom_idx}

            # Turnover vs previous timestamp's book (fraction of names changed).
            long_turn = _turnover(prev_long, top_syms)
            short_turn = _turnover(prev_short, bottom_syms)
            ts_turnover = 0.5 * (long_turn + short_turn)
            turnovers.append(ts_turnover)

            # Net returns: long-short and long-only, charged turnover*cost.
            ls_gross = spread
            ls_net = ls_gross - cost * (long_turn + short_turn)
            net_spreads.append(ls_net)
            ts_ls_net.append(ls_net)

            lo_net = top_ret - cost * long_turn
            ts_lo_net.append(lo_net)

            prev_long, prev_short = top_syms, bottom_syms

            # Persist OOS predictions (mandate §50).
            for i in range(n):
                pred_rows.append(
                    {
                        "ts": pd.Timestamp(ts).isoformat(),
                        "symbol": symbols[i],
                        "window": widx,
                        "model": model_name,
                        "label": label_col,
                        "feature_set": feature_set_name,
                        "horizon": horizon,
                        "score": float(score[i]),
                        "realized": float(realized[i]),
                        "bucket": (
                            "TOP" if i in set(top_idx.tolist())
                            else "BOTTOM" if i in set(bottom_idx.tolist())
                            else "MID"
                        ),
                    }
                )

        preds_df = pd.DataFrame(pred_rows)
        return (
            _safe_mean(ics), _safe_mean(rank_ics), _safe_mean(spreads),
            _safe_mean(long_onlys), _safe_mean(turnovers), _safe_mean(net_spreads),
            ts_ls_net, ts_lo_net, prev_long, prev_short, preds_df,
        )

    def _aggregate(
        self, windows, ls_net_series, long_only_net_series,
        label_col, model_name, feature_set_name, horizon,
    ) -> CrossSectionalResult:
        n_pred = 0
        res = CrossSectionalResult(
            label=label_col, model=model_name, feature_set=feature_set_name,
            horizon=horizon, n_windows=len(windows), n_oos_timestamps=0,
            n_oos_predictions=0, windows=[w.__dict__ for w in windows],
        )
        if not windows:
            return res

        pearson_ics = np.array([w.mean_xs_ic for w in windows], dtype=float)
        rank_ics = np.array([w.mean_xs_rank_ic for w in windows], dtype=float)
        spreads = np.array([w.top_bottom_spread for w in windows], dtype=float)
        long_onlys = np.array([w.long_only_top_mean for w in windows], dtype=float)
        turns = np.array([w.turnover for w in windows], dtype=float)

        res.n_oos_timestamps = int(sum(w.n_test_ts for w in windows))

        # ── PRIMARY: cross-sectional RANK IC (robust) ─────────────────────
        res.mean_rank_ic = round(float(np.mean(rank_ics)), 6)
        res.median_rank_ic = round(float(np.median(rank_ics)), 6)
        res.worst_rank_ic = round(float(np.min(rank_ics)), 6)
        res.best_rank_ic = round(float(np.max(rank_ics)), 6)
        res.rank_ic_std = round(float(np.std(rank_ics)), 6)
        res.positive_window_fraction = round(float(np.mean(rank_ics > 0)), 4)

        # ── SECONDARY: Pearson IC (outlier-sensitive diagnostic) ──────────
        res.mean_pearson_ic = round(float(np.mean(pearson_ics)), 6)
        res.pearson_vs_rank_gap = round(
            abs(res.mean_pearson_ic - res.mean_rank_ic), 6
        )
        # A large Pearson-vs-rank gap means the Pearson IC is inflated by a few
        # extreme daily returns and is NOT evidence of a broad monotone edge.
        res.outlier_dominated = bool(
            abs(res.mean_pearson_ic) > 0.15
            and res.pearson_vs_rank_gap > 2.0 * max(abs(res.mean_rank_ic), 1e-9)
        )

        res.mean_top_bottom_spread = round(float(np.mean(spreads)), 6)
        res.median_top_bottom_spread = round(float(np.median(spreads)), 6)
        res.mean_long_only_top = round(float(np.mean(long_onlys)), 6)
        res.mean_turnover = round(float(np.mean(turns)), 4)

        # Annualised gross/net long-short spread (per-timestamp mean * periods).
        periods = self.periods_per_year / max(1, horizon)
        gross_series = np.array([w.top_bottom_spread for w in windows], dtype=float)
        res.gross_spread_annual = round(float(np.mean(gross_series)) * periods, 6)

        ls = np.array(ls_net_series, dtype=float)
        lo = np.array(long_only_net_series, dtype=float)
        res.n_oos_predictions = int(len(ls))
        if len(ls) > 1 and np.std(ls) > 1e-12:
            res.net_sharpe_ls = round(
                float(np.mean(ls) / np.std(ls)) * math.sqrt(periods), 4
            )
            res.net_spread_annual = round(float(np.mean(ls)) * periods, 6)
        if len(lo) > 1 and np.std(lo) > 1e-12:
            res.net_sharpe_long_only = round(
                float(np.mean(lo) / np.std(lo)) * math.sqrt(periods), 4
            )

        # Cost break-even (mandate §84): gross per-timestamp spread expressed in
        # bps of round-trip cost that would zero out the net edge.
        mean_gross = float(np.mean(spreads))
        mean_turn = float(np.mean(turns)) if np.mean(turns) > 0 else 1.0
        # net = gross - (cost_frac) * (2*turn)  -> breakeven cost_frac = gross / (2*turn)
        breakeven_frac = mean_gross / (2.0 * mean_turn) if mean_turn > 0 else 0.0
        res.cost_breakeven_bps = round(breakeven_frac * 10_000.0, 2)
        return res


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _turnover(prev: set, cur: set) -> float:
    """Fraction of the book that changed between two consecutive rebalances."""
    if not cur:
        return 0.0
    if not prev:
        return 1.0
    changed = len(cur.symmetric_difference(prev)) / (2.0 * len(cur))
    return float(min(1.0, changed))


def _safe_mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _is_classifier(model: Any) -> bool:
    """Heuristic: a model is a classifier if it exposes predict_proba AND its
    predict output is expected to be a probability. Our baselines/advanced
    models all expose predict() returning a score in [0,1] for classifiers and
    a real-valued target for regressors; we detect by class name."""
    name = type(model).__name__.lower()
    return any(k in name for k in ("logistic", "lightgbm", "xgboost", "catboost", "classifier"))
