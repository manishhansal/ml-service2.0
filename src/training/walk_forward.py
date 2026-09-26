"""
src.training.walk_forward — WalkForwardValidator (P0-007).

Anchored/rolling walk-forward out-of-sample validation. The history is divided
into >= 5 sequential windows. For each window:
    train on all data before the window (purged + embargoed),
    then evaluate on the window (true OOS — never seen in training).
Time only advances; the final test window is never reused for selection.

Per-window metrics: rank_ic (Spearman, PRIMARY), pearson_ic (diagnostic),
hit rate, net Sharpe (after cost), mean net return, max drawdown, turnover
proxy, sample count.

IC methodology (HIGH-4 fix):
    ``ic_mean`` in WalkForwardReport is now the MEAN OF PER-WINDOW SPEARMAN
    RANK IC, not the pooled Pearson IC that was previously reported.
    Spearman rank IC is the standard institutional metric: it is outlier-robust
    and does not require distributional assumptions.  The pooled Pearson IC is
    still computed and stored per-window as ``pearson_ic`` for diagnostic use.

    When symbol metadata (``symbols`` array aligned to rows) is provided,
    ``WalkForwardValidator`` additionally computes the per-timestamp cross-
    sectional Rank IC from ``reconciliation/ic.py`` and exposes it as
    ``xs_rank_ic_mean`` in the report — the most honest economic metric.

Per-fold normalization (CRITICAL-2 fix):
    When a ``FeatureNormalizer`` factory (``normalizer_factory``) is supplied
    to ``validate()``, the normalizer is fitted on the training fold and applied
    to both train and test folds before any model fitting or scoring.  This
    eliminates the look-ahead contamination that occurred when the normalizer
    was fitted on the full dataset before splitting.

    The normalizer fitted on the last (largest) fold is returned in the report
    as ``final_fold_normalizer`` so that the orchestrator can include it in the
    production model artifact.

Requirements: P0-007, Phase 31, 10_VALIDATION_FRAMEWORK.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.logging_config import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252


@dataclass
class WindowResult:
    """OOS metrics for a single walk-forward window."""

    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    # PRIMARY metric: Spearman rank IC of predictions vs continuous returns
    rank_ic: float
    # Diagnostic: Pearson IC (outlier-sensitive, kept for backward-compat logging)
    ic: float
    hit_rate: float
    net_sharpe: float
    mean_net_return: float
    max_drawdown: float
    turnover: float
    # Cross-sectional rank IC (populated when symbols are provided)
    xs_rank_ic: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class WalkForwardReport:
    """Aggregate walk-forward validation report."""

    n_windows: int
    windows: list[WindowResult] = field(default_factory=list)
    # PRIMARY: mean of per-window Spearman rank IC (HIGH-4 fix)
    ic_mean: float = 0.0
    ic_median: float = 0.0
    ic_worst: float = 0.0
    ic_std: float = 0.0
    positive_ic_fraction: float = 0.0
    # Pearson IC (diagnostic only — do not use for champion selection)
    pearson_ic_mean: float = 0.0
    # Cross-sectional rank IC mean (most honest economic IC)
    xs_rank_ic_mean: float = 0.0
    net_sharpe_mean: float = 0.0
    worst_drawdown: float = 0.0
    cost_bps: float = 10.0
    # Normalizer fitted on the final (largest) training fold.
    # Set only when normalizer_factory was passed to validate().
    # The orchestrator must include this in the production model artifact.
    final_fold_normalizer: Any = None  # FeatureNormalizer | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_windows": self.n_windows,
            "ic_mean": self.ic_mean,
            "ic_median": self.ic_median,
            "ic_worst": self.ic_worst,
            "ic_std": self.ic_std,
            "positive_ic_fraction": self.positive_ic_fraction,
            "pearson_ic_mean": self.pearson_ic_mean,
            "xs_rank_ic_mean": self.xs_rank_ic_mean,
            "net_sharpe_mean": self.net_sharpe_mean,
            "worst_drawdown": self.worst_drawdown,
            "cost_bps": self.cost_bps,
            "windows": [w.to_dict() for w in self.windows],
        }


class WalkForwardValidator:
    """
    Walk-forward out-of-sample validator with per-fold normalization.

    Usage (basic)::

        wf = WalkForwardValidator(n_windows=5, embargo_days=10, cost_bps=10)
        report = wf.validate(
            X, y_label, returns, timestamps,
            model_factory=lambda: build_estimator("lightgbm"),
        )

    Usage (with per-fold normalization)::

        from src.features.normalizer import FeatureNormalizer
        report = wf.validate(
            X_raw, y_label, returns, timestamps,
            model_factory=lambda: build_estimator("lightgbm"),
            normalizer_factory=lambda: FeatureNormalizer(),
        )
        # report.ic_mean is now Spearman rank IC (not Pearson)
        # report.final_fold_normalizer is the normalizer for the production artifact

    ``y_label`` is the (0/1) classification target used to fit; ``returns`` is
    the realized net-of-nothing forward return used to score IC and Sharpe.

    ``symbols`` (optional): array aligned to rows giving symbol identifier.
    When provided, per-timestamp cross-sectional Rank IC is also computed.
    """

    def __init__(
        self,
        n_windows: int = 5,
        embargo_days: int = 10,
        cost_bps: float = 10.0,
        anchored: bool = True,
    ) -> None:
        if n_windows < 5:
            raise ValueError("WalkForwardValidator requires n_windows >= 5 (Phase 31).")
        self.n_windows = n_windows
        self.embargo_days = embargo_days
        self.cost_bps = cost_bps
        self.anchored = anchored

    def validate(
        self,
        X: np.ndarray,
        y_label: np.ndarray,
        returns: np.ndarray,
        timestamps: pd.DatetimeIndex,
        model_factory: Callable[[], Any],
        normalizer_factory: Callable[[], Any] | None = None,
        symbols: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> WalkForwardReport:
        """
        Run walk-forward validation.

        Args:
            X:                  Feature matrix (raw, un-normalized).
            y_label:            Binary classification labels.
            returns:            Continuous realized returns (for IC/Sharpe scoring).
            timestamps:         UTC DatetimeIndex aligned to rows.
            model_factory:      Callable returning a fresh unfitted estimator.
            normalizer_factory: Optional callable returning a fresh FeatureNormalizer.
                                When supplied, a new normalizer is fitted on each
                                training fold and applied to both train and test —
                                eliminating look-ahead from pre-split normalization.
            symbols:            Optional array of symbol strings aligned to rows.
                                When supplied, per-timestamp cross-sectional Rank IC
                                is computed as an additional quality metric.
            feature_names:      Optional list of feature column names. When supplied
                                with normalizer_factory, the fitted normalizer uses
                                named columns matching inference-time DataFrames.

        Returns:
            WalkForwardReport with ``ic_mean`` = mean Spearman rank IC.
            ``final_fold_normalizer`` is set when normalizer_factory was supplied.
        """
        n = len(X)
        if not (n == len(y_label) == len(returns) == len(timestamps)):
            raise ValueError("X, y_label, returns, timestamps must be equal length.")

        order = np.argsort(timestamps.values)
        X = X[order]
        y_label = np.asarray(y_label)[order]
        returns = np.asarray(returns, dtype=float)[order]
        ts = timestamps[order]
        syms = np.asarray(symbols)[order] if symbols is not None else None
        # Column names for normalizer DataFrames — use named cols when available
        # so the fitted specs match inference-time string column names.
        _col_names: list | None = feature_names

        # Divide into (n_windows + 1) blocks: block 0 is the initial train seed.
        total_blocks = self.n_windows + 1
        edges = np.linspace(0, n, total_blocks + 1, dtype=int)
        cost = self.cost_bps / 10_000.0

        windows: list[WindowResult] = []
        last_normalizer: Any = None

        for w in range(1, total_blocks):
            test_lo, test_hi = edges[w], edges[w + 1]
            if test_hi - test_lo < 5:
                continue

            train_hi = test_lo
            train_lo = 0 if self.anchored else edges[w - 1]

            # Embargo: drop the last `embargo_days` of training before test start.
            test_start_date = ts[test_lo]
            if self.embargo_days > 0:
                cutoff = test_start_date - pd.Timedelta(days=self.embargo_days)
                train_mask = np.asarray(ts[train_lo:train_hi] <= cutoff)
                train_idx = np.arange(train_lo, train_hi)[train_mask]
            else:
                train_idx = np.arange(train_lo, train_hi)

            if len(train_idx) < 30:
                continue

            test_idx = np.arange(test_lo, test_hi)

            # ── Per-fold normalization (CRITICAL-2 fix) ───────────────────
            # Fit normalizer on THIS fold's training data only, then apply to
            # both train and test.  This prevents future test data from
            # contaminating the normalizer statistics.
            X_train_fold = X[train_idx]
            X_test_fold  = X[test_idx]

            fold_normalizer: Any = None
            if normalizer_factory is not None:
                try:
                    fold_normalizer = normalizer_factory()
                    X_train_fold_df = pd.DataFrame(X_train_fold, columns=_col_names)
                    fold_normalizer.fit(X_train_fold_df)
                    X_train_fold = fold_normalizer.transform(X_train_fold_df).to_numpy(dtype=float)
                    X_test_fold  = fold_normalizer.transform(
                        pd.DataFrame(X_test_fold, columns=_col_names)
                    ).to_numpy(dtype=float)
                    last_normalizer = fold_normalizer
                except Exception as exc:
                    logger.warning(
                        "walk_forward_normalizer_failed_skipping",
                        window=w, error=str(exc),
                    )
                    # Fall back to un-normalized data for this fold
                    X_train_fold = X[train_idx]
                    X_test_fold  = X[test_idx]
                    fold_normalizer = None

            model = model_factory()
            model.fit(X_train_fold, y_label[train_idx])
            preds = np.asarray(model.predict(X_test_fold), dtype=float)
            test_ret = returns[test_idx]

            # Cross-sectional Rank IC (if symbols provided)
            xs_ic = 0.0
            if syms is not None:
                xs_ic = self._cross_sectional_rank_ic(
                    preds, test_ret, ts[test_idx], syms[test_idx]
                )

            windows.append(
                self._score_window(
                    w - 1, preds, test_ret, y_label[test_idx],
                    ts[train_idx], ts[test_idx], len(train_idx), cost, xs_ic,
                )
            )

        report = self._aggregate(windows)
        report.final_fold_normalizer = last_normalizer
        return report

    # ── Scoring ────────────────────────────────────────────────────────────

    def _score_window(
        self,
        idx: int,
        preds: np.ndarray,
        test_ret: np.ndarray,
        test_label: np.ndarray,
        train_ts: pd.DatetimeIndex,
        test_ts: pd.DatetimeIndex,
        n_train: int,
        cost: float,
        xs_ic: float = 0.0,
    ) -> WindowResult:
        # PRIMARY: Spearman rank IC (outlier-robust, no distributional assumption)
        rank_ic = self._safe_corr(preds, test_ret, method="spearman")
        # Diagnostic: Pearson IC (kept for backward-compat logging only)
        pearson_ic = self._safe_corr(preds, test_ret, method="pearson")

        # Position = centered prediction (long above 0.5, short below).
        position = np.sign(preds - 0.5)
        hit = float(np.mean((position * np.sign(test_ret)) > 0)) if len(test_ret) else 0.0

        # Turnover proxy = mean absolute change in position.
        turnover = float(np.mean(np.abs(np.diff(position)))) if len(position) > 1 else 0.0

        gross = position * test_ret
        net = gross - cost * np.abs(position)
        mean_net = float(np.mean(net)) if len(net) else 0.0
        std_net = float(np.std(net))
        net_sharpe = (mean_net / std_net * np.sqrt(TRADING_DAYS)) if std_net > 1e-12 else 0.0

        max_dd = self._max_drawdown(net)

        return WindowResult(
            window_index=idx,
            train_start=str(train_ts.min().date()) if len(train_ts) else "",
            train_end=str(train_ts.max().date()) if len(train_ts) else "",
            test_start=str(test_ts.min().date()),
            test_end=str(test_ts.max().date()),
            n_train=n_train,
            n_test=len(test_ret),
            rank_ic=round(rank_ic, 6),
            ic=round(pearson_ic, 6),      # backward-compat field — Pearson diagnostic
            hit_rate=round(hit, 4),
            net_sharpe=round(net_sharpe, 4),
            mean_net_return=round(mean_net, 6),
            max_drawdown=round(max_dd, 6),
            turnover=round(turnover, 4),
            xs_rank_ic=round(xs_ic, 6),
        )

    def _aggregate(self, windows: list[WindowResult]) -> WalkForwardReport:
        report = WalkForwardReport(n_windows=len(windows), windows=windows, cost_bps=self.cost_bps)
        if not windows:
            return report

        # PRIMARY aggregation uses Spearman rank IC (HIGH-4 fix)
        rank_ics  = np.array([w.rank_ic  for w in windows])
        sharpes   = np.array([w.net_sharpe for w in windows])
        dds       = np.array([w.max_drawdown for w in windows])
        pearson_ics = np.array([w.ic for w in windows])
        xs_ics    = np.array([w.xs_rank_ic for w in windows])

        report.ic_mean             = round(float(np.mean(rank_ics)), 6)
        report.ic_median           = round(float(np.median(rank_ics)), 6)
        report.ic_worst            = round(float(np.min(rank_ics)), 6)
        report.ic_std              = round(float(np.std(rank_ics)), 6)
        report.positive_ic_fraction = round(float(np.mean(rank_ics > 0)), 4)
        report.pearson_ic_mean     = round(float(np.mean(pearson_ics)), 6)
        report.xs_rank_ic_mean     = round(float(np.mean(xs_ics)), 6)
        report.net_sharpe_mean     = round(float(np.mean(sharpes)), 4)
        report.worst_drawdown      = round(float(np.min(dds)), 6)

        logger.info(
            "walk_forward_complete",
            n_windows=len(windows),
            rank_ic_mean=report.ic_mean,
            pearson_ic_mean=report.pearson_ic_mean,
            xs_rank_ic_mean=report.xs_rank_ic_mean,
            net_sharpe_mean=report.net_sharpe_mean,
        )
        return report

    # ── Cross-sectional IC helper ──────────────────────────────────────────

    @staticmethod
    def _cross_sectional_rank_ic(
        preds: np.ndarray,
        returns: np.ndarray,
        timestamps: pd.DatetimeIndex,
        symbols: np.ndarray,
        min_symbols: int = 3,
    ) -> float:
        """Compute mean per-timestamp cross-sectional Spearman Rank IC.

        This is the economically honest IC: at each date T, rank all symbols
        by their model score, then measure rank correlation with their realized
        forward returns at T.

        Returns 0.0 when fewer than ``min_symbols`` unique symbols appear at
        any timestamp (insufficient cross-section).
        """
        df = pd.DataFrame({
            "ts":     timestamps,
            "symbol": symbols,
            "pred":   preds,
            "ret":    returns,
        })
        per_ts: list[float] = []
        for _, grp in df.groupby("ts"):
            if len(grp) < min_symbols:
                continue
            s = grp["pred"].to_numpy(float)
            r = grp["ret"].to_numpy(float)
            if np.std(s) < 1e-12 or np.std(r) < 1e-12:
                continue
            ic_val, _ = spearmanr(s, r)
            if np.isfinite(ic_val):
                per_ts.append(float(ic_val))
        return float(np.mean(per_ts)) if per_ts else 0.0

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
        if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        if method == "spearman":
            r, _ = spearmanr(a, b)
        else:
            r = np.corrcoef(a, b)[0, 1]
        return float(r) if np.isfinite(r) else 0.0

    @staticmethod
    def _max_drawdown(net_returns: np.ndarray) -> float:
        if len(net_returns) == 0:
            return 0.0
        equity = np.cumprod(1.0 + net_returns)
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        return float(np.min(dd))
