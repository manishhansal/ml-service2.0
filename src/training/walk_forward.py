"""
src.training.walk_forward — WalkForwardValidator (P0-007).

Anchored/rolling walk-forward out-of-sample validation. The history is divided
into >= 5 sequential windows. For each window:
    train on all data before the window (purged + embargoed),
    then evaluate on the window (true OOS — never seen in training).
Time only advances; the final test window is never reused for selection.

Per-window metrics: IC (Spearman), rank IC, hit rate, net Sharpe (after cost),
mean net return, max drawdown, turnover proxy, sample count.

Aggregate: mean/median/worst IC, fraction of windows with positive IC,
mean net Sharpe, worst-window drawdown.

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
    ic: float
    rank_ic: float
    hit_rate: float
    net_sharpe: float
    mean_net_return: float
    max_drawdown: float
    turnover: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class WalkForwardReport:
    """Aggregate walk-forward validation report."""

    n_windows: int
    windows: list[WindowResult] = field(default_factory=list)
    ic_mean: float = 0.0
    ic_median: float = 0.0
    ic_worst: float = 0.0
    ic_std: float = 0.0
    positive_ic_fraction: float = 0.0
    net_sharpe_mean: float = 0.0
    worst_drawdown: float = 0.0
    cost_bps: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_windows": self.n_windows,
            "ic_mean": self.ic_mean,
            "ic_median": self.ic_median,
            "ic_worst": self.ic_worst,
            "ic_std": self.ic_std,
            "positive_ic_fraction": self.positive_ic_fraction,
            "net_sharpe_mean": self.net_sharpe_mean,
            "worst_drawdown": self.worst_drawdown,
            "cost_bps": self.cost_bps,
            "windows": [w.to_dict() for w in self.windows],
        }


class WalkForwardValidator:
    """
    Walk-forward out-of-sample validator.

    Usage::

        wf = WalkForwardValidator(n_windows=5, embargo_days=10, cost_bps=10)
        report = wf.validate(
            X, y_label, returns, timestamps,
            model_factory=lambda: build_estimator("lightgbm"),
        )

    ``y_label`` is the (0/1) classification target used to fit; ``returns`` is
    the realized net-of-nothing forward return used to score IC and Sharpe.
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
    ) -> WalkForwardReport:
        n = len(X)
        if not (n == len(y_label) == len(returns) == len(timestamps)):
            raise ValueError("X, y_label, returns, timestamps must be equal length.")

        order = np.argsort(timestamps.values)
        X = X[order]
        y_label = np.asarray(y_label)[order]
        returns = np.asarray(returns, dtype=float)[order]
        ts = timestamps[order]

        # Divide into (n_windows + 1) blocks: block 0 is the initial train seed.
        total_blocks = self.n_windows + 1
        edges = np.linspace(0, n, total_blocks + 1, dtype=int)
        cost = self.cost_bps / 10_000.0

        windows: list[WindowResult] = []
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

            model = model_factory()
            model.fit(X[train_idx], y_label[train_idx])
            preds = np.asarray(model.predict(X[test_idx]), dtype=float)
            test_ret = returns[test_idx]

            windows.append(
                self._score_window(
                    w - 1, preds, test_ret, y_label[test_idx],
                    ts[train_idx], ts[test_idx], len(train_idx), cost,
                )
            )

        return self._aggregate(windows)

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
    ) -> WindowResult:
        # IC / rank IC between prediction and realized forward return.
        ic = self._safe_corr(preds, test_ret, method="pearson")
        rank_ic = self._safe_corr(preds, test_ret, method="spearman")

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
            ic=round(ic, 6),
            rank_ic=round(rank_ic, 6),
            hit_rate=round(hit, 4),
            net_sharpe=round(net_sharpe, 4),
            mean_net_return=round(mean_net, 6),
            max_drawdown=round(max_dd, 6),
            turnover=round(turnover, 4),
        )

    def _aggregate(self, windows: list[WindowResult]) -> WalkForwardReport:
        report = WalkForwardReport(n_windows=len(windows), windows=windows, cost_bps=self.cost_bps)
        if not windows:
            return report
        ics = np.array([w.ic for w in windows])
        sharpes = np.array([w.net_sharpe for w in windows])
        dds = np.array([w.max_drawdown for w in windows])
        report.ic_mean = round(float(np.mean(ics)), 6)
        report.ic_median = round(float(np.median(ics)), 6)
        report.ic_worst = round(float(np.min(ics)), 6)
        report.ic_std = round(float(np.std(ics)), 6)
        report.positive_ic_fraction = round(float(np.mean(ics > 0)), 4)
        report.net_sharpe_mean = round(float(np.mean(sharpes)), 4)
        report.worst_drawdown = round(float(np.min(dds)), 6)
        logger.info(
            "walk_forward_complete",
            n_windows=len(windows),
            ic_mean=report.ic_mean,
            net_sharpe_mean=report.net_sharpe_mean,
        )
        return report

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
