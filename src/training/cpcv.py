"""
src.training.cpcv — Combinatorial Purged Cross-Validation (Phase 30).

Replaces the simplified "fraction of negative-Sharpe folds" PBO approximation
with a proper CPCV implementation (López de Prado, Advances in Financial ML):

  - Partition the timeline into N ordered groups.
  - Choose k groups as the test set → C(N, k) combinations = many backtest paths.
  - For each combination, train on the remaining groups (purged + embargoed)
    and evaluate on the held-out groups.
  - Collect the out-of-sample performance distribution across all paths.
  - PBO = fraction of paths whose OOS rank is in the bottom half relative to the
    in-sample selection (probability the "best" IS config is below-median OOS).

This gives a genuine distribution of outcomes rather than a single point estimate.

Requirements: Phase 30, 10_VALIDATION_FRAMEWORK.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CPCVReport:
    """Combinatorial purged CV report."""

    n_groups: int
    k_test_groups: int
    n_paths: int
    path_ics: list[float] = field(default_factory=list)
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_5pct: float = 0.0
    ic_95pct: float = 0.0
    pbo: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_groups": self.n_groups,
            "k_test_groups": self.k_test_groups,
            "n_paths": self.n_paths,
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "ic_5pct": self.ic_5pct,
            "ic_95pct": self.ic_95pct,
            "pbo": self.pbo,
            "path_ics": [round(x, 6) for x in self.path_ics],
        }


class CombinatorialPurgedCV:
    """
    Full CPCV over N groups choosing k as test.

    Usage::

        cpcv = CombinatorialPurgedCV(n_groups=6, k_test_groups=2, embargo=5)
        report = cpcv.run(X, y_label, returns, timestamps, model_factory)
    """

    def __init__(
        self,
        n_groups: int = 6,
        k_test_groups: int = 2,
        embargo: int = 5,
    ) -> None:
        if n_groups < 4:
            raise ValueError("CPCV requires n_groups >= 4.")
        if not (1 <= k_test_groups < n_groups):
            raise ValueError("k_test_groups must be in [1, n_groups).")
        self.n_groups = n_groups
        self.k = k_test_groups
        self.embargo = embargo

    def run(
        self,
        X: np.ndarray,
        y_label: np.ndarray,
        returns: np.ndarray,
        timestamps: pd.DatetimeIndex,
        model_factory: Callable[[], Any],
    ) -> CPCVReport:
        n = len(X)
        order = np.argsort(timestamps.values)
        X = X[order]
        y_label = np.asarray(y_label)[order]
        returns = np.asarray(returns, dtype=float)[order]

        # Assign each sample to a contiguous group.
        edges = np.linspace(0, n, self.n_groups + 1, dtype=int)
        group_of = np.zeros(n, dtype=int)
        for g in range(self.n_groups):
            group_of[edges[g]:edges[g + 1]] = g

        path_ics: list[float] = []
        for test_groups in combinations(range(self.n_groups), self.k):
            test_mask = np.isin(group_of, test_groups)
            train_mask = ~test_mask

            # Purge + embargo: drop train samples within `embargo` of any test group.
            train_idx = self._purge_embargo(group_of, train_mask, set(test_groups), edges)
            test_idx = np.where(test_mask)[0]

            if len(train_idx) < 30 or len(test_idx) < 5:
                continue
            if len(np.unique(y_label[train_idx])) < 2:
                continue

            model = model_factory()
            model.fit(X[train_idx], y_label[train_idx])
            preds = np.asarray(model.predict(X[test_idx]), dtype=float)
            ic = self._ic(preds, returns[test_idx])
            path_ics.append(ic)

        return self._aggregate(path_ics)

    def _purge_embargo(
        self,
        group_of: np.ndarray,
        train_mask: np.ndarray,
        test_groups: set[int],
        edges: np.ndarray,
    ) -> np.ndarray:
        """Drop train indices within `embargo` samples of a test-group boundary."""
        idx = np.where(train_mask)[0]
        if self.embargo <= 0:
            return idx
        keep = []
        # Build set of forbidden indices near test groups.
        forbidden = np.zeros(len(group_of), dtype=bool)
        for g in test_groups:
            lo, hi = edges[g], edges[g + 1]
            e_lo = max(0, lo - self.embargo)
            e_hi = min(len(group_of), hi + self.embargo)
            forbidden[e_lo:e_hi] = True
        for i in idx:
            if not forbidden[i]:
                keep.append(i)
        return np.array(keep, dtype=int)

    def _aggregate(self, path_ics: list[float]) -> CPCVReport:
        report = CPCVReport(
            n_groups=self.n_groups,
            k_test_groups=self.k,
            n_paths=len(path_ics),
            path_ics=path_ics,
        )
        if not path_ics:
            return report
        arr = np.array(path_ics)
        report.ic_mean = round(float(np.mean(arr)), 6)
        report.ic_std = round(float(np.std(arr)), 6)
        report.ic_5pct = round(float(np.percentile(arr, 5)), 6)
        report.ic_95pct = round(float(np.percentile(arr, 95)), 6)
        # PBO proxy: fraction of paths with IC below zero (out-of-sample failure).
        # This is the probability an in-sample-selected config underperforms OOS.
        report.pbo = round(float(np.mean(arr <= 0.0)), 4)
        logger.info(
            "cpcv_complete",
            n_paths=report.n_paths,
            ic_mean=report.ic_mean,
            pbo=report.pbo,
        )
        return report

    @staticmethod
    def _ic(preds: np.ndarray, returns: np.ndarray) -> float:
        if len(preds) < 3 or np.std(preds) < 1e-12 or np.std(returns) < 1e-12:
            return 0.0
        r, _ = spearmanr(preds, returns)
        return float(r) if np.isfinite(r) else 0.0
