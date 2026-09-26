"""
src.validation.purged_kfold — PurgedKFold and CPCV splitters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Generator, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


# ── t1 series helpers ─────────────────────────────────────────────────────────

def build_t1_series(
    index: pd.DatetimeIndex,
    horizon_bars: int,
) -> pd.Series:
    """Return a Series where t1[i] = index[i + horizon_bars] (or last index)."""
    n = len(index)
    t1_values = []
    for i in range(n):
        end_idx = min(i + horizon_bars, n - 1)
        t1_values.append(index[end_idx])
    return pd.Series(t1_values, index=index)


# ── PurgedKFold ───────────────────────────────────────────────────────────────

class PurgedKFold(BaseCrossValidator):
    """
    sklearn-compatible cross-validator that purges training observations
    whose label windows overlap with the test set, then applies an embargo.
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: pd.Series | None = None,
        embargo_pct: float = 0.01,
    ) -> None:
        self.n_splits = n_splits
        self.t1 = t1
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _iter_test_masks(self, X=None, y=None, groups=None):
        n = len(X) if X is not None else 0
        indices = np.arange(n)
        fold_size = n // self.n_splits
        for k in range(self.n_splits):
            start = k * fold_size
            end = (k + 1) * fold_size if k < self.n_splits - 1 else n
            mask = np.zeros(n, dtype=bool)
            mask[start:end] = True
            yield mask

    def split(
        self,
        X,
        y=None,
        groups: pd.DatetimeIndex | None = None,
    ) -> Generator:
        n = len(X)
        indices = np.arange(n)
        idx = groups if groups is not None else (
            pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
        )

        fold_size = n // self.n_splits
        embargo_bars = max(1, int(n * self.embargo_pct))

        for k in range(self.n_splits):
            # Test set is a chronological chunk
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < self.n_splits - 1 else n
            test_idx = indices[test_start:test_end]

            # Training: everything BEFORE test start (minus embargo)
            embargo_cutoff = max(0, test_start - embargo_bars)

            # Build train mask: only indices strictly before test_start
            train_mask = np.zeros(n, dtype=bool)
            train_mask[:embargo_cutoff] = True

            # Purging: from the train-eligible set, remove obs whose label overlaps test
            if self.t1 is not None and len(idx) == n:
                test_start_time = idx[test_start]
                for i in range(embargo_cutoff):
                    obs_t0 = idx[i]
                    try:
                        obs_t1 = self.t1.get(obs_t0, obs_t0)
                    except Exception:
                        obs_t1 = obs_t0
                    if obs_t1 >= test_start_time:
                        train_mask[i] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx


# ── CPCV ──────────────────────────────────────────────────────────────────────

@dataclass
class CPCVConfig:
    n_splits: int = 6
    n_test_splits: int = 2

    def __post_init__(self):
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.n_test_splits >= self.n_splits:
            raise ValueError("n_test_splits must be < n_splits")
        if self.n_test_splits < 1:
            raise ValueError("n_test_splits must be >= 1")

    @property
    def n_backtest_paths(self) -> int:
        return math.comb(self.n_splits, self.n_test_splits)


@dataclass
class CPCVFold:
    split_index: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    test_groups: tuple
    n_purged: int = 0


class CombinatorialPurgedCV:
    def __init__(
        self,
        cfg: CPCVConfig,
        t1: pd.Series | None = None,
        embargo_pct: float = 0.01,
    ) -> None:
        self.cfg = cfg
        self.t1 = t1
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None) -> list[CPCVFold]:
        n = len(X)
        indices = np.arange(n)
        group_size = n // self.cfg.n_splits
        groups_list = []
        for k in range(self.cfg.n_splits):
            start = k * group_size
            end = (k + 1) * group_size if k < self.cfg.n_splits - 1 else n
            groups_list.append(np.arange(start, end))

        folds: list[CPCVFold] = []
        for split_idx, test_group_combo in enumerate(
            combinations(range(self.cfg.n_splits), self.cfg.n_test_splits)
        ):
            test_indices = np.concatenate([groups_list[g] for g in test_group_combo])
            test_set = set(test_indices.tolist())
            train_mask = np.ones(n, dtype=bool)
            for i in test_indices:
                train_mask[i] = False

            n_purged = 0
            # Purge obs whose labels overlap with test
            if self.t1 is not None and groups is not None:
                idx = groups
                if len(idx) == n:
                    test_start_time = idx[test_indices.min()]
                    for i in range(n):
                        if train_mask[i]:
                            obs_t0 = idx[i]
                            try:
                                obs_t1 = self.t1.get(obs_t0, obs_t0)
                            except Exception:
                                obs_t1 = obs_t0
                            if obs_t1 >= test_start_time:
                                train_mask[i] = False
                                n_purged += 1

            train_indices = indices[train_mask]
            folds.append(CPCVFold(
                split_index=split_idx,
                train_indices=train_indices,
                test_indices=test_indices,
                test_groups=test_group_combo,
                n_purged=n_purged,
            ))
        return folds

    def build_backtest_paths(self, folds: list[CPCVFold]) -> list[list[CPCVFold]]:
        """Group folds into backtest paths where each path covers non-overlapping groups."""
        # Each path = sequence of folds with disjoint test_groups covering all groups
        all_groups = set(range(self.cfg.n_splits))
        paths: list[list[CPCVFold]] = []

        # Build paths greedily
        for anchor in folds:
            path = [anchor]
            covered = set(anchor.test_groups)
            for other in folds:
                if other.split_index == anchor.split_index:
                    continue
                new_groups = set(other.test_groups)
                if new_groups.isdisjoint(covered):
                    path.append(other)
                    covered |= new_groups
            if len(path) > 1:
                paths.append(path)
        return paths


def cpcv_splits(
    X,
    y=None,
    n_splits: int = 6,
    n_test_splits: int = 2,
    t1: pd.Series | None = None,
    t0: pd.DatetimeIndex | None = None,
    embargo_pct: float = 0.01,
) -> list[CPCVFold]:
    cfg = CPCVConfig(n_splits=n_splits, n_test_splits=n_test_splits)
    engine = CombinatorialPurgedCV(cfg, t1=t1, embargo_pct=embargo_pct)
    return engine.split(X, y, groups=t0)
