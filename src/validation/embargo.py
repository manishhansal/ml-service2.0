"""
src.validation.embargo — Embargo period utilities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd


class EmbargoUnit(str, Enum):
    BARS = "bars"
    MINUTES = "minutes"
    DAYS = "days"


@dataclass
class EmbargoConfig:
    size: int          # number of units
    unit: EmbargoUnit = EmbargoUnit.BARS
    bars_per_unit: float = 1.0   # conversion factor

    def __post_init__(self):
        if self.size < 0:
            raise ValueError(f"Embargo size must be >= 0, got {self.size}")

    @classmethod
    def bars(cls, n: int) -> "EmbargoConfig":
        return cls(size=n, unit=EmbargoUnit.BARS, bars_per_unit=1.0)

    @classmethod
    def minutes(cls, n: int, bars_per_minute: float = 1.0) -> "EmbargoConfig":
        return cls(size=n, unit=EmbargoUnit.MINUTES, bars_per_unit=bars_per_minute)

    @classmethod
    def days(cls, n: int, bars_per_day: float = 1.0) -> "EmbargoConfig":
        return cls(size=n, unit=EmbargoUnit.DAYS, bars_per_unit=bars_per_day)

    def to_bars(self) -> int:
        return max(0, int(math.ceil(self.size * self.bars_per_unit)))


class EmbargoApplier:
    def __init__(self, cfg: EmbargoConfig) -> None:
        self.cfg = cfg

    def apply_to_indices(
        self,
        train: Iterable[int],
        candidates: Iterable[int],
    ) -> list[int]:
        """Remove candidates within `embargo_bars` of the last training index."""
        n_bars = self.cfg.to_bars()
        train_list = list(train)
        cand_list = list(candidates)
        if not train_list or n_bars == 0:
            return cand_list
        last_train = max(train_list)
        cutoff = last_train + n_bars
        return [c for c in cand_list if c > cutoff]

    def apply_to_fold(
        self,
        train: Iterable[int],
        val: Iterable[int],
        test: Iterable[int],
    ) -> tuple[list[int], list[int], list[int]]:
        """Apply embargo between train→val and val→test."""
        train_list = list(train)
        val_list = list(val)
        test_list = list(test)

        n_bars = self.cfg.to_bars()
        purged_val = self.apply_to_indices(train_list, val_list)
        purged_test = self.apply_to_indices(val_list, test_list) if val_list else test_list
        return train_list, purged_val, purged_test


def purge_train_indices_by_t1(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    index: pd.DatetimeIndex,
    t1: pd.Series,
) -> list[int]:
    """Remove training indices whose label end time >= val start time."""
    if len(val_idx) == 0:
        return train_idx.tolist()
    val_start_time = index[val_idx[0]]
    purged = []
    for i in train_idx:
        obs_t0 = index[i]
        try:
            obs_t1 = t1.get(obs_t0, obs_t0)
        except Exception:
            obs_t1 = obs_t0
        if obs_t1 < val_start_time:
            purged.append(int(i))
    return purged


def compute_embargo_times(
    index: pd.DatetimeIndex,
    pct: float = 0.01,
) -> pd.Series:
    """Return embargo end times for each observation (pct of series length)."""
    n = len(index)
    embargo_bars = max(1, int(n * pct))
    result = {}
    for i, t in enumerate(index):
        end_idx = min(i + embargo_bars, n - 1)
        result[t] = index[end_idx]
    return pd.Series(result)


def detect_label_overlap(
    index: pd.DatetimeIndex,
    t1: pd.Series,
) -> dict:
    """Count pairs of adjacent observations whose label windows overlap."""
    n = len(index)
    n_overlapping = 0
    for i in range(n - 1):
        t0_i = index[i]
        t1_i = t1.get(t0_i, t0_i)
        t0_next = index[i + 1]
        if t1_i >= t0_next:
            n_overlapping += 1
    total_pairs = max(1, n - 1)
    return {
        "n_overlapping_pairs": n_overlapping,
        "overlap_fraction": n_overlapping / total_pairs,
        "n_total_pairs": total_pairs,
    }
