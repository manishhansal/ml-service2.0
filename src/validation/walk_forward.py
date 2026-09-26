"""
src.validation.walk_forward — Walk-forward split utilities.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Generator, Iterator

import numpy as np
import pandas as pd


@dataclass
class WalkForwardFold:
    fold_index: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    expanding: bool = False

    # Optional datetime labels (populated by walk_forward_splits_on_dates)
    train_start_dt: pd.Timestamp | None = None
    train_end_dt: pd.Timestamp | None = None
    val_start_dt: pd.Timestamp | None = None
    val_end_dt: pd.Timestamp | None = None
    test_start_dt: pd.Timestamp | None = None
    test_end_dt: pd.Timestamp | None = None

    @property
    def train_indices(self) -> np.ndarray:
        return np.arange(self.train_start, self.train_end)

    @property
    def val_indices(self) -> np.ndarray:
        return np.arange(self.val_start, self.val_end)

    @property
    def test_indices(self) -> np.ndarray:
        return np.arange(self.test_start, self.test_end)

    def n_train(self) -> int:
        return self.train_end - self.train_start

    def n_val(self) -> int:
        return self.val_end - self.val_start

    def n_test(self) -> int:
        return self.test_end - self.test_start


@dataclass
class WalkForwardConfig:
    train_bars: int
    val_bars: int
    test_bars: int
    step_bars: int | None = None
    expanding: bool = False

    def __post_init__(self):
        if self.step_bars is None:
            self.step_bars = self.test_bars


class WalkForwardValidator:
    def __init__(self, cfg: WalkForwardConfig) -> None:
        self.cfg = cfg

    def split(self, n: int) -> list[WalkForwardFold]:
        return walk_forward_splits(
            n_periods=n,
            train_bars=self.cfg.train_bars,
            val_bars=self.cfg.val_bars,
            test_bars=self.cfg.test_bars,
            step_bars=self.cfg.step_bars or self.cfg.test_bars,
            expanding=self.cfg.expanding,
        )

    def iter_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        folds: list[WalkForwardFold],
    ) -> Iterator[tuple[WalkForwardFold, dict]]:
        for fold in folds:
            yield fold, {
                "train": (X[fold.train_indices], y[fold.train_indices]),
                "val": (X[fold.val_indices], y[fold.val_indices]),
                "test": (X[fold.test_indices], y[fold.test_indices]),
            }


def walk_forward_splits(
    n_periods: int,
    train_bars: int,
    val_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    expanding: bool = False,
) -> list[WalkForwardFold]:
    if step_bars is None:
        step_bars = test_bars
    window = train_bars + val_bars + test_bars
    if n_periods < window:
        raise ValueError(
            f"Time series too short: {n_periods} < {window} "
            f"(train={train_bars}+val={val_bars}+test={test_bars}). Raise n_periods."
        )

    folds = []
    fold_idx = 0
    start = 0
    while True:
        train_start = 0 if expanding else start
        train_end = start + train_bars
        val_end = train_end + val_bars
        test_end = val_end + test_bars
        if test_end > n_periods:
            break
        folds.append(WalkForwardFold(
            fold_index=fold_idx,
            train_start=train_start,
            train_end=train_end,
            val_start=train_end,
            val_end=val_end,
            test_start=val_end,
            test_end=test_end,
            expanding=expanding,
        ))
        fold_idx += 1
        start += step_bars

    return folds


def walk_forward_splits_on_dates(
    index: pd.DatetimeIndex,
    train_bars: int,
    val_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    expanding: bool = False,
) -> list[WalkForwardFold]:
    folds = walk_forward_splits(
        n_periods=len(index),
        train_bars=train_bars,
        val_bars=val_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        expanding=expanding,
    )
    for f in folds:
        f.train_start_dt = index[f.train_start]
        f.train_end_dt = index[f.train_end - 1]
        f.val_start_dt = index[f.val_start]
        f.val_end_dt = index[f.val_end - 1]
        f.test_start_dt = index[f.test_start]
        f.test_end_dt = index[f.test_end - 1]
    return folds


def save_fold_manifest(
    folds: list[WalkForwardFold],
    path: Path,
    cfg: WalkForwardConfig | None = None,
) -> Path:
    """Save fold manifest to JSON; never overwrites existing file."""
    path = Path(path)
    if path.exists():
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%f")
        path = parent / f"{stem}_{ts}{suffix}"

    manifest = {
        "n_folds": len(folds),
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "config": {
            "train_bars": cfg.train_bars if cfg else None,
            "val_bars": cfg.val_bars if cfg else None,
            "test_bars": cfg.test_bars if cfg else None,
        } if cfg else {},
        "folds": [
            {
                "fold_index": f.fold_index,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "val_start": f.val_start,
                "val_end": f.val_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "n_train": f.n_train(),
                "n_val": f.n_val(),
                "n_test": f.n_test(),
            }
            for f in folds
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path
