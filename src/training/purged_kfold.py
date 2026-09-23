"""
PurgedKFoldSplitter — time-series cross-validator with look-ahead purging and embargo.

Algorithm:
1. Sort samples chronologically by timestamps.
2. Divide the dataset into (n_splits + 1) equal windows.
3. The first window is the initial training "burn-in" (never used for validation).
4. For each of the remaining n_splits windows (k = 1 .. n_splits):
   a. Validation fold = window k.
   b. Training candidates = all samples strictly BEFORE the validation window start
      (i.e. windows 0 .. k-1).
   c. EMBARGO: remove training samples whose timestamps fall within
      [val_start - embargo_days, val_start).
   d. PURGE: additionally remove training samples whose label forward window
      (t + label_horizon_days) overlaps the validation window start.
   e. Yield the remaining training indices and the validation indices.

Design contract:
- embargo_days >= 5 (minimum enforced by constructor)
- embargo_days = 0 disables embargo (still purges overlapping labels)
- All returned arrays are np.ndarray of integer indices
- Training indices are always chronologically before validation indices
- No index appears in both train and validation sets
- Exactly n_splits folds are produced

Requirements: Req 3.1, Req 3.2, Req 3.3
"""
from __future__ import annotations

from datetime import timedelta
from typing import Iterator

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

MIN_EMBARGO_DAYS: int = 5


class PurgedKFoldSplitter:
    """
    Walk-forward time-series cross-validator with purging and embargo.

    The dataset is divided into (n_splits + 1) internal windows. The first
    window serves as the initial training burn-in so that every validation
    fold has at least one window of prior data.  This guarantees exactly
    n_splits folds are produced, all with non-empty training sets and
    strictly chronological ordering.

    Parameters
    ----------
    n_splits : int
        Number of temporal folds to produce (minimum 2).
    embargo_days : int
        Number of calendar days to embargo between training and validation.
        Must be >= 5 or == 0 (to disable). Default: 10.
    label_horizon_days : int
        Length of the label look-forward window in calendar days.
        Training samples whose label window overlaps the validation fold
        are purged. Default: 1.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 10,
        label_horizon_days: int = 1,
    ) -> None:
        if embargo_days != 0 and embargo_days < MIN_EMBARGO_DAYS:
            raise ValueError(
                f"embargo_days must be >= {MIN_EMBARGO_DAYS} or exactly 0 to disable. "
                f"Got {embargo_days}."
            )
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.label_horizon_days = label_horizon_days

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged+embargoed train/validation splits.

        Produces exactly ``self.n_splits`` folds.  Each fold has:
        - training indices chronologically before the validation window
        - validation indices not appearing in training indices

        Yields
        ------
        train_indices : np.ndarray
            Integer indices into X, y (and timestamps) for the training set.
        val_indices : np.ndarray
            Integer indices into X, y (and timestamps) for the validation set.
        """
        n = len(timestamps)

        # Sort by timestamp for deterministic fold boundaries
        sort_order = np.argsort(timestamps)
        sorted_dates = timestamps[sort_order]

        # Divide into (n_splits + 1) windows so that even fold 0 has prior data
        total_windows = self.n_splits + 1
        window_sizes = np.full(total_windows, n // total_windows, dtype=int)
        window_sizes[: n % total_windows] += 1  # distribute remainder to early windows

        window_boundaries = np.concatenate([[0], np.cumsum(window_sizes)])

        # Validation folds are windows 1 .. n_splits (window 0 is burn-in)
        for fold_k in range(1, total_windows):
            val_win_start = window_boundaries[fold_k]
            val_win_end = window_boundaries[fold_k + 1]

            # Validation: sorted positions for window fold_k
            val_sorted_pos = np.arange(val_win_start, val_win_end)
            val_dates = sorted_dates[val_sorted_pos]
            val_original_idx = sort_order[val_sorted_pos]

            val_start_date = val_dates.min()
            val_end_date = val_dates.max()

            # Training candidates: all samples BEFORE the validation window
            # (strict walk-forward — never use future data)
            train_sorted_pos = np.arange(0, val_win_start)
            train_candidate_dates = sorted_dates[train_sorted_pos]
            train_candidate_original_idx = sort_order[train_sorted_pos]

            # Embargo mask: exclude samples within embargo_days before val_start
            if self.embargo_days > 0:
                embargo_cutoff = val_start_date - timedelta(days=self.embargo_days)
                embargo_mask = train_candidate_dates <= embargo_cutoff
            else:
                embargo_mask = np.ones(len(train_candidate_dates), dtype=bool)

            # Purge mask: exclude samples whose label window overlaps val window.
            # A sample at time t has its label at t + label_horizon_days;
            # if that is >= val_start_date, it leaks forward into the validation fold.
            if self.label_horizon_days > 0:
                purge_cutoff = val_start_date - timedelta(days=self.label_horizon_days)
                purge_mask = train_candidate_dates <= purge_cutoff
            else:
                purge_mask = np.ones(len(train_candidate_dates), dtype=bool)

            # Combined: must pass both embargo AND purge
            combined_mask = embargo_mask & purge_mask
            train_original_idx = train_candidate_original_idx[combined_mask]

            if len(train_original_idx) == 0:
                # Embargo/purge removed everything — fall back to all pre-validation
                # candidates so the fold is non-empty.  Log a warning.
                logger.warning(
                    "purged_kfold_empty_train_fold_fallback",
                    fold_k=fold_k - 1,  # 0-indexed from caller perspective
                    n_splits=self.n_splits,
                    embargo_days=self.embargo_days,
                    val_start=str(val_start_date.date()),
                )
                train_original_idx = train_candidate_original_idx

            logger.debug(
                "purged_kfold_fold",
                fold=fold_k - 1,  # 0-indexed from caller perspective
                n_train=len(train_original_idx),
                n_val=len(val_original_idx),
                val_start=str(val_start_date.date()),
                val_end=str(val_end_date.date()),
                embargo_removed=int(combined_mask.size - combined_mask.sum()),
            )

            yield train_original_idx, val_original_idx
