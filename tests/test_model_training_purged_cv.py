"""
test_model_training_purged_cv.py

TDD property-based tests for PurgedKFoldSplitter.
Written BEFORE PurgedKFoldSplitter implementation (red phase).

Properties:
  Property 6a: zero temporal overlap — |t_val - t_train| > embargo_period for all pairs
  Property 6b: embargo removes samples — len(purged_train) < len(full_train)
  Property 6c: at least n_splits folds are produced

Requirements: Req 3.1, Req 3.2, Req 3.3, Req 18.9
"""
from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings as h_settings
from hypothesis import strategies as st

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")


def _skip_if_not_implemented():
    """Skip if PurgedKFoldSplitter is not yet implemented."""
    try:
        from src.training.purged_kfold import PurgedKFoldSplitter

        return PurgedKFoldSplitter
    except (ImportError, ModuleNotFoundError):
        pytest.skip("PurgedKFoldSplitter not yet implemented — TDD red phase")


def _make_time_series(n_samples: int = 200) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Create synthetic X, y, and timestamps for testing."""
    np.random.seed(42)
    X = np.random.randn(n_samples, 10)
    y = np.random.randn(n_samples)
    dates = pd.bdate_range(start="2022-01-03", periods=n_samples)  # business days
    return X, y, dates


@st.composite
def time_series_params(draw):
    """Hypothesis strategy for variable-length time series parameters."""
    n_samples = draw(st.integers(min_value=100, max_value=300))
    n_splits = draw(st.integers(min_value=3, max_value=6))
    embargo_days = draw(st.integers(min_value=5, max_value=20))
    return n_samples, n_splits, embargo_days


class TestPurgedKFoldSplitterImport:
    """Verify the module can be imported."""

    def test_purged_kfold_splitter_importable(self):
        """PurgedKFoldSplitter must be importable."""
        try:
            from src.training.purged_kfold import PurgedKFoldSplitter

            splitter = PurgedKFoldSplitter(n_splits=5, embargo_days=10)
            assert splitter is not None
        except (ImportError, ModuleNotFoundError):
            pytest.skip("PurgedKFoldSplitter not yet implemented — TDD red phase")


class TestZeroTemporalOverlap:
    """
    Property 6a: Zero temporal overlap between training and validation folds.
    For all fold k, for ALL pairs (t_train, t_val):
        |t_val - t_train| > embargo_period (in trading days)

    Validates: Requirements 3.1, 3.2, 3.3
    """

    def test_no_overlap_fixed_case(self):
        """No train timestamp is within the embargo period of any validation timestamp."""
        PurgedKFoldSplitter = _skip_if_not_implemented()
        X, y, dates = _make_time_series(200)
        splitter = PurgedKFoldSplitter(n_splits=5, embargo_days=10)

        for train_idx, val_idx in splitter.split(X, y, dates):
            train_dates = dates[train_idx]
            val_dates = dates[val_idx]

            val_start = val_dates.min()

            # All training samples that precede the validation window must be
            # at least embargo_days calendar days before the validation start.
            for t_train in train_dates:
                if t_train < val_start:
                    embargo_boundary = val_start - timedelta(days=10)
                    assert t_train <= embargo_boundary, (
                        f"Training sample {t_train.date()} is within embargo of "
                        f"validation start {val_start.date()} "
                        f"(embargo_boundary={embargo_boundary.date()})"
                    )

    def test_validation_does_not_appear_in_training(self):
        """No validation index should appear in the training set."""
        PurgedKFoldSplitter = _skip_if_not_implemented()
        X, y, dates = _make_time_series(200)
        splitter = PurgedKFoldSplitter(n_splits=5, embargo_days=10)

        for train_idx, val_idx in splitter.split(X, y, dates):
            train_set = set(train_idx.tolist())
            val_set = set(val_idx.tolist())
            overlap = train_set.intersection(val_set)
            assert not overlap, (
                f"Index overlap between train and validation sets: {overlap}"
            )

    @given(params=time_series_params())
    @h_settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=5000)
    def test_no_temporal_overlap_hypothesis(self, params: tuple) -> None:
        """
        Property 6a (Hypothesis): For all valid time series lengths, splits, and
        embargo periods, no train sample is within the embargo window of any
        validation sample.

        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        n_samples, n_splits, embargo_days = params
        PurgedKFoldSplitter = _skip_if_not_implemented()

        X, y, dates = _make_time_series(n_samples)
        splitter = PurgedKFoldSplitter(n_splits=n_splits, embargo_days=embargo_days)

        for train_idx, val_idx in splitter.split(X, y, dates):
            train_set = set(train_idx.tolist())
            val_set = set(val_idx.tolist())

            # No index can appear in both sets
            assert not train_set.intersection(val_set), (
                "Train and validation index sets must be disjoint"
            )

            if len(train_idx) > 0 and len(val_idx) > 0:
                train_dates = dates[train_idx]
                val_dates = dates[val_idx]
                val_start = val_dates.min()

                # All training samples before the embargo boundary must be outside
                # the embargo zone (> embargo_days calendar days before val_start)
                for t_train in train_dates:
                    if t_train < val_start:
                        distance_days = (val_start - t_train).days
                        assert distance_days >= embargo_days, (
                            f"Training sample {t_train.date()} is only {distance_days} days "
                            f"before validation start {val_start.date()} "
                            f"(embargo={embargo_days} days required)"
                        )


class TestSamplesRemovedByEmbargo:
    """
    Property 6b: Embargo removes samples.
    When embargo_days > 0, the purged training fold has fewer samples
    than the unpurged training fold.

    Validates: Requirements 3.2, 3.3
    """

    def test_embargo_removes_training_samples(self):
        """With embargo=10, purged train fold must be smaller than naive train fold."""
        PurgedKFoldSplitter = _skip_if_not_implemented()
        X, y, dates = _make_time_series(200)

        splitter_purged = PurgedKFoldSplitter(n_splits=5, embargo_days=10)
        splitter_no_embargo = PurgedKFoldSplitter(n_splits=5, embargo_days=0)

        purged_counts = [len(t) for t, _ in splitter_purged.split(X, y, dates)]
        no_embargo_counts = [len(t) for t, _ in splitter_no_embargo.split(X, y, dates)]

        # At least some folds should have fewer training samples after purging
        assert any(
            p < n for p, n in zip(purged_counts, no_embargo_counts)
        ), "Embargo should remove at least some training samples from some folds"

    def test_minimum_embargo_enforced(self):
        """embargo_days must be at least 5 (design requirement)."""
        PurgedKFoldSplitter = _skip_if_not_implemented()

        with pytest.raises((ValueError, AssertionError)):
            # Creating a splitter with embargo < 5 should raise
            PurgedKFoldSplitter(n_splits=5, embargo_days=4)


class TestFoldCount:
    """Property 6c: At least n_splits folds are produced."""

    def test_correct_number_of_folds(self):
        """split() must yield exactly n_splits folds."""
        PurgedKFoldSplitter = _skip_if_not_implemented()
        X, y, dates = _make_time_series(200)

        for n_splits in [3, 5, 7]:
            splitter = PurgedKFoldSplitter(n_splits=n_splits, embargo_days=10)
            folds = list(splitter.split(X, y, dates))
            assert len(folds) == n_splits, (
                f"Expected {n_splits} folds, got {len(folds)}"
            )

    def test_all_folds_non_empty(self):
        """All folds must have at least 1 training and 1 validation sample."""
        PurgedKFoldSplitter = _skip_if_not_implemented()
        X, y, dates = _make_time_series(200)
        splitter = PurgedKFoldSplitter(n_splits=5, embargo_days=10)

        for i, (train_idx, val_idx) in enumerate(splitter.split(X, y, dates)):
            assert len(train_idx) > 0, f"Fold {i}: empty training set"
            assert len(val_idx) > 0, f"Fold {i}: empty validation set"


class TestChronologicalOrdering:
    """Training folds must come before validation folds chronologically."""

    def test_training_before_validation_chronologically(self):
        """All training timestamps must be before the validation fold."""
        PurgedKFoldSplitter = _skip_if_not_implemented()
        X, y, dates = _make_time_series(200)
        splitter = PurgedKFoldSplitter(n_splits=5, embargo_days=10)

        for train_idx, val_idx in splitter.split(X, y, dates):
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            max_train_date = dates[train_idx].max()
            min_val_date = dates[val_idx].min()
            assert max_train_date < min_val_date, (
                f"Training extends into validation window: "
                f"max_train={max_train_date.date()}, min_val={min_val_date.date()}"
            )
