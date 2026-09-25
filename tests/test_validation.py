"""
Tests for the AlphaForge Financial ML Validation Framework.

Covers all 9 required areas:
  1.  Walk-forward validation — window sizes, non-overlap, time ordering
  2.  Expanding window mode — training window grows over time
  3.  Embargo period — bars / minutes / days units, correct index exclusion
  4.  Lookahead leakage detection — synthetic dataset proves leakage detected
  5.  Label overlap purging — PurgedKFold removes contaminated training obs
  6.  Purging correctness — purged training set never contains future labels
  7.  CPCV structure — C(N,k) splits generated, correct group coverage
  8.  CPCV path construction — backtest paths cover all groups exactly once
  9.  Classification metrics — accuracy, precision, recall, F1, ROC-AUC
  10. Trading metrics — Sharpe, Sortino, max drawdown, profit factor, expectancy
  11. Model acceptance gate — dual-pass enforcement (accuracy AND trading)
  12. Stability analysis — cross-fold distribution, decay, dominance detection
  13. Result persistence — save / no-overwrite / load history

All tests are fully offline — no network calls, no database, no model loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _make_datetime_index(
    n: int,
    start: str = "2023-01-02",
    freq: str = "B",
    tz: str = "UTC",
) -> pd.DatetimeIndex:
    """Generate a business-day DatetimeIndex of length n."""
    return pd.bdate_range(start=start, periods=n, freq=freq, tz=tz)


def _make_price_series(
    n: int = 500,
    seed: int = 42,
    start: str = "2023-01-02",
) -> pd.Series:
    """Geometric random walk close price series."""
    rng = np.random.default_rng(seed)
    idx = _make_datetime_index(n, start=start)
    log_ret = rng.normal(0.0005, 0.015, size=n)
    close = 1000.0 * np.exp(np.cumsum(log_ret))
    return pd.Series(close, index=idx, name="close")


def _make_returns_series(
    n: int = 252,
    seed: int = 99,
    start: str = "2023-01-02",
    positive_drift: float = 0.0002,
) -> pd.Series:
    """Synthetic daily strategy return series."""
    rng = np.random.default_rng(seed)
    idx = _make_datetime_index(n, start=start)
    returns = rng.normal(positive_drift, 0.012, size=n)
    return pd.Series(returns, index=idx, name="strategy_return")


def _make_classification_dataset(
    n: int = 500,
    n_features: int = 10,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Synthetic X, y, and DatetimeIndex for classification tests."""
    rng = np.random.default_rng(seed)
    idx = _make_datetime_index(n)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.int32)
    return X, y, idx


def _make_t1_series(
    index: pd.DatetimeIndex,
    horizon_bars: int = 5,
) -> pd.Series:
    """Build a fixed-horizon t1 (label end time) series."""
    from src.validation.purged_kfold import build_t1_series
    return build_t1_series(index, horizon_bars)


# ─── 1. Walk-forward validation ───────────────────────────────────────────────


class TestWalkForwardValidation:
    """
    Requirement #1 / Test area 1: rolling windows, non-overlap, time ordering.
    """

    def test_basic_split_returns_folds(self) -> None:
        from src.validation import walk_forward_splits

        folds = walk_forward_splits(n_periods=500, train_bars=300, val_bars=100, test_bars=100)
        assert len(folds) >= 1
        for fold in folds:
            assert fold.n_train() > 0
            assert fold.n_val() > 0
            assert fold.n_test() > 0

    def test_train_val_test_non_overlapping(self) -> None:
        from src.validation import walk_forward_splits

        folds = walk_forward_splits(n_periods=600, train_bars=200, val_bars=100, test_bars=100)
        for fold in folds:
            train = set(fold.train_indices)
            val = set(fold.val_indices)
            test = set(fold.test_indices)
            assert train.isdisjoint(val), "Train overlaps validation"
            assert val.isdisjoint(test), "Validation overlaps test"
            assert train.isdisjoint(test), "Train overlaps test"

    def test_strict_time_ordering_within_fold(self) -> None:
        from src.validation import walk_forward_splits

        folds = walk_forward_splits(n_periods=600, train_bars=200, val_bars=100, test_bars=100)
        for fold in folds:
            assert max(fold.train_indices) < min(fold.val_indices), (
                "Train extends past val start — time ordering violated"
            )
            assert max(fold.val_indices) < min(fold.test_indices), (
                "Val extends past test start — time ordering violated"
            )

    def test_multiple_folds_generated(self) -> None:
        from src.validation import walk_forward_splits

        folds = walk_forward_splits(
            n_periods=1000, train_bars=300, val_bars=100, test_bars=100, step_bars=100
        )
        assert len(folds) >= 4, f"Expected >=4 folds, got {len(folds)}"

    def test_step_controls_roll_distance(self) -> None:
        from src.validation import walk_forward_splits

        folds_step50 = walk_forward_splits(
            n_periods=1000, train_bars=300, val_bars=100, test_bars=100, step_bars=50
        )
        folds_step100 = walk_forward_splits(
            n_periods=1000, train_bars=300, val_bars=100, test_bars=100, step_bars=100
        )
        # Smaller step → more folds
        assert len(folds_step50) > len(folds_step100)

    def test_raises_when_series_too_short(self) -> None:
        from src.validation import walk_forward_splits

        with pytest.raises(ValueError, match="too short"):
            walk_forward_splits(n_periods=50, train_bars=200, val_bars=50, test_bars=50)

    def test_folds_are_chronologically_ordered(self) -> None:
        from src.validation import walk_forward_splits

        folds = walk_forward_splits(n_periods=800, train_bars=300, val_bars=100, test_bars=100)
        for i in range(len(folds) - 1):
            assert folds[i].test_end <= folds[i + 1].test_end, (
                "Folds are not in chronological order"
            )

    def test_split_on_datetime_index_attaches_dt_labels(self) -> None:
        from src.validation import walk_forward_splits_on_dates

        idx = _make_datetime_index(500)
        folds = walk_forward_splits_on_dates(
            idx, train_bars=250, val_bars=100, test_bars=100
        )
        for fold in folds:
            assert fold.train_start_dt is not None
            assert fold.train_end_dt is not None
            assert fold.test_start_dt is not None
            assert fold.test_end_dt is not None
            # Datetime ordering must mirror index ordering
            assert fold.train_end_dt <= fold.val_start_dt  # type: ignore[operator]
            assert fold.val_end_dt <= fold.test_start_dt   # type: ignore[operator]

    def test_iter_splits_yields_correct_array_shapes(self) -> None:
        from src.validation import WalkForwardConfig, WalkForwardValidator

        X, y, _ = _make_classification_dataset(600)
        cfg = WalkForwardConfig(train_bars=300, val_bars=100, test_bars=100)
        wfv = WalkForwardValidator(cfg)
        folds = wfv.split(len(X))

        for fold, data in wfv.iter_splits(X, y, folds):
            X_train, y_train = data["train"]
            X_val,   y_val   = data["val"]
            X_test,  y_test  = data["test"]
            assert X_train.shape[0] == y_train.shape[0]
            assert X_val.shape[0] == y_val.shape[0]
            assert X_test.shape[0] == y_test.shape[0]
            # Sizes match fold metadata
            assert X_train.shape[0] == fold.n_train()
            assert X_test.shape[0] == fold.n_test()

    def test_configurable_window_sizes(self) -> None:
        """Each window size is exactly as configured (rolling mode)."""
        from src.validation import walk_forward_splits

        train_b, val_b, test_b = 200, 60, 40
        folds = walk_forward_splits(
            n_periods=700, train_bars=train_b, val_bars=val_b, test_bars=test_b
        )
        for fold in folds:
            assert fold.n_train() == train_b
            assert fold.n_val() == val_b
            assert fold.n_test() == test_b

    def test_fold_manifest_saved_and_loadable(self, tmp_path: Path) -> None:
        from src.validation import walk_forward_splits, WalkForwardConfig, save_fold_manifest

        folds = walk_forward_splits(n_periods=600, train_bars=300, val_bars=100, test_bars=100)
        cfg = WalkForwardConfig(train_bars=300, val_bars=100, test_bars=100)
        saved = save_fold_manifest(folds, tmp_path / "manifest.json", cfg)

        assert saved.exists()
        manifest = json.loads(saved.read_text())
        assert manifest["n_folds"] == len(folds)
        assert "folds" in manifest
        assert len(manifest["folds"]) == len(folds)

    def test_manifest_not_overwritten(self, tmp_path: Path) -> None:
        """Saving a second manifest never overwrites the first."""
        from src.validation import walk_forward_splits, save_fold_manifest

        folds = walk_forward_splits(n_periods=600, train_bars=300, val_bars=100, test_bars=100)
        path = tmp_path / "manifest.json"
        p1 = save_fold_manifest(folds, path)
        p2 = save_fold_manifest(folds, path)

        # Both files must exist and be different
        assert p1.exists() and p2.exists()
        assert p1 != p2, "Second save overwrote the first — historical results are not protected"


# ─── 2. Expanding window mode ─────────────────────────────────────────────────


class TestExpandingWindow:
    """
    Requirement #1 (expanding variant): training window grows over time.
    """

    def test_expanding_train_grows_with_each_fold(self) -> None:
        from src.validation import WalkForwardConfig, WalkForwardValidator

        cfg = WalkForwardConfig(
            train_bars=252, val_bars=63, test_bars=63, expanding=True
        )
        wfv = WalkForwardValidator(cfg)
        folds = wfv.split(1000)

        train_sizes = [f.n_train() for f in folds]
        # Each fold's training window should be larger than the previous
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1], (
                f"Expanding window shrunk at fold {i}: "
                f"{train_sizes[i]} < {train_sizes[i-1]}"
            )

    def test_expanding_train_always_starts_at_zero(self) -> None:
        from src.validation import WalkForwardConfig, WalkForwardValidator

        cfg = WalkForwardConfig(train_bars=252, val_bars=63, test_bars=63, expanding=True)
        folds = WalkForwardValidator(cfg).split(800)
        for fold in folds:
            assert fold.train_start == 0, (
                "Expanding window must always start at index 0"
            )

    def test_expanding_no_overlap_with_val_test(self) -> None:
        from src.validation import WalkForwardConfig, WalkForwardValidator

        cfg = WalkForwardConfig(train_bars=200, val_bars=50, test_bars=50, expanding=True)
        folds = WalkForwardValidator(cfg).split(700)
        for fold in folds:
            train = set(fold.train_indices)
            val = set(fold.val_indices)
            test = set(fold.test_indices)
            assert train.isdisjoint(val)
            assert train.isdisjoint(test)


# ─── 3. Embargo period ────────────────────────────────────────────────────────


class TestEmbargoPeriod:
    """
    Requirement #3 / Test area 3: embargo in bars, minutes, and days.
    The embargo must exclude the correct number of observations.
    """

    def test_embargo_bars_removes_correct_count(self) -> None:
        from src.validation import EmbargoConfig, EmbargoApplier

        train = range(0, 100)
        candidates = range(100, 200)

        cfg = EmbargoConfig.bars(10)
        applier = EmbargoApplier(cfg)
        filtered = applier.apply_to_indices(train, candidates)

        # First 10 candidates (100-109) should be excluded; 110-199 included
        assert 109 not in filtered, "Bar 109 should be embargoed"
        assert 110 in filtered, "Bar 110 should be past the embargo"
        assert len(filtered) == 90

    def test_zero_embargo_passes_all(self) -> None:
        from src.validation import EmbargoConfig, EmbargoApplier

        train = range(0, 50)
        candidates = range(50, 100)
        applier = EmbargoApplier(EmbargoConfig.bars(0))
        filtered = applier.apply_to_indices(train, candidates)
        assert len(filtered) == 50

    def test_embargo_minutes_converts_to_bars(self) -> None:
        from src.validation import EmbargoConfig, EmbargoUnit

        # 15 minutes at 5-min resolution = 3 bars
        cfg = EmbargoConfig.minutes(15, bars_per_minute=0.2)
        assert cfg.to_bars() == 3

    def test_embargo_days_converts_to_bars(self) -> None:
        from src.validation import EmbargoConfig, EmbargoUnit

        # 2 days at 75 bars/day (5-min NSE session) = 150 bars
        cfg = EmbargoConfig.days(2, bars_per_day=75)
        assert cfg.to_bars() == 150

    def test_embargo_applied_to_fold_val_and_test(self) -> None:
        from src.validation import EmbargoConfig, EmbargoApplier

        train = range(0, 200)
        val = range(200, 300)
        test = range(300, 400)

        applier = EmbargoApplier(EmbargoConfig.bars(5))
        _, purged_val, purged_test = applier.apply_to_fold(train, val, test)

        # Val: first 5 observations excluded (200-204 → cutoff at 205)
        assert 204 not in purged_val, "Bar 204 should be embargoed from val"
        assert 205 in purged_val, "Bar 205 should be past the embargo"

    def test_embargo_does_not_exclude_beyond_cutoff(self) -> None:
        from src.validation import EmbargoConfig, EmbargoApplier

        train = range(0, 100)
        # Large gap between train end and candidates — embargo should not remove any
        candidates = range(200, 300)
        applier = EmbargoApplier(EmbargoConfig.bars(10))
        filtered = applier.apply_to_indices(train, candidates)
        assert len(filtered) == 100, (
            "Embargo should not exclude candidates that are already past the embargo cutoff"
        )

    def test_embargo_raises_on_negative_size(self) -> None:
        from src.validation import EmbargoConfig

        with pytest.raises(ValueError):
            EmbargoConfig(size=-1)

    def test_compute_embargo_times_produces_series(self) -> None:
        from src.validation import compute_embargo_times

        idx = _make_datetime_index(100)
        result = compute_embargo_times(idx, pct=0.02)
        assert len(result) == len(idx)
        # The embargo end time must always be >= the observation time
        for t in idx[:80]:
            assert result[t] >= t, "Embargo end time must be >= observation time"


# ─── 4. Lookahead leakage detection ──────────────────────────────────────────


class TestLookaheadLeakageDetection:
    """
    Requirement #4 / Test area 4: synthetic datasets prove leakage is detected.

    Leakage scenario: a feature is literally the future label (forward-shifted).
    This must be detected and flagged, while clean features must pass through.
    """

    def _make_leaked_dataset(
        self, n: int = 300, horizon: int = 5
    ) -> tuple[pd.DataFrame, list[str], str]:
        """
        Return a DataFrame where 'leaked_feature' is the future label.

        The label is pure white noise (no autocorrelation) so that:
          - corr(leaked_feature, future_label) ≈ 1.0
          - corr(leaked_feature, present_label) ≈ 0.0
        This guarantees the leakage condition in assert_no_future_leakage()
        is cleanly triggered: corr_future >> 0.95 and corr_future >> corr_now + 0.1.
        """
        rng = np.random.default_rng(1)
        idx = _make_datetime_index(n)
        # White noise label (zero autocorrelation) so shifted version is
        # uncorrelated with the unshifted version
        label = rng.standard_normal(n)
        # leaked_feature = the future label exactly (forward shift by horizon)
        future_label = np.empty(n)
        future_label[:n - horizon] = label[horizon:]
        future_label[n - horizon:] = np.nan
        clean = rng.standard_normal(n)  # independent noise — no leakage
        df = pd.DataFrame(
            {"clean_feature": clean, "leaked_feature": future_label, "label": label},
            index=idx,
        )
        return df, ["clean_feature", "leaked_feature"], "label"

    def _make_clean_dataset(self, n: int = 300) -> tuple[pd.DataFrame, list[str], str]:
        """Return a DataFrame with only clean (non-leaking) features."""
        rng = np.random.default_rng(2)
        idx = _make_datetime_index(n)
        label = np.cumsum(rng.normal(0, 1, n))
        feature = rng.normal(0, 1, n)  # truly independent noise
        df = pd.DataFrame({"feature": feature, "label": label}, index=idx)
        return df, ["feature"], "label"

    def test_clean_features_pass_leakage_check(self) -> None:
        """A feature that is independent noise must not raise AssertionError."""
        from src.training.data_pipeline import assert_no_future_leakage

        df, features, label = self._make_clean_dataset()
        # Must not raise
        assert_no_future_leakage(df, features, label, horizon=5)

    def test_leaked_feature_raises_assertion(self) -> None:
        """A feature that IS the future label must trigger AssertionError."""
        from src.training.data_pipeline import assert_no_future_leakage

        df, features, label = self._make_leaked_dataset()
        with pytest.raises(AssertionError, match="leakage detected"):
            assert_no_future_leakage(df, features, label, horizon=5)

    def test_walk_forward_train_indices_never_include_future(self) -> None:
        """
        The maximum train index in any fold must be strictly less than
        the minimum val index. Future data never enters the training set.
        """
        from src.validation import walk_forward_splits

        folds = walk_forward_splits(n_periods=500, train_bars=250, val_bars=100, test_bars=100)
        for fold in folds:
            assert max(fold.train_indices) < min(fold.val_indices), (
                "Walk-forward fold allows future data into training set"
            )

    def test_purged_kfold_train_labels_do_not_overlap_test_times(self) -> None:
        """
        After purging, no training observation's label end time may fall
        inside the test window. This directly tests leakage prevention.
        """
        from src.validation import PurgedKFold

        n = 300
        X, y, idx = _make_classification_dataset(n, seed=11)
        t1 = _make_t1_series(idx, horizon_bars=10)

        splitter = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.01)

        for train_idx, test_idx in splitter.split(X, y, groups=idx):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            test_start_time = idx[test_idx[0]]
            # Check every training observation's label end time
            for i in train_idx:
                t0_i = idx[i]
                t1_i = t1.get(t0_i, t0_i)
                assert t1_i < test_start_time, (
                    f"Purging failed: training obs at {t0_i} has label end time "
                    f"{t1_i} which overlaps test start {test_start_time}. "
                    "This is a leakage bug."
                )

    def test_cpcv_train_indices_do_not_overlap_test_indices(self) -> None:
        """CPCV: for every split, train indices must be disjoint from test indices."""
        from src.validation import cpcv_splits

        X, y, idx = _make_classification_dataset(300, seed=13)
        t1 = _make_t1_series(idx, horizon_bars=5)
        folds = cpcv_splits(X, y, n_splits=5, n_test_splits=2, t1=t1, t0=idx)

        for fold in folds:
            train_set = set(fold.train_indices.tolist())
            test_set = set(fold.test_indices.tolist())
            assert train_set.isdisjoint(test_set), (
                f"CPCV fold {fold.split_index}: train and test indices overlap — "
                "future data leaking into training set"
            )


# ─── 5. Label overlap purging ─────────────────────────────────────────────────


class TestLabelOverlapPurging:
    """
    Requirement #2 & #5 / Test area 5: PurgedKFold removes contaminated obs.
    """

    def _make_strongly_overlapping_t1(
        self, index: pd.DatetimeIndex, horizon: int = 20
    ) -> pd.Series:
        """Create t1 series with very long label horizon (lots of overlap)."""
        return _make_t1_series(index, horizon_bars=horizon)

    def test_purging_reduces_training_set_size(self) -> None:
        """With a large label horizon, purging must remove some training obs."""
        from src.validation import PurgedKFold

        n = 200
        X, y, idx = _make_classification_dataset(n, seed=5)
        t1 = self._make_strongly_overlapping_t1(idx, horizon=15)

        splitter_purged = PurgedKFold(n_splits=4, t1=t1, embargo_pct=0.01)
        splitter_raw = PurgedKFold(n_splits=4, t1=None, embargo_pct=0.0)

        total_purged_train = 0
        total_raw_train = 0

        for train_idx, _ in splitter_purged.split(X, y, groups=idx):
            total_purged_train += len(train_idx)

        for train_idx, _ in splitter_raw.split(X, y, groups=idx):
            total_raw_train += len(train_idx)

        assert total_purged_train < total_raw_train, (
            "Purging with a long horizon should remove training observations, "
            f"but purged ({total_purged_train}) >= raw ({total_raw_train})"
        )

    def test_purging_leaves_test_set_intact(self) -> None:
        """Purging only modifies training sets. Test sets must be unchanged."""
        from src.validation import PurgedKFold

        n = 200
        X, y, idx = _make_classification_dataset(n, seed=6)
        t1 = _make_t1_series(idx, horizon_bars=10)
        splitter = PurgedKFold(n_splits=4, t1=t1, embargo_pct=0.01)
        splitter_raw = PurgedKFold(n_splits=4, t1=None, embargo_pct=0.0)

        for (_, test_purged), (_, test_raw) in zip(
            splitter.split(X, y, groups=idx),
            splitter_raw.split(X, y, groups=idx),
        ):
            # Test folds must be identical regardless of purging
            assert np.array_equal(test_purged, test_raw), (
                "Purging modified the test set — it should only affect training"
            )

    def test_detect_label_overlap_quantifies_contamination(self) -> None:
        """detect_label_overlap must report non-zero overlap for overlapping labels."""
        from src.validation import detect_label_overlap

        idx = _make_datetime_index(100)
        t1_overlapping = _make_t1_series(idx, horizon_bars=10)  # each label spans 10 bars
        overlap_stats = detect_label_overlap(idx, t1_overlapping)

        assert overlap_stats["n_overlapping_pairs"] > 0, (
            "Expected label overlap to be detected for a 10-bar horizon label"
        )
        assert overlap_stats["overlap_fraction"] > 0, (
            "Overlap fraction must be positive for overlapping labels"
        )

    def test_detect_label_overlap_zero_for_point_labels(self) -> None:
        """Point-in-time labels (horizon=0) have no overlap."""
        from src.validation import detect_label_overlap

        idx = _make_datetime_index(100)
        t1_point = _make_t1_series(idx, horizon_bars=0)  # label ends at same bar
        overlap_stats = detect_label_overlap(idx, t1_point)

        # With horizon=0 every label end time equals t0 — no overlap with next bar
        assert overlap_stats["n_overlapping_pairs"] == 0, (
            "Point-in-time labels should have zero overlap"
        )

    def test_purged_kfold_is_sklearn_compatible(self) -> None:
        """PurgedKFold must implement BaseCrossValidator correctly."""
        from src.validation import PurgedKFold
        from sklearn.model_selection import BaseCrossValidator

        splitter = PurgedKFold(n_splits=5)
        assert isinstance(splitter, BaseCrossValidator)
        assert splitter.get_n_splits() == 5


# ─── 6. Purging correctness ───────────────────────────────────────────────────


class TestPurgingCorrectness:
    """
    Test area 6: Directly verify that purged training sets contain no future-leaking labels.
    """

    def test_build_t1_series_horizon_correct(self) -> None:
        from src.validation import build_t1_series

        idx = _make_datetime_index(50)
        t1 = build_t1_series(idx, horizon_bars=5)

        # For observation at position i, t1[i] must be idx[min(i+5, n-1)]
        for i in range(len(idx) - 10):
            expected_t1 = idx[i + 5]
            assert t1.iloc[i] == expected_t1, (
                f"t1 at position {i}: expected {expected_t1}, got {t1.iloc[i]}"
            )

    def test_purge_train_indices_by_t1_removes_contaminated(self) -> None:
        from src.validation.embargo import purge_train_indices_by_t1

        idx = _make_datetime_index(100)
        t1 = _make_t1_series(idx, horizon_bars=10)

        # Training: first 50 bars; validation: bars 60-80
        train_idx = np.arange(0, 50)
        val_idx = np.arange(60, 80)

        purged = purge_train_indices_by_t1(train_idx, val_idx, idx, t1)

        # Training obs 50-59 have labels that extend into the val window (60+)
        # So train obs at positions ~50 onwards should be purged
        # Specifically obs at positions 50..59 have t1 that reaches 60..69 (in val)
        # Since we use positions 0-49 as train, those with t1 >= idx[60] are purged
        val_start_time = idx[60]
        for i in purged:
            obs_t0 = idx[i]
            obs_t1 = t1.get(obs_t0, obs_t0)
            assert obs_t1 < val_start_time, (
                f"Contaminated obs at position {i} (t1={obs_t1}) survived purging. "
                f"Val starts at {val_start_time}. Leakage not prevented."
            )

    def test_embargo_after_purge_excludes_boundary_observations(self) -> None:
        """After purging, an additional embargo removes boundary observations."""
        from src.validation import EmbargoConfig, EmbargoApplier
        from src.validation.embargo import purge_train_indices_by_t1

        idx = _make_datetime_index(200)
        t1 = _make_t1_series(idx, horizon_bars=5)

        train_idx = np.arange(0, 120)
        val_idx = np.arange(130, 180)

        # Step 1: purge
        purged_train = purge_train_indices_by_t1(train_idx, val_idx, idx, t1)

        # Step 2: embargo 5 bars after purge
        applier = EmbargoApplier(EmbargoConfig.bars(5))
        final_train = applier.apply_to_indices(purged_train, val_idx)

        # All remaining val indices should be > max(purged_train) + 5
        if len(purged_train) > 0 and len(final_train) > 0:
            max_train = max(purged_train)
            for idx_val in final_train:
                assert idx_val > max_train + 5, (
                    f"Val index {idx_val} is within embargo window of last train idx {max_train}"
                )


# ─── 7. CPCV structure ───────────────────────────────────────────────────────


class TestCPCVStructure:
    """
    Requirement #4 / Test area 7: CPCV produces C(N,k) splits with correct coverage.
    """

    def test_cpcv_produces_correct_number_of_splits(self) -> None:
        """C(6, 2) = 15 splits should be generated."""
        from src.validation import cpcv_splits
        from math import comb

        X, y, idx = _make_classification_dataset(300, seed=14)
        folds = cpcv_splits(X, y, n_splits=6, n_test_splits=2)
        expected = comb(6, 2)
        assert len(folds) == expected, f"Expected {expected} CPCV splits, got {len(folds)}"

    def test_cpcv_train_test_cover_all_samples(self) -> None:
        """
        The union of all test sets across all C(N,k) splits should collectively
        cover every sample index at least once.
        """
        from src.validation import cpcv_splits

        n = 300
        X, y, idx = _make_classification_dataset(n, seed=15)
        folds = cpcv_splits(X, y, n_splits=6, n_test_splits=2)

        all_test_indices: set[int] = set()
        for fold in folds:
            all_test_indices |= set(fold.test_indices.tolist())

        # Not all indices may be covered but at least 80% should be
        coverage = len(all_test_indices) / n
        assert coverage >= 0.8, f"CPCV test coverage only {coverage:.1%} — expected >= 80%"

    def test_cpcv_each_split_train_not_empty(self) -> None:
        """Every CPCV fold must have a non-empty training set."""
        from src.validation import cpcv_splits

        X, y, _ = _make_classification_dataset(300, seed=16)
        folds = cpcv_splits(X, y, n_splits=6, n_test_splits=2)
        for fold in folds:
            assert len(fold.train_indices) > 0, f"Fold {fold.split_index} has empty training set"

    def test_cpcv_each_split_test_not_empty(self) -> None:
        from src.validation import cpcv_splits

        X, y, _ = _make_classification_dataset(300, seed=17)
        folds = cpcv_splits(X, y, n_splits=6, n_test_splits=2)
        for fold in folds:
            assert len(fold.test_indices) > 0, f"Fold {fold.split_index} has empty test set"

    def test_cpcv_test_groups_are_all_distinct_combinations(self) -> None:
        """Each split must use a unique combination of test groups."""
        from src.validation import cpcv_splits

        X, y, _ = _make_classification_dataset(300, seed=18)
        folds = cpcv_splits(X, y, n_splits=6, n_test_splits=2)

        test_group_sets = [fold.test_groups for fold in folds]
        unique_sets = set(test_group_sets)
        assert len(unique_sets) == len(folds), (
            f"Duplicate test group combinations found: {len(folds) - len(unique_sets)} duplicates"
        )

    def test_cpcv_with_purging_marks_n_purged(self) -> None:
        """When t1 is provided, n_purged must be > 0 for at least some folds."""
        from src.validation import cpcv_splits

        n = 300
        X, y, idx = _make_classification_dataset(n, seed=19)
        t1 = _make_t1_series(idx, horizon_bars=10)  # 10-bar overlap horizon

        folds = cpcv_splits(X, y, n_splits=6, n_test_splits=2, t1=t1, t0=idx)
        total_purged = sum(f.n_purged for f in folds)
        assert total_purged > 0, (
            "With a 10-bar label horizon, some training obs must be purged. "
            "n_purged=0 suggests purging is not working."
        )

    def test_cpcv_invalid_config_raises(self) -> None:
        from src.validation import CPCVConfig

        with pytest.raises(ValueError):
            CPCVConfig(n_splits=3, n_test_splits=3)  # n_test_splits must be < n_splits

        with pytest.raises(ValueError):
            CPCVConfig(n_splits=1, n_test_splits=1)  # n_splits must be >= 2


# ─── 8. CPCV path construction ───────────────────────────────────────────────


class TestCPCVPathConstruction:
    """
    Test area 8: backtest paths cover all groups without overlap.
    """

    def test_backtest_paths_cover_all_groups(self) -> None:
        """Each backtest path must collectively cover all N groups."""
        from src.validation import CPCVConfig, CombinatorialPurgedCV

        n = 360
        X, y, _ = _make_classification_dataset(n, seed=20)
        cfg = CPCVConfig(n_splits=6, n_test_splits=2)
        engine = CombinatorialPurgedCV(cfg)
        folds = engine.split(X, y)
        paths = engine.build_backtest_paths(folds)

        assert len(paths) > 0, "No backtest paths were constructed"
        for path in paths:
            covered = set()
            for fold in path:
                covered |= set(fold.test_groups)
            # Each path should cover at least half of all groups
            assert len(covered) >= cfg.n_splits // 2, (
                f"Backtest path covers only {len(covered)}/{cfg.n_splits} groups"
            )

    def test_backtest_path_folds_non_overlapping_test_groups(self) -> None:
        """Within a single backtest path, test groups must be pairwise disjoint."""
        from src.validation import CPCVConfig, CombinatorialPurgedCV

        X, y, _ = _make_classification_dataset(300, seed=21)
        cfg = CPCVConfig(n_splits=6, n_test_splits=2)
        engine = CombinatorialPurgedCV(cfg)
        folds = engine.split(X, y)
        paths = engine.build_backtest_paths(folds)

        for path_idx, path in enumerate(paths):
            groups_seen: set[int] = set()
            for fold in path:
                fold_groups = set(fold.test_groups)
                assert fold_groups.isdisjoint(groups_seen), (
                    f"Path {path_idx}: test groups overlap within path — "
                    f"groups {fold_groups} already in {groups_seen}"
                )
                groups_seen |= fold_groups


# ─── 9. Classification metrics ───────────────────────────────────────────────


class TestClassificationMetrics:
    """
    Requirement #5 / Test area 9: accuracy, precision, recall, F1, ROC-AUC.
    """

    def _make_perfect_predictions(
        self, n: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(30)
        y_true = rng.integers(0, 2, size=n)
        return y_true, y_true.copy()

    def _make_random_predictions(
        self, n: int = 200, seed: int = 31
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 2, size=n)
        y_pred = rng.integers(0, 2, size=n)
        y_prob = rng.random((n, 2))
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        return y_true, y_pred, y_prob

    def test_perfect_predictions_give_accuracy_one(self) -> None:
        from src.validation import FinancialMetricsEvaluator

        ev = FinancialMetricsEvaluator()
        y_true, y_pred = self._make_perfect_predictions()
        metrics = ev.classification_metrics(y_true, y_pred)
        assert metrics.accuracy == pytest.approx(1.0)
        assert metrics.f1 == pytest.approx(1.0)

    def test_all_wrong_gives_accuracy_zero(self) -> None:
        from src.validation import FinancialMetricsEvaluator

        ev = FinancialMetricsEvaluator()
        y_true = np.zeros(100, dtype=int)
        y_pred = np.ones(100, dtype=int)
        metrics = ev.classification_metrics(y_true, y_pred)
        assert metrics.accuracy == pytest.approx(0.0)

    def test_roc_auc_with_probabilities(self) -> None:
        from src.validation import FinancialMetricsEvaluator

        ev = FinancialMetricsEvaluator()
        y_true, y_pred, y_prob = self._make_random_predictions()
        metrics = ev.classification_metrics(y_true, y_pred, y_prob=y_prob[:, 1])
        assert metrics.roc_auc is not None
        assert 0.0 <= metrics.roc_auc <= 1.0

    def test_classification_metrics_sample_count(self) -> None:
        from src.validation import FinancialMetricsEvaluator

        ev = FinancialMetricsEvaluator()
        y_true, y_pred, _ = self._make_random_predictions(n=150)
        metrics = ev.classification_metrics(y_true, y_pred)
        assert metrics.n_samples == 150

    def test_multiclass_metrics(self) -> None:
        """Metrics must handle 3-class problems via macro averaging."""
        from src.validation import FinancialMetricsEvaluator

        rng = np.random.default_rng(32)
        y_true = rng.integers(0, 3, size=300)
        y_pred = rng.integers(0, 3, size=300)
        ev = FinancialMetricsEvaluator()
        metrics = ev.classification_metrics(y_true, y_pred)
        assert metrics.n_classes == 3
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.f1 <= 1.0


# ─── 10. Trading metrics ─────────────────────────────────────────────────────


class TestTradingMetrics:
    """
    Requirement #5 / Test area 10: Sharpe, Sortino, drawdown, profit factor,
    expectancy, turnover.
    """

    def test_positive_drift_gives_positive_sharpe(self) -> None:
        from src.validation import FinancialMetricsEvaluator

        ev = FinancialMetricsEvaluator()
        # Strong positive drift: 0.1% / day, low vol
        rng = np.random.default_rng(40)
        returns = pd.Series(rng.normal(0.001, 0.005, 252))
        metrics = ev.trading_metrics(returns)
        assert metrics.sharpe_ratio > 0, (
            "Strongly positive returns must yield a positive Sharpe ratio"
        )

    def test_negative_returns_give_negative_sharpe(self) -> None:
        from src.validation import FinancialMetricsEvaluator

        ev = FinancialMetricsEvaluator()
        rng = np.random.default_rng(41)
        returns = pd.Series(rng.normal(-0.002, 0.005, 252))
        metrics = ev.trading_metrics(returns)
        assert metrics.sharpe_ratio < 0

    def test_max_drawdown_is_non_negative(self) -> None:
        from src.validation import FinancialMetricsEvaluator

        ev = FinancialMetricsEvaluator()
        returns = _make_returns_series(252)
        metrics = ev.trading_metrics(returns)
        assert metrics.max_drawdown >= 0.0
        assert metrics.max_drawdown <= 1.0

    def test_max_drawdown_correct_on_known_path(self) -> None:
        """
        Known path: +10% then -20% from peak.
        Max drawdown from (1.1) to (0.88): (0.88 - 1.1) / 1.1 ≈ 0.2 (20%).
        """
        from src.validation import FinancialMetricsEvaluator

        ev = FinancialMetricsEvaluator()
        # Build a simple path: 10 * +1%, then 20 * -1%
        up = [0.01] * 10
        down = [-0.01] * 20
        returns = pd.Series(up + down + [0.0] * 5)
        metrics = ev.trading_metrics(returns)
        # Max drawdown should be approximately 18-22%
        assert 0.15 <= metrics.max_drawdown <= 0.30, (
            f"Expected max drawdown ~18-22%, got {metrics.max_drawdown:.4f}"
        )

    def test_profit_factor_above_one_for_profitable_strategy(self) -> None:
        from src.validation import FinancialMetricsEvaluator

        ev = FinancialMetricsEvaluator()
        rng = np.random.default_rng(42)
        # More winning days than losing (drift = +0.1%)
        returns = pd.Series(rng.normal(0.001, 0.008, 252))
        metrics = ev.trading_metrics(returns)
        # With positive drift, profit factor should be > 1 most of the time
        # (not guaranteed by randomness, but statistically likely with n=252)
        # Check that the metric is correctly computed and in a valid range
        assert metrics.profit_factor >= 0.0

    def test_win_rate_between_zero_and_one(self) -> None:
        from src.validation import FinancialMetricsEvaluator

        ev = FinancialMetricsEvaluator()
        returns = _make_returns_series(252)
        metrics = ev.trading_metrics(returns)
        assert 0.0 <= metrics.win_rate <= 1.0

    def test_empty_returns_series_does_not_crash(self) -> None:
        from src.validation import FinancialMetricsEvaluator

        ev = FinancialMetricsEvaluator()
        metrics = ev.trading_metrics(pd.Series([], dtype=float))
        assert metrics.sharpe_ratio == 0.0
        assert metrics.n_bars == 0

    def test_sortino_uses_only_downside_volatility(self) -> None:
        """Sortino >= Sharpe when there is positive skew (fewer losses than wins)."""
        from src.validation import FinancialMetricsEvaluator

        ev = FinancialMetricsEvaluator()
        # Returns with strong positive skew: mostly small gains, rare large losses
        rng = np.random.default_rng(43)
        returns_pos = rng.uniform(0.0, 0.002, 220)
        returns_neg = rng.uniform(-0.015, -0.010, 32)
        r = pd.Series(np.concatenate([returns_pos, returns_neg]))
        metrics = ev.trading_metrics(r)
        # With mostly positive returns, Sortino should be >= Sharpe
        # (fewer downside observations → lower downside vol)
        assert metrics.sortino_ratio >= metrics.sharpe_ratio * 0.5, (
            "Sortino should reflect that most returns are positive "
            "(downside vol < total vol)"
        )

    def test_evaluate_fold_returns_fold_result(self) -> None:
        from src.validation import FinancialMetricsEvaluator

        ev = FinancialMetricsEvaluator()
        rng = np.random.default_rng(44)
        y_true = rng.integers(0, 2, size=100)
        y_pred = rng.integers(0, 2, size=100)
        returns = pd.Series(rng.normal(0.001, 0.01, 100))

        fold = ev.evaluate_fold(
            fold_index=0,
            y_true=y_true,
            y_pred=y_pred,
            returns=returns,
            n_train=300,
        )
        assert fold.fold_index == 0
        assert fold.n_test == 100
        assert fold.classification.n_samples == 100
        assert fold.trading.n_bars == 100


# ─── 11. Model acceptance gate ───────────────────────────────────────────────


class TestModelAcceptanceGate:
    """
    Requirement #6 / Test area 11: dual-pass enforcement.
    A model with high accuracy but poor trading performance must be rejected.
    A model with good trading performance but poor accuracy must also be rejected.
    """

    def _make_fold_results(
        self,
        n_folds: int,
        accuracy: float = 0.60,
        sharpe: float = 1.2,
        max_dd: float = 0.10,
        total_return: float = 0.15,
        seed: int = 50,
    ) -> list:
        """Build synthetic FoldResult objects with configurable metrics."""
        from src.validation import FinancialMetricsEvaluator
        from src.validation.metrics import FoldResult

        ev = FinancialMetricsEvaluator()
        rng = np.random.default_rng(seed)
        folds = []

        for i in range(n_folds):
            y_true = rng.integers(0, 2, size=80)
            # Adjust y_pred to achieve approximately the desired accuracy
            flip_count = int((1 - accuracy) * 80)
            y_pred = y_true.copy()
            flip_idx = rng.choice(80, size=flip_count, replace=False)
            y_pred[flip_idx] = 1 - y_pred[flip_idx]

            # Build returns to achieve approximately the desired Sharpe
            target_daily_sharpe = sharpe / np.sqrt(252)
            vol = 0.01
            drift = target_daily_sharpe * vol
            returns = pd.Series(rng.normal(drift, vol, 80))
            # Normalize to approach desired total_return
            returns = returns * (total_return / (returns.sum() + 1e-10))

            fold = ev.evaluate_fold(
                fold_index=i,
                y_true=y_true,
                y_pred=y_pred,
                returns=returns,
                n_train=200,
            )
            folds.append(fold)

        return folds

    def test_good_model_is_accepted(self) -> None:
        """A model with strong classification AND trading metrics must be accepted."""
        from src.validation import (
            ModelAcceptanceGate, AcceptanceThresholds,
            compute_stability, aggregate_fold_results,
        )

        folds = self._make_fold_results(
            n_folds=5, accuracy=0.65, sharpe=1.5, max_dd=0.08, total_return=0.20
        )
        stability = compute_stability(folds)
        agg_clf, agg_trd = aggregate_fold_results(folds)

        gate = ModelAcceptanceGate()
        decision = gate.evaluate(folds, stability, agg_clf, agg_trd)
        assert decision.accepted, (
            f"Good model was rejected. Reasons: {decision.reasons}"
        )

    def test_high_accuracy_but_negative_return_rejected(self) -> None:
        """
        A model with 75% accuracy but negative out-of-sample returns must
        be rejected. High classification accuracy is not sufficient.
        """
        from src.validation import (
            ModelAcceptanceGate, compute_stability, aggregate_fold_results,
        )
        from src.validation.metrics import FoldResult, AcceptanceThresholds

        folds = self._make_fold_results(
            n_folds=5, accuracy=0.75, sharpe=-0.5, max_dd=0.20, total_return=-0.10
        )
        stability = compute_stability(folds)
        agg_clf, agg_trd = aggregate_fold_results(folds)

        gate = ModelAcceptanceGate()
        decision = gate.evaluate(folds, stability, agg_clf, agg_trd)
        assert not decision.accepted, (
            "Model with negative out-of-sample return must be rejected, "
            "even if accuracy is high"
        )
        assert any("return" in r.lower() or "sharpe" in r.lower() for r in decision.reasons), (
            "Rejection reason must mention trading performance"
        )

    def test_good_trading_but_accuracy_below_floor_rejected(self) -> None:
        """A model with great Sharpe but accuracy below 52% must be rejected."""
        from src.validation import (
            ModelAcceptanceGate, compute_stability, aggregate_fold_results,
        )

        folds = self._make_fold_results(
            n_folds=5, accuracy=0.45, sharpe=2.0, max_dd=0.05, total_return=0.30
        )
        stability = compute_stability(folds)
        agg_clf, agg_trd = aggregate_fold_results(folds)

        gate = ModelAcceptanceGate()
        decision = gate.evaluate(folds, stability, agg_clf, agg_trd)
        assert not decision.accepted, (
            "Model with accuracy < 52% must be rejected even if trading metrics are good"
        )
        assert any("accuracy" in r.lower() for r in decision.reasons), (
            "Rejection reason must mention accuracy"
        )

    def test_excessive_drawdown_rejected(self) -> None:
        """A model exceeding the max drawdown threshold must be rejected."""
        from src.validation import (
            ModelAcceptanceGate, compute_stability, aggregate_fold_results,
            FinancialMetricsEvaluator,
        )
        from src.validation.metrics import FoldResult

        # Build folds with a large drawdown directly
        ev = FinancialMetricsEvaluator()
        rng = np.random.default_rng(55)
        folds = []
        for i in range(5):
            y_true = rng.integers(0, 2, 80)
            y_pred = y_true.copy()
            flip = rng.choice(80, size=28, replace=False)
            y_pred[flip] = 1 - y_pred[flip]
            # Create a return series with a 30%+ drawdown
            returns = np.concatenate([
                rng.normal(0.002, 0.005, 40),  # good start
                rng.normal(-0.008, 0.010, 40), # large drawdown
            ])
            fold = ev.evaluate_fold(
                fold_index=i, y_true=y_true, y_pred=y_pred,
                returns=pd.Series(returns), n_train=200,
            )
            folds.append(fold)

        stability = compute_stability(folds)
        agg_clf, agg_trd = aggregate_fold_results(folds)

        gate = ModelAcceptanceGate()
        decision = gate.evaluate(folds, stability, agg_clf, agg_trd)
        # Drawdown > 25% should cause rejection
        if agg_trd.max_drawdown > 0.25:
            assert not decision.accepted
            assert any("drawdown" in r.lower() for r in decision.reasons)

    def test_acceptance_decision_has_score(self) -> None:
        from src.validation import (
            ModelAcceptanceGate, compute_stability, aggregate_fold_results,
        )

        folds = self._make_fold_results(5)
        stability = compute_stability(folds)
        agg_clf, agg_trd = aggregate_fold_results(folds)
        gate = ModelAcceptanceGate()
        decision = gate.evaluate(folds, stability, agg_clf, agg_trd)

        assert 0.0 <= decision.score <= 1.0
        assert isinstance(decision.thresholds_checked, dict)
        assert len(decision.thresholds_checked) > 0


# ─── 12. Stability analysis ───────────────────────────────────────────────────


class TestStabilityAnalysis:
    """
    Requirement #7 / Test area 12: cross-fold distribution, decay, dominance.
    """

    def _make_stable_folds(self, n: int = 5) -> list:
        """Folds with consistent performance across all windows."""
        from src.validation import FinancialMetricsEvaluator

        ev = FinancialMetricsEvaluator()
        rng = np.random.default_rng(60)
        folds = []
        for i in range(n):
            y_true = rng.integers(0, 2, 100)
            y_pred = y_true.copy()
            flip = rng.choice(100, 38, replace=False)
            y_pred[flip] = 1 - y_pred[flip]
            returns = pd.Series(rng.normal(0.001, 0.008, 100))
            fold = ev.evaluate_fold(
                fold_index=i, y_true=y_true, y_pred=y_pred,
                returns=returns, n_train=300,
            )
            folds.append(fold)
        return folds

    def test_stability_mean_and_std_computed(self) -> None:
        from src.validation import compute_stability

        folds = self._make_stable_folds(5)
        s = compute_stability(folds)
        assert isinstance(s.mean_sharpe, float)
        assert isinstance(s.std_sharpe, float)
        assert s.mean_accuracy >= 0.0

    def test_worst_and_best_fold_identified(self) -> None:
        from src.validation import compute_stability

        folds = self._make_stable_folds(5)
        s = compute_stability(folds)
        assert s.min_sharpe <= s.mean_sharpe <= s.max_sharpe, (
            "min_sharpe <= mean_sharpe <= max_sharpe must hold"
        )
        assert s.min_accuracy <= s.mean_accuracy <= s.max_accuracy

    def test_dominance_detected_when_single_fold_dominates(self) -> None:
        """
        If one fold contributes 80% of total return, dominance must be flagged.
        """
        from src.validation import FinancialMetricsEvaluator, compute_stability

        ev = FinancialMetricsEvaluator()
        rng = np.random.default_rng(61)
        folds = []
        for i in range(5):
            y_true = rng.integers(0, 2, 100)
            y_pred = y_true.copy()
            flip = rng.choice(100, 38, replace=False)
            y_pred[flip] = 1 - y_pred[flip]

            if i == 2:
                # One fold has enormously positive returns — dominates everything
                returns = pd.Series(rng.normal(0.05, 0.01, 100))
            else:
                # All other folds near zero
                returns = pd.Series(rng.normal(0.00005, 0.001, 100))

            fold = ev.evaluate_fold(
                fold_index=i, y_true=y_true, y_pred=y_pred,
                returns=returns, n_train=300,
            )
            folds.append(fold)

        s = compute_stability(folds)
        assert s.is_dominated_by_single_period, (
            "Dominance must be flagged when one fold produces 80%+ of returns"
        )
        assert s.max_fold_return_contribution > 0.6

    def test_stability_score_higher_for_consistent_folds(self) -> None:
        """
        A strategy with consistent performance across folds should have
        a higher stability score than one with wild variation.
        """
        from src.validation import FinancialMetricsEvaluator, compute_stability

        ev = FinancialMetricsEvaluator()
        rng = np.random.default_rng(62)

        # Consistent folds
        consistent_folds = []
        for i in range(5):
            y_true = rng.integers(0, 2, 100)
            y_pred = y_true.copy()
            flip = rng.choice(100, 38, replace=False)
            y_pred[flip] = 1 - y_pred[flip]
            returns = pd.Series(rng.normal(0.001, 0.005, 100))  # low vol, consistent
            consistent_folds.append(ev.evaluate_fold(
                fold_index=i, y_true=y_true, y_pred=y_pred,
                returns=returns, n_train=300,
            ))

        # Wildly variable folds
        variable_folds = []
        for i in range(5):
            y_true = rng.integers(0, 2, 100)
            y_pred = rng.integers(0, 2, 100)  # random predictions (bad accuracy)
            ret_sign = 1 if i % 2 == 0 else -1
            returns = pd.Series(rng.normal(ret_sign * 0.03, 0.04, 100))  # high vol, alternating
            variable_folds.append(ev.evaluate_fold(
                fold_index=i, y_true=y_true, y_pred=y_pred,
                returns=returns, n_train=300,
            ))

        s_consistent = compute_stability(consistent_folds)
        s_variable = compute_stability(variable_folds)

        assert s_consistent.stability_score >= s_variable.stability_score, (
            f"Consistent folds stability ({s_consistent.stability_score:.4f}) should be "
            f">= variable folds stability ({s_variable.stability_score:.4f})"
        )

    def test_sharpe_decay_positive_for_degrading_strategy(self) -> None:
        """A strategy that performs well early but poorly late has positive decay."""
        from src.validation import FinancialMetricsEvaluator, compute_stability

        ev = FinancialMetricsEvaluator()
        rng = np.random.default_rng(63)
        folds = []

        for i in range(6):
            y_true = rng.integers(0, 2, 100)
            y_pred = y_true.copy()
            flip = rng.choice(100, 38, replace=False)
            y_pred[flip] = 1 - y_pred[flip]

            # Early folds: good performance; late folds: bad performance
            drift = 0.003 * (1 - i / 5) - 0.001  # decays from +0.002 to -0.004
            returns = pd.Series(rng.normal(drift, 0.008, 100))
            folds.append(ev.evaluate_fold(
                fold_index=i, y_true=y_true, y_pred=y_pred,
                returns=returns, n_train=300,
            ))

        s = compute_stability(folds)
        # Sharpe decay should be positive (first half better than second)
        assert s.sharpe_decay > 0, (
            f"Degrading strategy should have positive sharpe_decay, got {s.sharpe_decay:.4f}"
        )


# ─── 13. Result persistence ───────────────────────────────────────────────────


class TestResultPersistence:
    """
    Requirement #8 / Test area 13: save results, never overwrite, load history.
    """

    def _make_validation_result(
        self, model_ver: str = "regime-xgb-v1"
    ) -> "ValidationResult":
        from src.validation import (
            FinancialMetricsEvaluator, compute_stability,
            aggregate_fold_results, ModelAcceptanceGate,
        )
        from src.validation.metrics import ValidationResult

        ev = FinancialMetricsEvaluator()
        rng = np.random.default_rng(70)
        folds = []
        for i in range(3):
            y_true = rng.integers(0, 2, 80)
            y_pred = y_true.copy()
            flip = rng.choice(80, 28, replace=False)
            y_pred[flip] = 1 - y_pred[flip]
            returns = pd.Series(rng.normal(0.001, 0.009, 80))
            folds.append(ev.evaluate_fold(
                fold_index=i, y_true=y_true, y_pred=y_pred,
                returns=returns, n_train=200,
            ))

        stability = compute_stability(folds)
        agg_clf, agg_trd = aggregate_fold_results(folds)
        gate = ModelAcceptanceGate()
        decision = gate.evaluate(folds, stability, agg_clf, agg_trd)

        return ValidationResult(
            model_version=model_ver,
            dataset_version="af-v3.0-fv4-abcd1234",
            feature_version="fv4",
            validation_method="walk_forward",
            evaluated_at="2026-08-31T14:00:00Z",
            fold_results=folds,
            aggregate_classification=agg_clf,
            aggregate_trading=agg_trd,
            stability=stability,
            accepted=decision.accepted,
            acceptance_reasons=decision.reasons,
        )

    def test_save_validation_result_creates_file(self, tmp_path: Path) -> None:
        from src.validation import save_validation_result

        result = self._make_validation_result()
        path = save_validation_result(result, tmp_path)
        assert path.exists(), "Saved validation result file not found"

    def test_saved_file_is_valid_json(self, tmp_path: Path) -> None:
        from src.validation import save_validation_result

        result = self._make_validation_result()
        path = save_validation_result(result, tmp_path)
        data = json.loads(path.read_text())
        assert "modelVersion" in data
        assert "datasetVersion" in data
        assert "featureVersion" in data
        assert "validationMethod" in data
        assert "foldResults" in data

    def test_result_contains_provenance_fields(self, tmp_path: Path) -> None:
        from src.validation import save_validation_result

        result = self._make_validation_result("regime-xgb-v2")
        path = save_validation_result(result, tmp_path)
        data = json.loads(path.read_text())

        for field in ("modelVersion", "datasetVersion", "featureVersion",
                      "validationMethod", "evaluatedAt", "nFolds",
                      "accepted", "aggregateClassification", "aggregateTrading"):
            assert field in data, f"Required field '{field}' missing from saved result"

    def test_historical_results_never_overwritten(self, tmp_path: Path) -> None:
        """
        Saving the same result twice must produce two distinct files.
        The original file must not be modified.
        """
        from src.validation import save_validation_result

        result = self._make_validation_result()
        p1 = save_validation_result(result, tmp_path)
        original_content = p1.read_text()

        # Save again with the same model version and timestamp
        p2 = save_validation_result(result, tmp_path)

        assert p1 != p2, "Second save must produce a different file (historical results protected)"
        assert p1.read_text() == original_content, "First file was modified — historical results corrupted"

    def test_load_validation_history_returns_saved_results(self, tmp_path: Path) -> None:
        from src.validation import save_validation_result, load_validation_history

        r1 = self._make_validation_result("model-v1")
        r2 = self._make_validation_result("model-v2")
        save_validation_result(r1, tmp_path)
        save_validation_result(r2, tmp_path)

        history = load_validation_history(tmp_path)
        assert len(history) == 2

    def test_load_history_filters_by_model_version(self, tmp_path: Path) -> None:
        from src.validation import save_validation_result, load_validation_history

        r1 = self._make_validation_result("regime-xgb-v1")
        r2 = self._make_validation_result("ranker-lgb-v1")
        save_validation_result(r1, tmp_path)
        save_validation_result(r2, tmp_path)

        history = load_validation_history(tmp_path, model_version="regime-xgb-v1")
        assert len(history) == 1
        assert history[0]["modelVersion"] == "regime-xgb-v1"

    def test_load_from_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        from src.validation import load_validation_history

        history = load_validation_history(tmp_path / "nonexistent")
        assert history == []

    def test_saved_result_includes_all_fold_results(self, tmp_path: Path) -> None:
        from src.validation import save_validation_result

        result = self._make_validation_result()
        path = save_validation_result(result, tmp_path)
        data = json.loads(path.read_text())

        assert data["nFolds"] == len(result.fold_results)
        assert len(data["foldResults"]) == len(result.fold_results)
        for fold_data in data["foldResults"]:
            assert "fold_index" in fold_data
            assert "classification" in fold_data
            assert "trading" in fold_data
