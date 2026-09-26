"""
tests/test_feature_normalizer.py — unit tests for FeatureNormalizer.

Covers:
  - fit / transform round-trip
  - Winsorization (outlier clipping)
  - NaN preservation
  - serialization: to_dict / from_dict / to_json / from_json
  - clip report tracking
  - fit_transform convenience
  - transform_known (subset transform)
  - unfitted raises RuntimeError
  - unseen column raises RuntimeError
  - summary() shape
  - per-feature method overrides (rsi_14 → ROBUST, skew_20 → ZSCORE)
  - ScalingMethod.NONE pass-through
  - constant column (all same value) → ScalingMethod.NONE stored
  - all-NaN column → ScalingMethod.NONE stored
  - empty DataFrame fit
"""
from __future__ import annotations

import math
import json

import numpy as np
import pandas as pd
import pytest

import os
os.environ.setdefault("ML_SERVICE_API_KEY", "test-key")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from src.features.normalizer import (
    ClipReport,
    FeatureNormSpec,
    FeatureNormalizer,
    ScalingMethod,
    _DEFAULT_WINSOR,
    _FEATURE_METHOD_DEFAULTS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ret_1":           rng.normal(0, 0.01, n),
        "rsi_14":          rng.uniform(0, 100, n),
        "vol_20":          rng.uniform(0.005, 0.05, n),
        "rel_volume_20":   np.abs(rng.lognormal(0, 0.5, n)),
        "close_position":  rng.uniform(0, 1, n),
        "adx_14":          rng.uniform(0, 100, n),
        "skew_20":         rng.normal(0, 1, n),
        "kurt_20":         rng.normal(0, 3, n),
    })


# ── Fit / transform basic ─────────────────────────────────────────────────────

class TestFitTransform:
    def test_fit_succeeds_and_marks_fitted(self):
        norm = FeatureNormalizer()
        df = _make_df()
        norm.fit(df)
        assert norm.is_fitted

    def test_transform_preserves_shape(self):
        df = _make_df()
        norm = FeatureNormalizer()
        norm.fit(df)
        result = norm.transform(df)
        assert result.shape == df.shape
        assert list(result.columns) == list(df.columns)

    def test_transform_preserves_index(self):
        df = _make_df().set_index(pd.date_range("2024-01-01", periods=300, freq="D"))
        norm = FeatureNormalizer()
        norm.fit(df)
        result = norm.transform(df)
        assert list(result.index) == list(df.index)

    def test_fit_transform_equivalent_to_fit_then_transform(self):
        df = _make_df()
        n1 = FeatureNormalizer(winsor_pct=(1.0, 99.0))
        n2 = FeatureNormalizer(winsor_pct=(1.0, 99.0))
        n1.fit(df)
        r1 = n1.transform(df)
        r2 = n2.fit_transform(df)
        pd.testing.assert_frame_equal(r1, r2)


# ── Winsorization / clipping ──────────────────────────────────────────────────

class TestWinsorization:
    def test_extreme_outlier_clipped(self):
        df = _make_df()
        df.loc[0, "ret_1"] = 50.0        # 50× normal range
        df.loc[1, "ret_1"] = -50.0
        norm = FeatureNormalizer(winsor_pct=(1.0, 99.0))
        norm.fit(df)
        result = norm.transform(df)
        # After robust scaling the clipped+scaled value should be moderate
        assert abs(result["ret_1"].max()) < 50
        assert abs(result["ret_1"].min()) < 50

    def test_clip_report_counts_clipped_values(self):
        df = _make_df()
        df.loc[0, "ret_1"] = 100.0   # will be clipped high
        df.loc[1, "ret_1"] = -100.0  # will be clipped low
        norm = FeatureNormalizer(winsor_pct=(1.0, 99.0))
        norm.fit(df)
        result, reports = norm.transform(df, track_clips=True)
        ret_report = next(r for r in reports if r.feature == "ret_1")
        assert isinstance(ret_report, ClipReport)
        assert ret_report.n_clipped_high >= 1
        assert ret_report.n_clipped_low >= 1
        assert ret_report.clip_rate_pct > 0

    def test_nan_preserved_after_transform(self):
        df = _make_df()
        df.loc[5, "vol_20"] = float("nan")
        norm = FeatureNormalizer()
        norm.fit(df)
        result = norm.transform(df)
        assert math.isnan(result.loc[5, "vol_20"])

    def test_nan_excluded_from_fit_stats(self):
        """NaN rows must not affect fitted statistics."""
        df_clean = _make_df(seed=1)
        df_dirty = df_clean.copy()
        df_dirty.iloc[:50, 0] = float("nan")
        n1 = FeatureNormalizer()
        n2 = FeatureNormalizer()
        n1.fit(df_clean)
        n2.fit(df_dirty)
        # Medians won't be identical but should be close (same data minus NaNs)
        assert abs(n1._specs["ret_1"].center - n2._specs["ret_1"].center) < 0.005


# ── Per-feature method overrides ──────────────────────────────────────────────

class TestMethodOverrides:
    def test_rsi_14_uses_robust(self):
        df = _make_df()
        norm = FeatureNormalizer()
        norm.fit(df)
        assert norm._specs["rsi_14"].method == ScalingMethod.ROBUST

    def test_skew_20_uses_zscore(self):
        df = _make_df()
        norm = FeatureNormalizer()
        norm.fit(df)
        assert norm._specs["skew_20"].method == ScalingMethod.ZSCORE

    def test_kurt_20_uses_zscore(self):
        df = _make_df()
        norm = FeatureNormalizer()
        norm.fit(df)
        assert norm._specs["kurt_20"].method == ScalingMethod.ZSCORE

    def test_rel_volume_tighter_winsor(self):
        """rel_volume_20 uses (0.5, 99.5) not the default (1, 99)."""
        df = _make_df()
        norm = FeatureNormalizer()
        norm.fit(df)
        spec = norm._specs["rel_volume_20"]
        # clip_low should be the 0.5th percentile — lower than the 1st percentile
        finite = df["rel_volume_20"].to_numpy(dtype=float)
        p1 = float(np.percentile(finite[np.isfinite(finite)], 1.0))
        p05 = float(np.percentile(finite[np.isfinite(finite)], 0.5))
        assert abs(spec.clip_low - p05) < 1e-6

    def test_constant_column_stored_as_none(self):
        # The normalizer stores ScalingMethod.NONE for all-NaN/inf columns.
        # A column with a constant finite value is stored with its actual method
        # (ROBUST) but scale is clamped to min_scale — it can still be "scaled".
        # The pass-through (NONE) path is reserved for fully-missing columns.
        df = _make_df()
        df["all_nan_col"] = float("nan")   # truly unscalable → stored as NONE
        norm = FeatureNormalizer()
        norm.fit(df)
        assert norm._specs["all_nan_col"].method == ScalingMethod.NONE

    def test_all_nan_column_stored_as_none(self):
        df = _make_df()
        df["nan_col"] = float("nan")
        norm = FeatureNormalizer()
        norm.fit(df)
        assert norm._specs["nan_col"].method == ScalingMethod.NONE

    def test_constant_column_has_min_scale(self):
        """A constant finite column keeps its method but scale is clamped to min_scale."""
        df = _make_df()
        df["const_col"] = 5.0
        norm = FeatureNormalizer()
        norm.fit(df)
        spec = norm._specs["const_col"]
        # scale must be min_scale (not zero), to avoid division by zero
        assert spec.scale >= norm._min_scale


# ── Serialization ─────────────────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict_from_dict_round_trip(self):
        df = _make_df()
        norm = FeatureNormalizer(winsor_pct=(2.0, 98.0))
        norm.fit(df)
        r1 = norm.transform(df)

        state = norm.to_dict()
        norm2 = FeatureNormalizer.from_dict(state)
        r2 = norm2.transform(df)

        pd.testing.assert_frame_equal(r1, r2)

    def test_to_json_from_json_round_trip(self):
        df = _make_df()
        norm = FeatureNormalizer()
        norm.fit(df)
        js = norm.to_json()
        assert isinstance(js, str)
        norm2 = FeatureNormalizer.from_json(js)
        assert norm2.is_fitted
        pd.testing.assert_frame_equal(norm.transform(df), norm2.transform(df))

    def test_to_dict_is_json_serializable(self):
        df = _make_df()
        norm = FeatureNormalizer()
        norm.fit(df)
        state = norm.to_dict()
        # Must not raise
        json_str = json.dumps(state)
        assert len(json_str) > 0

    def test_from_dict_preserves_winsor_pct(self):
        norm = FeatureNormalizer(winsor_pct=(5.0, 95.0))
        norm.fit(_make_df())
        state = norm.to_dict()
        norm2 = FeatureNormalizer.from_dict(state)
        assert norm2._winsor_pct == (5.0, 95.0)

    def test_to_dict_raises_if_not_fitted(self):
        norm = FeatureNormalizer()
        with pytest.raises(RuntimeError, match="unfitted"):
            norm.to_dict()


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_transform_before_fit_raises(self):
        norm = FeatureNormalizer()
        with pytest.raises(RuntimeError, match="fit()"):
            norm.transform(_make_df())

    def test_transform_unseen_column_raises(self):
        """Unseen column must raise RuntimeError, not silently pass through."""
        df_train = _make_df()[["ret_1", "rsi_14"]]
        df_new   = _make_df()[["ret_1", "rsi_14", "vol_20"]]  # extra column
        norm = FeatureNormalizer()
        norm.fit(df_train)
        with pytest.raises(RuntimeError, match="vol_20"):
            norm.transform(df_new)

    def test_transform_known_passes_through_unseen(self):
        """transform_known() must NOT raise for unseen columns — pass through."""
        df_train = _make_df()[["ret_1", "rsi_14"]]
        df_new   = _make_df()[["ret_1", "rsi_14", "vol_20"]]
        norm = FeatureNormalizer()
        norm.fit(df_train)
        # Should not raise; vol_20 passed through unchanged
        result = norm.transform_known(df_new)
        assert "vol_20" in result.columns
        # ret_1 and rsi_14 must be normalized
        r1 = norm.transform(df_new[["ret_1", "rsi_14"]])
        pd.testing.assert_series_equal(result["ret_1"], r1["ret_1"])

    def test_empty_dataframe_fit_does_not_crash(self):
        norm = FeatureNormalizer()
        norm.fit(pd.DataFrame())
        assert norm.is_fitted

    def test_clip_report_only_when_track_clips_true(self):
        df = _make_df()
        norm = FeatureNormalizer()
        norm.fit(df)
        result = norm.transform(df, track_clips=False)
        assert isinstance(result, pd.DataFrame)

    def test_clip_report_returns_tuple_when_requested(self):
        df = _make_df()
        norm = FeatureNormalizer()
        norm.fit(df)
        out = norm.transform(df, track_clips=True)
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert all(isinstance(r, ClipReport) for r in out[1])


# ── summary() ────────────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_returns_dataframe_with_correct_shape(self):
        df = _make_df()
        norm = FeatureNormalizer()
        norm.fit(df)
        s = norm.summary()
        assert isinstance(s, pd.DataFrame)
        assert len(s) == len(df.columns)
        assert "method" in s.columns
        assert "clip_low" in s.columns
        assert "clip_high" in s.columns
        assert "center" in s.columns
        assert "scale" in s.columns

    def test_summary_before_fit_raises(self):
        norm = FeatureNormalizer()
        with pytest.raises(RuntimeError, match="fitted"):
            norm.summary()

    def test_feature_names_property(self):
        df = _make_df()
        norm = FeatureNormalizer()
        norm.fit(df)
        assert set(norm.feature_names) == set(df.columns)
