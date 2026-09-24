"""
test_feature_factory.py — Phase C regression tests for FeatureFactory.

Key properties:
- PIT-safety: feature[t] depends only on bars <= t (verified by the truncation
  invariant: recomputing on data[:k] yields identical feature values for rows < k).
- Missing values are NaN, never silently zero.
- Deterministic / idempotent.
- Declared schema is respected.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.features.factory import FeatureFactory

UTC = timezone.utc


def _ohlcv(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(
        [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    )
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_schema_respected():
    factory = FeatureFactory()
    features, _ = factory.build(_ohlcv())
    assert list(features.columns) == FeatureFactory.FEATURE_NAMES


def test_idempotent():
    factory = FeatureFactory()
    df = _ohlcv()
    f1, _ = factory.build(df)
    f2, _ = factory.build(df)
    pd.testing.assert_frame_equal(f1, f2)


def test_pit_safety_truncation_invariant():
    """
    Feature values for early rows must NOT change when future bars are appended.
    Recompute on a truncated prefix and compare the overlapping region.
    """
    factory = FeatureFactory()
    df = _ohlcv(n=120)
    full, _ = factory.build(df)
    prefix, _ = factory.build(df.iloc[:80])
    # Compare rows 0..79 for a rolling feature that is fully warmed up.
    common = full.iloc[:80][["ret_5", "rsi_14", "vol_20", "macd_hist"]]
    pref = prefix[["ret_5", "rsi_14", "vol_20", "macd_hist"]]
    pd.testing.assert_frame_equal(common, pref)


def test_missing_values_are_nan_not_zero():
    factory = FeatureFactory()
    df = _ohlcv(n=120)
    features, _ = factory.build(df)
    # ret_20 needs 20 bars → first 20 rows must be NaN, not 0.
    assert features["ret_20"].iloc[:20].isna().all()
    assert not (features["ret_20"].iloc[:20] == 0).any()


def test_availability_metadata():
    factory = FeatureFactory()
    df = _ohlcv(n=120)
    _, avail = factory.build(df)
    assert avail.total_rows == 120
    # vol_20 has ~20 missing warmup rows.
    assert avail.per_feature_missing["vol_20"] >= 19
    assert 0.0 <= avail.missing_fraction("vol_20") <= 1.0


def test_no_inf_values():
    factory = FeatureFactory()
    features, _ = factory.build(_ohlcv())
    # No infinities should remain (replaced with NaN).
    assert not np.isinf(features.to_numpy(dtype=float)[~np.isnan(features.to_numpy(dtype=float))]).any()


def test_warmed_up_features_finite():
    factory = FeatureFactory()
    features, _ = factory.build(_ohlcv(n=200))
    tail = features.iloc[-1]
    # After 200 bars, all features should be defined (finite).
    assert tail.notna().all(), f"NaN features at tail: {tail[tail.isna()].index.tolist()}"
