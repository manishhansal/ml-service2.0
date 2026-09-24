"""
test_dataset_builder.py — dataset construction + freeze + leakage validation.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.data.dataset_builder import DatasetBuilder
from src.data.labels import LabelConfig

UTC = timezone.utc


def _ohlcv(n=260, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(
        [datetime(2022, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    )
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.abs(close) + 10
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_build_freeze_and_load(tmp_path):
    builder = DatasetBuilder(output_root=tmp_path, run_leakage_validation=False)
    meta = builder.build(
        {"NIFTY": _ohlcv(seed=1), "RELIANCE": _ohlcv(seed=2)},
        LabelConfig(label_type="triple_barrier", horizon=5, cost_bps=10),
        timeframe="1d",
    )
    assert meta.row_count > 0
    assert meta.dataset_hash
    assert meta.feature_count == 24
    assert meta.code_sha  # git sha or "unknown"
    X, y, ts = builder.load(meta.dataset_id)
    assert X.shape[0] == meta.row_count
    assert len(y) == meta.row_count
    assert len(ts) == meta.row_count
    # No NaN should remain in the frozen feature matrix.
    assert not np.isnan(X).any()


def test_metadata_persisted(tmp_path):
    builder = DatasetBuilder(output_root=tmp_path, run_leakage_validation=False)
    meta = builder.build(
        {"NIFTY": _ohlcv(seed=1)},
        LabelConfig(label_type="fixed_horizon", horizon=3),
    )
    loaded = builder.load_metadata(meta.dataset_id)
    assert loaded.dataset_id == meta.dataset_id
    assert loaded.dataset_hash == meta.dataset_hash
    assert "NIFTY" in loaded.universe
    assert loaded.label_quality["n_resolved"] > 0


def test_dataset_hash_deterministic(tmp_path):
    b1 = DatasetBuilder(output_root=tmp_path / "a", run_leakage_validation=False)
    b2 = DatasetBuilder(output_root=tmp_path / "b", run_leakage_validation=False)
    cfg = LabelConfig(label_type="triple_barrier", horizon=5)
    m1 = b1.build({"NIFTY": _ohlcv(seed=7)}, cfg)
    m2 = b2.build({"NIFTY": _ohlcv(seed=7)}, cfg)
    assert m1.dataset_hash == m2.dataset_hash


def test_leakage_validation_runs(tmp_path):
    builder = DatasetBuilder(output_root=tmp_path, run_leakage_validation=True)
    meta = builder.build(
        {"NIFTY": _ohlcv(seed=3)},
        LabelConfig(label_type="triple_barrier", horizon=5),
    )
    # PIT-safe features should pass leakage validation.
    assert meta.leakage_validated is True
    assert meta.pit_status == "PIT_VALIDATED"


def test_leakage_validator_catches_egregious_leak():
    """A feature that literally equals the forward label must be flagged."""
    from src.features.leakage_validator import LeakageValidator, PITViolationError

    n = 120
    idx = pd.RangeIndex(n)
    rng = np.random.default_rng(0)
    fwd_return = pd.Series(rng.normal(0, 0.01, n), index=idx)
    # Leaky feature = the future return itself (perfect look-ahead).
    leaky = pd.DataFrame({"leaky_feature": fwd_return.shift(-1)}, index=idx)
    validator = LeakageValidator(threshold=0.95)
    with pytest.raises(PITViolationError):
        validator.validate(leaky, fwd_return)
