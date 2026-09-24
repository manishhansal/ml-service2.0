"""
test_label_factory.py — Phase D regression tests for LabelFactory (P0-002).

Verifies fixed-horizon, triple-barrier, vol-adjusted, and meta-labels, plus
PIT-safety (labels look forward and the last `horizon` rows are unresolved),
cost incorporation, MAE/MFE, and label-quality diagnostics.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.data.labels import LabelConfig, LabelFactory, label_quality_report

UTC = timezone.utc


def _ohlcv(closes, highs=None, lows=None):
    n = len(closes)
    idx = pd.DatetimeIndex(
        [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    )
    highs = highs or [c * 1.01 for c in closes]
    lows = lows or [c * 0.99 for c in closes]
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes,
         "volume": [1000] * n},
        index=idx,
    )


def test_fixed_horizon_forward_looking_and_pit_safe():
    closes = [100, 101, 102, 103, 104, 105]
    df = _ohlcv(closes)
    factory = LabelFactory(LabelConfig(label_type="fixed_horizon", horizon=2, cost_bps=0))
    labels = factory.build(df)
    # Row 0: (102-100)/100 = 0.02 → UP
    assert labels["realized_return"].iloc[0] == pytest.approx(0.02)
    assert labels["label"].iloc[0] == 1
    # Last `horizon` rows have no forward data → unresolved.
    assert pd.isna(labels["label"].iloc[-1])
    assert labels["outcome"].iloc[-1] == "UNRESOLVED"


def test_fixed_horizon_incorporates_cost():
    closes = [100, 100.5, 101]  # +1% over horizon 2
    df = _ohlcv(closes)
    factory = LabelFactory(LabelConfig(label_type="fixed_horizon", horizon=2, cost_bps=50))
    labels = factory.build(df)
    gross = labels["realized_return"].iloc[0]
    net = labels["realized_return_net"].iloc[0]
    assert net == pytest.approx(gross - 0.005)


def test_triple_barrier_target_hit():
    # Sharp upward move should hit the +2% upper barrier.
    closes = [100, 103, 103, 103]
    highs = [100, 103, 103, 103]
    lows = [100, 102, 102, 102]
    df = _ohlcv(closes, highs, lows)
    factory = LabelFactory(LabelConfig(
        label_type="triple_barrier", horizon=3,
        upper_barrier_pct=0.02, lower_barrier_pct=0.02, cost_bps=0))
    labels = factory.build(df)
    assert labels["outcome"].iloc[0] == "TARGET_HIT"
    assert labels["label"].iloc[0] == 1


def test_triple_barrier_stop_hit():
    closes = [100, 97, 97, 97]
    highs = [100, 98, 98, 98]
    lows = [100, 97, 97, 97]
    df = _ohlcv(closes, highs, lows)
    factory = LabelFactory(LabelConfig(
        label_type="triple_barrier", horizon=3,
        upper_barrier_pct=0.02, lower_barrier_pct=0.02, cost_bps=0))
    labels = factory.build(df)
    assert labels["outcome"].iloc[0] == "STOP_HIT"
    assert labels["label"].iloc[0] == 0


def test_triple_barrier_mae_mfe():
    closes = [100, 101, 102, 101]
    highs = [100, 103, 104, 102]
    lows = [100, 98, 99, 100]
    df = _ohlcv(closes, highs, lows)
    factory = LabelFactory(LabelConfig(
        label_type="triple_barrier", horizon=3,
        upper_barrier_pct=0.10, lower_barrier_pct=0.10, cost_bps=0))
    labels = factory.build(df)
    # MFE should be positive (price rose to 104), MAE negative (fell to 98).
    assert labels["mfe"].iloc[0] > 0
    assert labels["mae"].iloc[0] < 0


def test_vol_adjusted_barrier_runs():
    np.random.seed(0)
    closes = list(100 + np.cumsum(np.random.randn(60)))
    df = _ohlcv(closes)
    factory = LabelFactory(LabelConfig(
        label_type="vol_adjusted_barrier", horizon=5, vol_window=20, cost_bps=10))
    labels = factory.build(df)
    assert "upper_barrier" in labels.columns
    assert labels["label"].notna().sum() > 0


def test_meta_labels():
    closes = [100, 102, 103, 103, 103, 103]
    highs = [100, 103, 103, 103, 103, 103]
    lows = [100, 100, 100, 100, 100, 100]
    df = _ohlcv(closes, highs, lows)
    factory = LabelFactory(LabelConfig(
        label_type="triple_barrier", horizon=3,
        upper_barrier_pct=0.02, lower_barrier_pct=0.02, cost_bps=0))
    primary = pd.Series(1, index=df.index)  # always long
    meta = factory.build_meta_labels(df, primary)
    assert "label" in meta.columns
    assert "primary_side" in meta.columns
    assert meta["label"].iloc[0] in (0, 1)


def test_label_quality_report():
    closes = list(range(100, 140))
    df = _ohlcv(closes)
    factory = LabelFactory(LabelConfig(label_type="fixed_horizon", horizon=3, cost_bps=0))
    labels = factory.build(df)
    report = label_quality_report(labels)
    assert report["n_resolved"] > 0
    assert 0.0 <= report["class_balance_positive"] <= 1.0
    assert "label_autocorrelation_lag1" in report
