"""
test_label_factory.py — Phase D regression tests for LabelFactory (P0-002).

Verifies fixed-horizon, triple-barrier, vol-adjusted, and meta-labels, plus
PIT-safety (labels look forward and the last `horizon` rows are unresolved),
cost incorporation, MAE/MFE, and label-quality diagnostics.

Execution model tests (mandate §20):
  - Tests with execution_model="close_to_close" verify the old numeric behaviour
    (entry=close[T], exit=close[T+h]).
  - Tests with execution_model="next_open" (default) verify the economically
    valid behaviour (entry=open[T+1], exit=open[T+1+h]).
  - is_economic_evidence=False for close_to_close datasets.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.data.labels import LabelConfig, LabelFactory, label_quality_report

UTC = timezone.utc


def _ohlcv(closes, highs=None, lows=None, opens=None):
    n = len(closes)
    idx = pd.DatetimeIndex(
        [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    )
    highs = highs or [c * 1.01 for c in closes]
    lows = lows or [c * 0.99 for c in closes]
    opens = opens or closes  # default: open = close (for legacy tests using c2c mode)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [1000] * n},
        index=idx,
    )


# ── Fixed-horizon ──────────────────────────────────────────────────────────────


def test_fixed_horizon_close_to_close():
    """close_to_close mode: entry=close[T], exit=close[T+h]. Research diagnostic only."""
    closes = [100, 101, 102, 103, 104, 105]
    df = _ohlcv(closes)
    factory = LabelFactory(LabelConfig(
        label_type="fixed_horizon", horizon=2, cost_bps=0,
        execution_model="close_to_close",
    ))
    labels = factory.build(df)
    # Row 0: (close[2]-close[0])/close[0] = (102-100)/100 = 0.02
    assert labels["realized_return"].iloc[0] == pytest.approx(0.02)
    assert labels["label"].iloc[0] == 1
    # is_economic_evidence must be False for close_to_close (mandate §20)
    assert labels["is_economic_evidence"].iloc[0] == False  # noqa: E712
    # Last `horizon` rows have no forward data → unresolved
    assert pd.isna(labels["label"].iloc[-1])
    assert labels["outcome"].iloc[-1] == "UNRESOLVED"


def test_fixed_horizon_next_open():
    """next_open mode (default, mandate §20): entry=open[T+1], exit=open[T+1+h]."""
    # open[T+1]=101, open[T+1+2]=open[T+3]=103
    # realized = (103-101)/101 ≈ 0.0198
    opens = [100, 101, 102, 103, 104, 105]
    closes = [100, 101, 102, 103, 104, 105]
    df = _ohlcv(closes, opens=opens)
    factory = LabelFactory(LabelConfig(
        label_type="fixed_horizon", horizon=2, cost_bps=0,
        execution_model="next_open",
    ))
    labels = factory.build(df)
    expected = (103 - 101) / 101
    assert labels["realized_return"].iloc[0] == pytest.approx(expected)
    # is_economic_evidence must be True for next_open (mandate §20)
    assert labels["is_economic_evidence"].iloc[0] == True  # noqa: E712
    # execution_model column preserved
    assert labels["execution_model"].iloc[0] == "next_open"


def test_fixed_horizon_default_is_next_open():
    """Default LabelConfig must use next_open (mandate §20)."""
    cfg = LabelConfig(label_type="fixed_horizon", horizon=2)
    assert cfg.execution_model == "next_open"


def test_fixed_horizon_incorporates_cost():
    """Cost is subtracted from both modes."""
    closes = [100, 100.5, 101]
    df = _ohlcv(closes)
    factory = LabelFactory(LabelConfig(
        label_type="fixed_horizon", horizon=2, cost_bps=50,
        execution_model="close_to_close",
    ))
    labels = factory.build(df)
    gross = labels["realized_return"].iloc[0]
    net = labels["realized_return_net"].iloc[0]
    assert net == pytest.approx(gross - 0.005)


# ── Triple-barrier ─────────────────────────────────────────────────────────────


def test_triple_barrier_target_hit():
    """Sharp upward move should hit the +2% upper barrier (close_to_close for simplicity)."""
    closes = [100, 103, 103, 103]
    highs = [100, 103, 103, 103]
    lows = [100, 102, 102, 102]
    df = _ohlcv(closes, highs, lows)
    factory = LabelFactory(LabelConfig(
        label_type="triple_barrier", horizon=3,
        upper_barrier_pct=0.02, lower_barrier_pct=0.02, cost_bps=0,
        execution_model="close_to_close",
    ))
    labels = factory.build(df)
    assert labels["outcome"].iloc[0] == "TARGET_HIT"
    assert labels["label"].iloc[0] == 1


def test_triple_barrier_stop_hit():
    """Sharp downward move should hit the -2% lower barrier."""
    closes = [100, 97, 97, 97]
    highs = [100, 98, 98, 98]
    lows = [100, 97, 97, 97]
    df = _ohlcv(closes, highs, lows)
    factory = LabelFactory(LabelConfig(
        label_type="triple_barrier", horizon=3,
        upper_barrier_pct=0.02, lower_barrier_pct=0.02, cost_bps=0,
        execution_model="close_to_close",
    ))
    labels = factory.build(df)
    assert labels["outcome"].iloc[0] == "STOP_HIT"
    assert labels["label"].iloc[0] == 0


def test_triple_barrier_mae_mfe():
    """MAE should be negative, MFE positive when price oscillates."""
    closes = [100, 101, 102, 101]
    highs = [100, 103, 104, 102]
    lows = [100, 98, 99, 100]
    df = _ohlcv(closes, highs, lows)
    factory = LabelFactory(LabelConfig(
        label_type="triple_barrier", horizon=3,
        upper_barrier_pct=0.10, lower_barrier_pct=0.10, cost_bps=0,
        execution_model="close_to_close",
    ))
    labels = factory.build(df)
    assert labels["mfe"].iloc[0] > 0
    assert labels["mae"].iloc[0] < 0


def test_triple_barrier_next_open_economic_evidence():
    """next_open triple-barrier labels carry is_economic_evidence=True."""
    closes = [100, 102, 103, 103, 103, 103]
    highs = [100, 103, 103, 103, 103, 103]
    lows = [100, 100, 100, 100, 100, 100]
    opens = [100, 101, 102, 103, 103, 103]
    df = _ohlcv(closes, highs, lows, opens)
    factory = LabelFactory(LabelConfig(
        label_type="triple_barrier", horizon=3,
        upper_barrier_pct=0.02, lower_barrier_pct=0.02, cost_bps=0,
        execution_model="next_open",
    ))
    labels = factory.build(df)
    # is_economic_evidence should be True for all rows
    assert labels["is_economic_evidence"].iloc[0] == True  # noqa: E712
    assert labels["execution_model"].iloc[0] == "next_open"


def test_vol_adjusted_barrier_runs():
    np.random.seed(0)
    closes = list(100 + np.cumsum(np.random.randn(60)))
    df = _ohlcv(closes)
    factory = LabelFactory(LabelConfig(
        label_type="vol_adjusted_barrier", horizon=5, vol_window=20, cost_bps=10,
        execution_model="close_to_close",
    ))
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
        upper_barrier_pct=0.02, lower_barrier_pct=0.02, cost_bps=0,
        execution_model="close_to_close",
    ))
    primary = pd.Series(1, index=df.index)  # always long
    meta = factory.build_meta_labels(df, primary)
    assert "label" in meta.columns
    assert "primary_side" in meta.columns
    assert meta["label"].iloc[0] in (0, 1)
    assert "execution_model" in meta.columns
    assert "is_economic_evidence" in meta.columns


# ── Execution model invariants (mandate §20) ──────────────────────────────────


def test_close_to_close_is_not_economic_evidence():
    """close_to_close labels MUST carry is_economic_evidence=False (mandate §20)."""
    closes = [100, 101, 102, 103]
    df = _ohlcv(closes)
    factory = LabelFactory(LabelConfig(
        label_type="fixed_horizon", horizon=1, cost_bps=0,
        execution_model="close_to_close",
    ))
    labels = factory.build(df)
    assert all(labels["is_economic_evidence"] == False)  # noqa: E712


def test_next_open_is_economic_evidence():
    """next_open labels MUST carry is_economic_evidence=True (mandate §20)."""
    closes = [100, 101, 102, 103]
    df = _ohlcv(closes)
    factory = LabelFactory(LabelConfig(
        label_type="fixed_horizon", horizon=1, cost_bps=0,
        execution_model="next_open",
    ))
    labels = factory.build(df)
    assert all(labels["is_economic_evidence"] == True)  # noqa: E712


def test_next_open_requires_open_column():
    """next_open mode raises ValueError if 'open' column is missing."""
    closes = [100, 101, 102, 103]
    idx = pd.DatetimeIndex(
        [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(4)]
    )
    df = pd.DataFrame({"close": closes, "volume": [1000] * 4}, index=idx)
    factory = LabelFactory(LabelConfig(
        label_type="fixed_horizon", horizon=1, cost_bps=0,
        execution_model="next_open",
    ))
    with pytest.raises(ValueError, match="open"):
        factory.build(df)


def test_execution_model_column_preserved():
    """execution_model column is present in output for both modes."""
    closes = [100, 101, 102, 103]
    df = _ohlcv(closes)
    for mode in ("close_to_close", "next_open"):
        factory = LabelFactory(LabelConfig(
            label_type="fixed_horizon", horizon=1, cost_bps=0,
            execution_model=mode,
        ))
        labels = factory.build(df)
        assert "execution_model" in labels.columns
        assert all(labels["execution_model"] == mode)


# ── Label quality ─────────────────────────────────────────────────────────────


def test_label_quality_report():
    closes = list(range(100, 140))
    df = _ohlcv(closes)
    factory = LabelFactory(LabelConfig(
        label_type="fixed_horizon", horizon=3, cost_bps=0,
        execution_model="close_to_close",
    ))
    labels = factory.build(df)
    report = label_quality_report(labels)
    assert report["n_resolved"] > 0
    assert 0.0 <= report["class_balance_positive"] <= 1.0
    assert "label_autocorrelation_lag1" in report
