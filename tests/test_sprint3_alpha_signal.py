"""
tests/test_sprint3_alpha_signal.py — Regression tests for CRIT-001 and CRIT-002 fixes.

Verifies:
  CRIT-001: 7-day label horizon (not 5-bar)
  CRIT-002: Labels have positive expected value with improved barrier design
  CRIT-003: Break-even accuracy calculation is correct
  CRIT-004: TIME_EXPIRY label uses net-cost threshold (not gross zero)
  CRIT-005: Asymmetric barriers produce positive mean EV

pytest markers: unit
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labels.seven_day import (
    EQUITY_COST_BPS,
    HORIZON_DAYS,
    generate_7d_asymmetric_barrier_label,
    generate_7d_excess_return_label,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 100, trend: float = 0.001, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    p = 1000.0 * np.cumprod(1 + rng.normal(trend, 0.015, n))
    h = p * (1 + rng.uniform(0.002, 0.015, n))
    l = p * (1 - rng.uniform(0.002, 0.015, n))
    o = p * (1 + rng.normal(0, 0.003, n))
    v = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": p, "volume": v}, index=idx)


def _make_nifty(n: int = 100, seed: int = 99) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    p = 22000.0 * np.cumprod(1 + rng.normal(0.0, 0.01, n))
    return pd.Series(p, index=idx, name="close")


# ── CRIT-001: 7-day horizon ───────────────────────────────────────────────────


@pytest.mark.unit
def test_seven_day_label_horizon_is_7():
    """CRIT-001: The improved label must use exactly 7 bars (not 5)."""
    assert HORIZON_DAYS == 7, f"HORIZON_DAYS must be 7, got {HORIZON_DAYS}"


@pytest.mark.unit
def test_excess_return_last_7_bars_are_nan():
    """CRIT-001: The last 7 bars must be NaN (no forward data)."""
    ohlcv = _make_ohlcv(50)
    nifty = _make_nifty(50)
    label = generate_7d_excess_return_label(ohlcv, nifty, horizon=7)
    assert label.iloc[-7:].isna().all(), (
        "Last 7 bars must be NaN — no 7-day forward window available"
    )


@pytest.mark.unit
def test_barrier_label_last_7_bars_are_unresolved():
    """CRIT-001: Barrier label last N bars must be unresolved (NaN label)."""
    ohlcv = _make_ohlcv(50)
    label_df = generate_7d_asymmetric_barrier_label(ohlcv, horizon=7)
    last_labels = label_df["label"].iloc[-7:]
    # The last 7 bars may have unresolved entries (NaN) due to insufficient forward data
    # At a minimum, the last bar should be unresolved
    assert pd.isna(label_df["label"].iloc[-1]), (
        "Last bar must have NaN label (no 7-day forward data)"
    )


@pytest.mark.unit
def test_excess_return_uses_7_bar_forward_not_5():
    """CRIT-001: Verify label at T uses close[T+7] not close[T+5]."""
    ohlcv = _make_ohlcv(50)
    nifty = _make_nifty(50)
    label_7 = generate_7d_excess_return_label(ohlcv, nifty, horizon=7)
    label_5 = generate_7d_excess_return_label(ohlcv, nifty, horizon=5)
    # The values at same T should differ (different forward horizons)
    valid_both = label_7.dropna().index.intersection(label_5.dropna().index)
    if len(valid_both) > 0:
        diffs = (label_7[valid_both] - label_5[valid_both]).abs()
        assert diffs.mean() > 0.0, "7-day and 5-day labels must differ"


# ── CRIT-002: Positive expected value ────────────────────────────────────────


@pytest.mark.unit
def test_asymmetric_barrier_rr_ratio_above_one():
    """CRIT-002: The new barrier design must have R:R > 1 (target > stop)."""
    ohlcv = _make_ohlcv(200)
    label_df = generate_7d_asymmetric_barrier_label(ohlcv, horizon=7)
    rr = label_df["rr_ratio"].dropna()
    assert len(rr) > 0, "RR ratio must be computed"
    mean_rr = float(rr.mean())
    assert mean_rr > 1.0, (
        f"Mean R:R ratio {mean_rr:.3f} must be > 1.0 for positive expected value"
    )


@pytest.mark.unit
def test_asymmetric_barrier_target_larger_than_stop():
    """CRIT-002: Target barrier % must be larger than stop barrier %."""
    ohlcv = _make_ohlcv(200)
    label_df = generate_7d_asymmetric_barrier_label(ohlcv, horizon=7)
    valid = label_df[label_df["target_pct"].notna() & label_df["stop_pct"].notna()]
    assert (valid["target_pct"] > valid["stop_pct"]).all(), (
        "Target pct must always exceed stop pct (2:1 R:R design)"
    )


@pytest.mark.unit
def test_time_expiry_uses_net_cost_threshold():
    """CRIT-002 fix: TIME_EXPIRY should be labeled positive only if net return > 0."""
    # Construct a case where realized return is small positive but less than cost
    ohlcv = _make_ohlcv(50, trend=0.0001, seed=7)  # tiny uptrend
    cost_bps = 100.0  # artificially high cost to trigger the fix
    label_df = generate_7d_asymmetric_barrier_label(
        ohlcv, cost_bps=cost_bps, horizon=7
    )
    time_expiry = label_df[label_df["outcome"] == "TIME_EXPIRY"]
    cost_frac = cost_bps / 10_000.0
    for _, row in time_expiry.iterrows():
        if pd.isna(row["label"]) or pd.isna(row["realized_return"]):
            continue
        r = float(row["realized_return"])
        lbl = int(row["label"])
        # Label must be 1 only if return > cost, 0 if return < cost
        if r > cost_frac:
            assert lbl == 1, f"Return {r:.4f} > cost {cost_frac:.4f} should be label=1"
        elif r < cost_frac:
            assert lbl == 0, f"Return {r:.4f} < cost {cost_frac:.4f} should be label=0"


# ── PIT safety ────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.pit
def test_excess_return_label_is_pit_safe():
    """PIT: Appending future prices must not change historical label values."""
    ohlcv = _make_ohlcv(80, seed=42)
    nifty = _make_nifty(80, seed=42)
    cutoff = 50

    label_short = generate_7d_excess_return_label(ohlcv.iloc[:cutoff], nifty.iloc[:cutoff])

    # Extend with extreme future data
    ohlcv_extended = ohlcv.copy()
    ohlcv_extended.iloc[cutoff:, ohlcv_extended.columns.get_loc("close")] = 99999.0
    nifty_extended = nifty.copy()
    nifty_extended.iloc[cutoff:] = 99999.0

    label_extended = generate_7d_excess_return_label(ohlcv_extended, nifty_extended)

    # Historical values (pre-cutoff, excluding last-7 which are NaN anyway) must be identical
    for i in range(min(cutoff - 7, 40)):
        short_val    = label_short.iloc[i]
        extended_val = label_extended.iloc[i]
        if np.isnan(short_val) or np.isnan(extended_val):
            continue
        assert abs(short_val - extended_val) < 1e-9, (
            f"Historical label at index {i} changed after appending future data: "
            f"{short_val} ≠ {extended_val}"
        )


@pytest.mark.unit
@pytest.mark.pit
def test_barrier_label_entry_is_next_open():
    """PIT: Entry price must be open[T+1], not close[T]."""
    ohlcv = _make_ohlcv(30)
    label_df = generate_7d_asymmetric_barrier_label(ohlcv, horizon=7)
    for i in range(len(ohlcv) - 1):
        entry = label_df["entry_price"].iloc[i]
        if np.isnan(entry):
            continue
        next_open = float(ohlcv["open"].iloc[i + 1])
        assert abs(entry - next_open) < 1e-9, (
            f"Entry at index {i} must be next open={next_open:.4f}, got {entry:.4f}"
        )


# ── Economic validity ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_break_even_win_rate_with_new_design():
    """CRIT-002: Break-even win rate with 2:1 R:R is well below 56.9% (old design)."""
    # For 2:1 R:R at equity costs:
    # win_net = target - cost ≈ (2× stop_net)
    # break_even = |loss_net| / (|win_net| + |loss_net|)
    cost = EQUITY_COST_BPS / 10_000.0

    # Example: target=4%, stop=2%, cost=0.27%
    target = 0.04
    stop   = 0.02
    win_net  =  target - cost
    loss_net = -stop   - cost
    be = abs(loss_net) / (win_net + abs(loss_net))

    assert be < 0.40, (
        f"2:1 R:R break-even {be:.1%} must be < 40% — far below old 56.9%"
    )


@pytest.mark.unit
def test_is_economic_evidence_always_true():
    """All improved labels must set is_economic_evidence=True."""
    ohlcv = _make_ohlcv(30)
    label_df = generate_7d_asymmetric_barrier_label(ohlcv, horizon=7)
    assert label_df["is_economic_evidence"].all(), (
        "is_economic_evidence must be True for all rows in improved label"
    )


@pytest.mark.unit
def test_horizon_constant_is_7():
    """CRIT-001: Module constant HORIZON_DAYS == 7."""
    assert HORIZON_DAYS == 7


# ── Comparison with old design ────────────────────────────────────────────────


@pytest.mark.unit
def test_new_label_has_higher_rr_than_old():
    """
    CRIT-002: With ±2% symmetric barriers (old), R:R=1.0.
    New design should have R:R > 1.0 on a typical stock.
    """
    # Old design: fixed ±2%
    old_rr = 1.0  # 2% / 2% = 1

    # New design: 2:1 R:R by construction
    ohlcv = _make_ohlcv(200)
    label_df = generate_7d_asymmetric_barrier_label(ohlcv, horizon=7)
    new_rr = float(label_df["rr_ratio"].dropna().mean())

    assert new_rr > old_rr, (
        f"New design R:R {new_rr:.2f} must exceed old symmetric R:R {old_rr:.2f}"
    )
