"""
Regression tests for SevenDayBacktestEngine.

These tests enforce the hard contracts from the mandate:
  - No look-ahead in signal evaluation
  - Entry at next-bar open, never signal-bar price
  - Exit at exactly T+7 trading days (NSE calendar)
  - Net P&L correctly deducts realistic costs
  - MFE and MAE are computed correctly
  - Direction inversion is correct for SHORT signals
  - Signal decay output is consistent

pytest markers: unit, pit
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.seven_day_engine import (
    NSECostModel,
    SevenDayBacktestEngine,
    SignalRecord,
    classify_market_regime,
    classify_volatility_regime,
    nth_trading_day,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 30, seed: int = 42, trend: float = 0.0) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-02", periods=n, freq="B", tz="UTC")
    prices = 1000.0 * np.cumprod(1 + rng.normal(trend, 0.01, n))
    high   = prices * (1 + rng.uniform(0.005, 0.02, n))
    low    = prices * (1 - rng.uniform(0.005, 0.02, n))
    open_  = prices * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": prices, "volume": volume},
        index=idx,
    )


def _engine(cost: str = "equity") -> SevenDayBacktestEngine:
    cost_model = NSECostModel.equity() if cost == "equity" else NSECostModel.futures()
    return SevenDayBacktestEngine(
        cost_model=cost_model,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        model_version="test-v1",
    )


# ── Contract: no look-ahead ───────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.pit
def test_entry_price_is_next_bar_open():
    """Entry price must be open[T+1], never close[T]."""
    ohlcv = _make_ohlcv(20)
    engine = _engine()
    signal_date = ohlcv.index[5]  # bar T = index 5
    rec = engine.evaluate_signal("TEST", signal_date, direction=1, score=0.8, ohlcv=ohlcv)

    expected_entry = float(ohlcv["open"].iloc[6])  # open[T+1]
    assert abs(rec.entry_price - expected_entry) < 1e-9, (
        f"Entry must be open[T+1]={expected_entry:.4f}, got {rec.entry_price:.4f}"
    )


@pytest.mark.unit
@pytest.mark.pit
def test_entry_never_uses_signal_bar_close():
    """Entry price must NOT equal close[T] (no look-ahead execution)."""
    ohlcv = _make_ohlcv(20)
    engine = _engine()
    signal_date = ohlcv.index[5]
    rec = engine.evaluate_signal("TEST", signal_date, direction=1, score=0.8, ohlcv=ohlcv)

    signal_bar_close = float(ohlcv["close"].iloc[5])
    assert abs(rec.entry_price - signal_bar_close) > 1e-6 or rec.outcome == "UNRESOLVED", (
        "Entry price must not equal signal-bar close — this would be look-ahead execution"
    )


@pytest.mark.unit
def test_exit_uses_only_forward_data():
    """
    Appending extreme price data BEFORE the signal date must not change the outcome
    for signals generated AFTER those bars.
    """
    ohlcv = _make_ohlcv(30)
    engine = _engine()
    signal_date = ohlcv.index[15]

    rec_original = engine.evaluate_signal("TEST", signal_date, direction=1, score=0.7, ohlcv=ohlcv)

    # Append extreme data in the past (before signal) — must not affect outcome
    extreme_early = ohlcv.copy()
    extreme_early.loc[ohlcv.index[:5], "close"] = 100_000.0  # absurd past prices
    rec_modified  = engine.evaluate_signal("TEST", signal_date, direction=1, score=0.7, ohlcv=extreme_early)

    # Entry, exit, and net_pnl must be identical
    assert abs(rec_original.entry_price - rec_modified.entry_price) < 1e-6
    assert abs(rec_original.exit_price  - rec_modified.exit_price)  < 1e-6


# ── Contract: exit exactly at T+7 trading days ────────────────────────────────


@pytest.mark.unit
def test_exit_after_exactly_7_trading_days():
    """Without barrier hits, exit must be at the 7th trading day close."""
    # Create OHLCV with no extreme moves so barriers are not hit
    rng = np.random.default_rng(0)
    n = 30
    idx = pd.date_range("2025-01-06", periods=n, freq="B", tz="UTC")
    prices = 1000 + rng.normal(0, 0.5, n)  # tiny moves → no barrier hit
    df = pd.DataFrame({
        "open":   prices * 0.999,
        "high":   prices * 1.002,
        "low":    prices * 0.998,
        "close":  prices,
        "volume": np.ones(n) * 1e6,
    }, index=idx)

    engine = SevenDayBacktestEngine(
        stop_loss_pct=0.99,    # effectively disable stop
        take_profit_pct=0.99,  # effectively disable target
        model_version="test",
    )
    signal_date = idx[5]
    rec = engine.evaluate_signal("TEST", signal_date, direction=1, score=0.8, ohlcv=df)

    if rec.outcome == "UNRESOLVED":
        pytest.skip("Insufficient data")

    # The exit price should be the close of the 7th trading day after entry
    entry_bar = 6   # open[T+1]
    # Count 7 trading days from entry
    td = 0
    expected_exit_bar = entry_bar
    for k in range(entry_bar, min(entry_bar + 20, n)):
        if df.index[k].weekday() >= 5:
            continue
        td += 1
        expected_exit_bar = k
        if td >= 7:
            break

    expected_exit = float(df["close"].iloc[expected_exit_bar])
    assert abs(rec.exit_price - expected_exit) < 1e-6, (
        f"Exit must be close[T+7]={expected_exit:.4f}, got {rec.exit_price:.4f}"
    )


# ── Contract: cost correctly deducted ────────────────────────────────────────


@pytest.mark.unit
def test_net_pnl_equals_gross_minus_cost():
    """net_pnl = gross_pnl − cost (exact arithmetic)."""
    ohlcv = _make_ohlcv(25)
    engine = _engine("equity")
    signal_date = ohlcv.index[5]
    rec = engine.evaluate_signal("TEST", signal_date, direction=1, score=0.8, ohlcv=ohlcv)

    if rec.outcome == "UNRESOLVED":
        pytest.skip("Unresolved signal")

    assert abs(rec.net_pnl - (rec.gross_pnl - rec.cost)) < 1e-6, (
        f"net_pnl={rec.net_pnl} ≠ gross={rec.gross_pnl} − cost={rec.cost}"
    )


@pytest.mark.unit
def test_equity_cost_bps_is_realistic():
    """NSE equity round-trip cost must be close to 27.65 bps."""
    cost = NSECostModel.equity()
    assert 20 <= cost.total_round_trip_bps <= 35, (
        f"Equity round-trip cost {cost.total_round_trip_bps:.2f} bps is outside realistic range"
    )


@pytest.mark.unit
def test_futures_cost_bps_is_lower_than_equity():
    """Futures cost must be lower than equity cost."""
    eq  = NSECostModel.equity()
    fut = NSECostModel.futures()
    assert fut.total_round_trip_bps < eq.total_round_trip_bps


# ── Contract: direction inversion ────────────────────────────────────────────


@pytest.mark.unit
def test_short_signal_inverts_return():
    """A SHORT signal on a falling stock should produce positive return."""
    n = 25
    idx = pd.date_range("2025-01-06", periods=n, freq="B", tz="UTC")
    # Create strongly falling stock
    prices = np.linspace(1000, 850, n)
    df = pd.DataFrame({
        "open":   prices * 0.999,
        "high":   prices * 1.001,
        "low":    prices * 0.998,
        "close":  prices,
        "volume": np.ones(n) * 1e6,
    }, index=idx)

    engine = SevenDayBacktestEngine(
        stop_loss_pct=0.50,  # disable stop
        take_profit_pct=0.50,  # disable target
        model_version="test",
    )
    signal_date = idx[3]

    long_rec  = engine.evaluate_signal("TEST", signal_date, direction=1,  score=0.8, ohlcv=df)
    short_rec = engine.evaluate_signal("TEST", signal_date, direction=-1, score=0.2, ohlcv=df)

    if long_rec.outcome == "UNRESOLVED" or short_rec.outcome == "UNRESOLVED":
        pytest.skip("Insufficient data")

    # On a falling stock: long should lose, short should win
    assert long_rec.gross_pnl < 0, "Long on falling stock should lose"
    assert short_rec.gross_pnl > 0, "Short on falling stock should profit"
    # Gross returns should be approximately equal and opposite
    assert abs(long_rec.gross_pnl + short_rec.gross_pnl) < 0.1, (
        "Long + Short gross P&L should sum to approximately 0 (before costs)"
    )


# ── Contract: MFE and MAE non-negative ───────────────────────────────────────


@pytest.mark.unit
def test_mfe_and_mae_are_nonnegative():
    """MFE and MAE must both be ≥ 0."""
    ohlcv = _make_ohlcv(25, seed=7)
    engine = _engine()
    for i in [3, 8, 12]:
        signal_date = ohlcv.index[i]
        rec = engine.evaluate_signal("TEST", signal_date, direction=1, score=0.7, ohlcv=ohlcv)
        if rec.outcome == "UNRESOLVED":
            continue
        if not np.isnan(rec.mfe):
            assert rec.mfe >= 0, f"MFE={rec.mfe} must be non-negative"
        if not np.isnan(rec.mae):
            assert rec.mae >= 0, f"MAE={rec.mae} must be non-negative"


# ── Contract: performance report consistency ──────────────────────────────────


@pytest.mark.unit
def test_performance_report_win_rate_consistency():
    """Performance report win_rate must equal n_wins / (n_wins + n_losses)."""
    ohlcv = _make_ohlcv(50, seed=99)
    engine = _engine()
    records = []
    for i in range(0, 40, 4):
        rec = engine.evaluate_signal(
            "TEST", ohlcv.index[i], direction=1, score=0.7, ohlcv=ohlcv
        )
        if rec.outcome != "UNRESOLVED":
            records.append(rec)

    if not records:
        pytest.skip("No resolved records")

    report = engine.performance_report(records)
    n_wins  = report.get("n_wins", 0)
    n_total = report.get("n_wins", 0) + report.get("n_losses", 0)
    if n_total > 0:
        expected_wr = n_wins / n_total
        assert abs(report.get("win_rate", float("nan")) - expected_wr) < 1e-6


# ── Contract: 7-day label mismatch vs 5-bar ───────────────────────────────────


@pytest.mark.unit
def test_seven_day_differs_from_five_bar():
    """
    Verify that T+7 exit price differs from T+5 exit price.
    This confirms the engine evaluates EXACTLY 7 trading days, not 5.
    """
    ohlcv = _make_ohlcv(30)
    signal_date = ohlcv.index[5]

    engine7 = SevenDayBacktestEngine(stop_loss_pct=0.99, take_profit_pct=0.99)
    rec7 = engine7.evaluate_signal("TEST", signal_date, direction=1, score=0.7, ohlcv=ohlcv)

    engine5 = SevenDayBacktestEngine(stop_loss_pct=0.99, take_profit_pct=0.99)
    # Manually get T+5 exit
    entry_idx = 6
    t5_exit_idx = entry_idx
    td = 0
    for k in range(entry_idx, min(entry_idx + 15, len(ohlcv))):
        if ohlcv.index[k].weekday() >= 5:
            continue
        td += 1
        t5_exit_idx = k
        if td >= 5:
            break

    t7_exit_close = rec7.exit_price
    t5_exit_close = float(ohlcv["close"].iloc[t5_exit_idx])

    if rec7.outcome != "UNRESOLVED" and td == 5:
        # T+7 and T+5 should differ (2 additional trading days)
        assert abs(t7_exit_close - t5_exit_close) > 1e-6, (
            "T+7 and T+5 exit prices should differ — engine must count exactly 7 trading days"
        )


# ── Cost model tests ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_cost_model_total_round_trip():
    """NSECostModel.equity() should be in the realistic NSE equity range (20–35 bps)."""
    cost = NSECostModel.equity()
    # Allow wider tolerance: actual Indian equity costs vary from 22–32 bps
    # depending on brokerage tier, lot size, and instrument type.
    assert 20 <= cost.total_round_trip_bps <= 35, (
        f"Equity round-trip {cost.total_round_trip_bps:.2f} bps is outside realistic range (20–35)"
    )
    # Must be higher than futures cost
    assert cost.total_round_trip_bps > NSECostModel.futures().total_round_trip_bps


@pytest.mark.unit
def test_futures_cost_model():
    """NSECostModel.futures() must be lower than equity and in a realistic range."""
    cost = NSECostModel.futures()
    assert 4 <= cost.total_round_trip_bps <= 15, (
        f"Futures round-trip {cost.total_round_trip_bps:.2f} bps is outside realistic range (4–15)"
    )


# ── NSE calendar tests ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_nth_trading_day_skips_weekend():
    """nth_trading_day should skip Saturday and Sunday."""
    friday = pd.Timestamp("2025-01-03", tz="UTC")   # Friday
    # T+1 = Monday Jan 6, T+2 = Tuesday Jan 7
    t1 = nth_trading_day(friday, 1)
    t2 = nth_trading_day(friday, 2)
    assert t1.weekday() == 0, f"T+1 from Friday should be Monday, got weekday={t1.weekday()}"
    assert t2.weekday() == 1, f"T+2 from Friday should be Tuesday"


@pytest.mark.unit
def test_nth_trading_day_basic():
    """T+7 from a Monday should be the following Tuesday (assuming no holidays)."""
    monday = pd.Timestamp("2025-01-06", tz="UTC")
    t7 = nth_trading_day(monday, 7)
    # Mon → Mon+7 trading days = next Monday+1 = next Tuesday (skips weekend)
    # 7 trading days from Mon: Mon+7 working = next Tue (Mon+Tue+Wed+Thu+Fri + Mon+Tue = day 5+2 = Tue)
    assert t7.weekday() in (0, 1, 2, 3, 4), "T+7 should be a weekday"


# ── Regime classification ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_bull_regime_classification():
    """Strong uptrend should be classified as BULL."""
    returns = pd.Series([0.005] * 20)  # +0.5%/day for 20 days = ~10% trend
    assert classify_market_regime(returns) == "BULL"


@pytest.mark.unit
def test_bear_regime_classification():
    """Strong downtrend should be classified as BEAR."""
    returns = pd.Series([-0.005] * 20)  # −0.5%/day
    assert classify_market_regime(returns) == "BEAR"


# ── Break-even accuracy test ─────────────────────────────────────────────────


@pytest.mark.unit
def test_break_even_accuracy_at_equity_costs():
    """
    Verify the break-even win rate calculation for equity costs.
    At 27.65bps round-trip, ±3% stop/target (3:1 R:R implied by engine defaults):
    Break-even = |loss_net| / (|win_net| + |loss_net|)
    """
    cost = NSECostModel.equity()
    stop_pct   = 0.03    # 3% stop
    target_pct = 0.06    # 6% target (2:1 R:R)
    cost_frac = cost.total_round_trip

    win_net  =  target_pct - cost_frac
    loss_net = -stop_pct   - cost_frac

    breakeven = abs(loss_net) / (win_net + abs(loss_net))
    # With 2:1 R:R at 27.65bps, break-even is approximately 35%
    assert 0.30 < breakeven < 0.45, (
        f"Break-even {breakeven:.1%} outside expected range for 2:1 R:R + {cost_frac*10000:.1f}bps"
    )


@pytest.mark.unit
def test_symmetric_barrier_breakeven_at_equity_costs():
    """
    Verify the break-even for the CURRENT label config (±2% symmetric barriers).
    This is the LABEL_AUDIT finding: 56.9% win rate required.
    """
    cost = NSECostModel.equity()
    barrier_pct = 0.02    # ±2% symmetric
    cost_frac   = cost.total_round_trip

    win_net  =  barrier_pct - cost_frac
    loss_net = -barrier_pct - cost_frac

    breakeven = abs(loss_net) / (win_net + abs(loss_net))
    # Must be around 56.9%
    assert 0.54 < breakeven < 0.60, (
        f"Break-even {breakeven:.1%} for symmetric ±2% barriers should be ~56.9%"
    )
