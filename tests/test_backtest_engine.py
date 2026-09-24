"""
test_backtest_engine.py — Phase K cost-aware backtest tests.

Key properties:
- Execution uses NEXT-bar open, never the signal-bar close.
- Costs strictly reduce net return below gross.
- Higher costs monotonically reduce net return.
- Profitability metrics are computed correctly.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    BacktestEngine,
    CostModel,
    cost_sensitivity_analysis,
)

UTC = timezone.utc


def _trending_prices(n=100, drift=0.005, seed=0):
    rng = np.random.default_rng(seed)
    rets = drift + rng.normal(0, 0.002, n)
    close = 100 * np.cumprod(1 + rets)
    open_ = np.concatenate([[100.0], close[:-1]])  # open = prev close
    idx = pd.DatetimeIndex([datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)])
    return pd.DataFrame({"open": open_, "close": close}, index=idx)


def test_costs_reduce_net_below_gross():
    prices = _trending_prices()
    signals = np.ones(len(prices))  # always long
    engine = BacktestEngine(CostModel())
    report = engine.run(prices, signals)
    assert report.net_return < report.gross_return
    assert report.cost_bps_round_trip > 0


def test_zero_cost_gross_equals_net():
    prices = _trending_prices()
    signals = np.ones(len(prices))
    engine = BacktestEngine(CostModel(brokerage_bps=0, fees_bps=0, half_spread_bps=0, slippage_bps=0))
    report = engine.run(prices, signals)
    assert report.net_return == pytest.approx(report.gross_return, abs=1e-9)


def test_higher_costs_reduce_returns():
    prices = _trending_prices(seed=2)
    # Alternate position to force many round trips → cost sensitivity visible.
    signals = np.array([1 if i % 4 < 2 else 0 for i in range(len(prices))], dtype=float)
    results = cost_sensitivity_analysis(prices, signals, bps_levels=(5, 10, 20))
    r5 = results["5bps"]["net_return"]
    r10 = results["10bps"]["net_return"]
    r20 = results["20bps"]["net_return"]
    assert r5 >= r10 >= r20  # monotonic decrease


def test_next_bar_execution_not_signal_close():
    # Signal on bar 0, price jumps on bar 1 open. The trade must fill at bar1 open.
    idx = pd.DatetimeIndex([datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(4)])
    prices = pd.DataFrame({
        "open": [100.0, 110.0, 110.0, 110.0],
        "close": [100.0, 110.0, 110.0, 110.0],
    }, index=idx)
    signals = np.array([1.0, 1.0, 0.0, 0.0])  # enter at bar0 decision → fill bar1 open=110
    engine = BacktestEngine(CostModel(brokerage_bps=0, fees_bps=0, half_spread_bps=0, slippage_bps=0))
    report = engine.run(prices, signals)
    # Entry at 110 (bar1 open), exit at 110 → ~0 gross (no lookahead capture of the jump).
    assert report.n_trades == 1
    assert abs(report.trades[0].entry_price - 110.0) < 1e-9


def test_profit_metrics_present():
    prices = _trending_prices(drift=0.004, seed=3)
    signals = np.ones(len(prices))
    report = BacktestEngine(CostModel()).run(prices, signals)
    d = report.to_dict()
    for key in ("sharpe", "sortino", "max_drawdown", "win_rate", "profit_factor", "turnover"):
        assert key in d


def test_no_trades_safe():
    prices = _trending_prices()
    signals = np.zeros(len(prices))  # never trade
    report = BacktestEngine(CostModel()).run(prices, signals)
    assert report.n_trades == 0
    assert report.net_return == 0.0
