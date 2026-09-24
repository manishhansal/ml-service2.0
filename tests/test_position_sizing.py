"""
test_position_sizing.py — Phase L position sizing + risk limit tests.

Key property (Phase 74): confidence NEVER bypasses hard caps.
"""
from __future__ import annotations

import pytest

from src.backtest.position_sizing import (
    PortfolioRiskState,
    PositionSizer,
    RiskLimits,
    fractional_kelly,
)


def test_kelly_positive_edge():
    # p=0.6, win=2%, loss=1% → positive edge.
    f = fractional_kelly(0.6, 0.02, -0.01, kelly_fraction=1.0)
    assert f > 0


def test_kelly_negative_edge_zero():
    # p=0.4, win=1%, loss=1% → negative edge → 0.
    f = fractional_kelly(0.4, 0.01, -0.01, kelly_fraction=1.0)
    assert f == 0.0


def test_fractional_scaling():
    full = fractional_kelly(0.6, 0.02, -0.01, kelly_fraction=1.0)
    quarter = fractional_kelly(0.6, 0.02, -0.01, kelly_fraction=0.25)
    assert quarter == pytest.approx(full * 0.25)


def test_position_cap_binds():
    sizer = PositionSizer(RiskLimits(max_position_weight=0.05, kelly_fraction=1.0))
    # Very high edge would want a large weight, but cap is 5%.
    result = sizer.size(prob_target=0.95, win_return=0.05, loss_return=-0.01)
    assert result.weight <= 0.05
    assert result.binding_constraint == "MAX_POSITION_WEIGHT"


def test_confidence_never_bypasses_hard_cap():
    """Even at prob=0.99, the hard position cap holds."""
    sizer = PositionSizer(RiskLimits(max_position_weight=0.10, kelly_fraction=1.0))
    result = sizer.size(prob_target=0.99, win_return=0.10, loss_return=-0.01)
    assert result.weight <= 0.10


def test_gross_exposure_cap():
    sizer = PositionSizer(RiskLimits(max_gross_exposure=0.5, max_position_weight=1.0, kelly_fraction=1.0))
    result = sizer.size(prob_target=0.9, win_return=0.05, loss_return=-0.01, current_gross=0.45)
    assert result.weight <= 0.05  # only 0.05 headroom left


def test_sector_cap():
    sizer = PositionSizer(RiskLimits(max_sector_weight=0.2, max_position_weight=1.0, kelly_fraction=1.0))
    result = sizer.size(prob_target=0.9, win_return=0.05, loss_return=-0.01, current_sector_weight=0.18)
    assert result.weight <= 0.02


def test_drawdown_throttle_reduces_size():
    sizer = PositionSizer(RiskLimits(max_drawdown_throttle=0.20, kelly_fraction=1.0, max_position_weight=1.0))
    no_dd = sizer.size(prob_target=0.6, win_return=0.02, loss_return=-0.01, current_drawdown=0.0)
    with_dd = sizer.size(prob_target=0.6, win_return=0.02, loss_return=-0.01, current_drawdown=-0.10)
    assert with_dd.weight < no_dd.weight


def test_vol_targeting_scales_down():
    sizer = PositionSizer(RiskLimits(vol_target=0.15, kelly_fraction=1.0, max_position_weight=1.0))
    low_vol = sizer.size(prob_target=0.6, win_return=0.02, loss_return=-0.01, realized_vol=0.10)
    high_vol = sizer.size(prob_target=0.6, win_return=0.02, loss_return=-0.01, realized_vol=0.45)
    assert high_vol.weight < low_vol.weight


def test_portfolio_risk_state_gating():
    limits = RiskLimits()
    state = PortfolioRiskState(n_positions=10, gross_exposure=0.5)
    ok, reason = state.can_add(limits, max_positions=10)
    assert ok is False
    assert reason == "MAX_POSITIONS_REACHED"

    state2 = PortfolioRiskState(n_positions=2, gross_exposure=0.3, current_drawdown=-0.05)
    ok2, reason2 = state2.can_add(limits, max_positions=10)
    assert ok2 is True
