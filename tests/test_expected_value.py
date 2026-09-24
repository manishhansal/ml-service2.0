"""
test_expected_value.py — Phase M expected-value gate tests.
"""
from __future__ import annotations

import pytest

from src.meta.engine import MetaDecisionEngine
from src.meta.expected_value import compute_expected_value
from src.schemas.base import PredictionProvenance


def test_positive_edge_viable():
    ev = compute_expected_value(
        prob_target=0.6, prob_stop=0.3, target_return=0.02, stop_return=-0.01,
        cost_bps=10, slippage_bps=2,
    )
    assert ev.is_viable is True
    assert ev.expected_net_edge > 0
    assert ev.reason == "POSITIVE_NET_EDGE"


def test_negative_edge_not_viable():
    # Low P(target), high P(stop) → negative edge.
    ev = compute_expected_value(
        prob_target=0.2, prob_stop=0.7, target_return=0.02, stop_return=-0.02,
        cost_bps=10, slippage_bps=2,
    )
    assert ev.is_viable is False
    assert ev.expected_net_edge < 0
    assert ev.reason == "INSUFFICIENT_EDGE"


def test_costs_can_flip_marginal_edge():
    # Small gross edge that survives at 0 cost but dies at high cost.
    lo = compute_expected_value(0.52, 0.48, 0.01, -0.01, cost_bps=0, slippage_bps=0)
    hi = compute_expected_value(0.52, 0.48, 0.01, -0.01, cost_bps=50, slippage_bps=20)
    assert lo.expected_net_edge > hi.expected_net_edge


def test_probability_invariant_enforced():
    ev = compute_expected_value(0.7, 0.6, 0.02, -0.01)  # sum > 1
    assert ev.prob_target + ev.prob_stop <= 1.0 + 1e-9


def _outputs():
    return [
        {"model_id": "m1", "action": "BUY", "confidence": 0.8, "direction": 1,
         "provenance": PredictionProvenance.TRAINED_MODEL.value},
        {"model_id": "m2", "action": "BUY", "confidence": 0.75, "direction": 1,
         "provenance": PredictionProvenance.TRAINED_MODEL.value},
        {"model_id": "m3", "action": "BUY", "confidence": 0.7, "direction": 1,
         "provenance": PredictionProvenance.TRAINED_MODEL.value},
    ]


def test_engine_rejects_insufficient_edge():
    engine = MetaDecisionEngine()
    # prob_stop_hit=0.55 stays below the 0.65 abstention threshold so the
    # EV gate (not the stop-probability abstention) is what rejects the trade.
    # With P(target)=0.30, P(stop)=0.55 and symmetric barriers the net edge
    # is clearly negative after costs.
    out = engine.decide(
        model_outputs=_outputs(), symbol="NIFTY", regime="bull",
        data_quality=0.95,
        risk_context={
            "prob_target_hit": 0.30, "prob_stop_hit": 0.55,
            "target_return": 0.02, "stop_return": -0.02,
            "cost_bps": 10, "slippage_bps": 2,
        },
    )
    assert out.action == "NO_TRADE"
    assert "INSUFFICIENT_EDGE" in out.reason_codes
    assert out.abstention is True


def test_engine_accepts_positive_edge():
    engine = MetaDecisionEngine()
    out = engine.decide(
        model_outputs=_outputs(), symbol="NIFTY", regime="bull",
        data_quality=0.95,
        risk_context={
            "prob_target_hit": 0.65, "prob_stop_hit": 0.25,
            "target_return": 0.03, "stop_return": -0.01,
            "cost_bps": 10, "slippage_bps": 2,
        },
    )
    assert out.action == "BUY"
    assert "INSUFFICIENT_EDGE" not in out.reason_codes
    assert out.expected_net_edge is not None and out.expected_net_edge > 0


def test_engine_backward_compatible_no_risk_context():
    engine = MetaDecisionEngine()
    out = engine.decide(model_outputs=_outputs(), symbol="NIFTY", regime="bull", data_quality=0.95)
    # No EV gate applied when no risk context supplied.
    assert out.action == "BUY"
