"""
test_decision_trace_integrity.py — evidence-chain integrity guard (mandate §7–§10).

A tradeable (BUY/SELL) decision must never be persisted with an incomplete
evidence chain. These tests pin the exact failure the LIVE certification
exposed: a "BUY" reconstructed with data_confidence=0, feature_as_of=None,
P_target=0, P_stop=0 and no reason codes. Such a trace must be DOWNGRADED to
NO_TRADE with EVIDENCE_CHAIN_INCOMPLETE, never trusted as a trade.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.explainability.decision_trace import (
    TRADEABLE_ACTIONS,
    DecisionTrace,
    DecisionTraceStore,
)

UTC = UTC
PRED = datetime(2024, 1, 2, 9, 30, tzinfo=UTC)


def _complete_buy(**overrides) -> DecisionTrace:
    base = dict(
        signal_id="sig-ok",
        symbol="NIFTY",
        prediction_timestamp=PRED.isoformat(),
        feature_as_of=(PRED - timedelta(minutes=1)).isoformat(),
        data_as_of=(PRED - timedelta(minutes=1)).isoformat(),
        data_confidence_score=88,
        action="BUY",
        confidence=0.6,
        provenance="trained_model",
        expected_net_edge=0.003,
        prob_target_hit=0.6,
        prob_stop_hit=0.3,
        reason_codes=["POSITIVE_NET_EDGE"],
        feature_snapshot={"rsi_14": 55.0},
    )
    base.update(overrides)
    return DecisionTrace(**base)


def test_complete_buy_passes_integrity():
    trace = _complete_buy()
    assert trace.integrity_violations() == []
    assert trace.is_evidence_complete()
    assert trace.enforce_integrity().action == "BUY"


def test_reproduce_live_bug_buy_is_downgraded():
    """The exact live-certification defect: a BUY with an empty evidence chain."""
    bad = DecisionTrace(
        signal_id="paper-1",
        symbol="NIFTY",
        prediction_timestamp=PRED.isoformat(),
        action="BUY",
        confidence=0.17,
        provenance="trained_model",
        expected_net_edge=0.003,
        # feature_as_of=None, data_as_of=None, data_confidence_score=0,
        # prob_target_hit=0, prob_stop_hit=0, reason_codes=[]  ← all defaults
    )
    violations = bad.integrity_violations()
    assert "MISSING_FEATURE_AS_OF" in violations
    assert "MISSING_DATA_AS_OF" in violations
    assert "MISSING_DATA_CONFIDENCE" in violations
    assert "DEGENERATE_BARRIER_PROBABILITIES" in violations
    assert "MISSING_REASON_CODES" in violations

    enforced = bad.enforce_integrity()
    assert enforced.action == "NO_TRADE"
    assert enforced.abstention is True
    assert enforced.abstention_reason == "EVIDENCE_CHAIN_INCOMPLETE"
    assert "EVIDENCE_CHAIN_INCOMPLETE" in enforced.reason_codes
    assert enforced.suggested_position_size_pct == 0.0


@pytest.mark.parametrize(
    "override,expected_code",
    [
        ({"feature_as_of": None}, "MISSING_FEATURE_AS_OF"),
        ({"data_as_of": None}, "MISSING_DATA_AS_OF"),
        ({"data_confidence_score": 0}, "MISSING_DATA_CONFIDENCE"),
        ({"prob_target_hit": 0.0, "prob_stop_hit": 0.0}, "DEGENERATE_BARRIER_PROBABILITIES"),
        ({"prob_target_hit": 0.8, "prob_stop_hit": 0.5}, "BARRIER_PROBABILITY_SUM_EXCEEDS_ONE"),
        ({"expected_net_edge": None}, "MISSING_EXPECTED_NET_EDGE"),
        ({"expected_net_edge": 0.0}, "NON_POSITIVE_EXPECTED_NET_EDGE"),
        ({"expected_net_edge": -0.01}, "NON_POSITIVE_EXPECTED_NET_EDGE"),
        ({"reason_codes": []}, "MISSING_REASON_CODES"),
    ],
)
def test_each_missing_field_triggers_downgrade(override, expected_code):
    trace = _complete_buy(**override)
    violations = trace.integrity_violations()
    assert expected_code in violations, violations
    assert trace.enforce_integrity().action == "NO_TRADE"


def test_pit_order_violation_detected():
    trace = _complete_buy(feature_as_of=(PRED + timedelta(minutes=5)).isoformat())
    assert "PIT_ORDER_VIOLATION_FEATURE_AFTER_PREDICTION" in trace.integrity_violations()
    assert trace.enforce_integrity().action == "NO_TRADE"


def test_store_persists_downgraded_action(tmp_path):
    store = DecisionTraceStore(tmp_path / "t.jsonl")
    bad = DecisionTrace(
        signal_id="paper-x", symbol="NIFTY",
        prediction_timestamp=PRED.isoformat(), action="BUY",
        confidence=0.5, provenance="trained_model", expected_net_edge=0.001,
    )
    written = store.record(bad)
    assert written.action == "NO_TRADE"
    reloaded = store.get("paper-x")
    assert reloaded.action == "NO_TRADE"
    assert "EVIDENCE_CHAIN_INCOMPLETE" in reloaded.reason_codes


def test_no_trade_only_needs_timestamp_and_reason():
    """NO_TRADE is non-directional: barrier probs / edge are not required."""
    trace = DecisionTrace(
        signal_id="nt", symbol="NIFTY",
        prediction_timestamp=PRED.isoformat(),
        action="NO_TRADE", abstention=True,
        reason_codes=["INSUFFICIENT_EDGE"],
    )
    assert trace.integrity_violations() == []
    assert trace.enforce_integrity().action == "NO_TRADE"


def test_no_trade_missing_reason_is_annotated_not_promoted():
    trace = DecisionTrace(
        signal_id="nt2", symbol="NIFTY",
        prediction_timestamp=PRED.isoformat(), action="NO_TRADE",
    )
    assert "MISSING_REASON_CODES" in trace.integrity_violations()
    enforced = trace.enforce_integrity()
    assert enforced.action == "NO_TRADE"
    assert "MISSING_REASON_CODES" in enforced.reason_codes


def test_tradeable_actions_constant():
    assert TRADEABLE_ACTIONS == frozenset({"BUY", "SELL"})
