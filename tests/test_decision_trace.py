"""
test_decision_trace.py — Phase S decision trace + reconstruction tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.explainability.decision_trace import (
    DecisionTrace,
    DecisionTraceStore,
    trace_from_meta_output,
)
from src.meta.engine import MetaDecisionEngine
from src.schemas.base import PredictionProvenance

UTC = timezone.utc


def test_feature_hash_deterministic():
    f = {"a": 1.0, "b": 2.0}
    h1 = DecisionTrace.compute_feature_hash(f)
    h2 = DecisionTrace.compute_feature_hash({"b": 2.0, "a": 1.0})  # order-independent
    assert h1 == h2


def test_trace_record_and_reconstruct(tmp_path):
    store = DecisionTraceStore(tmp_path / "traces.jsonl")
    trace = DecisionTrace(
        signal_id="sig-1", symbol="NIFTY",
        prediction_timestamp=datetime(2024, 1, 2, tzinfo=UTC).isoformat(),
        feature_snapshot={"rsi_14": 60.0, "ret_5": 0.01},
        models_used=["logistic", "lightgbm"],
        action="BUY", confidence=0.62, provenance="trained_model",
        expected_net_edge=0.004, prob_target_hit=0.6, prob_stop_hit=0.3,
        reason_codes=["POSITIVE_NET_EDGE"],
    )
    store.record(trace)
    loaded = store.get("sig-1")
    assert loaded is not None
    assert loaded.action == "BUY"
    assert loaded.feature_hash  # auto-computed
    text = store.reconstruct("sig-1")
    assert "NIFTY" in text
    assert "BUY" in text
    assert "Expected net edge" in text


def test_reconstruct_missing_signal(tmp_path):
    store = DecisionTraceStore(tmp_path / "traces.jsonl")
    assert "No decision trace" in store.reconstruct("nope")


def test_abstention_reason_reconstructed(tmp_path):
    store = DecisionTraceStore(tmp_path / "traces.jsonl")
    trace = DecisionTrace(
        signal_id="sig-2", symbol="NIFTY",
        prediction_timestamp=datetime(2024, 1, 2, tzinfo=UTC).isoformat(),
        action="NO_TRADE", abstention=True,
        reason_codes=["INSUFFICIENT_EDGE"], abstention_reason="INSUFFICIENT_EDGE",
    )
    store.record(trace)
    text = store.reconstruct("sig-2")
    assert "Abstained" in text
    assert "INSUFFICIENT_EDGE" in text


def test_trace_from_meta_output_end_to_end(tmp_path):
    """Reconstruct a real MetaDecisionEngine decision from persisted evidence."""
    engine = MetaDecisionEngine()
    outputs = [
        {"model_id": m, "action": "BUY", "confidence": 0.75, "direction": 1,
         "provenance": PredictionProvenance.TRAINED_MODEL.value}
        for m in ("m1", "m2", "m3")
    ]
    ts = datetime(2024, 1, 2, 9, 30, tzinfo=UTC)
    meta = engine.decide(
        model_outputs=outputs, symbol="NIFTY", regime="bull", data_quality=0.95,
        risk_context={
            "prob_target_hit": 0.65, "prob_stop_hit": 0.25,
            "target_return": 0.03, "stop_return": -0.01, "cost_bps": 10, "slippage_bps": 2,
        },
    )
    # Populate PIT timestamps (normally set by the signal API layer).
    meta = meta.model_copy(update={
        "prediction_timestamp": ts,
        "feature_as_of": ts - timedelta(minutes=1),
    })
    trace = trace_from_meta_output(
        meta, feature_snapshot={"rsi_14": 58.0, "macd_hist": 0.001}, regime="bull",
    )
    store = DecisionTraceStore(tmp_path / "traces.jsonl")
    store.record(trace)

    reconstructed = store.reconstruct(meta.signal_id)
    assert meta.symbol in reconstructed
    assert "BUY" in reconstructed
    # The persisted trace must contain enough to answer "why trade?".
    got = store.get(meta.signal_id)
    assert got.expected_net_edge is not None
    assert got.data_confidence_score == 95
    assert got.regime == "bull"
    assert set(got.models_used) == {"m1", "m2", "m3"}
