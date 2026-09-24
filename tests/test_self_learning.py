"""
test_self_learning.py — Phase R controlled online-learning loop tests.

Verifies the retraining DECISION logic and that the champion is never mutated
directly by the loop (promotion to champion still requires gated approval).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.data.feedback import FeedbackStore
from src.schemas.meta import FeedbackRecord
from src.training.self_learning import SelfLearningLoop

UTC = timezone.utc


def _rec(i, reason="TARGET_HIT"):
    return FeedbackRecord(
        signal_id=f"s{i}", symbol="NIFTY",
        prediction_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        action="BUY", entry_price=100.0, exit_price=101.0,
        realized_return=0.01, realized_return_net=0.008, exit_reason=reason,
    )


def _store(tmp_path, n):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    for i in range(n):
        store.append(_rec(i))
    return store


def test_no_trigger_when_insufficient_samples(tmp_path):
    loop = SelfLearningLoop(_store(tmp_path, 5), min_new_samples=30)
    decision = loop.should_retrain()
    assert decision.should_retrain is False
    assert decision.reason == "NO_TRIGGER"


def test_trigger_on_sufficient_samples(tmp_path):
    loop = SelfLearningLoop(_store(tmp_path, 40), min_new_samples=30)
    decision = loop.should_retrain()
    assert decision.should_retrain is True
    assert "SUFFICIENT_NEW_OUTCOMES" in decision.triggers


def test_trigger_on_drift(tmp_path):
    loop = SelfLearningLoop(_store(tmp_path, 5), min_new_samples=30)
    decision = loop.should_retrain(drift_action="TRAIN_CHALLENGER")
    assert decision.should_retrain is True
    assert "DRIFT_HIGH" in decision.triggers


def test_trigger_on_performance_degradation(tmp_path):
    loop = SelfLearningLoop(_store(tmp_path, 5), min_new_samples=30)
    decision = loop.should_retrain(performance_degraded=True)
    assert decision.should_retrain is True
    assert "PERFORMANCE_DEGRADATION" in decision.triggers


def test_rate_limit_blocks_retrain(tmp_path):
    loop = SelfLearningLoop(_store(tmp_path, 40), min_new_samples=30,
                            min_hours_between_retrains=24.0)
    first = loop.should_retrain()
    assert first.should_retrain is True
    loop.mark_retrain_triggered()
    # Immediately after triggering, a second retrain is rate-limited.
    second = loop.should_retrain(drift_action="TRAIN_CHALLENGER")
    assert second.should_retrain is False
    assert second.reason == "RATE_LIMITED"


def test_mark_retrain_resets_new_sample_counter(tmp_path):
    store = _store(tmp_path, 40)
    loop = SelfLearningLoop(store, min_new_samples=30, min_hours_between_retrains=0.0)
    loop.mark_retrain_triggered()
    # After consuming, new-sample count is 0 → no sample-based trigger.
    decision = loop.should_retrain()
    assert "SUFFICIENT_NEW_OUTCOMES" not in decision.triggers


def test_run_cycle_stops_at_shadow(tmp_path):
    """The loop registers a challenger to SHADOW but never promotes to champion."""
    store = _store(tmp_path, 40)
    loop = SelfLearningLoop(store, min_new_samples=30, min_hours_between_retrains=0.0)

    class _FakeReport:
        passed_acceptance = True
        champion_version = "1.0.0-x"
        def to_dict(self):
            return {"champion": "logistic", "champion_version": self.champion_version}

    class _FakeOrch:
        def train(self, *a, **k):
            return _FakeReport()

    events = {"challenger": None, "shadow": False, "champion": None}

    class _FakeLifecycle:
        def register_challenger(self, name, version):
            events["challenger"] = version
        def promote_to_shadow(self, name):
            events["shadow"] = True
        def promote_shadow_to_champion(self, *a, **k):
            events["champion"] = "SHOULD_NOT_BE_CALLED"

    result = loop.run_cycle(_FakeOrch(), _FakeLifecycle(), "market_regime", "ds-1")
    assert result["stage"] == "SHADOW"
    assert events["challenger"] == "1.0.0-x"
    assert events["shadow"] is True
    # Champion promotion NEVER auto-invoked by the loop.
    assert events["champion"] is None
