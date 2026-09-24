"""
test_feedback.py — Phase Q outcome resolution + feedback store tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.data.feedback import FeedbackStore, OutcomeResolver
from src.schemas.meta import FeedbackRecord

UTC = timezone.utc


def _path(start_price, prices, step_min=5):
    t0 = datetime(2024, 1, 1, 9, 15, tzinfo=UTC)
    return [(t0 + timedelta(minutes=step_min * i), p) for i, p in enumerate(prices)]


def test_target_hit_long():
    resolver = OutcomeResolver(cost_bps=10)
    res = resolver.resolve(
        signal_id="s1", symbol="NIFTY", direction=1,
        entry_price=100.0, target_price=102.0, stop_price=98.0,
        price_path=_path(100, [100.5, 101.0, 102.5, 103.0]),
    )
    assert res.exit_reason == "TARGET_HIT"
    assert res.exit_price == 102.0
    assert res.realized_return > 0
    assert res.resolved is True


def test_stop_hit_long():
    resolver = OutcomeResolver(cost_bps=10)
    res = resolver.resolve(
        signal_id="s2", symbol="NIFTY", direction=1,
        entry_price=100.0, target_price=102.0, stop_price=98.0,
        price_path=_path(100, [99.5, 98.0, 97.0]),
    )
    assert res.exit_reason == "STOP_HIT"
    assert res.realized_return < 0


def test_time_expiry():
    resolver = OutcomeResolver(cost_bps=10)
    res = resolver.resolve(
        signal_id="s3", symbol="NIFTY", direction=1,
        entry_price=100.0, target_price=105.0, stop_price=95.0,
        price_path=_path(100, [100.2, 100.5, 100.3, 100.4]),
    )
    assert res.exit_reason == "TIME_EXPIRY"  # neither barrier touched


def test_stop_hit_short():
    resolver = OutcomeResolver(cost_bps=10)
    res = resolver.resolve(
        signal_id="s4", symbol="NIFTY", direction=-1,
        entry_price=100.0, target_price=98.0, stop_price=102.0,
        price_path=_path(100, [100.5, 101.0, 102.5]),
    )
    assert res.exit_reason == "STOP_HIT"
    assert res.realized_return < 0  # short loses when price rises


def test_missing_execution():
    resolver = OutcomeResolver()
    res = resolver.resolve(
        signal_id="s5", symbol="NIFTY", direction=1,
        entry_price=100.0, target_price=102.0, stop_price=98.0,
        price_path=[],
    )
    assert res.resolved is False
    assert res.exit_reason == "MISSING_EXECUTION"


def test_mae_mfe_computed():
    resolver = OutcomeResolver()
    res = resolver.resolve(
        signal_id="s6", symbol="NIFTY", direction=1,
        entry_price=100.0, target_price=110.0, stop_price=90.0,
        price_path=_path(100, [102.0, 98.0, 103.0, 101.0]),
    )
    assert res.mfe > 0  # rose to 103
    assert res.mae < 0  # fell to 98


def test_forced_exit():
    resolver = OutcomeResolver()
    res = resolver.resolve(
        signal_id="s7", symbol="NIFTY", direction=1,
        entry_price=100.0, target_price=110.0, stop_price=90.0,
        price_path=_path(100, [100.5, 101.0]),
        forced_exit=True, manual_exit_price=100.8,
    )
    assert res.exit_reason == "FORCED_EXIT"
    assert res.exit_price == 100.8


# ── Feedback store ────────────────────────────────────────────────────────────


def _record(signal_id, symbol="NIFTY", net=0.01, reason="TARGET_HIT"):
    return FeedbackRecord(
        signal_id=signal_id, symbol=symbol,
        prediction_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        action="BUY", entry_price=100.0, exit_price=101.0,
        realized_return=net, realized_return_net=net, exit_reason=reason,
    )


def test_feedback_store_append_and_count(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append(_record("s1"))
    store.append(_record("s2"))
    assert store.count() == 2


def test_feedback_store_immutable_append_only(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append(_record("s1", net=0.02))
    store.append(_record("s1", net=0.05))  # same id, appended not overwritten
    recs = store.all_records()
    assert len(recs) == 2  # both retained


def test_feedback_summary(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append(_record("s1", net=0.02, reason="TARGET_HIT"))
    store.append(_record("s2", net=-0.01, reason="STOP_HIT"))
    store.append(_record("s3", net=0.03, reason="TARGET_HIT"))
    summary = store.summary()
    assert summary["n_resolved"] == 3
    assert summary["win_rate"] == pytest.approx(2 / 3, abs=0.01)
    assert summary["exit_reasons"]["TARGET_HIT"] == 2


def test_feedback_for_symbol(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.append(_record("s1", symbol="NIFTY"))
    store.append(_record("s2", symbol="BANKNIFTY"))
    assert len(store.for_symbol("NIFTY")) == 1
