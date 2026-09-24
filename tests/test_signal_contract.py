"""
test_signal_contract.py — Phase T signal expiry + ML->AlphaForge contract tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.meta.signal import (
    SignalExecutability,
    attach_expiry,
    is_executable,
    validate_signal_contract,
)
from src.schemas.base import PredictionProvenance
from src.schemas.meta import MetaOutput

UTC = timezone.utc


def _buy_signal(**overrides):
    base = dict(
        action="BUY", confidence=0.7, uncertainty=0.3,
        agreement=0.7, agreement_ratio=0.7,
        provenance=PredictionProvenance.TRAINED_MODEL,
        expected_net_edge=0.004, symbol="NIFTY",
    )
    base.update(overrides)
    return MetaOutput(**base)


def test_executable_within_window():
    ts = datetime(2024, 1, 2, 9, 30, tzinfo=UTC)
    sig = attach_expiry(_buy_signal(), ts, ttl_seconds=900)
    now = ts + timedelta(minutes=5)
    assert is_executable(sig, now=now) == SignalExecutability.EXECUTABLE


def test_expired_signal():
    ts = datetime(2024, 1, 2, 9, 30, tzinfo=UTC)
    sig = attach_expiry(_buy_signal(), ts, ttl_seconds=900)
    now = ts + timedelta(minutes=20)  # past 15-min TTL
    assert is_executable(sig, now=now) == SignalExecutability.EXPIRED


def test_exact_expiry_boundary_is_expired():
    ts = datetime(2024, 1, 2, 9, 30, tzinfo=UTC)
    sig = attach_expiry(_buy_signal(), ts, ttl_seconds=900)
    at_expiry = sig.expires_at  # exactly at expiry
    assert is_executable(sig, now=at_expiry) == SignalExecutability.EXPIRED


def test_future_signal_not_yet_valid():
    ts = datetime(2024, 1, 2, 9, 30, tzinfo=UTC)
    sig = attach_expiry(_buy_signal(), ts, ttl_seconds=900)
    before = ts - timedelta(minutes=1)
    assert is_executable(sig, now=before) == SignalExecutability.NOT_YET_VALID


def test_no_trade_non_directional():
    sig = MetaOutput(action="NO_TRADE", confidence=0.0, uncertainty=1.0,
                     agreement=0.0, agreement_ratio=0.0)
    assert is_executable(sig) == SignalExecutability.NON_DIRECTIONAL


def test_missing_timestamps():
    sig = _buy_signal()  # no expiry attached
    assert is_executable(sig) == SignalExecutability.MISSING_TIMESTAMPS


def test_heuristic_not_live_eligible_when_required():
    ts = datetime(2024, 1, 2, 9, 30, tzinfo=UTC)
    sig = attach_expiry(_buy_signal(provenance=PredictionProvenance.HEURISTIC), ts)
    now = ts + timedelta(minutes=5)
    assert is_executable(sig, now=now, require_live_eligible=True) == SignalExecutability.NOT_LIVE_ELIGIBLE
    # Without the requirement it is executable in paper/research mode.
    assert is_executable(sig, now=now, require_live_eligible=False) == SignalExecutability.EXECUTABLE


def test_attach_expiry_enforces_pit_invariant():
    ts = datetime(2024, 1, 2, 9, 30, tzinfo=UTC)
    with pytest.raises(ValueError):
        attach_expiry(_buy_signal(), ts, feature_as_of=ts + timedelta(minutes=1))


def test_valid_contract():
    ts = datetime(2024, 1, 2, 9, 30, tzinfo=UTC)
    sig = attach_expiry(_buy_signal(), ts, feature_as_of=ts - timedelta(minutes=1))
    valid, problems = validate_signal_contract(sig)
    assert valid is True
    assert problems == []


def test_contract_missing_expiry():
    sig = _buy_signal()  # no expires_at / prediction_timestamp
    valid, problems = validate_signal_contract(sig)
    assert valid is False
    assert "MISSING_EXPIRES_AT" in problems
    assert "MISSING_PREDICTION_TIMESTAMP" in problems


def test_contract_directional_needs_edge():
    ts = datetime(2024, 1, 2, 9, 30, tzinfo=UTC)
    sig = attach_expiry(_buy_signal(expected_net_edge=None), ts)
    valid, problems = validate_signal_contract(sig)
    assert valid is False
    assert "MISSING_EXPECTED_NET_EDGE" in problems
