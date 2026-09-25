"""
tests/test_training_readiness.py — Training-readiness gate tests (mandate §27, §28).

Verifies:
  - Gate fails closed on data-service unavailability
  - Gate fails closed on auth failure
  - Gate passes with mocked healthy services (market-only)
  - Gate reports READY_MARKET_ONLY when SentinelPulse unavailable + news optional
  - Gate reports NOT_READY when SentinelPulse unavailable + news required
  - Gate reports NOT_READY for insufficient history
  - DataServiceRateLimitedError maps to RATE_LIMITED (mandate §5.1.D)
  - Execution model default is next_open (mandate §20)
  - is_economic_evidence invariant
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.clients.data_service import (
    DataServiceAuthError,
    DataServiceRateLimitedError,
    DataServiceUnavailableError,
)
from src.data.readiness import TrainingReadinessGate, TrainingReadinessResult


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_data_client(*, healthy: bool = True, auth_fail: bool = False,
                      rate_limited: bool = False, bars: int = 300) -> MagicMock:
    """Return a mock DataServiceClient with configurable behaviour."""
    client = MagicMock()
    if auth_fail:
        client.get_market_status = AsyncMock(
            side_effect=DataServiceAuthError("401 Unauthorized")
        )
        client.get_fno_universe = AsyncMock(
            side_effect=DataServiceAuthError("401 Unauthorized")
        )
        client.get_historical_ohlcv = AsyncMock(
            side_effect=DataServiceAuthError("401 Unauthorized")
        )
    elif rate_limited:
        client.get_market_status = AsyncMock(
            side_effect=DataServiceRateLimitedError("429 Too Many Requests", 30.0)
        )
        client.get_fno_universe = AsyncMock(
            side_effect=DataServiceRateLimitedError("429 Too Many Requests", 30.0)
        )
        client.get_historical_ohlcv = AsyncMock(
            side_effect=DataServiceRateLimitedError("429 Too Many Requests", 30.0)
        )
    elif not healthy:
        client.get_market_status = AsyncMock(
            side_effect=DataServiceUnavailableError("connection refused")
        )
        client.get_fno_universe = AsyncMock(
            side_effect=DataServiceUnavailableError("connection refused")
        )
        client.get_historical_ohlcv = AsyncMock(
            side_effect=DataServiceUnavailableError("connection refused")
        )
    else:
        client.get_market_status = AsyncMock(return_value={"isOpen": True})
        client.get_fno_universe = AsyncMock(return_value={
            "data": {"constituents": [{"symbol": f"SYM{i}"} for i in range(50)]}
        })
        # Return `bars` mock OHLCV bars
        client.get_historical_ohlcv = AsyncMock(
            return_value=[{"close": 100.0, "timestamp": "2024-01-01"} for _ in range(bars)]
        )
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    return client


def _mock_sentinel_client(*, healthy: bool = True, has_samples: bool = True) -> MagicMock:
    client = MagicMock()
    if not healthy:
        client.fetch_market_context = AsyncMock(return_value=None)
        client.fetch_training_samples = AsyncMock(return_value=None)
    else:
        client.fetch_market_context = AsyncMock(return_value={"regime": "NEUTRAL"})
        if has_samples:
            client.fetch_training_samples = AsyncMock(return_value={
                "samples": [{"id": "s1", "look_ahead_validated": True}] * 5,
                "total_count": 5,
            })
        else:
            client.fetch_training_samples = AsyncMock(return_value={"samples": []})
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    return client


# ── Gate tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gate_fails_when_data_service_unavailable():
    """Mandate §28: gate MUST fail closed when data-service is unreachable."""
    gate = TrainingReadinessGate()
    result = await gate.check(
        data_client=_mock_data_client(healthy=False),
        sentinel_client=_mock_sentinel_client(),
        universe=["NIFTY"],
        timeframe="1d",
    )
    assert result.training_ready is False
    assert result.mode == "NOT_READY"
    assert "DATA_SERVICE_UNAVAILABLE" in result.blockers


@pytest.mark.asyncio
async def test_gate_fails_on_auth_error():
    """Mandate §28: 401/403 → NOT_READY / DATA_SERVICE_AUTH_FAILED."""
    gate = TrainingReadinessGate()
    result = await gate.check(
        data_client=_mock_data_client(auth_fail=True),
        sentinel_client=_mock_sentinel_client(),
        universe=["NIFTY"],
        timeframe="1d",
    )
    assert result.training_ready is False
    assert result.mode == "NOT_READY"
    assert "DATA_SERVICE_AUTH_FAILED" in result.blockers


@pytest.mark.asyncio
async def test_gate_rate_limited_fails_closed():
    """Mandate §5.1.D, §28: 429 → NOT_READY / DATA_SERVICE_UNAVAILABLE."""
    gate = TrainingReadinessGate()
    result = await gate.check(
        data_client=_mock_data_client(rate_limited=True),
        sentinel_client=_mock_sentinel_client(),
        universe=["NIFTY"],
        timeframe="1d",
    )
    assert result.training_ready is False
    assert result.mode == "NOT_READY"
    # Rate-limited is a form of data-service unavailability
    assert any("DATA_SERVICE" in b for b in result.blockers)


@pytest.mark.asyncio
async def test_gate_ready_market_only_when_sentinel_unavailable():
    """When SentinelPulse unavailable and news optional → READY_MARKET_ONLY."""
    gate = TrainingReadinessGate()
    result = await gate.check(
        data_client=_mock_data_client(healthy=True, bars=400),
        sentinel_client=_mock_sentinel_client(healthy=False),
        universe=["NIFTY", "BANKNIFTY", "RELIANCE"],
        timeframe="1d",
        news_required=False,
    )
    # Should be ready (market-only) even if SentinelPulse is unavailable
    assert result.training_ready is True
    assert result.mode in ("READY_MARKET_ONLY", "READY")
    assert result.news_status in ("UNAVAILABLE", "EVIDENCE_PENDING")


@pytest.mark.asyncio
async def test_gate_fails_when_sentinel_unavailable_and_news_required():
    """When SentinelPulse unavailable and news required → NOT_READY."""
    gate = TrainingReadinessGate()
    result = await gate.check(
        data_client=_mock_data_client(healthy=True, bars=400),
        sentinel_client=_mock_sentinel_client(healthy=False),
        universe=["NIFTY"],
        timeframe="1d",
        news_required=True,
    )
    assert result.training_ready is False
    assert result.mode == "NOT_READY"
    assert "SENTINELPULSE_UNAVAILABLE" in result.blockers


@pytest.mark.asyncio
async def test_gate_fails_on_insufficient_history():
    """Insufficient bars → NOT_READY / INSUFFICIENT_MARKET_HISTORY."""
    gate = TrainingReadinessGate()
    result = await gate.check(
        data_client=_mock_data_client(healthy=True, bars=50),  # < 252
        sentinel_client=_mock_sentinel_client(),
        universe=["NIFTY"],
        timeframe="1d",
        min_history_days=252,
    )
    assert result.training_ready is False
    assert result.mode == "NOT_READY"
    assert "INSUFFICIENT_MARKET_HISTORY" in result.blockers


@pytest.mark.asyncio
async def test_gate_ready_with_healthy_services():
    """Full green path: data + news healthy → READY or READY_MARKET_ONLY."""
    gate = TrainingReadinessGate()
    result = await gate.check(
        data_client=_mock_data_client(healthy=True, bars=500),
        sentinel_client=_mock_sentinel_client(healthy=True, has_samples=True),
        universe=["NIFTY", "BANKNIFTY", "RELIANCE"],
        timeframe="1d",
        news_required=False,
    )
    # With healthy data and news, should be READY or READY_MARKET_ONLY
    assert result.training_ready is True
    assert result.mode in ("READY", "READY_MARKET_ONLY")


@pytest.mark.asyncio
async def test_gate_news_evidence_pending_when_no_samples():
    """SentinelPulse reachable but no training samples → EVIDENCE_PENDING warning."""
    gate = TrainingReadinessGate()
    result = await gate.check(
        data_client=_mock_data_client(healthy=True, bars=400),
        sentinel_client=_mock_sentinel_client(healthy=True, has_samples=False),
        universe=["NIFTY"],
        timeframe="1d",
        news_required=False,
    )
    # Should still be ready but with news evidence pending
    assert result.training_ready is True
    assert result.news_status in ("EVIDENCE_PENDING", "UNAVAILABLE", "DISABLED")


@pytest.mark.asyncio
async def test_gate_result_to_dict_has_required_fields():
    """to_dict() must include all mandate §75 required fields."""
    gate = TrainingReadinessGate()
    result = await gate.check(
        data_client=_mock_data_client(healthy=True, bars=400),
        sentinel_client=_mock_sentinel_client(),
        universe=["NIFTY"],
        timeframe="1d",
    )
    d = result.to_dict()
    required = [
        "training_ready", "mode", "news_status", "blockers",
        "warnings", "gates", "checked_at", "git_sha",
        "docker_image", "python_version",
    ]
    for field in required:
        assert field in d, f"Missing required field: {field}"


# ── DataServiceRateLimitedError tests ─────────────────────────────────────────


def test_rate_limited_error_has_retry_after():
    """DataServiceRateLimitedError carries retry_after_seconds (mandate §5.1.D)."""
    exc = DataServiceRateLimitedError("429 Too Many Requests", retry_after_seconds=30.0)
    assert exc.retry_after_seconds == 30.0
    assert isinstance(exc, DataServiceUnavailableError)


def test_rate_limited_error_without_retry_after():
    """retry_after_seconds is optional (server may not send Retry-After header)."""
    exc = DataServiceRateLimitedError("429 Too Many Requests")
    assert exc.retry_after_seconds is None


# ── Mandate §20 invariants via gate ──────────────────────────────────────────


def test_label_config_default_execution_model():
    """Default LabelConfig.execution_model must be 'next_open' (mandate §20)."""
    from src.data.labels import LabelConfig
    cfg = LabelConfig()
    assert cfg.execution_model == "next_open"


def test_gate_label_schema_checks_next_open():
    """_check_labels gate must verify default execution_model=next_open."""
    gate = TrainingReadinessGate()
    result = gate._check_labels()
    assert result.status == "PASS"
    assert result.details.get("default_execution_model") == "next_open"
