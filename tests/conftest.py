"""
tests/conftest.py — shared fixtures for the ml-service2.0 test suite.

Provides:
  - Environment variable bootstrapping (must happen before any src imports).
  - ``app_client``       — FastAPI TestClient (session-scoped).
  - ``auth_headers``     — standard X-API-KEY header dict.
  - ``mock_ohlcv_response``  — realistic data-service2.0 OHLCV payload.
  - ``mock_live_quote``      — realistic data-service2.0 live-quote envelope.
  - ``mock_sentinel_pulse``  — realistic SentinelPulse news impact vector.
  - ``mock_data_client``     — AsyncMock DataServiceClient pre-wired with fixtures.
  - ``mock_news_client``     — AsyncMock SentinelPulseClient pre-wired with fixtures.
  - ``pit_safe_timestamp``   — a UTC datetime well within trading hours for PIT tests.
  - ``past_timestamp``       — a UTC datetime 15 minutes BEFORE pit_safe_timestamp.
  - ``future_timestamp``     — a UTC datetime 30 minutes AFTER pit_safe_timestamp
                               (use to trigger PIT violations in tests).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Env-var bootstrap ─────────────────────────────────────────────────────────
# Must be set before any `src.*` import so pydantic-settings does not raise
# on missing required fields.
#
# The API-key vars are FORCED (assignment, not setdefault) so the test suite is
# hermetic: the whole suite hardcodes the "test-key-for-testing" X-API-KEY, so
# the app's configured key must match regardless of any ML_SERVICE_API_KEY value
# the caller may have exported in their shell. Without this, running the suite
# with a different ML_SERVICE_API_KEY in the environment produces spurious 401s
# on every authenticated endpoint test.
os.environ["ML_SERVICE_API_KEY"] = "test-key-for-testing"
os.environ["DATA_SERVICE_API_KEY"] = "test-data-key"
os.environ.setdefault("DATA_SERVICE_2_URL", "http://localhost:8200")
os.environ.setdefault("SENTINEL_PULSE_URL", "http://localhost:3001")


# ── FastAPI test client ───────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def app_client():  # type: ignore[no-untyped-def]
    """Session-scoped TestClient for integration tests within the same process."""
    from fastapi.testclient import TestClient

    from src.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Standard auth headers for API calls."""
    return {"X-API-KEY": "test-key-for-testing"}


# ── Canonical timestamps ──────────────────────────────────────────────────────


@pytest.fixture
def pit_safe_timestamp() -> datetime:
    """A UTC datetime representing the PIT boundary for feature computation.

    Chosen to be a non-controversial NSE trading session time
    (2024-01-15 09:30 UTC = 15:00 IST, within market hours).
    """
    return datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def past_timestamp(pit_safe_timestamp: datetime) -> datetime:
    """15 minutes BEFORE the PIT boundary — valid source data timestamp."""
    from datetime import timedelta
    return pit_safe_timestamp - timedelta(minutes=15)


@pytest.fixture
def future_timestamp(pit_safe_timestamp: datetime) -> datetime:
    """30 minutes AFTER the PIT boundary — triggers PIT violations when used as source data."""
    from datetime import timedelta
    return pit_safe_timestamp + timedelta(minutes=30)


# ── data-service2.0 mock payloads ─────────────────────────────────────────────


@pytest.fixture
def mock_ohlcv_bars() -> list[dict[str, Any]]:
    """210 daily OHLCV bars ending 2024-01-15 — sufficient for Alpha158 (needs ≥ 200).

    Prices simulate a mild bull-trend in NIFTY50 (18000 → 22300 over ~840 days).
    Volume follows a random-walk around 1.5M shares/day.
    """
    import math

    bars: list[dict[str, Any]] = []
    base_close = 18000.0
    base_vol = 1_500_000.0

    for i in range(210):
        # Gentle upward drift with sin-wave noise
        trend = base_close + i * 20.0
        noise = 150.0 * math.sin(i * 0.3)
        close = round(trend + noise, 2)
        open_ = round(close - 50.0 + (i % 7) * 15.0, 2)
        high = round(max(open_, close) + 80.0, 2)
        low = round(min(open_, close) - 60.0, 2)
        vol = round(base_vol + (i % 5) * 200_000.0, 0)
        vwap = round((open_ + high + low + close) / 4, 2)

        # Timestamps: 210 trading days ending 2024-01-15
        day_offset = 210 - i
        # Use approximate 1-day spacing (skip weekends in real impl; test data is fine without)
        ts_epoch = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc).timestamp()
        bar_ts = datetime.fromtimestamp(ts_epoch - day_offset * 86400, tz=timezone.utc)

        bars.append({
            "timestamp": bar_ts.isoformat(),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "vwap": vwap,
        })

    return bars


@pytest.fixture
def mock_live_quote(past_timestamp: datetime) -> dict[str, Any]:
    """Realistic data-service2.0 live-quote envelope for NIFTY.

    Mirrors the JSON shape parsed by ``DataServiceClient.get_live_quote()``:
      ``data``     — OHLCV fields + instrument metadata
      ``metadata`` — quality gates (score, signalEngineAllowed, dataAsOf)

    Source timestamp (``dataAsOf``) is set to ``past_timestamp`` — 15 minutes
    before the PIT boundary — so PIT validation passes by default.
    """
    return {
        "data": {
            "instrumentId": "NSE:NIFTY:IDX",
            "symbol": "NIFTY",
            "close": 22_150.35,
            "open": 22_050.10,
            "high": 22_250.80,
            "low": 21_980.45,
            "volume": 1_234_567.0,
            "vwap": 22_110.50,
            "india_vix": 13.45,
            "put_call_ratio": 0.87,
            "change_pct": 0.47,
            "previous_close": 22_046.30,
        },
        "metadata": {
            "quality": {
                "score": 88,
                "signalEngineAllowed": True,
            },
            "dataAsOf": past_timestamp.isoformat(),
        },
    }


@pytest.fixture
def mock_live_quote_low_confidence(past_timestamp: datetime) -> dict[str, Any]:
    """Live-quote envelope with DataConfidenceScore=45 (below threshold of 70).

    Used to test ``PredictionProvenance.INSUFFICIENT_EVIDENCE`` branch.
    """
    return {
        "data": {
            "instrumentId": "NSE:NIFTY:IDX",
            "symbol": "NIFTY",
            "close": 22_150.35,
            "volume": 500_000.0,
        },
        "metadata": {
            "quality": {
                "score": 45,
                "signalEngineAllowed": True,
            },
            "dataAsOf": past_timestamp.isoformat(),
        },
    }


@pytest.fixture
def mock_live_quote_signal_blocked(past_timestamp: datetime) -> dict[str, Any]:
    """Live-quote envelope with ``signalEngineAllowed=False``.

    Used to test ``PredictionProvenance.UNAVAILABLE`` branch.
    """
    return {
        "data": {
            "instrumentId": "NSE:NIFTY:IDX",
            "symbol": "NIFTY",
            "close": 22_150.35,
            "volume": 500_000.0,
        },
        "metadata": {
            "quality": {
                "score": 92,
                "signalEngineAllowed": False,
            },
            "dataAsOf": past_timestamp.isoformat(),
        },
    }


# ── SentinelPulse mock payloads ───────────────────────────────────────────────


@pytest.fixture
def mock_sentinel_pulse(past_timestamp: datetime) -> dict[str, Any]:
    """Realistic SentinelPulse news impact vector for NIFTY.

    Shape mirrors ``GET /api/v1/alphaforge/news-context/NIFTY``:
      - ``news_impact_score``  : 0.512 (moderate impact)
      - ``impact_direction``   : ``"BULLISH"``
      - ``impact_confidence``  : 0.73
      - ``sentiment``          : five-axis breakdown
      - ``market_regime``      : ``"RISK_ON"``
      - ``as_of``              : ``past_timestamp`` (PIT-safe)

    Source timestamp is set to ``past_timestamp`` — 15 minutes before the
    PIT boundary — so PIT validation passes by default.
    """
    return {
        "instrument": "NIFTY",
        "as_of": past_timestamp.isoformat(),
        "news_impact_score": 0.512,
        "impact_direction": "BULLISH",
        "impact_confidence": 0.73,
        "sentiment": {
            "overall": 0.43,
            "market": 0.38,
            "company": 0.51,
            "macro": -0.05,
            "risk": -0.09,
        },
        "market_regime": "RISK_ON",
        "event_tags": ["RBI_POLICY", "FII_INFLOW"],
    }


@pytest.fixture
def mock_sentinel_pulse_bearish(past_timestamp: datetime) -> dict[str, Any]:
    """SentinelPulse payload with BEARISH impact for negative-signal tests."""
    return {
        "instrument": "NIFTY",
        "as_of": past_timestamp.isoformat(),
        "news_impact_score": 0.71,
        "impact_direction": "BEARISH",
        "impact_confidence": 0.82,
        "sentiment": {
            "overall": -0.55,
            "market": -0.48,
            "company": -0.61,
            "macro": -0.30,
            "risk": 0.65,
        },
        "market_regime": "RISK_OFF",
        "event_tags": ["GLOBAL_SELLOFF", "FII_OUTFLOW"],
    }


@pytest.fixture
def mock_sentinel_pulse_neutral() -> dict[str, Any]:
    """SentinelPulse payload with NEUTRAL impact — represents quiet news day."""
    return {
        "instrument": "NIFTY",
        "as_of": datetime(2024, 1, 15, 9, 14, 0, tzinfo=timezone.utc).isoformat(),
        "news_impact_score": 0.05,
        "impact_direction": "NEUTRAL",
        "impact_confidence": 0.60,
        "sentiment": {
            "overall": 0.02,
            "market": 0.01,
            "company": 0.03,
            "macro": -0.01,
            "risk": -0.02,
        },
        "market_regime": "NEUTRAL",
        "event_tags": [],
    }


# ── Pre-wired mock clients ────────────────────────────────────────────────────


@pytest.fixture
def mock_data_client(
    mock_live_quote: dict[str, Any],
    mock_ohlcv_bars: list[dict[str, Any]],
) -> AsyncMock:
    """AsyncMock DataServiceClient pre-wired with healthy mock responses.

    ``get_live_quote``       → ``mock_live_quote`` (score=88, allowed=True)
    ``get_historical_ohlcv`` → ``mock_ohlcv_bars`` (210 bars, Alpha158-ready)
    """
    client = AsyncMock()
    client.get_live_quote = AsyncMock(return_value=mock_live_quote)
    client.get_historical_ohlcv = AsyncMock(return_value=mock_ohlcv_bars)
    return client


@pytest.fixture
def mock_news_client(mock_sentinel_pulse: dict[str, Any]) -> AsyncMock:
    """AsyncMock SentinelPulseClient pre-wired with a healthy bullish news context."""
    client = AsyncMock()
    client.fetch_news_context = AsyncMock(return_value=mock_sentinel_pulse)
    return client


@pytest.fixture
def mock_news_client_unavailable() -> AsyncMock:
    """AsyncMock SentinelPulseClient that returns None (simulates unreachable service)."""
    client = AsyncMock()
    client.fetch_news_context = AsyncMock(return_value=None)
    return client


# ── Canonical model-output helpers ───────────────────────────────────────────


def make_model_output(
    model_id: str,
    action: str = "WAIT",
    direction: int = 0,
    confidence: float = 0.5,
    provenance: str = "trained_model",
) -> dict[str, Any]:
    """Build a single model-output dict in the format expected by MetaDecisionEngine.

    Args:
        model_id:   Unique model identifier.
        action:     One of ``BUY | SELL | WAIT | NO_TRADE``.
        direction:  -1 (SELL), 0 (WAIT/neutral), +1 (BUY).
        confidence: Float in [0, 1].
        provenance: ``PredictionProvenance`` value string.
    """
    return {
        "model_id": model_id,
        "action": action,
        "confidence": confidence,
        "direction": direction,
        "provenance": provenance,
    }


@pytest.fixture
def all_buy_outputs() -> list[dict[str, Any]]:
    """7 models all saying BUY with TRAINED_MODEL provenance — maximum agreement."""
    return [
        make_model_output(f"model_{i}", action="BUY", direction=1, confidence=0.80)
        for i in range(7)
    ]


@pytest.fixture
def all_unavailable_outputs() -> list[dict[str, Any]]:
    """7 models all with UNAVAILABLE provenance — triggers absorbing-identity (NO_TRADE)."""
    return [
        make_model_output(f"model_{i}", action="WAIT", direction=0, provenance="unavailable")
        for i in range(7)
    ]


@pytest.fixture
def low_agreement_outputs() -> list[dict[str, Any]]:
    """3 BUY + 4 WAIT from 7 models — agreement_ratio = 3/7 ≈ 0.43 < 0.5 → abstention."""
    return [
        make_model_output(f"model_{i}", action="BUY" if i < 3 else "WAIT", direction=1 if i < 3 else 0)
        for i in range(7)
    ]
