"""Chaos-matrix tests (mandate §64): every upstream failure mode must FAIL
CLOSED (raise / block), never fabricate data or silently continue.

The DataServiceClient is the single ingress for market data. When it cannot
establish data integrity, the inference/decision path must not proceed — the
honest outcome is a blocked request (NO_TRADE), never a synthetic substitute.
"""
from __future__ import annotations

import os

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.clients.data_service import (
    DataServiceAuthError,
    DataServiceClient,
    DataServiceUnavailableError,
)


def _client() -> DataServiceClient:
    return DataServiceClient(base_url="http://test-ds", api_key="k")


def _resp(status: int, json_data=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data or {}
    if status >= 400:
        r.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err", request=MagicMock(), response=r
        )
    else:
        r.raise_for_status.return_value = None
    return r


class TestChaosHttpStatusMatrix:
    """§64: DataService 401/403/429/500/503 must raise, never return fake data."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 403])
    async def test_auth_errors_raise_auth_error(self, status):
        c = _client()
        c._client = MagicMock()
        c._client.get = AsyncMock(return_value=_resp(status))
        with pytest.raises(DataServiceAuthError):
            await c.get_historical_ohlcv("NIFTY", interval="1d")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    async def test_server_errors_raise_unavailable(self, status):
        c = _client()
        c._client = MagicMock()
        c._client.get = AsyncMock(return_value=_resp(status))
        with pytest.raises(DataServiceUnavailableError):
            await c.get_historical_ohlcv("NIFTY", interval="1d")

    @pytest.mark.asyncio
    async def test_timeout_raises_unavailable(self):
        c = _client()
        c._client = MagicMock()
        c._client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(DataServiceUnavailableError):
            await c.get_historical_ohlcv("NIFTY", interval="1d")

    @pytest.mark.asyncio
    async def test_connect_error_raises_unavailable(self):
        c = _client()
        c._client = MagicMock()
        c._client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(DataServiceUnavailableError):
            await c.get_historical_ohlcv("NIFTY", interval="1d")

    @pytest.mark.asyncio
    async def test_not_connected_raises_unavailable(self):
        """DB/Redis/MLflow/model-missing analogue: no client => cannot proceed."""
        c = _client()
        c._client = None  # connect() never called (dependency down)
        with pytest.raises(DataServiceUnavailableError):
            await c.get_historical_ohlcv("NIFTY", interval="1d")


class TestCircuitBreakerFailsClosed:
    @pytest.mark.asyncio
    async def test_open_circuit_fast_fails_without_fabricating(self):
        c = _client()
        c._client = MagicMock()
        for _ in range(3):
            c._record_failure()
        assert c._is_circuit_open() is True
        with pytest.raises(DataServiceUnavailableError):
            await c.get_historical_ohlcv("NIFTY", interval="1d")

    def test_repeated_failures_open_circuit(self):
        c = _client()
        for _ in range(3):
            c._record_failure()
        assert c._is_circuit_open() is True


class TestNoSyntheticFallback:
    @pytest.mark.asyncio
    async def test_failure_never_returns_data(self):
        """A failed fetch must RAISE — it must never return an empty/fake list."""
        c = _client()
        c._client = MagicMock()
        c._client.get = AsyncMock(return_value=_resp(503))
        raised = False
        try:
            await c.get_historical_ohlcv("NIFTY", interval="1d")
        except DataServiceUnavailableError:
            raised = True
        assert raised, "A 503 must raise, never silently return data (§64, §97)."
