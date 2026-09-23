"""
Mock-based tests for src/clients/data_service.py.
"""
from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_mock_response(status_code: int = 200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def make_client():
    from src.clients.data_service import DataServiceClient

    client = DataServiceClient(
        base_url="http://test-data-service",
        api_key="test-api-key",
    )
    return client


# ---------------------------------------------------------------------------
# Circuit-breaker state
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_circuit_initially_closed(self):
        client = make_client()
        assert client._is_circuit_open() is False

    def test_circuit_opens_after_threshold_failures(self):
        client = make_client()
        for _ in range(3):
            client._record_failure()
        assert client._is_circuit_open() is True

    def test_record_success_resets_counter(self):
        client = make_client()
        client._record_failure()
        client._record_failure()
        client._record_success()
        assert client._consecutive_failures == 0
        assert client._is_circuit_open() is False

    def test_circuit_open_blocks_get_request(self):
        from src.clients.data_service import DataServiceUnavailableError

        async def _run():
            client = make_client()
            for _ in range(3):
                client._record_failure()
            assert client._is_circuit_open()

            with pytest.raises(DataServiceUnavailableError, match="circuit is open"):
                await client._get("/v1/test")

        asyncio.run(_run())

    def test_circuit_open_blocks_post_request(self):
        from src.clients.data_service import DataServiceUnavailableError

        async def _run():
            client = make_client()
            for _ in range(3):
                client._record_failure()

            with pytest.raises(DataServiceUnavailableError, match="circuit is open"):
                await client._post("/v1/test", {"key": "val"})

        asyncio.run(_run())

    def test_no_client_raises_unavailable_get(self):
        from src.clients.data_service import DataServiceUnavailableError

        async def _run():
            client = make_client()
            # client._client is None by default (connect() not called)
            with pytest.raises(DataServiceUnavailableError, match="connect()"):
                await client._get("/v1/test")

        asyncio.run(_run())

    def test_no_client_raises_unavailable_post(self):
        from src.clients.data_service import DataServiceUnavailableError

        async def _run():
            client = make_client()
            with pytest.raises(DataServiceUnavailableError, match="connect()"):
                await client._post("/v1/test", {})

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# _get() — HTTP responses
# ---------------------------------------------------------------------------


class TestGetMethod:
    def test_get_200_returns_json(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_mock_response(200, {"result": "ok"})
            client._client = mock_http

            result = await client._get("/v1/test")
            assert result == {"result": "ok"}

        asyncio.run(_run())

    def test_get_401_raises_auth_error(self):
        from src.clients.data_service import DataServiceAuthError

        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_mock_response(401)
            client._client = mock_http

            with pytest.raises(DataServiceAuthError):
                await client._get("/v1/test")

        asyncio.run(_run())

    def test_get_403_raises_auth_error(self):
        from src.clients.data_service import DataServiceAuthError

        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_mock_response(403)
            client._client = mock_http

            with pytest.raises(DataServiceAuthError):
                await client._get("/v1/test")

        asyncio.run(_run())

    def test_get_timeout_increments_failure_counter(self):
        from src.clients.data_service import DataServiceUnavailableError

        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.side_effect = httpx.TimeoutException("timed out")
            client._client = mock_http

            with pytest.raises(DataServiceUnavailableError):
                await client._get("/v1/test")

            assert client._consecutive_failures == 1

        asyncio.run(_run())

    def test_get_connect_error_raises_unavailable(self):
        from src.clients.data_service import DataServiceUnavailableError

        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.side_effect = httpx.ConnectError("refused")
            client._client = mock_http

            with pytest.raises(DataServiceUnavailableError):
                await client._get("/v1/test")

        asyncio.run(_run())

    def test_get_500_raises_unavailable_and_records_failure(self):
        from src.clients.data_service import DataServiceUnavailableError

        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_mock_response(500)
            client._client = mock_http

            with pytest.raises(DataServiceUnavailableError):
                await client._get("/v1/test")

            assert client._consecutive_failures == 1

        asyncio.run(_run())

    def test_get_success_resets_failure_counter(self):
        async def _run():
            client = make_client()
            client._consecutive_failures = 2
            mock_http = AsyncMock()
            mock_http.get.return_value = make_mock_response(200, {"ok": True})
            client._client = mock_http

            await client._get("/v1/test")
            assert client._consecutive_failures == 0

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# _post() — HTTP responses
# ---------------------------------------------------------------------------


class TestPostMethod:
    def test_post_200_returns_json(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.post.return_value = make_mock_response(200, {"created": True})
            client._client = mock_http

            result = await client._post("/v1/test", {"key": "value"})
            assert result == {"created": True}

        asyncio.run(_run())

    def test_post_401_raises_auth_error(self):
        from src.clients.data_service import DataServiceAuthError

        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.post.return_value = make_mock_response(401)
            client._client = mock_http

            with pytest.raises(DataServiceAuthError):
                await client._post("/v1/test", {})

        asyncio.run(_run())

    def test_post_timeout_raises_unavailable(self):
        from src.clients.data_service import DataServiceUnavailableError

        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.post.side_effect = httpx.TimeoutException("timed out")
            client._client = mock_http

            with pytest.raises(DataServiceUnavailableError):
                await client._post("/v1/test", {})

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------


class TestQualityGates:
    def test_signal_engine_not_allowed_raises(self):
        from src.clients.data_service import DataServiceClient, SignalEngineNotAllowedError

        client = make_client()
        resp = {
            "metadata": {
                "quality": {
                    "signalEngineAllowed": False,
                    "DataConfidenceScore": 65,
                }
            }
        }
        with pytest.raises(SignalEngineNotAllowedError):
            client._check_quality_gates(resp, "NIFTY", "2024-01-01T00:00:00Z")

    def test_low_confidence_score_raises(self):
        from src.clients.data_service import DataServiceClient, LowDataConfidenceError

        client = make_client()
        resp = {
            "metadata": {
                "quality": {
                    "signalEngineAllowed": True,
                    "DataConfidenceScore": 50,  # below default threshold of 70
                }
            }
        }
        with pytest.raises(LowDataConfidenceError):
            client._check_quality_gates(resp, "NIFTY", "2024-01-01T00:00:00Z")

    def test_high_confidence_no_exception(self):
        client = make_client()
        resp = {
            "metadata": {
                "quality": {
                    "signalEngineAllowed": True,
                    "DataConfidenceScore": 90,
                }
            }
        }
        # Should not raise
        client._check_quality_gates(resp, "NIFTY", "2024-01-01T00:00:00Z")

    def test_flat_payload_signal_engine_not_allowed(self):
        from src.clients.data_service import SignalEngineNotAllowedError

        client = make_client()
        resp = {"signalEngineAllowed": False}
        with pytest.raises(SignalEngineNotAllowedError):
            client._check_quality_gates(resp, "NIFTY", "2024-01-01T00:00:00Z")

    def test_no_quality_fields_no_exception(self):
        client = make_client()
        resp = {"data": {"price": 100.0}}
        # No quality metadata → should not raise
        client._check_quality_gates(resp, "NIFTY", "2024-01-01T00:00:00Z")


# ---------------------------------------------------------------------------
# Public API methods
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_get_historical_ohlcv_3m_raises_value_error(self):
        async def _run():
            client = make_client()
            with pytest.raises(ValueError, match="3m"):
                await client.get_historical_ohlcv("NIFTY", interval="3m")

        asyncio.run(_run())

    def test_get_market_status_calls_get(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_mock_response(200, {"isOpen": True})
            client._client = mock_http

            result = await client.get_market_status()
            assert result == {"isOpen": True}

        asyncio.run(_run())

    def test_get_live_quote_triggers_quality_gate(self):
        """Live quote with signalEngineAllowed=False should raise."""
        from src.clients.data_service import SignalEngineNotAllowedError

        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_mock_response(
                200,
                {
                    "signalEngineAllowed": False,
                    "data": {"price": 19000.0},
                },
            )
            client._client = mock_http

            with pytest.raises(SignalEngineNotAllowedError):
                await client.get_live_quote("NIFTY")

        asyncio.run(_run())

    def test_connect_creates_client(self):
        async def _run():
            client = make_client()
            assert client._client is None
            await client.connect()
            assert client._client is not None
            await client.disconnect()

        asyncio.run(_run())

    def test_disconnect_without_connect_is_safe(self):
        async def _run():
            client = make_client()
            # Should not raise
            await client.disconnect()

        asyncio.run(_run())

    def test_get_fno_universe_happy_path(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_mock_response(
                200, {"instruments": ["NIFTY", "BANKNIFTY"]}
            )
            client._client = mock_http

            result = await client.get_fno_universe()
            assert "instruments" in result

        asyncio.run(_run())

    def test_get_option_chain_happy_path(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_mock_response(
                200, {"metadata": {"quality": {"signalEngineAllowed": True, "DataConfidenceScore": 95}}, "calls": [], "puts": []}
            )
            client._client = mock_http

            result = await client.get_option_chain("NIFTY")
            assert "calls" in result

        asyncio.run(_run())

    def test_get_crypto_klines_happy_path(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_mock_response(
                200, {"data": [{"open": 50000, "close": 51000}]}
            )
            client._client = mock_http

            result = await client.get_crypto_klines("BTCUSDT")
            assert isinstance(result, list)

        asyncio.run(_run())

    def test_evaluate_data_quality_happy_path(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.post.return_value = make_mock_response(200, {"score": 85})
            client._client = mock_http

            result = await client.evaluate_data_quality({"key": "value"})
            assert result == {"score": 85}

        asyncio.run(_run())
