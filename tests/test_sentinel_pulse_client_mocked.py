"""
Mock-based tests for src/clients/sentinel_pulse.py.
"""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_http_response(status_code: int, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


VALID_NEWS_CTX = {
    "news_impact_score": 0.7,
    "impact_direction": "BULLISH",
    "impact_confidence": 0.85,
    "sentiment": {
        "overall": 0.4,
        "market": 0.3,
        "company": 0.5,
        "macro": 0.2,
        "risk": -0.1,
    },
}


def make_client():
    from src.clients.sentinel_pulse import SentinelPulseClient

    return SentinelPulseClient(
        base_url="http://sentinel-pulse-test",
        api_key="test-sp-key",
    )


# ---------------------------------------------------------------------------
# _validate_news_context()
# ---------------------------------------------------------------------------


class TestValidateNewsContext:
    def test_valid_context_returns_true(self):
        from src.clients.sentinel_pulse import SentinelPulseClient

        client = SentinelPulseClient()
        assert client._validate_news_context(VALID_NEWS_CTX) is True

    def test_missing_top_level_field_returns_false(self):
        from src.clients.sentinel_pulse import SentinelPulseClient

        client = SentinelPulseClient()
        ctx = dict(VALID_NEWS_CTX)
        del ctx["news_impact_score"]
        assert client._validate_news_context(ctx) is False

    def test_null_top_level_field_returns_false(self):
        from src.clients.sentinel_pulse import SentinelPulseClient

        client = SentinelPulseClient()
        ctx = dict(VALID_NEWS_CTX)
        ctx["impact_direction"] = None
        assert client._validate_news_context(ctx) is False

    def test_missing_sentiment_sub_field_returns_false(self):
        from src.clients.sentinel_pulse import SentinelPulseClient

        client = SentinelPulseClient()
        ctx = {**VALID_NEWS_CTX, "sentiment": {"overall": 0.1, "market": 0.1}}  # missing company/macro/risk
        assert client._validate_news_context(ctx) is False

    def test_null_sentiment_value_returns_false(self):
        from src.clients.sentinel_pulse import SentinelPulseClient

        client = SentinelPulseClient()
        ctx = {
            **VALID_NEWS_CTX,
            "sentiment": {
                "overall": None,  # null
                "market": 0.3,
                "company": 0.5,
                "macro": 0.2,
                "risk": -0.1,
            },
        }
        assert client._validate_news_context(ctx) is False

    def test_non_dict_sentiment_returns_false(self):
        from src.clients.sentinel_pulse import SentinelPulseClient

        client = SentinelPulseClient()
        ctx = {**VALID_NEWS_CTX, "sentiment": "bad_value"}
        assert client._validate_news_context(ctx) is False


# ---------------------------------------------------------------------------
# _fetch_with_retry()
# ---------------------------------------------------------------------------


class TestFetchWithRetry:
    def test_returns_none_when_client_not_connected(self):
        async def _run():
            client = make_client()
            # _client is None
            result = await client._fetch_with_retry("/api/v1/test")
            assert result is None

        asyncio.run(_run())

    def test_returns_none_on_401(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_http_response(401)
            client._client = mock_http

            result = await client._fetch_with_retry("/api/v1/test")
            assert result is None
            # Only called once — no retry on 401
            assert mock_http.get.call_count == 1

        asyncio.run(_run())

    def test_returns_none_on_404(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_http_response(404)
            client._client = mock_http

            result = await client._fetch_with_retry("/api/v1/test")
            assert result is None

        asyncio.run(_run())

    def test_returns_data_on_200_success_true(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_http_response(
                200, {"success": True, "data": {"key": "value"}}
            )
            client._client = mock_http

            result = await client._fetch_with_retry("/api/v1/test")
            assert result == {"key": "value"}

        asyncio.run(_run())

    def test_returns_none_on_200_success_false(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_http_response(
                200, {"success": False, "error": "not found"}
            )
            client._client = mock_http

            result = await client._fetch_with_retry("/api/v1/test")
            assert result is None

        asyncio.run(_run())

    def test_retries_on_timeout_then_succeeds(self):
        """First call times out, second call succeeds."""
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.side_effect = [
                httpx.TimeoutException("timeout"),
                make_http_response(200, {"success": True, "data": {"retried": True}}),
            ]
            client._client = mock_http

            result = await client._fetch_with_retry("/api/v1/test")
            assert result == {"retried": True}
            assert mock_http.get.call_count == 2

        asyncio.run(_run())

    def test_all_retries_exhausted_returns_none(self):
        """All 3 attempts time out → returns None."""
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.side_effect = httpx.TimeoutException("timeout")
            client._client = mock_http

            result = await client._fetch_with_retry("/api/v1/test")
            assert result is None
            assert mock_http.get.call_count == 3

        asyncio.run(_run())

    def test_unexpected_status_all_attempts_returns_none(self):
        """500 on all attempts → None after 3 tries."""
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_http_response(500)
            client._client = mock_http

            result = await client._fetch_with_retry("/api/v1/test")
            assert result is None
            assert mock_http.get.call_count == 3

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# fetch_news_context()
# ---------------------------------------------------------------------------


class TestFetchNewsContext:
    def test_returns_none_on_fetch_failure(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_http_response(404)
            client._client = mock_http

            result = await client.fetch_news_context("NIFTY")
            assert result is None

        asyncio.run(_run())

    def test_returns_none_on_missing_mandatory_fields(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            # Returns data missing mandatory fields
            mock_http.get.return_value = make_http_response(
                200,
                {"success": True, "data": {"incomplete": True}},
            )
            client._client = mock_http

            result = await client.fetch_news_context("NIFTY")
            assert result is None

        asyncio.run(_run())

    def test_returns_data_when_valid(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_http_response(
                200, {"success": True, "data": VALID_NEWS_CTX}
            )
            client._client = mock_http

            result = await client.fetch_news_context("NIFTY")
            assert result == VALID_NEWS_CTX

        asyncio.run(_run())

    def test_cache_hit_skips_http_call(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            client._client = mock_http

            # Pre-populate cache
            client._news_cache.set("NIFTY", VALID_NEWS_CTX)

            result = await client.fetch_news_context("NIFTY")
            assert result == VALID_NEWS_CTX
            mock_http.get.assert_not_called()

        asyncio.run(_run())

    def test_result_cached_after_success(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_http_response(
                200, {"success": True, "data": VALID_NEWS_CTX}
            )
            client._client = mock_http

            await client.fetch_news_context("NIFTY")
            # Verify result is now in cache
            cached = client._news_cache.get("NIFTY")
            assert cached is not None

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Other public API methods
# ---------------------------------------------------------------------------


class TestOtherPublicAPI:
    def test_fetch_market_regime_happy_path(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_http_response(
                200, {"success": True, "data": {"regime": "BULL"}}
            )
            client._client = mock_http

            result = await client.fetch_market_regime()
            assert result == {"regime": "BULL"}

        asyncio.run(_run())

    def test_fetch_market_context_happy_path(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_http_response(
                200, {"success": True, "data": {"macro": "positive"}}
            )
            client._client = mock_http

            result = await client.fetch_market_context()
            assert result == {"macro": "positive"}

        asyncio.run(_run())

    def test_fetch_pit_features_look_ahead_false_returns_none(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_http_response(
                200,
                {"success": True, "data": {"look_ahead_validated": False, "features": [1, 2, 3]}},
            )
            client._client = mock_http

            result = await client.fetch_pit_features("NIFTY_asset_id")
            assert result is None

        asyncio.run(_run())

    def test_fetch_pit_features_look_ahead_true_returns_data(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_http_response(
                200,
                {
                    "success": True,
                    "data": {
                        "look_ahead_validated": True,
                        "features": [0.1, 0.2, 0.3],
                    },
                },
            )
            client._client = mock_http

            result = await client.fetch_pit_features("NIFTY_asset_id")
            assert result is not None
            assert result["look_ahead_validated"] is True

        asyncio.run(_run())

    def test_fetch_training_samples_filters_invalid(self):
        """Samples with look_ahead_validated=False are filtered out."""
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_http_response(
                200,
                {
                    "success": True,
                    "data": {
                        "samples": [
                            {"asset_id": "A", "look_ahead_validated": True, "features": [1]},
                            {"asset_id": "B", "look_ahead_validated": False, "features": [2]},
                            {"asset_id": "C", "look_ahead_validated": True, "features": [3]},
                        ],
                        "cursor": None,
                    },
                },
            )
            client._client = mock_http

            result = await client.fetch_training_samples()
            assert result is not None
            assert len(result["samples"]) == 2
            assert all(s["look_ahead_validated"] for s in result["samples"])

        asyncio.run(_run())

    def test_fetch_historical_reactions_happy_path(self):
        async def _run():
            client = make_client()
            mock_http = AsyncMock()
            mock_http.get.return_value = make_http_response(
                200,
                {"success": True, "data": {"reactions": [{"event": "earnings"}]}},
            )
            client._client = mock_http

            result = await client.fetch_historical_reactions(asset_id="NIFTY")
            assert result is not None
            assert "reactions" in result

        asyncio.run(_run())

    def test_connect_and_disconnect_lifecycle(self):
        async def _run():
            client = make_client()
            assert client._client is None
            await client.connect()
            assert client._client is not None
            await client.disconnect()
            assert client._client is None

        asyncio.run(_run())
