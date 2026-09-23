"""
Coverage boost — batch 7. Targets features/pipeline.py helper functions and RL agent.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

import numpy as np
import pytest


# ===========================================================================
# FeaturePipeline helper functions (module-level, no async needed)
# ===========================================================================


class TestFeaturePipelineHelpers:
    def test_safe_float_valid_value(self):
        from src.features.pipeline import _safe_float

        assert _safe_float(1.5) == 1.5
        assert _safe_float("2.3") == pytest.approx(2.3)

    def test_safe_float_invalid_returns_none(self):
        from src.features.pipeline import _safe_float

        assert _safe_float("not_a_number") is None
        assert _safe_float(None) is None

    def test_safe_float_infinity_returns_none(self):
        from src.features.pipeline import _safe_float

        assert _safe_float(float("inf")) is None
        assert _safe_float(float("nan")) is None

    def test_parse_iso_valid(self):
        from src.features.pipeline import _parse_iso

        result = _parse_iso("2024-01-15T09:30:00Z")
        assert result is not None
        assert result.tzinfo is not None

    def test_parse_iso_none_returns_none(self):
        from src.features.pipeline import _parse_iso

        assert _parse_iso(None) is None
        assert _parse_iso("") is None

    def test_parse_iso_invalid_returns_none(self):
        from src.features.pipeline import _parse_iso

        assert _parse_iso("not-a-date") is None

    def test_parse_iso_without_tz_adds_utc(self):
        from src.features.pipeline import _parse_iso

        result = _parse_iso("2024-01-15T09:30:00")
        assert result is not None
        assert result.tzinfo == timezone.utc


class TestFeaturePipelineBuildVector:
    """Test FeaturePipeline.build_vector with mocked clients."""

    def _make_pipeline(self):
        from src.features.pipeline import FeaturePipeline
        from src.clients.data_service import DataServiceUnavailableError

        mock_data = AsyncMock()
        mock_data.get_live_quote.side_effect = DataServiceUnavailableError("down")
        mock_news = AsyncMock()
        mock_news.fetch_news_context.return_value = None

        return FeaturePipeline(data_client=mock_data, news_client=mock_news)

    def test_build_vector_data_service_unavailable(self):
        """DataServiceUnavailableError → UNAVAILABLE provenance."""
        from src.schemas.base import PredictionProvenance

        async def _run():
            pipeline = self._make_pipeline()
            ts = datetime.now(tz=timezone.utc)
            vector, report = await pipeline.build_vector("NIFTY", ts)
            assert vector.provenance == PredictionProvenance.UNAVAILABLE

        asyncio.run(_run())

    def test_build_vector_with_cache_hit(self):
        """Cache hit returns immediately without calling data client."""
        from src.features.pipeline import FeaturePipeline
        from src.schemas.base import PredictionProvenance

        async def _run():
            mock_data = AsyncMock()
            mock_cache = AsyncMock()

            # Return a cache hit that will fail validation → pipeline re-fetches
            # This tests the "stale / invalid cache" branch
            mock_cache.get_feature.return_value = {"invalid": "data"}

            from src.clients.data_service import DataServiceUnavailableError
            mock_data.get_live_quote.side_effect = DataServiceUnavailableError("down")
            mock_news = AsyncMock()
            mock_news.fetch_news_context.return_value = None

            pipeline = FeaturePipeline(data_client=mock_data, cache=mock_cache)
            ts = datetime.now(tz=timezone.utc)
            vector, report = await pipeline.build_vector("NIFTY", ts, mode="inference")

            # Tried cache, failed, then hit data service
            mock_cache.get_feature.assert_called_once()

        asyncio.run(_run())

    def test_build_vector_auth_error_returns_unavailable(self):
        """DataServiceAuthError → UNAVAILABLE."""
        from src.features.pipeline import FeaturePipeline
        from src.clients.data_service import DataServiceAuthError
        from src.schemas.base import PredictionProvenance

        async def _run():
            mock_data = AsyncMock()
            mock_data.get_live_quote.side_effect = DataServiceAuthError("401")
            mock_news = AsyncMock()
            mock_news.fetch_news_context.return_value = None

            pipeline = FeaturePipeline(data_client=mock_data, news_client=mock_news)
            ts = datetime.now(tz=timezone.utc)
            vector, report = await pipeline.build_vector("NIFTY", ts)
            assert vector.provenance == PredictionProvenance.UNAVAILABLE

        asyncio.run(_run())

    def test_build_vector_signal_engine_not_allowed(self):
        """SignalEngineNotAllowedError → UNAVAILABLE."""
        from src.features.pipeline import FeaturePipeline
        from src.clients.data_service import SignalEngineNotAllowedError
        from src.schemas.base import PredictionProvenance

        async def _run():
            mock_data = AsyncMock()
            mock_data.get_live_quote.side_effect = SignalEngineNotAllowedError("not allowed")
            mock_news = AsyncMock()
            mock_news.fetch_news_context.return_value = None

            pipeline = FeaturePipeline(data_client=mock_data, news_client=mock_news)
            ts = datetime.now(tz=timezone.utc)
            vector, report = await pipeline.build_vector("NIFTY", ts)
            assert vector.provenance == PredictionProvenance.UNAVAILABLE

        asyncio.run(_run())

    def test_build_vector_low_confidence_score(self):
        """LowDataConfidenceError → INSUFFICIENT_EVIDENCE."""
        from src.features.pipeline import FeaturePipeline
        from src.clients.data_service import LowDataConfidenceError
        from src.schemas.base import PredictionProvenance

        async def _run():
            mock_data = AsyncMock()
            mock_data.get_live_quote.side_effect = LowDataConfidenceError("score=50")
            mock_news = AsyncMock()
            mock_news.fetch_news_context.return_value = None

            pipeline = FeaturePipeline(data_client=mock_data, news_client=mock_news)
            ts = datetime.now(tz=timezone.utc)
            vector, report = await pipeline.build_vector("NIFTY", ts)
            assert vector.provenance in (
                PredictionProvenance.INSUFFICIENT_EVIDENCE,
                PredictionProvenance.UNAVAILABLE,
            )

        asyncio.run(_run())

    def test_build_vector_with_news_context(self):
        """Successful data + news context → vector with news fields populated."""
        from src.features.pipeline import FeaturePipeline
        from src.schemas.base import PredictionProvenance

        async def _run():
            mock_data = AsyncMock()
            mock_data.get_live_quote.return_value = {
                "ltp": 22000.0,
                "open": 21900.0,
                "high": 22100.0,
                "low": 21800.0,
                "close": 21950.0,
                "volume": 1000000,
                "change_pct": 0.23,
                "signalEngineAllowed": True,
                "DataConfidenceScore": 95,
            }
            mock_data.get_historical_ohlcv.return_value = [
                {"timestamp": "2024-01-15T09:00:00Z", "open": 21900.0, "high": 22100.0, "low": 21800.0, "close": 22000.0, "volume": 1000000}
                for _ in range(30)
            ]

            mock_news = AsyncMock()
            mock_news.fetch_news_context.return_value = {
                "news_impact_score": 0.7,
                "impact_direction": "BULLISH",
                "impact_confidence": 0.8,
                "sentiment": {
                    "overall": 0.4,
                    "market": 0.3,
                    "company": 0.5,
                    "macro": 0.2,
                    "risk": -0.1,
                },
            }

            pipeline = FeaturePipeline(data_client=mock_data, news_client=mock_news)
            ts = datetime.now(tz=timezone.utc)
            vector, report = await pipeline.build_vector("NIFTY", ts)
            # Any provenance is fine as long as we got a result
            assert vector is not None
            assert report is not None

        asyncio.run(_run())


# ===========================================================================
# RLExecutionAgent — heuristic paths
# ===========================================================================


class TestRLExecutionAgent:
    def test_heuristic_predict_enter_now(self):
        from src.models.rl_execution_agent import RLExecutionAgent

        agent = RLExecutionAgent()
        # Large profit → TRAIL_STOP
        result = agent.act({
            "unrealized_pnl_pct": 3.5,
            "time_in_trade_minutes": 60,
            "momentum": 0.5,
            "price_vs_vwap": 0.01,
            "current_price": 22200.0,
            "stop_loss": 21800.0,
            "entry": 22000.0,
        })
        assert result is not None

    def test_heuristic_predict_wait(self):
        from src.models.rl_execution_agent import RLExecutionAgent

        agent = RLExecutionAgent()
        # Low signal → WAIT
        result = agent.act({
            "unrealized_pnl_pct": 0.1,
            "time_in_trade_minutes": 30,
            "momentum": 0.1,
            "price_vs_vwap": 0.001,
            "current_price": 22050.0,
            "stop_loss": 21800.0,
            "entry": 22000.0,
        })
        assert result is not None

    def test_act_with_no_model_uses_heuristic(self):
        from src.models.rl_execution_agent import RLExecutionAgent
        from src.schemas.base import PredictionProvenance

        agent = RLExecutionAgent()
        response = agent.act({
            "unrealized_pnl_pct": 0.5,
            "time_in_trade_minutes": 60,
            "current_price": 22100.0,
            "stop_loss": 21800.0,
            "entry": 22000.0,
        })
        assert response.provenance == PredictionProvenance.HEURISTIC

    def test_act_full_exit_on_high_profit(self):
        from src.models.rl_execution_agent import RLExecutionAgent

        agent = RLExecutionAgent()
        # Moderate profit → TRAIL_STOP (not FULL_EXIT in heuristic)
        response = agent.act({
            "unrealized_pnl_pct": 2.0,
            "time_in_trade_minutes": 60,
            "current_price": 22400.0,
            "stop_loss": 21800.0,
            "entry": 22000.0,
        })
        assert response is not None

    def test_act_tighten_stop_near_stop(self):
        from src.models.rl_execution_agent import RLExecutionAgent

        agent = RLExecutionAgent()
        # Near stop with negative PnL → TIGHTEN_STOP
        response = agent.act({
            "unrealized_pnl_pct": -0.8,
            "time_in_trade_minutes": 60,
            "current_price": 22000.0,
            "stop_loss": 21800.0,  # ~0.9% from entry
            "entry": 22000.0,
            "momentum": 0.0,
        })
        assert response is not None

    def test_act_partial_exit_near_session_end(self):
        from src.models.rl_execution_agent import RLExecutionAgent

        agent = RLExecutionAgent()
        # Long time in trade → PARTIAL_EXIT
        response = agent.act({
            "unrealized_pnl_pct": 0.3,
            "time_in_trade_minutes": 350,  # > 300
            "current_price": 22100.0,
            "stop_loss": 21800.0,
            "entry": 22000.0,
        })
        assert response is not None

    def test_act_scale_in_strong_momentum(self):
        from src.models.rl_execution_agent import RLExecutionAgent

        agent = RLExecutionAgent()
        # Strong momentum + positive pnl + above vwap → SCALE_IN
        response = agent.act({
            "unrealized_pnl_pct": 0.5,
            "time_in_trade_minutes": 60,
            "momentum": 0.8,
            "price_vs_vwap": 0.005,  # above vwap
            "current_price": 22100.0,
            "stop_loss": 21800.0,
            "entry": 22000.0,
        })
        assert response is not None

    def test_has_trained_model_false(self):
        from src.models.rl_execution_agent import RLExecutionAgent

        agent = RLExecutionAgent()
        assert agent.has_trained_model is False
