"""
Mock-based tests for src/meta/llm_reasoner.py.
"""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BULLISH_CTX = {
    "news_impact_score": 0.8,
    "impact_direction": "BULLISH",
    "impact_confidence": 0.9,
    "sentiment": {
        "overall": 0.5,
        "market": 0.4,
        "company": 0.6,
        "macro": 0.3,
        "risk": -0.1,
    },
}

BEARISH_CTX = {
    "news_impact_score": 0.7,
    "impact_direction": "BEARISH",
    "impact_confidence": 0.8,
    "sentiment": {
        "overall": -0.4,
        "market": -0.3,
        "company": -0.5,
        "macro": -0.2,
        "risk": 0.2,
    },
}

NEUTRAL_CTX = {
    "news_impact_score": 0.1,
    "impact_direction": "NEUTRAL",
    "impact_confidence": 0.3,
    "sentiment": {
        "overall": 0.01,
        "market": 0.02,
        "company": -0.01,
        "macro": 0.0,
        "risk": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Heuristic signal tests (LLM unavailable path)
# ---------------------------------------------------------------------------


class TestHeuristicSignal:
    def test_bullish_direction(self):
        from src.meta.llm_reasoner import LLMNewsReasoner

        with patch("src.meta.llm_reasoner.LLM_AVAILABLE", False):
            reasoner = LLMNewsReasoner()
            signal = reasoner._heuristic_signal(BULLISH_CTX)

        assert signal.direction == 1
        assert signal.confidence >= 0.0

    def test_bearish_direction(self):
        from src.meta.llm_reasoner import LLMNewsReasoner

        with patch("src.meta.llm_reasoner.LLM_AVAILABLE", False):
            reasoner = LLMNewsReasoner()
            signal = reasoner._heuristic_signal(BEARISH_CTX)

        assert signal.direction == -1

    def test_neutral_direction(self):
        from src.meta.llm_reasoner import LLMNewsReasoner

        with patch("src.meta.llm_reasoner.LLM_AVAILABLE", False):
            reasoner = LLMNewsReasoner()
            signal = reasoner._heuristic_signal(NEUTRAL_CTX)

        assert signal.direction == 0

    def test_heuristic_confidence_bounded(self):
        from src.meta.llm_reasoner import LLMNewsReasoner

        reasoner = LLMNewsReasoner()
        signal = reasoner._heuristic_signal(BULLISH_CTX)
        assert 0.0 <= signal.confidence <= 1.0


class TestNeutralSignal:
    def test_neutral_signal_direction_zero(self):
        from src.meta.llm_reasoner import LLMNewsReasoner

        reasoner = LLMNewsReasoner()
        sig = reasoner._neutral_signal()
        assert sig.direction == 0
        assert sig.confidence == 0.0

    def test_neutral_signal_rationale_empty_or_str(self):
        from src.meta.llm_reasoner import LLMNewsReasoner

        reasoner = LLMNewsReasoner()
        sig = reasoner._neutral_signal()
        assert isinstance(sig.rationale, str)


# ---------------------------------------------------------------------------
# reason() — async tests via asyncio.run()
# ---------------------------------------------------------------------------


class TestReasonAsync:
    def test_reason_heuristic_path_returns_news_signal(self):
        """LLM unavailable → heuristic path, direction=1 for BULLISH."""
        from src.meta.llm_reasoner import LLMNewsReasoner

        async def _run():
            with patch("src.meta.llm_reasoner.LLM_AVAILABLE", False):
                reasoner = LLMNewsReasoner(timeout_ms=5000)
                signal, codes = await reasoner.reason(BULLISH_CTX, "NIFTY")
            assert signal.direction == 1
            assert codes == []

        asyncio.run(_run())

    def test_reason_bearish_heuristic(self):
        from src.meta.llm_reasoner import LLMNewsReasoner

        async def _run():
            with patch("src.meta.llm_reasoner.LLM_AVAILABLE", False):
                reasoner = LLMNewsReasoner(timeout_ms=5000)
                signal, codes = await reasoner.reason(BEARISH_CTX, "NIFTY")
            assert signal.direction == -1

        asyncio.run(_run())

    def test_reason_timeout_path_adds_news_timeout(self):
        """_run_inference sleeps 10s, timeout is 50ms → NEWS_TIMEOUT."""
        from src.meta.llm_reasoner import LLMNewsReasoner

        async def _run():
            reasoner = LLMNewsReasoner(timeout_ms=50)

            async def slow_inference(ctx, sym):
                await asyncio.sleep(10)
                return reasoner._neutral_signal()

            with patch.object(reasoner, "_run_inference", side_effect=slow_inference):
                signal, codes = await reasoner.reason(BULLISH_CTX, "NIFTY")

            assert "NEWS_TIMEOUT" in codes
            assert signal.direction == 0
            assert signal.confidence == 0.0

        asyncio.run(_run())

    def test_reason_cache_fallback_on_timeout(self):
        """Pre-populate cache, then force timeout → NEWS_CACHE_FALLBACK."""
        from src.meta.llm_reasoner import LLMNewsReasoner
        from src.meta.llm_reasoner import LRUCache

        async def _run():
            cache = LRUCache(max_size=100, ttl_seconds=60)
            reasoner = LLMNewsReasoner(timeout_ms=50, cache=cache)

            # Pre-populate cache
            from src.schemas.meta import NewsSignal

            cached_signal = NewsSignal(direction=1, confidence=0.7, rationale="cached")
            cache.set("llm:NIFTY", cached_signal.model_dump())

            async def slow_inference(ctx, sym):
                await asyncio.sleep(10)
                raise asyncio.TimeoutError()

            with patch.object(reasoner, "_run_inference", side_effect=slow_inference):
                signal, codes = await reasoner.reason(BULLISH_CTX, "NIFTY")

            assert "NEWS_CACHE_FALLBACK" in codes
            assert signal.direction == 1

        asyncio.run(_run())

    def test_reason_exception_path_returns_news_timeout(self):
        """_run_inference raises RuntimeError → falls back to NEWS_TIMEOUT."""
        from src.meta.llm_reasoner import LLMNewsReasoner

        async def _run():
            reasoner = LLMNewsReasoner(timeout_ms=5000)

            async def failing_inference(ctx, sym):
                raise RuntimeError("inference failure")

            with patch.object(reasoner, "_run_inference", side_effect=failing_inference):
                signal, codes = await reasoner.reason(BULLISH_CTX, "NIFTY")

            assert "NEWS_TIMEOUT" in codes

        asyncio.run(_run())

    def test_reason_exception_with_cache_fallback(self):
        """Exception + cache present → NEWS_CACHE_FALLBACK."""
        from src.meta.llm_reasoner import LLMNewsReasoner, LRUCache
        from src.schemas.meta import NewsSignal

        async def _run():
            cache = LRUCache(max_size=100, ttl_seconds=60)
            reasoner = LLMNewsReasoner(timeout_ms=5000, cache=cache)

            cached_signal = NewsSignal(direction=-1, confidence=0.6, rationale="cached")
            cache.set("llm:RELIANCE", cached_signal.model_dump())

            async def failing_inference(ctx, sym):
                raise RuntimeError("boom")

            with patch.object(reasoner, "_run_inference", side_effect=failing_inference):
                signal, codes = await reasoner.reason(BULLISH_CTX, "RELIANCE")

            assert "NEWS_CACHE_FALLBACK" in codes
            assert signal.direction == -1

        asyncio.run(_run())

    def test_reason_result_cached_on_success(self):
        """Successful inference result should be written to cache."""
        from src.meta.llm_reasoner import LLMNewsReasoner

        async def _run():
            with patch("src.meta.llm_reasoner.LLM_AVAILABLE", False):
                reasoner = LLMNewsReasoner(timeout_ms=5000)
                signal, _ = await reasoner.reason(BULLISH_CTX, "NIFTY")

            # Cache should now have the signal
            cached = reasoner._cache.get("llm:NIFTY")
            assert cached is not None

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# LLM pipeline path (mocked transformers)
# ---------------------------------------------------------------------------


class TestLLMPipelinePath:
    def test_llm_positive_label_returns_direction_1(self):
        """Mock FinBERT pipeline returning positive label → direction=1."""
        from src.meta.llm_reasoner import LLMNewsReasoner

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [[{"label": "positive", "score": 0.9}]]

        async def _run():
            with patch("src.meta.llm_reasoner.LLM_AVAILABLE", True):
                reasoner = LLMNewsReasoner(timeout_ms=5000)
                reasoner._initialized = True
                reasoner._pipeline = mock_pipeline
                signal = await reasoner._llm_inference(
                    {"latest_event": {"headline": "Market rallies strongly"}}
                )
            assert signal.direction == 1
            assert signal.confidence == 0.9

        asyncio.run(_run())

    def test_llm_negative_label_returns_direction_minus_1(self):
        from src.meta.llm_reasoner import LLMNewsReasoner

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [[{"label": "negative", "score": 0.85}]]

        async def _run():
            with patch("src.meta.llm_reasoner.LLM_AVAILABLE", True):
                reasoner = LLMNewsReasoner(timeout_ms=5000)
                reasoner._initialized = True
                reasoner._pipeline = mock_pipeline
                signal = await reasoner._llm_inference(
                    {"latest_event": {"headline": "Market crashes badly"}}
                )
            assert signal.direction == -1

        asyncio.run(_run())

    def test_llm_neutral_label_returns_direction_0(self):
        from src.meta.llm_reasoner import LLMNewsReasoner

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [[{"label": "neutral", "score": 0.75}]]

        async def _run():
            with patch("src.meta.llm_reasoner.LLM_AVAILABLE", True):
                reasoner = LLMNewsReasoner(timeout_ms=5000)
                reasoner._initialized = True
                reasoner._pipeline = mock_pipeline
                signal = await reasoner._llm_inference(
                    {"latest_event": {"headline": "Market steady"}}
                )
            assert signal.direction == 0

        asyncio.run(_run())

    def test_llm_no_headline_uses_impact_score(self):
        """When no headline in context, uses news_impact_score fallback text."""
        from src.meta.llm_reasoner import LLMNewsReasoner

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [[{"label": "positive", "score": 0.6}]]

        async def _run():
            with patch("src.meta.llm_reasoner.LLM_AVAILABLE", True):
                reasoner = LLMNewsReasoner(timeout_ms=5000)
                reasoner._initialized = True
                reasoner._pipeline = mock_pipeline
                # No latest_event headline
                signal = await reasoner._llm_inference({"news_impact_score": 0.5})
            assert signal.direction == 1

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# _initialize_pipeline()
# ---------------------------------------------------------------------------


class TestInitializePipeline:
    def test_returns_false_when_llm_unavailable(self):
        from src.meta.llm_reasoner import LLMNewsReasoner

        with patch("src.meta.llm_reasoner.LLM_AVAILABLE", False):
            reasoner = LLMNewsReasoner()
            result = reasoner._initialize_pipeline()

        assert result is False

    def test_initializes_only_once(self):
        from src.meta.llm_reasoner import LLMNewsReasoner

        with patch("src.meta.llm_reasoner.LLM_AVAILABLE", False):
            reasoner = LLMNewsReasoner()
            r1 = reasoner._initialize_pipeline()
            r2 = reasoner._initialize_pipeline()

        assert r1 == r2
        assert reasoner._initialized is True


# ---------------------------------------------------------------------------
# reason_sync()
# ---------------------------------------------------------------------------


class TestReasonSync:
    def test_reason_sync_when_loop_not_running(self):
        from src.meta.llm_reasoner import LLMNewsReasoner

        with patch("src.meta.llm_reasoner.LLM_AVAILABLE", False):
            reasoner = LLMNewsReasoner(timeout_ms=5000)
            signal, codes = reasoner.reason_sync(BULLISH_CTX, "NIFTY")

        # Should return a signal without raising
        assert signal is not None
        assert signal.direction in (-1, 0, 1)
