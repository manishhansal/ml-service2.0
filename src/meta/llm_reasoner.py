"""
LLMNewsReasoner — FinGPT/FinBERT based news sentiment signal for the MetaDecisionEngine.

Design (Req 10.5, Req 10.9):
- Processes SentinelPulse news context using FinBERT or FinGPT via LangChain
- 120ms timeout for LLM inference (settings.llm_inference_timeout_ms)
- On timeout: use cached signal if available (NEWS_CACHE_FALLBACK)
- If no cache: return neutral signal (NEWS_TIMEOUT reason code)
- Produces: direction ∈ {-1, 0, 1}, confidence ∈ [0,1], rationale (≤ 30 words)

Requirements: Req 10.5, Req 10.9
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from src.cache.redis_cache import LRUCache
from src.config import settings
from src.logging_config import get_logger
from src.schemas.meta import NewsSignal

logger = get_logger(__name__)

LLM_AVAILABLE = False
try:
    from transformers import pipeline as hf_pipeline
    LLM_AVAILABLE = True
except ImportError:
    pass


class LLMNewsReasoner:
    """
    FinBERT/FinGPT-based news sentiment reasoning for the MetaDecisionEngine.

    Usage::
        reasoner = LLMNewsReasoner()
        signal = await reasoner.reason(news_context, symbol="NIFTY")
        # signal.direction: -1 (bearish), 0 (neutral), +1 (bullish)
    """

    def __init__(
        self,
        model_name: str | None = None,
        timeout_ms: int | None = None,
        cache: LRUCache | None = None,
    ) -> None:
        self._model_name = model_name if model_name is not None else settings.llm_model_name
        self._timeout_ms = timeout_ms if timeout_ms is not None else settings.llm_inference_timeout_ms
        self._cache: LRUCache = cache if cache is not None else LRUCache(max_size=500, ttl_seconds=60)
        self._pipeline: Any = None
        self._initialized = False

        # Try to load the model lazily on first use

    def _initialize_pipeline(self) -> bool:
        """Initialize the FinBERT/FinGPT pipeline lazily."""
        if self._initialized:
            return self._pipeline is not None

        self._initialized = True

        if not LLM_AVAILABLE:
            logger.warning("llm_news_reasoner_transformers_not_available")
            return False

        try:
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self._model_name,
                return_all_scores=True,
                device=-1,  # CPU
            )
            logger.info("llm_news_reasoner_loaded", model=self._model_name)
            return True
        except Exception as exc:
            logger.warning("llm_news_reasoner_load_failed", model=self._model_name, error=str(exc))
            return False

    async def reason(
        self,
        news_context: dict[str, Any],
        symbol: str,
    ) -> tuple[NewsSignal, list[str]]:
        """
        Produce a NewsSignal from SentinelPulse news context.

        Args:
            news_context: SentinelPulse news context dict
            symbol:       Instrument symbol for cache keying

        Returns:
            (NewsSignal, reason_codes)
            reason_codes may include NEWS_CACHE_FALLBACK or NEWS_TIMEOUT
        """
        cache_key = f"llm:{symbol}"
        reason_codes: list[str] = []

        start = time.perf_counter()

        try:
            # Use asyncio.wait_for for timeout enforcement
            signal = await asyncio.wait_for(
                self._run_inference(news_context, symbol),
                timeout=self._timeout_ms / 1000.0,
            )

            # Cache the result for 60 seconds
            self._cache.set(cache_key, signal.model_dump())
            return signal, reason_codes

        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "llm_news_reasoner_timeout",
                symbol=symbol,
                elapsed_ms=round(elapsed_ms, 1),
                timeout_ms=self._timeout_ms,
            )

            # Try cache fallback
            cached = self._cache.get(cache_key)
            if cached:
                reason_codes.append("NEWS_CACHE_FALLBACK")
                try:
                    return NewsSignal.model_validate(cached), reason_codes
                except Exception:
                    pass

            # No cache → neutral signal per Req 10.9
            reason_codes.append("NEWS_TIMEOUT")
            return self._neutral_signal(), reason_codes

        except Exception as exc:
            logger.warning(
                "llm_news_reasoner_error",
                symbol=symbol,
                error=str(exc),
            )

            cached = self._cache.get(cache_key)
            if cached:
                reason_codes.append("NEWS_CACHE_FALLBACK")
                try:
                    return NewsSignal.model_validate(cached), reason_codes
                except Exception:
                    pass

            reason_codes.append("NEWS_TIMEOUT")
            return self._neutral_signal(), reason_codes

    async def _run_inference(
        self, news_context: dict[str, Any], symbol: str
    ) -> NewsSignal:
        """
        Run FinBERT/FinGPT inference on the news context.
        Falls back to a heuristic based on sentiment scores when LLM is unavailable.
        """
        # Yield once so that asyncio.wait_for can observe the timeout even on
        # the fast heuristic path (no external I/O).
        await asyncio.sleep(0)

        # Try LLM first
        if self._initialize_pipeline() and self._pipeline is not None:
            return await self._llm_inference(news_context)

        # Heuristic fallback using SentinelPulse sentiment scores
        return self._heuristic_signal(news_context)

    async def _llm_inference(self, news_context: dict[str, Any]) -> NewsSignal:
        """Run FinBERT sentiment classification on the news headline."""
        headline = news_context.get("latest_event", {}).get("headline", "")
        if not headline:
            # Use composite news description
            headline = f"News impact score: {news_context.get('news_impact_score', 0.5):.2f}"

        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._pipeline, headline)

        # FinBERT returns labels: positive/negative/neutral
        label_map = {"positive": 1, "negative": -1, "neutral": 0}
        best_label = max(results[0], key=lambda x: x["score"])

        direction = label_map.get(best_label["label"].lower(), 0)
        confidence = float(best_label["score"])
        rationale = f"FinBERT: {best_label['label']} ({confidence:.0%}) for {headline[:50]}"

        return NewsSignal(
            direction=direction,
            confidence=confidence,
            rationale=rationale[:200],
        )

    def _heuristic_signal(self, news_context: dict[str, Any]) -> NewsSignal:
        """Heuristic signal from SentinelPulse sentiment scores."""
        news_impact = float(news_context.get("news_impact_score", 0.0) or 0.0)
        impact_direction = str(news_context.get("impact_direction", "NEUTRAL") or "NEUTRAL").upper()
        sentiment_overall = float((news_context.get("sentiment") or {}).get("overall", 0.0) or 0.0)

        # Map direction
        if impact_direction == "BULLISH" or sentiment_overall > 0.15:
            direction = 1
        elif impact_direction == "BEARISH" or sentiment_overall < -0.15:
            direction = -1
        else:
            direction = 0

        confidence = min(1.0, abs(sentiment_overall) + news_impact * 0.3)
        rationale = (
            f"Heuristic: {impact_direction} sentiment={sentiment_overall:.2f} "
            f"impact={news_impact:.2f}"
        )

        return NewsSignal(
            direction=direction,
            confidence=max(0.1, confidence),
            rationale=rationale[:200],
        )

    def _neutral_signal(self) -> NewsSignal:
        """Return a neutral NewsSignal when inference fails (Req 10.9 NEWS_TIMEOUT case).

        Per Req 10.9: when no cached signal exists, direction=0 and confidence=0.0.
        """
        return NewsSignal(direction=0, confidence=0.0, rationale="")

    def reason_sync(
        self, news_context: dict[str, Any], symbol: str
    ) -> tuple[NewsSignal, list[str]]:
        """Synchronous wrapper for use in non-async contexts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't use asyncio.run() when loop is already running
                # Fall back to heuristic
                return self._heuristic_signal(news_context), []
            return loop.run_until_complete(self.reason(news_context, symbol))
        except Exception:
            return self._neutral_signal(), ["NEWS_TIMEOUT"]
