"""
pipeline.py

FeaturePipeline — PIT-correct feature vector assembly for ml-service2.0.

Fetches market data from data-service2.0 and news context from SentinelPulse,
validates quality gates, checks PIT boundaries, computes Qlib Alpha158 factors,
and assembles a ``FeatureVector`` with the appropriate ``PredictionProvenance``.

Quality gate priority (highest to lowest):
  1. data-service2.0 unreachable / auth error → UNAVAILABLE
  2. signalEngineAllowed=false               → UNAVAILABLE
  3. DataConfidenceScore < threshold          → INSUFFICIENT_EVIDENCE
  4. SentinelPulse unavailable               → degrade news to neutral, continue
  5. All gates pass                          → TRAINED_MODEL (provenance set here;
                                               downstream models may downgrade to HEURISTIC)

PIT validation rule:
  Any source datum whose ``dataAsOf`` timestamp is >= the requested ``timestamp``
  (the PIT boundary) constitutes a violation.  Violations are counted and recorded
  in ``FeatureQualityReport.pit_violations_count``; the offending data fields are
  still used (they represent the most-recent available data before the call was
  made), but the violation is flagged so downstream consumers can gate on it.

Degraded-mode news substitution (Req 1.11):
  When SentinelPulse returns None or raises, all numeric news fields → 0.0,
  directional field → ImpactDirection.NEUTRAL, ``sentinel_available`` → False.
  This must NOT block inference from proceeding.

Requirements: Req 1.11, Req 2.1, Req 2.6, Req 2.7, Req 2.8, Req 2.12
"""
from __future__ import annotations

import asyncio
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd

from src.cache.redis_cache import RedisCache
from src.clients.data_service import (
    DataServiceAuthError,
    DataServiceClient,
    DataServiceUnavailableError,
    LowDataConfidenceError,
    SignalEngineNotAllowedError,
)
from src.clients.sentinel_pulse import SentinelPulseClient
from src.config import settings
from src.features.qlib_engine import QlibFeatureEngine
from src.logging_config import get_logger
from src.schemas.base import ImpactDirection, PredictionProvenance
from src.schemas.features import FeatureQualityReport, FeatureVector

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Sentinel-degraded-mode neutral values (Req 1.11)
# ---------------------------------------------------------------------------

_SENTINEL_NEUTRAL: dict[str, Any] = {
    "news_impact_score": 0.0,
    "impact_direction": ImpactDirection.NEUTRAL,
    "impact_confidence": 0.0,
    "sentiment_overall": 0.0,
    "sentiment_market": 0.0,
    "sentiment_company": 0.0,
    "sentiment_macro": 0.0,
    "sentiment_risk": 0.0,
    "market_regime_nlp": "NEUTRAL",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(v: Any) -> float | None:
    """Convert *v* to float; return None when the value is missing or non-finite."""
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _parse_iso(ts_str: str | None) -> datetime | None:
    """Parse an ISO-8601 string into a UTC-aware datetime, or return None."""
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        # Python 3.11 fromisoformat handles 'Z' suffix; older versions need the replace
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------


class FeaturePipeline:
    """
    PIT-correct feature vector assembly from data-service2.0 and SentinelPulse.

    Inject mock clients in tests::

        pipeline = FeaturePipeline(data_client=mock_data, news_client=mock_news)
        vector, report = await pipeline.build_vector("NIFTY", ts, mode="inference")
    """

    def __init__(
        self,
        data_client: DataServiceClient | None = None,
        news_client: SentinelPulseClient | None = None,
        qlib_engine: QlibFeatureEngine | None = None,
        cache: RedisCache | None = None,
    ) -> None:
        self._data = data_client
        self._news = news_client
        self._qlib = qlib_engine or QlibFeatureEngine()
        self._cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_vector(
        self,
        symbol: str,
        timestamp: datetime,
        mode: Literal["inference", "backtest"] = "inference",
        pit_date: str | None = None,
    ) -> tuple[FeatureVector, FeatureQualityReport]:
        """
        Build a PIT-correct feature vector for *symbol* at *timestamp*.

        Args:
            symbol:    NSE/NFO instrument symbol (e.g. ``"NIFTY"``).
            timestamp: UTC PIT boundary — all source data must be strictly
                       before this point in time.
            mode:      ``"inference"`` (with Redis cache) or ``"backtest"``
                       (passes ``pit_date`` to data-service2.0, skips cache).
            pit_date:  Optional ``YYYY-MM-DD`` string for backtest PIT mode.

        Returns:
            A 2-tuple ``(FeatureVector, FeatureQualityReport)``.
        """
        t_start = time.perf_counter()
        batch_id = str(uuid.uuid4())

        # Normalise timestamp to UTC-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # ── Step 1: Redis cache (inference only) ──────────────────────────────
        if mode == "inference" and self._cache is not None:
            bucket = self._bucket(timestamp, settings.feature_cache_ttl)
            cached = await self._cache.get_feature(symbol, bucket)
            if cached is not None:
                try:
                    vector = FeatureVector.model_validate(cached)
                    report = FeatureQualityReport(
                        batch_id=batch_id,
                        timestamp=datetime.now(tz=timezone.utc),
                        pit_violations_count=0,
                        sentinel_pulse_available=vector.sentinel_available,
                        data_service_available=True,
                        processing_time_ms=0.0,
                    )
                    return vector, report
                except Exception:
                    pass  # stale / invalid cache — recompute below

        # ── Step 2: Fetch market data from data-service2.0 ────────────────────
        (
            data_service_available,
            provenance,
            confidence_score,
            signal_engine_allowed,
            market_data,
            pit_violations,
            data_families_unavailable,
        ) = await self._fetch_market_data(symbol, timestamp, pit_date)

        # ── Step 3: Fetch news context from SentinelPulse ────────────────────
        sentinel_available, news_fields, news_families_unavailable = (
            await self._fetch_news_context(symbol)
        )

        unavailable_families = data_families_unavailable + news_families_unavailable

        # ── Step 4: Compute Qlib Alpha158 (only when market data is usable) ──
        alpha158_factors: dict[str, float] = {}
        if data_service_available and provenance not in (
            PredictionProvenance.UNAVAILABLE,
            PredictionProvenance.INSUFFICIENT_EVIDENCE,
        ):
            alpha158_factors = await self._compute_alpha158(symbol, pit_date)

        # ── Step 5: Assemble FeatureVector ────────────────────────────────────
        vector = FeatureVector(
            symbol=symbol,
            timestamp=timestamp,
            pit_validated=(pit_violations == 0),
            data_confidence_score=confidence_score,
            signal_engine_allowed=(signal_engine_allowed and data_service_available),
            # OHLCV fields from market data (may be None if unavailable)
            close=_safe_float(market_data.get("close") or market_data.get("ltp")),
            open=_safe_float(market_data.get("open")),
            high=_safe_float(market_data.get("high")),
            low=_safe_float(market_data.get("low")),
            volume=_safe_float(market_data.get("volume")),
            vwap=_safe_float(market_data.get("vwap")),
            india_vix=_safe_float(market_data.get("india_vix")),
            put_call_ratio=_safe_float(
                market_data.get("pcr") or market_data.get("put_call_ratio")
            ),
            # SentinelPulse news features
            news_impact_score=float(news_fields["news_impact_score"]),
            impact_direction=news_fields["impact_direction"],
            impact_confidence=float(news_fields["impact_confidence"]),
            sentiment_overall=float(news_fields["sentiment_overall"]),
            sentiment_market=float(news_fields["sentiment_market"]),
            sentiment_company=float(news_fields["sentiment_company"]),
            sentiment_macro=float(news_fields["sentiment_macro"]),
            sentiment_risk=float(news_fields["sentiment_risk"]),
            market_regime_nlp=str(news_fields["market_regime_nlp"]),
            sentinel_available=sentinel_available,
            # Qlib factors
            alpha158_factors=alpha158_factors,
            # Provenance (quality-gate result)
            provenance=provenance,
        )

        # ── Step 6: Cache (inference only) ────────────────────────────────────
        if mode == "inference" and self._cache is not None and data_service_available:
            try:
                bucket = self._bucket(timestamp, settings.feature_cache_ttl)
                await self._cache.set_feature(symbol, bucket, vector.model_dump())
            except Exception as exc:
                logger.warning("feature_cache_write_error", symbol=symbol, error=str(exc))

        # ── Step 7: Assemble FeatureQualityReport ─────────────────────────────
        processing_ms = (time.perf_counter() - t_start) * 1_000
        report = FeatureQualityReport(
            batch_id=batch_id,
            timestamp=datetime.now(tz=timezone.utc),
            pit_violations_count=pit_violations,
            unavailable_families=unavailable_families,
            sentinel_pulse_available=sentinel_available,
            data_service_available=data_service_available,
            processing_time_ms=processing_ms,
        )

        return vector, report

    async def build_batch(
        self,
        symbols: list[str],
        timestamp: datetime,
        mode: Literal["inference", "backtest"] = "inference",
    ) -> tuple[list[FeatureVector], FeatureQualityReport]:
        """
        Build feature vectors for a list of symbols concurrently.

        Uses ``asyncio.gather`` so up to 500 symbols complete within the
        normal SLA (≤ 60 s under normal network conditions).
        """
        results = await asyncio.gather(
            *[self.build_vector(s, timestamp, mode) for s in symbols],
            return_exceptions=False,
        )

        vectors = [r[0] for r in results]

        # Aggregate quality reports
        total_pit = sum(r[1].pit_violations_count for r in results)
        all_unavailable: list[str] = []
        for r in results:
            all_unavailable.extend(r[1].unavailable_families)

        agg_report = FeatureQualityReport(
            batch_id=str(uuid.uuid4()),
            timestamp=datetime.now(tz=timezone.utc),
            pit_violations_count=total_pit,
            unavailable_families=list(set(all_unavailable)),
            data_service_available=all(r[1].data_service_available for r in results),
            sentinel_pulse_available=all(r[1].sentinel_pulse_available for r in results),
        )
        return vectors, agg_report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bucket(self, ts: datetime, ttl_seconds: int) -> str:
        """Return a cache-bucket key by flooring *ts* to *ttl_seconds* granularity."""
        epoch = int(ts.timestamp())
        return str(epoch - (epoch % ttl_seconds))

    async def _fetch_market_data(
        self,
        symbol: str,
        timestamp: datetime,
        pit_date: str | None,
    ) -> tuple[bool, PredictionProvenance, int, bool, dict[str, Any], int, list[str]]:
        """
        Fetch live quote from data-service2.0 and apply quality gates.

        Returns:
            (data_service_available, provenance, confidence_score,
             signal_engine_allowed, market_data_dict,
             pit_violations_count, unavailable_families)
        """
        if self._data is None:
            logger.warning("feature_pipeline_no_data_client", symbol=symbol)
            return False, PredictionProvenance.UNAVAILABLE, 0, False, {}, 0, ["market_data"]

        try:
            raw_quote = await self._data.get_live_quote(symbol)
        except (DataServiceUnavailableError, DataServiceAuthError) as exc:
            logger.warning(
                "feature_pipeline_data_service_unavailable",
                symbol=symbol,
                error=str(exc),
            )
            return False, PredictionProvenance.UNAVAILABLE, 0, False, {}, 0, ["market_data"]
        except SignalEngineNotAllowedError:
            # Quality gate: signalEngineAllowed=false → UNAVAILABLE (Req 1.5)
            logger.warning("feature_pipeline_signal_engine_not_allowed", symbol=symbol)
            return True, PredictionProvenance.UNAVAILABLE, 0, False, {}, 0, []
        except LowDataConfidenceError as exc:
            # Quality gate: DataConfidenceScore < threshold → INSUFFICIENT_EVIDENCE (Req 1.7)
            logger.warning(
                "feature_pipeline_low_confidence",
                symbol=symbol,
                error=str(exc),
            )
            # Extract confidence score from the exception message if possible
            return True, PredictionProvenance.INSUFFICIENT_EVIDENCE, 0, True, {}, 0, []

        # ── Parse quality metadata ─────────────────────────────────────────────
        metadata: dict[str, Any] = raw_quote.get("metadata") or {}
        quality: dict[str, Any] = metadata.get("quality") or {}

        confidence_score = int(quality.get("score", quality.get("DataConfidenceScore", 0)) or 0)

        signal_engine_val = quality.get("signalEngineAllowed")
        if signal_engine_val is None:
            signal_engine_val = raw_quote.get("signalEngineAllowed", True)
        signal_engine_allowed = bool(signal_engine_val)

        # ── PIT validation ─────────────────────────────────────────────────────
        pit_violations = 0
        data_as_of_str: str | None = (
            metadata.get("dataAsOf")
            or metadata.get("data_timestamp_ms")
        )
        if isinstance(data_as_of_str, int):
            # Millisecond epoch — convert to ISO string
            source_ts = datetime.fromtimestamp(data_as_of_str / 1000, tz=timezone.utc)
            if source_ts >= timestamp:
                pit_violations += 1
                logger.warning(
                    "pit_violation_source_data_after_boundary",
                    symbol=symbol,
                    source_ts=source_ts.isoformat(),
                    pit_boundary=timestamp.isoformat(),
                )
        else:
            source_ts = _parse_iso(data_as_of_str)
            if source_ts is not None and source_ts >= timestamp:
                pit_violations += 1
                logger.warning(
                    "pit_violation_source_data_after_boundary",
                    symbol=symbol,
                    source_ts=source_ts.isoformat(),
                    pit_boundary=timestamp.isoformat(),
                )

        # ── Apply quality gates (gates are now re-checked here because the
        #    DataServiceClient may or may not raise — depends on mock vs real) ──
        if not signal_engine_allowed:
            provenance = PredictionProvenance.UNAVAILABLE
        elif confidence_score < settings.min_confidence_score:
            provenance = PredictionProvenance.INSUFFICIENT_EVIDENCE
        else:
            # Healthy — eligible for model inference
            provenance = PredictionProvenance.TRAINED_MODEL

        market_data: dict[str, Any] = raw_quote.get("data") or raw_quote

        return (
            True,
            provenance,
            confidence_score,
            signal_engine_allowed,
            market_data,
            pit_violations,
            [],
        )

    async def _fetch_news_context(
        self,
        symbol: str,
    ) -> tuple[bool, dict[str, Any], list[str]]:
        """
        Fetch news context from SentinelPulse with degraded-mode substitution.

        Returns:
            (sentinel_available, news_fields_dict, unavailable_families)
        """
        if self._news is None:
            return False, _SENTINEL_NEUTRAL.copy(), ["news_context"]

        try:
            news_ctx = await self._news.fetch_news_context(symbol)
        except Exception as exc:
            logger.warning(
                "feature_pipeline_sentinel_unavailable",
                symbol=symbol,
                error=str(exc),
            )
            return False, _SENTINEL_NEUTRAL.copy(), ["news_context"]

        if news_ctx is None:
            # SentinelPulse returned None (unreachable / missing mandatory fields)
            return False, _SENTINEL_NEUTRAL.copy(), ["news_context"]

        # Parse real values from the response
        sentiment: dict[str, Any] = news_ctx.get("sentiment") or {}

        # Parse impact_direction — handle casing differences
        raw_direction = news_ctx.get("impact_direction", "NEUTRAL")
        try:
            direction = ImpactDirection(str(raw_direction).upper())
        except ValueError:
            direction = ImpactDirection.NEUTRAL

        news_fields: dict[str, Any] = {
            "news_impact_score": float(news_ctx.get("news_impact_score") or 0.0),
            "impact_direction": direction,
            "impact_confidence": float(news_ctx.get("impact_confidence") or 0.0),
            "sentiment_overall": float(sentiment.get("overall") or 0.0),
            "sentiment_market": float(sentiment.get("market") or 0.0),
            "sentiment_company": float(sentiment.get("company") or 0.0),
            "sentiment_macro": float(sentiment.get("macro") or 0.0),
            "sentiment_risk": float(sentiment.get("risk") or 0.0),
            "market_regime_nlp": str(
                news_ctx.get("market_regime") or news_ctx.get("market_regime_nlp") or "NEUTRAL"
            ),
        }

        return True, news_fields, []

    async def _compute_alpha158(
        self,
        symbol: str,
        pit_date: str | None,
    ) -> dict[str, float]:
        """
        Fetch OHLCV history and compute Qlib Alpha158 factors.

        Returns an empty dict silently if history is insufficient or
        computation fails — downstream consumers treat an empty dict as
        missing factors and proceed normally.
        """
        if self._data is None:
            return {}

        try:
            hist = await self._data.get_historical_ohlcv(
                symbol,
                interval="1d",
                pit_date=pit_date,
            )
        except Exception as exc:
            logger.warning(
                "alpha158_ohlcv_fetch_failed",
                symbol=symbol,
                error=str(exc),
            )
            return {}

        if not hist or len(hist) < 200:
            return {}

        try:
            ohlcv_df = pd.DataFrame(hist)
            # Normalise column names to lower-case
            ohlcv_df.columns = [c.lower() for c in ohlcv_df.columns]
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(set(ohlcv_df.columns)):
                return {}
            return self._qlib.compute_alpha158(ohlcv_df, symbol)
        except Exception as exc:
            logger.warning(
                "alpha158_computation_failed",
                symbol=symbol,
                error=str(exc),
            )
            return {}
