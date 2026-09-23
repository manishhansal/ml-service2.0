"""
HTTP client for SentinelPulse NLP/news intelligence service.

SentinelPulse runs on port 3001 and exposes:
- ``GET /api/v1/alphaforge/news-context/:instrument``  — news context for a symbol
- ``GET /api/v1/news/regime``                          — market regime classification
- ``GET /api/v1/alphaforge/context/market``            — macro market context
- ``GET /api/v1/ml/features/asset/:assetId``           — PIT-correct ML feature vector
- ``GET /api/v1/ml/training/samples``                  — paginated training samples
- ``GET /api/v1/ml/historical-reactions``              — historical event reactions

All public methods degrade gracefully — they return ``None`` (never raise) when
SentinelPulse is unreachable, so the calling FeaturePipeline can apply
degraded-mode substitution (Req 1.11).

Authentication:
    SentinelPulse uses Bearer token authentication:
    ``Authorization: Bearer <api_key>``

Caching:
    In-process LRU cache — max 500 entries, TTL = settings.news_context_cache_ttl (90 s).

Retry logic (exponential backoff):
    Up to 3 attempts, wait [0 s, 0.5 s, 1.0 s] before retries 2 and 3.
    Only retries on :exc:`httpx.TimeoutException` and :exc:`httpx.ConnectError`.
    Auth errors (401) are NOT retried.

Usage::

    client = SentinelPulseClient()
    await client.connect()

    ctx = await client.fetch_news_context("NIFTY")
    # Returns dict or None if unreachable / missing mandatory fields

    await client.disconnect()
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx

from src.cache.redis_cache import LRUCache
from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

# ── Field validation constants ────────────────────────────────────────────────

MANDATORY_NEWS_FIELDS = (
    "news_impact_score",
    "impact_direction",
    "impact_confidence",
)

MANDATORY_SENTIMENT_KEYS = (
    "overall",
    "market",
    "company",
    "macro",
    "risk",
)

# Retry delays (seconds) before each attempt (index 0 = first attempt, no wait)
_RETRY_DELAYS = (0.0, 0.5, 1.0)


class SentinelPulseClient:
    """
    HTTP client for SentinelPulse NLP/news intelligence service.

    All methods degrade gracefully — they return None (not raise) when
    SentinelPulse is unreachable, so the calling FeaturePipeline can
    apply degraded-mode substitution.

    Caching:
        In-process LRU cache: max 500 entries, TTL = settings.news_context_cache_ttl (90s)

    Retry logic (exponential backoff):
        Up to 3 attempts, wait [0.5s, 1.0s, 2.0s]
        Only retries on httpx.TimeoutException and httpx.ConnectError
        Auth errors (401) are NOT retried
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._base_url = (base_url or settings.sentinel_pulse_url).rstrip("/")
        self._api_key = api_key or settings.sentinel_pulse_api_key
        self._client: httpx.AsyncClient | None = None
        self._news_cache: LRUCache = LRUCache(
            max_size=500,
            ttl_seconds=settings.news_context_cache_ttl,
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the shared httpx.AsyncClient. Call once at startup."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=10.0,
        )
        logger.info("sentinel_pulse_client_connected", base_url=self._base_url)

    async def disconnect(self) -> None:
        """Close the shared httpx.AsyncClient. Call at shutdown."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("sentinel_pulse_client_disconnected")

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate_news_context(self, data: dict[str, Any]) -> bool:
        """
        Return True if all mandatory fields are present and non-null.
        Return False if any mandatory field is missing or null.

        Mandatory top-level fields:
            news_impact_score, impact_direction, impact_confidence

        Mandatory sentiment sub-fields:
            overall, market, company, macro, risk

        Per Req 1.10: missing mandatory fields are treated as unreachable.
        """
        # Check top-level mandatory fields
        for field in MANDATORY_NEWS_FIELDS:
            if data.get(field) is None:
                return False

        # Check sentiment sub-fields
        sentiment = data.get("sentiment", {})
        if not isinstance(sentiment, dict):
            return False
        for key in MANDATORY_SENTIMENT_KEYS:
            if sentiment.get(key) is None:
                return False

        return True

    # ── Public API methods ─────────────────────────────────────────────────────

    async def fetch_news_context(self, instrument: str) -> dict[str, Any] | None:
        """
        Fetch news context for an instrument.

        Endpoint: GET /api/v1/alphaforge/news-context/:instrument

        Per Req 1.9, extracts news_impact_score, impact_direction,
        impact_confidence, and all sentiment sub-fields to pass into the
        Feature_Pipeline as first-class features.

        Per Req 1.10, if any mandatory field is missing the response is treated
        equivalently to SentinelPulse being unreachable.

        Returns:
            dict with news context fields if successful and all mandatory fields
            are present.
            None if SentinelPulse is unreachable, returns 404, or mandatory
            fields are missing.

        Caches successful results in the in-process LRU cache (TTL = news_context_cache_ttl).
        """
        # Check LRU cache first
        cached = self._news_cache.get(instrument)
        if cached is not None:
            return cached  # type: ignore[return-value]

        # Fetch from API with retry
        data = await self._fetch_with_retry(
            f"/api/v1/alphaforge/news-context/{instrument}"
        )
        if data is None:
            return None

        # Validate mandatory fields (Req 1.10)
        if not self._validate_news_context(data):
            logger.warning(
                "sentinel_pulse_missing_mandatory_fields",
                instrument=instrument,
                fields_present=list(data.keys()),
            )
            return None

        # Cache and return
        self._news_cache.set(instrument, data)
        return data

    async def fetch_market_regime(self) -> dict[str, Any] | None:
        """
        Fetch the current market regime classification.

        Endpoint: GET /api/v1/news/regime

        Returns:
            dict with regime data, or None if unreachable.
        """
        return await self._fetch_with_retry("/api/v1/news/regime")

    async def fetch_market_context(self) -> dict[str, Any] | None:
        """
        Fetch macro market context.

        Endpoint: GET /api/v1/alphaforge/context/market

        Returns:
            dict with market context, or None if unreachable.
        """
        return await self._fetch_with_retry("/api/v1/alphaforge/context/market")

    async def fetch_pit_features(
        self,
        asset_id: str,
        feature_version: str = "latest",
        as_of: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Fetch PIT-correct ML feature vector for a single asset.

        Endpoint: GET /api/v1/ml/features/asset/:assetId

        Per Req 1.13 and 1.14, the ML_Service SHALL source news feature vectors
        exclusively from SentinelPulse PIT-correct endpoints. Vectors where
        look_ahead_validated is not True are rejected.

        Args:
            asset_id:        Instrument asset identifier.
            feature_version: Feature version to request (default: "latest").
            as_of:           ISO-8601 UTC timestamp for PIT-correct retrieval.

        Returns:
            dict with feature vector if look_ahead_validated is True.
            None if look_ahead_validated is False/absent, or on any failure.

        Training pipeline callers MUST check this flag (Req 1.14).
        """
        params: dict[str, str] = {"feature_version": feature_version}
        if as_of is not None:
            params["as_of"] = as_of

        data = await self._fetch_with_retry(
            f"/api/v1/ml/features/asset/{asset_id}",
            params=params,
        )
        if data is None:
            return None

        # Validate PIT correctness flag (Req 1.14)
        if not data.get("look_ahead_validated", False):
            logger.warning(
                "sentinel_pulse_look_ahead_not_validated",
                asset_id=asset_id,
                as_of=as_of,
            )
            return None

        return data

    async def fetch_training_samples(
        self,
        asset_id: str | None = None,
        feature_version: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        min_importance: float = 0.0,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Fetch paginated training samples.

        Endpoint: GET /api/v1/ml/training/samples

        Per Req 1.13, the ML_Service SHALL source news feature vectors
        exclusively from SentinelPulse PIT-correct endpoints. Per Req 1.14
        and 1.15, each sample must have look_ahead_validated=True; invalid
        samples are filtered out (not returned).

        Args:
            asset_id:        Optional instrument filter.
            feature_version: Optional feature version filter.
            start_date:      ISO-8601 start date for sample range.
            end_date:        ISO-8601 end date for sample range.
            min_importance:  Minimum feature importance threshold.
            limit:           Maximum number of samples to return (default 100).
            cursor:          Pagination cursor from a previous response.

        Returns:
            dict with ``samples`` list (only look_ahead_validated=True entries)
            and pagination metadata, or None on failure.
        """
        params: dict[str, str] = {
            "limit": str(limit),
            "min_importance": str(min_importance),
        }
        if asset_id is not None:
            params["asset_id"] = asset_id
        if feature_version is not None:
            params["feature_version"] = feature_version
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        if cursor is not None:
            params["cursor"] = cursor

        data = await self._fetch_with_retry("/api/v1/ml/training/samples", params=params)
        if data is None:
            return None

        # Filter out samples that lack look_ahead_validated=True (Req 1.14 / 1.15)
        raw_samples: list[Any] = data.get("samples", [])
        valid_samples: list[Any] = []
        for sample in raw_samples:
            if not isinstance(sample, dict):
                continue
            if sample.get("look_ahead_validated", False):
                valid_samples.append(sample)
            else:
                logger.warning(
                    "sentinel_pulse_training_sample_not_pit_validated",
                    asset_id=sample.get("asset_id"),
                    sample_timestamp=sample.get("timestamp"),
                )

        data["samples"] = valid_samples
        return data

    async def fetch_historical_reactions(
        self,
        asset_id: str | None = None,
        event_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any] | None:
        """
        Fetch historical event reactions.

        Endpoint: GET /api/v1/ml/historical-reactions

        Per Req 1.13, the ML_Service SHALL source news feature vectors
        exclusively from SentinelPulse PIT-correct endpoints.

        Args:
            asset_id:   Optional instrument filter.
            event_type: Optional event type filter.
            start_date: ISO-8601 start date.
            end_date:   ISO-8601 end date.
            limit:      Maximum number of reactions to return (default 100).

        Returns:
            dict with historical reactions, or None on failure.
        """
        params: dict[str, str] = {"limit": str(limit)}
        if asset_id is not None:
            params["asset_id"] = asset_id
        if event_type is not None:
            params["event_type"] = event_type
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date

        return await self._fetch_with_retry("/api/v1/ml/historical-reactions", params=params)

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _fetch_with_retry(
        self,
        path: str,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """
        Perform a GET request with exponential backoff retry.

        Retry policy:
            3 total attempts.
            Delays before each attempt: [0 s, 0.5 s, 1.0 s].
            Retries only on httpx.TimeoutException and httpx.ConnectError.
            401 auth errors abort immediately without retry (Req 1.4).

        Returns:
            response["data"] dict on HTTP 200 with success=true.
            None on 404, 401, unrecoverable error, or all retries exhausted.

        Never raises — always returns None on failure (degrade gracefully).
        """
        if self._client is None:
            logger.warning("sentinel_pulse_client_not_connected", path=path)
            return None

        max_attempts = len(_RETRY_DELAYS)

        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            if delay > 0.0:
                await asyncio.sleep(delay)

            try:
                response = await self._client.get(path, params=params)

                if response.status_code == 401:
                    # Auth errors are never retried (Req 1.4 pattern)
                    logger.warning(
                        "sentinel_pulse_auth_error",
                        path=path,
                        status=response.status_code,
                    )
                    return None

                if response.status_code == 404:
                    logger.warning(
                        "sentinel_pulse_not_found",
                        path=path,
                        status=response.status_code,
                    )
                    return None

                if response.status_code != 200:
                    logger.warning(
                        "sentinel_pulse_unexpected_status",
                        path=path,
                        attempt=attempt,
                        status=response.status_code,
                    )
                    if attempt < max_attempts:
                        continue
                    return None

                body: dict[str, Any] = response.json()
                if not body.get("success", False):
                    logger.warning(
                        "sentinel_pulse_success_false",
                        path=path,
                        error=body.get("error"),
                    )
                    return None

                return body.get("data")

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                logger.warning(
                    "sentinel_pulse_request_error",
                    path=path,
                    attempt=attempt,
                    error=str(exc),
                    exc_type=type(exc).__name__,
                )
                if attempt < max_attempts:
                    continue
                return None

        return None
