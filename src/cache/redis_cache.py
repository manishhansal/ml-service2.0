"""
Redis async cache layer for ml-service2.0.

Provides:
- ``RedisCache``: async Redis client with typed helpers for every key pattern
  defined in the design (feature vectors, news context, signals, LLM results,
  model status, drift reports).
- ``LRUCache``: thread-safe in-process LRU cache used for SHAP explainer
  instances (no TTL) and news context (TTL = news_context_cache_ttl seconds).

Usage::

    # at FastAPI lifespan startup
    cache = RedisCache()
    await cache.connect()

    # within a request handler
    hit = await cache.get_feature("NIFTY", "20250101093000")
    if hit is None:
        ...
        await cache.set_feature("NIFTY", "20250101093000", vector_dict)

    # in-process LRU (no await)
    lru = LRUCache(max_size=500, ttl_seconds=90)
    lru.set("NIFTY", news_context_dict)
    value = lru.get("NIFTY")   # returns None if expired or absent
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from threading import Lock
from typing import Any

import redis.asyncio as aioredis

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

# ── Key pattern catalogue (documentation only) ────────────────────────────────

KEY_PATTERNS: dict[str, str] = {
    "feature": "feature:{symbol}:{bucket}",    # TTL = settings.feature_cache_ttl  (default 60 s)
    "news": "news:{symbol}",                   # TTL = settings.news_context_cache_ttl (default 90 s)
    "signal": "signal:{symbol}:latest",        # TTL = 300 s
    "llm_news": "llm_news:{symbol}",           # TTL = 60 s
    "model_status": "model:status",            # no TTL — persists until explicit delete
    "drift_latest": "drift:latest",            # TTL = 86 400 s
}


# ── RedisCache ────────────────────────────────────────────────────────────────


class RedisCache:
    """Async Redis cache with typed helpers for every key pattern.

    Pool-based — one connection pool per process.  All public methods degrade
    gracefully on Redis errors: they log at WARNING level and return ``None``
    (reads) or silently skip (writes) rather than propagating exceptions into
    the prediction path.

    Lifecycle::

        cache = RedisCache()
        await cache.connect()   # call once at FastAPI lifespan startup
        ...
        await cache.disconnect() # call at FastAPI lifespan shutdown
    """

    def __init__(self, url: str | None = None) -> None:
        """
        Args:
            url: Redis connection URL.  Defaults to ``settings.redis_url``.
        """
        self._url: str = url or settings.redis_url
        self._client: aioredis.Redis | None = None  # type: ignore[type-arg]

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the async connection pool.

        Safe to call multiple times — subsequent calls are no-ops if the pool
        is already open.
        """
        if self._client is not None:
            return
        self._client = aioredis.from_url(
            self._url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
        logger.info("redis_cache_connected", url=self._url)

    async def disconnect(self) -> None:
        """Close the connection pool.

        Call once at FastAPI lifespan shutdown.  Safe to call when not
        connected.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("redis_cache_disconnected")

    # ── Core primitives ────────────────────────────────────────────────────────

    async def get(self, key: str) -> Any | None:
        """Return the JSON-decoded value for *key*, or ``None`` on miss/error.

        Args:
            key: Redis key string.

        Returns:
            Decoded Python object, or ``None`` if the key is absent or if a
            Redis error occurs.
        """
        if self._client is None:
            logger.warning("redis_get_skipped_not_connected", key=key)
            return None
        try:
            raw: str | None = await self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis_get_error", key=key, error=str(exc))
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """JSON-encode *value* and store it under *key* with an optional TTL.

        Args:
            key:   Redis key string.
            value: Any JSON-serialisable Python object.
            ttl:   Expiry in seconds.  ``None`` means no expiry (key persists
                   until explicitly deleted).
        """
        if self._client is None:
            logger.warning("redis_set_skipped_not_connected", key=key)
            return
        try:
            serialised = json.dumps(value)
            if ttl is not None:
                await self._client.set(key, serialised, ex=ttl)
            else:
                await self._client.set(key, serialised)
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis_set_error", key=key, error=str(exc))

    async def delete(self, key: str) -> None:
        """Delete *key* from Redis.

        No-op (with a warning) if Redis is not connected or an error occurs.
        """
        if self._client is None:
            logger.warning("redis_delete_skipped_not_connected", key=key)
            return
        try:
            await self._client.delete(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis_delete_error", key=key, error=str(exc))

    async def exists(self, key: str) -> bool:
        """Return ``True`` if *key* exists in Redis, ``False`` otherwise.

        Returns ``False`` on Redis errors (fail-safe).
        """
        if self._client is None:
            return False
        try:
            count: int = await self._client.exists(key)
            return count > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis_exists_error", key=key, error=str(exc))
            return False

    # ── Typed helpers ──────────────────────────────────────────────────────────

    async def get_feature(self, symbol: str, bucket: str) -> Any | None:
        """Get a cached ``FeatureVector`` JSON blob.

        Args:
            symbol: Instrument symbol, e.g. ``"NIFTY"``.
            bucket: Timestamp bucketed to the nearest cache interval, e.g.
                    ``"20250101093000"``.

        Returns:
            Decoded dict or ``None`` on cache miss.
        """
        return await self.get(f"feature:{symbol}:{bucket}")

    async def set_feature(self, symbol: str, bucket: str, value: Any) -> None:
        """Store a ``FeatureVector`` JSON blob with the configured feature TTL.

        TTL is taken from ``settings.feature_cache_ttl`` (default 60 s).
        """
        await self.set(
            f"feature:{symbol}:{bucket}",
            value,
            ttl=settings.feature_cache_ttl,
        )

    async def get_news(self, symbol: str) -> Any | None:
        """Get the cached SentinelPulse news context for *symbol*."""
        return await self.get(f"news:{symbol}")

    async def set_news(self, symbol: str, value: Any) -> None:
        """Store SentinelPulse news context with the configured news TTL.

        TTL is taken from ``settings.news_context_cache_ttl`` (default 90 s).
        """
        await self.set(f"news:{symbol}", value, ttl=settings.news_context_cache_ttl)

    async def get_llm_news(self, symbol: str) -> Any | None:
        """Get the cached LLM news signal (FinBERT/FinGPT result) for *symbol*."""
        return await self.get(f"llm_news:{symbol}")

    async def set_llm_news(self, symbol: str, value: Any) -> None:
        """Store a LLM news signal with a fixed 60 s TTL."""
        await self.set(f"llm_news:{symbol}", value, ttl=60)

    async def get_signal(self, symbol: str) -> Any | None:
        """Get the latest cached ``MetaOutput`` signal for *symbol*."""
        return await self.get(f"signal:{symbol}:latest")

    async def set_signal(self, symbol: str, value: Any) -> None:
        """Store a ``MetaOutput`` signal with a fixed 300 s TTL."""
        await self.set(f"signal:{symbol}:latest", value, ttl=300)

    async def get_model_status(self) -> Any | None:
        """Get the current model-load status dict (no TTL)."""
        return await self.get("model:status")

    async def set_model_status(self, value: Any) -> None:
        """Store the model-load status dict without an expiry (persists until deleted)."""
        await self.set("model:status", value, ttl=None)

    async def get_drift_latest(self) -> Any | None:
        """Get the latest drift report (refreshed daily)."""
        return await self.get("drift:latest")

    async def set_drift_latest(self, value: Any) -> None:
        """Store the drift report with a 24-hour TTL."""
        await self.set("drift:latest", value, ttl=86_400)


# ── LRUCache ──────────────────────────────────────────────────────────────────


class LRUCache:
    """Thread-safe in-process LRU cache with optional per-entry TTL.

    Designed for two use-cases:

    * **SHAP explainer instances** — ``max_size=100``, ``ttl_seconds=None``
      (entries are evicted only when the cache is full or explicitly cleared,
      e.g. on model reload).
    * **News context** — ``max_size=500``, ``ttl_seconds=90`` (entries expire
      after 90 seconds, matching ``settings.news_context_cache_ttl``).

    Internally, an :class:`~collections.OrderedDict` is used with
    ``move_to_end`` to maintain recency order.  A :class:`threading.Lock`
    guards all mutations, making the cache safe for use from multiple threads
    or concurrent asyncio tasks (since asyncio tasks run on a single thread,
    the lock is a no-op overhead in practice but keeps the class correct when
    used in threaded contexts such as tests or background threads).

    Each stored entry is a ``(value, inserted_at)`` tuple where
    ``inserted_at`` is ``time.monotonic()`` at insertion time.
    """

    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Args:
            max_size:    Maximum number of entries.  When the cache is full,
                         the least-recently-used entry is evicted to make room.
            ttl_seconds: Optional entry lifetime in seconds (wall time via
                         :func:`time.monotonic`).  ``None`` means entries
                         never expire due to age.
        """
        self._max_size: int = max_size
        self._ttl: int | None = ttl_seconds
        # OrderedDict preserves insertion order; we move accessed keys to the
        # end so the *first* entry is always the least-recently-used one.
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock: Lock = Lock()

    # ── Public interface ───────────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        """Return the cached value for *key*, or ``None`` on miss / expiry.

        If the entry exists but its TTL has elapsed it is evicted immediately
        before returning ``None``.

        Args:
            key: Cache key string.

        Returns:
            The stored value, or ``None``.
        """
        with self._lock:
            if key not in self._cache:
                return None

            value, inserted_at = self._cache[key]

            # TTL check
            if self._ttl is not None:
                age = time.monotonic() - inserted_at
                if age > self._ttl:
                    del self._cache[key]
                    return None

            # Promote to most-recently-used position
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key*, evicting the LRU entry if at capacity.

        If *key* already exists its value is updated in-place and it is
        promoted to the most-recently-used position.

        Args:
            key:   Cache key string.
            value: Any Python object.
        """
        with self._lock:
            if key in self._cache:
                # Update existing entry and promote it
                self._cache.move_to_end(key)
                self._cache[key] = (value, time.monotonic())
                return

            # Evict LRU entry if at capacity
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # pop the oldest (front) item

            self._cache[key] = (value, time.monotonic())

    def delete(self, key: str) -> None:
        """Remove *key* from the cache.  No-op if the key is absent.

        Args:
            key: Cache key string.
        """
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Evict all entries from the cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        """Return the number of entries currently in the cache (including expired ones)."""
        with self._lock:
            return len(self._cache)
