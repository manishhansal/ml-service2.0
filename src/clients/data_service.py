"""
HTTP REST client for data-service2.0.

Implements every public endpoint consumed by ml-service2.0, along with:
  - X-API-KEY authentication on every request (Req 1.3)
  - Circuit-breaker (3 consecutive failures → open for 60 s; half-open retry)
  - signalEngineAllowed quality gate (Req 1.5, Req 1.6)
  - DataConfidenceScore quality gate (Req 1.7)
  - 3m interval ban for Indian market data (Req 1.8)
  - 401/403 abort-and-log without retry (Req 1.4)

Usage::

    client = DataServiceClient()
    await client.connect()
    quote = await client.get_live_quote("RELIANCE")
    await client.disconnect()
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from src.config import settings
from src.logging_config import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical intervals permitted for Indian market data (Req 1.8)
# ---------------------------------------------------------------------------
CANONICAL_INTERVALS: list[str] = [
    "1m", "5m", "10m", "15m", "30m",
    "1h", "1d", "1w", "1M",
]


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class DataServiceAuthError(Exception):
    """Raised when data-service2.0 returns HTTP 401 or 403.

    The caller must NOT retry; return PredictionProvenance.UNAVAILABLE (Req 1.4).
    """


class SignalEngineNotAllowedError(Exception):
    """Raised when the response payload has signalEngineAllowed == false (Req 1.5)."""


class LowDataConfidenceError(Exception):
    """Raised when DataConfidenceScore is below settings.min_confidence_score (Req 1.7)."""


class DataServiceUnavailableError(Exception):
    """Raised on connection failures, timeouts, or while the circuit is open."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class DataServiceClient:
    """Async HTTP client for data-service2.0 with auth, circuit-breaker, and quality gates.

    Lifecycle::

        client = DataServiceClient()
        await client.connect()      # call once at application startup
        ...
        await client.disconnect()   # call at application shutdown
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._base_url: str = (base_url or settings.data_service_2_url).rstrip("/")
        self._api_key: str = api_key or settings.data_service_api_key
        self._client: httpx.AsyncClient | None = None

        # Circuit-breaker state
        self._circuit_open_until: float = 0.0   # monotonic timestamp
        self._consecutive_failures: int = 0
        self._circuit_threshold: int = 3         # failures before opening
        self._circuit_timeout: float = 60.0      # seconds before half-open retry

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the shared httpx.AsyncClient.  Call once at startup."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"X-API-KEY": self._api_key},
            timeout=10.0,  # 10 s per Req 2.7
        )

    async def disconnect(self) -> None:
        """Close the underlying HTTP connection pool.  Call at shutdown."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Circuit-breaker helpers ────────────────────────────────────────────────

    def _is_circuit_open(self) -> bool:
        """Return True when the circuit is open and the timeout has not expired.

        Returns False (half-open) when the timeout has passed, allowing one
        probe request through.
        """
        if self._circuit_open_until == 0.0:
            return False
        return time.monotonic() < self._circuit_open_until

    def _record_success(self) -> None:
        """Reset the failure counter and close the circuit after a successful call."""
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _record_failure(self) -> None:
        """Increment failure counter; open the circuit once the threshold is reached."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._circuit_threshold:
            self._circuit_open_until = time.monotonic() + self._circuit_timeout
            log.warning(
                "data_service_circuit_opened",
                consecutive_failures=self._consecutive_failures,
                circuit_timeout_s=self._circuit_timeout,
            )

    # ── Quality-gate helpers ──────────────────────────────────────────────────

    def _check_quality_gates(
        self,
        response_data: dict[str, Any],
        instrument: str,
        request_timestamp: str,
    ) -> None:
        """Inspect the response payload for quality-gate fields and raise on violations.

        The method probes two common locations that data-service2.0 uses for
        quality metadata:
          - ``response["metadata"]["quality"]``  (standard envelope)
          - ``response["data"]``                  (flat payloads like live quote)

        Args:
            response_data:      Parsed JSON response body.
            instrument:         Symbol/instrument being requested (for log context).
            request_timestamp:  ISO-8601 request timestamp (for log context).

        Raises:
            SignalEngineNotAllowedError: If ``signalEngineAllowed`` is ``False``.
            LowDataConfidenceError:     If ``DataConfidenceScore`` < threshold.
        """
        # Locate quality metadata — try both envelope locations.
        quality: dict[str, Any] = {}
        metadata = response_data.get("metadata") or {}
        if isinstance(metadata, dict):
            quality = metadata.get("quality") or {}

        # Flat payloads nest quality fields directly inside "data"
        data_block = response_data.get("data") or {}
        if isinstance(data_block, dict) and not quality:
            quality = data_block

        # ── signalEngineAllowed (Req 1.5, Req 1.6) ───────────────────────────
        signal_engine_allowed: bool | None = quality.get("signalEngineAllowed")
        if signal_engine_allowed is None:
            signal_engine_allowed = response_data.get("signalEngineAllowed")

        if signal_engine_allowed is False:
            confidence_received: int = int(
                quality.get("score", quality.get("DataConfidenceScore", 0))
            )
            log.warning(
                "signal_engine_not_allowed",
                instrument=instrument,
                request_timestamp=request_timestamp,
                data_confidence_score=confidence_received,
            )
            raise SignalEngineNotAllowedError(
                f"signalEngineAllowed=false for instrument={instrument!r} "
                f"at {request_timestamp}; DataConfidenceScore={confidence_received}"
            )

        # ── DataConfidenceScore (Req 1.7) ─────────────────────────────────────
        raw_score = quality.get("score") or quality.get("DataConfidenceScore")
        if raw_score is None:
            raw_score = response_data.get("DataConfidenceScore")

        if raw_score is not None:
            score = int(raw_score)
            threshold = settings.min_confidence_score
            if score < threshold:
                raise LowDataConfidenceError(
                    f"DataConfidenceScore={score} below threshold={threshold} "
                    f"for instrument={instrument!r} at {request_timestamp}"
                )

    # ── Internal HTTP helpers ─────────────────────────────────────────────────

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Internal GET with circuit-breaker, auth handling, and error mapping.

        Raises:
            DataServiceUnavailableError: Circuit is open, or a connection/timeout error occurred.
            DataServiceAuthError:        HTTP 401 or 403 received (no retry).
        """
        if self._is_circuit_open():
            raise DataServiceUnavailableError(
                "data-service2.0 circuit is open — fast-failing request"
            )

        if self._client is None:
            raise DataServiceUnavailableError(
                "DataServiceClient.connect() has not been called"
            )

        try:
            response = await self._client.get(path, params=params)
        except httpx.TimeoutException as exc:
            self._record_failure()
            raise DataServiceUnavailableError(
                f"Timeout calling data-service2.0 GET {path}: {exc}"
            ) from exc
        except httpx.ConnectError as exc:
            self._record_failure()
            raise DataServiceUnavailableError(
                f"Connection error calling data-service2.0 GET {path}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            self._record_failure()
            raise DataServiceUnavailableError(
                f"HTTP error calling data-service2.0 GET {path}: {exc}"
            ) from exc

        # Auth errors — abort immediately, no retry (Req 1.4)
        if response.status_code in (401, 403):
            instrument = str(params or path)
            log.warning(
                "data_service_auth_error",
                credential_type="X-API-KEY",
                instrument=instrument,
                status_code=response.status_code,
                path=path,
            )
            raise DataServiceAuthError(
                f"data-service2.0 returned {response.status_code} for "
                f"GET {path} (credential: X-API-KEY)"
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            self._record_failure()
            raise DataServiceUnavailableError(
                f"data-service2.0 returned {response.status_code} for GET {path}: {exc}"
            ) from exc

        self._record_success()
        return response.json()  # type: ignore[no-any-return]

    async def _post(
        self,
        path: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Internal POST with the same circuit-breaker and auth logic as ``_get``.

        Raises:
            DataServiceUnavailableError: Circuit is open, or a connection/timeout error occurred.
            DataServiceAuthError:        HTTP 401 or 403 received (no retry).
        """
        if self._is_circuit_open():
            raise DataServiceUnavailableError(
                "data-service2.0 circuit is open — fast-failing request"
            )

        if self._client is None:
            raise DataServiceUnavailableError(
                "DataServiceClient.connect() has not been called"
            )

        try:
            response = await self._client.post(path, json=body)
        except httpx.TimeoutException as exc:
            self._record_failure()
            raise DataServiceUnavailableError(
                f"Timeout calling data-service2.0 POST {path}: {exc}"
            ) from exc
        except httpx.ConnectError as exc:
            self._record_failure()
            raise DataServiceUnavailableError(
                f"Connection error calling data-service2.0 POST {path}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            self._record_failure()
            raise DataServiceUnavailableError(
                f"HTTP error calling data-service2.0 POST {path}: {exc}"
            ) from exc

        # Auth errors — abort immediately, no retry (Req 1.4)
        if response.status_code in (401, 403):
            log.warning(
                "data_service_auth_error",
                credential_type="X-API-KEY",
                instrument=path,
                status_code=response.status_code,
                path=path,
            )
            raise DataServiceAuthError(
                f"data-service2.0 returned {response.status_code} for "
                f"POST {path} (credential: X-API-KEY)"
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            self._record_failure()
            raise DataServiceUnavailableError(
                f"data-service2.0 returned {response.status_code} for POST {path}: {exc}"
            ) from exc

        self._record_success()
        return response.json()  # type: ignore[no-any-return]

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_live_quote(
        self,
        symbol: str,
        exchange: str = "NSE",
    ) -> dict[str, Any]:
        """Fetch the current live quote for *symbol*.

        Endpoint: ``GET /v1/india/quotes/{symbol}?exchange={exchange}``

        Applies quality gates after fetching (Req 1.5, Req 1.7).

        Args:
            symbol:   NSE/NFO instrument symbol (e.g. ``"RELIANCE"``).
            exchange: Exchange code (default ``"NSE"``).

        Returns:
            Parsed response JSON dict.

        Raises:
            DataServiceUnavailableError: Network/circuit failure.
            DataServiceAuthError:        401/403.
            SignalEngineNotAllowedError: signalEngineAllowed is false.
            LowDataConfidenceError:      DataConfidenceScore below threshold.
        """
        from datetime import datetime, timezone  # local import for timestamp

        path = f"/v1/india/quotes/{symbol}"
        params: dict[str, Any] = {"exchange": exchange}
        data = await self._get(path, params=params)
        self._check_quality_gates(
            data,
            instrument=symbol,
            request_timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )
        return data

    async def get_historical_ohlcv(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: str = "1d",
        from_date: str | None = None,
        to_date: str | None = None,
        pit_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch historical OHLCV bars for *symbol*.

        Endpoint: ``GET /v1/india/historical``

        Appends ``?pit_date=YYYY-MM-DD`` when *pit_date* is provided, activating
        data-service2.0's backtest / PIT mode (Req 2.x).

        Args:
            symbol:    NSE/NFO instrument symbol.
            exchange:  Exchange code (default ``"NSE"``).
            interval:  Bar interval.  Must be one of the canonical intervals.
                       ``"3m"`` is permanently banned for Indian data (Req 1.8).
            from_date: ISO-8601 date string for the start of the range.
            to_date:   ISO-8601 date string for the end of the range.
            pit_date:  Point-in-time boundary date (``YYYY-MM-DD``) for backtest
                       mode.  When supplied, only data with
                       ``availableAtMs ≤ T`` is returned.

        Returns:
            List of OHLCV bar dicts.

        Raises:
            ValueError:                  If ``interval == "3m"`` (Req 1.8).
            DataServiceUnavailableError: Network/circuit failure.
            DataServiceAuthError:        401/403.
            SignalEngineNotAllowedError: signalEngineAllowed is false.
            LowDataConfidenceError:      DataConfidenceScore below threshold.
        """
        from datetime import datetime, timezone  # local import for timestamp

        if interval == "3m":
            raise ValueError(
                "The '3m' interval is permanently banned for Indian market data (Req 1.8). "
                f"Use one of: {CANONICAL_INTERVALS}"
            )

        params: dict[str, Any] = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
        }
        if from_date is not None:
            params["from_date"] = from_date
        if to_date is not None:
            params["to_date"] = to_date
        if pit_date is not None:
            params["pit_date"] = pit_date

        data = await self._get("/v1/india/historical", params=params)
        self._check_quality_gates(
            data,
            instrument=symbol,
            request_timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

        # The response may wrap bars in a "data" list
        bars = data.get("data") or data.get("bars") or data
        if isinstance(bars, list):
            return bars  # type: ignore[return-value]
        return [bars]  # type: ignore[list-item]

    async def get_option_chain(
        self,
        underlying: str,
        expiry: str | None = None,
        exchange: str = "NSE",
    ) -> dict[str, Any]:
        """Fetch the option chain for *underlying*.

        Endpoint: ``GET /v1/india/option-chain``

        Args:
            underlying: Index or equity symbol (e.g. ``"NIFTY"``, ``"BANKNIFTY"``).
            expiry:     Option expiry date (``YYYY-MM-DD``).  If None, the
                        nearest expiry is returned.
            exchange:   Exchange code (default ``"NSE"``).

        Returns:
            Parsed option-chain response dict.

        Raises:
            DataServiceUnavailableError: Network/circuit failure.
            DataServiceAuthError:        401/403.
            SignalEngineNotAllowedError: signalEngineAllowed is false.
            LowDataConfidenceError:      DataConfidenceScore below threshold.
        """
        from datetime import datetime, timezone

        params: dict[str, Any] = {
            "underlying": underlying,
            "exchange": exchange,
        }
        if expiry is not None:
            params["expiry"] = expiry

        data = await self._get("/v1/india/option-chain", params=params)
        self._check_quality_gates(
            data,
            instrument=underlying,
            request_timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )
        return data

    async def get_market_status(self) -> dict[str, Any]:
        """Fetch current NSE market session status.

        Endpoint: ``GET /v1/india/market/status``

        Returns:
            Market-status response dict (isOpen, sessionType, nextOpenTs, …).

        Raises:
            DataServiceUnavailableError: Network/circuit failure.
            DataServiceAuthError:        401/403.
        """
        return await self._get("/v1/india/market/status")

    async def get_fno_universe(self) -> dict[str, Any]:
        """Fetch the current F&O tradeable universe from data-service2.0.

        Endpoint: ``GET /v1/instruments/fno-universe``

        Returns:
            F&O universe response dict containing the list of active instruments.

        Raises:
            DataServiceUnavailableError: Network/circuit failure.
            DataServiceAuthError:        401/403.
        """
        return await self._get("/v1/instruments/fno-universe")

    async def get_crypto_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch Binance OHLCV kline bars for a crypto *symbol*.

        Endpoint: ``GET /v1/crypto/{symbol}/klines``

        Note: The ``3m`` interval ban applies only to Indian market data.
        Crypto intervals follow Binance conventions.

        Args:
            symbol:   Binance pair symbol (e.g. ``"BTCUSDT"``).
            interval: Binance interval string (e.g. ``"1h"``, ``"4h"``, ``"1d"``).
            limit:    Maximum number of bars to return (default 500).

        Returns:
            List of kline/OHLCV bar dicts.

        Raises:
            DataServiceUnavailableError: Network/circuit failure.
            DataServiceAuthError:        401/403.
        """
        params: dict[str, Any] = {
            "interval": interval,
            "limit": limit,
        }
        data = await self._get(f"/v1/crypto/{symbol}/klines", params=params)

        # Normalise: response may wrap klines in "data" or "klines"
        klines = data.get("data") or data.get("klines") or data
        if isinstance(klines, list):
            return klines  # type: ignore[return-value]
        return [klines]  # type: ignore[list-item]

    async def evaluate_data_quality(
        self,
        data: dict[str, Any],
        min_confidence_score: int | None = None,
    ) -> dict[str, Any]:
        """Submit a data batch to data-service2.0's quality evaluation endpoint.

        Endpoint: ``POST /v1/quality/evaluate``

        Args:
            data:                 The data batch payload to evaluate.
            min_confidence_score: Optional override for the minimum acceptable
                                  DataConfidenceScore.  Defaults to
                                  ``settings.min_confidence_score``.

        Returns:
            Quality evaluation response dict.

        Raises:
            DataServiceUnavailableError: Network/circuit failure.
            DataServiceAuthError:        401/403.
        """
        body: dict[str, Any] = {"data": data}
        if min_confidence_score is not None:
            body["min_confidence_score"] = min_confidence_score
        else:
            body["min_confidence_score"] = settings.min_confidence_score

        return await self._post("/v1/quality/evaluate", body=body)
