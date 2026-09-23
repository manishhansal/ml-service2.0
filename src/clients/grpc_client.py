"""
gRPC streaming client for data-service2.0's MarketDataFeeder service.

Implements:
  - Connection timeout: settings.grpc_connection_timeout (default 5 s)  (Req 1.17)
  - Per-call deadline: settings.grpc_per_call_deadline (default 2 s)     (Req 1.17)
  - Stream reconnect: up to settings.grpc_max_reconnect_attempts          (Req 1.18)
    with 1-second delay between each attempt
  - Graceful degradation: raises GRPCStreamError after exhausting retries
    so the caller can return PredictionProvenance.UNAVAILABLE              (Req 1.18)

Usage::

    client = GRPCMarketDataClient()
    await client.connect()

    async for bar in client.stream_ticks("NIFTY", "NSE", "5m"):
        # process bar dict (timestamp_ms, open, high, low, close, ...)
        ...

    snapshot = await client.get_feature_snapshot("RELIANCE")
    await client.disconnect()
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from src.config import settings
from src.logging_config import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class GRPCStreamError(Exception):
    """Raised when a gRPC stream fails and cannot be recovered after retries."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GRPCMarketDataClient:
    """
    gRPC streaming client for data-service2.0's MarketDataFeeder service.

    Lifecycle::

        client = GRPCMarketDataClient()
        await client.connect()      # call once at application startup
        ...
        await client.disconnect()   # call at application shutdown

    The client degrades gracefully when grpcio is not installed: ``connect()``
    becomes a no-op and ``is_available`` returns ``False``, allowing callers to
    fall back to the REST client transparently.
    """

    def __init__(self, host: str | None = None, port: int = 50051) -> None:
        # Derive hostname from the data_service_2_url setting, stripping the
        # scheme and any trailing port that was already there.
        raw_host = (
            settings.data_service_2_url
            .replace("https://", "")
            .replace("http://", "")
            .split(":")[0]
        )
        self._host: str = host or raw_host
        self._port: int = port
        self._channel: Any = None
        self._stub: Any = None
        self._connected: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Establish the gRPC channel with connection timeout.

        Uses ``grpc.aio`` if available; silently degrades to a no-op stub when
        grpcio is not installed so the service can still run without it.

        Connection timeout: ``settings.grpc_connection_timeout`` seconds.
        """
        try:
            import grpc.aio as grpc_aio  # type: ignore[import]
            from src.clients.market_data_pb2_grpc import MarketDataFeederStub

            target = f"{self._host}:{self._port}"
            self._channel = grpc_aio.insecure_channel(
                target,
                options=[
                    (
                        "grpc.connect_timeout_ms",
                        int(settings.grpc_connection_timeout * 1000),
                    ),
                    # 50 MB max message — generous upper bound for option-chain payloads
                    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                ],
            )
            self._stub = MarketDataFeederStub(self._channel)
            self._connected = True
            log.info("grpc_client_connected", target=target)
        except ImportError:
            log.warning("grpc_not_available_falling_back_to_rest")
            self._connected = False

    async def disconnect(self) -> None:
        """Close the gRPC channel and release resources."""
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None
            self._connected = False

    # ── Public API ────────────────────────────────────────────────────────────

    async def stream_ticks(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: str = "5m",
        pit_timestamp_ms: int = 0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream OHLCV bars for *symbol* via gRPC server-streaming.

        Per Req 1.17: connection timeout = ``settings.grpc_connection_timeout`` (5 s).
        Per Req 1.18: on stream interruption, retry up to
        ``settings.grpc_max_reconnect_attempts`` (3) with 1-second delays.
        After all retries fail, raises :exc:`GRPCStreamError`.

        Args:
            symbol:           Instrument symbol (e.g. ``"NIFTY"``).
            exchange:         Exchange code (default ``"NSE"``).
            interval:         Bar interval — must never be ``"3m"`` for NSE/NFO
                              data (Req 1.8).
            pit_timestamp_ms: >0 activates backtest / PIT mode (UTC epoch ms
                              boundary).  0 = live streaming mode.

        Yields:
            dict with OHLCV bar fields: ``timestamp_ms``, ``open``, ``high``,
            ``low``, ``close``, ``volume``, ``vwap``, ``oi``,
            ``volume_unavailable``, ``provider``, ``confidence_score``,
            ``signal_engine_allowed``.

        Raises:
            ValueError:      If ``interval == "3m"`` (Indian market ban,
                             Req 1.8).
            GRPCStreamError: After all reconnect attempts are exhausted
                             (Req 1.18).
        """
        # Indian market 3 m interval ban (Req 1.8)
        if interval == "3m":
            raise ValueError(
                "The '3m' interval is permanently banned for Indian market data (Req 1.8)."
            )

        if not self._connected or self._stub is None:
            raise GRPCStreamError(
                "gRPC client not connected — call connect() first or use REST fallback"
            )

        from src.clients.market_data_pb2 import TickStreamRequest  # type: ignore[import]

        request = TickStreamRequest(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            pit_timestamp_ms=pit_timestamp_ms,
        )

        attempts_remaining = settings.grpc_max_reconnect_attempts

        while True:
            try:
                async for response in self._stub.StreamTicks(
                    request,
                    timeout=settings.grpc_per_call_deadline,
                ):
                    bar = response.bar
                    yield {
                        "timestamp_ms": bar.timestamp_ms,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "vwap": bar.vwap,
                        "oi": bar.oi,
                        "volume_unavailable": bar.volume_unavailable,
                        "provider": bar.provider,
                        "confidence_score": bar.confidence_score,
                        "signal_engine_allowed": bar.signal_engine_allowed,
                    }
                    if response.is_final:
                        return  # clean backtest-mode end-of-stream
                return  # live stream ended normally

            except Exception as exc:
                log.warning(
                    "grpc_stream_interrupted",
                    symbol=symbol,
                    error=str(exc),
                    attempts_remaining=attempts_remaining,
                )
                if attempts_remaining <= 0:
                    raise GRPCStreamError(
                        f"gRPC stream for {symbol!r} failed after all reconnect "
                        f"attempts: {exc}"
                    ) from exc

                attempts_remaining -= 1
                await asyncio.sleep(1.0)  # Req 1.18: 1-second delay between retries
                # Re-establish the channel before the next attempt
                await self.connect()

    async def get_feature_snapshot(
        self,
        symbol: str,
        pit_date: str = "",
    ) -> dict[str, Any] | None:
        """Fetch a feature snapshot via unary gRPC RPC.

        Per Req 1.17: per-call deadline = ``settings.grpc_per_call_deadline`` (2 s).
        Returns ``None`` gracefully when gRPC is unavailable or the call fails,
        allowing the caller to fall back to the REST path.

        Args:
            symbol:   Instrument symbol (e.g. ``"RELIANCE"``).
            pit_date: ``YYYY-MM-DD`` date string; empty string = live snapshot.

        Returns:
            dict with feature fields, or ``None`` on any failure.
        """
        if not self._connected or self._stub is None:
            return None

        try:
            from src.clients.market_data_pb2 import FeatureRequest  # type: ignore[import]

            request = FeatureRequest(symbol=symbol, pit_date=pit_date)
            response = await self._stub.GetFeatureSnapshot(
                request,
                timeout=settings.grpc_per_call_deadline,
            )
            return {
                "symbol": response.symbol,
                "data_timestamp_ms": response.data_timestamp_ms,
                "confidence_score": response.confidence_score,
                "signal_engine_allowed": response.signal_engine_allowed,
                "india_vix": response.india_vix,
                "nifty_change_pct": response.nifty_change_pct,
                "banknifty_change_pct": response.banknifty_change_pct,
                "put_call_ratio": response.put_call_ratio,
                "advance_decline_ratio": response.advance_decline_ratio,
                "fii_net_cr": response.fii_net_cr,
                "dii_net_cr": response.dii_net_cr,
                "option_chain_json": response.option_chain_json,
                "market_breadth_json": response.market_breadth_json,
            }
        except Exception as exc:
            log.warning(
                "grpc_feature_snapshot_failed",
                symbol=symbol,
                error=str(exc),
            )
            return None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Return ``True`` when the gRPC channel is connected and the stub is ready."""
        return self._connected and self._stub is not None
