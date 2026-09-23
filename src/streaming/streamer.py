"""
SignalStreamer — WebSocket signal broadcast for ml-service2.0.

Manages WebSocket connections and broadcasts MetaOutput signals to
alpha-forge and other consumers via WS /v2/stream/signals.

Design:
- Connection manager with active connection tracking
- Graceful disconnect handling (no errors on client disconnect)
- Thread-safe broadcast to all connected clients
- JSON serialization of SignalEvent payloads

Requirements: Req 15.7, Req 12
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from src.logging_config import get_logger
from src.schemas.streaming import SignalEvent

logger = get_logger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self) -> None:
        self._active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await ws.accept()
        self._active.append(ws)
        logger.info("websocket_client_connected", n_clients=len(self._active))

    async def disconnect(self, ws: WebSocket) -> None:
        """Remove a WebSocket from the active list."""
        if ws in self._active:
            self._active.remove(ws)
        logger.info("websocket_client_disconnected", n_clients=len(self._active))

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all connected clients, removing stale connections."""
        stale: list[WebSocket] = []
        for ws in list(self._active):
            try:
                await ws.send_json(message)
            except Exception:
                stale.append(ws)
        for ws in stale:
            await self.disconnect(ws)

    @property
    def client_count(self) -> int:
        """Return the number of currently active WebSocket connections."""
        return len(self._active)


class SignalStreamer:
    """
    Broadcasts MetaOutput signals to connected WebSocket clients.

    Usage::
        streamer = SignalStreamer()
        # In a WebSocket handler:
        await streamer.manager.connect(websocket)
        # When a new signal is produced:
        await streamer.broadcast_signal(meta_output, symbol)
    """

    def __init__(self) -> None:
        self.manager = ConnectionManager()

    async def broadcast_signal(
        self,
        meta_output: Any,  # MetaOutput instance or dict
        symbol: str,
    ) -> None:
        """Broadcast a MetaOutput signal to all connected clients.

        Assembles a ``SignalEvent`` from the ``meta_output`` (either a Pydantic
        model instance or a plain dict) and broadcasts it as JSON to every
        registered WebSocket client.  Stale / disconnected clients are pruned
        automatically inside ``ConnectionManager.broadcast``.
        """
        if self.manager.client_count == 0:
            return

        # Normalise to dict regardless of whether a Pydantic model was passed
        if hasattr(meta_output, "model_dump"):
            meta_dict = meta_output.model_dump()
        else:
            meta_dict = dict(meta_output)

        event = SignalEvent(
            event_id=str(uuid.uuid4()),
            symbol=symbol,
            timestamp=datetime.now(tz=timezone.utc),
            action=meta_dict.get("action", "WAIT"),
            confidence=float(meta_dict.get("confidence", 0.5)),
            provenance=str(meta_dict.get("provenance", "unavailable")),
            reason_codes=meta_dict.get("reason_codes", []),
            latency_ms=float(meta_dict.get("latency_ms", 0.0)),
            meta_output=meta_dict,
        )

        await self.manager.broadcast(event.model_dump(mode="json"))

        logger.debug(
            "signal_broadcast",
            symbol=symbol,
            action=event.action,
            n_clients=self.manager.client_count,
        )
