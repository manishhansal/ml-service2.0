"""
Mock-based tests for src/streaming/streamer.py.
"""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------


class TestConnectionManager:
    def test_connect_accepts_websocket_and_increments_count(self):
        from src.streaming.streamer import ConnectionManager

        async def _run():
            ws = AsyncMock()
            cm = ConnectionManager()
            await cm.connect(ws)
            assert cm.client_count == 1
            ws.accept.assert_called_once()

        asyncio.run(_run())

    def test_disconnect_removes_websocket(self):
        from src.streaming.streamer import ConnectionManager

        async def _run():
            ws = AsyncMock()
            cm = ConnectionManager()
            await cm.connect(ws)
            await cm.disconnect(ws)
            assert cm.client_count == 0

        asyncio.run(_run())

    def test_disconnect_on_unknown_ws_does_not_raise(self):
        from src.streaming.streamer import ConnectionManager

        async def _run():
            cm = ConnectionManager()
            ws = AsyncMock()
            # Never connected — disconnect should be a no-op
            await cm.disconnect(ws)
            assert cm.client_count == 0

        asyncio.run(_run())

    def test_client_count_initial_zero(self):
        from src.streaming.streamer import ConnectionManager

        cm = ConnectionManager()
        assert cm.client_count == 0

    def test_broadcast_sends_to_all_clients(self):
        from src.streaming.streamer import ConnectionManager

        async def _run():
            ws1 = AsyncMock()
            ws2 = AsyncMock()
            cm = ConnectionManager()
            await cm.connect(ws1)
            await cm.connect(ws2)
            msg = {"action": "BUY", "symbol": "NIFTY"}
            await cm.broadcast(msg)
            ws1.send_json.assert_called_once_with(msg)
            ws2.send_json.assert_called_once_with(msg)

        asyncio.run(_run())

    def test_broadcast_removes_stale_client(self):
        """If send_json raises, the stale client should be removed."""
        from src.streaming.streamer import ConnectionManager

        async def _run():
            ws_good = AsyncMock()
            ws_stale = AsyncMock()
            ws_stale.send_json.side_effect = RuntimeError("connection closed")

            cm = ConnectionManager()
            await cm.connect(ws_good)
            await cm.connect(ws_stale)
            assert cm.client_count == 2

            await cm.broadcast({"action": "WAIT"})

            # Stale client removed, good client still present
            assert cm.client_count == 1
            ws_good.send_json.assert_called_once()

        asyncio.run(_run())

    def test_multiple_connect_disconnect_cycles(self):
        from src.streaming.streamer import ConnectionManager

        async def _run():
            cm = ConnectionManager()
            ws = AsyncMock()
            await cm.connect(ws)
            await cm.connect(ws)  # connects twice
            assert cm.client_count == 2
            await cm.disconnect(ws)
            assert cm.client_count == 1

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# SignalStreamer
# ---------------------------------------------------------------------------


class TestSignalStreamer:
    def test_broadcast_signal_with_zero_clients_returns_immediately(self):
        """No crash when no clients connected."""
        from src.streaming.streamer import SignalStreamer

        async def _run():
            streamer = SignalStreamer()
            assert streamer.manager.client_count == 0
            # Should return without calling broadcast
            await streamer.broadcast_signal({"action": "BUY", "confidence": 0.8}, "NIFTY")

        asyncio.run(_run())

    def test_broadcast_signal_with_dict_meta_output(self):
        """dict meta_output goes through dict() path."""
        from src.streaming.streamer import SignalStreamer

        async def _run():
            streamer = SignalStreamer()
            ws = AsyncMock()
            await streamer.manager.connect(ws)

            meta_output = {
                "action": "BUY",
                "confidence": 0.75,
                "provenance": "trained_model",
                "reason_codes": ["IC_OK"],
                "latency_ms": 12.5,
            }
            await streamer.broadcast_signal(meta_output, "RELIANCE")

            ws.send_json.assert_called_once()
            sent = ws.send_json.call_args[0][0]
            assert sent["symbol"] == "RELIANCE"
            assert sent["action"] == "BUY"

        asyncio.run(_run())

    def test_broadcast_signal_with_pydantic_model_meta_output(self):
        """meta_output with model_dump() (Pydantic model) is handled."""
        from src.streaming.streamer import SignalStreamer

        async def _run():
            streamer = SignalStreamer()
            ws = AsyncMock()
            await streamer.manager.connect(ws)

            # Create a mock that has model_dump()
            meta_output = MagicMock()
            meta_output.model_dump.return_value = {
                "action": "SELL",
                "confidence": 0.6,
                "provenance": "trained_model",
                "reason_codes": [],
                "latency_ms": 8.0,
            }

            await streamer.broadcast_signal(meta_output, "BANKNIFTY")

            meta_output.model_dump.assert_called_once()
            ws.send_json.assert_called_once()
            sent = ws.send_json.call_args[0][0]
            assert sent["action"] == "SELL"

        asyncio.run(_run())

    def test_broadcast_signal_event_id_is_unique(self):
        """Each broadcast should produce a unique event_id."""
        from src.streaming.streamer import SignalStreamer

        async def _run():
            streamer = SignalStreamer()
            ws = AsyncMock()
            await streamer.manager.connect(ws)

            meta = {"action": "WAIT", "confidence": 0.5}
            await streamer.broadcast_signal(meta, "NIFTY")
            await streamer.broadcast_signal(meta, "NIFTY")

            calls = ws.send_json.call_args_list
            id1 = calls[0][0][0]["event_id"]
            id2 = calls[1][0][0]["event_id"]
            assert id1 != id2

        asyncio.run(_run())

    def test_broadcast_signal_includes_confidence_and_symbol(self):
        from src.streaming.streamer import SignalStreamer

        async def _run():
            streamer = SignalStreamer()
            ws = AsyncMock()
            await streamer.manager.connect(ws)

            meta = {"action": "BUY", "confidence": 0.92, "reason_codes": ["IC_PASS"]}
            await streamer.broadcast_signal(meta, "WIPRO")

            sent = ws.send_json.call_args[0][0]
            assert sent["symbol"] == "WIPRO"
            assert sent["confidence"] == 0.92

        asyncio.run(_run())
