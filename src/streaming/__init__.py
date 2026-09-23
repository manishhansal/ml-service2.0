"""
Streaming package for ml-service2.0.

Exports:
    ConnectionManager — tracks and manages active WebSocket connections.
    SignalStreamer     — broadcasts MetaOutput signals to connected WS clients.
"""
from src.streaming.streamer import ConnectionManager, SignalStreamer

__all__ = [
    "ConnectionManager",
    "SignalStreamer",
]
