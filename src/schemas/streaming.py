"""
Streaming schemas for ml-service2.0.

SignalEvent — WebSocket message emitted by the SignalStreamer for each new
              MetaDecisionEngine output.  Sent to alpha-forge via
              ``WS /v2/stream/signals``.

The full ``MetaOutput`` is nested as a plain ``dict[str, Any]`` to avoid a
circular import between this module and ``src/schemas/meta.py``.  Callers
that need a typed ``MetaOutput`` can do:

    from src.schemas.meta import MetaOutput
    typed = MetaOutput.model_validate(event.meta_output)
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import Field

from src.schemas.base import BaseSchema

if TYPE_CHECKING:
    from src.schemas.meta import MetaOutput


class SignalEvent(BaseSchema):
    """WebSocket message broadcast by SignalStreamer for every new trading signal.

    ``action`` mirrors ``MetaOutput.action`` (BUY | SELL | WAIT | NO_TRADE).
    ``provenance`` mirrors ``MetaOutput.provenance`` serialised as its string
    value so downstream consumers do not need to import the enum.
    ``meta_output`` carries the complete ``MetaOutput`` payload as a raw dict
    to avoid circular schema imports.
    """

    event_id: str  # UUID — unique per emission
    symbol: str
    timestamp: datetime  # UTC — when the signal was generated
    action: str  # "BUY" | "SELL" | "WAIT" | "NO_TRADE"
    confidence: float  # [0, 1] — mirrors MetaOutput.confidence
    provenance: str  # PredictionProvenance.value — e.g. "trained_model"
    reason_codes: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0  # end-to-end latency from request to signal emit
    # Full MetaOutput serialised as a dict; use MetaOutput.model_validate() to re-hydrate
    meta_output: dict[str, Any] = Field(default_factory=dict)
