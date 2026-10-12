"""
Pydantic schemas and enums for the Market Structure Scanner.

All models inherit from ``src.schemas.base.BaseSchema`` (``strict=True, frozen=True``).
Enums follow the ``(str, Enum)`` pattern used throughout the service so values serialise
as plain strings in JSON responses.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from src.schemas.base import BaseSchema


# ── Enums ─────────────────────────────────────────────────────────────────────


class MSSignalType(str, Enum):
    """Type of market structure signal detected."""

    BOS_BULLISH = "BOS_BULLISH"
    BOS_BEARISH = "BOS_BEARISH"
    CHOCH_BULLISH = "CHOCH_BULLISH"
    CHOCH_BEARISH = "CHOCH_BEARISH"


class TrendState(str, Enum):
    """Trend state of the symbol after the signal fires."""

    UNDEFINED = "UNDEFINED"
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


# ── Primary domain object ──────────────────────────────────────────────────────


class MSSignal(BaseSchema):
    """
    A single BoS or ChoCh market structure signal.

    Default field values use ``Field(default=...)`` explicitly because the
    base class sets ``frozen=True``, which is compatible with Pydantic V2
    field defaults when declared via ``Field``.
    """

    # Identity
    symbol: str
    signal_type: MSSignalType
    timeframe: str

    # Break detail
    broken_level: float
    close_at_break: float
    break_timestamp: datetime
    break_bar_index: int

    # Context
    trend_state: TrendState          # State AFTER this signal
    is_choch: bool
    swing_count: int

    # Scored / optional fields
    break_magnitude: float = Field(default=0.0)
    conviction_score: float = Field(default=0.0)
    invalidated: bool = Field(default=False)
    multi_tf_confluence: bool = Field(default=False)

    # Intraday trading context (computed from daily zone + last bar data)
    intraday_entry_price: float = Field(default=0.0, description="Intraday limit entry price at zone boundary")
    intraday_target_price: float = Field(default=0.0, description="First intraday profit target (1.5R from entry)")
    intraday_stop_price: float = Field(default=0.0, description="Intraday stop loss level")
    intraday_rr_ratio: float = Field(default=0.0, description="Risk:reward ratio for the intraday setup")
    proximity_pct: float = Field(default=999.0, description="% distance of current close from intraday entry")
    is_actionable: bool = Field(default=False, description="True when price is within 2% of the entry level")


# ── Scan aggregates ────────────────────────────────────────────────────────────


class MSScanResult(BaseSchema):
    """
    Internal result returned by the market structure scanner before being
    mapped to the HTTP response schema.
    """

    signals: list[MSSignal]
    scanned_at: datetime
    n_symbols_scanned: int


class MSScanResponse(BaseSchema):
    """
    Full API response for the market structure scanner endpoint.
    """

    scanned_at: datetime
    n_symbols_scanned: int
    n_signals: int
    signals: list[MSSignal]
