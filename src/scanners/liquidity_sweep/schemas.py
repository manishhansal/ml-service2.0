"""
Pydantic schemas and enums for the Liquidity Sweep Scanner.

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


class SweepDirection(str, Enum):
    """Direction of a liquidity sweep event."""

    BULLISH = "bullish"   # swept lows (stop-hunt below support, reversal up)
    BEARISH = "bearish"   # swept highs (stop-hunt above resistance, reversal down)


class SweepLifecycle(str, Enum):
    """Lifecycle state for a detected sweep signal."""

    ACTIVE = "ACTIVE"
    CONFIRMED = "CONFIRMED"
    INVALIDATED = "INVALIDATED"


# ── Primary domain objects ────────────────────────────────────────────────────


class SweepSignal(BaseSchema):
    """
    A single detected liquidity sweep / stop-hunt reversal event.

    ``lifecycle_state`` begins as ACTIVE and transitions to CONFIRMED once
    the reversal candle closes in the opposite direction, or INVALIDATED if
    price continues through the swept level.
    """

    # Identity
    symbol: str
    direction: SweepDirection
    timeframe: str

    # Swept level and zone
    swept_level: float
    sweep_zone_high: float
    sweep_zone_low: float

    # Timing
    sweep_timestamp: datetime
    sweep_bar_index: int

    # Lifecycle
    lifecycle_state: SweepLifecycle = Field(default=SweepLifecycle.ACTIVE)

    # Liquidity pool metadata
    equal_levels_count: int = Field(
        default=1,
        description="Number of equal highs/lows clustered at the swept level",
    )

    # Scoring
    sweep_magnitude: float = Field(
        default=0.0,
        description="Distance swept beyond the level, normalised by ATR",
    )
    volume_ratio: float = Field(
        default=0.0,
        description="Sweep bar volume divided by the rolling average volume",
    )
    conviction_score: float = Field(
        default=0.0,
        description="Composite score combining sweep magnitude, volume ratio, and equal-levels count",
    )

    # Confirmation
    confirmed_timestamp: datetime | None = Field(
        default=None,
        description="Timestamp of the reversal confirmation candle (k+1 bar)",
    )

    # Multi-TF confluence
    multi_tf_confluence: bool = Field(
        default=False,
        description="True when the same swept level appears across multiple timeframes within confluence_atr_multiplier distance",
    )

    # Intraday trading context (computed from daily zone + last bar data)
    intraday_entry_price: float = Field(default=0.0, description="Intraday limit entry price at zone boundary")
    intraday_target_price: float = Field(default=0.0, description="First intraday profit target (1.5R from entry)")
    intraday_stop_price: float = Field(default=0.0, description="Intraday stop loss level")
    intraday_rr_ratio: float = Field(default=0.0, description="Risk:reward ratio for the intraday setup")
    proximity_pct: float = Field(default=999.0, description="% distance of current close from intraday entry")
    is_actionable: bool = Field(default=False, description="True when price is within 2% of the entry level")


class SweepScanResult(BaseSchema):
    """
    Internal result returned by the sweep scanner before being mapped to the
    HTTP response schema.
    """

    signals: list[SweepSignal]
    scanned_at: datetime
    n_symbols_scanned: int


class SweepScanResponse(BaseSchema):
    """
    Full API response for ``GET /v2/scanners/liquidity-sweeps``.
    """

    scanned_at: datetime
    n_symbols_scanned: int
    n_signals: int
    signals: list[SweepSignal]
