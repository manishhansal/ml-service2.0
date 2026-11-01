"""
Pydantic schemas and enums for the Fair Value Gap Scanner.

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


class FVGDirection(str, Enum):
    """Direction of a fair value gap zone."""

    BULLISH = "bullish"
    BEARISH = "bearish"


class FVGLifecycle(str, Enum):
    """Lifecycle state for a fair value gap zone."""

    UNFILLED = "UNFILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    EXPIRED = "EXPIRED"


# ── Primary domain object ──────────────────────────────────────────────────────


class FVGZone(BaseSchema):
    """
    A detected fair value gap zone produced by the 3-candle imbalance pattern.

    Maps directly to the REST response fields for ``GET /v2/scanners/fair-value-gaps``.
    Default field values use ``Field(default=...)`` explicitly because the
    base class sets ``frozen=True``, which is compatible with Pydantic V2
    field defaults when declared via ``Field``.
    """

    # Identity
    symbol: str
    direction: FVGDirection
    timeframe: str

    # Zone bounds
    fvg_high: float
    fvg_low: float
    fvg_midpoint: float
    fvg_timestamp: datetime         # ISO-8601 UTC timestamp of the FVG formation

    # Bar indices (position in the OHLCV series)
    candle1_bar_index: int          # Index of the first candle (before the impulse)
    candle3_bar_index: int          # Index of the third candle (after the impulse)

    # Impulse context
    impulse_atr: float              # ATR value at the impulse candle

    # Lifecycle
    lifecycle_state: FVGLifecycle = Field(default=FVGLifecycle.UNFILLED)
    fill_timestamp: datetime | None = Field(default=None)

    # Scores
    gap_size_score: float = Field(default=0.0)      # gap_size / ATR normalised score
    volume_score: float = Field(default=0.0)         # impulse volume / avg_volume score
    combined_score: float = Field(default=0.0)       # composite ranking score

    # Intraday trading context (computed from daily zone + last bar data)
    intraday_entry_price: float = Field(default=0.0, description="Intraday limit entry price at zone boundary")
    intraday_target_price: float = Field(default=0.0, description="First intraday profit target (1.5R from entry)")
    intraday_stop_price: float = Field(default=0.0, description="Intraday stop loss level")
    intraday_rr_ratio: float = Field(default=0.0, description="Risk:reward ratio for the intraday setup")
    proximity_pct: float = Field(default=999.0, description="% distance of current close from intraday entry")
    is_actionable: bool = Field(default=False, description="True when price is within 2% of the entry level")


# ── Scan aggregates ────────────────────────────────────────────────────────────


class FVGScanResult(BaseSchema):
    """
    Internal result returned by ``FVGScanner.scan()`` before being mapped to the
    HTTP response schema.
    """

    zones: list[FVGZone]
    scanned_at: datetime
    n_symbols_scanned: int


class FVGScanResponse(BaseSchema):
    """
    Full API response for ``GET /v2/scanners/fair-value-gaps``.
    """

    # Top-level summary
    scanned_at: datetime
    n_symbols_scanned: int
    n_zones: int

    # Per-zone list
    zones: list[FVGZone]
