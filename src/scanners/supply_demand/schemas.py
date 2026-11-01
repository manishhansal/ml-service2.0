"""
Pydantic schemas and enums for the Supply & Demand Zone Scanner.

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


class SDZoneType(str, Enum):
    """Four pattern types describing entry-leg / departure-leg direction combinations."""

    RBR = "RBR"  # Rally-Base-Rally → DEMAND
    DBD = "DBD"  # Drop-Base-Drop  → SUPPLY
    RBD = "RBD"  # Rally-Base-Drop → SUPPLY
    DBR = "DBR"  # Drop-Base-Rally → DEMAND


class SDDirection(str, Enum):
    """Whether the zone acts as a supply (overhead resistance) or demand (underlying support)."""

    SUPPLY = "SUPPLY"
    DEMAND = "DEMAND"


class SDLifecycle(str, Enum):
    """Lifecycle state of a supply & demand zone."""

    FRESH = "FRESH"
    TESTED = "TESTED"
    USED = "USED"
    BROKEN = "BROKEN"


# ── Domain objects ─────────────────────────────────────────────────────────────


class SDZone(BaseSchema):
    """
    Fully detected and ranked supply & demand zone.

    Default field values use ``Field(default=...)`` explicitly because the
    base class sets ``frozen=True``, which is compatible with Pydantic V2
    field defaults when declared via ``Field``.
    """

    # Identity
    symbol: str
    direction: SDDirection
    zone_type: SDZoneType
    timeframe: str

    # Zone bounds
    zone_high: float
    zone_low: float

    # Timestamps
    entry_leg_timestamp: datetime
    departure_timestamp: datetime

    # Bar indices
    base_start_bar_index: int
    base_end_bar_index: int
    departure_bar_index: int

    # Lifecycle
    lifecycle_state: SDLifecycle = Field(default=SDLifecycle.FRESH)
    zone_strength: float = Field(default=1.0)
    broken_timestamp: datetime | None = Field(default=None)

    # Multi-TF confluence
    multi_tf_confluence: bool = Field(default=False)

    # Scores
    departure_strength: float = Field(default=0.0)
    combined_score: float = Field(default=0.0)

    # Intraday trading context (computed from daily zone + last bar data)
    intraday_entry_price: float = Field(default=0.0, description="Intraday limit entry price at zone boundary")
    intraday_target_price: float = Field(default=0.0, description="First intraday profit target (1.5R from entry)")
    intraday_stop_price: float = Field(default=0.0, description="Intraday stop loss level")
    intraday_rr_ratio: float = Field(default=0.0, description="Risk:reward ratio for the intraday setup")
    proximity_pct: float = Field(default=999.0, description="% distance of current close from intraday entry")
    is_actionable: bool = Field(default=False, description="True when price is within 2% of the entry level")


class SDScanResult(BaseSchema):
    """
    Internal result returned by the SD scanner before being mapped to the HTTP response.
    """

    zones: list[SDZone]
    scanned_at: datetime
    n_symbols_scanned: int


class SDScanResponse(BaseSchema):
    """
    Full API response for ``GET /v2/scanners/supply-demand``.
    """

    scanned_at: datetime
    n_symbols_scanned: int
    n_zones: int
    zones: list[SDZone]
