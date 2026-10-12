"""
Pydantic schemas and enums for the Order Block Scanner.

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


class OBZoneDirection(str, Enum):
    """Direction of an order block zone."""

    BULLISH = "bullish"
    BEARISH = "bearish"


class OBZoneLifecycle(str, Enum):
    """Lifecycle state for an order block zone (Req 6)."""

    ACTIVE = "ACTIVE"
    TESTED = "TESTED"
    RETESTED = "RETESTED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class OBMarketRegime(str, Enum):
    """Broad market regime produced by the BreadthTracker (Req 5)."""

    BULLISH_SWEEP = "BULLISH_SWEEP"
    BEARISH_SWEEP = "BEARISH_SWEEP"
    MIXED = "MIXED"


# ── Intermediate objects ───────────────────────────────────────────────────────


class OBCandidate(BaseSchema):
    """
    Unconfirmed order block candidate — output of ``OrderBlockDetector``.

    Becomes an ``OBCandidateConfirmed`` after ``StructureValidator`` confirms a BoS.
    """

    symbol: str
    ob_bar_index: int           # Index i of the OB candle in the OHLCV series
    ob_timestamp: datetime
    direction: OBZoneDirection
    ob_high: float
    ob_low: float
    ob_open: float
    ob_close: float
    ob_volume: float
    impulse_bar_index: int      # Index j of the first impulse candle after i
    impulse_atr: float          # ATR14 value at bar j
    timeframe: str


class OBCandidateConfirmed(OBCandidate):
    """
    ``OBCandidate`` that passed ``StructureValidator`` — carries BoS evidence (Req 2.5).
    """

    bos_bar_index: int          # Index j where close crossed the swing level
    bos_timestamp: datetime
    bos_level: float            # The swing_high or swing_low that was broken


# ── Primary domain object ──────────────────────────────────────────────────────


class OBZone(BaseSchema):
    """
    Fully confirmed, filtered, and ranked order block zone.

    Maps directly to the REST response fields specified in Req 8 AC5.
    Default field values use ``Field(default=...)`` explicitly because the
    base class sets ``frozen=True``, which is compatible with Pydantic V2
    field defaults when declared via ``Field``.
    """

    # Identity
    symbol: str
    direction: OBZoneDirection
    timeframe: str

    # Zone bounds
    ob_high: float
    ob_low: float
    ob_timestamp: datetime      # ISO-8601 UTC (serialised by Pydantic)

    # BoS evidence
    bos_timestamp: datetime

    # Bar indices — integer positions in the OHLCV series (for fast backtesting)
    ob_bar_index: int = Field(default=0, description="Integer position of the OB candle in the series")
    bos_bar_index: int = Field(default=0, description="Integer position of the BoS confirmation candle")

    # Scores
    volume_score: float                # ob_volume / avg_volume_20
    move_strength_ratio: float         # |close_j - open_j| / ATR50_j
    breadth_alignment_score: float

    # Lifecycle
    lifecycle_state: OBZoneLifecycle = Field(default=OBZoneLifecycle.ACTIVE)
    retest_count: int = Field(default=0)
    invalidation_timestamp: datetime | None = Field(default=None)

    # Multi-TF confluence
    multi_tf_confluence: bool = Field(default=False)

    # Intraday trading context (computed from daily zone + last bar data)
    intraday_entry_price: float = Field(default=0.0, description="Intraday limit entry price at zone boundary")
    intraday_target_price: float = Field(default=0.0, description="First intraday profit target (1.5R from entry)")
    intraday_stop_price: float = Field(default=0.0, description="Intraday stop loss level")
    intraday_rr_ratio: float = Field(default=0.0, description="Risk:reward ratio for the intraday setup")
    proximity_pct: float = Field(default=999.0, description="% distance of current close from intraday entry")
    is_actionable: bool = Field(default=False, description="True when price is within 2% of the entry level")


# ── Breadth and scan aggregates ────────────────────────────────────────────────


class BreadthResult(BaseSchema):
    """
    Market-breadth metrics produced by ``BreadthTracker`` (Req 5).

    ``alignment_scores`` maps each symbol to the breadth value that matches
    the direction of its active zones.
    """

    bullish_breadth: float
    bearish_breadth: float
    market_regime: OBMarketRegime
    alignment_scores: dict[str, float]


class OrderBlockScanResult(BaseSchema):
    """
    Internal result returned by ``ObScanner.scan()`` before being mapped to the
    HTTP response schema.
    """

    zones: list[OBZone]
    breadth: BreadthResult
    scanned_at: datetime
    n_symbols_scanned: int


class OrderBlockScanResponse(BaseSchema):
    """
    Full API response for ``GET /v2/scanners/order-blocks``.

    Maps to Req 8 AC5 (per-zone fields) and Req 8 AC6 (top-level fields).
    """

    # Top-level summary (Req 8 AC6)
    scanned_at: datetime
    n_symbols_scanned: int
    market_regime: OBMarketRegime
    bullish_breadth: float      # in [0.0, 1.0]
    bearish_breadth: float      # in [0.0, 1.0]
    n_zones: int

    # Per-zone list (Req 8 AC5)
    zones: list[OBZone]
