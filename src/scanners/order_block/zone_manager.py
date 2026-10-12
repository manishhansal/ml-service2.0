"""
ZoneManager — lifecycle state-machine for OBZone objects.

Implements ``update_zone_lifecycle()``, which advances (or holds) an OBZone
through the five-state lifecycle defined in Req 6:

    ACTIVE → TESTED → RETESTED
           ↘         ↙
            INVALIDATED  (terminal — any state)
    ACTIVE → EXPIRED

All transitions produce a *new* OBZone via ``.model_copy(update={...})``;
the input zone is never mutated (frozen Pydantic model).
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.scanners.order_block.config import OrderBlockConfig
from src.scanners.order_block.schemas import OBZone, OBZoneDirection, OBZoneLifecycle


def update_zone_lifecycle(
    zone: OBZone,
    candle_low: float,
    candle_high: float,
    candle_close: float,
    candle_bar_index: int,
    creation_bar_index: int,
    config: OrderBlockConfig,
) -> OBZone:
    """
    Advance ``zone`` through its lifecycle state machine based on a single
    incoming candle.  Returns early (without copying) when no transition fires.

    Steps (executed in order; returns on first triggered transition):

    1. **Terminal guard** — INVALIDATED zones are returned immediately.
    2. **Invalidation** — applies to all non-INVALIDATED states (including
       EXPIRED).  Closes beyond the zone boundary invalidate permanently.
    3. **Expiry** — ACTIVE zones older than ``config.zone_expiry_bars`` without
       a test event expire.
    4. **Test transition** — an ACTIVE zone whose price boundary is touched by
       the candle (without closing through it) moves to TESTED.
    5. **Retest transition** — a TESTED or RETESTED zone touched again increments
       ``retest_count`` and moves to (or stays at) RETESTED.

    Parameters
    ----------
    zone:
        The current OBZone to update.
    candle_low / candle_high / candle_close:
        OHLCV values for the incoming candle.
    candle_bar_index:
        Absolute bar index of the incoming candle in the OHLCV series.
    creation_bar_index:
        Absolute bar index at which this zone was first created (used for
        expiry age calculation).
    config:
        Scanner configuration; ``zone_expiry_bars`` is read here.

    Returns
    -------
    OBZone
        Either the same object (no transition) or a new frozen copy with
        updated lifecycle fields.
    """

    # ── Step 1: Terminal guard ─────────────────────────────────────────────────
    if zone.lifecycle_state == OBZoneLifecycle.INVALIDATED:
        return zone

    # ── Step 2: Invalidation (all non-INVALIDATED states, including EXPIRED) ───
    if zone.direction == OBZoneDirection.BULLISH:
        if candle_close <= zone.ob_low:
            return zone.model_copy(update={
                "lifecycle_state": OBZoneLifecycle.INVALIDATED,
                "invalidation_timestamp": datetime.now(tz=timezone.utc),
            })
    else:  # BEARISH
        if candle_close >= zone.ob_high:
            return zone.model_copy(update={
                "lifecycle_state": OBZoneLifecycle.INVALIDATED,
                "invalidation_timestamp": datetime.now(tz=timezone.utc),
            })

    # ── Step 3: Expiry (ACTIVE only) ──────────────────────────────────────────
    if zone.lifecycle_state == OBZoneLifecycle.ACTIVE:
        if (candle_bar_index - creation_bar_index) > config.zone_expiry_bars:
            return zone.model_copy(update={"lifecycle_state": OBZoneLifecycle.EXPIRED})

    # ── Step 4: Test transition (ACTIVE → TESTED) ─────────────────────────────
    if zone.lifecycle_state == OBZoneLifecycle.ACTIVE:
        if zone.direction == OBZoneDirection.BULLISH:
            # Candle low enters the zone from above; close holds above ob_low
            if zone.ob_low <= candle_low <= zone.ob_high and candle_close > zone.ob_low:
                return zone.model_copy(update={
                    "lifecycle_state": OBZoneLifecycle.TESTED,
                    "retest_count": 1,
                })
        else:  # BEARISH
            # Candle high enters the zone from below; close holds below ob_high
            if zone.ob_low <= candle_high <= zone.ob_high and candle_close < zone.ob_high:
                return zone.model_copy(update={
                    "lifecycle_state": OBZoneLifecycle.TESTED,
                    "retest_count": 1,
                })

    # ── Step 5: Retest transition (TESTED/RETESTED → RETESTED) ───────────────
    if zone.lifecycle_state in (OBZoneLifecycle.TESTED, OBZoneLifecycle.RETESTED):
        if zone.direction == OBZoneDirection.BULLISH:
            if zone.ob_low <= candle_low <= zone.ob_high and candle_close > zone.ob_low:
                return zone.model_copy(update={
                    "lifecycle_state": OBZoneLifecycle.RETESTED,
                    "retest_count": zone.retest_count + 1,
                })
        else:  # BEARISH
            if zone.ob_low <= candle_high <= zone.ob_high and candle_close < zone.ob_high:
                return zone.model_copy(update={
                    "lifecycle_state": OBZoneLifecycle.RETESTED,
                    "retest_count": zone.retest_count + 1,
                })

    # ── Default: no transition ────────────────────────────────────────────────
    return zone
