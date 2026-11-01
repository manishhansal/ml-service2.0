"""
SDZoneManager — lifecycle state machine for SDZone objects.

State machine:
  FRESH → TESTED (zone_strength 0.7): candle enters zone without closing through it
  TESTED → USED (zone_strength 0.4): second entry without closing through
  Any → BROKEN (zone_strength 0.0): close beyond far side of zone
  BROKEN: terminal — no further transitions

"Far side" rules:
  DEMAND zone: close < zone_low → BROKEN
  SUPPLY zone: close > zone_high → BROKEN

"Enters zone" (without breaking):
  candle_low <= zone_high AND candle_high >= zone_low
  AND NOT (closing through far side)

All mutations use .model_copy(update={...}) — SDZone is frozen=True.
"""
from __future__ import annotations
from datetime import datetime, timezone
from src.scanners.supply_demand.config import SDConfig
from src.scanners.supply_demand.schemas import SDDirection, SDLifecycle, SDZone


def update_sd_lifecycle(
    zone: SDZone,
    candle_low: float,
    candle_high: float,
    candle_close: float,
    bar_index: int,
    config: SDConfig,
) -> SDZone:
    """
    Advance zone through lifecycle state machine based on incoming candle.

    Parameters
    ----------
    zone: SDZone in its current lifecycle state
    candle_low, candle_high, candle_close: OHLC values for incoming candle
    bar_index: current bar index (reserved for future use)
    config: SDConfig (reserved for future use)

    Returns
    -------
    SDZone with updated lifecycle_state and zone_strength
    """
    # Terminal guard
    if zone.lifecycle_state == SDLifecycle.BROKEN:
        return zone

    # Check if candle closes through the far side of zone (BROKEN)
    if zone.direction == SDDirection.DEMAND:
        closes_through = candle_close < zone.zone_low
    else:  # SUPPLY
        closes_through = candle_close > zone.zone_high

    if closes_through:
        return zone.model_copy(update={
            "lifecycle_state": SDLifecycle.BROKEN,
            "zone_strength": 0.0,
            "broken_timestamp": datetime.now(tz=timezone.utc),
        })

    # Check if candle enters the zone (without closing through)
    candle_enters_zone = (candle_low <= zone.zone_high) and (candle_high >= zone.zone_low)

    if not candle_enters_zone:
        return zone

    # State transitions on zone entry
    if zone.lifecycle_state == SDLifecycle.FRESH:
        return zone.model_copy(update={
            "lifecycle_state": SDLifecycle.TESTED,
            "zone_strength": 0.7,
        })
    elif zone.lifecycle_state == SDLifecycle.TESTED:
        return zone.model_copy(update={
            "lifecycle_state": SDLifecycle.USED,
            "zone_strength": 0.4,
        })
    # USED stays USED on further entries (zone degrades but doesn't break)

    return zone
