"""
FVGZoneManager — lifecycle state machine for FVGZone objects.

State machine:
  UNFILLED → PARTIALLY_FILLED: candle's price range intersects the FVG zone
  PARTIALLY_FILLED → FILLED: candle closes on the far side of the midpoint
  UNFILLED → EXPIRED: bar_age > fvg_expiry_bars without any interaction
  FILLED, EXPIRED: terminal states (no further transitions)
  
All mutations use .model_copy(update={...}) — FVGZone is frozen=True.
"""
from __future__ import annotations
from datetime import datetime, timezone
from src.scanners.fair_value_gap.config import FVGConfig
from src.scanners.fair_value_gap.schemas import FVGDirection, FVGLifecycle, FVGZone


def update_fvg_lifecycle(
    zone: FVGZone,
    candle_low: float,
    candle_high: float,
    candle_close: float,
    bar_index: int,
    creation_bar_index: int,
    config: FVGConfig,
) -> FVGZone:
    """
    Advance zone through its lifecycle state machine based on incoming candle.
    
    Parameters
    ----------
    zone: FVGZone
    candle_low, candle_high, candle_close: OHLC values for the incoming candle
    bar_index: current bar's position in the series
    creation_bar_index: bar index of candle3 (when zone was detectable)
    config: FVGConfig with fvg_expiry_bars
    
    Returns
    -------
    FVGZone with updated lifecycle_state and fill_timestamp as needed
    """
    # Terminal guard — FILLED and EXPIRED never change
    if zone.lifecycle_state in (FVGLifecycle.FILLED, FVGLifecycle.EXPIRED):
        return zone
    
    # Check if candle intersects the FVG zone
    candle_enters_zone = (candle_low <= zone.fvg_high) and (candle_high >= zone.fvg_low)
    
    if zone.lifecycle_state == FVGLifecycle.UNFILLED:
        # Expiry check: no interaction and age > expiry threshold
        bar_age = bar_index - creation_bar_index
        if bar_age > config.fvg_expiry_bars and not candle_enters_zone:
            return zone.model_copy(update={"lifecycle_state": FVGLifecycle.EXPIRED})
        
        # If candle enters zone, check if it closes on far side of midpoint
        if candle_enters_zone:
            # Check for immediate fill (close crosses midpoint)
            crosses_midpoint = (
                (zone.direction == FVGDirection.BULLISH and candle_close < zone.fvg_midpoint)
                or (zone.direction == FVGDirection.BEARISH and candle_close > zone.fvg_midpoint)
            )
            if crosses_midpoint:
                return zone.model_copy(update={
                    "lifecycle_state": FVGLifecycle.FILLED,
                    "fill_timestamp": datetime.now(tz=timezone.utc),
                })
            else:
                return zone.model_copy(update={"lifecycle_state": FVGLifecycle.PARTIALLY_FILLED})
    
    elif zone.lifecycle_state == FVGLifecycle.PARTIALLY_FILLED:
        # PARTIALLY_FILLED: check if close crosses midpoint → FILLED
        if candle_enters_zone:
            crosses_midpoint = (
                (zone.direction == FVGDirection.BULLISH and candle_close < zone.fvg_midpoint)
                or (zone.direction == FVGDirection.BEARISH and candle_close > zone.fvg_midpoint)
            )
            if crosses_midpoint:
                return zone.model_copy(update={
                    "lifecycle_state": FVGLifecycle.FILLED,
                    "fill_timestamp": datetime.now(tz=timezone.utc),
                })
    
    return zone
