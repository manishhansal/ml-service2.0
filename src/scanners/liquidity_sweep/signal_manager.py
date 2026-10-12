"""
SweepSignalManager — lifecycle state machine for SweepSignal objects.

State machine:
  ACTIVE → CONFIRMED: next close continues in the reversal direction
  ACTIVE → INVALIDATED: close crosses back through the swept_level
  CONFIRMED → INVALIDATED: subsequent close crosses the swept_level
  INVALIDATED: terminal — no further transitions

All mutations use .model_copy(update={...}) — SweepSignal is frozen=True.
"""
from __future__ import annotations
from datetime import datetime, timezone
from src.scanners.liquidity_sweep.config import SweepConfig
from src.scanners.liquidity_sweep.schemas import SweepDirection, SweepLifecycle, SweepSignal


def update_sweep_lifecycle(
    signal: SweepSignal,
    next_candle_close: float,
    bar_index: int,
    config: SweepConfig,
) -> SweepSignal:
    """
    Advance signal through its lifecycle state machine.
    
    Parameters
    ----------
    signal: SweepSignal in its current lifecycle state
    next_candle_close: close price of the next candle after the sweep
    bar_index: absolute bar index of the next candle
    config: SweepConfig (reserved for future extension)
    
    Returns
    -------
    SweepSignal with updated lifecycle_state
    """
    # Terminal guard
    if signal.lifecycle_state == SweepLifecycle.INVALIDATED:
        return signal
    
    swept = signal.swept_level
    
    if signal.direction == SweepDirection.BULLISH:
        # Bullish sweep: swept below, close recovered above
        # Invalidated if close drops back below swept_level
        if next_candle_close < swept:
            return signal.model_copy(update={
                "lifecycle_state": SweepLifecycle.INVALIDATED,
            })
        # Confirmed if still above swept level (reversal continues)
        if signal.lifecycle_state == SweepLifecycle.ACTIVE:
            if next_candle_close > swept:
                return signal.model_copy(update={
                    "lifecycle_state": SweepLifecycle.CONFIRMED,
                    "confirmed_timestamp": datetime.now(tz=timezone.utc),
                })
    
    else:  # BEARISH sweep
        # Bearish sweep: swept above, close recovered below
        # Invalidated if close pops back above swept_level
        if next_candle_close > swept:
            return signal.model_copy(update={
                "lifecycle_state": SweepLifecycle.INVALIDATED,
            })
        # Confirmed if still below swept level
        if signal.lifecycle_state == SweepLifecycle.ACTIVE:
            if next_candle_close < swept:
                return signal.model_copy(update={
                    "lifecycle_state": SweepLifecycle.CONFIRMED,
                    "confirmed_timestamp": datetime.now(tz=timezone.utc),
                })
    
    return signal
