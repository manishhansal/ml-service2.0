"""
SweepScorer — computes scores for SweepSignal objects and sorts them by conviction_score descending.

Formulas:
  sweep_magnitude  = abs(wick_extreme - swept_level) / ATR14[sweep_bar_index]
  volume_ratio     = volume[sweep_bar_index] / avg_volume_20[sweep_bar_index]
  conviction_score = sweep_magnitude * volume_ratio * equal_levels_count

For a BULLISH sweep, wick_extreme = low[sweep_bar_index] (the low that swept below the level).
For a BEARISH sweep, wick_extreme = high[sweep_bar_index].

Sort order: primary descending conviction_score, tie-break descending sweep_timestamp.
"""
from __future__ import annotations
import math
import pandas as pd
from src.scanners.liquidity_sweep.config import SweepConfig
from src.scanners.liquidity_sweep.schemas import SweepDirection, SweepSignal


def rank_sweeps(
    signals: list[SweepSignal],
    config: SweepConfig,
    df_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> list[SweepSignal]:
    """
    Compute conviction scores for each SweepSignal and return sorted list.
    
    If df_by_symbol is None (or signal symbol not found), use existing signal 
    scores as-is and just sort.
    """
    if not signals:
        return []

    scored: list[SweepSignal] = []

    for signal in signals:
        df = (df_by_symbol or {}).get(signal.symbol) if df_by_symbol else None
        k = signal.sweep_bar_index

        if df is None or k >= len(df):
            # No DataFrame available: recompute conviction from existing fields
            conviction = signal.sweep_magnitude * signal.volume_ratio * signal.equal_levels_count
            scored.append(signal.model_copy(update={"conviction_score": conviction}))
            continue

        # ATR14 at sweep bar
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=config.atr_window, min_periods=1).mean()
        atr_k = float(atr.iloc[k])

        # Sweep magnitude
        if signal.direction == SweepDirection.BULLISH:
            wick_extreme = float(df["low"].iloc[k])
        else:
            wick_extreme = float(df["high"].iloc[k])

        if atr_k > 0:
            sweep_magnitude = abs(wick_extreme - signal.swept_level) / atr_k
        else:
            sweep_magnitude = 0.0

        # Volume ratio
        avg_vol = df["volume"].rolling(window=config.avg_volume_window, min_periods=1).mean()
        avg_vol_k = float(avg_vol.iloc[k])
        vol_k     = float(df["volume"].iloc[k])
        
        if avg_vol_k > 0:
            volume_ratio = vol_k / avg_vol_k
        else:
            volume_ratio = 0.0

        # Conviction score
        conviction_score = sweep_magnitude * volume_ratio * signal.equal_levels_count

        scored.append(signal.model_copy(update={
            "sweep_magnitude": sweep_magnitude,
            "volume_ratio": volume_ratio,
            "conviction_score": conviction_score,
        }))

    # Sort: primary descending conviction_score, tie-break descending sweep_timestamp
    return sorted(
        scored,
        key=lambda s: (-s.conviction_score, -s.sweep_timestamp.timestamp()),
    )
