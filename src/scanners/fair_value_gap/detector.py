"""
FVGDetector — identifies 3-candle Fair Value Gap imbalance zones.

A Fair Value Gap (FVG) is a 3-candle pattern where the middle candle's move is
so impulsive that the ranges of Candle_1 and Candle_3 do not overlap:
- Bullish FVG: low[i+2] > high[i] AND close[i+2] > high[i]
- Bearish FVG: high[i+2] < low[i] AND close[i+2] < low[i]

All ATR computations use trailing rolling windows (min_periods=1, never center=True).
"""
from __future__ import annotations
import math
from datetime import datetime
import pandas as pd
from src.scanners.fair_value_gap.config import FVGConfig
from src.scanners.fair_value_gap.schemas import FVGDirection, FVGZone

_OHLC_COLS = ["open", "high", "low", "close"]


def detect_fvg_zones(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: FVGConfig,
) -> list[FVGZone]:
    """Detect Fair Value Gap zones in an OHLCV series."""

    # Guard: need at least 3 bars
    if len(df) < 3:
        return []
    # Guard: all-NaN OHLC
    if df[_OHLC_COLS].isna().all().all():
        return []

    # Compute ATR14 (trailing, never center=True)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=config.atr_window, min_periods=1).mean()

    # Compute avg_volume_20
    avg_vol = df["volume"].rolling(window=20, min_periods=1).mean()

    results: list[FVGZone] = []

    for i in range(len(df) - 2):
        c1, c2, c3 = i, i + 1, i + 2

        # NaN guard: skip if any OHLCV is NaN at c1, c2, or c3
        if not pd.notna(df.iloc[c1][_OHLC_COLS]).all():
            continue
        if not pd.notna(df.iloc[c2][_OHLC_COLS]).all():
            continue
        if not pd.notna(df.iloc[c3][_OHLC_COLS]).all():
            continue

        # Impulse candle filter (Candle_2 body must exceed N × ATR14)
        body2 = abs(float(df["close"].iloc[c2]) - float(df["open"].iloc[c2]))
        atr_c2 = float(atr.iloc[c2])
        if body2 < config.impulse_atr_multiplier * atr_c2:
            continue

        high_c1 = float(df["high"].iloc[c1])
        low_c1  = float(df["low"].iloc[c1])
        low_c3  = float(df["low"].iloc[c3])
        high_c3 = float(df["high"].iloc[c3])
        close_c3 = float(df["close"].iloc[c3])
        atr_c3  = float(atr.iloc[c3])

        direction: FVGDirection | None = None
        fvg_high: float = 0.0
        fvg_low: float  = 0.0

        # Bullish FVG: low[c3] > high[c1] AND close[c3] > high[c1]
        if low_c3 > high_c1 and close_c3 > high_c1:
            fvg_high = low_c3
            fvg_low  = high_c1
            direction = FVGDirection.BULLISH
        # Bearish FVG: high[c3] < low[c1] AND close[c3] < low[c1]
        elif high_c3 < low_c1 and close_c3 < low_c1:
            fvg_high = low_c1
            fvg_low  = high_c3
            direction = FVGDirection.BEARISH
        else:
            continue

        # Gap size filter
        gap_size = fvg_high - fvg_low
        if gap_size < config.min_gap_atr_multiplier * atr_c3:
            continue

        zone = FVGZone(
            symbol=symbol,
            direction=direction,
            timeframe=timeframe,
            fvg_high=fvg_high,
            fvg_low=fvg_low,
            fvg_midpoint=(fvg_high + fvg_low) / 2.0,
            fvg_timestamp=df.index[c3],
            candle1_bar_index=c1,
            candle3_bar_index=c3,
            impulse_atr=atr_c2,
        )
        results.append(zone)

    return results
