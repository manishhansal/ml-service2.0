"""
FVGScorer — computes scores for FVGZone objects and sorts them by combined_score descending.

Formulas:
  gap_size_score = (fvg_high - fvg_low) / ATR14[candle3_bar_index]
  volume_score   = volume[candle2_bar_index] / avg_volume_20[candle2_bar_index]
  combined_score = gap_size_score * volume_score

Sort order: primary descending combined_score, tie-break descending fvg_timestamp.
All zone metadata fields are preserved unchanged (only scores are added/updated).
"""
from __future__ import annotations
import math
import pandas as pd
from src.scanners.fair_value_gap.config import FVGConfig
from src.scanners.fair_value_gap.schemas import FVGZone


def rank_fvg_zones(
    zones: list[FVGZone],
    df_by_symbol: dict[str, pd.DataFrame],
    config: FVGConfig,
) -> list[FVGZone]:
    """
    Compute scores for each FVGZone and return sorted list.

    Parameters
    ----------
    zones: confirmed FVGZone objects to score
    df_by_symbol: mapping of symbol → OHLCV DataFrame
    config: FVGConfig with atr_window

    Returns
    -------
    list[FVGZone] sorted by combined_score descending (tie-break: fvg_timestamp descending)
    """
    if not zones:
        return []

    scored: list[FVGZone] = []

    for zone in zones:
        df = df_by_symbol.get(zone.symbol)

        if df is None or len(df) < 3:
            scored.append(zone.model_copy(update={
                "gap_size_score": 0.0,
                "volume_score": 0.0,
                "combined_score": 0.0,
            }))
            continue

        c2 = zone.candle1_bar_index + 1  # Candle_2 is c1 + 1
        c3 = zone.candle3_bar_index

        # Bounds check
        if c2 >= len(df) or c3 >= len(df):
            scored.append(zone.model_copy(update={
                "gap_size_score": 0.0,
                "volume_score": 0.0,
                "combined_score": 0.0,
            }))
            continue

        # ATR14 at candle3
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=config.atr_window, min_periods=1).mean()
        atr_c3 = float(atr.iloc[c3])

        # Gap size score
        gap_size = zone.fvg_high - zone.fvg_low
        if atr_c3 > 0:
            gap_size_score = gap_size / atr_c3
        else:
            gap_size_score = 0.0

        # Volume score: volume of candle2 / avg_volume_20 at candle2
        avg_vol = df["volume"].rolling(window=20, min_periods=1).mean()
        avg_vol_c2 = float(avg_vol.iloc[c2])
        vol_c2 = float(df["volume"].iloc[c2])

        if avg_vol_c2 > 0:
            volume_score = vol_c2 / avg_vol_c2
        else:
            volume_score = 0.0

        # Combined score
        combined_score = gap_size_score * volume_score

        scored.append(zone.model_copy(update={
            "gap_size_score": gap_size_score,
            "volume_score": volume_score,
            "combined_score": combined_score,
        }))

    # Sort: primary descending combined_score, tie-break descending fvg_timestamp
    return sorted(
        scored,
        key=lambda z: (-z.combined_score, -z.fvg_timestamp.timestamp()),
    )
