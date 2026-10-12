"""
SDScorer — computes scores for SDZone objects and sorts them by combined_score descending.

Formulas:
  departure_strength = |close[dep] - open[dep]| / ATR14[dep]
  zone_freshness     = zone.zone_strength (already set: 1.0/0.7/0.4/0.0)
  combined_score     = departure_strength * zone_freshness

Sort order: primary descending combined_score, tie-break descending departure_timestamp.
"""
from __future__ import annotations
import math
import pandas as pd
from src.scanners.supply_demand.config import SDConfig
from src.scanners.supply_demand.schemas import SDZone


def rank_sd_zones(
    zones: list[SDZone],
    df_by_symbol: dict[str, pd.DataFrame],
    config: SDConfig,
) -> list[SDZone]:
    """
    Compute departure_strength and combined_score for each SDZone and return sorted list.
    """
    if not zones:
        return []

    scored: list[SDZone] = []

    for zone in zones:
        df = df_by_symbol.get(zone.symbol)
        dep = zone.departure_bar_index

        if df is None or dep >= len(df):
            scored.append(zone.model_copy(update={
                "departure_strength": 0.0,
                "combined_score": 0.0,
            }))
            continue

        # ATR14 at departure bar
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=config.atr_window, min_periods=1).mean()
        atr_dep = float(atr.iloc[dep])

        # Departure strength
        dep_body = abs(float(df["close"].iloc[dep]) - float(df["open"].iloc[dep]))
        if atr_dep > 0:
            departure_strength = dep_body / atr_dep
        else:
            departure_strength = 0.0

        # Combined score = departure_strength × zone_freshness (zone.zone_strength)
        combined_score = departure_strength * zone.zone_strength

        scored.append(zone.model_copy(update={
            "departure_strength": departure_strength,
            "combined_score": combined_score,
        }))

    # Sort: primary descending combined_score, tie-break descending departure_timestamp
    return sorted(
        scored,
        key=lambda z: (-z.combined_score, -z.departure_timestamp.timestamp()),
    )
