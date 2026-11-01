"""
MSScorer — computes scores for MSSignal objects and sorts them by conviction_score descending.

Formulas:
  break_magnitude  = abs(close_at_break - broken_level) / ATR14[break_bar_index]
  choch_multiplier = 1.5 if is_choch else 1.0
  conviction_score = break_magnitude * max(swing_count, 1) * choch_multiplier

Sort order: primary descending conviction_score, tie-break descending break_timestamp.
"""
from __future__ import annotations
import math
import pandas as pd
from src.scanners.market_structure.config import MSConfig
from src.scanners.market_structure.schemas import MSSignal


def rank_ms_signals(
    signals: list[MSSignal],
    config: MSConfig,
    df_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> list[MSSignal]:
    """
    Compute conviction scores for each MSSignal and return sorted list.

    If df_by_symbol is None (or symbol not found), use existing break_magnitude to
    recompute conviction_score from existing fields.
    """
    if not signals:
        return []

    scored: list[MSSignal] = []

    for signal in signals:
        df = (df_by_symbol or {}).get(signal.symbol) if df_by_symbol else None
        k = signal.break_bar_index
        choch_mult = 1.5 if signal.is_choch else 1.0
        swing_cnt = max(signal.swing_count, 1)

        if df is None or k >= len(df):
            # Recompute conviction from existing break_magnitude field
            conviction = signal.break_magnitude * swing_cnt * choch_mult
            scored.append(signal.model_copy(update={"conviction_score": conviction}))
            continue

        # ATR14 at break bar
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=config.atr_window, min_periods=1).mean()
        atr_k = float(atr.iloc[k])

        # Break magnitude
        if atr_k > 0 and not math.isnan(atr_k):
            break_magnitude = abs(signal.close_at_break - signal.broken_level) / atr_k
        else:
            break_magnitude = 0.0

        # Conviction score
        conviction_score = break_magnitude * swing_cnt * choch_mult

        scored.append(signal.model_copy(update={
            "break_magnitude": break_magnitude,
            "conviction_score": conviction_score,
        }))

    # Sort: primary descending conviction_score, tie-break descending break_timestamp
    return sorted(
        scored,
        key=lambda s: (-s.conviction_score, -s.break_timestamp.timestamp()),
    )
