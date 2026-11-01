"""
StructureValidator — confirms order block candidates via Break of Structure (BoS).

For each ``OBCandidate``, the validator computes a trailing swing_high/swing_low
reference level at the OB bar ``i`` and scans the confirmation window
``[i+1, i + max_confirmation_bars]`` for the first bar whose *close* strictly
crosses that level.

No-lookahead guarantee (Req 9.4):
  ``swing_high[i]`` uses a trailing rolling window of ``2*swing_lookback+1`` bars
  ending at bar ``i``.  The window never extends beyond bar ``i``, so modifying any
  bar at index ``> i`` cannot alter the reference level at ``i``.

BoS rule (Req 2.3):
  Only the *close* counts — a wick that penetrates the reference level but closes
  on the wrong side does NOT constitute a BoS.
"""
from __future__ import annotations

import math

import pandas as pd

from src.scanners.order_block.config import OrderBlockConfig
from src.scanners.order_block.schemas import (
    OBCandidate,
    OBCandidateConfirmed,
    OBZoneDirection,
)


def validate_structure(
    candidates: list[OBCandidate],
    df: pd.DataFrame,
    config: OrderBlockConfig,
) -> list[OBCandidateConfirmed]:
    """
    Validate each ``OBCandidate`` by checking for a close-based Break of Structure.

    Parameters
    ----------
    candidates:
        Unconfirmed order block candidates from ``OrderBlockDetector``.
    df:
        The same OHLCV ``DataFrame`` used to detect the candidates.  Must have
        columns ``open``, ``high``, ``low``, ``close``, ``volume`` and a
        UTC-aware ``DatetimeIndex``.
    config:
        Scanner parameters.  The active ``max_confirmation_bars`` value is read
        directly from *config*; the caller (``ObScanner``) is responsible for
        applying per-timeframe overrides before invoking this function.

    Returns
    -------
    list[OBCandidateConfirmed]
        Candidates confirmed by a close-based BoS within the confirmation window.
        Candidates for which no BoS is found within the window are treated as
        **expired** and excluded from the returned list.
    """
    if not candidates:
        return []

    # ── Step 1: Compute trailing swing_high and swing_low ONCE ───────────────
    #
    # A window of ``2 * swing_lookback + 1`` trailing bars centred at the
    # *right* edge gives the structural swing level including bar i itself.
    # This is the "Swing_High computed over bars [i - lookback, i]" from Req 2.1.
    # ``min_periods = swing_lookback + 1`` ensures the series is NaN for bars
    # that don't have enough history.
    swing_window = config.swing_lookback * 2 + 1
    min_periods = config.swing_lookback + 1

    # NEVER center=True — purely backward-looking (Req 2.4, 9.4)
    swing_high = df["high"].rolling(window=swing_window, min_periods=min_periods).max()
    swing_low = df["low"].rolling(window=swing_window, min_periods=min_periods).min()

    # ── Step 2: Process each candidate ───────────────────────────────────────
    confirmed: list[OBCandidateConfirmed] = []
    n = len(df)

    for cand in candidates:
        i = cand.ob_bar_index

        # Derive reference level at bar i (backward-looking; bar i is the
        # right edge of the rolling window, so only bars ≤ i are used).
        if cand.direction == OBZoneDirection.BULLISH:
            ref = float(swing_high.iloc[i])
        else:
            ref = float(swing_low.iloc[i])

        # Skip if insufficient history to compute a reference level
        if math.isnan(ref):
            continue

        # Confirmation window: [i+1, i + max_confirmation_bars] inclusive (Req 2.6)
        window_start = i + 1
        window_end = min(i + config.max_confirmation_bars, n - 1)

        if window_start > window_end:
            # OB candle is too close to the end of the series — treat as expired
            continue

        # Scan for the FIRST close-based BoS (wicks don't count — Req 2.3)
        bos_j: int | None = None
        for j in range(window_start, window_end + 1):
            close_j = df["close"].iloc[j]
            if pd.isna(close_j):
                continue  # skip NaN bars

            if cand.direction == OBZoneDirection.BULLISH:
                if float(close_j) > ref:
                    bos_j = j
                    break
            else:  # BEARISH
                if float(close_j) < ref:
                    bos_j = j
                    break

        if bos_j is None:
            # No BoS found within the window → expired, exclude (Req 2.6)
            continue

        # Build OBCandidateConfirmed, preserving all OBCandidate fields and
        # appending BoS evidence (Req 2.5)
        confirmed.append(
            OBCandidateConfirmed(
                symbol=cand.symbol,
                ob_bar_index=cand.ob_bar_index,
                ob_timestamp=cand.ob_timestamp,
                direction=cand.direction,
                ob_high=cand.ob_high,
                ob_low=cand.ob_low,
                ob_open=cand.ob_open,
                ob_close=cand.ob_close,
                ob_volume=cand.ob_volume,
                impulse_bar_index=cand.impulse_bar_index,
                impulse_atr=cand.impulse_atr,
                timeframe=cand.timeframe,
                bos_bar_index=bos_j,
                bos_timestamp=df.index[bos_j],
                bos_level=ref,
            )
        )

    return confirmed
