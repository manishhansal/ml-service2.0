"""
OrderBlockDetector — identifies candidate order block candles in an OHLCV series.

An order block (OB) is the last opposing candle immediately before an impulsive
directional move, representing an institutional entry zone.

Detection algorithm
-------------------
1. Guard: return [] for tiny or all-NaN inputs.
2. Compute ATR14 using true range and a **trailing** rolling window (never center=True).
3. Identify impulse bars: candles whose body exceeds ``config.impulse_multiplier × ATR``.
4. For each impulse bar j, scan backward from j-1 to max(0, j-lookback_bars) and take
   the **first** (most-recent) opposing candle as the order block.
5. Skip any bar where open/high/low/close contains NaN.
6. Deduplicate: if two impulse bars claim the same OB candle, keep the one with the
   larger impulse body (highest conviction).
"""
from __future__ import annotations

import pandas as pd

from src.scanners.order_block.config import OrderBlockConfig
from src.scanners.order_block.schemas import OBCandidate, OBZoneDirection

# OHLC columns checked for NaN guards
_OHLC_COLS = ["open", "high", "low", "close"]


def detect_candidates(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: OrderBlockConfig,
) -> list[OBCandidate]:
    """
    Detect order block candidates from an OHLCV DataFrame.

    Parameters
    ----------
    df:
        DataFrame with columns ``open``, ``high``, ``low``, ``close``, ``volume``
        and a timezone-aware UTC DatetimeIndex.
    symbol:
        Ticker symbol (propagated verbatim to all returned candidates).
    timeframe:
        Timeframe string — e.g. ``"1h"``, ``"5m"`` (propagated verbatim).
    config:
        Scanner parameters controlling ATR window, impulse multiplier, and
        lookback window.

    Returns
    -------
    list[OBCandidate]
        Deduplicated list of confirmed order block candidates.  Each element
        maps uniquely to one OB candle index; ties are resolved in favour of the
        impulse bar with the larger body size (highest conviction).
    """
    # ── Guard: minimum bar count ───────────────────────────────────────────────
    if len(df) < 15:
        return []

    # ── Guard: entirely empty OHLC data ───────────────────────────────────────
    if df[_OHLC_COLS].isna().all().all():
        return []

    # ── Extract column series ─────────────────────────────────────────────────
    high = df["high"]
    low = df["low"]
    close = df["close"]
    open_price = df["open"]

    # ── ATR14 — strictly backward-looking (never center=True) ────────────────
    # True range: TR = max(H-L, |H-prev_C|, |L-prev_C|)
    # .shift(1) makes prev_close lag by 1 bar — no forward leakage.
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=config.atr_window, min_periods=1).mean()

    # ── Impulse masks ─────────────────────────────────────────────────────────
    body_size = (close - open_price).abs()
    impulse_bull = (body_size > config.impulse_multiplier * atr) & (close > open_price)
    impulse_bear = (body_size > config.impulse_multiplier * atr) & (close < open_price)

    n = len(df)

    # ob_bar_index → (OBCandidate, conviction = body_size at j)
    best: dict[int, tuple[OBCandidate, float]] = {}

    for j in range(1, n):
        is_bull_impulse = bool(impulse_bull.iloc[j])
        is_bear_impulse = bool(impulse_bear.iloc[j])

        if not is_bull_impulse and not is_bear_impulse:
            continue

        # NaN guard for the impulse bar itself
        if not pd.notna(df.iloc[j][_OHLC_COLS]).all():
            continue

        body_j = float(body_size.iloc[j])

        # Backward scan: j-1 down to max(0, j - lookback_bars), inclusive.
        # The first opposing candle found is the most-recent (last) one.
        scan_start = j - 1
        scan_end_excl = max(0, j - config.lookback_bars) - 1  # range() exclusive bound

        ob_idx: int | None = None
        direction: OBZoneDirection = OBZoneDirection.BULLISH  # placeholder; set when found

        for i in range(scan_start, scan_end_excl, -1):
            # NaN guard for the potential OB candle
            if not pd.notna(df.iloc[i][_OHLC_COLS]).all():
                continue

            o_i = float(df.iloc[i]["open"])
            c_i = float(df.iloc[i]["close"])

            if is_bull_impulse:
                # Bullish impulse: OB is the last **bearish** candle (close < open)
                if c_i < o_i:
                    ob_idx = i
                    direction = OBZoneDirection.BULLISH
                    break
            else:
                # Bearish impulse: OB is the last **bullish** candle (close > open)
                if c_i > o_i:
                    ob_idx = i
                    direction = OBZoneDirection.BEARISH
                    break

        if ob_idx is None:
            continue

        # Deduplication: if this OB candle was already claimed by another impulse bar,
        # only keep the entry with the larger body_j (higher conviction).
        if ob_idx in best and body_j <= best[ob_idx][1]:
            continue

        row_ob = df.iloc[ob_idx]
        candidate = OBCandidate(
            symbol=symbol,
            ob_bar_index=ob_idx,
            ob_timestamp=df.index[ob_idx],
            direction=direction,
            ob_high=float(row_ob["high"]),
            ob_low=float(row_ob["low"]),
            ob_open=float(row_ob["open"]),
            ob_close=float(row_ob["close"]),
            ob_volume=float(row_ob["volume"]),
            impulse_bar_index=j,
            impulse_atr=float(atr.iloc[j]),
            timeframe=timeframe,
        )
        best[ob_idx] = (candidate, body_j)

    return [entry[0] for entry in best.values()]
