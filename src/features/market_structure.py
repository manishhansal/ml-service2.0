"""
Market structure features for Smart Money Concepts (SMC) analysis.

Covers: Fair Value Gaps (FVG), Order Blocks, Break of Structure (BOS),
Change of Character (CHOCH), liquidity sweeps, and supply/demand zones.
"""

import numpy as np
import pandas as pd


def detect_fair_value_gaps(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> dict[str, pd.Series]:
    """
    Detect Fair Value Gaps (FVGs) — 3-candle imbalances where the middle
    candle's range doesn't overlap with the surrounding candles.

    Returns:
        bullish_fvg_count: rolling count of unfilled bullish FVGs (5-bar window)
        bearish_fvg_count: rolling count of unfilled bearish FVGs (5-bar window)
        fvg_score: net FVG score in [-1, 1]
    """
    n = len(close)
    bullish_fvg = pd.Series(0, index=close.index)
    bearish_fvg = pd.Series(0, index=close.index)

    for i in range(2, n):
        # Bullish FVG: candle[i]'s low > candle[i-2]'s high (gap up imbalance)
        if low.iloc[i] > high.iloc[i - 2]:
            bullish_fvg.iloc[i] = 1

        # Bearish FVG: candle[i]'s high < candle[i-2]'s low (gap down imbalance)
        if high.iloc[i] < low.iloc[i - 2]:
            bearish_fvg.iloc[i] = 1

    # Rolling count over last 10 bars
    window = 10
    bull_count = bullish_fvg.rolling(window=window, min_periods=1).sum()
    bear_count = bearish_fvg.rolling(window=window, min_periods=1).sum()

    total = bull_count + bear_count
    fvg_score = pd.Series(0.0, index=close.index)
    mask = total > 0
    fvg_score[mask] = (bull_count[mask] - bear_count[mask]) / total[mask]

    return {
        "bullish_fvg_count": bull_count,
        "bearish_fvg_count": bear_count,
        "fvg_score": fvg_score,
    }


def detect_order_blocks(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> dict[str, pd.Series]:
    """
    Detect Order Blocks (OBs) — the last opposing candle before an impulsive
    move (institutional entry zone).

    Bullish OB: last bearish candle before a strong bullish impulse
    Bearish OB: last bullish candle before a strong bearish impulse

    Returns:
        bullish_ob_count: rolling count of bullish OBs nearby
        bearish_ob_count: rolling count of bearish OBs nearby
        ob_score: net score in [-1, 1]
    """
    n = len(close)
    bullish_ob = pd.Series(0, index=close.index)
    bearish_ob = pd.Series(0, index=close.index)

    # Impulse threshold: move > 1.5 * ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=1).mean()

    for i in range(2, n):
        move = close.iloc[i] - close.iloc[i - 1]
        impulse_threshold = 1.5 * atr.iloc[i]

        # Bullish impulse (big green candle)
        if move > impulse_threshold:
            # Look back for the last bearish candle
            if close.iloc[i - 1] < open_price.iloc[i - 1]:
                bullish_ob.iloc[i] = 1

        # Bearish impulse (big red candle)
        if move < -impulse_threshold:
            # Look back for the last bullish candle
            if close.iloc[i - 1] > open_price.iloc[i - 1]:
                bearish_ob.iloc[i] = 1

    window = 10
    bull_count = bullish_ob.rolling(window=window, min_periods=1).sum()
    bear_count = bearish_ob.rolling(window=window, min_periods=1).sum()

    total = bull_count + bear_count
    ob_score = pd.Series(0.0, index=close.index)
    mask = total > 0
    ob_score[mask] = (bull_count[mask] - bear_count[mask]) / total[mask]

    return {
        "bullish_ob_count": bull_count,
        "bearish_ob_count": bear_count,
        "ob_score": ob_score,
    }


def detect_bos_choch(
    high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 5
) -> dict[str, pd.Series]:
    """
    Detect Break of Structure (BOS) and Change of Character (CHOCH).

    Causal definitions — every value at bar t depends ONLY on bars ≤ t
    ----------------------------------------------------------------
    Swing high at bar t:
        The highest high over the window [t - (lookback*2), t] (trailing).
        This is the most recent "significant high" a trader could observe
        at close of bar t without seeing any future data.

    Swing low at bar t:
        The lowest low over the window [t - (lookback*2), t] (trailing).

    BOS (Break of Structure):
        At bar t, if the local trend is UP (recent highs are rising) and
        close[t] exceeds the swing-high level recorded lookback bars ago,
        a bullish BOS is flagged.  Symmetric for DOWN trend / swing low.
        Interpretation: trend continuation — existing momentum confirmed.

    CHOCH (Change of Character):
        At bar t, if the local trend is UP but close[t] breaks BELOW the
        swing-low level recorded lookback bars ago, a bearish CHOCH is
        flagged (potential reversal).  Symmetric for DOWN trend.

    Why trailing windows are correct:
        Using a centered rolling window incorporates bars AFTER t into the
        "swing level" at t — those bars are unknowable at decision time.

    Leakage guarantee:
        After this fix, the feature value at position t cannot change if
        any bar at position > t is modified.  Tests in test_phase3a.py
        verify this invariant.

    Returns:
        bos_net:       rolling net bullish minus bearish BOS events (10-bar)
        choch_net:     rolling net bullish minus bearish CHOCH events (10-bar)
        structure_score: combined market structure score in [-1, 1]
    """
    n = len(close)
    bos_bull = pd.Series(0, index=close.index)
    bos_bear = pd.Series(0, index=close.index)
    choch_bull = pd.Series(0, index=close.index)
    choch_bear = pd.Series(0, index=close.index)

    # ── Causal swing levels ───────────────────────────────────────────────
    # Trailing window of (lookback*2 + 1) bars ending at bar t.
    # min_periods = lookback + 1 so early bars produce NaN rather than
    # a meaningless single-bar "swing".
    swing_window = lookback * 2 + 1
    swing_high = high.rolling(window=swing_window, min_periods=lookback + 1).max()
    swing_low  = low.rolling(window=swing_window,  min_periods=lookback + 1).min()

    # Reference level: the swing measured lookback bars ago (the level a
    # trade would have been watching as a key structural reference).
    prev_swing_high = swing_high.shift(lookback)
    prev_swing_low  = swing_low.shift(lookback)

    # ── Local trend direction ─────────────────────────────────────────────
    # Trend at bar t: compare the high/low at the START vs END of the
    # trailing lookback window — purely backward-looking.
    trend = pd.Series(0, index=close.index)
    for i in range(lookback * 2, n):
        recent_highs = high.iloc[i - lookback : i]
        recent_lows  = low.iloc[i - lookback : i]
        if len(recent_highs) > 0:
            if recent_highs.iloc[-1] > recent_highs.iloc[0]:
                trend.iloc[i] = 1   # Uptrend structure
            elif recent_lows.iloc[-1] < recent_lows.iloc[0]:
                trend.iloc[i] = -1  # Downtrend structure

    # ── BOS / CHOCH detection ─────────────────────────────────────────────
    prev_high_arr = prev_swing_high.to_numpy()
    prev_low_arr  = prev_swing_low.to_numpy()
    close_arr     = close.to_numpy()
    trend_arr     = trend.to_numpy()

    for i in range(lookback * 2, n):
        ph = prev_high_arr[i]
        pl = prev_low_arr[i]
        c  = close_arr[i]
        t  = trend_arr[i]

        if np.isnan(ph) or np.isnan(pl):
            continue

        # BOS: break in the direction of the trend
        if t == 1 and c > ph:
            bos_bull.iloc[i] = 1
        elif t == -1 and c < pl:
            bos_bear.iloc[i] = 1

        # CHOCH: break AGAINST the trend (potential reversal)
        if t == 1 and c < pl:
            choch_bear.iloc[i] = 1
        elif t == -1 and c > ph:
            choch_bull.iloc[i] = 1

    window = 10
    bos_net   = bos_bull.rolling(window).sum()   - bos_bear.rolling(window).sum()
    choch_net = choch_bull.rolling(window).sum()  - choch_bear.rolling(window).sum()

    # Structure score: BOS confirms trend, CHOCH signals reversal
    max_events    = float(window)
    structure_score = ((bos_net * 0.6 + choch_net * 0.4) / max_events).clip(-1, 1)

    return {
        "bos_net":        bos_net,
        "choch_net":      choch_net,
        "structure_score": structure_score,
    }


def detect_liquidity_sweeps(
    high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 20
) -> pd.Series:
    """
    Detect liquidity sweeps: price wicks beyond swing levels then closes back.
    Positive: bullish sweep (dipped below support then closed above → trapped shorts)
    Negative: bearish sweep (spiked above resistance then closed below → trapped longs)
    """
    n = len(close)
    sweep_score = pd.Series(0.0, index=close.index)

    for i in range(lookback, n):
        window_high = high.iloc[i - lookback:i].max()
        window_low = low.iloc[i - lookback:i].min()

        # Bullish sweep: low went below prior support but close is above it
        if low.iloc[i] < window_low and close.iloc[i] > window_low:
            sweep_score.iloc[i] = 1.0

        # Bearish sweep: high went above prior resistance but close is below it
        elif high.iloc[i] > window_high and close.iloc[i] < window_high:
            sweep_score.iloc[i] = -1.0

    return sweep_score
