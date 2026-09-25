"""
Market Structure Feature Family — Phase 3D.

Implements BOS, CHOCH, FVG, Order Blocks, and Liquidity Sweeps.

Causality rules (MANDATORY)
----------------------------
1. Fair Value Gaps:
   FVG at bar i is identified using bars [i-2, i-1, i] only.
   No future bars are needed. formation_time = bar i.

2. Order Blocks:
   An OB at bar i is the last opposing candle *before* an impulse at bar i.
   Known at bar i. No future confirmation required.

3. BOS / CHOCH:
   Swing levels are computed from trailing windows only (max/min of past bars).
   prev_swing_high = rolling_max(...).shift(lookback) — all past data.
   No centered rolling windows.

4. Liquidity Sweeps:
   Detected using trailing rolling high/low windows.

These implementations mirror the causal-corrected versions in
features/market_structure.py (which were fixed in Phase 3A).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .momentum import _require_utc_index, compute_atr


# ── Fair Value Gaps ───────────────────────────────────────────────────────────

def detect_fair_value_gaps(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    count_window: int = 10,
) -> dict[str, pd.Series]:
    """
    Detect Fair Value Gaps using only bars [i-2, i-1, i].

    Bullish FVG at bar i: low[i] > high[i-2]   (gap up imbalance)
    Bearish FVG at bar i: high[i] < low[i-2]   (gap down imbalance)

    The fill status of an FVG (did price return to close it?) is a FUTURE
    outcome and is NOT included here.  The feature captures the formation
    of the imbalance at bar i.

    Returns
    -------
    bullish_fvg_count : rolling `count_window`-bar sum of bullish FVGs
    bearish_fvg_count : rolling `count_window`-bar sum of bearish FVGs
    fvg_score         : net score in [-1, 1]
    """
    _require_utc_index(close, "close")
    n = len(close)
    h = high.astype(float)
    l = low.astype(float)

    bull_fvg = pd.Series(0.0, index=close.index)
    bear_fvg = pd.Series(0.0, index=close.index)

    # Vectorised: uses only bars i-2 and i — no future data
    bull_fvg.iloc[2:] = (l.iloc[2:].values > h.iloc[:-2].values).astype(float)
    bear_fvg.iloc[2:] = (h.iloc[2:].values < l.iloc[:-2].values).astype(float)

    bull_cnt = bull_fvg.rolling(window=count_window, min_periods=1).sum()
    bear_cnt = bear_fvg.rolling(window=count_window, min_periods=1).sum()
    total    = (bull_cnt + bear_cnt).replace(0, np.nan)
    fvg_score = (bull_cnt - bear_cnt) / total
    fvg_score = fvg_score.fillna(0.0)

    return {
        "bullish_fvg_count": bull_cnt,
        "bearish_fvg_count": bear_cnt,
        "fvg_score":         fvg_score,
    }


# ── Order Blocks ──────────────────────────────────────────────────────────────

def detect_order_blocks(
    open_price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 14,
    impulse_mult: float = 1.5,
    count_window: int = 10,
) -> dict[str, pd.Series]:
    """
    Detect Order Blocks (institutional entry zones).

    Bullish OB at bar i: bar i is a large bullish candle (move > 1.5×ATR)
        AND bar i-1 was bearish (close[i-1] < open[i-1]).
    Bearish OB at bar i: bar i is a large bearish candle AND bar i-1 was bullish.

    Detection is fully causal: uses only data at and before bar i.

    The mitigation state (did price return to the OB zone?) is a future
    outcome and is NOT included as a feature.

    Returns
    -------
    bullish_ob_count : rolling count of bullish OBs
    bearish_ob_count : rolling count of bearish OBs
    ob_score         : net score in [-1, 1]
    """
    _require_utc_index(close, "close")
    h   = high.astype(float)
    l   = low.astype(float)
    c   = close.astype(float)
    o   = open_price.astype(float)
    n   = len(c)

    atr = compute_atr(h, l, c, atr_period)

    bull_ob = pd.Series(0.0, index=c.index)
    bear_ob = pd.Series(0.0, index=c.index)

    for i in range(1, n):
        move      = c.iloc[i] - c.iloc[i - 1]
        threshold = impulse_mult * atr.iloc[i]
        if np.isnan(threshold):
            continue
        # Bullish impulse bar: big up move AND prior bar was bearish
        if move > threshold and c.iloc[i - 1] < o.iloc[i - 1]:
            bull_ob.iloc[i] = 1.0
        # Bearish impulse bar: big down move AND prior bar was bullish
        elif move < -threshold and c.iloc[i - 1] > o.iloc[i - 1]:
            bear_ob.iloc[i] = 1.0

    bull_cnt  = bull_ob.rolling(window=count_window, min_periods=1).sum()
    bear_cnt  = bear_ob.rolling(window=count_window, min_periods=1).sum()
    total     = (bull_cnt + bear_cnt).replace(0, np.nan)
    ob_score  = ((bull_cnt - bear_cnt) / total).fillna(0.0)

    return {
        "bullish_ob_count": bull_cnt,
        "bearish_ob_count": bear_cnt,
        "ob_score":         ob_score,
    }


# ── BOS / CHOCH ───────────────────────────────────────────────────────────────

def detect_bos_choch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 5,
    roll_window: int = 10,
) -> dict[str, pd.Series]:
    """
    Break of Structure (BOS) and Change of Character (CHOCH).

    Causal swing definitions
    ------------------------
    swing_high[t] = max(high[t - 2*lookback .. t])   (trailing window)
    swing_low[t]  = min(low[t - 2*lookback .. t])    (trailing window)
    prev_swing_high[t] = swing_high[t - lookback]    (reference level lookback bars ago)

    BOS (trend continuation):
        - Uptrend + close breaks above prev_swing_high → Bullish BOS
        - Downtrend + close breaks below prev_swing_low → Bearish BOS

    CHOCH (potential reversal):
        - Uptrend + close breaks below prev_swing_low → Bearish CHOCH
        - Downtrend + close breaks above prev_swing_high → Bullish CHOCH

    All swing levels use trailing windows.  The feature at bar t
    CANNOT be changed by any bar at t+1 or later.

    Returns
    -------
    bos_net        : rolling net BOS events
    choch_net      : rolling net CHOCH events
    structure_score: combined score in [-1, 1]
    """
    _require_utc_index(close, "close")
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)
    n = len(c)

    swing_win  = lookback * 2 + 1
    swing_high = h.rolling(window=swing_win, min_periods=lookback + 1).max()
    swing_low  = l.rolling(window=swing_win, min_periods=lookback + 1).min()

    # Reference levels lookback bars ago
    prev_sh = swing_high.shift(lookback)
    prev_sl = swing_low.shift(lookback)

    # Local trend: compare recent highs/lows at start vs end of trailing window
    trend = pd.Series(0, index=c.index)
    for i in range(lookback * 2, n):
        recent_h = h.iloc[i - lookback : i]
        recent_l = l.iloc[i - lookback : i]
        if len(recent_h) < 2:
            continue
        if recent_h.iloc[-1] > recent_h.iloc[0]:
            trend.iloc[i] = 1
        elif recent_l.iloc[-1] < recent_l.iloc[0]:
            trend.iloc[i] = -1

    bos_bull   = pd.Series(0.0, index=c.index)
    bos_bear   = pd.Series(0.0, index=c.index)
    choch_bull = pd.Series(0.0, index=c.index)
    choch_bear = pd.Series(0.0, index=c.index)

    ph_arr = prev_sh.to_numpy()
    pl_arr = prev_sl.to_numpy()
    c_arr  = c.to_numpy()
    t_arr  = trend.to_numpy()

    for i in range(lookback * 2, n):
        ph = ph_arr[i]
        pl = pl_arr[i]
        cv = c_arr[i]
        tv = t_arr[i]
        if np.isnan(ph) or np.isnan(pl):
            continue
        if tv == 1 and cv > ph:
            bos_bull.iloc[i] = 1.0
        elif tv == -1 and cv < pl:
            bos_bear.iloc[i] = 1.0
        if tv == 1 and cv < pl:
            choch_bear.iloc[i] = 1.0
        elif tv == -1 and cv > ph:
            choch_bull.iloc[i] = 1.0

    bos_net   = bos_bull.rolling(roll_window).sum()   - bos_bear.rolling(roll_window).sum()
    choch_net = choch_bull.rolling(roll_window).sum() - choch_bear.rolling(roll_window).sum()

    max_ev       = float(roll_window)
    struct_score = ((bos_net * 0.6 + choch_net * 0.4) / max_ev).clip(-1.0, 1.0)

    return {
        "bos_net":         bos_net,
        "choch_net":       choch_net,
        "structure_score": struct_score,
    }


# ── Liquidity sweeps ──────────────────────────────────────────────────────────

def detect_liquidity_sweeps(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """
    Detect liquidity sweeps: price wicks beyond swing levels then closes back.

    +1 (Bullish sweep): bar's low dips below the trailing `lookback`-bar low
       but closes above it → trapped shorts, potential bullish reversal.
    -1 (Bearish sweep): bar's high exceeds the trailing `lookback`-bar high
       but closes below it → trapped longs, potential bearish reversal.
     0: no sweep.

    Trailing window excludes bar i itself via rolling(lookback).shift(1).
    """
    _require_utc_index(close, "close")
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)

    # Use prior N bars (exclude bar i itself)
    prior_high = h.rolling(window=lookback, min_periods=lookback).max().shift(1)
    prior_low  = l.rolling(window=lookback, min_periods=lookback).min().shift(1)

    bullish = (l < prior_low) & (c > prior_low)
    bearish = (h > prior_high) & (c < prior_high)

    result = pd.Series(0.0, index=c.index)
    result[bullish] =  1.0
    result[bearish] = -1.0
    return result
