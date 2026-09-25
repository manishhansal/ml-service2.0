"""
Volume & Liquidity Feature Family — Phase 3D.

Talib-free pure pandas/numpy implementations.
All functions return pd.Series with NaN for insufficient history.

VWAP semantics
--------------
For daily bars: rolling-window VWAP (trailing N bars).
For intraday bars: session-reset VWAP (resets at each calendar date).

The old cumsum()-based VWAP in technical.py was incorrect for daily bars
(accumulated from bar 0, not a session). This module provides the correct
rolling implementation.

Missing data policy
-------------------
No silent 0.0 substitution. NaN is propagated when data is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .momentum import _require_utc_index


# ── Relative volume ───────────────────────────────────────────────────────────

def compute_relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Today's volume / N-day rolling average volume.
    NaN for first `period - 1` bars or when avg_vol = 0.
    """
    _require_utc_index(volume, "volume")
    v   = volume.astype(float)
    avg = v.rolling(window=period, min_periods=period).mean()
    return v / avg.replace(0, np.nan)


def compute_volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    """Z-score of volume relative to its rolling mean/std."""
    _require_utc_index(volume, "volume")
    v   = volume.astype(float)
    mu  = v.rolling(window=period, min_periods=period).mean()
    sig = v.rolling(window=period, min_periods=period).std(ddof=1).replace(0, np.nan)
    return (v - mu) / sig


def compute_volume_breakout(
    volume: pd.Series,
    period: int = 20,
    threshold: float = 1.5,
) -> pd.Series:
    """Binary: 1.0 if volume >= threshold × avg, else 0.0. NaN when avg unavailable."""
    rel = compute_relative_volume(volume, period)
    return (rel >= threshold).astype(float).where(~rel.isna(), other=np.nan)


def compute_volume_trend(volume: pd.Series, period: int = 10) -> pd.Series:
    """
    Volume trend: recent N-bar avg / prior N-bar avg.
    NaN for first `2*period - 1` bars.
    """
    _require_utc_index(volume, "volume")
    v      = volume.astype(float)
    recent = v.rolling(window=period, min_periods=period).mean()
    prior  = v.shift(period).rolling(window=period, min_periods=period).mean()
    return recent / prior.replace(0, np.nan)


def compute_volume_price_confirmation(
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Volume-price confirmation:
    +1 = rising price + rising volume (trend confirmation)
    -1 = rising price + falling volume (divergence)
    0  = flat
    NaN when either series has insufficient history.
    """
    _require_utc_index(close, "close")
    price_dir = close.astype(float).diff(period).apply(
        lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
    )
    vol_avg = volume.astype(float).rolling(window=period, min_periods=period).mean()
    vol_dir = vol_avg.diff().apply(
        lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
    )
    result = price_dir * vol_dir
    # First period bars have no prior price diff
    result.iloc[:period] = np.nan
    return result


# ── VWAP ──────────────────────────────────────────────────────────────────────

def compute_vwap_distance_pct(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    period: int = 20,
    mode: str = "rolling",
) -> pd.Series:
    """
    % distance from VWAP: (close - vwap) / vwap × 100.

    Modes
    -----
    "rolling" (default, correct for daily bars):
        VWAP at bar t = sum(tp × vol, [t-N+1..t]) / sum(vol, [t-N+1..t]).
        Fully causal trailing window.

    "intraday" (correct for sub-daily bars):
        VWAP resets at each calendar date.
        Requires a DatetimeIndex with sub-daily frequency.
        Falls back to "rolling" if index date changes cannot be detected.

    The old cumsum() implementation in technical.py accumulated from bar 0
    of whatever window was passed in — that was semantically a multi-month
    VWAP, not a trading-session VWAP.  This function is the corrected version.
    """
    _require_utc_index(close, "close")
    tp     = (high.astype(float) + low.astype(float) + close.astype(float)) / 3.0
    tp_vol = tp * volume.astype(float)

    if mode == "intraday":
        dates = close.index.date
        date_series = pd.Series(dates, index=close.index)
        cum_tp_vol = tp_vol.groupby(date_series).cumsum()
        cum_vol    = volume.astype(float).groupby(date_series).cumsum()
        vwap       = cum_tp_vol / cum_vol.replace(0, np.nan)
    else:  # rolling (default, correct for daily bars)
        roll_tp_vol = tp_vol.rolling(window=period, min_periods=1).sum()
        roll_vol    = volume.astype(float).rolling(window=period, min_periods=1).sum()
        vwap        = roll_tp_vol / roll_vol.replace(0, np.nan)

    vwap_safe = vwap.replace(0, np.nan)
    return ((close.astype(float) - vwap_safe) / vwap_safe) * 100.0


# ── OBV / Accumulation ────────────────────────────────────────────────────────

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume (cumulative).
    OBV[t] += volume if close > prev_close else -= volume.
    """
    _require_utc_index(close, "close")
    c   = close.astype(float)
    v   = volume.astype(float)
    dir_v = pd.Series(
        np.where(c.diff() > 0, v, np.where(c.diff() < 0, -v, 0.0)),
        index=c.index,
    )
    return dir_v.cumsum()


def compute_obv_zscore(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """
    OBV z-score: (OBV - rolling_mean_OBV) / rolling_std_OBV.
    More cross-stock comparable than raw OBV.
    """
    _require_utc_index(close, "close")
    obv = compute_obv(close, volume)
    mu  = obv.rolling(window=period, min_periods=period).mean()
    sig = obv.rolling(window=period, min_periods=period).std(ddof=1).replace(0, np.nan)
    return (obv - mu) / sig


def compute_cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Chaikin Money Flow [-1, 1].
    CMF = sum(MF_vol, N) / sum(vol, N)
    """
    _require_utc_index(close, "close")
    hl_range     = (high.astype(float) - low.astype(float)).replace(0, np.nan)
    mf_mult      = ((close.astype(float) - low.astype(float))
                    - (high.astype(float) - close.astype(float))) / hl_range
    mf_vol       = mf_mult * volume.astype(float)
    return (mf_vol.rolling(window=period, min_periods=period).sum()
            / volume.astype(float).rolling(window=period, min_periods=period).sum().replace(0, np.nan))


def compute_accumulation_distribution(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Accumulation/Distribution Line (cumulative)."""
    _require_utc_index(close, "close")
    hl_range = (high.astype(float) - low.astype(float)).replace(0, np.nan)
    mf_mult  = ((close.astype(float) - low.astype(float))
                - (high.astype(float) - close.astype(float))) / hl_range
    # fillna(0) here is economically defensible: HL range=0 (doji) → no CLV
    mf_mult  = mf_mult.fillna(0.0)
    return (mf_mult * volume.astype(float)).cumsum()


def compute_force_index(
    close: pd.Series,
    volume: pd.Series,
    period: int = 13,
) -> pd.Series:
    """
    Elder's Force Index: price_change × volume, smoothed over `period` bars.
    """
    _require_utc_index(close, "close")
    fi = close.astype(float).diff() * volume.astype(float)
    return fi.ewm(span=period, adjust=False).mean()


def compute_volume_oscillator(
    volume: pd.Series,
    fast: int = 5,
    slow: int = 20,
) -> pd.Series:
    """
    Volume Oscillator: (EMA_fast - EMA_slow) / EMA_slow × 100.
    NaN while EMA_slow is warming up.
    """
    _require_utc_index(volume, "volume")
    v      = volume.astype(float)
    fast_e = v.ewm(span=fast, adjust=False).mean()
    slow_e = v.ewm(span=slow, adjust=False).mean()
    return ((fast_e - slow_e) / slow_e.replace(0, np.nan)) * 100.0


# ── Liquidity proxies ─────────────────────────────────────────────────────────

def compute_amihud_illiquidity(
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Amihud (2002) illiquidity proxy: rolling average of |return| / dollar_volume.

    High values → illiquid (price moves a lot per unit of trading).
    Low values → liquid.

    This is a PROXY for true price impact; actual spread data is needed
    for a true measure.  Labeled as PROXY in the feature registry.

    dollar_volume = close × volume (approximate INR turnover).
    """
    _require_utc_index(close, "close")
    c   = close.astype(float)
    v   = volume.astype(float)
    ret = c.pct_change().abs()
    dvol = c * v  # INR proxy
    raw  = ret / dvol.replace(0, np.nan)
    return raw.rolling(window=period, min_periods=period).mean()


def compute_volume_profile_score(
    close: pd.Series,
    volume: pd.Series,
    high: pd.Series,
    low: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """
    Position of current close relative to volume-weighted price distribution.
    +1: at top of high-volume zone, -1: at bottom.
    Purely trailing; NaN for first `lookback` bars.
    """
    _require_utc_index(close, "close")
    n      = len(close)
    result = np.full(n, np.nan)
    c_arr  = close.astype(float).to_numpy()
    v_arr  = volume.astype(float).to_numpy()
    h_arr  = high.astype(float).to_numpy()
    l_arr  = low.astype(float).to_numpy()

    for i in range(lookback, n):
        wc = c_arr[i - lookback : i]
        wv = v_arr[i - lookback : i]
        wh = h_arr[i - lookback : i]
        wl = l_arr[i - lookback : i]
        total_v = wv.sum()
        if total_v <= 0:
            continue
        vwap_w    = (wc * wv).sum() / total_v
        low_min   = wl.min()
        high_max  = wh.max()
        price_rng = high_max - low_min
        if price_rng <= 0:
            continue
        position      = (c_arr[i] - low_min) / price_rng
        vwap_position = (vwap_w   - low_min) / price_rng
        result[i]     = float(np.clip(2.0 * (position - vwap_position), -1.0, 1.0))

    return pd.Series(result, index=close.index, dtype=float)
