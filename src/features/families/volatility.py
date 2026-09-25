"""
Volatility Feature Family — Phase 3D.

All estimators use only trailing data — no forward look.
All functions return pd.Series with NaN for insufficient history.

Causal guarantee
----------------
Every computation uses rolling(window=N, min_periods=N) so the first
N-1 bars return NaN rather than spurious values from short windows.

Talib-free: uses only numpy / pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .momentum import _require_utc_index, _require_positive, compute_atr


# ── Realised volatility estimators ───────────────────────────────────────────

def compute_realized_vol(
    close: pd.Series,
    period: int = 20,
    annualize: bool = True,
    trading_days: float = 252.0,
) -> pd.Series:
    """
    Close-to-close realised volatility: rolling std of log returns.

    Parameters
    ----------
    close         : Close price series with UTC DatetimeIndex.
    period        : Rolling window in bars.
    annualize     : If True, multiply by sqrt(trading_days).
    trading_days  : Annualisation factor (252 for daily).

    Returns
    -------
    pd.Series of floats in % (e.g. 25.0 = 25% annualised vol).
    NaN for first `period` bars.
    """
    _require_utc_index(close, "close")
    c       = _require_positive(close, "close")
    log_ret = np.log(c / c.shift(1))
    vol     = log_ret.rolling(window=period, min_periods=period).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(trading_days)
    return vol * 100.0  # express as %


def compute_parkinson_vol(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
    annualize: bool = True,
    trading_days: float = 252.0,
) -> pd.Series:
    """
    Parkinson (high-low) volatility estimator.

    vol² = (1 / 4 ln 2) × (ln H/L)²    per bar
    Rolling mean of per-bar variance → sqrt → annualise.

    More efficient than close-to-close for stable intraday processes.
    Underestimates true vol on days with large overnight gaps.
    """
    _require_utc_index(high, "high")
    h = _require_positive(high, "high")
    l = _require_positive(low, "low")
    ln_hl   = np.log(h / l.replace(0, np.nan))
    park_sq = ln_hl ** 2 / (4.0 * np.log(2.0))
    vol     = park_sq.rolling(window=period, min_periods=period).mean().apply(
        lambda x: np.sqrt(x) if x >= 0 else np.nan
    )
    if annualize:
        vol = vol * np.sqrt(trading_days)
    return vol * 100.0  # %


def compute_atr_pct(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    ATR(period) as percentage of close price.
    Cross-stock comparable (₹10 stock and ₹10000 stock on same scale).
    NaN when close <= 0.
    """
    _require_utc_index(close, "close")
    atr = compute_atr(high, low, close, period)
    c   = _require_positive(close, "close")
    return (atr / c) * 100.0


# ── Volatility regime ─────────────────────────────────────────────────────────

def compute_vol_percentile(
    close: pd.Series,
    vol_period: int = 20,
    pct_window: int = 60,
    annualize: bool = True,
) -> pd.Series:
    """
    Rolling percentile of realised volatility vs its own history.

    At bar t: what % of the past `pct_window` realised-vol observations
    are below the current realised vol?

    Uses only past data — never future volatility observations.
    Returns NaN until `vol_period + pct_window` bars are available.
    """
    _require_utc_index(close, "close")
    rv = compute_realized_vol(close, vol_period, annualize)
    result = np.full(len(rv), np.nan)
    arr    = rv.to_numpy(dtype=float)
    total  = vol_period + pct_window

    for i in range(total - 1, len(arr)):
        window = arr[i - pct_window + 1 : i + 1]
        if np.all(np.isnan(window)):
            continue
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        current = arr[i]
        if np.isnan(current):
            continue
        result[i] = float((valid < current).sum() / len(valid) * 100.0)

    return pd.Series(result, index=close.index, dtype=float)


def compute_vol_regime(
    close: pd.Series,
    vol_period: int = 20,
    pct_window: int = 60,
) -> pd.Series:
    """
    Volatility regime: 0=low, 1=normal, 2=high, 3=extreme.
    Based on rolling percentile:
        0: pct < 25
        1: 25 <= pct < 60
        2: 60 <= pct < 90
        3: pct >= 90

    Returns NaN when history is insufficient.
    NEVER returns a fabricated regime when VIX/vol data is absent.
    """
    pct = compute_vol_percentile(close, vol_period, pct_window)
    result = pd.Series(np.nan, index=close.index, dtype=float)
    valid  = ~pct.isna()
    result[valid & (pct < 25)]  = 0.0
    result[valid & (pct >= 25) & (pct < 60)] = 1.0
    result[valid & (pct >= 60) & (pct < 90)] = 2.0
    result[valid & (pct >= 90)] = 3.0
    return result


def compute_vol_zscore(
    close: pd.Series,
    vol_period: int = 20,
    zscore_window: int = 60,
) -> pd.Series:
    """
    Z-score of realised volatility vs its rolling mean/std.
    Measures how extreme current volatility is relative to recent history.
    """
    _require_utc_index(close, "close")
    rv  = compute_realized_vol(close, vol_period)
    mu  = rv.rolling(window=zscore_window, min_periods=zscore_window).mean()
    sig = rv.rolling(window=zscore_window, min_periods=zscore_window).std(ddof=1).replace(0, np.nan)
    return (rv - mu) / sig


def compute_vol_expansion_ratio(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fast: int = 5,
    slow: int = 20,
) -> pd.Series:
    """
    ATR(fast) / ATR(slow): measures short-term vs long-term volatility.
    > 1 = volatility expanding (breakout potential).
    < 1 = volatility contracting (consolidation).
    """
    _require_utc_index(close, "close")
    atr_f = compute_atr(high, low, close, fast)
    atr_s = compute_atr(high, low, close, slow)
    return atr_f / atr_s.replace(0, np.nan)


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Money Flow Index [0, 100] — volume-weighted RSI equivalent.
    NaN for first `period` bars.
    """
    _require_utc_index(close, "close")
    tp  = (high.astype(float) + low.astype(float) + close.astype(float)) / 3.0
    mf  = tp * volume.astype(float)
    up_mf   = pd.Series(np.where(tp.diff() >= 0, mf, 0.0), index=close.index)
    down_mf = pd.Series(np.where(tp.diff() < 0, mf, 0.0), index=close.index)
    pos_mf  = up_mf.rolling(window=period, min_periods=period).sum()
    neg_mf  = down_mf.rolling(window=period, min_periods=period).sum()
    mfr     = pos_mf / neg_mf.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + mfr))
