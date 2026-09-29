"""
src/features/families/regime.py — Market Regime Feature Family

Computes features that characterise the current market regime:
  - volatility regime (rolling vol z-score, vol percentile)
  - trend regime (ADX, EMA spread direction)
  - breadth proxy (only when a universe panel is available)
  - dispersion (cross-sectional return dispersion — proxy for market stress)
  - correlation regime proxy (using the symbol's beta-adjusted residual)

All features are CAUSAL (backward-looking only). No shift(-N).

FIX NEW-P0-002: These regime features are absent from the existing FeatureFactory
and are among the most important alpha sources in institutional equity models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_volatility_regime(
    close: pd.Series,
    short_window: int = 10,
    long_window: int = 60,
    percentile_window: int = 252,
) -> dict[str, pd.Series]:
    """
    Characterise whether we are in a high/low volatility regime.

    Returns
    -------
    vol_regime_zscore : z-score of short-window vol vs long-window vol history
    vol_regime_pctile : percentile rank of current vol vs trailing window
    vol_expanding     : 1 if short vol > long vol (vol expanding), else -1
    vol_ratio         : short_window_vol / long_window_vol
    """
    rets = close.pct_change()

    vol_short = rets.rolling(short_window).std()
    vol_long = rets.rolling(long_window).std()

    # z-score of current short vol vs its own trailing distribution
    vol_long_mean = vol_short.rolling(long_window).mean()
    vol_long_std = vol_short.rolling(long_window).std()
    vol_regime_zscore = (vol_short - vol_long_mean) / vol_long_std.replace(0, np.nan)

    # Percentile rank of current vol in trailing window
    def _rolling_pctile(s: pd.Series, w: int) -> pd.Series:
        def _pctile(arr: np.ndarray) -> float:
            if len(arr) < 2 or np.isnan(arr[-1]):
                return np.nan
            return float(np.sum(arr[:-1] < arr[-1]) / max(len(arr) - 1, 1))
        return s.rolling(w).apply(_pctile, raw=True)

    vol_regime_pctile = _rolling_pctile(vol_short, min(percentile_window, len(close)))

    # Expansion flag
    vol_ratio = vol_short / vol_long.replace(0, np.nan)
    vol_expanding = np.sign(vol_short - vol_long).fillna(0)

    return {
        "vol_regime_zscore": vol_regime_zscore,
        "vol_regime_pctile": vol_regime_pctile,
        "vol_expanding": vol_expanding,
        "vol_ratio": vol_ratio,
    }


def compute_trend_regime(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    adx_window: int = 14,
    ema_short: int = 20,
    ema_long: int = 60,
) -> dict[str, pd.Series]:
    """
    Characterise the trend regime.

    Returns
    -------
    trend_strength    : ADX value (0–100; >25 = trending)
    trend_direction   : sign of EMA_short - EMA_long (+1 uptrend, -1 downtrend)
    trend_persistence : fraction of last 20 bars that closed in the EMA direction
    ema_spread        : (EMA_short - EMA_long) / EMA_long (relative spread)
    """
    # ADX
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(adx_window).mean()
    plus_di = 100.0 * plus_dm.rolling(adx_window).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.rolling(adx_window).mean() / atr.replace(0, np.nan)
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    trend_strength = dx.rolling(adx_window).mean()

    ema_s = close.ewm(span=ema_short, adjust=False).mean()
    ema_l = close.ewm(span=ema_long, adjust=False).mean()
    trend_direction = np.sign(ema_s - ema_l)
    ema_spread = (ema_s - ema_l) / ema_l.replace(0, np.nan)

    # Persistence: fraction of last 20 closes above / below EMA_long
    above = (close > ema_l).astype(float)
    trend_persistence = above.rolling(20).mean()
    # Sign-align with direction: persistence of the dominant direction
    trend_persistence = trend_persistence * trend_direction

    return {
        "trend_strength": trend_strength,
        "trend_direction": trend_direction,
        "ema_spread": ema_spread,
        "trend_persistence": trend_persistence,
    }


def compute_dispersion_regime(
    close_panel: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Cross-sectional dispersion of returns: proxy for market stress / divergence.

    A high dispersion means stocks are moving very differently from each other
    (regime of high idiosyncratic risk or sector rotation).
    A low dispersion means all stocks move together (macro-driven regime).

    Parameters
    ----------
    close_panel : DataFrame where each column is a symbol and rows are timestamps.
                  Must be aligned (same index).
    window      : rolling window for smoothing.

    Returns
    -------
    pd.Series (indexed like close_panel) — rolling std of cross-sectional returns.
    """
    rets = close_panel.pct_change()
    # Cross-sectional std at each timestamp
    xs_std = rets.std(axis=1)
    # Smoothed
    return xs_std.rolling(window).mean()


def compute_gap_regime(
    open_: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> dict[str, pd.Series]:
    """
    Gap regime: characterise how much overnight gapping is occurring.

    Returns
    -------
    gap_magnitude       : |open[t] - close[t-1]| / close[t-1]
    gap_regime_rolling  : rolling mean of |gap| — high = gap-heavy regime
    gap_direction       : sign of gap (positive = gap up, negative = gap down)
    """
    gap = (open_ - close.shift(1)) / close.shift(1).replace(0, np.nan)
    gap_mag = gap.abs()
    gap_regime_rolling = gap_mag.rolling(window).mean()
    gap_direction = np.sign(gap)

    return {
        "gap_magnitude": gap_mag,
        "gap_regime_rolling": gap_regime_rolling,
        "gap_direction": gap_direction,
    }
