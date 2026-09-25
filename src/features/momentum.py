"""
Momentum and trend features.

Covers: Multi-period returns, rate of change, trend strength, relative
strength vs index, sector momentum, and breakout detection.
"""

import numpy as np
import pandas as pd


def compute_returns(close: pd.Series, periods: list[int] | None = None) -> dict[str, pd.Series]:
    """Compute returns over multiple periods."""
    if periods is None:
        periods = [1, 2, 3, 5, 10, 20, 60]
    result = {}
    for p in periods:
        result[f"return_{p}d"] = close.pct_change(p) * 100
    return result


def compute_rate_of_change(close: pd.Series, period: int = 12) -> pd.Series:
    """Price Rate of Change (ROC)."""
    return ((close - close.shift(period)) / close.shift(period)) * 100


def compute_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """Raw momentum (price difference)."""
    return close - close.shift(period)


def compute_relative_strength_vs_index(
    stock_close: pd.Series, index_close: pd.Series, period: int = 20
) -> pd.Series:
    """
    Relative strength ratio vs benchmark index.
    RS > 1: outperforming the index
    RS < 1: underperforming the index
    """
    stock_return = stock_close.pct_change(period)
    index_return = index_close.pct_change(period)
    # Mansfield RS: stock performance / index performance
    rs = (1 + stock_return) / (1 + index_return)
    return rs


def compute_sector_momentum(
    sector_closes: dict[str, pd.Series], period: int = 5
) -> pd.Series:
    """Average return of the sector over the period."""
    returns = []
    for sym, close in sector_closes.items():
        ret = close.pct_change(period).iloc[-1] if len(close) > period else 0.0
        returns.append(ret)
    return pd.Series([np.mean(returns)] * max(len(next(iter(sector_closes.values()))), 1))


def compute_trend_strength(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Trend strength in [-1, 1] based on linear regression slope normalized.
    +1 = strong uptrend, -1 = strong downtrend.
    """
    result = pd.Series(0.0, index=close.index)
    x = np.arange(period)

    for i in range(period, len(close)):
        y = close.iloc[i - period:i].values
        if len(y) == period and not np.any(np.isnan(y)):
            # Normalized slope: slope * period / mean_price
            slope = np.polyfit(x, y, 1)[0]
            mean_price = np.mean(y)
            if mean_price > 0:
                result.iloc[i] = np.clip((slope * period / mean_price) * 5, -1, 1)

    return result


def compute_atr_expansion(
    high: pd.Series, low: pd.Series, close: pd.Series,
    fast_period: int = 5, slow_period: int = 20
) -> pd.Series:
    """
    ATR expansion rate: fast ATR / slow ATR.
    > 1 means volatility is expanding (breakout conditions).
    """
    from .technical import compute_atr

    atr_fast = compute_atr(high, low, close, fast_period)
    atr_slow = compute_atr(high, low, close, slow_period)
    return atr_fast / atr_slow.replace(0, np.nan)


def compute_breakout_score(
    close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series,
    lookback: int = 20
) -> pd.Series:
    """
    Breakout detection score in [-1, 1].
    +1: strong breakout above resistance with volume
    -1: strong breakdown below support with volume
    """
    from .volume import compute_relative_volume

    rolling_high = high.rolling(window=lookback).max()
    rolling_low = low.rolling(window=lookback).min()
    rel_vol = compute_relative_volume(volume, lookback)

    # Above resistance
    above = (close > rolling_high.shift(1)).astype(float)
    # Below support
    below = (close < rolling_low.shift(1)).astype(float)

    # Volume confirms
    vol_confirm = (rel_vol > 1.2).astype(float) * 0.5 + 0.5

    score = (above - below) * vol_confirm
    return score.clip(-1, 1)


def compute_gap_pct(open_price: pd.Series, prev_close: pd.Series) -> pd.Series:
    """Gap % from previous close to today's open."""
    return ((open_price - prev_close) / prev_close) * 100


def compute_distance_from_high(close: pd.Series, period: int = 52 * 5) -> pd.Series:
    """% distance from N-day high (for weekly use period=52*5 for 52-week)."""
    rolling_high = close.rolling(window=period, min_periods=1).max()
    return ((close - rolling_high) / rolling_high) * 100


def compute_distance_from_low(close: pd.Series, period: int = 52 * 5) -> pd.Series:
    """% distance from N-day low."""
    rolling_low = close.rolling(window=period, min_periods=1).min()
    return ((close - rolling_low) / rolling_low.replace(0, np.nan)) * 100


def compute_higher_highs_higher_lows(
    high: pd.Series, low: pd.Series, period: int = 5
) -> pd.Series:
    """
    Count of consecutive higher-highs and higher-lows.
    Positive = uptrend structure, Negative = downtrend structure.
    """
    result = pd.Series(0.0, index=high.index)

    for i in range(period, len(high)):
        hh_count = 0
        hl_count = 0
        lh_count = 0
        ll_count = 0
        for j in range(1, period):
            idx = i - period + j + 1
            prev_idx = idx - 1
            if high.iloc[idx] > high.iloc[prev_idx]:
                hh_count += 1
            else:
                lh_count += 1
            if low.iloc[idx] > low.iloc[prev_idx]:
                hl_count += 1
            else:
                ll_count += 1

        bull = (hh_count + hl_count) / (2 * (period - 1))
        bear = (lh_count + ll_count) / (2 * (period - 1))
        result.iloc[i] = bull - bear

    return result
