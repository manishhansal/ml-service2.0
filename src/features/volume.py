"""
Volume and flow-based features.

Covers: Relative Volume, Volume Breakout, Volume Profile approximation,
Delivery %, VWAP distance, volume-weighted momentum.
"""

import numpy as np
import pandas as pd


def compute_relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume relative to N-day moving average."""
    avg_vol = volume.rolling(window=period).mean()
    return volume / avg_vol.replace(0, np.nan)


def compute_volume_breakout(volume: pd.Series, period: int = 20, threshold: float = 1.5) -> pd.Series:
    """Binary flag: 1 if volume >= threshold * avg, 0 otherwise."""
    rel_vol = compute_relative_volume(volume, period)
    return (rel_vol >= threshold).astype(float)


def compute_volume_trend(volume: pd.Series, period: int = 10) -> pd.Series:
    """Volume trend: ratio of recent avg volume to prior avg."""
    recent = volume.rolling(window=period).mean()
    prior = volume.shift(period).rolling(window=period).mean()
    return recent / prior.replace(0, np.nan)


def compute_volume_price_confirmation(
    close: pd.Series, volume: pd.Series, period: int = 20
) -> pd.Series:
    """
    Volume-price confirmation score.
    +1: rising price + rising volume (strong trend)
    -1: rising price + falling volume (weak/divergent trend)
    """
    price_direction = close.diff(period).apply(lambda x: 1 if x > 0 else -1)
    vol_direction = volume.rolling(period).mean().diff().apply(lambda x: 1 if x > 0 else -1)
    return (price_direction * vol_direction).astype(float)


def compute_vwap_distance_pct(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    period: int = 20,
    mode: str = "rolling",
) -> pd.Series:
    """
    % distance from VWAP — with explicit, correct semantics per bar type.

    Causal guarantee
    ----------------
    The VWAP value at bar t uses ONLY bars ≤ t.  No future bar can affect
    the value at any past bar.

    Parameters
    ----------
    close, high, low, volume : standard OHLCV series
    period : lookback window length (bars) used in "rolling" mode.
             Ignored in "intraday" mode (resets per session).
    mode : "rolling"   — rolling N-bar VWAP.
               Each bar's VWAP = Σ(tp × vol) over [t-N+1 .. t] /
                                  Σ(vol)        over [t-N+1 .. t].
               Suitable for daily bars where there is no meaningful
               "session" boundary in the input data.
               This replaces the previous cumsum() implementation that
               accumulated from bar 0 of the passed window and produced
               a multi-month average masquerading as "intraday VWAP".
           "intraday" — session-anchored VWAP.
               Resets when the DatetimeIndex date changes.
               Suitable for sub-daily bars (1-min, 5-min) where the series
               contains multiple trading sessions.

    Returns
    -------
    pd.Series of floats: (close - vwap) / vwap × 100.
    Positive → close above VWAP; negative → close below VWAP.
    NaN where the rolling window has insufficient data.

    Previous behaviour (WRONG for daily bars)
    -----------------------------------------
    The old implementation used cumsum() over whatever series was passed in.
    For a 200-bar lookback window passed from compute_stock_features(), this
    produced a 200-day cumulative VWAP — a noisy proxy for "average cost
    over the last 10 months", NOT an intraday VWAP.  That feature was
    semantically misleading and highly autocorrelated with the price trend.
    """
    typical_price = (high + low + close) / 3

    if mode == "intraday":
        # Session-reset VWAP: accumulate within each calendar date.
        # Requires a DatetimeIndex; falls back to rolling if unavailable.
        if not isinstance(close.index, pd.DatetimeIndex):
            # Can't determine session boundaries — fall back to rolling.
            mode = "rolling"
        else:
            tp_vol = typical_price * volume
            # Group by date, compute cumulative sum within each group.
            dates = close.index.date
            date_series = pd.Series(dates, index=close.index)
            cum_tp_vol = tp_vol.groupby(date_series).cumsum()
            cum_vol    = volume.groupby(date_series).cumsum()
            vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
            return ((close - vwap) / vwap.replace(0, np.nan)) * 100

    # "rolling" mode (default, and fallback from intraday)
    # Uses a trailing N-bar window — purely causal.
    tp_vol = typical_price * volume
    rolling_tp_vol = tp_vol.rolling(window=period, min_periods=1).sum()
    rolling_vol    = volume.rolling(window=period, min_periods=1).sum()
    vwap = rolling_tp_vol / rolling_vol.replace(0, np.nan)
    return ((close - vwap) / vwap.replace(0, np.nan)) * 100


def compute_volume_profile_score(
    close: pd.Series, volume: pd.Series, high: pd.Series, low: pd.Series,
    lookback: int = 20
) -> pd.Series:
    """
    Simplified volume profile score.
    Measures where current price sits relative to the volume-weighted
    price distribution over the lookback period.
    +1: at the top of the high-volume zone (volume node support below)
    -1: at the bottom of the high-volume zone (volume node resistance above)
    """
    scores = pd.Series(0.0, index=close.index)

    for i in range(lookback, len(close)):
        window_close = close.iloc[i - lookback:i]
        window_vol = volume.iloc[i - lookback:i]
        window_high = high.iloc[i - lookback:i]
        window_low = low.iloc[i - lookback:i]

        # Volume-weighted average price in the window
        vwap_window = (window_close * window_vol).sum() / window_vol.sum()
        price_range = window_high.max() - window_low.min()

        if price_range > 0:
            position = (close.iloc[i] - window_low.min()) / price_range
            vwap_position = (vwap_window - window_low.min()) / price_range
            scores.iloc[i] = 2 * (position - vwap_position)

    return scores.clip(-1, 1)


def compute_accumulation_distribution(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Accumulation/Distribution Line."""
    hl_range = high - low
    mf_multiplier = ((close - low) - (high - close)) / hl_range.replace(0, np.nan)
    mf_multiplier = mf_multiplier.fillna(0)
    ad = (mf_multiplier * volume).cumsum()
    return ad


def compute_force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Elder's Force Index smoothed over `period`."""
    fi = close.diff() * volume
    return fi.ewm(span=period, adjust=False).mean()


def compute_volume_oscillator(volume: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    """Volume Oscillator: (fast EMA - slow EMA) / slow EMA * 100."""
    fast_ema = volume.ewm(span=fast, adjust=False).mean()
    slow_ema = volume.ewm(span=slow, adjust=False).mean()
    return ((fast_ema - slow_ema) / slow_ema.replace(0, np.nan)) * 100


def compute_vpin(
    bars: list[dict],
    bucket_size: float = 50,
    n_buckets: int = 50,
) -> dict:
    """
    Compute VPIN (Volume-synchronized Probability of Informed Trading)
    from 5-min OHLCV bars.

    Uses the tick rule to classify each bar's volume into buy/sell fractions:
      - close > prev_close  → 85% buy
      - close < prev_close  → 15% buy (85% sell)
      - close == prev_close → 50% buy

    Volume accumulates into fixed-size buckets. For each completed bucket:
      bucket_vpin = |buy_vol − sell_vol| / bucket_size

    The current VPIN is the mean of the last `n_buckets` bucket values.

    Args:
        bars:        List of dicts with keys open/high/low/close/volume.
        bucket_size: Volume units per bucket (must be > 0).
        n_buckets:   Number of recent buckets to average for current_vpin.

    Returns:
        {
          "vpin_series":  list[float]  — one value per completed bucket,
          "current_vpin": float        — mean of last n_buckets values.
        }

    Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.8
    """
    buckets: list[float] = []
    current_buy_vol = 0.0
    current_sell_vol = 0.0
    current_total = 0.0

    for i, bar in enumerate(bars):
        if i == 0:
            # For the first bar, use the bar's own open as the reference
            # price so the tick rule works correctly when open != close.
            prev_close = bar.get("open", bar["close"])
        else:
            prev_close = bars[i - 1]["close"]

        if bar["close"] > prev_close:
            buy_fraction = 0.85
        elif bar["close"] < prev_close:
            buy_fraction = 0.15
        else:
            buy_fraction = 0.5

        remaining_vol = bar["volume"]

        while remaining_vol > 0:
            # How much space is left in the current bucket?
            space = bucket_size - current_total

            if remaining_vol < space:
                # Bar doesn't complete the current bucket — just accumulate
                current_buy_vol += remaining_vol * buy_fraction
                current_sell_vol += remaining_vol * (1.0 - buy_fraction)
                current_total += remaining_vol
                remaining_vol = 0.0
            else:
                # Fill the bucket to completion using `space` units of this bar
                current_buy_vol += space * buy_fraction
                current_sell_vol += space * (1.0 - buy_fraction)
                current_total += space
                remaining_vol -= space

                # Emit the completed bucket
                bucket_vpin = abs(current_buy_vol - current_sell_vol) / bucket_size
                buckets.append(bucket_vpin)

                # Reset for the next bucket
                current_buy_vol = 0.0
                current_sell_vol = 0.0
                current_total = 0.0

    if not buckets:
        return {"vpin_series": [], "current_vpin": 0.0}

    recent = buckets[-n_buckets:]
    current_vpin = sum(recent) / len(recent)
    return {"vpin_series": buckets, "current_vpin": current_vpin}
