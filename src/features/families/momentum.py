"""
Momentum & Trend Feature Family — Phase 3D.

All functions return pd.Series aligned to the input index.
NaN values indicate insufficient history or missing data — never silent defaults.

Causal guarantee
----------------
Every rolling window uses only bars [0 .. t] at bar t.
No shift(-N), no center=True, no forward look.

Talib-free: uses only numpy / pandas.
Production code in features/engineer.py may use talib equivalents;
these functions are the reference implementations for testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Internal helpers ──────────────────────────────────────────────────────────

def _require_utc_index(series: pd.Series, name: str) -> None:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"Series '{name}' must have a DatetimeIndex.")
    if series.index.tz is None:
        raise ValueError(
            f"Series '{name}' has a naive DatetimeIndex. "
            "Localize to UTC before calling feature functions."
        )


def _require_positive(series: pd.Series, name: str) -> pd.Series:
    """Return series with non-positive values replaced by NaN."""
    s = series.astype(float).copy()
    s[s <= 0] = np.nan
    return s


# ── Price / Momentum ──────────────────────────────────────────────────────────

def compute_returns(
    close: pd.Series,
    periods: list[int] | None = None,
) -> dict[str, pd.Series]:
    """
    Multi-period close-to-close returns as percentages.

    Returns dict of {f"return_{p}d": pd.Series} for each period p.
    NaN for the first p bars (insufficient history).

    Parameters
    ----------
    close   : Close price series with UTC DatetimeIndex.
    periods : List of lookback periods. Default: [1, 2, 3, 5, 10, 20, 60].
    """
    _require_utc_index(close, "close")
    c = _require_positive(close, "close")
    if periods is None:
        periods = [1, 2, 3, 5, 10, 20, 60]
    return {f"return_{p}d": c.pct_change(p) * 100 for p in periods}


def compute_log_returns(close: pd.Series, period: int = 1) -> pd.Series:
    """
    Log return over `period` bars.
    log(close[t] / close[t-period])
    """
    _require_utc_index(close, "close")
    c = _require_positive(close, "close")
    return np.log(c / c.shift(period))


def compute_rate_of_change(close: pd.Series, period: int = 12) -> pd.Series:
    """
    Price Rate of Change as percentage.
    (close[t] - close[t-period]) / close[t-period] × 100
    First `period` bars are NaN.
    """
    _require_utc_index(close, "close")
    c = _require_positive(close, "close")
    shifted = c.shift(period)
    return ((c - shifted) / shifted) * 100


def compute_relative_strength_vs_index(
    stock_close: pd.Series,
    index_close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Mansfield Relative Strength vs benchmark index.
    RS = (1 + stock_return_N) / (1 + index_return_N)

    Returns NaN when index_close is None or misaligned — NEVER returns 1.0 silently.
    """
    _require_utc_index(stock_close, "stock_close")
    if index_close is None or len(index_close) == 0:
        return pd.Series(np.nan, index=stock_close.index, dtype=float)

    s = _require_positive(stock_close, "stock_close")
    i = _require_positive(index_close, "index_close").reindex(s.index)

    stock_ret  = s.pct_change(period)
    index_ret  = i.pct_change(period)
    denom      = 1.0 + index_ret
    rs         = (1.0 + stock_ret) / denom.replace(0, np.nan)
    return rs


def compute_trend_strength(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Linear regression slope over `period` bars, normalised to [-1, 1].

    At bar t: OLS slope of close[t-period+1..t] vs bar index [0..period-1],
    then normalised: slope × period / mean_price, clipped to [-1, 1].

    Returns NaN for bars with insufficient history (< period bars available).
    """
    _require_utc_index(close, "close")
    c = close.astype(float)
    n = len(c)
    result = np.full(n, np.nan)
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var  = ((x - x_mean) ** 2).sum()

    for i in range(period - 1, n):
        y = c.iloc[i - period + 1 : i + 1].to_numpy()
        if np.any(np.isnan(y)) or np.any(y <= 0):
            continue
        y_mean = y.mean()
        if y_mean <= 0:
            continue
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
        norm_slope = slope * period / y_mean
        result[i] = float(np.clip(norm_slope * 5.0, -1.0, 1.0))

    return pd.Series(result, index=c.index, dtype=float)


def compute_momentum_slope(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Momentum slope: annualised linear regression slope as % of mean price.
    More interpretable than raw trend_strength.
    """
    _require_utc_index(close, "close")
    c = close.astype(float)
    n = len(c)
    result = np.full(n, np.nan)
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var  = ((x - x_mean) ** 2).sum()

    for i in range(period - 1, n):
        y = c.iloc[i - period + 1 : i + 1].to_numpy()
        if np.any(np.isnan(y)) or np.any(y <= 0):
            continue
        y_mean = y.mean()
        slope  = ((x - x_mean) * (y - y_mean)).sum() / x_var
        result[i] = (slope / y_mean) * 252.0 * 100.0  # annualised %

    return pd.Series(result, index=c.index, dtype=float)


def compute_momentum_t_stat(close: pd.Series, period: int = 20) -> pd.Series:
    """
    t-statistic of the linear regression slope over `period` bars.
    High |t-stat| = statistically persistent trend.
    Distinguishes trend from noise.
    """
    _require_utc_index(close, "close")
    c = close.astype(float)
    n = len(c)
    result = np.full(n, np.nan)
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var  = ((x - x_mean) ** 2).sum()

    for i in range(period - 1, n):
        y = c.iloc[i - period + 1 : i + 1].to_numpy()
        if np.any(np.isnan(y)) or period < 4:
            continue
        y_mean = y.mean()
        slope  = ((x - x_mean) * (y - y_mean)).sum() / x_var
        resid  = y - (slope * x + (y_mean - slope * x_mean))
        sse    = (resid ** 2).sum()
        if sse <= 0 or x_var <= 0:
            continue
        se_slope = np.sqrt(sse / (period - 2) / x_var)
        result[i] = slope / se_slope if se_slope > 0 else np.nan

    return pd.Series(result, index=c.index, dtype=float)


def compute_return_consistency(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Return consistency: fraction of up-days in the last `period` bars.
    [0, 1]: 1 = all up, 0 = all down, 0.5 = equal.

    Economically meaningful: +15% over 20 days with 14/20 up-days is
    more persistent than the same return with 2 explosive days.
    """
    _require_utc_index(close, "close")
    daily_ret = close.astype(float).pct_change()
    up_days   = (daily_ret > 0).astype(float)
    return up_days.rolling(window=period, min_periods=period).mean()


def compute_up_day_fraction(close: pd.Series, period: int = 20) -> pd.Series:
    """Fraction of up days in lookback. Alias for compute_return_consistency."""
    return compute_return_consistency(close, period)


def compute_down_day_fraction(close: pd.Series, period: int = 20) -> pd.Series:
    """Fraction of down days in lookback."""
    _require_utc_index(close, "close")
    daily_ret = close.astype(float).pct_change()
    down_days = (daily_ret < 0).astype(float)
    return down_days.rolling(window=period, min_periods=period).mean()


def compute_distance_from_high(close: pd.Series, period: int = 252) -> pd.Series:
    """
    % distance from N-day rolling high (trailing).
    Value is <= 0 (negative or zero).
    NaN for first `period` bars.
    """
    _require_utc_index(close, "close")
    c = _require_positive(close, "close")
    rolling_high = c.rolling(window=period, min_periods=period).max()
    return ((c - rolling_high) / rolling_high) * 100


def compute_distance_from_low(close: pd.Series, period: int = 252) -> pd.Series:
    """
    % distance from N-day rolling low (trailing).
    Value is >= 0 (positive or zero).
    NaN for first `period` bars.
    """
    _require_utc_index(close, "close")
    c = _require_positive(close, "close")
    rolling_low = c.rolling(window=period, min_periods=period).min()
    return ((c - rolling_low) / rolling_low) * 100


def compute_gap_pct(
    open_price: pd.Series,
    prev_close: pd.Series | None = None,
    close: pd.Series | None = None,
) -> pd.Series:
    """
    Overnight gap as % of previous close.
    gap_pct = (open[t] - close[t-1]) / close[t-1] × 100

    Pass either prev_close (aligned) or close (shifted internally).
    First bar is NaN (no prior close).
    """
    if prev_close is not None:
        _require_utc_index(open_price, "open_price")
        o = open_price.astype(float)
        p = _require_positive(prev_close, "prev_close")
        return ((o - p) / p) * 100
    elif close is not None:
        _require_utc_index(close, "close")
        c = _require_positive(close, "close")
        o = open_price.astype(float)
        return ((o - c.shift(1)) / c.shift(1)) * 100
    else:
        raise ValueError("Supply either prev_close or close.")


def compute_higher_highs_higher_lows(
    high: pd.Series, low: pd.Series, period: int = 5
) -> pd.Series:
    """
    Fraction of HH+HL swings minus LH+LL in last `period` bars.
    Range: [-1, 1]. +1 = all bars making HH+HL.
    """
    _require_utc_index(high, "high")
    n = len(high)
    result = np.full(n, np.nan)

    for i in range(period, n):
        hh = hl = lh = ll = 0
        for j in range(1, period):
            idx  = i - period + j + 1
            pidx = idx - 1
            if idx >= n or pidx < 0:
                break
            if high.iloc[idx] > high.iloc[pidx]:
                hh += 1
            else:
                lh += 1
            if low.iloc[idx] > low.iloc[pidx]:
                hl += 1
            else:
                ll += 1
        denom = 2 * (period - 1)
        if denom > 0:
            result[i] = ((hh + hl) - (lh + ll)) / denom

    return pd.Series(result, index=high.index, dtype=float)


# ── Trend ─────────────────────────────────────────────────────────────────────

def compute_ema(close: pd.Series, period: int) -> pd.Series:
    """
    Exponential moving average with span=period.
    Equivalent to talib.EMA (k = 2/(period+1)).
    NaN for first `period - 1` bars.
    """
    _require_utc_index(close, "close")
    return close.astype(float).ewm(span=period, adjust=False).mean()


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average. NaN for first `period - 1` bars."""
    _require_utc_index(close, "close")
    return close.astype(float).rolling(window=period, min_periods=period).mean()


def compute_ema_stack_score(close: pd.Series) -> pd.Series:
    """
    EMA alignment score in [-1, 1].
    +1 = perfect bull stack (EMA8 > EMA13 > EMA21 > EMA55 > EMA200).
    -1 = perfect bear stack.

    NaN while EMA200 is warming up (first 199 bars).
    """
    _require_utc_index(close, "close")
    c = close.astype(float)
    emas = {p: c.ewm(span=p, adjust=False).mean() for p in [8, 13, 21, 55, 200]}
    pairs = [(8, 13), (13, 21), (21, 55), (55, 200)]
    score = pd.Series(0.0, index=c.index)
    for fast, slow in pairs:
        score += np.where(emas[fast] > emas[slow], 1.0,
                          np.where(emas[fast] < emas[slow], -1.0, 0.0))
    score /= len(pairs)

    # Mark NaN while EMA200 is still warming up
    ema200_nan = emas[200].isna()
    score[ema200_nan] = np.nan
    return score


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD line, signal line, histogram.
    Returns (macd_line, signal_line, histogram).
    """
    _require_utc_index(close, "close")
    c     = close.astype(float)
    ema_f = c.ewm(span=fast, adjust=False).mean()
    ema_s = c.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_f - ema_s
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Wilder ATR (trailing EWM, alpha = 1/period).
    Equivalent to talib.ATR.
    NaN for first bar (no prev close) and warming-up bars.
    """
    _require_utc_index(close, "close")
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average Directional Index — pure pandas, no talib.

    Uses Wilder smoothing (EWM alpha = 1/period).
    """
    _require_utc_index(close, "close")
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)

    up_move   = h.diff()
    down_move = -l.diff()

    plus_dm  = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                         index=c.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=c.index)

    atr14    = compute_atr(h, l, c, period)
    plus_di  = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean()  / atr14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr14.replace(0, np.nan))

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx


# ── Mean Reversion ────────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI — pure pandas (ewm alpha = 1/period).
    Equivalent to talib.RSI within 0.01% after warm-up.
    Range: [0, 100]. NaN for first period bars.
    """
    _require_utc_index(close, "close")
    delta = close.astype(float).diff()
    gain  = delta.clip(lower=0.0)
    loss  = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs   = avg_gain / avg_loss.replace(0, np.nan)
    rsi  = 100.0 - (100.0 / (1.0 + rs))
    rsi.iloc[:period] = np.nan
    return rsi


def compute_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands: (upper, middle, lower).
    Middle = SMA(period).
    NaN for first `period - 1` bars.
    """
    _require_utc_index(close, "close")
    c   = close.astype(float)
    mid = c.rolling(window=period, min_periods=period).mean()
    std = c.rolling(window=period, min_periods=period).std(ddof=1)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def compute_bollinger_position(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Position within Bollinger Bands: 0 = at lower, 1 = at upper.
    NaN when band_width = 0 or insufficient history.
    """
    upper, _, lower = compute_bollinger_bands(close, period)
    band_width = (upper - lower).replace(0, np.nan)
    return (close.astype(float) - lower) / band_width


def compute_zscore_price(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Rolling z-score of price: (close - rolling_mean) / rolling_std.
    Trailing window, no future data.
    """
    _require_utc_index(close, "close")
    c   = close.astype(float)
    mu  = c.rolling(window=period, min_periods=period).mean()
    sig = c.rolling(window=period, min_periods=period).std(ddof=1).replace(0, np.nan)
    return (c - mu) / sig


def compute_zscore_return(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Rolling z-score of daily returns: (ret - rolling_mean_ret) / rolling_std_ret.
    """
    _require_utc_index(close, "close")
    ret = close.astype(float).pct_change()
    mu  = ret.rolling(window=period, min_periods=period).mean()
    sig = ret.rolling(window=period, min_periods=period).std(ddof=1).replace(0, np.nan)
    return (ret - mu) / sig


def compute_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Williams %R in [-100, 0].
    NaN for first `period - 1` bars.
    """
    _require_utc_index(close, "close")
    h = high.astype(float).rolling(window=period, min_periods=period).max()
    l = low.astype(float).rolling(window=period, min_periods=period).min()
    hl_range = (h - l).replace(0, np.nan)
    return ((h - close.astype(float)) / hl_range) * -100.0


def compute_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Commodity Channel Index.
    CCI = (typical_price - rolling_mean_tp) / (0.015 × mean_deviation)
    """
    _require_utc_index(close, "close")
    tp  = (high.astype(float) + low.astype(float) + close.astype(float)) / 3.0
    mu  = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    ).replace(0, np.nan)
    return (tp - mu) / (0.015 * mad)


def compute_stochastic_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
) -> pd.Series:
    """
    Stochastic RSI fast-k: (RSI - min_RSI) / (max_RSI - min_RSI) × 100.
    NaN when insufficient history.
    """
    _require_utc_index(close, "close")
    rsi      = compute_rsi(close, rsi_period)
    rsi_min  = rsi.rolling(window=stoch_period, min_periods=stoch_period).min()
    rsi_max  = rsi.rolling(window=stoch_period, min_periods=stoch_period).max()
    rsi_rng  = (rsi_max - rsi_min).replace(0, np.nan)
    return ((rsi - rsi_min) / rsi_rng) * 100.0


def compute_atr_expansion(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fast_period: int = 5,
    slow_period: int = 20,
) -> pd.Series:
    """
    ATR expansion ratio: ATR(fast) / ATR(slow).
    > 1 means volatility expanding (breakout conditions).
    """
    atr_fast = compute_atr(high, low, close, fast_period)
    atr_slow = compute_atr(high, low, close, slow_period)
    return atr_fast / atr_slow.replace(0, np.nan)


def compute_breakout_score(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """
    Breakout / breakdown score in [-1, 1] with volume confirmation.

    +1: close above prior `lookback`-bar high AND relative volume > 1.2.
    -1: close below prior `lookback`-bar low AND relative volume > 1.2.
    Uses shift(1) on rolling high/low so bar t only sees bars [0..t-1].
    """
    _require_utc_index(close, "close")
    c = close.astype(float)
    h = high.astype(float)
    l = low.astype(float)

    # Prior high/low: shift(1) so we don't include bar t itself
    prior_high = h.rolling(window=lookback, min_periods=lookback).max().shift(1)
    prior_low  = l.rolling(window=lookback, min_periods=lookback).min().shift(1)

    # Relative volume
    avg_vol    = volume.astype(float).rolling(window=lookback, min_periods=lookback).mean()
    rel_vol    = volume.astype(float) / avg_vol.replace(0, np.nan)
    vol_confirm = (rel_vol > 1.2).astype(float) * 0.5 + 0.5

    above  = (c > prior_high).astype(float)
    below  = (c < prior_low).astype(float)
    score  = (above - below) * vol_confirm
    return score.clip(-1.0, 1.0)
