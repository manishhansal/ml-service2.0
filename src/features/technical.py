"""
Technical indicator features computed from OHLCV data.

Covers: RSI, MACD, ADX, Bollinger Bands, EMA stack, Stochastic RSI,
Williams %R, CCI, MFI, ATR, OBV, and custom derived indicators.

All core indicator loops have been replaced with TA-Lib vectorised C-backed
calls (Requirement 6.1). New candlestick pattern and HT_TRENDLINE features
added (Requirement 6.2). Existing public function signatures are preserved so
callers in engineer.py do not break.
"""

import numpy as np
import pandas as pd
import talib


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_f64(series: pd.Series) -> np.ndarray:
    """Convert a pandas Series to a contiguous float64 numpy array."""
    return np.ascontiguousarray(series.to_numpy(dtype=np.float64))


def _last_finite(arr: np.ndarray) -> float:
    """Return the last finite value from a numpy array, or 0.0."""
    for val in reversed(arr):
        if np.isfinite(val):
            return float(val)
    return 0.0


def _wrap(arr: np.ndarray, index: "pd.Index") -> pd.Series:
    """Wrap a numpy array back into a pandas Series with the given index."""
    return pd.Series(arr, index=index, dtype=np.float64)


# ---------------------------------------------------------------------------
# Core indicators — TA-Lib backed (Requirement 6.1)
# ---------------------------------------------------------------------------

def compute_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI via talib.RSI (vectorised C-backed).

    Signature unchanged — callers pass a pd.Series and receive a pd.Series.
    talib.RSI uses Wilder's smoothing (alpha = 1/period), matching the
    previous ewm(alpha=1/period) implementation within 0.01% after warm-up.
    """
    arr = _to_f64(closes)
    result = talib.RSI(arr, timeperiod=period)
    return _wrap(result, closes.index)


def compute_macd(
    closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram via talib.MACD."""
    arr = _to_f64(closes)
    macd_line, signal_line, histogram = talib.MACD(
        arr, fastperiod=fast, slowperiod=slow, signalperiod=signal
    )
    idx = closes.index
    return _wrap(macd_line, idx), _wrap(signal_line, idx), _wrap(histogram, idx)


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average Directional Index (ADX) via talib.ADX."""
    result = talib.ADX(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod=period)
    return _wrap(result, close.index)


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Wilder ATR via talib.ATR (vectorised C-backed).

    Matches the previous ewm(alpha=1/period) implementation within 0.01%
    after warm-up (Requirement 6.1).
    """
    result = talib.ATR(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod=period)
    return _wrap(result, close.index)


def compute_bollinger_bands(
    closes: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Upper band, middle band, lower band via talib.BBANDS."""
    arr = _to_f64(closes)
    upper, middle, lower = talib.BBANDS(
        arr, timeperiod=period, nbdevup=num_std, nbdevdn=num_std, matype=0
    )
    idx = closes.index
    return _wrap(upper, idx), _wrap(middle, idx), _wrap(lower, idx)


def compute_bollinger_position(closes: pd.Series, period: int = 20) -> pd.Series:
    """Position within Bollinger Bands: 0 = lower, 1 = upper."""
    upper, _, lower = compute_bollinger_bands(closes, period)
    band_width = upper - lower
    return (closes - lower) / band_width.replace(0, np.nan)


def compute_ema_stack_score(closes: pd.Series) -> pd.Series:
    """
    EMA alignment score in [-1, 1] via talib.EMA (Requirement 6.1).
    +1 = perfect bull stack (8 > 13 > 21 > 55 > 200)
    -1 = perfect bear stack (8 < 13 < 21 < 55 < 200)

    talib.EMA uses k = 2/(period+1), matching pandas ewm(span=period,
    adjust=False) to within 1e-6 (Requirement 6.6).
    """
    arr = _to_f64(closes)
    emas: dict[int, np.ndarray] = {}
    for p in [8, 13, 21, 55, 200]:
        emas[p] = talib.EMA(arr, timeperiod=p)

    pairs = [(8, 13), (13, 21), (21, 55), (55, 200)]
    score = np.zeros(len(arr), dtype=np.float64)
    for fast, slow in pairs:
        f_arr = emas[fast]
        s_arr = emas[slow]
        score += np.where(f_arr > s_arr, 1.0, np.where(f_arr < s_arr, -1.0, 0.0))
    score /= len(pairs)
    return _wrap(score, closes.index)


def compute_stochastic_rsi(
    closes: pd.Series, rsi_period: int = 14, stoch_period: int = 14
) -> pd.Series:
    """Stochastic RSI (fastk) in [0, 100] via talib.STOCHRSI."""
    arr = _to_f64(closes)
    fastk, _ = talib.STOCHRSI(
        arr,
        timeperiod=rsi_period,
        fastk_period=stoch_period,
        fastd_period=3,
        fastd_matype=0,
    )
    return _wrap(fastk, closes.index)


def compute_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Williams %R in [-100, 0] via talib.WILLR."""
    result = talib.WILLR(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod=period)
    return _wrap(result, close.index)


def compute_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """Commodity Channel Index via talib.CCI."""
    result = talib.CCI(_to_f64(high), _to_f64(low), _to_f64(close), timeperiod=period)
    return _wrap(result, close.index)


def compute_mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    """Money Flow Index (volume-weighted RSI) via talib.MFI."""
    result = talib.MFI(
        _to_f64(high), _to_f64(low), _to_f64(close), _to_f64(volume), timeperiod=period
    )
    return _wrap(result, close.index)


def compute_cmf(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
) -> pd.Series:
    """Chaikin Money Flow (pure-Python; no direct TA-Lib equivalent)."""
    hl_range = high - low
    mf_multiplier = ((close - low) - (high - close)) / hl_range.replace(0, np.nan)
    mf_volume = mf_multiplier * volume
    return mf_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume via talib.OBV."""
    result = talib.OBV(_to_f64(close), _to_f64(volume))
    return _wrap(result, close.index)


def compute_obv_trend(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """OBV trend: slope of OBV over `period` bars, normalized."""
    obv = compute_obv(close, volume)
    obv_sma = obv.rolling(window=period).mean()
    obv_std = obv.rolling(window=period).std().replace(0, np.nan)
    return (obv - obv_sma) / obv_std


def compute_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """
    Cumulative VWAP (intraday reset assumed by caller grouping by day).
    For daily bars this is an approximation using typical price * volume.
    """
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def compute_supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 10, multiplier: float = 3.0
) -> pd.Series:
    """
    Supertrend indicator: returns +1 for uptrend, -1 for downtrend.

    Uses talib.ATR internally; the trend-flip logic remains in Python
    (no TA-Lib equivalent for the full Supertrend algorithm).
    """
    atr = compute_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    direction = pd.Series(1, index=close.index, dtype=float)

    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                lower_band.iloc[i] = lower_band.iloc[i - 1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                upper_band.iloc[i] = upper_band.iloc[i - 1]

    return direction


# ---------------------------------------------------------------------------
# New candlestick pattern features (Requirement 6.2)
# ---------------------------------------------------------------------------

def compute_cdl_engulfing(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """
    Engulfing candlestick pattern (bullish or bearish).
    Returns +100 for bullish engulfing, -100 for bearish, 0 otherwise.
    Binary feature for ML: 1 if any engulfing occurred, else 0.
    """
    result = talib.CDLENGULFING(
        _to_f64(open_), _to_f64(high), _to_f64(low), _to_f64(close)
    )
    return _wrap((result != 0).astype(np.float64), close.index)


def compute_cdl_hammer(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """
    Hammer candlestick pattern.
    Returns 1.0 if a hammer is detected on that bar, else 0.0.
    """
    result = talib.CDLHAMMER(
        _to_f64(open_), _to_f64(high), _to_f64(low), _to_f64(close)
    )
    return _wrap((result != 0).astype(np.float64), close.index)


def compute_cdl_doji(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """
    Doji candlestick pattern.
    Returns 1.0 if a doji is detected on that bar, else 0.0.
    """
    result = talib.CDLDOJI(
        _to_f64(open_), _to_f64(high), _to_f64(low), _to_f64(close)
    )
    return _wrap((result != 0).astype(np.float64), close.index)


def compute_ht_trendline_dev(close: pd.Series) -> pd.Series:
    """
    HT_TRENDLINE deviation: (close - HT_TRENDLINE) / HT_TRENDLINE.

    Captures how far price has deviated from the Hilbert Transform
    instantaneous trendline (Requirement 6.2).
    """
    arr = _to_f64(close)
    ht = talib.HT_TRENDLINE(arr)
    # Avoid division by zero; where ht == 0 emit 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        dev = np.where(ht != 0.0, (arr - ht) / ht, 0.0)
    dev = np.where(np.isfinite(dev), dev, 0.0)
    return _wrap(dev, close.index)
