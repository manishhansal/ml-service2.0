"""
SweepDetector -- identifies liquidity sweep / stop-hunt reversal events.

A bullish sweep: wick below trailing swing_low, close back above swing_low, high volume.
A bearish sweep: wick above trailing swing_high, close back below swing_high, high volume.

Optional reversal confirmation: next candle closes further in the reversal direction.
A minimum sweep depth filter (min_sweep_magnitude_atr_multiplier) rejects micro-wicks
that barely pierce the level -- these tend to be noise rather than genuine stop-hunts.

Fixes applied:
- volume_ratio is pre-computed and stored on the signal during detection
- min_sweep_magnitude_atr_multiplier rejects micro-wicks (< 0.25 * ATR14 depth)
- volume bypass on zero-avg-vol symbols: require vol_k > 0 when avg_v_k == 0
- equal_levels count uses raw candle lows/highs, not the rolling sw_low/sw_high series
"""
from __future__ import annotations
import pandas as pd
from src.scanners.liquidity_sweep.config import SweepConfig
from src.scanners.liquidity_sweep.schemas import SweepDirection, SweepLifecycle, SweepSignal


def _count_equal_levels(
    candle_series: pd.Series,
    level: float,
    k: int,
    lookback: int,
    tolerance: float,
) -> int:
    """Count actual candle lows/highs within tolerance of level in the lookback window.

    Uses the raw OHLC series (df["low"] or df["high"]), not the rolling swing-min/max
    series, to avoid plateau inflation where a persistent extreme inflates the count.
    """
    count = 0
    for j in range(max(0, k - lookback + 1), k + 1):
        if abs(float(candle_series.iloc[j]) - level) <= tolerance:
            count += 1
    return max(count, 1)


def detect_sweeps(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: SweepConfig,
) -> list[SweepSignal]:
    """Detect liquidity sweep events in an OHLCV series."""
    min_bars = config.swing_lookback + 1
    if config.require_reversal_confirmation:
        min_bars += 1
    if len(df) < min_bars:
        return []

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=config.atr_window, min_periods=1).mean()

    sw_high = df["high"].rolling(window=config.swing_lookback, min_periods=1).max()
    sw_low  = df["low"].rolling(window=config.swing_lookback, min_periods=1).min()
    avg_vol = df["volume"].rolling(window=config.avg_volume_window, min_periods=1).mean()

    n = len(df)
    upper_k = n - 2 if config.require_reversal_confirmation else n - 1
    results: list[SweepSignal] = []

    for k in range(config.swing_lookback, upper_k + 1):
        if not pd.notna(df.iloc[k][["high", "low", "close", "volume"]]).all():
            continue

        vol_k   = float(df["volume"].iloc[k])
        avg_v_k = float(avg_vol.iloc[k])
        atr_k   = float(atr.iloc[k])
        close_k = float(df["close"].iloc[k])
        low_k   = float(df["low"].iloc[k])
        high_k  = float(df["high"].iloc[k])

        # Volume confirmation
        if avg_v_k > 0:
            if vol_k < config.volume_confirmation_multiplier * avg_v_k:
                continue
        else:
            # avg_vol is zero (e.g. index with no volume history).
            # Require the sweep bar itself to have non-zero volume.
            if vol_k == 0:
                continue

        sw_low_prev  = float(sw_low.iloc[k - 1])
        sw_high_prev = float(sw_high.iloc[k - 1])

        bullish_sweep = (low_k < sw_low_prev) and (close_k > sw_low_prev)
        bearish_sweep = (high_k > sw_high_prev) and (close_k < sw_high_prev)

        if not bullish_sweep and not bearish_sweep:
            continue

        # Minimum sweep depth filter -- reject micro-wicks that barely pierce the level
        if atr_k > 0:
            if bullish_sweep:
                sweep_extension = sw_low_prev - low_k
            else:
                sweep_extension = high_k - sw_high_prev
            if sweep_extension < config.min_sweep_magnitude_atr_multiplier * atr_k:
                continue

        # Reversal confirmation
        if config.require_reversal_confirmation:
            if k + 1 >= n:
                continue
            if not pd.notna(df.iloc[k + 1]["close"]):
                continue
            next_close = float(df["close"].iloc[k + 1])
            if bullish_sweep and next_close <= close_k:
                continue
            if bearish_sweep and next_close >= close_k:
                continue

        if bullish_sweep:
            direction   = SweepDirection.BULLISH
            swept_level = sw_low_prev
        else:
            direction   = SweepDirection.BEARISH
            swept_level = sw_high_prev

        sweep_zone_half = 0.5 * atr_k if atr_k > 0 else abs(swept_level * 0.005)

        tolerance = max(config.equal_level_tolerance_pct * float(df["close"].iloc[k]), 0.1 * atr_k)
        if bullish_sweep:
            eq_count = _count_equal_levels(df["low"], swept_level, k, config.swing_lookback, tolerance)
        else:
            eq_count = _count_equal_levels(df["high"], swept_level, k, config.swing_lookback, tolerance)

        # Pre-compute volume_ratio and store on signal for immediate API filtering
        volume_ratio = vol_k / avg_v_k if avg_v_k > 0 else float("inf")

        signal = SweepSignal(
            symbol=symbol,
            direction=direction,
            timeframe=timeframe,
            swept_level=swept_level,
            sweep_zone_high=swept_level + sweep_zone_half,
            sweep_zone_low=swept_level  - sweep_zone_half,
            sweep_timestamp=df.index[k],
            sweep_bar_index=k,
            equal_levels_count=eq_count,
            volume_ratio=volume_ratio,
        )
        results.append(signal)

    return results
