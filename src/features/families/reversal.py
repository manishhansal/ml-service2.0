"""
src.features.families.reversal — Reversal/Mean-Reversion Feature Family.

Why momentum misses reversals
-------------------------------
A pure momentum model predicts that trends continue.  It assigns high SHORT
scores to stocks in downtrends and high LONG scores to stocks in uptrends.
It cannot identify:
  - Oversold bounces: stocks that have fallen too far and are due a recovery
    (ADANIENT -8% over 5 days → bounced +5% on day 6)
  - Overbought crashes: stocks that have risen too fast and mean-revert
    (IEX +6% over 5 days → dropped -4.5% on day 6)
  - Volume-confirmed reversals: heavy selling on a down day (capitulation)
    followed by a sharp reversal the next day

Reversal Features (Group F — 12 new features)
----------------------------------------------
These features capture the conditions that PRECEDE a reversal and are
deliberately designed to COMPLEMENT the momentum features, not replace them.

The LightGBM model will learn which combination of momentum + reversal features
predicts the best outcome.  In training data:
  - Momentum features dominate in trending regimes
  - Reversal features dominate in mean-reverting regimes
  LightGBM allocates feature importance to each group by regime automatically.

Features computed:
    rsi_oversold_flag      1 if RSI < 30 (potential LONG reversal)
    rsi_overbought_flag    1 if RSI > 70 (potential SHORT reversal)
    rsi_extreme_distance   signed distance from extreme (+ = oversold, - = overbought)
    bb_pct_b               Bollinger %B: 0=at lower band, 1=at upper band
    bb_below_lower         1 if price < lower Bollinger Band (oversold squeeze)
    bb_above_upper         1 if price > upper Bollinger Band (overbought squeeze)
    consec_down_bars       count of consecutive down closes (3+ → oversold)
    consec_up_bars         count of consecutive up closes (3+ → overbought)
    vol_spike_ratio        today's volume / 20-day avg volume (>2 = abnormal activity)
    price_from_20d_low_pct (close - 20d_low) / 20d_low: 0% = at 20d low = oversold
    price_from_20d_high_pct (close - 20d_high) / 20d_high: 0% = at 20d high = overbought
    stoch_reversal_signal  +1=stoch %K crossed above %D from oversold, -1=crossed below from OB

PIT safety:
    All features use only data available at or before bar i.
    No forward-looking operations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_reversal_features(
    close: pd.Series,
    high:  pd.Series,
    low:   pd.Series,
    volume: pd.Series,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    stoch_k_period: int = 14,
    stoch_d_period: int = 3,
) -> dict[str, pd.Series]:
    """
    Compute all reversal/mean-reversion features.

    Args:
        close, high, low, volume: OHLCV series aligned by date index.
        rsi_period:    RSI lookback (default 14).
        bb_period:     Bollinger Bands SMA period (default 20).
        bb_std:        Bollinger Bands standard deviation multiplier (default 2).
        stoch_k_period: Stochastic %K period (default 14).
        stoch_d_period: Stochastic %D smoothing period (default 3).

    Returns:
        Dict of feature_name → pd.Series, all aligned to ``close.index``.
    """
    feats: dict[str, pd.Series] = {}

    # ── RSI ───────────────────────────────────────────────────────────────────
    delta   = close.diff()
    gain    = delta.clip(lower=0)
    loss    = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=rsi_period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))

    feats["rsi_oversold_flag"]     = (rsi < 30).astype(float)
    feats["rsi_overbought_flag"]   = (rsi > 70).astype(float)
    # Positive = oversold (distance below 50), negative = overbought (above 50)
    feats["rsi_extreme_distance"]  = (50 - rsi) / 50.0   # normalised [-1, +1]

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    bb_sma  = close.rolling(bb_period).mean()
    bb_std_ = close.rolling(bb_period).std()
    bb_upper = bb_sma + bb_std * bb_std_
    bb_lower = bb_sma - bb_std * bb_std_
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)

    bb_pct_b = (close - bb_lower) / bb_range  # 0=lower band, 0.5=middle, 1=upper
    feats["bb_pct_b"]           = bb_pct_b
    feats["bb_below_lower"]     = (close < bb_lower).astype(float)
    feats["bb_above_upper"]     = (close > bb_upper).astype(float)

    # ── Consecutive directional bars ─────────────────────────────────────────
    direction = np.sign(close.diff())

    consec_down = pd.Series(0.0, index=close.index)
    consec_up   = pd.Series(0.0, index=close.index)
    cnt_d = cnt_u = 0
    for i, d in enumerate(direction):
        if d < 0:
            cnt_d += 1; cnt_u = 0
        elif d > 0:
            cnt_u += 1; cnt_d = 0
        else:
            cnt_d = cnt_u = 0
        consec_down.iloc[i] = cnt_d
        consec_up.iloc[i]   = cnt_u

    feats["consec_down_bars"] = consec_down  # 3+ = oversold territory
    feats["consec_up_bars"]   = consec_up    # 3+ = overbought territory

    # ── Volume spike ──────────────────────────────────────────────────────────
    avg_vol = volume.rolling(20).mean().replace(0, np.nan)
    feats["vol_spike_ratio"] = (volume / avg_vol).clip(upper=10.0)

    # ── Price distance from 20-day extremes ───────────────────────────────────
    low_20d  = low.rolling(20).min()
    high_20d = high.rolling(20).max()
    feats["price_from_20d_low_pct"]  = ((close - low_20d)  / low_20d.replace(0, np.nan)).clip(-1, 5)
    feats["price_from_20d_high_pct"] = ((close - high_20d) / high_20d.replace(0, np.nan)).clip(-1, 0.5)

    # ── Stochastic %K/%D crossover signal ────────────────────────────────────
    lowest_low   = low.rolling(stoch_k_period).min()
    highest_high = high.rolling(stoch_k_period).max()
    hl_range     = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = 100 * (close - lowest_low) / hl_range
    stoch_d = stoch_k.rolling(stoch_d_period).mean()

    # Bullish crossover from oversold: %K crosses above %D when both < 20
    prev_k = stoch_k.shift(1); prev_d = stoch_d.shift(1)
    bullish_cross = ((prev_k <= prev_d) & (stoch_k > stoch_d) & (stoch_k < 30)).astype(float)
    bearish_cross = ((prev_k >= prev_d) & (stoch_k < stoch_d) & (stoch_k > 70)).astype(float)
    feats["stoch_reversal_signal"] = bullish_cross - bearish_cross  # +1, -1 or 0

    return feats


REVERSAL_FEATURE_NAMES = [
    "rsi_oversold_flag",
    "rsi_overbought_flag",
    "rsi_extreme_distance",
    "bb_pct_b",
    "bb_below_lower",
    "bb_above_upper",
    "consec_down_bars",
    "consec_up_bars",
    "vol_spike_ratio",
    "price_from_20d_low_pct",
    "price_from_20d_high_pct",
    "stoch_reversal_signal",
]
