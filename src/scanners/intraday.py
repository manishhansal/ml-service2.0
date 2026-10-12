"""
intraday.py — shared utility for computing intraday trading levels from daily zones.

Daily zones (Order Blocks, FVGs, S&D zones, etc.) mark institutional levels where
price tends to reverse intraday.  This module converts each zone into a concrete
intraday trade setup: entry, target, stop, and R:R ratio.

Computation rules
-----------------
Bullish setup (demand zones, bullish OBs, bullish FVGs, BOS_BULLISH):
  entry  = zone_high   (buy limit at the top of the demand zone — price dips in)
  target = zone_high + (zone_height * 1.5)   (1.5R reward)
  stop   = zone_low  - (atr * 0.5)           (buffer below zone)

Bearish setup (supply zones, bearish OBs, bearish FVGs, BOS_BEARISH):
  entry  = zone_low    (sell limit at the bottom of the supply zone — price rallies in)
  target = zone_low  - (zone_height * 1.5)
  stop   = zone_high + (atr * 0.5)

For Liquidity Sweep:
  Bullish sweep (swept lows): entry = swept_level, target = swept_level + atr*3, stop = sweep_zone_low
  Bearish sweep (swept highs): entry = swept_level, target = swept_level - atr*3, stop = sweep_zone_high

proximity_pct = abs(current_close - entry) / current_close * 100
is_actionable = proximity_pct <= 1.5  (within 1.5% of entry — intraday-ready)

The 1.5% threshold ensures signals are only marked actionable when price is
genuinely close to the entry level.  For intraday trading a 2%+ gap from
entry means price must move significantly before triggering — increasing
the risk of a missed or partial fill.
"""
from __future__ import annotations

# Proximity threshold for intraday actionability
_ACTIONABLE_THRESHOLD_PCT = 1.5


def compute_zone_intraday(
    is_bullish: bool,
    zone_high: float,
    zone_low: float,
    current_close: float,
    atr: float,
) -> dict:
    """Compute intraday levels for a zone-based signal (OB, FVG, S&D)."""
    zone_height = max(zone_high - zone_low, atr * 0.1)  # floor to avoid zero-height

    if is_bullish:
        entry  = zone_high
        target = zone_high + zone_height * 1.5
        stop   = zone_low - atr * 0.5
    else:
        entry  = zone_low
        target = zone_low - zone_height * 1.5
        stop   = zone_high + atr * 0.5

    risk   = abs(entry - stop)
    reward = abs(target - entry)
    rr     = round(reward / risk, 2) if risk > 0 else 0.0
    prox   = round(abs(current_close - entry) / current_close * 100, 2) if current_close > 0 else 999.0

    return {
        "intraday_entry_price":  round(entry, 2),
        "intraday_target_price": round(target, 2),
        "intraday_stop_price":   round(stop, 2),
        "intraday_rr_ratio":     rr,
        "proximity_pct":         prox,
        "is_actionable":         prox <= _ACTIONABLE_THRESHOLD_PCT,
    }


def compute_sweep_intraday(
    is_bullish: bool,
    swept_level: float,
    sweep_zone_high: float,
    sweep_zone_low: float,
    current_close: float,
    atr: float,
) -> dict:
    """Compute intraday levels for a Liquidity Sweep signal."""
    if is_bullish:
        entry  = swept_level
        target = swept_level + atr * 3.0   # 3 ATR target above swept level
        stop   = sweep_zone_low            # bottom of sweep zone
    else:
        entry  = swept_level
        target = swept_level - atr * 3.0
        stop   = sweep_zone_high

    risk   = abs(entry - stop)
    reward = abs(target - entry)
    rr     = round(reward / risk, 2) if risk > 0 else 0.0
    prox   = round(abs(current_close - entry) / current_close * 100, 2) if current_close > 0 else 999.0

    return {
        "intraday_entry_price":  round(entry, 2),
        "intraday_target_price": round(target, 2),
        "intraday_stop_price":   round(stop, 2),
        "intraday_rr_ratio":     rr,
        "proximity_pct":         prox,
        "is_actionable":         prox <= _ACTIONABLE_THRESHOLD_PCT,
    }


def compute_ms_intraday(
    is_bullish: bool,
    broken_level: float,
    current_close: float,
    atr: float,
    is_choch: bool,
) -> dict:
    """Compute intraday levels for a Market Structure signal (BoS/ChoCh).

    The broken_level is the key price: traders buy/sell a retest of that level.
    ChoCh gets a 1.5x multiplier on the target (higher conviction reversal).
    """
    target_multiplier = 3.0 if is_choch else 2.0

    if is_bullish:
        entry  = broken_level              # buy retest of broken resistance-turned-support
        target = broken_level + atr * target_multiplier
        stop   = broken_level - atr * 0.75
    else:
        entry  = broken_level              # sell retest of broken support-turned-resistance
        target = broken_level - atr * target_multiplier
        stop   = broken_level + atr * 0.75

    risk   = abs(entry - stop)
    reward = abs(target - entry)
    rr     = round(reward / risk, 2) if risk > 0 else 0.0
    prox   = round(abs(current_close - entry) / current_close * 100, 2) if current_close > 0 else 999.0

    return {
        "intraday_entry_price":  round(entry, 2),
        "intraday_target_price": round(target, 2),
        "intraday_stop_price":   round(stop, 2),
        "intraday_rr_ratio":     rr,
        "proximity_pct":         prox,
        "is_actionable":         prox <= _ACTIONABLE_THRESHOLD_PCT,
    }
