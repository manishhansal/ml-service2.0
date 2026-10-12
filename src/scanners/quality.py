"""
quality.py — signal quality filters for all SMC scanners.

Implements four gating layers applied after initial detection:

1. Liquidity gate    : skip symbols with avg daily value < min_daily_value_crore
2. Volume gate       : today's volume must be ≥ min_vol_ratio × 20-day average
3. Volatility gate   : ATR(14) as % of price must be in [min_atr_pct, max_atr_pct]
4. Trend + actionable: only emit signals matching market direction AND within entry proximity

All signals are INTRADAY — they are only shown when price is close to the entry level
and the stock has sufficient volume and volatility to trade that day.

Market direction is determined by the fraction of all loaded symbols where
close > SMA20. If > BULLISH_BIAS_THRESHOLD (55%) → BULLISH bias.
If < BEARISH_BIAS_THRESHOLD (45%) → BEARISH bias. Else NEUTRAL.
"""
from __future__ import annotations

import pandas as pd

BULLISH_BIAS_THRESHOLD  = 0.55   # > 55% stocks above SMA20 → bullish market
BEARISH_BIAS_THRESHOLD  = 0.45   # < 45% stocks above SMA20 → bearish market
DEFAULT_SMA_PERIOD      = 50     # symbol-level trend SMA period
DEFAULT_BIAS_SMA        = 20     # market breadth SMA period
DEFAULT_MIN_VALUE_CRORE = 50.0   # minimum avg daily value in crore INR
DEFAULT_MIN_VOL_RATIO   = 0.75   # today's volume must be ≥ 75% of 20-day avg
DEFAULT_MIN_ATR_PCT     = 0.5    # minimum ATR(14)/price — stock must move enough
DEFAULT_MAX_ATR_PCT     = 8.0    # maximum ATR(14)/price — avoid panic/circuit stocks

# Indices don't use equity volume conventions — exempt from volume/volatility gates
_INDEX_SYMBOLS = frozenset({"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"})


def compute_market_direction(
    all_dfs: dict[str, pd.DataFrame],
    sma_period: int = DEFAULT_BIAS_SMA,
) -> str:
    """
    Return "BULLISH", "BEARISH", or "NEUTRAL" based on the fraction of
    symbols trading above their SMA{sma_period}.
    """
    above = 0
    total = 0
    for df in all_dfs.values():
        if len(df) < sma_period:
            continue
        sma = df["close"].rolling(sma_period, min_periods=sma_period).mean()
        if pd.notna(sma.iloc[-1]):
            total += 1
            if float(df["close"].iloc[-1]) > float(sma.iloc[-1]):
                above += 1
    if total == 0:
        return "NEUTRAL"
    ratio = above / total
    if ratio > BULLISH_BIAS_THRESHOLD:
        return "BULLISH"
    if ratio < BEARISH_BIAS_THRESHOLD:
        return "BEARISH"
    return "NEUTRAL"


def compute_symbol_trend(
    df: pd.DataFrame,
    sma_period: int = DEFAULT_SMA_PERIOD,
) -> str:
    """
    Return "BULLISH", "BEARISH", or "NEUTRAL" for a single symbol.

    Rules:
      - BULLISH: close > SMA50 AND close > SMA20 (uptrend confirmed on two SMAs)
      - BEARISH: close < SMA50 AND close < SMA20
      - NEUTRAL: mixed
    """
    if len(df) < sma_period:
        return "NEUTRAL"
    sma50 = df["close"].rolling(sma_period, min_periods=sma_period).mean()
    sma20 = df["close"].rolling(20, min_periods=20).mean()
    close = float(df["close"].iloc[-1])
    s50   = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None
    s20   = float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else None

    if s50 is None or s20 is None:
        return "NEUTRAL"
    if close > s50 and close > s20:
        return "BULLISH"
    if close < s50 and close < s20:
        return "BEARISH"
    return "NEUTRAL"


def passes_liquidity_filter(
    df: pd.DataFrame,
    min_daily_value_crore: float = DEFAULT_MIN_VALUE_CRORE,
) -> bool:
    """
    Return True if the symbol's average daily turnover exceeds the threshold.
    Turnover = avg(volume * close) over last 20 bars, in crore INR (1 crore = 1e7).
    Indices are always liquid — exempt from this gate.
    """
    if len(df) < 5:
        return False
    tail = df.tail(20)
    avg_value = float((tail["volume"] * tail["close"]).mean()) / 1e7  # crore
    return avg_value >= min_daily_value_crore


def passes_intraday_volume(
    df: pd.DataFrame,
    symbol: str = "",
    min_vol_ratio: float = DEFAULT_MIN_VOL_RATIO,
) -> bool:
    """
    Return True if today's volume is at least min_vol_ratio × 20-day average.

    This ensures the stock is actively trading TODAY — a prerequisite for
    intraday entries.  Thin trading days produce false entries and wide
    bid-ask spreads.

    Indices are exempt (their 'volume' column represents futures OI, not
    spot liquidity; they are always considered active).
    """
    if symbol in _INDEX_SYMBOLS:
        return True
    if len(df) < 22:
        return True   # insufficient history — don't block new listings
    # Use the 20 bars BEFORE today so we don't include today in the average
    avg_vol = float(df["volume"].iloc[-21:-1].mean())
    today_vol = float(df["volume"].iloc[-1])
    if avg_vol <= 0:
        return True
    return today_vol >= avg_vol * min_vol_ratio


def passes_intraday_volatility(
    df: pd.DataFrame,
    symbol: str = "",
    min_atr_pct: float = DEFAULT_MIN_ATR_PCT,
    max_atr_pct: float = DEFAULT_MAX_ATR_PCT,
) -> bool:
    """
    Return True if the symbol's ATR(14) as a % of price is within [min, max].

    - Too low  (< 0.5%): stock is too flat — intraday targets won't be hit.
    - Too high (> 8.0%): stock is in a circuit / panic mode — entries are
      unreliable and stop placement is difficult.

    Indices are exempt (always within reasonable range).
    """
    if symbol in _INDEX_SYMBOLS:
        return True
    if len(df) < 15:
        return True
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14, min_periods=7).mean().iloc[-1])
    price = float(df["close"].iloc[-1])
    if price <= 0:
        return True
    atr_pct = atr14 / price * 100
    return min_atr_pct <= atr_pct <= max_atr_pct


def is_signal_trend_aligned(
    signal_direction_is_bullish: bool,
    symbol_trend: str,
    market_direction: str,
) -> bool:
    """
    Return True if the signal direction aligns with the market directional bias.

    Market direction takes priority:
      BULLISH market  → only bullish signals pass
      BEARISH market  → only bearish signals pass
      NEUTRAL market  → signal must also match the symbol's own SMA50+SMA20 trend

    In BULLISH/BEARISH regimes the individual symbol SMA is intentionally ignored
    because SMAs lag and may not yet reflect the market turn.
    """
    if market_direction == "BULLISH" and not signal_direction_is_bullish:
        return False
    if market_direction == "BEARISH" and signal_direction_is_bullish:
        return False

    # In NEUTRAL market only: use symbol's own trend as a tiebreaker
    if market_direction == "NEUTRAL":
        if symbol_trend == "BULLISH" and not signal_direction_is_bullish:
            return False
        if symbol_trend == "BEARISH" and signal_direction_is_bullish:
            return False
        # NEUTRAL symbol in NEUTRAL market: let R:R and actionable filters decide

    return True


def passes_quality_filters(
    item: object,
    symbol_trend: str,
    market_direction: str,
    min_rr_ratio: float = 1.5,
    require_actionable: bool = True,
    max_proximity_pct: float | None = None,
) -> bool:
    """
    Master quality gate for intraday SMC signals.

    Checks (in order):
    1. Trend alignment — signal direction must match market direction
    2. Minimum R:R ratio
    3. Actionable gate — price must be within the entry proximity threshold
       (is_actionable is required for ALL market regimes, not just NEUTRAL)
    4. Optional max_proximity_pct hard cap

    Returns True if all gates pass.
    """
    # ── Direction detection ──────────────────────────────────────────────────
    dirn = (getattr(item, "direction", None)
            or getattr(item, "signal_type", None))
    dirn_val = getattr(dirn, "value", str(dirn)).lower() if dirn else ""
    is_bullish = ("bull" in dirn_val or "demand" in dirn_val)

    # ── Trend alignment ───────────────────────────────────────────────────────
    if not is_signal_trend_aligned(is_bullish, symbol_trend, market_direction):
        return False

    # ── R:R filter ────────────────────────────────────────────────────────────
    rr = getattr(item, "intraday_rr_ratio", 0.0) or 0.0
    if rr < min_rr_ratio:
        return False

    # ── Actionable gate (always enforced for intraday) ────────────────────────
    # For intraday trading price must be near the entry level right now —
    # stale signals where price has already moved away are not tradeable.
    if require_actionable:
        if not getattr(item, "is_actionable", False):
            return False

    # ── Proximity hard cap ───────────────────────────────────────────────────
    if max_proximity_pct is not None:
        prox = getattr(item, "proximity_pct", 999.0) or 999.0
        if prox > max_proximity_pct:
            return False

    return True
