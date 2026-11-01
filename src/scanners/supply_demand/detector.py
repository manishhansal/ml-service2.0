"""
SDDetector — identifies Supply & Demand zones from OHLCV consolidation-departure patterns.

Zone types:
  RBR (Rally-Base-Rally) → DEMAND
  DBD (Drop-Base-Drop)   → SUPPLY
  RBD (Rally-Base-Drop)  → SUPPLY
  DBR (Drop-Base-Rally)  → DEMAND
"""
from __future__ import annotations
import pandas as pd
from src.scanners.supply_demand.config import SDConfig
from src.scanners.supply_demand.schemas import SDDirection, SDLifecycle, SDZone, SDZoneType

_OHLC_COLS = ["open", "high", "low", "close"]

_ZONE_TYPE_MAP = {
    (True,  True):  (SDZoneType.RBR, SDDirection.DEMAND),
    (False, False): (SDZoneType.DBD, SDDirection.SUPPLY),
    (True,  False): (SDZoneType.RBD, SDDirection.SUPPLY),
    (False, True):  (SDZoneType.DBR, SDDirection.DEMAND),
}


def detect_sd_zones(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: SDConfig,
) -> list[SDZone]:
    """Detect Supply & Demand zones in an OHLCV series."""
    if len(df) < 5:
        return []
    if df[_OHLC_COLS].isna().all().all():
        return []

    # ATR14 — strictly backward-looking
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=config.atr_window, min_periods=1).mean()

    body = (df["close"] - df["open"]).abs()
    n = len(df)
    results: list[SDZone] = []
    b = 1  # base_start (entry_leg is at b-1)

    while b < n - config.min_base_candles:
        # Require valid entry_leg at b-1
        if not pd.notna(df.iloc[b - 1][_OHLC_COLS]).all():
            b += 1
            continue

        # Scan forward for base candles starting at b
        base_end = b
        while base_end < min(b + config.max_base_candles, n):
            atr_k = float(atr.iloc[base_end]) if float(atr.iloc[base_end]) > 0 else 1.0
            if float(body.iloc[base_end]) >= config.base_body_threshold * atr_k:
                break  # candle too large → base ends here
            if not pd.notna(df.iloc[base_end][_OHLC_COLS]).all():
                break  # NaN ends the base
            base_end += 1

        base_len = base_end - b
        if base_len < config.min_base_candles:
            b += 1
            continue

        dep = base_end  # departure candle index
        if dep >= n:
            break

        # NaN guard for departure and all base candles
        if not pd.notna(df.iloc[dep][_OHLC_COLS]).all():
            b += 1
            continue
        has_nan = any(
            not pd.notna(df.iloc[k][_OHLC_COLS]).all()
            for k in range(b, base_end)
        )
        if has_nan:
            b += 1
            continue

        # Departure body must be impulsive
        atr_dep = float(atr.iloc[dep])
        if atr_dep <= 0:
            b += 1
            continue
        dep_body = float(body.iloc[dep])
        if dep_body < config.departure_body_threshold * atr_dep:
            b += 1
            continue

        # Zone bounds from base candles
        zone_high = float(df["high"].iloc[b:base_end].max())
        zone_low  = float(df["low"].iloc[b:base_end].min())

        # Minimum zone width — compare against the average True Range of the
        # base candles themselves (not atr_dep, which is inflated by the
        # departure's own large move). This gives a fair measure of volatility
        # during the consolidation period.
        base_trs = tr.iloc[b:base_end]
        base_atr = float(base_trs.mean()) if len(base_trs) > 0 and float(base_trs.mean()) > 0 else float(atr_dep)
        if zone_high - zone_low < config.min_zone_width_multiplier * base_atr:
            b += 1
            continue

        # Zone type classification
        entry_bullish = float(df["close"].iloc[b - 1]) > float(df["open"].iloc[b - 1])
        dep_bullish   = float(df["close"].iloc[dep]) > float(df["open"].iloc[dep])
        zone_type, direction = _ZONE_TYPE_MAP[(entry_bullish, dep_bullish)]

        zone = SDZone(
            symbol=symbol,
            direction=direction,
            zone_type=zone_type,
            timeframe=timeframe,
            zone_high=zone_high,
            zone_low=zone_low,
            entry_leg_timestamp=df.index[b - 1],
            departure_timestamp=df.index[dep],
            base_start_bar_index=b,
            base_end_bar_index=base_end - 1,
            departure_bar_index=dep,
        )
        results.append(zone)
        b = dep + 1  # advance past departure

    return results
