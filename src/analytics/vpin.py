"""
VPIN — Volume-synchronized Probability of Informed Trading.

Algorithm:
1. Aggregate OHLCV bars into equal-volume buckets.
2. Estimate buy/sell volume within each bucket using the tick rule:
     close > open → 85% buy volume
     close < open → 15% buy volume (85% sell)
     close == open → 50% buy / 50% sell
3. VPIN = mean(|V_buy - V_sell| / V_bucket) over a rolling window of
   the last n_buckets *completed* buckets.

Only completed buckets (volume == bucket_size) contribute to vpin_series.
Partial trailing buckets are excluded from the series and VPIN calculation.

Classification thresholds:
  - toxic:    vpin >= 0.7
  - elevated: 0.3 <= vpin < 0.7
  - benign:   vpin < 0.3

Validates: Requirements 15.5
"""
from __future__ import annotations

from typing import Any


def compute_vpin(
    bars: list[dict[str, Any]],
    bucket_size: float = 50.0,
    n_buckets: int = 50,
) -> dict[str, Any]:
    """
    Compute VPIN from a list of OHLCV bars.

    Parameters
    ----------
    bars        : List of {open, high, low, close, volume} dicts.
    bucket_size : Target volume per bucket. Only filled buckets (reaching
                  exactly bucket_size volume) contribute to vpin_series.
    n_buckets   : Rolling window of completed buckets used to compute
                  current_vpin (mean of last n_buckets bucket VPINs).

    Returns
    -------
    dict with keys:
        "current_vpin"  : float — mean VPIN over the last n_buckets completed
                          buckets; 0.0 if no bucket has been completed.
        "vpin_series"   : list[float] — per-completed-bucket VPIN values.
        "buckets"       : list[dict] — all completed volume buckets with
                          buy/sell split and per-bucket VPIN.
    """
    if not bars:
        return {"current_vpin": 0.0, "vpin_series": [], "buckets": []}

    completed_buckets: list[dict] = []

    # Accumulator for the bucket currently being filled
    current_buy = 0.0
    current_sell = 0.0
    current_total = 0.0

    for bar in bars:
        volume = float(bar.get("volume", 0) or 0)
        if volume <= 0:
            continue

        open_price = float(bar.get("open", 0) or 0)
        close_price = float(bar.get("close", 0) or 0)

        # Tick rule: assign buy/sell fraction
        if close_price > open_price:
            buy_frac = 0.85
        elif close_price < open_price:
            buy_frac = 0.15
        else:
            buy_frac = 0.50

        bar_buy = volume * buy_frac
        bar_sell = volume * (1.0 - buy_frac)
        remaining_buy = bar_buy
        remaining_sell = bar_sell
        remaining_total = volume

        while remaining_total > 1e-12:
            space_in_bucket = bucket_size - current_total

            if space_in_bucket <= 1e-12:
                # Flush completed bucket
                bucket_vpin = (
                    abs(current_buy - current_sell) / current_total
                    if current_total > 0
                    else 0.0
                )
                completed_buckets.append(
                    {
                        "buy_volume": current_buy,
                        "sell_volume": current_sell,
                        "total_volume": current_total,
                        "imbalance": abs(current_buy - current_sell),
                        "vpin": bucket_vpin,
                    }
                )
                current_buy = 0.0
                current_sell = 0.0
                current_total = 0.0
                space_in_bucket = bucket_size

            fill = min(remaining_total, space_in_bucket)
            frac = fill / remaining_total

            current_buy += remaining_buy * frac
            current_sell += remaining_sell * frac
            current_total += fill

            remaining_buy *= 1.0 - frac
            remaining_sell *= 1.0 - frac
            remaining_total -= fill

            # Flush if bucket is exactly full
            if abs(current_total - bucket_size) < 1e-9:
                bucket_vpin = (
                    abs(current_buy - current_sell) / current_total
                    if current_total > 0
                    else 0.0
                )
                completed_buckets.append(
                    {
                        "buy_volume": current_buy,
                        "sell_volume": current_sell,
                        "total_volume": current_total,
                        "imbalance": abs(current_buy - current_sell),
                        "vpin": bucket_vpin,
                    }
                )
                current_buy = 0.0
                current_sell = 0.0
                current_total = 0.0

    # Partial trailing bucket is intentionally excluded from the series.

    if not completed_buckets:
        return {"current_vpin": 0.0, "vpin_series": [], "buckets": completed_buckets}

    vpin_series = [b["vpin"] for b in completed_buckets]

    # current_vpin = mean of last n_buckets bucket VPINs
    window = vpin_series[-n_buckets:]
    current_vpin = sum(window) / len(window) if window else 0.0
    current_vpin = min(1.0, current_vpin)

    return {
        "current_vpin": current_vpin,
        "vpin_series": vpin_series,
        "buckets": completed_buckets,
    }
