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

Performance contract (Req 16.5):
  - Endpoint p95 < 100ms for payloads up to 500 bars.
  - bucket_size is auto-scaled to total_volume / (n_buckets * 2) when the
    raw bucket_size would produce > MAX_BUCKETS completed buckets, so the
    function never iterates over more than MAX_BUCKETS items regardless of
    the input volume scale.

Classification thresholds:
  - toxic:    vpin >= 0.7
  - elevated: 0.3 <= vpin < 0.7
  - benign:   vpin < 0.3

Validates: Requirements 15.5
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Hard cap on completed buckets — prevents O(volume/bucket_size) slowdowns
# when callers use small bucket_size relative to bar volume.
MAX_BUCKETS: int = 500


def compute_vpin(
    bars: list[dict[str, Any]],
    bucket_size: float = 50.0,
    n_buckets: int = 50,
) -> dict[str, Any]:
    """
    Compute VPIN from a list of OHLCV bars (numpy-vectorised, O(N_bars)).

    ``bucket_size`` is auto-scaled when it would produce more than
    ``MAX_BUCKETS`` completed buckets, keeping runtime predictable for
    any volume scale.

    Parameters
    ----------
    bars        : List of {open, high, low, close, volume} dicts.
    bucket_size : Target volume per bucket.  Auto-scaled when needed.
    n_buckets   : Rolling window of completed buckets for current_vpin.

    Returns
    -------
    dict with keys:
        "current_vpin"  : float
        "vpin_series"   : list[float]
        "buckets"       : list[dict]
    """
    if not bars:
        return {"current_vpin": 0.0, "vpin_series": [], "buckets": []}

    # ── 1. Extract arrays ────────────────────────────────────────────────────
    n = len(bars)
    volumes = np.empty(n, dtype=np.float64)
    opens   = np.empty(n, dtype=np.float64)
    closes  = np.empty(n, dtype=np.float64)

    for i, bar in enumerate(bars):
        volumes[i] = float(bar.get("volume", 0) or 0)
        opens[i]   = float(bar.get("open",   0) or 0)
        closes[i]  = float(bar.get("close",  0) or 0)

    # Drop zero-volume bars
    mask    = volumes > 0
    volumes = volumes[mask]
    opens   = opens[mask]
    closes  = closes[mask]

    if volumes.size == 0:
        return {"current_vpin": 0.0, "vpin_series": [], "buckets": []}

    # ── 2. Auto-scale bucket_size to cap completed bucket count ─────────────
    total_volume = float(volumes.sum())
    effective_bucket_size = max(bucket_size, total_volume / MAX_BUCKETS)

    # ── 3. Tick rule ─────────────────────────────────────────────────────────
    buy_fracs = np.where(closes > opens, 0.85,
                np.where(closes < opens, 0.15, 0.50))
    buy_vols  = volumes * buy_fracs
    sell_vols = volumes * (1.0 - buy_fracs)

    # ── 4. Assign each bar to its bucket via cumulative volume ───────────────
    cum_vol      = np.cumsum(volumes)
    bucket_index = (cum_vol / effective_bucket_size).astype(np.int64)

    # Number of completed buckets (exclude the partial trailing one)
    total_buckets_raw = int(total_volume / effective_bucket_size)
    last_complete = total_buckets_raw

    # A bar partially in the last complete bucket and partially beyond:
    # we use fractional assignment below.
    if last_complete <= 0:
        return {"current_vpin": 0.0, "vpin_series": [], "buckets": []}

    # ── 5. Accumulate buy/sell per completed bucket ───────────────────────────
    # For bars that fall entirely within one completed bucket: direct add.
    # For bars that cross a bucket boundary: proportional split.
    bucket_buy  = np.zeros(last_complete, dtype=np.float64)
    bucket_sell = np.zeros(last_complete, dtype=np.float64)

    cum_vol_start = np.concatenate([[0.0], cum_vol[:-1]])
    b_end_idx     = bucket_index  # bucket index at bar's trailing edge

    for i in range(len(volumes)):
        b_start = int(cum_vol_start[i] / effective_bucket_size)
        b_end   = int(min(b_end_idx[i], last_complete - 1))

        if b_start >= last_complete:
            continue  # bar is entirely in the partial trailing bucket

        if b_start == b_end:
            # Bar fully within one bucket
            bucket_buy[b_start]  += buy_vols[i]
            bucket_sell[b_start] += sell_vols[i]
        else:
            # Bar spans multiple buckets — split by volume fraction
            v = volumes[i]
            cv0 = cum_vol_start[i]
            for b in range(b_start, min(b_end + 1, last_complete)):
                lo = b       * effective_bucket_size
                hi = (b + 1) * effective_bucket_size
                # How much of this bar's volume falls in bucket b?
                overlap_start = max(cv0, lo)
                overlap_end   = min(cv0 + v, hi)
                overlap       = max(0.0, overlap_end - overlap_start)
                if overlap <= 0:
                    continue
                frac = overlap / v if v > 0 else 0.0
                bucket_buy[b]  += buy_vols[i]  * frac
                bucket_sell[b] += sell_vols[i] * frac

    # ── 6. Per-bucket VPIN ───────────────────────────────────────────────────
    bucket_total = bucket_buy + bucket_sell
    safe_total   = np.where(bucket_total > 0, bucket_total, 1.0)
    bucket_vpin  = np.abs(bucket_buy - bucket_sell) / safe_total
    bucket_vpin  = np.where(bucket_total > 0, bucket_vpin, 0.0)

    # ── 7. Current VPIN (rolling mean over last n_buckets) ───────────────────
    window_vpins = bucket_vpin[-n_buckets:]
    current_vpin = float(window_vpins.mean()) if window_vpins.size > 0 else 0.0
    current_vpin = min(1.0, current_vpin)

    # ── 8. Output structures ─────────────────────────────────────────────────
    vpin_series = bucket_vpin.tolist()
    buckets = [
        {
            "buy_volume":   float(bucket_buy[i]),
            "sell_volume":  float(bucket_sell[i]),
            "total_volume": float(bucket_total[i]),
            "imbalance":    float(abs(bucket_buy[i] - bucket_sell[i])),
            "vpin":         float(bucket_vpin[i]),
        }
        for i in range(last_complete)
    ]

    return {
        "current_vpin": current_vpin,
        "vpin_series":  vpin_series,
        "buckets":      buckets,
    }
