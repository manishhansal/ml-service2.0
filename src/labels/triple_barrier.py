"""
src.labels.triple_barrier — Triple-barrier label V2.

generate_risk_labels_v2():
  Entry point used by data_pipeline.py generate_risk_labels().

  For each bar i:
    stop   = close[i] - stop_atr_mult  * atr[i]
    target = close[i] + target_atr_mult * atr[i]
    Scan forward `lookforward` bars using high/low.
    If high[j] >= target → TARGET_HIT  → stop_hit=0, target_hit=1
    If low[j]  <= stop   → STOP_HIT    → stop_hit=1, target_hit=0
    If both hit same bar → CONSERVATIVE_SL → stop_hit=1, target_hit=0
    If neither → TIME_EXPIRY   → stop_hit=0, target_hit=0
    MAE = max adverse excursion = max(0, close[i] - min(low[i+1..i+h])) / close[i]
           clipped to [0, 20]

  Tail bars (last `lookforward`): NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_risk_labels_v2(
    df: pd.DataFrame,
    atr_series: pd.Series,
    stop_atr_mult: float = 1.4,
    target_atr_mult: float = 2.0,
    lookforward: int = 20,
    ambiguity_policy: str = "CONSERVATIVE_SL",
    symbol: str = "",
) -> tuple[pd.Series, pd.Series, pd.Series, list]:
    """
    Triple-barrier risk labels.

    Returns:
        stop_hit    : pd.Series[float] — 1.0 if stop hit, 0.0 otherwise, NaN for tail
        target_hit  : pd.Series[float] — 1.0 if target hit
        mae         : pd.Series[float] — max adverse excursion % (clipped 0–20)
        events      : list (event metadata, empty for compat)
    """
    n = len(df)
    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy() if "high" in df.columns else close.copy()
    low = df["low"].astype(float).to_numpy() if "low" in df.columns else close.copy()
    atr = atr_series.astype(float).to_numpy()

    stop_arr = np.full(n, np.nan)
    target_arr = np.full(n, np.nan)
    mae_arr = np.full(n, np.nan)

    for i in range(n - lookforward):
        c = close[i]
        a = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else c * 0.02
        stop_lvl = c - stop_atr_mult * a
        target_lvl = c + target_atr_mult * a

        stop_hit_val = 0.0
        target_hit_val = 0.0
        fw_high = high[i + 1: i + 1 + lookforward]
        fw_low = low[i + 1: i + 1 + lookforward]

        first_stop = next(
            (j for j, l in enumerate(fw_low) if l <= stop_lvl), None
        )
        first_target = next(
            (j for j, h in enumerate(fw_high) if h >= target_lvl), None
        )

        if first_stop is None and first_target is None:
            pass  # TIME_EXPIRY
        elif first_stop is None:
            target_hit_val = 1.0
        elif first_target is None:
            stop_hit_val = 1.0
        else:
            if ambiguity_policy == "CONSERVATIVE_SL" and first_stop <= first_target:
                stop_hit_val = 1.0
            elif first_target < first_stop:
                target_hit_val = 1.0
            else:
                stop_hit_val = 1.0

        # MAE: max adverse excursion
        if len(fw_low) > 0:
            min_low = np.min(fw_low)
            mae_val = max(0.0, (c - min_low) / c * 100.0)
            mae_val = min(mae_val, 20.0)
        else:
            mae_val = 0.0

        stop_arr[i] = stop_hit_val
        target_arr[i] = target_hit_val
        mae_arr[i] = mae_val

    idx = df.index
    return (
        pd.Series(stop_arr, index=idx, dtype=float),
        pd.Series(target_arr, index=idx, dtype=float),
        pd.Series(mae_arr, index=idx, dtype=float),
        [],
    )
