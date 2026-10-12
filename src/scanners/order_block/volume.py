"""
VolumeRanker — ranks order block zones by relative volume at detection time.

For each zone the volume score is computed as:

    volume_score = ob_volume / avg_volume_N

where ``avg_volume_N`` is the trailing rolling mean of the ``volume`` column
over ``config.avg_volume_window`` bars, evaluated at the bar whose timestamp
matches ``z.ob_timestamp``.

A score > 1.0 means the OB bar traded *above* average volume, which implies
higher conviction behind the zone.

Edge cases
----------
- Symbol not found in ``df_by_symbol`` → ``volume_score = 0.0``
- ``avg_vol_at_ts`` is NaN, zero, or the timestamp is not in the index → ``volume_score = 0.0``
- Empty ``zones`` list → return ``[]``

Sort order
----------
1. Descending ``volume_score`` (highest conviction first).
2. Tie-break: descending ``ob_timestamp`` (most recent zone first).
"""
from __future__ import annotations

import math

import pandas as pd

from src.scanners.order_block.config import OrderBlockConfig
from src.scanners.order_block.schemas import OBZone


def rank_by_volume(
    zones: list[OBZone],
    df_by_symbol: dict[str, pd.DataFrame],
    config: OrderBlockConfig,
) -> list[OBZone]:
    """
    Compute a ``volume_score`` for every zone and return them sorted descending.

    Parameters
    ----------
    zones:
        Fully confirmed OBZone objects to rank.  The list is not mutated
        (``OBZone`` is frozen).
    df_by_symbol:
        Mapping from symbol → OHLCV ``DataFrame`` with a UTC ``DatetimeIndex``.
        Each DataFrame must contain a ``"volume"`` column.
    config:
        Scanner parameters; uses ``avg_volume_window`` (rolling window size).

    Returns
    -------
    list[OBZone]
        The same zones with updated ``volume_score`` fields, sorted by
        descending ``volume_score`` then descending ``ob_timestamp``.
    """
    if not zones:
        return []

    scored: list[OBZone] = []

    for zone in zones:
        df = df_by_symbol.get(zone.symbol)

        if df is None:
            volume_score = 0.0
        else:
            # Trailing rolling mean — strictly backward-looking (min_periods=1
            # ensures we always get a value even at the start of the series)
            avg_vol: pd.Series = df["volume"].rolling(
                window=config.avg_volume_window, min_periods=1
            ).mean()

            # Look up the raw OB-bar volume at the zone's detection timestamp
            try:
                ob_vol_at_ts = df["volume"].loc[zone.ob_timestamp]
            except KeyError:
                ob_vol_at_ts = None

            # Look up the rolling average at the OB bar's timestamp
            try:
                avg_vol_at_ts = avg_vol.loc[zone.ob_timestamp]
            except KeyError:
                avg_vol_at_ts = None

            if (
                ob_vol_at_ts is None
                or avg_vol_at_ts is None
                or (isinstance(avg_vol_at_ts, float) and math.isnan(avg_vol_at_ts))
                or avg_vol_at_ts == 0.0
            ):
                volume_score = 0.0
            else:
                volume_score = float(ob_vol_at_ts) / float(avg_vol_at_ts)

        scored.append(zone.model_copy(update={"volume_score": volume_score}))

    # Sort: primary descending volume_score, tie-break descending ob_timestamp
    return sorted(
        scored,
        key=lambda z: (-z.volume_score, -z.ob_timestamp.timestamp()),
    )
