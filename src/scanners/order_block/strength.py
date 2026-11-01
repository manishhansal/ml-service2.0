"""
StrengthFilter — filters order block candidates by impulse move strength.

A candidate passes when its impulse bar body (|close - open| at the impulse bar)
is at least ``config.strength_multiplier`` times the ATR50 baseline at that bar.
Candidates that fail the filter are excluded and a DEBUG log event is emitted
so the caller can trace why a candidate was dropped.

Algorithm
---------
1. Guard: return ``[]`` immediately if ``candidates`` is empty.
2. Compute ATR50 using a **trailing** rolling window (``center=False``,
   ``min_periods=1``) so no future bars contaminate the baseline.
3. Per candidate (keyed by ``impulse_bar_index = j``):
   - ``body = |close[j] − open[j]|``
   - ``atr_val = atr50[j]``
   - Skip (exclude, DEBUG log) if ``atr_val == 0.0`` or ``math.isnan(atr_val)``
   - ``ratio = body / atr_val``
   - Include when ``ratio >= config.strength_multiplier``; else DEBUG-log and exclude
4. Return the filtered ``list[OBCandidateConfirmed]``.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
import structlog

from src.scanners.order_block.config import OrderBlockConfig
from src.scanners.order_block.schemas import OBCandidateConfirmed

# Module-level default logger — tests may override via the ``logger`` parameter.
_logger = structlog.get_logger(__name__)


def filter_by_strength(
    candidates: list[OBCandidateConfirmed],
    df: pd.DataFrame,
    config: OrderBlockConfig,
    logger: Any | None = None,
) -> list[OBCandidateConfirmed]:
    """
    Filter ``candidates`` by impulse move strength relative to ATR50.

    Parameters
    ----------
    candidates:
        Confirmed order block candidates to evaluate.
    df:
        Per-symbol OHLCV ``DataFrame`` with a UTC ``DatetimeIndex``.
    config:
        Scanner parameters; uses ``atr_strength_window`` and ``strength_multiplier``.
    logger:
        Optional structlog bound-logger override.  Defaults to the module-level
        logger when ``None``.

    Returns
    -------
    list[OBCandidateConfirmed]
        Only the candidates whose impulse bar body/ATR50 ratio meets the
        ``config.strength_multiplier`` threshold (inclusive).
    """
    if not candidates:
        return []

    log = logger if logger is not None else _logger

    # ── ATR50 — strictly backward-looking (never center=True) ────────────────
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr50 = tr.rolling(window=config.atr_strength_window, min_periods=1).mean()

    passed: list[OBCandidateConfirmed] = []

    for candidate in candidates:
        j = candidate.impulse_bar_index

        body = abs(float(df["close"].iloc[j]) - float(df["open"].iloc[j]))
        atr_val = float(atr50.iloc[j])

        # Guard: zero or NaN ATR — cannot compute a meaningful ratio
        if atr_val == 0.0 or math.isnan(atr_val):
            log.debug(
                "strength_filter_skipped_zero_atr",
                symbol=candidate.symbol,
                bar_index=j,
                move_strength_ratio=None,
            )
            continue

        ratio = body / atr_val

        if ratio >= config.strength_multiplier:
            passed.append(candidate)
        else:
            log.debug(
                "strength_filter_excluded",
                symbol=candidate.symbol,
                bar_index=j,
                move_strength_ratio=ratio,
            )

    return passed
