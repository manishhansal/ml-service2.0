"""
BreadthTracker — computes market-breadth metrics across all scanned symbols.

For each symbol the tracker counts whether an ACTIVE order block zone exists in
the bullish or bearish direction, then expresses those counts as fractions of the
total symbol universe (``total_symbols``).

Algorithm
---------
1. Guard: ``total_symbols == 0`` → return a zero-breadth ``BreadthResult`` with
   ``market_regime = OBMarketRegime.MIXED``.
2. ``bull_symbols`` = number of symbols that have ≥ 1 zone where
   ``direction == BULLISH`` **and** ``lifecycle_state == ACTIVE``.
3. ``bear_symbols`` = analogous for ``direction == BEARISH`` and ``ACTIVE``.
4. ``bullish_breadth = bull_symbols / total_symbols``
   ``bearish_breadth = bear_symbols / total_symbols``
5. ``market_regime``:
   - ``bullish_breadth > 0.6``  → ``BULLISH_SWEEP``  (strict inequality)
   - ``bearish_breadth > 0.6``  → ``BEARISH_SWEEP``
   - otherwise                  → ``MIXED``
6. ``alignment_scores``: for each symbol in ``zones_by_symbol``:
   - ACTIVE BULLISH zone present → value = ``bullish_breadth``
   - ACTIVE BEARISH zone present → value = ``bearish_breadth``
   - neither                     → symbol not included in the dict
7. Return ``BreadthResult``.
"""
from __future__ import annotations

from src.scanners.order_block.schemas import (
    BreadthResult,
    OBMarketRegime,
    OBZone,
    OBZoneDirection,
    OBZoneLifecycle,
)


def compute_breadth(
    zones_by_symbol: dict[str, list[OBZone]],
    total_symbols: int,
) -> BreadthResult:
    """
    Compute market-breadth metrics from the active order block zones.

    Parameters
    ----------
    zones_by_symbol:
        Mapping of symbol → list of ``OBZone`` objects (may include zones in
        any lifecycle state; only ``ACTIVE`` ones are counted).
    total_symbols:
        The total number of symbols in the scan universe, used as the
        denominator for breadth fractions.

    Returns
    -------
    BreadthResult
        Bullish/bearish breadth fractions, market regime, and per-symbol
        alignment scores.
    """
    # ── Guard: empty universe ────────────────────────────────────────────────
    if total_symbols == 0:
        return BreadthResult(
            bullish_breadth=0.0,
            bearish_breadth=0.0,
            market_regime=OBMarketRegime.MIXED,
            alignment_scores={},
        )

    # ── Count symbols with at least one ACTIVE zone per direction ────────────
    bull_symbols: int = 0
    bear_symbols: int = 0

    for zones in zones_by_symbol.values():
        has_active_bull = any(
            z.direction == OBZoneDirection.BULLISH
            and z.lifecycle_state == OBZoneLifecycle.ACTIVE
            for z in zones
        )
        has_active_bear = any(
            z.direction == OBZoneDirection.BEARISH
            and z.lifecycle_state == OBZoneLifecycle.ACTIVE
            for z in zones
        )
        if has_active_bull:
            bull_symbols += 1
        if has_active_bear:
            bear_symbols += 1

    # ── Breadth fractions ────────────────────────────────────────────────────
    bullish_breadth: float = bull_symbols / total_symbols
    bearish_breadth: float = bear_symbols / total_symbols

    # ── Market regime (strictly > 0.6) ───────────────────────────────────────
    if bullish_breadth > 0.6:
        regime = OBMarketRegime.BULLISH_SWEEP
    elif bearish_breadth > 0.6:
        regime = OBMarketRegime.BEARISH_SWEEP
    else:
        regime = OBMarketRegime.MIXED

    # ── Per-symbol alignment scores ──────────────────────────────────────────
    alignment_scores: dict[str, float] = {}

    for symbol, zones in zones_by_symbol.items():
        has_active_bull = any(
            z.direction == OBZoneDirection.BULLISH
            and z.lifecycle_state == OBZoneLifecycle.ACTIVE
            for z in zones
        )
        if has_active_bull:
            alignment_scores[symbol] = bullish_breadth
            continue

        has_active_bear = any(
            z.direction == OBZoneDirection.BEARISH
            and z.lifecycle_state == OBZoneLifecycle.ACTIVE
            for z in zones
        )
        if has_active_bear:
            alignment_scores[symbol] = bearish_breadth
        # else: symbol has no active zones — omit from alignment_scores

    return BreadthResult(
        bullish_breadth=bullish_breadth,
        bearish_breadth=bearish_breadth,
        market_regime=regime,
        alignment_scores=alignment_scores,
    )
