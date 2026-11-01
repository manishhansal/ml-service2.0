"""
ObScanner — top-level orchestrator for the Order Block Scanner pipeline.

Runs every sub-component (Detector → Validator → Filter → Ranker → ZoneManager
→ BreadthTracker → Confluence) against the full F&O symbol universe and returns
a single ``OrderBlockScanResult`` that the ScannerAPI can serve directly.

Supported timeframes: ``"5m"``, ``"15m"``, ``"1h"``, ``"1d"``
Each timeframe has a per-config ``max_confirmation_bars`` override that is applied
before passing the config to ``StructureValidator``.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import structlog

from src.scanners.order_block.breadth import compute_breadth
from src.scanners.order_block.config import OrderBlockConfig
from src.scanners.order_block.detector import detect_candidates
from src.scanners.order_block.schemas import (
    OBCandidateConfirmed,
    OBZone,
    OBZoneDirection,
    OBZoneLifecycle,
    OBMarketRegime,
    OrderBlockScanResult,
    BreadthResult,
)
from src.scanners.order_block.strength import filter_by_strength
from src.scanners.order_block.structure import validate_structure
from src.scanners.order_block.volume import rank_by_volume
from src.scanners.order_block.zone_manager import update_zone_lifecycle

_logger = structlog.get_logger(__name__)

# Supported timeframe values — used for validation and per-TF config overrides.
SUPPORTED_TIMEFRAMES = frozenset({"5m", "15m", "1h", "1d"})


# ── Module-level helper ────────────────────────────────────────────────────────


def _candidate_to_zone(
    c: OBCandidateConfirmed,
    df: pd.DataFrame,
    config: OrderBlockConfig,
) -> OBZone:
    """
    Convert a confirmed ``OBCandidateConfirmed`` into an ``OBZone``.

    ``volume_score`` and ``breadth_alignment_score`` are set to ``0.0`` here
    and overwritten later by ``rank_by_volume`` and the breadth assignment step.
    ``move_strength_ratio`` is computed from the impulse bar body / ATR50.

    Parameters
    ----------
    c:
        The confirmed candidate to convert.
    df:
        Per-symbol OHLCV ``DataFrame`` (same one used for detection).
    config:
        Scanner config; uses ``atr_strength_window`` for ATR50.

    Returns
    -------
    OBZone
        A new ``OBZone`` with ``lifecycle_state = ACTIVE``.
    """
    j = c.impulse_bar_index

    # Compute ATR50 (trailing, no look-ahead) for move_strength_ratio
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr50 = tr.rolling(config.atr_strength_window, min_periods=1).mean()

    atr_val = float(atr50.iloc[j])
    body = abs(float(df["close"].iloc[j]) - float(df["open"].iloc[j]))
    ratio = body / atr_val if atr_val > 0 else 0.0

    return OBZone(
        symbol=c.symbol,
        direction=c.direction,
        timeframe=c.timeframe,
        ob_high=c.ob_high,
        ob_low=c.ob_low,
        ob_timestamp=c.ob_timestamp,
        ob_bar_index=c.ob_bar_index,
        bos_timestamp=c.bos_timestamp,
        bos_bar_index=c.bos_bar_index,
        volume_score=0.0,          # set by rank_by_volume
        move_strength_ratio=ratio,
        breadth_alignment_score=0.0,  # set after breadth computation
    )


# ── ObScanner ─────────────────────────────────────────────────────────────────


class ObScanner:
    """
    Top-level orchestrator for the Order Block Scanner pipeline.

    Parameters
    ----------
    config:
        Optional ``OrderBlockConfig`` instance.  Defaults to a fresh
        ``OrderBlockConfig()`` (reads from env vars / .env).
    """

    def __init__(self, config: OrderBlockConfig | None = None) -> None:
        self.config = config or OrderBlockConfig()

    # ── Public interface ───────────────────────────────────────────────────────

    def scan(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        timeframe: str,
        existing_zones: dict[str, list[OBZone]] | None = None,
    ) -> OrderBlockScanResult:
        """
        Run the full Order Block Scanner pipeline over all symbols.

        Parameters
        ----------
        ohlcv_by_symbol:
            Mapping of symbol → OHLCV ``DataFrame`` (UTC ``DatetimeIndex``,
            columns ``open``, ``high``, ``low``, ``close``, ``volume``).
        timeframe:
            Candle aggregation interval.  Must be one of ``"5m"``, ``"15m"``,
            ``"1h"``, ``"1d"``; raises ``ValueError`` otherwise.
        existing_zones:
            Optional mapping of symbol → existing ``OBZone`` list from the
            previous scan.  If provided, zone lifecycles are updated before
            merging with newly detected zones.

        Returns
        -------
        OrderBlockScanResult
            Ranked, breadth-annotated scan result with zone list.

        Raises
        ------
        ValueError
            If ``timeframe`` is not in ``SUPPORTED_TIMEFRAMES``.
        """
        start_ts = time.time()

        # ── Step 1: Validate timeframe ─────────────────────────────────────────
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. Valid values: 5m, 15m, 1h, 1d"
            )

        # ── Step 2: Build per-timeframe effective config ───────────────────────
        tf_map: dict[str, int] = {
            "5m": self.config.tf_5m_confirmation_bars,
            "15m": self.config.tf_15m_confirmation_bars,
            "1h": self.config.tf_1h_confirmation_bars,
            "1d": self.config.tf_1d_confirmation_bars,
        }
        effective_config = self.config.model_copy(
            update={"max_confirmation_bars": tf_map[timeframe]}
        )

        # ── Step 3: Per-symbol detection pipeline ─────────────────────────────
        # new_zones_by_symbol collects freshly detected zones before any merge
        new_zones_by_symbol: dict[str, list[OBZone]] = {}

        for symbol, df in ohlcv_by_symbol.items():
            try:
                candidates = detect_candidates(df, symbol, timeframe, effective_config)
                confirmed = validate_structure(candidates, df, effective_config)
                filtered = filter_by_strength(confirmed, df, effective_config)
                new_zones = [_candidate_to_zone(c, df, effective_config) for c in filtered]
                new_zones_by_symbol[symbol] = new_zones
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    "scanner_symbol_error",
                    symbol=symbol,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                new_zones_by_symbol[symbol] = []

        # ── Step 4: Lifecycle update for existing zones ────────────────────────
        # Merge updated existing zones (non-INVALIDATED) + new zones
        all_zones_by_symbol: dict[str, list[OBZone]] = {}

        for symbol, df in ohlcv_by_symbol.items():
            existing = (existing_zones or {}).get(symbol, [])
            new_for_sym = new_zones_by_symbol.get(symbol, [])

            if existing and len(df) > 0:
                last = df.iloc[-1]
                bar_idx = len(df) - 1

                updated_existing: list[OBZone] = []
                for z in existing:
                    # Approximate creation bar index from ob_timestamp position in df
                    try:
                        creation_bar_index = df.index.get_loc(z.ob_timestamp)
                        # get_loc may return a slice or array for duplicates; fall back
                        if not isinstance(creation_bar_index, int):
                            creation_bar_index = 0
                    except KeyError:
                        creation_bar_index = 0

                    updated_z = update_zone_lifecycle(
                        z,
                        float(last["low"]),
                        float(last["high"]),
                        float(last["close"]),
                        bar_idx,
                        creation_bar_index,
                        effective_config,
                    )
                    updated_existing.append(updated_z)

                # Keep non-INVALIDATED existing zones and merge with new ones
                non_invalidated = [
                    z for z in updated_existing
                    if z.lifecycle_state != OBZoneLifecycle.INVALIDATED
                ]
                all_zones_by_symbol[symbol] = non_invalidated + new_for_sym
            else:
                all_zones_by_symbol[symbol] = new_for_sym

        # ── Step 5: Flatten all zones ──────────────────────────────────────────
        all_zones: list[OBZone] = [
            z for zones in all_zones_by_symbol.values() for z in zones
        ]

        # ── Step 6: Global volume ranking ──────────────────────────────────────
        all_zones = rank_by_volume(all_zones, ohlcv_by_symbol, effective_config)

        # ── Step 7: Multi-TF confluence ────────────────────────────────────────
        all_zones = self._compute_multi_tf_confluence(all_zones, effective_config)

        # ── Step 8: Breadth computation ────────────────────────────────────────
        # Rebuild per-symbol mapping after ranking / confluence updates
        zones_by_symbol: dict[str, list[OBZone]] = {}
        for z in all_zones:
            zones_by_symbol.setdefault(z.symbol, []).append(z)

        breadth_result = compute_breadth(zones_by_symbol, len(ohlcv_by_symbol))

        # ── Step 9: Assign breadth alignment scores ───────────────────────────
        all_zones = [
            z.model_copy(
                update={
                    "breadth_alignment_score": breadth_result.alignment_scores.get(
                        z.symbol, 0.0
                    )
                }
            )
            for z in all_zones
        ]

        # ── Step 10: Compute intraday trading levels ──────────────────────────
        from src.scanners.intraday import compute_zone_intraday  # local import to avoid circular

        intraday_zones: list[OBZone] = []
        for z in all_zones:
            df = ohlcv_by_symbol.get(z.symbol)
            if df is not None and len(df) > 0:
                current_close = float(df["close"].iloc[-1])
                prev_close = df["close"].shift(1)
                tr = pd.concat(
                    [
                        df["high"] - df["low"],
                        (df["high"] - prev_close).abs(),
                        (df["low"] - prev_close).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                atr = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
                is_bullish = z.direction == OBZoneDirection.BULLISH
                levels = compute_zone_intraday(is_bullish, z.ob_high, z.ob_low, current_close, atr)
                intraday_zones.append(z.model_copy(update=levels))
            else:
                intraday_zones.append(z)
        all_zones = intraday_zones

        # ── Step 11: Scanner-specific pre-filter (volume_score) ───────────────
        all_zones = [z for z in all_zones if z.volume_score >= self.config.min_volume_score]

        # ── Step 12: Quality gate (trend + liquidity + R:R) ───────────────────
        from src.scanners.quality import (
            compute_market_direction,
            compute_symbol_trend,
            passes_liquidity_filter,
            passes_quality_filters,
        )
        _market_dir = compute_market_direction(ohlcv_by_symbol)
        _qf: list[OBZone] = []
        for _z in all_zones:
            _df = ohlcv_by_symbol.get(_z.symbol)
            if _df is None:
                continue
            if not passes_liquidity_filter(_df, self.config.min_daily_value_crore):
                continue
            if self.config.require_trend_alignment:
                _trend = compute_symbol_trend(_df)
                if not passes_quality_filters(_z, _trend, _market_dir, self.config.min_rr_ratio):
                    continue
            elif (_z.intraday_rr_ratio or 0.0) < self.config.min_rr_ratio:
                continue
            _qf.append(_z)
        all_zones = _qf

        # ── Step 13: Log completion ────────────────────────────────────────────
        _logger.info(
            "scan_complete",
            symbols=len(ohlcv_by_symbol),
            n_zones=len(all_zones),
            regime=breadth_result.market_regime.value,
            elapsed_ms=int((time.time() - start_ts) * 1000),
        )

        # ── Step 14: Return result ─────────────────────────────────────────────
        return OrderBlockScanResult(
            zones=all_zones,
            breadth=breadth_result,
            scanned_at=datetime.now(tz=timezone.utc),
            n_symbols_scanned=len(ohlcv_by_symbol),
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_multi_tf_confluence(
        self,
        zones: list[OBZone],
        config: OrderBlockConfig,
    ) -> list[OBZone]:
        """
        Tag zones that have multi-timeframe confluence between ``1h`` and ``1d``.

        For each symbol and direction, pair every ``1h`` zone with every ``1d``
        zone and check whether their midpoints are within
        ``config.confluence_price_pct`` of each other.  Both matched zones are
        updated with ``multi_tf_confluence = True``.

        Midpoint formula: ``(ob_high + ob_low) / 2``
        Proximity check: ``abs(mid_1h - mid_1d) / mid_1d <= confluence_price_pct``

        Parameters
        ----------
        zones:
            Full zone list (any timeframe, any direction).
        config:
            Scanner config; uses ``confluence_price_pct``.

        Returns
        -------
        list[OBZone]
            The same zone list with ``multi_tf_confluence`` updated where
            applicable.  Zones that do not match remain unchanged.
        """
        if not zones:
            return zones

        # Build index: symbol → direction → timeframe → [zone indices]
        # We track indices into the ``zones`` list so we can do an in-place
        # replacement without losing the original ordering.

        # symbol → direction → timeframe → list[int]  (indices into zones)
        from collections import defaultdict

        idx: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for i, z in enumerate(zones):
            idx[z.symbol][z.direction.value]["1h" if z.timeframe == "1h" else z.timeframe].append(i)
            # Re-index properly: only store for 1h and 1d
        # Reset and redo properly
        idx = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for i, z in enumerate(zones):
            idx[z.symbol][z.direction.value][z.timeframe].append(i)

        # Collect indices that should have multi_tf_confluence = True
        confluence_indices: set[int] = set()

        for symbol, dir_map in idx.items():
            for direction, tf_map in dir_map.items():
                zones_1h = tf_map.get("1h", [])
                zones_1d = tf_map.get("1d", [])

                if not zones_1h or not zones_1d:
                    continue

                for i_1h in zones_1h:
                    z1h = zones[i_1h]
                    mid_1h = (z1h.ob_high + z1h.ob_low) / 2.0

                    for i_1d in zones_1d:
                        z1d = zones[i_1d]
                        mid_1d = (z1d.ob_high + z1d.ob_low) / 2.0

                        # Guard against zero mid_1d (degenerate zone)
                        if mid_1d == 0.0:
                            continue

                        proximity = abs(mid_1h - mid_1d) / mid_1d
                        if proximity <= config.confluence_price_pct:
                            confluence_indices.add(i_1h)
                            confluence_indices.add(i_1d)

        if not confluence_indices:
            return zones

        # Rebuild zone list with confluence flags applied
        updated: list[OBZone] = []
        for i, z in enumerate(zones):
            if i in confluence_indices:
                updated.append(z.model_copy(update={"multi_tf_confluence": True}))
            else:
                updated.append(z)

        return updated
