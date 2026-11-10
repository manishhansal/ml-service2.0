"""
ScannerAPI — FastAPI router for Smart Money scanner endpoints.

Exposes:
  - ``GET /v2/scanners/order-blocks``      – Order Block zones
  - ``GET /v2/scanners/fair-value-gaps``   – Fair Value Gap zones
  - ``GET /v2/scanners/supply-demand``     – Supply & Demand zones
  - ``GET /v2/scanners/liquidity-sweeps``  – Liquidity Sweep signals
  - ``GET /v2/scanners/market-structure``  – Market Structure (BoS/ChoCh) signals
  - ``GET /v2/scanners/registry``          – Live registry with signal timestamps

Authentication is handled globally by ``api_key_middleware`` in ``main.py``;
this router adds zero auth code.

HTTP behaviour:
  - 200: success (even if zones/signals list is empty — never 404)
  - 401: missing/invalid X-API-KEY (enforced by global middleware)
  - 422: invalid query parameters (FastAPI Query validation auto-raises this)
"""
from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from src.scanners.order_block.config import OrderBlockConfig
from src.scanners.order_block.schemas import OBZoneLifecycle, OrderBlockScanResponse
from src.scanners.order_block.scanner import ObScanner
from src.scanners.fair_value_gap.config import FVGConfig
from src.scanners.fair_value_gap.schemas import FVGLifecycle, FVGScanResponse
from src.scanners.fair_value_gap.scanner import FVGScanner
from src.scanners.supply_demand.config import SDConfig
from src.scanners.supply_demand.schemas import SDLifecycle, SDScanResponse
from src.scanners.supply_demand.scanner import SDScanner
from src.scanners.liquidity_sweep.config import SweepConfig
from src.scanners.liquidity_sweep.schemas import SweepLifecycle, SweepScanResponse
from src.scanners.liquidity_sweep.scanner import SweepScanner
from src.scanners.market_structure.config import MSConfig
from src.scanners.market_structure.schemas import MSScanResponse, MSSignalType
from src.scanners.market_structure.scanner import MSScanner
from src.scanners.data_loader import load_1d_data
from src.scanners import registry as signal_registry

import time as _time
from typing import Any

router = APIRouter(tags=["scanners"])

VALID_TIMEFRAMES = frozenset({"5m", "15m", "1h", "1d"})
VALID_SD_DIRECTIONS = frozenset({"supply", "demand"})
VALID_SD_ZONE_TYPES = frozenset({"RBR", "DBD", "RBD", "DBR"})
VALID_MS_SIGNAL_TYPES = frozenset({"BOS_BULLISH", "BOS_BEARISH", "CHOCH_BULLISH", "CHOCH_BEARISH"})

# ── Scan result cache ────────────────────────────────────────────────────────
# Prevents expensive re-scans within a 5-minute window. Consistent with the
# background task cadence (every 5 min during market hours in main.py).
_SCAN_CACHE: dict[str, tuple[float, Any]] = {}
_SCAN_CACHE_TTL = 300  # seconds


def _cached_scan(key: str, scanner: Any, dfs: dict, timeframe: str) -> Any:
    """Return cached scan result if fresh, otherwise run scanner and cache."""
    now = _time.monotonic()
    cached = _SCAN_CACHE.get(key)
    if cached and (now - cached[0]) < _SCAN_CACHE_TTL:
        return cached[1]
    result = scanner.scan(dfs, timeframe=timeframe)
    _SCAN_CACHE[key] = (now, result)
    return result


_config = OrderBlockConfig()
_scanner = ObScanner(config=_config)

_fvg_config = FVGConfig()
_fvg_scanner = FVGScanner(config=_fvg_config)

_sd_config = SDConfig()
_sd_scanner = SDScanner(config=_sd_config)

_sweep_config = SweepConfig()
_sweep_scanner = SweepScanner(config=_sweep_config)

_ms_config = MSConfig()
_ms_scanner = MSScanner(config=_ms_config)

_MS_BARS = 250   # limit MS lookback for speed (≈ 1 year of daily data)


def _fetch_ohlcv(timeframe: str) -> dict:
    """
    Load OHLCV data from local parquet files.

    For the 1d timeframe (the primary scanner timeframe) this reads from
    ``data/1d/1d/*.parquet``.  Other timeframes are not yet supported and
    fall back to 1d data so scanners always return results.
    """
    # All scanners operate on daily data. Intraday timeframes are not yet
    # available (no 5m/15m/1h parquets), so any timeframe request falls
    # through to the 1d data — the scanner will tag zones with the requested
    # timeframe label for client display purposes.
    return load_1d_data()


# ── Order Block Scanner ─────────────────────────────────────────────────────

@router.get("/scanners/order-blocks")
async def get_order_blocks(
    limit: int = Query(default=50, ge=1, le=200),
    direction: str | None = Query(default=None),
    timeframe: str | None = Query(default=None),
    include_invalidated: bool = Query(default=False),
    actionable_only: bool = Query(default=False),
) -> JSONResponse:
    """
    Return ranked confirmed order block zones across the F&O universe.

    Query parameters:
      - limit (int, 1-200, default 50): max zones to return
      - direction ("bullish"|"bearish", optional): filter by direction
      - timeframe ("5m"|"15m"|"1h"|"1d", optional): filter by timeframe
      - include_invalidated (bool, default false): include INVALIDATED zones
      - actionable_only (bool, default false): only return zones within 2% of their entry price

    Auth: X-API-KEY header (enforced globally by api_key_middleware in main.py)
    422 returned automatically by FastAPI when limit < 1 or > 200.
    """
    # Validate direction
    if direction is not None and direction not in ("bullish", "bearish"):
        return JSONResponse(
            status_code=422,
            content={"detail": f"direction must be 'bullish' or 'bearish', got '{direction}'"},
        )
    # Validate timeframe
    if timeframe is not None and timeframe not in VALID_TIMEFRAMES:
        return JSONResponse(
            status_code=422,
            content={
                "detail": (
                    f"timeframe must be one of {sorted(VALID_TIMEFRAMES)}, got '{timeframe}'"
                )
            },
        )

    # Fetch OHLCV data — loads 1d parquets for the full F&O universe
    ohlcv_by_symbol = _fetch_ohlcv(timeframe or "1d")

    # Run the full scanner pipeline (cached for 5 min to avoid re-scan cost)
    result = _cached_scan("order_block", _scanner, ohlcv_by_symbol, timeframe or "1d")

    # Apply filters
    zones = result.zones
    if not include_invalidated:
        zones = [z for z in zones if z.lifecycle_state != OBZoneLifecycle.INVALIDATED]
    if direction:
        zones = [z for z in zones if z.direction.value == direction]
    if timeframe:
        zones = [z for z in zones if z.timeframe == timeframe]
    if actionable_only:
        zones = [z for z in zones if z.is_actionable]
    zones = zones[:limit]

    # Build response
    response = OrderBlockScanResponse(
        scanned_at=result.scanned_at,
        n_symbols_scanned=result.n_symbols_scanned,
        market_regime=result.breadth.market_regime,
        bullish_breadth=result.breadth.bullish_breadth,
        bearish_breadth=result.breadth.bearish_breadth,
        n_zones=len(zones),
        zones=zones,
    )
    return JSONResponse(content=response.model_dump(mode="json"))


# ── Fair Value Gap Scanner ──────────────────────────────────────────────────

@router.get("/scanners/fair-value-gaps")
async def get_fair_value_gaps(
    limit: int = Query(default=50, ge=1, le=200),
    direction: str | None = Query(default=None),
    timeframe: str | None = Query(default=None),
    include_filled: bool = Query(default=False),
    actionable_only: bool = Query(default=False),
) -> JSONResponse:
    """Return ranked active Fair Value Gap zones across the F&O universe."""
    if direction is not None and direction not in ("bullish", "bearish"):
        return JSONResponse(
            status_code=422,
            content={"detail": f"direction must be 'bullish' or 'bearish', got '{direction}'"},
        )
    if timeframe is not None and timeframe not in VALID_TIMEFRAMES:
        return JSONResponse(
            status_code=422,
            content={"detail": f"timeframe must be one of {sorted(VALID_TIMEFRAMES)}, got '{timeframe}'"},
        )

    ohlcv_by_symbol = _fetch_ohlcv(timeframe or "1d")
    result = _cached_scan("fvg", _fvg_scanner, ohlcv_by_symbol, timeframe or "1d")

    zones = result.zones
    if not include_filled:
        zones = [z for z in zones if z.lifecycle_state not in (FVGLifecycle.FILLED, FVGLifecycle.EXPIRED)]
    if direction:
        zones = [z for z in zones if z.direction.value == direction]
    if timeframe:
        zones = [z for z in zones if z.timeframe == timeframe]
    if actionable_only:
        zones = [z for z in zones if z.is_actionable]
    zones = zones[:limit]

    response = FVGScanResponse(
        scanned_at=result.scanned_at,
        n_symbols_scanned=result.n_symbols_scanned,
        n_zones=len(zones),
        zones=zones,
    )
    return JSONResponse(content=response.model_dump(mode="json"))


# ── Supply & Demand Scanner ─────────────────────────────────────────────────

@router.get("/scanners/supply-demand")
async def get_supply_demand(
    limit: int = Query(default=50, ge=1, le=200),
    direction: str | None = Query(default=None),
    timeframe: str | None = Query(default=None),
    include_broken: bool = Query(default=False),
    zone_type: str | None = Query(default=None),
    actionable_only: bool = Query(default=False),
) -> JSONResponse:
    """Return ranked Supply & Demand zones across the F&O universe."""
    if direction is not None and direction not in VALID_SD_DIRECTIONS:
        return JSONResponse(
            status_code=422,
            content={"detail": f"direction must be 'supply' or 'demand', got '{direction}'"},
        )
    if timeframe is not None and timeframe not in VALID_TIMEFRAMES:
        return JSONResponse(
            status_code=422,
            content={"detail": f"timeframe must be one of {sorted(VALID_TIMEFRAMES)}, got '{timeframe}'"},
        )
    if zone_type is not None and zone_type not in VALID_SD_ZONE_TYPES:
        return JSONResponse(
            status_code=422,
            content={"detail": f"zone_type must be one of {sorted(VALID_SD_ZONE_TYPES)}, got '{zone_type}'"},
        )

    ohlcv_by_symbol = _fetch_ohlcv(timeframe or "1d")
    result = _cached_scan("supply_demand", _sd_scanner, ohlcv_by_symbol, timeframe or "1d")

    zones = result.zones
    if not include_broken:
        zones = [z for z in zones if z.lifecycle_state != SDLifecycle.BROKEN]
    if direction:
        zones = [z for z in zones if z.direction.value.lower() == direction.lower()]
    if timeframe:
        zones = [z for z in zones if z.timeframe == timeframe]
    if zone_type:
        zones = [z for z in zones if z.zone_type.value == zone_type]
    if actionable_only:
        zones = [z for z in zones if z.is_actionable]
    zones = zones[:limit]

    response = SDScanResponse(
        scanned_at=result.scanned_at,
        n_symbols_scanned=result.n_symbols_scanned,
        n_zones=len(zones),
        zones=zones,
    )
    return JSONResponse(content=response.model_dump(mode="json"))


# ── Liquidity Sweep Scanner ─────────────────────────────────────────────────

@router.get("/scanners/liquidity-sweeps")
async def get_liquidity_sweeps(
    limit: int = Query(default=50, ge=1, le=200),
    direction: str | None = Query(default=None),
    timeframe: str | None = Query(default=None),
    min_conviction: float | None = Query(default=None),
    actionable_only: bool = Query(default=False),
) -> JSONResponse:
    """Return ranked Liquidity Sweep signals across the F&O universe."""
    if direction is not None and direction not in ("bullish", "bearish"):
        return JSONResponse(
            status_code=422,
            content={"detail": f"direction must be 'bullish' or 'bearish', got '{direction}'"},
        )
    if timeframe is not None and timeframe not in VALID_TIMEFRAMES:
        return JSONResponse(
            status_code=422,
            content={"detail": f"timeframe must be one of {sorted(VALID_TIMEFRAMES)}, got '{timeframe}'"},
        )

    ohlcv_by_symbol = _fetch_ohlcv(timeframe or "1d")
    result = _cached_scan("liquidity_sweep", _sweep_scanner, ohlcv_by_symbol, timeframe or "1d")

    signals = result.signals
    signals = [s for s in signals if s.lifecycle_state != SweepLifecycle.INVALIDATED]
    if direction:
        signals = [s for s in signals if s.direction.value == direction]
    if timeframe:
        signals = [s for s in signals if s.timeframe == timeframe]
    if min_conviction is not None:
        signals = [s for s in signals if s.conviction_score >= min_conviction]
    if actionable_only:
        signals = [s for s in signals if s.is_actionable]
    signals = signals[:limit]

    response = SweepScanResponse(
        scanned_at=result.scanned_at,
        n_symbols_scanned=result.n_symbols_scanned,
        n_signals=len(signals),
        signals=signals,
    )
    return JSONResponse(content=response.model_dump(mode="json"))


# ── Market Structure Scanner ─────────────────────────────────────────────────

@router.get("/scanners/market-structure")
async def get_market_structure(
    limit: int = Query(default=50, ge=1, le=200),
    timeframe: str | None = Query(default=None),
    signal_type: str | None = Query(default=None),
    actionable_only: bool = Query(default=False),
) -> JSONResponse:
    """Return ranked Market Structure signals (BoS/ChoCh) across the F&O universe."""
    if timeframe is not None and timeframe not in VALID_TIMEFRAMES:
        return JSONResponse(
            status_code=422,
            content={"detail": f"timeframe must be one of {sorted(VALID_TIMEFRAMES)}, got '{timeframe}'"},
        )
    if signal_type is not None and signal_type not in VALID_MS_SIGNAL_TYPES:
        return JSONResponse(
            status_code=422,
            content={"detail": f"signal_type must be one of {sorted(VALID_MS_SIGNAL_TYPES)}, got '{signal_type}'"},
        )

    ohlcv_by_symbol = _fetch_ohlcv(timeframe or "1d")
    # Limit MS to last 250 bars for speed (≈ 1 year of daily data)
    ms_dfs = {sym: df.iloc[-_MS_BARS:] if len(df) > _MS_BARS else df
              for sym, df in ohlcv_by_symbol.items()}
    result = _cached_scan("market_structure", _ms_scanner, ms_dfs, timeframe or "1d")

    signals = result.signals
    signals = [s for s in signals if not s.invalidated]
    if timeframe:
        signals = [s for s in signals if s.timeframe == timeframe]
    if signal_type:
        signals = [s for s in signals if s.signal_type.value == signal_type]
    if actionable_only:
        signals = [s for s in signals if s.is_actionable]
    signals = signals[:limit]

    response = MSScanResponse(
        scanned_at=result.scanned_at,
        n_symbols_scanned=result.n_symbols_scanned,
        n_signals=len(signals),
        signals=signals,
    )
    return JSONResponse(content=response.model_dump(mode="json"))


# ── Signal Registry ─────────────────────────────────────────────────────────

@router.get("/scanners/registry")
async def get_scanner_registry(
    request: Request,
    scanner_type: str | None = Query(default=None),
    include_closed: bool = Query(default=False),
    date: str | None = Query(default=None, description="Closed signals date (YYYY-MM-DD IST, default today)"),
    limit: int = Query(default=200, ge=1, le=2000),
) -> JSONResponse:
    """
    Return the live signal registry — all active signals with timestamps.

    Each record includes:
      - ``id``:             stable signal ID (survives re-scans)
      - ``scanner_type``:   "order_block" | "fvg" | "supply_demand" | "liquidity_sweep" | "market_structure"
      - ``symbol``, ``direction``, ``entry``, ``target``, ``stop``, ``rr``
      - ``proximity_pct``:  % distance from current close to entry
      - ``is_actionable``:  True when within 2 % of entry
      - ``is_choch``:       True for ChoCh MS signals
      - ``first_seen_at``:  ISO 8601 UTC when signal first appeared
      - ``last_seen_at``:   ISO 8601 UTC when signal was last confirmed
      - ``closed_at``:      ISO 8601 UTC when signal closed (null if active)
      - ``close_reason``:   "expired" | "invalidated" | "eod" | null

    Query parameters:
      - ``scanner_type``    (optional): filter by scanner type
      - ``include_closed``  (bool):     also return today's closed signals
      - ``date``            (optional): closed signals date (default today IST)
      - ``limit``           (int):      max total records returned
    """
    cache = request.app.state.cache

    active  = await signal_registry.get_active(cache)
    closed: list[dict] = []
    if include_closed:
        closed = await signal_registry.get_closed(cache, date=date)

    # Apply scanner_type filter
    if scanner_type:
        active = [r for r in active if r.get("scanner_type") == scanner_type]
        closed = [r for r in closed if r.get("scanner_type") == scanner_type]

    meta = await signal_registry.get_last_scan_meta(cache)

    return JSONResponse(content={
        "scanned_at":        meta.get("scanned_at") if meta else None,
        "n_active":          len(active),
        "n_closed_today":    len(closed),
        "last_scan_meta":    meta,
        "active_signals":    active[:limit],
        "closed_signals":    closed[:limit],
    })
