"""
FastAPI application entry-point for ml-service2.0.

Lifecycle:
    startup  — configure structlog, verify audit-log integrity, connect Redis
    shutdown — close Redis pool

Middleware (applied innermost → outermost):
    1. X-Request-ID   — generate/propagate trace ID; bind to structlog context
    2. X-API-KEY      — authenticate all non-exempt paths (returns 401 on failure)
    3. CORSMiddleware  — allow origins from settings.allowed_origins

Exception handlers:
    RequestValidationError  → 422  (Pydantic V2 field-level errors)
    Exception               → 500  (unhandled — logged at ERROR level)

Usage::

    uvicorn src.main:app --host 0.0.0.0 --port 8100 --workers 4
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.audit.logger import AuditLogger, AuditLogViolation
from src.cache.redis_cache import RedisCache
from src.config import settings
from src.logging_config import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)

logger = get_logger(__name__)

# ── Module-level singletons — created once, shared across all requests ─────────
cache = RedisCache()
audit = AuditLogger()


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan: startup → yield → shutdown.

    Startup sequence:
        1. Configure structlog at the log level from settings.
        2. Verify audit log integrity — refuse to start if tampered.
        3. Connect to Redis.
        4. Expose cache + audit on ``app.state`` for use in route handlers.

    Shutdown sequence:
        1. Disconnect from Redis.
    """
    # 1. Structured logging — idempotent, safe to call multiple times
    configure_logging(settings.log_level)

    # 2. Audit-log integrity check — abort on tamper detection
    try:
        audit._check_append_only()
    except AuditLogViolation as exc:
        logger.critical("audit_log_violation_at_startup", error=str(exc))
        raise  # refuse to start with a tampered audit log

    # 3. Redis
    await cache.connect()

    # 4. Expose singletons on app.state so route handlers can reach them
    app.state.cache = cache
    app.state.audit = audit

    logger.info(
        "ml_service_started",
        port=settings.port,
        deployment_mode=settings.deployment_mode.value,
        version="2.0.0",
    )

    # ── File-watcher: auto-broadcast scores whenever latest_scores.json changes ──
    # Polls mtime every 2 s (negligible CPU). When the autorun writes a new
    # snapshot (every ~30 s), we push it to all connected WebSocket clients
    # immediately — no polling from the browser needed.
    import asyncio as _aio
    import json as _json
    from pathlib import Path as _Path

    _LATEST_PATH_WATCH = _Path(__file__).parent.parent / "artifacts" / "live_session" / "latest_scores.json"

    async def _watch_and_broadcast() -> None:
        last_mtime: float = 0.0
        while True:
            try:
                if _LATEST_PATH_WATCH.exists():
                    mtime = _LATEST_PATH_WATCH.stat().st_mtime
                    if mtime != last_mtime:
                        last_mtime = mtime
                        try:
                            snapshot = _json.loads(_LATEST_PATH_WATCH.read_text())
                            await _signal_streamer.broadcast_scores(snapshot)
                        except Exception as _be:
                            logger.debug("broadcast_scores_error", error=str(_be))
            except Exception:
                pass
            await _aio.sleep(2)   # check every 2 s

    _watcher_task = _aio.create_task(_watch_and_broadcast())
    logger.info("signal_file_watcher_started", path=str(_LATEST_PATH_WATCH))

    # ── Live SMC scanner: re-run all 5 scanners every 5 minutes during market hours ──
    # Scans are run in a thread pool so the CPU-bound scanner work doesn't
    # block the event loop.  Results are persisted to the Redis signal registry
    # so the /v2/scanners/registry endpoint always returns fresh data.

    async def _live_smc_scanner_task() -> None:
        import asyncio as _a
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td, time as _t
        from src.scanners.data_loader import load_1d_data
        from src.scanners import registry as _registry
        from src.scanners.order_block.config import OrderBlockConfig
        from src.scanners.order_block.scanner import ObScanner
        from src.scanners.fair_value_gap.config import FVGConfig
        from src.scanners.fair_value_gap.scanner import FVGScanner
        from src.scanners.supply_demand.config import SDConfig
        from src.scanners.supply_demand.scanner import SDScanner
        from src.scanners.liquidity_sweep.config import SweepConfig
        from src.scanners.liquidity_sweep.scanner import SweepScanner
        from src.scanners.market_structure.config import MSConfig
        from src.scanners.market_structure.scanner import MSScanner

        _IST       = _tz(_td(hours=5, minutes=30))
        _INTERVAL  = 300   # seconds between scans
        _MS_BARS   = 250   # limit MS lookback

        _ob_scanner    = ObScanner(OrderBlockConfig())
        _fvg_scanner   = FVGScanner(FVGConfig())
        _sd_scanner    = SDScanner(SDConfig())
        _sweep_scanner = SweepScanner(SweepConfig())
        _ms_scanner    = MSScanner(MSConfig())

        # Track whether we've done the morning clear for today
        _cleared_date_ref: list[str | None] = [None]   # mutable container for nonlocal

        def _is_market_hours() -> bool:
            now = _dt.now(_IST)
            if now.weekday() >= 5:     # Sat/Sun
                return False
            # NSE opens 9:15 IST — scan starts 1 min after open, closes after 15:30
            return _t(9, 15) <= now.time() <= _t(15, 45)

        def _run_all_scanners(dfs: dict) -> dict:
            """CPU-bound: run all 5 scanners and return their results."""
            from src.scanners.quality import (   # noqa: PLC0415
                passes_intraday_volume, passes_intraday_volatility
            )
            # Apply intraday pre-filter: only scan symbols with today's volume
            # and sufficient volatility — removes illiquid/halted/flat stocks
            dfs_filtered = {
                sym: df
                for sym, df in dfs.items()
                if passes_intraday_volume(df, symbol=sym)
                and passes_intraday_volatility(df, symbol=sym)
            }
            dfs_ms = {sym: df.iloc[-_MS_BARS:] if len(df) > _MS_BARS else df
                      for sym, df in dfs_filtered.items()}
            return {
                "order_block":      _ob_scanner.scan(dfs_filtered, timeframe="1d"),
                "fvg":              _fvg_scanner.scan(dfs_filtered, timeframe="1d"),
                "supply_demand":    _sd_scanner.scan(dfs_filtered, timeframe="1d"),
                "liquidity_sweep":  _sweep_scanner.scan(dfs_filtered, timeframe="1d"),
                "market_structure": _ms_scanner.scan(dfs_ms,        timeframe="1d"),
                "_n_filtered":      len(dfs_filtered),
            }

        logger.info("live_smc_scanner_task_started", interval_s=_INTERVAL)

        # Initial scan 60 seconds after startup to let everything settle
        await _a.sleep(60)

        while True:
            try:
                now_ist = _dt.now(_IST)
                today_str = now_ist.date().isoformat()

                if _is_market_hours():
                    _cache = app.state.cache

                    # ── Morning clear: discard stale overnight signals ────────
                    # Runs once per trading day, at the first scan after 9:15 IST.
                    if _cleared_date_ref[0] != today_str:
                        n_cleared = await _registry.clear_for_new_day(_cache)
                        _cleared_date_ref[0] = today_str
                        if n_cleared:
                            logger.info("live_smc_morning_clear",
                                        n_cleared=n_cleared, date=today_str)

                    # Load data and run scanners in a thread (CPU-bound work)
                    dfs     = await _a.to_thread(load_1d_data)
                    results = await _a.to_thread(_run_all_scanners, dfs)

                    n_filtered = results.pop("_n_filtered", len(dfs))

                    # Update registry for each scanner type
                    attr_map = {
                        "order_block":      ("zones",   results["order_block"]),
                        "fvg":              ("zones",   results["fvg"]),
                        "supply_demand":    ("zones",   results["supply_demand"]),
                        "liquidity_sweep":  ("signals", results["liquidity_sweep"]),
                        "market_structure": ("signals", results["market_structure"]),
                    }
                    n_total = 0
                    for stype, (attr, res) in attr_map.items():
                        items = getattr(res, attr, [])
                        await _registry.record_scan(stype, items, _cache)
                        n_total += len(items)

                    # Persist scan metadata
                    scanned_at = _dt.now(_tz.utc).isoformat()
                    await _registry.set_last_scan_meta(_cache, {
                        "scanned_at":      scanned_at,
                        "n_symbols":       n_filtered,
                        "n_total_signals": n_total,
                    })

                    logger.info(
                        "live_smc_scan_complete",
                        n_symbols=n_filtered,
                        n_signals=n_total,
                    )
                else:
                    # ── EOD close: settle all signals at market close ─────────
                    # Window: 15:30–16:00 IST. Wider than the market close (15:30)
                    # to account for the 5-min loop interval — the task may wake
                    # up at 15:50:xx which would miss a 15:50:00 upper bound.
                    # A daily flag prevents double-firing.
                    if (now_ist.weekday() < 5
                            and _t(15, 30) <= now_ist.time() <= _t(16, 0)):
                        _cache = app.state.cache
                        n_closed = await _registry.close_all_active(_cache, reason="eod")
                        if n_closed:
                            logger.info("live_smc_eod_close", n_closed=n_closed)

            except Exception as _exc:
                logger.error("live_smc_scanner_error", error=str(_exc), exc_info=True)

            await _a.sleep(_INTERVAL)

    _smc_task = _aio.create_task(_live_smc_scanner_task())
    logger.info("live_smc_scanner_task_created")

    yield  # ── application runs ──────────────────────────────────────────────

    # Shutdown — cancel file-watcher and SMC scanner cleanly
    _watcher_task.cancel()
    _smc_task.cancel()
    for t in (_watcher_task, _smc_task):
        try:
            await t
        except _aio.CancelledError:
            pass
    await cache.disconnect()
    logger.info("ml_service_shutdown")


# ── FastAPI application ────────────────────────────────────────────────────────

# Hide interactive docs in VALIDATED_ML_ONLY mode
_docs_url = "/docs" if settings.deployment_mode.value != "validated_ml_only" else None
_redoc_url = "/redoc" if settings.deployment_mode.value != "validated_ml_only" else None

app = FastAPI(
    title="ml-service2.0",
    description=(
        "Institutional-grade standalone ML microservice for AlphaForge. "
        "Provides inference, training, monitoring, and portfolio optimization."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url=_docs_url,
    redoc_url=_redoc_url,
)


# ── CORS ──────────────────────────────────────────────────────────────────────

_allowed_origins: list[str] = (
    settings.allowed_origins
    if isinstance(settings.allowed_origins, list)
    else [settings.allowed_origins]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── X-API-KEY authentication middleware ───────────────────────────────────────

# Paths that do not require an API key
_EXEMPT_PATHS: frozenset[str] = frozenset(
    {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}
)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next: Any) -> Response:
    """Enforce X-API-KEY on all paths except those in ``_EXEMPT_PATHS``.

    Returns HTTP 401 immediately on missing or invalid key without invoking
    downstream middleware or route handlers.
    """
    if request.url.path not in _EXEMPT_PATHS:
        api_key = request.headers.get("X-API-KEY")
        if not api_key or api_key != settings.ml_service_api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid X-API-KEY header"},
            )
    return await call_next(request)


# ── X-Request-ID middleware ───────────────────────────────────────────────────


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Response:
    """Propagate or generate an ``X-Request-ID`` trace header.

    The ID is bound into the structlog context so every log statement emitted
    within the request automatically carries ``request_id``.  Context is
    cleared in the ``finally`` block to prevent bleed between requests.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    bind_request_context(request_id=request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        clear_request_context()


# ── Global exception handlers ─────────────────────────────────────────────────


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Map Pydantic V2 ``RequestValidationError`` → HTTP 422.

    Returns the full list of field-level errors so callers can identify
    exactly which fields are invalid without making a second request.
    """
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Catch-all for unhandled exceptions — returns HTTP 500.

    Logs the exception at ERROR level with the request path so it can be
    correlated via ``X-Request-ID`` in structured log aggregation pipelines.
    In normal operation this handler should never fire; its presence prevents
    unformatted 500 stack traces from leaking to callers.
    """
    logger.error(
        "unhandled_exception",
        error=str(exc),
        exc_info=True,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Health endpoint ───────────────────────────────────────────────────────────


@app.get("/health", tags=["health"])
async def health() -> dict[str, Any]:
    """Service health check.  Does not require ``X-API-KEY``.

    Returns::

        {
          "status":    "healthy",
          "timestamp": <unix milliseconds>,
          "version":   "2.0.0"
        }
    """
    return {
        "status": "healthy",
        "timestamp": int(time.time() * 1000),
        "version": "2.0.0",
    }


@app.get("/metrics", tags=["observability"])
async def metrics() -> Response:
    """Prometheus metrics endpoint (FIX NEW-P3-002).

    Does not require ``X-API-KEY`` — standard Prometheus scraper convention.
    Exposes: predictions_total, signals_total, no_trade_total, model_errors_total,
    inference_latency_seconds, data_quality_score, drift_score_max,
    model_health, alpha_ic_decay_fraction, drawdown_fraction, drawdown_state,
    forward_paper_signals_total, forward_paper_resolved_total.

    GET /metrics
    """
    from src.monitoring.metrics import metrics_output  # noqa: PLC0415
    content, content_type = metrics_output()
    return Response(content=content, media_type=content_type)


# ── Router registration ───────────────────────────────────────────────────────
# Routers are imported here (after `app` is created) to avoid circular imports
# while keeping the module layout consistent with the design spec.

from src.api import analytics, explain, meta, monitoring, predict, scanners as scanners_api, signals, training  # noqa: E402
from src.streaming.streamer import SignalStreamer  # noqa: E402

app.include_router(predict.router, prefix="/v2")
app.include_router(meta.router, prefix="/v2")
app.include_router(analytics.router, prefix="/v2")
app.include_router(explain.router, prefix="/v2")
app.include_router(signals.router, prefix="/v2")
app.include_router(training.router)
app.include_router(monitoring.router)
app.include_router(scanners_api.router, prefix="/v2")

# ── Signal streaming ─────────────────────────────────────────────────────────

_signal_streamer = SignalStreamer()


@app.websocket("/v2/stream/signals")
async def ws_stream_signals(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time signal streaming.

    Path: ``WS /v2/stream/signals``

    Authentication is enforced via the ``?api_key=`` query parameter on
    WebSocket upgrade (HTTP headers are unavailable for WS handshakes in most
    browser/client implementations).

    Signals are pushed by ``_signal_streamer.broadcast_signal()`` whenever
    the ``MetaDecisionEngine`` produces a new output; this handler only keeps
    the connection alive and handles heartbeat messages sent by the client.
    """
    # Authenticate via query parameter — WebSocket upgrade cannot carry arbitrary
    # HTTP headers in all clients, so api_key is accepted as a query param.
    api_key = websocket.query_params.get("api_key")
    if not api_key or api_key != settings.ml_service_api_key:
        await websocket.close(code=1008)  # 1008 = Policy Violation
        logger.warning("websocket_auth_failure", path="/v2/stream/signals")
        return

    await _signal_streamer.manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive; actual signals are pushed via broadcast_signal()
            data = await websocket.receive_text()
            # Echo back a heartbeat so clients can detect dead connections
            await websocket.send_json({"type": "heartbeat", "data": data})
    except (WebSocketDisconnect, Exception):
        await _signal_streamer.manager.disconnect(websocket)
