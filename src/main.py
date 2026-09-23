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

    yield  # ── application runs ──────────────────────────────────────────────

    # Shutdown
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
    {"/health", "/docs", "/openapi.json", "/redoc"}
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


# ── Router registration ───────────────────────────────────────────────────────
# Routers are imported here (after `app` is created) to avoid circular imports
# while keeping the module layout consistent with the design spec.

from src.api import analytics, explain, meta, monitoring, predict, training  # noqa: E402
from src.streaming.streamer import SignalStreamer  # noqa: E402

app.include_router(predict.router, prefix="/v2")
app.include_router(meta.router, prefix="/v2")
app.include_router(analytics.router, prefix="/v2")
app.include_router(explain.router, prefix="/v2")
app.include_router(training.router)
app.include_router(monitoring.router)

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
