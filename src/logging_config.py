"""
Structlog configuration for ml-service2.0.

Configures structured JSON logging with mandatory fields on every log entry.
Provides helpers for request-scoped context binding, log retrieval, and
operation timing.

Usage::

    from src.logging_config import configure_logging, get_logger, bind_request_context

    configure_logging(log_level="INFO")
    bind_request_context(request_id="abc-123", model_name="market_regime")
    log = get_logger(__name__)
    log.info("prediction_complete", latency_ms=12.5, provenance="trained_model")
"""

from __future__ import annotations

import logging
import time
from types import TracebackType
from typing import TYPE_CHECKING

import structlog
from structlog.types import FilteringBoundLogger

if TYPE_CHECKING:
    pass  # avoid circular imports at runtime

# ── Module-level flag so configure_logging() is idempotent ────────────────────
_configured: bool = False


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog globally.

    Safe to call multiple times — subsequent calls are no-ops once the first
    configuration has been applied.  Pass ``log_level="DEBUG"`` to enable the
    human-readable ``ConsoleRenderer``; all other levels use ``JSONRenderer``.

    Args:
        log_level: Minimum log level string (``DEBUG``, ``INFO``, ``WARNING``,
                   ``ERROR``, ``CRITICAL``).  Case-insensitive.
    """
    global _configured  # noqa: PLW0603
    if _configured:
        return

    # Configure the stdlib root logger so third-party libraries route through
    # structlog at the same level.
    numeric_level = logging.getLevelName(log_level.upper())
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
    )

    # Choose renderer based on log level: DEBUG → human-friendly, else JSON.
    use_dev_renderer = log_level.upper() == "DEBUG"
    renderer: structlog.types.Processor
    if use_dev_renderer:
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    def _add_logger_name(
        logger: object,
        method_name: str,
        event_dict: structlog.types.EventDict,
    ) -> structlog.types.EventDict:
        """
        Inject the logger name from the structlog bound logger's positional key.

        structlog.stdlib.add_logger_name only works with stdlib-backed loggers;
        this processor handles the PrintLogger case used here.
        """
        record = event_dict.get("_record")
        if record is not None:
            event_dict["logger"] = record.name
        elif hasattr(logger, "_logger") and hasattr(logger._logger, "name"):
            event_dict["logger"] = logger._logger.name
        else:
            # Fall back to the logger's repr or class name
            event_dict.setdefault("logger", type(logger).__name__)
        return event_dict

    structlog.configure(
        processors=[
            # Merge in any context-variables bound via bind_request_context()
            structlog.contextvars.merge_contextvars,
            # Add ISO-8601 UTC timestamp
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            # Add the log level as a string field ("level")
            structlog.processors.add_log_level,
            # Add the logger name ("logger") — custom processor for PrintLogger
            _add_logger_name,
            # Render tracebacks inline when an exception is attached
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            # Final renderer: JSON in production, ConsoleRenderer in dev
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configured = True


# ── Request-scoped context helpers ────────────────────────────────────────────


def bind_request_context(
    request_id: str,
    model_name: str = "",
    provenance: str = "",
) -> None:
    """
    Bind mandatory log fields for the current async task / thread.

    Every subsequent log statement in the same async context will automatically
    include these fields without the caller having to pass them explicitly.

    Args:
        request_id:  The ``X-Request-ID`` header value (or a generated UUID).
        model_name:  Name of the model being invoked (e.g. ``"market_regime"``).
        provenance:  ``PredictionProvenance`` string value for the current call.
    """
    # Import here to avoid a circular dependency at module load time
    from src.config import settings  # noqa: PLC0415

    structlog.contextvars.bind_contextvars(
        service="ml-service2.0",
        version="2.0.0",
        request_id=request_id,
        model_name=model_name,
        provenance=provenance,
        deployment_mode=settings.deployment_mode.value,
    )


def clear_request_context() -> None:
    """
    Remove all context variables bound by :func:`bind_request_context`.

    Should be called at the end of every request (e.g. in FastAPI middleware)
    so context does not bleed into subsequent requests served by the same
    worker/coroutine.
    """
    structlog.contextvars.clear_contextvars()


# ── Logger factory ────────────────────────────────────────────────────────────


def get_logger(name: str = __name__) -> FilteringBoundLogger:
    """
    Return a structlog bound logger for *name*.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A :class:`structlog.BoundLogger` instance.
    """
    return structlog.get_logger(name)


# ── Timing context manager ────────────────────────────────────────────────────


class TimingLogger:
    """
    Context manager that logs how long an operation took in milliseconds.

    Example::

        log = get_logger(__name__)
        with TimingLogger("feature_pipeline_build", log):
            vector = await pipeline.build_vector(...)

    The ``__exit__`` log entry has the form::

        {"event": "operation_completed", "operation": "<name>", "latency_ms": 12.34}
    """

    def __init__(self, operation: str, logger: FilteringBoundLogger) -> None:
        self.operation = operation
        self._logger = logger
        self._start: float = 0.0

    def __enter__(self) -> TimingLogger:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self._logger.info(
            "operation_completed",
            operation=self.operation,
            latency_ms=round(elapsed_ms, 3),
        )
