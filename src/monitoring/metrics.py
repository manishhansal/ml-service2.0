"""
src/monitoring/metrics.py — Prometheus metrics for ml-service2.0

FIX NEW-P3-002: All key operational metrics exported via /metrics endpoint.

Counters / Gauges / Histograms:
  predictions_total         — total prediction requests by action + provenance
  signals_total             — signals emitted by action
  no_trade_total            — NO_TRADE decisions by reason_code
  model_errors_total        — model-level errors by model_name
  data_quality_gauge        — current DataConfidenceScore per symbol (last seen)
  drift_score_gauge         — current max PSI drift score
  model_health_gauge        — model health state (1=HEALTHY, 0.75=WARN, 0.3=DEGRADED, 0=DISABLED)
  inference_latency_seconds — prediction latency histogram
  alpha_decay_gauge         — alpha specialist IC decay fraction per specialist
  drawdown_gauge            — current portfolio drawdown fraction
  drawdown_state_gauge      — drawdown state (3=NORMAL,2=CAUTION,1=DEFENSIVE,0=HALTED)
  forward_paper_signals_total   — signals written to forward paper store
  forward_paper_resolved_total  — signals resolved in forward paper store

Usage (from route handlers or middleware)::

    from src.monitoring.metrics import (
        predictions_total, inference_latency_seconds, record_prediction,
    )
    record_prediction(action="BUY", provenance="trained_model", latency_s=0.012)
"""
from __future__ import annotations

from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from prometheus_client import REGISTRY as DEFAULT_REGISTRY

# ── Registry ──────────────────────────────────────────────────────────────────
# Use the default registry so /metrics includes Python process metrics too.
REGISTRY = DEFAULT_REGISTRY

# ── Counters ──────────────────────────────────────────────────────────────────

predictions_total = Counter(
    "ml_service_predictions_total",
    "Total prediction requests",
    ["action", "provenance"],
    registry=REGISTRY,
)

signals_total = Counter(
    "ml_service_signals_total",
    "Signals emitted by the MetaDecisionEngine",
    ["action"],
    registry=REGISTRY,
)

no_trade_total = Counter(
    "ml_service_no_trade_total",
    "NO_TRADE decisions by primary reason code",
    ["reason_code"],
    registry=REGISTRY,
)

model_errors_total = Counter(
    "ml_service_model_errors_total",
    "Model-level errors during inference",
    ["model_name", "error_type"],
    registry=REGISTRY,
)

forward_paper_signals_total = Counter(
    "ml_service_forward_paper_signals_total",
    "Signals written to the forward paper store",
    [],
    registry=REGISTRY,
)

forward_paper_resolved_total = Counter(
    "ml_service_forward_paper_resolved_total",
    "Signals resolved in the forward paper store",
    ["outcome"],
    registry=REGISTRY,
)

# ── Gauges ────────────────────────────────────────────────────────────────────

data_quality_gauge = Gauge(
    "ml_service_data_quality_score",
    "Most recent DataConfidenceScore (0–100) per symbol",
    ["symbol"],
    registry=REGISTRY,
)

drift_score_gauge = Gauge(
    "ml_service_drift_score_max",
    "Maximum PSI drift score across all features (current window)",
    [],
    registry=REGISTRY,
)

model_health_gauge = Gauge(
    "ml_service_model_health",
    "Model health weight multiplier (1.0=HEALTHY, 0.75=WARN, 0.30=DEGRADED, 0.0=DISABLED)",
    ["model_name"],
    registry=REGISTRY,
)

alpha_decay_gauge = Gauge(
    "ml_service_alpha_ic_decay_fraction",
    "IC decay fraction for each alpha specialist (0=healthy, 1=fully decayed)",
    ["alpha_name"],
    registry=REGISTRY,
)

drawdown_gauge = Gauge(
    "ml_service_drawdown_fraction",
    "Current portfolio drawdown fraction (0.0–1.0)",
    [],
    registry=REGISTRY,
)

drawdown_state_gauge = Gauge(
    "ml_service_drawdown_state",
    "Drawdown state: 3=NORMAL 2=CAUTION 1=DEFENSIVE 0=HALTED",
    [],
    registry=REGISTRY,
)

# ── Histograms ────────────────────────────────────────────────────────────────

inference_latency_seconds = Histogram(
    "ml_service_inference_latency_seconds",
    "End-to-end prediction latency in seconds",
    ["symbol"],
    buckets=(0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0, 5.0),
    registry=REGISTRY,
)

feature_computation_seconds = Histogram(
    "ml_service_feature_computation_seconds",
    "Feature vector computation latency in seconds",
    [],
    buckets=(0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.500),
    registry=REGISTRY,
)

# ── Convenience helpers ────────────────────────────────────────────────────────

_DRAWDOWN_STATE_VALUES = {
    "NORMAL": 3.0,
    "CAUTION": 2.0,
    "DEFENSIVE": 1.0,
    "HALTED": 0.0,
}


def record_prediction(
    action: str,
    provenance: str,
    latency_s: float,
    symbol: str = "unknown",
) -> None:
    """Record one completed prediction — updates counter and latency histogram."""
    predictions_total.labels(action=action, provenance=provenance).inc()
    signals_total.labels(action=action).inc()
    inference_latency_seconds.labels(symbol=symbol).observe(latency_s)


def record_no_trade(reason_code: str) -> None:
    """Record a NO_TRADE decision with its primary reason code."""
    no_trade_total.labels(reason_code=reason_code).inc()


def record_model_error(model_name: str, error_type: str = "inference_error") -> None:
    """Record a model-level error."""
    model_errors_total.labels(model_name=model_name, error_type=error_type).inc()


def update_model_health(model_name: str, weight_multiplier: float) -> None:
    """Push the current model health weight multiplier to Prometheus."""
    model_health_gauge.labels(model_name=model_name).set(weight_multiplier)


def update_drift_score(max_psi: float) -> None:
    """Push the current max PSI drift score."""
    drift_score_gauge.set(max_psi)


def update_drawdown(drawdown_fraction: float, state: str) -> None:
    """Push the current drawdown state."""
    drawdown_gauge.set(max(0.0, min(1.0, drawdown_fraction)))
    drawdown_state_gauge.set(_DRAWDOWN_STATE_VALUES.get(state.upper(), 3.0))


def update_alpha_decay(alpha_name: str, ic_decay_fraction: float) -> None:
    """Push the IC decay fraction for an alpha specialist."""
    alpha_decay_gauge.labels(alpha_name=alpha_name).set(max(0.0, min(1.0, ic_decay_fraction)))


def update_data_quality(symbol: str, score: float) -> None:
    """Push the most recent DataConfidenceScore for a symbol."""
    data_quality_gauge.labels(symbol=symbol).set(max(0.0, min(100.0, score)))


def metrics_output() -> tuple[bytes, str]:
    """Return raw Prometheus metrics bytes and content type for the /metrics endpoint."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
