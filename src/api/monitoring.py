"""
Monitoring endpoints for ml-service2.0.

Endpoints:
    GET  /monitoring/drift                          → drift report dict
    GET  /monitoring/performance                    → NannyML CBPE estimates dict
    GET  /monitoring/alerts                         → list[DriftAlert] (last 50)
    POST /monitoring/performance/{model_name}/clear → {"cleared": bool, "model": str, "was_blocked": bool}

In-memory state:
    _alerts             — ordered list of DriftAlert objects; capped at last 50 on read
    _performance_blocks — map of model_name → is_blocked flag (Req 12.10)

Helper:
    emit_drift_alert(alert)  — append a DriftAlert to the in-memory store (Req 12.6, 12.7)
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from src.monitoring.drift_monitor import DriftMonitor
from src.schemas.monitoring import DriftAlert

router = APIRouter(tags=["monitoring"])

# ── In-memory state ───────────────────────────────────────────────────────────

_drift_monitor = DriftMonitor()
_alerts: list[DriftAlert] = []
_performance_blocks: dict[str, bool] = {}  # model_name → is_blocked


# ── Helper ────────────────────────────────────────────────────────────────────


def emit_drift_alert(alert: DriftAlert) -> None:
    """Add a DriftAlert to the in-memory alert store (Req 12.6, 12.7)."""
    _alerts.append(alert)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/monitoring/drift")
async def monitoring_drift() -> dict[str, Any]:
    """Latest feature drift report for all models.

    Returns PSI threshold configuration and a count of active alerts.
    Populate the 'models' key by running drift analysis via DriftMonitor.
    """
    return {
        "models": {},
        "last_run": None,
        "psi_threshold_medium": 0.2,
        "psi_threshold_high": 0.25,
        "total_alerts": len(_alerts),
        "note": "Run drift analysis to populate this report",
    }


@router.get("/monitoring/performance")
async def monitoring_performance() -> dict[str, Any]:
    """NannyML CBPE estimated IC + ECE per model.

    Returns current performance-block state alongside an empty model map until
    NannyML analysis has been run.
    """
    return {
        "models": {},
        "last_run": None,
        "performance_blocks": _performance_blocks,
        "note": "NannyML CBPE integration available via DriftMonitor.run_nannyml_cbpe()",
    }


@router.get("/monitoring/alerts")
async def monitoring_alerts() -> list[Any]:
    """Active drift/performance alerts. Returns the last 50 alerts."""
    return [a.model_dump() for a in _alerts[-50:]]


@router.post("/monitoring/performance/{model_name}/clear")
async def monitoring_performance_clear(model_name: str) -> dict[str, Any]:
    """Manually clear a performance-degradation block for *model_name* (Req 12.10).

    Idempotent — returns success whether or not the model was blocked.
    """
    was_blocked = _performance_blocks.get(model_name, False)
    _performance_blocks[model_name] = False
    return {
        "cleared": True,
        "model": model_name,
        "was_blocked": was_blocked,
    }
