"""
Analytics and features endpoint for ml-service2.0.

Endpoints wired here:
    POST        /analytics/greeks      -> list[dict]     (100ms p95) [stub]
    GET/POST    /analytics/gex         -> GexResponse    (100ms p95) [stub]
    POST        /analytics/vpin        -> VpinResponse   (100ms p95)
    POST        /analytics/vol-surface -> VolSurfResponse (100ms p95)
    GET         /features/quality      -> FeatureQualityReport (2s p95)

Full analytics implementations arrive in a later phase.
Until then every unimplemented computation endpoint returns HTTP 503.
/features/quality returns the latest FeatureQualityReport if one has been
produced by the pipeline, or an empty report skeleton otherwise.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pydantic
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.analytics.vpin import compute_vpin
from src.analytics.vol_surface import build_iv_surface, compute_term_structure, fit_svi

router = APIRouter(tags=["analytics"])

_NOT_IMPLEMENTED = JSONResponse(
    status_code=503,
    content={"detail": "Analytics module not yet implemented"},
)

# Updated by FeaturePipeline after each build_vector / build_batch call.
# The pipeline imports this module and sets this variable so that the
# /features/quality endpoint reflects real pipeline output.
_latest_quality_report: dict[str, Any] | None = None


# -- Greeks -------------------------------------------------------------------


@router.post("/analytics/greeks")
async def analytics_greeks(request: Any = None) -> JSONResponse:
    """Black-Scholes/Black-76 Greeks calculation.  POST /v2/analytics/greeks — 503 stub."""
    return _NOT_IMPLEMENTED


# -- Gamma Exposure ------------------------------------------------------------


@router.get("/analytics/gex")
@router.post("/analytics/gex")
async def analytics_gex(request: Any = None) -> JSONResponse:
    """Dealer Gamma Exposure computation.  GET|POST /v2/analytics/gex — 503 stub."""
    return _NOT_IMPLEMENTED


# -- VPIN ---------------------------------------------------------------------

# VPIN endpoint uses a minimal typed dict body to avoid Pydantic v2 deep
# validation overhead on the bars list, keeping p95 latency under 100ms.


class VpinRequest(pydantic.BaseModel):
    """Request body for POST /v2/analytics/vpin — minimal validation for speed."""

    model_config = pydantic.ConfigDict(strict=False)

    symbol: str = "NIFTY"
    bars: list[Any]          # Accept raw list — VPIN compute_vpin handles dicts
    bucket_size: float = 50.0
    n_buckets: int = 50


@router.post("/analytics/vpin")
async def analytics_vpin_post(req: VpinRequest) -> dict:
    """
    Volume-synchronized Probability of Informed Trading.

    POST /v2/analytics/vpin — responds within 100ms at p95.

    Validates: Requirements 15.5
    """
    if not req.bars:
        return {
            "symbol": req.symbol,
            "vpin": 0.0,
            "bucketHistory": [],
            "classification": "benign",
            "available": True,
        }

    result = compute_vpin(req.bars, bucket_size=req.bucket_size, n_buckets=req.n_buckets)
    vpin = result["current_vpin"]

    if vpin >= 0.7:
        classification = "toxic"
    elif vpin >= 0.3:
        classification = "elevated"
    else:
        classification = "benign"

    return {
        "symbol": req.symbol,
        "vpin": round(vpin, 6),
        "bucketHistory": [round(v, 6) for v in result["vpin_series"][-20:]],
        "classification": classification,
        "available": True,
    }


# Keep GET stub for backward compat
@router.get("/analytics/vpin")
async def analytics_vpin_get() -> JSONResponse:
    """GET /v2/analytics/vpin — use POST instead."""
    return _NOT_IMPLEMENTED


# -- Volatility Surface -------------------------------------------------------


class VolSurfaceSnapshotItem(pydantic.BaseModel):
    """Per-expiry snapshot data."""

    strikes: list[float]
    ivs: list[float]
    forward: float
    days_to_expiry: float
    atm_iv: float


class VolSurfaceRequest(pydantic.BaseModel):
    """Request body for POST /v2/analytics/vol-surface."""

    symbol: str = "NIFTY"
    snapshots_by_expiry: dict[str, VolSurfaceSnapshotItem]


@router.post("/analytics/vol-surface")
async def analytics_vol_surface_post(req: VolSurfaceRequest) -> dict:
    """
    SVI volatility surface fitting + term structure.

    POST /v2/analytics/vol-surface — responds within 100ms at p95.

    Validates: Requirements 15.5
    """
    snapshots_raw = {k: v.model_dump() for k, v in req.snapshots_by_expiry.items()}

    iv_by_expiry = build_iv_surface(snapshots_raw)

    svi_params: dict[str, Any] = {}
    for expiry, snapshot in snapshots_raw.items():
        try:
            svi_params[expiry] = fit_svi(
                strikes=snapshot["strikes"],
                ivs=snapshot["ivs"],
                forward=snapshot["forward"],
            )
        except Exception:
            svi_params[expiry] = None

    term_structure = compute_term_structure(snapshots_raw)

    return {
        "symbol": req.symbol,
        "expiries": list(snapshots_raw.keys()),
        "ivByExpiry": iv_by_expiry,
        "termStructure": [
            {"daysToExpiry": e["days_to_expiry"], "atmIv": e["atm_iv"]}
            for e in term_structure
        ],
        "sviParams": svi_params,
        "available": True,
    }


# Keep GET stub for backward compat
@router.get("/analytics/vol-surface")
async def analytics_vol_surface_get() -> JSONResponse:
    """GET /v2/analytics/vol-surface — use POST instead."""
    return _NOT_IMPLEMENTED


# -- Feature Quality ----------------------------------------------------------


@router.get("/features/quality")
async def features_quality() -> dict[str, Any]:
    """Return the latest FeatureQualityReport.

    Returns the most-recent report produced by FeaturePipeline.build_vector or
    build_batch.  Falls back to an empty/zero skeleton until the pipeline has
    run at least once.

    Response time SLA: <= 2 seconds p95 (satisfied trivially — no I/O here).
    """
    if _latest_quality_report is not None:
        return _latest_quality_report
    return {
        "batch_id": str(uuid.uuid4()),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "total_features_requested": 0,
        "missing_count": 0,
        "imputed_count": 0,
        "rejected_count": 0,
        "unavailable_families": [],
        "pit_violations_count": 0,
        "discarded_backtest_records": 0,
        "processing_time_ms": 0.0,
        "sentinel_pulse_available": True,
        "data_service_available": True,
        "note": "FeaturePipeline has not yet produced a report — empty skeleton",
    }
