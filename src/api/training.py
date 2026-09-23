"""
Training endpoints for ml-service2.0.

Endpoints:
    POST /training/run              → TrainingRunStatus (queued acknowledgement)
    GET  /training/status/{run_id}  → TrainingRunStatus

Training runs are accepted asynchronously; the POST endpoint returns a QUEUED
status immediately.  Actual training execution happens offline or via a
separate worker process.  Status is retrievable via GET for the lifetime of
the process.

Requirements: Req 15.4
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from src.schemas.registry import TrainingConfig, TrainingRunStatus
from src.training.pipeline import TrainingPipeline

router = APIRouter(tags=["training"])

# In-memory registry of submitted training runs (keyed by run_id).
# Populated by POST /training/run; queried by GET /training/status/{run_id}.
_training_registry: dict[str, TrainingRunStatus] = {}

# Shared pipeline instance — used for lifecycle validation and status look-ups.
_pipeline = TrainingPipeline()


@router.post("/training/run", response_model=TrainingRunStatus)
async def training_run(config: TrainingConfig) -> TrainingRunStatus:
    """Submit a training run request.

    Accepts a ``TrainingConfig`` and returns a ``TrainingRunStatus`` with
    ``status="QUEUED"``.  The run is registered so its status can be polled
    via ``GET /training/status/{run_id}``.

    POST /training/run
    """
    run_id = str(uuid.uuid4())
    status = TrainingRunStatus(
        run_id=run_id,
        model_name=config.model_name,
        status="QUEUED",
        started_at=datetime.now(tz=timezone.utc),
        current_stage=f"Queued for {config.model_name} training",
    )
    _training_registry[run_id] = status
    return status


@router.get("/training/status/{run_id}", response_model=TrainingRunStatus)
async def training_status(run_id: str) -> TrainingRunStatus:
    """Poll the status of a previously submitted training run.

    Returns ``TrainingRunStatus`` for the given ``run_id``, or HTTP 404 when
    the run is not found in the in-memory registry.

    GET /training/status/{run_id}
    """
    status = _training_registry.get(run_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Training run {run_id} not found",
        )
    return status
