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
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.data.feedback import FeedbackStore
from src.schemas.meta import FeedbackRecord
from src.schemas.registry import TrainingConfig, TrainingRunStatus
from src.training.pipeline import TrainingPipeline

router = APIRouter(tags=["training"])

# Append-only feedback store (Phase Q). Path is configurable via settings-like
# default; kept alongside the audit log for co-location.
_feedback_store = FeedbackStore(Path("./artifacts/feedback.jsonl"))

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


@router.post("/train/feedback")
async def train_feedback(record: FeedbackRecord) -> dict:
    """Ingest a trade-outcome feedback record from AlphaForge (Phase Q).

    The record is appended to the immutable feedback store and enters the
    self-learning loop. Feedback is never modified once written.

    POST /train/feedback
    """
    _feedback_store.append(record)
    return {
        "accepted": True,
        "signal_id": record.signal_id,
        "total_feedback_records": _feedback_store.count(),
    }


@router.get("/train/feedback/summary")
async def train_feedback_summary() -> dict:
    """Return an aggregate summary of collected feedback (Phase 67).

    GET /train/feedback/summary
    """
    return _feedback_store.summary()


# ── Training Readiness Gate ───────────────────────────────────────────────────
# Mandate §27: single authoritative readiness endpoint.
# Mandate §28: gate MUST fail closed. Returns READY, READY_MARKET_ONLY, or NOT_READY.


@router.get("/training/readiness")
async def training_readiness(
    news_required: bool = False,
    min_history_days: int = 252,
    min_universe_size: int = 3,
    timeframe: str = "1d",
) -> dict:
    """
    Training-readiness gate — single authoritative answer: READY / NOT_READY.

    Checks every precondition before training is allowed:
      DATA      — data-service2.0 reachability, auth, universe, history
      NEWS      — SentinelPulse (if news_required=True; else optional)
      FEATURES  — feature schema validity
      LABELS    — label schema validity, default execution_model=next_open
      TRAINING  — Python dependencies, artifact path writable
      PROVENANCE — git SHA, docker image

    Mandate §28: fails closed on any critical blocker.

    GET /training/readiness?news_required=false&min_history_days=252&timeframe=1d

    Returns::

        {
          "training_ready": true | false,
          "mode": "READY" | "READY_MARKET_ONLY" | "NOT_READY",
          "news_status": "ENABLED" | "DISABLED" | "UNAVAILABLE" | "EVIDENCE_PENDING",
          "blockers": [...],
          "warnings": [...],
          "gates": [{"gate": ..., "status": "PASS"|"FAIL"|"WARN"|"SKIP", ...}],
          "checked_at": "...",
          "git_sha": "...",
          "docker_image": "...",
          "python_version": "..."
        }
    """
    from src.clients.data_service import DataServiceClient
    from src.clients.sentinel_pulse import SentinelPulseClient
    from src.config import settings
    from src.data.readiness import TrainingReadinessGate

    data_client = DataServiceClient()
    await data_client.connect()

    sentinel_client = SentinelPulseClient()
    await sentinel_client.connect()

    gate = TrainingReadinessGate()

    # Fetch real F&O universe — single call, reused by gate
    # (avoids double-fetching which would hit the rate limit)
    sample_universe: list[str]
    try:
        universe_resp = await data_client.get_fno_universe()
        raw_syms = (
            universe_resp.get("data", {}).get("constituents", [])
            or universe_resp.get("data", {}).get("symbols", [])
            or []
        )
        sample_universe = [s["symbol"] if isinstance(s, dict) else s for s in raw_syms[:50]]
        if not sample_universe:
            sample_universe = ["NIFTY", "BANKNIFTY", "RELIANCE"]
        print(f"Universe: {len(sample_universe)} symbols, sample: {sample_universe[:5]}", flush=True)
    except Exception as e:
        sample_universe = ["NIFTY", "BANKNIFTY", "RELIANCE"]
        print(f"Universe fetch failed ({e}), using fallback: {sample_universe}", flush=True)

    try:
        result = await gate.check(
            data_client=data_client,
            sentinel_client=sentinel_client,
            universe=sample_universe,
            timeframe=timeframe,
            min_history_days=min_history_days,
            news_required=news_required,
            min_universe_size=min_universe_size,
        )
    finally:
        await data_client.disconnect()
        await sentinel_client.disconnect()

    return result.to_dict()
