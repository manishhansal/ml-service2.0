"""
Training endpoints for ml-service2.0.

FIX NEW-P1-003: /training/run now routes to TrainingOrchestrator (walk-forward
+ CPCV + calibration + baseline comparison) rather than the legacy
TrainingPipeline (purged-kfold only, no CPCV, inferior Sharpe computation).

Endpoints:
    POST /training/run              → TrainingRunStatus (async acknowledgement)
    GET  /training/status/{run_id}  → TrainingRunStatus
    POST /training/run/sync         → TrainingReport (synchronous, for scripts)
    POST /train/feedback            → feedback ingestion
    GET  /train/feedback/summary    → aggregate feedback summary
    GET  /training/readiness        → gate check before training

Training flow (TrainingOrchestrator):
    frozen dataset (DatasetBuilder)
    → WalkForwardValidator (5 windows, per-fold normalisation)
    → CombinatorialPurgedCV (PBO estimate)
    → Calibration (Platt/isotonic, ECE)
    → Baseline comparison gate  (champion must beat naive baseline + margin)
    → Champion selection (parsimony: baseline preferred within 0.005 IC)
    → ModelRegistry (CHALLENGER stage — never auto-promoted to PRODUCTION)

The legacy TrainingPipeline is still available for backward-compat scripts
that call it directly; the API path always uses the Orchestrator.

Requirements: Req 15.4, FIX NEW-P1-003, mandate §75.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from src.data.feedback import FeedbackStore
from src.schemas.meta import FeedbackRecord
from src.schemas.registry import TrainingConfig, TrainingRunStatus
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["training"])

# ── Singletons ────────────────────────────────────────────────────────────────
_feedback_store = FeedbackStore(Path("./artifacts/feedback.jsonl"))
_training_registry: dict[str, TrainingRunStatus] = {}
_training_results: dict[str, dict[str, Any]] = {}   # run_id → full TrainingReport dict


# ── Extended config schema for Orchestrator ──────────────────────────────────

class OrchestratorTrainingConfig(BaseModel):
    """Extended config for TrainingOrchestrator endpoint."""
    model_name: str
    dataset_id: str
    candidate_names: list[str] = Field(
        default=["logistic", "lightgbm", "xgboost"],
        description="Estimator names to evaluate. Parsimony: baseline wins within 0.005 IC margin.",
    )
    n_windows: int = Field(default=5, ge=5, description="Walk-forward windows (min 5).")
    embargo_days: int = Field(default=10, ge=5)
    cost_bps: float = Field(default=27.65, description="Indian primary cost scenario (27.65 bps).")
    horizon_bars: int = Field(default=5, ge=1, description="Label horizon for correct Sharpe annualization.")
    baseline_ic: float | None = Field(
        default=None,
        description=(
            "OOS IC of the best naive baseline. When set, champion must exceed "
            "this by parsimony_margin (0.005) to pass acceptance (mandate §47)."
        ),
    )
    register_champion: bool = Field(
        default=True,
        description="Register champion artifact in ModelRegistry at CHALLENGER stage.",
    )
    notes: str = ""


# ── Background training task ──────────────────────────────────────────────────

async def _run_orchestrator_training(
    run_id: str,
    config: OrchestratorTrainingConfig,
) -> None:
    """Execute TrainingOrchestrator in the background; update registry on completion."""
    _training_registry[run_id].status = "RUNNING"
    _training_registry[run_id].current_stage = "initialising_orchestrator"

    try:
        from src.data.dataset_builder import DatasetBuilder
        from src.registry.registry import ModelRegistry
        from src.training.orchestrator import TrainingOrchestrator

        # DatasetBuilder needs an output root — use the existing datasets dir
        builder = DatasetBuilder(output_root=Path("./artifacts/datasets"))
        registry = ModelRegistry()
        orch = TrainingOrchestrator(
            dataset_builder=builder,
            registry=registry,
            n_windows=config.n_windows,
            embargo_days=config.embargo_days,
            cost_bps=config.cost_bps,
            horizon_bars=config.horizon_bars,
        )

        _training_registry[run_id].current_stage = "walk_forward_validation"

        # Run in a thread so the async event loop isn't blocked
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(
            None,
            lambda: orch.train(
                model_name=config.model_name,
                dataset_id=config.dataset_id,
                candidate_names=config.candidate_names,
                register_champion=config.register_champion,
                baseline_ic=config.baseline_ic,
            ),
        )

        # Store result
        _training_results[run_id] = report.to_dict()

        status_str = "COMPLETED" if report.passed_acceptance else "COMPLETED_REJECTED"
        _training_registry[run_id].status = status_str
        _training_registry[run_id].completed_at = datetime.now(tz=timezone.utc)
        _training_registry[run_id].current_stage = (
            f"champion={report.champion} ic={report.champion_ic_mean:.4f}"
            if report.passed_acceptance
            else f"rejected: {report.rejection_reason}"
        )
        _training_registry[run_id].ic_mean = report.champion_ic_mean

        logger.info(
            "api_training_run_completed",
            run_id=run_id,
            model_name=config.model_name,
            passed=report.passed_acceptance,
            champion=report.champion,
            ic_mean=report.champion_ic_mean,
        )

    except Exception as exc:
        _training_registry[run_id].status = "FAILED"
        _training_registry[run_id].completed_at = datetime.now(tz=timezone.utc)
        _training_registry[run_id].error_message = str(exc)
        logger.error(
            "api_training_run_failed",
            run_id=run_id,
            error=str(exc),
            exc_info=True,
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/training/run", response_model=TrainingRunStatus)
async def training_run(
    config: OrchestratorTrainingConfig,
    background_tasks: BackgroundTasks,
) -> TrainingRunStatus:
    """
    Submit an async training run using **TrainingOrchestrator** (FIX NEW-P1-003).

    Immediately returns ``QUEUED`` status; training runs in the background.
    Poll ``GET /training/status/{run_id}`` for completion.

    The orchestrator executes:
    - Walk-forward validation (5+ windows, per-fold normalization)
    - Combinatorial purged CV (PBO estimate)
    - Calibration (Platt/isotonic, ECE gate)
    - Baseline comparison gate (mandate §47)
    - Champion selection (parsimony: baseline preferred within 0.005 IC margin)
    - Artifact registration at CHALLENGER stage (never auto-promoted to PRODUCTION)

    POST /training/run
    """
    run_id = str(uuid.uuid4())
    status = TrainingRunStatus(
        run_id=run_id,
        model_name=config.model_name,
        status="QUEUED",
        started_at=datetime.now(tz=timezone.utc),
        current_stage=f"queued: {config.model_name} on {config.dataset_id}",
    )
    _training_registry[run_id] = status

    background_tasks.add_task(_run_orchestrator_training, run_id, config)

    logger.info(
        "api_training_run_queued",
        run_id=run_id,
        model_name=config.model_name,
        dataset_id=config.dataset_id,
        candidates=config.candidate_names,
    )
    return status


@router.post("/training/run/sync")
async def training_run_sync(config: OrchestratorTrainingConfig) -> dict[str, Any]:
    """
    Synchronous training run — blocks until complete; returns full TrainingReport.

    Use for scripts and CI pipelines. Same orchestrator path as async endpoint.
    Not suitable for production API calls (may take several minutes).

    POST /training/run/sync
    """
    from src.data.dataset_builder import DatasetBuilder
    from src.registry.registry import ModelRegistry
    from src.training.orchestrator import TrainingOrchestrator

    builder = DatasetBuilder(output_root=Path("./artifacts/datasets"))
    registry = ModelRegistry()
    orch = TrainingOrchestrator(
        dataset_builder=builder,
        registry=registry,
        n_windows=config.n_windows,
        embargo_days=config.embargo_days,
        cost_bps=config.cost_bps,
        horizon_bars=config.horizon_bars,
    )

    try:
        report = orch.train(
            model_name=config.model_name,
            dataset_id=config.dataset_id,
            candidate_names=config.candidate_names,
            register_champion=config.register_champion,
            baseline_ic=config.baseline_ic,
        )
        return {
            "status": "COMPLETED" if report.passed_acceptance else "REJECTED",
            "report": report.to_dict(),
            "pipeline": "TrainingOrchestrator",
            "fix": "NEW-P1-003",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/training/status/{run_id}", response_model=TrainingRunStatus)
async def training_status(run_id: str) -> TrainingRunStatus:
    """Poll the status of a previously submitted training run.

    GET /training/status/{run_id}
    """
    status = _training_registry.get(run_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Training run {run_id!r} not found",
        )
    return status


@router.get("/training/result/{run_id}")
async def training_result(run_id: str) -> dict[str, Any]:
    """Return the full TrainingReport for a completed run.

    GET /training/result/{run_id}
    """
    result = _training_results.get(run_id)
    if result is None:
        status = _training_registry.get(run_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
        if status.status in ("QUEUED", "RUNNING"):
            raise HTTPException(status_code=202, detail=f"Run {run_id!r} still in progress")
        raise HTTPException(status_code=404, detail=f"No result for run {run_id!r}")
    return result


@router.post("/train/feedback")
async def train_feedback(record: FeedbackRecord) -> dict:
    """Ingest a trade-outcome feedback record from AlphaForge.

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
    """Return aggregate feedback summary.

    GET /train/feedback/summary
    """
    return _feedback_store.summary()


@router.get("/training/readiness")
async def training_readiness(
    news_required: bool = False,
    min_history_days: int = 252,
    min_universe_size: int = 3,
    timeframe: str = "1d",
) -> dict:
    """Training-readiness gate (mandate §27, §28).

    GET /training/readiness
    """
    from src.clients.data_service import DataServiceClient
    from src.clients.sentinel_pulse import SentinelPulseClient
    from src.data.readiness import TrainingReadinessGate

    data_client = DataServiceClient()
    await data_client.connect()
    sentinel_client = SentinelPulseClient()
    await sentinel_client.connect()
    gate = TrainingReadinessGate()

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
    except Exception as e:
        sample_universe = ["NIFTY", "BANKNIFTY", "RELIANCE"]

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
