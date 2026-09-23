"""
Registry, training, and promotion schemas for ml-service2.0.

ModelArtifact      — Versioned model artifact with SHA-256 integrity metadata.
GateEvaluation     — Result of a single promotion gate evaluation.
PromotionDecision  — Output of the six-gate ModelPromotion pipeline.
TrainingConfig     — Input configuration for a training run.
TrainingRunStatus  — Live status of an in-progress or completed training run.
TrainingRun        — Append-only audit log entry for a completed training run.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import Field

from src.schemas.base import (
    BaseSchema,
    EvidenceLevel,
    GateResult,
    ModelLifecycleStage,
    PredictionProvenance,
    PromotionOutcome,
)


class ModelArtifact(BaseSchema):
    """Versioned model artifact stored in the immutable artifact store.

    Loaded by ``ModelRegistry`` at startup; SHA-256 checksum is validated
    against the on-disk file before the model is allowed to serve predictions.
    """

    model_name: str  # e.g. "market_regime", "stock_ranker"
    version: str  # e.g. "1.0.0" or "1.2.3-online-2025-07-15"
    stage: ModelLifecycleStage
    artifact_path: str  # absolute filesystem path to the model file
    sha256_checksum: str  # hex string — validated at load time
    training_date: str  # ISO-8601 date string (YYYY-MM-DD)
    training_dataset_hash: str  # SHA-256 of the training dataset
    ic_mean: float = 0.0  # mean Spearman IC across validation folds
    sharpe_net: float = 0.0  # net Sharpe after 10bp transaction costs
    pbo: float = 0.0  # probability of backtest overfitting (CPCV)
    provenance: PredictionProvenance = PredictionProvenance.TRAINED_MODEL
    ope_sharpe: Optional[float] = None  # OPE estimated Sharpe — RL agents only
    consecutive_online_updates: int = 0  # resets on full retrain
    metadata: dict[str, Any] = Field(default_factory=dict)


class GateEvaluation(BaseSchema):
    """Result of evaluating a single promotion gate.

    ``gate_name`` is one of: DATA, PREDICTIVE, CALIBRATION, EXECUTION, RISK,
    STABILITY.  ``details`` carries gate-specific numeric metrics (e.g.
    challenger IC, champion IC, delta) for audit log serialisation.
    """

    gate_name: str  # "DATA" | "PREDICTIVE" | "CALIBRATION" | "EXECUTION" | "RISK" | "STABILITY"
    result: GateResult
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class PromotionDecision(BaseSchema):
    """Output of the six-gate ModelPromotion pipeline for a challenger/champion pair.

    ``outcome`` is PROMOTE only when all six gates return PASS *and* a valid
    ``approval_token`` has been issued by an authorised reviewer.  Any FAIL
    produces REJECTED; any INSUFFICIENT_EVIDENCE produces BLOCKED.
    """

    challenger_id: str
    champion_id: Optional[str] = None  # None when no champion exists yet
    outcome: PromotionOutcome
    gate_evaluations: list[GateEvaluation]
    approval_policy: str = ""  # "HUMAN_APPROVAL_REQUIRED" or "AUTOMATIC"
    timestamp: datetime
    reviewer_identity: Optional[str] = None  # populated when human-approved
    blocked_gates: list[str] = Field(default_factory=list)  # gate names with INSUFFICIENT_EVIDENCE


class TrainingConfig(BaseSchema):
    """Input configuration supplied to ``POST /training/run``.

    Validated by the TrainingPipeline before any data is fetched.  The
    ``embargo_period_days`` minimum of 10 matches the design default; the
    ``n_trials`` minimum of 50 matches the Optuna HPO requirement.
    """

    model_name: str
    feature_version: str = "latest"
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    embargo_period_days: int = Field(default=10, ge=5)  # min 5 per design
    n_trials: int = Field(default=50, ge=50)  # min 50 Optuna trials
    notes: str = ""


class TrainingRunStatus(BaseSchema):
    """Live status snapshot returned by ``GET /training/status/{run_id}``.

    ``status`` follows the lifecycle: RUNNING → COMPLETED | FAILED | ABORTED.
    ``current_stage`` is a human-readable description of the active pipeline
    step (e.g. "purged_kfold_fold_3_of_5", "optuna_trial_42_of_50").
    """

    run_id: str
    model_name: str
    status: str  # "RUNNING" | "COMPLETED" | "FAILED" | "ABORTED"
    started_at: datetime
    completed_at: Optional[datetime] = None
    current_stage: str = ""
    ic_mean: Optional[float] = None
    error_message: Optional[str] = None


class TrainingRun(BaseSchema):
    """Append-only audit log entry written at the end of every training run.

    Each entry is immutable once written.  ``gate_results`` maps gate name
    to GateResult string value (e.g. ``{"DATA": "pass", "PREDICTIVE": "fail"}``).
    ``approval_token`` is included when ``outcome`` is PROMOTE so that the
    token issuance event is co-located with the training run record.
    """

    run_id: str
    event_type: str = "training_run"
    model_name: str
    model_version: str
    started_at: datetime
    completed_at: datetime
    dataset_hash: str  # SHA-256 of training dataset
    training_date_range: tuple[str, str]  # (start_date, end_date) ISO-8601
    validation_date_range: tuple[str, str]  # (start_date, end_date) ISO-8601
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    ic_per_fold: list[float] = Field(default_factory=list)
    ic_mean: float = 0.0
    sharpe_net: float = 0.0
    max_drawdown: float = 0.0
    pbo: float = 0.0
    gate_results: dict[str, str] = Field(default_factory=dict)  # gate_name → GateResult value
    outcome: str = ""  # mirrors PromotionOutcome value
    approval_token: Optional[str] = None
    reviewer_identity: Optional[str] = None
