"""
src/schemas — public re-exports for all ml-service2.0 Pydantic schemas and enums.

Import from this package rather than from individual sub-modules so that
internal module reorganisation never breaks callers.
"""
from __future__ import annotations

from src.schemas.base import (
    BaseSchema,
    DeploymentMode,
    DriftSeverity,
    EvidenceLevel,
    ExecutionAction,
    GateResult,
    ImpactDirection,
    IVRegime,
    MarketRegime,
    ModelLifecycleStage,
    PredictionProvenance,
    PromotionOutcome,
    RecommendedAction,
    TradingStrategy,
)
from src.schemas.features import FeatureQualityReport, FeatureVector
from src.schemas.meta import (
    ConfidenceDecomposition,
    MetaDecideRequest,
    MetaFitResponse,
    MetaOutput,
    MetaOutputExplainability,
    NewsSignal,
    OOSRecord,
)
from src.schemas.monitoring import DriftAlert
from src.schemas.registry import (
    GateEvaluation,
    ModelArtifact,
    PromotionDecision,
    TrainingConfig,
    TrainingRun,
    TrainingRunStatus,
)
from src.schemas.streaming import SignalEvent

__all__ = [
    # ── Base ──────────────────────────────────────────────────────────────────
    "BaseSchema",
    # ── Enums ─────────────────────────────────────────────────────────────────
    "DeploymentMode",
    "DriftSeverity",
    "EvidenceLevel",
    "ExecutionAction",
    "GateResult",
    "ImpactDirection",
    "IVRegime",
    "MarketRegime",
    "ModelLifecycleStage",
    "PredictionProvenance",
    "PromotionOutcome",
    "RecommendedAction",
    "TradingStrategy",
    # ── Feature schemas ───────────────────────────────────────────────────────
    "FeatureVector",
    "FeatureQualityReport",
    # ── Meta schemas ──────────────────────────────────────────────────────────
    "NewsSignal",
    "ConfidenceDecomposition",
    "MetaOutputExplainability",
    "MetaOutput",
    "MetaDecideRequest",
    "OOSRecord",
    "MetaFitResponse",
    # ── Registry / training schemas ───────────────────────────────────────────
    "ModelArtifact",
    "GateEvaluation",
    "PromotionDecision",
    "TrainingConfig",
    "TrainingRunStatus",
    "TrainingRun",
    # ── Monitoring schemas ────────────────────────────────────────────────────
    "DriftAlert",
    # ── Streaming schemas ─────────────────────────────────────────────────────
    "SignalEvent",
]
