"""
src.validation — Financial ML validation framework.

Exports all public symbols so tests can do:
    from src.validation import walk_forward_splits, WalkForwardConfig, ...
"""

# Walk-forward splits
from src.validation.walk_forward import (
    WalkForwardFold,
    WalkForwardConfig,
    WalkForwardValidator,
    walk_forward_splits,
    walk_forward_splits_on_dates,
    save_fold_manifest,
)

# Embargo
from src.validation.embargo import (
    EmbargoConfig,
    EmbargoUnit,
    EmbargoApplier,
    compute_embargo_times,
    detect_label_overlap,
    purge_train_indices_by_t1,
)

# PurgedKFold + CPCV
from src.validation.purged_kfold import (
    PurgedKFold,
    CPCVConfig,
    CPCVFold,
    CombinatorialPurgedCV,
    cpcv_splits,
    build_t1_series,
)

# Metrics, evaluator, acceptance gate, persistence
from src.validation.metrics import (
    ClassificationMetrics,
    TradingMetrics,
    FoldResult,
    StabilityResult,
    AcceptanceThresholds,
    AcceptanceDecision,
    ValidationResult,
    FinancialMetricsEvaluator,
    ModelAcceptanceGate,
    compute_stability,
    aggregate_fold_results,
    save_validation_result,
    load_validation_history,
)

__all__ = [
    # walk-forward
    "WalkForwardFold",
    "WalkForwardConfig",
    "WalkForwardValidator",
    "walk_forward_splits",
    "walk_forward_splits_on_dates",
    "save_fold_manifest",
    # embargo
    "EmbargoConfig",
    "EmbargoUnit",
    "EmbargoApplier",
    "compute_embargo_times",
    "detect_label_overlap",
    "purge_train_indices_by_t1",
    # purged kfold / cpcv
    "PurgedKFold",
    "CPCVConfig",
    "CPCVFold",
    "CombinatorialPurgedCV",
    "cpcv_splits",
    "build_t1_series",
    # metrics
    "ClassificationMetrics",
    "TradingMetrics",
    "FoldResult",
    "StabilityResult",
    "AcceptanceThresholds",
    "AcceptanceDecision",
    "ValidationResult",
    "FinancialMetricsEvaluator",
    "ModelAcceptanceGate",
    "compute_stability",
    "aggregate_fold_results",
    "save_validation_result",
    "load_validation_history",
]
