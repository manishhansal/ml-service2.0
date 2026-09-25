"""
AbstentionPolicy — determines when the MetaDecisionEngine should abstain from trading.

Five abstention conditions (Req 10.7):
1. agreement_ratio < 0.5
2. data_quality < 0.6
3. mean_confidence < 0.35
4. risk.prob_stop_hit > 0.65
5. number of available (non-UNAVAILABLE) models < 3

When any condition is met, the engine returns NO_TRADE with abstention=True.

Requirements: Req 10.7
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)

AGREEMENT_THRESHOLD = 0.5
DATA_QUALITY_THRESHOLD = 0.6
MEAN_CONFIDENCE_THRESHOLD = 0.35
PROB_STOP_HIT_THRESHOLD = 0.65
MIN_AVAILABLE_MODELS = 3


@dataclass
class AbstentionContext:
    """Input context for the AbstentionPolicy."""
    agreement_ratio: float = 0.0
    data_quality: float = 1.0
    mean_confidence: float = 0.5
    prob_stop_hit: float = 0.0
    n_available_models: int = 7
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AbstentionResult:
    """Result of the AbstentionPolicy check."""
    should_abstain: bool
    reason_codes: list[str]

    @property
    def triggered_reasons(self) -> str:
        return ", ".join(self.reason_codes) if self.reason_codes else "none"


class AbstentionPolicy:
    """
    Evaluates five conditions and returns whether the engine should abstain.

    Usage::
        policy = AbstentionPolicy()
        ctx = AbstentionContext(agreement_ratio=0.4, data_quality=0.5, ...)
        result = policy.check(ctx)
        if result.should_abstain:
            # return NO_TRADE
    """

    def check(self, context: AbstentionContext) -> AbstentionResult:
        """
        Evaluate all five abstention conditions.

        Returns:
            AbstentionResult with should_abstain=True if ANY condition is met.
            reason_codes lists all triggered conditions.
        """
        reason_codes: list[str] = []

        # Condition 1: agreement_ratio < 0.5
        if context.agreement_ratio < AGREEMENT_THRESHOLD:
            reason_codes.append("LOW_AGREEMENT")

        # Condition 2: data_quality < 0.6
        if context.data_quality < DATA_QUALITY_THRESHOLD:
            reason_codes.append("LOW_DATA_QUALITY")

        # Condition 3: mean_confidence < 0.35
        if context.mean_confidence < MEAN_CONFIDENCE_THRESHOLD:
            reason_codes.append("LOW_CONFIDENCE")

        # Condition 4: prob_stop_hit > 0.65
        if context.prob_stop_hit > PROB_STOP_HIT_THRESHOLD:
            reason_codes.append("HIGH_STOP_PROBABILITY")

        # Condition 5: fewer than 3 available models
        if context.n_available_models < MIN_AVAILABLE_MODELS:
            reason_codes.append("INSUFFICIENT_MODELS")

        should_abstain = len(reason_codes) > 0

        if should_abstain:
            logger.info(
                "abstention_triggered",
                reasons=reason_codes,
                agreement_ratio=context.agreement_ratio,
                n_available_models=context.n_available_models,
            )

        return AbstentionResult(
            should_abstain=should_abstain,
            reason_codes=reason_codes,
        )

    def check_from_dict(self, context_dict: dict[str, Any]) -> AbstentionResult:
        """Convenience wrapper accepting a plain dict."""
        ctx = AbstentionContext(
            agreement_ratio=float(context_dict.get("agreement_ratio", 0.0)),
            data_quality=float(context_dict.get("data_quality", 1.0)),
            mean_confidence=float(context_dict.get("mean_confidence", 0.5)),
            prob_stop_hit=float(context_dict.get("prob_stop_hit", 0.0)),
            n_available_models=int(context_dict.get("n_available_models", 7)),
        )
        return self.check(ctx)


# ── Backward-compatibility alias ──────────────────────────────────────────────
# Some test files and the existing meta_model.py import AbstentionDecision
# rather than AbstentionResult. Both names refer to the same dataclass.
AbstentionDecision = AbstentionResult

# Also export AbstentionInputs as an alias for AbstentionContext
AbstentionInputs = AbstentionContext


# AbstentionKind — classifies the reason for abstention
from enum import Enum


class AbstentionKind(str, Enum):
    """Reason for a MetaDecisionEngine abstention."""

    LOW_AGREEMENT = "low_agreement"
    LOW_DATA_QUALITY = "low_data_quality"
    HIGH_UNCERTAINTY = "high_uncertainty"
    STALE_DATA = "stale_data"
    MODEL_UNAVAILABLE = "model_unavailable"
    REGIME_MISMATCH = "regime_mismatch"
    CALIBRATION_MISSING = "calibration_missing"
    MANUAL_OVERRIDE = "manual_override"


# AbstentionThresholds — configuration for AbstentionPolicy
from dataclasses import dataclass as _dc2


@_dc2
class AbstentionThresholds:
    """Configurable thresholds for the AbstentionPolicy."""

    min_agreement_ratio: float = 0.5
    min_data_quality: float = 0.6
    max_uncertainty: float = 0.35
    max_stale_seconds: float = 3600.0
