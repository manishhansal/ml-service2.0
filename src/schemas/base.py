"""
Core enums and base Pydantic schema for ml-service2.0.

All enums are defined as (str, Enum) so their values serialize as plain strings
in JSON responses.  Every schema that inherits from BaseSchema gets strict
Pydantic V2 validation and immutability by default.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


# ── Enums ─────────────────────────────────────────────────────────────────────


class PredictionProvenance(str, Enum):
    """Classifies the source and evidence quality of every model prediction."""

    TRAINED_MODEL = "trained_model"
    HEURISTIC = "heuristic"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNAVAILABLE = "unavailable"

    @property
    def is_live_eligible(self) -> bool:
        """Only TRAINED_MODEL predictions are eligible for live capital deployment."""
        return self == PredictionProvenance.TRAINED_MODEL


class DeploymentMode(str, Enum):
    """Controls heuristic fallback behaviour across the service."""

    RESEARCH = "research"
    PAPER = "paper"
    SHADOW = "shadow"
    VALIDATED_ML_ONLY = "validated_ml_only"


class MarketRegime(str, Enum):
    """Six discrete market regimes produced by the RegimeClassifier."""

    STRONG_BULL = "strong_bull"
    BULL = "bull"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    BEAR = "bear"
    CRASH = "crash"


class TradingStrategy(str, Enum):
    """Eight trading strategies the StrategySelector can recommend."""

    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    VWAP_BOUNCE = "vwap_bounce"
    RANGE_TRADING = "range_trading"
    SCALPING = "scalping"
    VOLATILITY_BREAKOUT = "volatility_breakout"


class ExecutionAction(str, Enum):
    """Discrete action space for the RL execution agent."""

    ENTER_NOW = "enter_now"
    WAIT = "wait"
    SCALE_IN = "scale_in"
    PARTIAL_EXIT = "partial_exit"
    FULL_EXIT = "full_exit"
    TIGHTEN_STOP = "tighten_stop"
    TRAIL_STOP = "trail_stop"


class IVRegime(str, Enum):
    """Implied-volatility regime classification (mandatory input for StrategySelector)."""

    CRUSH = "CRUSH"
    STABLE = "STABLE"
    SPIKE = "SPIKE"


class ModelLifecycleStage(str, Enum):
    """Linear promotion stages a model must pass through before reaching production."""

    HYPOTHESIS = "hypothesis"
    BACKTEST = "backtest"
    CHALLENGER = "challenger"
    SHADOW = "shadow"
    APPROVED = "approved"
    PRODUCTION = "production"


class PromotionOutcome(str, Enum):
    """Outcome produced by the six-gate ModelPromotion pipeline."""

    PROMOTE = "promote"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    FINAL_OOS_CONTAMINATED = "final_oos_contaminated"


class GateResult(str, Enum):
    """Result of a single promotion gate evaluation."""

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DriftSeverity(str, Enum):
    """Severity levels emitted by the DriftMonitor."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImpactDirection(str, Enum):
    """Directional label for a SentinelPulse news impact signal."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class RecommendedAction(str, Enum):
    """Operator recommendation included in drift alert payloads."""

    MONITOR = "MONITOR"
    RETRAIN = "RETRAIN"
    ROLLBACK = "ROLLBACK"


class EvidenceLevel(str, Enum):
    """Evidence quality tier used in the DATA promotion gate."""

    LEVEL_A = "LEVEL_A"
    LEVEL_B = "LEVEL_B"
    LEVEL_C = "LEVEL_C"
    LEVEL_D = "LEVEL_D"


# ── Base schema ───────────────────────────────────────────────────────────────


class BaseSchema(BaseModel):
    """
    Root Pydantic V2 base class for all ml-service2.0 schemas.

    - ``strict=True``  prevents silent type coercion (e.g. "1" → 1).
    - ``frozen=True``  makes every instance effectively immutable, which
      supports idempotence guarantees required by Properties 2, 10, 11, 13.
    """

    model_config = ConfigDict(strict=True, frozen=True)


# ── resolve_action helper (used by tests and MetaDecisionEngine) ──────────────

def resolve_action(
    provenance: "PredictionProvenance",
    proposed_action: str,
    deployment_mode: "DeploymentMode",
) -> "tuple[str, PredictionProvenance]":
    """Resolve a proposed trading action against deployment-mode constraints.

    In VALIDATED_ML_ONLY mode, any non-trained-model provenance is downgraded
    to NO_TRADE with INSUFFICIENT_EVIDENCE provenance.

    Returns:
        (action, effective_provenance) tuple.
    """
    if deployment_mode == DeploymentMode.VALIDATED_ML_ONLY:
        if provenance != PredictionProvenance.TRAINED_MODEL:
            return "NO_TRADE", PredictionProvenance.INSUFFICIENT_EVIDENCE
    return proposed_action, provenance
