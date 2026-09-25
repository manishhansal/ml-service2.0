"""
Decision Policy for the Meta Decision Engine.

Responsibilities
----------------
1. Confidence decomposition
   The final confidence is broken into five orthogonal components:
     - signalConfidence   : how strong the directional signal is
     - modelAgreement     : how well the models agree
     - dataQuality        : quality of input features
     - regimeConfidence   : certainty of the regime classifier
     - executionQuality   : suitability of current conditions for execution

2. Reason codes
   Every output carries a list of machine-readable reason codes explaining
   what drove the decision and what constrained it.

3. Explainability
   Top positive factors, top negative factors, and list of disagreeing models
   are surfaced so the frontend can render a human-readable rationale.

4. Final decision assembly
   Maps the weighted ensemble score + confidence decomposition into a
   concrete action: BUY | SELL | WAIT | NO_TRADE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import numpy as np
import structlog

from .ensemble import EnsembleResult, ModelSignal

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TradeAction(str, Enum):
    """Final meta-decision actions."""

    BUY      = "BUY"
    SELL     = "SELL"
    WAIT     = "WAIT"
    NO_TRADE = "NO_TRADE"


class ReasonCode(str, Enum):
    """Machine-readable reason codes attached to every decision."""

    # Positive signals
    STRONG_CONSENSUS       = "strong_consensus"
    REGIME_ALIGNED         = "regime_aligned"
    HIGH_MODEL_CONFIDENCE  = "high_model_confidence"
    GOOD_RISK_REWARD       = "good_risk_reward"
    MOMENTUM_CONFIRMED     = "momentum_confirmed"
    LOW_IV_ENVIRONMENT     = "low_iv_environment"

    # Negative signals / constraints
    MODELS_DISAGREE        = "models_disagree"
    WEAK_SIGNAL            = "weak_signal"
    HIGH_RISK              = "high_risk"
    REGIME_UNCERTAIN       = "regime_uncertain"
    DATA_QUALITY_LOW       = "data_quality_low"
    HIGH_UNCERTAINTY       = "high_uncertainty"
    REGIME_UNFAVORABLE     = "regime_unfavorable"
    IV_SPIKE               = "iv_spike"
    EXECUTION_CONDITIONS_POOR = "execution_conditions_poor"

    # Informational
    CALIBRATION_APPLIED    = "calibration_applied"
    ABSTENTION_TRIGGERED   = "abstention_triggered"
    FALLBACK_USED          = "fallback_used"


# ---------------------------------------------------------------------------
# Confidence decomposition
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceDecomposition:
    """
    Five-component breakdown of the final confidence score.

    Each component ∈ [0, 1].  The final confidence is a weighted
    combination of all five so that weaknesses in any dimension are
    transparently reflected.

    Example
    -------
    Final Confidence = 0.78
      signalConfidence   = 0.82
      modelAgreement     = 0.91
      dataQuality        = 0.97
      regimeConfidence   = 0.82
      executionQuality   = 0.76
    """

    signal_confidence: float    # strength of the raw directional signal
    model_agreement: float      # agreement ratio across models
    data_quality: float         # input feature quality
    regime_confidence: float    # regime classifier certainty
    execution_quality: float    # execution suitability (time, spread, IV)

    # Component weights — tuned so agreement and signal matter most
    _WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {
            "signal_confidence": 0.30,
            "model_agreement":   0.25,
            "data_quality":      0.15,
            "regime_confidence": 0.20,
            "execution_quality": 0.10,
        },
        repr=False,
    )

    @property
    def final_confidence(self) -> float:
        """Weighted composite confidence ∈ [0, 1]."""
        w = self._WEIGHTS
        return float(np.clip(
            w["signal_confidence"]   * self.signal_confidence
            + w["model_agreement"]   * self.model_agreement
            + w["data_quality"]      * self.data_quality
            + w["regime_confidence"] * self.regime_confidence
            + w["execution_quality"] * self.execution_quality,
            0.0, 1.0,
        ))

    def to_dict(self) -> dict[str, float]:
        return {
            "signalConfidence":  round(self.signal_confidence, 4),
            "modelAgreement":    round(self.model_agreement, 4),
            "dataQuality":       round(self.data_quality, 4),
            "regimeConfidence":  round(self.regime_confidence, 4),
            "executionQuality":  round(self.execution_quality, 4),
            "finalConfidence":   round(self.final_confidence, 4),
        }


# ---------------------------------------------------------------------------
# Per-model contribution for explainability
# ---------------------------------------------------------------------------

class ModelContribution(NamedTuple):
    """A single model's contribution to the final decision."""

    model_name: str
    direction: int         # +1 | 0 | -1
    confidence: float
    weight: float          # normalised weight in the ensemble
    contribution: float    # weight × direction × confidence  ∈ [-1, 1]
    agreed_with_decision: bool


@dataclass
class ExplainabilityReport:
    """
    Human-readable explanation of the meta decision.

    Attributes
    ----------
    top_positive_factors : reasons supporting the action
    top_negative_factors : reasons constraining or opposing the action
    disagreeing_models   : models that pointed in the minority direction
    contributing_models  : full breakdown by model
    """

    top_positive_factors: list[str]
    top_negative_factors: list[str]
    disagreeing_models: list[str]
    contributing_models: list[ModelContribution]

    def to_dict(self) -> dict:
        return {
            "topPositiveFactors": self.top_positive_factors,
            "topNegativeFactors": self.top_negative_factors,
            "disagreeingModels":  self.disagreeing_models,
            "contributingModels": [
                {
                    "model":                c.model_name,
                    "direction":            c.direction,
                    "confidence":           round(c.confidence, 4),
                    "weight":               round(c.weight, 4),
                    "contribution":         round(c.contribution, 4),
                    "agreedWithDecision":   c.agreed_with_decision,
                }
                for c in self.contributing_models
            ],
        }


# ---------------------------------------------------------------------------
# DecisionPolicy
# ---------------------------------------------------------------------------

@dataclass
class DecisionPolicyConfig:
    """Tunable thresholds for the decision policy."""

    # Minimum |ensemble_score| for BUY or SELL
    min_action_score: float  = 0.15
    # Minimum final_confidence for BUY or SELL
    min_action_confidence: float = 0.55
    # Score must be positive for BUY, negative for SELL
    # (enforced automatically)

    # Execution quality: penalise last 15 min of session (high spread)
    session_close_penalty_minutes: int = 15
    # Full session length in minutes (NSE: 375)
    session_length_minutes: int = 375


class DecisionPolicy:
    """
    Assembles the final decision from ensemble results and confidence decomposition.

    Parameters
    ----------
    config : DecisionPolicyConfig (defaults used if None)
    """

    def __init__(self, config: DecisionPolicyConfig | None = None) -> None:
        self.config = config or DecisionPolicyConfig()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build_decision(
        self,
        *,
        ensemble_result: EnsembleResult,
        signals: list[ModelSignal],
        data_quality: float,
        regime_confidence: float,
        prob_stop_hit: float,
        time_of_day_minutes: int | None = None,
        iv_regime: str | None = None,
        abstention_kind: str = "none",
    ) -> tuple["DecisionOutput", ConfidenceDecomposition, ExplainabilityReport]:
        """
        Build the complete decision output.

        Parameters
        ----------
        ensemble_result      : output from EnsembleWeighter.compute()
        signals              : original list of ModelSignal (for explainability)
        data_quality         : composite data quality ∈ [0, 1]
        regime_confidence    : regime classifier confidence ∈ [0, 1]
        prob_stop_hit        : risk model P(stop hit) ∈ [0, 1]
        time_of_day_minutes  : minutes since market open (0–375)
        iv_regime            : "CRUSH" | "STABLE" | "SPIKE" from IVClassifier
        abstention_kind      : "none" | "wait" | "no_trade" from AbstentionPolicy

        Returns
        -------
        (DecisionOutput, ConfidenceDecomposition, ExplainabilityReport)
        """
        cfg = self.config
        er = ensemble_result

        # ── 1. Confidence decomposition ──────────────────────────────────
        # Phase 3F fix (BUG #6): signal_confidence was abs(weighted_score),
        # which is a score magnitude — NOT a probability.  Now uses
        # weighted_confidence from EnsembleResult, which is the
        # calibration-quality-weighted mean of calibrated model confidences.
        # This is a proper [0,1] probability-like quantity.
        signal_conf = float(np.clip(er.weighted_confidence, 0.0, 1.0))
        exec_quality = self._execution_quality(
            time_of_day_minutes=time_of_day_minutes,
            iv_regime=iv_regime,
            prob_stop_hit=prob_stop_hit,
        )

        decomp = ConfidenceDecomposition(
            signal_confidence=signal_conf,
            model_agreement=er.disagreement.agreement_ratio,
            data_quality=data_quality,
            regime_confidence=regime_confidence,
            execution_quality=exec_quality,
        )

        final_confidence = decomp.final_confidence

        # ── 2. Reason codes ──────────────────────────────────────────────
        reason_codes = self._collect_reason_codes(
            ensemble_result=er,
            decomp=decomp,
            prob_stop_hit=prob_stop_hit,
            iv_regime=iv_regime,
            abstention_kind=abstention_kind,
        )

        # ── 3. Action ────────────────────────────────────────────────────
        action = self._determine_action(
            score=er.weighted_score,
            confidence=final_confidence,
            abstention_kind=abstention_kind,
            cfg=cfg,
        )

        # ── 4. Explainability report ─────────────────────────────────────
        explain = self._build_explainability(
            signals=signals,
            ensemble_result=er,
            action=action,
        )

        # ── 5. Uncertainty ───────────────────────────────────────────────
        uncertainty = float(np.clip(
            1.0 - final_confidence
            + 0.1 * er.disagreement.prediction_variance
            + 0.05 * er.disagreement.confidence_spread,
            0.0, 1.0,
        ))

        output = DecisionOutput(
            action=action,
            confidence=final_confidence,
            uncertainty=uncertainty,
            agreement=er.disagreement.agreement_ratio,
            reason_codes=[r.value for r in reason_codes],
            contributing_models=er.active_models,
            ensemble_score=er.weighted_score,
            decomposition=decomp.to_dict(),
        )

        logger.info(
            "meta_decision",
            action=action.value,
            confidence=round(final_confidence, 3),
            agreement=round(er.disagreement.agreement_ratio, 3),
            uncertainty=round(uncertainty, 3),
            reasons=[r.value for r in reason_codes],
        )

        return output, decomp, explain

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_action(
        score: float,
        confidence: float,
        abstention_kind: str,
        cfg: DecisionPolicyConfig,
    ) -> TradeAction:
        """Map ensemble score + abstention kind to a TradeAction."""
        if abstention_kind == "no_trade":
            return TradeAction.NO_TRADE
        if abstention_kind == "wait":
            return TradeAction.WAIT
        if (
            abs(score) < cfg.min_action_score
            or confidence < cfg.min_action_confidence
        ):
            return TradeAction.WAIT
        return TradeAction.BUY if score > 0 else TradeAction.SELL

    @staticmethod
    def _execution_quality(
        time_of_day_minutes: int | None,
        iv_regime: str | None,
        prob_stop_hit: float,
    ) -> float:
        """
        Estimate execution quality ∈ [0, 1] from session timing,
        IV environment, and stop-hit probability.
        """
        score = 1.0

        # Session timing: opening (0-15 min) and closing (360-375 min) are noisier
        if time_of_day_minutes is not None:
            if time_of_day_minutes < 15 or time_of_day_minutes > 360:
                score *= 0.75
            elif time_of_day_minutes > 345:
                score *= 0.88

        # IV environment
        if iv_regime == "SPIKE":
            score *= 0.80   # wide spreads, slippage likely
        elif iv_regime == "CRUSH":
            score *= 0.92   # options cheapening — fine for equity trades

        # Risk-adjusted: high stop probability hurts execution quality
        score *= max(0.5, 1.0 - prob_stop_hit * 0.4)

        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _collect_reason_codes(
        ensemble_result: EnsembleResult,
        decomp: ConfidenceDecomposition,
        prob_stop_hit: float,
        iv_regime: str | None,
        abstention_kind: str,
    ) -> list[ReasonCode]:
        codes: list[ReasonCode] = []
        er = ensemble_result

        # Positive signals
        if er.disagreement.agreement_ratio >= 0.75:
            codes.append(ReasonCode.STRONG_CONSENSUS)
        if decomp.regime_confidence >= 0.70:
            codes.append(ReasonCode.REGIME_ALIGNED)
        if decomp.signal_confidence >= 0.65:
            codes.append(ReasonCode.HIGH_MODEL_CONFIDENCE)
        if prob_stop_hit <= 0.35:
            codes.append(ReasonCode.GOOD_RISK_REWARD)
        if iv_regime == "STABLE":
            codes.append(ReasonCode.LOW_IV_ENVIRONMENT)

        # Negative signals
        if er.disagreement.direction_disagreement > 0.35:
            codes.append(ReasonCode.MODELS_DISAGREE)
        if abs(er.weighted_score) < 0.20:
            codes.append(ReasonCode.WEAK_SIGNAL)
        if prob_stop_hit > 0.65:
            codes.append(ReasonCode.HIGH_RISK)
        if decomp.regime_confidence < 0.55:
            codes.append(ReasonCode.REGIME_UNCERTAIN)
        if decomp.data_quality < 0.65:
            codes.append(ReasonCode.DATA_QUALITY_LOW)
        if decomp.signal_confidence < 0.50:
            codes.append(ReasonCode.HIGH_UNCERTAINTY)
        if decomp.execution_quality < 0.70:
            codes.append(ReasonCode.EXECUTION_CONDITIONS_POOR)
        if iv_regime == "SPIKE":
            codes.append(ReasonCode.IV_SPIKE)

        # Informational
        if abstention_kind in ("wait", "no_trade"):
            codes.append(ReasonCode.ABSTENTION_TRIGGERED)

        return codes

    @staticmethod
    def _build_explainability(
        signals: list[ModelSignal],
        ensemble_result: EnsembleResult,
        action: TradeAction,
    ) -> ExplainabilityReport:
        """Build a human-readable explainability report."""
        er = ensemble_result
        action_direction = 1 if action == TradeAction.BUY else (-1 if action == TradeAction.SELL else 0)

        contributions: list[ModelContribution] = []
        for sig in signals:
            if not sig.is_available:
                continue
            weight = er.weights.get(sig.model_name, 0.0)
            contrib = weight * sig.direction * sig.confidence
            contributions.append(ModelContribution(
                model_name=sig.model_name,
                direction=sig.direction,
                confidence=sig.confidence,
                weight=weight,
                contribution=contrib,
                agreed_with_decision=(
                    action_direction == 0
                    or sig.direction == action_direction
                    or sig.direction == 0
                ),
            ))

        # Sort by absolute contribution
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)

        # Top positive factors: models aligned with the decision, sorted by contribution
        top_positive = [
            f"{c.model_name} (confidence={c.confidence:.2f}, "
            f"weight={c.weight:.2f}, contribution={c.contribution:+.3f})"
            for c in contributions
            if c.contribution > 0
        ][:3]

        # Top negative factors: models opposing or dampening the decision
        top_negative = [
            f"{c.model_name} (confidence={c.confidence:.2f}, "
            f"weight={c.weight:.2f}, contribution={c.contribution:+.3f})"
            for c in contributions
            if c.contribution < 0 or not c.agreed_with_decision
        ][:3]

        disagreeing = er.penalised_models[:]

        return ExplainabilityReport(
            top_positive_factors=top_positive,
            top_negative_factors=top_negative,
            disagreeing_models=disagreeing,
            contributing_models=contributions,
        )


# ---------------------------------------------------------------------------
# DecisionOutput — the final output schema of the meta engine
# ---------------------------------------------------------------------------

@dataclass
class DecisionOutput:
    """
    Final output of the Meta Decision Engine for one evaluation.

    Attributes
    ----------
    action             : BUY | SELL | WAIT | NO_TRADE
    confidence         : composite confidence ∈ [0, 1]
    uncertainty        : complementary uncertainty measure ∈ [0, 1]
    agreement          : fraction of models agreeing on action ∈ [0, 1]
    reason_codes       : machine-readable list of ReasonCode values
    contributing_models: model names that contributed to the decision
    ensemble_score     : raw weighted aggregate score ∈ [-1, 1]
    decomposition      : dict with the five confidence components
    """

    action: TradeAction
    confidence: float
    uncertainty: float
    agreement: float
    reason_codes: list[str]
    contributing_models: list[str]
    ensemble_score: float
    decomposition: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "action":             self.action.value,
            "confidence":         round(self.confidence, 4),
            "uncertainty":        round(self.uncertainty, 4),
            "agreement":          round(self.agreement, 4),
            "reasonCodes":        self.reason_codes,
            "contributingModels": self.contributing_models,
            "ensembleScore":      round(self.ensemble_score, 4),
            "decomposition":      {k: round(v, 4) for k, v in self.decomposition.items()},
        }
