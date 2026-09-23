"""
ModelPromotion — six-gate champion/challenger promotion pipeline.

Gates (evaluated in order):
  DATA         — OOS data quality and integrity
  PREDICTIVE   — IC improvement over champion
  CALIBRATION  — Brier score comparison
  EXECUTION    — Net return and turnover
  RISK         — Maximum drawdown
  STABILITY    — IC decay and feature drift

Each gate returns: PASS | FAIL | INSUFFICIENT_EVIDENCE

The outcome is:
  PROMOTE               — all gates PASS (requires human approvalToken)
  BLOCKED               — any gate returns INSUFFICIENT_EVIDENCE
  REJECTED              — any gate FAILS
  FINAL_OOS_CONTAMINATED — challenger used final OOS data (permanently blocked)

Requirements: Req 13.1 through Req 13.14
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.audit.logger import AuditLogger
from src.config import settings
from src.logging_config import get_logger
from src.schemas.base import (
    EvidenceLevel,
    GateResult,
    ModelLifecycleStage,
    PromotionOutcome,
)
from src.schemas.registry import GateEvaluation, ModelArtifact, PromotionDecision

logger = get_logger(__name__)

# ── Gate thresholds ────────────────────────────────────────────────────────────

MIN_OOS_OBSERVATIONS = 60
MAX_DRAWDOWN_WITHOUT_CHAMPION = 0.20  # 20%
BRIER_SCORE_THRESHOLD_WITHOUT_CHAMPION = 0.25
PREDICTIVE_GATE_IC_THRESHOLD_NO_CHAMPION = 0.0
CALIBRATION_TOLERANCE = 0.01  # Brier can be at most 0.01 worse than champion
RISK_TOLERANCE_PP = 0.02  # max drawdown may be 2 percentage points worse than champion
TURNOVER_MAX_RELATIVE_INCREASE = 0.20  # challenger turnover <= champion * 1.20

# Evidence levels that cause immediate DATA gate failure (Req 13.12)
_BELOW_MINIMUM_EVIDENCE = {EvidenceLevel.LEVEL_C.value, EvidenceLevel.LEVEL_D.value}

# IC decay / drift values that trigger STABILITY gate failure (Req 13.8)
_FAILING_IC_DECAY_STATUSES = {"FAILED", "SIGNIFICANT_DECAY"}
_FAILING_DRIFT_SEVERITIES = {"HIGH", "CRITICAL"}


class ModelPromotion:
    """
    Evaluates a challenger model against a champion (or bootstrap criteria)
    across six structured promotion gates.

    Usage::

        promotion = ModelPromotion()
        decision = promotion.evaluate_all_gates(challenger_dict, champion_dict)
        if decision.outcome == PromotionOutcome.PROMOTE:
            ...  # present approvalToken workflow
    """

    def __init__(self, audit_logger: AuditLogger | None = None) -> None:
        self._audit = audit_logger or AuditLogger()

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate_all_gates(
        self,
        challenger: dict[str, Any],
        champion: dict[str, Any] | None = None,
    ) -> PromotionDecision:
        """
        Evaluate all six gates and return a ``PromotionDecision``.

        The gates are evaluated in order: DATA → PREDICTIVE → CALIBRATION →
        EXECUTION → RISK → STABILITY.  If the challenger is permanently
        contaminated (``final_oos_used_for_selection=True``), the DATA gate
        returns FAIL and the overall outcome is ``FINAL_OOS_CONTAMINATED``
        regardless of other gate results (Req 13.13).

        Args:
            challenger: Evaluation metrics dict with keys:
                - ``model_name``               str
                - ``version``                  str
                - ``ic_mean``                  float  — mean Spearman IC
                - ``brier_score``              float  — calibration score
                - ``max_drawdown``             float  — fraction (e.g. 0.15 = 15%)
                - ``oos_count``                int    — number of OOS ticker–date pairs
                - ``look_ahead_validated``     bool
                - ``evidence_level``           str    — EvidenceLevel value
                - ``final_oos_used_for_selection`` bool
                - ``ic_decay_status``          str    — "OK"|"FAILED"|"SIGNIFICANT_DECAY"
                - ``feature_drift_severity``   str    — "LOW"|"MEDIUM"|"HIGH"|"CRITICAL"
                - ``net_return_validation``    float  — net return on validation partition
                - ``turnover_validation``      float  — turnover on validation partition
                - ``sharpe_net``               float  (optional, informational)
            champion: Same structure as *challenger*, or ``None`` when no
                      champion exists yet.

        Returns:
            ``PromotionDecision`` with outcome, per-gate evaluations, and
            blocked gate names.
        """
        # Check for permanent OOS contamination before running gates (Req 13.13)
        final_oos_contaminated = bool(challenger.get("final_oos_used_for_selection", False))

        gates: list[GateEvaluation] = [
            self._evaluate_data_gate(challenger, final_oos_contaminated),
            self._evaluate_predictive_gate(challenger, champion),
            self._evaluate_calibration_gate(challenger, champion),
            self._evaluate_execution_gate(challenger, champion),
            self._evaluate_risk_gate(challenger, champion),
            self._evaluate_stability_gate(challenger),
        ]

        # Determine outcome
        outcome, blocked_gates = self._determine_outcome(gates)

        # Contaminated challengers are permanently blocked regardless of other gates
        if final_oos_contaminated:
            outcome = PromotionOutcome.FINAL_OOS_CONTAMINATED

        challenger_id = (
            f"{challenger.get('model_name', 'unknown')}-"
            f"{challenger.get('version', '0.0.0')}"
        )
        champion_id: str | None = None
        if champion:
            champion_id = (
                f"{champion.get('model_name', 'unknown')}-"
                f"{champion.get('version', '0.0.0')}"
            )

        decision = PromotionDecision(
            challenger_id=challenger_id,
            champion_id=champion_id,
            outcome=outcome,
            gate_evaluations=gates,
            approval_policy=(
                "HUMAN_APPROVAL_REQUIRED"
                if outcome == PromotionOutcome.PROMOTE
                else "N/A"
            ),
            timestamp=datetime.now(tz=timezone.utc),
            blocked_gates=blocked_gates,
        )

        # Write to the immutable audit log (Req 13.14)
        try:
            self._audit.log_promotion_decision(
                challenger_id=challenger_id,
                champion_id=champion_id,
                outcome=outcome.value,
                gate_results={g.gate_name: g.result.value for g in gates},
                approval_policy=decision.approval_policy,
                blocked_gates=blocked_gates,
            )
        except Exception as exc:
            logger.warning("promotion_audit_log_failed", error=str(exc))

        return decision

    # ── Gate implementations ───────────────────────────────────────────────────

    def _evaluate_data_gate(
        self, challenger: dict[str, Any], final_oos_contaminated: bool
    ) -> GateEvaluation:
        """
        DATA gate — enforces OOS data quality and integrity (Req 13.3, 13.12, 13.13).

        FAIL conditions (in priority order):
        1. ``final_oos_used_for_selection`` is True  → permanent contamination
        2. Evidence level is LEVEL_C or LEVEL_D      → insufficient quality
        3. ``look_ahead_validated`` is False          → integrity failure
        4. OOS count < 60                            → insufficient evidence
        """
        gate_name = "DATA"

        # (1) Final OOS contamination — permanent disqualification (Req 13.13)
        if final_oos_contaminated:
            return GateEvaluation(
                gate_name=gate_name,
                result=GateResult.FAIL,
                reason="Challenger is permanently disqualified: final OOS partition was used during training/selection (FINAL_OOS_CONTAMINATED)",
                details={"final_oos_used_for_selection": True},
            )

        # (2) Evidence level must be at least LEVEL_B (Req 13.12)
        evidence_level = challenger.get("evidence_level", EvidenceLevel.LEVEL_D.value)
        if evidence_level in _BELOW_MINIMUM_EVIDENCE:
            return GateEvaluation(
                gate_name=gate_name,
                result=GateResult.FAIL,
                reason=f"Evidence level {evidence_level!r} is below the LEVEL_B minimum required for promotion",
                details={"evidence_level": evidence_level},
            )

        # (3) Look-ahead validation must pass (Req 13.3)
        if not challenger.get("look_ahead_validated", False):
            return GateEvaluation(
                gate_name=gate_name,
                result=GateResult.FAIL,
                reason="look_ahead_validated is False — data integrity check failed",
                details={"look_ahead_validated": False},
            )

        # (4) OOS observation count must meet minimum (Req 13.3)
        oos_count = int(challenger.get("oos_count", 0))
        if oos_count < MIN_OOS_OBSERVATIONS:
            return GateEvaluation(
                gate_name=gate_name,
                result=GateResult.INSUFFICIENT_EVIDENCE,
                reason=(
                    f"OOS observation count {oos_count} is below the minimum "
                    f"{MIN_OOS_OBSERVATIONS} required for promotion"
                ),
                details={"oos_count": oos_count, "required": MIN_OOS_OBSERVATIONS},
            )

        return GateEvaluation(
            gate_name=gate_name,
            result=GateResult.PASS,
            reason="Data quality and integrity checks passed",
            details={
                "oos_count": oos_count,
                "evidence_level": evidence_level,
                "look_ahead_validated": True,
            },
        )

    def _evaluate_predictive_gate(
        self,
        challenger: dict[str, Any],
        champion: dict[str, Any] | None,
    ) -> GateEvaluation:
        """
        PREDICTIVE gate — IC must exceed champion IC by configured margin,
        or be strictly positive when no champion exists (Req 13.4).
        """
        gate_name = "PREDICTIVE"
        challenger_ic = float(challenger.get("ic_mean", 0.0))

        if champion is not None:
            champion_ic = float(champion.get("ic_mean", 0.0))
            margin: float = settings.predictive_gate_margin
            required = champion_ic + margin

            if challenger_ic > required:
                return GateEvaluation(
                    gate_name=gate_name,
                    result=GateResult.PASS,
                    reason=(
                        f"Challenger IC {challenger_ic:.4f} exceeds "
                        f"champion IC {champion_ic:.4f} + margin {margin:.4f} = {required:.4f}"
                    ),
                    details={
                        "challenger_ic": challenger_ic,
                        "champion_ic": champion_ic,
                        "margin": margin,
                        "required": required,
                    },
                )
            return GateEvaluation(
                gate_name=gate_name,
                result=GateResult.FAIL,
                reason=(
                    f"Challenger IC {challenger_ic:.4f} does not exceed "
                    f"champion IC {champion_ic:.4f} + margin {margin:.4f} = {required:.4f}"
                ),
                details={
                    "challenger_ic": challenger_ic,
                    "champion_ic": champion_ic,
                    "margin": margin,
                    "required": required,
                },
            )

        # No champion: IC must be strictly positive (Req 13.4)
        if challenger_ic > PREDICTIVE_GATE_IC_THRESHOLD_NO_CHAMPION:
            return GateEvaluation(
                gate_name=gate_name,
                result=GateResult.PASS,
                reason=f"Challenger IC {challenger_ic:.4f} > 0 (no champion exists)",
                details={"challenger_ic": challenger_ic},
            )
        return GateEvaluation(
            gate_name=gate_name,
            result=GateResult.FAIL,
            reason=f"Challenger IC {challenger_ic:.4f} <= 0 (no champion exists; IC must be positive)",
            details={"challenger_ic": challenger_ic},
        )

    def _evaluate_calibration_gate(
        self,
        challenger: dict[str, Any],
        champion: dict[str, Any] | None,
    ) -> GateEvaluation:
        """
        CALIBRATION gate — Brier score must not be more than 0.01 worse than
        champion's; or must be < 0.25 when no champion exists (Req 13.5).
        """
        gate_name = "CALIBRATION"
        challenger_brier = float(challenger.get("brier_score", 1.0))

        if champion is not None:
            champion_brier = float(champion.get("brier_score", 1.0))
            threshold = champion_brier + CALIBRATION_TOLERANCE

            if challenger_brier <= threshold:
                return GateEvaluation(
                    gate_name=gate_name,
                    result=GateResult.PASS,
                    reason=(
                        f"Challenger Brier {challenger_brier:.4f} <= "
                        f"champion {champion_brier:.4f} + tolerance {CALIBRATION_TOLERANCE} = {threshold:.4f}"
                    ),
                    details={
                        "challenger_brier": challenger_brier,
                        "champion_brier": champion_brier,
                        "tolerance": CALIBRATION_TOLERANCE,
                    },
                )
            return GateEvaluation(
                gate_name=gate_name,
                result=GateResult.FAIL,
                reason=(
                    f"Challenger Brier {challenger_brier:.4f} exceeds "
                    f"champion {champion_brier:.4f} + tolerance {CALIBRATION_TOLERANCE} = {threshold:.4f}"
                ),
                details={
                    "challenger_brier": challenger_brier,
                    "champion_brier": champion_brier,
                    "tolerance": CALIBRATION_TOLERANCE,
                },
            )

        # No champion: absolute threshold (Req 13.5)
        if challenger_brier < BRIER_SCORE_THRESHOLD_WITHOUT_CHAMPION:
            return GateEvaluation(
                gate_name=gate_name,
                result=GateResult.PASS,
                reason=(
                    f"Challenger Brier {challenger_brier:.4f} < "
                    f"{BRIER_SCORE_THRESHOLD_WITHOUT_CHAMPION} (no champion exists)"
                ),
                details={"challenger_brier": challenger_brier},
            )
        return GateEvaluation(
            gate_name=gate_name,
            result=GateResult.FAIL,
            reason=(
                f"Challenger Brier {challenger_brier:.4f} >= "
                f"{BRIER_SCORE_THRESHOLD_WITHOUT_CHAMPION} (no champion exists)"
            ),
            details={"challenger_brier": challenger_brier},
        )

    def _evaluate_execution_gate(
        self,
        challenger: dict[str, Any],
        champion: dict[str, Any] | None,
    ) -> GateEvaluation:
        """
        EXECUTION gate — net return must be positive; if a champion exists,
        turnover increase must not exceed 20% in relative terms (Req 13.6).
        """
        gate_name = "EXECUTION"
        net_return = float(challenger.get("net_return_validation", 0.0))

        # Net return must be strictly positive (Req 13.6)
        if net_return <= 0.0:
            return GateEvaluation(
                gate_name=gate_name,
                result=GateResult.FAIL,
                reason=f"Net return {net_return:.4f} is not positive on the held-out validation partition",
                details={"net_return": net_return},
            )

        # With champion: check turnover increase (Req 13.6)
        if champion is not None:
            challenger_turnover = float(challenger.get("turnover_validation", 0.0))
            champion_turnover = float(champion.get("turnover_validation", 0.0))

            if champion_turnover > 0.0:
                max_allowed_turnover = champion_turnover * (1.0 + TURNOVER_MAX_RELATIVE_INCREASE)
                if challenger_turnover > max_allowed_turnover:
                    return GateEvaluation(
                        gate_name=gate_name,
                        result=GateResult.FAIL,
                        reason=(
                            f"Challenger turnover {challenger_turnover:.4f} exceeds "
                            f"champion {champion_turnover:.4f} by more than "
                            f"{TURNOVER_MAX_RELATIVE_INCREASE * 100:.0f}% "
                            f"(max allowed: {max_allowed_turnover:.4f})"
                        ),
                        details={
                            "challenger_turnover": challenger_turnover,
                            "champion_turnover": champion_turnover,
                            "max_allowed_turnover": max_allowed_turnover,
                        },
                    )

        return GateEvaluation(
            gate_name=gate_name,
            result=GateResult.PASS,
            reason=f"Execution gate passed — net return {net_return:.4f} is positive",
            details={"net_return": net_return},
        )

    def _evaluate_risk_gate(
        self,
        challenger: dict[str, Any],
        champion: dict[str, Any] | None,
    ) -> GateEvaluation:
        """
        RISK gate — max drawdown must not exceed champion's by more than 2
        percentage points; or must be <= 20% when no champion exists (Req 13.7).
        """
        gate_name = "RISK"
        challenger_dd = float(challenger.get("max_drawdown", 0.0))

        if champion is not None:
            champion_dd = float(champion.get("max_drawdown", 0.0))
            threshold = champion_dd + RISK_TOLERANCE_PP

            if challenger_dd <= threshold:
                return GateEvaluation(
                    gate_name=gate_name,
                    result=GateResult.PASS,
                    reason=(
                        f"Challenger max drawdown {challenger_dd:.4f} <= "
                        f"champion {champion_dd:.4f} + tolerance {RISK_TOLERANCE_PP} = {threshold:.4f}"
                    ),
                    details={
                        "challenger_dd": challenger_dd,
                        "champion_dd": champion_dd,
                        "tolerance_pp": RISK_TOLERANCE_PP,
                    },
                )
            return GateEvaluation(
                gate_name=gate_name,
                result=GateResult.FAIL,
                reason=(
                    f"Challenger max drawdown {challenger_dd:.4f} exceeds "
                    f"champion {champion_dd:.4f} + tolerance {RISK_TOLERANCE_PP} = {threshold:.4f}"
                ),
                details={
                    "challenger_dd": challenger_dd,
                    "champion_dd": champion_dd,
                    "tolerance_pp": RISK_TOLERANCE_PP,
                },
            )

        # No champion: absolute threshold of 20% (Req 13.7)
        if challenger_dd <= MAX_DRAWDOWN_WITHOUT_CHAMPION:
            return GateEvaluation(
                gate_name=gate_name,
                result=GateResult.PASS,
                reason=(
                    f"Challenger max drawdown {challenger_dd:.4f} <= "
                    f"{MAX_DRAWDOWN_WITHOUT_CHAMPION} (no champion exists)"
                ),
                details={"challenger_dd": challenger_dd},
            )
        return GateEvaluation(
            gate_name=gate_name,
            result=GateResult.FAIL,
            reason=(
                f"Challenger max drawdown {challenger_dd:.4f} > "
                f"{MAX_DRAWDOWN_WITHOUT_CHAMPION} (no champion exists)"
            ),
            details={"challenger_dd": challenger_dd},
        )

    def _evaluate_stability_gate(self, challenger: dict[str, Any]) -> GateEvaluation:
        """
        STABILITY gate — IC decay status and feature drift severity must not
        exceed acceptable thresholds (Req 13.8).
        """
        gate_name = "STABILITY"
        ic_decay = str(challenger.get("ic_decay_status", "OK"))
        drift_severity = str(challenger.get("feature_drift_severity", "LOW"))

        if ic_decay in _FAILING_IC_DECAY_STATUSES:
            return GateEvaluation(
                gate_name=gate_name,
                result=GateResult.FAIL,
                reason=f"IC decay status is {ic_decay!r} — model stability is insufficient for promotion",
                details={
                    "ic_decay_status": ic_decay,
                    "feature_drift_severity": drift_severity,
                },
            )

        if drift_severity in _FAILING_DRIFT_SEVERITIES:
            return GateEvaluation(
                gate_name=gate_name,
                result=GateResult.FAIL,
                reason=f"Feature drift severity is {drift_severity!r} — model stability is insufficient for promotion",
                details={
                    "ic_decay_status": ic_decay,
                    "feature_drift_severity": drift_severity,
                },
            )

        return GateEvaluation(
            gate_name=gate_name,
            result=GateResult.PASS,
            reason="Stability checks passed",
            details={
                "ic_decay_status": ic_decay,
                "feature_drift_severity": drift_severity,
            },
        )

    # ── Outcome assembly ───────────────────────────────────────────────────────

    def _determine_outcome(
        self,
        gates: list[GateEvaluation],
    ) -> tuple[PromotionOutcome, list[str]]:
        """
        Derive the final PromotionOutcome from the gate evaluations.

        Priority: FAIL > INSUFFICIENT_EVIDENCE > all PASS.

        Returns:
            A tuple of (outcome, blocked_gate_names).
            ``blocked_gate_names`` is only populated when the outcome is BLOCKED.
        """
        blocked_gates = [
            g.gate_name for g in gates if g.result == GateResult.INSUFFICIENT_EVIDENCE
        ]
        failed_gates = [g.gate_name for g in gates if g.result == GateResult.FAIL]

        if failed_gates:
            # FAIL takes priority over INSUFFICIENT_EVIDENCE (Req 13.10 by implication)
            logger.info(
                "promotion_gates_failed",
                failed_gates=failed_gates,
                blocked_gates=blocked_gates,
            )
            return PromotionOutcome.REJECTED, blocked_gates

        if blocked_gates:
            logger.info("promotion_gates_blocked", blocked_gates=blocked_gates)
            return PromotionOutcome.BLOCKED, blocked_gates

        # All gates passed — requires human approval (Req 13.9)
        logger.info("promotion_all_gates_passed")
        return PromotionOutcome.PROMOTE, []
