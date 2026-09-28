"""
src.training.self_learning — controlled outcome-driven adaptation loop (Phase R).

Self-learning = outcome-driven CONTROLLED adaptation, NOT uncontrolled model
mutation. The champion is NEVER updated directly. The loop is:

    feedback (FeedbackStore)
        -> retraining decision (enough new resolved outcomes? drift? decay?)
        -> build fresh dataset incorporating outcomes
        -> train challenger (TrainingOrchestrator)
        -> validate (walk-forward + CPCV + calibration)
        -> shadow (ChampionChallengerManager)
        -> six-gate promotion + approval
        -> champion

This module owns the RETRAINING DECISION and wiring; it delegates training to
the orchestrator and promotion to the lifecycle manager. It enforces guards:
minimum new samples, maximum update frequency, and never promoting without the
full gate pass.

Requirements: Phase R, Phase 49, Phase 68, 12_ONLINE_LEARNING_SPEC.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.data.feedback import FeedbackStore
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RetrainingDecision:
    """Whether to trigger a challenger retrain and why."""

    should_retrain: bool
    reason: str
    n_new_resolved: int
    triggers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class SelfLearningLoop:
    """
    Decides when to spawn a challenger from accumulated feedback, and drives the
    controlled promotion path. NEVER mutates the champion directly.

    Usage::

        loop = SelfLearningLoop(feedback_store, min_new_samples=30)
        decision = loop.should_retrain(
            drift_action="MONITOR", performance_degraded=False,
        )
        if decision.should_retrain:
            loop.mark_retrain_triggered()
    """

    def __init__(
        self,
        feedback_store: FeedbackStore,
        min_new_samples: int = 30,
        min_hours_between_retrains: float = 24.0,
    ) -> None:
        self._feedback = feedback_store
        self.min_new_samples = min_new_samples
        self.min_hours_between_retrains = min_hours_between_retrains
        self._last_retrain_at: datetime | None = None
        self._consumed_count: int = 0

    def should_retrain(
        self,
        drift_action: str = "MONITOR",
        performance_degraded: bool = False,
    ) -> RetrainingDecision:
        """
        Decide whether to trigger a challenger retrain.

        Triggers (any of):
          - enough NEW resolved outcomes since last retrain (>= min_new_samples)
          - drift action is TRAIN_CHALLENGER
          - performance degradation detected
        Guards:
          - respect min_hours_between_retrains (rate limit)
          - CRITICAL drift (BLOCK) does not itself trigger a retrain — it blocks
            the affected model; a challenger is spawned via TRAIN_CHALLENGER.
        """
        resolved = self._feedback.resolved_records()
        n_new = len(resolved) - self._consumed_count
        triggers: list[str] = []

        if n_new >= self.min_new_samples:
            triggers.append("SUFFICIENT_NEW_OUTCOMES")
        if drift_action == "TRAIN_CHALLENGER":
            triggers.append("DRIFT_HIGH")
        if performance_degraded:
            triggers.append("PERFORMANCE_DEGRADATION")

        if not triggers:
            return RetrainingDecision(False, "NO_TRIGGER", n_new, [])

        # Rate limit.
        if self._last_retrain_at is not None:
            elapsed_h = (datetime.now(tz=UTC) - self._last_retrain_at).total_seconds() / 3600
            if elapsed_h < self.min_hours_between_retrains:
                return RetrainingDecision(
                    False, "RATE_LIMITED", n_new, triggers,
                )

        return RetrainingDecision(True, "RETRAIN_TRIGGERED", n_new, triggers)

    def mark_retrain_triggered(self) -> None:
        """Record that a retrain was launched; resets the new-sample counter."""
        self._last_retrain_at = datetime.now(tz=UTC)
        self._consumed_count = len(self._feedback.resolved_records())
        logger.info(
            "self_learning_retrain_triggered",
            consumed=self._consumed_count,
            at=self._last_retrain_at.isoformat(),
        )

    def run_cycle(
        self,
        orchestrator: Any,
        lifecycle_manager: Any,
        model_name: str,
        dataset_id: str,
        drift_action: str = "MONITOR",
        performance_degraded: bool = False,
        candidate_names: list[str] | None = None,
        champion_ic: float | None = None,
    ) -> dict[str, Any]:
        """
        Execute one controlled self-learning cycle end-to-end.

        FIX NEW-P1-004: Before promoting a challenger to SHADOW, verify it beats
        the current champion's IC by at least the orchestrator's parsimony_margin.
        A challenger that barely clears the minimum gates but does not beat the
        current production champion must NOT be auto-promoted to SHADOW.

        Args:
            orchestrator:        TrainingOrchestrator instance.
            lifecycle_manager:   Champion/challenger lifecycle manager.
            model_name:          Model family name.
            dataset_id:          Frozen dataset ID to train on.
            drift_action:        Drift recommendation ('MONITOR', 'TRAIN_CHALLENGER').
            performance_degraded: Whether performance degradation was detected.
            candidate_names:     Estimator names to evaluate.
            champion_ic:         Current champion's OOS IC (for comparison gate).
                                 If None, the comparison gate is skipped.

        Returns a dict describing what happened.
        The champion is only ever changed through the lifecycle manager's gated
        promotion — this method stops at registering the challenger + promoting
        it to SHADOW. Promotion to champion still requires the six-gate +
        approval-token step, which is intentionally NOT auto-approved here.
        """
        decision = self.should_retrain(drift_action, performance_degraded)
        result: dict[str, Any] = {"decision": decision.to_dict()}
        if not decision.should_retrain:
            return result

        self.mark_retrain_triggered()

        # Pass champion_ic as baseline so orchestrator enforces the comparison gate.
        # FIX NEW-P1-004: challenger must beat champion by parsimony_margin.
        report = orchestrator.train(
            model_name, dataset_id,
            candidate_names=candidate_names or ["logistic", "lightgbm"],
            register_champion=True,
            baseline_ic=champion_ic,  # NEW: comparison gate
        )
        result["training"] = report.to_dict()

        if report.passed_acceptance and report.champion_version:
            lifecycle_manager.register_challenger(model_name, report.champion_version)
            lifecycle_manager.promote_to_shadow(model_name)
            result["challenger_registered"] = report.champion_version
            result["stage"] = "SHADOW"
            logger.info(
                "self_learning_challenger_promoted_to_shadow",
                model_name=model_name,
                version=report.champion_version,
                challenger_ic=getattr(report, "champion_ic_mean", None),
                champion_ic=champion_ic,
            )
        else:
            result["stage"] = "REJECTED"
            result["rejection_reason"] = report.rejection_reason
            logger.info(
                "self_learning_challenger_rejected",
                model_name=model_name,
                reason=report.rejection_reason,
            )
        # NOTE: promotion SHADOW -> CHAMPION is intentionally NOT performed here.
        # It requires the six-gate promotion + human approval token.

        # ── Phil integration: update feature_weights.json after retrain ──────
        # When a challenger is accepted, automatically run a score threshold
        # sweep on the resolved feedback records and update feature_weights.json.
        # This is the "playbook self-edit after retro" pattern from Phil.
        if report.passed_acceptance:
            try:
                from src.analytics.feature_weight_manager import FeatureWeightManager
                from src.analytics.score_threshold_sweep import ScoreThresholdSweep

                resolved = self._feedback.resolved_records()
                if len(resolved) >= 20:
                    # Build scores list from feedback (direction-as-score proxy)
                    scores_proxy = [
                        {"symbol": r.symbol, "score": 0.75 if r.action == "BUY" else 0.25,
                         "direction": 1 if r.action == "BUY" else -1}
                        for r in resolved if r.action in ("BUY", "SELL")
                    ]
                    outcomes = [
                        {"symbol": r.symbol,
                         "net_pct": float(r.realized_return_net or 0) * 100,
                         "direction": 1 if r.action == "BUY" else -1}
                        for r in resolved if r.action in ("BUY", "SELL")
                        and r.realized_return_net is not None
                    ]
                    if scores_proxy and outcomes:
                        sweep = ScoreThresholdSweep()
                        sweep_report = sweep.run(
                            scores=scores_proxy,
                            resolved_outcomes=outcomes,
                            date=datetime.now(tz=UTC).strftime("%Y-%m-%d"),
                        )
                        if sweep_report.optimal_threshold > 0:
                            mgr = FeatureWeightManager()
                            mgr.update_from_sweep(
                                optimal_threshold=sweep_report.optimal_threshold,
                                win_rate=sweep_report.optimal_win_rate,
                                evidence=f"SelfLearningLoop after {len(resolved)} outcomes: {sweep_report.reasoning}",
                            )
                            result["feature_weights_updated"] = {
                                "threshold": sweep_report.optimal_threshold,
                                "win_rate":  sweep_report.optimal_win_rate,
                            }
                            logger.info(
                                "self_learning_feature_weights_updated",
                                threshold=sweep_report.optimal_threshold,
                                n_outcomes=len(resolved),
                            )
            except Exception as e:
                logger.warning("self_learning_feature_weight_update_failed", error=str(e))

        return result
