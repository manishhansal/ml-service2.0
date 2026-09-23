"""
MetaDecisionEngine — final Go/No-Go trading decision engine.

Combines outputs from all base models (regime, ranker, strategy, risk,
price forecaster, IV classifier, RL agent) via:
1. Calibration (CalibrationLayer)
2. Ensemble weighting (EnsembleWeighter)
3. Abstention check (AbstentionPolicy)
4. LLM news reasoning (LLMNewsReasoner — optional)
5. ConfidenceDecomposition (ConfidenceDecomposer)
6. XAI explainability assembly (MetaOutputExplainability)

Key properties (from TDD tests):
- Absorbing identity: all UNAVAILABLE → NO_TRADE + UNAVAILABLE provenance
- Idempotence: same inputs → same MetaOutput (no stochastic state mutations)
- Abstention invariant: agreement_ratio < 0.5 → NO_TRADE/WAIT + abstention=True
- confidence + uncertainty <= 1.0

agreement_ratio is computed as:
    max(n_buy, n_sell) / n_available
where WAIT (direction=0) outputs are treated as non-directional votes and do
NOT contribute to the plurality count.  This ensures that a split of
3 BUY + 4 WAIT yields agreement_ratio = 3/7 ≈ 0.43 (< 0.5 → abstain) rather
than incorrectly counting WAIT as the plurality.

Requirements: Req 10.1 through Req 10.12
"""
from __future__ import annotations

from typing import Any

from src.logging_config import get_logger
from src.meta.abstention import AbstentionContext, AbstentionPolicy
from src.meta.calibration import CalibrationLayer, ConfidenceDecomposer
from src.meta.ensemble import EnsembleWeighter
from src.schemas.base import PredictionProvenance
from src.schemas.meta import (
    ConfidenceDecomposition,
    MetaOutput,
    MetaOutputExplainability,
    NewsSignal,
)

logger = get_logger(__name__)

# If news sentiment strongly conflicts with the ensemble direction at this
# confidence level, the engine overrides to WAIT instead of BUY/SELL.
NEWS_CONFLICT_OVERRIDE_THRESHOLD = 0.7


class MetaDecisionEngine:
    """
    Combines all base model outputs into a final trading decision.

    The engine is designed to be **stateless with respect to single decide()
    calls** — it mutates no shared state during inference, ensuring idempotence
    (Property 13).  The injected sub-components (CalibrationLayer,
    EnsembleWeighter, etc.) may hold fitted state, but they are not mutated
    during ``decide()``.

    Usage::

        engine = MetaDecisionEngine()
        result = engine.decide(
            model_outputs=[...],
            symbol="NIFTY",
            regime="bull",
        )
        # result.action in {"BUY", "SELL", "WAIT", "NO_TRADE"}
    """

    def __init__(
        self,
        calibration: CalibrationLayer | None = None,
        weighter: EnsembleWeighter | None = None,
        abstention_policy: AbstentionPolicy | None = None,
        decomposer: ConfidenceDecomposer | None = None,
        llm_reasoner: Any = None,
    ) -> None:
        self._calibration = calibration or CalibrationLayer()
        self._weighter = weighter or EnsembleWeighter()
        self._abstention = abstention_policy or AbstentionPolicy()
        self._decomposer = decomposer or ConfidenceDecomposer()
        self._llm_reasoner = llm_reasoner  # optional — LLMNewsReasoner or compatible

    # ── Public API ────────────────────────────────────────────────────────────

    def decide(
        self,
        model_outputs: list[dict[str, Any]],
        symbol: str,
        regime: str = "sideways",
        news_context: dict[str, Any] | None = None,
        risk_context: dict[str, Any] | None = None,
    ) -> MetaOutput:
        """
        Produce a final MetaOutput from all base model outputs.

        Algorithm
        ---------
        1. Separate available vs UNAVAILABLE model outputs.
        2. Apply absorbing identity (all UNAVAILABLE → NO_TRADE).
        3. Apply fewer-than-3 quorum guard (< 3 available → NO_TRADE).
        4. Calibrate raw scores via CalibrationLayer.
        5. Compute IC-proportional ensemble weights via EnsembleWeighter.
        6. Compute directional agreement_ratio (BUY/SELL votes only).
        7. Compute ensemble score and mean confidence.
        8. Evaluate AbstentionPolicy on five conditions.
        9. Apply optional news conflict override.
        10. Determine final action.
        11. Compute confidence + uncertainty (guarantee: sum ≤ 1.0).
        12. Determine weakest-link provenance.
        13. Assemble XAI explainability block.
        14. Return frozen MetaOutput.

        Args:
            model_outputs: List of dicts, each with keys:
                model_id (str), action (str), confidence (float),
                direction (int: -1|0|1), provenance (str).
            symbol:        Instrument symbol (e.g. "NIFTY").
            regime:        Current market regime string for IC lookup.
            news_context:  Optional SentinelPulse news dict.
            risk_context:  Optional dict with ``prob_stop_hit`` key.

        Returns:
            A frozen :class:`MetaOutput` instance.
        """
        # ── Step 1: Partition outputs ─────────────────────────────────────────
        unavailable_value = PredictionProvenance.UNAVAILABLE.value
        available_outputs = [
            o for o in model_outputs
            if o.get("provenance") != unavailable_value
        ]
        n_available = len(available_outputs)

        # ── Step 2: Absorbing identity — all models UNAVAILABLE ───────────────
        if n_available == 0:
            return MetaOutput(
                action="NO_TRADE",
                confidence=0.0,
                uncertainty=1.0,
                agreement=0.0,
                agreement_ratio=0.0,
                provenance=PredictionProvenance.UNAVAILABLE,
                reason_codes=["ALL_MODELS_UNAVAILABLE"],
                abstention=True,
                symbol=symbol,
            )

        # ── Step 3: Quorum guard (Req 10.7 — INSUFFICIENT_MODELS) ────────────
        if n_available < 3:
            return MetaOutput(
                action="NO_TRADE",
                confidence=0.0,
                uncertainty=1.0,
                agreement=0.0,
                agreement_ratio=0.0,
                provenance=self._weakest_provenance(available_outputs),
                reason_codes=["INSUFFICIENT_MODELS"],
                abstention=True,
                symbol=symbol,
            )

        # ── Step 4: Calibrate raw scores ──────────────────────────────────────
        calibrated_confidences: list[float] = []
        for o in available_outputs:
            model_id = str(o.get("model_id", "unknown"))
            raw = float(o.get("confidence", 0.5))
            cal = self._calibration.calibrate(model_id, raw)
            calibrated_confidences.append(cal)

        # ── Step 5: Ensemble weights ──────────────────────────────────────────
        model_ids = [str(o.get("model_id", f"m_{i}")) for i, o in enumerate(available_outputs)]
        weights = self._weighter.compute_weights(
            model_ids=model_ids,
            regime=regime,
        )

        # ── Step 6: Directional agreement (BUY/SELL votes only) ───────────────
        #
        # WAIT (direction=0) is non-directional and does NOT contribute to the
        # plurality count.  This matches the test expectation that
        # 3 BUY + 4 WAIT yields agreement_ratio = 3/7 (< 0.5 → abstain).
        directions = [int(o.get("direction", 0)) for o in available_outputs]
        buy_count = sum(1 for d in directions if d > 0)
        sell_count = sum(1 for d in directions if d < 0)

        # agreement_ratio = fraction of ALL available models that voted for the
        # dominant directional signal (BUY or SELL).  WAIT votes reduce this
        # fraction naturally because n_available is the denominator.
        if buy_count == 0 and sell_count == 0:
            # All models said WAIT → no directional signal
            agreement_ratio = 0.0
            plurality_direction = 0
        elif buy_count >= sell_count:
            agreement_ratio = float(buy_count) / float(n_available)
            plurality_direction = 1 if buy_count > 0 else 0
        else:
            agreement_ratio = float(sell_count) / float(n_available)
            plurality_direction = -1

        # ── Step 7: Ensemble score and mean confidence ────────────────────────
        weighted_conf_sum = 0.0
        total_weight = 0.0
        for i, mid in enumerate(model_ids):
            w = weights.get(mid, 1.0 / n_available)
            weighted_conf_sum += calibrated_confidences[i] * w
            total_weight += w

        mean_confidence = (
            weighted_conf_sum / total_weight if total_weight > 0.0
            else sum(calibrated_confidences) / max(n_available, 1)
        )
        mean_confidence = max(0.0, min(1.0, mean_confidence))

        # Signed ensemble score: positive = bullish, negative = bearish
        ensemble_score = mean_confidence * float(plurality_direction) * agreement_ratio

        # ── Step 8: Abstention policy ─────────────────────────────────────────
        prob_stop_hit = 0.0
        if risk_context:
            prob_stop_hit = float(risk_context.get("prob_stop_hit", 0.0))

        abstention_ctx = AbstentionContext(
            agreement_ratio=agreement_ratio,
            data_quality=1.0,       # simplified: assumes good data quality
            mean_confidence=mean_confidence,
            prob_stop_hit=prob_stop_hit,
            n_available_models=n_available,
        )
        abstention_result = self._abstention.check(abstention_ctx)
        reason_codes: list[str] = list(abstention_result.reason_codes)

        # ── Step 9: Optional news sentiment override ──────────────────────────
        news_signal: NewsSignal | None = None
        if news_context is not None and self._llm_reasoner is not None:
            news_signal, news_codes = self._llm_reasoner.reason_sync(
                news_context, symbol
            )
            reason_codes.extend(news_codes)

        # ── Step 10: Determine final action ───────────────────────────────────
        if abstention_result.should_abstain:
            action = "NO_TRADE"
        else:
            # Check news conflict override
            if (
                news_signal is not None
                and float(news_signal.confidence) >= NEWS_CONFLICT_OVERRIDE_THRESHOLD
                and int(news_signal.direction) != 0
                and int(news_signal.direction) != plurality_direction
            ):
                # Strong conflicting news signal → override to WAIT
                action = "WAIT"
                reason_codes.append("NEWS_CONFLICT_OVERRIDE")
            elif plurality_direction > 0:
                action = "BUY"
            elif plurality_direction < 0:
                action = "SELL"
            else:
                action = "WAIT"

        # ── Step 11: Confidence and uncertainty ───────────────────────────────
        #
        # final_confidence = mean_confidence * agreement_ratio (0 when no agreement).
        # final_uncertainty = 1 − final_confidence  (guarantee: sum = 1.0 ≤ 1.0).
        final_confidence = float(mean_confidence * agreement_ratio)
        final_confidence = max(0.0, min(1.0, final_confidence))
        final_uncertainty = max(0.0, min(1.0, 1.0 - final_confidence))
        # Guard: floating-point can push sum slightly above 1.0
        if final_confidence + final_uncertainty > 1.0 + 1e-12:
            final_uncertainty = 1.0 - final_confidence

        # ── Step 12: Weakest-link provenance ──────────────────────────────────
        meta_provenance = self._weakest_provenance(available_outputs)

        # ── Step 13: XAI explainability assembly ──────────────────────────────
        explainability = self._build_explainability(
            available_outputs=available_outputs,
            weights=weights,
            model_ids=model_ids,
            plurality_direction=plurality_direction,
            news_signal=news_signal,
        )

        # ── Step 14: Confidence decomposition ─────────────────────────────────
        worst_ece = max(
            (self._calibration.get_ece(mid) for mid in model_ids),
            default=0.0,
        )
        decomp = self._decomposer.decompose(
            base_confidence=mean_confidence,
            agreement_ratio=agreement_ratio,
            data_quality=1.0,
            regime_confidence=mean_confidence,
            calibration_ece=worst_ece,
        )

        return MetaOutput(
            action=action,
            confidence=final_confidence,
            uncertainty=final_uncertainty,
            agreement=final_confidence,          # mirrors confidence (0–1 scale)
            agreement_ratio=agreement_ratio,
            ensemble_score=ensemble_score,
            reason_codes=reason_codes,
            contributing_models=model_ids,
            abstention=abstention_result.should_abstain,
            provenance=meta_provenance,
            news_sentiment_signal=news_signal,
            decomposition=decomp,
            explainability=explainability,
            symbol=symbol,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _weakest_provenance(
        available_outputs: list[dict[str, Any]],
    ) -> PredictionProvenance:
        """
        Return the weakest provenance across available model outputs.

        Ordering (strongest → weakest):
            TRAINED_MODEL > HEURISTIC > INSUFFICIENT_EVIDENCE > UNAVAILABLE

        "Weakest link" means the output provenance is constrained by the
        least-reliable contributing model.
        """
        _rank: dict[str, int] = {
            PredictionProvenance.TRAINED_MODEL.value: 4,
            PredictionProvenance.HEURISTIC.value: 3,
            PredictionProvenance.INSUFFICIENT_EVIDENCE.value: 2,
            PredictionProvenance.UNAVAILABLE.value: 1,
        }
        _from_rank: dict[int, PredictionProvenance] = {
            v: k  # type: ignore[misc]
            for k, v in {
                PredictionProvenance.TRAINED_MODEL: 4,
                PredictionProvenance.HEURISTIC: 3,
                PredictionProvenance.INSUFFICIENT_EVIDENCE: 2,
                PredictionProvenance.UNAVAILABLE: 1,
            }.items()
        }

        if not available_outputs:
            return PredictionProvenance.UNAVAILABLE

        min_rank = min(
            _rank.get(
                str(o.get("provenance", PredictionProvenance.UNAVAILABLE.value)),
                1,
            )
            for o in available_outputs
        )
        return _from_rank[min_rank]

    @staticmethod
    def _build_explainability(
        available_outputs: list[dict[str, Any]],
        weights: dict[str, float],
        model_ids: list[str],
        plurality_direction: int,
        news_signal: NewsSignal | None,
    ) -> MetaOutputExplainability:
        """
        Assemble the XAI explainability block for the MetaOutput (Task 53.2).

        ``top_features``       — up to 5 highest-weight contributing model IDs,
                                 sorted by weight descending.
        ``conflicting_models`` — model IDs whose direction contradicts the
                                 ensemble plurality direction.
        ``rationale``          — brief natural language summary (≤ 50 words).
        """
        # Top features: model IDs sorted by weight, capped at 5
        sorted_models = sorted(
            model_ids,
            key=lambda mid: weights.get(mid, 0.0),
            reverse=True,
        )
        top_features = sorted_models[:5]

        # Conflicting models: those whose direction != plurality
        conflicting_models: list[str] = []
        if plurality_direction != 0:
            for o, mid in zip(available_outputs, model_ids):
                direction = int(o.get("direction", 0))
                if direction != 0 and direction != plurality_direction:
                    conflicting_models.append(mid)

        # Rationale: concise natural-language summary
        direction_label = (
            "BUY" if plurality_direction > 0
            else ("SELL" if plurality_direction < 0 else "WAIT")
        )
        n_conflicting = len(conflicting_models)
        n_agreeing = sum(
            1 for o in available_outputs
            if int(o.get("direction", 0)) == plurality_direction and plurality_direction != 0
        )

        if news_signal is not None and news_signal.direction != 0:
            news_label = "bullish" if news_signal.direction > 0 else "bearish"
            rationale = (
                f"{n_agreeing}/{len(available_outputs)} models signal {direction_label}. "
                f"News sentiment: {news_label} ({news_signal.confidence:.0%}). "
                f"{n_conflicting} conflicting model(s)."
            )
        else:
            rationale = (
                f"{n_agreeing}/{len(available_outputs)} models signal {direction_label}. "
                f"{n_conflicting} conflicting model(s)."
            )

        # Truncate to ≤ 200 chars (schema constraint on rationale)
        rationale = rationale[:200]

        return MetaOutputExplainability(
            top_features=top_features,
            conflicting_models=conflicting_models,
            rationale=rationale,
        )
