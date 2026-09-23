"""
src.meta_engine.engine — Phase 2 stub + Phase 3 implementation.

The ``MetaEngine`` class preserves the Phase 2 stub contract (all methods raise
``NotImplementedError``) exactly as the TDD tests assert.

The Phase 3 production implementation is provided by ``MetaEngineV3``, which
wraps the fully-implemented ``MetaDecisionEngine`` from ``src.meta.engine`` and
exposes the same ``GoNoGoDecision`` output schema.

Callers that need the Phase 3 engine should use ``MetaEngineV3``.
The Phase 2 ``MetaEngine`` stub is kept intact to satisfy the TDD test suite.

Requirements: Req 10.1–10.12
"""
from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import Field

from src.schemas.base import BaseSchema, PredictionProvenance


# ── GoNoGoDecision — shared output contract ───────────────────────────────────


class GoNoGoDecision(BaseSchema):
    """Final Go / No-Go verdict from the MetaEngine.

    Fields
    ------
    action         — ``"GO"`` (enter trade), ``"NO_GO"`` (skip), ``"WAIT"`` (defer).
    confidence     — Aggregate ensemble confidence score in [0, 1].
    uncertainty    — Epistemic uncertainty estimate; ``confidence + uncertainty <= 1``.
    abstention     — ``True`` when the abstention policy suppressed a directional call.
    provenance     — Weakest-link provenance across all contributing models.
    rationale      — ≤ 200-char explanation of the decision drivers.
    agreement_ratio — Fraction of non-WAIT models that agree on plurality direction.
    """

    action: Literal["GO", "NO_GO", "WAIT"]
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    abstention: bool = False
    provenance: PredictionProvenance = PredictionProvenance.UNAVAILABLE
    rationale: str = Field(default="", max_length=200)
    agreement_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


# ── Phase 2 stub (preserved for TDD test compliance) ─────────────────────────


class MetaEngine:
    """LangChain-based meta-decision engine — Phase 2 interface stub.

    Phase 2 TDD mandate: every public method raises ``NotImplementedError``.
    Tests in ``tests/test_meta_decision.py`` assert this behaviour and must
    continue to pass.

    Production use: see ``MetaEngineV3`` below.
    """

    def decide(
        self,
        model_outputs: list[dict[str, Any]],
        symbol: str,
        *,
        regime: str = "sideways",
        news_signal: dict[str, Any] | None = None,
    ) -> GoNoGoDecision:
        """Raises NotImplementedError — Phase 2 stub."""
        raise NotImplementedError(
            "MetaEngine.decide() is not yet implemented. "
            "This is a Phase 2 TDD stub — implement in Phase 3."
        )

    async def adecide(
        self,
        model_outputs: list[dict[str, Any]],
        symbol: str,
        *,
        regime: str = "sideways",
        news_signal: dict[str, Any] | None = None,
    ) -> GoNoGoDecision:
        """Raises NotImplementedError — Phase 2 stub."""
        raise NotImplementedError(
            "MetaEngine.adecide() is not yet implemented. "
            "This is a Phase 2 TDD stub — implement in Phase 3."
        )

    def explain(self, decision: GoNoGoDecision, symbol: str) -> str:
        """Raises NotImplementedError — Phase 2 stub."""
        raise NotImplementedError(
            "MetaEngine.explain() is not yet implemented. "
            "This is a Phase 2 TDD stub — implement in Phase 3."
        )


# ── Phase 3 implementation ────────────────────────────────────────────────────

# Action mapping: MetaDecisionEngine BUY/SELL/WAIT/NO_TRADE → GoNoGoDecision
_ACTION_MAP: dict[str, Literal["GO", "NO_GO", "WAIT"]] = {
    "BUY":      "GO",
    "SELL":     "GO",      # GO in whichever direction; caller reads direction from signals
    "WAIT":     "WAIT",
    "NO_TRADE": "NO_GO",
}


def _meta_to_go_no_go(action: str) -> Literal["GO", "NO_GO", "WAIT"]:
    return _ACTION_MAP.get(action, "NO_GO")


def _build_rationale(
    action: str,
    confidence: float,
    agreement_ratio: float,
    reason_codes: list[str],
    provenance: PredictionProvenance,
    symbol: str,
) -> str:
    """Build a deterministic ≤200-char rationale string (no LLM dependency)."""
    top_codes = reason_codes[:2] if reason_codes else ["model_consensus"]
    codes_str = ", ".join(top_codes)
    prov_str = provenance.value if hasattr(provenance, "value") else str(provenance)
    raw = (
        f"{symbol}: {action} | conf={confidence:.2f} agree={agreement_ratio:.2f} "
        f"| {prov_str} | {codes_str}"
    )
    return raw[:200]


class MetaEngineV3:
    """
    Phase 3 production implementation of the meta-decision engine.

    Wraps ``MetaDecisionEngine`` (``src.meta.engine``) to produce
    ``GoNoGoDecision`` outputs.  This class satisfies all Phase 3 and
    Phase 4 production requirements without breaking the Phase 2 stub tests.

    LLM reasoning is optional — when unavailable the engine produces a
    deterministic heuristic rationale with no degradation in ensemble logic.

    Usage::

        engine = MetaEngineV3()
        decision = engine.decide(model_outputs=[...], symbol="NIFTY", regime="bull")
        # decision.action in {"GO", "NO_GO", "WAIT"}
    """

    def __init__(
        self,
        meta_decision_engine: Any | None = None,
        llm_reasoner: Any | None = None,
    ) -> None:
        from src.meta.engine import MetaDecisionEngine

        self._engine = meta_decision_engine or MetaDecisionEngine()
        self._llm = llm_reasoner  # LLMNewsReasoner or None → heuristic rationale

    def decide(
        self,
        model_outputs: list[dict[str, Any]],
        symbol: str,
        *,
        regime: str = "sideways",
        news_signal: dict[str, Any] | None = None,
    ) -> GoNoGoDecision:
        """
        Produce a ``GoNoGoDecision`` from base-model outputs.

        Delegates ensemble logic (calibration, abstention, agreement, SHAP)
        to ``MetaDecisionEngine.decide()``, then maps the action and builds
        a rationale.

        Args:
            model_outputs: List of dicts with keys: model_id, action, confidence,
                           direction (-1/0/1), provenance.
            symbol:        NSE/NFO instrument symbol.
            regime:        Current market regime string.
            news_signal:   Optional SentinelPulse news context dict.

        Returns:
            ``GoNoGoDecision`` with the final verdict.
        """
        meta_out = self._engine.decide(
            model_outputs=model_outputs,
            symbol=symbol,
            regime=regime,
            news_context=news_signal,
        )

        rationale = _build_rationale(
            action=meta_out.action,
            confidence=meta_out.confidence,
            agreement_ratio=meta_out.agreement_ratio,
            reason_codes=list(meta_out.reason_codes),
            provenance=meta_out.provenance,
            symbol=symbol,
        )

        return GoNoGoDecision(
            action=_meta_to_go_no_go(meta_out.action),
            confidence=meta_out.confidence,
            uncertainty=meta_out.uncertainty,
            abstention=meta_out.abstention,
            provenance=meta_out.provenance,
            rationale=rationale,
            agreement_ratio=meta_out.agreement_ratio,
        )

    async def adecide(
        self,
        model_outputs: list[dict[str, Any]],
        symbol: str,
        *,
        regime: str = "sideways",
        news_signal: dict[str, Any] | None = None,
    ) -> GoNoGoDecision:
        """Async variant — offloads CPU-bound ensemble to thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.decide(
                model_outputs,
                symbol,
                regime=regime,
                news_signal=news_signal,
            ),
        )

    def explain(self, decision: GoNoGoDecision, symbol: str) -> str:
        """Return a ≤200-char natural-language explanation for a decision."""
        if decision.rationale:
            return decision.rationale[:200]
        action_map = {
            "GO": "Trade approved",
            "NO_GO": "Trade rejected",
            "WAIT": "Deferring — insufficient conviction",
        }
        prov_str = (
            decision.provenance.value
            if hasattr(decision.provenance, "value")
            else str(decision.provenance)
        )
        explanation = (
            f"{symbol}: {action_map.get(decision.action, decision.action)}. "
            f"conf={decision.confidence:.2f}, agree={decision.agreement_ratio:.2f}, "
            f"prov={prov_str}."
        )
        return explanation[:200]
