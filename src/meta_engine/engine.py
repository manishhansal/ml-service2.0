"""
src.meta_engine.engine — Phase 2 stub for the LangChain-based meta-decision engine.

This stub defines the public interface contract that the TDD tests in
``tests/test_meta_decision.py`` drive.  All methods raise ``NotImplementedError``
until Phase 3 implementation.

Interface contract
------------------
``MetaEngine.decide(model_outputs, symbol, *, regime, news_signal)``
  - Accepts a list of base-model output dicts and returns a ``GoNoGoDecision``.
  - Must be deterministic (same inputs → same output) — tested by Property 13.
  - Must return ``GoNoGoDecision(action="NO_TRADE")`` when all inputs have
    UNAVAILABLE provenance — Property 12 (absorbing identity).
  - Must set ``abstention=True`` when agreement_ratio < 0.5 — Property 14.

``GoNoGoDecision``
  - Pydantic model representing the final Go / No-Go verdict.
  - ``action``: Literal["GO", "NO_GO", "WAIT"]
  - ``confidence``: float in [0, 1]
  - ``rationale``: str (≤ 200 chars, LangChain-generated)
  - ``abstention``: bool
  - ``provenance``: PredictionProvenance

Phase 3 will replace the ``raise NotImplementedError`` bodies with the full
LangChain + MetaDecisionEngine integration.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from src.schemas.base import BaseSchema, PredictionProvenance


# ── Phase 2 output contract ───────────────────────────────────────────────────


class GoNoGoDecision(BaseSchema):
    """Final Go / No-Go verdict from the MetaEngine.

    This is the Phase-2 interface contract.  Phase 3 will enrich this
    with the full ``MetaOutput`` fields; for now, these five fields are
    the minimum needed by the TDD test suite.

    Fields
    ------
    action         — ``"GO"`` (enter trade), ``"NO_GO"`` (skip), ``"WAIT"`` (defer).
    confidence     — Aggregate ensemble confidence score in [0, 1].
    uncertainty    — Epistemic uncertainty estimate; ``confidence + uncertainty <= 1``.
    abstention     — ``True`` when the abstention policy suppressed a directional call.
    provenance     — Weakest-link provenance across all contributing models.
    rationale      — ≤ 200-char LangChain-generated natural-language explanation.
    agreement_ratio — Fraction of non-WAIT models that agree on the plurality direction.
    """

    action: Literal["GO", "NO_GO", "WAIT"]
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    abstention: bool = False
    provenance: PredictionProvenance = PredictionProvenance.UNAVAILABLE
    rationale: str = Field(default="", max_length=200)
    agreement_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


# ── Phase 2 stub ──────────────────────────────────────────────────────────────


class MetaEngine:
    """LangChain-based meta-decision engine — Phase 2 interface stub.

    Phase 2 TDD mandate: every public method raises ``NotImplementedError``.
    The test suite imports this class and asserts the expected behaviour via
    ``pytest.raises(NotImplementedError)`` until Phase 3 provides the real
    implementation.

    Usage (Phase 3)::

        engine = MetaEngine()
        decision = engine.decide(
            model_outputs=[...],
            symbol="NIFTY",
            regime="bull",
        )
        # decision.action in {"GO", "NO_GO", "WAIT"}
    """

    def decide(
        self,
        model_outputs: list[dict[str, Any]],
        symbol: str,
        *,
        regime: str = "sideways",
        news_signal: dict[str, Any] | None = None,
    ) -> GoNoGoDecision:
        """Produce a Go/No-Go trading decision from a list of base-model outputs.

        Args:
            model_outputs: List of dicts, each containing keys:
                           ``model_id``, ``action``, ``confidence``,
                           ``direction`` (-1/0/1), ``provenance``.
            symbol:        NSE/NFO instrument symbol (e.g. ``"NIFTY"``).
            regime:        Current market regime string (MarketRegime value).
            news_signal:   Optional dict with keys ``direction``, ``confidence``,
                           ``rationale`` from the LLM news reasoner.

        Returns:
            ``GoNoGoDecision`` with the final trading verdict.

        Raises:
            NotImplementedError: Until Phase 3 implementation.
        """
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
        """Async variant of ``decide()`` for use inside FastAPI route handlers.

        Raises:
            NotImplementedError: Until Phase 3 implementation.
        """
        raise NotImplementedError(
            "MetaEngine.adecide() is not yet implemented. "
            "This is a Phase 2 TDD stub — implement in Phase 3."
        )

    def explain(self, decision: GoNoGoDecision, symbol: str) -> str:
        """Return a LangChain-generated natural-language explanation for a decision.

        Args:
            decision: The ``GoNoGoDecision`` to explain.
            symbol:   Instrument symbol for context.

        Returns:
            A string ≤ 200 characters summarising the key drivers.

        Raises:
            NotImplementedError: Until Phase 3 implementation.
        """
        raise NotImplementedError(
            "MetaEngine.explain() is not yet implemented. "
            "This is a Phase 2 TDD stub — implement in Phase 3."
        )
