"""
test_meta_decision_engine.py

TDD tests for the MetaDecisionEngine.
Written BEFORE MetaDecisionEngine implementation (red phase).

Properties:
  Property 12: absorbing identity — all UNAVAILABLE inputs → NO_TRADE
  Property 13: idempotence — same input always produces same output
  Property 14: abstention invariant — agreement_ratio < 0.5 → action in {WAIT, NO_TRADE}
               and abstention == True
  Invariant:   confidence + uncertainty <= 1.0

**Validates: Requirements 10.1, 10.4, 10.7, 10.12, 17.3, 18.4**
"""
from __future__ import annotations

import os
from typing import Any

import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as h_settings
from hypothesis import strategies as st

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from src.schemas.base import PredictionProvenance
from src.schemas.meta import MetaOutput


# ── Helpers ───────────────────────────────────────────────────────────────────


def _skip_if_not_implemented() -> Any:
    """Return MetaDecisionEngine class, or skip the test if not yet implemented."""
    try:
        from src.meta.engine import MetaDecisionEngine  # type: ignore[import]

        return MetaDecisionEngine
    except (ImportError, ModuleNotFoundError):
        pytest.skip("MetaDecisionEngine not yet implemented — TDD red phase")


def _unavailable_output(model_id: str) -> dict[str, Any]:
    """Construct a single model output with UNAVAILABLE provenance."""
    return {
        "model_id": model_id,
        "action": "WAIT",
        "confidence": 0.5,
        "direction": 0,
        "provenance": PredictionProvenance.UNAVAILABLE.value,
    }


def _all_unavailable_outputs(n: int = 7) -> list[dict[str, Any]]:
    """Build a list of *n* model outputs all with UNAVAILABLE provenance."""
    return [_unavailable_output(f"model_{i}") for i in range(n)]


def _trained_output(model_id: str, direction: int, confidence: float = 0.7) -> dict[str, Any]:
    """Construct a single model output with TRAINED_MODEL provenance."""
    action = "BUY" if direction > 0 else ("SELL" if direction < 0 else "WAIT")
    return {
        "model_id": model_id,
        "action": action,
        "confidence": confidence,
        "direction": direction,
        "provenance": PredictionProvenance.TRAINED_MODEL.value,
    }


def _outputs_with_agreement(agreement_fraction: float, n_total: int = 7) -> list[dict[str, Any]]:
    """
    Build model outputs where exactly ``round(n_total * agreement_fraction)``
    models agree on BUY and the rest emit SELL.  This gives an agreement_ratio
    of ``max(n_buy, n_sell) / n_total`` from the engine's perspective.
    """
    n_buy = round(n_total * agreement_fraction)
    outputs: list[dict[str, Any]] = []
    for i in range(n_total):
        direction = 1 if i < n_buy else -1
        outputs.append(_trained_output(f"model_{i}", direction))
    return outputs


# ── Hypothesis strategies ─────────────────────────────────────────────────────

# Strategy for a single UNAVAILABLE model output dict.
_st_unavailable_output = st.fixed_dictionaries(
    {
        "model_id": st.text(min_size=1, max_size=32),
        "action": st.just("WAIT"),
        "confidence": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        "direction": st.just(0),
        "provenance": st.just(PredictionProvenance.UNAVAILABLE.value),
    }
)

# Strategy for a valid model output with TRAINED_MODEL provenance and a fixed direction.
_st_trained_output = st.fixed_dictionaries(
    {
        "model_id": st.text(min_size=1, max_size=32),
        "action": st.sampled_from(["BUY", "SELL", "WAIT"]),
        "confidence": st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
        "direction": st.integers(min_value=-1, max_value=1),
        "provenance": st.just(PredictionProvenance.TRAINED_MODEL.value),
    }
)

# A list of 7 UNAVAILABLE outputs — used for Property 12.
_st_all_unavailable = st.lists(
    _st_unavailable_output,
    min_size=7,
    max_size=7,
)

# A list of 7 TRAINED_MODEL outputs — used for Property 13.
_st_all_trained = st.lists(
    _st_trained_output,
    min_size=7,
    max_size=7,
)


# ── Property 12: Absorbing Identity ──────────────────────────────────────────


class TestAbsorbingIdentity:
    """
    Property 12: meta_decide({all models UNAVAILABLE}) == NO_TRADE

    **Validates: Requirements 10.12**
    """

    def test_all_unavailable_returns_no_trade_unit(self) -> None:
        """Unit check: 7 UNAVAILABLE models → action == NO_TRADE."""
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        result = engine.decide(model_outputs=_all_unavailable_outputs(), symbol="NIFTY")

        assert result.action == "NO_TRADE", (
            f"Expected NO_TRADE when all models UNAVAILABLE, got {result.action}"
        )

    def test_all_unavailable_sets_unavailable_provenance(self) -> None:
        """When all models are UNAVAILABLE provenance must propagate to MetaOutput."""
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        result = engine.decide(model_outputs=_all_unavailable_outputs(), symbol="NIFTY")

        assert result.provenance == PredictionProvenance.UNAVAILABLE, (
            f"Expected UNAVAILABLE provenance, got {result.provenance}"
        )

    def test_fewer_than_3_available_returns_no_trade(self) -> None:
        """AbstentionPolicy: fewer than 3 non-UNAVAILABLE models → NO_TRADE (Req 10.7)."""
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        # 5 UNAVAILABLE + 2 TRAINED_MODEL (< 3 available)
        outputs: list[dict[str, Any]] = [_unavailable_output(f"m_{i}") for i in range(5)] + [
            _trained_output("m_5", direction=1),
            _trained_output("m_6", direction=1),
        ]

        result = engine.decide(model_outputs=outputs, symbol="NIFTY")

        assert result.action == "NO_TRADE", (
            f"Expected NO_TRADE with < 3 available models, got {result.action}"
        )

    @given(model_outputs=_st_all_unavailable)
    @h_settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_absorbing_identity_property(self, model_outputs: list[dict[str, Any]]) -> None:
        """
        Property 12 (Hypothesis): for ANY list of 7 UNAVAILABLE model outputs,
        action MUST be NO_TRADE and provenance MUST be UNAVAILABLE.

        **Validates: Requirements 10.12**
        """
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        result = engine.decide(model_outputs=model_outputs, symbol="NIFTY")

        assert result.action == "NO_TRADE", (
            f"Absorbing identity violated: all UNAVAILABLE but action={result.action}"
        )
        assert result.provenance == PredictionProvenance.UNAVAILABLE, (
            f"Absorbing identity violated: provenance={result.provenance}"
        )


# ── Property 13: Idempotence ──────────────────────────────────────────────────


class TestIdempotence:
    """
    Property 13: meta_decide(x) == meta_decide(x) for all inputs.

    **Validates: Requirements 17.3, 18.4**
    """

    def test_same_inputs_produce_same_action(self) -> None:
        """Two successive identical calls must yield the same action."""
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        outputs = _outputs_with_agreement(0.86)  # 6/7 agree BUY

        r1 = engine.decide(model_outputs=outputs, symbol="NIFTY")
        r2 = engine.decide(model_outputs=outputs, symbol="NIFTY")

        assert r1.action == r2.action, (
            f"Idempotence violated on action: first={r1.action}, second={r2.action}"
        )

    def test_same_inputs_produce_same_confidence(self) -> None:
        """Two successive identical calls must yield the same confidence (within 1e-9)."""
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        outputs = _outputs_with_agreement(0.86)

        r1 = engine.decide(model_outputs=outputs, symbol="NIFTY")
        r2 = engine.decide(model_outputs=outputs, symbol="NIFTY")

        assert abs(r1.confidence - r2.confidence) < 1e-9, (
            f"Confidence not idempotent: {r1.confidence} vs {r2.confidence}"
        )

    def test_same_inputs_produce_same_agreement_ratio(self) -> None:
        """Two successive identical calls must yield the same agreement_ratio."""
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        outputs = _outputs_with_agreement(0.86)

        r1 = engine.decide(model_outputs=outputs, symbol="NIFTY")
        r2 = engine.decide(model_outputs=outputs, symbol="NIFTY")

        assert r1.agreement_ratio == r2.agreement_ratio, (
            "agreement_ratio not idempotent: "
            f"{r1.agreement_ratio} vs {r2.agreement_ratio}"
        )

    def test_unavailable_inputs_idempotent(self) -> None:
        """Idempotence must hold for all-UNAVAILABLE inputs too."""
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        outputs = _all_unavailable_outputs()

        r1 = engine.decide(model_outputs=outputs, symbol="NIFTY")
        r2 = engine.decide(model_outputs=outputs, symbol="NIFTY")

        assert r1.action == r2.action
        assert abs(r1.confidence - r2.confidence) < 1e-9

    @given(model_outputs=_st_all_trained)
    @h_settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_idempotence_property(self, model_outputs: list[dict[str, Any]]) -> None:
        """
        Property 13 (Hypothesis): for any list of 7 TRAINED_MODEL outputs,
        calling decide() twice with the same inputs must produce the same action
        and confidence.

        **Validates: Requirements 17.3, 18.4**
        """
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()

        r1 = engine.decide(model_outputs=model_outputs, symbol="NIFTY")
        r2 = engine.decide(model_outputs=model_outputs, symbol="NIFTY")

        assert r1.action == r2.action, (
            f"Idempotence violated: action={r1.action} vs {r2.action}"
        )
        assert abs(r1.confidence - r2.confidence) < 1e-9, (
            f"Confidence not idempotent: {r1.confidence} vs {r2.confidence}"
        )


# ── Property 14: Abstention Invariant ────────────────────────────────────────


class TestAbstentionInvariant:
    """
    Property 14: agreement_ratio < 0.5 → action ∈ {WAIT, NO_TRADE} AND abstention == True

    **Validates: Requirements 10.4, 10.7, 18.4**
    """

    def test_low_agreement_triggers_abstention_unit(self) -> None:
        """
        3 BUY vs 4 WAIT among 7 available models gives BUY agreement_ratio = 3/7 ≈ 0.43 < 0.5.
        Expect action ∈ {WAIT, NO_TRADE} and abstention == True.
        """
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        # 3 models say BUY (direction=1), 4 say WAIT (direction=0)
        outputs: list[dict[str, Any]] = [
            _trained_output(f"m_{i}", direction=1 if i < 3 else 0)
            for i in range(7)
        ]

        result = engine.decide(model_outputs=outputs, symbol="NIFTY")

        assert result.action in ("WAIT", "NO_TRADE"), (
            f"Expected WAIT or NO_TRADE for agreement_ratio < 0.5, got {result.action}"
        )
        assert result.abstention is True, (
            f"Expected abstention=True when agreement_ratio < 0.5, got {result.abstention}"
        )

    def test_equal_buy_sell_split_triggers_abstention(self) -> None:
        """
        Equal BUY/SELL split (3 vs 3, 1 WAIT) → plurality = 3/7 ≈ 0.43 < 0.5.
        Expect abstention.
        """
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        outputs: list[dict[str, Any]] = (
            [_trained_output(f"buy_{i}", direction=1) for i in range(3)]
            + [_trained_output(f"sell_{i}", direction=-1) for i in range(3)]
            + [_trained_output("neutral_0", direction=0)]
        )

        result = engine.decide(model_outputs=outputs, symbol="NIFTY")

        assert result.action in ("WAIT", "NO_TRADE"), (
            f"Equal BUY/SELL split should produce WAIT/NO_TRADE, got {result.action}"
        )
        assert result.abstention is True, (
            f"Expected abstention=True for equal BUY/SELL split"
        )

    @given(n_agreeing=st.integers(min_value=0, max_value=3))
    @h_settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_abstention_invariant_hypothesis(self, n_agreeing: int) -> None:
        """
        Property 14 (Hypothesis): when the plurality direction is held by fewer than
        half of the available models (n_agreeing ∈ [0, 3] out of 7), the engine must
        set action ∈ {WAIT, NO_TRADE} and abstention == True.

        **Validates: Requirements 10.4, 10.7, 18.4**
        """
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        n_total = 7
        # n_agreeing models vote BUY, the rest vote WAIT (neutral)
        outputs: list[dict[str, Any]] = [
            _trained_output(f"m_{i}", direction=1 if i < n_agreeing else 0)
            for i in range(n_total)
        ]

        result = engine.decide(model_outputs=outputs, symbol="NIFTY")

        # BUY agreement_ratio = n_agreeing / n_total; max 3/7 ≈ 0.43 < 0.5
        agreement = n_agreeing / n_total
        assert agreement < 0.5, "Test setup invariant broken"

        assert result.action in ("WAIT", "NO_TRADE"), (
            f"abstention invariant violated: agreement_ratio={agreement:.3f} < 0.5 "
            f"but action={result.action}"
        )
        assert result.abstention is True, (
            f"abstention flag not set: agreement_ratio={agreement:.3f} < 0.5 "
            f"but abstention={result.abstention}"
        )


# ── Confidence + Uncertainty Invariant ───────────────────────────────────────


class TestConfidenceUncertaintyInvariant:
    """confidence + uncertainty <= 1.0 for all valid MetaOutput instances."""

    def test_sum_at_most_one_for_high_agreement(self) -> None:
        """confidence + uncertainty must be ≤ 1.0 on a high-agreement scenario."""
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        result = engine.decide(
            model_outputs=_outputs_with_agreement(0.86), symbol="NIFTY"
        )

        total = result.confidence + result.uncertainty
        assert total <= 1.0 + 1e-9, (
            f"confidence ({result.confidence}) + uncertainty ({result.uncertainty}) = "
            f"{total:.6f} > 1.0"
        )

    def test_sum_at_most_one_for_all_unavailable(self) -> None:
        """confidence + uncertainty must be ≤ 1.0 even when all models are UNAVAILABLE."""
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        result = engine.decide(model_outputs=_all_unavailable_outputs(), symbol="NIFTY")

        total = result.confidence + result.uncertainty
        assert total <= 1.0 + 1e-9, (
            f"confidence ({result.confidence}) + uncertainty ({result.uncertainty}) = "
            f"{total:.6f} > 1.0"
        )

    def test_confidence_in_unit_interval(self) -> None:
        """confidence must lie in [0, 1]."""
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        result = engine.decide(
            model_outputs=_outputs_with_agreement(0.71), symbol="NIFTY"
        )

        assert 0.0 <= result.confidence <= 1.0, (
            f"confidence={result.confidence} outside [0, 1]"
        )

    def test_uncertainty_in_unit_interval(self) -> None:
        """uncertainty must lie in [0, 1]."""
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        result = engine.decide(model_outputs=_all_unavailable_outputs(), symbol="NIFTY")

        assert 0.0 <= result.uncertainty <= 1.0, (
            f"uncertainty={result.uncertainty} outside [0, 1]"
        )

    @given(model_outputs=_st_all_trained)
    @h_settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_confidence_uncertainty_bound_hypothesis(
        self, model_outputs: list[dict[str, Any]]
    ) -> None:
        """
        Hypothesis: for any mix of TRAINED_MODEL outputs,
        confidence + uncertainty <= 1.0.

        **Validates: Requirements 10.1**
        """
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        result = engine.decide(model_outputs=model_outputs, symbol="NIFTY")

        total = result.confidence + result.uncertainty
        assert total <= 1.0 + 1e-9, (
            f"confidence ({result.confidence}) + uncertainty ({result.uncertainty}) = "
            f"{total:.6f} > 1.0"
        )


# ── MetaOutput Schema Validation ─────────────────────────────────────────────


class TestMetaOutputSchema:
    """decide() must always return a fully-formed MetaOutput."""

    def test_decide_returns_meta_output_instance(self) -> None:
        """Return type must be MetaOutput."""
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        result = engine.decide(
            model_outputs=_outputs_with_agreement(0.86), symbol="NIFTY"
        )

        assert isinstance(result, MetaOutput), (
            f"decide() must return MetaOutput, got {type(result).__name__}"
        )

    def test_meta_output_has_required_fields(self) -> None:
        """All required MetaOutput fields must be present and accessible."""
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        result = engine.decide(model_outputs=_all_unavailable_outputs(), symbol="NIFTY")

        required_fields = [
            "action",
            "confidence",
            "uncertainty",
            "agreement",
            "agreement_ratio",
            "provenance",
            "reason_codes",
            "abstention",
        ]
        for field in required_fields:
            assert hasattr(result, field), f"MetaOutput missing required field: {field}"

    def test_action_is_valid_literal(self) -> None:
        """action must be one of the four allowed literals."""
        MetaDecisionEngine = _skip_if_not_implemented()

        engine = MetaDecisionEngine()
        result = engine.decide(model_outputs=_all_unavailable_outputs(), symbol="NIFTY")

        assert result.action in ("BUY", "SELL", "WAIT", "NO_TRADE"), (
            f"action={result.action!r} is not a valid MetaOutput action"
        )
