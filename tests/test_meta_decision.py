"""
tests/test_meta_decision.py — TDD: MetaEngine Go/No-Go contract validation.

This file drives the Phase-2 ``MetaEngine`` stub interface defined in
``src/meta_engine/engine.py``.  All tests that call ``MetaEngine.decide()``
are expected to raise ``NotImplementedError`` until Phase 3 provides the
implementation.

Additionally, this file tests the *existing* ``MetaDecisionEngine`` from
``src/meta/engine.py`` to document its Go/No-Go contract for Phase 3 reviewers
and provide regression protection.

Test categories
---------------
1. Phase-2 MetaEngine stub     — verifies the stub raises NotImplementedError
                                  and that ``GoNoGoDecision`` schema is correct.
2. Go/No-Go contract (existing) — tests ``MetaDecisionEngine.decide()``
                                   for the three core properties that Phase 3 must preserve.
3. Drift detector stub          — ``DriftDetector`` must raise NotImplementedError.

Markers: @pytest.mark.unit, @pytest.mark.tdd
"""
from __future__ import annotations

import os
from typing import Any

import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as h_settings
from hypothesis import strategies as st
from pydantic import ValidationError

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from src.meta_engine.engine import GoNoGoDecision, MetaEngine
from src.monitoring.drift import DriftDetector, DriftReport
from src.schemas.base import PredictionProvenance


# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_output(
    model_id: str,
    action: str = "WAIT",
    direction: int = 0,
    confidence: float = 0.5,
    provenance: str = "trained_model",
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "action": action,
        "confidence": confidence,
        "direction": direction,
        "provenance": provenance,
    }


def _all_unavailable(n: int = 7) -> list[dict[str, Any]]:
    return [_make_output(f"m_{i}", action="WAIT", direction=0, provenance="unavailable") for i in range(n)]


def _all_buy(n: int = 7) -> list[dict[str, Any]]:
    return [_make_output(f"m_{i}", action="BUY", direction=1, confidence=0.80) for i in range(n)]


def _low_agreement(n_total: int = 7, n_agreeing: int = 3) -> list[dict[str, Any]]:
    """n_agreeing BUY + (n_total - n_agreeing) WAIT — agreement_ratio < 0.5."""
    return [
        _make_output(f"m_{i}", action="BUY" if i < n_agreeing else "WAIT", direction=1 if i < n_agreeing else 0)
        for i in range(n_total)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Phase-2 MetaEngine stub — all methods must raise NotImplementedError
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.tdd
class TestMetaEngineStub:
    """``MetaEngine`` is a Phase-2 stub — all public methods raise NotImplementedError."""

    def test_decide_raises_not_implemented(self) -> None:
        """``decide()`` must raise NotImplementedError (Phase 2 TDD stub)."""
        engine = MetaEngine()
        with pytest.raises(NotImplementedError, match="Phase 2"):
            engine.decide(model_outputs=_all_buy(), symbol="NIFTY")

    def test_decide_with_unavailable_inputs_raises_not_implemented(self) -> None:
        """``decide()`` must raise NotImplementedError even for all-UNAVAILABLE inputs."""
        engine = MetaEngine()
        with pytest.raises(NotImplementedError):
            engine.decide(model_outputs=_all_unavailable(), symbol="NIFTY")

    @pytest.mark.asyncio
    async def test_adecide_raises_not_implemented(self) -> None:
        """Async ``adecide()`` must also raise NotImplementedError."""
        engine = MetaEngine()
        with pytest.raises(NotImplementedError):
            await engine.adecide(model_outputs=_all_buy(), symbol="NIFTY")

    def test_explain_raises_not_implemented(self) -> None:
        """``explain()`` must raise NotImplementedError until Phase 3."""
        engine = MetaEngine()
        dummy_decision = GoNoGoDecision(
            action="NO_GO",
            confidence=0.0,
            uncertainty=1.0,
            provenance=PredictionProvenance.UNAVAILABLE,
        )
        with pytest.raises(NotImplementedError):
            engine.explain(dummy_decision, "NIFTY")

    @given(n_models=st.integers(min_value=1, max_value=10))
    @h_settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_decide_always_raises_for_any_input_size(self, n_models: int) -> None:
        """Hypothesis: decide() raises NotImplementedError for any number of model outputs."""
        engine = MetaEngine()
        outputs = [_make_output(f"m_{i}", action="BUY", direction=1) for i in range(n_models)]
        with pytest.raises(NotImplementedError):
            engine.decide(model_outputs=outputs, symbol="NIFTY")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GoNoGoDecision schema — contract validation
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.tdd
class TestGoNoGoDecisionSchema:
    """``GoNoGoDecision`` Pydantic model must enforce its field constraints."""

    def test_go_decision_parses(self) -> None:
        decision = GoNoGoDecision(
            action="GO",
            confidence=0.82,
            uncertainty=0.18,
            provenance=PredictionProvenance.TRAINED_MODEL,
        )
        assert decision.action == "GO"
        assert decision.confidence == 0.82

    def test_no_go_decision_parses(self) -> None:
        decision = GoNoGoDecision(
            action="NO_GO",
            confidence=0.0,
            uncertainty=1.0,
            provenance=PredictionProvenance.UNAVAILABLE,
        )
        assert decision.action == "NO_GO"
        assert decision.abstention is False

    def test_wait_decision_with_abstention_parses(self) -> None:
        decision = GoNoGoDecision(
            action="WAIT",
            confidence=0.30,
            uncertainty=0.70,
            abstention=True,
            provenance=PredictionProvenance.HEURISTIC,
        )
        assert decision.abstention is True
        assert decision.action == "WAIT"

    def test_invalid_action_rejected(self) -> None:
        """Action 'BUY' is not a valid GoNoGoDecision action (must be GO/NO_GO/WAIT)."""
        with pytest.raises(ValidationError):
            GoNoGoDecision(
                action="BUY",  # type: ignore[arg-type]
                confidence=0.8,
                uncertainty=0.2,
            )

    def test_confidence_above_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GoNoGoDecision(action="GO", confidence=1.01, uncertainty=0.0)  # type: ignore[arg-type]

    def test_confidence_below_0_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GoNoGoDecision(action="GO", confidence=-0.01, uncertainty=0.0)  # type: ignore[arg-type]

    def test_rationale_max_length_enforced(self) -> None:
        """rationale must be ≤ 200 characters."""
        long_rationale = "x" * 201
        with pytest.raises(ValidationError):
            GoNoGoDecision(
                action="GO",
                confidence=0.8,
                uncertainty=0.2,
                rationale=long_rationale,
            )

    def test_rationale_200_chars_is_valid(self) -> None:
        """rationale of exactly 200 characters must be accepted."""
        decision = GoNoGoDecision(
            action="WAIT",
            confidence=0.5,
            uncertainty=0.5,
            rationale="x" * 200,
        )
        assert len(decision.rationale) == 200

    def test_agreement_ratio_defaults_to_zero(self) -> None:
        """agreement_ratio defaults to 0.0 when not provided."""
        decision = GoNoGoDecision(action="NO_GO", confidence=0.0, uncertainty=1.0)
        assert decision.agreement_ratio == 0.0

    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        uncertainty=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @h_settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_confidence_uncertainty_pairs_parse(
        self, confidence: float, uncertainty: float
    ) -> None:
        """Hypothesis: any (confidence, uncertainty) pair in [0,1] must parse."""
        decision = GoNoGoDecision(
            action="WAIT",
            confidence=confidence,
            uncertainty=uncertainty,
        )
        assert 0.0 <= decision.confidence <= 1.0
        assert 0.0 <= decision.uncertainty <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Existing MetaDecisionEngine — Go/No-Go contract regression
# ═══════════════════════════════════════════════════════════════════════════════


def _get_meta_decision_engine() -> Any:
    """Import MetaDecisionEngine; skip if not yet importable."""
    try:
        from src.meta.engine import MetaDecisionEngine
        return MetaDecisionEngine
    except (ImportError, ModuleNotFoundError):
        pytest.skip("MetaDecisionEngine not importable — skip regression tests")


@pytest.mark.unit
@pytest.mark.tdd
class TestMetaDecisionEngineGoNoGoContract:
    """
    Regression tests for ``MetaDecisionEngine.decide()`` that document the
    Go/No-Go contract Phase 3 must preserve.

    These tests mirror the properties in ``tests/test_meta_decision_engine.py``
    but are phrased in terms of the Phase-2 interface: the focus is on whether
    the engine's decision is safe for capital deployment (Go) or not (No-Go).
    """

    def test_all_unavailable_is_not_go(self) -> None:
        """When all models are UNAVAILABLE, the decision must never be 'GO' / 'BUY'."""
        MetaDecisionEngine = _get_meta_decision_engine()
        engine = MetaDecisionEngine()
        result = engine.decide(model_outputs=_all_unavailable(), symbol="NIFTY")

        # The existing engine returns MetaOutput with action in {BUY, SELL, WAIT, NO_TRADE}
        # A safe decision for UNAVAILABLE provenance must be NO_TRADE
        assert result.action == "NO_TRADE", (
            f"All-UNAVAILABLE inputs must yield NO_TRADE, got {result.action}"
        )

    def test_all_unavailable_sets_unavailable_provenance(self) -> None:
        MetaDecisionEngine = _get_meta_decision_engine()
        engine = MetaDecisionEngine()
        result = engine.decide(model_outputs=_all_unavailable(), symbol="NIFTY")
        assert result.provenance == PredictionProvenance.UNAVAILABLE

    def test_low_agreement_triggers_abstention(self) -> None:
        """3/7 agreement ratio triggers abstention — not safe for live capital."""
        MetaDecisionEngine = _get_meta_decision_engine()
        engine = MetaDecisionEngine()
        result = engine.decide(model_outputs=_low_agreement(), symbol="NIFTY")

        assert result.abstention is True, (
            f"Low agreement (3/7) must set abstention=True, got {result.abstention}"
        )
        assert result.action in ("WAIT", "NO_TRADE"), (
            f"Low agreement must produce WAIT or NO_TRADE, got {result.action}"
        )

    def test_confidence_and_uncertainty_bounded(self) -> None:
        """confidence + uncertainty must always be <= 1.0."""
        MetaDecisionEngine = _get_meta_decision_engine()
        engine = MetaDecisionEngine()
        result = engine.decide(model_outputs=_all_buy(), symbol="NIFTY")

        total = result.confidence + result.uncertainty
        assert total <= 1.0 + 1e-9, (
            f"confidence ({result.confidence}) + uncertainty ({result.uncertainty}) "
            f"= {total:.6f} > 1.0"
        )

    def test_idempotence_for_buy_scenario(self) -> None:
        """Identical inputs must always produce identical outputs (idempotence)."""
        MetaDecisionEngine = _get_meta_decision_engine()
        engine = MetaDecisionEngine()
        outputs = _all_buy()

        r1 = engine.decide(model_outputs=outputs, symbol="NIFTY")
        r2 = engine.decide(model_outputs=outputs, symbol="NIFTY")

        assert r1.action == r2.action, (
            f"Idempotence violated: {r1.action} vs {r2.action}"
        )
        assert abs(r1.confidence - r2.confidence) < 1e-9

    def test_idempotence_for_unavailable_scenario(self) -> None:
        """Idempotence must hold for all-UNAVAILABLE inputs too."""
        MetaDecisionEngine = _get_meta_decision_engine()
        engine = MetaDecisionEngine()
        outputs = _all_unavailable()

        r1 = engine.decide(model_outputs=outputs, symbol="NIFTY")
        r2 = engine.decide(model_outputs=outputs, symbol="NIFTY")

        assert r1.action == r2.action

    @given(
        model_outputs=st.lists(
            st.fixed_dictionaries({
                "model_id": st.text(min_size=1, max_size=20),
                "action": st.sampled_from(["BUY", "SELL", "WAIT"]),
                "confidence": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                "direction": st.integers(min_value=-1, max_value=1),
                "provenance": st.just(PredictionProvenance.UNAVAILABLE.value),
            }),
            min_size=7,
            max_size=7,
        )
    )
    @h_settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_all_unavailable_hypothesis_never_produces_buy_or_sell(
        self, model_outputs: list[dict[str, Any]]
    ) -> None:
        """Hypothesis: 7 UNAVAILABLE models MUST yield NO_TRADE (absorbing identity)."""
        MetaDecisionEngine = _get_meta_decision_engine()
        engine = MetaDecisionEngine()
        result = engine.decide(model_outputs=model_outputs, symbol="NIFTY")

        assert result.action == "NO_TRADE", (
            f"Absorbing identity violated: all UNAVAILABLE but action={result.action}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DriftDetector stub — all methods must raise NotImplementedError
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.tdd
class TestDriftDetectorStub:
    """``DriftDetector`` is a Phase-2 stub — all methods raise NotImplementedError."""

    def test_detect_raises_not_implemented(self) -> None:
        detector = DriftDetector()
        with pytest.raises(NotImplementedError, match="Phase 2"):
            detector.detect(reference=None, current=None)

    def test_should_retrain_raises_not_implemented(self) -> None:
        from datetime import timezone
        from datetime import datetime
        import uuid
        detector = DriftDetector()
        dummy_report = DriftReport(
            report_id=str(uuid.uuid4()),
            detected_at=datetime.now(tz=timezone.utc),
        )
        with pytest.raises(NotImplementedError):
            detector.should_retrain(dummy_report)

    def test_get_feature_importance_raises_not_implemented(self) -> None:
        from datetime import timezone
        from datetime import datetime
        import uuid
        detector = DriftDetector()
        dummy_report = DriftReport(
            report_id=str(uuid.uuid4()),
            detected_at=datetime.now(tz=timezone.utc),
        )
        with pytest.raises(NotImplementedError):
            detector.get_feature_importance(dummy_report)

    def test_drift_report_schema_is_valid(self) -> None:
        """DriftReport Pydantic model must be instantiable with defaults."""
        from datetime import timezone
        from datetime import datetime
        import uuid
        report = DriftReport(
            report_id=str(uuid.uuid4()),
            detected_at=datetime.now(tz=timezone.utc),
        )
        assert report.n_drifted == 0
        assert report.drift_fraction == 0.0

    def test_drift_detector_threshold_is_stored(self) -> None:
        """Constructor threshold must be stored on the instance."""
        detector = DriftDetector(threshold=0.15)
        assert detector.threshold == 0.15

    def test_drift_detector_default_threshold(self) -> None:
        """Default threshold must be 0.1 (PSI standard)."""
        detector = DriftDetector()
        assert detector.threshold == 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Integration: conftest fixtures work with MetaDecisionEngine
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestConftestFixturesIntegrity:
    """Verify the Phase-2 conftest fixtures are internally consistent."""

    def test_all_buy_outputs_fixture(self, all_buy_outputs: list[dict[str, Any]]) -> None:
        """all_buy_outputs fixture must have 7 BUY outputs with TRAINED_MODEL provenance."""
        assert len(all_buy_outputs) == 7
        for output in all_buy_outputs:
            assert output["action"] == "BUY"
            assert output["direction"] == 1
            assert output["provenance"] == "trained_model"

    def test_all_unavailable_outputs_fixture(
        self, all_unavailable_outputs: list[dict[str, Any]]
    ) -> None:
        """all_unavailable_outputs fixture must have 7 UNAVAILABLE outputs."""
        assert len(all_unavailable_outputs) == 7
        for output in all_unavailable_outputs:
            assert output["provenance"] == "unavailable"

    def test_low_agreement_outputs_fixture(
        self, low_agreement_outputs: list[dict[str, Any]]
    ) -> None:
        """low_agreement_outputs must have 3 BUY + 4 WAIT = 7 total."""
        assert len(low_agreement_outputs) == 7
        buys = [o for o in low_agreement_outputs if o["action"] == "BUY"]
        waits = [o for o in low_agreement_outputs if o["action"] == "WAIT"]
        assert len(buys) == 3
        assert len(waits) == 4

    def test_mock_sentinel_pulse_fixture(
        self, mock_sentinel_pulse: dict[str, Any]
    ) -> None:
        """mock_sentinel_pulse fixture must have all mandatory SentinelPulse fields."""
        required = {"instrument", "as_of", "news_impact_score", "impact_direction",
                    "impact_confidence", "sentiment"}
        assert required.issubset(mock_sentinel_pulse.keys())
        assert mock_sentinel_pulse["news_impact_score"] == 0.512
        assert mock_sentinel_pulse["impact_direction"] == "BULLISH"

    def test_mock_live_quote_fixture(self, mock_live_quote: dict[str, Any]) -> None:
        """mock_live_quote fixture must contain data + metadata.quality."""
        assert "data" in mock_live_quote
        assert "metadata" in mock_live_quote
        quality = mock_live_quote["metadata"]["quality"]
        assert quality["score"] == 88
        assert quality["signalEngineAllowed"] is True
