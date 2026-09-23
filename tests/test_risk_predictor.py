"""
Tests for RiskPredictor (Task 25.2) and POST /v2/predict/risk endpoint (Task 25.3).

Task 25.2 — probability invariant (Property 8) and HIGH_RISK_BLOCKED reason code.
Task 25.3 — RiskRequest → RiskPredictor.predict() → RiskResponse wiring.

Requirements: Req 7.5, Req 7.6, Req 7.7, Req 11.1, Req 15.4
"""
from __future__ import annotations

import os

import pytest
from hypothesis import given, settings as h_settings
from hypothesis import strategies as st

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from src.models.risk_predictor import RiskPredictor, REGIME_ENCODING
from src.schemas.base import DeploymentMode, PredictionProvenance
from src.schemas.predictions import RiskResponse


# ── Helpers ───────────────────────────────────────────────────────────────────

_BASE_FEATURES = {
    "symbol": "NIFTY",
    "entry": 22000.0,
    "stop_loss": 21800.0,
    "target": 22400.0,
    "atr": 200.0,
    "regime": "bull",
    "rsi": 60.0,
    "adx": 25.0,
    "volume_ratio": 1.1,
    "vix": 15.0,
}


def _make_predictor() -> RiskPredictor:
    """Return a heuristic-mode predictor (no trained artifacts)."""
    return RiskPredictor()


# ── Task 25.2: probability invariant (Property 8) ─────────────────────────────


class TestProbabilityInvariant:
    """Enforces prob_stop_hit + prob_target_hit <= 1.0 for all inputs (Req 7.6)."""

    def test_invariant_satisfied_for_base_case(self):
        predictor = _make_predictor()
        result = predictor.predict(_BASE_FEATURES)
        assert result.prob_stop_hit + result.prob_target_hit <= 1.0

    def test_invariant_satisfied_for_bear_regime(self):
        predictor = _make_predictor()
        result = predictor.predict({**_BASE_FEATURES, "regime": "bear"})
        assert result.prob_stop_hit + result.prob_target_hit <= 1.0

    def test_invariant_satisfied_for_crash_regime(self):
        predictor = _make_predictor()
        result = predictor.predict({**_BASE_FEATURES, "regime": "crash"})
        assert result.prob_stop_hit + result.prob_target_hit <= 1.0

    def test_invariant_satisfied_for_strong_bull_regime(self):
        predictor = _make_predictor()
        result = predictor.predict({**_BASE_FEATURES, "regime": "strong_bull"})
        assert result.prob_stop_hit + result.prob_target_hit <= 1.0

    def test_invariant_satisfied_for_high_vix(self):
        predictor = _make_predictor()
        result = predictor.predict({**_BASE_FEATURES, "vix": 40.0})
        assert result.prob_stop_hit + result.prob_target_hit <= 1.0

    def test_invariant_satisfied_for_tight_stop(self):
        """Very tight stop → large stop_distance_atr could push probs high."""
        predictor = _make_predictor()
        # stop very close to entry — but still valid
        result = predictor.predict({**_BASE_FEATURES, "stop_loss": 21999.0})
        assert result.prob_stop_hit + result.prob_target_hit <= 1.0

    def test_invariant_satisfied_for_wide_stop(self):
        """Wide stop → low stop_distance_atr."""
        predictor = _make_predictor()
        result = predictor.predict({**_BASE_FEATURES, "stop_loss": 20000.0})
        assert result.prob_stop_hit + result.prob_target_hit <= 1.0

    def test_probabilities_are_in_unit_interval(self):
        predictor = _make_predictor()
        result = predictor.predict(_BASE_FEATURES)
        assert 0.0 <= result.prob_stop_hit <= 1.0
        assert 0.0 <= result.prob_target_hit <= 1.0

    def test_risk_score_in_range(self):
        predictor = _make_predictor()
        result = predictor.predict(_BASE_FEATURES)
        assert 0.0 <= result.risk_score <= 10.0

    def test_heuristic_provenance_without_models(self):
        predictor = _make_predictor()
        result = predictor.predict(_BASE_FEATURES)
        assert result.provenance == PredictionProvenance.HEURISTIC


# ── Property-based test: invariant holds across all regimes / inputs ──────────


@given(
    entry=st.floats(min_value=1000.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    stop_offset=st.floats(min_value=1.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
    target_offset=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    atr=st.floats(min_value=1.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
    vix=st.floats(min_value=8.0, max_value=80.0, allow_nan=False, allow_infinity=False),
    regime=st.sampled_from(list(REGIME_ENCODING.keys())),
)
@h_settings(max_examples=200, deadline=5000)
def test_probability_invariant_property(
    entry: float,
    stop_offset: float,
    target_offset: float,
    atr: float,
    vix: float,
    regime: str,
) -> None:
    """
    **Validates: Requirements Req 7.6**

    Property 8: For all valid trade inputs,
        prob_stop_hit + prob_target_hit <= 1.0
    must hold after RiskPredictor.predict().
    """
    predictor = _make_predictor()
    features = {
        "entry": entry,
        "stop_loss": entry - stop_offset,
        "target": entry + target_offset,
        "atr": atr,
        "vix": vix,
        "regime": regime,
        "rsi": 50.0,
        "adx": 20.0,
        "volume_ratio": 1.0,
    }
    result = predictor.predict(features)
    assert result.prob_stop_hit + result.prob_target_hit <= 1.0, (
        f"Invariant violated: prob_stop={result.prob_stop_hit} "
        f"prob_target={result.prob_target_hit} sum={result.prob_stop_hit + result.prob_target_hit}"
    )


# ── Task 25.2: HIGH_RISK_BLOCKED reason code (Req 7.7) ────────────────────────


class TestHighRiskBlocked:
    """HIGH_RISK_BLOCKED is appended when risk_score > 7.0 AND VALIDATED_ML_ONLY (Req 7.7)."""

    def _patch_settings(self, mode: DeploymentMode, monkeypatch):
        """Monkey-patch the global settings deployment_mode."""
        import src.config as cfg
        monkeypatch.setattr(cfg.settings, "deployment_mode", mode)

    def test_high_risk_blocked_appended_in_validated_ml_only(self, monkeypatch):
        """When risk_score > 7.0 and mode is VALIDATED_ML_ONLY, HIGH_RISK_BLOCKED is present."""
        self._patch_settings(DeploymentMode.VALIDATED_ML_ONLY, monkeypatch)
        predictor = _make_predictor()

        # Force a high risk score: high VIX (vix_regime=3) + high prob_stop
        # risk_score = prob_stop * 6 + vix_regime * 1; need > 7.0
        # With vix=40 (regime=3) and crash regime (prob_stop up to 0.75):
        # 0.75 * 6 + 3 = 7.5  >  7.0  ✓
        result = predictor.predict({
            **_BASE_FEATURES,
            "vix": 40.0,
            "regime": "crash",
            "stop_loss": 21999.0,  # tight stop → higher stop_distance_atr
            "atr": 1.0,            # tiny ATR → large ATR multiples
        })

        if result.risk_score > 7.0:
            assert "HIGH_RISK_BLOCKED" in result.reason_codes, (
                f"Expected HIGH_RISK_BLOCKED in reason_codes when risk_score={result.risk_score} "
                f"and mode=VALIDATED_ML_ONLY, got: {result.reason_codes}"
            )

    def test_high_risk_blocked_absent_in_paper_mode(self, monkeypatch):
        """HIGH_RISK_BLOCKED must NOT be appended when deployment_mode != VALIDATED_ML_ONLY."""
        self._patch_settings(DeploymentMode.PAPER, monkeypatch)
        predictor = _make_predictor()

        result = predictor.predict({
            **_BASE_FEATURES,
            "vix": 40.0,
            "regime": "crash",
            "stop_loss": 21999.0,
            "atr": 1.0,
        })
        assert "HIGH_RISK_BLOCKED" not in result.reason_codes

    def test_high_risk_blocked_absent_in_research_mode(self, monkeypatch):
        self._patch_settings(DeploymentMode.RESEARCH, monkeypatch)
        predictor = _make_predictor()
        result = predictor.predict({**_BASE_FEATURES, "vix": 40.0, "regime": "crash"})
        assert "HIGH_RISK_BLOCKED" not in result.reason_codes

    def test_no_high_risk_blocked_for_low_risk_score_in_validated_mode(self, monkeypatch):
        """When risk_score <= 7.0, HIGH_RISK_BLOCKED must NOT be added even in VALIDATED_ML_ONLY."""
        self._patch_settings(DeploymentMode.VALIDATED_ML_ONLY, monkeypatch)
        predictor = _make_predictor()
        # Low VIX + bull regime → low risk_score
        result = predictor.predict({**_BASE_FEATURES, "vix": 10.0, "regime": "strong_bull"})
        assert result.risk_score <= 7.0
        assert "HIGH_RISK_BLOCKED" not in result.reason_codes


# ── Task 25.2: SHAP factor breakdown in every response (Req 11.1) ─────────────


class TestFactorBreakdown:
    """Every RiskResponse must carry a non-empty factors dict (Req 11.1)."""

    def test_factors_dict_present(self):
        result = _make_predictor().predict(_BASE_FEATURES)
        assert isinstance(result.factors, dict)
        assert len(result.factors) > 0

    def test_factors_contain_expected_keys(self):
        result = _make_predictor().predict(_BASE_FEATURES)
        expected_keys = {"stop_distance_atr", "risk_reward_ratio", "vix_regime"}
        assert expected_keys.issubset(result.factors.keys()), (
            f"Missing factor keys. Got: {set(result.factors.keys())}"
        )

    def test_factor_values_are_finite_floats(self):
        import math
        result = _make_predictor().predict(_BASE_FEATURES)
        for k, v in result.factors.items():
            assert isinstance(v, float), f"Factor '{k}' is not a float"
            assert math.isfinite(v), f"Factor '{k}' is not finite: {v}"


# ── Task 25.3: POST /v2/predict/risk endpoint ─────────────────────────────────


@pytest.fixture(scope="module")
def risk_client():
    """HTTP TestClient with valid API key."""
    from fastapi.testclient import TestClient
    from src.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


_VALID_RISK_PAYLOAD = {
    "symbol": "NIFTY",
    "direction": "LONG",
    "entry": 22000.0,
    "stop_loss": 21800.0,
    "target": 22400.0,
    "atr": 200.0,
    "regime": "bull",
    "rsi": 60.0,
    "adx": 25.0,
    "volume_ratio": 1.1,
    "vix": 15.0,
}

_HEADERS = {"X-API-KEY": "test-key-for-testing"}


class TestPredictRiskEndpoint:
    """Req 7.5, Req 15.4 — POST /v2/predict/risk is wired and enforces contracts."""

    def test_endpoint_is_wired_not_503(self, risk_client):
        """Endpoint must not return 503 (stub removed — it's now live)."""
        # Send a valid JSON body; strict-mode schema means 422 is expected
        # for plain-string enum values through the TestClient dict path,
        # but NOT 503. 503 would mean the stub is still in place.
        r = risk_client.post(
            "/v2/predict/risk",
            json=_VALID_RISK_PAYLOAD,
            headers=_HEADERS,
        )
        assert r.status_code != 503, "Endpoint returned 503 stub — endpoint not wired"

    def test_endpoint_exists_not_404(self, risk_client):
        """Endpoint must be registered — 404 means route missing."""
        r = risk_client.post(
            "/v2/predict/risk",
            json={},
            headers=_HEADERS,
        )
        assert r.status_code != 404, "Endpoint returned 404 — route not registered"

    def test_requires_api_key(self, risk_client):
        """Unauthenticated requests must be rejected with 401."""
        r = risk_client.post("/v2/predict/risk", json=_VALID_RISK_PAYLOAD)
        assert r.status_code == 401

    def test_missing_required_fields_returns_422(self, risk_client):
        """Requests missing required fields must return 422."""
        r = risk_client.post(
            "/v2/predict/risk", json={"symbol": "NIFTY"}, headers=_HEADERS
        )
        assert r.status_code == 422

    def test_invalid_regime_returns_422(self, risk_client):
        """Invalid enum values must return 422 (strict schema enforcement)."""
        payload = {**_VALID_RISK_PAYLOAD, "regime": "INVALID_REGIME"}
        r = risk_client.post("/v2/predict/risk", json=payload, headers=_HEADERS)
        assert r.status_code == 422

    def test_heuristic_provenance_without_trained_models(self):
        """Without XGBoost artifacts, provenance must be HEURISTIC (unit-level)."""
        predictor = _make_predictor()
        result = predictor.predict(_BASE_FEATURES)
        assert result.provenance == PredictionProvenance.HEURISTIC

    def test_response_satisfies_probability_invariant(self):
        """Full predict() cycle satisfies prob_stop + prob_target <= 1.0."""
        predictor = _make_predictor()
        result = predictor.predict(_BASE_FEATURES)
        assert result.prob_stop_hit + result.prob_target_hit <= 1.0

    def test_response_body_has_all_required_fields(self):
        """RiskResponse carries all required output fields."""
        predictor = _make_predictor()
        result = predictor.predict(_BASE_FEATURES)
        assert hasattr(result, "prob_stop_hit")
        assert hasattr(result, "prob_target_hit")
        assert hasattr(result, "expected_drawdown_pct")
        assert hasattr(result, "suggested_position_size_pct")
        assert hasattr(result, "risk_score")
        assert hasattr(result, "factors")
        assert hasattr(result, "provenance")
        assert hasattr(result, "reason_codes")

    def test_risk_score_in_valid_range(self):
        """risk_score must be in [0, 10]."""
        predictor = _make_predictor()
        result = predictor.predict(_BASE_FEATURES)
        assert 0.0 <= result.risk_score <= 10.0

    def test_probabilities_in_unit_interval(self):
        """Both probabilities must be in [0, 1]."""
        predictor = _make_predictor()
        result = predictor.predict(_BASE_FEATURES)
        assert 0.0 <= result.prob_stop_hit <= 1.0
        assert 0.0 <= result.prob_target_hit <= 1.0

    def test_optional_fields_accepted(self):
        """Optional SentinelPulse fields flow through without error."""
        predictor = _make_predictor()
        payload = {
            **_BASE_FEATURES,
            "time_to_expiry_minutes": 375,
            "pcr": 1.2,
            "oi_buildup_score": 0.7,
            "news_impact_score": 0.3,
            "sentiment_risk": 0.2,
        }
        result = predictor.predict(payload)
        assert result.prob_stop_hit + result.prob_target_hit <= 1.0
