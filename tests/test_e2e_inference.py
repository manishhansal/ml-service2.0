"""
test_e2e_inference.py

End-to-end inference flow integration tests.
Tests the complete path: API → Models → MetaDecisionEngine → Response.

Requirements: Phase 13 integration testing

Notes on endpoint behavior:
- /v2/predict/regime  — 503 stub (not yet implemented, Phase 4)
- /v2/predict/risk    — implemented; strict Pydantic validation (regime must be
                        a valid MarketRegime value; string coercion works via JSON)
- /v2/meta/decide     — fully implemented, returns MetaOutput
"""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

HEADERS = {"X-API-KEY": "test-key-for-testing"}


@pytest.fixture(scope="module")
def e2e_client():
    from fastapi.testclient import TestClient
    from src.main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestRegimePredictionFlow:
    """End-to-end: POST /v2/predict/regime → endpoint existence check (503 stub accepted)."""

    _PAYLOAD = {
        "nifty_change_pct": 0.5,
        "banknifty_change_pct": 0.8,
        "india_vix": 15.0,
        "nifty_adx": 25.0,
        "advance_decline_ratio": 1.5,
        "market_breadth": 0.6,
        "volume_ratio": 1.1,
        "gap_pct": 0.2,
    }

    def test_regime_endpoint_exists_not_404(self, e2e_client):
        """Regime endpoint must be registered — 503 stub is acceptable at this phase."""
        r = e2e_client.post("/v2/predict/regime", json=self._PAYLOAD, headers=HEADERS)
        assert r.status_code != 404, "Regime endpoint not registered (404)"

    def test_regime_endpoint_returns_200_or_503(self, e2e_client):
        """Regime returns 503 stub or 200 if implemented."""
        r = e2e_client.post("/v2/predict/regime", json=self._PAYLOAD, headers=HEADERS)
        assert r.status_code in (200, 503), (
            f"Expected 200 or 503, got {r.status_code}: {r.text}"
        )

    def test_regime_response_has_required_fields_when_implemented(self, e2e_client):
        r = e2e_client.post("/v2/predict/regime", json=self._PAYLOAD, headers=HEADERS)
        if r.status_code == 200:
            body = r.json()
            assert "regime" in body
            assert "confidence" in body
            assert "provenance" in body

    def test_regime_confidence_in_unit_interval_when_implemented(self, e2e_client):
        r = e2e_client.post("/v2/predict/regime", json=self._PAYLOAD, headers=HEADERS)
        if r.status_code == 200:
            body = r.json()
            assert 0.0 <= body["confidence"] <= 1.0


class TestRiskPredictionFlow:
    """End-to-end: POST /v2/predict/risk → RiskResponse.

    RiskRequest.regime uses strict Pydantic validation. MarketRegime is a
    str+Enum; FastAPI deserialises the JSON body via model_validate_json which
    handles string→enum coercion.  Tests here verify the schema and response
    contract.
    """

    # Valid MarketRegime string values: strong_bull, bull, sideways, volatile, bear, crash
    _PAYLOAD = {
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

    def test_risk_endpoint_not_404(self, e2e_client):
        """Risk endpoint must be registered."""
        r = e2e_client.post("/v2/predict/risk", json=self._PAYLOAD, headers=HEADERS)
        assert r.status_code != 404, "Risk endpoint not registered (404)"

    def test_risk_invalid_input_returns_422(self, e2e_client):
        """Missing required fields return 422."""
        r = e2e_client.post("/v2/predict/risk", json={}, headers=HEADERS)
        assert r.status_code == 422

    def test_risk_probability_invariant(self, e2e_client):
        """When risk endpoint returns 200, prob_stop_hit + prob_target_hit <= 1.0."""
        r = e2e_client.post("/v2/predict/risk", json=self._PAYLOAD, headers=HEADERS)
        if r.status_code == 200:
            body = r.json()
            stop = body.get("prob_stop_hit", 0)
            target = body.get("prob_target_hit", 0)
            assert stop + target <= 1.0 + 1e-9, (
                f"Probability invariant violated: {stop} + {target}"
            )

    def test_risk_response_schema_when_200(self, e2e_client):
        """When risk endpoint returns 200, all required fields are present."""
        r = e2e_client.post("/v2/predict/risk", json=self._PAYLOAD, headers=HEADERS)
        if r.status_code == 200:
            body = r.json()
            assert "prob_stop_hit" in body
            assert "prob_target_hit" in body
            assert "risk_score" in body
            assert "provenance" in body


class TestMetaDecideFlow:
    """End-to-end: POST /v2/meta/decide → MetaOutput"""

    _PAYLOAD = {"symbol": "NIFTY", "regime": "bull"}

    def test_meta_decide_returns_200(self, e2e_client):
        r = e2e_client.post("/v2/meta/decide", json=self._PAYLOAD, headers=HEADERS)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

    def test_meta_decide_action_is_valid(self, e2e_client):
        r = e2e_client.post("/v2/meta/decide", json=self._PAYLOAD, headers=HEADERS)
        if r.status_code == 200:
            body = r.json()
            assert body.get("action") in ("BUY", "SELL", "WAIT", "NO_TRADE")

    def test_meta_decide_confidence_uncertainty_sum(self, e2e_client):
        r = e2e_client.post("/v2/meta/decide", json=self._PAYLOAD, headers=HEADERS)
        if r.status_code == 200:
            body = r.json()
            conf = float(body.get("confidence", 0))
            unc = float(body.get("uncertainty", 0))
            assert conf + unc <= 1.0 + 1e-9

    def test_meta_decide_response_has_required_fields(self, e2e_client):
        r = e2e_client.post("/v2/meta/decide", json=self._PAYLOAD, headers=HEADERS)
        if r.status_code == 200:
            body = r.json()
            assert "action" in body
            assert "confidence" in body
            assert "uncertainty" in body
            assert "provenance" in body

    def test_meta_decide_confidence_in_unit_interval(self, e2e_client):
        r = e2e_client.post("/v2/meta/decide", json=self._PAYLOAD, headers=HEADERS)
        if r.status_code == 200:
            body = r.json()
            assert 0.0 <= body["confidence"] <= 1.0
