"""
test_security.py

Security and boundary tests.

Requirements: Req 15.2, Req 15.10, Phase 13 integration testing
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

VALID_HEADERS = {"X-API-KEY": "test-key-for-testing"}


@pytest.fixture(scope="module")
def sec_client():
    from fastapi.testclient import TestClient
    from src.main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestAuthEnforcement:
    """All guarded endpoints must reject requests without a valid API key."""

    GUARDED_ENDPOINTS = [
        ("POST", "/v2/predict/regime"),
        ("POST", "/v2/predict/risk"),
        ("POST", "/v2/meta/decide"),
        ("GET", "/v2/models/status"),
        ("GET", "/monitoring/alerts"),
    ]

    @pytest.mark.parametrize("method,path", GUARDED_ENDPOINTS)
    def test_no_api_key_returns_401(self, sec_client, method: str, path: str):
        r = sec_client.request(method, path, json={})
        assert r.status_code == 401

    @pytest.mark.parametrize("method,path", GUARDED_ENDPOINTS)
    def test_wrong_api_key_returns_401(self, sec_client, method: str, path: str):
        r = sec_client.request(method, path, json={}, headers={"X-API-KEY": "wrong-key"})
        assert r.status_code == 401

    def test_health_exempt_from_auth(self, sec_client):
        r = sec_client.get("/health")
        assert r.status_code == 200

    def test_empty_api_key_returns_401(self, sec_client):
        r = sec_client.get("/v2/models/status", headers={"X-API-KEY": ""})
        assert r.status_code == 401


class TestInputBoundary:
    """Boundary inputs must return 422, not 500."""

    def test_extreme_float_values_handled(self, sec_client):
        payload = {
            "symbol": "NIFTY",
            "direction": "LONG",
            "entry": 22000.0,
            "stop_loss": 21800.0,
            "target": 22400.0,
            "atr": 200.0,
            "regime": "bull",
            "rsi": 1e308,  # extreme value
            "adx": 25.0,
            "volume_ratio": 1.1,
            "vix": 15.0,
        }
        r = sec_client.post("/v2/predict/risk", json=payload, headers=VALID_HEADERS)
        assert r.status_code in (200, 422), f"Unexpected status {r.status_code}"

    def test_empty_body_returns_422(self, sec_client):
        r = sec_client.post("/v2/predict/risk", json={}, headers=VALID_HEADERS)
        assert r.status_code == 422

    def test_unknown_origin_cors_not_allowed(self, sec_client):
        """Requests from unknown origins should not have wildcard CORS allow headers."""
        r = sec_client.get(
            "/health",
            headers={"Origin": "https://malicious-origin.example.com"},
        )
        # The response should not have Access-Control-Allow-Origin: *
        assert (
            "access-control-allow-origin" not in r.headers
            or r.headers.get("access-control-allow-origin") != "*"
        )


class TestPredictionErrorHandling:
    """Prediction endpoints must never return 500 for malformed input."""

    @pytest.mark.parametrize("path,payload", [
        ("/v2/predict/regime", {"invalid_field": "value"}),
        ("/v2/predict/strategy", {"regime": "invalid_regime_value"}),
        ("/v2/predict/execution", {"symbol": "NIFTY"}),  # missing required
    ])
    def test_malformed_input_returns_422_not_500(
        self, sec_client, path: str, payload: dict
    ):
        r = sec_client.post(path, json=payload, headers=VALID_HEADERS)
        assert r.status_code != 500, f"{path} returned 500 for malformed input"
        assert r.status_code in (200, 422, 503), (
            f"Unexpected status {r.status_code} for {path}"
        )
