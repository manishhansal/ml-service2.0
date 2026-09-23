"""
API contract tests for ml-service2.0.

Task 8.1: Schema round-trip serialization tests.
Task 8.2: HTTP auth and status integration tests (added later).

TDD Phase: Tests written BEFORE full implementation.
Requirements: Req 19.1, Req 19.2, Req 19.3, Req 19.4, Req 15.2, Req 15.8, Req 15.9
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings as h_settings
from hypothesis import strategies as st

# Import all schemas
from src.schemas.base import (
    DriftSeverity,
    EvidenceLevel,
    ExecutionAction,
    GateResult,
    ImpactDirection,
    IVRegime,
    MarketRegime,
    ModelLifecycleStage,
    PredictionProvenance,
    PromotionOutcome,
    RecommendedAction,
    TradingStrategy,
)
from src.schemas.features import FeatureQualityReport, FeatureVector
from src.schemas.meta import ConfidenceDecomposition, MetaOutput, NewsSignal
from src.schemas.monitoring import DriftAlert
from src.schemas.predictions import (
    ExecutionDecision,
    ExplainResponse,
    FeatureContribution,
    IVRegimeResponse,
    PriceRegimeResponse,
    PortfolioResponse,
    RankingResponse,
    RegimePredictionResponse,
    RiskResponse,
    StrategyResponse,
)
from src.schemas.registry import (
    GateEvaluation,
    ModelArtifact,
    PromotionDecision,
    TrainingRun,
)
from src.schemas.streaming import SignalEvent


# ─── Property: round-trip serialisation ──────────────────────────────────────


class TestSchemaRoundTrip:
    """
    Property 20: parse(serialize(instance)) == instance for all schemas.
    Property 21: parse(serialize(parse(x))) == parse(x) for all valid schemas.
    """

    def test_regime_prediction_response_round_trip(self):
        """RegimePredictionResponse survives JSON round-trip."""
        instance = RegimePredictionResponse(
            regime=MarketRegime.BULL,
            confidence=0.75,
            probabilities={"bull": 0.75, "sideways": 0.25},
            features_used=10,
            model_version="1.0.0",
            provenance=PredictionProvenance.TRAINED_MODEL,
        )
        serialized = instance.model_dump_json()
        parsed = RegimePredictionResponse.model_validate_json(serialized)
        assert parsed == instance, "RegimePredictionResponse round-trip failed"

    def test_meta_output_round_trip(self):
        """MetaOutput survives JSON round-trip."""
        decomp = ConfidenceDecomposition(
            base_confidence=0.7,
            calibration_quality=0.8,
            agreement_bonus=0.1,
            data_quality_factor=0.9,
            regime_confidence_factor=0.75,
        )
        instance = MetaOutput(
            action="BUY",
            confidence=0.82,
            uncertainty=0.18,
            agreement=0.85,
            agreement_ratio=0.85,
            ensemble_score=0.82,
            provenance=PredictionProvenance.TRAINED_MODEL,
            decomposition=decomp,
        )
        serialized = instance.model_dump_json()
        parsed = MetaOutput.model_validate_json(serialized)
        assert parsed == instance, "MetaOutput round-trip failed"

    def test_risk_response_round_trip(self):
        instance = RiskResponse(
            prob_stop_hit=0.3,
            prob_target_hit=0.5,
            expected_drawdown_pct=2.5,
            suggested_position_size_pct=1.0,
            risk_score=4.0,
            factors={"vix_regime": 0.2},
            provenance=PredictionProvenance.HEURISTIC,
        )
        parsed = RiskResponse.model_validate_json(instance.model_dump_json())
        assert parsed == instance

    def test_drift_alert_round_trip(self):
        from datetime import datetime, timezone

        instance = DriftAlert(
            alert_id="test-alert-001",
            model_name="market_regime",
            feature_name="india_vix",
            severity=DriftSeverity.HIGH,
            psi_value=0.28,
            reference_mean=15.0,
            reference_std=3.0,
            current_mean=22.0,
            current_std=5.0,
            recommended_action=RecommendedAction.RETRAIN,
            triggered_at=datetime(2025, 1, 15, 9, 30, 0, tzinfo=timezone.utc),
        )
        parsed = DriftAlert.model_validate_json(instance.model_dump_json())
        assert parsed == instance

    def test_model_artifact_round_trip(self):
        instance = ModelArtifact(
            model_name="stock_ranker",
            version="1.0.0",
            stage=ModelLifecycleStage.PRODUCTION,
            artifact_path="/artifacts/stock_ranker/1.0.0/model.txt",
            sha256_checksum="abc123" * 10,
            training_date="2025-01-01",
            training_dataset_hash="def456" * 10,
            provenance=PredictionProvenance.TRAINED_MODEL,
        )
        parsed = ModelArtifact.model_validate_json(instance.model_dump_json())
        assert parsed == instance

    def test_signal_event_round_trip(self):
        from datetime import datetime, timezone

        instance = SignalEvent(
            event_id="evt-001",
            symbol="NIFTY",
            timestamp=datetime(2025, 1, 15, 9, 30, 0, tzinfo=timezone.utc),
            action="BUY",
            confidence=0.82,
            provenance="trained_model",
        )
        parsed = SignalEvent.model_validate_json(instance.model_dump_json())
        assert parsed == instance

    def test_feature_quality_report_round_trip(self):
        from datetime import datetime, timezone

        instance = FeatureQualityReport(
            batch_id="batch-001",
            timestamp=datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc),
            total_features_requested=158,
            missing_count=5,
        )
        parsed = FeatureQualityReport.model_validate_json(instance.model_dump_json())
        assert parsed == instance


# ─── Property: idempotent parse-serialize cycle ───────────────────────────────


class TestIdempotentParseSerialize:
    """Property 21: parse(serialize(parse(x))) == parse(x) for all valid schemas."""

    def test_regime_response_idempotent(self):
        json_str = json.dumps(
            {
                "regime": "bull",
                "confidence": 0.75,
                "probabilities": {"bull": 0.75, "sideways": 0.25},
                "features_used": 10,
                "model_version": "1.0.0",
                "provenance": "trained_model",
                "shap_top10": [],
            }
        )
        first_parse = RegimePredictionResponse.model_validate_json(json_str)
        second_parse = RegimePredictionResponse.model_validate_json(
            first_parse.model_dump_json()
        )
        assert first_parse == second_parse, (
            "Idempotent parse-serialize failed for RegimePredictionResponse"
        )

    def test_meta_output_idempotent(self):
        json_str = json.dumps(
            {
                "action": "WAIT",
                "confidence": 0.5,
                "uncertainty": 0.5,
                "agreement": 0.6,
                "agreement_ratio": 0.6,
                "provenance": "heuristic",
            }
        )
        first = MetaOutput.model_validate_json(json_str)
        second = MetaOutput.model_validate_json(first.model_dump_json())
        assert first == second


# ─── Property: invalid enum values return 422 ─────────────────────────────────


class TestInvalidEnumValidation:
    """Property: unknown enum values in requests raise Pydantic ValidationError."""

    def test_invalid_market_regime_raises(self):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError) as exc_info:
            RegimePredictionResponse(
                regime="invalid_regime",  # not a valid MarketRegime
                confidence=0.5,
                probabilities={},
                features_used=0,
                model_version="1.0.0",
            )
        # Verify the error mentions the field
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("regime",) for e in errors)

    def test_invalid_trading_strategy_raises(self):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            StrategyResponse(
                strategy="invalid_strategy",
                confidence=0.5,
                alternatives=[],
                rationale="test",
            )

    def test_invalid_provenance_raises(self):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            RegimePredictionResponse(
                regime=MarketRegime.BULL,
                confidence=0.5,
                probabilities={},
                features_used=0,
                model_version="1.0.0",
                provenance="not_a_valid_provenance",  # invalid enum value
            )

    def test_unknown_provenance_string_in_signal_event(self):
        """Unknown provenance in SignalEvent is stored as string (not enum) — should not raise."""
        from datetime import datetime, timezone

        # SignalEvent.provenance is str, not enum — unknown values are OK
        instance = SignalEvent(
            event_id="evt-002",
            symbol="RELIANCE",
            timestamp=datetime(2025, 1, 15, 9, 30, 0, tzinfo=timezone.utc),
            action="WAIT",
            confidence=0.5,
            provenance="unknown_future_provenance",  # str field — should NOT raise
        )
        assert instance.provenance == "unknown_future_provenance"


# ─── Property: PredictionProvenance.is_live_eligible ─────────────────────────


class TestPredictionProvenance:
    def test_only_trained_model_is_live_eligible(self):
        assert PredictionProvenance.TRAINED_MODEL.is_live_eligible is True
        assert PredictionProvenance.HEURISTIC.is_live_eligible is False
        assert PredictionProvenance.INSUFFICIENT_EVIDENCE.is_live_eligible is False
        assert PredictionProvenance.UNAVAILABLE.is_live_eligible is False

    def test_provenance_serializes_as_string(self):
        """PredictionProvenance values serialize as their string values, not enum names."""
        assert PredictionProvenance.TRAINED_MODEL.value == "trained_model"
        assert PredictionProvenance.HEURISTIC.value == "heuristic"


# ─── Property: risk invariant ─────────────────────────────────────────────────


class TestRiskInvariants:
    def test_prob_stop_plus_target_can_be_at_most_one(self):
        """Risk response allows prob_stop_hit + prob_target_hit <= 1.0."""
        instance = RiskResponse(
            prob_stop_hit=0.4,
            prob_target_hit=0.5,  # 0.4 + 0.5 = 0.9 <= 1.0 ✓
            expected_drawdown_pct=2.0,
            suggested_position_size_pct=1.0,
            risk_score=5.0,
            factors={},
        )
        assert instance.prob_stop_hit + instance.prob_target_hit <= 1.0 + 1e-9


# ─── HTTP integration tests ────────────────────────────────────────────────────
# NOTE: These tests use FastAPI's TestClient with the app directly.
# The app stub returns 503 for unimplemented endpoints — that's expected.
# These tests only verify auth, routing, and schema validation at the HTTP layer.
#
# NB: monitoring/* and training/* routers are included WITHOUT a /v2 prefix in
# src/main.py, so their paths are /monitoring/... and /training/... directly.

import httpx
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def test_client():
    """TestClient that does NOT follow lifespan (avoids Redis/audit startup)."""
    import os

    os.environ["ML_SERVICE_API_KEY"] = "test-key-for-testing"
    os.environ["DATA_SERVICE_API_KEY"] = "test-data-key"
    # Import after env vars are set
    from src.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestHealthEndpoint:
    """Health endpoint must return 200 without API key."""

    def test_health_returns_200(self, test_client):
        r = test_client.get("/health")
        assert r.status_code == 200

    def test_health_returns_healthy_status(self, test_client):
        r = test_client.get("/health")
        body = r.json()
        assert body["status"] == "healthy"
        assert "timestamp" in body
        assert body["version"] == "2.0.0"

    def test_health_does_not_require_api_key(self, test_client):
        """Health endpoint is exempt from X-API-KEY check."""
        r = test_client.get("/health")  # no API key
        assert r.status_code == 200


class TestAPIKeyAuthentication:
    """All guarded endpoints require X-API-KEY header."""

    # monitoring/* and training/* have no /v2 prefix — they are at root level
    GUARDED_ENDPOINTS = [
        ("GET", "/v2/models/status"),
        ("GET", "/v2/models/registry"),
        ("GET", "/v2/features/quality"),
        ("GET", "/monitoring/drift"),
        ("GET", "/monitoring/performance"),
        ("GET", "/monitoring/alerts"),
    ]

    @pytest.mark.parametrize("method,path", GUARDED_ENDPOINTS)
    def test_returns_401_without_api_key(self, test_client, method, path):
        r = test_client.request(method, path)
        assert r.status_code == 401, (
            f"Expected 401 for {method} {path}, got {r.status_code}"
        )

    @pytest.mark.parametrize("method,path", GUARDED_ENDPOINTS)
    def test_returns_not_401_with_valid_api_key(self, test_client, method, path):
        r = test_client.request(
            method, path, headers={"X-API-KEY": "test-key-for-testing"}
        )
        # Should NOT be 401 (may be 200, 503, etc. depending on implementation stage)
        assert r.status_code != 401, (
            f"Expected non-401 for {method} {path} with valid key, got {r.status_code}"
        )

    def test_returns_401_with_wrong_api_key(self, test_client):
        r = test_client.get("/v2/models/status", headers={"X-API-KEY": "wrong-key"})
        assert r.status_code == 401

    def test_returns_401_with_empty_api_key(self, test_client):
        r = test_client.get("/v2/models/status", headers={"X-API-KEY": ""})
        assert r.status_code == 401


class TestRequestIDPropagation:
    """X-Request-ID header must be echoed back in responses."""

    def test_request_id_is_echoed(self, test_client):
        r = test_client.get(
            "/health", headers={"X-Request-ID": "my-request-id-123"}
        )
        assert r.headers.get("X-Request-ID") == "my-request-id-123"

    def test_request_id_generated_when_absent(self, test_client):
        r = test_client.get("/health")
        # Should have a generated X-Request-ID in the response
        assert "X-Request-ID" in r.headers
        assert len(r.headers["X-Request-ID"]) > 0


class TestValidationErrorHandling:
    """Invalid request bodies must return 422 with field-level errors."""

    def test_invalid_regime_value_returns_422(self, test_client):
        payload = {
            "regime": "NOT_A_VALID_REGIME",  # invalid enum
            "symbol": "NIFTY",
            "rsi": 60.0,
            "adx": 25.0,
            "atr_pct": 1.5,
            "volume_ratio": 1.2,
            "vwap_distance_pct": 0.3,
            "bollinger_position": 0.5,
            "trend_strength": 0.4,
            "volatility_rank": 0.6,
            "time_of_day_minutes": 120,
            "iv_regime": "STABLE",
        }
        r = test_client.post(
            "/v2/predict/strategy",
            json=payload,
            headers={"X-API-KEY": "test-key-for-testing"},
        )
        assert r.status_code == 422

    def test_missing_required_field_returns_422(self, test_client):
        """A request missing required fields should return 422."""
        # RiskRequest requires symbol, direction, entry, stop_loss, target, atr, regime, etc.
        payload = {"symbol": "NIFTY"}  # missing required fields
        r = test_client.post(
            "/v2/predict/risk",
            json=payload,
            headers={"X-API-KEY": "test-key-for-testing"},
        )
        assert r.status_code == 422


class TestPredictionEndpointsExist:
    """All prediction endpoints must exist (503 stub is acceptable at this stage)."""

    POST_ENDPOINTS = [
        "/v2/predict/regime",
        "/v2/predict/rankings",
        "/v2/predict/strategy",
        "/v2/predict/risk",
        "/v2/predict/portfolio",
        "/v2/predict/portfolio-v2",
        "/v2/predict/execution",
        "/v2/predict/price-regime",
        "/v2/predict/iv-regime",
        "/v2/meta/decide",
        "/v2/analytics/greeks",
        "/v2/analytics/gex",
        "/v2/analytics/vpin",
        "/v2/analytics/vol-surface",
    ]

    @pytest.mark.parametrize("path", POST_ENDPOINTS)
    def test_endpoint_exists_not_404(self, test_client, path):
        """Endpoint must exist — 503 is acceptable, 404 is not."""
        r = test_client.post(
            path,
            json={},
            headers={"X-API-KEY": "test-key-for-testing"},
        )
        assert r.status_code != 404, (
            f"Endpoint {path} returned 404 — endpoint not registered"
        )
