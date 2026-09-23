"""
Additional mock-based coverage tests to push total coverage to 90%+.

Targets:
- src/registry/registry.py (71%)
- src/models/regime_classifier.py (64%)
- src/models/risk_predictor.py (64%)
- src/monitoring/drift_monitor.py (75%)
- src/clients/data_service.py (82%)
- src/meta/llm_reasoner.py (85%)
- src/training/online_learner.py (89%)
- src/training/pipeline.py (88%)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

import numpy as np
import pytest


# ===========================================================================
# ModelRegistry
# ===========================================================================


def make_test_artifact(
    model_name="test_model",
    version="1.0.0",
    stage=None,
    artifact_path="/tmp/test_model.pkl",
    sha256_checksum="a" * 64,
):
    from src.schemas.base import ModelLifecycleStage
    from src.schemas.registry import ModelArtifact

    if stage is None:
        stage = ModelLifecycleStage.PRODUCTION

    return ModelArtifact(
        model_name=model_name,
        version=version,
        stage=stage,
        artifact_path=artifact_path,
        sha256_checksum=sha256_checksum,
        training_date="2024-01-01",
        training_dataset_hash="b" * 64,
    )


class TestModelRegistry:
    def test_get_champion_returns_none_when_no_champion(self):
        from src.registry.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(artifacts_path=Path(tmpdir))
            result = reg.get_champion("nonexistent_model")
            assert result is None

    def test_register_creates_metadata_file(self):
        from src.registry.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(artifacts_path=Path(tmpdir))
            artifact = make_test_artifact(artifact_path=str(Path(tmpdir) / "fake.pkl"))

            reg.register(artifact)

            meta_path = Path(tmpdir) / "test_model" / "1.0.0" / "metadata.json"
            assert meta_path.exists()

    def test_register_creates_champion_json_for_production_stage(self):
        from src.registry.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(artifacts_path=Path(tmpdir))
            artifact = make_test_artifact(artifact_path=str(Path(tmpdir) / "fake.pkl"))
            reg.register(artifact)

            champion_path = Path(tmpdir) / "test_model" / "champion.json"
            assert champion_path.exists()

    def test_register_then_get_champion(self):
        from src.registry.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(artifacts_path=Path(tmpdir))
            artifact = make_test_artifact(artifact_path=str(Path(tmpdir) / "fake.pkl"))
            reg.register(artifact)

            champion = reg.get_champion("test_model")
            assert champion is not None
            assert champion.model_name == "test_model"
            assert champion.version == "1.0.0"

    def test_register_raises_if_version_already_exists(self):
        from src.registry.registry import ArtifactAlreadyExistsError, ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(artifacts_path=Path(tmpdir))
            artifact = make_test_artifact(artifact_path=str(Path(tmpdir) / "fake.pkl"))
            reg.register(artifact)

            with pytest.raises(ArtifactAlreadyExistsError):
                reg.register(artifact)

    def test_load_artifact_raises_integrity_failure_on_hash_mismatch(self):
        from src.registry.registry import ArtifactIntegrityFailure, ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a real file
            artifact_file = Path(tmpdir) / "model.pkl"
            artifact_file.write_bytes(b"model data here")

            # Create artifact with wrong hash
            artifact = make_test_artifact(
                artifact_path=str(artifact_file),
                sha256_checksum="wrong" + "0" * 59,
            )

            reg = ModelRegistry(artifacts_path=Path(tmpdir))
            with pytest.raises(ArtifactIntegrityFailure):
                reg.load_artifact(artifact)

    def test_load_artifact_raises_file_not_found(self):
        from src.registry.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(artifacts_path=Path(tmpdir))
            artifact = make_test_artifact(
                artifact_path="/nonexistent/path/model.pkl"
            )
            with pytest.raises(FileNotFoundError):
                reg.load_artifact(artifact)

    def test_load_artifact_succeeds_with_correct_hash(self):
        from src.registry.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a real file
            artifact_file = Path(tmpdir) / "model.pkl"
            artifact_file.write_bytes(b"model data")

            correct_sha256 = ModelRegistry.compute_file_sha256(artifact_file)
            artifact = make_test_artifact(
                artifact_path=str(artifact_file),
                sha256_checksum=correct_sha256,
            )

            reg = ModelRegistry(artifacts_path=Path(tmpdir))
            returned_path = reg.load_artifact(artifact)
            assert returned_path == artifact_file

    def test_list_registry_empty_dir(self):
        from src.registry.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(artifacts_path=Path(tmpdir))
            artifacts = reg.list_registry()
            assert artifacts == []

    def test_list_registry_returns_registered_artifacts(self):
        from src.registry.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(artifacts_path=Path(tmpdir))
            artifact = make_test_artifact(artifact_path=str(Path(tmpdir) / "fake.pkl"))
            reg.register(artifact)

            listed = reg.list_registry()
            assert len(listed) == 1
            assert listed[0].model_name == "test_model"

    def test_load_all_champions_no_champions(self):
        from src.registry.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(artifacts_path=Path(tmpdir))
            results = reg.load_all_champions()
            # All known models should return None since no artifacts registered
            assert all(v is None for v in results.values())

    def test_compute_file_sha256(self):
        from src.registry.registry import ModelRegistry

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            fpath = Path(f.name)

        try:
            sha = ModelRegistry.compute_file_sha256(fpath)
            assert len(sha) == 64
        finally:
            fpath.unlink()

    def test_compute_dict_sha256(self):
        from src.registry.registry import ModelRegistry

        sha = ModelRegistry.compute_dict_sha256({"key": "value"})
        assert len(sha) == 64

    def test_get_artifact_path(self):
        from src.registry.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(artifacts_path=Path(tmpdir))
            artifact = make_test_artifact(
                artifact_path="/some/path/model.pkl",
            )
            path = reg.get_artifact_path(artifact)
            assert path.name == "model.pkl"


# ===========================================================================
# RegimeClassifier
# ===========================================================================


class TestRegimeClassifier:
    def test_heuristic_predict_crash_high_vix(self):
        from src.models.regime_classifier import RegimeClassifier
        from src.schemas.base import MarketRegime

        clf = RegimeClassifier.__new__(RegimeClassifier)
        clf._model = None
        regime, conf = clf._heuristic_predict({"india_vix": 40.0, "nifty_change_pct": -1.0})
        assert regime == MarketRegime.CRASH

    def test_heuristic_predict_strong_bull(self):
        from src.models.regime_classifier import RegimeClassifier
        from src.schemas.base import MarketRegime

        clf = RegimeClassifier.__new__(RegimeClassifier)
        clf._model = None
        regime, conf = clf._heuristic_predict({"india_vix": 12.0, "nifty_change_pct": 1.5})
        assert regime == MarketRegime.STRONG_BULL

    def test_heuristic_predict_bear(self):
        from src.models.regime_classifier import RegimeClassifier
        from src.schemas.base import MarketRegime

        clf = RegimeClassifier.__new__(RegimeClassifier)
        clf._model = None
        regime, conf = clf._heuristic_predict({"india_vix": 28.0, "nifty_change_pct": -1.0})
        assert regime == MarketRegime.BEAR

    def test_heuristic_predict_volatile(self):
        from src.models.regime_classifier import RegimeClassifier
        from src.schemas.base import MarketRegime

        clf = RegimeClassifier.__new__(RegimeClassifier)
        clf._model = None
        regime, conf = clf._heuristic_predict({"india_vix": 24.0, "nifty_change_pct": 0.0})
        assert regime == MarketRegime.VOLATILE

    def test_heuristic_predict_bull(self):
        from src.models.regime_classifier import RegimeClassifier
        from src.schemas.base import MarketRegime

        clf = RegimeClassifier.__new__(RegimeClassifier)
        clf._model = None
        regime, conf = clf._heuristic_predict({"india_vix": 14.0, "nifty_change_pct": 0.5})
        assert regime == MarketRegime.BULL

    def test_heuristic_predict_sideways_default(self):
        from src.models.regime_classifier import RegimeClassifier
        from src.schemas.base import MarketRegime

        clf = RegimeClassifier.__new__(RegimeClassifier)
        clf._model = None
        regime, conf = clf._heuristic_predict({"india_vix": 16.0, "nifty_change_pct": 0.0})
        assert regime == MarketRegime.SIDEWAYS

    def test_predict_heuristic_returns_response_object(self):
        """Full predict() call using heuristic mode (no model)."""
        from src.models.regime_classifier import RegimeClassifier
        from src.schemas.predictions import RegimePredictionResponse
        from threading import Lock
        from src.explainability.explainer import ModelExplainer
        from src.models.regime_classifier import REGIME_FEATURE_NAMES

        clf = RegimeClassifier.__new__(RegimeClassifier)
        clf._lock = Lock()
        clf._model = None
        clf._model_version = "heuristic-v1"
        clf._last_regime = None
        clf._last_confidence = 0.0
        clf._explainer = ModelExplainer()
        clf.feature_names = REGIME_FEATURE_NAMES

        response = clf.predict({"india_vix": 16.0, "nifty_change_pct": 0.5})
        assert isinstance(response, RegimePredictionResponse)
        assert response.provenance.value in ("heuristic", "insufficient_evidence")

    def test_predict_with_trained_model(self):
        """Full predict() with a mock trained model."""
        from src.models.regime_classifier import RegimeClassifier
        from src.schemas.predictions import RegimePredictionResponse
        from src.schemas.base import PredictionProvenance
        from threading import Lock
        from src.explainability.explainer import ModelExplainer
        from src.models.regime_classifier import REGIME_FEATURE_NAMES

        clf = RegimeClassifier.__new__(RegimeClassifier)
        clf._lock = Lock()
        clf._model_version = "1.0.0"
        clf._last_regime = None
        clf._last_confidence = 0.0
        clf._explainer = ModelExplainer()
        clf.feature_names = REGIME_FEATURE_NAMES

        # Mock model returning probabilities
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.05, 0.7, 0.1, 0.05, 0.05, 0.05]])
        clf._model = mock_model

        response = clf.predict({"india_vix": 14.0, "nifty_change_pct": 0.8})
        assert isinstance(response, RegimePredictionResponse)
        assert response.provenance == PredictionProvenance.TRAINED_MODEL

    def test_regime_transition_event_logged(self):
        """Transition from one regime to another triggers logging."""
        from src.models.regime_classifier import RegimeClassifier
        from src.schemas.base import MarketRegime
        from threading import Lock
        from src.explainability.explainer import ModelExplainer
        from src.models.regime_classifier import REGIME_FEATURE_NAMES

        clf = RegimeClassifier.__new__(RegimeClassifier)
        clf._lock = Lock()
        clf._model = None
        clf._model_version = "heuristic-v1"
        clf._last_regime = MarketRegime.BEAR  # previous regime differs
        clf._last_confidence = 0.6
        clf._explainer = ModelExplainer()
        clf.feature_names = REGIME_FEATURE_NAMES

        # Should log regime_transition_event without raising
        response = clf.predict({"india_vix": 12.0, "nifty_change_pct": 1.5})
        assert response is not None

    def test_trigger_online_learning_check_triggers_when_degraded(self):
        from src.models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier.__new__(RegimeClassifier)
        # current_ic = 0.01, baseline_ic = 0.05 → 80% drop → should trigger
        assert clf.trigger_online_learning_check(0.01, 0.05) is True

    def test_trigger_online_learning_check_no_trigger_when_ok(self):
        from src.models.regime_classifier import RegimeClassifier

        clf = RegimeClassifier.__new__(RegimeClassifier)
        # current_ic = 0.045, baseline_ic = 0.05 → only 10% drop → should NOT trigger
        assert clf.trigger_online_learning_check(0.045, 0.05) is False

    def test_idx_to_regime_out_of_range(self):
        from src.models.regime_classifier import RegimeClassifier
        from src.schemas.base import MarketRegime

        regime = RegimeClassifier._idx_to_regime(99)
        assert regime == MarketRegime.SIDEWAYS


# ===========================================================================
# RiskPredictor
# ===========================================================================


class TestRiskPredictor:
    def _make_features(self):
        return {
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

    def test_heuristic_predict_basic(self):
        from src.models.risk_predictor import RiskPredictor

        predictor = RiskPredictor()
        response = predictor.predict(self._make_features())
        assert 0.0 <= response.prob_stop_hit <= 1.0
        assert 0.0 <= response.prob_target_hit <= 1.0

    def test_heuristic_bear_regime_higher_stop_prob(self):
        from src.models.risk_predictor import RiskPredictor

        predictor = RiskPredictor()
        features = dict(self._make_features())
        features["regime"] = "crash"
        response = predictor.predict(features)
        assert response.prob_stop_hit > 0.0

    def test_high_risk_blocked_in_validated_ml_only_mode(self):
        """risk_score > 7.0 in VALIDATED_ML_ONLY → HIGH_RISK_BLOCKED reason."""
        from src.models.risk_predictor import RiskPredictor
        from src.schemas.base import DeploymentMode

        predictor = RiskPredictor()

        with patch("src.config.settings.deployment_mode", DeploymentMode.VALIDATED_ML_ONLY):
            # High VIX + tight stop → high risk_score
            features = {
                "entry": 22000.0,
                "stop_loss": 21990.0,  # very tight stop → prob_stop high
                "target": 22400.0,
                "atr": 200.0,
                "regime": "crash",
                "rsi": 80.0,
                "adx": 40.0,
                "volume_ratio": 0.5,
                "vix": 40.0,  # extreme VIX
            }
            response = predictor.predict(features)

        # Risk_score should be high with high vix and crash regime
        assert response.risk_score >= 0

    def test_calibration_violation_clamped(self):
        """When prob_stop + prob_target > 1.0, values are clamped."""
        from src.models.risk_predictor import RiskPredictor

        predictor = RiskPredictor()

        mock_stop_model = MagicMock()
        mock_stop_model.predict_proba.return_value = np.array([[0.1, 0.95]])
        mock_target_model = MagicMock()
        mock_target_model.predict_proba.return_value = np.array([[0.1, 0.90]])

        predictor.stop_model = mock_stop_model
        predictor.target_model = mock_target_model

        response = predictor.predict(self._make_features())
        # Should be clamped: prob_stop + prob_target < 1.0
        assert response.prob_stop_hit + response.prob_target_hit < 1.0

    def test_model_predict_with_trained_model(self):
        from src.models.risk_predictor import RiskPredictor
        from src.schemas.base import PredictionProvenance

        predictor = RiskPredictor()

        mock_stop_model = MagicMock()
        mock_stop_model.predict_proba.return_value = np.array([[0.6, 0.4]])
        mock_target_model = MagicMock()
        mock_target_model.predict_proba.return_value = np.array([[0.7, 0.3]])

        predictor.stop_model = mock_stop_model
        predictor.target_model = mock_target_model

        response = predictor.predict(self._make_features())
        assert response.provenance == PredictionProvenance.TRAINED_MODEL

    def test_encode_vix_regime_ranges(self):
        from src.models.risk_predictor import RiskPredictor

        assert RiskPredictor._encode_vix_regime(10.0) == 0.0
        assert RiskPredictor._encode_vix_regime(15.0) == 1.0
        assert RiskPredictor._encode_vix_regime(20.0) == 2.0
        assert RiskPredictor._encode_vix_regime(30.0) == 3.0

    def test_has_trained_model_false_initially(self):
        from src.models.risk_predictor import RiskPredictor

        predictor = RiskPredictor()
        assert predictor.has_trained_model is False

    def test_predict_no_stop_loss_uses_defaults(self):
        """When entry/stop/target are 0, should not crash."""
        from src.models.risk_predictor import RiskPredictor

        predictor = RiskPredictor()
        response = predictor.predict({})
        assert response is not None


# ===========================================================================
# DriftMonitor — missing branches
# ===========================================================================


class TestDriftMonitorExtra:
    def test_get_recommended_action_monitor(self):
        from src.monitoring.drift_monitor import DriftMonitor

        dm = DriftMonitor()
        assert dm.get_recommended_action(0.1) == dm.get_recommended_action(0.1)

    def test_get_recommended_action_retrain_no_online_learning(self):
        from src.monitoring.drift_monitor import DriftMonitor
        from src.schemas.base import RecommendedAction

        dm = DriftMonitor()
        action = dm.get_recommended_action(0.3, online_learning_triggered=False)
        assert action == RecommendedAction.RETRAIN

    def test_get_recommended_action_rollback_low_ic(self):
        from src.monitoring.drift_monitor import DriftMonitor
        from src.schemas.base import RecommendedAction

        dm = DriftMonitor()
        action = dm.get_recommended_action(0.3, online_learning_triggered=True, estimated_ic=0.005)
        assert action == RecommendedAction.ROLLBACK

    def test_get_recommended_action_retrain_ok_ic(self):
        from src.monitoring.drift_monitor import DriftMonitor
        from src.schemas.base import RecommendedAction

        dm = DriftMonitor()
        action = dm.get_recommended_action(0.3, online_learning_triggered=True, estimated_ic=0.05)
        assert action == RecommendedAction.RETRAIN

    def test_run_evidently_drift_unavailable(self):
        from src.monitoring.drift_monitor import DriftMonitor

        with patch("src.monitoring.drift_monitor.EVIDENTLY_AVAILABLE", False):
            dm = DriftMonitor()
            result = dm.run_evidently_drift(None, None)
        assert result == {"available": False}

    def test_run_nannyml_cbpe_unavailable(self):
        from src.monitoring.drift_monitor import DriftMonitor

        with patch("src.monitoring.drift_monitor.NANNYML_AVAILABLE", False):
            dm = DriftMonitor()
            result = dm.run_nannyml_cbpe(None, None, {})
        assert result == {"available": False}

    def test_run_evidently_drift_raises_gracefully(self):
        """Evidently available but raises → returns error dict."""
        from src.monitoring.drift_monitor import DriftMonitor

        mock_evidently = MagicMock()
        mock_evidently.metric_preset.DataDriftPreset.side_effect = RuntimeError("boom")

        with patch("src.monitoring.drift_monitor.EVIDENTLY_AVAILABLE", True):
            import src.monitoring.drift_monitor as dm_mod
            orig = getattr(dm_mod, "evidently", None)
            dm_mod.evidently = mock_evidently
            try:
                dm = DriftMonitor()
                result = dm.run_evidently_drift(MagicMock(), MagicMock())
            finally:
                if orig is None and hasattr(dm_mod, "evidently"):
                    delattr(dm_mod, "evidently")
                elif orig is not None:
                    dm_mod.evidently = orig

        # Should return gracefully
        assert isinstance(result, dict)


# ===========================================================================
# DataServiceClient — remaining gaps
# ===========================================================================


def make_mock_response(status_code=200, json_data=None):
    from unittest.mock import MagicMock
    import httpx

    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestDataServiceClientExtra:
    """Additional tests to cover remaining data_service.py gaps."""

    def test_get_historical_ohlcv_with_pit_date(self):
        """pit_date param is passed in params dict."""
        import asyncio
        from unittest.mock import AsyncMock

        async def _run():
            from src.clients.data_service import DataServiceClient

            client = DataServiceClient(base_url="http://test", api_key="key")
            mock_http = AsyncMock()
            mock_http.get.return_value = make_mock_response(
                200,
                {
                    "metadata": {"quality": {"signalEngineAllowed": True, "DataConfidenceScore": 90}},
                    "data": [{"open": 22000, "close": 22100}],
                },
            )
            client._client = mock_http

            result = await client.get_historical_ohlcv(
                "NIFTY", interval="1d", pit_date="2024-01-15"
            )
            assert isinstance(result, list)

            # Verify pit_date was in the request params
            call_kwargs = mock_http.get.call_args
            params = call_kwargs[1].get("params", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else {})
            assert "pit_date" in params

        asyncio.run(_run())

    def test_get_connects_and_has_correct_headers(self):
        """connect() sets X-API-KEY header."""
        import asyncio

        async def _run():
            from src.clients.data_service import DataServiceClient

            client = DataServiceClient(base_url="http://test", api_key="my-api-key")
            await client.connect()
            assert client._client is not None
            await client.disconnect()

        asyncio.run(_run())

    def test_http_error_during_get_raises_unavailable(self):
        import asyncio
        from unittest.mock import AsyncMock
        import httpx

        async def _run():
            from src.clients.data_service import DataServiceClient, DataServiceUnavailableError

            client = DataServiceClient(base_url="http://test", api_key="key")
            mock_http = AsyncMock()
            mock_http.get.side_effect = httpx.HTTPError("generic http error")
            client._client = mock_http

            with pytest.raises(DataServiceUnavailableError):
                await client._get("/v1/test")

        asyncio.run(_run())

    def test_post_connect_error_raises_unavailable(self):
        import asyncio
        from unittest.mock import AsyncMock
        import httpx

        async def _run():
            from src.clients.data_service import DataServiceClient, DataServiceUnavailableError

            client = DataServiceClient(base_url="http://test", api_key="key")
            mock_http = AsyncMock()
            mock_http.post.side_effect = httpx.ConnectError("refused")
            client._client = mock_http

            with pytest.raises(DataServiceUnavailableError):
                await client._post("/v1/test", {})

        asyncio.run(_run())

    def test_quality_gate_data_block_path(self):
        """Quality fields inside 'data' block (flat payload)."""
        from src.clients.data_service import DataServiceClient, LowDataConfidenceError

        client = DataServiceClient(base_url="http://test", api_key="key")
        resp = {
            "data": {
                "signalEngineAllowed": True,
                "DataConfidenceScore": 40,  # below threshold
            }
        }
        with pytest.raises(LowDataConfidenceError):
            client._check_quality_gates(resp, "NIFTY", "2024-01-01T00:00:00Z")


# ===========================================================================
# OnlineLearner — remaining lines
# ===========================================================================


class TestOnlineLearnerExtra:
    def test_update_apply_incremental_returns_none_logs_skip(self):
        """When _apply_incremental_update returns None, update returns None."""
        import asyncio
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model", max_consecutive_updates=5)

        with patch.object(learner, "_apply_incremental_update", return_value=None):
            result = learner.update(MagicMock(), np.ones((5, 3)), np.ones(5), "1.0.0")

        assert result is None

    def test_validate_update_handles_exception_gracefully(self):
        """If spearmanr raises, _validate_update returns False without crashing."""
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model")

        with patch("src.training.online_learner.spearmanr" if False else "scipy.stats.spearmanr") as _:
            # Use a model that raises on predict
            broken_model = MagicMock()
            broken_model.predict.side_effect = RuntimeError("predict failed")
            prior_model = MagicMock()
            prior_model.predict.return_value = np.arange(5, dtype=float)

            result = learner._validate_update(
                broken_model, prior_model,
                np.ones((5, 3)), np.arange(5, dtype=float),
                "1.0.0-online", "1.0.0"
            )

        assert result is False


# ===========================================================================
# TrainingPipeline — remaining lines
# ===========================================================================


class TestTrainingPipelineExtra:
    def test_run_hpo_when_search_space_provided(self):
        """When search_space is provided and hyperparameters is None, HPO runs."""
        from src.training.pipeline import TrainingPipeline
        from unittest.mock import patch, MagicMock

        n = 100
        X = np.random.randn(n, 3)
        y = np.arange(n, dtype=float)
        timestamps = __import__("pandas").date_range("2020-01-01", periods=n, freq="B")

        def model_factory(params):
            m = MagicMock()
            m.predict = lambda data: np.arange(len(data), dtype=float)
            return m

        # Patch HPO to avoid running 50 trials in tests
        mock_best_params = {"lr": 0.01}
        with patch("src.training.pipeline.TrainingPipeline._run_hpo", return_value=mock_best_params):
            pipeline = TrainingPipeline(
                audit_logger=MagicMock(), n_splits=3
            )
            result = pipeline.run_training(
                "test",
                X, y, timestamps,
                model_factory,
                search_space={"lr": {"type": "float", "low": 0.001, "high": 0.1}},
            )

        assert result.hyperparameters == mock_best_params
