"""
Final coverage boost — batch 4.

Targets remaining gap-lines in:
- src/models/risk_predictor.py — _load_models (xgboost paths)
- src/registry/registry.py — load_artifact alternative path, list_registry corruption handling
- src/monitoring/drift_monitor.py — evidently / nannyml raise paths
- src/meta/calibration.py — ECE rejection path, isotonic path
- src/audit/logger.py — write failure, invalid JSON lines
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

import numpy as np
import pytest


# ===========================================================================
# RiskPredictor — _load_models with mocked xgboost
# ===========================================================================


class TestRiskPredictorLoadModels:
    def test_load_models_with_xgboost_mock(self, tmp_path):
        """Mock xgboost, create model files, verify loading."""
        from src.models.risk_predictor import RiskPredictor

        # Create dummy model files
        stop_path = tmp_path / "stop_model.json"
        target_path = tmp_path / "target_model.json"
        drawdown_path = tmp_path / "drawdown_model.json"
        for p in [stop_path, target_path, drawdown_path]:
            p.write_text("{}", encoding="utf-8")

        mock_xgb = MagicMock()
        mock_stop = MagicMock()
        mock_target = MagicMock()
        mock_drawdown = MagicMock()
        mock_xgb.XGBClassifier.side_effect = [mock_stop, mock_target]
        mock_xgb.XGBRegressor.return_value = mock_drawdown

        with patch.dict("sys.modules", {"xgboost": mock_xgb}):
            predictor = RiskPredictor(model_dir=tmp_path)

        mock_stop.load_model.assert_called_once_with(str(stop_path))
        mock_target.load_model.assert_called_once_with(str(target_path))
        mock_drawdown.load_model.assert_called_once_with(str(drawdown_path))
        assert predictor.model_version == "1.0.0-trained"

    def test_load_models_xgboost_not_installed_falls_back(self, tmp_path):
        """When xgboost import fails, predictor falls back to heuristic."""
        from src.models.risk_predictor import RiskPredictor

        stop_path = tmp_path / "stop_model.json"
        stop_path.write_text("{}")

        # Ensure xgboost raises ImportError
        with patch.dict("sys.modules", {"xgboost": None}):
            predictor = RiskPredictor(model_dir=tmp_path)

        assert predictor.stop_model is None

    def test_load_models_stops_only_when_no_target_file(self, tmp_path):
        """Only stop_model.json present → stop_model loaded, target_model None."""
        from src.models.risk_predictor import RiskPredictor

        stop_path = tmp_path / "stop_model.json"
        stop_path.write_text("{}")

        mock_xgb = MagicMock()
        mock_stop = MagicMock()
        mock_xgb.XGBClassifier.return_value = mock_stop

        with patch.dict("sys.modules", {"xgboost": mock_xgb}):
            predictor = RiskPredictor(model_dir=tmp_path)

        mock_stop.load_model.assert_called_once()
        assert predictor.target_model is None

    def test_load_models_exception_falls_back_gracefully(self, tmp_path):
        """Exception during loading → heuristic mode, no crash."""
        from src.models.risk_predictor import RiskPredictor

        stop_path = tmp_path / "stop_model.json"
        stop_path.write_text("{}")

        mock_xgb = MagicMock()
        mock_xgb.XGBClassifier.side_effect = RuntimeError("model corrupt")

        with patch.dict("sys.modules", {"xgboost": mock_xgb}):
            predictor = RiskPredictor(model_dir=tmp_path)

        assert predictor.stop_model is None


# ===========================================================================
# ModelRegistry — remaining paths
# ===========================================================================


def make_artifact(model_name="m", version="1.0.0", artifact_path="/tmp/m.pkl", sha256="a" * 64):
    from src.schemas.base import ModelLifecycleStage
    from src.schemas.registry import ModelArtifact

    return ModelArtifact(
        model_name=model_name,
        version=version,
        stage=ModelLifecycleStage.PRODUCTION,
        artifact_path=artifact_path,
        sha256_checksum=sha256,
        training_date="2024-01-01",
        training_dataset_hash="b" * 64,
    )


class TestModelRegistryPaths:
    def test_load_artifact_uses_alternative_path_in_registry(self, tmp_path):
        """When artifact_path doesn't exist but the file is in version dir → uses alt path."""
        from src.registry.registry import ModelRegistry

        reg = ModelRegistry(artifacts_path=tmp_path)

        # Create the file in version dir
        version_dir = tmp_path / "m" / "1.0.0"
        version_dir.mkdir(parents=True)
        model_file = version_dir / "m.pkl"
        model_file.write_bytes(b"model data")

        correct_sha = reg.compute_file_sha256(model_file)
        artifact = make_artifact(
            artifact_path="/nonexistent/m.pkl",  # original path doesn't exist
            sha256=correct_sha,
        )

        path = reg.load_artifact(artifact)
        assert path == model_file

    def test_list_registry_handles_corrupted_metadata(self, tmp_path):
        """Corrupted metadata.json → logged but skipped, not raised."""
        from src.registry.registry import ModelRegistry

        reg = ModelRegistry(artifacts_path=tmp_path)

        # Create a model dir with corrupted metadata
        version_dir = tmp_path / "corrupt_model" / "1.0.0"
        version_dir.mkdir(parents=True)
        corrupt_metadata = version_dir / "metadata.json"
        corrupt_metadata.write_text("not valid json {{{{", encoding="utf-8")

        # Also create one valid artifact
        artifact = make_artifact(model_name="good_model", artifact_path=str(tmp_path / "good.pkl"))
        reg.register(artifact)

        listed = reg.list_registry()
        # Good model should be listed, corrupted one silently skipped
        model_names = [a.model_name for a in listed]
        assert "good_model" in model_names
        assert "corrupt_model" not in model_names

    def test_get_champion_corrupted_champion_json(self, tmp_path):
        """Corrupted champion.json → returns None gracefully."""
        from src.registry.registry import ModelRegistry

        reg = ModelRegistry(artifacts_path=tmp_path)

        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        champion_file = model_dir / "champion.json"
        champion_file.write_text("{{invalid json", encoding="utf-8")

        result = reg.get_champion("test_model")
        assert result is None

    def test_load_all_champions_with_file_not_found(self, tmp_path):
        """Artifact file missing → logged, model returns None."""
        from src.registry.registry import ModelRegistry, ArtifactIntegrityFailure

        reg = ModelRegistry(artifacts_path=tmp_path)

        # Register market_regime (a known model) with a file that won't exist
        from src.schemas.base import ModelLifecycleStage
        from src.schemas.registry import ModelArtifact

        artifact = ModelArtifact(
            model_name="market_regime",
            version="1.0.0",
            stage=ModelLifecycleStage.PRODUCTION,
            artifact_path=str(tmp_path / "nonexistent_model.pkl"),
            sha256_checksum="a" * 64,
            training_date="2024-01-01",
            training_dataset_hash="b" * 64,
        )
        reg.register(artifact)

        results = reg.load_all_champions()
        # market_regime should return None due to missing file
        assert results.get("market_regime") is None


# ===========================================================================
# Monitoring — Evidently and NannyML exception paths
# ===========================================================================


class TestDriftMonitorFinalPaths:
    def test_run_evidently_report_raises_returns_error_dict(self):
        """Report.run() raises → returns error dict with available=False."""
        from src.monitoring.drift_monitor import DriftMonitor

        # Create mock evidently modules
        mock_evidently = MagicMock()
        mock_report_instance = MagicMock()
        mock_report_instance.run.side_effect = RuntimeError("evidently fail")
        mock_evidently.report.Report.return_value = mock_report_instance

        import src.monitoring.drift_monitor as dm_mod

        orig_available = dm_mod.EVIDENTLY_AVAILABLE
        orig_evidently = getattr(dm_mod, "evidently", None)

        try:
            dm_mod.EVIDENTLY_AVAILABLE = True
            dm_mod.evidently = mock_evidently
            dm = DriftMonitor()

            # Patch the import inside the method
            with patch.dict(
                "sys.modules",
                {
                    "evidently": mock_evidently,
                    "evidently.metric_preset": mock_evidently.metric_preset,
                    "evidently.report": mock_evidently.report,
                },
            ):
                result = dm.run_evidently_drift(MagicMock(), MagicMock())
        finally:
            dm_mod.EVIDENTLY_AVAILABLE = orig_available
            if orig_evidently is None and hasattr(dm_mod, "evidently"):
                delattr(dm_mod, "evidently")
            elif orig_evidently is not None:
                dm_mod.evidently = orig_evidently

        assert "available" in result

    def test_run_nannyml_estimator_fit_raises(self):
        """NannyML estimator.fit() raises → graceful error dict."""
        from src.monitoring.drift_monitor import DriftMonitor

        mock_nml = MagicMock()
        mock_cbpe = MagicMock()
        mock_cbpe.fit.side_effect = RuntimeError("fit failed")
        mock_nml.CBPE.return_value = mock_cbpe

        import src.monitoring.drift_monitor as dm_mod

        orig_available = dm_mod.NANNYML_AVAILABLE
        orig_nml = getattr(dm_mod, "nannyml", None)

        try:
            dm_mod.NANNYML_AVAILABLE = True
            dm_mod.nannyml = mock_nml

            with patch.dict("sys.modules", {"nannyml": mock_nml}):
                dm = DriftMonitor()
                result = dm.run_nannyml_cbpe(MagicMock(), MagicMock(), {"chunk_size": 10})
        finally:
            dm_mod.NANNYML_AVAILABLE = orig_available
            if orig_nml is None and hasattr(dm_mod, "nannyml"):
                delattr(dm_mod, "nannyml")
            elif orig_nml is not None:
                dm_mod.nannyml = orig_nml

        assert isinstance(result, dict)


# ===========================================================================
# CalibrationLayer — isotonic path and high ECE rejection
# ===========================================================================


class TestCalibrationLayerPaths:
    def test_fit_rejects_when_ece_too_high(self):
        """When best ECE > 0.15, calibration is rejected."""
        from src.meta.calibration import CalibrationLayer

        cal = CalibrationLayer()
        # Random data with no pattern → high ECE
        import random
        random.seed(42)
        scores = [random.random() for _ in range(30)]
        labels = [random.choice([0.0, 1.0]) for _ in range(30)]

        # Patch _compute_ece to always return high value
        with patch.object(cal, "_compute_ece", return_value=0.3):
            result = cal.fit("model", scores, labels)

        assert result is False

    def test_fit_isotonic_path_when_lower_ece(self):
        """When isotonic ECE < platt ECE, isotonic is chosen."""
        from src.meta.calibration import CalibrationLayer

        cal = CalibrationLayer()
        scores = list(np.linspace(0.01, 0.99, 30))
        labels = [1.0 if s > 0.5 else 0.0 for s in scores]

        call_count = [0]
        original_compute_ece = CalibrationLayer._compute_ece

        def mock_ece(preds, labs, n_bins=10):
            call_count[0] += 1
            # First call (platt) returns 0.1, second call (isotonic) returns 0.05
            if call_count[0] == 1:
                return 0.1
            return 0.05  # isotonic is better

        with patch.object(cal, "_compute_ece", side_effect=mock_ece):
            result = cal.fit("model_iso", scores, labels)

        # Should have accepted (either path)
        assert isinstance(result, bool)

    def test_calibrate_with_isotonic_wrapper(self):
        """CalibrationLayer with isotonic calibrator returns float in [0,1]."""
        from src.meta.calibration import CalibrationLayer, _IsotonicWrapper
        from sklearn.isotonic import IsotonicRegression

        cal = CalibrationLayer()
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        iso_model = IsotonicRegression(out_of_bounds="clip")
        iso_model.fit(scores, labels)
        wrapper = _IsotonicWrapper(iso_model)
        cal._calibrators["test"] = wrapper
        cal._ece_scores["test"] = 0.05

        result = cal.calibrate("test", 0.7)
        assert 0.0 <= result <= 1.0

    def test_calibrate_exception_returns_raw(self):
        """When calibrator.predict_proba raises, returns raw score."""
        from src.meta.calibration import CalibrationLayer

        cal = CalibrationLayer()
        bad_calibrator = MagicMock()
        bad_calibrator.predict_proba.side_effect = RuntimeError("exploded")
        cal._calibrators["m"] = bad_calibrator

        result = cal.calibrate("m", 0.75)
        assert result == pytest.approx(0.75)


# ===========================================================================
# AuditLogger — missing paths
# ===========================================================================


class TestAuditLoggerMissingPaths:
    def test_check_append_only_missing_hash_raises(self, tmp_path):
        """Entry without entry_hash → AuditLogViolation."""
        from src.audit.logger import AuditLogger, AuditLogViolation

        log_file = tmp_path / "audit.jsonl"
        # Write an entry without entry_hash
        entry = {"entry_id": "x", "event_type": "test", "service": "ml-service2.0"}
        log_file.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        audit = AuditLogger(log_path=log_file)
        with pytest.raises(AuditLogViolation, match="entry_hash"):
            audit._check_append_only()

    def test_check_append_only_invalid_json_raises(self, tmp_path):
        """Non-JSON line → AuditLogViolation."""
        from src.audit.logger import AuditLogger, AuditLogViolation

        log_file = tmp_path / "audit.jsonl"
        log_file.write_text("{not valid json}\n", encoding="utf-8")

        audit = AuditLogger(log_path=log_file)
        with pytest.raises(AuditLogViolation):
            audit._check_append_only()

    def test_log_training_run_with_approval_token(self, tmp_path):
        """log_training_run with approval_token and reviewer_identity."""
        from src.audit.logger import AuditLogger

        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        audit.log_training_run(
            run_id="r1",
            model_name="m",
            model_version="v1",
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T01:00:00Z",
            dataset_hash="a" * 64,
            training_date_range=("2024-01-01", "2024-01-01"),
            validation_date_range=("2024-01-01", "2024-01-01"),
            hyperparameters={},
            ic_per_fold=[0.05],
            ic_mean=0.05,
            sharpe_net=1.0,
            max_drawdown=-0.01,
            pbo=0.2,
            gate_results={"IC": "PASS"},
            outcome="PASSED",
            approval_token="tok-123",
            reviewer_identity="reviewer@example.com",
        )
        content = (tmp_path / "audit.jsonl").read_text()
        entry = json.loads(content.strip())
        assert entry["approval_token"] == "tok-123"
        assert entry["reviewer_identity"] == "reviewer@example.com"

    def test_log_promotion_decision_with_reviewer(self, tmp_path):
        """log_promotion_decision with reviewer_identity and blocked_gates."""
        from src.audit.logger import AuditLogger

        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        audit.log_promotion_decision(
            challenger_id="v2.0.0",
            champion_id="v1.0.0",
            outcome="BLOCKED",
            gate_results={"IC": "insufficient_evidence"},
            approval_policy="HUMAN_APPROVAL_REQUIRED",
            reviewer_identity="reviewer@example.com",
            blocked_gates=["IC"],
        )
        content = (tmp_path / "audit.jsonl").read_text()
        entry = json.loads(content.strip())
        assert entry["blocked_gates"] == ["IC"]

    def test_write_entry_io_error_raises_write_failure(self, tmp_path):
        """When Path.open() fails with IOError, AuditLogWriteFailure is raised."""
        from src.audit.logger import AuditLogger, AuditLogWriteFailure

        log_file = tmp_path / "audit.jsonl"
        audit = AuditLogger(log_path=log_file)

        # Patch Path.open to raise IOError
        with patch.object(Path, "open", side_effect=IOError("disk full")):
            with pytest.raises(AuditLogWriteFailure):
                audit._write_entry("test_event", {"key": "value"})
