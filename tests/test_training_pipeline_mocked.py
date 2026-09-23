"""
Mock-based tests for src/training/pipeline.py and src/training/mlflow_tracker.py.

Uses unittest.mock to avoid live services (MLflow, external data).
"""
from __future__ import annotations

import os

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.schemas.base import ModelLifecycleStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 200


def _make_timestamps(n: int = N) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _make_monotonic_dataset(n: int = N):
    """Return X, y, timestamps where predictions will equal y → perfect IC."""
    timestamps = _make_timestamps(n)
    X = np.random.randn(n, 5)
    y = np.arange(n, dtype=float)
    return X, y, timestamps


def make_model_factory():
    """Factory that produces a mock model whose predict == arange(len)."""

    def factory(params):
        model = MagicMock()
        model.predict = lambda X: np.arange(len(X), dtype=float)
        return model

    return factory


def make_mock_audit_logger():
    audit = MagicMock()
    audit.log_training_run = MagicMock()
    return audit


# ---------------------------------------------------------------------------
# TrainingPipeline — basic plumbing
# ---------------------------------------------------------------------------


class TestTrainingPipelineBasic:
    def test_len_mismatch_raises_training_aborted(self):
        from src.training.pipeline import TrainingAbortedError, TrainingPipeline

        pipeline = TrainingPipeline(audit_logger=make_mock_audit_logger(), n_splits=3)
        X = np.random.randn(50, 5)
        y = np.random.randn(30)  # length mismatch
        ts = _make_timestamps(50)
        with pytest.raises(TrainingAbortedError):
            pipeline.run_training("m", X, y, ts, make_model_factory())

    def test_timestamps_len_mismatch_raises(self):
        from src.training.pipeline import TrainingAbortedError, TrainingPipeline

        pipeline = TrainingPipeline(audit_logger=make_mock_audit_logger(), n_splits=3)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        ts = _make_timestamps(40)  # mismatch
        with pytest.raises(TrainingAbortedError):
            pipeline.run_training("m", X, y, ts, make_model_factory())

    def test_get_status_unknown_run_id_returns_none(self):
        from src.training.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(audit_logger=make_mock_audit_logger())
        assert pipeline.get_status("nonexistent-run") is None

    def test_result_has_run_id(self):
        from src.training.pipeline import TrainingPipeline

        X, y, ts = _make_monotonic_dataset()
        pipeline = TrainingPipeline(audit_logger=make_mock_audit_logger(), n_splits=3)
        result = pipeline.run_training("test_model", X, y, ts, make_model_factory())
        assert result.run_id
        assert len(result.run_id) > 0

    def test_dataset_hash_populated(self):
        from src.training.pipeline import TrainingPipeline

        X, y, ts = _make_monotonic_dataset()
        pipeline = TrainingPipeline(audit_logger=make_mock_audit_logger(), n_splits=3)
        result = pipeline.run_training("test_model", X, y, ts, make_model_factory())
        assert len(result.dataset_hash) == 64  # sha256 hex


# ---------------------------------------------------------------------------
# TrainingPipeline — acceptance gates
# ---------------------------------------------------------------------------


class TestAcceptanceGates:
    def test_happy_path_passed_lifecycle_challenger(self):
        """With monotonically increasing predict == y, IC is perfect → PASSED."""
        from src.training.pipeline import TrainingPipeline

        X, y, ts = _make_monotonic_dataset()
        pipeline = TrainingPipeline(audit_logger=make_mock_audit_logger(), n_splits=3)
        result = pipeline.run_training("test_model", X, y, ts, make_model_factory())
        assert result.outcome == "PASSED"
        assert result.lifecycle_stage == ModelLifecycleStage.CHALLENGER
        assert result.gate_results["IC"] == "PASS"
        assert result.gate_results["SHARPE"] == "PASS"
        assert result.gate_results["PBO"] == "PASS"
        assert result.rejection_reason == ""

    def test_ic_gate_rejection(self):
        """When IC < 0.02 threshold → REJECTED with IC_BELOW_THRESHOLD."""
        from src.training.pipeline import TrainingPipeline

        X, y, ts = _make_monotonic_dataset()

        # Model predicts opposite direction → poor IC
        def bad_factory(params):
            model = MagicMock()
            # Predict reversed → strongly negative IC
            model.predict = lambda data: -np.arange(len(data), dtype=float)
            return model

        pipeline = TrainingPipeline(audit_logger=make_mock_audit_logger(), n_splits=3)
        result = pipeline.run_training("test_model", X, y, ts, bad_factory)
        assert result.outcome == "REJECTED"
        # IC should be negative (well below 0.02 threshold)
        assert result.ic_mean < 0.02
        assert result.rejection_reason in ("IC_BELOW_THRESHOLD", "NEGATIVE_NET_SHARPE")

    def test_sharpe_gate_rejection(self):
        """Force negative Sharpe by patching _evaluate_folds."""
        from src.training.pipeline import TrainingPipeline

        X, y, ts = _make_monotonic_dataset()
        pipeline = TrainingPipeline(audit_logger=make_mock_audit_logger(), n_splits=3)

        # Patch _evaluate_folds to return high IC but negative sharpes
        with patch.object(
            pipeline,
            "_evaluate_folds",
            return_value=([0.1, 0.1, 0.1], [-1.0, -2.0, -1.5]),
        ):
            result = pipeline.run_training("test_model", X, y, ts, make_model_factory())

        assert result.outcome == "REJECTED"
        assert result.rejection_reason == "NEGATIVE_NET_SHARPE"
        assert result.gate_results["IC"] == "PASS"
        assert result.gate_results["SHARPE"] == "FAIL"

    def test_pbo_gate_rejection(self):
        """Force high PBO by patching _evaluate_folds."""
        from src.training.pipeline import TrainingPipeline

        X, y, ts = _make_monotonic_dataset()
        pipeline = TrainingPipeline(audit_logger=make_mock_audit_logger(), n_splits=3)

        # High IC, positive mean Sharpe, but 3 of 4 folds negative → PBO=0.75 > 0.5
        with patch.object(
            pipeline,
            "_evaluate_folds",
            return_value=([0.1, 0.1, 0.1, 0.1], [2.0, -0.5, -0.5, -0.5]),
        ):
            result = pipeline.run_training(
                "test_model", X, y, ts, make_model_factory()
            )

        # PBO = 3/4 = 0.75 > max_pbo=0.5 → should be rejected
        assert result.outcome == "REJECTED"
        assert result.rejection_reason == "HIGH_PBO"

    def test_all_gates_skipped_reported_in_gate_results(self):
        """IC FAIL means SHARPE and PBO show SKIP."""
        from src.training.pipeline import TrainingPipeline

        X, y, ts = _make_monotonic_dataset()
        pipeline = TrainingPipeline(audit_logger=make_mock_audit_logger(), n_splits=3)

        with patch.object(
            pipeline,
            "_evaluate_folds",
            return_value=([-0.5, -0.5, -0.5], [-1.0, -1.0, -1.0]),
        ):
            result = pipeline.run_training("test_model", X, y, ts, make_model_factory())

        assert result.gate_results["IC"] == "FAIL"
        assert result.gate_results["SHARPE"] == "SKIP"
        assert result.gate_results["PBO"] == "SKIP"

    def test_completed_at_is_set(self):
        from src.training.pipeline import TrainingPipeline

        X, y, ts = _make_monotonic_dataset()
        pipeline = TrainingPipeline(audit_logger=make_mock_audit_logger(), n_splits=3)
        result = pipeline.run_training("test_model", X, y, ts, make_model_factory())
        assert result.completed_at is not None

    def test_audit_logger_called(self):
        from src.training.pipeline import TrainingPipeline

        audit = make_mock_audit_logger()
        X, y, ts = _make_monotonic_dataset()
        pipeline = TrainingPipeline(audit_logger=audit, n_splits=3)
        pipeline.run_training("test_model", X, y, ts, make_model_factory())
        audit.log_training_run.assert_called_once()


# ---------------------------------------------------------------------------
# TrainingPipeline — lifecycle validation
# ---------------------------------------------------------------------------


class TestLifecycleTransition:
    def test_forward_transitions_valid(self):
        from src.training.pipeline import TrainingPipeline

        assert TrainingPipeline.validate_lifecycle_transition(
            ModelLifecycleStage.HYPOTHESIS, ModelLifecycleStage.BACKTEST
        )
        assert TrainingPipeline.validate_lifecycle_transition(
            ModelLifecycleStage.BACKTEST, ModelLifecycleStage.CHALLENGER
        )
        assert TrainingPipeline.validate_lifecycle_transition(
            ModelLifecycleStage.CHALLENGER, ModelLifecycleStage.SHADOW
        )
        assert TrainingPipeline.validate_lifecycle_transition(
            ModelLifecycleStage.SHADOW, ModelLifecycleStage.APPROVED
        )
        assert TrainingPipeline.validate_lifecycle_transition(
            ModelLifecycleStage.APPROVED, ModelLifecycleStage.PRODUCTION
        )

    def test_same_stage_is_valid(self):
        from src.training.pipeline import TrainingPipeline

        for stage in ModelLifecycleStage:
            assert TrainingPipeline.validate_lifecycle_transition(stage, stage)

    def test_backward_transitions_invalid(self):
        from src.training.pipeline import TrainingPipeline

        assert not TrainingPipeline.validate_lifecycle_transition(
            ModelLifecycleStage.PRODUCTION, ModelLifecycleStage.HYPOTHESIS
        )
        assert not TrainingPipeline.validate_lifecycle_transition(
            ModelLifecycleStage.CHALLENGER, ModelLifecycleStage.BACKTEST
        )
        assert not TrainingPipeline.validate_lifecycle_transition(
            ModelLifecycleStage.SHADOW, ModelLifecycleStage.HYPOTHESIS
        )

    def test_skip_ahead_is_valid(self):
        from src.training.pipeline import TrainingPipeline

        assert TrainingPipeline.validate_lifecycle_transition(
            ModelLifecycleStage.HYPOTHESIS, ModelLifecycleStage.PRODUCTION
        )


# ---------------------------------------------------------------------------
# Static helper methods
# ---------------------------------------------------------------------------


class TestStaticHelpers:
    def test_compute_pbo_all_positive(self):
        from src.training.pipeline import TrainingPipeline

        assert TrainingPipeline._compute_pbo([1.0, 2.0, 3.0]) == 0.0

    def test_compute_pbo_all_negative(self):
        from src.training.pipeline import TrainingPipeline

        assert TrainingPipeline._compute_pbo([-1.0, -2.0]) == 1.0

    def test_compute_pbo_half_negative(self):
        from src.training.pipeline import TrainingPipeline

        pbo = TrainingPipeline._compute_pbo([1.0, -1.0])
        assert abs(pbo - 0.5) < 1e-9

    def test_compute_pbo_empty_list(self):
        from src.training.pipeline import TrainingPipeline

        assert TrainingPipeline._compute_pbo([]) == 0.0

    def test_compute_max_drawdown_all_positive(self):
        from src.training.pipeline import TrainingPipeline

        # Monotonically increasing → drawdown should be ~0
        dd = TrainingPipeline._compute_max_drawdown([1.0, 2.0, 3.0])
        assert dd <= 0.0  # drawdown is always <= 0

    def test_compute_max_drawdown_empty(self):
        from src.training.pipeline import TrainingPipeline

        assert TrainingPipeline._compute_max_drawdown([]) == 0.0

    def test_compute_max_drawdown_with_decline(self):
        from src.training.pipeline import TrainingPipeline

        dd = TrainingPipeline._compute_max_drawdown([2.0, 1.0, 0.5, -1.0])
        assert dd < 0.0  # definite drawdown when sequence declines


# ---------------------------------------------------------------------------
# HPO path — explicit hyperparameters bypass HPO
# ---------------------------------------------------------------------------


class TestHPOPath:
    def test_explicit_hyperparameters_bypass_hpo(self):
        from src.training.pipeline import TrainingPipeline

        X, y, ts = _make_monotonic_dataset()
        pipeline = TrainingPipeline(audit_logger=make_mock_audit_logger(), n_splits=3)
        explicit_params = {"max_depth": 5, "lr": 0.01}
        result = pipeline.run_training(
            "test_model", X, y, ts, make_model_factory(), hyperparameters=explicit_params
        )
        assert result.hyperparameters == explicit_params

    def test_no_search_space_no_params_returns_empty(self):
        from src.training.pipeline import TrainingPipeline

        X, y, ts = _make_monotonic_dataset()
        pipeline = TrainingPipeline(audit_logger=make_mock_audit_logger(), n_splits=3)
        result = pipeline.run_training(
            "test_model", X, y, ts, make_model_factory()
        )
        assert result.hyperparameters == {}


# ---------------------------------------------------------------------------
# MLflowTracker
# ---------------------------------------------------------------------------


class TestMLflowTracker:
    def _make_result(self):
        from src.training.pipeline import TrainingResult
        from src.schemas.base import ModelLifecycleStage

        r = TrainingResult()
        r.model_name = "test"
        r.run_id = "abc123"
        r.hyperparameters = {"lr": 0.01}
        r.ic_mean = 0.05
        r.sharpe_net = 1.2
        r.max_drawdown = -0.03
        r.pbo = 0.2
        r.ic_per_fold = [0.04, 0.06]
        r.dataset_hash = "a" * 64
        r.outcome = "PASSED"
        r.rejection_reason = ""
        r.lifecycle_stage = ModelLifecycleStage.CHALLENGER
        return r

    def test_log_when_mlflow_unavailable_returns_none(self):
        from src.training.mlflow_tracker import MLflowTracker

        with patch("src.training.mlflow_tracker.MLFLOW_AVAILABLE", False):
            tracker = MLflowTracker()
            result = tracker.log_training_result(self._make_result())
        assert result is None

    def test_log_when_mlflow_available_logs_metrics(self):
        import src.training.mlflow_tracker as tracker_mod
        from src.training.mlflow_tracker import MLflowTracker

        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_run.info.run_id = "mlflow-run-id"
        mock_mlflow.start_run.return_value = mock_run

        # Inject the mock mlflow module attribute and flip the flag
        original_available = tracker_mod.MLFLOW_AVAILABLE
        original_mlflow = getattr(tracker_mod, "mlflow", None)
        try:
            tracker_mod.MLFLOW_AVAILABLE = True
            tracker_mod.mlflow = mock_mlflow
            tracker = MLflowTracker()
            run_id = tracker.log_training_result(self._make_result())
        finally:
            tracker_mod.MLFLOW_AVAILABLE = original_available
            if original_mlflow is None:
                if hasattr(tracker_mod, "mlflow"):
                    delattr(tracker_mod, "mlflow")
            else:
                tracker_mod.mlflow = original_mlflow

        assert run_id == "mlflow-run-id"
        mock_mlflow.log_metric.assert_called()
        mock_mlflow.log_param.assert_called()

    def test_log_exception_returns_none(self):
        """When mlflow raises, log_training_result should return None gracefully."""
        import src.training.mlflow_tracker as tracker_mod
        from src.training.mlflow_tracker import MLflowTracker

        mock_mlflow = MagicMock()
        mock_mlflow.set_experiment.side_effect = RuntimeError("mlflow error")

        original_available = tracker_mod.MLFLOW_AVAILABLE
        original_mlflow = getattr(tracker_mod, "mlflow", None)
        try:
            tracker_mod.MLFLOW_AVAILABLE = True
            tracker_mod.mlflow = mock_mlflow
            tracker = MLflowTracker()
            result = tracker.log_training_result(self._make_result())
        finally:
            tracker_mod.MLFLOW_AVAILABLE = original_available
            if original_mlflow is None:
                if hasattr(tracker_mod, "mlflow"):
                    delattr(tracker_mod, "mlflow")
            else:
                tracker_mod.mlflow = original_mlflow

        assert result is None

    def test_init_with_mlflow_unavailable(self):
        """Init when MLFLOW_AVAILABLE=False should not raise."""
        from src.training.mlflow_tracker import MLflowTracker

        with patch("src.training.mlflow_tracker.MLFLOW_AVAILABLE", False):
            tracker = MLflowTracker()
        assert tracker is not None
