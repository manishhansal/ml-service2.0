"""
tests/test_api_training_orchestrator.py
-----------------------------------------
FIX NEW-P1-003: /training/run now uses TrainingOrchestrator, not legacy TrainingPipeline.
FIX NEW-P3-002: /metrics endpoint present and returns Prometheus format.
"""
from __future__ import annotations

import inspect
import pytest


class TestTrainingApiUsesOrchestrator:
    """API training route must delegate to TrainingOrchestrator (NEW-P1-003)."""

    def test_training_run_endpoint_exists(self):
        from src.api.training import router
        paths = [r.path for r in router.routes]
        assert "/training/run" in paths, "POST /training/run must exist"

    def test_training_run_sync_endpoint_exists(self):
        from src.api.training import router
        paths = [r.path for r in router.routes]
        assert "/training/run/sync" in paths, "POST /training/run/sync must exist"

    def test_training_result_endpoint_exists(self):
        from src.api.training import router
        paths = [r.path for r in router.routes]
        assert "/training/result/{run_id}" in paths

    def test_api_training_source_imports_orchestrator(self):
        """The training API module must reference TrainingOrchestrator, not legacy pipeline."""
        import src.api.training as training_module
        source = inspect.getsource(training_module)
        assert "TrainingOrchestrator" in source, (
            "api/training.py must import and use TrainingOrchestrator (FIX NEW-P1-003)"
        )

    def test_api_training_does_not_use_legacy_pipeline_for_run(self):
        """The /training/run endpoint must NOT call TrainingPipeline.run_training()."""
        import src.api.training as training_module
        source = inspect.getsource(training_module._run_orchestrator_training)
        # The background task function must use Orchestrator, not legacy pipeline
        assert "TrainingOrchestrator" in source
        # The legacy pipeline's main method must NOT be called
        assert "run_training" not in source, (
            "_run_orchestrator_training must not call run_training() (legacy pipeline method)"
        )

    def test_orchestrator_config_has_baseline_ic_param(self):
        from src.api.training import OrchestratorTrainingConfig
        sig = inspect.signature(OrchestratorTrainingConfig)
        assert "baseline_ic" in sig.parameters, (
            "OrchestratorTrainingConfig must accept baseline_ic for mandate §47"
        )

    def test_orchestrator_config_has_horizon_bars(self):
        from src.api.training import OrchestratorTrainingConfig
        sig = inspect.signature(OrchestratorTrainingConfig)
        assert "horizon_bars" in sig.parameters, (
            "OrchestratorTrainingConfig must accept horizon_bars for correct Sharpe annualization"
        )

    def test_orchestrator_config_defaults(self):
        from src.api.training import OrchestratorTrainingConfig
        cfg = OrchestratorTrainingConfig(
            model_name="test",
            dataset_id="ds-1",
        )
        assert cfg.n_windows >= 5
        assert cfg.embargo_days >= 5
        assert cfg.cost_bps == 27.65  # Indian primary cost scenario
        assert cfg.horizon_bars == 5  # default label horizon
        assert cfg.baseline_ic is None  # optional — skips gate when None
        assert "logistic" in cfg.candidate_names
        assert "lightgbm" in cfg.candidate_names

    def test_feedback_endpoint_still_present(self):
        """Feedback endpoint must not have been removed during refactor."""
        from src.api.training import router
        paths = [r.path for r in router.routes]
        assert "/train/feedback" in paths
        assert "/train/feedback/summary" in paths

    def test_readiness_endpoint_still_present(self):
        from src.api.training import router
        paths = [r.path for r in router.routes]
        assert "/training/readiness" in paths


class TestPrometheusMetricsEndpoint:
    """FIX NEW-P3-002: /metrics endpoint available and functional."""

    def test_metrics_module_imports(self):
        from src.monitoring.metrics import (
            predictions_total, signals_total, no_trade_total,
            model_errors_total, inference_latency_seconds,
            data_quality_gauge, drift_score_gauge, model_health_gauge,
            alpha_decay_gauge, drawdown_gauge, drawdown_state_gauge,
            forward_paper_signals_total, forward_paper_resolved_total,
        )
        # All metrics must be importable
        assert predictions_total is not None

    def test_record_prediction_increments_counter(self):
        from src.monitoring.metrics import predictions_total, record_prediction
        before = predictions_total.labels(
            action="BUY_TEST_COUNTER", provenance="trained_model"
        )._value.get()
        record_prediction("BUY_TEST_COUNTER", "trained_model", latency_s=0.01, symbol="TEST")
        after = predictions_total.labels(
            action="BUY_TEST_COUNTER", provenance="trained_model"
        )._value.get()
        assert after == before + 1

    def test_record_no_trade_increments_counter(self):
        from src.monitoring.metrics import no_trade_total, record_no_trade
        before = no_trade_total.labels(reason_code="TEST_REASON")._value.get()
        record_no_trade("TEST_REASON")
        after = no_trade_total.labels(reason_code="TEST_REASON")._value.get()
        assert after == before + 1

    def test_update_drawdown_sets_gauge(self):
        from src.monitoring.metrics import drawdown_gauge, drawdown_state_gauge, update_drawdown
        update_drawdown(0.05, "CAUTION")
        assert abs(drawdown_gauge._value.get() - 0.05) < 1e-9
        assert drawdown_state_gauge._value.get() == 2.0  # CAUTION=2

    def test_update_model_health_sets_gauge(self):
        from src.monitoring.metrics import model_health_gauge, update_model_health
        update_model_health("test_model", 0.75)
        assert abs(model_health_gauge.labels(model_name="test_model")._value.get() - 0.75) < 1e-9

    def test_update_alpha_decay_sets_gauge(self):
        from src.monitoring.metrics import alpha_decay_gauge, update_alpha_decay
        update_alpha_decay("MOMENTUM_ALPHA", 0.35)
        val = alpha_decay_gauge.labels(alpha_name="MOMENTUM_ALPHA")._value.get()
        assert abs(val - 0.35) < 1e-9

    def test_metrics_output_returns_prometheus_format(self):
        from src.monitoring.metrics import metrics_output
        content, content_type = metrics_output()
        assert isinstance(content, bytes)
        assert len(content) > 0
        assert "text/plain" in content_type
        # Prometheus format: lines starting with # HELP or # TYPE
        text = content.decode("utf-8")
        assert "# HELP" in text
        assert "ml_service_predictions_total" in text

    def test_metrics_endpoint_in_main(self):
        """The /metrics route must be registered in the FastAPI app."""
        from src.main import app
        # app.routes contains both APIRoute objects (with .path) and
        # Include/Mount objects (without .path). Iterate safely.
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/metrics" in paths, (
            f"/metrics endpoint must be registered in main.py. Found paths: {paths}"
        )

    def test_metrics_exempt_from_api_key(self):
        """The /metrics path must be in _EXEMPT_PATHS so scrapers don't need a key."""
        from src.main import _EXEMPT_PATHS
        assert "/metrics" in _EXEMPT_PATHS, (
            "/metrics must be exempt from API key auth (Prometheus scrapers don't send keys)"
        )
