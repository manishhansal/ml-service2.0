"""
MLflow experiment tracking for ml-service2.0 training pipeline.

Provides:
  MLflowTracker — logs training runs (hyperparameters, metrics, artifacts)
                  to the configured MLFLOW_TRACKING_URI.

Requirements: Req 3.8
"""
from __future__ import annotations

from typing import Any

from src.config import settings
from src.logging_config import get_logger
from src.training.pipeline import TrainingResult

logger = get_logger(__name__)

MLFLOW_AVAILABLE = False
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    pass


class MLflowTracker:
    """
    Wraps MLflow experiment tracking for training runs.

    Usage::

        tracker = MLflowTracker()
        with tracker.start_run(model_name="market_regime", run_id=result.run_id):
            tracker.log_result(result)
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self._run = None

        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(self._tracking_uri)
            except Exception as exc:
                logger.warning("mlflow_set_tracking_uri_failed", error=str(exc))

    def log_training_result(
        self,
        result: TrainingResult,
        experiment_name: str | None = None,
    ) -> str | None:
        """
        Log a TrainingResult to MLflow.

        Returns the MLflow run ID, or None if MLflow is unavailable.
        """
        if not MLFLOW_AVAILABLE:
            logger.debug("mlflow_not_available_skipping_log")
            return None

        try:
            exp_name = experiment_name or f"ml-service2/{result.model_name}"
            mlflow.set_experiment(exp_name)

            with mlflow.start_run(run_name=result.run_id) as run:
                # Log hyperparameters
                for key, value in result.hyperparameters.items():
                    mlflow.log_param(key, value)

                # Log scalar metrics
                mlflow.log_metric("ic_mean", result.ic_mean)
                mlflow.log_metric("sharpe_net", result.sharpe_net)
                mlflow.log_metric("max_drawdown", result.max_drawdown)
                mlflow.log_metric("pbo", result.pbo)

                # Log per-fold ICs
                for i, ic in enumerate(result.ic_per_fold):
                    mlflow.log_metric(f"ic_fold_{i}", ic, step=i)

                # Log dataset hash as a tag
                mlflow.set_tag("dataset_hash", result.dataset_hash[:16])
                mlflow.set_tag("outcome", result.outcome)
                mlflow.set_tag("rejection_reason", result.rejection_reason or "")
                mlflow.set_tag("lifecycle_stage", result.lifecycle_stage.value)

                logger.info(
                    "mlflow_run_logged",
                    model_name=result.model_name,
                    mlflow_run_id=run.info.run_id,
                    outcome=result.outcome,
                )

                return run.info.run_id

        except Exception as exc:
            logger.warning("mlflow_log_failed", error=str(exc))
            return None
