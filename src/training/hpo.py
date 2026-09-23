"""
Optuna hyperparameter optimization for ml-service2.0 training pipeline.

Provides:
  HyperparameterOptimizer — Optuna-based HPO with configurable objective,
                            minimum trial count enforcement (50+), and
                            MLflow experiment tracking integration.

Design:
- Objective metric: mean Spearman IC across all held-out validation folds.
- Minimum 50 trials enforced (settings.optuna_n_trials, default 50).
- Pruning: Optuna MedianPruner to stop unpromising trials early.
- All trials logged to MLflow when MLFLOW_TRACKING_URI is configured.

Requirements: Req 3.6, Req 3.7
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

OPTUNA_AVAILABLE = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    pass


class HyperparameterOptimizer:
    """
    Optuna-backed HPO with minimum trial enforcement and MLflow logging.

    Usage::

        optimizer = HyperparameterOptimizer(model_name="market_regime")

        def objective(trial):
            max_depth = trial.suggest_int("max_depth", 3, 8)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)
            # ... train model, return mean IC across folds
            return mean_ic

        best_params, best_value = optimizer.optimize(objective, n_trials=50)
    """

    def __init__(
        self,
        model_name: str,
        direction: str = "maximize",
        n_trials: int | None = None,
        timeout: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.model_name = model_name
        self.direction = direction
        self.n_trials = max(n_trials or settings.optuna_n_trials, 50)  # enforce minimum 50
        self.timeout = timeout
        self.n_jobs = n_jobs
        self._study: Any = None

    def optimize(
        self,
        objective: Callable,
        n_trials: int | None = None,
        sampler: Any = None,
        pruner: Any = None,
    ) -> tuple[dict[str, Any], float]:
        """
        Run HPO and return (best_params, best_value).

        Args:
            objective: Callable that takes an optuna.Trial and returns a float.
            n_trials:  Override number of trials (must be >= 50).
            sampler:   Custom Optuna sampler (default: TPESampler).
            pruner:    Custom Optuna pruner (default: MedianPruner).

        Returns:
            (best_params dict, best_value float)
        """
        effective_n_trials = max(n_trials or self.n_trials, 50)

        if not OPTUNA_AVAILABLE:
            logger.warning(
                "optuna_not_available_returning_defaults",
                model_name=self.model_name,
            )
            return {}, 0.0

        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner

        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler or TPESampler(seed=42),
            pruner=pruner or MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            study_name=f"{self.model_name}_hpo",
        )
        self._study = study

        def wrapped_objective(trial: Any) -> float:
            try:
                return objective(trial)
            except Exception as exc:
                logger.warning(
                    "optuna_trial_failed",
                    trial_number=trial.number,
                    error=str(exc),
                )
                return float("-inf") if self.direction == "maximize" else float("inf")

        study.optimize(
            wrapped_objective,
            n_trials=effective_n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_value = study.best_value

        logger.info(
            "optuna_optimization_complete",
            model_name=self.model_name,
            n_trials=len(study.trials),
            best_value=best_value,
            best_params=best_params,
        )

        return best_params, best_value

    def get_best_params(self) -> dict[str, Any]:
        """Return best params from the last optimization run."""
        if self._study is None:
            return {}
        return self._study.best_params

    def get_trial_history(self) -> list[dict[str, Any]]:
        """Return all trial results for MLflow logging."""
        if self._study is None:
            return []
        return [
            {
                "trial_number": t.number,
                "params": t.params,
                "value": t.value,
                "state": t.state.name,
            }
            for t in self._study.trials
        ]


# ── Hyperparameter search spaces per model ─────────────────────────────────────

REGIME_CLASSIFIER_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "max_depth": {"type": "int", "low": 3, "high": 8},
    "learning_rate": {"type": "float", "low": 1e-3, "high": 0.1, "log": True},
    "n_estimators": {"type": "int", "low": 100, "high": 500},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
    "min_child_weight": {"type": "int", "low": 1, "high": 20},
    "gamma": {"type": "float", "low": 0.0, "high": 0.5},
}

STOCK_RANKER_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "num_leaves": {"type": "int", "low": 31, "high": 255},
    "learning_rate": {"type": "float", "low": 1e-3, "high": 0.1, "log": True},
    "n_estimators": {"type": "int", "low": 100, "high": 1000},
    "min_child_samples": {"type": "int", "low": 5, "high": 100},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
}

RISK_PREDICTOR_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "max_depth": {"type": "int", "low": 3, "high": 7},
    "learning_rate": {"type": "float", "low": 1e-3, "high": 0.1, "log": True},
    "n_estimators": {"type": "int", "low": 100, "high": 400},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
}


def suggest_params_from_space(trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
    """
    Suggest hyperparameters from a search space definition.

    Args:
        trial:        Optuna trial object.
        search_space: Dict mapping param_name → {type, low, high, ...}.

    Returns:
        Dict of suggested param_name → value.
    """
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params
