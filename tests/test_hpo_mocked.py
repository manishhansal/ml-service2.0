"""
Mock-based tests for src/training/hpo.py.

Uses real Optuna (in deps) for optimization tests.
"""
from __future__ import annotations

import os

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from unittest.mock import patch


class TestHyperparameterOptimizerNoOptuna:
    def test_returns_empty_params_when_optuna_unavailable(self):
        from src.training.hpo import HyperparameterOptimizer

        with patch("src.training.hpo.OPTUNA_AVAILABLE", False):
            optimizer = HyperparameterOptimizer("test_model")
            best_params, best_value = optimizer.optimize(lambda t: 0.5, n_trials=50)

        assert best_params == {}
        assert best_value == 0.0

    def test_get_best_params_before_optimize_returns_empty(self):
        from src.training.hpo import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer("test_model")
        assert optimizer.get_best_params() == {}

    def test_get_trial_history_before_optimize_returns_empty(self):
        from src.training.hpo import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer("test_model")
        assert optimizer.get_trial_history() == []


class TestHyperparameterOptimizerWithOptuna:
    """Real Optuna tests (no mocking of Optuna itself)."""

    def test_simple_optimization_returns_params_and_value(self):
        from src.training.hpo import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer("test_model")
        best_params, best_value = optimizer.optimize(
            lambda trial: trial.suggest_float("x", 0.0, 1.0),
            n_trials=50,
        )
        assert "x" in best_params
        assert 0.0 <= best_params["x"] <= 1.0
        assert isinstance(best_value, float)

    def test_minimum_50_trials_enforced_even_if_5_passed(self):
        """Passing n_trials=5 should still run >=50 trials."""
        from src.training.hpo import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer("test_model")
        optimizer.optimize(
            lambda trial: trial.suggest_float("x", 0.0, 1.0),
            n_trials=5,  # below minimum
        )
        # The study should have >= 50 trials
        history = optimizer.get_trial_history()
        assert len(history) >= 50

    def test_constructor_enforces_minimum_n_trials(self):
        """n_trials=1 at construction should be clamped to 50."""
        from src.training.hpo import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer("test_model", n_trials=1)
        assert optimizer.n_trials >= 50

    def test_get_best_params_after_optimize(self):
        from src.training.hpo import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer("test_model")
        optimizer.optimize(
            lambda trial: trial.suggest_float("lr", 1e-4, 0.1, log=True),
            n_trials=50,
        )
        params = optimizer.get_best_params()
        assert "lr" in params

    def test_get_trial_history_returns_list_of_dicts(self):
        from src.training.hpo import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer("test_model")
        optimizer.optimize(
            lambda trial: trial.suggest_int("n", 1, 10),
            n_trials=50,
        )
        history = optimizer.get_trial_history()
        assert isinstance(history, list)
        assert len(history) >= 50
        for entry in history[:3]:
            assert "trial_number" in entry
            assert "params" in entry
            assert "value" in entry
            assert "state" in entry

    def test_failing_objective_still_returns_best_params(self):
        """When objective raises, should return float('-inf') and continue."""
        from src.training.hpo import HyperparameterOptimizer

        call_count = {"n": 0}

        def flaky_objective(trial):
            x = trial.suggest_float("x", 0.0, 1.0)
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                raise ValueError("simulated failure")
            return x

        optimizer = HyperparameterOptimizer("test_model")
        best_params, best_value = optimizer.optimize(flaky_objective, n_trials=50)
        # Should still return params from successful trials
        assert isinstance(best_params, dict)


class TestSuggestParamsFromSpace:
    def test_int_param(self):
        from src.training.hpo import suggest_params_from_space
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        space = {"depth": {"type": "int", "low": 2, "high": 10}}
        params = suggest_params_from_space(trial, space)
        assert "depth" in params
        assert 2 <= params["depth"] <= 10

    def test_float_param(self):
        from src.training.hpo import suggest_params_from_space
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        space = {"lr": {"type": "float", "low": 0.001, "high": 0.1}}
        params = suggest_params_from_space(trial, space)
        assert "lr" in params
        assert 0.001 <= params["lr"] <= 0.1

    def test_float_log_param(self):
        from src.training.hpo import suggest_params_from_space
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        space = {"alpha": {"type": "float", "low": 1e-5, "high": 1.0, "log": True}}
        params = suggest_params_from_space(trial, space)
        assert "alpha" in params
        assert 1e-5 <= params["alpha"] <= 1.0

    def test_categorical_param(self):
        from src.training.hpo import suggest_params_from_space
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        space = {"criterion": {"type": "categorical", "choices": ["gini", "entropy"]}}
        params = suggest_params_from_space(trial, space)
        assert params["criterion"] in ("gini", "entropy")

    def test_mixed_search_space(self):
        from src.training.hpo import suggest_params_from_space
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        space = {
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
            "booster": {"type": "categorical", "choices": ["gbtree", "dart"]},
        }
        params = suggest_params_from_space(trial, space)
        assert len(params) == 3

    def test_predefined_regime_classifier_space(self):
        """Smoke-test the REGIME_CLASSIFIER_SEARCH_SPACE constant."""
        from src.training.hpo import REGIME_CLASSIFIER_SEARCH_SPACE, suggest_params_from_space
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = suggest_params_from_space(trial, REGIME_CLASSIFIER_SEARCH_SPACE)
        assert "max_depth" in params
        assert "learning_rate" in params
