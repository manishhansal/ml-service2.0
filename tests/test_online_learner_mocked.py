"""
Mock-based tests for src/training/online_learner.py.
"""
from __future__ import annotations

import os

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 20


def make_mock_xgb_model(predict_values=None):
    model = MagicMock()
    model.__class__.__name__ = "XGBClassifier"
    model.get_booster.return_value = MagicMock()
    if predict_values is None:
        predict_values = np.arange(N, dtype=float)
    model.predict.return_value = predict_values
    return model


def make_mock_lgbm_model(predict_values=None):
    model = MagicMock()
    model.__class__.__name__ = "LGBMClassifier"
    if predict_values is None:
        predict_values = np.arange(N, dtype=float)
    model.predict.return_value = predict_values
    return model


def make_mock_generic_model(predict_values=None):
    model = MagicMock()
    model.__class__.__name__ = "RandomForestClassifier"
    if predict_values is None:
        predict_values = np.arange(N, dtype=float)
    model.predict.return_value = predict_values
    return model


def make_data():
    X = np.random.randn(N, 5)
    y = np.arange(N, dtype=float)
    return X, y


# ---------------------------------------------------------------------------
# Version management
# ---------------------------------------------------------------------------


class TestVersionManagement:
    def test_generate_next_version_contains_online(self):
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model")
        new_v = learner.generate_next_version("1.0.0")
        assert "online" in new_v

    def test_generate_next_version_differs_from_prior(self):
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model")
        prior = "1.0.0"
        new_v = learner.generate_next_version(prior)
        assert new_v != prior

    def test_versions_are_unique_across_consecutive_calls(self):
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model")
        versions = set()
        current = "1.0.0"
        for _ in range(5):
            current = learner.generate_next_version(current)
            versions.add(current)
        assert len(versions) == 5

    def test_reset_consecutive_updates(self):
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model")
        learner.increment_consecutive_updates()
        learner.increment_consecutive_updates()
        learner.reset_consecutive_updates()
        assert learner._consecutive_updates == 0

    def test_requires_full_retrain_true_at_max(self):
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model", max_consecutive_updates=3)
        for _ in range(3):
            learner.increment_consecutive_updates()
        assert learner.requires_full_retrain() is True

    def test_requires_full_retrain_false_below_max(self):
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model", max_consecutive_updates=5)
        learner.increment_consecutive_updates()
        assert learner.requires_full_retrain() is False


# ---------------------------------------------------------------------------
# update() — guard conditions
# ---------------------------------------------------------------------------


class TestUpdateGuards:
    def test_update_returns_none_when_full_retrain_required(self):
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model", max_consecutive_updates=0)
        # max=0 means requires_full_retrain() is immediately True
        X, y = make_data()
        result = learner.update(make_mock_xgb_model(), X, y, "1.0.0")
        assert result is None

    def test_update_returns_none_when_model_is_none(self):
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model")
        X, y = make_data()
        result = learner.update(None, X, y, "1.0.0")
        assert result is None


# ---------------------------------------------------------------------------
# _apply_incremental_update()
# ---------------------------------------------------------------------------


class TestApplyIncrementalUpdate:
    def test_xgb_model_calls_get_booster_update(self):
        """XGBoost path — patch the xgboost import so it works without the package."""
        from src.training.online_learner import OnlineLearner

        mock_xgb = MagicMock()
        mock_dmatrix_instance = MagicMock()
        mock_xgb.DMatrix.return_value = mock_dmatrix_instance

        learner = OnlineLearner("test_model")
        model = make_mock_xgb_model()
        X, y = make_data()

        with patch.dict("sys.modules", {"xgboost": mock_xgb}):
            updated = learner._apply_incremental_update(model, X, y)

        assert updated is not None
        model.get_booster.assert_called_once()
        model.get_booster.return_value.update.assert_called_once()

    def test_lgbm_model_calls_fit_with_init_model(self):
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model")
        model = make_mock_lgbm_model()
        X, y = make_data()

        updated = learner._apply_incremental_update(model, X, y)
        assert updated is not None
        model.fit.assert_called_once()
        call_kwargs = model.fit.call_args
        assert call_kwargs[1].get("init_model") == model or (
            len(call_kwargs[0]) >= 3 and call_kwargs[0][2] == model
        )

    def test_generic_model_calls_fit(self):
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model")
        model = make_mock_generic_model()
        X, y = make_data()

        updated = learner._apply_incremental_update(model, X, y)
        assert updated is not None
        model.fit.assert_called_once()

    def test_exception_in_update_returns_none(self):
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model")
        model = MagicMock()
        model.__class__.__name__ = "XGBClassifier"
        model.get_booster.side_effect = RuntimeError("boom")
        X, y = make_data()

        # Patch xgboost so the import doesn't fail before we hit get_booster
        mock_xgb = MagicMock()
        with patch.dict("sys.modules", {"xgboost": mock_xgb}):
            result = learner._apply_incremental_update(model, X, y)

        assert result is None


# ---------------------------------------------------------------------------
# update() — with validation
# ---------------------------------------------------------------------------


class TestUpdateWithValidation:
    def test_update_accepted_when_new_ic_above_prior(self):
        """Updated model predicts same as prior (perfect correlation) → accepted."""
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model", max_consecutive_updates=5)
        X, y = make_data()
        val_y = np.arange(N, dtype=float)

        prior_model = make_mock_xgb_model(predict_values=val_y)
        updated_model_mock = make_mock_xgb_model(predict_values=val_y + 0.1)

        # Patch _apply_incremental_update to return updated_model_mock
        with patch.object(
            learner, "_apply_incremental_update", return_value=updated_model_mock
        ):
            result = learner.update(
                prior_model,
                X,
                y,
                "1.0.0",
                validation_X=X,
                validation_y=val_y,
            )

        # updated IC >= prior IC (both perfect) → accepted
        assert result is not None
        updated_m, new_version = result
        assert "online" in new_version

    def test_update_rejected_when_new_ic_below_prior(self):
        """Updated model predicts worse than prior → rejected."""
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model", max_consecutive_updates=5)
        X, y = make_data()
        val_y = np.arange(N, dtype=float)

        # Prior model has perfect IC
        prior_model = make_mock_xgb_model(predict_values=val_y)
        # Updated model has inverted predictions → negative IC
        updated_model_mock = make_mock_xgb_model(predict_values=-val_y)

        with patch.object(
            learner, "_apply_incremental_update", return_value=updated_model_mock
        ):
            result = learner.update(
                prior_model,
                X,
                y,
                "1.0.0",
                validation_X=X,
                validation_y=val_y,
            )

        assert result is None

    def test_update_without_validation_data_increments_counter(self):
        """No validation data → update accepted, counter incremented."""
        from src.training.online_learner import OnlineLearner

        learner = OnlineLearner("test_model", max_consecutive_updates=5)
        X, y = make_data()
        model = make_mock_xgb_model()

        mock_xgb = MagicMock()
        with patch.dict("sys.modules", {"xgboost": mock_xgb}):
            result = learner.update(model, X, y, "1.0.0")

        assert result is not None
        assert learner._consecutive_updates == 1

    def test_update_with_audit_logger_calls_log_online_update(self):
        """Accepted update should call audit.log_online_update."""
        from src.training.online_learner import OnlineLearner

        audit = MagicMock()
        learner = OnlineLearner("test_model", max_consecutive_updates=5, audit_logger=audit)
        X, y = make_data()
        val_y = np.arange(N, dtype=float)

        prior_model = make_mock_xgb_model(predict_values=val_y)
        updated_model_mock = make_mock_xgb_model(predict_values=val_y)

        with patch.object(
            learner, "_apply_incremental_update", return_value=updated_model_mock
        ):
            learner.update(
                prior_model, X, y, "1.0.0",
                validation_X=X, validation_y=val_y
            )

        audit.log_online_update.assert_called_once()

    def test_update_with_audit_logger_calls_rejected_on_ic_regression(self):
        """IC regression should call audit.log_online_update_rejected."""
        from src.training.online_learner import OnlineLearner

        audit = MagicMock()
        learner = OnlineLearner("test_model", max_consecutive_updates=5, audit_logger=audit)
        X, y = make_data()
        val_y = np.arange(N, dtype=float)

        prior_model = make_mock_xgb_model(predict_values=val_y)
        updated_model_mock = make_mock_xgb_model(predict_values=-val_y)

        with patch.object(
            learner, "_apply_incremental_update", return_value=updated_model_mock
        ):
            learner.update(
                prior_model, X, y, "1.0.0",
                validation_X=X, validation_y=val_y
            )

        audit.log_online_update_rejected.assert_called_once()
