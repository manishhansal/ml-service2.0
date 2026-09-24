"""
src.models.estimators — baseline + advanced estimators with a uniform interface.

Every estimator exposes:
    fit(X, y)                 -> None
    predict(X)                -> np.ndarray  (score / signal in [0,1] or real)
    predict_proba(X)          -> np.ndarray  (n, 2) when classification-capable

Baselines (Phase F, mandatory):
    LogisticBaseline          — L2 logistic regression (classification)
    RidgeBaseline             — ridge regression (regression → sigmoid to [0,1])
    NaiveMomentum             — signal = sign of a momentum feature
    NaiveMeanReversion        — signal = -sign of a momentum feature

Advanced (Phase G):
    LightGBMModel             — gradient-boosted trees (classification)
    XGBoostModel              — gradient-boosted trees (classification)
    CatBoostModel             — optional; only if catboost installed

Model selection is done by OOS evidence in the training pipeline, NOT by
defaulting to the most complex model. A baseline that ties the advanced model
wins by parsimony.

Requirements: Phase 18 (baselines first), Phase 19 (real ML), Phase 65 (ablation).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class BaseEstimator:
    """Uniform estimator interface used by the training pipeline."""

    name: str = "base"
    is_classifier: bool = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.clip(self.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


# ── Baselines ───────────────────────────────────────────────────────────────


class LogisticBaseline(BaseEstimator):
    name = "logistic"

    def __init__(self, C: float = 1.0, **kwargs: Any) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        self._model = LogisticRegression(C=C, max_iter=1000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticBaseline:
        Xs = self._scaler.fit_transform(X)
        yb = (np.asarray(y) > 0).astype(int)
        # Guard against single-class targets.
        if len(np.unique(yb)) < 2:
            self._degenerate = float(yb.mean())
        else:
            self._degenerate = None
            self._model.fit(Xs, yb)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "_degenerate", None) is not None:
            return np.full(len(X), self._degenerate)
        Xs = self._scaler.transform(X)
        return self._model.predict_proba(Xs)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.predict(X)
        return np.column_stack([1.0 - p, p])


class RidgeBaseline(BaseEstimator):
    name = "ridge"
    is_classifier = False

    def __init__(self, alpha: float = 1.0, **kwargs: Any) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        self._model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeBaseline:
        Xs = self._scaler.fit_transform(X)
        self._model.fit(Xs, np.asarray(y, dtype=float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scaler.transform(X)
        raw = self._model.predict(Xs)
        # Map to [0,1] via sigmoid so it fits the common probabilistic interface.
        return _sigmoid(raw - np.mean(raw))


class NaiveMomentum(BaseEstimator):
    """Signal = the momentum feature at ``feature_index`` mapped to [0,1]."""

    name = "naive_momentum"

    def __init__(self, feature_index: int = 1, **kwargs: Any) -> None:
        self._idx = feature_index

    def fit(self, X: np.ndarray, y: np.ndarray) -> NaiveMomentum:
        col = X[:, self._idx]
        self._mean = float(np.nanmean(col))
        self._std = float(np.nanstd(col)) or 1.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = (X[:, self._idx] - self._mean) / self._std
        return _sigmoid(z)


class NaiveMeanReversion(BaseEstimator):
    """Signal = inverse of the momentum feature (fade the move)."""

    name = "naive_mean_reversion"

    def __init__(self, feature_index: int = 1, **kwargs: Any) -> None:
        self._idx = feature_index

    def fit(self, X: np.ndarray, y: np.ndarray) -> NaiveMeanReversion:
        col = X[:, self._idx]
        self._mean = float(np.nanmean(col))
        self._std = float(np.nanstd(col)) or 1.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = (X[:, self._idx] - self._mean) / self._std
        return _sigmoid(-z)


# ── Advanced tree models ──────────────────────────────────────────────────────


class LightGBMModel(BaseEstimator):
    name = "lightgbm"

    def __init__(self, **params: Any) -> None:
        import lightgbm as lgb

        defaults = dict(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_lambda=1.0, random_state=42, verbosity=-1,
        )
        defaults.update(params)
        self._model = lgb.LGBMClassifier(**defaults)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LightGBMModel:
        yb = (np.asarray(y) > 0).astype(int)
        if len(np.unique(yb)) < 2:
            self._degenerate = float(yb.mean())
        else:
            self._degenerate = None
            self._model.fit(X, yb)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "_degenerate", None) is not None:
            return np.full(len(X), self._degenerate)
        return self._model.predict_proba(X)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.predict(X)
        return np.column_stack([1.0 - p, p])

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return getattr(self._model, "feature_importances_", None)


class XGBoostModel(BaseEstimator):
    name = "xgboost"

    def __init__(self, **params: Any) -> None:
        import xgboost as xgb

        defaults = dict(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        defaults.update(params)
        self._model = xgb.XGBClassifier(**defaults)

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostModel:
        yb = (np.asarray(y) > 0).astype(int)
        if len(np.unique(yb)) < 2:
            self._degenerate = float(yb.mean())
        else:
            self._degenerate = None
            self._model.fit(X, yb)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "_degenerate", None) is not None:
            return np.full(len(X), self._degenerate)
        return self._model.predict_proba(X)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.predict(X)
        return np.column_stack([1.0 - p, p])

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return getattr(self._model, "feature_importances_", None)


def catboost_available() -> bool:
    try:
        import catboost  # noqa: F401
        return True
    except ImportError:
        return False


class CatBoostModel(BaseEstimator):
    name = "catboost"

    def __init__(self, **params: Any) -> None:
        from catboost import CatBoostClassifier

        defaults = dict(
            iterations=200, depth=4, learning_rate=0.05,
            l2_leaf_reg=3.0, random_seed=42, verbose=False,
        )
        defaults.update(params)
        self._model = CatBoostClassifier(**defaults)

    def fit(self, X: np.ndarray, y: np.ndarray) -> CatBoostModel:
        yb = (np.asarray(y) > 0).astype(int)
        if len(np.unique(yb)) < 2:
            self._degenerate = float(yb.mean())
        else:
            self._degenerate = None
            self._model.fit(X, yb)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "_degenerate", None) is not None:
            return np.full(len(X), self._degenerate)
        return self._model.predict_proba(X)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.predict(X)
        return np.column_stack([1.0 - p, p])


# ── Registry / factory ────────────────────────────────────────────────────────

BASELINE_MODELS = {
    "logistic": LogisticBaseline,
    "ridge": RidgeBaseline,
    "naive_momentum": NaiveMomentum,
    "naive_mean_reversion": NaiveMeanReversion,
}

ADVANCED_MODELS = {
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
}
if catboost_available():
    ADVANCED_MODELS["catboost"] = CatBoostModel


def build_estimator(name: str, **params: Any) -> BaseEstimator:
    """Instantiate an estimator by name from the baseline or advanced registries."""
    registry = {**BASELINE_MODELS, **ADVANCED_MODELS}
    if name not in registry:
        raise ValueError(f"Unknown estimator {name!r}. Available: {sorted(registry)}")
    return registry[name](**params)


def all_model_names() -> list[str]:
    return list(BASELINE_MODELS) + list(ADVANCED_MODELS)
