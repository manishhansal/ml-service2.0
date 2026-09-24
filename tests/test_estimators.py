"""
test_estimators.py — Phase F/G baseline + advanced estimator tests.

Every estimator must fit, predict in [0,1], expose predict_proba summing to 1,
handle degenerate single-class targets, and learn a genuinely predictive
relationship (advanced models must recover a known signal).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.estimators import (
    ADVANCED_MODELS,
    BASELINE_MODELS,
    all_model_names,
    build_estimator,
)


def _learnable(n=400, seed=0):
    """Feature 0 carries a genuine signal: y = 1 when x0 > 0 (with noise)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 5))
    logits = 2.0 * X[:, 0] + 0.5 * X[:, 1]
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y


@pytest.mark.parametrize("name", list(BASELINE_MODELS) + list(ADVANCED_MODELS))
def test_estimator_fit_predict_range(name):
    X, y = _learnable()
    est = build_estimator(name)
    est.fit(X, y)
    preds = est.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


@pytest.mark.parametrize("name", list(BASELINE_MODELS) + list(ADVANCED_MODELS))
def test_predict_proba_sums_to_one(name):
    X, y = _learnable()
    est = build_estimator(name).fit(X, y)
    proba = est.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("name", ["logistic", "lightgbm", "xgboost"])
def test_advanced_models_recover_signal(name):
    """AUC on the learnable dataset must clearly beat random (0.5)."""
    from sklearn.metrics import roc_auc_score

    X, y = _learnable(n=800, seed=1)
    split = 600
    est = build_estimator(name).fit(X[:split], y[:split])
    preds = est.predict(X[split:])
    auc = roc_auc_score(y[split:], preds)
    assert auc > 0.65, f"{name} AUC={auc:.3f} did not beat random"


@pytest.mark.parametrize("name", list(BASELINE_MODELS) + list(ADVANCED_MODELS))
def test_degenerate_single_class(name):
    X = np.random.default_rng(0).normal(0, 1, (50, 5))
    y = np.ones(50)  # single class
    est = build_estimator(name).fit(X, y)
    preds = est.predict(X)
    assert np.all(np.isfinite(preds))


def test_naive_momentum_vs_mean_reversion_opposite():
    X, y = _learnable()
    mom = build_estimator("naive_momentum", feature_index=0).fit(X, y).predict(X)
    rev = build_estimator("naive_mean_reversion", feature_index=0).fit(X, y).predict(X)
    # They should be roughly mirror images around 0.5.
    np.testing.assert_allclose(mom + rev, 1.0, atol=1e-6)


def test_unknown_estimator_raises():
    with pytest.raises(ValueError):
        build_estimator("does_not_exist")


def test_all_model_names_nonempty():
    names = all_model_names()
    assert "logistic" in names
    assert "lightgbm" in names
    assert "xgboost" in names
