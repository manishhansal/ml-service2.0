"""
test_walk_forward.py — Phase J WalkForwardValidator tests (P0-007).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.models.estimators import build_estimator
from src.training.walk_forward import WalkForwardValidator

UTC = timezone.utc


def _dataset(n=1000, seed=0, signal_strength=2.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 5))
    # Forward return depends on feature 0 → a learnable, PIT-safe signal.
    returns = signal_strength * 0.01 * X[:, 0] + rng.normal(0, 0.01, n)
    y_label = (returns > 0).astype(int)
    ts = pd.DatetimeIndex(
        [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    )
    return X, y_label, returns, ts


def test_requires_min_5_windows():
    with pytest.raises(ValueError):
        WalkForwardValidator(n_windows=3)


def test_produces_5_windows():
    X, y, r, ts = _dataset()
    wf = WalkForwardValidator(n_windows=5, embargo_days=5, cost_bps=10)
    report = wf.validate(X, y, r, ts, lambda: build_estimator("logistic"))
    assert report.n_windows == 5
    assert len(report.windows) == 5


def test_windows_are_chronological_and_oos():
    X, y, r, ts = _dataset()
    wf = WalkForwardValidator(n_windows=5, embargo_days=5)
    report = wf.validate(X, y, r, ts, lambda: build_estimator("logistic"))
    for i in range(1, len(report.windows)):
        prev = report.windows[i - 1]
        cur = report.windows[i]
        # Test windows advance in time.
        assert cur.test_start >= prev.test_start
        # Train ends before test starts.
        assert cur.train_end <= cur.test_start


def test_learnable_signal_yields_positive_ic():
    X, y, r, ts = _dataset(signal_strength=3.0, seed=1)
    wf = WalkForwardValidator(n_windows=5, embargo_days=5, cost_bps=10)
    report = wf.validate(X, y, r, ts, lambda: build_estimator("lightgbm"))
    # With a genuine signal, mean OOS IC should be clearly positive.
    assert report.ic_mean > 0.02
    assert report.positive_ic_fraction >= 0.6


def test_random_signal_yields_near_zero_ic():
    # Pure noise: returns independent of features → IC ~ 0, must NOT be forced positive.
    rng = np.random.default_rng(5)
    n = 1000
    X = rng.normal(0, 1, (n, 5))
    returns = rng.normal(0, 0.01, n)  # no dependence on X
    y = (returns > 0).astype(int)
    ts = pd.DatetimeIndex([datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)])
    wf = WalkForwardValidator(n_windows=5, embargo_days=5)
    report = wf.validate(X, y, returns, ts, lambda: build_estimator("logistic"))
    assert abs(report.ic_mean) < 0.05  # honest near-zero result


def test_report_serializable():
    X, y, r, ts = _dataset()
    wf = WalkForwardValidator(n_windows=5)
    report = wf.validate(X, y, r, ts, lambda: build_estimator("logistic"))
    d = report.to_dict()
    assert "ic_mean" in d and "windows" in d
    assert len(d["windows"]) == report.n_windows
