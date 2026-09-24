"""
test_cpcv.py — Phase 30 Combinatorial Purged CV tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import comb

import numpy as np
import pandas as pd
import pytest

from src.models.estimators import build_estimator
from src.training.cpcv import CombinatorialPurgedCV

UTC = timezone.utc


def _dataset(n=900, seed=0, signal=2.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 5))
    returns = signal * 0.01 * X[:, 0] + rng.normal(0, 0.01, n)
    y = (returns > 0).astype(int)
    ts = pd.DatetimeIndex([datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)])
    return X, y, returns, ts


def test_validates_params():
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=3)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=6, k_test_groups=6)


def test_number_of_paths_matches_combinations():
    X, y, r, ts = _dataset()
    cpcv = CombinatorialPurgedCV(n_groups=6, k_test_groups=2, embargo=0)
    report = cpcv.run(X, y, r, ts, lambda: build_estimator("logistic"))
    assert report.n_paths == comb(6, 2)  # 15 paths


def test_signal_yields_positive_mean_ic_and_low_pbo():
    X, y, r, ts = _dataset(signal=3.0, seed=1)
    cpcv = CombinatorialPurgedCV(n_groups=6, k_test_groups=2, embargo=5)
    report = cpcv.run(X, y, r, ts, lambda: build_estimator("lightgbm"))
    assert report.ic_mean > 0.02
    assert report.pbo < 0.5  # most paths OOS-positive


def test_noise_yields_high_pbo():
    rng = np.random.default_rng(7)
    n = 900
    X = rng.normal(0, 1, (n, 5))
    returns = rng.normal(0, 0.01, n)
    y = (returns > 0).astype(int)
    ts = pd.DatetimeIndex([datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)])
    cpcv = CombinatorialPurgedCV(n_groups=6, k_test_groups=2, embargo=5)
    report = cpcv.run(X, y, returns, ts, lambda: build_estimator("logistic"))
    # Pure noise → PBO should be near/above 0.5 (no durable edge).
    assert report.pbo >= 0.35


def test_report_has_distribution():
    X, y, r, ts = _dataset()
    cpcv = CombinatorialPurgedCV(n_groups=6, k_test_groups=2)
    report = cpcv.run(X, y, r, ts, lambda: build_estimator("logistic"))
    d = report.to_dict()
    assert "ic_5pct" in d and "ic_95pct" in d
    assert len(d["path_ics"]) == report.n_paths
