"""
test_drift_reference.py — Phase P drift monitoring with reference distributions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.performance_drift import PerformanceDriftTracker
from src.monitoring.reference import (
    DriftAction,
    DriftGate,
    ReferenceDistribution,
    ReferenceDistributionStore,
)


def _frame(n=500, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "f1": rng.normal(0 + shift, 1, n),
        "f2": rng.normal(5 + shift, 2, n),
    })


def test_reference_store_roundtrip(tmp_path):
    store = ReferenceDistributionStore(tmp_path)
    ref = ReferenceDistributionStore.from_training(
        "market_regime", "1.0.0", _frame(), np.random.default_rng(0).uniform(0, 1, 500)
    )
    store.save(ref)
    loaded = store.load("market_regime", "1.0.0")
    assert loaded is not None
    assert loaded.model_name == "market_regime"
    assert "f1" in loaded.feature_samples


def test_no_drift_low_severity(tmp_path):
    ref = ReferenceDistributionStore.from_training(
        "m", "1", _frame(seed=1), np.random.default_rng(1).uniform(0, 1, 500)
    )
    gate = DriftGate()
    # Same distribution → LOW / MONITOR.
    result = gate.evaluate(ref, _frame(seed=2))
    assert result.severity == "LOW"
    assert result.action == DriftAction.MONITOR.value


def test_large_shift_triggers_action(tmp_path):
    ref = ReferenceDistributionStore.from_training(
        "m", "1", _frame(seed=1), np.random.default_rng(1).uniform(0, 1, 500)
    )
    gate = DriftGate()
    # Large distribution shift → HIGH or CRITICAL.
    result = gate.evaluate(ref, _frame(shift=5.0, seed=3))
    assert result.severity in ("HIGH", "CRITICAL")
    assert result.action in (DriftAction.TRAIN_CHALLENGER.value, DriftAction.BLOCK.value)
    assert len(result.drifted_features) > 0


def test_prediction_drift_detected():
    ref = ReferenceDistribution(
        model_name="m", version="1",
        feature_samples={"f1": np.random.default_rng(0).normal(0, 1, 500).tolist()},
        prediction_samples=np.random.default_rng(0).uniform(0.4, 0.6, 500).tolist(),
    )
    gate = DriftGate()
    frame = pd.DataFrame({"f1": np.random.default_rng(1).normal(0, 1, 300)})
    # Predictions shifted to a very different range → prediction PSI high.
    shifted_preds = np.random.default_rng(2).uniform(0.0, 0.05, 300)
    result = gate.evaluate(ref, frame, current_predictions=shifted_preds)
    assert result.prediction_psi > 0.25


def test_severity_action_mapping():
    gate = DriftGate()
    assert gate._action("LOW") == "MONITOR"
    assert gate._action("MEDIUM") == "ALERT"
    assert gate._action("HIGH") == "TRAIN_CHALLENGER"
    assert gate._action("CRITICAL") == "BLOCK"


# ── Performance drift ────────────────────────────────────────────────────────


def test_performance_tracker_no_degradation():
    tracker = PerformanceDriftTracker(baseline_ic=0.05, window=200, min_samples=30)
    rng = np.random.default_rng(0)
    # Predictions that correlate with outcomes → maintains IC.
    for _ in range(150):
        p = rng.uniform(0, 1)
        o = 1.0 if rng.uniform(0, 1) < p else 0.0
        tracker.record(p, o)
    status = tracker.assess()
    assert status.n == 150
    assert status.degraded is False


def test_performance_tracker_detects_degradation():
    tracker = PerformanceDriftTracker(baseline_ic=0.10, window=200, min_samples=30)
    rng = np.random.default_rng(1)
    # Random predictions unrelated to outcomes → IC collapses.
    for _ in range(150):
        tracker.record(rng.uniform(0, 1), float(rng.integers(0, 2)))
    status = tracker.assess()
    assert status.degraded is True
    assert status.action == "TRIGGER_CHALLENGER"


def test_performance_tracker_insufficient_data():
    tracker = PerformanceDriftTracker(baseline_ic=0.05, min_samples=30)
    tracker.record(0.6, 1.0)
    status = tracker.assess()
    assert status.action == "INSUFFICIENT_DATA"
