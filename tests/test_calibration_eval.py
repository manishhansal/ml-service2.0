"""
test_calibration_eval.py — Phase I calibration metrics + gate tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.meta.calibration_eval import (
    calibration_gate,
    evaluate_calibration,
)


def test_perfectly_calibrated_low_ece():
    # Predictions equal true probabilities → low ECE.
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(float)
    m = evaluate_calibration(p, y)
    assert m.ece < 0.05
    assert m.n == n
    assert 0.0 <= m.brier <= 1.0


def test_miscalibrated_high_ece():
    # Always predict 0.9 but true rate is 0.1 → large ECE.
    n = 1000
    p = np.full(n, 0.9)
    y = np.zeros(n)
    y[:100] = 1.0  # 10% positive
    m = evaluate_calibration(p, y)
    assert m.ece > 0.5


def test_gate_passes_good_calibration():
    rng = np.random.default_rng(1)
    n = 3000
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(float)
    m = evaluate_calibration(p, y)
    passed, reason = calibration_gate(m, max_ece=0.10)
    assert passed is True
    assert reason == "CALIBRATION_OK"


def test_gate_fails_poor_calibration():
    n = 1000
    p = np.full(n, 0.9)
    y = np.zeros(n)
    m = evaluate_calibration(p, y)
    passed, reason = calibration_gate(m, max_ece=0.10)
    assert passed is False
    assert "ECE_TOO_HIGH" in reason


def test_gate_fails_insufficient_samples():
    m = evaluate_calibration([0.5, 0.6], [1, 0])
    passed, reason = calibration_gate(m)
    assert passed is False
    assert reason == "INSUFFICIENT_CALIBRATION_SAMPLES"


def test_reliability_curve_present():
    rng = np.random.default_rng(2)
    n = 2000
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(float)
    m = evaluate_calibration(p, y)
    assert len(m.reliability) > 0
    for b in m.reliability:
        assert "avg_confidence" in b and "avg_accuracy" in b
