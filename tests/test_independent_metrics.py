"""
test_independent_metrics.py — independent metric recomputation (mandate §37,§73,§74).

These tests pin the SECOND-OPINION calculator used by the certification harness
to cross-check model-reported metrics, and the adversarial ECE detector that
flags a suspicious "perfect" ECE=0.0.
"""
from __future__ import annotations

from src.analytics.independent_metrics import (
    brier_score,
    compare,
    expected_calibration_error,
    net_consistency,
    pearson_ic,
    rank_ic,
    sharpe,
    win_rate,
)


def test_rank_ic_perfect_monotone():
    preds = [0.1, 0.2, 0.3, 0.4, 0.5]
    outs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert rank_ic(preds, outs) == 1.0
    assert pearson_ic(preds, outs) == 1.0


def test_rank_ic_zero_on_noise_is_small():
    preds = [0.5, 0.5, 0.5, 0.5]
    outs = [1.0, 0.0, 1.0, 0.0]
    # constant predictions -> zero variance -> IC 0
    assert rank_ic(preds, outs) == 0.0


def test_brier_bounds():
    # perfect probabilistic predictions
    assert brier_score([1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) == 0.0
    # worst case
    assert brier_score([0.0, 1.0], [1.0, 0.0]) == 1.0


def test_ece_flags_constant_predictions_as_suspicious():
    # 200 identical predictions => spurious ECE, must be flagged (mandate §37)
    preds = [0.5] * 200
    outs = [1.0 if i % 2 == 0 else 0.0 for i in range(200)]
    res = expected_calibration_error(preds, outs)
    assert res.ece_is_suspicious
    assert res.suspicious_reason in ("CONSTANT_PREDICTIONS", "SINGLE_POPULATED_BIN")


def test_ece_flags_too_few_samples():
    res = expected_calibration_error([0.2, 0.8], [0.0, 1.0])
    assert res.ece_is_suspicious
    assert "TOO_FEW_SAMPLES" in res.suspicious_reason


def test_ece_not_suspicious_on_spread_predictions():
    # Well-spread, well-calibrated predictions across bins, enough samples.
    preds, outs = [], []
    for i in range(200):
        p = (i % 10) / 10.0 + 0.05
        preds.append(p)
        outs.append(1.0 if (i * 7) % 10 < p * 10 else 0.0)
    res = expected_calibration_error(preds, outs)
    assert not res.ece_is_suspicious
    assert res.n_nonempty_bins > 1


def test_net_consistency_detects_bad_accounting():
    gross = [0.010, 0.020, -0.005]
    cost = [0.001, 0.001, 0.001]
    good_net = [0.009, 0.019, -0.006]
    res = net_consistency(gross, cost, good_net)
    assert res.n_inconsistent == 0
    assert res.aggregate_consistent

    bad_net = [0.009, 0.030, -0.006]  # second trade inconsistent
    res2 = net_consistency(gross, cost, bad_net)
    assert res2.n_inconsistent == 1
    assert not res2.aggregate_consistent


def test_sharpe_and_win_rate():
    rets = [0.01, -0.005, 0.02, -0.01, 0.015]
    assert isinstance(sharpe(rets), float)
    assert win_rate(rets) == 0.6


def test_compare_agreement():
    assert compare(0.042, 0.0415)["agrees"]
    assert not compare(0.042, 0.0)["agrees"]
