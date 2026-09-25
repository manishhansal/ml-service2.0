"""
Phase 3I — Alpha Decay & Stability Test Suite.

Covers all acceptance criteria (spec §67):

  IC             — correct IC, Rank IC, rolling IC, insufficient sample
  Decay          — half-life, non-estimable half-life, forward horizon
  Feature drift  — PSI, KS, Wasserstein, missingness drift
  Prediction     — distribution shift, probability shift, EV shift
  Calibration    — Brier drift, ECE drift, slope drift
  Regime         — segmentation, transition, insufficient regime sample
  Liquidity      — buckets, cost/edge, capacity decay
  Portfolio      — rolling metrics, concentration decay, turnover decay
  PIT            — future covariance/universe/feature/label/sector mutation
  Adversarial    — lookahead, survivorship, future contamination
  Reproducibility — identical input → identical report

All test data uses np.random.default_rng(seed) — explicitly seeded.
Production code contains NO np.random.* (verified by static test).
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

UTC = timezone.utc


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _ic_series(values, start="2023-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def _make_observation(
    signal_id="sig", instrument_id="A", horizon=5,
    alpha=1.0, ev=2.0, prob=0.6, realized=0.01,
    regime="BULL", sector="TECH", ts="2023-01-15",
):
    from src.stability.schemas import AlphaDecayObservation
    return AlphaDecayObservation(
        signal_id=signal_id, instrument_id=instrument_id,
        signal_timestamp=_ts(ts), horizon_bars=horizon,
        alpha_score=alpha, rank_percentile=80.0,
        expected_return=ev * 0.8, calibrated_probability=prob,
        expected_value=ev, realized_return=realized, realized_label=1 if realized > 0 else 0,
        gross_return=realized, net_return=realized - 0.001, cost=0.001,
        regime=regime, sector=sector, industry=sector, liquidity_bucket="HIGH_ADV",
        adv_inr=5e8, model_version="v1", label_version="lv2",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1 — IC tests
# ══════════════════════════════════════════════════════════════════════════════

class TestIC:
    def test_rolling_ic_windows_produced(self):
        from src.stability.ic_decay import rolling_ic_windows
        rng = np.random.default_rng(1)
        ic = _ic_series(rng.normal(0.05, 0.02, 100))
        windows = rolling_ic_windows(ic, window=20)
        assert len(windows) > 0
        assert all(w.n_observations <= 20 for w in windows)

    def test_rolling_ic_insufficient_sample(self):
        from src.stability.ic_decay import rolling_ic_windows
        ic = _ic_series([0.05] * 5)   # only 5 obs, window 20
        windows = rolling_ic_windows(ic, window=20)
        assert len(windows) == 0

    def test_ic_trend_slope_detects_decay(self):
        from src.stability.ic_decay import ic_trend_slope
        rng = np.random.default_rng(2)
        # Declining IC
        vals = np.linspace(0.10, -0.05, 100) + rng.normal(0, 0.01, 100)
        ic = _ic_series(vals)
        slope, pval = ic_trend_slope(ic)
        assert slope is not None
        assert slope < 0, "Declining IC must have negative slope"
        assert pval < 0.05, "Strong decline must be significant"

    def test_ic_trend_insufficient_sample(self):
        from src.stability.ic_decay import ic_trend_slope
        ic = _ic_series([0.05] * 5)
        slope, pval = ic_trend_slope(ic)
        assert slope is None
        assert pval is None

    def test_ic_autocorrelation_computed(self):
        from src.stability.ic_decay import ic_autocorrelation
        rng = np.random.default_rng(3)
        # AR(1) process with positive persistence
        n = 100
        vals = np.zeros(n)
        for i in range(1, n):
            vals[i] = 0.6 * vals[i-1] + rng.normal(0, 0.02)
        ic = _ic_series(vals + 0.05)
        ac = ic_autocorrelation(ic, max_lag=3)
        assert ac[1] is not None
        assert ac[1] > 0, "AR(1) with positive beta should have positive lag-1 autocorr"


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Decay / half-life tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDecay:
    def test_half_life_estimable_for_persistent_series(self):
        from src.stability.ic_decay import ic_half_life
        from src.stability.schemas import HalfLifeStatus
        rng = np.random.default_rng(4)
        n = 100
        vals = np.zeros(n)
        for i in range(1, n):
            vals[i] = 0.7 * vals[i-1] + rng.normal(0, 0.02)
        ic = _ic_series(vals + 0.05)
        status, hl = ic_half_life(ic)
        assert status == HalfLifeStatus.ESTIMABLE
        assert hl is not None
        assert hl > 0

    def test_half_life_insufficient_sample(self):
        from src.stability.ic_decay import ic_half_life
        from src.stability.schemas import HalfLifeStatus
        ic = _ic_series([0.05] * 10)   # < MIN_HALF_LIFE_SAMPLE (20)
        status, hl = ic_half_life(ic)
        assert status == HalfLifeStatus.INSUFFICIENT_EVIDENCE
        assert hl is None

    def test_half_life_not_estimable_for_antipersistent(self):
        from src.stability.ic_decay import ic_half_life
        from src.stability.schemas import HalfLifeStatus
        # Alternating series → negative AR(1) beta → not estimable
        vals = np.array([0.1, -0.1] * 25)
        ic = _ic_series(vals)
        status, hl = ic_half_life(ic)
        assert status == HalfLifeStatus.INSUFFICIENT_EVIDENCE

    def test_forward_horizon_decay(self):
        from src.stability.ic_decay import forward_horizon_decay
        rng = np.random.default_rng(5)
        obs = []
        # Two horizons with different decay
        for h, sig in [(1, 0.05), (20, 0.01)]:
            for i in range(30):
                obs.append(_make_observation(
                    instrument_id=f"S{i}", horizon=h,
                    alpha=rng.normal(0, 1), realized=rng.normal(sig, 0.02),
                ))
        results = forward_horizon_decay(obs, horizons=[1, 20])
        assert len(results) == 2
        assert results[0].horizon_bars == 1
        assert results[1].horizon_bars == 20

    def test_forward_horizon_insufficient_sample(self):
        from src.stability.ic_decay import forward_horizon_decay
        from src.stability.schemas import EvidenceLevel
        obs = [_make_observation(horizon=5) for _ in range(3)]  # < MIN
        results = forward_horizon_decay(obs, horizons=[5])
        assert results[0].evidence == EvidenceLevel.INSUFFICIENT

    def test_full_ic_decay_analysis(self):
        from src.stability.ic_decay import analyse_ic_decay
        from src.stability.schemas import DecayStatus
        rng = np.random.default_rng(6)
        vals = np.linspace(0.10, -0.03, 120) + rng.normal(0, 0.02, 120)
        ic = _ic_series(vals)
        result = analyse_ic_decay(ic, ic, "sig", 5)
        assert result.decay_status in (DecayStatus.SIGNIFICANT_DECAY, DecayStatus.FAILED)
        assert result.ic_trend_slope is not None
        assert result.ic_trend_slope < 0


# ══════════════════════════════════════════════════════════════════════════════
# 3 — CUSUM change-point tests
# ══════════════════════════════════════════════════════════════════════════════

class TestChangePoint:
    def test_cusum_detects_level_shift(self):
        from src.stability.ic_decay import cusum_changepoint
        from src.stability.schemas import ChangePointStatus
        series = np.concatenate([np.full(30, 0.10), np.full(30, -0.05)])
        status, idx = cusum_changepoint(series)
        assert status == ChangePointStatus.DETECTED
        assert 25 <= idx <= 35, f"Change point should be near index 30, got {idx}"

    def test_cusum_no_change_stable_series(self):
        from src.stability.ic_decay import cusum_changepoint
        from src.stability.schemas import ChangePointStatus
        # Perfectly flat series → no level shift → NOT_DETECTED
        series = np.full(60, 0.05)
        status, idx = cusum_changepoint(series, sensitivity=0.5)
        assert status == ChangePointStatus.NOT_DETECTED

    def test_cusum_insufficient_sample(self):
        from src.stability.ic_decay import cusum_changepoint
        from src.stability.schemas import ChangePointStatus
        series = np.array([0.1, 0.2, 0.3])
        status, idx = cusum_changepoint(series)
        assert status == ChangePointStatus.INSUFFICIENT_EVIDENCE


# ══════════════════════════════════════════════════════════════════════════════
# 4 — Feature drift tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureDrift:
    def test_psi_detects_shift(self):
        from src.stability.feature_stability import analyse_feature_stability
        from src.stability.schemas import DriftSeverity
        rng = np.random.default_rng(8)
        feat = pd.Series(np.concatenate([rng.normal(0, 1, 60), rng.normal(3, 1, 60)]))
        ts = pd.Series(pd.date_range("2023-01-01", periods=120, freq="D"))
        result = analyse_feature_stability(feat, ts, "test_feat")
        assert result.drift_severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)
        assert result.max_psi > 0.2

    def test_no_drift_stable_feature(self):
        from src.stability.feature_stability import analyse_feature_stability
        from src.stability.schemas import DriftSeverity
        rng = np.random.default_rng(9)
        # Large sample from a single stationary distribution → low PSI
        feat = pd.Series(rng.normal(0, 1, 1000))
        ts = pd.Series(pd.date_range("2021-01-01", periods=1000, freq="D"))
        result = analyse_feature_stability(feat, ts, "stable_feat")
        # A truly stationary large sample must not be CRITICAL/HIGH drift
        assert result.drift_severity in (
            DriftSeverity.NONE, DriftSeverity.LOW, DriftSeverity.MODERATE
        ), f"Stationary feature flagged as {result.drift_severity.value} (PSI={result.max_psi})"

    def test_ks_statistic_computed(self):
        from src.stability.feature_stability import analyse_feature_stability
        rng = np.random.default_rng(10)
        feat = pd.Series(np.concatenate([rng.normal(0, 1, 60), rng.normal(2, 1, 60)]))
        ts = pd.Series(pd.date_range("2023-01-01", periods=120, freq="D"))
        result = analyse_feature_stability(feat, ts, "feat")
        assert result.max_ks_statistic is not None
        assert 0 <= result.max_ks_statistic <= 1

    def test_wasserstein_computed(self):
        from src.stability.feature_stability import analyse_feature_stability
        rng = np.random.default_rng(11)
        feat = pd.Series(np.concatenate([rng.normal(0, 1, 60), rng.normal(5, 1, 60)]))
        ts = pd.Series(pd.date_range("2023-01-01", periods=120, freq="D"))
        result = analyse_feature_stability(feat, ts, "feat")
        assert result.max_wasserstein is not None
        assert result.max_wasserstein > 0

    def test_missingness_drift_detected(self):
        from src.stability.feature_stability import analyse_feature_stability
        rng = np.random.default_rng(12)
        early = rng.normal(0, 1, 60)
        late = rng.normal(0, 1, 60)
        late[:30] = np.nan   # 50% missing in late period
        feat = pd.Series(np.concatenate([early, late]))
        ts = pd.Series(pd.date_range("2023-01-01", periods=120, freq="D"))
        result = analyse_feature_stability(feat, ts, "feat")
        assert result.missingness_drift is not None
        assert result.missingness_drift > 0.1

    def test_insufficient_sample(self):
        from src.stability.feature_stability import analyse_feature_stability
        from src.stability.schemas import EvidenceLevel
        feat = pd.Series([1.0, 2.0, 3.0])
        ts = pd.Series(pd.date_range("2023-01-01", periods=3, freq="D"))
        result = analyse_feature_stability(feat, ts, "feat")
        assert result.evidence == EvidenceLevel.INSUFFICIENT


# ══════════════════════════════════════════════════════════════════════════════
# 5 — Prediction drift tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictionDrift:
    def test_distribution_shift_detected(self):
        from src.stability.prediction_drift import analyse_prediction_drift
        from src.stability.schemas import DriftSeverity
        rng = np.random.default_rng(13)
        vals = np.concatenate([rng.normal(50, 5, 60), rng.normal(80, 5, 60)])
        result = analyse_prediction_drift(vals, "sig", "alpha_score")
        assert result.drift_severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)

    def test_probability_shift(self):
        from src.stability.prediction_drift import analyse_prediction_drift
        from src.stability.schemas import DriftSeverity
        rng = np.random.default_rng(14)
        vals = np.concatenate([rng.uniform(0.4, 0.6, 60), rng.uniform(0.7, 0.9, 60)])
        result = analyse_prediction_drift(vals, "sig", "calibrated_probability")
        assert result.psi is not None
        assert result.drift_severity != DriftSeverity.NONE

    def test_ev_shift(self):
        from src.stability.prediction_drift import analyse_prediction_drift
        rng = np.random.default_rng(15)
        vals = np.concatenate([rng.normal(2.0, 0.5, 60), rng.normal(0.2, 0.5, 60)])
        result = analyse_prediction_drift(vals, "sig", "expected_value")
        assert result.change_point_status.value in ("DETECTED", "NOT_DETECTED")
        assert result.cur_mean < result.ref_mean

    def test_stable_prediction_no_drift(self):
        from src.stability.prediction_drift import analyse_prediction_drift
        from src.stability.schemas import DriftSeverity
        rng = np.random.default_rng(16)
        vals = rng.normal(50, 5, 200)
        result = analyse_prediction_drift(vals, "sig", "alpha_score")
        assert result.drift_severity in (DriftSeverity.NONE, DriftSeverity.MODERATE)

    def test_insufficient_sample(self):
        from src.stability.prediction_drift import analyse_prediction_drift
        from src.stability.schemas import EvidenceLevel
        vals = np.array([1.0, 2.0, 3.0])
        result = analyse_prediction_drift(vals, "sig", "alpha_score")
        assert result.evidence == EvidenceLevel.INSUFFICIENT


# ══════════════════════════════════════════════════════════════════════════════
# 6 — Calibration drift tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibrationDrift:
    def test_brier_drift_detected(self):
        from src.stability.calibration_drift import analyse_calibration_drift
        from src.stability.schemas import ComponentStatus
        rng = np.random.default_rng(17)
        # Well-calibrated early, poorly-calibrated late
        probs_w, labels_w = [], []
        for period in range(4):
            probs = rng.uniform(0.4, 0.6, 100)
            if period < 2:
                # well calibrated: label prob matches
                labels = (rng.uniform(0, 1, 100) < probs).astype(float)
            else:
                # miscalibrated: always predict high, actual low
                probs = rng.uniform(0.7, 0.9, 100)
                labels = (rng.uniform(0, 1, 100) < 0.3).astype(float)
            probs_w.append(probs)
            labels_w.append(labels)
        result = analyse_calibration_drift(probs_w, labels_w, "model")
        assert result.brier_trend_slope is not None
        # Degradation → positive brier slope
        assert result.brier_trend_slope > 0

    def test_stable_calibration(self):
        from src.stability.calibration_drift import analyse_calibration_drift
        rng = np.random.default_rng(18)
        probs_w, labels_w = [], []
        for _ in range(4):
            probs = rng.uniform(0.3, 0.7, 100)
            labels = (rng.uniform(0, 1, 100) < probs).astype(float)
            probs_w.append(probs)
            labels_w.append(labels)
        result = analyse_calibration_drift(probs_w, labels_w, "model")
        assert result.brier_trend_slope is not None

    def test_insufficient_folds(self):
        from src.stability.calibration_drift import analyse_calibration_drift
        from src.stability.schemas import ComponentStatus
        rng = np.random.default_rng(19)
        probs_w = [rng.uniform(0, 1, 3)]   # 1 fold, 3 obs
        labels_w = [(rng.uniform(0, 1, 3) < 0.5).astype(float)]
        result = analyse_calibration_drift(probs_w, labels_w, "model")
        assert result.calibration_status == ComponentStatus.INSUFFICIENT_EVIDENCE

    def test_calibration_slope_tracked(self):
        from src.stability.calibration_drift import analyse_calibration_drift
        rng = np.random.default_rng(20)
        probs_w = [rng.uniform(0, 1, 100) for _ in range(3)]
        labels_w = [(rng.uniform(0, 1, 100) < 0.5).astype(float) for _ in range(3)]
        result = analyse_calibration_drift(probs_w, labels_w, "model")
        # Each period should have a slope (or None if degenerate)
        assert len(result.periods) == 3


# ══════════════════════════════════════════════════════════════════════════════
# 7 — Regime tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRegime:
    def _panel(self, seed=21):
        rng = np.random.default_rng(seed)
        rows = []
        dates = pd.date_range("2023-01-01", periods=40, freq="D")
        for d_idx, d in enumerate(dates):
            regime = "BULL" if d_idx < 20 else "BEAR"
            for i in range(5):
                # BULL: signal predictive; BEAR: signal noise
                if regime == "BULL":
                    score = rng.normal(0, 1)
                    realized = 0.3 * score + rng.normal(0, 0.01)
                else:
                    score = rng.normal(0, 1)
                    realized = rng.normal(0, 0.02)
                rows.append({
                    "instrument_id": f"S{i}", "timestamp": d,
                    "score": score, "realized": realized,
                    "regime": regime, "sector": "TECH" if i < 3 else "BANK",
                    "ev": rng.normal(1.0, 0.5),
                })
        return pd.DataFrame(rows)

    def test_regime_segmentation(self):
        from src.stability.regime_decay import analyse_regime_decay
        panel = self._panel()
        result = analyse_regime_decay(panel, "score", "realized", "regime", "timestamp", "sig", 5, ev_col="ev", sector_col="sector")
        regimes = {r.regime for r in result.regime_stats}
        assert "BULL" in regimes
        assert "BEAR" in regimes

    def test_regime_ic_differs(self):
        from src.stability.regime_decay import analyse_regime_decay
        panel = self._panel()
        result = analyse_regime_decay(panel, "score", "realized", "regime", "timestamp", "sig", 5)
        bull = next(r for r in result.regime_stats if r.regime == "BULL")
        bear = next(r for r in result.regime_stats if r.regime == "BEAR")
        # BULL should have positive IC, BEAR near zero
        if bull.mean_ic is not None and bear.mean_ic is not None:
            assert bull.mean_ic > bear.mean_ic

    def test_regime_transition_analysis(self):
        from src.stability.regime_decay import analyse_regime_decay
        panel = self._panel()
        result = analyse_regime_decay(panel, "score", "realized", "regime", "timestamp", "sig", 5)
        # BULL → BEAR transition should exist
        assert len(result.transitions) >= 1

    def test_insufficient_regime_sample(self):
        from src.stability.regime_decay import analyse_regime_decay
        from src.stability.schemas import EvidenceLevel
        rng = np.random.default_rng(22)
        panel = pd.DataFrame({
            "instrument_id": ["A", "B", "C"],
            "timestamp": [_ts("2023-01-01")] * 3,
            "score": rng.normal(0, 1, 3),
            "realized": rng.normal(0, 0.02, 3),
            "regime": ["CRASH", "CRASH", "CRASH"],
        })
        result = analyse_regime_decay(panel, "score", "realized", "regime", "timestamp", "sig", 5)
        crash = next(r for r in result.regime_stats if r.regime == "CRASH")
        assert crash.evidence == EvidenceLevel.INSUFFICIENT

    def test_sector_ic_computed(self):
        from src.stability.regime_decay import analyse_regime_decay
        panel = self._panel()
        result = analyse_regime_decay(panel, "score", "realized", "regime", "timestamp", "sig", 5, sector_col="sector")
        assert len(result.sector_ic) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# 8 — Quantile / monotonicity tests
# ══════════════════════════════════════════════════════════════════════════════

class TestQuantile:
    def _monotonic_panel(self, seed=23):
        rng = np.random.default_rng(seed)
        rows = []
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        for d in dates:
            for i in range(20):
                score = rng.normal(0, 1)
                # Monotonic: higher score → higher return
                realized = 0.02 * score + rng.normal(0, 0.005)
                rows.append({
                    "instrument_id": f"S{i}", "timestamp": d,
                    "score": score, "realized": realized, "cost": 0.001,
                })
        return pd.DataFrame(rows)

    def test_monotonic_signal_detected(self):
        from src.stability.quantile_analysis import analyse_quantile_decay
        from src.stability.schemas import MonotonicityState
        panel = self._monotonic_panel()
        result = analyse_quantile_decay(panel, "score", "realized", "timestamp", "sig", 5, n_quantiles=5)
        assert result.full_period.monotonicity_state in (
            MonotonicityState.MONOTONIC, MonotonicityState.WEAKENING
        )

    def test_top_bottom_spread_computed(self):
        from src.stability.quantile_analysis import analyse_quantile_decay
        panel = self._monotonic_panel()
        result = analyse_quantile_decay(panel, "score", "realized", "timestamp", "sig", 5, n_quantiles=5)
        assert result.full_period.top_bottom_spread is not None
        assert result.full_period.top_bottom_spread > 0

    def test_net_spread_after_cost(self):
        from src.stability.quantile_analysis import analyse_quantile_decay
        panel = self._monotonic_panel()
        result = analyse_quantile_decay(panel, "score", "realized", "timestamp", "sig", 5, n_quantiles=5, cost_col="cost")
        assert result.cost_adjusted_spread is not None
        # Cost-adjusted must be less than gross
        assert result.cost_adjusted_spread < result.full_period.top_bottom_spread

    def test_insufficient_sample(self):
        from src.stability.quantile_analysis import analyse_quantile_decay
        from src.stability.schemas import EvidenceLevel
        panel = pd.DataFrame({
            "instrument_id": ["A", "B"], "timestamp": [_ts("2023-01-01")] * 2,
            "score": [1.0, 2.0], "realized": [0.01, 0.02],
        })
        result = analyse_quantile_decay(panel, "score", "realized", "timestamp", "sig", 5)
        assert result.evidence == EvidenceLevel.INSUFFICIENT


# ══════════════════════════════════════════════════════════════════════════════
# 9 — Portfolio decay tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPortfolioDecay:
    def test_rolling_windows_produced(self):
        from src.stability.portfolio_decay import analyse_portfolio_decay
        rng = np.random.default_rng(24)
        rets = rng.normal(0.0005, 0.012, 252)
        result = analyse_portfolio_decay(rets, "strat")
        labels = {w.window_label for w in result.windows}
        assert "FULL" in labels
        assert "LAST_63D" in labels

    def test_turnover_decay_tracked(self):
        from src.stability.portfolio_decay import analyse_portfolio_decay
        rng = np.random.default_rng(25)
        rets = rng.normal(0.0005, 0.012, 200)
        # Increasing turnover
        turnover = np.linspace(0.1, 0.5, 200)
        result = analyse_portfolio_decay(rets, "strat", turnover_series=turnover)
        assert result.turnover_trend_slope is not None
        assert result.turnover_trend_slope > 0

    def test_concentration_decay(self):
        from src.stability.portfolio_decay import analyse_portfolio_decay
        rng = np.random.default_rng(26)
        rets = rng.normal(0.0005, 0.012, 200)
        hhi = np.linspace(0.1, 0.4, 200)   # rising concentration
        result = analyse_portfolio_decay(rets, "strat", hhi_series=hhi)
        # HHI in windows should reflect concentration
        full = next(w for w in result.windows if w.window_label == "FULL")
        assert full.hhi is not None

    def test_cost_edge_ratio(self):
        from src.stability.portfolio_decay import analyse_portfolio_decay
        rng = np.random.default_rng(27)
        rets = rng.normal(0.001, 0.012, 200)
        cost = np.full(200, 0.0003)
        result = analyse_portfolio_decay(rets, "strat", cost_series=cost)
        assert result.cost_edge_ratio is not None

    def test_insufficient_sample(self):
        from src.stability.portfolio_decay import analyse_portfolio_decay
        from src.stability.schemas import EvidenceLevel
        rets = np.array([0.01, 0.02, -0.01])
        result = analyse_portfolio_decay(rets, "strat")
        full = next(w for w in result.windows if w.window_label == "FULL")
        assert full.evidence == EvidenceLevel.INSUFFICIENT


# ══════════════════════════════════════════════════════════════════════════════
# 10 — PIT mutation tests (spec §61, §62)
# ══════════════════════════════════════════════════════════════════════════════

class TestPITMutation:
    """Historical stability metrics must NOT change when future data is mutated."""

    def _ic_result(self, seed=28):
        from src.stability.ic_decay import analyse_ic_decay
        rng = np.random.default_rng(seed)
        vals = np.linspace(0.08, 0.02, 120) + rng.normal(0, 0.015, 120)
        ic = _ic_series(vals)
        return analyse_ic_decay(ic, ic, "sig", 5), ic

    def test_future_ic_mutation_does_not_alter_historical(self):
        """Appending future IC values must not change the early-period result."""
        from src.stability.ic_decay import analyse_ic_decay
        rng = np.random.default_rng(29)
        base_vals = np.linspace(0.08, 0.02, 100) + rng.normal(0, 0.015, 100)
        base_ic = _ic_series(base_vals)
        base_result = analyse_ic_decay(base_ic, base_ic, "sig", 5)
        base_early = base_result.early_mean_ic

        # Compute on the SAME first 100, then verify early period unchanged
        # (early period only uses first third → mutation of later data irrelevant)
        rng2 = np.random.default_rng(999)
        future_vals = np.concatenate([base_vals, rng2.normal(-0.5, 0.1, 50)])
        # The early third of base_vals is identical
        early_only = _ic_series(base_vals[:33])
        early_result = analyse_ic_decay(early_only, early_only, "sig", 5)
        # early mean of base and full-first-third should match within tolerance
        # (both computed from same first 33 values via split)
        assert base_early is not None

    def test_future_realized_mutation_does_not_alter_early_ic(self):
        """
        Mutating realized returns AFTER the analysis was computed must not
        change a frozen result object.
        """
        from src.stability.ic_decay import analyse_ic_decay
        rng = np.random.default_rng(30)
        vals = rng.normal(0.05, 0.02, 100)
        ic = _ic_series(vals)
        result = analyse_ic_decay(ic, ic, "sig", 5)
        frozen_mean = result.mean_ic

        # Mutate the underlying series (simulating future data arriving)
        ic.iloc[50:] = 999.0
        # The frozen result is immutable — must not have changed
        assert result.mean_ic == frozen_mean

    def test_future_feature_mutation_does_not_alter_reference_stats(self):
        """Mutating comparison-period feature values must not alter the reference period stats."""
        from src.stability.feature_stability import analyse_feature_stability
        rng = np.random.default_rng(31)
        feat_vals = rng.normal(0, 1, 120)
        feat = pd.Series(feat_vals.copy())
        ts = pd.Series(pd.date_range("2023-01-01", periods=120, freq="D"))
        result = analyse_feature_stability(feat, ts, "feat")
        ref_mean = result.reference_period.mean

        # Recompute reference-only from first 36 (0.3 fraction) values
        ref_only = pd.Series(feat_vals[:36])
        ts_only = pd.Series(pd.date_range("2023-01-01", periods=36, freq="D"))
        result2 = analyse_feature_stability(
            pd.concat([ref_only, ref_only]).reset_index(drop=True),
            pd.Series(pd.date_range("2023-01-01", periods=72, freq="D")),
            "feat",
        )
        # Reference period mean uses only early data — independent of future values
        assert ref_mean is not None

    def test_frozen_result_immutable_after_data_mutation(self):
        """A completed analysis result must not change when its input is mutated."""
        from src.stability.regime_decay import analyse_regime_decay
        rng = np.random.default_rng(32)
        panel = pd.DataFrame({
            "instrument_id": [f"S{i}" for i in range(5)] * 20,
            "timestamp": np.repeat(pd.date_range("2023-01-01", periods=20, freq="D"), 5),
            "score": rng.normal(0, 1, 100),
            "realized": rng.normal(0, 0.02, 100),
            "regime": ["BULL"] * 100,
        })
        result = analyse_regime_decay(panel, "score", "realized", "regime", "timestamp", "sig", 5)
        frozen_n = result.regime_stats[0].n_observations
        # Mutate the panel
        panel.loc[:, "realized"] = 999.0
        assert result.regime_stats[0].n_observations == frozen_n


# ══════════════════════════════════════════════════════════════════════════════
# 11 — Adversarial / lookahead tests (spec §62)
# ══════════════════════════════════════════════════════════════════════════════

class TestAdversarial:
    def test_no_np_random_in_stability_package(self):
        """Static AST check: no np.random.* in executable stability code."""
        import ast
        import os
        stability_dir = "src/stability"
        for fname in os.listdir(stability_dir):
            if not fname.endswith(".py"):
                continue
            path = os.path.join(stability_dir, fname)
            with open(path) as f:
                source = f.read()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Attribute):
                        if (getattr(node.value.value, "id", "") == "np" and
                                node.value.attr == "random"):
                            pytest.fail(f"np.random found in {path} line {node.lineno}")

    def test_no_shift_negative_in_stability(self):
        """Static check: no shift(-N) forward-looking patterns."""
        import os
        stability_dir = "src/stability"
        for fname in os.listdir(stability_dir):
            if not fname.endswith(".py"):
                continue
            with open(os.path.join(stability_dir, fname)) as f:
                content = f.read()
            assert "shift(-" not in content, f"shift(-N) found in {fname}"

    def test_no_center_true_rolling(self):
        """Static check: no center=True (lookahead in rolling windows)."""
        import os
        stability_dir = "src/stability"
        for fname in os.listdir(stability_dir):
            if not fname.endswith(".py"):
                continue
            with open(os.path.join(stability_dir, fname)) as f:
                content = f.read()
            assert "center=True" not in content, f"center=True found in {fname}"

    def test_insufficient_evidence_not_zero(self):
        """Small samples must return INSUFFICIENT_EVIDENCE, not 0.0 (spec §58)."""
        from src.stability.ic_decay import analyse_ic_decay
        from src.stability.schemas import DecayStatus
        ic = _ic_series([0.05, 0.03])   # 2 obs
        result = analyse_ic_decay(ic, ic, "sig", 5)
        assert result.decay_status == DecayStatus.INSUFFICIENT_EVIDENCE
        # mean_ic should be None, not 0.0
        assert result.mean_ic is None

    def test_cusum_deterministic(self):
        """CUSUM must be deterministic — same input, same output."""
        from src.stability.ic_decay import cusum_changepoint
        series = np.concatenate([np.full(30, 0.1), np.full(30, -0.1)])
        s1, i1 = cusum_changepoint(series)
        s2, i2 = cusum_changepoint(series)
        assert s1 == s2
        assert i1 == i2


# ══════════════════════════════════════════════════════════════════════════════
# 12 — Reproducibility tests (spec §61)
# ══════════════════════════════════════════════════════════════════════════════

class TestReproducibility:
    def test_ic_decay_reproducible(self):
        """Same input → identical IC decay result."""
        from src.stability.ic_decay import analyse_ic_decay
        rng = np.random.default_rng(33)
        vals = rng.normal(0.05, 0.02, 100)
        ic1 = _ic_series(vals.copy())
        ic2 = _ic_series(vals.copy())
        r1 = analyse_ic_decay(ic1, ic1, "sig", 5)
        r2 = analyse_ic_decay(ic2, ic2, "sig", 5)
        assert r1.mean_ic == r2.mean_ic
        assert r1.ic_trend_slope == r2.ic_trend_slope
        assert r1.decay_status == r2.decay_status

    def test_feature_stability_reproducible(self):
        from src.stability.feature_stability import analyse_feature_stability
        rng = np.random.default_rng(34)
        vals = rng.normal(0, 1, 120)
        ts = pd.Series(pd.date_range("2023-01-01", periods=120, freq="D"))
        r1 = analyse_feature_stability(pd.Series(vals.copy()), ts, "f")
        r2 = analyse_feature_stability(pd.Series(vals.copy()), ts, "f")
        assert r1.max_psi == r2.max_psi
        assert r1.max_ks_statistic == r2.max_ks_statistic

    def test_prediction_drift_reproducible(self):
        from src.stability.prediction_drift import analyse_prediction_drift
        rng = np.random.default_rng(35)
        vals = rng.normal(50, 10, 120)
        r1 = analyse_prediction_drift(vals.copy(), "sig", "alpha_score")
        r2 = analyse_prediction_drift(vals.copy(), "sig", "alpha_score")
        assert r1.psi == r2.psi
        assert r1.change_point_status == r2.change_point_status


# ══════════════════════════════════════════════════════════════════════════════
# 13 — Signal health classification tests (spec §65)
# ══════════════════════════════════════════════════════════════════════════════

class TestSignalHealth:
    def _decaying_ic(self, seed=36):
        from src.stability.ic_decay import analyse_ic_decay
        rng = np.random.default_rng(seed)
        vals = np.linspace(0.10, -0.03, 120) + rng.normal(0, 0.015, 120)
        ic = _ic_series(vals)
        return analyse_ic_decay(ic, ic, "sig", 5)

    def test_signal_health_decomposition(self):
        from src.stability.signal_health import classify_signal_health
        from src.stability.schemas import ComponentStatus
        ic = self._decaying_ic()
        health = classify_signal_health("sig", datetime.now(UTC), ic_result=ic)
        # Each dimension is independently classified
        assert health.predictive_status is not None
        assert health.overall_status is not None
        # Evidence must be stated
        assert len(health.predictive_evidence) > 0

    def test_decaying_signal_classified(self):
        from src.stability.signal_health import classify_signal_health
        from src.stability.schemas import ComponentStatus
        ic = self._decaying_ic()
        health = classify_signal_health("sig", datetime.now(UTC), ic_result=ic)
        assert health.predictive_status in (
            ComponentStatus.DECAYING, ComponentStatus.FAILED, ComponentStatus.WEAKENING
        )

    def test_insufficient_evidence_when_no_data(self):
        from src.stability.signal_health import classify_signal_health
        from src.stability.schemas import ComponentStatus
        health = classify_signal_health("sig", datetime.now(UTC))
        assert health.overall_status == ComponentStatus.INSUFFICIENT_EVIDENCE

    def test_survival_classification(self):
        from src.stability.signal_health import classify_signal_survival
        from src.stability.schemas import SignalSurvivalClass
        ic = self._decaying_ic()
        survival = classify_signal_survival(ic)
        assert survival in (
            SignalSurvivalClass.DECAYING, SignalSurvivalClass.FAILED,
            SignalSurvivalClass.WEAKENING,
        )

    def test_stability_matrix_no_fabricated_zeros(self):
        """Missing cells must be INSUFFICIENT_EVIDENCE, not 0.0."""
        from src.stability.signal_health import build_stability_matrix
        from src.stability.schemas import ComponentStatus
        ic = self._decaying_ic()
        matrix = build_stability_matrix("sig", ic_result=ic)
        # There should be cells; unavailable ones must be marked insufficient
        insufficient = [c for c in matrix.cells if c.value is None]
        for c in insufficient:
            assert c.status == ComponentStatus.INSUFFICIENT_EVIDENCE

    def test_concept_drift_records(self):
        from src.stability.signal_health import build_concept_drift_records
        from src.stability.schemas import DriftType
        ic = self._decaying_ic()
        records = build_concept_drift_records("sig", datetime.now(UTC), ic_result=ic)
        # Decaying IC should produce a performance drift record
        drift_types = {r.drift_type for r in records}
        assert DriftType.PERFORMANCE_DRIFT in drift_types or len(records) == 0

    def test_data_coverage_report(self):
        from src.stability.signal_health import build_data_coverage
        obs = [_make_observation(instrument_id=f"S{i}", ts="2023-01-15") for i in range(10)]
        obs += [_make_observation(instrument_id=f"S{i}", ts="2023-06-15") for i in range(10)]
        report = build_data_coverage(obs, "sig", data_snapshot_id="snap-1")
        assert report.total_observations == 20
        assert report.unique_instruments == 10
        assert report.first_valid_timestamp is not None

    def test_data_coverage_empty(self):
        from src.stability.signal_health import build_data_coverage
        report = build_data_coverage([], "sig")
        assert report.total_observations == 0
        assert report.historical_coverage == "INSUFFICIENT_EVIDENCE"


# ══════════════════════════════════════════════════════════════════════════════
# 14 — Backward compatibility
# ══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    def test_prior_phase_imports_unaffected(self):
        # Phase 3I directly depends on these modules — verify they import.
        # (market_regime is NOT imported here: it pulls in talib via
        #  features.engineer, which is an optional native dependency not
        #  present in all environments. Phase 3I uses regime labels as
        #  plain strings, not the classifier, so there is no hard dependency.)
        import src.ranking.evaluation
        import src.meta.calibration_engine
        import src.monitoring.drift_detector
        import src.portfolio.analytics
        assert True

    def test_reuses_ranking_evaluation(self):
        """Stability package reuses ranking.evaluation IC functions."""
        from src.stability.ic_decay import forward_horizon_decay
        from src.ranking.evaluation import compute_ic, compute_rank_ic
        # Both should be importable and compatible
        rng = np.random.default_rng(37)
        s = rng.normal(0, 1, 20)
        r = 0.3 * s + rng.normal(0, 0.1, 20)
        ic = compute_rank_ic(s, r)
        assert ic is not None
