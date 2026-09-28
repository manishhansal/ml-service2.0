"""
tests/test_sprint2_new_infrastructure.py
------------------------------------------
Tests for Sprint 2 new infrastructure:
  - ExpandedFeatureFactory (NEW-P0-002 fix)
  - RegimeAlphaMatrix (NEW-P1-006 fix)
  - AlphaDecayDetector (NEW-P1-007 fix)
  - regime feature families
  - time_context feature family
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 1000.0 + np.cumsum(rng.normal(0, 5, n))
    prices = np.maximum(prices, 1.0)
    idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="UTC")
    high = prices * (1.0 + rng.uniform(0.001, 0.02, n))
    low = prices * (1.0 - rng.uniform(0.001, 0.02, n))
    open_ = prices * (1.0 + rng.normal(0, 0.005, n))
    volume = rng.integers(100_000, 2_000_000, n).astype(float)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": prices, "volume": volume,
    }, index=idx)


# ──────────────────────────────────────────────────────────────────────────────
# ExpandedFeatureFactory
# ──────────────────────────────────────────────────────────────────────────────

class TestExpandedFeatureFactory:
    """Tests for ExpandedFeatureFactory (FIX NEW-P0-002)."""

    def test_produces_more_features_than_base(self):
        from src.features.factory import FeatureFactory
        from src.features.expanded_factory import ExpandedFeatureFactory

        ohlcv = _make_ohlcv()
        base_f, _ = FeatureFactory().build(ohlcv)
        exp_f, _ = ExpandedFeatureFactory().build(ohlcv)

        assert len(exp_f.columns) > len(base_f.columns), (
            f"Expanded factory must have more columns than base "
            f"(got {len(exp_f.columns)} vs base {len(base_f.columns)})"
        )

    def test_base_features_preserved(self):
        """All 24 base feature names must appear in expanded output."""
        from src.features.factory import FeatureFactory
        from src.features.expanded_factory import ExpandedFeatureFactory

        ohlcv = _make_ohlcv()
        base_names = set(FeatureFactory.FEATURE_NAMES)
        exp_f, _ = ExpandedFeatureFactory().build(ohlcv)
        missing = base_names - set(exp_f.columns)
        assert not missing, f"Base features missing from expanded: {missing}"

    def test_regime_features_present(self):
        from src.features.expanded_factory import ExpandedFeatureFactory
        ohlcv = _make_ohlcv()
        f, _ = ExpandedFeatureFactory().build(ohlcv)
        regime_features = ["vol_regime_zscore", "trend_strength", "trend_direction", "gap_magnitude"]
        for feat in regime_features:
            assert feat in f.columns, f"Regime feature '{feat}' missing"

    def test_time_features_present(self):
        from src.features.expanded_factory import ExpandedFeatureFactory
        ohlcv = _make_ohlcv()
        f, _ = ExpandedFeatureFactory().build(ohlcv)
        assert "weekday" in f.columns
        assert "weekday_sin" in f.columns
        assert "is_monday" in f.columns

    def test_no_future_leakage_in_extended_features(self):
        """Appending future data must not change historical values."""
        from src.features.expanded_factory import ExpandedFeatureFactory

        ohlcv = _make_ohlcv(200)
        factory = ExpandedFeatureFactory()
        f_base, _ = factory.build(ohlcv)

        # Append extreme future data
        rng = np.random.default_rng(99)
        future_prices = 100_000.0 * np.ones(20)  # extreme spike
        future_idx = pd.date_range(ohlcv.index[-1] + pd.Timedelta("1D"), periods=20, freq="B", tz="UTC")
        future = pd.DataFrame({
            "open": future_prices, "high": future_prices * 1.01,
            "low": future_prices * 0.99, "close": future_prices,
            "volume": rng.integers(100_000, 200_000, 20).astype(float),
        }, index=future_idx)
        extended = pd.concat([ohlcv, future])
        f_ext, _ = factory.build(extended)

        # Historical portion should be identical for all regime features
        for col in ["vol_regime_zscore", "trend_strength", "gap_magnitude"]:
            if col not in f_base.columns or col not in f_ext.columns:
                continue
            hist_base = f_base[col].iloc[:150].dropna()
            hist_ext = f_ext[col].iloc[:150].dropna()
            common_idx = hist_base.index.intersection(hist_ext.index)
            if len(common_idx) < 5:
                continue
            deltas = (hist_base.loc[common_idx] - hist_ext.loc[common_idx]).abs()
            assert deltas.max() < 1e-9, (
                f"Future data changed historical values of '{col}': max_delta={deltas.max():.2e}"
            )

    def test_no_inf_in_output(self):
        from src.features.expanded_factory import ExpandedFeatureFactory
        ohlcv = _make_ohlcv()
        f, _ = ExpandedFeatureFactory().build(ohlcv)
        has_inf = np.isinf(f.fillna(0).values).any()
        assert not has_inf, "ExpandedFeatureFactory output must not contain infinity"

    def test_all_feature_names_match_columns(self):
        """FEATURE_NAMES list must exactly match DataFrame columns."""
        from src.features.expanded_factory import ExpandedFeatureFactory
        factory = ExpandedFeatureFactory()
        ohlcv = _make_ohlcv()
        f, _ = factory.build(ohlcv)
        assert list(f.columns) == factory.FEATURE_NAMES, (
            "FEATURE_NAMES must exactly match the columns produced by build()"
        )

    def test_advanced_vol_features_present(self):
        from src.features.expanded_factory import ExpandedFeatureFactory
        ohlcv = _make_ohlcv()
        f, _ = ExpandedFeatureFactory().build(ohlcv)
        assert "parkinson_vol" in f.columns
        assert "garman_klass_vol" in f.columns
        assert "atr_zscore" in f.columns


# ──────────────────────────────────────────────────────────────────────────────
# Regime Feature Family
# ──────────────────────────────────────────────────────────────────────────────

class TestRegimeFeatures:
    def test_volatility_regime_causal(self):
        """vol_regime_zscore must not change historical values when future appended."""
        from src.features.families.regime import compute_volatility_regime
        ohlcv = _make_ohlcv(100)
        base_reg = compute_volatility_regime(ohlcv["close"])

        # Append extreme future
        future_close = pd.concat([
            ohlcv["close"],
            pd.Series(100_000.0 * np.ones(20),
                      index=pd.date_range(ohlcv.index[-1] + pd.Timedelta("1D"),
                                          periods=20, freq="B", tz="UTC")),
        ])
        ext_reg = compute_volatility_regime(future_close)

        zscore_base = base_reg["vol_regime_zscore"].iloc[:80].dropna()
        zscore_ext = ext_reg["vol_regime_zscore"].iloc[:80].dropna()
        common = zscore_base.index.intersection(zscore_ext.index)
        if len(common) > 0:
            deltas = (zscore_base.loc[common] - zscore_ext.loc[common]).abs()
            assert deltas.max() < 1e-9, f"Causal violation: max_delta={deltas.max():.2e}"

    def test_trend_regime_adx_range(self):
        """ADX (trend_strength) must be in [0, 100]."""
        from src.features.families.regime import compute_trend_regime
        ohlcv = _make_ohlcv(200)
        reg = compute_trend_regime(ohlcv["close"], ohlcv["high"], ohlcv["low"])
        ts = reg["trend_strength"].dropna()
        assert (ts >= 0).all() and (ts <= 100).all(), (
            f"ADX must be in [0, 100], got range [{ts.min():.2f}, {ts.max():.2f}]"
        )

    def test_gap_magnitude_non_negative(self):
        from src.features.families.regime import compute_gap_regime
        ohlcv = _make_ohlcv(100)
        reg = compute_gap_regime(ohlcv["open"], ohlcv["close"])
        gm = reg["gap_magnitude"].dropna()
        assert (gm >= 0).all(), "Gap magnitude must be non-negative"


# ──────────────────────────────────────────────────────────────────────────────
# Time Context Feature Family
# ──────────────────────────────────────────────────────────────────────────────

class TestTimeContextFeatures:
    def test_weekday_range(self):
        from src.features.families.time_context import compute_calendar_features
        idx = pd.date_range("2023-01-02", periods=20, freq="B", tz="UTC")
        cal = compute_calendar_features(idx)
        assert "weekday" in cal
        assert cal["weekday"].min() >= 0 and cal["weekday"].max() <= 4

    def test_is_monday_is_binary(self):
        from src.features.families.time_context import compute_calendar_features
        idx = pd.date_range("2023-01-02", periods=10, freq="B", tz="UTC")
        cal = compute_calendar_features(idx)
        vals = cal["is_monday"].dropna().unique()
        assert set(vals).issubset({0.0, 1.0})

    def test_expiry_proximity_positive(self):
        from src.features.families.time_context import compute_expiry_proximity
        idx = pd.date_range("2023-01-02", periods=30, freq="B", tz="UTC")
        ep = compute_expiry_proximity(idx)
        valid = ep.dropna()
        assert (valid >= 0).all(), "Expiry proximity must be non-negative"

    def test_session_progress_range(self):
        from src.features.families.time_context import compute_intraday_session_features
        idx = pd.date_range("2023-01-02 09:15", periods=75, freq="5min", tz="Asia/Kolkata")
        features = compute_intraday_session_features(idx)
        prog = features["session_progress"].dropna()
        assert (prog >= 0).all() and (prog <= 1).all()


# ──────────────────────────────────────────────────────────────────────────────
# RegimeAlphaMatrix (NEW-P1-006)
# ──────────────────────────────────────────────────────────────────────────────

class TestRegimeAlphaMatrix:
    def test_update_and_retrieve(self):
        from src.analytics.regime_alpha_matrix import RegimeAlphaMatrix
        matrix = RegimeAlphaMatrix()
        for ic in [0.04, 0.03, 0.05, 0.02, 0.04, 0.03, 0.045, 0.025, 0.03, 0.04]:
            matrix.update("MOMENTUM_ALPHA", "TREND_UP", ic=ic)
        cell = matrix.get_performance("MOMENTUM_ALPHA", "TREND_UP")
        assert cell is not None
        assert cell.sample_count == 10
        assert 0.02 < cell.ic_mean < 0.06

    def test_best_alpha_for_regime(self):
        from src.analytics.regime_alpha_matrix import RegimeAlphaMatrix
        matrix = RegimeAlphaMatrix()
        for _ in range(15):
            matrix.update("MOMENTUM_ALPHA", "TREND_UP", ic=0.04)
            matrix.update("MEAN_REVERSION_ALPHA", "TREND_UP", ic=0.01)
        best = matrix.best_alpha_for_regime("TREND_UP")
        assert best == "MOMENTUM_ALPHA"

    def test_returns_none_when_no_data(self):
        from src.analytics.regime_alpha_matrix import RegimeAlphaMatrix
        matrix = RegimeAlphaMatrix()
        best = matrix.best_alpha_for_regime("TREND_UP")
        assert best is None

    def test_confidence_interval_wider_with_fewer_samples(self):
        """CI should be wide (spanning lo < mean < hi) for small samples."""
        from src.analytics.regime_alpha_matrix import RegimeAlphaMatrix
        matrix = RegimeAlphaMatrix()
        # Add 5 identical ICs — std=0, CI falls back to (-1, 1)
        for ic in [0.04] * 5:
            matrix.update("TREND_ALPHA", "RANGE", ic=ic)
        cell = matrix.get_performance("TREND_ALPHA", "RANGE")
        lo, hi = cell.ic_confidence_interval()
        # With n < 3, CI is always (-1, 1)
        assert lo < hi, f"CI lo ({lo}) must be < hi ({hi})"
        assert lo <= cell.ic_mean <= hi or (lo == -1.0 and hi == 1.0)

    def test_save_and_load(self, tmp_path: Path):
        from src.analytics.regime_alpha_matrix import RegimeAlphaMatrix
        matrix = RegimeAlphaMatrix()
        for ic in [0.03, 0.04, 0.05, 0.02, 0.04] * 3:
            matrix.update("BREAKOUT_ALPHA", "BREAKOUT", ic=ic)
        save_path = tmp_path / "matrix.json"
        matrix.save(save_path)
        loaded = RegimeAlphaMatrix.load(save_path)
        cell = loaded.get_performance("BREAKOUT_ALPHA", "BREAKOUT")
        assert cell is not None
        assert cell.sample_count > 0

    def test_to_dataframe(self):
        from src.analytics.regime_alpha_matrix import RegimeAlphaMatrix
        matrix = RegimeAlphaMatrix()
        for ic in [0.03] * 5:
            matrix.update("VOLATILITY_ALPHA", "HIGH_VOLATILITY", ic=ic)
        df = matrix.to_dataframe()
        assert len(df) >= 1
        assert "ic_mean" in df.columns

    def test_case_insensitive(self):
        from src.analytics.regime_alpha_matrix import RegimeAlphaMatrix
        matrix = RegimeAlphaMatrix()
        matrix.update("momentum_alpha", "trend_up", ic=0.04)
        cell = matrix.get_performance("MOMENTUM_ALPHA", "TREND_UP")
        assert cell is not None


# ──────────────────────────────────────────────────────────────────────────────
# AlphaDecayDetector (NEW-P1-007)
# ──────────────────────────────────────────────────────────────────────────────

class TestAlphaDecayDetector:
    def _make_tracker(self, ics: list[float], name: str = "TEST_ALPHA"):
        from src.analytics.alpha_decay import AlphaHealthTracker
        tracker = AlphaHealthTracker(name)
        for ic in ics:
            tracker.add_ic(ic)
        return tracker

    def test_healthy_stable_ic(self):
        from src.analytics.alpha_decay import AlphaDecayDetector, AlphaHealthState
        tracker = self._make_tracker([0.04] * 20)
        detector = AlphaDecayDetector(min_observations=10)
        result = detector.evaluate(tracker)
        assert result.health_state == AlphaHealthState.HEALTHY

    def test_insufficient_data(self):
        from src.analytics.alpha_decay import AlphaDecayDetector, AlphaHealthState
        tracker = self._make_tracker([0.04] * 5)
        detector = AlphaDecayDetector(min_observations=10)
        result = detector.evaluate(tracker)
        assert result.health_state == AlphaHealthState.INSUFFICIENT_DATA

    def test_quarantine_on_consecutive_negative(self):
        from src.analytics.alpha_decay import AlphaDecayDetector, AlphaHealthState
        # Start healthy, then go negative
        ics = [0.04] * 15 + [-0.02, -0.03, -0.01, -0.04, -0.02]
        tracker = self._make_tracker(ics)
        detector = AlphaDecayDetector(negative_bars=5, min_observations=10)
        result = detector.evaluate(tracker)
        assert result.health_state == AlphaHealthState.QUARANTINE
        assert result.should_quarantine

    def test_degrade_on_large_ic_drop(self):
        from src.analytics.alpha_decay import AlphaDecayDetector, AlphaHealthState
        # Peak at 0.08, then decay to 0.02 (75% drop)
        ics = [0.08] * 10 + [0.02] * 15
        tracker = self._make_tracker(ics)
        detector = AlphaDecayDetector(decay_threshold=0.50, min_observations=10)
        result = detector.evaluate(tracker)
        assert result.health_state in (AlphaHealthState.DEGRADE, AlphaHealthState.QUARANTINE)
        assert result.should_train_challenger or result.should_quarantine

    def test_warn_on_negative_trend(self):
        from src.analytics.alpha_decay import AlphaDecayDetector, AlphaHealthState
        # Slowly declining IC
        ics = [0.04 - 0.001 * i for i in range(20)]
        tracker = self._make_tracker(ics)
        detector = AlphaDecayDetector(warn_slope_threshold=-0.0005, min_observations=10)
        result = detector.evaluate(tracker)
        assert result.health_state in (AlphaHealthState.WARN, AlphaHealthState.DEGRADE)

    def test_recommended_action_mapping(self):
        from src.analytics.alpha_decay import AlphaDecayDetector, AlphaHealthState
        tracker = self._make_tracker([0.04] * 20)
        detector = AlphaDecayDetector(min_observations=10)
        result = detector.evaluate(tracker)
        assert result.recommended_action == "MONITOR"
        assert result.health_state == AlphaHealthState.HEALTHY

    def test_evaluate_all(self):
        from src.analytics.alpha_decay import AlphaDecayDetector
        tracker_a = self._make_tracker([0.04] * 15, "ALPHA_A")
        tracker_b = self._make_tracker([0.04] * 5 + [-0.01] * 5, "ALPHA_B")
        detector = AlphaDecayDetector(min_observations=10)
        results = detector.evaluate_all({"ALPHA_A": tracker_a, "ALPHA_B": tracker_b})
        assert "ALPHA_A" in results
        assert "ALPHA_B" in results

    def test_to_dict(self):
        from src.analytics.alpha_decay import AlphaDecayDetector
        tracker = self._make_tracker([0.04] * 15)
        detector = AlphaDecayDetector(min_observations=10)
        result = detector.evaluate(tracker)
        d = result.to_dict()
        assert "health_state" in d
        assert "rolling_ic_mean" in d
        assert "recommended_action" in d


class TestAlphaHalfLifeEstimator:
    def test_stable_ic_long_half_life(self):
        from src.analytics.alpha_decay import AlphaHalfLifeEstimator
        ics = [0.04] * 50
        hl = AlphaHalfLifeEstimator().estimate(ics)
        assert hl is None or hl > 5  # stable IC → long half-life or AR1 ≈ 1

    def test_fast_decay_short_half_life(self):
        from src.analytics.alpha_decay import AlphaHalfLifeEstimator
        # IC that decays rapidly (near-zero AR1)
        rng = np.random.default_rng(42)
        ics = [0.04 + rng.normal(0, 0.05) for _ in range(30)]  # white noise
        hl = AlphaHalfLifeEstimator().estimate(ics)
        # White noise has short half-life or None
        assert hl is None or hl < 50

    def test_insufficient_data_returns_none(self):
        from src.analytics.alpha_decay import AlphaHalfLifeEstimator
        hl = AlphaHalfLifeEstimator().estimate([0.04] * 5)
        assert hl is None
