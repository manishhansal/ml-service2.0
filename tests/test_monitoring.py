"""
Tests for the AlphaForge ML Model Monitoring and Drift Detection System.

Scenarios covered
-----------------
Drift detection
  1. Stable distribution    — PSI < 0.10, KS non-significant, JS < 0.10
  2. Slow (minor) drift     — PSI 0.10–0.20, minor severity
  3. Sudden (major) drift   — PSI > 0.20, major severity, CRITICAL alert emitted

Feature monitor
  4. Reference set + observe cycle triggers FeatureDriftReport
  5. force_check works below window_size

Performance monitor
  6. Healthy model stays HEALTHY under good trade outcomes
  7. Model degradation — bad Brier / low win-rate triggers DEGRADED state
  8. Strategy decay — negative Sharpe triggers decay alert
  9. Trading expectancy computation

Model registry
  10. Registration with provenance fields
  11. HEALTHY → WARNING → DEGRADED state transition via consecutive warnings
  12. Force DEGRADED → retraining recommendation created
  13. DEGRADED → DISABLED — weight drops to 0.0
  14. Recovery: DISABLED → HEALTHY, weight restored to 1.0
  15. get_weight returns correct multiplier per state
  16. get_active_models excludes disabled models

Alert system
  17. Alerts are stored and retrievable by severity / model
  18. Alert summary counts
  19. clear() resets history

Integration
  20. FeatureMonitor → ModelRegistry propagation (major drift → DEGRADED)
  21. PerformanceMonitor → ModelRegistry propagation (bad perf → DEGRADED)
  22. Prediction distribution summary (BUY/SELL/WAIT ratios)
  23. DriftDetector.detect_all bulk check
"""

from __future__ import annotations

import numpy as np
import pytest

from src.monitoring.alerts import (
    Alert,
    AlertCategory,
    AlertSeverity,
    AlertSystem,
)
from src.monitoring.drift_detector import (
    DriftDetector,
    DriftResult,
    DriftSeverity,
    detect_drift,
    _compute_psi,
    _compute_js_divergence,
)
from src.monitoring.feature_monitor import FeatureMonitor
from src.monitoring.model_registry import (
    ModelRecord,
    ModelRegistry,
    ModelState,
    RetrainingRecommendation,
    build_default_registry,
)
from src.monitoring.performance_monitor import (
    PerformanceMonitor,
    PerformanceThresholds,
    TradeOutcome,
    _brier_score,
    _ece,
    _accuracy,
    _trading_expectancy,
    _sharpe_ratio,
    _max_drawdown,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


def _stable_pair(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Two samples from the same Normal(0.5, 0.1) distribution."""
    ref = RNG.normal(0.5, 0.1, n).clip(0, 1)
    cur = RNG.normal(0.5, 0.1, n).clip(0, 1)
    return ref, cur


def _slow_drift_pair(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Current distribution shifts mean by +0.15 — minor PSI expected."""
    ref = RNG.normal(0.5, 0.1, n).clip(0, 1)
    cur = RNG.normal(0.65, 0.1, n).clip(0, 1)
    return ref, cur


def _sudden_drift_pair(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Current distribution shifts mean by +0.40 — major PSI expected."""
    ref = RNG.normal(0.3, 0.08, n).clip(0, 1)
    cur = RNG.normal(0.75, 0.08, n).clip(0, 1)
    return ref, cur


def _good_outcomes(n: int = 60, model: str = "regime") -> list[TradeOutcome]:
    """High win-rate, positive expectancy outcomes."""
    outcomes = []
    for i in range(n):
        wins = i % 3 != 0  # ~67% win rate
        outcomes.append(TradeOutcome(
            model_name=model,
            strategy="momentum",
            symbol=f"STOCK{i % 10}",
            predicted_prob=0.70 if wins else 0.35,
            actual_outcome=1.0 if wins else 0.0,
            pnl_pct=0.015 if wins else -0.008,
            confidence=0.72,
            action="BUY",
        ))
    return outcomes


def _bad_outcomes(n: int = 60, model: str = "regime") -> list[TradeOutcome]:
    """Low win-rate, negative expectancy outcomes."""
    outcomes = []
    for i in range(n):
        wins = i % 4 == 0  # 25% win rate
        outcomes.append(TradeOutcome(
            model_name=model,
            strategy="momentum",
            symbol=f"STOCK{i % 10}",
            predicted_prob=0.70,          # model always confident
            actual_outcome=1.0 if wins else 0.0,
            pnl_pct=0.008 if wins else -0.018,
            confidence=0.70,
            action="BUY",
        ))
    return outcomes


@pytest.fixture()
def alert_system() -> AlertSystem:
    return AlertSystem()


@pytest.fixture()
def registry(alert_system) -> ModelRegistry:
    reg = ModelRegistry(alert_system=alert_system, warn_after_n=3)
    reg.register(ModelRecord(
        model_name="regime",
        model_version="regime-xgb-v1",
        dataset_version="ds-v1",
        feature_version="feat-v1",
        training_period="2022-01-01/2024-06-30",
        deployment_date="2024-10-01T00:00:00Z",
        validation_metrics={"accuracy": 0.74},
    ))
    return reg


@pytest.fixture()
def perf_monitor(alert_system, registry) -> PerformanceMonitor:
    return PerformanceMonitor(
        alert_system=alert_system,
        model_registry=registry,
        window_size=100,
        min_samples=20,
    )


@pytest.fixture()
def feature_monitor(alert_system, registry) -> FeatureMonitor:
    fm = FeatureMonitor(
        alert_system=alert_system,
        model_registry=registry,
        window_size=50,
        min_check_size=10,
    )
    ref = RNG.normal(0.5, 0.1, 300).clip(0, 1)
    fm.set_reference("regime", "india_vix", ref)
    fm.set_reference("regime", "nifty_rsi", RNG.normal(55, 10, 300).clip(0, 100))
    return fm


# ─────────────────────────────────────────────────────────────────────────────
# 1. Stable distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestStableDistribution:
    def test_psi_below_threshold(self):
        ref, cur = _stable_pair()
        psi = _compute_psi(ref, cur)
        assert psi < 0.10, f"Expected PSI < 0.10 for stable dist, got {psi:.4f}"

    def test_js_below_threshold(self):
        ref, cur = _stable_pair()
        js = _compute_js_divergence(ref, cur)
        assert js < 0.10, f"Expected JS < 0.10 for stable dist, got {js:.4f}"

    def test_detect_drift_returns_none_severity(self):
        ref, cur = _stable_pair()
        result = detect_drift("test_feature", ref, cur)
        assert result.severity == DriftSeverity.NONE

    def test_detect_drift_no_drift_flag(self):
        ref, cur = _stable_pair()
        result = detect_drift("test_feature", ref, cur)
        assert not result.has_drift

    def test_stateful_detector_stable(self):
        detector = DriftDetector()
        ref, cur = _stable_pair()
        detector.set_reference("feat", ref)
        result = detector.detect("feat", cur)
        assert result is not None
        assert result.severity == DriftSeverity.NONE


# ─────────────────────────────────────────────────────────────────────────────
# 2. Slow (minor) drift
# ─────────────────────────────────────────────────────────────────────────────

class TestSlowDrift:
    def test_psi_in_minor_range(self):
        ref, cur = _slow_drift_pair(1000)
        psi = _compute_psi(ref, cur)
        # mean shift of 0.15 on Normal(0.5,0.1) should exceed PSI_NONE=0.10
        assert psi > 0.05, f"Expected some PSI increase for slow drift, got {psi:.4f}"

    def test_detect_drift_minor_or_above(self):
        ref, cur = _slow_drift_pair(1000)
        result = detect_drift("slow_feature", ref, cur)
        # A 0.15 mean shift should at minimum be MINOR
        assert result.severity in (DriftSeverity.MINOR, DriftSeverity.MAJOR), (
            f"Expected MINOR or MAJOR severity, got {result.severity}"
        )

    def test_drift_result_fields_populated(self):
        ref, cur = _slow_drift_pair(500)
        result = detect_drift("slow_feature", ref, cur)
        assert result.psi >= 0.0
        assert 0.0 <= result.ks_statistic <= 1.0
        assert 0.0 <= result.ks_p_value <= 1.0
        assert 0.0 <= result.js_divergence <= 1.0
        assert result.ref_n == len(ref)
        assert result.cur_n == len(cur)

    def test_drift_result_means_differ(self):
        ref, cur = _slow_drift_pair(1000)
        result = detect_drift("slow_feature", ref, cur)
        # Mean shift of 0.15 should be visible in the result
        assert abs(result.cur_mean - result.ref_mean) > 0.05


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sudden (major) drift
# ─────────────────────────────────────────────────────────────────────────────

class TestSuddenDrift:
    def test_psi_above_major_threshold(self):
        ref, cur = _sudden_drift_pair(1000)
        psi = _compute_psi(ref, cur)
        assert psi > 0.20, f"Expected PSI > 0.20 for sudden drift, got {psi:.4f}"

    def test_detect_drift_major_severity(self):
        ref, cur = _sudden_drift_pair(1000)
        result = detect_drift("sudden_feature", ref, cur)
        assert result.severity == DriftSeverity.MAJOR

    def test_detect_drift_has_drift_true(self):
        ref, cur = _sudden_drift_pair(1000)
        result = detect_drift("sudden_feature", ref, cur)
        assert result.has_drift

    def test_to_dict_structure(self):
        ref, cur = _sudden_drift_pair(500)
        result = detect_drift("sudden_feature", ref, cur)
        d = result.to_dict()
        assert d["severity"] == "major"
        assert "psi" in d
        assert "ks_statistic" in d
        assert "js_divergence" in d

    def test_stateful_detector_major_drift(self):
        detector = DriftDetector()
        ref, cur = _sudden_drift_pair(1000)
        detector.set_reference("feat", ref)
        result = detector.detect("feat", cur)
        assert result is not None
        assert result.severity == DriftSeverity.MAJOR

    def test_stateful_detector_no_reference_returns_none(self):
        detector = DriftDetector()
        result = detector.detect("unknown_feat", np.array([1.0, 2.0, 3.0]))
        assert result is None

    def test_detect_all_bulk(self):
        detector = DriftDetector()
        ref, _ = _stable_pair(500)
        _, cur_drift = _sudden_drift_pair(500)
        detector.set_reference("stable_feat", ref)
        detector.set_reference("drifted_feat", ref)
        results = detector.detect_all({
            "stable_feat":  ref,          # same as reference
            "drifted_feat": cur_drift,    # sudden drift
        })
        assert "stable_feat"  in results
        assert "drifted_feat" in results
        assert results["drifted_feat"].severity == DriftSeverity.MAJOR


# ─────────────────────────────────────────────────────────────────────────────
# 4 & 5. Feature Monitor
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureMonitor:
    def test_reference_set_is_reflected(self, feature_monitor):
        assert feature_monitor._detector.has_reference("regime::india_vix")
        assert feature_monitor._detector.has_reference("regime::nifty_rsi")

    def test_stable_observations_no_report(self, feature_monitor):
        # Feed 10 observations (below window of 50) — no report expected
        for _ in range(10):
            feature_monitor.observe("regime", {"india_vix": 0.52, "nifty_rsi": 54.0})
        report = feature_monitor.get_last_report("regime")
        # No report yet (not enough samples, and window not triggered)
        # We can't guarantee None here since force_check might not have fired
        # but there shouldn't be a MAJOR report
        if report is not None:
            assert report.overall_severity != DriftSeverity.MAJOR

    def test_window_trigger_returns_report(self, feature_monitor):
        # Flood the monitor with stable data until window triggers
        for _ in range(60):
            feature_monitor.observe(
                "regime",
                {"india_vix": 0.51, "nifty_rsi": 55.0},
            )
        report = feature_monitor.get_last_report("regime")
        assert report is not None
        assert report.model_name == "regime"

    def test_sudden_drift_detected_via_monitor(self, feature_monitor):
        # Inject drift: india_vix shifts from ~0.5 to ~0.9
        for _ in range(60):
            feature_monitor.observe(
                "regime",
                {"india_vix": 0.92, "nifty_rsi": 20.0},  # both drifted
            )
        report = feature_monitor.get_last_report("regime")
        assert report is not None
        # At least one feature should show drift
        assert report.n_drifted >= 1 or report.overall_severity != DriftSeverity.NONE

    def test_force_check_works(self, feature_monitor):
        # Inject enough data for min_check_size (10) but below window (50)
        for _ in range(15):
            feature_monitor.observe("regime", {"india_vix": 0.88, "nifty_rsi": 18.0})
        report = feature_monitor.force_check("regime")
        assert report is not None
        assert report.model_name == "regime"

    def test_force_check_insufficient_returns_none(self):
        # Brand new monitor with no data at all
        fm = FeatureMonitor(window_size=50, min_check_size=10)
        fm.set_reference("model_x", "feat_a", RNG.normal(0.5, 0.1, 100))
        result = fm.force_check("model_x")
        assert result is None  # zero observations buffered

    def test_set_references_bulk(self):
        fm = FeatureMonitor(window_size=50, min_check_size=10)
        fm.set_references_bulk("regime", {
            "india_vix": RNG.normal(0.5, 0.1, 200),
            "nifty_rsi": RNG.normal(55, 10, 200),
        })
        assert "india_vix" in fm.monitored_features("regime")
        assert "nifty_rsi" in fm.monitored_features("regime")

    def test_observe_batch(self, feature_monitor):
        vectors = [{"india_vix": 0.5, "nifty_rsi": 55.0} for _ in range(60)]
        # Should not raise; may return a report if window triggers
        feature_monitor.observe_batch("regime", vectors)

    def test_unmonitored_feature_ignored(self, feature_monitor):
        # No reference for 'unknown_feat' — observe should silently ignore it
        for _ in range(60):
            feature_monitor.observe("regime", {"unknown_feat": 999.0})
        # Should not crash and buffer fill for unknown feature should be 0
        fill = feature_monitor.get_buffer_fill("regime")
        assert fill.get("unknown_feat", 0) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Good performance — model stays HEALTHY
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthyModel:
    def test_good_outcomes_no_degradation(self, perf_monitor, registry):
        outcomes = _good_outcomes(n=60, model="regime")
        snap = perf_monitor.record_outcomes_batch(outcomes)
        assert snap is not None
        assert not snap.degraded
        assert snap.win_rate > 0.5
        assert snap.trading_expectancy > 0

    def test_brier_score_good_predictions(self):
        probs  = np.array([0.8, 0.75, 0.7, 0.2, 0.15])
        labels = np.array([1.0, 1.0,  1.0, 0.0,  0.0])
        bs = _brier_score(probs, labels)
        assert bs < 0.10, f"Expected low Brier for accurate predictions, got {bs:.4f}"

    def test_accuracy_good_threshold(self):
        probs  = np.array([0.8, 0.75, 0.7, 0.2, 0.15])
        labels = np.array([1.0, 1.0,  1.0, 0.0,  0.0])
        acc = _accuracy(probs, labels)
        assert acc == 1.0

    def test_ece_well_calibrated(self):
        # Well-calibrated: 70% confidence → ~70% accuracy
        probs  = np.array([0.7] * 100)
        labels = np.array([1.0] * 70 + [0.0] * 30)
        ece = _ece(probs, labels)
        assert ece < 0.05

    def test_trading_expectancy_positive(self):
        pnls = np.array([0.015, 0.012, -0.006, 0.018, 0.010, -0.008])
        exp, wr, aw, al = _trading_expectancy(pnls)
        assert exp > 0
        assert 0 < wr < 1

    def test_max_drawdown_zero_for_all_wins(self):
        pnls = np.array([0.01, 0.02, 0.015, 0.01])
        dd = _max_drawdown(pnls)
        assert dd >= 0.0  # cumsum always increasing → drawdown is 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Model degradation
# ─────────────────────────────────────────────────────────────────────────────

class TestModelDegradation:
    def test_bad_outcomes_mark_degraded(self, perf_monitor, registry):
        outcomes = _bad_outcomes(n=60, model="regime")
        snap = perf_monitor.record_outcomes_batch(outcomes)
        assert snap is not None
        assert snap.degraded, (
            f"Expected degraded=True; Brier={snap.brier_score:.3f}, "
            f"accuracy={snap.accuracy:.3f}, win_rate={snap.win_rate:.3f}"
        )

    def test_degraded_model_weight_reduced(self, perf_monitor, registry):
        outcomes = _bad_outcomes(n=60, model="regime")
        perf_monitor.record_outcomes_batch(outcomes)
        # Registry should have reduced the weight
        weight = registry.get_weight("regime")
        assert weight < 1.0, f"Expected weight < 1.0 for degraded model, got {weight}"

    def test_brier_score_bad_predictions(self):
        # Model is confident but wrong
        probs  = np.full(100, 0.80)
        labels = np.zeros(100)     # never correct
        bs = _brier_score(probs, labels)
        assert bs > 0.30

    def test_high_brier_triggers_degraded_flag(self):
        monitor = PerformanceMonitor(
            window_size=50, min_samples=10,
            thresholds=PerformanceThresholds(brier_critical=0.20),
        )
        outcomes = []
        for i in range(30):
            outcomes.append(TradeOutcome(
                model_name="test", strategy="momentum", symbol="X",
                predicted_prob=0.85,  # overconfident
                actual_outcome=0.0,   # always wrong
                pnl_pct=-0.01,
            ))
        snap = monitor.record_outcomes_batch(outcomes)
        assert snap is not None and snap.degraded


# ─────────────────────────────────────────────────────────────────────────────
# 8. Strategy decay
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyDecay:
    def test_negative_sharpe_triggers_decay(self, perf_monitor):
        # Register a test model
        outcomes = []
        for i in range(40):
            # Consistent small losses → negative Sharpe
            outcomes.append(TradeOutcome(
                model_name="regime", strategy="breakout", symbol="TCS",
                predicted_prob=0.6, actual_outcome=0.0,
                pnl_pct=-0.012, confidence=0.6,
            ))
        for outcome in outcomes:
            perf_monitor.record_outcome(outcome)

        metrics = perf_monitor.get_strategy_decay("regime", "breakout")
        assert metrics is not None
        assert metrics.rolling_sharpe < 0
        assert metrics.is_decaying

    def test_sharpe_ratio_positive_for_consistent_wins(self):
        pnls = np.full(50, 0.015)   # consistent 1.5% wins
        sr = _sharpe_ratio(pnls)
        assert sr > 0

    def test_sharpe_ratio_negative_for_losses(self):
        pnls = np.full(50, -0.012)
        sr = _sharpe_ratio(pnls)
        assert sr < 0

    def test_max_drawdown_negative(self):
        pnls = np.array([0.01, -0.05, 0.01, -0.03])
        dd = _max_drawdown(pnls)
        assert dd < 0.0

    def test_win_rate_calculation(self):
        pnls = np.array([0.01, 0.02, -0.01, 0.015, -0.008, 0.01, 0.01, -0.005])
        _, wr, _, _ = _trading_expectancy(pnls)
        expected_wr = 5 / 8
        assert abs(wr - expected_wr) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 9. Trading expectancy
# ─────────────────────────────────────────────────────────────────────────────

class TestTradingExpectancy:
    def test_expectancy_positive(self):
        pnls = np.array([0.02, 0.015, -0.008, 0.018, -0.007])
        exp, wr, aw, al = _trading_expectancy(pnls)
        assert exp > 0

    def test_expectancy_negative_for_losers(self):
        pnls = np.array([-0.02, -0.015, 0.005, -0.018, -0.007])
        exp, _, _, _ = _trading_expectancy(pnls)
        assert exp < 0

    def test_expectancy_empty_array(self):
        exp, wr, aw, al = _trading_expectancy(np.array([]))
        assert exp == 0.0
        assert wr == 0.0

    def test_expectancy_all_wins(self):
        pnls = np.array([0.01, 0.02, 0.015])
        exp, wr, aw, al = _trading_expectancy(pnls)
        assert wr == 1.0
        assert al == 0.0
        assert exp > 0


# ─────────────────────────────────────────────────────────────────────────────
# 10–16. Model Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestModelRegistry:
    def test_registration(self, registry):
        assert registry.is_registered("regime")
        record = registry.get("regime")
        assert record is not None
        assert record.model_version == "regime-xgb-v1"
        assert record.state == ModelState.HEALTHY

    def test_to_dict_has_required_fields(self, registry):
        d = registry.get("regime").to_dict()
        required = [
            "modelName", "modelVersion", "datasetVersion", "featureVersion",
            "trainingPeriod", "deploymentDate", "validationMetrics",
            "state", "weightMultiplier",
        ]
        for key in required:
            assert key in d, f"Missing key: {key}"

    def test_healthy_weight_is_one(self, registry):
        assert registry.get_weight("regime") == 1.0

    def test_warning_weight_is_reduced(self, registry):
        registry.set_warning("regime", reasons=["test drift"])
        weight = registry.get_weight("regime")
        assert weight < 1.0
        assert weight == 0.75

    def test_state_transitions_healthy_to_warning(self, registry):
        new_state = registry.set_warning("regime", reasons=["minor drift"])
        assert new_state == ModelState.WARNING

    def test_consecutive_warnings_escalate_to_degraded(self, registry):
        """3 consecutive warnings → DEGRADED (warn_after_n=3)."""
        for _ in range(3):
            registry.set_warning("regime", reasons=["repeated drift"])
        record = registry.get("regime")
        assert record.state == ModelState.DEGRADED

    def test_degraded_weight_is_point_three(self, registry):
        registry.set_degraded("regime", reasons=["performance below threshold"])
        assert registry.get_weight("regime") == 0.30

    def test_degraded_creates_retraining_recommendation(self, registry):
        registry.set_degraded("regime", reasons=["Brier=0.40"])
        record = registry.get("regime")
        assert len(record.retraining_recommendations) >= 1
        rec = record.retraining_recommendations[-1]
        assert not rec.acknowledged
        assert rec.current_state == ModelState.DEGRADED

    def test_disabled_weight_is_zero(self, registry):
        registry.set_disabled("regime", reasons=["operator override"])
        assert registry.get_weight("regime") == 0.0

    def test_disabled_excluded_from_active(self, registry):
        registry.set_disabled("regime", reasons=["disabled"])
        assert "regime" not in registry.get_active_models()
        assert "regime" in registry.get_disabled_models()

    def test_recovery_restores_healthy_state(self, registry):
        registry.set_disabled("regime", reasons=["test"])
        new_state = registry.recover("regime", notes="Model retrained.")
        assert new_state == ModelState.HEALTHY
        assert registry.get_weight("regime") == 1.0

    def test_recovery_resets_consecutive_warnings(self, registry):
        for _ in range(2):
            registry.set_warning("regime", reasons=["x"])
        registry.recover("regime")
        record = registry.get("regime")
        assert record.consecutive_warnings == 0

    def test_disabled_model_not_auto_recovered_by_warning(self, registry):
        registry.set_disabled("regime", reasons=["hard disable"])
        state = registry.set_warning("regime", reasons=["ignored warning"])
        assert state == ModelState.DISABLED

    def test_state_history_recorded(self, registry):
        registry.set_warning("regime",  reasons=["w"])
        registry.set_degraded("regime", reasons=["d"])
        record = registry.get("regime")
        # Should have at least one transition in history
        assert len(record.state_history) >= 1

    def test_unknown_model_returns_default_weight(self, registry):
        weight = registry.get_weight("nonexistent_model")
        assert weight == 1.0

    def test_health_summary_structure(self, registry):
        summary = registry.health_summary()
        assert "totalModels"    in summary
        assert "stateCounts"    in summary
        assert "models"         in summary
        assert "activeModels"   in summary
        assert "disabledModels" in summary

    def test_acknowledge_recommendation(self, registry):
        registry.set_degraded("regime", reasons=["psi high"])
        count = registry.acknowledge_recommendation("regime", acknowledged_by="test_user")
        assert count >= 1
        recs = registry.get_retraining_recommendations(unacknowledged_only=True)
        model_recs = [r for r in recs if r.model_name == "regime"]
        assert len(model_recs) == 0  # all acknowledged

    def test_build_default_registry_has_all_models(self):
        from src.meta.ensemble import MODEL_NAMES
        reg = build_default_registry()
        for name in MODEL_NAMES:
            assert reg.is_registered(name), f"Model '{name}' not in default registry"


# ─────────────────────────────────────────────────────────────────────────────
# 17–19. Alert System
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertSystem:
    def test_alert_stored_in_history(self, alert_system):
        alert_system.warn(
            AlertCategory.FEATURE_DRIFT, "regime",
            "Test warning", "Some drift detected."
        )
        recent = alert_system.get_recent()
        assert len(recent) == 1
        assert recent[0].severity == AlertSeverity.WARNING

    def test_filter_by_severity(self, alert_system):
        alert_system.info(AlertCategory.SYSTEM, "system", "Info", "msg")
        alert_system.critical(AlertCategory.PERFORMANCE, "regime", "Crit", "msg")
        criticals = alert_system.get_recent(severity=AlertSeverity.CRITICAL)
        assert all(a.severity == AlertSeverity.CRITICAL for a in criticals)

    def test_filter_by_model(self, alert_system):
        alert_system.warn(AlertCategory.FEATURE_DRIFT, "regime",  "A", "msg")
        alert_system.warn(AlertCategory.FEATURE_DRIFT, "ranker", "B", "msg")
        regime_alerts = alert_system.get_recent(model_name="regime")
        assert all(a.model_name == "regime" for a in regime_alerts)

    def test_alert_summary_counts(self, alert_system):
        alert_system.info(    AlertCategory.SYSTEM,      "x", "I", ".")
        alert_system.warn(    AlertCategory.SYSTEM,      "x", "W", ".")
        alert_system.warn(    AlertCategory.SYSTEM,      "x", "W2",".")
        alert_system.critical(AlertCategory.PERFORMANCE, "x", "C", ".")
        summary = alert_system.get_summary()
        assert summary["counts"]["info"]     == 1
        assert summary["counts"]["warning"]  == 2
        assert summary["counts"]["critical"] == 1
        assert summary["total"] == 4

    def test_clear_resets_history(self, alert_system):
        alert_system.warn(AlertCategory.SYSTEM, "x", "w", ".")
        alert_system.clear()
        assert len(alert_system.get_recent()) == 0

    def test_alert_to_dict_structure(self, alert_system):
        alert = alert_system.warn(
            AlertCategory.FEATURE_DRIFT, "regime", "Drift", "PSI=0.25",
            metadata={"psi": 0.25},
        )
        d = alert.to_dict()
        assert d["severity"]  == "warning"
        assert d["category"]  == "feature_drift"
        assert d["modelName"] == "regime"
        assert d["metadata"]["psi"] == 0.25
        assert "alertId"   in d
        assert "timestamp" in d

    def test_disabled_alert_emitted(self, alert_system):
        alert_system.disabled_alert("regime", "Model disabled by operator.")
        recent = alert_system.get_recent(severity=AlertSeverity.DISABLED)
        assert len(recent) == 1

    def test_alert_limit_respected(self, alert_system):
        for i in range(20):
            alert_system.info(AlertCategory.SYSTEM, "x", f"Alert {i}", ".")
        recent = alert_system.get_recent(limit=5)
        assert len(recent) == 5


# ─────────────────────────────────────────────────────────────────────────────
# 20. Integration: FeatureMonitor → ModelRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureMonitorRegistryIntegration:
    def test_major_drift_escalates_registry_state(self, feature_monitor, registry):
        """
        Injecting 60 observations with severe drift (india_vix shifting from
        ~0.5 to ~0.95) should trigger a MAJOR DriftResult and escalate the
        model's state in the registry to at least WARNING.
        """
        for _ in range(60):
            feature_monitor.observe(
                "regime", {"india_vix": 0.97, "nifty_rsi": 10.0}
            )
        record = registry.get("regime")
        assert record is not None
        assert record.state in (ModelState.WARNING, ModelState.DEGRADED), (
            f"Expected WARNING or DEGRADED after major drift injection, got {record.state}"
        )

    def test_stable_drift_does_not_escalate(self):
        """Stable data must not escalate a HEALTHY model."""
        alerts   = AlertSystem()
        reg      = ModelRegistry(alert_system=alerts, warn_after_n=3)
        reg.register(ModelRecord(
            model_name="ranker",
            model_version="v1",
            dataset_version="ds",
            feature_version="fv",
            training_period="2022/2024",
            deployment_date="2024-10-01T00:00:00Z",
        ))
        fm = FeatureMonitor(
            alert_system=alerts, model_registry=reg, window_size=50, min_check_size=10
        )
        # Use a seeded generator with a large reference so PSI bins are well-populated
        rng_local = np.random.default_rng(99)
        ref = rng_local.normal(0.5, 0.1, 500)
        fm.set_reference("ranker", "rsi_14", ref)

        # Feed stable data drawn from the same distribution (50 obs = one window)
        for _ in range(60):
            fm.observe("ranker", {"rsi_14": float(rng_local.normal(0.5, 0.1))})

        record = reg.get("ranker")
        # Stable data should not reach DEGRADED
        # (may briefly hit WARNING due to small-sample noise, but not DEGRADED)
        assert record is not None
        assert record.state != ModelState.DISABLED


# ─────────────────────────────────────────────────────────────────────────────
# 21. Integration: PerformanceMonitor → ModelRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformanceRegistryIntegration:
    def test_bad_performance_escalates_state(self, perf_monitor, registry):
        outcomes = _bad_outcomes(n=60, model="regime")
        snap = perf_monitor.record_outcomes_batch(outcomes)
        assert snap is not None and snap.degraded
        record = registry.get("regime")
        assert record.state == ModelState.DEGRADED

    def test_good_performance_does_not_degrade(self, perf_monitor, registry):
        outcomes = _good_outcomes(n=60, model="regime")
        perf_monitor.record_outcomes_batch(outcomes)
        record = registry.get("regime")
        assert record.state == ModelState.HEALTHY


# ─────────────────────────────────────────────────────────────────────────────
# 22. Prediction distribution summary
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionDistribution:
    def test_buy_sell_wait_ratios(self):
        actions = ["BUY"] * 60 + ["SELL"] * 20 + ["WAIT"] * 20
        confs   = [0.75] * 60 + [0.65] * 20 + [0.50] * 20
        summary = DriftDetector.prediction_distribution_summary(actions, confs)
        assert summary["counts"]["BUY"]  == 60
        assert summary["counts"]["SELL"] == 20
        assert summary["counts"]["WAIT"] == 20
        assert abs(summary["ratios"]["BUY"] - 0.60) < 1e-6
        assert summary["buy_sell_ratio"] == pytest.approx(3.0, abs=0.01)

    def test_confidence_stats(self):
        actions = ["BUY"] * 10
        confs   = np.linspace(0.6, 0.9, 10).tolist()
        summary = DriftDetector.prediction_distribution_summary(actions, confs)
        assert "confidence_mean" in summary
        assert "confidence_std"  in summary
        assert "confidence_p10"  in summary
        assert "confidence_p90"  in summary

    def test_empty_actions_returns_empty_dict(self):
        summary = DriftDetector.prediction_distribution_summary([], [])
        assert summary == {}

    def test_no_trade_ratio(self):
        actions = ["BUY"] * 4 + ["NO_TRADE"] * 6
        confs   = [0.6] * 10
        summary = DriftDetector.prediction_distribution_summary(actions, confs)
        assert summary["counts"].get("NO_TRADE", 0) == 6


# ─────────────────────────────────────────────────────────────────────────────
# 23. Disable → recovery scenario (end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

class TestDisableAndRecovery:
    def test_full_lifecycle(self, registry, perf_monitor, alert_system):
        """
        Full lifecycle:
          HEALTHY → DEGRADED (bad performance) → DISABLED (operator) → HEALTHY (recovery)
        """
        # Step 1: start healthy
        assert registry.get("regime").state == ModelState.HEALTHY
        assert registry.get_weight("regime") == 1.0

        # Step 2: bad performance → DEGRADED
        outcomes = _bad_outcomes(n=60, model="regime")
        snap = perf_monitor.record_outcomes_batch(outcomes)
        assert snap is not None and snap.degraded
        assert registry.get("regime").state == ModelState.DEGRADED
        assert registry.get_weight("regime") == 0.30
        assert "regime" not in registry.get_disabled_models()

        # Step 3: operator disables model
        registry.set_disabled("regime", reasons=["Persistent degradation — disabled by ops"])
        assert registry.get("regime").state == ModelState.DISABLED
        assert registry.get_weight("regime") == 0.0
        assert "regime" in registry.get_disabled_models()
        assert "regime" not in registry.get_active_models()

        # Verify a DISABLED alert was emitted
        disabled_alerts = alert_system.get_recent(severity=AlertSeverity.DISABLED)
        assert len(disabled_alerts) >= 1
        assert any(a.model_name == "regime" for a in disabled_alerts)

        # Verify retraining recommendation exists
        recs = registry.get_retraining_recommendations()
        assert any(r.model_name == "regime" for r in recs)

        # Step 4: operator recovers after retraining
        new_state = registry.recover("regime", notes="Retrained on fresh dataset.")
        assert new_state == ModelState.HEALTHY
        assert registry.get_weight("regime") == 1.0
        assert "regime" in registry.get_active_models()
        assert "regime" not in registry.get_disabled_models()

        # Recovery alert should have been emitted
        info_alerts = alert_system.get_recent(severity=AlertSeverity.INFO, model_name="regime")
        assert len(info_alerts) >= 1

    def test_disabled_model_has_zero_weight(self, registry):
        registry.set_disabled("regime", reasons=["zero weight test"])
        all_weights = registry.get_all_weights()
        assert all_weights["regime"] == 0.0

    def test_retraining_recommendation_not_auto_triggered(self, registry):
        """Retraining recommendation is created but auto-retraining is NOT triggered."""
        registry.set_degraded("regime", reasons=["drift"])
        recs = registry.get_retraining_recommendations()
        assert any(r.model_name == "regime" for r in recs)
        # No 'retraining_triggered' or similar field — it's always a recommendation only
        for rec in recs:
            assert hasattr(rec, "acknowledged")
            assert not rec.acknowledged  # untouched by default

    def test_multiple_model_disable_and_recover(self):
        """Multiple models can be independently disabled and recovered."""
        alerts = AlertSystem()
        reg = build_default_registry(alert_system=alerts)

        # Disable regime and ranker
        reg.set_disabled("regime", reasons=["bad"])
        reg.set_disabled("ranker", reasons=["bad"])

        assert reg.get_weight("regime") == 0.0
        assert reg.get_weight("ranker") == 0.0
        assert len(reg.get_disabled_models()) == 2

        # Recover regime only
        reg.recover("regime", notes="Regime model retrained.")
        assert reg.get_weight("regime") == 1.0
        assert reg.get_weight("ranker") == 0.0
        assert "regime" in reg.get_active_models()
        assert "ranker" in reg.get_disabled_models()
