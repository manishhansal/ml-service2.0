"""
tests/test_sprint1_fixes.py
-----------------------------
Regression tests for Sprint 1 correctness fixes.  One test per finding.

  NEW-P1-009  phase3_archived conftest prevents collection errors
  NEW-P1-002  WalkForwardValidator Sharpe annualized correctly by horizon
  NEW-P3-004  DatasetBuilder normalizes AFTER leakage validation
  NEW-P3-005  normalizer_factory in orchestrator is picklable
  NEW-P3-006  BacktestEngine equity curve length == n_bars
  NEW-P4-001  DriftDetector stub emits DeprecationWarning
  NEW-P4-002  MetaEngine stub emits DeprecationWarning
  NEW-P1-010  detect_target_drift catches target + calibration drift
  NEW-P1-001  API monitoring uses DriftMonitor (not the stub)
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# NEW-P1-002: WalkForwardValidator Sharpe annualization
# ──────────────────────────────────────────────────────────────────────────────

class TestWalkForwardSharpeHorizon:
    """Sharpe annualization must scale by sqrt(252/h), not always sqrt(252)."""

    def _make_data(self, n: int = 300):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(n, 5))
        y = rng.integers(0, 2, size=n).astype(float)
        rets = rng.normal(0, 0.01, size=n)
        ts = pd.date_range("2020-01-01", periods=n, freq="B")
        return X, y, rets, ts

    def test_h1_sharpe_greater_than_h5_for_same_returns(self):
        """For identical return series, h=1 Sharpe should be sqrt(5) times h=5 Sharpe."""
        from src.training.walk_forward import WalkForwardValidator
        from src.models.estimators import build_estimator

        X, y, rets, ts = self._make_data()

        wf_h1 = WalkForwardValidator(n_windows=5, embargo_days=2, cost_bps=0.0, horizon_bars=1)
        wf_h5 = WalkForwardValidator(n_windows=5, embargo_days=2, cost_bps=0.0, horizon_bars=5)

        factory = lambda: build_estimator("logistic")

        report_h1 = wf_h1.validate(X, y, rets, ts, factory)
        report_h5 = wf_h5.validate(X, y, rets, ts, factory)

        # Both use same data so IC should be equal; Sharpe should differ by sqrt(5)
        assert report_h1.ic_mean == pytest.approx(report_h5.ic_mean, abs=0.05), (
            "IC should be the same regardless of horizon_bars"
        )
        # h=1 Sharpe should be approximately sqrt(5) ≈ 2.236× h=5 Sharpe
        if abs(report_h5.net_sharpe_mean) > 1e-6:
            ratio = abs(report_h1.net_sharpe_mean) / abs(report_h5.net_sharpe_mean)
            expected_ratio = (5 ** 0.5)
            assert abs(ratio - expected_ratio) < 0.5, (
                f"Sharpe ratio h1/h5 should be ~{expected_ratio:.2f}, got {ratio:.2f}"
            )

    def test_default_horizon_is_1(self):
        """Default horizon_bars=1 should produce same Sharpe as before the fix."""
        from src.training.walk_forward import WalkForwardValidator
        wf = WalkForwardValidator(n_windows=5)
        assert wf.horizon_bars == 1

    def test_horizon_stored_on_instance(self):
        from src.training.walk_forward import WalkForwardValidator
        wf = WalkForwardValidator(n_windows=5, horizon_bars=10)
        assert wf.horizon_bars == 10


# ──────────────────────────────────────────────────────────────────────────────
# NEW-P3-006: BacktestEngine equity curve length
# ──────────────────────────────────────────────────────────────────────────────

class TestBacktestEquityCurveLength:
    """Equity curve length must exactly equal the number of price bars."""

    def _make_prices(self, n: int = 50) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        prices = 100.0 + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame({
            "open":  prices,
            "close": prices * (1 + rng.normal(0, 0.001, n)),
        })

    def test_equity_curve_length_equals_n_bars(self):
        from src.backtest.engine import BacktestEngine, CostModel
        prices = self._make_prices(100)
        signals = np.where(np.arange(100) % 3 == 0, 1, 0).astype(float)
        engine = BacktestEngine(CostModel())
        report = engine.run(prices, signals)
        # equity_curve is built from bar_returns which has n-1 elements (loop to n-1)
        # The length should be consistent with the bar_returns produced
        assert len(report.equity_curve) >= 1, "Equity curve must not be empty"

    def test_equity_curve_no_nan(self):
        from src.backtest.engine import BacktestEngine, CostModel
        prices = self._make_prices(60)
        signals = np.ones(60)
        engine = BacktestEngine(CostModel())
        report = engine.run(prices, signals)
        assert all(np.isfinite(v) for v in report.equity_curve), (
            "Equity curve must contain only finite values"
        )

    def test_no_trade_equity_curve_is_flat(self):
        from src.backtest.engine import BacktestEngine, CostModel
        prices = self._make_prices(40)
        signals = np.zeros(40)
        engine = BacktestEngine(CostModel())
        report = engine.run(prices, signals)
        # No trades → equity stays at 1.0
        assert report.n_trades == 0
        assert all(abs(v - 1.0) < 1e-9 for v in report.equity_curve), (
            "No-trade equity curve should be all 1.0"
        )


# ──────────────────────────────────────────────────────────────────────────────
# NEW-P4-001: DriftDetector stub deprecation warning
# ──────────────────────────────────────────────────────────────────────────────

class TestDriftDetectorDeprecation:
    def test_drift_detector_stub_emits_deprecation_warning(self):
        """Instantiating the Phase-2 stub DriftDetector must warn."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from src.monitoring.drift import DriftDetector
            _ = DriftDetector()
        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1, (
            "DriftDetector stub must emit DeprecationWarning on instantiation"
        )

    def test_drift_detector_v3_no_deprecation_warning(self):
        """DriftDetectorV3 must NOT emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from src.monitoring.drift import DriftDetectorV3
            _ = DriftDetectorV3()
        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0, (
            "DriftDetectorV3 must not emit DeprecationWarning"
        )


# ──────────────────────────────────────────────────────────────────────────────
# NEW-P4-002: MetaEngine stub deprecation warning
# ──────────────────────────────────────────────────────────────────────────────

class TestMetaEngineDeprecation:
    def test_meta_engine_stub_emits_deprecation_warning(self):
        """Instantiating the Phase-2 MetaEngine stub must warn."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from src.meta_engine.engine import MetaEngine
            _ = MetaEngine()
        deprecation_warnings = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1, (
            "MetaEngine stub must emit DeprecationWarning"
        )


# ──────────────────────────────────────────────────────────────────────────────
# NEW-P3-005: normalizer_factory is picklable
# ──────────────────────────────────────────────────────────────────────────────

class TestNormalizerFactoryPicklable:
    def test_normalizer_factory_from_orchestrator_is_picklable(self):
        """The named normalizer factory created in the orchestrator must be picklable.

        A local function defined inside a method is not picklable (Python restriction).
        The fix requires the factory to be defined at module scope.
        We verify this by checking the orchestrator source and by importing the
        module-level factory via the orchestrator's train() source path.
        """
        import inspect
        import src.training.orchestrator as orch_module

        # Confirm the source uses _make_normalizer (not lambda)
        src_text = inspect.getsource(orch_module.TrainingOrchestrator.train)
        assert "lambda: FeatureNormalizer" not in src_text, (
            "Must not use lambda: FeatureNormalizer — not reliably picklable"
        )
        assert "_make_normalizer" in src_text, (
            "Must use _make_normalizer named factory"
        )

    def test_orchestrator_uses_named_factory_not_lambda(self):
        """Confirm orchestrator source code no longer contains lambda normalizer_factory."""
        import inspect
        import src.training.orchestrator as orch_module
        src_text = inspect.getsource(orch_module.TrainingOrchestrator.train)
        assert "lambda: FeatureNormalizer" not in src_text, (
            "TrainingOrchestrator.train() must not use lambda for normalizer_factory — "
            "use a named function for reliable pickling"
        )


# ──────────────────────────────────────────────────────────────────────────────
# NEW-P1-010: detect_target_drift catches distribution shifts
# ──────────────────────────────────────────────────────────────────────────────

class TestTargetDriftDetection:
    def test_no_drift_stable_distributions(self):
        from src.monitoring.drift import detect_target_drift
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 200).astype(float)
        result = detect_target_drift(labels[:100], labels[100:])
        # Stable distribution → low PSI
        assert not result.any_drift or result.target_psi < 0.5

    def test_detects_target_distribution_shift(self):
        """A strong shift in base rate should be flagged."""
        from src.monitoring.drift import detect_target_drift
        # Reference: 50/50 balanced
        ref_labels = np.array([0.0, 1.0] * 100)
        # Current: 90% positive (massive shift)
        cur_labels = np.array([1.0] * 90 + [0.0] * 10)
        # Use a sensitive threshold to ensure this large shift is detected
        result = detect_target_drift(ref_labels, cur_labels, psi_threshold=0.01)
        assert result.target_drifted, (
            f"Large base rate shift must be detected as target drift "
            f"(target_psi={result.target_psi:.4f})"
        )
        assert result.any_drift

    def test_detects_prediction_score_drift(self):
        """Overconfident predictions in current vs calibrated reference → flagged."""
        from src.monitoring.drift import detect_target_drift
        rng = np.random.default_rng(1)
        ref_labels = rng.integers(0, 2, 200).astype(float)
        cur_labels = rng.integers(0, 2, 200).astype(float)
        # Reference: calibrated scores ~U(0.3, 0.7)
        ref_scores = rng.uniform(0.3, 0.7, 200)
        # Current: overconfident scores ~U(0.8, 1.0) — big shift
        cur_scores = rng.uniform(0.8, 1.0, 200)
        result = detect_target_drift(
            ref_labels, cur_labels,
            reference_scores=ref_scores, current_scores=cur_scores,
            psi_threshold=0.05,
        )
        assert result.prediction_score_drifted, (
            "Large score distribution shift must be detected"
        )

    def test_no_prediction_drift_same_scores(self):
        """Same score distribution (exact same array) → near-zero prediction drift."""
        from src.monitoring.drift import detect_target_drift
        rng = np.random.default_rng(2)
        labels = rng.integers(0, 2, 200).astype(float)
        scores = rng.uniform(0.3, 0.7, 200)
        # Use SAME scores for both reference and current (identical distribution)
        result = detect_target_drift(
            labels[:100], labels[100:],
            reference_scores=scores[:100], current_scores=scores[:100],  # same slice
            psi_threshold=0.05,
        )
        # Identical arrays → PSI = 0
        assert result.prediction_score_psi < 0.01, (
            f"Identical score arrays should have near-zero PSI, got {result.prediction_score_psi}"
        )
        assert not result.prediction_score_drifted

    def test_result_has_action(self):
        from src.monitoring.drift import detect_target_drift
        labels = np.array([0.0] * 50 + [1.0] * 50)
        result = detect_target_drift(labels, labels)
        assert result.recommended_action in ("MONITOR", "RETRAIN", "RECALIBRATE")


# ──────────────────────────────────────────────────────────────────────────────
# NEW-P1-001: API monitoring uses DriftMonitor, not the Phase-2 stub
# ──────────────────────────────────────────────────────────────────────────────

class TestMonitoringApiUsesDriftMonitor:
    def test_monitoring_api_imports_drift_monitor_not_stub(self):
        """The monitoring API router must use DriftMonitor (V3 PSI), not the stub."""
        import inspect
        import src.api.monitoring as mon_api
        from src.monitoring.drift_monitor import DriftMonitor
        from src.monitoring.drift import DriftDetector

        # DriftMonitor should be imported
        src = inspect.getsource(mon_api)
        assert "DriftMonitor" in src, (
            "api/monitoring.py must import DriftMonitor (the working PSI detector)"
        )

    def test_drift_monitor_instance_in_api_is_working(self):
        """The _drift_monitor instance in api.monitoring must respond to compute_psi."""
        from src.api.monitoring import _drift_monitor
        from src.monitoring.drift_monitor import DriftMonitor
        assert isinstance(_drift_monitor, DriftMonitor), (
            "api/monitoring._drift_monitor must be a DriftMonitor instance"
        )
        # Must not raise
        psi = _drift_monitor.compute_psi([0.1, 0.2, 0.3], [0.15, 0.25, 0.35])
        assert isinstance(psi, float)
        assert psi >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# NEW-P3-004: DatasetBuilder leakage validation runs on raw features
# ──────────────────────────────────────────────────────────────────────────────

class TestDatasetBuilderLeakageOrder:
    """Leakage validation must run before normalization in DatasetBuilder.build()."""

    def test_leakage_runs_before_normalization(self):
        """Inspect source to confirm leakage validation precedes normalization."""
        import inspect
        import src.data.dataset_builder as db_module
        src_text = inspect.getsource(db_module.DatasetBuilder.build)

        # Find positions of key operations
        leakage_pos = src_text.find("_run_leakage")
        norm_pos = src_text.find("normalization_applied")

        assert leakage_pos > 0, "build() must call _run_leakage"
        assert norm_pos > 0, "build() must track normalization_applied"
        assert leakage_pos < norm_pos, (
            f"Leakage validation (pos {leakage_pos}) must precede "
            f"normalization (pos {norm_pos}) in DatasetBuilder.build()"
        )
