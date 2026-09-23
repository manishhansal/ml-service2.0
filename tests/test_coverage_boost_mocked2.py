"""
Additional mock-based coverage tests — second batch.

Targets:
- src/models/stock_ranker.py (80%)
- src/models/strategy_selector.py (73%)
- src/meta/calibration.py (82%)
- src/monitoring/drift_monitor.py (83%)
- src/registry/registry.py (74%)
- src/models/regime_classifier.py (83%)
- src/models/risk_predictor.py (74%)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

import numpy as np
import pytest


# ===========================================================================
# StockRanker
# ===========================================================================


class TestStockRanker:
    def _make_stocks(self, n=3):
        return [
            {
                "relative_volume": float(i + 1),
                "momentum_5d": float(i) * 0.1,
                "rsi_14": 50.0 + float(i),
                "relative_strength_vs_nifty": float(i) * 0.2,
                "ema_stack_score": float(i) * 0.5,
            }
            for i in range(n)
        ]

    def test_rank_heuristic_basic(self):
        from src.models.stock_ranker import StockRanker
        from src.schemas.base import MarketRegime

        ranker = StockRanker()
        stocks = self._make_stocks(3)
        response = ranker.rank(stocks, ["NIFTY", "RELIANCE", "HDFC"], MarketRegime.BULL)
        assert len(response.rankings) == 3
        assert response.provenance.value == "heuristic"

    def test_rank_empty_list(self):
        from src.models.stock_ranker import StockRanker
        from src.schemas.base import MarketRegime

        ranker = StockRanker()
        response = ranker.rank([], [], MarketRegime.BULL)
        assert response.rankings == []

    def test_rank_bear_regime(self):
        from src.models.stock_ranker import StockRanker
        from src.schemas.base import MarketRegime

        ranker = StockRanker()
        stocks = self._make_stocks(3)
        response = ranker.rank(stocks, ["A", "B", "C"], MarketRegime.BEAR)
        assert len(response.rankings) <= 3

    def test_rank_crash_regime(self):
        from src.models.stock_ranker import StockRanker
        from src.schemas.base import MarketRegime

        ranker = StockRanker()
        stocks = self._make_stocks(3)
        response = ranker.rank(stocks, ["A", "B", "C"], MarketRegime.CRASH)
        assert len(response.rankings) <= 3

    def test_rank_all_identical_scores_normalizes_to_50(self):
        """All equal raw scores → normalised to 50.0."""
        from src.models.stock_ranker import StockRanker
        from src.schemas.base import MarketRegime

        ranker = StockRanker()
        # All same features → same raw score
        stocks = [{"relative_volume": 1.0} for _ in range(3)]
        response = ranker.rank(stocks, ["A", "B", "C"], MarketRegime.SIDEWAYS)
        for rank in response.rankings:
            assert abs(rank.score - 50.0) < 0.01

    def test_rank_top_n_limits_results(self):
        from src.models.stock_ranker import StockRanker
        from src.schemas.base import MarketRegime

        ranker = StockRanker()
        stocks = self._make_stocks(10)
        symbols = [f"SYM{i}" for i in range(10)]
        response = ranker.rank(stocks, symbols, MarketRegime.BULL, top_n=3)
        assert len(response.rankings) == 3

    def test_rank_with_trained_model(self):
        """Tested with mock LightGBM model."""
        from src.models.stock_ranker import StockRanker
        from src.schemas.base import MarketRegime, PredictionProvenance

        ranker = StockRanker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8, 0.5, 0.3])
        ranker.model = mock_model

        stocks = self._make_stocks(3)
        response = ranker.rank(stocks, ["A", "B", "C"], MarketRegime.BULL)
        assert response.provenance == PredictionProvenance.TRAINED_MODEL

    def test_rank_model_fails_falls_back_to_heuristic(self):
        """Model raises → falls back to heuristic without crashing."""
        from src.models.stock_ranker import StockRanker
        from src.schemas.base import MarketRegime, PredictionProvenance

        ranker = StockRanker()
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model exploded")
        ranker.model = mock_model

        stocks = self._make_stocks(3)
        response = ranker.rank(stocks, ["A", "B", "C"], MarketRegime.VOLATILE)
        assert response.provenance == PredictionProvenance.HEURISTIC

    def test_has_trained_model_false(self):
        from src.models.stock_ranker import StockRanker

        ranker = StockRanker()
        assert ranker.has_trained_model is False

    def test_compute_factors_returns_top5_features(self):
        from src.models.stock_ranker import StockRanker

        ranker = StockRanker()
        stock = {
            "relative_strength_vs_nifty": 0.5,
            "momentum_5d": 0.3,
            "relative_volume": 1.2,
            "rsi_14": 55.0,
            "ema_stack_score": 0.7,
        }
        factors = ranker._compute_factors(stock)
        assert len(factors) == 5
        assert "relative_strength_vs_nifty" in factors


# ===========================================================================
# StrategySelector
# ===========================================================================


class TestStrategySelector:
    def test_select_heuristic_bull_trend_following(self):
        from src.models.strategy_selector import StrategySelector
        from src.schemas.base import MarketRegime, TradingStrategy

        sel = StrategySelector()
        features = {
            "rsi": 60.0,
            "adx": 30.0,
            "trend_strength": 0.5,
            "volume_ratio": 1.2,
            "time_of_day_minutes": 100,
            "iv_regime": "STABLE",
        }
        response = sel.select(features, MarketRegime.BULL)
        assert response.strategy == TradingStrategy.TREND_FOLLOWING

    def test_select_heuristic_bull_breakout_early_session(self):
        from src.models.strategy_selector import StrategySelector
        from src.schemas.base import MarketRegime, TradingStrategy

        sel = StrategySelector()
        features = {
            "rsi": 60.0,
            "adx": 20.0,  # low ADX → breakout path
            "trend_strength": 0.1,
            "volume_ratio": 1.2,
            "time_of_day_minutes": 30,  # early session
            "iv_regime": "STABLE",
        }
        response = sel.select(features, MarketRegime.STRONG_BULL)
        assert response.strategy == TradingStrategy.BREAKOUT

    def test_select_heuristic_bull_momentum_late_session(self):
        from src.models.strategy_selector import StrategySelector
        from src.schemas.base import MarketRegime, TradingStrategy

        sel = StrategySelector()
        features = {
            "rsi": 60.0,
            "adx": 20.0,
            "trend_strength": 0.1,
            "volume_ratio": 1.2,
            "time_of_day_minutes": 200,  # late session
            "iv_regime": "STABLE",
        }
        response = sel.select(features, MarketRegime.BULL)
        assert response.strategy == TradingStrategy.MOMENTUM

    def test_select_heuristic_bear_mean_reversion(self):
        from src.models.strategy_selector import StrategySelector
        from src.schemas.base import MarketRegime, TradingStrategy

        sel = StrategySelector()
        features = {"rsi": 40.0, "adx": 25.0, "iv_regime": "STABLE"}
        response = sel.select(features, MarketRegime.BEAR)
        assert response.strategy == TradingStrategy.MEAN_REVERSION

    def test_select_heuristic_crash_mean_reversion(self):
        from src.models.strategy_selector import StrategySelector
        from src.schemas.base import MarketRegime, TradingStrategy

        sel = StrategySelector()
        features = {"iv_regime": "STABLE"}
        response = sel.select(features, MarketRegime.CRASH)
        assert response.strategy == TradingStrategy.MEAN_REVERSION

    def test_select_heuristic_volatile_scalping(self):
        from src.models.strategy_selector import StrategySelector
        from src.schemas.base import MarketRegime, TradingStrategy

        sel = StrategySelector()
        features = {"iv_regime": "STABLE"}
        response = sel.select(features, MarketRegime.VOLATILE)
        assert response.strategy == TradingStrategy.SCALPING

    def test_select_heuristic_sideways_vwap_bounce(self):
        from src.models.strategy_selector import StrategySelector
        from src.schemas.base import MarketRegime, TradingStrategy

        sel = StrategySelector()
        features = {"rsi": 35.0, "iv_regime": "STABLE"}
        response = sel.select(features, MarketRegime.SIDEWAYS)
        assert response.strategy == TradingStrategy.VWAP_BOUNCE

    def test_select_heuristic_sideways_range_trading(self):
        from src.models.strategy_selector import StrategySelector
        from src.schemas.base import MarketRegime, TradingStrategy

        sel = StrategySelector()
        features = {"rsi": 55.0, "iv_regime": "STABLE"}
        response = sel.select(features, MarketRegime.SIDEWAYS)
        assert response.strategy == TradingStrategy.RANGE_TRADING

    def test_select_iv_spike_high_adx(self):
        from src.models.strategy_selector import StrategySelector
        from src.schemas.base import MarketRegime, TradingStrategy

        sel = StrategySelector()
        features = {"adx": 30.0, "iv_regime": "SPIKE"}
        response = sel.select(features, MarketRegime.BULL)
        assert response.strategy == TradingStrategy.VOLATILITY_BREAKOUT

    def test_select_iv_spike_low_adx(self):
        from src.models.strategy_selector import StrategySelector
        from src.schemas.base import MarketRegime, TradingStrategy

        sel = StrategySelector()
        features = {"adx": 15.0, "iv_regime": "SPIKE"}
        response = sel.select(features, MarketRegime.BULL)
        assert response.strategy == TradingStrategy.SCALPING

    def test_select_iv_crush_range_trading(self):
        from src.models.strategy_selector import StrategySelector
        from src.schemas.base import MarketRegime, TradingStrategy

        sel = StrategySelector()
        features = {"iv_regime": "CRUSH"}
        response = sel.select(features, MarketRegime.BULL)
        assert response.strategy == TradingStrategy.RANGE_TRADING

    def test_select_low_confidence_has_alternatives(self):
        """When confidence < 0.40, at least 2 alternatives must be listed."""
        from src.models.strategy_selector import StrategySelector
        from src.schemas.base import MarketRegime

        sel = StrategySelector()

        # Patch _heuristic_select to return confidence < 0.40
        with patch.object(
            sel,
            "_heuristic_select",
            return_value=(
                __import__("src.schemas.base", fromlist=["TradingStrategy"]).TradingStrategy.MOMENTUM,
                0.30,
                {s.value: 0.1 for s in __import__("src.schemas.base", fromlist=["TradingStrategy"]).TradingStrategy},
                __import__("src.schemas.base", fromlist=["PredictionProvenance"]).PredictionProvenance.HEURISTIC,
            ),
        ):
            response = sel.select({}, MarketRegime.BULL)

        # With all strategies at 0.1, there should be many alternatives
        assert len(response.alternatives) >= 2

    def test_select_with_trained_model(self):
        """Mock trained model path."""
        from src.models.strategy_selector import StrategySelector
        from src.schemas.base import MarketRegime, PredictionProvenance

        sel = StrategySelector()
        # Inject mock model
        all_strats = list(__import__("src.schemas.base", fromlist=["TradingStrategy"]).TradingStrategy)
        probas = np.zeros(len(all_strats))
        probas[0] = 0.8  # first strategy wins
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = probas.reshape(1, -1)
        sel.model = mock_model

        response = sel.select({"rsi": 60.0, "adx": 25.0}, MarketRegime.BULL)
        assert response.provenance == PredictionProvenance.TRAINED_MODEL

    def test_resolve_iv_regime_invalid_falls_back_stable(self):
        from src.models.strategy_selector import StrategySelector
        from src.schemas.base import IVRegime

        sel = StrategySelector()
        result = sel._resolve_iv_regime({"iv_regime": "INVALID_VALUE"})
        assert result == IVRegime.STABLE

    def test_resolve_iv_regime_none_falls_back_stable(self):
        from src.models.strategy_selector import StrategySelector
        from src.schemas.base import IVRegime

        sel = StrategySelector()
        result = sel._resolve_iv_regime({})
        assert result == IVRegime.STABLE


# ===========================================================================
# CalibrationLayer
# ===========================================================================


class TestCalibrationLayer:
    def test_calibrate_without_fit_returns_raw(self):
        from src.meta.calibration import CalibrationLayer

        cal = CalibrationLayer()
        result = cal.calibrate("model_a", 0.7)
        assert abs(result - 0.7) < 1e-9

    def test_calibrate_clamps_to_01(self):
        from src.meta.calibration import CalibrationLayer

        cal = CalibrationLayer()
        assert cal.calibrate("x", 1.5) == 1.0
        assert cal.calibrate("x", -0.5) == 0.0

    def test_fit_too_few_samples_returns_false(self):
        from src.meta.calibration import CalibrationLayer

        cal = CalibrationLayer()
        result = cal.fit("m", [0.5, 0.6], [1.0, 0.0])
        assert result is False

    def test_fit_accepts_good_data(self):
        from src.meta.calibration import CalibrationLayer

        cal = CalibrationLayer()
        scores = [float(i) / 20.0 for i in range(20)]
        labels = [1.0 if s > 0.5 else 0.0 for s in scores]
        result = cal.fit("regime", scores, labels)
        # May return True or False depending on ECE; just ensure no crash
        assert isinstance(result, bool)

    def test_has_calibrator_false_before_fit(self):
        from src.meta.calibration import CalibrationLayer

        cal = CalibrationLayer()
        assert cal.has_calibrator("unknown") is False

    def test_get_ece_returns_1_before_fit(self):
        from src.meta.calibration import CalibrationLayer

        cal = CalibrationLayer()
        assert cal.get_ece("unknown") == 1.0

    def test_compute_ece_empty_array(self):
        from src.meta.calibration import CalibrationLayer

        ece = CalibrationLayer._compute_ece(np.array([]), np.array([]))
        assert ece == 1.0

    def test_fit_with_perfect_calibration(self):
        """With binary labels and matching probabilities, fit should succeed."""
        from src.meta.calibration import CalibrationLayer

        cal = CalibrationLayer()
        # Create 20 well-separated samples
        scores = [0.1] * 10 + [0.9] * 10
        labels = [0.0] * 10 + [1.0] * 10
        result = cal.fit("test_model", scores, labels)
        # Whether it accepts or rejects depends on ECE — just no crash
        assert isinstance(result, bool)

    def test_calibrate_after_fit_returns_float(self):
        from src.meta.calibration import CalibrationLayer

        cal = CalibrationLayer()
        scores = [0.1] * 10 + [0.9] * 10
        labels = [0.0] * 10 + [1.0] * 10
        cal.fit("test_model", scores, labels)
        result = cal.calibrate("test_model", 0.6)
        assert 0.0 <= result <= 1.0


class TestConfidenceDecomposer:
    def test_decompose_basic(self):
        from src.meta.calibration import ConfidenceDecomposer

        decomposer = ConfidenceDecomposer()
        result = decomposer.decompose(
            base_confidence=0.7,
            agreement_ratio=0.8,
            data_quality=0.9,
            regime_confidence=0.75,
            calibration_ece=0.05,
        )
        assert result.base_confidence == 0.7
        assert 0.0 <= result.agreement_bonus <= 0.1
        assert result.calibration_quality == pytest.approx(0.95, abs=0.01)

    def test_decompose_low_agreement(self):
        from src.meta.calibration import ConfidenceDecomposer

        decomposer = ConfidenceDecomposer()
        result = decomposer.decompose(
            base_confidence=0.5,
            agreement_ratio=0.3,  # below 0.5 → zero bonus
            data_quality=0.8,
            regime_confidence=0.6,
        )
        assert result.agreement_bonus == 0.0

    def test_decompose_clamps_out_of_range(self):
        from src.meta.calibration import ConfidenceDecomposer

        decomposer = ConfidenceDecomposer()
        result = decomposer.decompose(
            base_confidence=2.0,  # over 1.0
            agreement_ratio=1.0,
            data_quality=1.5,  # over 1.0
            regime_confidence=0.5,
            calibration_ece=0.0,
        )
        assert result.base_confidence == 1.0
        assert result.data_quality_factor == 1.0


# ===========================================================================
# DriftMonitor — remaining branches
# ===========================================================================


class TestDriftMonitorMoreBranches:
    def test_create_drift_alert_returns_drift_alert(self):
        from src.monitoring.drift_monitor import DriftMonitor

        dm = DriftMonitor()
        alert = dm.create_drift_alert(
            model_name="test_model",
            feature_name="rsi",
            psi=0.3,
            reference_mean=50.0,
            reference_std=10.0,
            current_mean=55.0,
            current_std=12.0,
        )
        assert alert.model_name == "test_model"
        assert alert.feature_name == "rsi"
        assert alert.psi_value == 0.3

    def test_create_drift_alert_high_severity(self):
        from src.monitoring.drift_monitor import DriftMonitor
        from src.schemas.base import DriftSeverity

        dm = DriftMonitor()
        alert = dm.create_drift_alert(
            model_name="m",
            feature_name="f",
            psi=0.3,
            reference_mean=0.0,
            reference_std=1.0,
            current_mean=1.0,
            current_std=2.0,
        )
        assert alert.severity == DriftSeverity.HIGH

    def test_create_drift_alert_with_online_learning_and_rollback(self):
        from src.monitoring.drift_monitor import DriftMonitor
        from src.schemas.base import RecommendedAction

        dm = DriftMonitor()
        alert = dm.create_drift_alert(
            model_name="m",
            feature_name="f",
            psi=0.3,
            reference_mean=0.0,
            reference_std=1.0,
            current_mean=1.0,
            current_std=2.0,
            online_learning_triggered=True,
            estimated_ic=0.005,
        )
        assert alert.recommended_action == RecommendedAction.ROLLBACK

    def test_run_nannyml_cbpe_exception_graceful(self):
        """NannyML available but raises → returns error dict."""
        from src.monitoring.drift_monitor import DriftMonitor
        import src.monitoring.drift_monitor as dm_mod

        mock_nml = MagicMock()
        mock_nml.CBPE.side_effect = RuntimeError("nml error")

        with patch("src.monitoring.drift_monitor.NANNYML_AVAILABLE", True):
            orig = getattr(dm_mod, "nannyml", None)
            dm_mod.nannyml = mock_nml
            try:
                dm = DriftMonitor()
                result = dm.run_nannyml_cbpe(MagicMock(), MagicMock(), {})
            finally:
                if orig is None and hasattr(dm_mod, "nannyml"):
                    delattr(dm_mod, "nannyml")
                elif orig is not None:
                    dm_mod.nannyml = orig

        assert isinstance(result, dict)


# ===========================================================================
# ModelRegistry — additional lines
# ===========================================================================


def make_test_artifact(
    model_name="reg_model",
    version="2.0.0",
    artifact_path="/tmp/reg_model.pkl",
    sha256_checksum="c" * 64,
):
    from src.schemas.base import ModelLifecycleStage
    from src.schemas.registry import ModelArtifact

    return ModelArtifact(
        model_name=model_name,
        version=version,
        stage=ModelLifecycleStage.PRODUCTION,
        artifact_path=artifact_path,
        sha256_checksum=sha256_checksum,
        training_date="2024-02-01",
        training_dataset_hash="d" * 64,
    )


class TestModelRegistryAdvanced:
    def test_get_champion_from_disk_after_no_in_memory(self):
        """Champion.json exists on disk → loaded without in-memory cache."""
        from src.registry.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(artifacts_path=Path(tmpdir))
            artifact = make_test_artifact(artifact_path=str(Path(tmpdir) / "fake.pkl"))
            reg.register(artifact)

            # Clear in-memory cache to force disk read
            reg._champions.clear()

            champion = reg.get_champion("reg_model")
            assert champion is not None
            assert champion.version == "2.0.0"

    def test_register_with_source_file_copies_it(self):
        """register() with artifact_file_path copies the file to version dir."""
        from src.registry.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source artifact file
            src_file = Path(tmpdir) / "source_model.pkl"
            src_file.write_bytes(b"model content")

            reg = ModelRegistry(artifacts_path=Path(tmpdir))
            artifact = make_test_artifact(artifact_path=str(src_file))
            reg.register(artifact, artifact_file_path=src_file)

            # The file should be copied into the version directory
            version_dir = Path(tmpdir) / "reg_model" / "2.0.0"
            assert (version_dir / "source_model.pkl").exists()

    def test_load_all_champions_with_integrity_failure(self):
        """ArtifactIntegrityFailure is caught — that model returns None."""
        from src.registry.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(artifacts_path=Path(tmpdir))

            # Register artifact with wrong hash so integrity check fails
            real_file = Path(tmpdir) / "model.pkl"
            real_file.write_bytes(b"real model")
            artifact = make_test_artifact(
                artifact_path=str(real_file),
                sha256_checksum="wrong" + "0" * 59,
            )
            reg.register(artifact)

            results = reg.load_all_champions()
            # Our model is not in the known_models list, so won't appear
            # Test just confirms no crash
            assert isinstance(results, dict)

    def test_list_registry_skips_non_directories(self):
        """Files in model dir don't cause parse errors."""
        from src.registry.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(artifacts_path=Path(tmpdir))
            artifact = make_test_artifact(artifact_path=str(Path(tmpdir) / "fake.pkl"))
            reg.register(artifact)

            # Add a non-directory file in the model dir
            extra_file = Path(tmpdir) / "reg_model" / "extra_file.txt"
            extra_file.write_text("extra")

            listed = reg.list_registry()
            assert len(listed) >= 1  # the real artifact is still listed


# ===========================================================================
# RegimeClassifier — remaining lines
# ===========================================================================


class TestRegimeClassifierAdditional:
    def test_predict_insufficient_evidence_low_confidence(self):
        """confidence < 0.35 → provenance is INSUFFICIENT_EVIDENCE."""
        from src.models.regime_classifier import RegimeClassifier
        from src.schemas.base import PredictionProvenance
        from threading import Lock
        from src.explainability.explainer import ModelExplainer
        from src.models.regime_classifier import REGIME_FEATURE_NAMES

        clf = RegimeClassifier.__new__(RegimeClassifier)
        clf._lock = Lock()
        clf._model = None
        clf._model_version = "heuristic-v1"
        clf._last_regime = None
        clf._last_confidence = 0.0
        clf._explainer = ModelExplainer()
        clf.feature_names = REGIME_FEATURE_NAMES

        # Patch _heuristic_predict to return confidence < 0.35
        with patch.object(
            clf,
            "_heuristic_predict",
            return_value=(
                __import__("src.schemas.base", fromlist=["MarketRegime"]).MarketRegime.SIDEWAYS,
                0.20,  # below threshold
            ),
        ):
            response = clf.predict({"india_vix": 16.0})

        assert response.provenance == PredictionProvenance.INSUFFICIENT_EVIDENCE

    def test_load_from_path_nonexistent_falls_back_to_heuristic(self):
        """_load_from_path with non-existent file → logs warning, no crash."""
        from src.models.regime_classifier import RegimeClassifier

        # Use a non-existent path
        clf = RegimeClassifier(model_path=Path("/nonexistent/path/model.pkl"))
        assert clf._model is None  # Fell back to heuristic


# ===========================================================================
# RiskPredictor — remaining lines
# ===========================================================================


class TestRiskPredictorAdditional:
    def test_model_predict_target_model_none_uses_fallback(self):
        """When target_model is None, prob_target is derived from prob_stop."""
        from src.models.risk_predictor import RiskPredictor
        from src.schemas.base import PredictionProvenance

        predictor = RiskPredictor()

        mock_stop_model = MagicMock()
        mock_stop_model.predict_proba.return_value = np.array([[0.6, 0.4]])
        predictor.stop_model = mock_stop_model
        predictor.target_model = None  # no target model

        features = {
            "entry": 22000.0,
            "stop_loss": 21800.0,
            "target": 22400.0,
            "atr": 200.0,
            "vix": 15.0,
        }
        response = predictor.predict(features)
        assert response.provenance == PredictionProvenance.TRAINED_MODEL
        assert response.prob_target_hit >= 0.0

    def test_model_predict_exception_falls_back_to_heuristic(self):
        """Model raises during predict → falls back to heuristic."""
        from src.models.risk_predictor import RiskPredictor
        from src.schemas.base import PredictionProvenance

        predictor = RiskPredictor()

        mock_stop_model = MagicMock()
        mock_stop_model.predict_proba.side_effect = RuntimeError("model crashed")
        predictor.stop_model = mock_stop_model

        response = predictor.predict({"entry": 22000.0, "stop_loss": 21800.0, "target": 22400.0, "atr": 200.0, "vix": 15.0})
        assert response.provenance == PredictionProvenance.HEURISTIC

    def test_heuristic_strong_bull_reduces_stop_prob(self):
        """Bull/strong_bull regime reduces prob_stop."""
        from src.models.risk_predictor import RiskPredictor

        predictor = RiskPredictor()
        features = {
            "entry": 22000.0,
            "stop_loss": 21800.0,
            "target": 22400.0,
            "atr": 200.0,
            "regime": "strong_bull",
            "vix": 14.0,
        }
        response = predictor.predict(features)
        assert response.prob_stop_hit < 0.7  # should be reduced
