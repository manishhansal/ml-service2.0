"""
tests/test_sprint3_alpha_signal.py
------------------------------------
Tests for M8-M11 new components:
  - AlphaSpecialist architecture (NEW-P2-001)
  - SignalPromotionEngine (NEW-P2-004)
  - ResearchTrialLedger de-duplication (NEW-P1-008)
"""
from __future__ import annotations

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# AlphaSpecialist / AlphaSpecialistRegistry (NEW-P2-001)
# ──────────────────────────────────────────────────────────────────────────────

class TestAlphaSpecialists:
    def _trend_features(self) -> dict:
        return {
            "ret_1": 0.01, "ret_5": 0.035, "ret_10": 0.04, "ret_20": 0.05,
            "ret_60": 0.08, "ret_3": 0.02, "vol_20": 0.012,
            "bb_zscore_20": 0.5, "rsi_14": 62.0, "price_zscore_20": 0.6,
            "trend_direction": 1.0, "trend_strength": 35.0,
            "vol_expanding": 1.0, "vol_ratio": 1.1, "rel_volume_20": 1.2,
            "gap_magnitude": 0.003, "ema_spread": 0.02,
            "vol_regime_zscore": 0.5, "vol_regime_pctile": 0.6,
            "vol_of_vol_20": 0.001, "parkinson_vol": 0.012,
        }

    def _mean_rev_features(self) -> dict:
        return {
            "ret_1": -0.02, "ret_5": -0.04, "ret_10": -0.03, "ret_20": -0.02,
            "ret_60": 0.01, "ret_3": -0.025, "vol_20": 0.012,
            "bb_zscore_20": -2.5, "rsi_14": 25.0, "price_zscore_20": -2.2,
            "trend_direction": -0.1, "trend_strength": 15.0,
            "vol_expanding": 0.0, "vol_ratio": 0.9, "rel_volume_20": 0.8,
            "gap_magnitude": 0.001, "ema_spread": -0.01,
            "vol_regime_zscore": -0.3, "vol_regime_pctile": 0.3,
            "vol_of_vol_20": 0.0005, "parkinson_vol": 0.010,
        }

    def test_momentum_alpha_produces_signal_in_trend(self):
        from src.alpha.base import MomentumAlpha
        alpha = MomentumAlpha()
        sig = alpha.generate(self._trend_features(), symbol="NIFTY", regime="TREND_UP")
        assert sig is not None, "MomentumAlpha must produce a signal in TREND_UP regime"
        assert sig.direction == 1  # uptrend → long
        assert sig.confidence > 0.5

    def test_momentum_alpha_no_signal_in_mean_reverting(self):
        from src.alpha.base import MomentumAlpha
        alpha = MomentumAlpha()
        sig = alpha.generate(self._trend_features(), symbol="NIFTY", regime="MEAN_REVERTING")
        # In MEAN_REVERTING, regime_fit is very low → should return None
        assert sig is None, "MomentumAlpha must not produce signal in MEAN_REVERTING regime"

    def test_mean_reversion_alpha_produces_signal_in_range(self):
        from src.alpha.base import MeanReversionAlpha
        alpha = MeanReversionAlpha()
        sig = alpha.generate(self._mean_rev_features(), symbol="RELIANCE", regime="RANGE")
        assert sig is not None, "MeanReversionAlpha must produce signal in RANGE regime"
        assert sig.direction == 1  # oversold bb → long

    def test_mean_reversion_alpha_no_signal_in_trend(self):
        from src.alpha.base import MeanReversionAlpha
        alpha = MeanReversionAlpha()
        sig = alpha.generate(self._trend_features(), symbol="RELIANCE", regime="TREND_UP")
        # Weak bb_zscore in trend regime → no signal
        assert sig is None or sig.confidence < 0.5

    def test_alpha_signal_fields_complete(self):
        from src.alpha.base import MomentumAlpha
        alpha = MomentumAlpha()
        sig = alpha.generate(self._trend_features(), regime="TREND_UP")
        assert sig is not None
        assert 0.0 <= sig.confidence <= 1.0
        assert sig.direction in (-1, 0, 1)
        assert sig.expected_holding_period > 0
        assert sig.risk_estimate >= 0
        assert -1.0 <= sig.alpha_score <= 1.0

    def test_alpha_signal_to_dict(self):
        from src.alpha.base import MomentumAlpha
        alpha = MomentumAlpha()
        sig = alpha.generate(self._trend_features(), regime="TREND_UP")
        assert sig is not None
        d = sig.to_dict()
        for key in ["alpha_name", "direction", "expected_return", "confidence",
                    "alpha_score", "regime_fit", "reason"]:
            assert key in d

    def test_alpha_signal_expected_net_return(self):
        from src.alpha.base import MomentumAlpha
        alpha = MomentumAlpha()
        sig = alpha.generate(self._trend_features(), regime="TREND_UP")
        if sig:
            # Net return should be gross minus cost
            assert sig.expected_net_return <= sig.expected_return

    def test_registry_generates_multiple_signals(self):
        from src.alpha.base import AlphaSpecialistRegistry
        registry = AlphaSpecialistRegistry.default()
        sigs = registry.generate_all(self._trend_features(), symbol="NIFTY", regime="TREND_UP")
        assert len(sigs) >= 1, "At least one specialist must produce a signal"

    def test_registry_rank_by_alpha_score(self):
        from src.alpha.base import AlphaSpecialistRegistry
        registry = AlphaSpecialistRegistry.default()
        sigs = registry.generate_all(self._trend_features(), symbol="NIFTY", regime="BREAKOUT")
        if len(sigs) >= 2:
            ranked = registry.rank_signals(sigs)
            scores = [abs(s.alpha_score) for s in ranked]
            assert scores == sorted(scores, reverse=True)

    def test_quarantined_specialist_excluded(self):
        from src.alpha.base import AlphaSpecialistRegistry, AlphaStatus
        registry = AlphaSpecialistRegistry.default()
        registry.set_status("MOMENTUM_ALPHA", AlphaStatus.QUARANTINED)
        sigs = registry.generate_all(self._trend_features(), regime="TREND_UP")
        names = [s.alpha_name for s in sigs]
        assert "MOMENTUM_ALPHA" not in names

    def test_no_signal_on_nan_features(self):
        from src.alpha.base import MomentumAlpha
        alpha = MomentumAlpha()
        # Key momentum features are NaN
        sig = alpha.generate({"ret_5": float("nan"), "ret_20": float("nan")})
        assert sig is None, "Must return None when key features are NaN"

    def test_regime_compatibility_values(self):
        from src.alpha.base import MomentumAlpha, MeanReversionAlpha
        mom = MomentumAlpha()
        rev = MeanReversionAlpha()
        # Momentum should be most compatible with TREND_UP
        assert mom.regime_compatibility("TREND_UP") > mom.regime_compatibility("RANGE")
        # MeanRev should be most compatible with RANGE
        assert rev.regime_compatibility("RANGE") > rev.regime_compatibility("TREND_UP")

    def test_breakout_alpha_triggers_on_high_volume(self):
        from src.alpha.base import BreakoutAlpha
        alpha = BreakoutAlpha()
        features = {
            "vol_expanding": 1.0, "vol_ratio": 1.8,
            "rel_volume_20": 2.5, "ret_1": 0.02,
            "gap_magnitude": 0.01, "ema_spread": 0.03,
        }
        sig = alpha.generate(features, regime="BREAKOUT")
        assert sig is not None
        assert sig.direction == 1  # positive ret → long

    def test_volatility_alpha_fades_extreme_vol(self):
        from src.alpha.base import VolatilityAlpha
        alpha = VolatilityAlpha()
        features = {
            "vol_regime_zscore": 3.0,  # extreme high vol
            "vol_regime_pctile": 0.95,
            "vol_of_vol_20": 0.005,
            "vol_20": 0.03,
            "parkinson_vol": 0.03,
        }
        sig = alpha.generate(features, regime="HIGH_VOLATILITY")
        assert sig is not None
        assert sig.direction == -1  # fade high vol (expect contraction)


# ──────────────────────────────────────────────────────────────────────────────
# SignalPromotionEngine (NEW-P2-004)
# ──────────────────────────────────────────────────────────────────────────────

class TestSignalPromotionEngine:
    """Tests for SignalPromotionEngine (mandate §26)."""

    def _make_trades(self, n: int, mean_ret: float = 0.005, seed: int = 42) -> list[dict]:
        rng = np.random.default_rng(seed)
        regimes = ["TREND_UP", "RANGE", "HIGH_VOLATILITY"]
        return [
            {
                "net_return": mean_ret + rng.normal(0, 0.002),
                "regime": regimes[i % len(regimes)],
                "cost_bps": 27.65,
            }
            for i in range(n)
        ]

    def test_promotes_when_all_gates_pass(self):
        from src.analytics.signal_promotion import (
            SignalPromotionEngine, SignalLifecycleStage, PromotionGateConfig,
        )
        engine = SignalPromotionEngine(
            gate_config=PromotionGateConfig(min_trades=10, min_regimes=2)
        )
        trades = self._make_trades(50, mean_ret=0.008)
        result = engine.evaluate(
            signal_family="MOMENTUM_ALPHA",
            current_stage=SignalLifecycleStage.SHADOW,
            trade_records=trades,
            calibration_ece=0.05,
        )
        assert result.decision == "PROMOTE", (
            f"Should promote but got: {result.decision} — {result.reason}"
        )
        assert result.stage_to == SignalLifecycleStage.PAPER_PRODUCTION

    def test_rejects_insufficient_trades(self):
        from src.analytics.signal_promotion import (
            SignalPromotionEngine, SignalLifecycleStage, PromotionGateConfig,
        )
        engine = SignalPromotionEngine(
            gate_config=PromotionGateConfig(min_trades=50)
        )
        trades = self._make_trades(10, mean_ret=0.01)
        result = engine.evaluate(
            "MOMENTUM_ALPHA",
            SignalLifecycleStage.SHADOW,
            trades,
        )
        assert result.decision == "REJECT"
        assert "G1_TRADE_COUNT" in result.gate_results
        assert "FAIL" in result.gate_results["G1_TRADE_COUNT"]

    def test_rejects_negative_expectancy(self):
        from src.analytics.signal_promotion import (
            SignalPromotionEngine, SignalLifecycleStage, PromotionGateConfig,
        )
        engine = SignalPromotionEngine(
            gate_config=PromotionGateConfig(min_trades=10)
        )
        trades = self._make_trades(50, mean_ret=-0.003)
        result = engine.evaluate(
            "MOMENTUM_ALPHA",
            SignalLifecycleStage.SHADOW,
            trades,
        )
        assert result.decision == "REJECT"
        assert "G2_NET_EXPECTANCY" in result.gate_results

    def test_rejects_poor_calibration(self):
        from src.analytics.signal_promotion import (
            SignalPromotionEngine, SignalLifecycleStage, PromotionGateConfig,
        )
        engine = SignalPromotionEngine(
            gate_config=PromotionGateConfig(min_trades=10, max_ece=0.05)
        )
        trades = self._make_trades(50, mean_ret=0.008)
        result = engine.evaluate(
            "MOMENTUM_ALPHA",
            SignalLifecycleStage.SHADOW,
            trades,
            calibration_ece=0.15,  # above max_ece=0.05
        )
        assert result.decision == "REJECT"
        assert "G3_CALIBRATION" in result.gate_results

    def test_promotion_stage_progression(self):
        from src.analytics.signal_promotion import (
            SignalPromotionEngine, SignalLifecycleStage,
        )
        engine = SignalPromotionEngine._next_stage
        assert engine(SignalLifecycleStage.SHADOW) == SignalLifecycleStage.PAPER_PRODUCTION
        assert engine(SignalLifecycleStage.PAPER_PRODUCTION) == SignalLifecycleStage.LIMITED_PRODUCTION
        assert engine(SignalLifecycleStage.LIMITED_PRODUCTION) == SignalLifecycleStage.PRODUCTION

    def test_demotion_on_negative_rolling_performance(self):
        from src.analytics.signal_promotion import (
            SignalPromotionEngine, SignalLifecycleStage,
        )
        engine = SignalPromotionEngine()
        recent = [{"net_return": -0.003, "regime": "RANGE"} for _ in range(25)]
        result = engine.evaluate_demotion(
            "MOMENTUM_ALPHA",
            SignalLifecycleStage.PRODUCTION,
            recent,
        )
        assert result.decision == "DEMOTE"
        assert result.stage_to == SignalLifecycleStage.DEGRADED

    def test_keep_on_positive_rolling_performance(self):
        from src.analytics.signal_promotion import (
            SignalPromotionEngine, SignalLifecycleStage,
        )
        engine = SignalPromotionEngine()
        recent = [{"net_return": 0.005, "regime": "TREND_UP"} for _ in range(20)]
        result = engine.evaluate_demotion(
            "MOMENTUM_ALPHA",
            SignalLifecycleStage.PRODUCTION,
            recent,
        )
        assert result.decision == "KEEP"

    def test_no_trade_records_returns_reject(self):
        from src.analytics.signal_promotion import (
            SignalPromotionEngine, SignalLifecycleStage,
        )
        engine = SignalPromotionEngine()
        result = engine.evaluate("TEST", SignalLifecycleStage.SHADOW, [])
        assert result.decision == "REJECT"
        assert result.reason == "NO_TRADE_RECORDS"

    def test_evidence_has_all_required_fields(self):
        from src.analytics.signal_promotion import (
            SignalPromotionEngine, SignalLifecycleStage,
        )
        engine = SignalPromotionEngine()
        trades = self._make_trades(15)
        result = engine.evaluate("TEST", SignalLifecycleStage.SHADOW, trades)
        d = result.to_dict()
        for key in ["signal_family", "n_trades", "net_expectancy",
                    "decision", "reason", "gate_results", "evaluated_at"]:
            assert key in d


# ──────────────────────────────────────────────────────────────────────────────
# ResearchTrialLedger de-duplication (NEW-P1-008)
# ──────────────────────────────────────────────────────────────────────────────

class TestResearchLedgerDuplication:
    def test_duplicate_blocked(self, tmp_path):
        from src.validation.ledger import ResearchTrialLedger, make_entry
        ledger = ResearchTrialLedger(path=tmp_path / "ledger.jsonl")

        entry1 = make_entry(
            experiment_date="2026-09-27", code_sha="abc", docker_image="img",
            dataset_hash="hash1", dataset_id="ds-1", universe="10-sym", timeframe="1d",
            features="BASE-24", label="triple_barrier", model="logistic",
            hyperparameters={}, cost_bps=10.0, execution_convention="next_open",
            training_period="2021-2024", validation_period="val",
            oos_period="oos", ic_mean=0.03, rank_ic_mean=None,
            net_sharpe=0.5, pbo=0.2, n_oos_windows=5,
            selection_status="EXPLORATORY", rejection_reason="",
            pre_registered=False, experiment_class="EXPLORATORY",
            reason_for_experiment="first run",
        )
        accepted = ledger.check_and_append(entry1)
        assert accepted, "First entry should be accepted"

        # Structurally identical second entry
        entry2 = make_entry(
            experiment_date="2026-09-28", code_sha="def", docker_image="img2",
            dataset_hash="hash2", dataset_id="ds-2",
            universe="10-sym", timeframe="1d",  # same
            features="BASE-24", label="triple_barrier", model="logistic",  # same
            hyperparameters={}, cost_bps=10.0, execution_convention="next_open",  # same
            training_period="2021-2024", validation_period="val",
            oos_period="oos", ic_mean=0.04, rank_ic_mean=None,
            net_sharpe=0.8, pbo=0.1, n_oos_windows=5,
            selection_status="EXPLORATORY", rejection_reason="",
            pre_registered=False, experiment_class="EXPLORATORY",
            reason_for_experiment="duplicate run",
        )
        accepted2 = ledger.check_and_append(entry2)
        assert not accepted2, "Duplicate experiment should be blocked"

        # Only 1 entry should be in the ledger
        assert len(ledger.all_entries()) == 1

    def test_allow_rerun_bypasses_dedup(self, tmp_path):
        from src.validation.ledger import ResearchTrialLedger, make_entry
        ledger = ResearchTrialLedger(path=tmp_path / "ledger.jsonl")

        def _make(reason):
            return make_entry(
                experiment_date="2026-09-27", code_sha="abc", docker_image="img",
                dataset_hash="h", dataset_id="ds-1", universe="sym", timeframe="1d",
                features="BASE", label="triple", model="logistic",
                hyperparameters={}, cost_bps=10.0, execution_convention="next_open",
                training_period="p", validation_period="v", oos_period="o",
                ic_mean=None, rank_ic_mean=None, net_sharpe=None, pbo=None,
                n_oos_windows=5, selection_status="EXPLORATORY", rejection_reason="",
                pre_registered=False, experiment_class="EXPLORATORY",
                reason_for_experiment=reason,
            )

        ledger.check_and_append(_make("first"))
        accepted = ledger.check_and_append(_make("explicit rerun"), allow_rerun=True)
        assert accepted, "allow_rerun=True must bypass de-duplication"
        assert len(ledger.all_entries()) == 2

    def test_research_adjusted_min_ic_increases_with_experiments(self, tmp_path):
        from src.validation.ledger import ResearchTrialLedger, make_entry
        ledger = ResearchTrialLedger(path=tmp_path / "ledger.jsonl")

        base_ic = ledger.research_adjusted_min_ic(0.02)
        assert base_ic == 0.02, "With 0 experiments, no adjustment"

        # Add 50 experiments with distinct hypotheses
        for i in range(50):
            e = make_entry(
                experiment_date="2026-09-27", code_sha="abc", docker_image="img",
                dataset_hash=f"h{i}", dataset_id=f"ds-{i}",
                universe=f"sym-{i}", timeframe="1d",
                features=f"F-{i}", label=f"L-{i}", model=f"M-{i}",
                hyperparameters={}, cost_bps=10.0, execution_convention="next_open",
                training_period="p", validation_period="v", oos_period="o",
                ic_mean=None, rank_ic_mean=None, net_sharpe=None, pbo=None,
                n_oos_windows=5, selection_status="EXPLORATORY", rejection_reason="",
                pre_registered=False, experiment_class="EXPLORATORY",
                reason_for_experiment=f"exp {i}",
            )
            ledger.append(e)

        adjusted = ledger.research_adjusted_min_ic(0.02)
        assert adjusted > 0.02, "Min IC should increase as more experiments are run"
        assert adjusted < 0.10, "Adjustment should be reasonable, not extreme"

    def test_fingerprint_different_for_different_configs(self, tmp_path):
        from src.validation.ledger import ResearchTrialLedger, make_entry
        ledger = ResearchTrialLedger(path=tmp_path / "ledger.jsonl")

        def _make(model):
            return make_entry(
                experiment_date="2026-09-27", code_sha="abc", docker_image="img",
                dataset_hash="h", dataset_id="ds-1", universe="sym", timeframe="1d",
                features="BASE", label="triple", model=model,
                hyperparameters={}, cost_bps=10.0, execution_convention="next_open",
                training_period="p", validation_period="v", oos_period="o",
                ic_mean=None, rank_ic_mean=None, net_sharpe=None, pbo=None,
                n_oos_windows=5, selection_status="EXPLORATORY", rejection_reason="",
                pre_registered=False, experiment_class="EXPLORATORY",
                reason_for_experiment="test",
            )

        e_lr = _make("logistic")
        e_lgb = _make("lightgbm")
        # Different models → different fingerprint → both accepted
        assert ledger._hypothesis_fingerprint(e_lr) != ledger._hypothesis_fingerprint(e_lgb)
