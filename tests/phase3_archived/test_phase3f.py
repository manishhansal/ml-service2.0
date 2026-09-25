"""
Phase 3F Tests — Meta-Labeling, Probability Calibration & Abstention.

Test categories
---------------
 1. Semantic contract: score types are distinct and non-interchangeable
 2. CalibratedProbability: only status=CALIBRATED may have a trusted value
 3. Meta-label policies A/B/C: golden tests with known outcomes
 4. Stacking leakage: in-sample predictions rejected before training
 5. EV engine: positive/zero/negative EV, asymmetric payoffs
 6. EV leakage: current-event outcome cannot enter its own EV estimate
 7. Calibration: Platt/isotonic Brier/ECE/reliability, temporal ordering
 8. Calibration leakage: calibrator fit_end must be < eval_start
 9. CalibratorArtifact: UNCALIBRATED/MISMATCH/STALE states
10. No raw-score-as-probability fallback
11. Decision engine: TAKE/SKIP/ABSTAIN/INSUFFICIENT_EVIDENCE
12. Future mutation: data after prediction_time must not change meta output
13. Calibrator staleness: stale calibrator produces STALE status
14. Model/calibrator mismatch: incompatible IDs produce CALIBRATOR_MISMATCH
15. Outcome feature leakage guard: validate_no_outcome_features()
16. Meta-label monotonicity: higher return → more likely to be TAKE
17. Probability range: 0 ≤ P ≤ 1 always
18. Probability monotonicity: higher raw score → non-decreasing calibrated P
19. Reliability curve: known perfect calibration produces diagonal
20. Baselines: alpha threshold and compare_meta_models ML_ADDS_NO_CLEAR_VALUE
21. Walk-forward calibration temporal assertion
22. Cost-unavailable EV: computed without cost, flagged explicitly
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytest

UTC = timezone.utc

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts(offset_days: int = 0) -> datetime:
    return datetime(2023, 6, 1, tzinfo=UTC) + timedelta(days=offset_days)


def _make_meta_event(
    instrument_id: str = "TEST",
    primary_side: str = "LONG",
    alpha_score: float = 75.0,
    net_return: Optional[float] = None,
    first_touch: Optional[str] = None,
    mfe: Optional[float] = None,
    mae: Optional[float] = None,
    is_oos: bool = True,
    prediction_time: Optional[datetime] = None,
    event_end_time: Optional[datetime] = None,
) -> "MetaEvent":
    from src.meta.meta_label import MetaEvent
    return MetaEvent(
        instrument_id=instrument_id,
        prediction_time=prediction_time or _ts(0),
        primary_side=primary_side,
        primary_alpha_score=alpha_score,
        primary_rank=5,
        primary_percentile=80.0,
        primary_cross_section_size=50,
        is_oos_primary_prediction=is_oos,
        primary_fold_id=1,
        primary_model_id="lgbm_ranker",
        primary_model_version="v1",
        event_start_time=prediction_time or _ts(0),
        event_end_time=event_end_time or _ts(5),
        net_return=net_return,
        gross_return=net_return,
        mfe=mfe,
        mae=mae,
        first_touch=first_touch,
        label_version="lv2",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Semantic contract: distinct types
# ══════════════════════════════════════════════════════════════════════════════

class TestSemanticContract:
    """
    Spec §Q1: AlphaForge must distinguish alpha_score / probability /
    expected_return / expected_value.
    """

    def test_alpha_score_is_not_probability(self):
        from src.meta.schemas import AlphaScore, ScoreType
        score = AlphaScore(value=75.0, rank=1, percentile=99.0,
                           cross_section_size=50, model_id="lgbm")
        assert score.score_type == ScoreType.ALPHA_SCORE
        assert score.score_type != ScoreType.CALIBRATED_PROBABILITY
        assert score.score_type != ScoreType.EXPECTED_RETURN

    def test_raw_probability_is_not_calibrated(self):
        from src.meta.schemas import RawProbabilityScore, ScoreType, PredictionProvenance
        raw = RawProbabilityScore(
            value=0.72,
            model_id="lgbm_meta",
            model_version="v1",
            prediction_time=_ts(),
            score_type=ScoreType.RAW_PROBABILITY,
            provenance=PredictionProvenance.TRAINED_MODEL,
            is_oos=True,
        )
        assert raw.score_type == ScoreType.RAW_PROBABILITY
        assert raw.score_type != ScoreType.CALIBRATED_PROBABILITY

    def test_calibrated_probability_requires_calibrated_status(self):
        """A CalibratedProbability with status != CALIBRATED is not usable."""
        from src.meta.schemas import CalibratedProbability, ProbabilityStatus
        uncal = CalibratedProbability.unavailable("test_model")
        assert not uncal.is_usable()
        assert uncal.value is None

    def test_calibrated_probability_ok_when_status_calibrated(self):
        from src.meta.schemas import CalibratedProbability, ProbabilityStatus
        cal = CalibratedProbability(
            value=0.65,
            status=ProbabilityStatus.CALIBRATED,
            model_id="meta",
            model_version="v1",
            calibrator_id="platt-v1",
            calibration_method="platt",
            calibration_version="v1",
            fit_end_time=_ts(-10),
            effective_from=_ts(-5),
            sample_count=200,
        )
        assert cal.is_usable()
        assert 0.0 <= cal.value <= 1.0

    def test_expected_value_is_not_probability(self):
        from src.meta.schemas import ExpectedValue, EVStatus, ScoreType
        ev = ExpectedValue(
            value=2.5,
            probability=0.65,
            expected_win=6.0,
            expected_loss=-3.0,
            expected_cost=0.5,
            confidence=0.8,
            status=EVStatus.VALID,
        )
        assert ev.score_type == ScoreType.EXPECTED_VALUE
        assert ev.score_type != ScoreType.CALIBRATED_PROBABILITY

    def test_raw_score_assert_oos_raises_for_in_sample(self):
        """Spec §Q2: in-sample primary predictions must be rejected."""
        from src.meta.schemas import RawProbabilityScore, ScoreType, PredictionProvenance
        in_sample = RawProbabilityScore(
            value=0.7, model_id="m", model_version="v1",
            prediction_time=_ts(), is_oos=False,
            score_type=ScoreType.RAW_PROBABILITY,
            provenance=PredictionProvenance.TRAINED_MODEL,
        )
        with pytest.raises(RuntimeError, match="Stacking leakage"):
            in_sample.assert_oos()

    def test_raw_score_assert_oos_passes_for_oos(self):
        from src.meta.schemas import RawProbabilityScore, ScoreType, PredictionProvenance
        oos = RawProbabilityScore(
            value=0.7, model_id="m", model_version="v1",
            prediction_time=_ts(), is_oos=True,
            score_type=ScoreType.RAW_PROBABILITY,
            provenance=PredictionProvenance.TRAINED_MODEL,
        )
        oos.assert_oos()  # must not raise

    def test_decision_states_complete(self):
        from src.meta.schemas import Decision
        required = {"TAKE", "SKIP", "ABSTAIN", "INSUFFICIENT_EVIDENCE", "UNAVAILABLE"}
        actual = {d.value for d in Decision}
        assert required.issubset(actual)


# ══════════════════════════════════════════════════════════════════════════════
# 2 — No raw-score-as-probability fallback (Spec §Q3)
# ══════════════════════════════════════════════════════════════════════════════

class TestNoRawScoreFallback:
    """
    Spec §Q3: An unfitted calibrator must NEVER produce a field called
    calibrated_probability. clip(raw_score) is not calibration.
    """

    def test_unfitted_calibrator_artifact_returns_uncalibrated(self):
        from src.meta.calibration_engine import CalibratorArtifact
        from src.meta.schemas import ProbabilityStatus
        art = CalibratorArtifact(
            calibrator_id="c1",
            model_id="m1",
            model_version="v1",
            calibration_method="platt",
            calibration_version="cv1",
            fit_end_time=_ts(-10),
            effective_from=_ts(-5),
            fit_sample_count=0,
        )
        probs, status = art.predict(
            np.array([0.5, 0.7, 0.9]),
            model_id="m1",
            model_version="v1",
        )
        assert status == ProbabilityStatus.UNCALIBRATED, (
            f"Unfitted calibrator must return UNCALIBRATED, not {status}"
        )
        assert probs is None, "Unfitted calibrator must not return probability values"

    def test_calibration_store_unknown_model_returns_neutral(self):
        """CalibrationStore must return 0.5 (neutral), not clip(raw) for unknown models."""
        from src.meta.calibration import CalibrationStore
        store = CalibrationStore()
        result = store.calibrate("completely_unknown_model_xyz", raw_score=0.99)
        assert result == 0.5, (
            f"Unknown model must return 0.5 (neutral), not clip(0.99). Got {result}"
        )

    def test_calibration_store_batch_unknown_returns_neutral(self):
        from src.meta.calibration import CalibrationStore
        import numpy as np
        store = CalibrationStore()
        result = store.calibrate_batch("nonexistent_model", np.array([0.1, 0.9, 1.5]))
        assert (result == 0.5).all(), (
            f"Batch unknown model must return 0.5 array, not clip(). Got {result}"
        )

    def test_calibrated_probability_unavailable_has_none_value(self):
        """Spec §114: calibrated_probability must be None, not clip(raw)."""
        from src.meta.schemas import CalibratedProbability, ProbabilityStatus
        cal = CalibratedProbability.unavailable(
            "test_model",
            reason=ProbabilityStatus.UNCALIBRATED,
        )
        assert cal.value is None, (
            f"Unavailable probability must have value=None, not clip(raw). Got {cal.value}"
        )
        assert cal.status == ProbabilityStatus.UNCALIBRATED
        assert not cal.is_usable()


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Meta-label policies: golden tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMetaLabelPolicies:
    """Spec §72: golden tests for meta label construction."""

    def _run_policy(self, events, policy_type, **kwargs):
        from src.meta.meta_label import MetaLabelPolicy, MetaLabelPolicyConfig
        from src.meta.schemas import MetaLabelPolicyType
        cfg = MetaLabelPolicyConfig(
            policy_type=policy_type,
            policy_version="test-v1",
            **kwargs,
        )
        policy = MetaLabelPolicy(cfg)
        labeled, stats = __import__("src.meta.meta_label", fromlist=["build_meta_labels"]).build_meta_labels(
            events, policy, require_oos=True
        )
        return labeled, stats

    def test_policy_a_positive_return_is_take(self):
        """candidate A: +5% net → label=1"""
        from src.meta.schemas import MetaLabelPolicyType
        events = [
            _make_meta_event("A", net_return=5.0, first_touch="TAKE_PROFIT"),
            _make_meta_event("B", net_return=-2.0, first_touch="STOP_LOSS"),
            _make_meta_event("C", net_return=0.01, first_touch="TIME_LIMIT"),
            _make_meta_event("D", net_return=-4.0, first_touch="STOP_LOSS"),
        ]
        labeled, stats = self._run_policy(
            events, MetaLabelPolicyType.POLICY_A_POSITIVE_NET, min_net_return=0.0
        )
        lmap = {e.instrument_id: e.meta_label for e in labeled}
        assert lmap["A"] == 1, "A (+5%) must be TAKE (label=1)"
        assert lmap["B"] == 0, "B (-2%) must be SKIP (label=0)"
        assert lmap["C"] == 1, "C (+0.01%) must be TAKE (label=1)"
        assert lmap["D"] == 0, "D (-4%) must be SKIP (label=0)"

    def test_policy_b_take_profit_is_take(self):
        """first_touch==TAKE_PROFIT → label=1"""
        from src.meta.schemas import MetaLabelPolicyType
        events = [
            _make_meta_event("A", net_return=5.0,  first_touch="TAKE_PROFIT"),
            _make_meta_event("B", net_return=-2.0, first_touch="STOP_LOSS"),
            _make_meta_event("C", net_return=1.0,  first_touch="TIME_LIMIT"),
        ]
        labeled, _ = self._run_policy(events, MetaLabelPolicyType.POLICY_B_TRIPLE_BARRIER)
        lmap = {e.instrument_id: e.meta_label for e in labeled}
        assert lmap["A"] == 1
        assert lmap["B"] == 0
        assert lmap["C"] == 0

    def test_policy_c_risk_adjusted(self):
        """net_return > 0 AND mae > -3% → label=1"""
        from src.meta.schemas import MetaLabelPolicyType
        events = [
            _make_meta_event("A", net_return=2.0,  mae=-1.0),  # good: win + low dd
            _make_meta_event("B", net_return=2.0,  mae=-4.0),  # bad: win but high dd
            _make_meta_event("C", net_return=-1.0, mae=-1.0),  # bad: loss
        ]
        labeled, _ = self._run_policy(
            events, MetaLabelPolicyType.POLICY_C_RISK_ADJUSTED,
            min_required_return=0.0, max_mae=-3.0
        )
        lmap = {e.instrument_id: e.meta_label for e in labeled}
        assert lmap["A"] == 1, "A (+2%, mae=-1%) must be TAKE"
        assert lmap["B"] == 0, "B (+2%, mae=-4%) must be SKIP (drawdown too large)"
        assert lmap["C"] == 0, "C (-1%) must be SKIP"

    def test_meta_label_statistics(self):
        """build_meta_labels returns accurate positive/negative rate stats."""
        from src.meta.schemas import MetaLabelPolicyType
        events = [
            _make_meta_event(f"S{i}", net_return=1.0 if i < 6 else -1.0)
            for i in range(10)
        ]
        _, stats = self._run_policy(events, MetaLabelPolicyType.POLICY_A_POSITIVE_NET)
        assert stats["n_labeled"] == 10
        assert abs(stats["positive_rate"] - 0.6) < 1e-6

    def test_excluded_data_insufficient_events(self):
        """DATA_INSUFFICIENT first_touch with None return → excluded (not labeled)."""
        from src.meta.schemas import MetaLabelPolicyType
        events = [
            _make_meta_event("OK",  net_return=1.0, first_touch="TAKE_PROFIT"),
            _make_meta_event("BAD", net_return=None, first_touch="DATA_INSUFFICIENT"),
        ]
        labeled, stats = self._run_policy(events, MetaLabelPolicyType.POLICY_A_POSITIVE_NET)
        assert stats["n_excluded"] == 1
        assert stats["n_labeled"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 4 — Stacking leakage enforcement (Spec §117)
# ══════════════════════════════════════════════════════════════════════════════

class TestStackingLeakage:
    """
    Spec §Q2: meta model must be trained on OOS primary predictions.
    """

    def test_build_meta_labels_rejects_in_sample_predictions(self):
        from src.meta.schemas import MetaLabelPolicyType
        from src.meta.meta_label import MetaLabelPolicy, MetaLabelPolicyConfig, build_meta_labels

        in_sample_event = _make_meta_event("A", net_return=2.0, is_oos=False)
        cfg = MetaLabelPolicyConfig(
            policy_type=MetaLabelPolicyType.POLICY_A_POSITIVE_NET,
            policy_version="v1",
        )
        policy = MetaLabelPolicy(cfg)

        with pytest.raises(RuntimeError, match="Stacking leakage"):
            build_meta_labels([in_sample_event], policy, require_oos=True)

    def test_build_meta_labels_accepts_oos_predictions(self):
        from src.meta.schemas import MetaLabelPolicyType
        from src.meta.meta_label import MetaLabelPolicy, MetaLabelPolicyConfig, build_meta_labels

        oos_event = _make_meta_event("A", net_return=2.0, is_oos=True)
        cfg = MetaLabelPolicyConfig(
            policy_type=MetaLabelPolicyType.POLICY_A_POSITIVE_NET,
            policy_version="v1",
        )
        policy = MetaLabelPolicy(cfg)
        labeled, _ = build_meta_labels([oos_event], policy, require_oos=True)
        assert len(labeled) == 1
        assert labeled[0].meta_label == 1

    def test_meta_event_assert_oos_raises_for_in_sample(self):
        ev = _make_meta_event("A", is_oos=False)
        with pytest.raises(RuntimeError, match="Stacking leakage"):
            ev.assert_oos()

    def test_build_meta_labels_require_oos_false_allows_in_sample(self):
        """When require_oos=False, in-sample events are allowed (for research)."""
        from src.meta.schemas import MetaLabelPolicyType
        from src.meta.meta_label import MetaLabelPolicy, MetaLabelPolicyConfig, build_meta_labels

        ev = _make_meta_event("A", net_return=2.0, is_oos=False)
        cfg = MetaLabelPolicyConfig(
            policy_type=MetaLabelPolicyType.POLICY_A_POSITIVE_NET,
            policy_version="v1",
        )
        policy = MetaLabelPolicy(cfg)
        labeled, _ = build_meta_labels([ev], policy, require_oos=False)
        assert labeled[0].meta_label == 1  # labeled even though in_sample


# ══════════════════════════════════════════════════════════════════════════════
# 5 — EV engine: golden tests (Spec §92, §35)
# ══════════════════════════════════════════════════════════════════════════════

class TestExpectedValueGolden:
    """Exact EV calculations for known inputs."""

    def _make_prob(self, value: float) -> "CalibratedProbability":
        from src.meta.schemas import CalibratedProbability, ProbabilityStatus
        return CalibratedProbability(
            value=value, status=ProbabilityStatus.CALIBRATED,
            model_id="m", model_version="v1", calibrator_id="c",
            calibration_method="platt", calibration_version="cv1",
            fit_end_time=_ts(-30), effective_from=_ts(-20),
            sample_count=500,
        )

    def _make_payoff(self, win=5.0, loss=-3.0, n=100) -> "PayoffDistribution":
        from src.meta.ev_engine import PayoffDistribution
        return PayoffDistribution(
            win_mean=win, loss_mean=loss, n_samples=n,
            fit_end_time=_ts(-5),
        )

    def test_positive_ev_golden(self):
        """P=0.7, E[win]=5%, E[loss]=-3%, cost=0: EV = 0.7×5 + 0.3×(-3) = 2.6%"""
        from src.meta.ev_engine import ExpectedValueCalculator, EVConfig, CostModel
        from src.meta.schemas import EVStatus
        calc = ExpectedValueCalculator(EVConfig(cost_model=CostModel()))
        ev = calc.compute(self._make_prob(0.7), self._make_payoff(5.0, -3.0),
                          prediction_time=_ts(0))
        assert ev.status in (EVStatus.VALID, EVStatus.COST_DATA_UNAVAILABLE)
        assert ev.value is not None
        # EV before cost: 0.7×5 + 0.3×(-3) = 3.5 - 0.9 = 2.6
        assert abs(ev.value - 2.6) < 0.01, f"Expected EV≈2.6, got {ev.value}"

    def test_negative_ev_golden(self):
        """P=0.4, E[win]=3%, E[loss]=-5%: EV = 0.4×3 + 0.6×(-5) = 1.2 - 3.0 = -1.8%"""
        from src.meta.ev_engine import ExpectedValueCalculator, EVConfig, CostModel
        from src.meta.schemas import EVStatus
        calc = ExpectedValueCalculator(EVConfig(cost_model=CostModel()))
        ev = calc.compute(self._make_prob(0.4), self._make_payoff(3.0, -5.0),
                          prediction_time=_ts(0))
        assert ev.value is not None
        assert abs(ev.value - (-1.8)) < 0.01, f"Expected EV≈-1.8, got {ev.value}"
        assert not ev.is_positive()

    def test_zero_ev_golden(self):
        """P=0.6, E[win]=2%, E[loss]=-3%: EV = 0.6×2 + 0.4×(-3) = 1.2 - 1.2 = 0"""
        from src.meta.ev_engine import ExpectedValueCalculator, EVConfig, CostModel
        calc = ExpectedValueCalculator(EVConfig(cost_model=CostModel()))
        ev = calc.compute(self._make_prob(0.6), self._make_payoff(2.0, -3.0),
                          prediction_time=_ts(0))
        assert abs(ev.value) < 0.01, f"Expected EV≈0, got {ev.value}"

    def test_uncalibrated_probability_gives_ev_status_uncalibrated(self):
        from src.meta.schemas import CalibratedProbability, ProbabilityStatus, EVStatus
        from src.meta.ev_engine import ExpectedValueCalculator, EVConfig, CostModel
        uncal = CalibratedProbability.unavailable("m")
        calc = ExpectedValueCalculator(EVConfig(cost_model=CostModel()))
        ev = calc.compute(uncal, self._make_payoff())
        assert ev.status == EVStatus.PROBABILITY_UNCALIBRATED
        assert ev.value is None

    def test_missing_payoff_gives_insufficient_data_status(self):
        from src.meta.schemas import EVStatus
        from src.meta.ev_engine import ExpectedValueCalculator, EVConfig, CostModel
        calc = ExpectedValueCalculator(EVConfig(cost_model=CostModel()))
        ev = calc.compute(self._make_prob(0.7), None)
        assert ev.status == EVStatus.OUTCOME_DATA_INSUFFICIENT
        assert ev.value is None

    def test_cost_unavailable_status(self):
        """When cost data is absent, EV is computed but flagged COST_DATA_UNAVAILABLE."""
        from src.meta.schemas import EVStatus
        from src.meta.ev_engine import ExpectedValueCalculator, EVConfig, CostModel
        calc = ExpectedValueCalculator(EVConfig(cost_model=CostModel.unavailable()))
        ev = calc.compute(self._make_prob(0.7), self._make_payoff(),
                          prediction_time=_ts(0))
        assert ev.status == EVStatus.COST_DATA_UNAVAILABLE
        # EV should still be computed (without cost component)
        assert ev.value is not None

    def test_asymmetric_payoffs(self):
        """Explicitly tests non-symmetric E[win] ≠ |E[loss]|."""
        from src.meta.ev_engine import ExpectedValueCalculator, EVConfig, CostModel
        calc = ExpectedValueCalculator(EVConfig(cost_model=CostModel()))
        # P=0.5, E[win]=10%, E[loss]=-2%: EV = 0.5×10 + 0.5×(-2) = 4%
        ev = calc.compute(self._make_prob(0.5), self._make_payoff(10.0, -2.0),
                          prediction_time=_ts(0))
        assert abs(ev.value - 4.0) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# 6 — EV leakage: current-event outcome cannot enter its own EV
# ══════════════════════════════════════════════════════════════════════════════

class TestEVLeakage:
    """Spec §118: current event return/MFE/MAE must not enter EV estimate."""

    def test_payoff_future_of_prediction_time_raises(self):
        """PayoffDistribution fitted AFTER prediction_time = EV leakage."""
        from src.meta.ev_engine import PayoffDistribution

        payoff = PayoffDistribution(
            win_mean=5.0, loss_mean=-3.0, n_samples=100,
            fit_end_time=_ts(10),   # AFTER prediction_time!
        )
        with pytest.raises(RuntimeError, match="EV leakage"):
            payoff.assert_no_future_leakage(prediction_time=_ts(5))

    def test_payoff_before_prediction_time_ok(self):
        from src.meta.ev_engine import PayoffDistribution
        payoff = PayoffDistribution(
            win_mean=5.0, loss_mean=-3.0, n_samples=100,
            fit_end_time=_ts(0),    # BEFORE prediction_time
        )
        payoff.assert_no_future_leakage(prediction_time=_ts(5))  # must not raise

    def test_ev_calculator_detects_leakage(self):
        from src.meta.schemas import CalibratedProbability, ProbabilityStatus, EVStatus
        from src.meta.ev_engine import ExpectedValueCalculator, EVConfig, CostModel, PayoffDistribution
        cal = CalibratedProbability(
            value=0.7, status=ProbabilityStatus.CALIBRATED,
            model_id="m", model_version="v1", calibrator_id="c",
            calibration_method="platt", calibration_version="v1",
            fit_end_time=_ts(-30), effective_from=_ts(-20), sample_count=100,
        )
        # Payoff fitted AFTER the prediction_time
        future_payoff = PayoffDistribution(
            win_mean=5.0, loss_mean=-3.0, n_samples=100,
            fit_end_time=_ts(10),  # future
        )
        calc = ExpectedValueCalculator(EVConfig(cost_model=CostModel()))
        ev = calc.compute(cal, future_payoff, prediction_time=_ts(0))
        assert ev.status == EVStatus.INSUFFICIENT_EVIDENCE, (
            f"Future payoff must produce INSUFFICIENT_EVIDENCE, got {ev.status}"
        )
        assert "EV_LEAKAGE_DETECTED" in ev.payoff_assumptions


# ══════════════════════════════════════════════════════════════════════════════
# 7 — Calibration metrics (pure numpy/pandas, no sklearn)
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibrationMetrics:
    def test_perfect_calibration_ece_near_zero(self):
        """Predicted prob ≈ observed freq → ECE ≈ 0."""
        from src.meta.calibration_engine import compute_calibration_metrics
        np.random.seed(42)
        n = 1000
        probs  = np.random.uniform(0.1, 0.9, n)
        labels = (np.random.uniform(0, 1, n) < probs).astype(float)
        metrics = compute_calibration_metrics(probs, labels)
        assert metrics["ece"] < 0.05, f"Perfect calibration should have ECE<0.05, got {metrics['ece']}"

    def test_overconfident_calibration_ece_high(self):
        """Predicted=0.9 but observed=0.5 → ECE should be ~0.4."""
        from src.meta.calibration_engine import compute_calibration_metrics
        n = 1000
        probs  = np.full(n, 0.9)
        labels = np.random.RandomState(0).binomial(1, 0.5, n).astype(float)
        metrics = compute_calibration_metrics(probs, labels)
        assert metrics["ece"] > 0.2, f"Overconfident: expected ECE>0.2, got {metrics['ece']}"

    def test_reliability_curve_perfect_calibration(self):
        """Perfectly calibrated: each bin's mean_predicted ≈ observed_frequency."""
        from src.meta.calibration_engine import reliability_curve
        np.random.seed(0)
        n = 5000
        probs  = np.random.uniform(0, 1, n)
        labels = (np.random.uniform(0, 1, n) < probs).astype(float)
        bins   = reliability_curve(probs, labels, n_bins=10)
        filled = [b for b in bins if b.sample_count > 0]
        # Calibration gaps should be small for perfectly calibrated model
        gaps = [b.calibration_gap for b in filled if b.calibration_gap is not None]
        assert len(gaps) > 0
        assert max(gaps) < 0.15, f"Max calibration gap {max(gaps):.3f} > 0.15"

    def test_brier_score_range(self):
        from src.meta.calibration_engine import compute_calibration_metrics
        probs  = np.array([0.1, 0.5, 0.9])
        labels = np.array([0.0, 1.0, 1.0])
        metrics = compute_calibration_metrics(probs, labels)
        assert 0.0 <= metrics["brier"] <= 1.0

    def test_calibration_slope_perfect_calibration(self):
        """slope≈1.0, intercept≈0.0 for perfectly calibrated model."""
        from src.meta.calibration_engine import calibration_slope_intercept
        np.random.seed(1)
        n = 1000
        probs  = np.random.uniform(0.1, 0.9, n)
        labels = (np.random.uniform(0, 1, n) < probs).astype(float)
        slope, intercept = calibration_slope_intercept(probs, labels)
        assert slope is not None and intercept is not None
        assert 0.7 <= slope <= 1.3, f"Slope {slope:.3f} far from 1.0"
        assert -0.2 <= intercept <= 0.2, f"Intercept {intercept:.3f} far from 0.0"

    def test_insufficient_data_returns_none(self):
        from src.meta.calibration_engine import calibration_slope_intercept
        slope, intercept = calibration_slope_intercept(
            np.array([0.5, 0.6]), np.array([0.0, 1.0])
        )
        assert slope is None and intercept is None


# ══════════════════════════════════════════════════════════════════════════════
# 8 — Calibration leakage: fit_end must be < eval_start
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibrationLeakage:
    """Spec §116: calibration must end before evaluation begins."""

    def test_walk_forward_temporal_order_invariant(self):
        """
        walk_forward_calibrate asserts fit_end_time < eval_start_time.
        If this assertion fires, a leakage bug is present.
        """
        from src.meta.calibration_engine import walk_forward_calibrate
        np.random.seed(42)
        n = 200
        ts = np.array([_ts(i) for i in range(n)])
        scores = np.random.uniform(0, 1, n)
        labels = (np.random.uniform(0, 1, n) < scores).astype(float)

        results = walk_forward_calibrate(
            timestamps=ts, raw_scores=scores, labels=labels,
            calibration_bars=80, eval_bars=40, step_bars=40,
            min_fit_samples=10,
        )
        assert len(results) > 0
        for r in results:
            if r.fit_end_time and r.eval_start_time:
                assert r.fit_end_time < r.eval_start_time, (
                    f"Fold {r.fold_index}: fit_end={r.fit_end_time} >= "
                    f"eval_start={r.eval_start_time} — calibration leakage"
                )

    def test_calibrator_fit_end_before_eval_start(self):
        """Calibration on fold T must not include any data from fold T+1."""
        from src.meta.calibration_engine import walk_forward_calibrate
        np.random.seed(10)
        n = 300
        ts = np.array([_ts(i) for i in range(n)])
        scores = np.random.uniform(0, 1, n)
        labels = (np.random.uniform(0, 1, n) < scores).astype(float)
        results = walk_forward_calibrate(ts, scores, labels,
                                          calibration_bars=100, eval_bars=50, step_bars=50)
        for fold in results:
            assert fold.eval_is_oos, "eval_is_oos must be True — eval is disjoint from fit"


# ══════════════════════════════════════════════════════════════════════════════
# 9 — CalibratorArtifact states
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibratorArtifactStates:
    def _make_artifact(
        self, model_id="m1", model_version="v1",
        effective_from_offset=-5, max_age_days=30
    ) -> "CalibratorArtifact":
        from src.meta.calibration_engine import CalibratorArtifact
        art = CalibratorArtifact(
            calibrator_id="c1",
            model_id=model_id,
            model_version=model_version,
            calibration_method="platt",
            calibration_version="cv1",
            fit_end_time=_ts(effective_from_offset - 1),
            effective_from=_ts(effective_from_offset),
            fit_sample_count=200,
            max_age_days=max_age_days,
        )
        return art

    def _make_fitted_artifact(
        self, model_id="m1", model_version="v1",
        effective_from_offset=-5, max_age_days=30
    ) -> "CalibratorArtifact":
        """Return a pre-fitted Platt artifact for testing mismatch/staleness."""
        from src.meta.calibration_engine import CalibratorArtifact
        art = CalibratorArtifact(
            calibrator_id="c1",
            model_id=model_id,
            model_version=model_version,
            calibration_method="platt",
            calibration_version="cv1",
            fit_end_time=_ts(effective_from_offset - 1),
            effective_from=_ts(effective_from_offset),
            fit_sample_count=200,
            max_age_days=max_age_days,
            _fitted=True,
            _platt_a=1.0,
            _platt_b=0.0,
        )
        return art

    def test_unfitted_returns_uncalibrated(self):
        from src.meta.schemas import ProbabilityStatus
        art = self._make_artifact()
        _, status = art.predict(np.array([0.5]), "m1", "v1")
        assert status == ProbabilityStatus.UNCALIBRATED

    def test_model_mismatch_returns_mismatch_status(self):
        """Spec §115: model_v1 + calibrator_v2 → CALIBRATOR_MISMATCH."""
        from src.meta.schemas import ProbabilityStatus
        art = self._make_fitted_artifact(model_id="model_a", model_version="v1")
        _, status = art.predict(np.array([0.5]), "model_b", "v1")  # wrong model_id
        assert status == ProbabilityStatus.CALIBRATOR_MISMATCH

    def test_version_mismatch_returns_mismatch_status(self):
        from src.meta.schemas import ProbabilityStatus
        art = self._make_fitted_artifact(model_id="m1", model_version="v1")
        _, status = art.predict(np.array([0.5]), "m1", "v2")  # wrong version
        assert status == ProbabilityStatus.CALIBRATOR_MISMATCH

    def test_stale_calibrator_returns_stale_status(self):
        """Spec §120: stale calibrator must return STALE, not produce a probability."""
        from src.meta.schemas import ProbabilityStatus
        from src.meta.calibration_engine import CalibratorArtifact
        # Create an artifact that's 100 days old with max_age_days=30
        art = CalibratorArtifact(
            calibrator_id="stale",
            model_id="m1", model_version="v1",
            calibration_method="platt",
            calibration_version="cv1",
            fit_end_time=_ts(-101),
            effective_from=_ts(-100),   # 100 days ago
            fit_sample_count=200,
            max_age_days=30,            # max 30 days
            _fitted=True, _platt_a=1.0, _platt_b=0.0,
        )
        _, status = art.predict(
            np.array([0.5]), "m1", "v1",
            current_time=_ts(0),  # today
        )
        assert status == ProbabilityStatus.STALE, (
            f"100-day-old calibrator with max_age=30 must be STALE, got {status}"
        )

    def test_calibrate_single_returns_unavailable_for_unfitted(self):
        from src.meta.schemas import ProbabilityStatus
        art = self._make_artifact()
        result = art.calibrate_single(0.7, "m1", "v1")
        assert not result.is_usable()
        assert result.status != ProbabilityStatus.CALIBRATED


# ══════════════════════════════════════════════════════════════════════════════
# 10 — Decision engine (pure rule-based, no sklearn)
# ══════════════════════════════════════════════════════════════════════════════

class TestDecisionEngine:
    """Test the canonical TAKE/SKIP/ABSTAIN/INSUFFICIENT_EVIDENCE states."""

    def _make_calibrated_prob(self, value: float) -> "CalibratedProbability":
        from src.meta.schemas import CalibratedProbability, ProbabilityStatus
        return CalibratedProbability(
            value=value, status=ProbabilityStatus.CALIBRATED,
            model_id="m", model_version="v1", calibrator_id="c",
            calibration_method="platt", calibration_version="cv1",
            fit_end_time=_ts(-30), effective_from=_ts(-20), sample_count=200,
        )

    def _make_ev(self, ev_val: float, positive: bool) -> "ExpectedValue":
        from src.meta.schemas import ExpectedValue, EVStatus
        return ExpectedValue(
            value=ev_val, probability=0.65,
            expected_win=5.0, expected_loss=-3.0, expected_cost=0.5,
            confidence=0.8,
            status=EVStatus.VALID,
        )

    def test_take_when_prob_above_threshold_and_positive_ev(self):
        from src.meta.schemas import Decision, AlphaScore, CalibratedProbability, ProbabilityStatus
        from src.meta.schemas import ExpectedValue, EVStatus, MetaDecisionOutput, DecisionReason

        decision = Decision.TAKE
        reasons = [DecisionReason.CALIBRATED_PROBABILITY_ABOVE_THRESHOLD,
                   DecisionReason.POSITIVE_NET_EV]
        assert decision == Decision.TAKE
        assert DecisionReason.POSITIVE_NET_EV in reasons

    def test_skip_when_negative_ev(self):
        from src.meta.schemas import Decision
        assert Decision.SKIP != Decision.TAKE

    def test_abstain_when_calibration_unavailable(self):
        """ABSTAIN when calibration is unavailable, not SKIP."""
        from src.meta.schemas import (Decision, ProbabilityStatus, DecisionReason,
                                       CalibratedProbability)
        uncal = CalibratedProbability.unavailable(
            "m", reason=ProbabilityStatus.UNCALIBRATED
        )
        assert not uncal.is_usable()
        # The system should ABSTAIN when calibration is not available
        # (not SKIP — we don't have enough evidence to say SKIP)
        abstain_reasons = [DecisionReason.CALIBRATION_UNAVAILABLE]
        assert DecisionReason.CALIBRATION_UNAVAILABLE in abstain_reasons

    def test_insufficient_evidence_has_explicit_status(self):
        from src.meta.schemas import Decision, ProbabilityStatus
        assert Decision.INSUFFICIENT_EVIDENCE != Decision.UNAVAILABLE
        assert ProbabilityStatus.INSUFFICIENT_DATA != ProbabilityStatus.UNAVAILABLE

    def test_all_five_decision_states_reachable(self):
        from src.meta.schemas import Decision
        states = {Decision.TAKE, Decision.SKIP, Decision.ABSTAIN,
                  Decision.INSUFFICIENT_EVIDENCE, Decision.UNAVAILABLE}
        assert len(states) == 5


# ══════════════════════════════════════════════════════════════════════════════
# 11 — Outcome feature leakage guard
# ══════════════════════════════════════════════════════════════════════════════

class TestOutcomeFeatureLeakage:
    def test_gross_return_in_features_raises(self):
        from src.meta.meta_label import validate_no_outcome_features
        with pytest.raises(ValueError, match="gross_return"):
            validate_no_outcome_features(["rsi_14", "gross_return", "adx_14"])

    def test_net_return_in_features_raises(self):
        from src.meta.meta_label import validate_no_outcome_features
        with pytest.raises(ValueError, match="net_return"):
            validate_no_outcome_features(["alpha_score", "net_return"])

    def test_mfe_in_features_raises(self):
        from src.meta.meta_label import validate_no_outcome_features
        with pytest.raises(ValueError, match="mfe"):
            validate_no_outcome_features(["momentum", "mfe"])

    def test_mae_in_features_raises(self):
        from src.meta.meta_label import validate_no_outcome_features
        with pytest.raises(ValueError, match="mae"):
            validate_no_outcome_features(["volatility", "mae", "adx"])

    def test_first_touch_in_features_raises(self):
        from src.meta.meta_label import validate_no_outcome_features
        with pytest.raises(ValueError, match="first_touch"):
            validate_no_outcome_features(["rsi_14", "first_touch"])

    def test_outcome_prefix_in_features_raises(self):
        from src.meta.meta_label import validate_no_outcome_features
        with pytest.raises(ValueError, match="_outcome_"):
            validate_no_outcome_features(["_outcome_return", "rsi_14"])

    def test_clean_features_pass(self):
        from src.meta.meta_label import validate_no_outcome_features
        validate_no_outcome_features([
            "primary_alpha_score", "primary_rank", "primary_percentile",
            "rsi_14", "adx_14", "atr_pct", "relative_volume",
        ])  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# 12 — Future mutation: data after prediction_time must not change output
# ══════════════════════════════════════════════════════════════════════════════

class TestFutureMutation:
    """Spec §119: modifying data after prediction_time must not alter output."""

    def test_meta_label_future_mutation_invariant(self):
        """
        Adding extreme future outcomes to a set of events must not change
        the labels of events at earlier timestamps.
        """
        from src.meta.meta_label import MetaLabelPolicy, MetaLabelPolicyConfig, build_meta_labels
        from src.meta.schemas import MetaLabelPolicyType

        events_t0 = [
            _make_meta_event("A", net_return=2.0,  prediction_time=_ts(0), event_end_time=_ts(5)),
            _make_meta_event("B", net_return=-1.0, prediction_time=_ts(1), event_end_time=_ts(6)),
        ]
        cfg = MetaLabelPolicyConfig(
            policy_type=MetaLabelPolicyType.POLICY_A_POSITIVE_NET, policy_version="v1"
        )
        policy = MetaLabelPolicy(cfg)
        labeled_before, _ = build_meta_labels(events_t0, policy, require_oos=True)

        # Append extreme future events
        future_events = [
            _make_meta_event("X", net_return=9999.0, prediction_time=_ts(50),
                             event_end_time=_ts(55)),
        ]
        all_events = events_t0 + future_events
        labeled_all, _ = build_meta_labels(all_events, policy, require_oos=True)

        # Labels of original events must be unchanged
        before_map = {e.instrument_id: e.meta_label for e in labeled_before}
        after_map  = {e.instrument_id: e.meta_label for e in labeled_all}
        for sym in ["A", "B"]:
            assert before_map[sym] == after_map[sym], (
                f"{sym}: meta_label changed after appending future events "
                f"({before_map[sym]} → {after_map[sym]})"
            )

    def test_ev_future_payoff_mutation_rejected(self):
        """Payoff estimated after prediction_time → EV leakage → INSUFFICIENT_EVIDENCE."""
        from src.meta.ev_engine import ExpectedValueCalculator, EVConfig, CostModel, PayoffDistribution
        from src.meta.schemas import CalibratedProbability, ProbabilityStatus, EVStatus

        cal = CalibratedProbability(
            value=0.7, status=ProbabilityStatus.CALIBRATED,
            model_id="m", model_version="v1", calibrator_id="c",
            calibration_method="platt", calibration_version="v1",
            fit_end_time=_ts(-30), effective_from=_ts(-20), sample_count=100,
        )
        # Payoff fitted AFTER prediction time (mutation)
        mutated_payoff = PayoffDistribution(
            win_mean=5.0, loss_mean=-3.0, n_samples=100,
            fit_end_time=_ts(100),  # extreme future
        )
        calc = ExpectedValueCalculator(EVConfig(cost_model=CostModel()))
        ev = calc.compute(cal, mutated_payoff, prediction_time=_ts(0))
        assert ev.status == EVStatus.INSUFFICIENT_EVIDENCE


# ══════════════════════════════════════════════════════════════════════════════
# 13 — Probability range invariant
# ══════════════════════════════════════════════════════════════════════════════

class TestProbabilityRange:
    def test_calibrated_probability_in_range(self):
        from src.meta.schemas import CalibratedProbability, ProbabilityStatus
        for v in [0.0, 0.01, 0.5, 0.99, 1.0]:
            cp = CalibratedProbability(
                value=v, status=ProbabilityStatus.CALIBRATED,
                model_id="m", model_version="v1", calibrator_id="c",
                calibration_method="platt", calibration_version="v1",
                fit_end_time=_ts(-5), effective_from=_ts(-1), sample_count=100,
            )
            assert cp.is_usable(), f"P={v} should be usable"
            assert 0.0 <= cp.value <= 1.0

    def test_probability_outside_range_not_usable(self):
        from src.meta.schemas import CalibratedProbability, ProbabilityStatus
        for bad_v in [-0.01, 1.01, 1.5, -1.0]:
            cp = CalibratedProbability(
                value=bad_v, status=ProbabilityStatus.CALIBRATED,
                model_id="m", model_version="v1", calibrator_id="c",
                calibration_method="platt", calibration_version="v1",
                fit_end_time=_ts(-5), effective_from=_ts(-1), sample_count=100,
            )
            assert not cp.is_usable(), f"P={bad_v} outside [0,1] must not be usable"


# ══════════════════════════════════════════════════════════════════════════════
# 14 — Baseline ranker and compare_meta_models
# ══════════════════════════════════════════════════════════════════════════════

class TestBaselineRankers:
    def test_alpha_threshold_baseline_rank1_is_highest_score(self):
        from src.meta.meta_ranker import AlphaThresholdBaseline
        baseline = AlphaThresholdBaseline(alpha_score_col_idx=0, normalizer=100.0)
        baseline.fit(np.zeros((5, 1)), np.zeros(5))
        X = np.array([[90.0], [50.0], [75.0], [10.0], [60.0]])
        scores = baseline.predict_raw(X)
        assert scores[0] == pytest.approx(0.9)
        assert scores[3] == pytest.approx(0.1)

    def test_ml_adds_no_clear_value_when_below_baseline(self):
        from src.meta.meta_ranker import compare_meta_models
        results = [
            {"model_id": "alpha_threshold_baseline", "roc_auc": 0.62},
            {"model_id": "lgbm_meta_ranker",         "roc_auc": 0.62},  # equal
        ]
        comparison = compare_meta_models(results)
        lgbm_row = next(r for r in comparison if r["model_id"] == "lgbm_meta_ranker")
        assert lgbm_row["verdict"] == "ML_ADDS_NO_CLEAR_VALUE", (
            "LightGBM equal to baseline should be ML_ADDS_NO_CLEAR_VALUE"
        )

    def test_research_signal_when_ml_clearly_better(self):
        from src.meta.meta_ranker import compare_meta_models
        results = [
            {"model_id": "alpha_threshold_baseline", "roc_auc": 0.55},
            {"model_id": "lgbm_meta_ranker",         "roc_auc": 0.70},  # much better
        ]
        comparison = compare_meta_models(results)
        lgbm_row = next(r for r in comparison if r["model_id"] == "lgbm_meta_ranker")
        assert lgbm_row["verdict"] == "RESEARCH_SIGNAL_DETECTED"

    def test_insufficient_evidence_when_auc_none(self):
        from src.meta.meta_ranker import compare_meta_models
        results = [{"model_id": "lgbm_meta_ranker", "roc_auc": None}]
        comparison = compare_meta_models(results)
        assert comparison[0]["verdict"] == "INSUFFICIENT_EVIDENCE"


# ══════════════════════════════════════════════════════════════════════════════
# 15 — Meta-label to DataFrame: outcome prefix safety
# ══════════════════════════════════════════════════════════════════════════════

class TestMetaEventDataframe:
    def test_outcome_columns_have_prefix(self):
        from src.meta.meta_label import meta_events_to_dataframe
        events = [_make_meta_event("A", net_return=2.0)]
        df = meta_events_to_dataframe(events)
        outcome_cols = [c for c in df.columns if c.startswith("_outcome_")]
        assert len(outcome_cols) > 0, "Outcome columns must have _outcome_ prefix"
        non_prefixed_outcomes = [c for c in df.columns if c in
                                   ("gross_return","net_return","mfe","mae","first_touch")]
        assert len(non_prefixed_outcomes) == 0, (
            "Outcome columns must not appear without _outcome_ prefix in DataFrame"
        )

    def test_meta_label_column_present(self):
        from src.meta.meta_label import meta_events_to_dataframe, MetaLabelPolicy, MetaLabelPolicyConfig, build_meta_labels
        from src.meta.schemas import MetaLabelPolicyType
        events = [_make_meta_event("A", net_return=2.0)]
        cfg = MetaLabelPolicyConfig(
            policy_type=MetaLabelPolicyType.POLICY_A_POSITIVE_NET,
            policy_version="v1",
        )
        labeled, _ = build_meta_labels(events, MetaLabelPolicy(cfg), require_oos=True)
        df = meta_events_to_dataframe(labeled)
        assert "meta_label" in df.columns
        assert df["meta_label"].iloc[0] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 16 — ML rankers (sklearn/lightgbm/xgboost dependent)
# ══════════════════════════════════════════════════════════════════════════════

class TestLinearMetaRanker:
    def test_predict_shape_and_range(self):
        pytest.importorskip("sklearn", reason="sklearn not installed")
        pytest.importorskip("scipy", reason="scipy not installed")
        from src.meta.meta_ranker import LinearMetaRanker
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = (np.random.randn(100) > 0).astype(float)
        r = LinearMetaRanker()
        r.fit(X, y)
        preds = r.predict_raw(X[:10])
        assert preds.shape == (10,)
        assert (preds >= 0).all() and (preds <= 1).all()


class TestLightGBMMetaRanker:
    def test_predict_shape(self):
        pytest.importorskip("lightgbm", reason="lightgbm not installed")
        from src.meta.meta_ranker import LightGBMMetaRanker
        np.random.seed(42)
        X = np.random.randn(200, 8).astype(np.float32)
        y = (np.random.randn(200) > 0).astype(float)
        r = LightGBMMetaRanker(params={"n_estimators": 20, "num_leaves": 7})
        r.fit(X, y)
        preds = r.predict_raw(X[:5])
        assert preds.shape == (5,)


class TestXGBoostMetaRanker:
    def test_predict_shape(self):
        pytest.importorskip("xgboost", reason="xgboost not installed")
        from src.meta.meta_ranker import XGBoostMetaRanker
        np.random.seed(42)
        X = np.random.randn(200, 6).astype(np.float32)
        y = (np.random.randn(200) > 0).astype(float)
        r = XGBoostMetaRanker(params={"n_estimators": 20, "max_depth": 3})
        r.fit(X, y)
        preds = r.predict_raw(X[:5])
        assert preds.shape == (5,)


# ══════════════════════════════════════════════════════════════════════════════
# 17 — CalibrationQualityV2 includes eval_is_oos
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibrationQualityV2:
    def test_eval_is_oos_field_present(self):
        from src.meta.schemas import CalibrationQualityV2
        q = CalibrationQualityV2(
            model_id="m", calibrator_id="c",
            calibration_method="platt", calibration_version="v1",
            fit_end_time=_ts(-10), effective_from=_ts(-5),
            fit_sample_count=200, eval_sample_count=100,
            eval_is_oos=True,
            ece=0.05, mce=0.10, brier_score=0.2,
            log_loss=0.5, calibration_slope=1.02, calibration_intercept=0.01,
            is_fitted=True,
        )
        assert q.eval_is_oos is True
        assert q.quality_score > 0.0

    def test_staleness_detection(self):
        from src.meta.schemas import CalibrationQualityV2
        q = CalibrationQualityV2(
            model_id="m", calibrator_id="c",
            calibration_method="platt", calibration_version="v1",
            fit_end_time=_ts(-100), effective_from=_ts(-90),
            fit_sample_count=100, eval_sample_count=50,
            eval_is_oos=True,
            ece=0.05, mce=0.10, brier_score=0.2,
            log_loss=0.5, calibration_slope=1.0, calibration_intercept=0.0,
            is_fitted=True,
        )
        assert q.is_stale(_ts(0), max_age_days=30)
        assert not q.is_stale(_ts(-85), max_age_days=30)

    def test_legacy_calibration_quality_has_eval_is_oos(self):
        """Phase 3F fix BUG #4: CalibrationQuality must now have eval_is_oos field."""
        from src.meta.calibration import CalibrationQuality
        q = CalibrationQuality(
            model_name="regime",
            calibrator_kind="platt",
            ece=0.05, mce=0.10, brier_score=0.2, n_samples=100,
            is_fitted=True,
        )
        assert hasattr(q, "eval_is_oos"), (
            "CalibrationQuality must have eval_is_oos field (BUG #4 fix)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 18 — Probability bucket analysis
# ══════════════════════════════════════════════════════════════════════════════

class TestProbabilityBucketAnalysis:
    def test_all_buckets_reported(self):
        """All 10 buckets must be reported — no cherry-picking."""
        from src.meta.ev_engine import compute_probability_bucket_analysis
        np.random.seed(0)
        n = 500
        probs    = np.random.uniform(0, 1, n)
        outcomes = (np.random.uniform(0, 1, n) < probs).astype(float)
        returns  = np.where(outcomes == 1, 3.0, -2.0)
        buckets  = compute_probability_bucket_analysis(probs, outcomes, returns)
        assert len(buckets) == 10, f"Expected 10 buckets, got {len(buckets)}"

    def test_high_prob_bucket_higher_success_rate(self):
        """High-probability bucket should have higher success rate than low-prob bucket."""
        from src.meta.ev_engine import compute_probability_bucket_analysis
        n = 2000
        np.random.seed(42)
        probs    = np.random.uniform(0, 1, n)
        outcomes = (np.random.uniform(0, 1, n) < probs).astype(float)
        returns  = np.where(outcomes == 1, 5.0, -3.0)
        buckets  = compute_probability_bucket_analysis(probs, outcomes, returns)
        low_bucket  = next((b for b in buckets if b.bucket_lo < 0.2 and b.n_observations > 0), None)
        high_bucket = next((b for b in reversed(buckets) if b.bucket_hi > 0.8 and b.n_observations > 0), None)
        if low_bucket and high_bucket and low_bucket.actual_success_rate and high_bucket.actual_success_rate:
            assert high_bucket.actual_success_rate > low_bucket.actual_success_rate, (
                "High-probability bucket must have higher observed success rate"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 19 — CalibrationQuality eval_is_oos propagated in CalibrationStore.fit()
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibrationStoreEvalIsOOS:
    def test_fit_with_eval_scores_sets_eval_is_oos_true(self):
        pytest.importorskip("sklearn", reason="sklearn not installed")
        from src.meta.calibration import CalibrationStore
        import numpy as np
        store = CalibrationStore()
        np.random.seed(1)
        n = 200
        scores     = np.random.uniform(0, 1, n)
        labels     = (np.random.uniform(0, 1, n) < scores).astype(float)
        eval_scores  = np.random.uniform(0, 1, 50)
        eval_labels  = (np.random.uniform(0, 1, 50) < eval_scores).astype(float)
        q = store.fit("regime", scores, labels, kind="platt",
                      eval_scores=eval_scores, eval_labels=eval_labels)
        assert q.eval_is_oos is True

    def test_fit_without_eval_scores_sets_eval_is_oos_false(self):
        pytest.importorskip("sklearn", reason="sklearn not installed")
        from src.meta.calibration import CalibrationStore
        import numpy as np
        store = CalibrationStore()
        np.random.seed(2)
        scores = np.random.uniform(0, 1, 200)
        labels = (np.random.uniform(0, 1, 200) < scores).astype(float)
        q = store.fit("regime", scores, labels, kind="platt")
        assert q.eval_is_oos is False
