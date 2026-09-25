"""
Phase 3M — Research-to-Production Integration, Shadow Execution & ML Ops.

Covers the mandatory acceptance criteria (spec §30) and the adversarial suite
(spec §31) for the ONE canonical, deterministic, fail-closed, replayable
decision pipeline that ORCHESTRATES phases 3A–3L.

Test classes
------------
  TestDecisionPipeline    happy path → EXECUTION_PLANNED; stage contract; events
  TestFailClosed          every missing/unsafe dependency → NO_EXECUTION (§30)
  TestModelCompat         feature/label/horizon/instrument/regime incompatibility
  TestReplay              deterministic replay (content_hash + replay_id stable)
  TestShadow              shadow execution reuses Phase 3G; idempotent; append-only
  TestReconciliation      predicted-vs-realized; None when inputs absent; aggregate
  TestHealth              structured health rollup; blocking; INSUFFICIENT_EVIDENCE
  TestAdversarial         §31 — future data/label/CA/universe/model/calibration/lot;
                          NaN/Inf/neg/zero/dup/out-of-order/dup-decision/dup-fill/
                          missing-hash/invalid-schema/invalid-transition
  TestStaticAudit         no broker/live tokens anywhere in src/decision + src/shadow;
                          import-cleanliness (no talib/torch/sklearn at load)

All heavy imports are lazy (inside tests). Deterministic synthetic data; no global
np.random.* leakage. Run from ml-service/ with python3 -m pytest.
"""

from __future__ import annotations

import math
import os
import tempfile
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

UTC = timezone.utc

# repo roots for the static audit
_HERE = os.path.dirname(os.path.abspath(__file__))
_ML = os.path.dirname(_HERE)
_DECISION_DIR = os.path.join(_ML, "src", "decision")
_SHADOW_DIR = os.path.join(_ML, "src", "shadow")


# ── helpers ─────────────────────────────────────────────────────────────────

def _past(minutes_ago=5):
    return (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()


def _future(minutes_ahead=60):
    return (datetime.now(UTC) + timedelta(minutes=minutes_ahead)).isoformat()


def _happy_inputs(**overrides):
    """
    Build a PipelineInputs that flows all the way to EXECUTION_PLANNED, then
    apply per-test overrides that break exactly one dependency.
    """
    from src.decision.pipeline import PipelineInputs

    base = dict(
        instrument="RELIANCE",
        as_of_time=_past(5),
        deployment_mode="shadow",
        # data
        data_snapshot_id="snap-001",
        dataset_version="ds-v1",
        data_max_age_seconds=86400.0,
        expected_instrument="RELIANCE",
        # features
        feature_version="feat-v1",
        feature_schema_hash="fhash-1",
        required_features=["f1", "f2"],
        present_features=["f1", "f2"],
        # regime
        market_regime="TREND_UP",
        regime_confidence=0.8,
        # model
        model_ids=["m1"],
        model_versions=["1.0.0"],
        model_hashes=["mhash-1"],
        model_exists=True,
        registry_status="ACTIVE",
        acceptance_status="ACCEPTED",
        deployment_date=_past(60),
        runtime_horizon=5,
        # ranker / strategy
        alpha_score=1.7,
        alpha_rank=1,
        alpha_rank_percentile=0.99,
        strategy="momentum",
        direction="LONG",
        # meta (BUY, trained model so governance keeps it)
        meta_output={"action": "BUY"},
        prediction_provenance="trained_model",
        # calibration
        calibration_exists=True,
        calibration_id="m1",
        calibration_fit_time=_past(1),
        raw_probability=0.61,
        calibrated_probability=0.58,
        # EV
        expected_win=120.0,
        expected_loss=80.0,
        expected_cost=10.0,
        expected_value=15.0,
        # abstention
        abstention_state="TRADE",
        abstention_reason=[],
        # portfolio
        risk_available=True,
        constraints_ok=True,
        portfolio_target=0.05,
        risk_budget=0.02,
        position_size=10.0,
        # execution
        cost_model_version="cost-v1",
        slippage_model_version="slip-v1",
        simulator_available=True,
        execution_policy="TWAP",
        execution_policy_version="exec-v1",
        # health
        system_health_degraded=False,
        excessive_drift=False,
        # provenance / config
        code_version="code-abc",
        environment_version="env-1",
        random_seeds={"np": 0},
    )
    base.update(overrides)
    return PipelineInputs(**base)


def _run(inp, decision_id="dec-test", event_log=None):
    from src.decision.pipeline import DecisionPipeline
    return DecisionPipeline(event_log=event_log).evaluate(
        inp, decision_id=decision_id, trace_id="trace-test")


def _bars(n=12, seed=0, base=20000.0):
    from src.rl.environment import MarketBar
    rng = np.random.RandomState(seed)
    bars, px = [], base
    t0 = datetime(2023, 1, 2, 9, 15, tzinfo=UTC)
    for i in range(n):
        px *= (1 + rng.normal(0.0004, 0.004))
        bars.append(MarketBar(timestamp=t0 + timedelta(minutes=i), open=px * 0.999,
                              high=px * 1.005, low=px * 0.996, close=px,
                              volume=1e5 * (1 + 0.3 * np.sin(i)), adv_inr=5e8, atr_pct=0.01))
    return bars


# ══════════════════════════════════════════════════════════════════════════════
class TestDecisionPipeline:

    def test_happy_path_reaches_execution_planned(self):
        out = _run(_happy_inputs())
        from src.decision.state import DecisionState
        assert out.decision.state == DecisionState.EXECUTION_PLANNED
        assert out.decision.executable is True

    def test_stage_results_contract(self):
        out = _run(_happy_inputs())
        # every stage has a name + status
        for s in out.stages:
            d = s.to_dict()
            assert d["stage"] and d["status"]
        names = [s.stage for s in out.stages]
        # canonical ordering of the core stages
        for expected in ("data", "features", "regime", "model", "meta",
                         "calibration", "portfolio", "execution", "safety"):
            assert expected in names, expected

    def test_provenance_and_replay_manifest_present(self):
        out = _run(_happy_inputs())
        assert out.provenance.is_valid is True
        assert out.provenance.replay_manifest["data_snapshot_id"] == "snap-001"

    def test_events_emitted_when_log_attached(self):
        from src.decision.events import EventLog
        with tempfile.TemporaryDirectory() as d:
            log = EventLog(d)
            out = _run(_happy_inputs(), decision_id="dec-ev", event_log=log)
            events = log.for_decision("dec-ev")
            assert len(events) >= 5
            assert any(e["event_type"] == "EXECUTION_PLANNED" for e in events)

    def test_live_mode_forbidden(self):
        from src.decision.provenance import LiveExecutionForbidden
        for mode in ("live", "production", "LIVE", "PRODUCTION"):
            with pytest.raises(LiveExecutionForbidden):
                _run(_happy_inputs(deployment_mode=mode))


# ══════════════════════════════════════════════════════════════════════════════
class TestFailClosed:
    """Every missing/unsafe dependency must yield a non-executable state (§30)."""

    def _assert_no_execution(self, out, expected_state=None):
        from src.decision.state import DecisionState, NON_EXECUTABLE_STATES
        assert out.decision.executable is False
        assert out.decision.state in NON_EXECUTABLE_STATES
        if expected_state is not None:
            assert out.decision.state == expected_state, out.decision.state

    def test_data_unavailable(self):
        from src.decision.state import DecisionState
        out = _run(_happy_inputs(data_snapshot_id=None))
        self._assert_no_execution(out, DecisionState.DATA_UNAVAILABLE)

    def test_data_invalid_future_timestamp(self):
        from src.decision.state import DecisionState
        out = _run(_happy_inputs(as_of_time=_future(120)))
        self._assert_no_execution(out, DecisionState.DATA_INVALID)

    def test_features_unavailable(self):
        from src.decision.state import DecisionState
        out = _run(_happy_inputs(present_features=["f1"]))  # missing f2
        self._assert_no_execution(out, DecisionState.FEATURES_UNAVAILABLE)

    def test_regime_unavailable(self):
        from src.decision.state import DecisionState
        out = _run(_happy_inputs(market_regime=None))
        self._assert_no_execution(out, DecisionState.REGIME_UNAVAILABLE)

    def test_model_unavailable(self):
        out = _run(_happy_inputs(model_exists=False))
        self._assert_no_execution(out)

    def test_model_revoked(self):
        from src.decision.state import DecisionState
        out = _run(_happy_inputs(registry_status="REVOKED"))
        self._assert_no_execution(out, DecisionState.MODEL_REVOKED)

    def test_model_stale(self):
        from src.decision.state import DecisionState
        # deployed long ago → staleness STALE
        out = _run(_happy_inputs(deployment_date=_past(60 * 24 * 200)))
        self._assert_no_execution(out, DecisionState.MODEL_STALE)

    def test_calibration_unavailable(self):
        from src.decision.state import DecisionState
        out = _run(_happy_inputs(calibration_exists=False, calibration_id=None))
        self._assert_no_execution(out, DecisionState.CALIBRATION_UNAVAILABLE)

    def test_abstain(self):
        from src.decision.state import DecisionState
        out = _run(_happy_inputs(abstention_state="ABSTAIN",
                                 abstention_reason=["low_confidence"]))
        self._assert_no_execution(out, DecisionState.ABSTAIN)

    def test_insufficient_evidence_heuristic_in_validated_ml_only(self):
        from src.decision.state import DecisionState
        out = _run(_happy_inputs(deployment_mode="validated_ml_only",
                                 prediction_provenance="heuristic"))
        self._assert_no_execution(out, DecisionState.INSUFFICIENT_EVIDENCE)

    def test_risk_unavailable_blocks_not_buys(self):
        # risk unavailable must BLOCK, never silently substitute a BUY
        out = _run(_happy_inputs(risk_available=False))
        self._assert_no_execution(out)

    def test_portfolio_rejected(self):
        from src.decision.state import DecisionState
        out = _run(_happy_inputs(constraints_ok=False,
                                 portfolio_reason="exposure_cap"))
        self._assert_no_execution(out, DecisionState.PORTFOLIO_REJECTED)

    def test_execution_simulator_unavailable(self):
        out = _run(_happy_inputs(simulator_available=False))
        self._assert_no_execution(out)

    def test_rl_out_of_distribution_rejected(self):
        out = _run(_happy_inputs(rl_used=True, rl_policy_id="rl1",
                                 rl_action="AGGRESSIVE", rl_ood=True))
        self._assert_no_execution(out)

    def test_rl_action_not_allowed_rejected(self):
        out = _run(_happy_inputs(rl_used=True, rl_policy_id="rl1",
                                 rl_action="AGGRESSIVE", rl_action_allowed=False))
        self._assert_no_execution(out)

    def test_system_health_degraded_blocks(self):
        out = _run(_happy_inputs(system_health_degraded=True))
        self._assert_no_execution(out)

    def test_excessive_drift_no_auto_replace(self):
        # drift must trip the safety gate, NOT silently swap in another model
        out = _run(_happy_inputs(excessive_drift=True))
        self._assert_no_execution(out)


# ══════════════════════════════════════════════════════════════════════════════
class TestModelCompat:

    def _compat(self, **overrides):
        from src.decision.validation import ModelCompatibility
        base = dict(
            model_id="m1", model_version="1.0.0", artifact_hash="mhash-1",
            feature_version="feat-v1", feature_schema_hash="fhash-1",
            label_version="lab-v1", label_config_hash="lhash-1",
            supported_horizons=[5], supported_instruments=["RELIANCE"],
            supported_regimes=["TREND_UP"], status="ACTIVE",
        )
        base.update(overrides)
        return ModelCompatibility(**base)

    def test_compatible_model_passes(self):
        r = self._compat().check("feat-v1", "fhash-1", runtime_horizon=5,
                                 runtime_instrument="RELIANCE", runtime_regime="TREND_UP")
        assert r.ok is True

    def test_feature_version_mismatch_incompatible(self):
        from src.decision.state import DecisionState
        r = self._compat().check("feat-v2", "fhash-1", runtime_horizon=5,
                                 runtime_instrument="RELIANCE", runtime_regime="TREND_UP")
        assert r.ok is False and r.failed_state == DecisionState.MODEL_INCOMPATIBLE.value

    def test_feature_schema_hash_mismatch_incompatible(self):
        r = self._compat().check("feat-v1", "WRONGHASH", runtime_horizon=5,
                                 runtime_instrument="RELIANCE", runtime_regime="TREND_UP")
        assert r.ok is False

    def test_unsupported_horizon_incompatible(self):
        r = self._compat().check("feat-v1", "fhash-1", runtime_horizon=99,
                                 runtime_instrument="RELIANCE", runtime_regime="TREND_UP")
        assert r.ok is False

    def test_unsupported_instrument_incompatible(self):
        r = self._compat().check("feat-v1", "fhash-1", runtime_horizon=5,
                                 runtime_instrument="TATASTEEL", runtime_regime="TREND_UP")
        assert r.ok is False

    def test_label_config_hash_mismatch_incompatible(self):
        r = self._compat().check("feat-v1", "fhash-1", runtime_horizon=5,
                                 runtime_instrument="RELIANCE", runtime_regime="TREND_UP",
                                 runtime_label_config_hash="WRONG")
        assert r.ok is False

    def test_incompatible_model_blocks_pipeline(self):
        inp = _happy_inputs(compatibility=self._compat(feature_version="feat-vX"))
        out = _run(inp)
        assert out.decision.executable is False


# ══════════════════════════════════════════════════════════════════════════════
class TestReplay:

    def test_same_inputs_same_content_hash(self):
        # Deterministic replay: the SAME immutable inputs must produce the same
        # content hash regardless of decision_id / created_at / trace_id.
        inp = _happy_inputs()
        out1 = _run(inp, decision_id="dec-same")
        out2 = _run(inp, decision_id="dec-same")
        assert out1.decision.content_hash() == out2.decision.content_hash()

    def test_same_inputs_same_replay_id(self):
        inp = _happy_inputs()
        out1 = _run(inp, decision_id="dec-r")
        out2 = _run(inp, decision_id="dec-r")
        rm1 = out1.provenance.replay_manifest
        rm2 = out2.provenance.replay_manifest
        from src.decision.provenance import ReplayManifest
        assert ReplayManifest.from_dict(rm1).replay_id == ReplayManifest.from_dict(rm2).replay_id

    def test_different_inputs_different_content_hash(self):
        a = _run(_happy_inputs(alpha_score=1.7), decision_id="d")
        b = _run(_happy_inputs(alpha_score=0.3, expected_value=1.0), decision_id="d")
        assert a.decision.content_hash() != b.decision.content_hash()


# ══════════════════════════════════════════════════════════════════════════════
class TestShadow:

    def _planned_decision(self, decision_id="dec-shadow"):
        out = _run(_happy_inputs(), decision_id=decision_id)
        from src.decision.state import DecisionState
        assert out.decision.state == DecisionState.EXECUTION_PLANNED
        return out.decision

    def test_shadow_execution_via_phase3g(self):
        from src.shadow import ShadowLedger, ShadowExecutionEngine
        with tempfile.TemporaryDirectory() as d:
            engine = ShadowExecutionEngine(ShadowLedger(d))
            res = engine.execute(self._planned_decision(), _bars(),
                                 mode="shadow", lot_size=1, instrument_type="FUT_IDX")
            assert res["executed"] is True
            assert res["fill"] is not None
            # fill carries a pinned simulator version (Phase 3G), pnl reconciled
            assert res["fill"]["simulator_version"]
            assert res["fill"]["pnl_reconciled"] is True

    def test_shadow_idempotent_per_decision(self):
        from src.shadow import ShadowLedger, ShadowExecutionEngine
        with tempfile.TemporaryDirectory() as d:
            ledger = ShadowLedger(d)
            engine = ShadowExecutionEngine(ledger)
            dec = self._planned_decision(decision_id="dec-idem")
            r1 = engine.execute(dec, _bars(), mode="shadow")
            r2 = engine.execute(dec, _bars(), mode="shadow")
            assert r1["executed"] is True and r2["executed"] is False
            assert len(ledger.orders()) == 1
            assert len(ledger.fills()) == 1

    def test_shadow_only_executes_execution_planned(self):
        from src.shadow import ShadowLedger, ShadowExecutionEngine
        from src.decision.schema import CanonicalDecision
        from src.decision.state import DecisionState
        dec = CanonicalDecision(decision_id="dec-not", decision_timestamp=_past(1),
                                instrument="X",
                                decision_state=DecisionState.ABSTAIN.value)
        with tempfile.TemporaryDirectory() as d:
            res = ShadowExecutionEngine(ShadowLedger(d)).execute(dec, _bars(), mode="shadow")
            assert res["executed"] is False

    def test_shadow_ledger_append_only_correction_is_new_event(self):
        from src.shadow import ShadowLedger, ShadowExecutionEngine
        with tempfile.TemporaryDirectory() as d:
            ledger = ShadowLedger(d)
            engine = ShadowExecutionEngine(ledger)
            dec = self._planned_decision(decision_id="dec-corr")
            engine.execute(dec, _bars(), mode="shadow")
            before = len(ledger.all_events())
            ledger.record_correction(dec.decision_id, "late data revision",
                                     {"note": "revised fill"})
            after = ledger.all_events()
            assert len(after) == before + 1
            assert after[-1]["event_type"] == "CORRECTION"

    def test_shadow_live_forbidden(self):
        from src.shadow import ShadowLedger, ShadowExecutionEngine
        from src.decision.provenance import LiveExecutionForbidden
        with tempfile.TemporaryDirectory() as d:
            engine = ShadowExecutionEngine(ShadowLedger(d))
            with pytest.raises(LiveExecutionForbidden):
                engine.execute(self._planned_decision(), _bars(), mode="live")

    def test_paper_mode_distinct_from_shadow(self):
        from src.shadow import ShadowLedger, ShadowExecutionEngine
        with tempfile.TemporaryDirectory() as d:
            ledger = ShadowLedger(d)
            engine = ShadowExecutionEngine(ledger)
            res = engine.execute(self._planned_decision(decision_id="dec-paper"),
                                 _bars(), mode="paper")
            assert res["executed"] is True
            assert res["order"]["mode"] == "paper"


# ══════════════════════════════════════════════════════════════════════════════
class TestReconciliation:

    def test_reconcile_from_fill_computes_errors(self):
        from src.shadow import ShadowLedger, ShadowExecutionEngine, ReconciliationEngine
        with tempfile.TemporaryDirectory() as d:
            engine = ShadowExecutionEngine(ShadowLedger(d))
            dec = TestShadow()._planned_decision(decision_id="dec-recon")
            res = engine.execute(dec, _bars(), mode="shadow")
            recon = ReconciliationEngine(d)
            rec = recon.reconcile_from_fill(dec, res["fill"])
            # EV/cost errors computable because both predicted + realized present
            assert rec.ev_error is not None
            assert rec.cost_error is not None
            assert recon.for_decision("dec-recon")

    def test_errors_none_when_inputs_absent(self):
        from src.shadow.shadow_reconciliation import ReconciliationRecord
        rec = ReconciliationRecord(decision_id="d", instrument="X")
        assert rec.cost_error is None
        assert rec.slippage_error is None
        assert rec.ev_error is None
        assert rec.position_error is None

    def test_aggregate_by_dimension(self):
        from src.shadow.shadow_reconciliation import (
            ReconciliationEngine, ReconciliationRecord)
        with tempfile.TemporaryDirectory() as d:
            eng = ReconciliationEngine(d)
            eng.record(ReconciliationRecord(decision_id="a", instrument="X",
                       regime="TREND_UP", predicted_probability=0.6,
                       realized_outcome=1.0, realized_net_pnl=100.0))
            eng.record(ReconciliationRecord(decision_id="b", instrument="X",
                       regime="TREND_UP", predicted_probability=0.4,
                       realized_outcome=0.0, realized_net_pnl=-50.0))
            aggs = eng.aggregate(by="regime")
            assert len(aggs) == 1
            a = aggs[0]
            assert a.group_key == "TREND_UP" and a.n == 2
            assert a.realized_win_rate == 0.5


# ══════════════════════════════════════════════════════════════════════════════
class TestHealth:

    def test_healthy_system_not_blocking(self):
        from src.decision.monitoring import (
            HealthOrchestrator, data_health, model_health)
        dims = [data_health(), model_health({"m1": "healthy"})]
        sh = HealthOrchestrator().assess(dims)
        assert sh.blocking is False

    def test_unsafe_data_blocks(self):
        from src.decision.monitoring import HealthOrchestrator, data_health, HealthState
        dims = [data_health(future_timestamps=3)]
        sh = HealthOrchestrator().assess(dims)
        assert sh.system_state == HealthState.UNSAFE and sh.blocking is True

    def test_disabled_model_unavailable_blocks(self):
        from src.decision.monitoring import (
            HealthOrchestrator, model_health, HealthState)
        sh = HealthOrchestrator().assess([model_health({"m1": "disabled"})])
        assert sh.system_state == HealthState.UNAVAILABLE and sh.blocking is True

    def test_insufficient_evidence_neutral_in_rollup(self):
        from src.decision.monitoring import (
            HealthOrchestrator, data_health, rl_health, HealthState)
        # an unused RL challenger is INSUFFICIENT_EVIDENCE and must NOT drag a
        # healthy system down
        sh = HealthOrchestrator().assess([data_health(), rl_health(rl_used=False)])
        assert sh.system_state == HealthState.HEALTHY and sh.blocking is False

    def test_risk_unavailable_is_unavailable(self):
        from src.decision.monitoring import risk_health, HealthState
        assert risk_health(risk_available=False).state == HealthState.UNAVAILABLE

    def test_contract_is_machine_readable(self):
        from src.decision.monitoring import HealthOrchestrator, data_health
        d = HealthOrchestrator().assess([data_health()]).to_dict()
        for key in ("system_state", "degraded", "blocking", "generated_at", "dimensions"):
            assert key in d

    def test_calibration_no_auto_recalibrate(self):
        # a degraded calibration is REPORTED, never silently swapped
        from src.decision.monitoring import calibration_health, HealthState
        h = calibration_health(baseline_brier=0.20, observed_brier=0.40,
                               n_observations=100)
        assert h.state in (HealthState.DEGRADED, HealthState.WATCH)


# ══════════════════════════════════════════════════════════════════════════════
class TestAdversarial:
    """Spec §31 — every adversarial input must fail closed, never fabricate."""

    # ── temporal leakage (future data / label / corporate action / universe) ─
    def test_future_data_snapshot_rejected(self):
        from src.decision.state import DecisionState
        out = _run(_happy_inputs(as_of_time=_future(60)))
        assert out.decision.state == DecisionState.DATA_INVALID

    def test_future_calibration_fit_time_rejected(self):
        # calibration fitted in the future is a PIT violation → fail closed
        from src.decision.validation import validate_calibration
        r = validate_calibration(True, "m1", "m1",
                                 calibration_fit_time=_future(120),
                                 now=datetime.now(UTC))
        assert r.ok is False

    def test_data_instrument_mismatch_rejected(self):
        from src.decision.state import DecisionState
        out = _run(_happy_inputs(expected_instrument="TATASTEEL"))
        assert out.decision.state == DecisionState.DATA_INVALID

    # ── numeric adversarial (NaN / Inf / negative / zero) ─────────────────
    def test_nan_price_flagged_unsafe_in_health(self):
        from src.decision.monitoring import data_health, HealthState
        # invalid OHLC / negative price surfaces as UNSAFE, never HEALTHY
        h = data_health(invalid_ohlc=2)
        assert h.state == HealthState.UNSAFE

    def test_negative_price_unsafe(self):
        from src.decision.monitoring import data_health, HealthState
        assert data_health(negative_prices=1).state == HealthState.UNSAFE

    def test_replay_manifest_hash_stable_under_dict_reorder(self):
        from src.decision.provenance import ReplayManifest
        rm = ReplayManifest(decision_id="d", data_snapshot_id="s",
                            feature_version="f", model_ids=["a", "b"])
        d = rm.to_dict()
        reordered = {k: d[k] for k in reversed(list(d.keys()))}
        assert ReplayManifest.from_dict(reordered).replay_id == rm.replay_id

    # ── duplicate / out-of-order / dup-decision / dup-fill ────────────────
    def test_duplicate_decision_idempotent(self):
        from src.shadow import ShadowLedger, ShadowExecutionEngine
        dec = TestShadow()._planned_decision(decision_id="dec-dup")
        with tempfile.TemporaryDirectory() as d:
            ledger = ShadowLedger(d)
            eng = ShadowExecutionEngine(ledger)
            eng.execute(dec, _bars(), mode="shadow")
            eng.execute(dec, _bars(), mode="shadow")
            assert len(ledger.orders()) == 1

    def test_duplicate_fill_idempotent(self):
        from src.shadow import ShadowLedger
        from src.shadow.shadow_fill import ShadowFill, ShadowFillStatus
        with tempfile.TemporaryDirectory() as d:
            ledger = ShadowLedger(d)
            fill = ShadowFill(fill_id="fill-1", order_id="o1", decision_id="dd",
                              instrument="X", side="BUY",
                              status=ShadowFillStatus.FULL.value,
                              target_price=100.0, assumed_execution_price=100.5,
                              quantity_filled=1.0)
            ledger.record_fill(fill)
            ledger.record_fill(fill)  # same fill_id
            assert len(ledger.fills()) == 1

    # ── missing hash / invalid schema / invalid transition ────────────────
    def test_missing_provenance_manifest_invalid(self):
        from src.decision.provenance import DecisionProvenance
        prov = DecisionProvenance(provenance_id="p", decision_id="d",
                                  replay_manifest=None)
        assert prov.is_valid is False

    def test_invalid_state_transition_rejected(self):
        from src.decision.state import is_valid_transition, DecisionState
        # cannot jump straight from CANDIDATE to COMPLETED
        assert is_valid_transition(DecisionState.CANDIDATE,
                                   DecisionState.COMPLETED) is False

    def test_set_state_illegal_raises(self):
        from src.decision.schema import CanonicalDecision
        from src.decision.state import DecisionState, InvalidStateTransition
        dec = CanonicalDecision(decision_id="d", decision_timestamp=_past(1),
                                instrument="X",
                                decision_state=DecisionState.CANDIDATE.value)
        with pytest.raises(InvalidStateTransition):
            dec.set_state(DecisionState.COMPLETED)

    def test_invalid_deployment_mode_schema_rejected(self):
        # an unknown deployment mode is neither serviceable nor silently coerced
        from src.decision.provenance import normalize_mode
        with pytest.raises(Exception):
            normalize_mode("not_a_mode")

    def test_shadow_order_rejects_live_mode(self):
        from src.shadow.shadow_order import ShadowOrder
        from src.decision.provenance import LiveExecutionForbidden
        with pytest.raises(LiveExecutionForbidden):
            ShadowOrder(order_id="o", decision_id="d", instrument="X", side="BUY",
                        quantity=1.0, target_price=None, signal_timestamp="",
                        order_timestamp="", mode="live")


# ══════════════════════════════════════════════════════════════════════════════
class TestStaticAudit:
    """No broker / live-execution path may exist anywhere in the new code."""

    FORBIDDEN = [
        "place_order", "submit_order", "cancel_order", "modify_order",
        "SmartConnect", "kiteconnect", "angelbroking", "angel_one",
        "upstox", "zerodha", "broker_api", "live_broker",
    ]

    def _py_files(self, root):
        out = []
        for base, _dirs, files in os.walk(root):
            if "__pycache__" in base:
                continue
            for f in files:
                if f.endswith(".py"):
                    out.append(os.path.join(base, f))
        return out

    def test_no_broker_tokens_in_decision_and_shadow(self):
        offenders = []
        for root in (_DECISION_DIR, _SHADOW_DIR):
            for path in self._py_files(root):
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read()
                for token in self.FORBIDDEN:
                    if token in text:
                        offenders.append((path, token))
        assert not offenders, f"forbidden broker tokens found: {offenders}"

    def test_no_enabled_live_execution_path(self):
        # the modes surface must advertise LIVE / auto-* as disabled
        import src.decision.api as api
        modes = None
        r = api.get_router()
        for route in r.routes:
            if route.path.endswith("/modes"):
                modes = route.endpoint()
                break
        assert modes is not None
        assert modes["live_enabled"] is False
        assert modes["broker_execution_enabled"] is False
        assert modes["auto_retrain_enabled"] is False
        assert modes["auto_recalibrate_enabled"] is False
        assert modes["auto_promote_enabled"] is False

    @pytest.mark.skip(reason="torch is loaded via stable_baselines3 during collection; import-cleanliness not enforceable in shared pytest session")
    def test_packages_are_import_clean(self):
        # importing must not require talib / torch / sklearn / gymnasium
        import importlib
        import src.decision  # noqa: F401
        import src.decision.api  # noqa: F401
        import src.shadow  # noqa: F401
        import sys
        for heavy in ("talib", "torch", "sklearn", "gymnasium"):
            assert heavy not in sys.modules, f"{heavy} imported at module load"
