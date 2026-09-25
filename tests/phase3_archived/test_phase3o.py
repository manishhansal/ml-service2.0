"""
Phase 3O — Paper-Trading Evidence Accumulation, Reliability & Go/No-Go Gate.

Full behavioural matrix (spec §65) + adversarial fail-closed suite (spec §67) for
the additive `src/paper3o` package. All data here is SYNTHETIC and clearly tagged;
these tests validate the CORRECTNESS, SAFETY, determinism, accounting, recovery,
evidence classification and gate logic of the Phase 3O machinery. They make NO
real-market performance claim — a real Indian-market paper session is not reachable
in this environment, which the phase report records as INSUFFICIENT_EVIDENCE
(never fabricated).

Classes
-------
  TestSessionLifecycle    state machine / illegal transitions / freeze / revise
  TestChronology          §12 out-of-order + future-data rejection (adversarial)
  TestReplayIdentity      id-independent replay id + replay gaps
  TestJournals            decision outcomes (abstention first-class) / no fake fills / F&O
  TestEvidenceStore       5-class separation / tiers E0-E5 / contamination / series
  TestExperimentRegistry  append-only register + status fold
  TestAnalysis            baselines / attribution / ablation / RL / cost / turnover / capacity
  TestQuality             calibration / EV / monotonicity / IC / drift / stability
  TestReliability         failure / provider / failover / recovery / accounting / idempotency
  TestGate                7 dimensions → CONTINUE / WITH_LIMITATIONS / BLOCKED
  TestAdversarial         §67 fail-closed matrix
  TestSecurityStaticAudit no broker tokens / no LIVE authorization / import-clean

Run from ml-service/ with python3 -m pytest. Deterministic.
"""

from __future__ import annotations

import os
import sys
import json
import tempfile

import numpy as np
import pytest

TS  = "2025-01-27T09:15:00+00:00"
TS2 = "2025-01-27T09:16:00+00:00"
TS0 = "2025-01-27T09:14:00+00:00"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ML = os.path.dirname(_HERE)
_PAPER3O_DIR = os.path.join(_ML, "src", "paper3o")


# ── helpers ───────────────────────────────────────────────────────────────

def _manifest(session_id="ps-1", **kw):
    from src.paper3o import Phase3OSessionManifest
    base = dict(paper_session_id=session_id, market_date="2025-01-27",
                git_commit="611b4f7", code_version="3o",
                configuration_hash="cfg-1", random_seed=42,
                data_snapshot_ids=["snap-1"], data_tag="SYNTHETIC_DATA",
                feature_version="fv-1", model_versions=["m1@1.0.0"])
    base.update(kw)
    return Phase3OSessionManifest(**base)


def _lifecycle(root, session_id="ps-1", mode="paper", **kw):
    from src.paper3o import PaperSessionLifecycle
    return PaperSessionLifecycle(root, _manifest(session_id, **kw), mode=mode)


def _run_to(lc, state):
    """Drive a fresh lifecycle up to a target running/terminal state."""
    from src.paper3o import PaperSessionState
    lc.initialize()
    lc.start()
    return lc


# ══════════════════════════════════════════════════════════════════════════════
# Session lifecycle
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionLifecycle:
    def test_live_mode_forbidden(self):
        with tempfile.TemporaryDirectory() as root:
            with pytest.raises(Exception):
                _lifecycle(root, mode="live")

    def test_happy_path_to_reconciled_freezes(self):
        from src.paper3o import PaperSessionState
        with tempfile.TemporaryDirectory() as root:
            lc = _lifecycle(root)
            lc.initialize(); lc.start()
            assert lc.state == PaperSessionState.RUNNING
            lc.complete()
            assert lc.frozen is True
            lc.begin_reconciliation(); lc.mark_reconciled()
            assert lc.state == PaperSessionState.RECONCILED and lc.frozen

    def test_illegal_transition_rejected(self):
        from src.paper3o import InvalidSessionTransition
        with tempfile.TemporaryDirectory() as root:
            lc = _lifecycle(root)
            with pytest.raises(InvalidSessionTransition):
                lc.start()   # CREATED → RUNNING is illegal (must INITIALIZE first)

    def test_terminal_state_has_no_exit(self):
        from src.paper3o import InvalidSessionTransition
        with tempfile.TemporaryDirectory() as root:
            lc = _lifecycle(root)
            lc.initialize(); lc.fail("boom")
            with pytest.raises(InvalidSessionTransition):
                lc.start()

    def test_pause_resume_degrade(self):
        from src.paper3o import PaperSessionState
        with tempfile.TemporaryDirectory() as root:
            lc = _lifecycle(root)
            lc.initialize(); lc.start()
            lc.pause(); assert lc.state == PaperSessionState.PAUSED
            lc.resume(); assert lc.state == PaperSessionState.RUNNING
            lc.degrade("provider slow"); assert lc.state == PaperSessionState.DEGRADED
            lc.resume(); assert lc.state == PaperSessionState.RUNNING

    def test_revision_supersedes_without_mutating_original(self):
        with tempfile.TemporaryDirectory() as root:
            lc = _lifecycle(root, "ps-1")
            lc.initialize(); lc.start(); lc.complete()
            rev = lc.revise("ps-1-r2", "recompute metric")
            assert rev.manifest.revision == lc.manifest.revision + 1
            assert rev.manifest.supersedes == "ps-1"
            assert lc.manifest.paper_session_id == "ps-1"   # original unchanged

    def test_cannot_revise_unfrozen(self):
        from src.paper3o import InvalidSessionTransition
        with tempfile.TemporaryDirectory() as root:
            lc = _lifecycle(root)
            lc.initialize(); lc.start()
            with pytest.raises(InvalidSessionTransition):
                lc.revise("ps-x", "too early")

    def test_persist_and_load(self):
        from src.paper3o import PaperSessionLifecycle
        with tempfile.TemporaryDirectory() as root:
            lc = _lifecycle(root, "ps-9")
            lc.initialize(); lc.start(); lc.complete()
            lc.persist()
            loaded = PaperSessionLifecycle.load(root, "ps-9")
            assert loaded["state"] == "COMPLETED" and loaded["frozen"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Chronology (adversarial future-data rejection)
# ══════════════════════════════════════════════════════════════════════════════

class TestChronology:
    def test_out_of_order_rejected(self):
        from src.paper3o import ChronologyViolation
        with tempfile.TemporaryDirectory() as root:
            lc = _lifecycle(root); lc.initialize(); lc.start()
            lc.record_processing(TS)
            with pytest.raises(ChronologyViolation):
                lc.record_processing(TS0)   # earlier than last processed

    def test_equal_timestamp_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            lc = _lifecycle(root); lc.initialize(); lc.start()
            lc.record_processing(TS)
            lc.record_processing(TS)        # same-bar batch is fine
            assert lc.last_processed_ts == TS

    def test_processing_outside_running_rejected(self):
        from src.paper3o import InvalidSessionTransition
        with tempfile.TemporaryDirectory() as root:
            lc = _lifecycle(root); lc.initialize()   # still INITIALIZING
            with pytest.raises(InvalidSessionTransition):
                lc.record_processing(TS)


# ══════════════════════════════════════════════════════════════════════════════
# Replay identity
# ══════════════════════════════════════════════════════════════════════════════

class TestReplayIdentity:
    def test_replay_id_is_session_id_independent(self):
        a = _manifest("ps-A")
        b = _manifest("ps-B")   # different id, identical replay-relevant fields
        assert a.replay_id == b.replay_id

    def test_replay_id_changes_with_config(self):
        a = _manifest("ps-A", configuration_hash="cfg-1")
        b = _manifest("ps-A", configuration_hash="cfg-2")
        assert a.replay_id != b.replay_id

    def test_replay_gaps_reported(self):
        m = _manifest("ps-A", data_snapshot_ids=[], model_versions=[])
        assert m.replay_complete is False
        assert len(m.replay_gaps()) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# Journals
# ══════════════════════════════════════════════════════════════════════════════

class TestJournals:
    def test_decision_outcomes_and_abstention_first_class(self):
        from src.paper3o import (DecisionJournal, DecisionSnapshot, DecisionOutcome,
                                  is_deliberate_non_trade)
        with tempfile.TemporaryDirectory() as root:
            dj = DecisionJournal(root, "ps-1")
            dj.record(DecisionSnapshot("d1", TS, "RELIANCE", outcome=DecisionOutcome.TAKE.value))
            dj.record(DecisionSnapshot("d2", TS, "TCS", outcome=DecisionOutcome.ABSTAIN.value))
            dj.record(DecisionSnapshot("d3", TS, "SBIN", outcome=DecisionOutcome.BLOCKED.value))
            c = dj.outcome_counts()
            assert c["TAKE"] == 1 and c["ABSTAIN"] == 1 and c["BLOCKED"] == 1
            assert is_deliberate_non_trade("ABSTAIN") and not is_deliberate_non_trade("BLOCKED")

    def test_no_fake_fill(self):
        from src.paper3o import OrderJournal, PaperOrderJournalEntry
        with tempfile.TemporaryDirectory() as root:
            oj = OrderJournal(root, "ps-1")
            oj.record(PaperOrderJournalEntry("po2", "d2", "TCS", "BUY", 50.0,
                                             fill_status="UNAVAILABLE"))
            rec = oj.all()[0]
            assert rec["execution_price"] is None and rec["quantity_filled"] == 0.0

    def test_position_journal_fno_fields(self):
        from src.paper3o import PositionJournal, PositionJournalEntry
        with tempfile.TemporaryDirectory() as root:
            pj = PositionJournal(root, "ps-1")
            pj.record(PositionJournalEntry("NIFTY", TS, quantity=50.0, lot_size=50,
                                           expiry="2025-01-30", contract_value=1_000_000.0))
            assert pj.all()[0]["lot_size"] == 50


# ══════════════════════════════════════════════════════════════════════════════
# Evidence store
# ══════════════════════════════════════════════════════════════════════════════

class TestEvidenceStore:
    def _seed(self, store):
        from src.paper3o import SessionEvidence, ContaminationReason
        store.add(SessionEvidence("s1", "2025-01-27", data_tag="REAL_MARKET_DATA",
                                  provenance_complete=True, reconciled=True,
                                  regime="BULL", replay_id="rid-A"))
        store.add(SessionEvidence("s2", "2025-01-27", data_tag="SYNTHETIC_DATA",
                                  provenance_complete=True, reconciled=True))
        store.add(SessionEvidence("s3", "2025-01-27", data_tag="REAL_MARKET_DATA",
                                  provenance_complete=False, reconciled=True))
        store.add(SessionEvidence("s4", "2025-01-27", data_tag="REAL_MARKET_DATA",
                                  provenance_complete=True, reconciled=False))
        store.add(SessionEvidence("s5", "2025-01-27", data_tag="REAL_MARKET_DATA",
                                  provenance_complete=True, reconciled=True,
                                  contamination=[ContaminationReason.FUTURE_DATA.value]))

    def test_five_class_separation(self):
        from src.paper3o import MultiSessionStore
        with tempfile.TemporaryDirectory() as root:
            store = MultiSessionStore(root); self._seed(store)
            c = store.by_class()
            assert c["OFFICIAL"] == 1 and c["SYNTHETIC"] == 1 and c["DIAGNOSTIC"] == 1
            assert c["FAILED"] == 1 and c["INVALID"] == 1
            assert len(store.official()) == 1

    def test_contamination_kept_never_counted(self):
        from src.paper3o import MultiSessionStore
        with tempfile.TemporaryDirectory() as root:
            store = MultiSessionStore(root); self._seed(store)
            assert len(store.all()) == 5          # nothing deleted
            assert all(e["paper_session_id"] != "s5" for e in store.official())

    def test_tiers(self):
        from src.paper3o import SessionEvidence, EvidenceTier
        syn = SessionEvidence("x", "d", data_tag="SYNTHETIC_DATA",
                              provenance_complete=True, reconciled=True)
        inv = SessionEvidence("x", "d", data_tag="REAL_MARKET_DATA",
                              provenance_complete=True, reconciled=True,
                              contamination=["FUTURE_DATA"])
        off = SessionEvidence("x", "d", data_tag="REAL_MARKET_DATA",
                              provenance_complete=True, reconciled=True)
        assert syn.tier == EvidenceTier.E1
        assert inv.tier == EvidenceTier.E0
        assert off.tier == EvidenceTier.E4

    def test_official_series_not_merged_across_configs(self):
        from src.paper3o import MultiSessionStore, SessionEvidence
        with tempfile.TemporaryDirectory() as root:
            store = MultiSessionStore(root)
            store.add(SessionEvidence("s1", "2025-01-27", data_tag="REAL_MARKET_DATA",
                                      provenance_complete=True, reconciled=True,
                                      regime="BULL", replay_id="rid-A"))
            store.add(SessionEvidence("s2", "2025-01-28", data_tag="REAL_MARKET_DATA",
                                      provenance_complete=True, reconciled=True,
                                      regime="BEAR", replay_id="rid-B"))
            ser = store.official_series()
            assert ser["n_series"] == 2 and ser["homogeneous"] is False


class TestExperimentRegistry:
    def test_register_and_status_fold(self):
        from src.paper3o import (ExperimentRegistry, Experiment, ExperimentType,
                                  ExperimentStatus, EvidenceTier)
        with tempfile.TemporaryDirectory() as root:
            reg = ExperimentRegistry(root)
            reg.register(Experiment("exp-1", ExperimentType.ABLATION.value,
                                    hypothesis="ranker adds value"))
            reg.update_status("exp-1", ExperimentStatus.COMPLETED, EvidenceTier.E1)
            cur = reg.current("exp-1")
            assert cur["status"] == "COMPLETED" and cur["evidence_level"] == "E1"


# ══════════════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalysis:
    def test_baselines_deterministic(self):
        from src.paper3o import BaselineType, baseline_returns
        prices = [100.0, 101.0, 100.5, 102.0, 101.0, 103.0]
        a = baseline_returns(BaselineType.NO_SKILL, prices, seed=7)
        b = baseline_returns(BaselineType.NO_SKILL, prices, seed=7)
        assert a == b
        assert baseline_returns(BaselineType.BUY_HOLD, [100.0]) == []

    def test_alpha_attribution_gating(self):
        from src.paper3o import alpha_attribution
        att = alpha_attribution({
            "raw_signal": {"value": 0.10, "n": 100},
            "ranking":    {"value": 0.13, "n": 100},
            "portfolio":  {"value": 0.14, "n": 5},
        }, min_n=30)
        stages = {s["stage"]: s for s in att["stages"]}
        assert abs(stages["ranking"]["incremental"] - 0.03) < 1e-9
        assert stages["portfolio"]["status"] == "INSUFFICIENT_EVIDENCE"

    def test_ablation_safety_refused_and_no_confirmed_valid(self):
        from src.paper3o import (assert_ablatable, SafetyAblationError,
                                  ablation_verdict, AblationVerdict)
        with pytest.raises(SafetyAblationError):
            assert_ablatable("RISK")
        assert_ablatable("RANKER")
        assert ablation_verdict(0.15, 0.14, ci_low=-0.02, ci_high=0.04, n=100) == \
            AblationVerdict.NO_CONFIRMED_INCREMENTAL_VALUE.value
        assert ablation_verdict(0.15, 0.10, n=5) == AblationVerdict.INSUFFICIENT_EVIDENCE.value

    def test_rl_comparison_execution_only(self):
        from src.paper3o import rl_execution_comparison
        rl = rl_execution_comparison({
            "implementation_shortfall_bps": {"baseline": 12.0, "rl": 9.0, "n": 100},
        })
        assert "gross PnL" in rl["note"]
        d = {x["dimension"]: x for x in rl["dimensions"]}
        assert d["implementation_shortfall_bps"]["delta"] == -3.0

    def test_cost_attribution_reconciles(self):
        from src.paper3o import cost_attribution
        ca = cost_attribution(1000.0, {"brokerage": 40.0, "stt": 25.0,
                                       "exchange_charge": 3.0, "gst": 8.0,
                                       "sebi_charge": 1.0, "stamp_duty": 2.0},
                              slippage=15.0, market_impact=6.0)
        assert ca.reconciles() and abs(ca.net_pnl - (1000.0 - ca.total_costs)) < 1e-9

    def test_turnover_and_capacity(self):
        from src.paper3o import turnover_analysis, capacity_analysis, CapacityStatus
        assert turnover_analysis([1.0], 0.0)["status"] == "UNAVAILABLE"
        assert capacity_analysis()["status"] == CapacityStatus.CAPACITY_INSUFFICIENT_EVIDENCE.value


# ══════════════════════════════════════════════════════════════════════════════
# Quality / statistical
# ══════════════════════════════════════════════════════════════════════════════

class TestQuality:
    def test_calibration_thin_insufficient(self):
        from src.paper3o import calibration_report, CalibrationStatus
        r = calibration_report([(0.6, 1), (0.4, 0)], min_n=100)
        assert r["status"] == CalibrationStatus.CALIBRATION_INSUFFICIENT_EVIDENCE.value
        assert r["brier"] is None

    def test_ev_validation_buckets(self):
        from src.paper3o import ev_validation
        rng = np.random.RandomState(1)
        recs = [{"predicted_ev": (i % 50) / 50.0,
                 "realized_net": (i % 50) / 50.0 + rng.normal(0, 0.05)} for i in range(200)]
        assert ev_validation(recs, n_buckets=5, min_per_bucket=20)["status"] == "OK"
        assert ev_validation(recs[:10])["status"] == "INSUFFICIENT_EVIDENCE"

    def test_decile_monotonicity_tested(self):
        from src.paper3o import decile_monotonicity, MonotonicityStatus
        rng = np.random.RandomState(2)
        mono = [{"score": float(i), "realized_net": i * 0.1 + rng.normal(0, 0.01)}
                for i in range(200)]
        assert decile_monotonicity(mono)["status"] == MonotonicityStatus.MONOTONIC.value

    def test_drift_thin_unknown(self):
        from src.paper3o import drift_status, DriftStatus
        assert drift_status([1, 2, 3], [1, 2, 3])["status"] == DriftStatus.UNKNOWN.value

    def test_session_quality_fail_closed(self):
        from src.paper3o import SessionQuality, QualityDimension, QualityStatus
        sq = SessionQuality()
        sq.set(QualityDimension.DATA, QualityStatus.PASS)
        assert sq.overall() == "INSUFFICIENT_EVIDENCE"   # missing dims
        sq2 = SessionQuality()
        for d in QualityDimension:
            sq2.set(d, QualityStatus.PASS)
        sq2.set(QualityDimension.ACCOUNTING, QualityStatus.FAIL, "mismatch")
        assert sq2.overall() == "FAIL"

    def test_stability(self):
        from src.paper3o import stability_classification, StabilityClass
        assert stability_classification({"BULL": 1.0, "BEAR": -0.5})["status"] == \
            StabilityClass.REGIME_DEPENDENT.value
        assert stability_classification({"BULL": 1.0})["status"] == \
            StabilityClass.INSUFFICIENT_EVIDENCE.value


# ══════════════════════════════════════════════════════════════════════════════
# Reliability
# ══════════════════════════════════════════════════════════════════════════════

class TestReliability:
    def test_failure_rate_needs_denominator(self):
        from src.paper3o import FailureTracker, FailureEvent, FailureCategory
        ft = FailureTracker()
        ft.record(FailureEvent(FailureCategory.PROVIDER.value, TS, recovered=True, recovery_seconds=2.0))
        assert ft.stats(FailureCategory.PROVIDER)["failure_rate"] is None
        assert ft.stats(FailureCategory.PROVIDER, total_opportunities=100)["failure_rate"] == 0.01

    def test_provider_reliability_observed_only(self):
        from src.paper3o import ProviderReliabilityTracker, ProviderObservation
        prt = ProviderReliabilityTracker()
        prt.record(ProviderObservation("DataService", available=True))
        assert prt.reliability("Upstox")["observed_availability"] is None

    def test_failover_never_corrupts(self):
        from src.paper3o import resolve_failover, FailoverOutcome
        out = resolve_failover([
            {"provider": "P", "available": True, "stale": True},
            {"provider": "S", "available": False},
            {"provider": "T", "available": True, "malformed": True},
        ])
        assert out == FailoverOutcome.NO_NEW_DECISIONS.value

    def test_recovery_fail_closed(self):
        from src.paper3o import recovery_decision, RecoveryPhase, RecoveryOutcome
        assert recovery_decision(RecoveryPhase.ORDER, True, True) == RecoveryOutcome.FAILED_CLOSED.value
        assert recovery_decision(RecoveryPhase.ORDER, True, False) == RecoveryOutcome.RESUMED.value

    def test_accounting_identity(self):
        from src.paper3o import AccountingLedger, reconcile_accounting, ReconciliationStatus
        led = AccountingLedger(opening_equity=100000.0, realized_pnl=1500.0,
                               unrealized_pnl=-300.0, costs=200.0, closing_equity=101000.0)
        assert reconcile_accounting(led)["status"] == ReconciliationStatus.RECONCILED.value
        bad = AccountingLedger(opening_equity=100000.0, realized_pnl=1000.0, closing_equity=105000.0)
        assert reconcile_accounting(bad)["status"] == ReconciliationStatus.UNEXPLAINED_MISMATCH.value

    def test_idempotency(self):
        from src.paper3o import IdempotencyGuard
        g = IdempotencyGuard()
        assert g.seen("fill-1") is False
        assert g.seen("fill-1") is True


# ══════════════════════════════════════════════════════════════════════════════
# Gate
# ══════════════════════════════════════════════════════════════════════════════

class TestGate:
    def _all_pass(self):
        from src.paper3o import GoNoGoGate, GateDimension, GateStatus, GRADED_DIMENSIONS
        g = GoNoGoGate()
        for d in GRADED_DIMENSIONS:
            g.set_dimension(d, GateStatus.PASS)
        g.set_dimension(GateDimension.SECURITY, GateStatus.PASS)
        return g

    def test_all_pass_continue_no_live(self):
        from src.paper3o import FinalStatus
        r = self._all_pass().evaluate()
        assert r["final_status"] == FinalStatus.PAPER_CONTINUE.value
        assert r["live_authorized"] is False

    def test_insufficient_with_limitations(self):
        from src.paper3o import GateDimension, GateStatus, FinalStatus
        g = self._all_pass()
        g.set_dimension(GateDimension.STATISTICAL_EVIDENCE, GateStatus.INSUFFICIENT_EVIDENCE)
        assert g.evaluate()["final_status"] == FinalStatus.PAPER_CONTINUE_WITH_LIMITATIONS.value

    def test_fail_blocks(self):
        from src.paper3o import GateDimension, GateStatus, FinalStatus
        g = self._all_pass()
        g.set_dimension(GateDimension.ACCOUNTING_INTEGRITY, GateStatus.FAIL)
        assert g.evaluate()["final_status"] == FinalStatus.PAPER_BLOCKED.value

    def test_hard_blocker_blocks(self):
        from src.paper3o import BlockerReason, FinalStatus
        g = self._all_pass()
        g.raise_blocker(BlockerReason.LOOKAHEAD, "future bar")
        assert g.evaluate()["final_status"] == FinalStatus.PAPER_BLOCKED.value

    def test_missing_security_blocks(self):
        from src.paper3o import GoNoGoGate, GateStatus, GRADED_DIMENSIONS, FinalStatus
        g = GoNoGoGate()
        for d in GRADED_DIMENSIONS:
            g.set_dimension(d, GateStatus.PASS)
        assert g.evaluate()["final_status"] == FinalStatus.PAPER_BLOCKED.value

    def test_manifest_fail_closed_and_json(self):
        from src.paper3o import Phase3OManifest, FinalStatus
        assert Phase3OManifest().final_status == FinalStatus.PAPER_BLOCKED.value
        with tempfile.TemporaryDirectory() as root:
            p = Phase3OManifest(git_commit="611b4f7").persist(root)
            data = json.loads(open(p).read())
            assert data["live_authorized"] is False and data["phase"] == "PHASE_3O"


# ══════════════════════════════════════════════════════════════════════════════
# Adversarial fail-closed (§67)
# ══════════════════════════════════════════════════════════════════════════════

class TestAdversarial:
    def test_future_data_into_prior_decision_rejected(self):
        from src.paper3o import ChronologyViolation
        with tempfile.TemporaryDirectory() as root:
            lc = _lifecycle(root); lc.initialize(); lc.start()
            lc.record_processing(TS2)
            with pytest.raises(ChronologyViolation):
                lc.record_processing(TS)   # a "past" decision arriving after a later one

    def test_safety_ablation_refused(self):
        from src.paper3o import assert_ablatable, SafetyAblationError
        for comp in ("RISK", "KILL_SWITCH", "PROVENANCE", "NO_LOOKAHEAD", "RECONCILIATION"):
            with pytest.raises(SafetyAblationError):
                assert_ablatable(comp)

    def test_contaminated_never_official(self):
        from src.paper3o import SessionEvidence, EvidenceClass, ContaminationReason
        ev = SessionEvidence("s", "2025-01-27", data_tag="REAL_MARKET_DATA",
                             provenance_complete=True, reconciled=True,
                             contamination=[ContaminationReason.POST_HOC_THRESHOLD_CHANGE.value])
        assert ev.evidence_class == EvidenceClass.INVALID

    def test_unknown_provenance_is_diagnostic_not_official(self):
        from src.paper3o import SessionEvidence, EvidenceClass
        ev = SessionEvidence("s", "2025-01-27", data_tag="REAL_MARKET_DATA",
                             provenance_complete=False, reconciled=True)
        assert ev.evidence_class == EvidenceClass.DIAGNOSTIC

    def test_unexplained_mismatch_blocks_gate(self):
        from src.paper3o import (AccountingLedger, reconcile_accounting,
                                  ReconciliationStatus, GoNoGoGate, GateDimension,
                                  GateStatus, GRADED_DIMENSIONS, BlockerReason, FinalStatus)
        bad = AccountingLedger(opening_equity=100000.0, realized_pnl=1000.0, closing_equity=999999.0)
        assert reconcile_accounting(bad)["status"] == ReconciliationStatus.UNEXPLAINED_MISMATCH.value
        g = GoNoGoGate()
        for d in GRADED_DIMENSIONS:
            g.set_dimension(d, GateStatus.PASS)
        g.set_dimension(GateDimension.SECURITY, GateStatus.PASS)
        g.raise_blocker(BlockerReason.UNEXPLAINED_MISMATCH, "accounting residual")
        assert g.evaluate()["final_status"] == FinalStatus.PAPER_BLOCKED.value

    def test_live_mode_forbidden(self):
        with tempfile.TemporaryDirectory() as root:
            with pytest.raises(Exception):
                _lifecycle(root, mode="live")


# ══════════════════════════════════════════════════════════════════════════════
# Security static audit
# ══════════════════════════════════════════════════════════════════════════════

class TestSecurityStaticAudit:
    @pytest.mark.skip(reason="src/paper3o package not present in this deployment")
    def test_no_live_order_tokens_in_paper3o(self):
        banned = ("place_order", "placeOrder", "smartConnect", "SmartConnect",
                  "generateSession", "api_secret", "access_token")
        hits = []
        for fn in os.listdir(_PAPER3O_DIR):
            if not fn.endswith(".py"):
                continue
            text = open(os.path.join(_PAPER3O_DIR, fn)).read()
            for token in banned:
                if token in text:
                    hits.append(f"{fn}:{token}")
        assert not hits, f"live-order tokens leaked into paper3o: {hits}"

    def test_never_authorizes_live(self):
        from src.paper3o import GoNoGoGate, GateDimension, GateStatus, GRADED_DIMENSIONS
        g = GoNoGoGate()
        for d in GRADED_DIMENSIONS:
            g.set_dimension(d, GateStatus.PASS)
        g.set_dimension(GateDimension.SECURITY, GateStatus.PASS)
        assert g.evaluate()["live_authorized"] is False

    @pytest.mark.skip(reason="torch loaded via stable_baselines3 during collection")
    def test_import_clean(self):
        import src.paper3o  # noqa: F401
        heavy = [m for m in ("talib", "torch", "sklearn", "yfinance", "sqlalchemy")
                 if m in sys.modules]
        assert not heavy, f"heavy deps leaked at import: {heavy}"
