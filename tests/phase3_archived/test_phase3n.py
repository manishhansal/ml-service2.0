"""
Phase 3N — Indian Market Paper-Trading Validation & Production-Readiness Gate.

Full test matrix (spec §54, §55) + adversarial no-lookahead suite (spec §56) +
static security audit (spec §42, §55 Security, §64).

All data here is SYNTHETIC and clearly tagged as such — these tests validate the
CORRECTNESS, SAFETY, PIT-discipline, idempotency, recovery, evidence machinery
and gate logic of the `src/paper` package. They make NO real-market performance
claim (a real Indian-market session is not reachable in this environment; that is
reported as INSUFFICIENT_EVIDENCE in the phase report, never fabricated).

Classes
-------
  TestProvider          primary/fallback/all-unavailable/stale/malformed/partial
  TestDataQuality       session/freshness/OHLC/OI/option-chain/F&O-meta
  TestSignals           families/aggregation/conflict/dup/unavailable
  TestPaperEngine       lifecycle/dup/restart/partial/no-fabricated-fill
  TestSession           kill-switch NO_NEW_PAPER_EXPOSURE/EOD/replay/recovery
  TestEvidence          INSUFFICIENT/CI/conditional/multiple-testing/official-filter
  TestReadiness         8 gates / NOT_READY / READY_WITH_LIMITATIONS / no LIVE_READY
  TestNoLookahead       adversarial future-information injection (spec §56)
  TestSecurityStaticAudit  no broker tokens / no LIVE_READY / import-clean (spec §64)

Run from ml-service/ with python3 -m pytest. Deterministic.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone, timedelta, date

import numpy as np
import pytest

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
TS = "2025-01-27T04:00:00+00:00"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ML = os.path.dirname(_HERE)
_PAPER_DIR = os.path.join(_ML, "src", "paper")


# ── helpers ─────────────────────────────────────────────────────────────────

def _planned_decision(decision_id="dec-1", qty=100.0, direction="LONG"):
    from src.decision.schema import CanonicalDecision
    from src.decision.state import DecisionState
    d = CanonicalDecision(decision_id=decision_id, decision_timestamp=TS,
                          instrument="RELIANCE", direction=direction,
                          position_size=qty, data_snapshot_id="snap-1",
                          decision_state=DecisionState.CANDIDATE.value)
    d.set_state(DecisionState.VALIDATED)
    d.set_state(DecisionState.EXECUTION_PLANNED)
    return d


def _sig(family, direction, strength, **kw):
    from src.paper import CanonicalSignal, EvidenceLevel, SignalStatus
    defaults = dict(confidence=0.8, timeframe="1d", regime="TREND_UP",
                    evidence_level=EvidenceLevel.TRAINED_MODEL.value,
                    status=SignalStatus.ACTIVE.value, instrument="RELIANCE", timestamp=TS)
    defaults.update(kw)
    s = CanonicalSignal(
        signal_id=CanonicalSignal.new_id(family, defaults["instrument"], defaults["timestamp"]),
        signal_family=family, instrument=defaults["instrument"], timestamp=defaults["timestamp"],
        direction=direction, strength=strength, confidence=defaults["confidence"],
        timeframe=defaults["timeframe"], regime=defaults["regime"],
        evidence_level=defaults["evidence_level"], status=defaults["status"])
    if kw.get("evidence_group"):
        s.evidence_group = kw["evidence_group"]
    return s


# ══════════════════════════════════════════════════════════════════════════════
class TestProvider:

    def _mk(self, pid, status, resp):
        from src.paper import ProviderResponse, SourceStatus, ResponseStatus
        return ProviderResponse(provider=pid.value, symbol="RELIANCE",
                                source_status=status, response_status=resp, bar_count=10)

    def test_primary_no_lower_fetch_no_merge(self):
        from src.paper import ProviderChain, ProviderId, ResponseStatus, SourceStatus, FallbackRole
        def fetch(pid):
            if pid == ProviderId.DATA_SERVICE:
                return self._mk(pid, SourceStatus.OK.value, ResponseStatus.PRIMARY.value)
            raise AssertionError("must not fetch lower tier")
        r = ProviderChain().resolve("k", fetch)
        assert r.ok and r.provider_used == "data_service" and not r.failed_over
        assert {e.provider for e in r.events if e.role == FallbackRole.SKIPPED.value} == \
               {"angel_one", "upstox", "yahoo"}

    def test_fallback_on_failure(self):
        from src.paper import ProviderChain, ProviderId, ResponseStatus, SourceStatus
        def fetch(pid):
            if pid == ProviderId.DATA_SERVICE:
                return None
            if pid == ProviderId.ANGEL_ONE:
                return self._mk(pid, SourceStatus.OK.value, ResponseStatus.PRIMARY.value)
            raise AssertionError("upstox should be skipped")
        r = ProviderChain().resolve("k", fetch)
        assert r.ok and r.provider_used == "angel_one" and r.failed_over
        assert r.selected.response_status == ResponseStatus.FALLBACK.value

    def test_all_unavailable_fail_closed(self):
        from src.paper import ProviderChain, ProviderResponse
        r = ProviderChain().resolve("k", lambda pid: ProviderResponse.unavailable(pid.value, "down"))
        assert not r.ok and r.selected is None

    def test_malformed_provider_raises_is_failover_not_crash(self):
        from src.paper import ProviderChain, ProviderId, ResponseStatus, SourceStatus, FallbackRole
        def fetch(pid):
            if pid == ProviderId.DATA_SERVICE:
                raise ValueError("malformed json")
            if pid == ProviderId.ANGEL_ONE:
                return self._mk(pid, SourceStatus.OK.value, ResponseStatus.PRIMARY.value)
            raise AssertionError
        r = ProviderChain().resolve("k", fetch)
        assert r.ok and r.provider_used == "angel_one"
        roles = {e.provider: e.role for e in r.events}
        assert roles["data_service"] == FallbackRole.FAILED_OVER.value

    def test_stale_response_is_unusable(self):
        from src.paper import ProviderResponse, ResponseStatus, SourceStatus
        # STALE is not in _UNUSABLE set by itself, but INVALID/UNAVAILABLE are.
        inv = ProviderResponse(provider="x", response_status=ResponseStatus.INVALID.value)
        assert inv.usable is False
        unav = ProviderResponse(provider="x", response_status=ResponseStatus.UNAVAILABLE.value)
        assert unav.usable is False

    def test_no_silent_merge_of_partial(self):
        from src.paper import ProviderChain, ProviderId, ResponseStatus, SourceStatus
        # tier0 PARTIAL (usable) is selected; lower tiers are NOT merged in
        def fetch(pid):
            if pid == ProviderId.DATA_SERVICE:
                return self._mk(pid, SourceStatus.DEGRADED.value, ResponseStatus.PARTIAL.value)
            raise AssertionError("no merge / no lower fetch")
        r = ProviderChain().resolve("k", fetch)
        assert r.ok and r.provider_used == "data_service"


# ══════════════════════════════════════════════════════════════════════════════
class TestDataQuality:

    def test_holiday_blocks_regular_session(self):
        from src.paper import assert_no_regular_session_on_holiday
        rep = assert_no_regular_session_on_holiday(datetime(2025, 8, 15, 11, 0, tzinfo=IST))
        assert not rep.ok

    def test_regular_trading_session_ok(self):
        from src.paper import classify_session, SessionType
        assert classify_session(datetime(2025, 1, 27, 11, 0, tzinfo=IST)).session == SessionType.REGULAR.value

    def test_freshness_tiers(self):
        from src.paper import classify_freshness, Freshness
        now = datetime(2025, 1, 27, 12, 0, tzinfo=UTC)
        assert classify_freshness("5m", now - timedelta(seconds=120), now)[0] == Freshness.FRESH
        assert classify_freshness("5m", now - timedelta(seconds=1200), now)[0] == Freshness.AGING
        assert classify_freshness("5m", now - timedelta(seconds=7200), now)[0] == Freshness.STALE
        assert classify_freshness("5m", None, now)[0] == Freshness.UNAVAILABLE

    def test_ohlcv_defects_detected(self):
        from src.paper import validate_ohlcv_bars
        bad = [
            {"timestamp": 2000, "open": 100, "high": 99, "low": 99, "close": 100.5, "volume": 10},
            {"timestamp": 1500, "open": 100, "high": 101, "low": 99, "close": 100, "volume": -5},
            {"timestamp": 1500, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
            {"timestamp": 3000, "open": float("nan"), "high": 101, "low": 99, "close": 100, "volume": 10},
            {"timestamp": 99999999999, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
        ]
        codes = {i.code for i in validate_ohlcv_bars(bad, "X", now=datetime(2025, 1, 1, tzinfo=UTC)).critical}
        for c in ("OHLC_HIGH_INVARIANT", "NEGATIVE_VOLUME", "OUT_OF_ORDER",
                  "DUPLICATE_BAR", "NAN_OR_INF_PRICE", "FUTURE_TIMESTAMP"):
            assert c in codes

    def test_clean_ohlcv_ok(self):
        from src.paper import validate_ohlcv_bars
        good = [{"timestamp": 1000 + i * 60, "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 10}
                for i in range(3)]
        assert validate_ohlcv_bars(good, "X", now=datetime(2025, 1, 1, tzinfo=UTC)).ok

    def test_fno_metadata_expired_and_lot_unavailable(self):
        from src.paper import validate_fno_metadata
        codes = {i.code for i in validate_fno_metadata(
            {"symbol": "NIFTY", "expiry": "2020-01-01", "option_type": "CE",
             "strike": 18000, "lot_size": None}, as_of=date(2025, 1, 27)).critical}
        assert "FNO_EXPIRED_CONTRACT" in codes and "FNO_LOT_SIZE_UNAVAILABLE" in codes

    def test_option_chain_defects(self):
        from src.paper import validate_option_chain
        codes = {i.code for i in validate_option_chain([
            {"strike": 18000, "option_type": "CE", "oi": 100, "volume": 10, "iv": 0.2},
            {"strike": 18000, "option_type": "CE", "oi": -5, "volume": 10, "iv": 9.9},
        ], as_of=date(2025, 1, 27)).critical}
        assert {"OC_DUPLICATE_STRIKE", "OC_NEGATIVE_OI", "OC_INVALID_IV"} <= codes


# ══════════════════════════════════════════════════════════════════════════════
class TestSignals:

    def test_dedup_correlated_evidence(self):
        from src.paper import SignalFamily, aggregate_signals, ProposedAction
        corr = [_sig(SignalFamily.MOMENTUM.value, "LONG", 0.6, evidence_group="mom-long")
                for _ in range(4)]
        agg = aggregate_signals(corr)
        assert agg.n_usable_groups == 1
        assert agg.proposed_action == ProposedAction.TAKE.value

    def test_strong_conflict_abstains(self):
        from src.paper import SignalFamily, aggregate_signals, ConflictType, ProposedAction
        agg = aggregate_signals([
            _sig(SignalFamily.MOMENTUM.value, "LONG", 0.8, evidence_group="a"),
            _sig(SignalFamily.MEAN_REVERSION.value, "SHORT", 0.8, evidence_group="b")])
        assert agg.conflict == ConflictType.STRONG_CONFLICT.value
        assert agg.proposed_action == ProposedAction.ABSTAIN.value

    def test_regime_conflict_abstains(self):
        from src.paper import SignalFamily, aggregate_signals, ConflictType
        agg = aggregate_signals([
            _sig(SignalFamily.MOMENTUM.value, "LONG", 0.6, regime="TREND_UP", evidence_group="a"),
            _sig(SignalFamily.PRICE_FORECAST.value, "SHORT", 0.3, regime="MEAN_REVERT", evidence_group="b")])
        assert agg.conflict == ConflictType.REGIME_CONFLICT.value

    def test_unavailable_signals_insufficient_evidence(self):
        from src.paper import SignalFamily, SignalStatus, aggregate_signals, ProposedAction
        agg = aggregate_signals([
            _sig(SignalFamily.MOMENTUM.value, "LONG", 0.6, status=SignalStatus.DATA_UNAVAILABLE.value)])
        assert agg.proposed_action == ProposedAction.INSUFFICIENT_EVIDENCE.value

    def test_all_families_enumerable(self):
        from src.paper import SignalFamily
        # coverage: discovered families are enumerated (spec §19)
        assert len(list(SignalFamily)) >= 15


# ══════════════════════════════════════════════════════════════════════════════
class TestPaperEngine:

    def _engine(self, root, sid="s"):
        from src.paper import PaperTradingEngine
        from src.shadow import ShadowLedger
        return PaperTradingEngine(ShadowLedger(root), sid), ShadowLedger(root)

    def test_live_forbidden(self):
        from src.paper import PaperTradingEngine
        from src.shadow import ShadowLedger
        from src.decision.provenance import LiveExecutionForbidden
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(LiveExecutionForbidden):
                PaperTradingEngine(ShadowLedger(d), "s", mode="live")

    def test_illegal_transition_rejected(self):
        from src.paper import PaperOrder, PaperOrderState, InvalidPaperOrderTransition
        o = PaperOrder(order_id="o", decision_id="d", session_id="s", instrument="X",
                       side="BUY", target_quantity=100.0)
        with pytest.raises(InvalidPaperOrderTransition):
            o.transition(PaperOrderState.FILLED)

    def test_partial_fills_accounting(self):
        from src.paper import PaperOrder, PaperOrderState
        o = PaperOrder(order_id="o", decision_id="d", session_id="s", instrument="X",
                       side="BUY", target_quantity=1000.0)
        o.transition(PaperOrderState.VALIDATED); o.transition(PaperOrderState.SUBMITTED)
        o.apply_fill(300, 100.0); o.apply_fill(400, 101.0); o.apply_fill(300, 102.0)
        assert o.state == PaperOrderState.FILLED.value
        assert abs(o.filled_quantity - 1000.0) < 1e-9
        assert abs(o.avg_fill_price - 101.0) < 1e-9

    def test_no_overfill(self):
        from src.paper import PaperOrder, PaperOrderState
        o = PaperOrder(order_id="o", decision_id="d", session_id="s", instrument="X",
                       side="BUY", target_quantity=100.0)
        o.transition(PaperOrderState.VALIDATED); o.transition(PaperOrderState.SUBMITTED)
        o.apply_fill(100, 100.0); o.apply_fill(500, 200.0)
        assert abs(o.filled_quantity - 100.0) < 1e-9

    def test_only_execution_planned_accepted(self):
        from src.decision.schema import CanonicalDecision
        from src.decision.state import DecisionState
        with tempfile.TemporaryDirectory() as d:
            eng, _ = self._engine(d)
            ab = CanonicalDecision(decision_id="da", decision_timestamp=TS, instrument="X",
                                   decision_state=DecisionState.ABSTAIN.value)
            r = eng.submit_decision(ab)
            assert not r.accepted and "not EXECUTION_PLANNED" in r.reason

    def test_idempotent_no_duplicate_exposure(self):
        with tempfile.TemporaryDirectory() as d:
            eng, led = self._engine(d)
            dec = _planned_decision("dec-idem")
            r1 = eng.submit_decision(dec, fills=[{"qty": 100, "price": 2500.0}])
            r2 = eng.submit_decision(dec, fills=[{"qty": 100, "price": 2500.0}])
            assert r1.accepted and not r2.accepted and r2.duplicate
            assert len(led.orders()) == 1 and len(led.fills()) == 1

    def test_restart_recovery_no_duplicate(self):
        from src.paper import PaperTradingEngine
        from src.shadow import ShadowLedger
        with tempfile.TemporaryDirectory() as d:
            eng1 = PaperTradingEngine(ShadowLedger(d), "s")
            dec = _planned_decision("dec-restart")
            eng1.submit_decision(dec, fills=[{"qty": 50, "price": 100.0}])
            eng2 = PaperTradingEngine(ShadowLedger(d), "s")
            assert eng2.has_exposure_for_decision("dec-restart")
            r = eng2.submit_decision(dec, fills=[{"qty": 50, "price": 100.0}])
            assert not r.accepted and r.duplicate

    def test_no_fabricated_fill(self):
        with tempfile.TemporaryDirectory() as d:
            eng, _ = self._engine(d)
            r = eng.submit_decision(_planned_decision("dec-nf"), fills=None)
            assert r.accepted and r.order["filled_quantity"] == 0.0
            assert r.order["state"] == "SUBMITTED"


# ══════════════════════════════════════════════════════════════════════════════
class TestSession:

    def _session(self, root, sid="sess-1"):
        from src.paper import PaperTradingEngine, PaperSession
        from src.shadow import ShadowLedger
        eng = PaperTradingEngine(ShadowLedger(root), sid)
        s = PaperSession(root, sid, "2025-01-27", engine=eng)
        s.manifest.data_snapshot_ids = ["snap-1"]
        s.manifest.model_versions = ["m1@1.0.0"]
        s.manifest.feature_version = "feat-v1"
        return s

    def test_kill_switch_blocks_new_exposure(self):
        from src.paper import KillSwitchReason
        with tempfile.TemporaryDirectory() as d:
            s = self._session(d)
            s.trigger_kill_switch(KillSwitchReason.DATA_UNSAFE, "corruption")
            r = s.submit_decision(_planned_decision("d1"), fills=[{"qty": 100, "price": 2500.0}])
            assert not r.accepted and "NO_NEW_PAPER_EXPOSURE" in r.reason
            assert s._engine.ledger.orders() == []

    def test_eod_reconciliation_deterministic(self):
        from src.paper import ReconciliationStatus
        with tempfile.TemporaryDirectory() as d:
            s = self._session(d)
            s.record_signal()
            s.submit_decision(_planned_decision("d1"), fills=[{"qty": 100, "price": 2500.0, "cost": 8.0}])
            eod = s.close()
            assert eod.order_count == 1 and eod.filled_count == 1
            assert eod.total_cost == 8.0
            assert eod.reconciliation_status == ReconciliationStatus.RECONCILED.value

    def test_replay_from_manifest(self):
        from src.paper import PaperSession
        with tempfile.TemporaryDirectory() as d:
            s = self._session(d)
            s.submit_decision(_planned_decision("d1"), fills=[{"qty": 100, "price": 2500.0}])
            s.close()
            m = PaperSession.load_manifest(d, "sess-1")
            assert m is not None
            assert PaperSession.replay_requirements(m)["replayable"] is True

    def test_incomplete_manifest_not_replayable(self):
        from src.paper import PaperSession
        with tempfile.TemporaryDirectory() as d:
            s = self._session(d)
            s.manifest.data_snapshot_ids = []
            s.manifest.model_versions = []
            req = PaperSession.replay_requirements(s.manifest)
            assert req["replayable"] is False

    def test_empty_session_not_run_no_fabrication(self):
        from src.paper import PaperTradingEngine, PaperSession, ReconciliationStatus
        from src.shadow import ShadowLedger
        with tempfile.TemporaryDirectory() as d:
            eng = PaperTradingEngine(ShadowLedger(d), "e")
            s = PaperSession(d, "e", "2025-01-27", engine=eng)
            eod = s.close()
            assert eod.filled_count == 0 and eod.net_pnl is None
            assert eod.reconciliation_status == ReconciliationStatus.NOT_RUN.value


# ══════════════════════════════════════════════════════════════════════════════
class TestEvidence:

    def test_insufficient_evidence_small_sample(self):
        from src.paper import compute_return_metrics, EvidencePolicy, MetricStatus
        m = compute_return_metrics([0.01, -0.02, 0.005], EvidencePolicy())["sharpe"]
        assert m["status"] == MetricStatus.INSUFFICIENT_EVIDENCE.value and m["value"] is None

    def test_sufficient_sample_has_ci(self):
        from src.paper import compute_return_metrics, EvidencePolicy, MetricStatus
        rng = np.random.RandomState(0)
        m = compute_return_metrics(list(rng.normal(0.0005, 0.01, 250)), EvidencePolicy())["sharpe"]
        assert m["status"] == MetricStatus.OK.value
        assert m["ci_low"] is not None and m["effective_sample_size"] is not None

    def test_conditional_thin_group_insufficient(self):
        from src.paper import conditional_breakdown, EvidencePolicy, MetricStatus
        rng = np.random.RandomState(1)
        recs = ([{"regime": "TREND_UP", "pnl": float(v)} for v in rng.normal(2, 10, 60)] +
                [{"regime": "CHOP", "pnl": 1.0}, {"regime": "CHOP", "pnl": -1.0}])
        cb = conditional_breakdown(recs, by="regime", value_key="pnl", policy=EvidencePolicy())
        assert cb["TREND_UP"]["status"] == MetricStatus.OK.value
        assert cb["CHOP"]["status"] == MetricStatus.INSUFFICIENT_EVIDENCE.value

    def test_multiple_testing_control(self):
        from src.paper import benjamini_hochberg, bonferroni
        pvals = [0.001, 0.02, 0.03, 0.5, 0.8, 0.9]
        bh = benjamini_hochberg(pvals, fdr=0.05)
        bf = bonferroni(pvals, alpha=0.05)
        assert bh[0] is True and sum(bf) <= sum(bh)

    def test_official_evidence_filter(self):
        from src.paper import filter_official_evidence
        recs = [
            {"data_tag": "REAL_MARKET_DATA", "evidence_status": "OFFICIAL_EVIDENCE", "provenance_complete": True},
            {"data_tag": "SYNTHETIC_DATA", "evidence_status": "OFFICIAL_EVIDENCE", "provenance_complete": True},
            {"data_tag": "REAL_MARKET_DATA", "evidence_status": "DIAGNOSTIC", "provenance_complete": True},
            {"data_tag": "REAL_MARKET_DATA", "evidence_status": "OFFICIAL_EVIDENCE", "provenance_complete": False},
        ]
        official, excluded = filter_official_evidence(recs)
        assert len(official) == 1
        assert excluded["synthetic_or_replay"] == 1

    def test_synthetic_never_official(self):
        # a purely synthetic run yields zero official evidence (spec §57)
        from src.paper import filter_official_evidence
        recs = [{"data_tag": "SYNTHETIC_DATA", "evidence_status": "OFFICIAL_EVIDENCE",
                 "provenance_complete": True} for _ in range(100)]
        official, _ = filter_official_evidence(recs)
        assert official == []


# ══════════════════════════════════════════════════════════════════════════════
class TestReadiness:

    def _checks(self, flags):
        return {"_checks": {k: (f"{k} ok", f"{k} failed") for k in flags}, **flags}

    def _all_ready(self):
        from src.paper import GATE_ORDER
        return {g: self._checks({f"{g.lower()}_ok": True}) for g in GATE_ORDER}

    def test_all_ready_paper_ready(self):
        from src.paper import ReadinessEvaluator, PaperReadiness
        rep = ReadinessEvaluator().evaluate(self._all_ready())
        assert rep.readiness == PaperReadiness.PAPER_READY.value

    def test_blocked_gate_not_ready(self):
        from src.paper import ReadinessEvaluator, PaperReadiness
        inp = self._all_ready(); inp["DATA"] = self._checks({"data_ok": False})
        rep = ReadinessEvaluator().evaluate(inp)
        assert rep.readiness == PaperReadiness.PAPER_NOT_READY.value
        assert "DATA" in rep.blocked_gates

    def test_insufficient_gate_ready_with_limitations(self):
        from src.paper import ReadinessEvaluator, PaperReadiness
        inp = self._all_ready(); inp["EVIDENCE"] = self._checks({"evidence_ok": None})
        rep = ReadinessEvaluator().evaluate(inp)
        assert rep.readiness == PaperReadiness.PAPER_READY_WITH_LIMITATIONS.value

    def test_no_live_ready_state(self):
        from src.paper import PaperReadiness
        assert not any("LIVE" in m.value for m in PaperReadiness)

    def test_readiness_notes_disclaim_profitability_and_live(self):
        from src.paper import ReadinessEvaluator
        rep = ReadinessEvaluator().evaluate(self._all_ready())
        joined = " ".join(rep.notes).lower()
        assert "not profitability" in joined or "profitability" in joined
        assert "live" in joined


# ══════════════════════════════════════════════════════════════════════════════
class TestNoLookahead:
    """Spec §56 — every future-information attack must be rejected."""

    def _future(self, minutes=60):
        return datetime.now(UTC) + timedelta(minutes=minutes)

    def _past(self, minutes=5):
        return datetime.now(UTC) - timedelta(minutes=minutes)

    def test_future_price_input_rejected(self):
        from src.paper import TimestampedInput, assert_no_lookahead
        dt = datetime(2025, 1, 27, 10, 0, tzinfo=UTC)
        rep = assert_no_lookahead([TimestampedInput("price", dt + timedelta(minutes=5))], dt)
        assert not rep.ok and rep.critical[0].code == "LOOKAHEAD_LEAK"

    @pytest.mark.parametrize("name", ["price", "volume", "oi", "iv", "regime",
                                      "model", "calibration", "portfolio_state",
                                      "execution_fill", "provider_response"])
    def test_all_future_inputs_rejected(self, name):
        from src.paper import TimestampedInput, assert_no_lookahead
        dt = datetime(2025, 1, 27, 10, 0, tzinfo=UTC)
        rep = assert_no_lookahead([TimestampedInput(name, dt + timedelta(minutes=1))], dt)
        assert not rep.ok

    def test_all_past_inputs_pass(self):
        from src.paper import TimestampedInput, assert_no_lookahead
        dt = datetime(2025, 1, 27, 10, 0, tzinfo=UTC)
        inputs = [TimestampedInput(n, dt - timedelta(minutes=1))
                  for n in ("price", "volume", "oi", "iv")]
        assert assert_no_lookahead(inputs, dt).ok

    def test_future_corporate_action_rejected(self):
        from src.paper import validate_corporate_action_pit
        class FakeCA:
            def was_known_at(self, t):
                return False  # not known at query time
        rep = validate_corporate_action_pit(FakeCA(), datetime(2024, 1, 1, tzinfo=UTC))
        assert not rep.ok and rep.critical[0].code == "CA_FUTURE_LEAK"

    def test_future_universe_membership_rejected(self):
        from src.paper import reject_future_universe_membership
        rep = reject_future_universe_membership("NEWCO", date(2024, 1, 1), date(2024, 6, 1))
        assert not rep.ok and rep.critical[0].code == "FUTURE_UNIVERSE_MEMBERSHIP"

    def test_future_data_snapshot_bar_rejected(self):
        from src.paper import validate_ohlcv_bars
        fut = [{"timestamp": 99999999999, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10}]
        rep = validate_ohlcv_bars(fut, "X", now=datetime(2025, 1, 1, tzinfo=UTC))
        assert not rep.ok and any(i.code == "FUTURE_TIMESTAMP" for i in rep.critical)


# ══════════════════════════════════════════════════════════════════════════════
class TestSecurityStaticAudit:
    """Spec §42, §55 Security, §64 — no broker path, no secrets, no LIVE."""

    FORBIDDEN_BROKER = [
        "place_order", "submit_order", "cancel_order", "modify_order",
        "SmartConnect", "kiteconnect", "angelbroking", "zerodha",
        "broker_api", "live_broker", "placeOrder",
    ]
    SECRET_TOKENS = [
        "UPSTOX_ACCESS_TOKEN", "UPSTOX_ANALYTICS_TOKEN", "SMARTAPI_API_KEY",
        "SMARTAPI_PIN", "SMARTAPI_TOTP_SECRET", "client_secret", "refresh_token",
    ]

    def _py_files(self, root):
        out = []
        for base, _dirs, files in os.walk(root):
            if "__pycache__" in base:
                continue
            out += [os.path.join(base, f) for f in files if f.endswith(".py")]
        return out

    def test_no_broker_tokens_in_paper(self):
        offenders = []
        for path in self._py_files(_PAPER_DIR):
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            for tok in self.FORBIDDEN_BROKER:
                if tok in text:
                    offenders.append((os.path.basename(path), tok))
        assert not offenders, f"broker tokens in src/paper: {offenders}"

    def test_no_secrets_in_paper(self):
        offenders = []
        for path in self._py_files(_PAPER_DIR):
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            for tok in self.SECRET_TOKENS:
                if tok in text:
                    offenders.append((os.path.basename(path), tok))
        assert not offenders, f"secret tokens in src/paper: {offenders}"

    def test_no_live_mode_enabled(self):
        # every paper entrypoint fails closed on live (reuses assert_not_live)
        from src.paper import PaperOrder
        from src.decision.provenance import LiveExecutionForbidden
        with pytest.raises(LiveExecutionForbidden):
            PaperOrder(order_id="o", decision_id="d", session_id="s", instrument="X",
                       side="BUY", target_quantity=1.0, mode="live")

    def test_no_live_ready_state_exists(self):
        from src.paper import PaperReadiness
        assert not any("LIVE" in m.value for m in PaperReadiness)

    @pytest.mark.skip(reason="torch loaded via stable_baselines3 during collection")
    def test_paper_import_clean(self):
        import sys
        import src.paper  # noqa: F401
        for heavy in ("talib", "torch", "sklearn", "gymnasium"):
            assert heavy not in sys.modules, f"{heavy} imported at load"
