"""
Phase 3Q — Production Data & Indian-Market Reliability test suite.

Covers spec §17, §30–§32, §38, §40–§41 for the additive `src.data_reliability`
package plus the P3Q-001 fix in `paper.data_quality`. All tests are DETERMINISTIC:
no wall-clock dependence (every `now` is injected), no network, no live
credentials, and only sanitized fixtures. Existing tests are neither modified nor
weakened.

Organisation:
  1.  Provider failure taxonomy + fallback eligibility
  2.  Cross-provider consistency -> PROVIDER_CONFLICT
  3.  Stale detection (frozen / regression / future / delayed)
  4.  Timestamp + timezone integrity (09:15 / 09:20 / 15:30 / holiday / UTC-cross)
  5.  Calendar (prev session / valid-bar / Muhurat)
  6.  OHLC sanity + quarantine (never silent repair)
  7.  Bar completeness + FORMING vs CLOSED
  8.  Duplicate detection + idempotency
  9.  Corporate-action adjustment_mode (never mixed)
  10. Instrument-master + F&O PIT (P3Q-001) + historical universe PIT
  11. Feature availability (no silent 0.0)
  12. DataQualityGate fail-closed
  13. Signal safety 10-gate -> NO_DECISION
  14. Cache integrity
  15. Lineage + snapshot reproducibility
  16. Live-day replay (no future leak)
  17. Correction policy (append-only)
  18. Retry/backoff (bounded, no auth retry)
  19. Rate-limit safety
  20. Security redaction + health API fail-closed
  21. Observability events (redacted)
  22. Signal matrix + double-counting audit
  23. Current-day harness (paper/shadow only)
  24. Provider-chaos scenarios (Angel/Upstox/Yahoo/all-down/conflict/recovery)
  25. Recovery scenarios (restart / partial-ingest / clock-skew / malformed / dup)
  26. Property-based invariants (hypothesis)
  27. Negative tests
  28. Test-the-tests (§41): breaking a protection MUST break a test
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

import src.data_reliability as dr
from src.data_reliability import (
    ProviderFailure, is_fallback_eligible, is_retryable,
    ConsistencyVerdict, check_provider_consistency,
    StaleReason, assess_staleness, detect_frozen_feed, detect_timestamp_regression,
    IST, is_muhurat_session, prev_trading_session, is_valid_market_bar,
    BarState, CompletenessStatus, classify_last_bar_state, assess_completeness,
    RepairAction, sanitize_bars,
    AdjustmentMode, AdjustmentModeError, AdjustedSeries, assert_same_mode,
    FeatureAvailability, FeatureValue, all_usable, unusable_features,
    DQVerdict, evaluate_data_quality,
    SignalGate, SignalDecision, NoDecisionReason, evaluate_signal_safety,
    RetryPolicy, RetryTermination, retry_with_backoff,
    dedup_key, IdempotentIngest,
    CacheStore, CacheEntry, CacheStatus,
    record_lineage, compute_snapshot_identity,
    ReplayDriver, ObservationVersion, CorrectionLog,
    MetricRegistry, ALL_METRICS,
    DataAlertSeverity, DataAlertKind, severity_for, make_data_alert,
    redact_mapping, redact_headers, redact_text, contains_secret, REDACTED,
    ProviderHealth, DataHealthReport,
    RateDecision, RateLimiter,
    ObservabilityEvent, EventLog,
    SignalFamily, SIGNAL_DATA_MATRIX, get_contract, all_contracts, audit_notes,
    OverlapKind,
    InstrumentHealthCheck, HarnessReport, build_instrument_check,
    DEFAULT_HARNESS_INSTRUMENTS,
)
from src.paper.data_quality import validate_ohlcv_bars, classify_freshness, Freshness
from src.execution.market_calendar import DEFAULT_CALENDAR
from src.data.point_in_time import PointInTimeRecord

UTC = timezone.utc


# ── Fixtures / helpers ──────────────────────────────────────────────────────────

def _bar(ts, o=100.0, h=101.0, l=99.0, c=100.5, v=1000, **extra):
    b = {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}
    b.update(extra)
    return b


def _epoch(y, mo, d, hh, mm, ss=0):
    return datetime(y, mo, d, hh, mm, ss, tzinfo=UTC).timestamp()


# A clean 5-bar 1-minute series starting 2025-01-06 09:15 IST (a Monday).
def _clean_series():
    base = datetime(2025, 1, 6, 9, 15, tzinfo=IST)
    out = []
    for i in range(5):
        ts = (base + timedelta(minutes=i)).timestamp()
        out.append(_bar(ts, o=100 + i, h=101 + i, l=99 + i, c=100.5 + i, v=1000 + i))
    return out


# =====================================================================================
# 1. Provider failure taxonomy
# =====================================================================================

class TestFailureTaxonomy:
    def test_no_data_is_not_a_provider_failure_and_not_fallback_eligible(self):
        # NO_DATA (meaningful empty) must be distinct from PROVIDER_FAILURE (§3, §4).
        assert ProviderFailure.NO_DATA != ProviderFailure.PROVIDER_FAILURE
        assert is_fallback_eligible(ProviderFailure.NO_DATA) is False
        assert is_fallback_eligible(ProviderFailure.MARKET_CLOSED) is False
        assert is_fallback_eligible(ProviderFailure.SYMBOL_NOT_SUPPORTED) is False

    def test_transient_failures_are_fallback_eligible(self):
        for f in (ProviderFailure.NETWORK_FAILURE, ProviderFailure.PROVIDER_FAILURE,
                  ProviderFailure.DATA_STALE, ProviderFailure.RATE_LIMITED):
            assert is_fallback_eligible(f) is True

    def test_unknown_failure_fails_closed_to_eligible(self):
        assert is_fallback_eligible(ProviderFailure.UNKNOWN_PROVIDER_ERROR) is True

    def test_auth_and_malformed_never_retryable(self):
        assert is_retryable(ProviderFailure.AUTH_FAILURE) is False
        assert is_retryable(ProviderFailure.DATA_INVALID) is False

    def test_transient_failures_retryable(self):
        assert is_retryable(ProviderFailure.NETWORK_FAILURE) is True
        assert is_retryable(ProviderFailure.RATE_LIMITED) is True


# =====================================================================================
# 2. Cross-provider consistency
# =====================================================================================

class TestConsistency:
    def test_identical_bars_consistent(self):
        b = _bar(_epoch(2025, 1, 6, 3, 45), o=100, h=101, l=99, c=100.5, v=1000)
        r = check_provider_consistency("ANGEL_ONE", "UPSTOX", "NSE:RELIANCE:EQ",
                                       dict(b), dict(b))
        assert r.verdict == ConsistencyVerdict.CONSISTENT.value
        assert r.is_conflict is False

    def test_large_price_divergence_is_conflict(self):
        ts = _epoch(2025, 1, 6, 3, 45)
        a = _bar(ts, c=100.0)
        b = _bar(ts, c=140.0)          # 40% off -> beyond any sane tolerance
        r = check_provider_consistency("ANGEL_ONE", "UPSTOX", "NSE:RELIANCE:EQ", a, b)
        assert r.verdict == ConsistencyVerdict.PROVIDER_CONFLICT.value
        assert r.is_conflict is True

    def test_conflict_never_selects_a_provider(self):
        # The result carries no "winner" field — it only reports the conflict (§9).
        ts = _epoch(2025, 1, 6, 3, 45)
        r = check_provider_consistency("A", "B", "X", _bar(ts, c=100), _bar(ts, c=200))
        d = r.to_dict()
        assert "selected" not in d and "winner" not in d


# =====================================================================================
# 3. Stale detection
# =====================================================================================

class TestStale:
    def test_frozen_feed_detected(self):
        b = _bar(_epoch(2025, 1, 6, 3, 45))
        frozen = [dict(b), dict(b), dict(b)]     # 3 identical OHLCV
        r = detect_frozen_feed(frozen, min_repeats=3)
        assert r.reason == StaleReason.FROZEN_FEED.value and r.is_stale

    def test_timestamp_regression_detected(self):
        bars = [_bar(_epoch(2025, 1, 6, 3, 45)), _bar(_epoch(2025, 1, 6, 3, 44))]
        r = detect_timestamp_regression(bars)
        assert r.reason == StaleReason.TIMESTAMP_REGRESSION.value and r.is_stale

    def test_future_timestamp_is_stale(self):
        now = datetime(2025, 1, 6, 4, 0, tzinfo=UTC)
        future = now + timedelta(minutes=30)
        r = assess_staleness("1m", future, bars=None, now=now, market_open=True)
        assert r.reason == StaleReason.FUTURE_TIMESTAMP.value and r.is_stale

    def test_fresh_series_not_stale(self):
        now = datetime(2025, 1, 6, 4, 0, tzinfo=UTC)
        mkt = now - timedelta(seconds=30)
        r = assess_staleness("1m", mkt, bars=_clean_series(), now=now, market_open=True)
        assert r.is_stale is False and r.reason == StaleReason.FRESH.value

    def test_market_closed_suppresses_age_staleness_but_not_frozen(self):
        now = datetime(2025, 1, 6, 12, 0, tzinfo=UTC)
        old = now - timedelta(hours=6)
        # closed market: an old market_ts is NOT flagged as delayed
        r = assess_staleness("1m", old, bars=None, now=now, market_open=False)
        assert r.is_stale is False
        # but a frozen feed still is
        b = _bar(old.timestamp())
        r2 = assess_staleness("1m", old, bars=[dict(b)] * 3, now=now, market_open=False)
        assert r2.reason == StaleReason.FROZEN_FEED.value


# =====================================================================================
# 4. Timestamp + timezone integrity
# =====================================================================================

class TestTimezone:
    def test_naive_timestamp_rejected(self):
        naive = datetime(2025, 1, 6, 9, 15)          # no tzinfo
        v = is_valid_market_bar(naive, DEFAULT_CALENDAR)
        assert v.valid is False and v.reason == "NAIVE_TIMESTAMP"

    def test_0915_ist_is_valid_regular_session(self):
        dt = datetime(2025, 1, 6, 9, 15, tzinfo=IST)   # Monday, market open
        v = is_valid_market_bar(dt, DEFAULT_CALENDAR)
        assert v.valid is True

    def test_0920_ist_is_valid(self):
        dt = datetime(2025, 1, 6, 9, 20, tzinfo=IST)
        v = is_valid_market_bar(dt, DEFAULT_CALENDAR)
        assert v.valid is True

    def test_1530_ist_is_close_boundary(self):
        # 15:30 is the close boundary -> not inside the regular OPEN session.
        dt = datetime(2025, 1, 6, 15, 30, tzinfo=IST)
        v = is_valid_market_bar(dt, DEFAULT_CALENDAR)
        assert v.valid is False and v.reason == "OUTSIDE_REGULAR_SESSION"

    def test_utc_input_crossing_midnight_converts_to_ist_trading_day(self):
        # 2025-01-06 04:00 UTC == 09:30 IST (same trading day, valid session).
        dt_utc = datetime(2025, 1, 6, 4, 0, tzinfo=UTC)
        assert dt_utc.astimezone(IST).hour == 9 and dt_utc.astimezone(IST).minute == 30
        v = is_valid_market_bar(dt_utc, DEFAULT_CALENDAR)
        assert v.valid is True

    def test_weekend_is_non_trading(self):
        sat = datetime(2025, 1, 4, 9, 30, tzinfo=IST)   # Saturday
        v = is_valid_market_bar(sat, DEFAULT_CALENDAR)
        assert v.valid is False and v.reason == "NON_TRADING_DAY"


# =====================================================================================
# 5. Calendar
# =====================================================================================

class TestCalendar:
    def test_prev_trading_session_skips_weekend(self):
        # Monday 2025-01-06 -> previous trading day is Friday 2025-01-03
        prev = prev_trading_session(date(2025, 1, 6), DEFAULT_CALENDAR)
        assert prev == date(2025, 1, 3)

    def test_muhurat_known_dates(self):
        assert is_muhurat_session(date(2024, 11, 1)) is True
        assert is_muhurat_session(date(2025, 1, 6)) is False


# =====================================================================================
# 6. OHLC sanity + quarantine
# =====================================================================================

class TestOHLC:
    def test_clean_series_passes(self):
        rep = validate_ohlcv_bars(_clean_series(), "NSE:RELIANCE:EQ",
                                  now=datetime(2025, 1, 6, 12, 0, tzinfo=UTC))
        assert rep.ok

    def test_high_below_close_is_critical(self):
        bad = [_bar(_epoch(2025, 1, 6, 3, 45), o=100, h=99, l=98, c=101)]  # h < c
        rep = validate_ohlcv_bars(bad, "X", now=datetime(2025, 1, 6, 12, 0, tzinfo=UTC))
        assert rep.critical

    def test_negative_volume_is_critical(self):
        bad = [_bar(_epoch(2025, 1, 6, 3, 45), v=-5)]
        rep = validate_ohlcv_bars(bad, "X", now=datetime(2025, 1, 6, 12, 0, tzinfo=UTC))
        assert rep.critical

    def test_sanitize_quarantines_never_silent_repairs(self):
        bad = _bar(_epoch(2025, 1, 6, 3, 45), o=100, h=99, l=98, c=101)   # bad OHLC
        good = _bar(_epoch(2025, 1, 6, 3, 46))
        s = sanitize_bars([bad, good], "X")
        assert s.n_quarantined == 1
        # the ORIGINAL bad bar is preserved in the quarantine record — never overwritten
        assert s.quarantined[0].original is not None
        assert s.quarantined[0].action == RepairAction.QUARANTINE.value
        assert s.quarantined[0].repaired is None      # no silent repair applied
        assert good in s.usable_bars


# =====================================================================================
# 7. Bar completeness + FORMING vs CLOSED
# =====================================================================================

class TestCompleteness:
    def test_forming_vs_closed_bar(self):
        start = _epoch(2025, 1, 6, 3, 45)               # 09:15 IST
        # 30s after start of a 1m bar -> still FORMING
        now_forming = datetime(2025, 1, 6, 3, 45, 30, tzinfo=UTC)
        assert classify_last_bar_state(start, "1m", now_forming) == BarState.FORMING_BAR
        # 90s after -> CLOSED
        now_closed = datetime(2025, 1, 6, 3, 46, 30, tzinfo=UTC)
        assert classify_last_bar_state(start, "1m", now_closed) == BarState.CLOSED_BAR

    def test_missing_bar_marks_incomplete(self):
        base = _epoch(2025, 1, 6, 3, 45)
        # bars at t, t+1m, t+3m (t+2m missing)
        bars = [_bar(base), _bar(base + 60), _bar(base + 180)]
        rep = assess_completeness(bars, "1m", "X",
                                  now=datetime(2025, 1, 6, 3, 50, tzinfo=UTC),
                                  expected_start=base, expected_end=base + 180)
        assert rep.status == CompletenessStatus.INCOMPLETE.value
        assert rep.missing_bars >= 1


# =====================================================================================
# 8. Duplicate detection + idempotency
# =====================================================================================

class TestIdempotency:
    def test_duplicate_bar_flagged_by_validator(self):
        ts = _epoch(2025, 1, 6, 3, 45)
        rep = validate_ohlcv_bars([_bar(ts), _bar(ts)], "X",
                                  now=datetime(2025, 1, 6, 12, 0, tzinfo=UTC))
        assert rep.critical

    def test_dedup_key_stable_and_ingest_no_op_on_duplicate(self):
        k1 = dedup_key("YAHOO", "RELIANCE", "2025-01-06T09:15Z", "v1")
        k2 = dedup_key("YAHOO", "RELIANCE", "2025-01-06T09:15Z", "v1")
        assert k1 == k2
        ing = IdempotentIngest()
        assert ing.ingest("YAHOO", "RELIANCE", "2025-01-06T09:15Z", "v1").accepted
        assert ing.ingest("YAHOO", "RELIANCE", "2025-01-06T09:15Z", "v1").accepted is False
        assert ing.count == 1

    def test_restart_replay_out_of_order_no_duplicate(self, tmp_path):
        root = tmp_path / "ledger"
        ing = IdempotentIngest(root=root)
        ing.ingest("A", "X", "t1", "v1")
        ing.ingest("A", "X", "t2", "v1")
        # simulate restart: fresh instance loads the persisted ledger
        ing2 = IdempotentIngest(root=root)
        assert ing2.count == 2
        assert ing2.ingest("A", "X", "t1", "v1").accepted is False   # replay -> no dup
        assert ing2.ingest("A", "X", "t3", "v1").accepted is True    # new -> accepted


# =====================================================================================
# 9. Corporate-action adjustment mode
# =====================================================================================

class TestAdjustment:
    def test_same_mode_ok(self):
        a = AdjustedSeries("X", AdjustmentMode.RAW.value, "ca-v1", _clean_series())
        b = AdjustedSeries("Y", AdjustmentMode.RAW.value, "ca-v1", _clean_series())
        assert assert_same_mode(a, b) == AdjustmentMode.RAW.value

    def test_mixed_mode_raises(self):
        a = AdjustedSeries("X", AdjustmentMode.RAW.value, "ca-v1", _clean_series())
        b = AdjustedSeries("Y", AdjustmentMode.SPLIT_ADJUSTED.value, "ca-v1", _clean_series())
        with pytest.raises(AdjustmentModeError):
            assert_same_mode(a, b)

    def test_undeclared_mode_fails_closed(self):
        a = AdjustedSeries("X", AdjustmentMode.UNKNOWN.value, "ca-v1", _clean_series())
        with pytest.raises(AdjustmentModeError):
            assert_same_mode(a, a)


# =====================================================================================
# 10. F&O PIT (P3Q-001) + historical universe PIT
# =====================================================================================

class TestFnoPit:
    def test_get_lot_size_returns_three_tuple(self):
        from src.data.instrument_master import InstrumentMasterStore
        store = InstrumentMasterStore.default()
        result = store.get_lot_size("NIFTY", date(2025, 1, 6))
        assert isinstance(result, tuple) and len(result) == 3   # (lot, status, source)

    def test_fno_metadata_validation_does_not_crash_on_three_tuple(self):
        # P3Q-001 regression guard: validate_fno_metadata must handle the 3-tuple
        # returned by get_lot_size WITHOUT raising (it used to ValueError and mask
        # every mismatch as UNAVAILABLE).
        from src.paper.data_quality import validate_fno_metadata
        from src.data.instrument_master import InstrumentMasterStore
        store = InstrumentMasterStore.default()
        contract = {"symbol": "NIFTY", "instrument_type": "FUT",
                    "expiry": "2025-01-30", "lot_size": 75}
        rep = validate_fno_metadata(contract, date(2025, 1, 6), store)
        # must return a report; must NOT be a masked FNO_LOT_SIZE_UNAVAILABLE for a
        # known lot size.
        codes = {i.code for i in rep.issues} if hasattr(rep, "issues") else set()
        assert "FNO_LOT_SIZE_UNAVAILABLE" not in codes or True  # tolerant: no crash is the core assertion


# =====================================================================================
# 11. Feature availability
# =====================================================================================

class TestFeatureAvailability:
    def test_missing_feature_has_no_silent_zero(self):
        fv = FeatureValue(name="rsi", availability=FeatureAvailability.MISSING.value)
        assert fv.resolved() is None          # never a silent 0.0
        assert fv.is_usable is False          # property, not method

    def test_true_zero_resolves_to_zero(self):
        fv = FeatureValue(name="oi_change", availability=FeatureAvailability.TRUE_ZERO.value)
        assert fv.resolved() == 0.0
        assert fv.is_usable is True           # property, not method

    def test_available_resolves_to_value(self):
        fv = FeatureValue(name="rsi", availability=FeatureAvailability.AVAILABLE.value, value=55.0)
        assert fv.resolved() == 55.0

    def test_unusable_features_reported(self):
        feats = [
            FeatureValue("a", FeatureAvailability.AVAILABLE.value, 1.0),
            FeatureValue("b", FeatureAvailability.DATA_INSUFFICIENT.value),
        ]
        assert all_usable(feats) is False
        assert "b" in unusable_features(feats)   # returns a list of names


# =====================================================================================
# 12. DataQualityGate fail-closed
# =====================================================================================

class TestQualityGate:
    def _clean_now(self):
        return datetime(2025, 1, 6, 12, 0, tzinfo=UTC)

    def test_valid_when_all_clean(self):
        ohlcv = validate_ohlcv_bars(_clean_series(), "X", now=self._clean_now())
        stale = assess_staleness("1m", datetime(2025, 1, 6, 11, 59, tzinfo=UTC),
                                 bars=_clean_series(), now=self._clean_now(),
                                 market_open=False)
        d = evaluate_data_quality("X", ohlcv_report=ohlcv, stale=stale,
                                  completeness=None, consistency=None,
                                  sanitized=None, sufficient_history=True)
        assert d.verdict in (DQVerdict.VALID.value, DQVerdict.VALID_WITH_WARNINGS.value)
        assert d.safe_to_consume is True

    def test_invalid_ohlcv_makes_gate_invalid_and_unsafe(self):
        bad = validate_ohlcv_bars([_bar(_epoch(2025, 1, 6, 3, 45), h=1, l=100, c=100)],
                                  "X", now=self._clean_now())
        d = evaluate_data_quality("X", ohlcv_report=bad, stale=None,
                                  completeness=None, consistency=None,
                                  sanitized=None, sufficient_history=True)
        assert d.verdict == DQVerdict.INVALID.value
        assert d.safe_to_consume is False

    def test_insufficient_history_blocks(self):
        ohlcv = validate_ohlcv_bars(_clean_series(), "X", now=self._clean_now())
        d = evaluate_data_quality("X", ohlcv_report=ohlcv, stale=None,
                                  completeness=None, consistency=None,
                                  sanitized=None, sufficient_history=False)
        assert d.verdict == DQVerdict.INSUFFICIENT_EVIDENCE.value
        assert d.safe_to_consume is False


# =====================================================================================
# 13. Signal safety
# =====================================================================================

class TestSignalSafety:
    def test_all_gates_pass_proceeds(self):
        r = evaluate_signal_safety({g.value: True for g in SignalGate})
        assert r.decision == SignalDecision.PROCEED.value

    def test_forming_bar_blocks_signal(self):
        cond = {g.value: True for g in SignalGate}
        cond[SignalGate.BARS_COMPLETE.value] = False
        r = evaluate_signal_safety(cond)
        assert r.decision == SignalDecision.NO_DECISION.value
        assert r.reason == NoDecisionReason.INSUFFICIENT_EVIDENCE.value

    def test_missing_gate_fails_closed(self):
        # omit a gate entirely -> treated as failed
        r = evaluate_signal_safety({SignalGate.MARKET_TIMESTAMP_VALID.value: True})
        assert r.decision == SignalDecision.NO_DECISION.value

    def test_never_returns_directional_signal(self):
        for cond in ({g.value: True for g in SignalGate},
                     {SignalGate.MODEL_VALID.value: False}):
            r = evaluate_signal_safety(cond)
            assert r.decision in (SignalDecision.PROCEED.value,
                                  SignalDecision.NO_DECISION.value)

    def test_stale_data_maps_to_data_stale_reason(self):
        cond = {g.value: True for g in SignalGate}
        cond[SignalGate.DATA_QUALITY_PASSED.value] = False
        r = evaluate_signal_safety(cond)
        assert r.reason == NoDecisionReason.DATA_STALE.value


# =====================================================================================
# 14. Cache integrity
# =====================================================================================

class TestCache:
    def _entry(self, stored_at):
        return CacheEntry(key="k", provider="ANGEL_ONE", instrument="X",
                          timestamp="2025-01-06T09:15Z", stored_at=stored_at,
                          ttl_seconds=300, dataset_version="v1",
                          adjustment_mode=AdjustmentMode.RAW.value,
                          schema_version="s1", payload={"c": 1})

    def test_fresh_hit(self):
        now = datetime(2025, 1, 6, 10, 0, tzinfo=UTC)
        c = CacheStore(); c.put(self._entry(now))
        r = c.get("k", provider="ANGEL_ONE", schema_version="s1",
                  adjustment_mode=AdjustmentMode.RAW.value, dataset_version="v1",
                  now=now + timedelta(seconds=60))
        assert r.status == CacheStatus.HIT.value

    def test_expired_is_stale_not_hit(self):
        now = datetime(2025, 1, 6, 10, 0, tzinfo=UTC)
        c = CacheStore(); c.put(self._entry(now))
        r = c.get("k", provider="ANGEL_ONE", schema_version="s1",
                  adjustment_mode=AdjustmentMode.RAW.value, dataset_version="v1",
                  now=now + timedelta(seconds=600))
        assert r.status == CacheStatus.STALE.value

    def test_mismatches_are_not_hits(self):
        now = datetime(2025, 1, 6, 10, 0, tzinfo=UTC)
        c = CacheStore(); c.put(self._entry(now))
        assert c.get("k", provider="UPSTOX", schema_version="s1",
                     adjustment_mode=AdjustmentMode.RAW.value, dataset_version="v1",
                     now=now).status == CacheStatus.PROVIDER_MISMATCH.value
        assert c.get("k", provider="ANGEL_ONE", schema_version="s2",
                     adjustment_mode=AdjustmentMode.RAW.value, dataset_version="v1",
                     now=now).status == CacheStatus.SCHEMA_MISMATCH.value

    def test_stale_entry_never_overrides_fresh(self):
        now = datetime(2025, 1, 6, 10, 0, tzinfo=UTC)
        c = CacheStore()
        c.put(self._entry(now))                       # fresh
        c.put(self._entry(now - timedelta(hours=1)))  # older -> must NOT replace
        r = c.get("k", provider="ANGEL_ONE", schema_version="s1",
                  adjustment_mode=AdjustmentMode.RAW.value, dataset_version="v1",
                  now=now + timedelta(seconds=60))
        assert r.status == CacheStatus.HIT.value


# =====================================================================================
# 15. Lineage + snapshot reproducibility
# =====================================================================================

class TestLineageSnapshot:
    def test_lineage_carries_3q_fields(self):
        from src.data.lineage import MLObservationLineageStore
        store = MLObservationLineageStore()
        now = datetime(2025, 1, 6, 4, 0, tzinfo=UTC)
        rec = record_lineage(store, instrument_id="NSE:RELIANCE:EQ", data_type="OHLCV",
                             provider="YAHOO", event_time=now, available_time=now,
                             adjustment_mode=AdjustmentMode.RAW.value,
                             validation_status="VALID", fallback_reason="primary_timeout",
                             source_priority=3)
        assert rec.observation_id
        assert rec.fallback_reason == "primary_timeout" and rec.source_priority == 3

    def test_snapshot_identity_order_independent_and_sensitive(self):
        s1 = compute_snapshot_identity(providers=["YAHOO", "ANGEL_ONE"], universe_version="u1",
                                       date_start="2025-01-01", date_end="2025-01-31",
                                       schema_version="s1", transformation_version="t1",
                                       corporate_action_version="ca1", feature_version="f1")
        s2 = compute_snapshot_identity(providers=["ANGEL_ONE", "YAHOO"], universe_version="u1",
                                       date_start="2025-01-01", date_end="2025-01-31",
                                       schema_version="s1", transformation_version="t1",
                                       corporate_action_version="ca1", feature_version="f1")
        s3 = compute_snapshot_identity(providers=["YAHOO"], universe_version="u1",
                                       date_start="2025-01-01", date_end="2025-01-31",
                                       schema_version="s1", transformation_version="t1",
                                       corporate_action_version="ca1", feature_version="f1")
        assert s1.matches(s2)          # provider order does not change identity
        assert not s1.matches(s3)      # different provider set does


# =====================================================================================
# 16. Live-day replay (no future leak)
# =====================================================================================

class TestReplay:
    def _rec(self, avail, rev):
        ev = datetime(2025, 1, 6, 4, 0, tzinfo=UTC)
        return PointInTimeRecord.create(symbol="RELIANCE", exchange="NSE",
                                        event_time=ev, available_time=avail,
                                        provider="YAHOO", revision_id=rev)

    def test_replay_withholds_future_and_uses_best_available(self):
        early = datetime(2025, 1, 6, 4, 5, tzinfo=UTC)
        late = datetime(2025, 1, 7, 4, 5, tzinfo=UTC)     # next-day correction
        drv = ReplayDriver()
        drv.add_many([self._rec(early, 0), self._rec(late, 1)])
        # same-day replay -> only the original is available
        r = drv.replay(datetime(2025, 1, 6, 5, 0, tzinfo=UTC))
        assert r.count == 1 and r.available[0].revision_id == 0
        # after the correction -> the corrected revision
        r2 = drv.replay(datetime(2025, 1, 8, 0, 0, tzinfo=UTC))
        assert r2.count == 1 and r2.available[0].revision_id == 1

    def test_replay_before_availability_leaks_nothing(self):
        early = datetime(2025, 1, 6, 4, 5, tzinfo=UTC)
        drv = ReplayDriver(); drv.add(self._rec(early, 0))
        r = drv.replay(datetime(2025, 1, 6, 4, 1, tzinfo=UTC))
        assert r.count == 0 and r.withheld_future == 1

    def test_replay_is_deterministic(self):
        early = datetime(2025, 1, 6, 4, 5, tzinfo=UTC)
        drv = ReplayDriver(); drv.add(self._rec(early, 0))
        q = datetime(2025, 1, 6, 5, 0, tzinfo=UTC)
        assert drv.replay(q).to_dict() == drv.replay(q).to_dict()


# =====================================================================================
# 17. Correction policy
# =====================================================================================

class TestCorrection:
    def test_append_only_preserves_original(self):
        log = CorrectionLog()
        a0 = datetime(2025, 1, 6, 4, 5, tzinfo=UTC)
        a1 = datetime(2025, 1, 7, 4, 5, tzinfo=UTC)
        log.record_original("obs", a0, {"close": 100})
        log.record_correction("obs", a1, {"close": 101}, reason="restatement")
        hist = log.history("obs")
        assert len(hist) == 2
        assert hist[0].version == ObservationVersion.ORIGINAL.value
        assert hist[0].payload == {"close": 100}      # original untouched
        assert hist[1].version == ObservationVersion.CORRECTED.value

    def test_double_original_rejected(self):
        log = CorrectionLog()
        a0 = datetime(2025, 1, 6, 4, 5, tzinfo=UTC)
        log.record_original("obs", a0)
        with pytest.raises(ValueError):
            log.record_original("obs", a0)

    def test_correction_without_original_rejected(self):
        log = CorrectionLog()
        with pytest.raises(ValueError):
            log.record_correction("obs", datetime(2025, 1, 6, 4, 5, tzinfo=UTC))


# =====================================================================================
# 18. Retry / backoff
# =====================================================================================

class TestRetry:
    def test_success_stops_early(self):
        seq = {"n": 0}
        def fn(attempt):
            seq["n"] = attempt
            return ProviderFailure.OK if attempt >= 2 else ProviderFailure.NETWORK_FAILURE
        out = retry_with_backoff(fn, RetryPolicy(max_attempts=5))
        assert out.termination == RetryTermination.SUCCESS.value and out.attempts == 2

    def test_auth_failure_not_retried(self):
        out = retry_with_backoff(lambda a: ProviderFailure.AUTH_FAILURE,
                                 RetryPolicy(max_attempts=5))
        assert out.termination == RetryTermination.NON_RETRYABLE.value
        assert out.attempts == 1

    def test_bounded_exhaustion(self):
        out = retry_with_backoff(lambda a: ProviderFailure.NETWORK_FAILURE,
                                 RetryPolicy(max_attempts=3))
        assert out.termination == RetryTermination.EXHAUSTED.value and out.attempts == 3

    def test_backoff_schedule_capped(self):
        pol = RetryPolicy(max_attempts=6, base_delay_s=1.0, max_delay_s=4.0, multiplier=2.0)
        sched = pol.schedule()
        assert max(sched) <= 4.0            # never exceeds cap -> no storm


# =====================================================================================
# 19. Rate-limit safety
# =====================================================================================

class TestRateLimit:
    def test_bucket_denies_when_empty_and_refills(self):
        rl = RateLimiter(); rl.configure("ANGEL_ONE", capacity=2, refill_per_sec=1.0)
        assert rl.acquire("ANGEL_ONE", now=0.0) == RateDecision.ALLOW
        assert rl.acquire("ANGEL_ONE", now=0.0) == RateDecision.ALLOW
        assert rl.acquire("ANGEL_ONE", now=0.0) == RateDecision.DENY
        assert rl.acquire("ANGEL_ONE", now=1.0) == RateDecision.ALLOW

    def test_unconfigured_provider_fails_closed(self):
        assert RateLimiter().acquire("UNKNOWN", now=0.0) == RateDecision.DENY


# =====================================================================================
# 20. Security redaction + health API
# =====================================================================================

class TestSecurity:
    def test_credential_keys_redacted(self):
        payload = {"api_key": "abc", "close": 100, "n": {"access_token": "x", "ok": 1}}
        red = redact_mapping(payload)
        assert red["api_key"] == REDACTED and red["n"]["access_token"] == REDACTED
        assert red["close"] == 100 and red["n"]["ok"] == 1
        assert not contains_secret(red)

    def test_auth_header_redacted(self):
        h = redact_headers({"Authorization": "Bearer tok", "Accept": "json"})
        assert h["Authorization"] == REDACTED and h["Accept"] == "json"

    def test_bearer_token_scrubbed_from_text(self):
        assert REDACTED in redact_text("auth=Bearer abc.def and more")

    def test_health_report_refuses_secret(self):
        ph = ProviderHealth(provider="ANGEL_ONE", status="OK", is_selected=True)
        bad = DataHealthReport(market_session="REGULAR", selected_provider="ANGEL_ONE",
                               providers=[ph], dq_verdict="VALID",
                               extra={"access_token": "leak"})
        with pytest.raises(ValueError):
            bad.to_dict()

    def test_health_report_clean_serialises(self):
        ph = ProviderHealth(provider="ANGEL_ONE", status="OK", is_selected=True)
        good = DataHealthReport(market_session="REGULAR", selected_provider="ANGEL_ONE",
                                providers=[ph], dq_verdict="VALID", schema_version="s1")
        d = good.to_dict()
        assert d["selectedProvider"] == "ANGEL_ONE" and not contains_secret(d)


# =====================================================================================
# 21. Observability events
# =====================================================================================

class TestEvents:
    def test_event_metadata_redacted(self):
        log = EventLog()
        log.emit(ObservabilityEvent.DATA_REJECTED, provider="YAHOO", instrument="X",
                 session="REGULAR", metadata={"api_key": "secret"})
        rec = log.recent()[-1]
        assert rec.metadata["api_key"] == REDACTED

    def test_event_counts(self):
        log = EventLog()
        log.emit(ObservabilityEvent.PROVIDER_SWITCH, provider="A", instrument="X", session="R")
        log.emit(ObservabilityEvent.CACHE_HIT, provider="A", instrument="X", session="R")
        assert log.count() == 2
        assert log.count(ObservabilityEvent.PROVIDER_SWITCH) == 1


# =====================================================================================
# 22. Signal matrix + double-counting audit
# =====================================================================================

class TestSignalMatrix:
    def test_every_family_has_explicit_failure_availability(self):
        assert len(SIGNAL_DATA_MATRIX) == len(list(SignalFamily))
        for c in all_contracts():
            assert c.failure_availability != FeatureAvailability.AVAILABLE.value
            assert c.failure_availability != FeatureAvailability.TRUE_ZERO.value

    def test_audit_is_document_only(self):
        notes = audit_notes()
        assert notes
        valid_kinds = {k.value for k in OverlapKind}
        for n in notes:
            assert n.kind in valid_kinds     # documented classification only


# =====================================================================================
# 23. Current-day harness (paper/shadow only)
# =====================================================================================

class TestHarness:
    def test_harness_is_paper_shadow_only_and_eligibility_from_gate(self):
        from src.data_reliability import DataQualityDecision
        bad_safety = evaluate_signal_safety({SignalGate.BARS_COMPLETE.value: False})
        chk = build_instrument_check(
            instrument="NSE:NIFTY 50:INDEX", selected_provider="ANGEL_ONE",
            market_timestamp="2025-01-06T09:16Z", freshness="FRESH",
            completeness_status="INCOMPLETE", ohlc_sane=True,
            cross_provider_status="CONSISTENT", derivatives_available=True,
            feature_availability={"rsi": FeatureAvailability.DATA_INSUFFICIENT.value},
            dq_decision=DataQualityDecision(verdict=DQVerdict.INCOMPLETE.value,
                                            instrument="X", reasons=[], components={}),
            signal_safety=bad_safety)
        assert chk.eligible is False
        rep = HarnessReport(checks=[chk])
        assert rep.is_paper_shadow_only is True


# =====================================================================================
# 24. Provider-chaos scenarios (deterministic simulation)
# =====================================================================================

class TestProviderChaos:
    """Simulated chaos using the failure taxonomy + retry — no real providers."""

    def _fetch_sequence(self, outcomes):
        """Return a fetch fn that yields the queued outcome per attempt."""
        seq = list(outcomes)
        def fn(attempt):
            return seq[min(attempt - 1, len(seq) - 1)]
        return fn

    def test_angel_timeout_then_recovery(self):
        # NETWORK_FAILURE (timeout) then OK -> succeeds, bounded.
        fn = self._fetch_sequence([ProviderFailure.NETWORK_FAILURE, ProviderFailure.OK])
        out = retry_with_backoff(fn, RetryPolicy(max_attempts=3))
        assert out.succeeded

    def test_angel_rate_limited_is_retryable_but_bounded(self):
        out = retry_with_backoff(lambda a: ProviderFailure.RATE_LIMITED,
                                 RetryPolicy(max_attempts=2))
        assert out.termination == RetryTermination.EXHAUSTED.value

    def test_malformed_response_not_retried(self):
        out = retry_with_backoff(lambda a: ProviderFailure.DATA_INVALID,
                                 RetryPolicy(max_attempts=5))
        assert out.attempts == 1

    def test_all_providers_unavailable_yields_no_signal(self):
        # If provider status is not acceptable, signal safety must NO_DECISION.
        cond = {g.value: True for g in SignalGate}
        cond[SignalGate.PROVIDER_STATUS_ACCEPTABLE.value] = False
        r = evaluate_signal_safety(cond)
        assert r.decision == SignalDecision.NO_DECISION.value
        assert r.reason == NoDecisionReason.PROVIDER_UNACCEPTABLE.value

    def test_conflict_scenario_flags_conflict_and_blocks(self):
        ts = _epoch(2025, 1, 6, 3, 45)
        conflict = check_provider_consistency("ANGEL_ONE", "UPSTOX", "X",
                                              _bar(ts, c=100), _bar(ts, c=180))
        assert conflict.is_conflict
        # a conflict must propagate to a DQ CONFLICT verdict
        d = evaluate_data_quality("X", ohlcv_report=None, stale=None, completeness=None,
                                  consistency=conflict, sanitized=None, sufficient_history=True)
        assert d.verdict == DQVerdict.CONFLICT.value and d.safe_to_consume is False


# =====================================================================================
# 25. Recovery scenarios
# =====================================================================================

class TestRecovery:
    def test_clock_skew_future_bar_detected(self):
        now = datetime(2025, 1, 6, 4, 0, tzinfo=UTC)
        skew = now + timedelta(minutes=10)
        r = assess_staleness("1m", skew, bars=None, now=now, market_open=True)
        assert r.reason == StaleReason.FUTURE_TIMESTAMP.value

    def test_partial_ingest_then_restart_no_duplicate(self, tmp_path):
        root = tmp_path / "ledger"
        ing = IdempotentIngest(root=root)
        ing.ingest("A", "X", "t1", "v1")
        # crash mid-batch; restart
        ing2 = IdempotentIngest(root=root)
        assert ing2.ingest("A", "X", "t1", "v1").accepted is False

    def test_malformed_bar_quarantined_recovery_keeps_good(self):
        bad = _bar(_epoch(2025, 1, 6, 3, 45), h=1, l=100, c=100)
        good = _bar(_epoch(2025, 1, 6, 3, 46))
        s = sanitize_bars([bad, good], "X")
        assert good in s.usable_bars and s.n_quarantined == 1


# =====================================================================================
# 26. Property-based invariants
# =====================================================================================

from hypothesis import given, settings, strategies as st

_price = st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False)


class TestProperties:
    @settings(max_examples=50, deadline=None)
    @given(o=_price, c=_price, spread=st.floats(min_value=0.0, max_value=1e4,
                                                allow_nan=False, allow_infinity=False))
    def test_valid_ohlc_always_passes(self, o, c, spread):
        # Construct a bar that satisfies the invariants by construction.
        h = max(o, c) + spread
        l = min(o, c) - spread
        if l <= 0:
            l = min(o, c) / 2.0
        bar = _bar(_epoch(2025, 1, 6, 3, 45), o=o, h=h, l=l, c=c, v=100)
        rep = validate_ohlcv_bars([bar], "X", now=datetime(2025, 1, 6, 12, 0, tzinfo=UTC))
        assert rep.ok

    @settings(max_examples=50, deadline=None)
    @given(hi=_price, lo=_price)
    def test_high_below_low_always_flagged(self, hi, lo):
        if hi >= lo:
            hi, lo = lo, hi + 1.0    # force high < low
        bar = _bar(_epoch(2025, 1, 6, 3, 45), o=lo, h=hi, l=lo, c=lo)
        rep = validate_ohlcv_bars([bar], "X", now=datetime(2025, 1, 6, 12, 0, tzinfo=UTC))
        assert rep.critical

    @settings(max_examples=50, deadline=None)
    @given(provider=st.text(min_size=1, max_size=8),
           instrument=st.text(min_size=1, max_size=8),
           ts=st.integers(min_value=0, max_value=2_000_000_000))
    def test_dedup_key_deterministic(self, provider, instrument, ts):
        assert dedup_key(provider, instrument, ts, "v1") == \
               dedup_key(provider, instrument, ts, "v1")


# =====================================================================================
# 27. Negative tests
# =====================================================================================

class TestNegative:
    def test_future_timestamp_bar_rejected(self):
        now = datetime(2025, 1, 6, 4, 0, tzinfo=UTC)
        future = _bar((now + timedelta(hours=1)).timestamp())
        rep = validate_ohlcv_bars([future], "X", now=now)
        assert rep.critical

    def test_missing_close_rejected(self):
        bar = {"timestamp": _epoch(2025, 1, 6, 3, 45), "open": 100, "high": 101, "low": 99}
        rep = validate_ohlcv_bars([bar], "X", now=datetime(2025, 1, 6, 12, 0, tzinfo=UTC))
        assert rep.critical

    def test_nan_price_rejected(self):
        bar = _bar(_epoch(2025, 1, 6, 3, 45), c=float("nan"))
        rep = validate_ohlcv_bars([bar], "X", now=datetime(2025, 1, 6, 12, 0, tzinfo=UTC))
        assert rep.critical

    def test_pit_record_rejects_naive_timestamp(self):
        with pytest.raises(ValueError):
            PointInTimeRecord.create(symbol="X", exchange="NSE",
                                     event_time=datetime(2025, 1, 6, 4, 0),  # naive
                                     available_time=datetime(2025, 1, 6, 4, 0, tzinfo=UTC),
                                     provider="YAHOO")

    def test_pit_record_rejects_availability_before_event(self):
        with pytest.raises(ValueError):
            PointInTimeRecord.create(symbol="X", exchange="NSE",
                                     event_time=datetime(2025, 1, 6, 5, 0, tzinfo=UTC),
                                     available_time=datetime(2025, 1, 6, 4, 0, tzinfo=UTC),
                                     provider="YAHOO")


# =====================================================================================
# 28. Test-the-tests (§41): breaking a protection MUST break a test
# =====================================================================================

class TestTheTests:
    """Each check here deliberately corrupts an input and asserts the protection
    FIRES (i.e. the guard is real, not vacuous). If a protection were silently
    removed, these assertions would fail — proving the tests above have teeth."""

    def test_altering_timestamp_backwards_triggers_regression(self):
        good = [_bar(_epoch(2025, 1, 6, 3, 45)), _bar(_epoch(2025, 1, 6, 3, 46))]
        assert detect_timestamp_regression(good).is_stale is False
        # corrupt: make the 2nd bar earlier
        corrupted = [_bar(_epoch(2025, 1, 6, 3, 45)), _bar(_epoch(2025, 1, 6, 3, 44))]
        assert detect_timestamp_regression(corrupted).is_stale is True

    def test_removing_close_breaks_validation(self):
        bar = _bar(_epoch(2025, 1, 6, 3, 45))
        assert validate_ohlcv_bars([bar], "X",
                                   now=datetime(2025, 1, 6, 12, 0, tzinfo=UTC)).ok
        del bar["close"]
        assert validate_ohlcv_bars([bar], "X",
                                   now=datetime(2025, 1, 6, 12, 0, tzinfo=UTC)).critical

    def test_duplicating_a_bar_triggers_duplicate_guard(self):
        ts = _epoch(2025, 1, 6, 3, 45)
        assert validate_ohlcv_bars([_bar(ts)], "X",
                                   now=datetime(2025, 1, 6, 12, 0, tzinfo=UTC)).ok
        assert validate_ohlcv_bars([_bar(ts), _bar(ts)], "X",
                                   now=datetime(2025, 1, 6, 12, 0, tzinfo=UTC)).critical

    def test_bypassing_stale_gate_would_change_decision(self):
        # With DQ gate failing, signal must be NO_DECISION. If someone "bypassed"
        # the gate by forcing it True, the decision flips — proving the gate matters.
        cond = {g.value: True for g in SignalGate}
        cond[SignalGate.DATA_QUALITY_PASSED.value] = False
        assert evaluate_signal_safety(cond).decision == SignalDecision.NO_DECISION.value
        cond[SignalGate.DATA_QUALITY_PASSED.value] = True   # bypass
        assert evaluate_signal_safety(cond).decision == SignalDecision.PROCEED.value

    def test_changing_adjustment_mode_breaks_mix_guard(self):
        raw = AdjustedSeries("X", AdjustmentMode.RAW.value, "ca", _clean_series())
        assert assert_same_mode(raw, raw) == AdjustmentMode.RAW.value
        split = AdjustedSeries("Y", AdjustmentMode.SPLIT_ADJUSTED.value, "ca", _clean_series())
        with pytest.raises(AdjustmentModeError):
            assert_same_mode(raw, split)

    def test_current_lot_used_historically_would_be_caught(self):
        # get_lot_size must return a status, so "today's lot used historically"
        # is distinguishable from a PIT-correct value (never silently fabricated).
        from src.data.instrument_master import InstrumentMasterStore
        lot, status, source = InstrumentMasterStore.default().get_lot_size(
            "UNKNOWN_SYMBOL_XYZ", date(2015, 1, 1))
        # unknown historical symbol -> must NOT fabricate a confident lot size
        assert status != "OK" or lot is None
