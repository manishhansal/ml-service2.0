"""
tests/test_intraday_gap_detection.py — unit tests for intraday gap detection.

Covers:
  - classify_intraday_gaps: full session, partial session, missing session
  - classify_intraday_gaps: within-session bar gaps
  - classify_intraday_gaps: weekend / holiday timestamps excluded
  - classify_intraday_gaps: empty input
  - is_trading_timestamp: valid / pre-open / post-close / weekend / holiday
  - expected_bars_in_session: correct counts per interval
  - _interval_to_minutes: all canonical intervals
  - normalize_bars integration: intraday gap fields populated
  - to_dict() JSON safety for intraday_gap_details (LOW-1 fix)
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

import os
os.environ.setdefault("ML_SERVICE_API_KEY", "test-key")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from src.validation.calendar import (
    _EXPECTED_BARS_PER_SESSION,
    _interval_to_minutes,
    classify_intraday_gaps,
    expected_bars_in_session,
    is_trading_timestamp,
)
from src.data.ingestion import IngestionValidationReport, normalize_bars

UTC = timezone.utc

# ── Helpers ───────────────────────────────────────────────────────────────────

def _session_ts(session_date: date, interval: str) -> list[datetime]:
    """Return a FULL set of bar timestamps for a single NSE session."""
    n = expected_bars_in_session(interval)
    minutes = _interval_to_minutes(interval)
    # Session opens at 09:15 IST = 03:45 UTC
    base = datetime(session_date.year, session_date.month, session_date.day,
                    3, 45, 0, tzinfo=UTC)
    return [base + timedelta(minutes=minutes * i) for i in range(n)]


def _make_bars(timestamps: list[datetime]) -> list[dict]:
    return [
        {
            "timestamp": ts.isoformat(),
            "open": 22000.0, "high": 22100.0,
            "low": 21900.0, "close": 22050.0,
            "volume": 500_000.0,
        }
        for ts in timestamps
    ]


# ── is_trading_timestamp ──────────────────────────────────────────────────────

class TestIsTradingTimestamp:
    # Normal trading day, 10:00 IST = 04:30 UTC
    _valid = datetime(2024, 10, 3, 4, 30, 0, tzinfo=UTC)
    # 08:30 IST = 03:00 UTC (pre-open)
    _pre_open = datetime(2024, 10, 3, 3, 0, 0, tzinfo=UTC)
    # 16:30 IST = 11:00 UTC (post-close)
    _post_close = datetime(2024, 10, 3, 11, 0, 0, tzinfo=UTC)
    # Saturday
    _weekend = datetime(2024, 10, 5, 5, 0, 0, tzinfo=UTC)
    # Oct 2 = Gandhi Jayanti holiday
    _holiday = datetime(2024, 10, 2, 5, 0, 0, tzinfo=UTC)

    def test_valid_intraday_ts(self):
        assert is_trading_timestamp(self._valid, "5m") is True

    def test_pre_open_rejected(self):
        assert is_trading_timestamp(self._pre_open, "5m") is False

    def test_post_close_rejected(self):
        assert is_trading_timestamp(self._post_close, "5m") is False

    def test_weekend_rejected(self):
        assert is_trading_timestamp(self._weekend, "5m") is False

    def test_holiday_rejected(self):
        assert is_trading_timestamp(self._holiday, "5m") is False

    def test_daily_interval_delegates_to_market_open(self):
        # For 1d, is_trading_timestamp just checks is_market_open
        assert is_trading_timestamp(self._valid, "1d") is True
        assert is_trading_timestamp(self._weekend, "1d") is False

    def test_naive_datetime_handled_as_utc(self):
        naive_ts = datetime(2024, 10, 3, 4, 30, 0)  # no tzinfo
        # Should not raise; treated as UTC
        result = is_trading_timestamp(naive_ts, "5m")
        assert isinstance(result, bool)

    def test_first_bar_of_session_valid(self):
        # 09:15 IST = 03:45 UTC — first valid bar
        first_bar = datetime(2024, 10, 3, 3, 45, 0, tzinfo=UTC)
        assert is_trading_timestamp(first_bar, "1m") is True

    def test_last_bar_boundary(self):
        # 15:30 IST = 10:00 UTC — _SESSION_CLOSE_UTC is exclusive
        last_bar = datetime(2024, 10, 3, 10, 0, 0, tzinfo=UTC)
        assert is_trading_timestamp(last_bar, "5m") is False


# ── expected_bars_in_session ──────────────────────────────────────────────────

class TestExpectedBars:
    def test_1m_is_375(self):
        assert expected_bars_in_session("1m") == 375

    def test_5m_is_75(self):
        assert expected_bars_in_session("5m") == 75

    def test_10m_is_38(self):
        assert expected_bars_in_session("10m") == 38

    def test_15m_is_25(self):
        assert expected_bars_in_session("15m") == 25

    def test_30m_is_13(self):
        assert expected_bars_in_session("30m") == 13

    def test_1h_is_7(self):
        assert expected_bars_in_session("1h") == 7

    def test_unknown_interval_returns_0(self):
        assert expected_bars_in_session("3m") == 0
        assert expected_bars_in_session("4h") == 0


# ── _interval_to_minutes ──────────────────────────────────────────────────────

class TestIntervalToMinutes:
    def test_canonical_intervals(self):
        assert _interval_to_minutes("1m") == 1
        assert _interval_to_minutes("5m") == 5
        assert _interval_to_minutes("10m") == 10
        assert _interval_to_minutes("15m") == 15
        assert _interval_to_minutes("30m") == 30
        assert _interval_to_minutes("1h") == 60
        assert _interval_to_minutes("1d") == 1440

    def test_unknown_returns_0(self):
        assert _interval_to_minutes("3m") == 0
        assert _interval_to_minutes("4h") == 0


# ── classify_intraday_gaps ────────────────────────────────────────────────────

class TestClassifyIntradayGaps:
    def test_empty_input_returns_zeros(self):
        result = classify_intraday_gaps([], "5m")
        assert result["n_sessions_expected"] == 0
        assert result["n_sessions_full"] == 0
        assert result["gap_rate_pct"] == 0.0

    def test_full_session_5m(self):
        # Oct 3 2024 = Thursday
        ts = _session_ts(date(2024, 10, 3), "5m")
        result = classify_intraday_gaps(ts, "5m")
        assert result["n_sessions_expected"] >= 1
        assert result["n_sessions_full"] >= 1
        assert result["n_sessions_partial"] == 0
        assert result["n_sessions_missing"] == 0
        assert result["n_intraday_gaps"] == 0
        assert result["gap_rate_pct"] == 0.0

    def test_partial_session_detected(self):
        # 65 / 75 bars → partial
        ts = _session_ts(date(2024, 10, 3), "5m")[:65]
        result = classify_intraday_gaps(ts, "5m")
        assert result["n_sessions_partial"] >= 1
        assert result["gap_rate_pct"] > 0.0

    def test_missing_session_detected(self):
        # Two sessions: Oct 3 and Oct 7 (skip Oct 4 Friday)
        ts_3  = _session_ts(date(2024, 10, 3), "5m")
        ts_7  = _session_ts(date(2024, 10, 7), "5m")
        # Oct 4 (Friday) is missing — it is a true missing session
        ts = ts_3 + ts_7
        result = classify_intraday_gaps(ts, "5m")
        # Oct 4 should be counted as a missing session
        assert result["n_sessions_missing"] >= 1

    def test_within_session_gap_detected(self):
        # Full session but skip 3 bars in the middle → within-session gap
        ts = _session_ts(date(2024, 10, 3), "5m")
        ts_with_gap = ts[:30] + ts[33:]  # skip bars 30, 31, 32
        result = classify_intraday_gaps(ts_with_gap, "5m")
        assert result["n_intraday_gaps"] >= 1
        assert len(result["intraday_gap_details"]) >= 1

    def test_weekend_timestamps_excluded(self):
        # Saturday bars are outside NSE session → filtered out → empty result
        sat = date(2024, 10, 5)
        ts = [
            datetime(2024, 10, 5, 4, 0, tzinfo=UTC) + timedelta(minutes=5 * i)
            for i in range(10)
        ]
        result = classify_intraday_gaps(ts, "5m")
        assert result["n_sessions_expected"] == 0

    def test_holiday_timestamps_excluded(self):
        # Oct 2 = Gandhi Jayanti
        ts = [
            datetime(2024, 10, 2, 4, 0, tzinfo=UTC) + timedelta(minutes=5 * i)
            for i in range(10)
        ]
        result = classify_intraday_gaps(ts, "5m")
        assert result["n_sessions_expected"] == 0

    def test_multiple_full_sessions_1h(self):
        ts = []
        for d in [date(2024, 10, 3), date(2024, 10, 4), date(2024, 10, 7)]:
            ts.extend(_session_ts(d, "1h"))
        result = classify_intraday_gaps(ts, "1h")
        assert result["n_sessions_full"] == 3
        assert result["n_sessions_missing"] == 0


# ── normalize_bars intraday gap field integration ─────────────────────────────

class TestNormalizeBarsIntradayFields:
    def test_5m_gap_fields_populated(self):
        ts = _session_ts(date(2024, 10, 3), "5m")
        bars = _make_bars(ts)
        report = IngestionValidationReport(symbol="NIFTY", interval="5m")
        normalize_bars(bars, "NIFTY", "5m", report)
        assert report.intraday_sessions_expected >= 1
        assert report.intraday_sessions_full >= 1
        assert report.intraday_gap_rate_pct == 0.0

    def test_partial_5m_session_flags_gap(self):
        ts = _session_ts(date(2024, 10, 3), "5m")[:60]  # partial
        bars = _make_bars(ts)
        report = IngestionValidationReport(symbol="NIFTY", interval="5m")
        normalize_bars(bars, "NIFTY", "5m", report)
        assert report.intraday_sessions_partial >= 1
        assert report.gaps_detected > 0

    def test_1h_gap_fields_populated(self):
        ts = _session_ts(date(2024, 10, 3), "1h")
        bars = _make_bars(ts)
        report = IngestionValidationReport(symbol="NIFTY", interval="1h")
        normalize_bars(bars, "NIFTY", "1h", report)
        assert report.intraday_sessions_expected >= 1

    def test_daily_interval_does_not_populate_intraday_fields(self):
        bars = [
            {
                "timestamp": datetime(2024, 10, d, 10, 0, tzinfo=UTC).isoformat(),
                "open": 22000.0, "high": 22100.0, "low": 21900.0,
                "close": 22050.0, "volume": 500_000.0,
            }
            for d in range(3, 8)   # Oct 3–7 (Mon–Fri)
        ]
        report = IngestionValidationReport(symbol="NIFTY", interval="1d")
        normalize_bars(bars, "NIFTY", "1d", report)
        # daily interval uses NSE calendar gap detection, not intraday
        assert report.intraday_sessions_expected == 0


# ── to_dict() JSON safety (LOW-1 fix) ─────────────────────────────────────────

class TestGapDetailsJSONSafety:
    def test_to_dict_is_json_serializable_with_intraday_details(self):
        ts = _session_ts(date(2024, 10, 3), "5m")
        ts_with_gap = ts[:30] + ts[33:]  # creates a within-session gap
        bars = _make_bars(ts_with_gap)
        report = IngestionValidationReport(symbol="NIFTY", interval="5m")
        normalize_bars(bars, "NIFTY", "5m", report)
        d = report.to_dict()
        # Must not raise TypeError
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_intraday_gap_details_are_strings_in_dict(self):
        ts = _session_ts(date(2024, 10, 3), "5m")
        ts_with_gap = ts[:30] + ts[33:]
        bars = _make_bars(ts_with_gap)
        report = IngestionValidationReport(symbol="NIFTY", interval="5m")
        normalize_bars(bars, "NIFTY", "5m", report)
        d = report.to_dict()
        for detail in d["intraday_gap_details"]:
            for v in detail.values():
                # No datetime or date objects allowed — must be primitive types
                assert not isinstance(v, (datetime, date)), (
                    f"Non-serializable type {type(v)} in gap detail: {v}"
                )
