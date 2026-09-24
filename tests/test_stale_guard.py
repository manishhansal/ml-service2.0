"""Tests for timeframe-specific stale-data blocking (mandate §62)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core.exceptions import StaleDataError
from src.features.stale_guard import StaleDataGuard


def _ts(**kw) -> datetime:
    return datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc) - timedelta(**kw)


class TestStaleDataGuard:
    def test_fresh_5m_data_passes(self):
        g = StaleDataGuard()
        now = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
        latest = now - timedelta(minutes=3)  # within 12m allowance
        g.check("5m", latest, now)  # must not raise

    def test_stale_5m_data_blocks(self):
        g = StaleDataGuard()
        now = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
        latest = now - timedelta(minutes=30)  # exceeds 12m allowance
        with pytest.raises(StaleDataError) as ei:
            g.check("5m", latest, now)
        assert ei.value.timeframe == "5m"
        assert ei.value.age_seconds > ei.value.max_age_seconds

    def test_thresholds_are_timeframe_specific(self):
        """A 30-minute-old quote is stale for 5m but fresh for 1d (§62)."""
        g = StaleDataGuard()
        now = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
        latest = now - timedelta(minutes=30)
        with pytest.raises(StaleDataError):
            g.check("5m", latest, now)
        g.check("1d", latest, now)  # daily tolerates 30 min easily

    def test_daily_is_session_aware(self):
        """Daily tolerates a weekend gap (session-aware, not one universal number)."""
        g = StaleDataGuard()
        now = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)  # Monday
        latest = now - timedelta(days=2, hours=1)  # Friday close-ish
        g.check("1d", latest, now)  # must not raise

    def test_future_dated_data_not_blocked_here(self):
        """Future-dated data is a PIT concern, not staleness — guard must not raise."""
        g = StaleDataGuard()
        now = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
        latest = now + timedelta(minutes=5)
        g.check("5m", latest, now)  # negative age -> no raise

    def test_evaluate_returns_age_and_limit(self):
        g = StaleDataGuard()
        now = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
        latest = now - timedelta(minutes=10)
        ok, age, limit = g.evaluate("5m", latest, now)
        # 10 min = 600s, 5m limit = 720s -> still fresh
        assert ok is True
        assert abs(age - 600) < 1.0
        assert limit == pytest.approx(720.0)

    def test_unknown_timeframe_uses_conservative_default(self):
        g = StaleDataGuard()
        now = datetime(2026, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
        latest = now - timedelta(hours=12)
        # default is 24h -> 12h passes
        g.check("7m", latest, now)
        latest2 = now - timedelta(hours=30)
        with pytest.raises(StaleDataError):
            g.check("7m", latest2, now)

    def test_naive_timestamps_treated_as_utc(self):
        g = StaleDataGuard()
        now = datetime(2026, 1, 5, 10, 0, 0)  # naive
        latest = datetime(2026, 1, 5, 9, 58, 0)  # naive, 2 min old
        g.check("5m", latest, now)  # must not raise
