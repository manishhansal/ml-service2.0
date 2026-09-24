"""
src.features.stale_guard — timeframe-specific stale-data blocking (mandate §62).

The previous certification flagged stale-data blocking as only PARTIAL. This
closes it: before any live inference/trade decision, the freshest available
market datum must be recent enough for the strategy's timeframe. If

    now - latest_data_timestamp > max_allowed_staleness(timeframe)

then the decision is blocked with ``STALE_MARKET_DATA`` (NO_TRADE).

The threshold is NOT a single universal number (§62): a 5m strategy tolerates
only a few minutes of staleness, whereas a daily strategy is session-aware and
tolerates an overnight gap. Thresholds are documented and configurable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.core.exceptions import StaleDataError

# Maximum allowed staleness (seconds) by timeframe. These are multiples of the
# bar interval plus a tolerance for feed latency; daily/weekly are session-aware
# (an overnight/weekend gap is normal, so the allowance spans a non-trading gap).
_DEFAULT_MAX_STALENESS_S: dict[str, float] = {
    "1m": 3 * 60,        # 3 bars
    "5m": 12 * 60,       # ~2.4 bars — strict intraday
    "10m": 25 * 60,
    "15m": 40 * 60,
    "30m": 75 * 60,
    "1h": 150 * 60,
    "1d": 4 * 24 * 3600,   # session-aware: tolerate weekend + a holiday
    "1w": 10 * 24 * 3600,
    "1M": 40 * 24 * 3600,
}


@dataclass
class StaleDataGuard:
    """Blocks trading on market data that is too old for the timeframe (§62).

    Usage::

        guard = StaleDataGuard()
        guard.check("5m", latest_data_ts, now)   # raises StaleDataError if stale
        # or, non-raising:
        ok, age, limit = guard.evaluate("5m", latest_data_ts, now)
    """

    max_staleness_s: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_MAX_STALENESS_S)
    )
    # Fallback for unknown timeframes: conservative (block if older than 1 day).
    default_max_staleness_s: float = 24 * 3600

    def limit_for(self, timeframe: str) -> float:
        return self.max_staleness_s.get(timeframe, self.default_max_staleness_s)

    def evaluate(
        self, timeframe: str, latest_data_ts: datetime, now: datetime | None = None
    ) -> tuple[bool, float, float]:
        """Return ``(is_fresh, age_seconds, max_age_seconds)`` without raising."""
        now = now or datetime.now(tz=timezone.utc)
        if latest_data_ts.tzinfo is None:
            latest_data_ts = latest_data_ts.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        age = (now - latest_data_ts).total_seconds()
        limit = self.limit_for(timeframe)
        return (age <= limit, age, limit)

    def check(
        self, timeframe: str, latest_data_ts: datetime, now: datetime | None = None
    ) -> None:
        """Raise :class:`StaleDataError` if the data is too old for *timeframe*.

        A negative age (data timestamped in the future) is NOT handled here —
        that is a PIT concern (:class:`LookAheadGuard`). This guard only blocks
        data that is too far in the past.
        """
        is_fresh, age, limit = self.evaluate(timeframe, latest_data_ts, now)
        if age < 0:
            # Future-dated data is a look-ahead concern, not staleness; let the
            # PIT guard handle it. Do not block here.
            return
        if not is_fresh:
            raise StaleDataError(timeframe=timeframe, age_seconds=age, max_age_seconds=limit)
