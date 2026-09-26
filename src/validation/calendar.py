"""
src.validation.calendar — NSE exchange calendar and gap classification (mandate §4).

Distinguishes between:
  - EXPECTED_MARKET_CLOSURE  (weekend / NSE holiday / pre-open / post-close slot)
  - TRUE_MISSING_SESSION     (trading session or intraday bar that should exist)

Live calendar enrichment (Gap 2 fix):
  ``refresh_from_market_status(data_client)`` calls data-service2.0's
  ``GET /v1/india/market/status`` endpoint to pull the authoritative holiday
  list.  The result is merged into the module-level ``_NSE_HOLIDAYS`` set so
  subsequent ``is_market_open`` calls benefit immediately.  The static set
  below is the fallback when the live call is unavailable.

Intraday session model (NSE):
  Regular session: 09:15–15:30 IST  →  03:45–10:00 UTC
  Pre-open:        09:00–09:15 IST  (no OHLCV bars; treated as non-trading)
  Post-close:      15:30–16:00 IST  (block deal; no regular bars)

  Expected bars per session by interval (regular session only):
    1m  → 375   (375 minutes 09:15–15:30 inclusive first bar, exclusive last)
    5m  →  75
    10m →  38   (375 / 10, rounded up to account for partial last bar)
    15m →  25
    30m →  13   (375 / 30 = 12.5 → 13 bars)
    1h  →   7   (375 / 60 = 6.25 → 7 bars)
    1d  →   1   (one daily bar per session)
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Any, Sequence

# ── NSE session constants (IST = UTC+5:30) ───────────────────────────────────

_IST = timezone(timedelta(hours=5, minutes=30))
_UTC = timezone.utc

# Regular session open/close in UTC
_SESSION_OPEN_UTC  = time(3, 45)   # 09:15 IST
_SESSION_CLOSE_UTC = time(10, 0)   # 15:30 IST

# Expected bars per full trading session for each intraday interval
# (partial last bar is counted; pre-open / post-close slots excluded)
_EXPECTED_BARS_PER_SESSION: dict[str, int] = {
    "1m":  375,
    "5m":   75,
    "10m":  38,   # ceil(375/10)
    "15m":  25,
    "30m":  13,   # ceil(375/30)
    "1h":    7,   # ceil(375/60)
}

# ── Gap classification ───────────────────────────────────────────────────────


class GapType(str, Enum):
    EXPECTED_MARKET_CLOSURE  = "EXPECTED_MARKET_CLOSURE"   # weekend / holiday
    TRUE_MISSING_SESSION     = "TRUE_MISSING_SESSION"       # missing full day
    MISSING_INTRADAY_BARS    = "MISSING_INTRADAY_BARS"      # partial gap within a session
    PARTIAL_SESSION          = "PARTIAL_SESSION"            # fewer bars than expected
    STALE_OBSERVATION        = "STALE_OBSERVATION"          # data but too old


# ── Static NSE holiday set (2021–2026) ───────────────────────────────────────
# Authoritative source: NSE published trading holiday calendar.
# This set is the FALLBACK when data-service2.0 is unavailable.
# Call ``refresh_from_market_status()`` at startup to load the live calendar.

_NSE_HOLIDAYS: set[date] = {
    # 2019
    date(2019, 3, 4),  date(2019, 3, 21), date(2019, 4, 17), date(2019, 4, 19),
    date(2019, 4, 29), date(2019, 6, 5),  date(2019, 8, 12), date(2019, 8, 15),
    date(2019, 9, 2),  date(2019, 10, 2), date(2019, 10, 8), date(2019, 10, 28),
    date(2019, 11, 12), date(2019, 12, 25),
    # 2020
    date(2020, 2, 21), date(2020, 3, 10), date(2020, 4, 2),  date(2020, 4, 6),
    date(2020, 4, 10), date(2020, 4, 14), date(2020, 5, 25), date(2020, 10, 2),
    date(2020, 11, 16), date(2020, 11, 30), date(2020, 12, 25),
    # 2021
    date(2021, 1, 26), date(2021, 3, 11), date(2021, 4, 2),  date(2021, 4, 14),
    date(2021, 5, 13), date(2021, 7, 21), date(2021, 8, 19), date(2021, 9, 10),
    date(2021, 10, 15), date(2021, 11, 4), date(2021, 11, 5), date(2021, 11, 19),
    # 2022
    date(2022, 1, 26), date(2022, 3, 1),  date(2022, 3, 18), date(2022, 4, 14),
    date(2022, 4, 15), date(2022, 5, 3),  date(2022, 8, 9),  date(2022, 8, 15),
    date(2022, 10, 2), date(2022, 10, 5), date(2022, 10, 24), date(2022, 10, 26),
    date(2022, 11, 8),
    # 2023
    date(2023, 1, 26), date(2023, 3, 7),  date(2023, 3, 30), date(2023, 4, 4),
    date(2023, 4, 7),  date(2023, 4, 14), date(2023, 5, 1),  date(2023, 6, 29),
    date(2023, 8, 15), date(2023, 9, 19), date(2023, 10, 2), date(2023, 10, 24),
    date(2023, 11, 14), date(2023, 11, 27), date(2023, 12, 25),
    # 2024
    date(2024, 1, 22), date(2024, 1, 26), date(2024, 3, 25), date(2024, 3, 29),
    date(2024, 4, 11), date(2024, 4, 14), date(2024, 4, 17), date(2024, 5, 23),
    date(2024, 6, 17), date(2024, 7, 17), date(2024, 8, 15), date(2024, 10, 2),
    date(2024, 10, 12), date(2024, 11, 1), date(2024, 11, 15), date(2024, 12, 25),
    # 2025
    date(2025, 2, 26), date(2025, 3, 14), date(2025, 4, 10), date(2025, 4, 14),
    date(2025, 4, 18), date(2025, 5, 1),  date(2025, 8, 15), date(2025, 8, 27),
    date(2025, 10, 2), date(2025, 10, 20), date(2025, 10, 21),
    date(2025, 11, 5), date(2025, 12, 25),
    # 2026 (partial — live refresh will extend this)
    date(2026, 1, 26), date(2026, 3, 4),  date(2026, 3, 20), date(2026, 4, 3),
    date(2026, 8, 15),
}

# asyncio.Lock for async-safe mutation of _NSE_HOLIDAYS and _live_calendar_loaded.
# Using asyncio.Lock (not threading.Lock) because refresh_from_market_status() is
# an async function running on the event loop — a threading.Lock would block the
# loop if contended.  A new lock is created lazily per event loop to avoid
# "attached to a different loop" errors in tests that spin up their own loops.
_HOLIDAY_LOCK: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    """Return the asyncio.Lock for the current event loop, creating it if needed."""
    global _HOLIDAY_LOCK
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if _HOLIDAY_LOCK is None:
        _HOLIDAY_LOCK = asyncio.Lock()
    return _HOLIDAY_LOCK


# Track whether a live refresh has succeeded this process lifetime.
# Written only while holding the asyncio lock (HIGH-3 fix).
_live_calendar_loaded: bool = False


# ── Live calendar refresh ────────────────────────────────────────────────────


async def refresh_from_market_status(data_client: Any) -> bool:
    """Enrich the in-process holiday set from data-service2.0's live calendar.

    Calls ``GET /v1/india/market/status`` which returns a ``holidayList`` array
    of ISO-8601 date strings for the current year.  Any dates not already in
    ``_NSE_HOLIDAYS`` are merged in atomically.

    HIGH-3 fix:
    - Uses ``asyncio.Lock`` (not ``threading.Lock``) so the event loop is never
      blocked when two coroutines call this concurrently at startup.
    - Both the holiday-set update AND the ``_live_calendar_loaded`` flag are
      written while the lock is held, eliminating the previous race where the
      flag was set outside the lock.

    Args:
        data_client: A connected ``DataServiceClient`` instance.

    Returns:
        True if the live calendar was loaded successfully, False on any failure
        (the static fallback remains active).
    """
    global _live_calendar_loaded
    try:
        status = await data_client.get_market_status()
        holiday_dates = _extract_holidays(status)
        if holiday_dates:
            async with _get_lock():                  # async lock — never blocks loop
                _NSE_HOLIDAYS.update(holiday_dates)
                _live_calendar_loaded = True         # written INSIDE lock (HIGH-3 fix)
            return True
        return False
    except Exception:
        return False


def _extract_holidays(status: dict[str, Any]) -> set[date]:
    """Parse holiday dates from a data-service2.0 market-status payload.

    Handles multiple envelope shapes:
      {"holidayList": ["2025-01-26", ...]}
      {"data": {"holidayList": [...]}}
      {"holidays": [...]}
      {"data": {"holidays": [...]}}
    """
    raw: list[Any] = (
        status.get("holidayList")
        or (status.get("data") or {}).get("holidayList")
        or status.get("holidays")
        or (status.get("data") or {}).get("holidays")
        or []
    )
    result: set[date] = set()
    for entry in raw:
        d = _parse_date_entry(entry)
        if d is not None:
            result.add(d)
    return result


def _parse_date_entry(entry: Any) -> date | None:
    """Convert a raw holiday entry (string, dict, or date) to a date object."""
    if isinstance(entry, date):
        return entry
    if isinstance(entry, str):
        try:
            return date.fromisoformat(entry[:10])
        except ValueError:
            return None
    if isinstance(entry, dict):
        # {"date": "2025-01-26", "description": "Republic Day"}
        raw = entry.get("date") or entry.get("tradingDate") or entry.get("holidayDate")
        if raw:
            return _parse_date_entry(raw)
    return None


def is_calendar_live() -> bool:
    """Return True when the holiday set has been enriched from a live source."""
    return _live_calendar_loaded


# ── Core calendar helpers ────────────────────────────────────────────────────


def is_market_open(d: date) -> bool:
    """Return True if NSE was/is expected to be open on *d* (date-level check).

    Reads _NSE_HOLIDAYS without a lock.  This is safe because:
    - CPython's GIL protects individual set membership tests.
    - _NSE_HOLIDAYS is only mutated during startup (refresh_from_market_status)
      which completes before serving begins.
    """
    if d.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    return d not in _NSE_HOLIDAYS


def is_trading_timestamp(ts: datetime, interval: str) -> bool:
    """Return True if *ts* (UTC-aware) falls inside a valid NSE bar slot.

    For intraday intervals a bar is valid when its timestamp falls within the
    regular session window [09:15 IST, 15:30 IST].  The first bar of the day
    opens at 09:15 IST; the last bar closes at 15:30 IST.

    For daily/weekly/monthly intervals the date-level ``is_market_open`` check
    is sufficient.
    """
    if interval in ("1d", "1w", "1M"):
        return is_market_open(ts.date())

    # Normalise to UTC
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_UTC)
    ts_utc = ts.astimezone(_UTC)

    # Date must be a trading day
    if not is_market_open(ts_utc.date()):
        return False

    # Bar open time must be within session [03:45 UTC, 10:00 UTC)
    bar_time = ts_utc.time().replace(second=0, microsecond=0)
    return _SESSION_OPEN_UTC <= bar_time < _SESSION_CLOSE_UTC


def expected_bars_in_session(interval: str) -> int:
    """Return the number of OHLCV bars expected in a full NSE trading session."""
    return _EXPECTED_BARS_PER_SESSION.get(interval, 0)


def expected_trading_days(first: date, last: date) -> set[date]:
    """Return the set of NSE trading days in the closed range [first, last]."""
    result: set[date] = set()
    cur = first
    while cur <= last:
        if is_market_open(cur):
            result.add(cur)
        cur += timedelta(days=1)
    return result


# ── Gap classification (daily) ────────────────────────────────────────────────


def classify_gap(d1: date, d2: date) -> list[dict[str, Any]]:
    """Classify every calendar day in the open interval (d1, d2) exclusive.

    Returns a list of dicts:
      {"date": str, "gap_type": GapType.value, "weekday": int}
    """
    result: list[dict[str, Any]] = []
    current = d1 + timedelta(days=1)
    while current < d2:
        gtype = (
            GapType.EXPECTED_MARKET_CLOSURE
            if not is_market_open(current)
            else GapType.TRUE_MISSING_SESSION
        )
        result.append({
            "date": current.isoformat(),
            "gap_type": gtype.value,
            "weekday": current.weekday(),
        })
        current += timedelta(days=1)
    return result


def classify_gaps_in_series(dates: Sequence[date]) -> dict[str, Any]:
    """Analyse a sorted sequence of *daily* trading dates for gaps.

    Returns a summary dict:
      n_expected_closures, n_true_missing_sessions,
      true_missing_dates, expected_closure_sample, n_total_gaps
    """
    sorted_dates = sorted(dates)
    n_expected = 0
    n_missing  = 0
    missing:   list[str] = []
    closures:  list[str] = []

    for i in range(1, len(sorted_dates)):
        gaps = classify_gap(sorted_dates[i - 1], sorted_dates[i])
        for g in gaps:
            if g["gap_type"] == GapType.TRUE_MISSING_SESSION.value:
                n_missing += 1
                missing.append(g["date"])
            else:
                n_expected += 1
                closures.append(g["date"])

    return {
        "n_expected_closures":     n_expected,
        "n_true_missing_sessions": n_missing,
        "true_missing_dates":      missing,
        "expected_closure_sample": closures[:10],
        "n_total_gaps":            n_expected + n_missing,
    }


# ── Gap classification (intraday) ────────────────────────────────────────────


def classify_intraday_gaps(
    timestamps: Sequence[datetime],
    interval: str,
) -> dict[str, Any]:
    """Classify gaps in an intraday timestamp series for a given NSE interval.

    Algorithm
    ---------
    1. Discard any timestamp outside valid trading hours or on non-trading days.
    2. Group remaining timestamps by calendar date (session).
    3. For each session:
       a. Count observed bars.
       b. Compare to ``expected_bars_in_session(interval)``.
       c. Classify the session as FULL, PARTIAL, or MISSING.
    4. Scan consecutive timestamps for gaps wider than the bar interval
       (within a session) — these are MISSING_INTRADAY_BARS.

    Args:
        timestamps: Sequence of UTC-aware datetimes from the ingested DataFrame.
        interval:   Bar interval string (e.g. "5m", "15m", "1h").

    Returns:
        dict with keys:
          n_sessions_expected    — trading days spanned by the data
          n_sessions_full        — sessions with expected number of bars
          n_sessions_partial     — sessions with fewer bars than expected
          n_sessions_missing     — expected sessions with zero bars
          n_intraday_gaps        — within-session bar gaps
          intraday_gap_details   — list of {"session_date", "gap_before_ts", "bars_missing"}
          session_summary        — per-date bar counts
          gap_rate_pct           — (missing + partial) / expected * 100
    """
    if not timestamps:
        return _empty_intraday_result()

    interval_minutes = _interval_to_minutes(interval)
    if interval_minutes <= 0:
        return _empty_intraday_result()

    expected_per_session = expected_bars_in_session(interval)

    # ── 1. Filter to valid trading timestamps ────────────────────────────────
    valid: list[datetime] = []
    for ts in timestamps:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_UTC)
        if is_trading_timestamp(ts, interval):
            valid.append(ts.astimezone(_UTC))

    if not valid:
        return _empty_intraday_result()

    valid.sort()

    # ── 2. Group by session date ─────────────────────────────────────────────
    sessions: dict[date, list[datetime]] = {}
    for ts in valid:
        d = ts.date()
        sessions.setdefault(d, []).append(ts)

    # ── 3. Determine expected sessions in the data span ──────────────────────
    first_date = valid[0].date()
    last_date  = valid[-1].date()
    all_expected = expected_trading_days(first_date, last_date)

    n_full    = 0
    n_partial = 0
    n_missing = 0
    session_summary: dict[str, int] = {}

    for session_date in sorted(all_expected):
        observed = len(sessions.get(session_date, []))
        session_summary[session_date.isoformat()] = observed
        if observed == 0:
            n_missing += 1
        elif observed < expected_per_session:
            n_partial += 1
        else:
            n_full += 1

    # ── 4. Within-session intraday gaps ──────────────────────────────────────
    n_intraday_gaps   = 0
    gap_details: list[dict[str, Any]] = []
    gap_threshold_s   = interval_minutes * 60 * 1.5   # 1.5× interval = a missing bar

    for session_date, session_ts in sorted(sessions.items()):
        for i in range(1, len(session_ts)):
            delta_s = (session_ts[i] - session_ts[i - 1]).total_seconds()
            if delta_s > gap_threshold_s:
                n_bars_missing = int(round(delta_s / (interval_minutes * 60))) - 1
                n_intraday_gaps += 1
                gap_details.append({
                    "session_date":  session_date.isoformat(),
                    "gap_before_ts": session_ts[i].isoformat(),
                    "bars_missing":  n_bars_missing,
                    "gap_seconds":   int(delta_s),
                })

    n_expected_total = len(all_expected)
    gap_rate = (
        (n_missing + n_partial) / n_expected_total * 100
        if n_expected_total > 0 else 0.0
    )

    return {
        "n_sessions_expected":  n_expected_total,
        "n_sessions_full":      n_full,
        "n_sessions_partial":   n_partial,
        "n_sessions_missing":   n_missing,
        "n_intraday_gaps":      n_intraday_gaps,
        "intraday_gap_details": gap_details[:50],   # cap for log safety
        "session_summary":      session_summary,
        "gap_rate_pct":         round(gap_rate, 2),
        "interval":             interval,
    }


def _empty_intraday_result() -> dict[str, Any]:
    return {
        "n_sessions_expected": 0, "n_sessions_full": 0,
        "n_sessions_partial": 0,  "n_sessions_missing": 0,
        "n_intraday_gaps": 0,     "intraday_gap_details": [],
        "session_summary": {},    "gap_rate_pct": 0.0, "interval": "",
    }


def _interval_to_minutes(interval: str) -> int:
    """Convert a canonical interval string to integer minutes. Returns 0 if unknown."""
    _MAP = {
        "1m": 1, "5m": 5, "10m": 10, "15m": 15,
        "30m": 30, "1h": 60, "1d": 1440, "1w": 10080,
    }
    return _MAP.get(interval, 0)
