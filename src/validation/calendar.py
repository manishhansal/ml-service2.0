"""
src.validation.calendar — NSE exchange calendar and gap classification (mandate §4).

Distinguishes between:
  - EXPECTED_MARKET_CLOSURE  (weekend / NSE holiday)
  - TRUE_MISSING_SESSION     (trading session with no data)

Never reports a weekend or holiday as a data gap.
"""
from __future__ import annotations

from datetime import date, timedelta
from enum import Enum
from typing import Sequence


class GapType(str, Enum):
    EXPECTED_MARKET_CLOSURE = "EXPECTED_MARKET_CLOSURE"
    TRUE_MISSING_SESSION = "TRUE_MISSING_SESSION"
    PARTIAL_SESSION = "PARTIAL_SESSION"
    STALE_OBSERVATION = "STALE_OBSERVATION"


# NSE holidays (approximate; expand with live calendar API if available).
# Sourced from NSE published holiday calendar.  This is NOT exhaustive for all
# years — it is a best-effort static set.  In production, this should be fetched
# from data-service2.0's /india/market-status endpoint.
_NSE_HOLIDAYS: set[date] = {
    # 2021
    date(2021, 1, 26), date(2021, 3, 11), date(2021, 4, 2), date(2021, 4, 14),
    date(2021, 5, 13), date(2021, 7, 21), date(2021, 8, 19), date(2021, 9, 10),
    date(2021, 10, 15), date(2021, 11, 4), date(2021, 11, 5), date(2021, 11, 19),
    # 2022
    date(2022, 1, 26), date(2022, 3, 1), date(2022, 3, 18), date(2022, 4, 14),
    date(2022, 4, 15), date(2022, 5, 3), date(2022, 8, 9), date(2022, 8, 15),
    date(2022, 10, 2), date(2022, 10, 5), date(2022, 10, 24), date(2022, 10, 26),
    date(2022, 11, 8),
    # 2023
    date(2023, 1, 26), date(2023, 3, 7), date(2023, 3, 30), date(2023, 4, 4),
    date(2023, 4, 7), date(2023, 4, 14), date(2023, 5, 1), date(2023, 6, 29),
    date(2023, 8, 15), date(2023, 9, 19), date(2023, 10, 2), date(2023, 10, 24),
    date(2023, 11, 14), date(2023, 11, 27), date(2023, 12, 25),
    # 2024
    date(2024, 1, 22), date(2024, 1, 26), date(2024, 3, 25), date(2024, 3, 29),
    date(2024, 4, 11), date(2024, 4, 14), date(2024, 4, 17), date(2024, 5, 23),
    date(2024, 6, 17), date(2024, 7, 17), date(2024, 8, 15), date(2024, 10, 2),
    date(2024, 10, 12), date(2024, 11, 1), date(2024, 11, 15), date(2024, 12, 25),
    # 2025
    date(2025, 2, 26), date(2025, 3, 14), date(2025, 4, 10), date(2025, 4, 14),
    date(2025, 4, 18), date(2025, 5, 1), date(2025, 8, 15), date(2025, 8, 27),
    date(2025, 10, 2), date(2025, 10, 2), date(2025, 10, 20), date(2025, 10, 21),
    date(2025, 11, 5), date(2025, 12, 25),
    # 2026 (partial — update as calendar is published)
    date(2026, 1, 26), date(2026, 3, 4), date(2026, 3, 20), date(2026, 4, 3),
    date(2026, 8, 15),
}


def is_market_open(d: date) -> bool:
    """Return True if NSE was/is expected to be open on *d*."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return d not in _NSE_HOLIDAYS


def classify_gap(d1: date, d2: date) -> list[dict[str, str]]:
    """
    Classify every calendar day in the open interval (d1, d2) exclusive.

    Returns a list of dicts describing each gap day:
      {"date": str, "gap_type": GapType, "weekday": int}

    A day is EXPECTED_MARKET_CLOSURE if it is a weekend or NSE holiday.
    A day is TRUE_MISSING_SESSION if it should have been a trading day.
    """
    result = []
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
    """
    Analyse a sorted sequence of trading dates for gaps.

    Returns a summary dict with:
      - n_expected_closures
      - n_true_missing_sessions
      - true_missing_dates: list of date strings
      - expected_closure_dates: list (weekends + holidays)
    """
    from typing import Any
    sorted_dates = sorted(dates)
    n_expected = 0
    n_missing = 0
    missing: list[str] = []
    closures: list[str] = []

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
        "n_expected_closures": n_expected,
        "n_true_missing_sessions": n_missing,
        "true_missing_dates": missing,
        "expected_closure_sample": closures[:10],  # first 10 for brevity
        "n_total_gaps": n_expected + n_missing,
    }
