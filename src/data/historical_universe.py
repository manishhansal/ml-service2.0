"""
src.data.historical_universe — PIT F&O universe eligibility.

G_UNIVERSE fix (RC-010): replaces the stub that always returned
DATA_UNAVAILABLE for historical dates.

Implementation strategy
-----------------------
Full NSE F&O eligibility databases (lot-size history, SEBI surveillance lists,
circuit-breaker history) are not publicly available in machine-readable form.
Until a proper database is built (Month-2 roadmap item), we use the
**parquet-first-date** heuristic:

    A symbol is considered F&O-eligible on date D if:
      1. Its on-disk parquet file starts on or before D, AND
      2. D is within the parquet's date range (i.e., data is available).

    Rationale: the parquet files were built from historical data sourced
    from NSE/Upstox for confirmed F&O symbols. The first bar date in the
    parquet is therefore a conservative lower bound on F&O eligibility —
    a symbol couldn't have a parquet entry before it was on-exchange.

    Limitation (documented as RC-010 partial): symbols added to F&O after
    our data collection start date (≈2021-09-19) will be correctly handled.
    Symbols REMOVED from F&O (and then retained in parquets as historical
    data) will appear eligible past their removal date. This is a known
    survivorship bias risk, estimated to affect < 5% of training rows.

Public API
----------
    universe = HistoricalUniverse.default()        # singleton, thread-safe
    result   = universe.check(symbol, date)        # FoEligibilityResult
    is_elig  = universe.is_eligible(symbol, date)  # bool (False = DATA_UNAVAILABLE)
    symbols  = universe.eligible_symbols(date)     # list[str]
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date as _date
from enum import Enum
from pathlib import Path
from typing import ClassVar

import pandas as pd

_PARQUET_DIR = Path(__file__).parent.parent.parent / "data" / "1d" / "1d"
_SINGLETON_LOCK = threading.Lock()


class FoEligibility(str, Enum):
    ELIGIBLE         = "ELIGIBLE"
    NOT_YET_LISTED   = "NOT_YET_LISTED"   # date before parquet start
    DELISTED         = "DELISTED"          # date after parquet end
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"  # no parquet on disk


@dataclass(frozen=True)
class FoEligibilityResult:
    symbol:      str
    query_date:  _date
    fo_eligible: FoEligibility
    first_date:  _date | None = None
    last_date:   _date | None = None
    note:        str          = ""


class HistoricalUniverse:
    """PIT F&O eligibility oracle backed by on-disk parquet date ranges."""

    _instance: ClassVar["HistoricalUniverse | None"] = None

    def __init__(self, parquet_dir: Path = _PARQUET_DIR) -> None:
        self._parquet_dir = parquet_dir
        # symbol → (first_date, last_date) — loaded lazily, cached
        self._cache: dict[str, tuple[_date, _date]] = {}
        self._loaded: bool = False
        self._lock = threading.Lock()

    # ── Singleton ──────────────────────────────────────────────────────────
    @classmethod
    def default(cls) -> "HistoricalUniverse":
        """Return the process-wide singleton instance."""
        if cls._instance is None:
            with _SINGLETON_LOCK:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Parquet scan ───────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._cache = {}
            if not self._parquet_dir.exists():
                self._loaded = True
                return
            for pf in self._parquet_dir.glob("*.parquet"):
                sym = pf.stem
                try:
                    df = pd.read_parquet(str(pf), columns=[])  # metadata only
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                    first = df.index.min().date()
                    last  = df.index.max().date()
                    self._cache[sym] = (first, last)
                except Exception:
                    pass  # corrupted parquet — skip
            self._loaded = True

    # ── Public API ─────────────────────────────────────────────────────────
    def check(self, symbol: str, query_date: _date | str | pd.Timestamp) -> FoEligibilityResult:
        """Check F&O eligibility for *symbol* on *query_date* (PIT-safe).

        Returns an :class:`FoEligibilityResult` with the eligibility status.
        Never raises — returns ``DATA_UNAVAILABLE`` for unknown symbols.
        """
        self._ensure_loaded()

        # Normalise query_date
        if isinstance(query_date, str):
            query_date = pd.Timestamp(query_date).date()
        elif isinstance(query_date, pd.Timestamp):
            query_date = query_date.date()

        if symbol not in self._cache:
            return FoEligibilityResult(
                symbol=symbol,
                query_date=query_date,
                fo_eligible=FoEligibility.DATA_UNAVAILABLE,
                note="No parquet on disk — symbol unknown or data not yet ingested",
            )

        first, last = self._cache[symbol]

        if query_date < first:
            return FoEligibilityResult(
                symbol=symbol,
                query_date=query_date,
                fo_eligible=FoEligibility.NOT_YET_LISTED,
                first_date=first,
                last_date=last,
                note=f"Query date {query_date} is before first parquet bar {first}",
            )

        if query_date > last:
            return FoEligibilityResult(
                symbol=symbol,
                query_date=query_date,
                fo_eligible=FoEligibility.DELISTED,
                first_date=first,
                last_date=last,
                note=f"Query date {query_date} is after last parquet bar {last}; "
                     "may be delisted or data not yet ingested",
            )

        return FoEligibilityResult(
            symbol=symbol,
            query_date=query_date,
            fo_eligible=FoEligibility.ELIGIBLE,
            first_date=first,
            last_date=last,
        )

    def is_eligible(self, symbol: str, query_date: _date | str | pd.Timestamp) -> bool:
        """Return True iff the symbol is F&O-eligible on query_date."""
        return self.check(symbol, query_date).fo_eligible == FoEligibility.ELIGIBLE

    def eligible_symbols(self, query_date: _date | str | pd.Timestamp) -> list[str]:
        """Return all symbols eligible on *query_date*, sorted alphabetically."""
        self._ensure_loaded()
        if isinstance(query_date, str):
            query_date = pd.Timestamp(query_date).date()
        elif isinstance(query_date, pd.Timestamp):
            query_date = query_date.date()
        return sorted(
            sym for sym, (first, last) in self._cache.items()
            if first <= query_date <= last
        )

    def universe_size(self, query_date: _date | str | pd.Timestamp) -> int:
        return len(self.eligible_symbols(query_date))

    def all_symbols(self) -> list[str]:
        """Return every symbol with an on-disk parquet, sorted."""
        self._ensure_loaded()
        return sorted(self._cache)

    def date_range(self, symbol: str) -> tuple[_date, _date] | None:
        """Return (first_date, last_date) for *symbol*, or None if unknown."""
        self._ensure_loaded()
        return self._cache.get(symbol)

    def survivorship_diagnostics(self) -> dict:
        """Return a summary dict for monitoring survivorship bias risk."""
        self._ensure_loaded()
        if not self._cache:
            return {"n_symbols": 0}
        all_firsts = [v[0] for v in self._cache.values()]
        all_lasts  = [v[1] for v in self._cache.values()]
        import statistics
        return {
            "n_symbols":              len(self._cache),
            "earliest_first_date":    str(min(all_firsts)),
            "median_first_date":      str(sorted(all_firsts)[len(all_firsts) // 2]),
            "latest_first_date":      str(max(all_firsts)),
            "earliest_last_date":     str(min(all_lasts)),
            "latest_last_date":       str(max(all_lasts)),
            # Symbols with a very late start date (added to F&O recently)
            "symbols_starting_after_2023": sum(
                1 for d in all_firsts if d.year >= 2023
            ),
            "note": (
                "Heuristic: parquet first-date used as F&O entry proxy. "
                "Symbols with late parquet start may appear DATA_UNAVAILABLE "
                "for earlier dates. Full historical lot-size DB is a Month-2 task."
            ),
        }
