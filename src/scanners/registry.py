"""
registry.py — signal registry for the live SMC scanner.

Tracks when each signal first appeared, when it was last confirmed by a
re-scan, and when it closed (with the close reason).

Timestamps
----------
signal_timestamp — the actual market event time (when the zone/break formed).
                   Set from the scanner schema's native timestamp field:
                     order_block      → ob_timestamp (candle where OB formed)
                     fvg              → fvg_timestamp (3rd candle of the imbalance)
                     supply_demand    → departure_timestamp (impulsive departure candle)
                     liquidity_sweep  → sweep_timestamp (the sweep bar)
                     market_structure → break_timestamp (bar where BoS/ChoCh fired)
                   For daily-timeframe data, this is a UTC datetime whose IST
                   equivalent is the trading date — e.g. 2026-10-08T18:30:00+00:00
                   = 2026-10-09 00:00 IST = "Oct 9" candle.

first_seen_at   — wall-clock time when this signal was first inserted into Redis
                   (i.e. when the background scanner task first ran after deploy).
last_seen_at    — wall-clock time when the signal was last confirmed by a re-scan.
closed_at       — wall-clock time when the signal was marked closed.

Close reasons
-------------
expired     — signal vanished from scan results or proximity exceeded 15 %
invalidated — scanner marked the zone/signal as invalidated/broken/filled
eod         — end-of-day sweep (called at 15:30 IST to close all active signals)

Storage layout (Redis keys via RedisCache)
-----------------------------------------
smc:active            → list[SignalRecord]  TTL = 1 day
smc:closed:<YYYY-MM-DD> → list[SignalRecord]  TTL = 3 days
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)

IST            = timezone(timedelta(hours=5, minutes=30))
_ACTIVE_KEY    = "smc:active"
_CLOSED_PREFIX = "smc:closed:"
_ACTIVE_TTL    = 86_400        # 1 day
_CLOSED_TTL    = 3 * 86_400    # 3 days
MAX_PROX_PCT   = 3.0           # intraday only — signals >3% from entry are too far to trade same-day

# Which schema field holds the actual pattern-formation timestamp per scanner type
_SIGNAL_TS_FIELD: dict[str, str] = {
    "order_block":      "ob_timestamp",           # candle where the OB formed
    "fvg":              "fvg_timestamp",           # 3rd candle of the imbalance
    "supply_demand":    "departure_timestamp",     # impulsive departure candle
    "liquidity_sweep":  "sweep_timestamp",         # the sweep bar
    "market_structure": "break_timestamp",         # bar where BoS/ChoCh fired
}


# ── Signal ID ─────────────────────────────────────────────────────────────────

def make_signal_id(scanner_type: str, item: Any) -> str:
    """Return a 12-hex stable ID for a signal across multiple scan cycles."""
    sym = item.symbol
    dirn = getattr(item, "direction", None) or getattr(item, "signal_type", None)
    dirn_val = getattr(dirn, "value", str(dirn)) if dirn else "?"

    # Primary price level used to distinguish two separate zones of the same type
    level_map = {
        "order_block":       "ob_high",
        "fvg":               "fvg_high",
        "supply_demand":     "zone_high",
        "liquidity_sweep":   "swept_level",
        "market_structure":  "broken_level",
    }
    attr = level_map.get(scanner_type, "")
    level = round(getattr(item, attr, 0.0) or 0.0, 1)

    raw = f"{scanner_type}:{sym}:{dirn_val}:{level}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Item serialization ────────────────────────────────────────────────────────

def _item_to_record(scanner_type: str, item: Any, now_iso: str) -> dict:
    """Convert a scanner result object to a plain dict for Redis storage."""
    dirn = getattr(item, "direction", None) or getattr(item, "signal_type", None)
    dirn_val = getattr(dirn, "value", str(dirn)) if dirn else "?"

    lc = getattr(item, "lifecycle_state", None)
    lc_val = getattr(lc, "value", str(lc)) if lc is not None else None

    # Capture the actual market event time (when the pattern formed),
    # not the wall-clock time the scanner ran.
    ts_field = _SIGNAL_TS_FIELD.get(scanner_type, "")
    raw_ts   = getattr(item, ts_field, None)
    signal_ts = (
        raw_ts.isoformat() if hasattr(raw_ts, "isoformat")
        else (str(raw_ts) if raw_ts else now_iso)
    )

    return {
        "id":               make_signal_id(scanner_type, item),
        "scanner_type":     scanner_type,
        "symbol":           item.symbol,
        "direction":        dirn_val,
        "entry":            getattr(item, "intraday_entry_price",  None),
        "target":           getattr(item, "intraday_target_price", None),
        "stop":             getattr(item, "intraday_stop_price",   None),
        "rr":               getattr(item, "intraday_rr_ratio",     None),
        "proximity_pct":    getattr(item, "proximity_pct",         None),
        "is_actionable":    getattr(item, "is_actionable",         False),
        "is_choch":         getattr(item, "is_choch",              False),
        "lifecycle_state":  lc_val,
        "invalidated":      getattr(item, "invalidated",           False),
        "conviction_score": getattr(item, "conviction_score",      None),
        # ── Timestamps ────────────────────────────────────────────────────────
        "signal_timestamp": signal_ts,   # actual candle date (UTC ISO-8601)
        "first_seen_at":    now_iso,     # when this scanner run inserted it
        "last_seen_at":     now_iso,     # updated on every re-scan
        "closed_at":        None,
        "close_reason":     None,
    }


def _update_record(rec: dict, item: Any, now_iso: str) -> dict:
    """Refresh mutable fields of an existing record from the latest scan item."""
    updated = dict(rec)
    updated["last_seen_at"]  = now_iso
    updated["proximity_pct"] = getattr(item, "proximity_pct", rec.get("proximity_pct"))
    updated["is_actionable"] = getattr(item, "is_actionable", rec.get("is_actionable", False))

    lc = getattr(item, "lifecycle_state", None)
    lc_val = getattr(lc, "value", str(lc)) if lc is not None else rec.get("lifecycle_state")
    updated["lifecycle_state"] = lc_val
    updated["invalidated"] = getattr(item, "invalidated", rec.get("invalidated", False))

    # Back-fill signal_timestamp if this is an old record that lacks it
    if not rec.get("signal_timestamp"):
        scanner_type = rec.get("scanner_type", "")
        ts_field = _SIGNAL_TS_FIELD.get(scanner_type, "")
        raw_ts   = getattr(item, ts_field, None)
        if hasattr(raw_ts, "isoformat"):
            updated["signal_timestamp"] = raw_ts.isoformat()
        elif raw_ts:
            updated["signal_timestamp"] = str(raw_ts)

    return updated


# ── Registry operations ───────────────────────────────────────────────────────

async def record_scan(
    scanner_type: str,
    items: list[Any],
    cache: Any,                        # RedisCache instance
) -> dict:
    """
    Upsert signals from one scanner run into the registry.

    - New signal      → added with first_seen_at = now
    - Existing signal → last_seen_at updated; proximity_pct refreshed
    - Active signal not in scan → closed as "expired"

    Returns a summary dict {new, updated, expired}.
    """
    now     = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    today   = now.astimezone(IST).date().isoformat()

    # Load active signals for this scanner type
    all_active: list[dict] = await cache.get(_ACTIVE_KEY) or []
    this_type  = {r["id"]: r for r in all_active if r.get("scanner_type") == scanner_type}
    other_type = [r for r in all_active if r.get("scanner_type") != scanner_type]

    # Filter scan items to reasonable proximity
    valid_items = [
        i for i in items
        if (getattr(i, "proximity_pct", 999.0) or 999.0) <= MAX_PROX_PCT
        and not getattr(i, "invalidated", False)
    ]

    # Build new snapshot
    new_snapshot: dict[str, dict] = {}
    n_new = n_updated = 0
    for item in valid_items:
        sid = make_signal_id(scanner_type, item)
        if sid in this_type:
            new_snapshot[sid] = _update_record(this_type[sid], item, now_iso)
            n_updated += 1
        else:
            new_snapshot[sid] = _item_to_record(scanner_type, item, now_iso)
            n_new += 1

    # Close signals that disappeared from the scan
    n_expired = 0
    closed_today: list[dict] = await cache.get(f"{_CLOSED_PREFIX}{today}") or []
    for sid, rec in this_type.items():
        if sid not in new_snapshot:
            closed_rec = dict(rec)
            closed_rec["closed_at"]   = now_iso
            closed_rec["close_reason"] = "expired"
            closed_today.append(closed_rec)
            n_expired += 1

    # Persist
    updated_active = other_type + list(new_snapshot.values())
    await cache.set(_ACTIVE_KEY, updated_active, ttl=_ACTIVE_TTL)
    if n_expired:
        await cache.set(f"{_CLOSED_PREFIX}{today}", closed_today, ttl=_CLOSED_TTL)

    logger.info(
        "smc_registry_updated: type=%s new=%d updated=%d expired=%d total_active=%d",
        scanner_type, n_new, n_updated, n_expired, len(updated_active),
    )
    return {"new": n_new, "updated": n_updated, "expired": n_expired}


async def clear_for_new_day(cache: Any) -> int:
    """
    Clear all active signals at market open (9:15 IST).

    Called once each morning to discard overnight stale signals so the day
    starts fresh.  Any active signals are moved to the previous day's closed
    list with close_reason = "overnight".

    Returns number of signals cleared.
    """
    now     = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    # Signals are cleared just before market opens — attribute them to yesterday
    yesterday = (now.astimezone(IST).date() - timedelta(days=1)).isoformat()

    all_active: list[dict] = await cache.get(_ACTIVE_KEY) or []
    if not all_active:
        return 0

    prev_day_closed: list[dict] = await cache.get(f"{_CLOSED_PREFIX}{yesterday}") or []
    for rec in all_active:
        closed_rec = dict(rec)
        closed_rec["closed_at"]    = now_iso
        closed_rec["close_reason"] = "overnight"
        prev_day_closed.append(closed_rec)

    await cache.set(_ACTIVE_KEY, [], ttl=_ACTIVE_TTL)
    await cache.set(f"{_CLOSED_PREFIX}{yesterday}", prev_day_closed, ttl=_CLOSED_TTL)
    # Also delete the scan meta so the UI shows no stale "last scan" time
    await cache.delete("smc:last_scan_meta")
    return len(all_active)


async def close_all_active(cache: Any, reason: str = "eod") -> int:
    """
    Close all currently active signals.  Used at end-of-day.
    Returns number of signals closed.
    """
    now     = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    today   = now.astimezone(IST).date().isoformat()

    all_active: list[dict] = await cache.get(_ACTIVE_KEY) or []
    if not all_active:
        return 0

    closed_today: list[dict] = await cache.get(f"{_CLOSED_PREFIX}{today}") or []
    for rec in all_active:
        closed_rec = dict(rec)
        closed_rec["closed_at"]    = now_iso
        closed_rec["close_reason"] = reason
        closed_today.append(closed_rec)

    await cache.set(_ACTIVE_KEY, [], ttl=_ACTIVE_TTL)
    await cache.set(f"{_CLOSED_PREFIX}{today}", closed_today, ttl=_CLOSED_TTL)
    return len(all_active)


async def get_active(cache: Any) -> list[dict]:
    """
    Return all active signals sorted by focus score (highest first).

    Focus score formula — designed so the most actionable, highest-conviction
    signals appear at the top of the scanner board:

      1. Actionable   (+1000)  — price within 2 % of entry RIGHT NOW
      2. ChoCh        (+400)   — change-of-character = trend reversal, highest conviction
      3. Scanner type (+150–80) — MS > Sweep > OB > S&D > FVG
      4. R:R × 30     (+30–120) — reward-to-risk quality
      5. Proximity    (+0–200)  — tighter proximity = more urgent (only when actionable)
    """
    raw = await cache.get(_ACTIVE_KEY) or []
    return sorted(raw, key=_focus_score, reverse=True)


_SCANNER_TYPE_WEIGHT: dict[str, float] = {
    "market_structure": 150.0,
    "liquidity_sweep":  130.0,
    "order_block":      110.0,
    "supply_demand":     90.0,
    "fvg":               80.0,
}


def _focus_score(sig: dict) -> float:
    """Compute a numeric priority score for a signal record (higher = more important)."""
    score = 0.0

    # ── Tier 1: Actionable — price is AT the entry level right now ─────────────
    if sig.get("is_actionable"):
        score += 1000.0

    # ── Tier 2: ChoCh signals indicate a full trend reversal ───────────────────
    if sig.get("is_choch"):
        score += 400.0

    # ── Tier 3: Scanner type weight ────────────────────────────────────────────
    score += _SCANNER_TYPE_WEIGHT.get(sig.get("scanner_type", ""), 0.0)

    # ── Tier 4: R:R ratio (quality of the setup) ───────────────────────────────
    rr = sig.get("rr") or 0.0
    score += float(rr) * 30.0

    # ── Tier 5: Proximity bonus (closer = more urgent; only when near entry) ────
    # IMPORTANT: use explicit None check — proximity_pct=0.0 is falsy in Python
    # so `sig.get("proximity_pct") or 999.0` incorrectly returns 999.0 for prox=0.
    prox_raw = sig.get("proximity_pct")
    prox = float(prox_raw) if prox_raw is not None else 999.0
    if prox <= 2.0:
        score += (2.0 - prox) * 100.0   # 0 % prox → +200, 2 % prox → +0
    elif prox <= 5.0:
        score += (5.0 - prox) * 10.0    # 2–5 % → small bonus

    return score


async def get_closed(cache: Any, date: str | None = None) -> list[dict]:
    """Return closed signals for *date* (ISO YYYY-MM-DD, default today IST)."""
    if date is None:
        date = datetime.now(timezone.utc).astimezone(IST).date().isoformat()
    raw = await cache.get(f"{_CLOSED_PREFIX}{date}") or []
    return sorted(raw, key=lambda x: x.get("closed_at", ""), reverse=True)


async def get_last_scan_meta(cache: Any) -> dict | None:
    """Return metadata about the most recent full scan run."""
    return await cache.get("smc:last_scan_meta")


async def set_last_scan_meta(cache: Any, meta: dict) -> None:
    """Persist metadata about the most recent full scan run (TTL: 1 hour)."""
    await cache.set("smc:last_scan_meta", meta, ttl=3600)
