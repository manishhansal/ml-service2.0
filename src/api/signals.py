"""
Live signal endpoints for ml-service2.0 → AlphaForge UI.

Endpoints:
    GET /v2/signals/latest   → LatestSignalsResponse   — most recent scoring snapshot
    GET /v2/signals/session  → SessionSummaryResponse  — full session log + all samples
    GET /v2/signals/history  → list[ForecastRecord]    — today's ForecastLedger records

Data source
-----------
These endpoints are read-only consumers of files written by
``scripts/autorun_till_close.py``:

  • ``artifacts/live_session/latest_scores.json``  — atomic snapshot, updated every 5 min
  • ``artifacts/live_session/session_summary.json``— full session (written at EOD close)
  • ``artifacts/forward_paper/forecasts.jsonl``    — ForecastLedger (all 218 symbols)

When no live session is running the endpoints return the most-recently-saved
snapshot (or a structured "no data" response) so the UI can always render
something meaningful.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["signals"])

# ── File paths (relative to repo root) ───────────────────────────────────────
_BASE = Path(__file__).parent.parent.parent  # repo root
_LATEST_PATH = _BASE / "artifacts" / "live_session" / "latest_scores.json"
_SESSION_PATH = _BASE / "artifacts" / "live_session" / "session_summary.json"
_FORECASTS_PATH = _BASE / "artifacts" / "forward_paper" / "forecasts.jsonl"
_AUTORUN_LOG = _BASE / "artifacts" / "live_session" / "autorun_log.jsonl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except Exception:
        return None


def _no_session_response() -> dict[str, Any]:
    """Structured placeholder returned when no live session has run today."""
    return {
        "session_date": None,
        "generated_at": None,
        "market_open": False,
        "model_version": None,
        "n_scored": 0,
        "n_long": 0,
        "n_short": 0,
        "nifty_chg": None,
        "nifty_ltp": None,
        "session_pnl": {"mean_net": None, "win_rate": None, "n_positions": 0},
        "signals": [],
        "stale": True,
        "message": "No live session snapshot found. Run `make session` to start autorun.",
    }


def _today_ist() -> str:
    """Return today's date string in IST (UTC+5:30)."""
    from datetime import timedelta
    ist = datetime.now(tz=timezone.utc) + timedelta(hours=5, minutes=30)
    return ist.strftime("%Y-%m-%d")


def _is_stale(snapshot: dict[str, Any]) -> bool:
    """A snapshot is stale if it was generated more than 15 minutes ago."""
    gen = snapshot.get("generated_at")
    if not gen:
        return True
    try:
        ts = datetime.fromisoformat(gen)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_mins = (datetime.now(tz=timezone.utc) - ts).total_seconds() / 60
        return age_mins > 15
    except Exception:
        return True


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/signals/latest")
async def get_latest_signals() -> JSONResponse:
    """Most recent LightGBM scoring snapshot — all 218 F&O symbols.

    Returns the content of ``artifacts/live_session/latest_scores.json``
    written by ``autorun_till_close.py`` every 5 minutes during a live
    session.  If no session has run today, returns a structured
    "no data" payload (HTTP 200, ``stale: true``) so the UI can render a
    meaningful empty state.

    Response shape
    ~~~~~~~~~~~~~~
    ::

        {
          "session_date": "2026-09-30",          // YYYY-MM-DD
          "generated_at": "2026-09-30T09:15:00Z",
          "market_open":  true,
          "model_version": "fs-2.0.0",
          "n_scored":  218,
          "n_long":     64,
          "n_short":   154,
          "nifty_chg": -0.42,
          "nifty_ltp": 22910.5,
          "session_pnl": {
            "mean_net": 0.8, "win_rate": 80.0, "n_positions": 26
          },
          "stale": false,
          "signals": [
            {
              "symbol":    "SBIN",
              "score":     0.0312,
              "direction": -1,       // +1 = LONG, -1 = SHORT
              "data_date": "2026-09-29",
              "rank":       1,
              "conviction": "S",     // S/A/B/C/D by distance from 0.5
              "live_pnl":  { "net_pct": 1.24, "entry": 994.1, "ltp": 982.7, ... }
            },
            ...
          ]
        }

    GET /v2/signals/latest
    """
    snapshot = _read_json(_LATEST_PATH)

    if snapshot is None:
        return JSONResponse(content=_no_session_response())

    # Normalize direction: 0 is a ForecastLedger sentinel; derive from score.
    for sig in snapshot.get("signals", []):
        raw = sig.get("direction", 0)
        if raw not in (1, -1):
            sig["direction"] = 1 if sig.get("score", 0.5) >= 0.5 else -1
        # Re-compute n_long/n_short after normalization
    sigs = snapshot.get("signals", [])
    snapshot["n_long"]  = sum(1 for s in sigs if s.get("direction") == 1)
    snapshot["n_short"] = sum(1 for s in sigs if s.get("direction") == -1)

    # Annotate staleness — useful for UI "last updated" badge
    snapshot["stale"] = _is_stale(snapshot)

    # Annotate whether the snapshot is from today's IST session
    today = _today_ist()
    snapshot["is_today"] = snapshot.get("session_date") == today

    return JSONResponse(content=snapshot)


@router.get("/signals/session")
async def get_session_summary() -> JSONResponse:
    """Full EOD session summary written at market close.

    Contains ``all_samples`` — the complete time-series of per-tick records —
    and the aggregated ``final_pnl``.  Only available after 15:30 IST.
    Before close, returns ``available: false``.

    GET /v2/signals/session
    """
    summary = _read_json(_SESSION_PATH)

    if summary is None:
        return JSONResponse(
            content={
                "available": False,
                "message": "Session summary written at market close (15:30 IST). Not yet available.",
            }
        )

    today = _today_ist()
    summary["is_today"] = summary.get("session_date") == today
    return JSONResponse(content=summary)


@router.get("/signals/history")
async def get_signal_history(
    date: str | None = None,
    limit: int = 218,
    status: str | None = None,
) -> JSONResponse:
    """ForecastLedger records for a session date.

    When ``date`` is omitted, defaults to today IST.  If today has no
    records (e.g. before the first session of the day), automatically
    falls back to the most recent date that does have records.

    Optional query parameters:
      - ``date``   — YYYY-MM-DD (default: today IST, with automatic fallback)
      - ``limit``  — max records returned (default: 218)
      - ``status`` — filter: "open" | "won" | "lost"

    GET /v2/signals/history?date=2026-09-29&status=won
    """
    if not _FORECASTS_PATH.exists():
        return JSONResponse(content={"records": [], "n": 0, "date": date or _today_ist(),
                                     "stats": {}, "is_fallback": False})

    requested_date = date or _today_ist()

    try:
        # ── Single-pass read: group all records by session_date ────────
        by_date: dict[str, list[dict]] = {}
        for line in _FORECASTS_PATH.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                d = rec.get("session_date", "")
                if d:
                    by_date.setdefault(d, []).append(rec)
            except Exception:
                continue

        # Auto-fallback: if requested date has no records, use latest available
        target_date  = requested_date
        is_fallback  = False
        if not by_date.get(target_date) and date is None and by_date:
            target_date = max(by_date.keys())
            is_fallback = (target_date != requested_date)

        STATUS_RANK = {"won": 3, "lost": 3, "open": 2, "superseded": 1}

        # ── Deduplicate: best status per symbol, then latest ts ────────
        seen: dict[str, dict] = {}
        for rec in by_date.get(target_date, []):
            sym = rec.get("symbol", "")
            if not sym:
                continue
            existing = seen.get(sym)
            if existing is None:
                seen[sym] = rec
                continue
            cur_rank = STATUS_RANK.get(rec.get("status", ""), 0)
            ex_rank  = STATUS_RANK.get(existing.get("status", ""), 0)
            if cur_rank > ex_rank:
                seen[sym] = rec
            elif cur_rank == ex_rank and rec.get("ts", "") > existing.get("ts", ""):
                seen[sym] = rec

        # ── Filter by status, sort, limit ──────────────────────────────
        deduped_all = list(seen.values())
        if status:
            deduped_all = [r for r in deduped_all if r.get("status") == status]
        deduped_all.sort(key=lambda r: abs(r.get("score", 0.5) - 0.5), reverse=True)
        deduped = deduped_all[:limit]

        # ── Session stats ──────────────────────────────────────────────
        resolved = [r for r in deduped if r.get("net_pct") is not None]
        stats: dict = {
            "n_total":    len(deduped),
            "n_resolved": len(resolved),
            "n_open":     sum(1 for r in deduped if r.get("status") == "open"),
            "n_won":      sum(1 for r in deduped if r.get("status") == "won"),
            "n_lost":     sum(1 for r in deduped if r.get("status") == "lost"),
        }
        if resolved:
            net_pcts = [r["net_pct"] for r in resolved]
            stats["win_rate"]          = round(sum(1 for p in net_pcts if p > 0) / len(net_pcts) * 100, 1)
            stats["mean_net"]          = round(sum(net_pcts) / len(net_pcts), 4)
            stats["best_net"]          = round(max(net_pcts), 4)
            stats["worst_net"]         = round(min(net_pcts), 4)
            bd_vals = [r.get("brier_delta") for r in resolved if r.get("brier_delta") is not None]
            stats["brier_delta_mean"]  = round(sum(bd_vals) / len(bd_vals), 6) if bd_vals else None

        # List available session dates for the date picker
        available_dates = sorted(by_date.keys(), reverse=True)

        return JSONResponse(content={
            "date":            target_date,
            "requested_date":  requested_date,
            "is_fallback":     is_fallback,
            "n":               len(deduped),
            "stats":           stats,
            "records":         deduped,
            "available_dates": available_dates,
        })
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "records": [], "n": 0, "date": requested_date},
        )

    target_date = date or _today_ist()

    try:
        # Read ALL records for the target date — limit applied AFTER deduplication.
        # The ForecastLedger appends records chronologically; resolved (won/lost)
        # records are written at EOD and sit at the end of the file.  Applying
        # limit during read would truncate before reaching them.
        all_records: list[dict] = []
        for line in _FORECASTS_PATH.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("session_date") != target_date:
                    continue
                all_records.append(rec)
            except Exception:
                continue

        # Deduplicate per symbol — precedence: won/lost > open > superseded.
        # When multiple records have the same status, keep the one with the
        # latest timestamp so re-scored sessions use the freshest data.
        STATUS_RANK = {"won": 3, "lost": 3, "open": 2, "superseded": 1}

        seen: dict[str, dict] = {}
        for rec in all_records:
            sym = rec.get("symbol", "")
            if not sym:
                continue
            existing = seen.get(sym)
            if existing is None:
                seen[sym] = rec
                continue
            # Prefer higher-ranked status; break ties by latest ts
            cur_rank = STATUS_RANK.get(rec.get("status", ""), 0)
            ex_rank  = STATUS_RANK.get(existing.get("status", ""), 0)
            if cur_rank > ex_rank:
                seen[sym] = rec
            elif cur_rank == ex_rank and rec.get("ts", "") > existing.get("ts", ""):
                seen[sym] = rec

        # Optional status filter applied AFTER deduplication
        deduped_all = list(seen.values())
        if status:
            deduped_all = [r for r in deduped_all if r.get("status") == status]

        # Sort by conviction (distance from 0.5), then apply limit
        return JSONResponse(content={
            "date":            target_date,
            "requested_date":  requested_date,
            "is_fallback":     is_fallback,
            "n":               len(deduped),
            "stats":           stats,
            "records":         deduped,
            "available_dates": available_dates,
        })
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "records": [], "n": 0, "date": requested_date},
        )


@router.get("/signals/autorun-log")
async def get_autorun_log(last_n: int = 10) -> JSONResponse:
    """Last N per-tick samples from the running session log.

    Useful for a mini time-series chart in the UI (NIFTY % change + model
    signal count over the session).  Returns up to ``last_n`` samples
    (default 10, max 100), most-recent first.

    GET /v2/signals/autorun-log?last_n=5
    """
    if not _AUTORUN_LOG.exists():
        return JSONResponse(content={"samples": [], "n": 0})

    try:
        lines = [l for l in _AUTORUN_LOG.read_text().splitlines() if l.strip()]
        tail = lines[-min(last_n, 100):]
        samples = []
        for line in reversed(tail):  # most-recent first
            try:
                samples.append(json.loads(line))
            except Exception:
                continue
        return JSONResponse(content={"samples": samples, "n": len(samples)})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc), "samples": []})
