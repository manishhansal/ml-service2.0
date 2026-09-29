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
    """Today's ForecastLedger records for all scored symbols.

    Optional query parameters:
      - ``date``   — YYYY-MM-DD to filter by session date (default: today IST)
      - ``limit``  — max records to return (default: 218)
      - ``status`` — filter by status: "open" | "won" | "lost" | "superseded"

    Each record has: id, ts, session_date, symbol, score, direction, est_prob,
    market_prior, data_date, nifty_chg_at_record, model_version, status,
    and after resolution: realized, net_pct, resolved_at, brier_delta.

    GET /v2/signals/history?date=2026-09-30&status=won
    """
    if not _FORECASTS_PATH.exists():
        return JSONResponse(content={"records": [], "n": 0, "date": date})

    target_date = date or _today_ist()

    try:
        records = []
        for line in _FORECASTS_PATH.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("session_date") != target_date:
                    continue
                if status and rec.get("status") != status:
                    continue
                records.append(rec)
                if len(records) >= limit:
                    break
            except Exception:
                continue

        # Deduplicate per symbol — keep latest non-superseded record
        seen: dict[str, dict] = {}
        for rec in records:
            sym = rec["symbol"]
            if rec.get("status") == "superseded":
                if sym not in seen:
                    seen[sym] = rec  # keep if nothing better yet
            else:
                seen[sym] = rec  # non-superseded always wins

        deduped = sorted(seen.values(), key=lambda r: abs(r.get("score", 0.5) - 0.5), reverse=True)

        return JSONResponse(content={
            "date": target_date,
            "n": len(deduped),
            "records": deduped,
        })
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "records": [], "n": 0},
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
