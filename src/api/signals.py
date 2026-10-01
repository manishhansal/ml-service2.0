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
    """A snapshot is stale if it was generated more than 2 minutes ago.

    Threshold lowered from 15 min → 2 min because the async refactor now
    runs a 30-second cycle; anything older than 2 minutes indicates the
    autorun has stopped or the data-service is unavailable.
    """
    gen = snapshot.get("generated_at")
    if not gen:
        return True
    try:
        ts = datetime.fromisoformat(gen)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_secs = (datetime.now(tz=timezone.utc) - ts).total_seconds()
        return age_secs > 120   # 2 minutes
    except Exception:
        return True


def _age_seconds(snapshot: dict[str, Any]) -> float | None:
    """Return how many seconds ago the snapshot was generated, or None."""
    gen = snapshot.get("generated_at")
    if not gen:
        return None
    try:
        ts = datetime.fromisoformat(gen)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(tz=timezone.utc) - ts).total_seconds()
    except Exception:
        return None


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

    # ── Direction handling ────────────────────────────────────────────────────
    # direction=0 is NOT a sentinel — it is a deliberate "no signal" state set
    # by the pipeline (weight manager threshold, sector_dimmed, stock_dampened
    # etc.).  The old normalization loop coerced all direction=0 to LONG/SHORT
    # using score >= 0.5, which is why AlphaForge showed 241 LONG / 44 SHORT
    # instead of the correct 40 LONG / 35 SHORT.
    #
    # Fix:
    #   1. Only fill direction when the key is genuinely absent (pre-pipeline
    #      ForecastLedger records that never ran through the full signal engine).
    #   2. Strip direction=0 signals from the response — the UI only renders
    #      actionable LONG/SHORT signals; neutral signals add noise.
    #   3. Use the file's top-level n_long / n_short (computed correctly by
    #      autorun) rather than recomputing from the mutated signal list.
    #   4. When market_open=False (post-close EOD snapshot) the post-close code
    #      calls score_all() WITHOUT the full pipeline, so direction is the raw
    #      score threshold (score>0.5→LONG). These are forward-looking scores
    #      for TOMORROW, not actionable intraday signals. Set all to direction=0
    #      so the UI shows "Market Closed / No active signals" correctly.
    all_signals: list[dict] = snapshot.get("signals", [])
    market_open: bool = snapshot.get("market_open", True)

    if not market_open:
        # Post-close snapshot: raw EOD scores without pipeline, not actionable.
        # Neutralize all directions so AlphaForge shows market-closed state.
        for sig in all_signals:
            sig["direction"] = 0
        snapshot["signals"]   = []   # no active signals when market is closed
        snapshot["n_long"]    = 0
        snapshot["n_short"]   = 0
        snapshot["n_neutral"] = len(all_signals)
        snapshot["stale"]     = False   # not stale — just market closed
        snapshot["age_seconds"] = _age_seconds(snapshot)
        today = _today_ist()
        snapshot["is_today"]  = snapshot.get("session_date") == today
        return JSONResponse(content=snapshot)

    # Step 1: fill missing direction for genuinely un-scored entries only
    for sig in all_signals:
        if "direction" not in sig:
            sig["direction"] = 1 if sig.get("score", 0.5) >= 0.5 else -1

    # Step 2: only expose actionable signals (direction ±1) to the UI.
    # Neutral signals (direction=0) carry reasons (filter_reason, sector_dimmed,
    # stock_dampened etc.) that are logged for diagnostics but are not trades.
    actionable = [s for s in all_signals if s.get("direction") in (1, -1)]
    snapshot["signals"] = actionable

    # Step 3: n_long / n_short from file are already correct; recompute only
    # to stay consistent with the filtered list we just built.
    snapshot["n_long"]  = sum(1 for s in actionable if s["direction"] == 1)
    snapshot["n_short"] = sum(1 for s in actionable if s["direction"] == -1)
    snapshot["n_neutral"] = len(all_signals) - len(actionable)   # informational

    # Annotate staleness — useful for UI "last updated" badge
    snapshot["stale"] = _is_stale(snapshot)
    snapshot["age_seconds"] = _age_seconds(snapshot)   # handy for UI "N secs ago"

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


@router.get("/signals/movers")
async def get_movers(limit: int = 10) -> JSONResponse:
    """F&O top gainers and losers from the latest live-quote snapshot.

    Reads ``artifacts/live_session/live_quotes.json`` (written by the autorun
    every 30 s) and returns the top-N gainers and top-N losers ranked by
    intraday ``changePct``.

    Query parameters:
      - ``limit`` — number of gainers *and* losers to return (default: 10)

    Response shape::

        {
          "gainers": [{"symbol": "KOTAKBANK", "ltp": 426.0, "changePct": 2.1}, ...],
          "losers":  [{"symbol": "M&M",       "ltp": 2840.0, "changePct": -3.4}, ...],
          "nifty":   {"ltp": 22466.0, "changePct": -0.87},
          "generated_at": "...",
          "age_seconds": 12.4,
          "stale": false
        }

    GET /v2/signals/movers
    """
    _LQ_PATH = _BASE / "artifacts" / "live_session" / "live_quotes.json"
    lq = _read_json(_LQ_PATH)

    if lq is None:
        return JSONResponse(content={
            "gainers": [], "losers": [], "nifty": None,
            "stale": True, "message": "Live quotes not yet available.",
        })

    quotes = lq.get("quotes", {})
    movers: list[dict] = []

    for sym, q in quotes.items():
        if not isinstance(q, dict):
            continue
        ltp = q.get("ltp")
        chg = q.get("changePct")
        if ltp is None or chg is None:
            continue
        try:
            movers.append({
                "symbol":    sym,
                "ltp":       float(ltp),
                "changePct": float(chg),
            })
        except (TypeError, ValueError):
            continue

    movers.sort(key=lambda x: x["changePct"], reverse=True)
    gainers = movers[:limit]
    losers  = list(reversed(movers[-limit:])) if len(movers) >= limit else list(reversed(movers))

    nifty_q = quotes.get("NIFTY")
    nifty   = (
        {"ltp": float(nifty_q.get("ltp", 0)), "changePct": float(nifty_q.get("changePct", 0))}
        if isinstance(nifty_q, dict) and nifty_q.get("ltp") else None
    )

    age = _age_seconds(lq)

    return JSONResponse(content={
        "gainers":      gainers,
        "losers":       losers,
        "nifty":        nifty,
        "generated_at": lq.get("generated_at"),
        "age_seconds":  age,
        "stale":        age is None or age > 120,
    })


@router.get("/signals/history")
async def get_signal_history(
    date: str | None = None,
    limit: int = 500,
    status: str | None = None,
) -> JSONResponse:
    """ForecastLedger records for a session date.

    When ``date`` is omitted, defaults to today IST.  If today has no
    records (e.g. before the first session of the day), automatically
    falls back to the most recent date that does have records.

    Optional query parameters:
      - ``date``   — YYYY-MM-DD (default: today IST, with automatic fallback)
      - ``limit``  — max records returned (default: 500, raised from 218 to
                     cover the full 285-symbol F&O universe)
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

        # ── Filter by status, sort, limit ──────────────────────────────────
        deduped_all = list(seen.values())
        if status:
            deduped_all = [r for r in deduped_all if r.get("status") == status]

        # Sort: resolved (won/lost) FIRST so they always appear before the limit
        # cutoff regardless of conviction. Within each group, sort by conviction.
        # Bug fix: old sort by conviction alone put 25/26 resolved records at
        # ranks 215-285 — below limit=218 — so History showed Won 0, Lost 1.
        deduped_all.sort(
            key=lambda r: (
                0 if r.get("status") in ("won", "lost") else 1,
                -abs(r.get("score", 0.5) - 0.5),
            )
        )

        # Stats over FULL deduped universe (before limit) so header totals are
        # always accurate regardless of how many records the UI page requests.
        all_resolved = [r for r in deduped_all if r.get("net_pct") is not None]
        stats: dict = {
            "n_total":    len(deduped_all),
            "n_resolved": len(all_resolved),
            "n_open":     sum(1 for r in deduped_all if r.get("status") == "open"),
            "n_won":      sum(1 for r in deduped_all if r.get("status") == "won"),
            "n_lost":     sum(1 for r in deduped_all if r.get("status") == "lost"),
        }
        if all_resolved:
            net_pcts = [r["net_pct"] for r in all_resolved]
            stats["win_rate"]         = round(sum(1 for p in net_pcts if p > 0) / len(net_pcts) * 100, 1)
            stats["mean_net"]         = round(sum(net_pcts) / len(net_pcts), 4)
            stats["best_net"]         = round(max(net_pcts), 4)
            stats["worst_net"]        = round(min(net_pcts), 4)
            bd_vals = [r.get("brier_delta") for r in all_resolved if r.get("brier_delta") is not None]
            stats["brier_delta_mean"] = round(sum(bd_vals) / len(bd_vals), 6) if bd_vals else None

        # Apply limit AFTER stats. Raised 218→500 to cover 285-symbol universe.
        deduped = deduped_all[:limit]

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
