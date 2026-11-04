#!/usr/bin/env python3
"""
_seed_forward_paper.py — Seed signals_v2.jsonl from latest_scores.json.

Takes the current scored signals, selects top/bottom 15% by score as LONG/SHORT,
and writes them to artifacts/forward_paper/signals_v2.jsonl so the next session's
autorun has forward paper positions to track live P&L against.

Safe to run multiple times (APPENDS new signals; existing symbols are skipped
so repeat runs on the same session_date are idempotent).

Usage:
    PYTHONPATH=. python3 scripts/_seed_forward_paper.py
    PYTHONPATH=. python3 scripts/_seed_forward_paper.py --top-pct 0.20
    PYTHONPATH=. python3 scripts/_seed_forward_paper.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE        = Path(__file__).parent.parent
SCORES_PATH = BASE / "artifacts/live_session/latest_scores.json"
SIGNAL_STORE = BASE / "artifacts/forward_paper/signals_v2.jsonl"

IST = timezone(timedelta(hours=5, minutes=30))
HORIZON_BARS = 5
COST_BPS     = 27.65


def _trading_days_ahead(from_date: datetime, n: int) -> datetime:
    """Approximate n trading days ahead (Mon-Fri, skips weekends)."""
    d = from_date
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-pct", type=float, default=0.15,
                        help="Top/bottom percentile to select as LONG/SHORT (default 0.15 = 15%%)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not SCORES_PATH.exists():
        print(f"ERROR: {SCORES_PATH} not found — run after autorun completes")
        return

    scores_doc = json.loads(SCORES_PATH.read_text())
    session_date = scores_doc.get("session_date", "")
    signals = scores_doc.get("signals", [])
    if not signals:
        print("No signals in latest_scores.json")
        return

    # Sort by raw score
    signals_sorted = sorted(signals, key=lambda s: s["score"])
    n = len(signals_sorted)
    cutoff = max(1, int(n * args.top_pct))

    short_candidates = signals_sorted[:cutoff]          # lowest scores = SHORT
    long_candidates  = signals_sorted[n - cutoff:]       # highest scores = LONG

    # Load existing signals to avoid duplicate entries for same session_date
    existing: set[str] = set()
    if SIGNAL_STORE.exists():
        for line in SIGNAL_STORE.read_text().splitlines():
            if line.strip():
                try:
                    s = json.loads(line)
                    if s.get("session_date") == session_date:
                        existing.add(s["symbol"])
                except Exception:
                    pass

    now_ist = datetime.now(IST)
    resolve_after = _trading_days_ahead(now_ist, HORIZON_BARS + 1).replace(
        hour=9, minute=15, second=0, microsecond=0
    ).isoformat()

    new_signals: list[dict] = []
    for s in short_candidates + long_candidates:
        sym = s["symbol"]
        direction = s["direction"]
        if direction == 0:
            continue
        if sym in existing:
            continue

        new_signals.append({
            "signal_id":       str(uuid.uuid4()),
            "symbol":          sym,
            "direction":       direction,
            "score":           s["score"],
            "conviction":      s.get("conviction", "D"),
            "signal_ts":       f"{session_date}T09:15:00+05:30",
            "session_date":    session_date,
            "created_at":      now_ist.isoformat(),
            "horizon_bars":    HORIZON_BARS,
            "resolve_after":   resolve_after,
            "cost_bps":        COST_BPS,
            "feature_schema":  scores_doc.get("model_version", "fs-2.0.0"),
            "status":          "NONE",
        })

    n_long  = sum(1 for s in new_signals if s["direction"] ==  1)
    n_short = sum(1 for s in new_signals if s["direction"] == -1)
    print(f"Session {session_date}: {len(signals)} scored → {n_long} LONG + {n_short} SHORT signals selected")
    print(f"Already in store for today: {len(existing)}")
    print(f"New signals to write: {len(new_signals)}")

    if args.dry_run:
        print("[DRY RUN] Not writing.")
        return

    if new_signals:
        with open(SIGNAL_STORE, "a") as f:
            for sig in new_signals:
                f.write(json.dumps(sig) + "\n")
        print(f"Appended {len(new_signals)} signals → {SIGNAL_STORE}")
    else:
        print("Nothing to write.")


if __name__ == "__main__":
    main()
