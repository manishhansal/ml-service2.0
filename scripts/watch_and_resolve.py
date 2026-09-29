#!/usr/bin/env python3
"""
Watches wall-clock time and automatically resolves forward paper signals
when their resolve_after timestamp passes. Runs the full resolution +
promotion evaluation cycle.

Usage:
    python3 scripts/watch_and_resolve.py

Keeps running until all signals are fully resolved (or stopped manually).
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SIGNALS_PATH  = Path("artifacts/forward_paper/signals.jsonl")
OUTCOMES_PATH = Path("artifacts/forward_paper/outcomes.jsonl")
CHECK_INTERVAL = 300   # check every 5 minutes


def load_signals():
    return [json.loads(l) for l in SIGNALS_PATH.read_text().splitlines() if l.strip()]


def load_resolved_ids():
    if not OUTCOMES_PATH.exists():
        return set()
    return {json.loads(l)["signal_id"] for l in OUTCOMES_PATH.read_text().splitlines() if l.strip()}


def run_resolve():
    result = subprocess.run(
        ["python3", "scripts/resolve_forward_paper.py"],
        capture_output=True, text=True
    )
    print(result.stdout[-1000:])
    if result.returncode != 0:
        print("[watch] resolve script error:", result.stderr[-500:])


def run_promotion():
    result = subprocess.run(
        ["python3", "scripts/run_signal_promotion.py"],
        capture_output=True, text=True
    )
    print(result.stdout[-800:])


def main():
    print(f"[watch] Starting forward paper watcher — {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    signals = load_signals()
    all_resolve_dates = sorted({datetime.fromisoformat(s["resolve_after"]) for s in signals})
    print(f"[watch] {len(signals)} signals across {len(all_resolve_dates)} batches")
    for d in all_resolve_dates:
        diff = (d - datetime.now(tz=timezone.utc)).total_seconds() / 3600
        print(f"  Batch: {d.strftime('%Y-%m-%d %H:%M UTC')} ({diff:+.1f}h)")

    while True:
        now = datetime.now(tz=timezone.utc)
        signals = load_signals()
        resolved_ids = load_resolved_ids()

        due = [s for s in signals
               if datetime.fromisoformat(s["resolve_after"]) <= now
               and s["signal_id"] not in resolved_ids]
        total_unresolved = len([s for s in signals if s["signal_id"] not in resolved_ids])

        print(f"\n[watch] {now.strftime('%H:%M UTC')} — {len(resolved_ids)} resolved, {len(due)} due, {total_unresolved} unresolved")

        if due:
            print(f"[watch] Running resolution for {len(due)} due signals...")
            run_resolve()

            resolved_ids_new = load_resolved_ids()
            newly_resolved = len(resolved_ids_new) - len(resolved_ids)
            print(f"[watch] +{newly_resolved} newly resolved → total {len(resolved_ids_new)}")

            if len(resolved_ids_new) >= 10:
                print("[watch] Running SignalPromotionEngine...")
                run_promotion()

        # Check if all done
        resolved_ids = load_resolved_ids()
        remaining = [s for s in signals if s["signal_id"] not in resolved_ids]
        pending_dates = [datetime.fromisoformat(s["resolve_after"]) for s in remaining
                         if datetime.fromisoformat(s["resolve_after"]) > now]
        if not pending_dates:
            print("[watch] All signals resolved. Exiting.")
            break

        nxt = min(pending_dates)
        wait_secs = max(CHECK_INTERVAL, (nxt - now).total_seconds() - 60)
        wake = datetime.now(tz=timezone.utc).fromtimestamp(
            datetime.now(tz=timezone.utc).timestamp() + wait_secs, tz=timezone.utc)
        print(f"[watch] Next check at {wake.strftime('%H:%M UTC')} (sleeping {wait_secs:.0f}s)")
        time.sleep(wait_secs)


if __name__ == "__main__":
    main()
