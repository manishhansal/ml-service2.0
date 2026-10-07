#!/usr/bin/env python3
"""
session_watchdog.py — Keeps autorun_till_close.py running until market close.

Monitors the latest_scores.json snapshot age. If it gets stale (>10 min
without an update), restarts autorun_till_close.py.

Automatically generates the final gap analysis report at market close.

Usage:
    PYTHONPATH=. python3 scripts/session_watchdog.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE        = Path(__file__).parent.parent
SCORES_PATH = BASE / "artifacts" / "live_session" / "latest_scores.json"
REPORTS_DIR = BASE / "reports" / "live"   # score_threshold_sweep + gap analysis land here
TRACKER_DIR = BASE / "artifacts" / "signal_tracker"
AUTORUN_CMD  = [str(BASE / ".venv/bin/python"), "-W", "ignore",
                str(BASE / "scripts/autorun_till_close.py")]
TRACKER_CMD  = [str(BASE / ".venv/bin/python"), "-W", "ignore",
                str(BASE / "scripts/signal_tracker.py")]

_autorun_proc:  subprocess.Popen | None = None
_tracker_proc:  subprocess.Popen | None = None


def ist_now() -> datetime:
    return datetime.now(tz=timezone.utc) + timedelta(hours=5, minutes=30)


def market_open() -> bool:
    now = ist_now()
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return (9, 15) <= t <= (15, 31)


def mins_to_close() -> float:
    now = ist_now()
    close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return max(0, (close - now).total_seconds() / 60)


def snapshot_age_mins() -> float:
    if not SCORES_PATH.exists():
        return 999
    snap = json.loads(SCORES_PATH.read_text())
    gen = snap.get("generated_at", "")
    if not gen:
        return 999
    ts = datetime.fromisoformat(gen)
    return (datetime.now(tz=timezone.utc) - ts).total_seconds() / 60


def autorun_alive() -> bool:
    global _autorun_proc
    if _autorun_proc is None:
        return False
    return _autorun_proc.poll() is None


def start_autorun() -> None:
    global _autorun_proc
    print(f"[watchdog] Starting autorun_till_close.py at {ist_now().strftime('%H:%M IST')}")
    env = {**os.environ, "PYTHONPATH": str(BASE)}
    _autorun_proc = subprocess.Popen(
        AUTORUN_CMD,
        cwd=str(BASE),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(f"[watchdog] PID={_autorun_proc.pid}")


def start_tracker() -> None:
    """Launch signal_tracker.py so tracker data is available for the EOD gap report."""
    global _tracker_proc
    if _tracker_proc is not None and _tracker_proc.poll() is None:
        return  # already running
    tracker_script = BASE / "scripts" / "signal_tracker.py"
    if not tracker_script.exists():
        print("[watchdog] signal_tracker.py not found — skipping tracker launch")
        return
    env = {**os.environ, "PYTHONPATH": str(BASE)}
    _tracker_proc = subprocess.Popen(
        TRACKER_CMD,
        cwd=str(BASE),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[watchdog] signal_tracker.py started (PID={_tracker_proc.pid})")


def generate_final_report() -> None:
    """Trigger the signal tracker to generate the final report."""
    session_date = ist_now().strftime("%Y-%m-%d")
    tracker_file = TRACKER_DIR / f"{session_date}.jsonl"
    if not tracker_file.exists():
        print("[watchdog] No tracker data — skipping report")
        return
    print(f"[watchdog] Generating final gap analysis report...")
    env = {**os.environ, "PYTHONPATH": str(BASE)}
    result = subprocess.run(
        [str(BASE / ".venv/bin/python"), "-c", f"""
import sys
sys.path.insert(0, '{BASE}')
from scripts.signal_tracker import generate_gap_report
from pathlib import Path
report = generate_gap_report('{session_date}')
p = Path('{REPORTS_DIR}/SIGNAL_GAP_ANALYSIS_{session_date}.md')
p.write_text(report)
print('Report saved to:', p)
print('Lines:', len(report.split('\\n')))
"""],
        cwd=str(BASE),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"[watchdog] Report error: {result.stderr[:200]}")


def _run_signal_ledger_safety(session_date: str) -> None:
    """ISSUE-15 safety net: run signal_ledger_daily.py at watchdog close.

    This ensures signal lifecycle data is persisted even if autorun_till_close.py
    crashed before reaching its post_close_signal_ledger() call.
    Idempotent: safe to run twice (will just find 0 positions to settle).
    """
    ledger_script = BASE / "scripts" / "signal_ledger_daily.py"
    if not ledger_script.exists():
        return
    print(f"[watchdog] Running signal ledger (safety net) for {session_date} ...")
    python_bin = str(BASE / ".venv/bin/python")
    if not os.path.exists(python_bin):
        python_bin = sys.executable
    env = {**os.environ, "PYTHONPATH": str(BASE)}
    result = subprocess.run(
        [python_bin, "-W", "ignore", str(ledger_script), "--date", session_date],
        cwd=str(BASE),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.stdout:
        # Print last 10 lines only
        lines = result.stdout.strip().splitlines()
        for line in lines[-10:]:
            print(f"[watchdog/ledger] {line}")
    if result.returncode != 0 and result.stderr:
        print(f"[watchdog] Signal ledger error: {result.stderr[:300]}")


def main():
    ist = ist_now()
    print("=" * 70)
    print(f"  SESSION WATCHDOG  |  {ist.strftime('%Y-%m-%d %H:%M IST')}")
    print(f"  Closes in: {mins_to_close():.0f} min")
    print("=" * 70)

    start_autorun()
    start_tracker()   # launch signal_tracker.py so EOD gap report has data
    close_reported = False
    session_date = ist_now().strftime("%Y-%m-%d")

    while True:
        now = ist_now()
        mins_left = mins_to_close()
        age = snapshot_age_mins()

        # Market closed
        if not market_open():
            print(f"\n[{now.strftime('%H:%M')}] Market CLOSED")
            if not close_reported:
                if _autorun_proc and _autorun_proc.poll() is None:
                    print("[watchdog] Waiting 3 min for autorun to finish EOD tasks...")
                    time.sleep(180)
                generate_final_report()
                # ISSUE-15 safety net: run signal_ledger_daily as watchdog-level fallback
                # in case autorun_till_close.py crashed before reaching post_close_signal_ledger()
                _run_signal_ledger_safety(session_date)
                close_reported = True
            break

        # Check if autorun needs restart — ISSUE-17: tightened from 12min/120s to 8min/60s
        if not autorun_alive() or age > 8:
            if not autorun_alive():
                print(f"[{now.strftime('%H:%M')}] Autorun died — restarting (snapshot age={age:.0f}min)")
            else:
                print(f"[{now.strftime('%H:%M')}] Snapshot stale ({age:.0f}min) — restarting autorun")
            if _autorun_proc and _autorun_proc.poll() is None:
                _autorun_proc.terminate()
                time.sleep(3)
            start_autorun()

        print(f"[{now.strftime('%H:%M')}] OK | snap_age={age:.0f}min | autorun={'alive' if autorun_alive() else 'dead'} | {mins_left:.0f}min left")
        time.sleep(60)   # ISSUE-17: check every 60s (was 120s)


if __name__ == "__main__":
    main()
