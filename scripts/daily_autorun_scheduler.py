#!/usr/bin/env python3
"""
daily_autorun_scheduler.py — Keeps session_watchdog running every trading day.

Runs forever:
  - On weekdays 09:00-15:35 IST: launches session_watchdog.py (which manages autorun)
  - At market close (15:35 IST): waits for watchdog to finish, then sleeps until next day
  - On weekends / holidays: sleeps until Monday 09:00 IST

Run once and leave it running:
    PYTHONPATH=. nohup python3 scripts/daily_autorun_scheduler.py &
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE = Path(__file__).parent.parent
WATCHDOG = BASE / "scripts" / "session_watchdog.py"
PYTHON = BASE / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

_proc: subprocess.Popen | None = None


def ist_now() -> datetime:
    return datetime.now(tz=timezone.utc) + timedelta(hours=5, minutes=30)


def market_open() -> bool:
    now = ist_now()
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return (9, 0) <= t <= (15, 35)


def secs_to_next_open() -> float:
    now = ist_now()
    candidate = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return (candidate - now).total_seconds()


def watchdog_alive() -> bool:
    return _proc is not None and _proc.poll() is None


def start_watchdog():
    global _proc
    ts = ist_now().strftime('%H:%M IST')
    print(f"[scheduler {ts}] Starting session_watchdog.py ...")
    env = {**os.environ, "PYTHONPATH": str(BASE)}
    _proc = subprocess.Popen(
        [str(PYTHON), "-W", "ignore", str(WATCHDOG)],
        cwd=str(BASE), env=env,
    )
    print(f"[scheduler] Watchdog PID={_proc.pid}")


def stop_watchdog():
    global _proc
    if watchdog_alive():
        print(f"[scheduler] Stopping watchdog PID={_proc.pid}")
        _proc.terminate()
        try:
            _proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


def check_upstox_token_expiry() -> None:
    """Warn if the Upstox token will expire within the next 12 hours."""
    try:
        import base64, json as _json
        env_path = BASE.parent / "data-service2.0" / ".env"
        if not env_path.exists():
            return
        for line in env_path.read_text().splitlines():
            if line.startswith("UPSTOX_ACCESS_TOKEN="):
                token = line.split("=", 1)[1].strip()
                parts = token.split(".")
                if len(parts) == 3:
                    pad = parts[1] + "=="
                    payload = _json.loads(base64.b64decode(pad))
                    from datetime import datetime as _dt
                    exp = _dt.fromtimestamp(payload["exp"])
                    hrs = (exp - _dt.now()).total_seconds() / 3600
                    if hrs < 12:
                        print(f"\n{'!'*60}")
                        print(f"  ⚠️  UPSTOX TOKEN EXPIRES IN {hrs:.1f}h ({exp.strftime('%Y-%m-%d %H:%M IST')})")
                        print("  Run: python3 scripts/refresh_upstox_token.py")
                        print(f"{'!'*60}\n")
                    else:
                        print(f"[scheduler] Upstox token valid for {hrs:.1f}h (expires {exp.strftime('%H:%M IST')})")
                break
    except Exception:
        pass  # non-critical — don't crash the scheduler


def main():
    print("=" * 60)
    print(f"  DAILY AUTORUN SCHEDULER  |  {ist_now().strftime('%Y-%m-%d %H:%M IST')}")
    print("  Launches session_watchdog.py every trading day.")
    print("=" * 60)

    check_upstox_token_expiry()

    while True:
        now = ist_now()

        if market_open():
            if not watchdog_alive():
                check_upstox_token_expiry()   # re-check at session start
                start_watchdog()
            else:
                pid_str = str(_proc.pid) if _proc else "?"
                print(f"[{now.strftime('%H:%M')}] Watchdog running (PID={pid_str}) OK")
            time.sleep(120)
        else:
            stop_watchdog()
            secs = secs_to_next_open()
            wake = ist_now() + timedelta(seconds=secs)
            wake_str = wake.strftime('%Y-%m-%d %H:%M IST')
            hours = secs / 3600
            print(f"[{now.strftime('%H:%M')}] Market closed. Sleeping {hours:.1f}h -> waking at {wake_str}")
            while secs > 0:
                chunk = min(300, secs)
                time.sleep(chunk)
                secs -= chunk
                if market_open():
                    break


if __name__ == "__main__":
    main()
