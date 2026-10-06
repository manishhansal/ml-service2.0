#!/usr/bin/env python3
"""
scripts/news_backfill_scheduler.py
--------------------------------------
Daemon that runs the daily SentinelPulse news backfill after market close
and triggers an automatic retrain once enough news history has accumulated.

Schedule
---------
  Daily at 16:00 IST (30 min after NSE market close at 15:30 IST):
    1. Run backfill_news_features.py  → appends one row per symbol
    2. Count total rows in MARKET.parquet (news history depth)
    3. If rows >= RETRAIN_THRESHOLD (90 rows = ~3 months) and no retrain
       in the last RETRAIN_COOLDOWN_DAYS days:
         → Run train_expanded_features.py (full retrain with news features)
         → Register new SHADOW model
    4. Log to logs/news_scheduler.log and artifacts/live_session/scheduler.jsonl

Safety
-------
  - Skips weekends and NSE holidays (from strategy/nse-calendar.json)
  - Retries backfill up to 3 times on failure (5-minute delay between retries)
  - Never blocks the main loop on a failed backfill; logs and continues
  - Lock file prevents concurrent runs (useful if started multiple times)

Usage
------
    # Run directly:
    PYTHONPATH=. python3 scripts/news_backfill_scheduler.py

    # Run in Docker (see docker-compose.yml news-scheduler service):
    docker-compose up news-scheduler

Environment variables
----------------------
    NEWS_BACKFILL_HOUR_IST   Hour to run backfill (default: 16)
    NEWS_BACKFILL_MIN_IST    Minute to run backfill (default: 0)
    RETRAIN_THRESHOLD        Min news rows to trigger retrain (default: 90)
    RETRAIN_COOLDOWN_DAYS    Min days between retrains (default: 30)
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta, date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────
BASE               = Path(__file__).parent.parent
LOG_FILE           = BASE / "logs" / "news_scheduler.log"
LEDGER_FILE        = BASE / "artifacts" / "live_session" / "news_scheduler.jsonl"
LOCK_FILE          = BASE / "logs" / ".news_scheduler.lock"
NEWS_DIR           = BASE / "data" / "news" / "1d"
CALENDAR_FILE      = BASE / "strategy" / "nse-calendar.json"

BACKFILL_HOUR_IST  = int(os.getenv("NEWS_BACKFILL_HOUR_IST", "16"))
BACKFILL_MIN_IST   = int(os.getenv("NEWS_BACKFILL_MIN_IST",  "0"))
RETRAIN_THRESHOLD  = int(os.getenv("RETRAIN_THRESHOLD",       "90"))
RETRAIN_COOLDOWN   = int(os.getenv("RETRAIN_COOLDOWN_DAYS",   "30"))
MAX_RETRIES        = 3
RETRY_SLEEP_SECS   = 300   # 5 minutes
LOOP_SLEEP_SECS    = 60    # poll every 60 seconds

IST = timezone(timedelta(hours=5, minutes=30))

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("news_scheduler")


# ── Helpers ───────────────────────────────────────────────────────────────────

def ist_now() -> datetime:
    return datetime.now(tz=IST)


def _nse_holidays() -> set[str]:
    """Load NSE holiday dates from the calendar file (YYYY-MM-DD strings)."""
    if not CALENDAR_FILE.exists():
        return set()
    try:
        cal = json.loads(CALENDAR_FILE.read_text())
        holidays: list[str] = []
        for entry in cal.get("events", []) if isinstance(cal, dict) else []:
            if isinstance(entry, dict) and entry.get("type") == "HOLIDAY":
                fire_at = entry.get("fire_at", "")
                if fire_at:
                    holidays.append(fire_at[:10])
        return set(holidays)
    except Exception:
        return set()


def is_trading_day(dt: datetime) -> bool:
    """Return True if dt falls on an NSE trading day (Mon-Fri, non-holiday)."""
    if dt.weekday() >= 5:   # Saturday=5, Sunday=6
        return False
    holidays = _nse_holidays()
    return dt.strftime("%Y-%m-%d") not in holidays


def _news_row_count() -> int:
    """Count rows in MARKET.parquet — one row per completed backfill day."""
    market_pf = NEWS_DIR / "MARKET.parquet"
    if not market_pf.exists():
        return 0
    try:
        import pandas as pd
        df = pd.read_parquet(market_pf)
        return len(df)
    except Exception:
        return 0


def _last_retrain_date() -> _date | None:
    """Return the date of the last scheduler-triggered retrain, or None."""
    if not LEDGER_FILE.exists():
        return None
    try:
        retrains = []
        for line in LEDGER_FILE.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("event") == "retrain_started":
                retrains.append(entry.get("date", ""))
        if not retrains:
            return None
        return _date.fromisoformat(max(retrains))
    except Exception:
        return None


def _log_event(event: str, **kwargs) -> None:
    """Append a structured event to the scheduler ledger."""
    record = {
        "event":     event,
        "timestamp": ist_now().isoformat(),
        "date":      ist_now().strftime("%Y-%m-%d"),
        **kwargs,
    }
    with LEDGER_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _run_backfill(today: str) -> bool:
    """Run backfill_news_features.py for today. Returns True on success."""
    cmd = [
        sys.executable,
        str(BASE / "scripts" / "backfill_news_features.py"),
        "--date", today,
    ]
    log.info(f"Running backfill for {today} ...")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE),
            capture_output=True,
            text=True,
            timeout=600,   # 10-minute hard limit
            env={**os.environ, "PYTHONPATH": str(BASE)},
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            # Find the "Done: saved=..." summary line
            summary = next((l for l in reversed(lines) if "Done:" in l), lines[-1] if lines else "")
            log.info(f"Backfill OK — {summary}")
            _log_event("backfill_ok", date=today, summary=summary)
            return True
        else:
            log.error(f"Backfill FAILED (exit {result.returncode}): {result.stderr[:500]}")
            _log_event("backfill_failed", date=today, exit_code=result.returncode,
                       stderr=result.stderr[:300])
            return False
    except subprocess.TimeoutExpired:
        log.error("Backfill TIMEOUT (10 min)")
        _log_event("backfill_timeout", date=today)
        return False
    except Exception as exc:
        log.error(f"Backfill ERROR: {exc}")
        _log_event("backfill_error", date=today, error=str(exc))
        return False


def _run_retrain() -> bool:
    """Run train_expanded_features.py. Returns True on success."""
    cmd = [
        sys.executable,
        str(BASE / "scripts" / "train_expanded_features.py"),
    ]
    log.info("Triggering retrain with news features ...")
    _log_event("retrain_started")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE),
            capture_output=True,
            text=True,
            timeout=7200,   # 2-hour hard limit for full training
            env={**os.environ, "PYTHONPATH": str(BASE), "OMP_NUM_THREADS": "1"},
        )
        if result.returncode == 0:
            log.info("Retrain completed successfully")
            _log_event("retrain_ok")
            # Run the register script to save the new model
            _register_new_model()
            return True
        else:
            log.error(f"Retrain FAILED (exit {result.returncode})")
            _log_event("retrain_failed", exit_code=result.returncode,
                       stderr=result.stderr[:300])
            return False
    except subprocess.TimeoutExpired:
        log.error("Retrain TIMEOUT (2 hr)")
        _log_event("retrain_timeout")
        return False
    except Exception as exc:
        log.error(f"Retrain ERROR: {exc}")
        _log_event("retrain_error", error=str(exc))
        return False


def _register_new_model() -> None:
    """Run register_fs4_model.py to register the freshly trained model."""
    cmd = [sys.executable, str(BASE / "scripts" / "register_fs4_model.py")]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE),
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "PYTHONPATH": str(BASE), "OMP_NUM_THREADS": "1"},
        )
        if result.returncode == 0:
            log.info("New model registered successfully")
            _log_event("model_registered")
        else:
            log.error(f"Model registration failed: {result.stderr[:200]}")
    except Exception as exc:
        log.error(f"Model registration error: {exc}")


# ── Main loop ─────────────────────────────────────────────────────────────────

_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    log.info(f"Signal {signum} received — shutting down gracefully")
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def main() -> None:
    log.info("=" * 60)
    log.info("  News Backfill Scheduler — starting")
    log.info(f"  Backfill time: {BACKFILL_HOUR_IST:02d}:{BACKFILL_MIN_IST:02d} IST")
    log.info(f"  Retrain threshold: {RETRAIN_THRESHOLD} rows")
    log.info(f"  Retrain cooldown:  {RETRAIN_COOLDOWN} days")
    log.info("=" * 60)

    # Lock file — prevent duplicate runs
    if LOCK_FILE.exists():
        pid_str = LOCK_FILE.read_text().strip()
        log.warning(f"Lock file exists (PID={pid_str}). Another instance may be running.")
        # Check if that PID is actually alive AND is this scheduler script
        is_live_scheduler = False
        try:
            pid_int = int(pid_str)
            os.kill(pid_int, 0)   # raises if dead
            # Verify the PID belongs to THIS script, not init/PID-1
            # In Docker containers PID=1 is always init — that's a stale lock.
            if pid_int == 1:
                log.info("Lock held by PID=1 (container init) — stale lock, removing.")
            else:
                cmdline_path = f"/proc/{pid_int}/cmdline"
                try:
                    import pathlib as _pl
                    cmd = _pl.Path(cmdline_path).read_text(errors="replace")
                    if "news_backfill_scheduler" in cmd:
                        is_live_scheduler = True
                except OSError:
                    pass  # /proc not available or process gone
                if is_live_scheduler:
                    log.error("Existing scheduler instance confirmed alive — exiting.")
                    sys.exit(1)
                else:
                    log.info("Lock PID is alive but not the scheduler — stale lock, removing.")
        except (ProcessLookupError, ValueError):
            log.info("Stale lock file (process gone) — removing.")
        LOCK_FILE.unlink(missing_ok=True)

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))

    _log_event("scheduler_started",
               backfill_time=f"{BACKFILL_HOUR_IST:02d}:{BACKFILL_MIN_IST:02d}",
               retrain_threshold=RETRAIN_THRESHOLD)

    last_run_date: _date | None = None

    try:
        while not _shutdown_requested:
            now = ist_now()
            today = now.date()

            # Check if we should run today
            should_run = (
                is_trading_day(now)
                and now.hour == BACKFILL_HOUR_IST
                and now.minute >= BACKFILL_MIN_IST
                and last_run_date != today
            )

            if should_run:
                log.info(f"[{now.strftime('%H:%M IST')}] Starting daily news backfill for {today}")
                last_run_date = today

                # ── Backfill with retry ───────────────────────────────────────
                success = False
                for attempt in range(1, MAX_RETRIES + 1):
                    success = _run_backfill(today.isoformat())
                    if success:
                        break
                    if attempt < MAX_RETRIES:
                        log.warning(f"Retry {attempt}/{MAX_RETRIES-1} in {RETRY_SLEEP_SECS//60}m ...")
                        time.sleep(RETRY_SLEEP_SECS)

                if not success:
                    log.error(f"All {MAX_RETRIES} backfill attempts failed for {today}")
                    # Continue to next day — don't block

                # ── Check retrain trigger ─────────────────────────────────────
                news_rows = _news_row_count()
                log.info(f"News history depth: {news_rows} rows (threshold: {RETRAIN_THRESHOLD})")

                if news_rows >= RETRAIN_THRESHOLD:
                    last_retrain = _last_retrain_date()
                    if last_retrain is None:
                        days_since = RETRAIN_COOLDOWN + 1  # force first retrain
                    else:
                        days_since = (today - last_retrain).days

                    if days_since >= RETRAIN_COOLDOWN:
                        log.info(
                            f"Retrain triggered: {news_rows} rows ≥ {RETRAIN_THRESHOLD}, "
                            f"{days_since}d since last retrain (cooldown={RETRAIN_COOLDOWN}d)"
                        )
                        _run_retrain()
                    else:
                        log.info(
                            f"Retrain suppressed: cooldown {days_since}/{RETRAIN_COOLDOWN}d"
                        )
                else:
                    rows_needed = RETRAIN_THRESHOLD - news_rows
                    log.info(
                        f"Retrain not yet due: need {rows_needed} more rows "
                        f"(~{rows_needed} trading days)"
                    )

            # ── Step 4: Daily options feature backfill at 15:45 IST ──────────
            # Options analytics (PCR, ATM IV) are only live — must run during/after market.
            OPTIONS_H, OPTIONS_M = 15, 45
            if (
                is_trading_day(now)
                and now.hour == OPTIONS_H
                and now.minute >= OPTIONS_M
                and last_run_date == today
            ):
                log.info("Running options feature backfill (15:45 IST) ...")
                try:
                    import subprocess as _sp
                    res = _sp.run(
                        [sys.executable,
                         str(BASE / "scripts" / "backfill_options_features.py"),
                         "--date", today.isoformat()],
                        cwd=str(BASE),
                        capture_output=True, text=True, timeout=300,
                        env={**os.environ, "PYTHONPATH": str(BASE)},
                    )
                    if res.returncode == 0:
                        log.info("Options backfill OK")
                        _log_event("options_backfill_ok", date=today.isoformat())
                    else:
                        log.warning(f"Options backfill failed: {res.stderr[:200]}")
                except Exception as exc:
                    log.warning(f"Options backfill error: {exc}")

            # Sleep until next check
            time.sleep(LOOP_SLEEP_SECS)

    finally:
        LOCK_FILE.unlink(missing_ok=True)
        _log_event("scheduler_stopped")
        log.info("News Backfill Scheduler stopped.")


if __name__ == "__main__":
    main()
