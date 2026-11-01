"""
eod_refresh.py — End-of-day data refresh and SMC scanner reset.

Polls Upstox until today's candle is available, then:
  1. Refreshes all F&O parquets with today's close
  2. Clears the SMC scanner Redis cache
  3. Runs the forward paper session resolver
  4. Prints a summary

Run after market close (15:30 IST). Data usually available by 19:00 IST.
Usage:
    python scripts/eod_refresh.py
    python scripts/eod_refresh.py --date 2026-10-12  # specific date
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))
PROJECT = Path(__file__).parent.parent

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_token() -> str:
    for line in Path("/Users/manishkumar/Desktop/data-service2.0/.env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            if k.strip() == "UPSTOX_ACCESS_TOKEN":
                tok = v.strip()
                if tok:
                    return tok
    raise SystemExit("UPSTOX_ACCESS_TOKEN not found")

def _test_candle_available(date_str: str, token: str) -> bool:
    """Return True if Upstox has 1d candle data for date_str (RELIANCE as probe)."""
    url = (
        "https://api.upstox.com/v2/historical-candle/"
        + urllib.parse.quote("NSE_EQ|INE002A01018", safe="")
        + f"/day/{date_str}/{date_str}"
    )
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept":        "application/json",
        "User-Agent":    "Mozilla/5.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        candles = data.get("data", {}).get("candles", [])
        return len(candles) > 0
    except Exception:
        return False

def _clear_smc_cache() -> bool:
    """Clear SMC scanner Redis cache so next scan uses fresh data."""
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        deleted = r.delete("smc:active", "smc:last_scan_meta")
        print(f"  Cleared {deleted} Redis keys (smc:active, smc:last_scan_meta)")
        return True
    except Exception as e:
        print(f"  Redis clear failed: {e} (manual: docker exec ml-service20-redis-1 redis-cli DEL smc:active)")
        return False

# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--no-wait", action="store_true", help="Don't poll — fail immediately if no data")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else datetime.now(IST).date()
    target_str  = str(target_date)
    token       = _load_token()

    print(f"EOD Refresh — {target_str}")
    print("=" * 60)

    # ── Step 1: Poll until data available ─────────────────────────────────────
    if not args.no_wait:
        print(f"\n[1/3] Waiting for Upstox to publish {target_str} data...")
        max_wait = 3 * 3600   # max 3 hours
        poll_interval = 300   # 5 min
        waited = 0
        while not _test_candle_available(target_str, token):
            now_ist = datetime.now(IST).strftime("%H:%M IST")
            print(f"  {now_ist}: data not yet available — retrying in {poll_interval//60} min...")
            if args.no_wait or waited >= max_wait:
                print("  Max wait exceeded — exiting.")
                return
            time.sleep(poll_interval)
            waited += poll_interval
        print(f"  Data available at {datetime.now(IST).strftime('%H:%M IST')}")
    else:
        if not _test_candle_available(target_str, token):
            print(f"No data available for {target_str} yet. Re-run after 19:00 IST.")
            return
        print(f"[1/3] Data available for {target_str}")

    # ── Step 2: Refresh parquets ───────────────────────────────────────────────
    print(f"\n[2/3] Refreshing parquets for {target_str}...")
    import subprocess
    result = subprocess.run(
        [str(PROJECT / ".venv/bin/python"), str(PROJECT / "scripts/refresh_parquets.py"),
         "--date", target_str],
        capture_output=True, text=True, cwd=str(PROJECT)
    )
    if result.returncode == 0:
        # Count updated
        lines = result.stdout.splitlines()
        for line in lines:
            if "Updated=" in line or "Total updated" in line:
                print(f"  {line.strip()}")
    else:
        print(f"  FAILED: {result.stderr[-200:]}")
        return

    # ── Step 3: Clear SMC scanner cache ───────────────────────────────────────
    print("\n[3/3] Clearing SMC scanner cache (forces rescan with new data)...")
    _clear_smc_cache()

    print()
    print("=" * 60)
    print(f"EOD refresh complete for {target_str}")
    print(f"Next steps:")
    print(f"  - Run monday_signals.py for tomorrow's signals")
    print(f"  - SMC scanner will auto-refresh at next market open")
    print("=" * 60)

if __name__ == "__main__":
    main()
