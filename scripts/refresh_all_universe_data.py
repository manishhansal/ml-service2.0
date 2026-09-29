#!/usr/bin/env python3
"""
refresh_all_universe_data.py — Comprehensive data refresh for all 218 F&O symbols.

Root causes of data staleness (Sep 2026):
  1. Upstox access token expired Sep 19 → CDN fallback returns only top-tier stocks
  2. CATCHUP_EOD_HOUR_IST=17 → EOD pass ran before Upstox CDN settled (now fixed to 19)
  3. backfill_india_1y.py --class ALL only covers Nifty50 (~58 symbols), not full F&O

This script:
  Step 1: Runs data-service backfill for ALL 218 F&O symbols individually
          (via docker exec backfill_india_1y.py --symbol X --class EQ)
          Symbols with Angel One coverage get fresh Sep 24-27 data.
          Symbols without Angel One coverage get whatever Upstox CDN has.

  Step 2: Calls /v1/india/historical for all 218 symbols and pulls fresh bars
          into local parquets.

  Step 3: Runs phantom bar deduplication on any newly added parquets.

  Step 4: Reports final data freshness status.

Usage:
    PYTHONPATH=. python3 scripts/refresh_all_universe_data.py
    # Or via Makefile:
    make refresh-data

Note on Upstox token:
    If the Upstox access token is expired, symbols not covered by Angel One
    will not get fresh data until the token is refreshed manually via OAuth.
    Run 'make check-token' to see token status.
    To refresh: log into https://api.upstox.com/v2/login and update
    UPSTOX_ACCESS_TOKEN in data-service2.0/.env, then restart data-service.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

PARQUET_DIR = Path("data/1d/1d")
env = {k.strip(): v.strip() for line in Path(".env").read_text().splitlines()
       if "=" in line and not line.strip().startswith("#")
       for k, _, v in [line.partition("=")]}
DATA_KEY = env.get("DATA_SERVICE_API_KEY", "")
DATA_URL  = env.get("DATA_SERVICE_2_URL", "http://localhost:8200")
CONTAINER = "data-service-api"


def check_token_status() -> dict:
    """Check if Upstox token is valid and report status."""
    import base64
    token = env.get("UPSTOX_ACCESS_TOKEN", "")
    if not token:
        return {"valid": False, "reason": "UPSTOX_ACCESS_TOKEN not set in .env"}
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp", 0)
        now = datetime.now(tz=timezone.utc).timestamp()
        if now > exp:
            expired_by = datetime.fromtimestamp(exp, tz=timezone.utc)
            return {"valid": False, "reason": f"EXPIRED since {expired_by.strftime('%Y-%m-%d %H:%M')} UTC",
                    "exp": exp}
        return {"valid": True, "expires": datetime.fromtimestamp(exp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    except Exception as e:
        return {"valid": False, "reason": f"could not decode: {e}"}


def get_all_symbols() -> list[str]:
    """Get all 218 symbols from local parquets."""
    return sorted([p.stem for p in PARQUET_DIR.glob("*.parquet")])


def backfill_symbol_via_docker(sym: str, lookback_days: int = 15) -> bool:
    """
    Trigger data-service backfill for a single symbol via docker exec.
    Returns True if at least 1 candle was stored.
    """
    try:
        result = subprocess.run(
            ["docker", "exec", CONTAINER, "python3",
             "/app/scripts/backfill_india_1y.py",
             "--symbol", sym, "--class", "EQ",
             "--intervals", "1d", f"--lookback-days={lookback_days}"],
            capture_output=True, text=True, timeout=30,
        )
        return "candles=" in result.stdout and "OK" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def get_historical_from_api(sym: str) -> list[dict]:
    """Fetch historical bars from data-service for a symbol."""
    url = f"{DATA_URL}/v1/india/historical?symbol={sym}&interval=1d"
    req = urllib.request.Request(url, headers={"X-API-KEY": DATA_KEY})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read().decode())
            return d.get("data", []) if isinstance(d, dict) else (d if isinstance(d, list) else [])
    except Exception:
        return []


def append_to_parquet(sym: str, bars: list[dict]) -> int:
    """Append new bars to parquet, return count added."""
    if not bars:
        return 0
    pf = PARQUET_DIR / f"{sym}.parquet"
    if not pf.exists():
        return 0
    try:
        df_new = pd.DataFrame(bars)
        if "time" in df_new.columns:
            df_new.index = pd.to_datetime(df_new["time"], unit="s", utc=True)
            df_new = df_new.drop(columns=["time"])
        for c in ("open","high","low","close","volume"):
            if c in df_new.columns:
                df_new[c] = pd.to_numeric(df_new[c], errors="coerce")
        df_new = df_new.dropna(subset=["close"])
        df_ex = pd.read_parquet(pf)
        if df_ex.index.tz is None:
            df_ex.index = df_ex.index.tz_localize("UTC")
        new = df_new[df_new.index > df_ex.index.max()]
        if new.empty:
            return 0
        common = [c for c in df_ex.columns if c in new.columns]
        pd.concat([df_ex[common], new[common]]).to_parquet(pf)
        return len(new)
    except Exception:
        return 0


def fix_phantom_bars(sym: str) -> int:
    """Remove any 03:45 UTC phantom bars that crept in with the new data."""
    pf = PARQUET_DIR / f"{sym}.parquet"
    try:
        df = pd.read_parquet(pf)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        n_before = len(df)
        df_clean = df[df.index.hour != 3]
        df_clean = df_clean[~df_clean.index.normalize().duplicated(keep="last")]
        n_removed = n_before - len(df_clean)
        if n_removed > 0:
            df_clean.to_parquet(pf)
        return n_removed
    except Exception:
        return 0


def main():
    print("=" * 70)
    print("  FULL UNIVERSE DATA REFRESH")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
    print("=" * 70)

    # --- Token check ---
    token_status = check_token_status()
    print(f"\n  Upstox token: {'✓ VALID' if token_status['valid'] else '✗ ' + token_status['reason']}")
    if not token_status["valid"]:
        print("  ⚠  Without a valid token, only Angel One-covered symbols will get fresh data.")
        print("  ⚠  Refresh: log into https://api.upstox.com/v2/login → update UPSTOX_ACCESS_TOKEN in data-service2.0/.env")

    # --- Get all symbols ---
    symbols = get_all_symbols()
    print(f"\n  Universe: {len(symbols)} symbols\n")

    # --- Step 1: Backfill via data-service for all symbols ---
    print("STEP 1: Backfill via data-service (Angel One + Upstox CDN)")
    print("-" * 60)
    backfill_ok = 0
    for i, sym in enumerate(symbols, 1):
        ok = backfill_symbol_via_docker(sym)
        if ok:
            backfill_ok += 1
            print(f"  [{i:3d}/{len(symbols)}] {sym:20s} ✓ backfilled")
        else:
            if i % 20 == 0:
                print(f"  [{i:3d}/{len(symbols)}] ... ({backfill_ok} backfilled so far)")
        time.sleep(0.1)

    print(f"\n  Backfilled: {backfill_ok}/{len(symbols)}")

    # --- Step 2: Pull fresh bars from data-service into parquets ---
    print("\nSTEP 2: Pull fresh bars into local parquets")
    print("-" * 60)
    total_new = 0
    updated = []
    for i, sym in enumerate(symbols, 1):
        bars = get_historical_from_api(sym)
        added = append_to_parquet(sym, bars)
        if added:
            total_new += added
            updated.append((sym, added))
        time.sleep(0.1)

    print(f"  Added: {total_new} bars across {len(updated)} symbols")
    for sym, n in updated[:10]:
        df = pd.read_parquet(PARQUET_DIR / f"{sym}.parquet")
        print(f"    {sym:20s}: +{n} bars  → {df.index.max().date()}")
    if len(updated) > 10:
        print(f"    ... and {len(updated)-10} more")

    # --- Step 3: Phantom bar cleanup ---
    print("\nSTEP 3: Phantom bar cleanup on updated parquets")
    print("-" * 60)
    total_phantom = 0
    for sym, _ in updated:
        removed = fix_phantom_bars(sym)
        if removed:
            total_phantom += removed
            print(f"  {sym:20s}: removed {removed} phantom rows")
    print(f"  Total phantom rows removed: {total_phantom}")

    # --- Step 4: Final status report ---
    print("\nSTEP 4: Final data freshness report")
    print("-" * 60)
    from collections import Counter
    cnt = Counter()
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        cnt[str(df.index.max().date())] += 1

    for date in sorted(cnt.keys(), reverse=True)[:6]:
        bar = "#" * (cnt[date] // 5)
        print(f"  {date}: {cnt[date]:3d} symbols  {bar}")

    fresh = sum(v for k, v in cnt.items() if k >= "2026-09-24")
    stale = sum(v for k, v in cnt.items() if k < "2026-09-24")
    print(f"\n  Fresh (Sep 24+): {fresh}/218  Stale: {stale}/218")
    print(f"\n  ✓ Refresh complete. Run 'make session' to start today's session.")

    if stale > 50:
        print(f"\n  ⚠  {stale} symbols still stale — refresh Upstox token and re-run.")
        print("     See: https://api.upstox.com/v2/login")


if __name__ == "__main__":
    main()
