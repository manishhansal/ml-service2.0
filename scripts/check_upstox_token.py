#!/usr/bin/env python3
"""
check_upstox_token.py — Check Upstox token status and data-service data freshness.

Run this before every market session to ensure data is fresh.
A stale or expired Upstox token means 154/218 symbols return outdated data.

Usage:
    PYTHONPATH=. python3 scripts/check_upstox_token.py
    # Or:
    make check-token
"""
from __future__ import annotations

import base64, json, sys, datetime
from pathlib import Path

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def check():
    env_path = Path("/Users/manishkumar/Desktop/data-service2.0/.env")
    if not env_path.exists():
        print(f"{RED}data-service2.0/.env not found{RESET}")
        return

    env = {k.strip(): v.strip() for line in env_path.read_text().splitlines()
           if "=" in line and not line.strip().startswith("#")
           for k, _, v in [line.partition("=")]}

    token = env.get("UPSTOX_ACCESS_TOKEN", "")
    now = datetime.datetime.now(tz=datetime.timezone.utc)

    print("=" * 60)
    print("  UPSTOX TOKEN + DATA FRESHNESS CHECK")
    print("=" * 60)

    # --- Token check ---
    if not token:
        print(f"\n  {RED}UPSTOX_ACCESS_TOKEN: NOT SET{RESET}")
        print("  Action: Set in data-service2.0/.env after OAuth flow")
        return

    try:
        parts = token.split(".")
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp_ts  = payload.get("exp", 0)
        iat_ts  = payload.get("iat", 0)
        exp_dt  = datetime.datetime.fromtimestamp(exp_ts, tz=datetime.timezone.utc)
        iat_dt  = datetime.datetime.fromtimestamp(iat_ts, tz=datetime.timezone.utc)

        if now > exp_dt:
            expired_ago = now - exp_dt
            print(f"\n  Upstox token:  {RED}EXPIRED{RESET} {expired_ago.days}d {expired_ago.seconds//3600}h ago")
            print(f"    Issued:  {iat_dt.strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"    Expired: {exp_dt.strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"    Impact:  {RED}154/218 symbols return stale Sep 24 data{RESET}")
            print(f"\n  {YELLOW}Fix:{RESET}")
            print("    1. Open https://api.upstox.com/v2/login in browser")
            print("    2. Complete OAuth flow → copy the new access_token")
            print("    3. Update UPSTOX_ACCESS_TOKEN in data-service2.0/.env")
            print("    4. Run: cd ../data-service2.0 && docker compose restart api worker scheduler")
            print("    5. Run: make refresh-data")
            token_ok = False
        else:
            remaining = exp_dt - now
            print(f"\n  Upstox token:  {GREEN}VALID{RESET} (expires in {remaining.seconds//3600}h {(remaining.seconds%3600)//60}m)")
            print(f"    Issued:  {iat_dt.strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"    Expires: {exp_dt.strftime('%Y-%m-%d %H:%M UTC')}")
            token_ok = True
    except Exception as e:
        print(f"\n  {RED}Could not decode token: {e}{RESET}")
        token_ok = False

    # --- Data freshness check ---
    import pandas as pd
    from collections import Counter
    from pathlib import Path as P
    PARQUET_DIR = P("/Users/manishkumar/Desktop/ml-service2.0/data/1d/1d")
    pqs = list(PARQUET_DIR.glob("*.parquet"))
    cnt = Counter()
    for p in pqs:
        df = pd.read_parquet(p)
        cnt[str(df.index.max().date())] += 1

    print(f"\n  Data freshness ({len(pqs)} symbols):")
    today = now.strftime("%Y-%m-%d")
    for date in sorted(cnt.keys(), reverse=True)[:5]:
        age_days = (now - datetime.datetime.fromisoformat(date + "T00:00:00+00:00")).days
        color = GREEN if age_days <= 2 else (YELLOW if age_days <= 5 else RED)
        bar = "#" * (cnt[date] // 5)
        print(f"    {color}{date}{RESET} ({age_days}d old): {cnt[date]:3d} symbols  {bar}")

    fresh = sum(v for k, v in cnt.items() if k >= "2026-09-24")
    stale = sum(v for k, v in cnt.items() if k < "2026-09-24")
    print(f"\n  Fresh: {GREEN}{fresh}/218{RESET}  Stale: {RED if stale>50 else YELLOW}{stale}/218{RESET}")

    # --- Recommendation ---
    print()
    if stale > 100 and not token_ok:
        print(f"  {RED}⚠  CRITICAL: {stale} stale symbols + expired token{RESET}")
        print("    Signals for stale symbols may be wrong direction!")
        print("    → Refresh Upstox token ASAP, then run: make refresh-data")
    elif stale > 50:
        print(f"  {YELLOW}⚠  WARNING: {stale} stale symbols{RESET}")
        print("    → Run: make refresh-data")
    elif stale > 0:
        print(f"  {YELLOW}Note: {stale} symbols behind by >4 days — run 'make refresh-data'{RESET}")
    else:
        print(f"  {GREEN}✓ All symbols fresh — ready for live session{RESET}")

    print()

if __name__ == "__main__":
    check()
