#!/usr/bin/env python3
"""
fast_backfill_recent.py — Parallel backfill of last N days for all F&O symbols.

Uses ThreadPoolExecutor to run docker exec backfill_india_1y.py in parallel
with a short lookback window. Designed to recover Sep 28-29 missing bars.

Usage:
    python3 scripts/fast_backfill_recent.py [--days 5] [--workers 8]
    make fast-backfill
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
PARQUET_DIR = ROOT / "data" / "1d" / "1d"
CONTAINER = "data-service-api"

# Read env
_env: dict[str, str] = {}
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            _env[k.strip()] = v.strip()

DS_ENV_FILE = Path("/Users/manishkumar/Desktop/data-service2.0/.env")
if DS_ENV_FILE.exists():
    for line in DS_ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            if k.strip() not in _env:
                _env[k.strip()] = v.strip()

DATA_URL = _env.get("DATA_SERVICE_2_URL", "http://localhost:8200")
DATA_KEY = _env.get("DATA_SERVICE_API_KEY", _env.get("CONSUMER_API_KEYS", "dev-key-local-1"))


# ---------------------------------------------------------------------------
# Token check
# ---------------------------------------------------------------------------

def check_token() -> dict:
    import base64
    token = _env.get("UPSTOX_ACCESS_TOKEN", "")
    if not token:
        return {"valid": False, "reason": "not set"}
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        exp = data.get("exp", 0)
        now = time.time()
        hrs = (exp - now) / 3600
        if now > exp:
            return {"valid": False, "reason": f"expired {abs(hrs):.1f} hrs ago", "hrs_remaining": hrs}
        return {"valid": True, "expires": datetime.fromtimestamp(exp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "hrs_remaining": hrs}
    except Exception as e:
        return {"valid": False, "reason": str(e)}


# ---------------------------------------------------------------------------
# Symbol list
# ---------------------------------------------------------------------------

def get_symbols() -> list[str]:
    """Get F&O universe from local parquets or cross-sectional cache."""
    cs = ROOT / "artifacts" / "cross_sectional" / "ohlcv_cache" / "universe_1d_1200d.parquet"
    if cs.exists():
        df = pd.read_parquet(cs)
        col = "symbol" if "symbol" in df.columns else "ticker"
        if col in df.columns:
            return sorted(df[col].unique().tolist())
    if PARQUET_DIR.exists():
        return sorted(p.stem for p in PARQUET_DIR.glob("*.parquet"))
    return []


# ---------------------------------------------------------------------------
# Backfill via docker exec
# ---------------------------------------------------------------------------

def backfill_one(sym: str, lookback_days: int) -> tuple[str, bool, str]:
    """
    Run backfill_india_1y.py for one symbol inside the data-service container.
    Returns (sym, success, detail).
    """
    try:
        result = subprocess.run(
            [
                "docker", "exec", CONTAINER,
                "python3", "/app/scripts/backfill_india_1y.py",
                "--symbol", sym,
                "--class", "EQ",
                "--intervals", "1d",
                f"--lookback-days={lookback_days}",
            ],
            capture_output=True, text=True, timeout=45,
        )
        stdout = result.stdout + result.stderr
        ok = result.returncode == 0 and ("candle" in stdout.lower() or "ok" in stdout.lower()
                                          or "stored" in stdout.lower() or "written" in stdout.lower()
                                          or "insert" in stdout.lower())
        # Extract candle count if present
        detail = ""
        for line in stdout.splitlines():
            if any(k in line.lower() for k in ("candle", "stored", "written", "insert", "ok", "error")):
                detail = line.strip()[-80:]
                break
        return sym, ok, detail
    except subprocess.TimeoutExpired:
        return sym, False, "TIMEOUT"
    except Exception as e:
        return sym, False, str(e)[:60]


# ---------------------------------------------------------------------------
# Fetch from API + append to parquet
# ---------------------------------------------------------------------------

def fetch_and_append(sym: str) -> tuple[str, int, str]:
    """
    Fetch recent bars from data-service API and append to local parquet.
    Returns (sym, bars_added, last_date).

    Bugs fixed:
    - URL-encodes symbols containing '&' (e.g. M&M, GVT&D) so they are not
      split into multiple query params by the HTTP layer.
    - Removes the over-broad phantom-bar filter (hour != 3) that incorrectly
      stripped legitimate NSE bars timestamped at 03:45 UTC (09:15 IST market
      open) for symbols whose data-service uses that convention.  The phantom
      bar filter belongs only in fix_phantom_bars.py (deduplication of bars
      with the SAME calendar date at both 18:30 UTC and 03:45 UTC).
    """
    import urllib.parse
    encoded_sym = urllib.parse.quote(sym, safe="")
    url = f"{DATA_URL}/v1/india/historical?symbol={encoded_sym}&interval=1d"
    req = urllib.request.Request(url, headers={"X-API-KEY": DATA_KEY})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = json.loads(r.read().decode())
            bars = raw.get("data", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    except Exception as e:
        return sym, 0, f"API error: {e}"

    if not bars:
        return sym, 0, "no bars"

    pf = PARQUET_DIR / f"{sym}.parquet"
    if not pf.exists():
        return sym, 0, "parquet missing"

    try:
        df_new = pd.DataFrame(bars)
        if "time" in df_new.columns:
            df_new.index = pd.to_datetime(df_new["time"], unit="s", utc=True)
            df_new = df_new.drop(columns=["time"])
        for c in ("open", "high", "low", "close", "volume"):
            if c in df_new.columns:
                df_new[c] = pd.to_numeric(df_new[c], errors="coerce")
        df_new = df_new.dropna(subset=["close"])

        df_ex = pd.read_parquet(pf)
        if df_ex.index.tz is None:
            df_ex.index = df_ex.index.tz_localize("UTC")

        cutoff = df_ex.index.max()
        new_rows = df_new[df_new.index > cutoff]
        if new_rows.empty:
            last = cutoff.strftime("%Y-%m-%d")
            return sym, 0, f"up-to-date ({last})"

        common = [c for c in df_ex.columns if c in new_rows.columns]
        merged = pd.concat([df_ex[common], new_rows[common]])
        # Deduplicate rows with the same calendar date (keep the most recent
        # timestamp per day).  Do NOT blanket-remove all hour==3 bars here —
        # that was the phantom-bar fix for a now-resolved ingestion bug, and it
        # incorrectly strips legitimate 03:45 UTC bars (09:15 IST market-open
        # timestamp) returned by data-service for some symbols.
        merged = merged[~merged.index.normalize().duplicated(keep="last")]
        merged.to_parquet(pf)

        last = merged.index.max().strftime("%Y-%m-%d")
        return sym, len(new_rows), last
    except Exception as e:
        return sym, 0, f"parquet error: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Fast parallel backfill for recent bars")
    ap.add_argument("--days", type=int, default=5, help="Lookback days (default 5)")
    ap.add_argument("--workers", type=int, default=8, help="Parallel docker exec workers (default 8)")
    ap.add_argument("--skip-backfill", action="store_true", help="Skip docker exec step, only pull from API")
    ap.add_argument("--skip-api", action="store_true", help="Skip API pull step")
    args = ap.parse_args()

    print("=" * 70)
    print("  FAST RECENT BACKFILL")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC  |  lookback={args.days}d  |  workers={args.workers}")
    print("=" * 70)

    # Token status
    tok = check_token()
    if tok["valid"]:
        print(f"\n  Token: VALID — {tok.get('hrs_remaining', 0):.1f} hrs remaining (expires {tok['expires']})")
    else:
        print(f"\n  Token: {tok['reason']}")
        print("  WARNING: Upstox backfill will fail. Only Angel One symbols will update.")

    symbols = get_symbols()
    print(f"  Universe: {len(symbols)} symbols\n")

    if not symbols:
        print("ERROR: no symbols found. Run from ml-service2.0 root.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 1: Parallel docker exec backfill
    # -----------------------------------------------------------------------
    if not args.skip_backfill:
        print(f"STEP 1: Parallel backfill via data-service ({args.workers} workers, {args.days}d lookback)")
        print("-" * 70)
        ok_count = 0
        fail_count = 0
        results: list[tuple[str, bool, str]] = []

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(backfill_one, sym, args.days): sym for sym in symbols}
            for i, future in enumerate(as_completed(futures), 1):
                sym, ok, detail = future.result()
                results.append((sym, ok, detail))
                if ok:
                    ok_count += 1
                else:
                    fail_count += 1
                if i % 20 == 0 or ok:
                    status = "✓" if ok else "✗"
                    print(f"  [{i:3d}/{len(symbols)}] {status} {sym:20s}  {detail[:60]}")

        print(f"\n  Backfill complete: {ok_count} OK, {fail_count} failed / {len(symbols)} total")
        print()

    # -----------------------------------------------------------------------
    # Step 2: Pull fresh bars from data-service API into parquets
    # -----------------------------------------------------------------------
    if not args.skip_api:
        print("STEP 2: Pull fresh bars from data-service API into parquets")
        print("-" * 70)
        total_added = 0
        updated: list[str] = []
        stale: list[str] = []

        with ThreadPoolExecutor(max_workers=min(args.workers, 12)) as pool:
            futures = {pool.submit(fetch_and_append, sym): sym for sym in symbols}
            for i, future in enumerate(as_completed(futures), 1):
                sym, added, last = future.result()
                if added > 0:
                    total_added += added
                    updated.append(sym)
                    print(f"  [{i:3d}/{len(symbols)}] + {sym:20s}  +{added} bars  last={last}")
                elif "up-to-date" not in last and "API error" not in last:
                    stale.append(f"{sym}({last})")

        print(f"\n  Bars added: {total_added}  |  Updated: {len(updated)}  |  Unchanged: {len(symbols) - len(updated)}")
        if stale:
            print(f"  Stale (no new bars): {', '.join(stale[:20])}")

    # -----------------------------------------------------------------------
    # Step 3: Freshness report
    # -----------------------------------------------------------------------
    print("\nSTEP 3: Data freshness summary")
    print("-" * 70)
    # IST thresholds (bars are timestamped at midnight IST in UTC)
    # Sep 29 IST start = Sep 28 18:30 UTC; Sep 28 IST start = Sep 27 18:30 UTC
    SEP29_IST_UTC = pd.Timestamp("2026-09-28 18:30:00", tz="UTC")
    SEP28_IST_UTC = pd.Timestamp("2026-09-27 18:30:00", tz="UTC")

    fresh_sep29_count, fresh_sep28_count, stale_syms_v2 = 0, 0, []
    if PARQUET_DIR.exists():
        for pf in sorted(PARQUET_DIR.glob("*.parquet")):
            try:
                df = pd.read_parquet(pf)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                last = df.index.max()
                if last >= SEP29_IST_UTC:
                    fresh_sep29_count += 1
                elif last >= SEP28_IST_UTC:
                    fresh_sep28_count += 1
                else:
                    last_ist = (last + pd.Timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
                    stale_syms_v2.append(f"{pf.stem}({last_ist} IST)")
            except Exception:
                stale_syms_v2.append(f"{pf.stem}(error)")

    total_syms = fresh_sep29_count + fresh_sep28_count + len(stale_syms_v2)
    print(f"  Sep 29 IST bar : {fresh_sep29_count} symbols  (via Angel One)")
    print(f"  Sep 28 IST bar : {fresh_sep28_count} symbols  (Upstox CDN settling)")
    print(f"  Fresh total    : {fresh_sep29_count + fresh_sep28_count}/{total_syms}")
    if stale_syms_v2:
        print(f"  Stale/error    : {', '.join(stale_syms_v2)}")
    else:
        print("  All symbols up to date!")

    print("\n" + "=" * 70)
    print("  BACKFILL COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
