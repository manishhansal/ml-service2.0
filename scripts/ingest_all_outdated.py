#!/usr/bin/env python3
"""
Ingest fresh bars for ALL symbols with outdated data.
With rate limit = 500/60s, we can make ~8 req/sec without hitting limits.
"""
from __future__ import annotations
import json, time, urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

PARQUET_DIR = Path("data/1d/1d")
env = {k.strip(): v.strip() for line in Path(".env").read_text().splitlines()
       if "=" in line and not line.strip().startswith("#")
       for k, _, v in [line.partition("=")]}
DATA_KEY = env.get("DATA_SERVICE_API_KEY", "")
DATA_URL = env.get("DATA_SERVICE_2_URL", "http://localhost:8200")


def get_historical(symbol: str) -> list[dict]:
    url = f"{DATA_URL}/v1/india/historical?symbol={symbol}&interval=1d"
    req = urllib.request.Request(url, headers={"X-API-KEY": DATA_KEY})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode())
            return d.get("data", []) if isinstance(d, dict) else (d if isinstance(d, list) else [])
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if "RATE_LIMIT" in body:
            return "RATE_LIMIT"
        return []
    except Exception:
        return []


def append_bars(symbol: str, bars: list[dict]) -> int:
    if not bars:
        return 0
    pf = PARQUET_DIR / f"{symbol}.parquet"
    try:
        df_new = pd.DataFrame(bars)
        if "time" in df_new.columns:
            df_new.index = pd.to_datetime(df_new["time"], unit="s", utc=True)
            df_new = df_new.drop(columns=["time"])
        elif "timestamp" in df_new.columns:
            df_new.index = pd.to_datetime(df_new["timestamp"], utc=True)
            df_new = df_new.drop(columns=["timestamp"])
        else:
            return 0
        for c in ("open","high","low","close","volume"):
            if c in df_new.columns:
                df_new[c] = pd.to_numeric(df_new[c], errors="coerce")
        df_new = df_new.dropna(subset=["close"])
    except Exception:
        return 0

    if not pf.exists():
        cols = [c for c in ("open","high","low","close","volume") if c in df_new.columns]
        df_new[cols].to_parquet(pf)
        return len(df_new)

    df_ex = pd.read_parquet(pf)
    if df_ex.index.tz is None:
        df_ex.index = df_ex.index.tz_localize("UTC")
    last = df_ex.index.max()
    new = df_new[df_new.index > last]
    if new.empty:
        return 0
    common = [c for c in df_ex.columns if c in new.columns]
    pd.concat([df_ex[common], new[common]]).to_parquet(pf)
    return len(new)


def main():
    # Find all symbols and their dates
    all_parquets = sorted(PARQUET_DIR.glob("*.parquet"))
    dates = {}
    for pf in all_parquets:
        df = pd.read_parquet(pf)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        dates[pf.stem] = str(df.index.max().date())

    outdated = [s for s, d in dates.items() if d < "2026-09-23"]
    print(f"Symbols to update: {len(outdated)} (last bar < 2026-09-23)")
    print(f"All symbols on disk: {len(all_parquets)}")
    print()

    total_new = 0
    updated = []
    failed = []

    for i, sym in enumerate(outdated, 1):
        print(f"  [{i:3d}/{len(outdated)}] {sym:20s}", end=" ", flush=True)
        result = get_historical(sym)
        if result == "RATE_LIMIT":
            print("RATE_LIMITED — sleeping 3s...", end=" ", flush=True)
            time.sleep(3)
            result = get_historical(sym)
        added = append_bars(sym, result)
        if added:
            total_new += added
            updated.append(sym)
            print(f"+{added}")
        else:
            failed.append(sym)
            print("(no new)")
        time.sleep(0.3)  # ~3 req/sec, well within 500/60s

    print(f"\nDone: {total_new} new bars, {len(updated)} symbols updated, {len(failed)} unchanged")
    if updated:
        print("Updated:")
        for s in updated[:10]:
            df = pd.read_parquet(PARQUET_DIR / f"{s}.parquet")
            if df.index.tz is None: df.index = df.index.tz_localize("UTC")
            print(f"  {s:20s} last={df.index.max().date()}")


if __name__ == "__main__":
    main()
