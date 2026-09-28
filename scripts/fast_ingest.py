#!/usr/bin/env python3
"""
Fast ingestion with workers paused.
Uses correct endpoint: GET /v1/india/historical?symbol=X&interval=1d
Appends only bars newer than existing parquet data.
"""
from __future__ import annotations
import json, time, urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

PARQUET_DIR = Path("data/1d/1d")
SIGNALS_PATH = Path("artifacts/forward_paper/signals.jsonl")

env = {k.strip(): v.strip() for line in Path(".env").read_text().splitlines()
       if "=" in line and not line.strip().startswith("#")
       for k, _, v in [line.partition("=")]}
DATA_KEY = env.get("DATA_SERVICE_API_KEY", "")
DATA_URL  = env.get("DATA_SERVICE_2_URL", "http://localhost:8200")


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
            print("  [RATE_LIMIT]", end=" ", flush=True)
        return []
    except Exception as e:
        print(f"  [ERR {e}]", end=" ", flush=True)
        return []


def append_new_bars(symbol: str, bars: list[dict]) -> int:
    if not bars:
        return 0
    pf = PARQUET_DIR / f"{symbol}.parquet"
    try:
        df_new = pd.DataFrame(bars)
        # Convert Unix timestamps to UTC DatetimeIndex
        if "time" in df_new.columns:
            df_new.index = pd.to_datetime(df_new["time"], unit="s", utc=True)
            df_new = df_new.drop(columns=["time"])
        elif "timestamp" in df_new.columns:
            df_new.index = pd.to_datetime(df_new["timestamp"], utc=True)
            df_new = df_new.drop(columns=["timestamp"])
        else:
            return 0
        df_new.index.name = None
        # Keep only OHLCV + known metadata cols
        keep = [c for c in df_new.columns if c in ("open","high","low","close","volume","oi","provider","volumeUnavailable","sourceType")]
        df_new = df_new[keep]
        for c in ("open","high","low","close","volume"):
            if c in df_new.columns:
                df_new[c] = pd.to_numeric(df_new[c], errors="coerce")
        df_new = df_new.dropna(subset=["close"])
    except Exception as e:
        print(f"  [PARSE:{e}]", end=" ", flush=True)
        return 0

    if not pf.exists():
        df_new.to_parquet(pf)
        return len(df_new)

    df_exist = pd.read_parquet(pf)
    if df_exist.index.tz is None:
        df_exist.index = df_exist.index.tz_localize("UTC")
    last = df_exist.index.max()
    new_rows = df_new[df_new.index > last]
    if new_rows.empty:
        return 0
    # Align columns
    for c in df_exist.columns:
        if c not in new_rows.columns:
            new_rows = new_rows.copy()
            new_rows[c] = None
    pd.concat([df_exist[df_exist.columns], new_rows[df_exist.columns]]).to_parquet(pf)
    return len(new_rows)


def main():
    signals = [json.loads(l) for l in SIGNALS_PATH.read_text().splitlines() if l.strip()]
    symbols = sorted({s["symbol"] for s in signals})
    # Also include broader universe
    extra = sorted([pf.stem for pf in PARQUET_DIR.glob("*.parquet")])
    all_syms = sorted(set(symbols) | set(extra))
    print(f"[fast_ingest] {len(all_syms)} symbols (workers paused, 1.5s between calls)")

    total_new = 0
    stats = {}
    for i, sym in enumerate(all_syms, 1):
        print(f"  [{i:3d}/{len(all_syms)}] {sym:20s}", end=" ", flush=True)
        bars = get_historical(sym)
        added = append_new_bars(sym, bars)
        total_new += added
        stats[sym] = added
        if added:
            print(f"+{added} bars")
        else:
            print("(no new)")
        time.sleep(1.5)   # 40 req/min — well within 100/min limit

    # Summary
    updated = {s: n for s, n in stats.items() if n > 0}
    print(f"\n[fast_ingest] Done — {total_new} new bars across {len(updated)} symbols")
    print("[fast_ingest] Updated symbols:")
    for s, n in sorted(updated.items()):
        pf = PARQUET_DIR / f"{s}.parquet"
        df = pd.read_parquet(pf)
        print(f"  {s:20s} last bar: {df.index.max().date()}  +{n} new")

    # Save log
    Path("artifacts/fast_ingest_log.json").write_text(
        json.dumps({"finished": datetime.now(tz=timezone.utc).isoformat(),
                    "total_new": total_new, "by_symbol": stats}, indent=2))


if __name__ == "__main__":
    main()
