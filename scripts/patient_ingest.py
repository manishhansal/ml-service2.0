#!/usr/bin/env python3
"""
Patient data ingestion for forward-paper symbols.
20-second sleep between calls stays within the shared rate limit
even with internal data-service workers consuming quota.
"""
from __future__ import annotations
import json, time, urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

PARQUET_DIR = Path("data/1d/1d")
SIGNALS_PATH = Path("artifacts/forward_paper/signals.jsonl")
LOG_PATH = Path("artifacts/ingest_log.json")
SLEEP = 20          # seconds between symbols — conservative but reliable

env = {k.strip(): v.strip() for line in Path(".env").read_text().splitlines()
       if "=" in line and not line.strip().startswith("#")
       for k, _, v in [line.partition("=")]}

DATA_KEY  = env.get("DATA_SERVICE_API_KEY", "")
DATA_URL  = env.get("DATA_SERVICE_2_URL", "http://localhost:8200")


def get_historical(symbol: str) -> list[dict]:
    url = f"{DATA_URL}/v1/india/historical/{symbol}/1d"
    req = urllib.request.Request(url, headers={"X-API-KEY": DATA_KEY})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read().decode()
    except urllib.error.HTTPError as e:
        raw = e.read().decode()

    if "RATE_LIMIT" in raw:
        return []                     # caller handles retry

    d = json.loads(raw) if raw.startswith(("{", "[")) else {}
    if isinstance(d, list):
        return d
    for key in ("data", "bars", "ohlcv", "candles"):
        if key in d and isinstance(d[key], list):
            return d[key]
    return []


def detect_ts_col(df: pd.DataFrame) -> str | None:
    for c in ("timestamp", "datetime", "date", "time", "t"):
        if c in df.columns:
            return c
    return None


def append_new_bars(symbol: str, bars: list[dict]) -> int:
    if not bars:
        return 0
    pf = PARQUET_DIR / f"{symbol}.parquet"
    try:
        df_new = pd.DataFrame(bars)
        df_new.columns = [c.lower() for c in df_new.columns]
        ts_col = detect_ts_col(df_new)
        if ts_col is None:
            return 0
        df_new.index = pd.to_datetime(df_new[ts_col], utc=True)
        df_new = df_new.drop(columns=[ts_col])
        for c in ("open", "high", "low", "close", "volume"):
            if c in df_new.columns:
                df_new[c] = pd.to_numeric(df_new[c], errors="coerce")
        req_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df_new.columns]
        df_new = df_new[req_cols].dropna(subset=["close"])
    except Exception as e:
        print(f"    parse error for {symbol}: {e}")
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
    common = [c for c in df_exist.columns if c in new_rows.columns]
    pd.concat([df_exist[common], new_rows[common]]).to_parquet(pf)
    return len(new_rows)


def main():
    signals = [json.loads(l) for l in SIGNALS_PATH.read_text().splitlines() if l.strip()]
    symbols = sorted({s["symbol"] for s in signals})
    print(f"[ingest] {len(symbols)} symbols — {SLEEP}s sleep between each")

    log = {"started": datetime.now(tz=timezone.utc).isoformat(), "results": {}}
    total_new = 0

    for i, sym in enumerate(symbols, 1):
        print(f"  [{i:2d}/{len(symbols)}] {sym}", end=" ", flush=True)
        bars = get_historical(sym)

        if not bars:
            # One retry after waiting
            print("rate-limited, sleeping 65s...", end=" ", flush=True)
            time.sleep(65)
            bars = get_historical(sym)

        added = append_new_bars(sym, bars)
        total_new += added
        print(f"+{added} bars" if added else "no new bars")
        log["results"][sym] = {"bars_fetched": len(bars), "new_bars": added}

        if i < len(symbols):
            time.sleep(SLEEP)

    log["finished"] = datetime.now(tz=timezone.utc).isoformat()
    log["total_new_bars"] = total_new
    LOG_PATH.write_text(json.dumps(log, indent=2))

    # Summary
    print(f"\n[ingest] Done: {total_new} new bars added")
    # Check latest dates
    print("[ingest] Latest bar dates:")
    for sym in symbols[:5]:
        pf = PARQUET_DIR / f"{sym}.parquet"
        if pf.exists():
            df = pd.read_parquet(pf)
            print(f"  {sym:20s} {df.index.max()}")


if __name__ == "__main__":
    main()
