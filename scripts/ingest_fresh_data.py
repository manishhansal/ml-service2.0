"""
scripts/ingest_fresh_data.py
------------------------------
Ingest fresh daily bars for the forward-paper-signal symbols so the
parquets extend past 2026-09-22 and signals can be resolved.

Calls data-service2.0 for each symbol, appending only new bars.
Rate-limit-aware: respects the 100 req/60s limit with built-in sleep.

Usage:
    PYTHONPATH=. python3 scripts/ingest_fresh_data.py
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_DIR = Path("data/1d/1d")
SIGNALS_PATH = Path("artifacts/forward_paper/signals.jsonl")
ENV_PATH = Path(".env")

RATE_LIMIT_REQUESTS = 90   # leave headroom below the 100/60s limit
SLEEP_BETWEEN_BATCHES = 65  # seconds to wait after each batch

# ── Read env ──────────────────────────────────────────────────────────────────
env = {}
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()

DATA_KEY = env.get("DATA_SERVICE_API_KEY", "")
DATA_URL = env.get("DATA_SERVICE_2_URL", "http://localhost:8200")


def fetch_historical(symbol: str, interval: str = "1d") -> list[dict]:
    url = f"{DATA_URL}/v1/india/historical/{symbol}/{interval}"
    req = urllib.request.Request(url, headers={"X-API-KEY": DATA_KEY})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode()
            d = json.loads(body)
            if isinstance(d, list):
                return d
            return d.get("data", d.get("bars", []))
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if "RATE_LIMIT" in body:
            raise RateLimitError(body)
        return []
    except Exception:
        return []


class RateLimitError(Exception):
    pass


def update_parquet(symbol: str, new_bars: list[dict]) -> int:
    """Append new bars to the symbol's parquet. Returns count of new rows added."""
    pf = PARQUET_DIR / f"{symbol}.parquet"
    if not new_bars:
        return 0

    # Build new DataFrame
    try:
        df_new = pd.DataFrame(new_bars)
        df_new.columns = [c.lower() for c in df_new.columns]
    except Exception:
        return 0

    # Detect timestamp column
    ts_col = next((c for c in ["timestamp", "datetime", "date", "time"] if c in df_new.columns), None)
    if ts_col is None:
        return 0
    try:
        df_new.index = pd.to_datetime(df_new[ts_col], utc=True)
        df_new = df_new.drop(columns=[ts_col])
    except Exception:
        return 0

    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col in df_new.columns:
            df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

    # Load existing parquet and find new rows
    if pf.exists():
        df_existing = pd.read_parquet(pf)
        if df_existing.index.tz is None:
            df_existing.index = df_existing.index.tz_localize("UTC")
        last_ts = df_existing.index.max()
        df_append = df_new[df_new.index > last_ts]
        if df_append.empty:
            return 0
        # Align columns
        common_cols = [c for c in df_existing.columns if c in df_append.columns]
        df_combined = pd.concat([df_existing[common_cols], df_append[common_cols]])
        df_combined.to_parquet(pf)
        return len(df_append)
    else:
        df_new[[c for c in required_cols if c in df_new.columns]].to_parquet(pf)
        return len(df_new)


def main() -> None:
    # Load symbols from forward paper signals
    if not SIGNALS_PATH.exists():
        print("No forward paper signals found")
        return

    signals = [json.loads(l) for l in SIGNALS_PATH.read_text().splitlines() if l.strip()]
    symbols = sorted({s["symbol"] for s in signals})
    print(f"Ingesting fresh data for {len(symbols)} symbols...")
    print(f"Rate limit: {RATE_LIMIT_REQUESTS} req/batch, sleep {SLEEP_BETWEEN_BATCHES}s between batches")

    total_new = 0
    errors = []
    batch_count = 0
    req_in_batch = 0

    for i, sym in enumerate(symbols, 1):
        if req_in_batch >= RATE_LIMIT_REQUESTS:
            print(f"  Rate limit reached. Sleeping {SLEEP_BETWEEN_BATCHES}s...")
            time.sleep(SLEEP_BETWEEN_BATCHES)
            req_in_batch = 0
            batch_count += 1

        try:
            bars = fetch_historical(sym, "1d")
            req_in_batch += 1
            added = update_parquet(sym, bars)
            if added > 0:
                total_new += added
                print(f"  [{i:3d}/{len(symbols)}] {sym:20s} +{added} new bars")
            else:
                print(f"  [{i:3d}/{len(symbols)}] {sym:20s} no new bars")
            time.sleep(0.7)  # ~1.4 req/s = well within 100/60s limit

        except RateLimitError:
            print(f"  [{i:3d}/{len(symbols)}] {sym:20s} RATE_LIMITED — sleeping {SLEEP_BETWEEN_BATCHES}s")
            time.sleep(SLEEP_BETWEEN_BATCHES)
            req_in_batch = 0
            # Retry once
            try:
                bars = fetch_historical(sym, "1d")
                req_in_batch += 1
                added = update_parquet(sym, bars)
                total_new += added
                print(f"            → retry OK: +{added} bars")
            except Exception as retry_exc:
                errors.append(f"{sym}: {retry_exc}")

        except Exception as exc:
            errors.append(f"{sym}: {exc}")
            print(f"  [{i:3d}/{len(symbols)}] {sym:20s} ERROR: {exc}")

    print()
    print(f"Ingestion complete: {total_new} new bars added across {len(symbols)} symbols")
    if errors:
        print(f"Errors ({len(errors)}): {errors[:5]}")

    # Check what the latest dates look like now
    print()
    print("Checking parquet dates for forward-paper symbols...")
    latest_dates = {}
    for sym in symbols[:5]:
        pf = PARQUET_DIR / f"{sym}.parquet"
        if pf.exists():
            df = pd.read_parquet(pf)
            latest_dates[sym] = str(df.index.max())
    for sym, dt in latest_dates.items():
        print(f"  {sym:20s} last bar: {dt}")


if __name__ == "__main__":
    main()
