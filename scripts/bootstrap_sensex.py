"""
bootstrap_sensex.py — Fetch 2 years of BSE SENSEX daily OHLCV from Upstox
and save as data/1d/1d/SENSEX.parquet so it's available for the SMC scanner.

Usage:
    python scripts/bootstrap_sensex.py
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT  = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data" / "1d" / "1d"
IST      = timezone(timedelta(hours=5, minutes=30))

INSTRUMENT_KEY = "BSE_INDEX|SENSEX"
CHUNK_MONTHS   = 3    # fetch 3 months at a time (Upstox limit)


def _load_token() -> str:
    for line in Path("/Users/manishkumar/Desktop/data-service2.0/.env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            if k.strip() == "UPSTOX_ACCESS_TOKEN":
                return v.strip()
    raise SystemExit("UPSTOX_ACCESS_TOKEN not found")


def fetch_chunk(ikey: str, from_date: str, to_date: str, token: str) -> list[list]:
    """Return candles for a date range (max ~3 months)."""
    url = (
        "https://api.upstox.com/v2/historical-candle/"
        + urllib.parse.quote(ikey, safe="")
        + f"/day/{to_date}/{from_date}"
    )
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept":        "application/json",
        "User-Agent":    "Mozilla/5.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read()).get("data", {}).get("candles", [])
    except Exception as e:
        print(f"  fetch error {from_date}→{to_date}: {e}")
        return []


def main() -> None:
    token    = _load_token()
    pf_path  = DATA_DIR / "SENSEX.parquet"
    today    = datetime.now(IST).date()
    end_date = today
    # Fetch 2 years back
    start_date = date(today.year - 2, today.month, today.day)

    print(f"Bootstrapping SENSEX data: {start_date} → {end_date}")
    print(f"Saving to: {pf_path}")

    all_candles: list[list] = []
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(
            date(chunk_start.year + (1 if chunk_start.month > 9 else 0),
                 ((chunk_start.month + 2) % 12) + 1 if chunk_start.month <= 9 else (chunk_start.month + 3) % 12,
                 1) - timedelta(days=1),
            end_date
        )
        # Simpler: just add 90 days
        chunk_end = min(chunk_start + timedelta(days=90), end_date)

        candles = fetch_chunk(INSTRUMENT_KEY, str(chunk_start), str(chunk_end), token)
        all_candles.extend(candles)
        print(f"  {chunk_start} → {chunk_end}: {len(candles)} candles")

        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(0.3)

    if not all_candles:
        print("No data received — aborting.")
        return

    # Build DataFrame
    rows = []
    for c in all_candles:
        # c = [timestamp_ist, open, high, low, close, volume, oi]
        ts_ist = pd.Timestamp(c[0])
        ts_utc = ts_ist.astimezone(timezone.utc)
        rows.append({
            "open":     float(c[1]),
            "high":     float(c[2]),
            "low":      float(c[3]),
            "close":    float(c[4]),
            "volume":   float(c[5]) if c[5] else 0.0,
            "provider": "upstox",
        })
        rows[-1]["_ts"] = ts_utc

    df = pd.DataFrame(rows)
    df = df.set_index(pd.DatetimeIndex([r["_ts"] for r in rows], tz="UTC"))
    df = df.drop(columns=["_ts"])
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    df.to_parquet(pf_path)
    print(f"\nSaved {len(df)} rows to {pf_path}")
    print(f"Date range: {df.index[0].astimezone(IST).date()} → {df.index[-1].astimezone(IST).date()}")
    print(f"Latest close: {df['close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
