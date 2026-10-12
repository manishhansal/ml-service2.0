"""
refresh_parquets.py — Fetch EOD OHLCV for NSE F&O symbols from Upstox
and append to parquet files.

Usage:
    # Fetch today's candle (default — auto-detects last trading day)
    python scripts/refresh_parquets.py

    # Fetch a specific date
    python scripts/refresh_parquets.py --date 2026-10-12

    # Fetch a range of dates (fills gaps)
    python scripts/refresh_parquets.py --from 2026-10-09 --to 2026-10-12

Run after market close (after 15:30 IST). Data is usually available by 16:00 IST.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT  = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data" / "1d" / "1d"
IST      = timezone(timedelta(hours=5, minutes=30))

# ── Auth ──────────────────────────────────────────────────────────────────────

def _load_token() -> str:
    env_path = Path("/Users/manishkumar/Desktop/data-service2.0/.env")
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            if k.strip() == "UPSTOX_ACCESS_TOKEN":
                tok = v.strip()
                if tok:
                    return tok
    raise SystemExit("UPSTOX_ACCESS_TOKEN not found in data-service .env")

# ── Instruments ───────────────────────────────────────────────────────────────

def _load_sym_map() -> dict[str, str]:
    gz = Path("/tmp/nse_instruments.json.gz")
    if not gz.exists() or (datetime.now() - datetime.fromtimestamp(gz.stat().st_mtime)).days > 1:
        print("Downloading NSE instruments master...")
        urllib.request.urlretrieve(
            "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
            gz,
        )
    with gzip.open(gz) as f:
        instruments = json.load(f)

    sym_map: dict[str, str] = {}
    for inst in instruments:
        if inst.get("segment") == "NSE_EQ" and inst.get("instrument_type") in ("EQ", "BE"):
            sym_map[inst["trading_symbol"]] = inst["instrument_key"]

    # Manual overrides / index mappings
    overrides = {
        "TATAMOTORS": "NSE_EQ|INE155A01022",
        "HFCL":       "NSE_EQ|INE548A01028",
        "LTI":        "NSE_EQ|INE214T01019",
    }
    sym_map.update(overrides)

    # Index instrument keys (NSE + BSE)
    index_map = {
        "NIFTY":      "NSE_INDEX|Nifty 50",
        "BANKNIFTY":  "NSE_INDEX|Nifty Bank",
        "FINNIFTY":   "NSE_INDEX|Nifty Fin Service",
        "MIDCPNIFTY": "NSE_INDEX|Nifty MidCap Select",
        "SENSEX":     "BSE_INDEX|SENSEX",       # BSE Sensex 30
    }
    sym_map.update(index_map)
    return sym_map

# ── Fetch candle ──────────────────────────────────────────────────────────────

def fetch_candle(instrument_key: str, date_str: str, token: str) -> dict | None:
    """Fetch 1d candle for instrument_key on date_str (YYYY-MM-DD IST)."""
    url = (
        "https://api.upstox.com/v2/historical-candle/"
        + urllib.parse.quote(instrument_key, safe="")
        + f"/day/{date_str}/{date_str}"
    )
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept":        "application/json",
        "User-Agent":    "Mozilla/5.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        candles = data.get("data", {}).get("candles", [])
        if not candles:
            return None
        c = candles[0]
        ts = pd.Timestamp(c[0])   # already has +05:30 timezone
        return {
            "timestamp_ist": ts,
            "open":   float(c[1]),
            "high":   float(c[2]),
            "low":    float(c[3]),
            "close":  float(c[4]),
            "volume": float(c[5]),
            "provider": "upstox",
        }
    except Exception as e:
        return None

# ── Date helpers ─────────────────────────────────────────────────────────────

def _last_trading_day() -> date:
    """Return the most recent Mon–Fri that isn't in the future."""
    today = datetime.now(IST).date()
    d = today
    while d.weekday() >= 5:   # skip Sat/Sun
        d -= timedelta(days=1)
    return d

def _trading_dates_between(from_date: date, to_date: date) -> list[date]:
    """Return all Mon–Fri dates between from_date and to_date inclusive."""
    dates = []
    d = from_date
    while d <= to_date:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill NSE F&O parquet data")
    parser.add_argument("--date",  type=str, default=None, help="Single date YYYY-MM-DD")
    parser.add_argument("--from",  dest="from_date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--to",    dest="to_date",   type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if date exists")
    args = parser.parse_args()

    # Determine target dates
    if args.date:
        target_dates = [date.fromisoformat(args.date)]
    elif args.from_date and args.to_date:
        target_dates = _trading_dates_between(
            date.fromisoformat(args.from_date),
            date.fromisoformat(args.to_date),
        )
    else:
        target_dates = [_last_trading_day()]

    token   = _load_token()
    sym_map = _load_sym_map()
    parquets = sorted(DATA_DIR.glob("*.parquet"))

    print(f"Refresh target dates : {[str(d) for d in target_dates]}")
    print(f"Symbols              : {len(parquets)}")
    print()

    grand_updated = grand_skipped = grand_nodata = grand_already = 0

    for target_date in target_dates:
        target_str = str(target_date)
        print(f"── {target_str} ──────────────────────────────────")
        updated = skipped = already_have = no_data = 0

        for pf in parquets:
            sym = pf.stem

            ikey = sym_map.get(sym)
            if not ikey:
                skipped += 1
                continue

            # Check existing coverage
            if not args.force:
                df_ex = pd.read_parquet(pf).drop(columns=["provider"], errors="ignore")
                if df_ex.index.tz is None:
                    df_ex.index = df_ex.index.tz_localize("UTC")
                last_ist = df_ex.index[-1].astimezone(IST).date()
                if last_ist >= target_date:
                    already_have += 1
                    continue

            row = fetch_candle(ikey, target_str, token)
            if row is None:
                no_data += 1
                continue

            # Store as UTC (IST midnight = prev day 18:30 UTC)
            ts_utc = row["timestamp_ist"].astimezone(timezone.utc)
            new_row = pd.DataFrame([{
                "open":     row["open"],
                "high":     row["high"],
                "low":      row["low"],
                "close":    row["close"],
                "volume":   row["volume"],
                "provider": row["provider"],
            }], index=pd.DatetimeIndex([ts_utc], tz="UTC"))

            df_full = pd.read_parquet(pf)
            if df_full.index.tz is None:
                df_full.index = df_full.index.tz_localize("UTC")

            df_combined = pd.concat([df_full, new_row])
            df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
            df_combined = df_combined.sort_index()
            df_combined.to_parquet(pf)

            updated += 1
            print(f"  [+] {sym:<14} close={row['close']:.2f}  vol={row['volume']:.0f}")
            time.sleep(0.12)   # ~8 req/s — well under 500/min limit

        print(f"  Updated={updated}  Already={already_have}  No-data={no_data}  Skipped={skipped}")
        grand_updated  += updated
        grand_skipped  += skipped
        grand_nodata   += no_data
        grand_already  += already_have

    print()
    print("=" * 60)
    print(f"  Total updated  : {grand_updated}")
    print(f"  Already had    : {grand_already}")
    print(f"  No data        : {grand_nodata}")
    print(f"  Skipped        : {grand_skipped}")
    print("=" * 60)

if __name__ == "__main__":
    main()
