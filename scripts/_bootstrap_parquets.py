#!/usr/bin/env python3
"""
_bootstrap_parquets.py — Create parquets from scratch for all NSE F&O symbols.

Fetches 2 years of 1d OHLCV from the Upstox historical-candle API (one range
call per symbol) and writes data/1d/1d/<SYMBOL>.parquet.

Safe to re-run: symbols that already have data through the --to date are skipped.

Usage:
    python3 scripts/_bootstrap_parquets.py
    python3 scripts/_bootstrap_parquets.py --from 2024-01-01 --to 2026-10-30
"""
from __future__ import annotations

import argparse
import gzip
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT  = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data" / "1d" / "1d"
IST      = timezone(timedelta(hours=5, minutes=30))

# Bypass SSL cert verification — Upstox API, internal tooling only
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

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
    raise SystemExit("UPSTOX_ACCESS_TOKEN not found in data-service2.0 .env")

# ── Instruments ───────────────────────────────────────────────────────────────

def _load_sym_map() -> dict[str, str]:
    gz = Path("/tmp/nse_instruments.json.gz")
    if not gz.exists() or (datetime.now() - datetime.fromtimestamp(gz.stat().st_mtime)).days > 1:
        print("Downloading NSE instruments master...")
        req = urllib.request.Request(
            "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            gz.write_bytes(resp.read())
    with gzip.open(gz) as f:
        instruments = json.load(f)

    sym_map: dict[str, str] = {}
    for inst in instruments:
        if inst.get("segment") == "NSE_EQ" and inst.get("instrument_type") in ("EQ", "BE"):
            sym_map[inst["trading_symbol"]] = inst["instrument_key"]

    sym_map.update({
        "TATAMOTORS": "NSE_EQ|INE155A01022",
        "HFCL":       "NSE_EQ|INE548A01028",
        "NIFTY":      "NSE_INDEX|Nifty 50",
        "BANKNIFTY":  "NSE_INDEX|Nifty Bank",
        "FINNIFTY":   "NSE_INDEX|Nifty Fin Service",
        "MIDCPNIFTY": "NSE_INDEX|Nifty MidCap Select",
        "SENSEX":     "BSE_INDEX|SENSEX",
    })
    return sym_map

# ── Fetch range ───────────────────────────────────────────────────────────────

def fetch_range(instrument_key: str, from_str: str, to_str: str, token: str) -> list[dict]:
    """Fetch all 1d candles between from_str and to_str inclusive."""
    url = (
        "https://api.upstox.com/v2/historical-candle/"
        + urllib.parse.quote(instrument_key, safe="")
        + f"/day/{to_str}/{from_str}"   # Upstox: to before from
    )
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept":        "application/json",
        "User-Agent":    "Mozilla/5.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=20, context=_SSL_CTX) as r:
            data = json.loads(r.read().decode())
        return data.get("data", {}).get("candles", [])
    except Exception:
        return []

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap NSE parquets from Upstox")
    parser.add_argument("--from", dest="from_date", default="2024-01-01")
    parser.add_argument("--to",   dest="to_date",   default=str(datetime.now(IST).date()))
    parser.add_argument("--force", action="store_true", help="Re-fetch even if parquet exists")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    token   = _load_token()
    sym_map = _load_sym_map()

    # Load F&O universe from the canonical universe module
    try:
        from src.scanners.universe import NSE_FO_STOCKS, NSE_INDICES
        fo_universe = (NSE_FO_STOCKS | NSE_INDICES) & set(sym_map.keys())
    except ImportError:
        fo_universe = set()

    # Determine symbol list: F&O universe + any existing parquets
    existing  = {f.stem for f in DATA_DIR.glob("*.parquet")}
    all_syms  = sorted((fo_universe | existing) & set(sym_map.keys()))

    if not all_syms:
        print("[bootstrap] F&O universe import failed — using full instruments master.")
        all_syms = sorted(sym_map.keys())

    print(f"Bootstrap: {args.from_date} → {args.to_date}  |  {len(all_syms)} symbols")
    print()

    to_date   = date.fromisoformat(args.to_date)
    updated   = skipped = nodata = 0

    for i, sym in enumerate(all_syms, 1):
        pf = DATA_DIR / f"{sym}.parquet"

        # Skip if already covers the full requested range (unless --force)
        if pf.exists() and not args.force:
            df_ex = pd.read_parquet(pf)
            if df_ex.index.tz is None:
                df_ex.index = df_ex.index.tz_localize("UTC")
            last_ist  = df_ex.index[-1].astimezone(IST).date()
            first_ist = df_ex.index[0].astimezone(IST).date()
            from_date_d = date.fromisoformat(args.from_date)
            if last_ist >= to_date and first_ist <= from_date_d + timedelta(days=7):
                skipped += 1
                continue

        ikey = sym_map.get(sym)
        if not ikey:
            nodata += 1
            continue

        candles = fetch_range(ikey, args.from_date, args.to_date, token)
        if not candles:
            nodata += 1
            print(f"  [skip] {sym}: no candles from Upstox")
            continue

        # Build dataframe
        rows = []
        for c in candles:
            try:
                ts = pd.Timestamp(c[0])
                rows.append({
                    "open":     float(c[1]),
                    "high":     float(c[2]),
                    "low":      float(c[3]),
                    "close":    float(c[4]),
                    "volume":   float(c[5]),
                    "provider": "upstox",
                })
                rows[-1]["_ts"] = ts.astimezone(timezone.utc)
            except Exception:
                continue

        if not rows:
            nodata += 1
            continue

        df = pd.DataFrame(rows)
        df = df.set_index(pd.DatetimeIndex([r["_ts"] for r in rows], tz="UTC"))
        df = df.drop(columns=["_ts"], errors="ignore")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Merge with existing data if parquet already exists
        if pf.exists():
            df_old = pd.read_parquet(pf)
            if df_old.index.tz is None:
                df_old.index = df_old.index.tz_localize("UTC")
            df = pd.concat([df_old, df])
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

        df.to_parquet(pf)
        updated += 1

        if i % 20 == 0 or i <= 3:
            last = str(df.index.max().date())
            print(f"  [{i:3d}/{len(all_syms)}] {sym:<14} bars={len(df)}  last={last}")

        time.sleep(0.12)   # ~8 req/s, within rate limit

    print()
    print("=" * 55)
    print(f"  Created/updated : {updated}")
    print(f"  Skipped (fresh) : {skipped}")
    print(f"  No data         : {nodata}")
    print("=" * 55)

if __name__ == "__main__":
    main()
