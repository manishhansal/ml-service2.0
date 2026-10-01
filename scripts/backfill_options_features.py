#!/usr/bin/env python3
"""
scripts/backfill_options_features.py
-----------------------------------------
Backfill daily options/IV features for all F&O-eligible symbols
into data/options/1d/{symbol}.parquet.

Data source: data-service2.0 GET /v1/india/optionchain (uses Angel One / Upstox)

Coverage: Data available for whatever the provider has.  For training,
this adds 5 Group I features with whatever historical depth exists.

Usage::
    PYTHONPATH=. python3 scripts/backfill_options_features.py
    PYTHONPATH=. python3 scripts/backfill_options_features.py --symbols NIFTY RELIANCE
    PYTHONPATH=. python3 scripts/backfill_options_features.py --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timezone, timedelta, date as _date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clients.data_service import DataServiceClient
from src.features.families.options_iv import (
    OPTIONS_FEATURE_NAMES,
    compute_options_features_from_chain,
)
from src.logging_config import get_logger

log = get_logger(__name__)

DAILY_DIR    = Path("data/1d/1d")
OPTIONS_DIR  = Path("data/options/1d")
RATE_SLEEP   = 0.5
MAX_RETRIES  = 2

# F&O universe (Nifty 50 + BankNifty + key midcaps always have options)
# Script auto-discovers from data/1d/1d but we always include indices
MANDATORY_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY"]


def _load_universe() -> list[str]:
    syms = [p.stem for p in DAILY_DIR.glob("*.parquet")]
    for s in MANDATORY_SYMBOLS:
        if s not in syms:
            syms.append(s)
    return sorted(syms)


def _save_row(symbol: str, row_date: _date, feats: dict[str, float]) -> None:
    OPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pf = OPTIONS_DIR / f"{symbol}.parquet"
    ts = pd.Timestamp(row_date).tz_localize("UTC")
    row_df = pd.DataFrame([feats], index=[ts], columns=OPTIONS_FEATURE_NAMES)
    row_df.index.name = "timestamp"

    if pf.exists():
        existing = pd.read_parquet(pf)
        if existing.index.tz is None:
            existing.index = existing.index.tz_localize("UTC")
        existing = existing[existing.index.normalize() != ts.normalize()]
        combined = pd.concat([existing, row_df]).sort_index()
    else:
        combined = row_df

    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(pf, index=True, compression="snappy")


async def _fetch_chain(
    client: DataServiceClient,
    symbol: str,
) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            chain = await client.get_option_chain(symbol)
            if chain:
                return chain
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                log.debug(f"Options chain failed {symbol}: {exc}")
            await asyncio.sleep(1.0)
    return None


async def backfill_all(
    symbols: list[str],
    row_date: _date,
    dry_run: bool = False,
) -> dict[str, int]:
    """Fetch option chain + compute features for all symbols."""
    client = DataServiceClient()
    await client.connect()
    results: dict[str, int] = {}
    t0 = time.monotonic()

    print(f"\n{'='*60}")
    print(f"  OPTIONS FEATURE BACKFILL — {row_date}")
    print(f"  Symbols: {len(symbols)}")
    if dry_run:
        print("  DRY RUN")
    print(f"{'='*60}\n")

    ok_count = skip_count = fail_count = 0

    for i, symbol in enumerate(symbols, 1):
        chain = await _fetch_chain(client, symbol)
        await asyncio.sleep(RATE_SLEEP)

        if chain is None:
            fail_count += 1
            if i % 50 == 0:
                elapsed = time.monotonic() - t0
                eta = (len(symbols) - i) * elapsed / i
                print(f"  [{i:3d}/{len(symbols)}] {symbol:20s} ✗ no chain  ETA={eta/60:.1f}m")
            results[symbol] = 0
            continue

        feats = compute_options_features_from_chain(chain)
        has_data = feats.get("put_call_ratio", 0) > 0 or feats.get("iv_atm_pct", 0) > 0

        if not dry_run:
            _save_row(symbol, row_date, feats)
            ok_count += 1
        else:
            skip_count += 1

        if has_data or i % 50 == 0 or i == len(symbols):
            elapsed = time.monotonic() - t0
            eta = (len(symbols) - i) * elapsed / i if i > 0 else 0
            pcr = feats.get("put_call_ratio", 0)
            iv  = feats.get("iv_atm_pct", 0)
            mark = "✓" if has_data else "·"
            if has_data or i % 50 == 0:
                print(f"  [{i:3d}/{len(symbols)}] {symbol:20s} {mark}  PCR={pcr:.2f}  IV={iv:.1f}%  ETA={eta/60:.1f}m")
        results[symbol] = 1 if has_data else 0

    await client.disconnect()
    elapsed = time.monotonic() - t0
    print(f"\n  Done: ok={ok_count} failed={fail_count} dry={skip_count}  time={elapsed:.1f}s")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill options features")
    p.add_argument("--symbols", nargs="+")
    p.add_argument("--date", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    symbols  = args.symbols if args.symbols else _load_universe()
    row_date = _date.fromisoformat(args.date) if args.date else _date.today()

    asyncio.run(backfill_all(symbols, row_date, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
