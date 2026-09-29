#!/usr/bin/env python3
"""
fix_phantom_bars.py — Remove phantom 03:45 UTC duplicate bars from all parquets.

Root cause (P0-DATA-001):
  14 of 218 parquets contain duplicate rows: for each date, both an 18:30 UTC
  row (real EOD, 9:15 PM IST next day = correct NSE EOD) and a 03:45 UTC row
  (fake, identical OHLCV values = phantom duplicate) exist.

  These phantom bars make up ~49.9% of all rows in affected parquets and corrupt
  every rolling indicator:
    - RSI(14) runs over ~7 real bars not 14
    - MACD, Bollinger, vol_20, stochastic: all halved window effects
    - ret_5, ret_20: cover 2.5 and 10 real days, not 5 and 20
    - trend_persistence: fraction of 20 bars in EMA direction → overstated

  This directly caused KAYNES to receive a 0.9321 LONG score (should be SHORT/FLAT).

Fix:
  Keep only 18:30 UTC rows (NSE EOD = real data). Remove all 03:45 UTC phantom rows.
  Also deduplicate on date in case other duplicate patterns exist.

Usage:
    PYTHONPATH=. python3 scripts/fix_phantom_bars.py [--dry-run]
    PYTHONPATH=. python3 scripts/fix_phantom_bars.py --symbols KAYNES GLENMARK ADANIENT
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

PARQUET_DIR = Path("data/1d/1d")


def fix_parquet(sym: str, dry_run: bool = False) -> dict:
    pf = PARQUET_DIR / f"{sym}.parquet"
    if not pf.exists():
        return {"symbol": sym, "status": "NOT_FOUND"}

    df = pd.read_parquet(pf)

    # Normalize timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    original_len = len(df)

    # ── Step 1: Remove 03:45 UTC phantom bars ────────────────────────────────
    # NSE EOD bars should always be at 18:30 UTC (midnight IST + 30 min)
    # or occasionally at 10:00 UTC (4:30 PM IST — for some providers).
    # 03:45 UTC = 9:15 IST (market OPEN time) → these are phantom duplicates.
    real_hours = df[df.index.hour != 3]   # remove 03:xx UTC rows
    phantom_count = original_len - len(real_hours)

    # ── Step 2: Also deduplicate on date (keep last occurrence per date) ──────
    # Guards against any other duplicate pattern
    deduped = real_hours[~real_hours.index.normalize().duplicated(keep="last")]
    extra_dups = len(real_hours) - len(deduped)

    final_df = deduped
    final_len = len(final_df)

    status = {
        "symbol": sym,
        "original_rows": original_len,
        "phantom_removed": phantom_count,
        "extra_dups_removed": extra_dups,
        "final_rows": final_len,
        "reduction_pct": round((original_len - final_len) / max(original_len, 1) * 100, 1),
        "last_date": str(final_df.index[-1].date()) if len(final_df) else "empty",
        "status": "FIXED" if phantom_count + extra_dups > 0 else "CLEAN",
        "dry_run": dry_run,
    }

    if not dry_run and (phantom_count + extra_dups > 0):
        final_df.to_parquet(pf)

    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be changed without writing")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="Specific symbols to fix (default: all 218)")
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = [pf.stem for pf in sorted(PARQUET_DIR.glob("*.parquet"))]

    print(f"{'DRY RUN — ' if args.dry_run else ''}Fixing phantom bars in {len(symbols)} parquets ...")
    print(f"{'Symbol':20s} {'Before':>8} {'Phantom':>8} {'ExtraDup':>9} {'After':>8} {'Status':>10}")
    print("-" * 72)

    fixed = 0
    clean = 0
    total_phantom = 0

    for sym in symbols:
        r = fix_parquet(sym, dry_run=args.dry_run)
        if r["status"] == "NOT_FOUND":
            print(f"  {sym:20s} NOT FOUND")
            continue
        marker = "◄ FIXED" if r["phantom_removed"] + r["extra_dups_removed"] > 0 else ""
        print(f"  {sym:20s} {r['original_rows']:>8} {r['phantom_removed']:>8} "
              f"{r['extra_dups_removed']:>9} {r['final_rows']:>8}  {r['status']:>10}  {marker}")
        if r["status"] == "FIXED":
            fixed += 1
            total_phantom += r["phantom_removed"]
        else:
            clean += 1

    print()
    print(f"Summary: {fixed} fixed, {clean} already clean, {total_phantom} phantom rows removed")
    if args.dry_run:
        print("(DRY RUN — no files were written)")
    else:
        print(f"✓ All parquets cleaned. Run 'make ingest' to fetch fresh bars.")


if __name__ == "__main__":
    main()
