#!/usr/bin/env python3
"""
scripts/refresh_universe_parquets.py
--------------------------------------
Refresh all daily 1d parquets from the data-service TimescaleDB and
download any DB symbols that are missing from data/1d/1d/.

Actions taken per symbol:
  1. If parquet missing:  fetch full history from DB → create parquet
  2. If parquet present but behind DB:  fetch gap → append + deduplicate
  3. If parquet present and up-to-date:  skip

Index/non-tradeable symbols are excluded (contain spaces, ^, VIX, SENSEX).

Usage::
    PYTHONPATH=. python3 scripts/refresh_universe_parquets.py
    PYTHONPATH=. python3 scripts/refresh_universe_parquets.py --dry-run
    PYTHONPATH=. python3 scripts/refresh_universe_parquets.py --symbols RELIANCE TCS
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clients.data_service import DataServiceClient
from src.logging_config import get_logger

log = get_logger(__name__)

PARQUET_DIR     = Path("data/1d/1d")
MIN_ROWS        = 252        # need at least 1 year to be useful for training
RATE_LIMIT_SLEEP = 0.25      # seconds between API calls
MAX_RETRIES      = 3
RETRY_SLEEP      = 5.0
HISTORY_FROM     = "2021-01-01"   # fetch from start of 2021 (5-yr window)

# Symbols to explicitly skip — index composites, volatility indices, etc.
SKIP_PATTERNS = [" ", "^", "VIX", "SENSEX", "NIFTY 50", "NIFTY AUTO",
                 "NIFTY BANK", "NIFTY FIN", "NIFTY FMCG", "NIFTY IT",
                 "NIFTY MID", "NIFTY PHARMA", "NIFTY REALTY", "INDIA VIX"]


def _is_tradeable(sym: str) -> bool:
    """Return True if symbol looks like a tradeable equity (not an index)."""
    for pat in SKIP_PATTERNS:
        if pat.upper() in sym.upper():
            return False
    return True


def _bars_to_df(bars: list[dict]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame()
    rows = []
    for b in bars:
        ts = b.get("time") or b.get("t") or b.get("timestamp")
        if ts is None:
            continue
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            try:
                dt = pd.Timestamp(ts, tz="UTC")
            except Exception:
                continue
        rows.append({
            "timestamp": dt,
            "open":   float(b.get("open",  0) or 0),
            "high":   float(b.get("high",  0) or 0),
            "low":    float(b.get("low",   0) or 0),
            "close":  float(b.get("close", 0) or 0),
            "volume": int(b.get("volume",  0) or 0),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df[df["close"] > 0]


def _load_existing(symbol: str) -> pd.DataFrame:
    pf = PARQUET_DIR / f"{symbol}.parquet"
    if not pf.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(pf)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    except Exception:
        return pd.DataFrame()


def _save(symbol: str, df: pd.DataFrame) -> None:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    pf = PARQUET_DIR / f"{symbol}.parquet"
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Deduplicate to one bar per IST calendar date.
    # The data-service DB stores both PROVIDER and DERIVED (aggregated) 1d bars
    # which have different UTC timestamps but map to the same NSE trading session.
    # We keep the LAST bar per IST date (= the closing bar).
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    ist_dates = df.index.tz_convert("Asia/Kolkata").normalize()
    if ist_dates.value_counts().max() > 1:
        groups: dict = {}
        for ts, row in df.sort_index().iterrows():
            ist_day = ts.tz_convert("Asia/Kolkata").normalize()
            groups[ist_day] = row
        clean_index = pd.DatetimeIndex(
            [d.tz_convert("UTC") for d in groups.keys()], name=df.index.name
        )
        df = pd.DataFrame(list(groups.values()), index=clean_index, columns=df.columns).sort_index()

    df.to_parquet(pf, index=True, compression="snappy")


async def _fetch_symbol(
    client: DataServiceClient,
    symbol: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            bars = await client.get_historical_ohlcv(
                symbol=symbol,
                exchange="NSE",
                interval="1d",
                from_date=from_date,
                to_date=to_date,
            )
            return _bars_to_df(bars)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                log.warning("refresh_fetch_failed", symbol=symbol, error=str(exc))
                return pd.DataFrame()
            await asyncio.sleep(RETRY_SLEEP * attempt)
    return pd.DataFrame()


async def refresh_all(
    symbols: list[str],
    dry_run: bool = False,
) -> dict[str, str]:
    """Refresh all parquets. Returns {symbol: status}."""
    today_str = date.today().isoformat()
    results: dict[str, str] = {}

    print(f"\n{'='*65}")
    print(f"  DAILY PARQUET REFRESH")
    print(f"  Symbols:    {len(symbols)}")
    print(f"  From:       {HISTORY_FROM}")
    print(f"  To:         {today_str}")
    print(f"  Output:     {PARQUET_DIR.resolve()}")
    if dry_run:
        print(f"  DRY RUN — no files written")
    print(f"{'='*65}\n")

    client = DataServiceClient()
    await client.connect()

    created = updated = skipped = failed = 0
    t0 = time.monotonic()

    for i, symbol in enumerate(symbols, 1):
        existing = _load_existing(symbol)

        if not existing.empty:
            last_date = existing.index.max()
            days_behind = (datetime.now(tz=timezone.utc) - last_date).days
            if days_behind <= 2:
                print(f"  [{i:3d}/{len(symbols)}] {symbol:20s}  SKIP (up-to-date, {len(existing)} bars)")
                results[symbol] = "skip"
                skipped += 1
                continue
            fetch_from = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            fetch_from = HISTORY_FROM

        print(f"  [{i:3d}/{len(symbols)}] {symbol:20s}  fetching {fetch_from}→{today_str}...",
              end="", flush=True)

        if dry_run:
            print(f"  DRY RUN")
            continue

        new_df = await _fetch_symbol(client, symbol, fetch_from, today_str)
        await asyncio.sleep(RATE_LIMIT_SLEEP)

        if new_df.empty:
            print(f"  ✗ no data")
            results[symbol] = "failed"
            failed += 1
            continue

        if not existing.empty:
            merged = pd.concat([existing, new_df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        else:
            merged = new_df

        if len(merged) < MIN_ROWS:
            print(f"  ✗ only {len(merged)} bars (<{MIN_ROWS} min)")
            results[symbol] = "too_few"
            failed += 1
            continue

        _save(symbol, merged)
        action = "created" if existing.empty else "updated"
        added  = len(new_df)
        total  = len(merged)

        elapsed = time.monotonic() - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta  = (len(symbols) - i) / rate if rate > 0 else 0
        print(f"  +{added:4d} bars  total={total:5d}  ETA={eta/60:.1f}m  [{action}]")
        results[symbol] = action
        if action == "created":
            created += 1
        else:
            updated += 1

    await client.disconnect()

    elapsed = time.monotonic() - t0
    print(f"\n{'='*65}")
    print(f"  Done: created={created} updated={updated} skipped={skipped} failed={failed}")
    print(f"  Total time: {elapsed/60:.1f}m")
    print(f"{'='*65}\n")
    return results


def _db_symbols() -> list[str]:
    """Query the DB for all tradeable NSE equity symbols with 1d bars."""
    import subprocess
    cmd = [
        "docker", "exec", "data-service-postgres",
        "psql", "-h", "127.0.0.1", "-p", "5432",
        "-U", "mds_user", "-d", "mds", "-t", "-A",
        "-c", """
SELECT REPLACE(instrument_id, 'NSE:', '')
FROM equity_candle
WHERE interval_str = '1d' AND exchange = 'NSE'
GROUP BY instrument_id
HAVING COUNT(*) >= 252
ORDER BY instrument_id;
"""
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    syms = [s.strip() for s in result.stdout.strip().splitlines() if s.strip()]
    return [s for s in syms if _is_tradeable(s)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh 1d parquets from data-service DB")
    parser.add_argument("--symbols", nargs="+", help="Override symbol list")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--existing-only", action="store_true",
                        help="Only refresh already on-disk symbols (no new ones)")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s for s in args.symbols if _is_tradeable(s)]
    elif args.existing_only:
        symbols = sorted(p.stem for p in PARQUET_DIR.glob("*.parquet"))
    else:
        # All DB symbols + existing on-disk symbols
        db_syms   = _db_symbols()
        disk_syms = [p.stem for p in PARQUET_DIR.glob("*.parquet")]
        symbols   = sorted(set(db_syms) | set(disk_syms))
        # Keep only tradeable
        symbols   = [s for s in symbols if _is_tradeable(s)]

    print(f"Universe: {len(symbols)} symbols")
    asyncio.run(refresh_all(symbols, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
