#!/usr/bin/env python3
"""
scripts/backfill_intraday.py
-----------------------------
Backfill 5-minute OHLCV bars for every symbol in the universe plus key
sector indices.  Saves one parquet per symbol under data/5m/5m/{symbol}.parquet

Design:
  - Reads the universe from data/1d/1d/*.parquet (same 218 symbols as daily model)
  - Also backfills sector indices: NIFTY, NSEBANK, CNXIT, MIDCPNIFTY
  - Fetches 5m bars in 28-day chunks to respect data-service rate limits
  - Merges with any existing parquet on disk (idempotent re-runs)
  - Respects BACKFILL_LOOKBACK_DAYS env var (default 1000 days ≈ 4yr of 5m data)

Usage::
    PYTHONPATH=. python3 scripts/backfill_intraday.py
    PYTHONPATH=. python3 scripts/backfill_intraday.py --symbols RELIANCE TCS
    PYTHONPATH=. python3 scripts/backfill_intraday.py --lookback-days 730
    PYTHONPATH=. python3 scripts/backfill_intraday.py --dry-run   # show plan only
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clients.data_service import DataServiceClient
from src.logging_config import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DAILY_DIR   = Path("data/1d/1d")
INTRADAY_DIR = Path("data/5m/5m")
INTERVAL    = "5m"
CHUNK_DAYS  = 28          # fetch window per API call (keeps payloads manageable)
RATE_LIMIT_SLEEP = 0.4    # seconds between API calls to avoid 429s
MAX_RETRIES  = 3
RETRY_SLEEP  = 5.0

# Sector indices to include even though they're not in the F&O universe
SECTOR_INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "CNXIT", "MIDCPNIFTY"]

DEFAULT_LOOKBACK_DAYS = int(os.getenv("BACKFILL_LOOKBACK_DAYS", "1825"))  # 5 years


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_universe() -> list[str]:
    """Load symbol list from daily parquet stems."""
    if not DAILY_DIR.exists():
        raise FileNotFoundError(f"Daily parquet dir not found: {DAILY_DIR}")
    symbols = sorted(p.stem for p in DAILY_DIR.glob("*.parquet"))
    if not symbols:
        raise ValueError(f"No parquet files found in {DAILY_DIR}")
    return symbols


def _bars_to_df(bars: list[dict]) -> pd.DataFrame:
    """Convert raw bar dicts from data-service into a normalised DataFrame."""
    if not bars:
        return pd.DataFrame()
    rows = []
    for b in bars:
        ts = b.get("time") or b.get("timestamp") or b.get("t")
        if ts is None:
            continue
        # ts can be unix epoch int or ISO string
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            try:
                dt = pd.Timestamp(ts, tz="UTC") if "Z" in str(ts) or "+" in str(ts) \
                     else pd.Timestamp(ts).tz_localize("UTC")
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
    df = df[df["close"] > 0]   # drop zero-price bars (data errors)
    return df


def _load_existing(symbol: str) -> pd.DataFrame:
    """Load existing on-disk parquet if present."""
    pf = INTRADAY_DIR / f"{symbol}.parquet"
    if not pf.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(pf)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    except Exception as exc:
        log.warning("backfill_load_existing_failed", symbol=symbol, error=str(exc))
        return pd.DataFrame()


def _save(symbol: str, df: pd.DataFrame) -> None:
    """Persist merged DataFrame to parquet."""
    INTRADAY_DIR.mkdir(parents=True, exist_ok=True)
    pf = INTRADAY_DIR / f"{symbol}.parquet"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(pf, index=True, compression="snappy")


def _date_chunks(start: datetime, end: datetime, chunk_days: int) -> list[tuple[str, str]]:
    """Split [start, end] into non-overlapping CHUNK_DAYS windows."""
    chunks: list[tuple[str, str]] = []
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=chunk_days), end)
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + timedelta(days=1)
    return chunks


async def _fetch_symbol(
    client: DataServiceClient,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch all 5m bars for symbol between start_date and end_date, chunked."""
    chunks = _date_chunks(
        datetime.fromisoformat(start_date),
        datetime.fromisoformat(end_date),
        CHUNK_DAYS,
    )
    all_dfs: list[pd.DataFrame] = []

    for from_d, to_d in chunks:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                bars = await client.get_historical_ohlcv(
                    symbol=symbol,
                    exchange="NSE",
                    interval=INTERVAL,
                    from_date=from_d,
                    to_date=to_d,
                )
                df_chunk = _bars_to_df(bars)
                if not df_chunk.empty:
                    all_dfs.append(df_chunk)
                await asyncio.sleep(RATE_LIMIT_SLEEP)
                break
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    log.warning(
                        "backfill_chunk_failed",
                        symbol=symbol,
                        from_date=from_d,
                        to_date=to_d,
                        error=str(exc),
                    )
                else:
                    await asyncio.sleep(RETRY_SLEEP * attempt)

    if not all_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


async def backfill_all(
    symbols: list[str],
    lookback_days: int,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Backfill 5m data for all symbols.  Returns {symbol: bars_added}.
    """
    end_date   = datetime.now(tz=timezone.utc).date()
    start_date = end_date - timedelta(days=lookback_days)

    print(f"\n{'='*65}")
    print(f"  INTRADAY 5m BACKFILL — {INTERVAL} bars")
    print(f"  Window : {start_date} → {end_date}  ({lookback_days} days)")
    print(f"  Symbols: {len(symbols)}")
    print(f"  Output : {INTRADAY_DIR.resolve()}")
    if dry_run:
        print(f"  DRY RUN: no files will be written")
    print(f"{'='*65}\n")

    if dry_run:
        for sym in symbols:
            existing = _load_existing(sym)
            last_bar = existing.index.max() if len(existing) else "—"
            print(f"  {sym:20s}  existing={len(existing):6d} bars  last={last_bar}")
        return {}

    client = DataServiceClient()
    await client.connect()

    results: dict[str, int] = {}
    t0 = time.monotonic()

    for i, symbol in enumerate(symbols, 1):
        # Determine actual start: if we have existing data, only fetch new bars
        existing = _load_existing(symbol)
        hist_gap_df = pd.DataFrame()   # historical gap data to prepend

        if not existing.empty:
            last_existing = existing.index.max()
            # Forward gap: fetch bars AFTER the last existing bar
            fetch_start = (last_existing + timedelta(days=1)).strftime("%Y-%m-%d")

            # Historical gap: fetch bars BEFORE the first existing bar if the
            # parquet doesn't go back as far as the target start date.
            # This handles the case where a previous backfill used a shorter
            # lookback window (e.g. 6-month or 1000-day window).
            first_existing = existing.index.min()
            target_start_dt = datetime.fromisoformat(str(start_date))
            # If first bar is more than 14 days later than our target start → gap exists
            gap_days = (first_existing.replace(tzinfo=None) - target_start_dt).days
            if gap_days > 14:
                hist_end = (first_existing - timedelta(days=1)).strftime("%Y-%m-%d")
                hist_start = str(start_date)
                print(f"  [{i:3d}/{len(symbols)}] {symbol:20s}  hist gap {hist_start}→{hist_end}...", end="", flush=True)
                if not dry_run:
                    hist_gap_df = await _fetch_symbol(client, symbol, hist_start, hist_end)
                    if not hist_gap_df.empty:
                        print(f" +{len(hist_gap_df)} hist bars", end="", flush=True)
                    await asyncio.sleep(RATE_LIMIT_SLEEP)
        else:
            fetch_start = str(start_date)

        fetch_end = str(end_date)

        if fetch_start > fetch_end:
            if hist_gap_df.empty:
                print(f"  [{i:3d}/{len(symbols)}] {symbol:20s}  up-to-date ({len(existing)} bars)")
                results[symbol] = 0
                continue
            # Only historical gap was fetched
            merged = pd.concat([hist_gap_df, existing])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            _save(symbol, merged)
            added = len(hist_gap_df)
            print(f"  [{i:3d}/{len(symbols)}] {symbol:20s}  hist-only +{added} bars  total={len(merged)}")
            results[symbol] = added
            continue

        print(f"  [{i:3d}/{len(symbols)}] {symbol:20s}  fetching {fetch_start}→{fetch_end}...", end="", flush=True)

        if dry_run:
            print()
            continue

        new_bars = await _fetch_symbol(client, symbol, fetch_start, fetch_end)

        if new_bars.empty and hist_gap_df.empty:
            print(f"  ✗ no data returned")
            results[symbol] = 0
            continue

        # Merge: historical gap + existing + new forward bars
        parts = [p for p in [hist_gap_df, existing, new_bars] if not p.empty]
        merged = pd.concat(parts)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()

        _save(symbol, merged)
        added = len(new_bars) + len(hist_gap_df)
        total = len(merged)
        elapsed = time.monotonic() - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta_secs = (len(symbols) - i) / rate if rate > 0 else 0
        print(f"  +{added:5d} bars  total={total:7d}  ETA={eta_secs/60:.1f}m")
        results[symbol] = added

    await client.disconnect()

    ok  = sum(1 for v in results.values() if v > 0)
    tot = sum(results.values())
    elapsed = time.monotonic() - t0
    print(f"\n{'='*65}")
    print(f"  Done: {ok}/{len(symbols)} symbols  |  {tot:,} total bars added  |  {elapsed/60:.1f}m")
    print(f"{'='*65}\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill 5m intraday bars")
    parser.add_argument(
        "--symbols", nargs="+",
        help="Override symbol list (default: all 218 from data/1d/1d/)",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
        help=f"How many calendar days back to fetch (default {DEFAULT_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--include-indices", action="store_true", default=True,
        help="Also backfill sector indices (NIFTY, NSEBANK, CNXIT, etc.)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without fetching or writing",
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = list(args.symbols)
    else:
        symbols = _load_universe()
        if args.include_indices:
            for idx in SECTOR_INDICES:
                if idx not in symbols:
                    symbols.append(idx)

    asyncio.run(backfill_all(
        symbols=symbols,
        lookback_days=args.lookback_days,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
