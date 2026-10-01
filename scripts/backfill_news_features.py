#!/usr/bin/env python3
"""
scripts/backfill_news_features.py
-----------------------------------
Backfill Group H (news/NLP) features from SentinelPulse into
data/news/1d/{symbol}.parquet and data/news/1d/MARKET.parquet.

Architecture
-------------
SentinelPulse ingests news continuously.  This script:
  1. Fetches the current market-level sentiment and saves it as today's row
     in data/news/1d/MARKET.parquet.
  2. For each symbol in the F&O universe, fetches the per-asset sentiment
     (endpoint: /api/v1/alphaforge/context/asset/NSE:{symbol}) and appends
     today's row to data/news/1d/{symbol}.parquet.
  3. Fetches the training events list (importance scores) and records the
     aggregate per day.

Run this script once a day (after market close) to accumulate history.
The DatasetBuilder reads these parquets and forward-fills news features
into the training dataset.  Dates before the first row get 0.0 (neutral).

Usage::
    PYTHONPATH=. python3 scripts/backfill_news_features.py
    PYTHONPATH=. python3 scripts/backfill_news_features.py --date 2026-09-30
    PYTHONPATH=. python3 scripts/backfill_news_features.py --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone, date as _date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clients.sentinel_pulse import SentinelPulseClient
from src.features.families.news import (
    NEWS_FEATURE_NAMES,
    news_features_from_context,
)
from src.logging_config import get_logger

log = get_logger(__name__)

NEWS_DIR     = Path("data/news/1d")
DAILY_DIR    = Path("data/1d/1d")
MARKET_SYM   = "MARKET"
RATE_SLEEP   = 0.25   # seconds between per-asset calls to avoid hammering SP
MAX_RETRIES  = 2


def _load_universe() -> list[str]:
    """Read symbol list from daily parquet stems."""
    return sorted(p.stem for p in DAILY_DIR.glob("*.parquet"))


def _save_row(symbol: str, row_date: _date, feats: dict[str, float]) -> None:
    """Append or update one row in data/news/1d/{symbol}.parquet."""
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    pf = NEWS_DIR / f"{symbol}.parquet"
    ts = pd.Timestamp(row_date).tz_localize("UTC")

    row_df = pd.DataFrame([feats], index=[ts], columns=NEWS_FEATURE_NAMES)
    row_df.index.name = "timestamp"

    if pf.exists():
        existing = pd.read_parquet(pf)
        if existing.index.tz is None:
            existing.index = existing.index.tz_localize("UTC")
        # Upsert: replace if date already exists, else append
        existing = existing[existing.index.normalize() != ts.normalize()]
        combined = pd.concat([existing, row_df]).sort_index()
    else:
        combined = row_df

    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(pf, index=True, compression="snappy")


async def _fetch_with_fallback(
    client: SentinelPulseClient,
    path: str,
    params: dict | None = None,
) -> dict | list | None:
    """Call _fetch_with_retry up to MAX_RETRIES times."""
    for attempt in range(MAX_RETRIES):
        result = await client._fetch_with_retry(path, params=params)
        if result is not None:
            return result
        await asyncio.sleep(0.5 * (attempt + 1))
    return None


async def backfill_date(
    row_date: _date,
    universe: list[str],
    dry_run: bool = False,
) -> dict[str, int]:
    """Backfill news features for one trading date. Returns {symbol: 1 if ok}."""
    client = SentinelPulseClient()
    await client.connect()
    results: dict[str, int] = {}
    t0 = time.monotonic()

    print(f"\n{'='*60}")
    print(f"  NEWS FEATURE BACKFILL — {row_date}")
    print(f"  Universe: {len(universe)} symbols")
    if dry_run:
        print("  DRY RUN — no files written")
    print(f"{'='*60}\n")

    # ── 1. Market-level features ──────────────────────────────────────────
    print("  [mkt] Fetching market context...", end="", flush=True)
    market_ctx = await _fetch_with_fallback(
        client, "/api/v1/alphaforge/context/market"
    )
    events_raw = await _fetch_with_fallback(
        client, "/api/v1/ml/training/events",
        params={"limit": "100"},
    )
    events_list: list[dict] = []
    if isinstance(events_raw, list):
        events_list = events_raw
    elif isinstance(events_raw, dict):
        events_list = events_raw.get("events", []) or []

    market_feats = news_features_from_context(
        market_ctx=market_ctx,
        asset_ctx=None,
        events=events_list,
    )

    if not dry_run:
        _save_row(MARKET_SYM, row_date, market_feats)

    mkt_summary = (
        f"mkt={market_feats['news_market_sentiment']:+.2f} "
        f"macro={market_feats['news_macro_sentiment']:+.2f} "
        f"risk={market_feats['news_risk_sentiment']:+.2f} "
        f"regime={market_feats['news_regime_score']:+.1f} "
        f"events={market_feats['news_event_importance']:.2f}"
    )
    print(f"  done  {mkt_summary}")
    results[MARKET_SYM] = 1

    # ── 2. Per-asset features ─────────────────────────────────────────────
    print(f"  Fetching per-asset context for {len(universe)} symbols...\n")
    saved = skipped = failed = 0

    for i, symbol in enumerate(universe, 1):
        nse_id = f"NSE:{symbol}"
        asset_ctx = await _fetch_with_fallback(
            client, f"/api/v1/alphaforge/context/asset/{nse_id}"
        )
        await asyncio.sleep(RATE_SLEEP)

        # Merge: market feats + asset-specific override
        asset_feats = news_features_from_context(
            market_ctx=market_ctx,
            asset_ctx=asset_ctx,
            events=events_list,
        )

        # Check if asset has real data (non-zero asset sentiment OR articles exist)
        has_data = (
            asset_ctx is not None
            and (
                (asset_ctx.get("article_count") or 0) > 0
                or asset_ctx.get("sentiment_summary", {}).get("overall") is not None
            )
        )

        if not dry_run:
            _save_row(symbol, row_date, asset_feats)
            saved += 1
        else:
            skipped += 1

        if has_data:
            print(
                f"  [{i:3d}/{len(universe)}] {symbol:20s}  "
                f"asset={asset_feats['news_asset_sentiment']:+.3f}  ✓ has articles"
            )
        elif i % 50 == 0 or i == len(universe):
            elapsed = time.monotonic() - t0
            eta = (len(universe) - i) * (elapsed / i)
            print(
                f"  [{i:3d}/{len(universe)}] {symbol:20s}  "
                f"neutral fill  ETA={eta/60:.1f}m"
            )
        results[symbol] = 1

    elapsed = time.monotonic() - t0
    print(f"\n  Done: saved={saved} skipped={skipped} failed={failed}  "
          f"time={elapsed:.1f}s")
    await client.disconnect()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill SentinelPulse news features")
    parser.add_argument("--date", default=None,
                        help="Date to backfill (YYYY-MM-DD, default: today)")
    parser.add_argument("--symbols", nargs="+",
                        help="Override symbol list")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    row_date = _date.fromisoformat(args.date) if args.date else _date.today()
    universe = args.symbols if args.symbols else _load_universe()

    asyncio.run(backfill_date(row_date, universe, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
