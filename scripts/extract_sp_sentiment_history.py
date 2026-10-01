#!/usr/bin/env python3
"""
scripts/extract_sp_sentiment_history.py
-----------------------------------------
Extract historical per-symbol-per-date news sentiment directly from the
SentinelPulse PostgreSQL database and save to data/news/1d/{symbol}.parquet.

This gives the model REAL TEMPORAL variance in Group H news features:
  - Instead of one Sep-30 snapshot applied to all training rows
  - Each training date gets the ACTUAL sentiment from articles published
    on that date (or nearest prior date via forward-fill)

The query joins:
    news_articles   (published_at, source_id)
    news_asset_links (asset_id → NSE:SYMBOL format)
    news_sentiment  (sentiment_score, market_sentiment, company_sentiment,
                     macro_sentiment, risk_sentiment, confidence)
    news_events     (importanceScore for event_importance feature)

Usage::
    PYTHONPATH=. python3 scripts/extract_sp_sentiment_history.py
    PYTHONPATH=. python3 scripts/extract_sp_sentiment_history.py --min-date 2021-01-01
    PYTHONPATH=. python3 scripts/extract_sp_sentiment_history.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

NEWS_DIR = Path("data/news/1d")
DB_CONN  = "postgresql://sentinel:sentinel_dev@localhost:5444/sentinel_pulse"

# Regime score mapping from qualitative signals
_REGIME_MAP = {"bullish": 1.0, "bull": 1.0, "positive": 0.5,
               "neutral": 0.0, "bearish": -1.0, "bear": -1.0, "negative": -0.5}


def _psql(sql: str) -> str:
    """Run a SQL query via docker exec psql and return stdout."""
    result = subprocess.run(
        ["docker", "exec", "data-service-postgres",
         "psql", "-h", "127.0.0.1", "-p", "5432",
         "-U", "sentinel", "-d", "sentinel_pulse",
         "-t", "-A", "-F", "\t", "-c", sql],
        capture_output=True, text=True, timeout=120,
    )
    return result.stdout.strip()


def extract_symbol_history(min_date: str = "2021-01-01") -> pd.DataFrame:
    """
    Query the SP DB for all per-symbol-per-date sentiment aggregates.
    Returns a DataFrame with columns:
        symbol, date, news_market_sentiment, news_macro_sentiment,
        news_risk_sentiment, news_asset_sentiment, news_event_importance,
        article_count
    """
    print("Querying SentinelPulse DB for historical sentiment by symbol+date ...")
    sql = f"""
SELECT
    REPLACE(al.asset_id, 'NSE:', '') AS symbol,
    DATE(na.published_at)            AS news_date,
    AVG(COALESCE(ns.market_sentiment,  0))::float AS market_sent,
    AVG(COALESCE(ns.macro_sentiment,   0))::float AS macro_sent,
    AVG(COALESCE(ns.risk_sentiment,    0))::float AS risk_sent,
    AVG(COALESCE(ns.company_sentiment, 0))::float AS company_sent,
    AVG(COALESCE(ns.sentiment_score,   0))::float AS overall_sent,
    MAX(COALESCE(
        (SELECT ne.importance::float FROM news_events ne
         WHERE ne.article_id = na.id LIMIT 1), 0.0
    ))                                             AS event_imp,
    COUNT(*)                                       AS n_articles
FROM news_articles na
JOIN news_asset_links al ON na.id = al.article_id
JOIN news_sentiment   ns ON na.id = ns.article_id
WHERE na.published_at >= '{min_date}'
  AND al.asset_id LIKE 'NSE:%'
  AND al.confidence >= 0.5
GROUP BY 1, 2
ORDER BY 1, 2;
"""
    raw = _psql(sql)
    if not raw:
        print("  No data returned from DB query.")
        return pd.DataFrame()

    rows = []
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        try:
            sym, dt, mk, mc, rk, cp, ov, ei, na_ = parts
            rows.append({
                "symbol":       sym.strip(),
                "date":         dt.strip(),
                "market_sent":  float(mk  or 0),
                "macro_sent":   float(mc  or 0),
                "risk_sent":    float(rk  or 0),
                "company_sent": float(cp  or 0),
                "overall_sent": float(ov  or 0),
                "event_imp":    float(ei  or 0),
                "n_articles":   int(na_   or 0),
            })
        except (ValueError, TypeError):
            continue

    if not rows:
        print("  Query returned rows but failed to parse.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"  {len(df)} symbol-date rows from {df['symbol'].nunique()} symbols")
    return df


def extract_market_history(min_date: str = "2021-01-01") -> pd.DataFrame:
    """
    Extract market-level (cross-symbol) daily sentiment.
    Used for the MARKET parquet which provides a fallback for symbols
    without direct asset links.
    """
    print("Querying market-level sentiment history ...")
    sql = f"""
SELECT
    DATE(na.published_at)              AS news_date,
    AVG(COALESCE(ns.market_sentiment,  0))::float AS market_sent,
    AVG(COALESCE(ns.macro_sentiment,   0))::float AS macro_sent,
    AVG(COALESCE(ns.risk_sentiment,    0))::float AS risk_sent,
    AVG(COALESCE(ns.sentiment_score,   0))::float AS overall_sent,
    COUNT(*)                                       AS n_articles
FROM news_articles na
JOIN news_sentiment ns ON na.id = ns.article_id
WHERE na.published_at >= '{min_date}'
GROUP BY 1
ORDER BY 1;
"""
    raw = _psql(sql)
    if not raw:
        return pd.DataFrame()

    rows = []
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        try:
            dt, mk, mc, rk, ov, na_ = parts
            rows.append({
                "date":         dt.strip(),
                "market_sent":  float(mk  or 0),
                "macro_sent":   float(mc  or 0),
                "risk_sent":    float(rk  or 0),
                "overall_sent": float(ov  or 0),
                "n_articles":   int(na_   or 0),
            })
        except (ValueError, TypeError):
            continue

    df = pd.DataFrame(rows)
    print(f"  {len(df)} market-level rows")
    return df


def _to_news_parquet(
    sym_df: pd.DataFrame,
    symbol: str,
    market_df: pd.DataFrame,
) -> None:
    """Save per-symbol parquet with Group H feature columns."""
    from src.features.families.news import NEWS_FEATURE_NAMES

    # Build UTC-indexed DataFrame
    rows = []
    for _, row in sym_df.iterrows():
        ts = pd.Timestamp(row["date"]).tz_localize("UTC")
        rows.append({
            "timestamp":             ts,
            "news_market_sentiment": float(np.clip(row["market_sent"], -1, 1)),
            "news_macro_sentiment":  float(np.clip(row["macro_sent"],  -1, 1)),
            "news_risk_sentiment":   float(np.clip(row["risk_sent"],   -1, 1)),
            "news_regime_score":     0.0,   # populated below from overall
            "news_asset_sentiment":  float(np.clip(row["company_sent"], -1, 1)),
            "news_event_importance": float(np.clip(row.get("event_imp", 0), 0, 1)),
        })
        # Rough regime from overall sentiment
        ov = float(row.get("overall_sent", 0))
        rows[-1]["news_regime_score"] = 1.0 if ov > 0.3 else (-1.0 if ov < -0.3 else 0.0)

    if not rows:
        return

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df = df[NEWS_FEATURE_NAMES]
    df = df[~df.index.duplicated(keep="last")]

    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    pf = NEWS_DIR / f"{symbol}.parquet"

    # Merge with any existing rows (e.g., today's snapshot from backfill_news_features)
    if pf.exists():
        existing = pd.read_parquet(pf)
        if existing.index.tz is None:
            existing.index = existing.index.tz_localize("UTC")
        # Keep existing for dates not in the DB query (forward data)
        combined = pd.concat([df, existing[~existing.index.isin(df.index)]])
        combined = combined.sort_index()
    else:
        combined = df

    combined.to_parquet(pf, index=True, compression="snappy")


def _market_parquet(market_df: pd.DataFrame) -> None:
    """Save market-level MARKET.parquet."""
    from src.features.families.news import NEWS_FEATURE_NAMES
    if market_df.empty:
        return

    rows = []
    for _, row in market_df.iterrows():
        ts = pd.Timestamp(row["date"]).tz_localize("UTC")
        ov = float(row.get("overall_sent", 0))
        rows.append({
            "timestamp":             ts,
            "news_market_sentiment": float(np.clip(row["market_sent"], -1, 1)),
            "news_macro_sentiment":  float(np.clip(row["macro_sent"],  -1, 1)),
            "news_risk_sentiment":   float(np.clip(row["risk_sent"],   -1, 1)),
            "news_regime_score":     1.0 if ov > 0.3 else (-1.0 if ov < -0.3 else 0.0),
            "news_asset_sentiment":  0.0,   # no specific asset for MARKET
            "news_event_importance": 0.0,
        })

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df = df[NEWS_FEATURE_NAMES]
    df = df[~df.index.duplicated(keep="last")]

    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    pf = NEWS_DIR / "MARKET.parquet"
    if pf.exists():
        existing = pd.read_parquet(pf)
        if existing.index.tz is None:
            existing.index = existing.index.tz_localize("UTC")
        combined = pd.concat([df, existing[~existing.index.isin(df.index)]])
        combined = combined.sort_index()
    else:
        combined = df
    combined.to_parquet(pf, index=True, compression="snappy")
    print(f"  MARKET.parquet: {len(combined)} rows  "
          f"({combined.index.min().date()} → {combined.index.max().date()})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-date", default="2021-01-01")
    p.add_argument("--dry-run",  action="store_true")
    args = p.parse_args()

    t0 = time.monotonic()
    print("=" * 60)
    print("  EXTRACTING HISTORICAL SP SENTIMENT → news parquets")
    print(f"  From: {args.min_date}")
    print("=" * 60)

    # 1. Extract market-level history
    market_df = extract_market_history(args.min_date)
    if not market_df.empty and not args.dry_run:
        _market_parquet(market_df)

    # 2. Extract per-symbol history
    sym_df = extract_symbol_history(args.min_date)
    if sym_df.empty:
        print("No per-symbol data found. Check asset_links table.")
        return

    symbols = sym_df["symbol"].unique()
    print(f"\nSaving {len(symbols)} symbol parquets ...")
    saved = 0
    for sym in sorted(symbols):
        rows = sym_df[sym_df["symbol"] == sym]
        if args.dry_run:
            print(f"  [DRY] {sym:15s}: {len(rows)} dates  "
                  f"({rows['date'].min()} → {rows['date'].max()})")
            saved += 1
            continue
        _to_news_parquet(rows, sym, market_df)
        saved += 1
        if saved % 5 == 0 or saved == len(symbols):
            print(f"  {saved}/{len(symbols)} saved  ({sym})")

    elapsed = time.monotonic() - t0
    print(f"\nDone: {saved} symbol parquets saved in {elapsed:.1f}s")

    # 3. Report coverage
    if not args.dry_run:
        from pathlib import Path as _P
        total_rows = 0
        min_dates  = []
        for pf in _P("data/news/1d").glob("*.parquet"):
            df2 = pd.read_parquet(pf)
            total_rows += len(df2)
            if len(df2) > 0:
                min_dates.append(df2.index.min())
        earliest = min(min_dates).date() if min_dates else "N/A"
        print(f"\nTotal news parquet rows: {total_rows:,}  earliest: {earliest}")
        print("Re-run train_expanded_features.py to use updated news features.")


if __name__ == "__main__":
    main()
