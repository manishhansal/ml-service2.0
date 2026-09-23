#!/usr/bin/env python3
"""
scripts/generate_dummy_data.py

Generates synthetic OHLCV and SentinelPulse fixtures for local development,
manual testing, and ML research notebooks.

Usage:
    python scripts/generate_dummy_data.py --output-dir artifacts/ --n-bars 500

Output files:
    artifacts/dummy_ohlcv_NIFTY.csv       — 500 daily OHLCV bars
    artifacts/dummy_sentinel_NIFTY.json   — 30 SentinelPulse news contexts
    artifacts/dummy_feature_vectors.json  — 30 FeatureVector snapshots (no ML logic)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── OHLCV generation ─────────────────────────────────────────────────────────


def generate_ohlcv(
    symbol: str,
    n_bars: int,
    start_price: float = 18000.0,
    seed: int = 42,
) -> list[dict]:
    """Generate ``n_bars`` synthetic daily OHLCV bars for ``symbol``."""
    random.seed(seed)
    bars = []
    close = start_price
    base_volume = 1_500_000.0
    start_date = datetime(2020, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    for i in range(n_bars):
        # Random walk with slight upward drift
        daily_return = random.gauss(0.0003, 0.012)
        close = max(close * (1 + daily_return), 100.0)

        open_ = round(close * (1 + random.uniform(-0.003, 0.003)), 2)
        high = round(max(open_, close) * (1 + random.uniform(0.001, 0.01)), 2)
        low = round(min(open_, close) * (1 - random.uniform(0.001, 0.01)), 2)
        volume = round(base_volume * random.uniform(0.6, 1.8), 0)
        vwap = round((open_ + high + low + close) / 4, 2)

        # Skip weekends (approximate)
        ts = start_date + timedelta(days=i + i // 5 * 2)

        bars.append({
            "timestamp": ts.isoformat(),
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": round(close, 2),
            "volume": volume,
            "vwap": vwap,
        })

    return bars


# ── SentinelPulse generation ──────────────────────────────────────────────────


def generate_sentinel_contexts(
    symbol: str,
    n_contexts: int,
    seed: int = 42,
) -> list[dict]:
    """Generate ``n_contexts`` synthetic SentinelPulse news contexts."""
    random.seed(seed + 1)
    contexts = []
    directions = ["BULLISH", "BEARISH", "NEUTRAL"]
    regimes = ["RISK_ON", "RISK_OFF", "NEUTRAL", "VOLATILE"]
    event_tags = ["RBI_POLICY", "FII_INFLOW", "EARNINGS", "GLOBAL_SELLOFF", "IPO", "RESULTS"]

    for i in range(n_contexts):
        direction = random.choice(directions)
        impact_score = round(random.uniform(0.05, 0.95), 3)
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)

        contexts.append({
            "instrument": symbol,
            "as_of": ts.isoformat(),
            "news_impact_score": impact_score,
            "impact_direction": direction,
            "impact_confidence": round(random.uniform(0.50, 0.95), 3),
            "sentiment": {
                "overall": round(random.uniform(-0.8, 0.8), 3),
                "market": round(random.uniform(-0.6, 0.6), 3),
                "company": round(random.uniform(-0.7, 0.7), 3),
                "macro": round(random.uniform(-0.5, 0.5), 3),
                "risk": round(random.uniform(-0.6, 0.6), 3),
            },
            "market_regime": random.choice(regimes),
            "event_tags": random.sample(event_tags, k=random.randint(0, 2)),
        })

    return contexts


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dummy market data fixtures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to write output files (default: artifacts/)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="NIFTY",
        help="Instrument symbol (default: NIFTY)",
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=500,
        help="Number of OHLCV bars to generate (default: 500)",
    )
    parser.add_argument(
        "--n-contexts",
        type=int,
        default=30,
        help="Number of SentinelPulse contexts to generate (default: 30)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # OHLCV
    bars = generate_ohlcv(args.symbol, args.n_bars)
    ohlcv_path = output_dir / f"dummy_ohlcv_{args.symbol}.json"
    ohlcv_path.write_text(json.dumps(bars, indent=2))
    print(f"✓ Wrote {len(bars)} OHLCV bars → {ohlcv_path}")

    # SentinelPulse
    contexts = generate_sentinel_contexts(args.symbol, args.n_contexts)
    sentinel_path = output_dir / f"dummy_sentinel_{args.symbol}.json"
    sentinel_path.write_text(json.dumps(contexts, indent=2))
    print(f"✓ Wrote {len(contexts)} SentinelPulse contexts → {sentinel_path}")

    print("\nDone. Use these fixtures with mock clients in tests or notebooks.")


if __name__ == "__main__":
    main()
