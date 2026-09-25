"""
scripts/run_readiness_gate.py — run the training-readiness gate against live services.

Usage (inside Docker):
    python3 scripts/run_readiness_gate.py [--timeframe 1d] [--news-required] [--min-history 252]

Exit codes:
    0 = READY or READY_MARKET_ONLY
    1 = NOT_READY (blockers present)

Note on rate limits: data-service2.0 enforces 100 req/60s.
This script minimises calls — it uses a fixed default universe list rather
than pre-fetching fno-universe, relying on the gate's _check_universe call
as the single fno-universe fetch.

Mandate §27, §28.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

sys.path.insert(0, "/app")

from src.clients.data_service import DataServiceClient
from src.clients.sentinel_pulse import SentinelPulseClient
from src.data.readiness import TrainingReadinessGate


async def main(
    timeframe: str = "1d",
    min_history_days: int = 252,
    news_required: bool = False,
    min_universe_size: int = 3,
) -> int:
    data_client = DataServiceClient()
    await data_client.connect()

    sentinel_client = SentinelPulseClient()
    await sentinel_client.connect()

    gate = TrainingReadinessGate()

    # Use a well-known default universe list.
    # The gate's _check_universe will call fno-universe (single call).
    # We deliberately avoid a separate pre-fetch to prevent double-hitting the
    # data-service rate limit (100 req/60s window).
    universe = [
        "NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "ICICIBANK",
        "INFY", "TCS", "SBIN", "AXISBANK", "KOTAKBANK",
    ]
    print(f"Checking universe membership for: {universe[:5]}...", flush=True)

    result = await gate.check(
        data_client=data_client,
        sentinel_client=sentinel_client,
        universe=universe,
        timeframe=timeframe,
        min_history_days=min_history_days,
        news_required=news_required,
        min_universe_size=min_universe_size,
    )

    await data_client.disconnect()
    await sentinel_client.disconnect()

    print(json.dumps(result.to_dict(), indent=2, default=str), flush=True)

    # Machine-readable summary
    print("\n" + "=" * 60, flush=True)
    print(f"TRAINING_READINESS: {result.mode}", flush=True)
    if result.training_ready:
        print("STATUS: READY", flush=True)
        print(f"NEWS:   {result.news_status}", flush=True)
    else:
        print("STATUS: NOT_READY", flush=True)
        for b in result.blockers:
            print(f"BLOCKER: {b}", flush=True)
    if result.warnings:
        for w in result.warnings:
            print(f"WARNING: {w}", flush=True)
    print("=" * 60, flush=True)

    return 0 if result.training_ready else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training-readiness gate")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--min-history", type=int, default=252)
    parser.add_argument("--news-required", action="store_true", default=False)
    parser.add_argument("--min-universe", type=int, default=3)
    args = parser.parse_args()

    rc = asyncio.run(
        main(
            timeframe=args.timeframe,
            min_history_days=args.min_history,
            news_required=args.news_required,
            min_universe_size=args.min_universe,
        )
    )
    sys.exit(rc)
