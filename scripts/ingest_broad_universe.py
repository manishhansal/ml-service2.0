"""
scripts/ingest_broad_universe.py — Ingest all 220 F&O symbols with rate-limit patience.

Strategy:
- Ingest 5 symbols at a time
- Wait 15 seconds between batches (gives rate limit headroom)
- Resume from checkpoint (skips already-ingested symbols)
- Stops on any auth failure (not rate limit)

Usage:
    python3 scripts/ingest_broad_universe.py --interval 1d [--max-symbols 220]
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

from src.clients.data_service import (
    DataServiceClient,
    DataServiceAuthError,
    DataServiceRateLimitedError,
    DataServiceUnavailableError,
)
from src.data.ingestion import DataIngestionPipeline


async def main(interval: str = "1d", max_symbols: int = 220) -> None:
    DS_URL  = os.environ.get("DATA_SERVICE_2_URL", "http://host.docker.internal:8200")
    DS_KEY  = os.environ.get("DATA_SERVICE_API_KEY", "")

    client = DataServiceClient(base_url=DS_URL, api_key=DS_KEY)
    await client.connect()

    # Fetch universe
    print("Fetching F&O universe...", flush=True)
    for attempt in range(5):
        try:
            resp = await client.get_fno_universe()
            symbols_raw = resp.get("data", {}).get("symbols", [])
            symbols = [s if isinstance(s, str) else str(s) for s in symbols_raw[:max_symbols]]
            if "NIFTY" not in symbols:
                symbols = ["NIFTY"] + symbols
            print(f"Universe: {len(symbols)} symbols", flush=True)
            break
        except DataServiceRateLimitedError as e:
            wait = (e.retry_after_seconds or 10.0) + 5.0
            print(f"  Rate limited, waiting {wait:.0f}s...", flush=True)
            await asyncio.sleep(wait)
    else:
        print("ERROR: Could not fetch universe after 5 attempts", flush=True)
        sys.exit(1)

    output_dir = Path("/app/data") / interval
    pipeline = DataIngestionPipeline(
        data_client=client,
        output_root=output_dir,
        max_concurrency=2,     # conservative: 2 concurrent
        rate_per_sec=2.0,      # conservative: 2/sec
    )

    # Load checkpoint to see what's already done
    checkpoint_path = output_dir / "_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        completed_keys = set(checkpoint.get("completed", {}).keys())
        already_done = {k.split(":")[1] for k in completed_keys if k.startswith(f"{interval}:")}
    else:
        already_done = set()

    remaining = [s for s in symbols if s not in already_done]
    print(f"Already ingested: {len(already_done)}, Remaining: {len(remaining)}", flush=True)

    if not remaining:
        print("All symbols already ingested!", flush=True)
        await client.disconnect()
        return

    # Ingest one symbol at a time with intelligent rate-limit handling.
    # Key insight: the DataServiceClient circuit breaker opens after 3 consecutive
    # failures and stays open for 60 seconds. We must detect circuit-open state
    # and wait 65s before retrying.
    symbol_sleep = 2.0      # baseline sleep between symbols
    circuit_sleep = 70.0    # wait when circuit opened or sustained rate limit
    consecutive_failures = 0
    total_ingested = len(already_done)
    total_failed = 0
    total_syms = len(remaining)

    for i, sym in enumerate(remaining, start=1):
        print(f"[{i}/{total_syms}] {sym}...", end=" ", flush=True)

        try:
            result = await pipeline.ingest(
                symbols=[sym],
                interval=interval,
                from_date="2019-01-01",
                to_date=None,
                exchange="NSE",
                resume=True,
            )
            if result.symbols_ingested:
                bars = result.reports[0].rows_clean if result.reports else 0
                total_ingested += 1
                consecutive_failures = 0
                print(f"OK ({bars} bars). Done: {total_ingested}/{len(symbols)}", flush=True)
            else:
                consecutive_failures += 1
                total_failed += 1
                print(f"FAILED ({consecutive_failures} consecutive). Done: {total_ingested}/{len(symbols)}", flush=True)
                if consecutive_failures >= 3:
                    # Circuit is likely open — wait for it to reset
                    print(f"  Circuit likely open. Waiting {circuit_sleep:.0f}s...", flush=True)
                    await asyncio.sleep(circuit_sleep)
                    consecutive_failures = 0
        except DataServiceAuthError:
            print("ERROR: Auth failure — stopping ingestion", flush=True)
            break
        except DataServiceRateLimitedError as exc:
            wait = (exc.retry_after_seconds or 10.0) + 5.0
            consecutive_failures += 1
            total_failed += 1
            print(f"RATE_LIMITED. Waiting {wait:.0f}s...", flush=True)
            await asyncio.sleep(wait)
            if consecutive_failures >= 3:
                print(f"  Sustained rate limit. Waiting {circuit_sleep:.0f}s for circuit reset...", flush=True)
                await asyncio.sleep(circuit_sleep)
                consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            total_failed += 1
            print(f"ERROR: {e}", flush=True)

        # Normal inter-symbol sleep (only if circuit was not opened)
        if i < total_syms and consecutive_failures < 3:
            await asyncio.sleep(symbol_sleep)

    await client.disconnect()
    print(f"\nIngestion complete: {total_ingested}/{len(symbols)} symbols, {total_failed} failed", flush=True)

    # Summary of what's in the data directory
    data_files = list((output_dir / interval).glob("*.parquet")) if (output_dir / interval).exists() else []
    print(f"Parquet files on disk: {len(data_files)}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--interval", default="1d")
    p.add_argument("--max-symbols", type=int, default=220)
    args = p.parse_args()
    asyncio.run(main(args.interval, args.max_symbols))
