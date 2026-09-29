#!/usr/bin/env python3
"""Test Angel One live connectivity from inside data-service container."""
import asyncio
import sys
sys.path.insert(0, "/app")


async def test():
    from src.core.settings import get_settings
    import redis.asyncio as aioredis
    from src.providers.adapters.angel_one import AngelOneAdapter

    s = get_settings()
    redis_client = aioredis.from_url(s.redis_url, decode_responses=True)

    # Force fresh auth
    key = f"mds:angel_one:jwt:{s.angel_one_client_id}"
    await redis_client.delete(key)
    print("Cleared old JWT, doing fresh auth...")

    adapter = AngelOneAdapter(
        api_key=s.angel_one_api_key,
        client_id=s.angel_one_client_id,
        totp_secret=s.angel_one_totp_secret,
        mpin=s.angel_one_mpin,
        redis_client=redis_client,
    )
    await adapter.ensure_authenticated()
    print("Auth complete")

    # Try fetching NIFTY quote via data-service internal quote endpoint
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(
            "http://localhost:8200/v1/india/quotes/NIFTY",
            headers={"X-API-KEY": s.consumer_api_keys.split(",")[0] if s.consumer_api_keys else "dev-key-local-1"},
        )
        data = r.json()
        dd = data.get("data", {}) or {}
        print(f"NIFTY via API: LTP={dd.get('ltp')} provider={dd.get('provider')} status={dd.get('marketStatus')}")

    # Also try direct Angel One quote
    try:
        from src.engines.market import MarketEngine
        engine = MarketEngine(angel_one_adapter=adapter)
        result = await engine.get_live_quote("NIFTY", "NSE")
        if result:
            print(f"NIFTY via MarketEngine: LTP={result.get('ltp')}")
        else:
            print("MarketEngine returned None")
    except Exception as e:
        print(f"MarketEngine error: {type(e).__name__}: {str(e)[:100]}")


asyncio.run(test())
