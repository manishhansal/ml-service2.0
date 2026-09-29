#!/usr/bin/env python3
"""Test MarketEngine live quote from inside data-service container."""
import asyncio
import sys
sys.path.insert(0, "/app")


async def test():
    from src.core.settings import get_settings
    import redis.asyncio as aioredis
    from src.providers.adapters.angel_one import AngelOneAdapter
    from src.engines.market_engine import MarketEngine

    s = get_settings()
    redis_client = aioredis.from_url(s.redis_url, decode_responses=True)
    adapter = AngelOneAdapter(
        api_key=s.angel_one_api_key,
        client_id=s.angel_one_client_id,
        totp_secret=s.angel_one_totp_secret,
        mpin=s.angel_one_mpin,
        redis_client=redis_client,
    )
    await adapter.ensure_authenticated()
    print("Angel One: authenticated")

    engine = MarketEngine(angel_one_adapter=adapter)
    symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "INFY", "TCS"]
    for sym in symbols:
        try:
            result = await engine.get_live_quote(sym, "NSE")
            if result:
                print(f"{sym}: LTP={result.get('ltp')} chg={result.get('changePct')}% prov={result.get('provider')}")
            else:
                print(f"{sym}: None returned")
        except Exception as e:
            print(f"{sym}: ERROR {type(e).__name__} {str(e)[:80]}")


asyncio.run(test())
