#!/usr/bin/env python3
"""Test Angel One live quote directly from inside data-service container."""
import asyncio
import sys
sys.path.insert(0, "/app")


async def test():
    from src.core.settings import get_settings
    s = get_settings()
    print(f"client_id: {s.angel_one_client_id[:4] if s.angel_one_client_id else 'NOT SET'}")
    print(f"api_key set: {bool(s.angel_one_api_key)}")

    from src.providers.adapters.angel_one import AngelOneAdapter
    import redis.asyncio as aioredis
    redis_client = aioredis.from_url(s.redis_url, decode_responses=True)
    adapter = AngelOneAdapter(
        api_key=s.angel_one_api_key,
        client_id=s.angel_one_client_id,
        totp_secret=s.angel_one_totp_secret,
        mpin=s.angel_one_mpin,
        redis_client=redis_client,
    )
    auth = await adapter.ensure_authenticated()
    print(f"ensure_authenticated returned: {auth}")
    # JWT loaded from Redis means we're authenticated - proceed regardless of return value
    quote = await adapter.get_quote("NIFTY", "NSE")
    if quote:
        print(f"NIFTY LTP: {quote.get('ltp')} changePct: {quote.get('changePct')}")
    else:
        print("NIFTY: quote returned None - checking connection...")


asyncio.run(test())
