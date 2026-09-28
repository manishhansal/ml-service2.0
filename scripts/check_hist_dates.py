#!/usr/bin/env python3
import json
from datetime import datetime, timezone
from pathlib import Path

raw = Path("/tmp/hist_resp.json").read_text()
if not raw.strip():
    print("EMPTY response")
    exit()
if 'RATE_LIMIT' in raw:
    d = json.loads(raw)
    print("RATE_LIMITED:", d['error']['retryAfterMs'])
    exit()

try:
    d = json.loads(raw)
except Exception as e:
    print("JSON parse error:", e)
    print("First 200:", raw[:200])
    exit()

bars = d.get('data', []) if isinstance(d, dict) else (d if isinstance(d, list) else [])
print(f"Total bars: {len(bars)}")
if bars:
    for b in bars[-7:]:
        t = b.get('time') or b.get('timestamp')
        if t:
            dt = datetime.fromtimestamp(int(t), tz=timezone.utc).date()
            print(f"  {dt}  open={b.get('open')}  close={b.get('close')}")
    print(f"\nLatest: {datetime.fromtimestamp(int(bars[-1].get('time',0)), tz=timezone.utc).date()}")
