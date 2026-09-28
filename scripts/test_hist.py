#!/usr/bin/env python3
import json, urllib.request
from pathlib import Path

env = {}
for line in Path('.env').read_text().splitlines():
    if '=' in line and not line.strip().startswith('#'):
        k, _, v = line.partition('=')
        env[k.strip()] = v.strip()

DATA_KEY = env.get('DATA_SERVICE_API_KEY', '')
url = 'http://localhost:8200/v1/india/historical/360ONE/1d'
req = urllib.request.Request(url, headers={'X-API-KEY': DATA_KEY})
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read().decode()
except Exception as e:
    raw = str(e)

print('Length:', len(raw))
if 'RATE_LIMIT' in raw:
    d = json.loads(raw)
    print('RATE_LIMITED retryAfter:', d.get('error', {}).get('retryAfterMs'))
elif raw.startswith('['):
    bars = json.loads(raw)
    print('Bars count:', len(bars))
    for b in bars[-3:]:
        ts = b.get('timestamp') or b.get('date') or b.get('datetime') or b.get('t')
        c = b.get('close') or b.get('c')
        print(f'  ts={ts}  close={c}')
elif raw.startswith('{'):
    d = json.loads(raw)
    print('Dict keys:', list(d.keys())[:8])
    bars = d.get('data', d.get('bars', d.get('ohlcv', [])))
    print('Bars count:', len(bars))
    for b in bars[-3:]:
        ts = b.get('timestamp') or b.get('date') or b.get('datetime')
        c = b.get('close') or b.get('c')
        print(f'  ts={ts}  close={c}')
else:
    print(raw[:200])
