import urllib.request, urllib.parse, json

token = None
for line in open('/Users/manishkumar/Desktop/data-service2.0/.env').readlines():
    if line.strip().startswith('UPSTOX_ACCESS_TOKEN='):
        token = line.split('=', 1)[1].strip()
        break

if not token:
    print("NO TOKEN"); exit(1)

print(f"Token: {token[:20]}...")
ikey = urllib.parse.quote('NSE_INDEX|Nifty 50', safe='')
headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}

tests = [
    ('single-day',  f'https://api.upstox.com/v2/historical-candle/{ikey}/day/2026-10-30/2026-10-30'),
    ('1mo-range',   f'https://api.upstox.com/v2/historical-candle/{ikey}/day/2026-10-30/2026-10-01'),
    ('2yr-range',   f'https://api.upstox.com/v2/historical-candle/{ikey}/day/2026-10-30/2024-01-01'),
]
for label, url in tests:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
        c = d.get('data', {}).get('candles', [])
        print(f"{label}: {len(c)} candles  status={d.get('status','?')}")
        if c:
            print(f"  oldest={c[-1][:2]}  newest={c[0][:2]}")
    except Exception as e:
        print(f"{label}: ERROR {e}")
