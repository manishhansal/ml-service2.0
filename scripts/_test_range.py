"""Quick test: how far back does Upstox return NIFTY daily data?"""
import ssl, json, urllib.request, urllib.parse

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

token = [l.split('=',1)[1].strip() for l in open('/Users/manishkumar/Desktop/data-service2.0/.env')
         if l.startswith('UPSTOX_ACCESS_TOKEN=')][0]
headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}

ikey = urllib.parse.quote('NSE_INDEX|Nifty 50', safe='')
for label, frm in [('2yr', '2024-01-01'), ('4yr', '2022-01-01'), ('5yr', '2021-01-01')]:
    url = f'https://api.upstox.com/v2/historical-candle/{ikey}/day/2026-10-30/{frm}'
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15, context=_CTX) as r:
            c = json.loads(r.read()).get('data', {}).get('candles', [])
            first = c[-1][0][:10] if c else '?'
            print(f'{label} ({frm}): {len(c)} bars, oldest={first}')
    except Exception as e:
        print(f'{label}: ERROR {e}')
