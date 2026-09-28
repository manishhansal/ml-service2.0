#!/usr/bin/env python3
"""Check for price mismatches between parquets and live quotes (data quality check)."""
import json
import time
import urllib.request
import pandas as pd
from pathlib import Path

BASE = Path('/Users/manishkumar/Desktop/ml-service2.0')
DATA_KEY = [l.split('=', 1)[1].strip() for l in (BASE / '.env').read_text().splitlines()
            if l.startswith('DATA_SERVICE_API_KEY=')][0]


def get_ltp(sym: str) -> float | None:
    try:
        req = urllib.request.Request(
            f'http://localhost:8200/v1/india/quotes/{sym}',
            headers={'X-API-KEY': DATA_KEY})
        with urllib.request.urlopen(req, timeout=6) as r:
            d = json.loads(r.read())
            ltp = d.get('data', {}).get('ltp')
            return float(ltp) if ltp else None
    except Exception:
        return None


mismatches = []
checked = 0
for pf in sorted((BASE / 'data/1d/1d').glob('*.parquet')):
    sym = pf.stem
    try:
        df = pd.read_parquet(pf)
        if 'close' not in df.columns or len(df) == 0:
            continue
        hist_close = float(df['close'].iloc[-1])
    except Exception:
        continue

    ltp = get_ltp(sym)
    time.sleep(0.15)
    checked += 1

    if ltp and hist_close > 0:
        ratio = abs(ltp - hist_close) / hist_close
        if ratio > 0.30:
            mismatches.append((sym, hist_close, ltp, ratio))

print(f'Checked: {checked} symbols')
print(f'Mismatches >30%: {len(mismatches)}')
for sym, h, ltp_val, r in sorted(mismatches, key=lambda x: -x[3]):
    print(f'  {sym:15s} hist_close={h:.2f}  live={ltp_val:.2f}  discrepancy={r:.0%}')

if not mismatches:
    print('  None found — all prices consistent!')
