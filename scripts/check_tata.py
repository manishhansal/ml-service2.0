#!/usr/bin/env python3
import pandas as pd, json
from pathlib import Path

BASE = Path('/Users/manishkumar/Desktop/ml-service2.0')
sigs = {}
for sp in [BASE/'artifacts/forward_paper/signals.jsonl', BASE/'artifacts/forward_paper/signals_v2.jsonl']:
    if sp.exists():
        for line in sp.read_text().splitlines():
            if line.strip():
                try:
                    s = json.loads(line)
                    sigs[s['symbol']] = s
                except Exception:
                    pass

# Check TATAMOTORS
tata = sigs.get('TATAMOTORS', {})
print(f"TATAMOTORS signal: dir={tata.get('direction')} entry={tata.get('entry_price')} ts={tata.get('signal_ts','?')[:20]}")

pf = BASE/'data/1d/1d/TATAMOTORS.parquet'
if pf.exists():
    df = pd.read_parquet(pf)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    print(f"Parquet rows: {len(df)} | date range: {df.index.min().date()} -> {df.index.max().date()}")
    if 'close' in df.columns:
        print(f"Last 3 closes: {df['close'].tail(3).tolist()}")
        print(f"Min close ever: {df['close'].min():.2f}  Max: {df['close'].max():.2f}")
else:
    print("No TATAMOTORS parquet!")
