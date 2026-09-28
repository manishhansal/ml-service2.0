#!/usr/bin/env python3
"""Full forensic diagnosis of universe, data gaps, rate limits, and all issues."""
import json, pandas as pd
from pathlib import Path
from collections import Counter

BASE = Path('.')

# ── 1. Forward paper signals ──────────────────────────────────────────────────
signals = [json.loads(l) for l in (BASE/'artifacts/forward_paper/signals.jsonl').read_text().splitlines() if l.strip()]
fp_syms = sorted({s['symbol'] for s in signals})
print(f"=== FORWARD PAPER ===")
print(f"Symbols in forward paper session: {len(fp_syms)}")

# ── 2. On-disk parquets ────────────────────────────────────────────────────────
parquets = sorted((BASE/'data/1d/1d').glob('*.parquet'))
disk_syms = [p.stem for p in parquets]
print(f"\n=== ON-DISK DATA ===")
print(f"Parquet files: {len(disk_syms)}")

# Date coverage
dates = {}
for pf in parquets:
    df = pd.read_parquet(pf)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    dates[pf.stem] = {
        'rows': len(df),
        'last': str(df.index.max().date()),
        'first': str(df.index.min().date()),
    }
last_date_counts = Counter(d['last'] for d in dates.values())
print("Date distribution (last bar):")
for dt, cnt in sorted(last_date_counts.items()):
    print(f"  {dt}: {cnt} symbols")

outdated = [s for s, d in dates.items() if d['last'] < '2026-09-23']
fresh    = [s for s, d in dates.items() if d['last'] >= '2026-09-23']
print(f"Outdated (last bar = 2026-09-22): {len(outdated)} symbols")
print(f"Fresh (last bar >= 2026-09-23):   {len(fresh)} symbols")

# ── 3. Training dataset ────────────────────────────────────────────────────────
ds_dirs = sorted(
    [d for d in (BASE/'artifacts/datasets').iterdir() if d.is_dir() and 'ds-1d-' in d.name],
    key=lambda d: d.name, reverse=True,
)
print(f"\n=== TRAINING DATASET ===")
if ds_dirs:
    meta = json.loads((ds_dirs[0]/'metadata.json').read_text())
    train_syms = meta['universe']
    print(f"Dataset: {ds_dirs[0].name}")
    print(f"Universe: {len(train_syms)} symbols")
    print(f"Rows: {meta['row_count']} | Features: {meta['feature_count']}")
    print(f"Schema: {meta.get('feature_schema_version','?')}")
    print(f"Survivorship: {meta.get('survivorship','?')}")
else:
    train_syms = []
    print("No dataset found!")

# ── 4. Why only 65 in forward paper? ──────────────────────────────────────────
print(f"\n=== WHY ONLY 65 SYMBOLS IN FORWARD PAPER? ===")
print(f"(Expected: 218 from training universe)")
in_train_not_fp = [s for s in train_syms if s not in fp_syms]
print(f"In training dataset but NOT in forward paper: {len(in_train_not_fp)}")
print(f"First 10 missing: {in_train_not_fp[:10]}")

# Find the forward paper script to understand why
fp_script = BASE / 'scripts/run_forward_paper_session.py'
if fp_script.exists():
    content = fp_script.read_text()
    # Find where universe is defined
    for line in content.splitlines():
        if 'symbol' in line.lower() or 'universe' in line.lower() or '65' in line:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                print(f"  FP script line: {stripped[:120]}")
else:
    print("  run_forward_paper_session.py not found!")

# ── 5. Full F&O universe vs what we have ─────────────────────────────────────
print(f"\n=== F&O UNIVERSE GAP ===")
print(f"On-disk: {len(disk_syms)} symbols")
print(f"Training dataset: {len(train_syms)} symbols")
print(f"Expected SEBI F&O: ~220 symbols")
print(f"Missing from disk vs expected: ~{220 - len(disk_syms)}")

not_in_disk = [s for s in train_syms if s not in disk_syms]
print(f"In training dataset but NOT on disk: {len(not_in_disk)} → {not_in_disk[:5]}")

# ── 6. Rate limit diagnosis ───────────────────────────────────────────────────
print(f"\n=== RATE LIMIT DIAGNOSIS ===")
print("Data-service rate limit: 100 req/60s (shared with workers)")
print("Workers consuming ~90-95 req/min continuously")
print("Root cause: data-service-worker + data-service-scheduler exhaust token bucket")
print("Fix: pause workers during bulk historical ingestion")
print("Or: use /v1/india/historical?symbol=X endpoint with correct URL")
print("Previous bug: used /v1/india/historical/SYMBOL/1d (404 Not Found!)")
