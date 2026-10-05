"""Forensic audit of forward paper net_pct values."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).parent.parent
fp = ROOT / "artifacts/forward_paper/forecasts.jsonl"

if not fp.exists():
    print("forecasts.jsonl not found"); exit()

records = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
print(f"Total records: {len(records)}")

by_date = defaultdict(list)
for r in records:
    by_date[r.get("session_date","?")].append(r)

for d in sorted(by_date):
    resolved = [r for r in by_date[d] if r.get("net_pct") is not None]
    if not resolved:
        continue
    net = [r["net_pct"] for r in resolved]
    print(f"\n{d}: {len(resolved)} resolved")
    print(f"  net_pct sample (raw): {net[:5]}")
    print(f"  net_pct mean (raw): {np.mean(net):.6f}")
    print(f"  net_pct mean (%):   {np.mean(net)*100:.4f}%")
    print(f"  max: {max(net):.6f}  min: {min(net):.6f}")

    # Print a few resolved records in full
    for r in resolved[:2]:
        print(f"\n  Sample record:")
        for k in ["symbol","direction","score","entry_price","exit_price","gross_pct","net_pct","status","session_date","ts"]:
            if k in r:
                print(f"    {k}: {r[k]}")

    # Manual P&L check
    for r in resolved[:1]:
        ep = r.get("entry_price", 0)
        xp = r.get("exit_price", 0)
        dr = r.get("direction", 1)
        if ep and xp:
            gross_manual = dr * (xp - ep) / ep
            net_manual = gross_manual - 27.35 / 10000
            print(f"\n  Manual P&L check for {r.get('symbol')}:")
            print(f"    entry={ep}, exit={xp}, direction={dr}")
            print(f"    gross_return={gross_manual:.6f} ({gross_manual*100:.4f}%)")
            print(f"    net_return={net_manual:.6f} ({net_manual*100:.4f}%)")
            print(f"    stored net_pct={r.get('net_pct'):.6f}")
            if abs(net_manual - r.get("net_pct",0)) > 0.01:
                print(f"    ⚠ MISMATCH: manual={net_manual:.6f} vs stored={r.get('net_pct'):.6f}")
            else:
                print(f"    ✓ MATCH (within tolerance)")
