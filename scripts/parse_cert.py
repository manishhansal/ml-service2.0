#!/usr/bin/env python3
import json
from pathlib import Path

r = json.loads(Path('reports/certification_run.json').read_text())
print("=== CERTIFICATION RUN SUMMARY ===")
print("Data:", r['data_source_class'])
print("Dataset ID:", r['dataset']['dataset_id'])
print("Rows:", r['dataset']['rows'], "| Universe:", r['dataset']['universe'])
print("Leakage OK:", r['dataset']['leakage_validated'])

t = r['training']
print()
print("=== TRAINING ===")
print("Champion:", t['champion'], "| IC_mean:", t['champion_ic_mean'])
print("PBO:", t['champion_pbo'], "| Net_Sharpe:", t['champion_net_sharpe'])
print("Passed:", t['passed_acceptance'], "| Reason:", t['rejection_reason'])
for c in t.get('candidates', []):
    print(f"  {c['name']:20s} wf_ic={c['wf_ic_mean']:.4f} cpcv_ic={c['cpcv_ic_mean']:.4f} pbo={c['cpcv_pbo']:.3f}")

bt = r.get('backtest', {})
print()
print("=== BACKTEST ===")
print("Net return:", bt.get('net_return'), "| Sharpe:", bt.get('sharpe'))
print("n_trades:", bt.get('n_trades'))

hs = r.get('historical_replay', {})
print()
print("=== HISTORICAL REPLAY ===")
print("total decisions:", hs.get('total_decisions'))
print("downgraded by guard:", hs.get('downgraded_by_guard'))
print("action distribution:", hs.get('action_distribution'))

fp = r.get('forward_paper', {})
print()
print("=== FORWARD PAPER ===")
print("status:", fp.get('status'))
print("reason:", fp.get('reason'))
print("n_signals:", fp.get('n_signals'))

cs = r.get('cost_sensitivity', {})
print()
print("=== COST SENSITIVITY ===")
for k, v in cs.items():
    print(f"  {k}: sharpe={v.get('sharpe', '?')} net_return={v.get('net_return', '?')}")
