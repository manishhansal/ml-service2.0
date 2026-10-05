"""Compare old vs new label designs on the live parquet data."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.labels.seven_day import compare_label_designs, EQUITY_COST_BPS

# Load a sample of symbols
pq_dir = ROOT / "data" / "1d" / "1d"
files = sorted(pq_dir.glob("*.parquet"))[:10]

print(f"Comparing label designs on {len(files)} symbols...")
print(f"Cost assumption: {EQUITY_COST_BPS} bps equity round-trip\n")

all_old_net, all_new_b_net, all_new_a = [], [], []

# Load NIFTY for excess return calculation
nifty_path = pq_dir / "NIFTY.parquet"
nifty_close = None
if nifty_path.exists():
    nifty_df = pd.read_parquet(str(nifty_path))
    nifty_close = nifty_df["close"]

for f in files:
    try:
        df = pd.read_parquet(str(f))
        if len(df) < 50:
            continue

        # Old label
        from src.data.labels import LabelConfig, LabelFactory
        old_cfg = LabelConfig(
            label_type="triple_barrier", horizon=5,
            upper_barrier_pct=0.02, lower_barrier_pct=0.02,
            cost_bps=EQUITY_COST_BPS, execution_model="next_open",
        )
        old_lbl = LabelFactory(old_cfg).build(df)
        old_net = old_lbl["realized_return_net"].dropna()
        all_old_net.extend(old_net.tolist())

        # New label B: 7d asymmetric barrier
        from src.labels.seven_day import generate_7d_asymmetric_barrier_label
        new_b = generate_7d_asymmetric_barrier_label(df, cost_bps=EQUITY_COST_BPS, horizon=7)
        new_b_net = new_b["realized_return_net"].dropna()
        all_new_b_net.extend(new_b_net.tolist())

        # New label A: excess return
        if nifty_close is not None:
            from src.labels.seven_day import generate_7d_excess_return_label
            new_a = generate_7d_excess_return_label(df, nifty_close, horizon=7)
            all_new_a.extend(new_a.dropna().tolist())
    except Exception as e:
        pass

print("=" * 60)
print("LABEL DESIGN COMPARISON")
print("=" * 60)

old = np.array(all_old_net)
if len(old) > 0:
    cost_old = EQUITY_COST_BPS / 10_000
    print(f"\nOLD (5-bar ±2% symmetric):")
    print(f"  n observations:     {len(old):,}")
    print(f"  mean net return:    {old.mean()*100:.4f}%  ← {'NEGATIVE EV' if old.mean()<0 else 'POSITIVE EV'}")
    print(f"  positive rate:      {(old>0).mean()*100:.1f}%")
    print(f"  break-even needed:  56.9%")
    print(f"  VERDICT:            {'FAILS' if (old>0).mean() < 0.569 else 'PASSES'} break-even test")

new_b = np.array(all_new_b_net)
if len(new_b) > 0:
    print(f"\nNEW B (7-day 2:1 asymmetric barrier):")
    print(f"  n observations:     {len(new_b):,}")
    print(f"  mean net return:    {new_b.mean()*100:.4f}%  ← {'NEGATIVE EV' if new_b.mean()<0 else 'POSITIVE EV'}")
    print(f"  positive rate:      {(new_b>0).mean()*100:.1f}%")
    be_new = abs(-0.02 - EQUITY_COST_BPS/10000) / (abs(0.04 - EQUITY_COST_BPS/10000) + abs(-0.02 - EQUITY_COST_BPS/10000))
    print(f"  break-even needed:  ~35% (R:R=2:1)")
    print(f"  VERDICT:            {'FAILS' if (new_b>0).mean() < 0.35 else 'PASSES'} break-even test")

new_a = np.array(all_new_a)
if len(new_a) > 0:
    print(f"\nNEW A (7-day vol-adjusted excess return):")
    print(f"  n observations:     {len(new_a):,}")
    print(f"  mean label value:   {new_a.mean():.4f}")
    print(f"  std label value:    {new_a.std():.4f}")
    print(f"  positive rate:      {(new_a>0).mean()*100:.1f}%")
    print(f"  VERDICT:            IC-BASED (target: IC > 0.02)")

print("\n" + "=" * 60)
print("KEY FINDING:")
if len(old) > 0 and old.mean() < 0:
    print(f"  OLD labels have NEGATIVE EV ({old.mean()*100:.4f}%/trade)")
    print(f"  Training on these labels optimises for noise, not alpha")
if len(new_b) > 0:
    improvement = (new_b.mean() - (old.mean() if len(old) > 0 else 0)) * 100
    print(f"  NEW 7d-barrier labels: {new_b.mean()*100:.4f}%/trade ({improvement:+.4f}% improvement)")
print("=" * 60)
