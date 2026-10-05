"""Forensic inspection script — run standalone."""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── 1. Inspect latest dataset ─────────────────────────────────────────────────
ds_dirs = sorted(ROOT.glob("artifacts/datasets/ds-1d-*"))
if not ds_dirs:
    print("No datasets found"); sys.exit(1)

latest = ds_dirs[-1]
print(f"Loading dataset: {latest.name}")
df = pd.read_parquet(str(latest / "data.parquet"))
print(f"Shape: {df.shape}")
print(f"Date range: {str(df.index.min())[:10]} -> {str(df.index.max())[:10]}")
nsymbols = df["symbol"].nunique() if "symbol" in df.columns else "unknown"
print(f"Symbols: {nsymbols}")

lbl = df["label"]
n1 = int((lbl == 1).sum())
n0 = int((lbl == 0).sum())
nna = int(lbl.isna().sum())
total = len(lbl)
print(f"\nLabel distribution:")
print(f"  1 (positive/target): {n1}  ({round(n1/total*100,1)}%)")
print(f"  0 (negative/stop):   {n0}  ({round(n0/total*100,1)}%)")
print(f"  NaN:                 {nna}")

if "execution_model" in df.columns:
    print(f"\nexecution_model: {df['execution_model'].unique().tolist()}")
if "is_economic_evidence" in df.columns:
    print(f"is_economic_evidence: {df['is_economic_evidence'].unique().tolist()}")

if "outcome" in df.columns:
    print(f"\nOutcome distribution:")
    print(df["outcome"].value_counts().to_dict())

if "realized_return" in df.columns:
    rr = df["realized_return"].dropna()
    print(f"\nRealized return (gross):")
    print(f"  mean: {round(float(rr.mean())*100,4)}%")
    print(f"  std:  {round(float(rr.std())*100,4)}%")
    print(f"  positive rate: {round(float((rr>0).mean())*100,2)}%")

if "realized_return_net" in df.columns:
    rrn = df["realized_return_net"].dropna()
    print(f"\nRealized return (net after 10bps):")
    print(f"  mean: {round(float(rrn.mean())*100,4)}%")
    print(f"  positive rate: {round(float((rrn>0).mean())*100,2)}%")

# Detect horizon from metadata
meta_path = latest / "metadata.json"
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    print(f"\nDataset metadata keys: {list(meta.keys())}")
    for k in ["label_horizon", "label_type", "execution_model", "feature_schema_version",
              "label_schema_version", "label_config"]:
        if k in meta:
            print(f"  {k}: {meta[k]}")

# ── 2. Check for look-ahead features ─────────────────────────────────────────
print("\n\n=== LOOK-AHEAD BIAS QUICK CHECK ===")
feature_cols = [c for c in df.columns if c not in
    ["label", "realized_return", "realized_return_net", "outcome",
     "execution_model", "is_economic_evidence", "symbol"]]

if "realized_return" in df.columns:
    future_ret = df["realized_return"].dropna()
    suspicious = []
    for col in feature_cols[:30]:  # check first 30
        try:
            feat = df[col].dropna()
            aligned = pd.concat([feat, future_ret], axis=1, keys=["f","r"]).dropna()
            if len(aligned) < 30:
                continue
            corr = float(aligned["f"].corr(aligned["r"]))
            if abs(corr) > 0.15:
                suspicious.append((col, round(corr, 4)))
        except Exception:
            pass
    if suspicious:
        print("SUSPICIOUS correlations with realized_return:")
        for col, c in sorted(suspicious, key=lambda x: -abs(x[1])):
            print(f"  {col}: {c}")
    else:
        print("No suspicious correlations found in first 30 features (|r|<0.15)")

# ── 3. Check parquet data files ────────────────────────────────────────────────
print("\n\n=== RAW DATA PARQUETS ===")
pq_files = list((ROOT / "data" / "1d" / "1d").glob("*.parquet"))
print(f"Total parquet files: {len(pq_files)}")
if pq_files:
    # Sample one
    sample_pq = pq_files[0]
    pq_df = pd.read_parquet(str(sample_pq))
    print(f"Sample parquet ({sample_pq.stem}): shape={pq_df.shape}, cols={list(pq_df.columns)}")
    print(f"  Date range: {str(pq_df.index.min())[:10]} -> {str(pq_df.index.max())[:10]}")
    print(f"  Timezone: {pq_df.index.tz}")
    # Check for phantom bars (03:45 UTC)
    if hasattr(pq_df.index, 'time'):
        times = pq_df.index.time
        import collections
        time_counts = collections.Counter(str(t) for t in times)
        print(f"  Top 5 bar times: {dict(list(time_counts.most_common(5)))}")

# ── 4. Forward paper signals ───────────────────────────────────────────────────
fp_path = ROOT / "artifacts" / "forward_paper" / "forecasts.jsonl"
if fp_path.exists():
    print(f"\n\n=== FORWARD PAPER SIGNALS ===")
    records = []
    for line in fp_path.read_text().splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    print(f"Total forecast records: {len(records)}")
    if records:
        # Group by session_date
        from collections import defaultdict
        by_date = defaultdict(list)
        for r in records:
            by_date[r.get("session_date","?")].append(r)
        print(f"Sessions: {sorted(by_date.keys())}")
        for d, recs in sorted(by_date.items()):
            statuses = [r.get("status","?") for r in recs]
            net_pcts = [r["net_pct"] for r in recs if r.get("net_pct") is not None]
            print(f"  {d}: {len(recs)} signals | status_counts={dict((s,statuses.count(s)) for s in set(statuses))}")
            if net_pcts:
                wins = sum(1 for p in net_pcts if p > 0)
                print(f"    Resolved: {len(net_pcts)} | win_rate={round(wins/len(net_pcts)*100,1)}% | mean_net={round(sum(net_pcts)/len(net_pcts)*100,3)}%")
