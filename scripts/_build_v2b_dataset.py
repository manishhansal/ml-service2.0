"""
v2b Dataset Builder: uses CROSS-SECTIONAL RANK of 7-day raw excess return.

Fix from v2: Remove vol-normalization from label.
  v2 label: (stock_7d_return - nifty_7d_return) / stock_vol  ← biases toward low-vol stocks
  v2b label: cross_sectional_rank(stock_7d_return - nifty_7d_return) in [0,1]

The rank-based label:
  - Removes vol bias entirely
  - Is inherently cross-sectional (maps directly to L/S portfolio construction)  
  - Has perfect label-to-strategy alignment (predict rank → trade by rank)
  - Centering at 0.5 (median) makes IC interpretation straightforward
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "artifacts" / "datasets" / "v2b_ranked"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_DIR = ROOT / "data" / "1d" / "1d"
HORIZON = 7


def main():
    print("="*60)
    print("  V2B DATASET BUILDER")
    print("  Label: cross-sectional rank of 7-day excess return")
    print("="*60)

    # Load feature dataset
    ds_files = sorted(ROOT.glob("artifacts/datasets/ds-1d-*/data.parquet"))
    if not ds_files:
        raise FileNotFoundError("No pre-built datasets found")
    df_full = pd.read_parquet(str(ds_files[-1]))
    if df_full.index.tz is None:
        df_full.index = df_full.index.tz_localize("UTC")
    print(f"Loaded dataset: {ds_files[-1].parent.name}")
    print(f"Shape: {df_full.shape}, Symbols: {df_full['symbol'].nunique()}")

    # Load NIFTY
    nifty_pq = PARQUET_DIR / "NIFTY.parquet"
    nifty_close = None
    if nifty_pq.exists():
        nf = pd.read_parquet(str(nifty_pq))
        if nf.index.tz is None:
            nf.index = nf.index.tz_localize("UTC")
        nifty_close = nf["close"]
        print(f"NIFTY loaded: {len(nifty_close)} bars")

    # Compute 7-day forward excess returns per symbol
    print("\nComputing 7-day forward returns per symbol...")
    label_parts = []
    pq_files = sorted(PARQUET_DIR.glob("*.parquet"))
    ds_symbols = set(df_full["symbol"].unique()) if "symbol" in df_full.columns else set()
    n_done = 0

    for pq in pq_files:
        sym = pq.stem
        if ds_symbols and sym not in ds_symbols:
            continue
        if sym in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"):
            continue

        try:
            raw = pd.read_parquet(str(pq))
            if raw.index.tz is None:
                raw.index = raw.index.tz_localize("UTC")
            close = raw["close"].astype(float)

            if len(close) < HORIZON + 5:
                continue

            # 7-day forward return (close to close)
            stock_fwd = close.shift(-HORIZON) / close - 1.0

            # NIFTY 7-day forward return
            if nifty_close is not None:
                nc = nifty_close.reindex(close.index, method="ffill")
                nifty_fwd = nc.shift(-HORIZON) / nc - 1.0
            else:
                nifty_fwd = pd.Series(0.0, index=close.index)

            # Raw excess return (no vol normalization)
            excess_ret = stock_fwd - nifty_fwd
            # Tail bars: no forward data
            excess_ret.iloc[-HORIZON:] = np.nan

            label_df = pd.DataFrame({"excess_ret": excess_ret}, index=close.index)
            label_df["symbol"] = sym
            label_parts.append(label_df)
            n_done += 1

            if n_done % 50 == 0:
                print(f"  {n_done} symbols processed...")

        except Exception as e:
            pass

    print(f"  Labels computed for {n_done} symbols")

    # Combine all excess returns
    exc_df = pd.concat(label_parts).sort_index()
    exc_df = exc_df[exc_df["excess_ret"].notna()]

    # Compute cross-sectional rank WITHIN each date (so each date maps to [0,1])
    print("\nComputing cross-sectional ranks...")
    exc_pivot = exc_df.pivot_table(
        index=exc_df.index,
        columns="symbol",
        values="excess_ret",
        aggfunc="last",
    )
    # Rank each row (date): percentile rank from 0 (worst) to 1 (best)
    rank_pivot = exc_pivot.rank(axis=1, pct=True)

    # Convert back to long format
    rank_long = rank_pivot.stack(future_stack=True).rename("label_v2b").reset_index()
    rank_long.columns = ["ts", "symbol", "label_v2b"]
    rank_long = rank_long.set_index("ts")

    # Also keep raw excess return for diagnostics
    exc_long = exc_pivot.stack(future_stack=True).rename("excess_ret_raw").reset_index()
    exc_long.columns = ["ts", "symbol", "excess_ret_raw"]
    exc_long = exc_long.set_index("ts")

    print(f"Rank label stats: mean={rank_long['label_v2b'].mean():.4f}, std={rank_long['label_v2b'].std():.4f}")
    print(f"Positive rate (rank > 0.5): {(rank_long['label_v2b'] > 0.5).mean():.1%}")

    # Merge onto feature dataset
    print("\nMerging onto feature dataset...")
    df_full_copy = df_full.copy()
    df_full_copy["_ts"] = df_full_copy.index

    rank_long["_ts"] = rank_long.index
    exc_long["_ts"] = exc_long.index

    # Merge
    merged = df_full_copy.merge(
        rank_long[["label_v2b", "_ts", "symbol"]],
        on=["_ts", "symbol"], how="left",
    )
    merged = merged.merge(
        exc_long[["excess_ret_raw", "_ts", "symbol"]],
        on=["_ts", "symbol"], how="left",
    )
    merged.index = df_full_copy.index
    merged = merged.drop(columns=["_ts"])

    # Remove rows without valid labels
    before = len(merged)
    merged = merged[merged["label_v2b"].notna()]
    after = len(merged)
    print(f"Valid labeled rows: {after:,} / {before:,}")

    # Replace label column
    if "label" in merged.columns:
        merged = merged.rename(columns={"label": "label_v1"})
    # Binary label: top 50% = 1 (outperformer), bottom 50% = 0 (underperformer)
    merged["label"] = (merged["label_v2b"] >= 0.5).astype(int)
    # Also add v2a label for comparison
    if "label_v2" in df_full.columns:
        merged["label_v2a"] = df_full["label_v2"].reindex(df_full_copy.index)

    print(f"\nFinal dataset shape: {merged.shape}")
    print(f"label_v2b stats: mean={merged['label_v2b'].mean():.4f}, std={merged['label_v2b'].std():.4f}")
    print(f"label (binary) positive: {merged['label'].mean():.1%}")
    print(f"excess_ret_raw mean: {merged['excess_ret_raw'].mean()*100:.4f}%")

    # Save
    out_parquet = OUT_DIR / "data.parquet"
    merged.to_parquet(str(out_parquet))

    metadata = {
        "dataset_id": "v2b_ranked",
        "label_type": "cross_sectional_rank_7d_excess_return",
        "label_description": "Percentile rank [0,1] of 7-day excess return vs NIFTY within universe per date",
        "label_horizon": HORIZON,
        "n_rows": len(merged),
        "date_start": str(merged.index.min()),
        "date_end": str(merged.index.max()),
        "n_symbols": int(merged["symbol"].nunique()) if "symbol" in merged.columns else 0,
        "label_mean": float(merged["label_v2b"].mean()),
        "label_std": float(merged["label_v2b"].std()),
        "binary_positive_rate": float(merged["label"].mean()),
        "key_fix": "Removed vol-normalization to eliminate defensive factor bias",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\n✓ Saved: {out_parquet}")
    print(f"✓ Saved: {OUT_DIR}/metadata.json")
    print(f"\nRun training with:")
    print(f"  python3 scripts/train_v2.py --dataset {out_parquet}")


if __name__ == "__main__":
    main()
