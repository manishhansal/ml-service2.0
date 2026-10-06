"""
Fast v2 dataset builder: loads existing features from pre-built dataset
and computes 7-day vol-adjusted excess return labels from raw parquets.

This avoids recomputing 84 features from scratch (which takes hours)
while getting accurate 7-day forward labels.

Outputs: artifacts/datasets/v2_labeled/data.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.labels.seven_day import generate_7d_excess_return_label
from src.logging_config import get_logger

logger = get_logger(__name__)

OUT_DIR = ROOT / "artifacts" / "datasets" / "v2_labeled"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_DIR = ROOT / "data" / "1d" / "1d"
HORIZON = 7
VOL_WINDOW = 20
CLIP = 5.0


def load_latest_dataset() -> pd.DataFrame:
    datasets = sorted(ROOT.glob("artifacts/datasets/ds-1d-*/data.parquet"))
    if not datasets:
        raise FileNotFoundError("No pre-built datasets found")
    ds = datasets[-1]
    print(f"Loading dataset: {ds.parent.name}")
    df = pd.read_parquet(str(ds))
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def load_nifty_close() -> pd.Series:
    nifty_pq = PARQUET_DIR / "NIFTY.parquet"
    if not nifty_pq.exists():
        # Fall back to BANKNIFTY or NIFTY50
        for fallback in ["BANKNIFTY.parquet", "NIFTY50.parquet"]:
            if (PARQUET_DIR / fallback).exists():
                nifty_pq = PARQUET_DIR / fallback
                break
    if not nifty_pq.exists():
        print("WARNING: NIFTY parquet not found — using zero excess return proxy")
        return None
    df = pd.read_parquet(str(nifty_pq))
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df["close"]


def main():
    print("="*60)
    print("  V2 DATASET BUILDER")
    print("  Computing 7-day vol-adjusted excess return labels")
    print("="*60)

    # Load existing feature dataset
    df_full = load_latest_dataset()
    print(f"Full dataset shape: {df_full.shape}")
    print(f"Date range: {str(df_full.index.min())[:10]} → {str(df_full.index.max())[:10]}")
    print(f"Symbols: {df_full['symbol'].nunique() if 'symbol' in df_full.columns else 'unknown'}")

    # Load NIFTY
    nifty_close = load_nifty_close()
    if nifty_close is not None:
        print(f"NIFTY loaded: {len(nifty_close)} bars")

    # Build 7-day labels per symbol from parquets
    print("\nComputing 7-day labels from parquet files...")
    label_parts = []
    pq_files = sorted(PARQUET_DIR.glob("*.parquet"))
    n_total = len(pq_files)
    n_done = 0

    # Filter to symbols in the dataset
    ds_symbols = set(df_full["symbol"].unique()) if "symbol" in df_full.columns else set()

    for pq in pq_files:
        sym = pq.stem
        if ds_symbols and sym not in ds_symbols:
            continue
        if sym in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"):
            continue  # indices don't need labels

        try:
            raw = pd.read_parquet(str(pq))
            if raw.index.tz is None:
                raw.index = raw.index.tz_localize("UTC")
            if len(raw) < HORIZON + VOL_WINDOW + 5:
                continue

            # Build minimal OHLCV
            ohlcv = raw[["open", "high", "low", "close", "volume"]].copy()

            # NIFTY close aligned to this symbol's dates
            if nifty_close is not None:
                nc = nifty_close.reindex(ohlcv.index, method="ffill")
            else:
                nc = pd.Series(22000.0, index=ohlcv.index)

            label_s = generate_7d_excess_return_label(
                stock_df=ohlcv,
                nifty_close=nc,
                horizon=HORIZON,
                vol_window=VOL_WINDOW,
                clip=CLIP,
            )
            label_df = pd.DataFrame({"label_v2": label_s, "close_for_label": raw["close"]})
            label_df["symbol"] = sym
            label_parts.append(label_df)
            n_done += 1

            if n_done % 50 == 0:
                print(f"  Processed {n_done} symbols...")

        except Exception as exc:
            logger.warning("v2_label_build_failed", sym=sym, error=str(exc))

    print(f"  Labels computed for {n_done} symbols")

    # Combine labels
    if not label_parts:
        raise RuntimeError("No labels built")

    labels_df = pd.concat(label_parts).sort_index()
    print(f"Labels shape: {labels_df.shape}")
    print(f"Label stats: mean={labels_df['label_v2'].mean():.4f}, "
          f"std={labels_df['label_v2'].std():.4f}, "
          f"positive={( labels_df['label_v2']>0).mean():.1%}")

    # Join labels onto feature dataset
    print("\nJoining labels onto feature dataset...")
    # Create a (timestamp, symbol) multi-index for joining
    if "symbol" in df_full.columns:
        df_full = df_full.copy()
        df_full["_ts"] = df_full.index
        labels_df["_ts"] = labels_df.index

        # Merge on (timestamp, symbol)
        merged = df_full.merge(
            labels_df[["label_v2", "_ts", "symbol"]],
            on=["_ts", "symbol"],
            how="left",
        )
        merged.index = df_full.index
        merged = merged.drop(columns=["_ts"])
    else:
        merged = df_full.join(labels_df[["label_v2"]], how="left")

    # Drop rows without valid labels (tail bars)
    before = len(merged)
    merged = merged[merged["label_v2"].notna()]
    after = len(merged)
    print(f"Rows with valid labels: {after:,} / {before:,} ({after/before:.1%})")

    # Also add old label as label_v1 for comparison
    if "label" in merged.columns:
        merged = merged.rename(columns={"label": "label_v1"})
    merged["label"] = np.sign(merged["label_v2"]).map({1.0: 1, -1.0: 0, 0.0: 0}).fillna(0).astype(int)

    print(f"\nFinal dataset shape: {merged.shape}")
    print(f"Label_v2 stats: mean={merged['label_v2'].mean():.4f}, std={merged['label_v2'].std():.4f}")
    print(f"Label (binary from v2 direction) balance: {merged['label'].mean():.1%} positive")

    # Save
    out_parquet = OUT_DIR / "data.parquet"
    merged.to_parquet(str(out_parquet))

    metadata = {
        "dataset_id": "v2_labeled",
        "label_type": "7d_vol_adjusted_excess_return",
        "label_horizon": HORIZON,
        "vol_window": VOL_WINDOW,
        "clip": CLIP,
        "n_rows": len(merged),
        "n_features": len([c for c in merged.columns if c not in ["label","label_v1","label_v2","symbol","realized_return","realized_return_net","outcome","execution_model","is_economic_evidence"]]),
        "date_start": str(merged.index.min()),
        "date_end":   str(merged.index.max()),
        "n_symbols":  merged["symbol"].nunique() if "symbol" in merged.columns else 0,
        "label_mean": float(merged["label_v2"].mean()),
        "label_std":  float(merged["label_v2"].std()),
        "positive_rate": float((merged["label_v2"]>0).mean()),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\n✓ Saved: {out_parquet}")
    print(f"✓ Saved: {OUT_DIR}/metadata.json")
    print(f"\nReady for: python3 scripts/train_v2.py --dataset {out_parquet}")


if __name__ == "__main__":
    main()
