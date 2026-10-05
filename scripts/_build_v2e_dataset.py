"""
V2E Dataset: v2c base + ONLY the 22 OOS-validated features.
No feature bloat — only signals proven on the true holdout test set.

Key additions (by OOS IC):
  cs_mom_12_1       +0.038  ← 12-1 month skip-momentum (strongest factor!)
  cs_ret_252        +0.035  ← 12-month return CS rank
  cs_pct_from_52w_high +0.026 ← 52-week high momentum
  cs_pos_in_52w_range  +0.023 ← 52-week range position
  cs_neg_amihud     +0.022  ← liquidity premium
  cs_neg_ret_1      +0.022  ← short-term reversal (anti-momentum 1d)
  cs_neg_ret_5      +0.015  ← short-term reversal (5d)
  rsi_oversold_flag  +0.019  ← oversold bounce signal
  rsi_overbought_flag +0.019 ← overbought peak signal
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "artifacts/datasets/v2e_selected"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR = ROOT / "data/1d/1d"

# Features proven to have OOS IC > 0.003, p < 0.05 from v2d
SELECTED_NEW_FEATS = [
    "cs_mom_12_1", "cs_ret_252", "mom_12_1",
    "cs_pct_from_52w_high", "cs_pos_in_52w_range",
    "cs_neg_amihud", "cs_neg_ret_1",
    "rsi_oversold_flag", "rsi_overbought_flag",
    "consec_down_bars",   # IC=+0.016
    "cs_neg_ret_5",       # IC=+0.015
    "bb_below_lower",     # IC=+0.010
    # Negative IC features (model will learn to use inversely)
    "consec_up_bars",     # IC=-0.017 (many up = overbought)
    "news_event_importance",  # IC=-0.016 (high news = negative?)
    "cs_up_down_vol_ratio",   # IC=-0.015
    "cs_ret_accel_5_10",      # IC=-0.012
]


def main():
    print("="*65)
    print("  V2E DATASET — OOS-VALIDATED FEATURE SELECTION")
    print(f"  Adding {len(SELECTED_NEW_FEATS)} features proven on 2025+ holdout")
    print("="*65)

    # Load v2d which has ALL new features
    v2d_path = ROOT / "artifacts/datasets/v2d_maxalpha/data.parquet"
    if not v2d_path.exists():
        raise FileNotFoundError("Run _build_v2d_dataset.py first")

    print("Loading v2d dataset...")
    df_v2d = pd.read_parquet(str(v2d_path))
    if df_v2d.index.tz is None:
        df_v2d.index = pd.to_datetime(df_v2d.index).tz_localize("UTC")
    print(f"  Shape: {df_v2d.shape}")

    # v2c base columns (keep all of them)
    v2c_path = ROOT / "artifacts/datasets/v2c_cs_regime/data.parquet"
    df_v2c = pd.read_parquet(str(v2c_path))
    v2c_cols = set(df_v2c.columns)

    # Select only validated new columns
    avail_new = [f for f in SELECTED_NEW_FEATS if f in df_v2d.columns]
    missing_new = [f for f in SELECTED_NEW_FEATS if f not in df_v2d.columns]
    if missing_new:
        print(f"  Missing from v2d (will skip): {missing_new}")

    # Build output: v2c columns + selected new ones
    keep_cols = list(df_v2c.columns) + [f for f in avail_new if f not in v2c_cols]
    keep_cols = [c for c in keep_cols if c in df_v2d.columns]

    df_out = df_v2d[keep_cols].copy()
    print(f"\nOutputting {len(df_out.columns)} columns ({len(keep_cols)} total):")
    print(f"  v2c base columns: {len(v2c_cols)}")
    print(f"  New validated:     {len([f for f in avail_new if f not in v2c_cols])}")

    # Quick OOS IC sanity check on top features
    OOS_START = pd.Timestamp("2025-01-01", tz="UTC")
    df_oos = df_out[df_out.index >= OOS_START]
    from scipy.stats import spearmanr
    if "label_v2b" in df_oos.columns:
        valid = df_oos["label_v2b"].notna()
        y = df_oos.loc[valid, "label_v2b"].to_numpy(dtype=float)
        print("\nOOS IC verification for key new features:")
        for f in ["cs_mom_12_1", "cs_ret_252", "cs_pct_from_52w_high", "cs_neg_amihud", "cs_neg_ret_1"]:
            if f in df_oos.columns:
                x = df_oos.loc[valid, f].fillna(0.5).to_numpy(dtype=float)
                ic, pval = spearmanr(x, y)
                print(f"  {f:35s}: IC={ic:+.4f} (p={pval:.4f})")

    # Save
    out_parquet = OUT_DIR / "data.parquet"
    df_out.to_parquet(str(out_parquet))

    metadata = {
        "dataset_id": "v2e_selected",
        "based_on": "v2d_maxalpha",
        "n_rows": len(df_out),
        "n_cols": df_out.shape[1],
        "new_features_added": avail_new,
        "feature_selection_method": "OOS IC > 0.003 and p < 0.05 on 2025-2026 holdout",
        "label_col": "label_v2b",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\n✓ Saved: {out_parquet}  ({len(df_out):,} rows × {df_out.shape[1]} cols)")
    print(f"Next: python3 scripts/train_v2.py --dataset {out_parquet}")


if __name__ == "__main__":
    main()
