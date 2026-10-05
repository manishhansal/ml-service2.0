"""
v2c Dataset Builder: adds cross-sectional features + regime features.

Fixes the momentum-regime bias from v2b:
  1. Cross-sectional rank features (idiosyncratic, regime-invariant)
  2. Regime indicators (allow model to condition on market state)
  3. Sector-relative momentum (less affected by market-wide momentum reversals)
  4. Beta-neutral signal components

New features added (computed cross-sectionally per date):
  cs_rank_ret_1d    - rank of 1-day return within universe [0,1]
  cs_rank_ret_5d    - rank of 5-day return within universe
  cs_rank_ret_20d   - rank of 20-day return within universe
  cs_rank_rs_nifty  - rank of relative strength vs NIFTY
  nifty_trend       - NIFTY 20d return (regime indicator: positive=momentum, negative=reversal)
  market_momentum   - fraction of universe stocks with positive 5d return (breadth)
  trend_regime      - 1 if nifty_20d > +3%, -1 if < -3%, 0 otherwise
  beta_adj_ret_5d   - beta-neutralized 5-day return (stock ret - 1.0 × nifty_ret)

These features give the model the ability to:
  - Distinguish idiosyncratic alpha from market beta
  - Learn regime-conditional patterns
  - Identify cross-sectional relative strength regardless of market direction
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "artifacts" / "datasets" / "v2c_cs_regime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_DIR = ROOT / "data" / "1d" / "1d"
HORIZON = 7


def main():
    print("="*65)
    print("  V2C DATASET BUILDER")
    print("  Cross-sectional + regime features for regime-invariant alpha")
    print("="*65)

    # Load v2b dataset (has rank labels + base features)
    v2b_path = ROOT / "artifacts" / "datasets" / "v2b_ranked" / "data.parquet"
    if not v2b_path.exists():
        raise FileNotFoundError("v2b_ranked dataset not found. Run _build_v2b_dataset.py first.")

    df = pd.read_parquet(str(v2b_path))
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    print(f"Loaded v2b dataset: {df.shape}")
    print(f"Symbols: {df['symbol'].nunique()}")

    # Load NIFTY for regime features
    nifty_pq = PARQUET_DIR / "NIFTY.parquet"
    nifty_close = None
    nifty_ret_20d = None
    if nifty_pq.exists():
        nf = pd.read_parquet(str(nifty_pq))
        if nf.index.tz is None:
            nf.index = nf.index.tz_localize("UTC")
        nifty_close = nf["close"]
        # 20-day momentum
        nifty_ret_20d = nifty_close.pct_change(20).rename("nifty_ret_20d_abs")
        print(f"NIFTY loaded: {len(nifty_close)} bars")

    # ── STEP 1: Compute cross-sectional rank features per date ────────────────
    print("\nComputing cross-sectional rank features...")

    # Pivot returns to dates × symbols
    def make_cs_rank(df_in: pd.DataFrame, ret_col: str, out_name: str) -> pd.DataFrame:
        """Compute cross-sectional rank of `ret_col` per date, returning (ts, symbol) dataframe."""
        if ret_col not in df_in.columns:
            return pd.DataFrame(columns=["symbol", out_name])
        pivot = df_in[["symbol", ret_col]].copy()
        pivot_wide = pivot.pivot_table(index=pivot.index, columns="symbol",
                                       values=ret_col, aggfunc="last")
        rank_wide = pivot_wide.rank(axis=1, pct=True)
        rank_long = rank_wide.stack(future_stack=True).rename(out_name).reset_index()
        rank_long.columns = ["ts", "symbol", out_name]
        return rank_long.set_index(["ts", "symbol"])[out_name]

    cs_features = {}

    # Cross-sectional ranks of returns
    for ret_col, out_name in [
        ("ret_1", "cs_rank_ret_1d"),
        ("ret_5", "cs_rank_ret_5d"),
        ("ret_20", "cs_rank_ret_20d"),
    ]:
        print(f"  Computing {out_name}...")
        rank_s = make_cs_rank(df, ret_col, out_name)
        cs_features[out_name] = rank_s

    # Cross-sectional rank of relative strength vs NIFTY
    if "nifty_ret_5d" in df.columns and "ret_5" in df.columns:
        df["_rs_nifty_5d"] = df["ret_5"] - df["nifty_ret_5d"]
        rank_s = make_cs_rank(df, "_rs_nifty_5d", "cs_rank_rs_nifty")
        cs_features["cs_rank_rs_nifty"] = rank_s
        df = df.drop(columns=["_rs_nifty_5d"])

    # Cross-sectional rank of RSI
    if "rsi_14" in df.columns:
        rank_s = make_cs_rank(df, "rsi_14", "cs_rank_rsi")
        cs_features["cs_rank_rsi"] = rank_s

    # ── STEP 2: Regime features ────────────────────────────────────────────────
    print("Computing regime features...")

    if nifty_ret_20d is not None:
        # Align nifty 20d return to dataset dates
        nifty_regime_20d = nifty_ret_20d.reindex(df.index, method="ffill")
        cs_features["nifty_trend_20d"] = nifty_regime_20d

        # Trend regime indicator (-1/0/+1)
        regime = pd.Series(0.0, index=nifty_regime_20d.index)
        regime[nifty_regime_20d > 0.03]  = 1.0   # momentum (NIFTY up >3%)
        regime[nifty_regime_20d < -0.03] = -1.0   # reversal (NIFTY down >3%)
        cs_features["trend_regime"] = regime

    # Market breadth: fraction of stocks with positive 5d return (per date)
    if "ret_5" in df.columns and "symbol" in df.columns:
        print("  Computing market_breadth...")
        ret5_pivot = df[["symbol", "ret_5"]].pivot_table(
            index=df.index, columns="symbol", values="ret_5", aggfunc="last"
        )
        breadth_per_day = (ret5_pivot > 0).mean(axis=1).rename("market_breadth_5d")
        cs_features["market_breadth_5d"] = breadth_per_day

    # Beta-adjusted return: stock_5d - nifty_5d (simple beta=1 neutralization)
    if "ret_5" in df.columns and "nifty_ret_5d" in df.columns:
        df["beta_adj_ret_5d"] = df["ret_5"] - df["nifty_ret_5d"]
        rank_s = make_cs_rank(df, "beta_adj_ret_5d", "cs_rank_beta_adj")
        cs_features["cs_rank_beta_adj"] = rank_s
        # Keep raw beta_adj too
        cs_features["beta_adj_ret_5d"] = df["beta_adj_ret_5d"]

    # Sector relative momentum
    if "sector_momentum" in df.columns and "ret_5" in df.columns:
        # Sector-relative: stock return minus average sector return
        df["_sector_rel"] = df["ret_5"] - df["sector_momentum"].fillna(df["ret_5"].median())
        rank_s = make_cs_rank(df, "_sector_rel", "cs_rank_sector_rel")
        cs_features["cs_rank_sector_rel"] = rank_s
        df = df.drop(columns=["_sector_rel"])

    # ── STEP 3: Join all new features onto the dataset ─────────────────────────
    print(f"\nJoining {len(cs_features)} new features onto dataset...")
    df_out = df.copy()
    # Save original index for restoration
    original_index = df_out.index.copy()
    df_out["_ts"] = df_out.index
    df_out["_orig_idx_pos"] = range(len(df_out))

    # Separate cs_features into (ts, symbol)-indexed and ts-only indexed
    for feat_name, feat_s in cs_features.items():
        try:
            if isinstance(feat_s, pd.Series) and isinstance(feat_s.index, pd.MultiIndex):
                # (ts, symbol) indexed — merge on both
                feat_df = feat_s.reset_index()
                feat_df.columns = ["_ts", "symbol", feat_name]
                df_out = df_out.merge(feat_df, on=["_ts", "symbol"], how="left")
            elif isinstance(feat_s, pd.Series):
                # ts-only indexed — reindex on _ts column
                ts_map = dict(zip(df_out["_ts"], range(len(df_out))))
                df_out[feat_name] = feat_s.reindex(df_out["_ts"]).values
            else:
                df_out[feat_name] = feat_s
        except Exception as e:
            print(f"  Warning: failed to join {feat_name}: {e}")

    # Restore DatetimeIndex (merge drops it)
    df_out.index = original_index
    df_out = df_out.drop(columns=["_ts", "_orig_idx_pos"], errors="ignore")

    print(f"Final dataset shape: {df_out.shape}")

    # Check for new features
    new_feats = [f for f in cs_features.keys() if f in df_out.columns]
    print(f"New features added: {new_feats}")

    for f in new_feats[:5]:
        non_nan = df_out[f].notna().mean()
        print(f"  {f}: mean={df_out[f].mean():.4f}, non_nan={non_nan:.1%}")

    # ── STEP 4: Save ───────────────────────────────────────────────────────────
    out_parquet = OUT_DIR / "data.parquet"
    df_out.to_parquet(str(out_parquet))

    metadata = {
        "dataset_id": "v2c_cs_regime",
        "based_on": "v2b_ranked",
        "new_features": new_feats,
        "n_rows": len(df_out),
        "n_total_cols": df_out.shape[1],
        "date_start": str(df_out.index.min()),
        "date_end": str(df_out.index.max()),
        "n_symbols": int(df_out["symbol"].nunique()) if "symbol" in df_out.columns else 0,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\n✓ Saved: {out_parquet}")
    print(f"  New features: {new_feats}")
    print(f"\nRun training with:")
    print(f"  python3 scripts/train_v2.py --dataset {out_parquet}")


if __name__ == "__main__":
    main()
