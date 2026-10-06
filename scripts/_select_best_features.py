"""
Feature selection by individual OOS IC.
Only keep new v2d features where individual IC > 0.003 in OOS period.
This prevents adding noisy features that overfit training.
"""
import pickle, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OOS_START = pd.Timestamp("2025-01-01", tz="UTC")

# Load v2d dataset
v2d = ROOT / "artifacts/datasets/v2d_maxalpha/data.parquet"
df = pd.read_parquet(str(v2d))
if df.index.tz is None:
    df.index = pd.to_datetime(df.index).tz_localize("UTC")
df_oos = df[df.index >= OOS_START].copy()
print(f"OOS rows: {len(df_oos):,}")

# Label for IC measurement
if "label_v2b" not in df_oos.columns:
    raise ValueError("label_v2b not found")

valid = df_oos["label_v2b"].notna()
df_v = df_oos[valid].copy()
y = df_v["label_v2b"].to_numpy(dtype=float)

# New v2d features (not in v2c)
from src.training.v2_pipeline import V2TrainingConfig
v2c_features = set([
    "ret_1", "ret_5", "ret_10", "ret_20", "log_ret_1",
    "vol_5", "vol_10", "vol_20", "atr_14_pct",
    "rel_volume_20", "volume_zscore_20", "vwap_distance_pct",
    "rsi_14", "macd_hist", "stoch_k_14",
    "ema_5_20", "ema_10_50", "adx_14",
    "hl_range_pct", "close_position", "gap_pct",
    "bb_zscore_20", "skew_20", "kurt_20",
    "ret_3", "ret_60", "mom_accel_5", "mom_accel_20",
    "ret_60_rel_vol", "vol_regime_zscore", "vol_regime_pctile",
    "vol_expanding", "vol_ratio", "trend_strength", "trend_direction",
    "ema_spread", "trend_persistence", "gap_magnitude",
    "gap_regime_rolling", "gap_direction",
    "weekday", "weekday_sin", "weekday_cos",
    "month_end_proximity", "quarter_end", "is_monday", "is_friday",
    "price_zscore_60", "price_zscore_20",
    "vol_norm_ret_5", "vol_norm_ret_20",
    "parkinson_vol", "garman_klass_vol", "vol_of_vol_20", "atr_zscore",
    "cs_rank_ret_1d", "cs_rank_ret_5d", "cs_rank_ret_20d",
    "cs_rank_rs_nifty", "cs_rank_rsi", "cs_rank_beta_adj",
    "nifty_trend_20d", "trend_regime", "market_breadth_5d", "beta_adj_ret_5d",
])

new_v2d_candidates = [c for c in df_v.columns if c not in v2c_features and
                       c not in {"symbol","label","label_v1","label_v2","label_v2b",
                                  "realized_return","realized_return_net","outcome",
                                  "execution_model","is_economic_evidence","excess_ret_raw"}]

print(f"\nMeasuring OOS IC for {len(new_v2d_candidates)} candidate features:")
print(f"{'Feature':35s} {'IC':>8s} {'p-val':>8s} {'Include?':>10s}")
print("-" * 65)

selected = []
IC_THRESHOLD = 0.003  # minimum individual OOS IC to include

for feat in sorted(new_v2d_candidates):
    if feat not in df_v.columns:
        continue
    x = df_v[feat].fillna(df_v[feat].median()).to_numpy(dtype=float)
    if np.std(x) < 1e-8:
        continue
    ic, pval = spearmanr(x, y)
    include = abs(ic) > IC_THRESHOLD and pval < 0.05
    mark = "✓ KEEP" if include else "✗ skip"
    print(f"  {feat:33s} {ic:+.4f}   {pval:.4f}   {mark}")
    if include:
        selected.append((feat, float(ic), float(pval)))

print(f"\nSelected {len(selected)} features with |IC| > {IC_THRESHOLD} and p < 0.05:")
for feat, ic, pval in sorted(selected, key=lambda x: -abs(x[1])):
    print(f"  {feat:35s} IC={ic:+.4f}")
