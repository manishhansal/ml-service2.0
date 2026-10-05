"""
TRUE MODEL INFERENCE ENGINE
Loads the actual trained LightGBM artifact (expanded_lgbm v1.0.0-20260928053134956099)
and generates M1 (real model) predictions for the full OOS period.

This script MUST NOT fall back to proxy/label scores.
If the model cannot be loaded → MODEL_ARTIFACT_REQUIRED error.

Mode P0 = label-proxy (forbidden for performance claims)
Mode M1 = actual model predictions (this script)
Mode P2 = portfolio execution of M1 signals
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "reports" / "forensic_cert_2026_10_01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT / "artifacts/registry/expanded_lgbm/1.0.0-20260928053134956099/model.pkl"
SHADOW_JSON = ROOT / "artifacts/registry/expanded_lgbm/shadow.json"

# ── Step 1: Load model — HARD FAIL if missing ─────────────────────────────────

if not MODEL_PATH.exists():
    print("MODEL_ARTIFACT_REQUIRED: artifact not found at", MODEL_PATH, file=sys.stderr)
    sys.exit(1)

print(f"Loading model from: {MODEL_PATH}")
with open(MODEL_PATH, "rb") as f:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_dict = pickle.load(f)

estimator = model_dict["estimator"]
feature_names: list[str] = model_dict["feature_names"]
fs_version: str = model_dict.get("feature_schema_version", "unknown")
normalizer_state: dict = model_dict.get("normalizer_state", {})
calibrator = model_dict.get("calibrator")

print(f"Model loaded: fs={fs_version}, features={len(feature_names)}")
print(f"Model type: {type(estimator).__name__}")
print(f"Has calibrator: {calibrator is not None}")

# ── Step 2: Load the ORIGINAL training dataset used for this model ────────────
# The model was trained on ds-1d-20260926202149-3f078494 (fs-2.0.0)
# We evaluate OOS period from 2025-01-01 onwards.

# Find the dataset that contains the fs-2.0.0 features
OOS_START = pd.Timestamp("2025-01-01", tz="UTC")

# Use the LATEST dataset since it includes all fs-2.0.0 features as a subset
datasets = sorted(ROOT.glob("artifacts/datasets/ds-1d-*/data.parquet"))
if not datasets:
    print("No datasets found", file=sys.stderr)
    sys.exit(1)

# Load most recent dataset
latest_ds = datasets[-1]
print(f"\nLoading OOS data from: {latest_ds.parent.name}")
df_full = pd.read_parquet(str(latest_ds))
print(f"Full dataset: {len(df_full):,} rows, {len(df_full.columns)} cols")

# Filter to OOS period
df_oos = df_full[df_full.index >= OOS_START].copy()
print(f"OOS period: {len(df_oos):,} rows ({str(OOS_START)[:10]} → {str(df_oos.index.max())[:10]})")

# Verify all model features are present
missing_features = set(feature_names) - set(df_oos.columns)
if missing_features:
    print(f"CRITICAL: Missing features: {missing_features}", file=sys.stderr)
    sys.exit(1)
print(f"All {len(feature_names)} model features present ✓")

# ── Step 3: Apply normalizer ──────────────────────────────────────────────────
# Extract only the model's feature columns in the correct order
X_oos = df_oos[feature_names].fillna(0.0).to_numpy(dtype=float)

# Apply normalizer state if available
if normalizer_state.get("fitted") and normalizer_state.get("specs"):
    print("Applying saved normalizer state...")
    try:
        from src.features.normalizer import FeatureNormalizer
        norm = FeatureNormalizer()
        norm.load_state(normalizer_state)
        X_df = pd.DataFrame(X_oos, columns=feature_names)
        X_df = norm.transform(X_df)
        X_oos = X_df.values
        print("Normalization applied ✓")
    except Exception as e:
        print(f"WARNING: Could not apply normalizer: {e} — using raw features")
        X_oos = df_oos[feature_names].fillna(0.0).to_numpy(dtype=float)

# ── Step 4: Generate TRUE MODEL predictions (M1) ──────────────────────────────
print(f"\nGenerating M1 predictions for {len(X_oos):,} OOS observations...")
try:
    raw_scores = estimator.predict(X_oos)
    print(f"Raw scores: mean={raw_scores.mean():.4f}, std={raw_scores.std():.4f}, "
          f"min={raw_scores.min():.4f}, max={raw_scores.max():.4f}")
except Exception as e:
    print(f"INFERENCE FAILED: {e}", file=sys.stderr)
    sys.exit(1)

# ── Step 5: Apply calibration (if available) ──────────────────────────────────
calibrated_scores = raw_scores.copy()
if calibrator is not None:
    try:
        # _IsotonicWrapper.predict_proba returns (n, 2) array
        cal_probs = calibrator.predict_proba(raw_scores.reshape(-1, 1))
        calibrated_scores = cal_probs[:, 1]  # positive-class probability
        print(f"Calibrated scores: mean={calibrated_scores.mean():.4f}, "
              f"std={calibrated_scores.std():.4f}")
    except Exception as e:
        print(f"WARNING: Calibration failed ({e}), using raw scores")
        calibrated_scores = raw_scores.copy()

# ── Step 6: Build result DataFrame ───────────────────────────────────────────
result_df = pd.DataFrame({
    "timestamp":         df_oos.index,
    "symbol":            df_oos["symbol"] if "symbol" in df_oos.columns else "UNKNOWN",
    "raw_score":         raw_scores,
    "calibrated_score":  calibrated_scores,
    "true_label":        df_oos["label"].fillna(-1).astype(int) if "label" in df_oos.columns else -1,
    "realized_return":   df_oos["realized_return"] if "realized_return" in df_oos.columns else np.nan,
    "realized_return_net": df_oos["realized_return_net"] if "realized_return_net" in df_oos.columns else np.nan,
    "outcome":           df_oos["outcome"] if "outcome" in df_oos.columns else "UNKNOWN",
})

# Direction: classify by cross-sectional ranking (top/bottom 20% = actionable)
# This is the CORRECT approach for a cross-sectional ranking model
# rather than threshold > 0.5
symbols = result_df["symbol"].unique().tolist()
dates = result_df["timestamp"].unique()

# Compute cross-sectional ranks per date
result_df["cs_rank_pct"] = np.nan
for dt in dates:
    mask = result_df["timestamp"] == dt
    scores_today = result_df.loc[mask, "calibrated_score"]
    if len(scores_today) > 0:
        ranks = scores_today.rank(pct=True, ascending=True)
        result_df.loc[mask, "cs_rank_pct"] = ranks

# Signal: top 20% = LONG (+1), bottom 20% = SHORT (-1), middle = HOLD (0)
result_df["direction_top20"] = 0
result_df.loc[result_df["cs_rank_pct"] >= 0.80, "direction_top20"] = 1
result_df.loc[result_df["cs_rank_pct"] <= 0.20, "direction_top20"] = -1

# Also compute simple threshold-based direction
result_df["direction_threshold"] = result_df["calibrated_score"].apply(
    lambda s: 1 if s > 0.55 else (-1 if s < 0.45 else 0)
)

print(f"\nDirection distribution (top/bottom 20%):")
print(f"  LONG:  {(result_df['direction_top20']==1).sum():,}")
print(f"  SHORT: {(result_df['direction_top20']==-1).sum():,}")
print(f"  HOLD:  {(result_df['direction_top20']==0).sum():,}")

print(f"\nDirection distribution (threshold 0.45/0.55):")
print(f"  LONG:  {(result_df['direction_threshold']==1).sum():,}")
print(f"  SHORT: {(result_df['direction_threshold']==-1).sum():,}")
print(f"  HOLD:  {(result_df['direction_threshold']==0).sum():,}")

# ── Step 7: Compute M1 predictive quality metrics ─────────────────────────────
from scipy.stats import spearmanr

print("\n\n=== M1 MODEL PREDICTIVE QUALITY ===")
valid_mask = result_df["true_label"].isin([0, 1]) & result_df["realized_return"].notna()
df_valid = result_df[valid_mask]

if len(df_valid) >= 100:
    # Binary accuracy
    pred_binary = (df_valid["calibrated_score"] >= 0.5).astype(int)
    accuracy = (pred_binary == df_valid["true_label"]).mean()
    print(f"Binary accuracy (threshold 0.5): {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Precision for actionable signals
    long_mask = df_valid["direction_top20"] == 1
    short_mask = df_valid["direction_top20"] == -1

    if long_mask.sum() > 0:
        long_prec = (df_valid.loc[long_mask, "realized_return"] > 0).mean()
        print(f"LONG precision (top 20%): {long_prec:.4f} ({long_prec*100:.1f}%)")

    if short_mask.sum() > 0:
        short_prec = (df_valid.loc[short_mask, "realized_return"] < 0).mean()
        print(f"SHORT precision (bottom 20%): {short_prec:.4f} ({short_prec*100:.1f}%)")

    # IC (Spearman rank correlation between scores and returns)
    ic_val, ic_pval = spearmanr(df_valid["calibrated_score"], df_valid["realized_return"])
    print(f"IC (Spearman, score vs realized_return): {ic_val:.4f} (p={ic_pval:.6f})")

    # Mean return by score quartile
    df_valid_copy = df_valid.copy()
    try:
        df_valid_copy["score_quartile"] = pd.qcut(
            df_valid_copy["calibrated_score"], q=4, duplicates="drop"
        )
    except Exception:
        df_valid_copy["score_quartile"] = pd.cut(
            df_valid_copy["calibrated_score"],
            bins=4, labels=["Q1","Q2","Q3","Q4"]
        )
    qret = df_valid_copy.groupby("score_quartile", observed=True)["realized_return"].agg(["mean","count"])
    print(f"\nMean return by score quartile:")
    for idx, row in qret.iterrows():
        print(f"  {idx}: mean={row['mean']*100:.3f}%  n={int(row['count'])}")

    # Expected value
    long_returns = df_valid.loc[long_mask, "realized_return_net"]
    short_returns = df_valid.loc[short_mask, "realized_return_net"] * -1  # invert for short
    all_actionable = pd.concat([long_returns, short_returns]).dropna()
    if len(all_actionable) > 0:
        ev = all_actionable.mean()
        print(f"\nExpected value (actionable signals, net): {ev*100:.4f}%/trade")
        print(f"Win rate (actionable, net): {(all_actionable > 0).mean()*100:.1f}%")
        print(f"n actionable signals: {len(all_actionable)}")
    
    # Calibration assessment
    print(f"\nCalibration:")
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    df_valid_copy["score_bin"] = pd.cut(df_valid_copy["calibrated_score"], bins=bins)
    cal_table = df_valid_copy.groupby("score_bin", observed=True)["true_label"].agg(["mean","count"])
    for idx, row in cal_table.iterrows():
        midpt = (idx.left + idx.right) / 2
        print(f"  score~{midpt:.2f}: actual_pos_rate={row['mean']:.3f}  n={int(row['count'])}")

print(f"\n\nSaving M1 predictions to {OUT_DIR}/")
result_df.to_parquet(str(OUT_DIR / "m1_oos_predictions.parquet"), index=False)
result_df.to_csv(str(OUT_DIR / "m1_oos_predictions.csv"), index=False)
print(f"Saved: m1_oos_predictions.parquet ({len(result_df):,} rows)")

# Summary json
summary = {
    "mode": "M1",
    "model_version": "1.0.0-20260928053134956099",
    "feature_schema": fs_version,
    "n_features": len(feature_names),
    "oos_start": str(OOS_START)[:10],
    "n_oos_rows": len(result_df),
    "n_symbols": result_df["symbol"].nunique() if "symbol" in result_df.columns else 0,
    "score_mean": float(calibrated_scores.mean()),
    "score_std": float(calibrated_scores.std()),
}
(OUT_DIR / "m1_inference_summary.json").write_text(json.dumps(summary, indent=2))
print("Saved: m1_inference_summary.json")
