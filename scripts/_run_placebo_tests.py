"""
PLACEBO AND NEGATIVE-CONTROL TESTS
Verifies that model performance is genuine, not an artifact.

Tests:
  P1: Label shuffle (randomize y) → IC should drop to ~0
  P2: Timestamp shuffle (destroy temporal relationship) → IC should drop to ~0
  P3: Random prediction baseline → IC should be ~0
  P4: Raw vs calibrated score comparison
  P5: Permutation IC test (bootstrap null distribution)
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "reports" / "forensic_cert_2026_10_01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT / "artifacts/registry/expanded_lgbm/1.0.0-20260928053134956099/model.pkl"
OOS_START = pd.Timestamp("2025-01-01", tz="UTC")
RNG_SEED = 42

print("="*70)
print("PLACEBO AND NEGATIVE-CONTROL TESTS")
print("="*70)

# Load model and data
print("\nLoading model...")
with open(MODEL_PATH, "rb") as f:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_dict = pickle.load(f)

estimator = model_dict["estimator"]
feature_names = model_dict["feature_names"]

datasets = sorted(ROOT.glob("artifacts/datasets/ds-1d-*/data.parquet"))
latest_ds = datasets[-1]
print(f"Loading dataset: {latest_ds.parent.name}")
df_full = pd.read_parquet(str(latest_ds))
df_oos = df_full[df_full.index >= OOS_START].copy()

valid_mask = df_oos["label"].isin([0, 1]) & df_oos["realized_return"].notna()
df_v = df_oos[valid_mask].copy()

X = df_v[feature_names].fillna(0.0).to_numpy(dtype=float)
y_true = df_v["realized_return"].to_numpy(dtype=float)
y_label = df_v["label"].astype(int).to_numpy()

print(f"OOS sample: {len(df_v):,} rows with valid labels")

# True model scores
true_scores = estimator.predict(X)
true_ic, true_pval = spearmanr(true_scores, y_true)
true_acc = ((true_scores >= 0.5).astype(int) == y_label).mean()

print(f"\n--- TRUE MODEL (M1) ---")
print(f"IC (raw score vs realized_return): {true_ic:.6f}")
print(f"IC p-value: {true_pval:.6f}")
print(f"Accuracy: {true_acc:.4f}")

results = {
    "true_model": {
        "ic": float(true_ic),
        "ic_pval": float(true_pval),
        "accuracy": float(true_acc),
        "n": len(df_v),
    }
}

# ── P1: LABEL SHUFFLE ─────────────────────────────────────────────────────────
print("\n--- P1: LABEL SHUFFLE (null hypothesis) ---")
rng = np.random.default_rng(RNG_SEED)
shuffled_ics = []
n_perms = 1000
for _ in range(n_perms):
    y_shuffled = rng.permutation(y_true)
    ic, _ = spearmanr(true_scores, y_shuffled)
    shuffled_ics.append(ic)

null_mean = np.mean(shuffled_ics)
null_std  = np.std(shuffled_ics)
# p-value: fraction of null ICs ≥ true IC
p_permutation = np.mean([ic >= true_ic for ic in shuffled_ics])

print(f"Null IC distribution: mean={null_mean:.6f}, std={null_std:.6f}")
print(f"True IC: {true_ic:.6f}")
print(f"Permutation p-value (one-sided, H1: IC > 0): {p_permutation:.4f}")
print(f"Z-score: {(true_ic - null_mean) / (null_std + 1e-10):.2f}")
print(f"ASSESSMENT: {'PASS (IC > null)' if p_permutation < 0.05 else 'FAIL (IC indistinguishable from null)'}")

results["p1_label_shuffle"] = {
    "null_ic_mean": float(null_mean),
    "null_ic_std": float(null_std),
    "p_permutation": float(p_permutation),
    "z_score": float((true_ic - null_mean) / (null_std + 1e-10)),
    "n_perms": n_perms,
    "assessment": "PASS" if p_permutation < 0.05 else "FAIL",
}

# ── P2: TIMESTAMP SHUFFLE ─────────────────────────────────────────────────────
print("\n--- P2: TIMESTAMP SHUFFLE (destroy temporal relationship) ---")
df_ts_shuffle = df_v.copy()
df_ts_shuffle["realized_return"] = rng.permutation(df_ts_shuffle["realized_return"].values)
X_ts = df_ts_shuffle[feature_names].fillna(0.0).to_numpy(dtype=float)
scores_ts = estimator.predict(X_ts)
ic_ts, pval_ts = spearmanr(scores_ts, df_ts_shuffle["realized_return"])
print(f"IC after timestamp shuffle: {ic_ts:.6f}")
print(f"Expected: ~0 (since features still reflect real prices but labels are scrambled)")
print(f"ASSESSMENT: {'Confirms no trivial pattern' if abs(ic_ts) < 0.02 else 'WARNING: pattern survives shuffle'}")

results["p2_timestamp_shuffle"] = {"ic": float(ic_ts), "pval": float(pval_ts)}

# ── P3: RANDOM PREDICTION BASELINE ────────────────────────────────────────────
print("\n--- P3: RANDOM PREDICTION BASELINE ---")
random_scores = rng.uniform(0, 1, len(y_true))
ic_rand, pval_rand = spearmanr(random_scores, y_true)
acc_rand = ((random_scores >= 0.5).astype(int) == y_label).mean()
print(f"Random IC: {ic_rand:.6f}  accuracy: {acc_rand:.4f}")
print(f"True model vs random: IC delta = {true_ic - ic_rand:.6f}")

results["p3_random_baseline"] = {"ic": float(ic_rand), "accuracy": float(acc_rand)}

# ── P4: RAW vs CALIBRATED ─────────────────────────────────────────────────────
print("\n--- P4: RAW vs CALIBRATED SCORE COMPARISON ---")
calibrator = model_dict.get("calibrator")
if calibrator is not None:
    try:
        cal_probs = calibrator.predict_proba(true_scores.reshape(-1, 1))
        cal_scores = cal_probs[:, 1]
        ic_cal, pval_cal = spearmanr(cal_scores, y_true)
        print(f"Raw score IC: {true_ic:.6f}")
        print(f"Calibrated score IC: {ic_cal:.6f}")
        print(f"Raw score std: {true_scores.std():.6f}")
        print(f"Calibrated score std: {cal_scores.std():.6f}")
        print(f"Calibration verdict: {'calibrator DESTROYS information' if ic_cal < true_ic else 'calibrator OK'}")
        results["p4_calibration"] = {
            "raw_ic": float(true_ic),
            "calibrated_ic": float(ic_cal),
            "raw_std": float(true_scores.std()),
            "calibrated_std": float(cal_scores.std()),
            "verdict": "DESTROYS_INFORMATION" if ic_cal < true_ic else "OK",
        }
    except Exception as e:
        print(f"Calibration test failed: {e}")

# ── P5: DIRECTION INVERSION TEST ──────────────────────────────────────────────
print("\n--- P5: DIRECTION INVERSION TEST ---")
inverted_scores = 1.0 - true_scores
ic_inv, _ = spearmanr(inverted_scores, y_true)
acc_inv = ((inverted_scores >= 0.5).astype(int) == y_label).mean()
print(f"Inverted score IC: {ic_inv:.6f}  accuracy: {acc_inv:.4f}")
print(f"Original IC: {true_ic:.6f}  accuracy: {true_acc:.4f}")
print(f"ASSESSMENT: {'Model direction is correct (original > inverted)' if true_ic > ic_inv else 'WARNING: Inverted is better!'}")

results["p5_direction_inversion"] = {
    "inverted_ic": float(ic_inv),
    "inverted_accuracy": float(acc_inv),
    "direction_correct": true_ic > ic_inv,
}

# ── P6: IN-SAMPLE vs OOS IC comparison ────────────────────────────────────────
print("\n--- P6: IN-SAMPLE vs TRUE OOS COMPARISON ---")
# Load the original training dataset
train_ds = ROOT / "artifacts/datasets/ds-1d-20260926202149-3f078494/data.parquet"
if train_ds.exists():
    df_train = pd.read_parquet(str(train_ds))
    # Take the "test" portion from walk-forward (last 20% chronologically)
    cutoff = df_train.index.min() + (df_train.index.max() - df_train.index.min()) * 0.8
    df_cv_test = df_train[df_train.index >= cutoff]
    valid_cv = df_cv_test["label"].isin([0, 1]) & df_cv_test["realized_return"].notna()
    df_cv_valid = df_cv_test[valid_cv].copy()
    if len(df_cv_valid) > 100:
        X_cv = df_cv_valid[feature_names].fillna(0.0).to_numpy(dtype=float)
        y_cv = df_cv_valid["realized_return"].to_numpy(dtype=float)
        scores_cv = estimator.predict(X_cv)
        ic_cv, pval_cv = spearmanr(scores_cv, y_cv)
        print(f"Within-training-period (last 20%) IC: {ic_cv:.6f}")
        print(f"True OOS (2025+) IC:                  {true_ic:.6f}")
        print(f"IC degradation from train-period to OOS: {ic_cv - true_ic:.6f}")
        results["p6_in_sample_vs_oos"] = {
            "training_period_last20pct_ic": float(ic_cv),
            "true_oos_2025_ic": float(true_ic),
            "degradation": float(ic_cv - true_ic),
        }
else:
    print("Original training dataset not found")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print("PLACEBO TEST SUMMARY")
print("="*70)
print(f"True model IC (OOS 2025+):    {true_ic:.6f}")
print(f"Null IC distribution:         mean={null_mean:.6f}, std={null_std:.6f}")
print(f"Permutation p-value:          {results['p1_label_shuffle']['p_permutation']:.4f}")
print(f"Calibration impact:           {'DESTROYS INFO' if results.get('p4_calibration', {}).get('verdict') == 'DESTROYS_INFORMATION' else 'OK'}")
print(f"Direction correct:            {results['p5_direction_inversion']['direction_correct']}")

print(f"\nVERDICT: {'MINIMAL REAL SIGNAL (but statistically significant)' if true_pval < 0.05 else 'NO STATISTICAL SIGNAL'}")

(OUT_DIR / "placebo_test_results.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved: placebo_test_results.json")
