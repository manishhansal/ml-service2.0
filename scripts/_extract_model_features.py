"""Extract feature names and model details from the shadow model artifact."""
import pickle
import json
import pathlib
import warnings
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
MODEL_PATH = ROOT / "artifacts/registry/expanded_lgbm/1.0.0-20260928053134956099/model.pkl"
DATASET_PATH = ROOT / "artifacts/datasets/ds-1d-20260926202149-3f078494/data.parquet"
LATEST_DS_PATH = ROOT / "artifacts/datasets/ds-1d-20261001034811-ebdf74af/data.parquet"

print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_dict = pickle.load(f)

feature_names = model_dict.get("feature_names", [])
fs_version = model_dict.get("feature_schema_version", "unknown")
estimator = model_dict.get("estimator")
calibrator = model_dict.get("calibrator")
normalizer_state = model_dict.get("normalizer_state", {})

print(f"\nFeature schema version: {fs_version}")
print(f"Number of features: {len(feature_names)}")
print(f"Feature names:\n  {feature_names}")

print(f"\nEstimator type: {type(estimator).__name__}")
print(f"Calibrator type: {type(calibrator).__name__}")
print(f"Normalization applied: {model_dict.get('normalization_applied')}")
print(f"Normalizer state keys: {list(normalizer_state.keys())}")

# Check if estimator has LGBM model inside
if hasattr(estimator, "_model"):
    lgbm = estimator._model
    print(f"\nInternal LGBM model type: {type(lgbm).__name__}")
    if hasattr(lgbm, "feature_importances_"):
        fi = lgbm.feature_importances_
        if fi is not None and len(fi) > 0:
            idx = np.argsort(fi)[::-1][:10]
            print("Top 10 features by importance:")
            for i in idx:
                if i < len(feature_names):
                    print(f"  {feature_names[i]}: {fi[i]:.0f}")

# Compare with current dataset
print("\n\n--- DATASET COMPATIBILITY CHECK ---")
import pandas as pd

# Check training dataset
if DATASET_PATH.exists():
    print(f"\nTraining dataset ({DATASET_PATH.name}):")
    meta = json.loads((DATASET_PATH.parent / "metadata.json").read_text())
    print(f"  feature_schema_version: {meta.get('feature_schema_version')}")
    print(f"  n_features: {meta.get('feature_count')}")
    df = pd.read_parquet(str(DATASET_PATH))
    print(f"  shape: {df.shape}")
    print(f"  Feature overlap: {len(set(feature_names) & set(df.columns))}/{len(feature_names)}")
    missing = set(feature_names) - set(df.columns)
    if missing:
        print(f"  MISSING from training dataset: {missing}")

# Check latest dataset
if LATEST_DS_PATH.exists():
    print(f"\nLatest dataset ({LATEST_DS_PATH.name}):")
    meta = json.loads((LATEST_DS_PATH.parent / "metadata.json").read_text())
    print(f"  feature_schema_version: {meta.get('feature_schema_version')}")
    print(f"  n_features: {meta.get('feature_count')}")
    df_latest = pd.read_parquet(str(LATEST_DS_PATH))
    print(f"  shape: {df_latest.shape}")
    print(f"  Feature overlap: {len(set(feature_names) & set(df_latest.columns))}/{len(feature_names)}")
    missing_latest = set(feature_names) - set(df_latest.columns)
    if missing_latest:
        print(f"  MISSING from latest dataset: {missing_latest}")
    else:
        print(f"  All {len(feature_names)} model features PRESENT in latest dataset ✓")

# Test inference
print("\n\n--- INFERENCE TEST ---")
try:
    if LATEST_DS_PATH.exists():
        df_test = df_latest[feature_names].iloc[:100].fillna(0.0).to_numpy(dtype=float)
        scores = estimator.predict(df_test)
        print(f"Inference test PASS: {len(scores)} scores, mean={scores.mean():.4f}, std={scores.std():.4f}")
        if calibrator is not None:
            cal_scores = calibrator.predict(scores.reshape(-1, 1))
            print(f"Calibrated scores: mean={cal_scores.mean():.4f}, std={cal_scores.std():.4f}")
except Exception as e:
    print(f"Inference test FAIL: {e}")
