"""Inspect all model artifacts."""
import pickle
import json
import pathlib
import hashlib

ROOT = pathlib.Path(__file__).parent.parent

def inspect_pkl(path):
    p = pathlib.Path(path)
    if not p.exists():
        return "NOT FOUND"
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        t = type(obj).__name__
        mod = type(obj).__module__
        sha = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        attrs = []
        if hasattr(obj, "num_trees"):
            attrs.append(f"trees={obj.num_trees()}")
        if hasattr(obj, "feature_name"):
            fn = obj.feature_name()
            attrs.append(f"features={len(fn)}: {fn[:3]}")
        if hasattr(obj, "params"):
            pr = obj.params
            for k in ["objective", "learning_rate", "num_leaves", "n_estimators"]:
                if k in pr:
                    attrs.append(f"{k}={pr[k]}")
        return f"{t}@{mod} sha={sha} | " + " | ".join(attrs)
    except Exception as e:
        return f"ERROR: {e}"

# All model paths to check
paths = [
    "artifacts/registry/expanded_lgbm/1.0.0-20260928053134956099/model.pkl",
    "artifacts/expanded_lgbm/1.0.0-20260928053134956099/model.pkl",
    "artifacts/expanded_lgbm/5.0.0-20261001033431388223/model.pkl",
    "artifacts/expanded_h5_fs-4_0_0/h5-20261001034834608555/model.pkl",
    "artifacts/expanded_h1_fs-4_0_0/h1-20261001034132928311/model.pkl",
    "artifacts/registry/stage_a_1d/1.0.0-20260925080931531542/model.pkl",
]

print("=" * 80)
print("MODEL ARTIFACT INVENTORY")
print("=" * 80)
for rel in paths:
    p = ROOT / rel
    print(f"\n{rel}")
    print(f"  exists: {p.exists()}")
    if p.exists():
        print(f"  size: {p.stat().st_size:,} bytes")
        print(f"  inspect: {inspect_pkl(str(p))}")
        meta = p.parent / "metadata.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            keys = ["model_version", "feature_schema_version", "label_schema_version",
                    "training_date_range", "n_samples", "n_features", "model_objective",
                    "dataset_id", "ic_mean", "sharpe_net", "pbo", "gate_results", "outcome"]
            for k in keys:
                if k in m:
                    print(f"  {k}: {m[k]}")

# Also check shadow.json
shadow = ROOT / "artifacts/registry/expanded_lgbm/shadow.json"
if shadow.exists():
    print(f"\n\nSHADOW CONFIG: {shadow}")
    s = json.loads(shadow.read_text())
    for k, v in s.items():
        print(f"  {k}: {v}")
