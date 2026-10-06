"""Deep-inspect the model artifact dict structure."""
import pickle
import json
import pathlib
import hashlib
import sys

ROOT = pathlib.Path(__file__).parent.parent

def inspect_model_dict(path):
    p = pathlib.Path(path)
    if not p.exists():
        print(f"NOT FOUND: {path}")
        return

    print(f"\n{'='*70}")
    print(f"ARTIFACT: {path}")
    print(f"Size: {p.stat().st_size:,} bytes")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    print(f"SHA256: {sha}")

    with open(p, "rb") as f:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = pickle.load(f)

    print(f"Type: {type(obj).__name__}")

    if isinstance(obj, dict):
        print(f"Dict keys: {list(obj.keys())}")
        for k, v in obj.items():
            vtype = type(v).__name__
            vmod  = type(v).__module__
            print(f"  [{k}] → {vtype}@{vmod}", end="")
            # Check if it's a LightGBM Booster
            if hasattr(v, "num_trees"):
                print(f" [LGBM Booster: {v.num_trees()} trees]", end="")
                fn = v.feature_name()
                print(f" features[{len(fn)}]={fn[:5]}...", end="")
            elif hasattr(v, "predict"):
                print(f" [has .predict()]", end="")
            elif isinstance(v, (list, tuple)):
                print(f" len={len(v)}", end="")
            elif isinstance(v, dict):
                print(f" sub-keys={list(v.keys())[:5]}", end="")
            elif isinstance(v, str):
                print(f" = '{v[:60]}'", end="")
            elif isinstance(v, (int, float)):
                print(f" = {v}", end="")
            print()

    # Try extracting the booster from common wrapper patterns
    booster = None
    if isinstance(obj, dict):
        for k in ["model", "booster", "estimator", "lgbm", "clf", "regressor"]:
            if k in obj:
                candidate = obj[k]
                if hasattr(candidate, "num_trees"):
                    booster = candidate
                    print(f"\nFound LGBM Booster at key '{k}'")
                    break
        # Try nested
        if booster is None:
            for k, v in obj.items():
                if hasattr(v, "num_trees"):
                    booster = v
                    print(f"\nFound LGBM Booster at key '{k}'")
                    break
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if hasattr(v2, "num_trees"):
                            booster = v2
                            print(f"\nFound LGBM Booster at key '{k}[{k2}]'")
                            break

    if booster is not None:
        print(f"\nLGBM Booster details:")
        print(f"  n_trees: {booster.num_trees()}")
        fn = booster.feature_name()
        print(f"  n_features: {len(fn)}")
        print(f"  features[:10]: {fn[:10]}")
        params = booster.params
        print(f"  params: {dict(list(params.items())[:8])}")
        # Get feature importance
        fi = booster.feature_importance(importance_type="gain")
        import numpy as np
        idx = np.argsort(fi)[::-1][:10]
        print(f"  Top 10 features by gain:")
        for i in idx:
            print(f"    {fn[i]}: {fi[i]:.2f}")
    else:
        print("\nNo LGBM Booster found in dict — checking if sklearn wrapper")
        if isinstance(obj, dict) and "feature_names" in obj:
            print(f"  feature_names: {obj['feature_names'][:10]}")

# Inspect the SHADOW model
inspect_model_dict("artifacts/registry/expanded_lgbm/1.0.0-20260928053134956099/model.pkl")
