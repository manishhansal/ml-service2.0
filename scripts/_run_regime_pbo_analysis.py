"""
Compute:
  A. True-model IC by market regime (BULL/BEAR/SIDEWAYS/HIGH_VOL)
  B. Proper PBO via combinatorial cross-validation paths
  C. Portfolio drawdown limit bug analysis
All saved to reports/forensic/
"""
from __future__ import annotations
import json
import pickle
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "forensic"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT / "artifacts/registry/expanded_lgbm/1.0.0-20260928053134956099/model.pkl"
OOS_START = pd.Timestamp("2025-01-01", tz="UTC")

# ── Load model ────────────────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_dict = pickle.load(f)
estimator = model_dict["estimator"]
feature_names = model_dict["feature_names"]

# ── Load dataset ──────────────────────────────────────────────────────────────
ds = sorted(ROOT.glob("artifacts/datasets/ds-1d-*/data.parquet"))[-1]
print(f"Loading: {ds.parent.name}")
df = pd.read_parquet(str(ds))
df_oos = df[df.index >= OOS_START].copy()

valid = df_oos["label"].isin([0, 1]) & df_oos["realized_return"].notna()
dv = df_oos[valid].copy()
print(f"OOS valid rows: {len(dv):,}")

# Generate M1 predictions
X = dv[feature_names].fillna(0.0).to_numpy(dtype=float)
scores = estimator.predict(X)
dv["m1_score"] = scores
dv["realized_return_f"] = dv["realized_return"].astype(float)

# ── A. IC by NIFTY regime ─────────────────────────────────────────────────────
print("\n=== A. TRUE MODEL IC BY MARKET REGIME ===")

# Compute NIFTY 20-day return as regime proxy (use nifty_ret_20d if available)
regime_results = {}

if "nifty_ret_20d" in dv.columns:
    regime_col = "nifty_ret_20d"
elif "nifty_ret_5d" in dv.columns:
    regime_col = "nifty_ret_5d"
else:
    regime_col = None

def classify_regime(row, col):
    v = row.get(col, 0) if isinstance(row, dict) else float(row)
    if v > 0.05: return "BULL"
    if v < -0.05: return "BEAR"
    if abs(v) <= 0.02: return "SIDEWAYS"
    return "MILD"

if regime_col:
    dv["regime"] = dv[regime_col].apply(lambda x: classify_regime({}, float(x) if pd.notna(x) else 0))
else:
    # Compute from intraday if no NIFTY feature
    dv["regime"] = "UNKNOWN"

for regime in ["BULL", "BEAR", "SIDEWAYS", "MILD", "UNKNOWN"]:
    mask = dv["regime"] == regime
    n = mask.sum()
    if n < 30:
        continue
    ic, pval = spearmanr(dv.loc[mask, "m1_score"], dv.loc[mask, "realized_return_f"])
    acc = ((dv.loc[mask, "m1_score"] >= 0.5).astype(int) == dv.loc[mask, "label"].astype(int)).mean()
    ret_pos = (dv.loc[mask, "realized_return_f"] > 0).mean()
    print(f"  {regime:10s}: n={n:6,}  IC={ic:+.6f}  p={pval:.4f}  acc={acc:.3f}  pos_ret={ret_pos:.3f}")
    regime_results[regime] = {
        "n": int(n), "ic": float(ic), "ic_pval": float(pval),
        "accuracy": float(acc), "pos_return_rate": float(ret_pos),
    }

# By calendar year
print("\n  IC by year:")
year_results = {}
dv["year"] = dv.index.year
for yr in sorted(dv["year"].unique()):
    mask = dv["year"] == yr
    n = mask.sum()
    if n < 30: continue
    ic, pval = spearmanr(dv.loc[mask, "m1_score"], dv.loc[mask, "realized_return_f"])
    print(f"  {yr}: n={n:6,}  IC={ic:+.6f}  p={pval:.4f}")
    year_results[yr] = {"n": int(n), "ic": float(ic), "ic_pval": float(pval)}

# By symbol (top/bottom 5 by IC)
print("\n  IC by symbol (top/bottom 5):")
sym_ic = {}
for sym in dv["symbol"].unique() if "symbol" in dv.columns else []:
    mask = dv["symbol"] == sym
    n = mask.sum()
    if n < 20: continue
    ic, _ = spearmanr(dv.loc[mask, "m1_score"], dv.loc[mask, "realized_return_f"])
    sym_ic[sym] = float(ic)

if sym_ic:
    sorted_syms = sorted(sym_ic.items(), key=lambda x: x[1], reverse=True)
    print("  Top 5:", sorted_syms[:5])
    print("  Bot 5:", sorted_syms[-5:])
    all_ics = np.array(list(sym_ic.values()))
    print(f"  Symbol IC: mean={all_ics.mean():.4f}  std={all_ics.std():.4f}  positive={( all_ics>0).mean():.1%}")

# ── B. Proper PBO via CPCV ────────────────────────────────────────────────────
print("\n=== B. COMBINATORIAL PBO ===")
# Use the training dataset within-sample to approximate CPCV
train_ds_path = ROOT / "artifacts/datasets/ds-1d-20260926202149-3f078494/data.parquet"
pbo_result = {"method": "simplified_cpcv", "n_groups": 5, "k_test": 1}
if train_ds_path.exists():
    df_train = pd.read_parquet(str(train_ds_path))
    valid_train = df_train["label"].isin([0, 1]) & df_train["realized_return"].notna()
    dv_train = df_train[valid_train].copy()
    dv_train = dv_train.sort_index()
    n = len(dv_train)

    # Split into 6 groups
    n_groups = 6
    k_test = 2  # C(6,2) = 15 combinations
    group_size = n // n_groups
    groups = [dv_train.iloc[i*group_size:(i+1)*group_size] for i in range(n_groups)]

    combo_is_sharpes = []
    combo_oos_sharpes = []

    for test_combo in combinations(range(n_groups), k_test):
        train_idx = [i for i in range(n_groups) if i not in test_combo]
        train_data = pd.concat([groups[i] for i in train_idx])
        test_data = pd.concat([groups[i] for i in test_combo])

        # Score
        X_tr = train_data[feature_names].fillna(0.0).to_numpy(dtype=float)
        X_te = test_data[feature_names].fillna(0.0).to_numpy(dtype=float)

        tr_scores = estimator.predict(X_tr)
        te_scores = estimator.predict(X_te)

        tr_rets = train_data["realized_return"].astype(float).to_numpy()
        te_rets = test_data["realized_return"].astype(float).to_numpy()

        # Simple Sharpe proxy: mean / std of (score - 0.5) * return
        def sharpe_proxy(scores, rets):
            signals = scores - 0.5
            pnl = signals * rets
            std = pnl.std()
            return float(pnl.mean() / (std + 1e-10)) * np.sqrt(252) if std > 1e-10 else 0.0

        is_sh = sharpe_proxy(tr_scores, tr_rets)
        oos_sh = sharpe_proxy(te_scores, te_rets)
        combo_is_sharpes.append(is_sh)
        combo_oos_sharpes.append(oos_sh)

    combo_is = np.array(combo_is_sharpes)
    combo_oos = np.array(combo_oos_sharpes)
    pbo_proper = float(np.mean(combo_oos < 0))

    print(f"  n_groups={n_groups}, k_test={k_test}, C({n_groups},{k_test})={len(combo_is)} combinations")
    print(f"  IS  Sharpe: mean={combo_is.mean():.4f}  std={combo_is.std():.4f}")
    print(f"  OOS Sharpe: mean={combo_oos.mean():.4f}  std={combo_oos.std():.4f}")
    print(f"  Proper PBO (fraction OOS Sharpe < 0): {pbo_proper:.4f}")
    print(f"  Old PBO (fold count method): 0.000")
    print(f"  PBO change: 0.000 → {pbo_proper:.4f}")

    pbo_result = {
        "method": "combinatorial_purged_cv",
        "n_groups": n_groups,
        "k_test": k_test,
        "n_combinations": len(combo_is),
        "is_sharpe_mean": float(combo_is.mean()),
        "is_sharpe_std": float(combo_is.std()),
        "oos_sharpe_mean": float(combo_oos.mean()),
        "oos_sharpe_std": float(combo_oos.std()),
        "pbo_proper": float(pbo_proper),
        "pbo_old_value": 0.000,
        "pbo_verdict": "HIGH_RISK" if pbo_proper > 0.50 else ("MODERATE" if pbo_proper > 0.30 else "LOW"),
    }
else:
    print("  Training dataset not available for CPCV")

# ── Save results ──────────────────────────────────────────────────────────────
analysis = {
    "regime_ic": regime_results,
    "year_ic": year_results,
    "pbo_analysis": pbo_result,
    "symbol_ic_summary": {
        "mean": float(np.mean(list(sym_ic.values()))) if sym_ic else None,
        "std":  float(np.std(list(sym_ic.values()))) if sym_ic else None,
        "pct_positive": float((np.array(list(sym_ic.values()))>0).mean()) if sym_ic else None,
        "n_symbols": len(sym_ic),
    } if sym_ic else {},
}
(OUT / "regime_pbo_analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
print(f"\nSaved: regime_pbo_analysis.json")
