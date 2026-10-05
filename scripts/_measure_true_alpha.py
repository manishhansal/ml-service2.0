"""
Measure TRUE ALPHA: IC between model predictions and actual executable returns.
  - Model predicts: rank of (close_T+7 - close_T) / close_T - nifty (close-to-close)
  - Trading uses: (close_T+7 - open_T+1) / open_T+1 - nifty (open-to-close)
  
The gap: close_T → open_T+1 is the overnight execution gap.
If this gap is adversely selected (high-score stocks gap down on open),
IC close-to-close may not translate to alpha from open-to-close.
"""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load latest v2c model
candidates = sorted(ROOT.glob("artifacts/v2_model/*/model.pkl"))
if not candidates:
    raise FileNotFoundError("No v2 model found")
latest = candidates[-1]
print(f"Loading model: {latest.parent.name}")
with open(latest, "rb") as f:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_dict = pickle.load(f)

estimator = model_dict["estimator"]
feature_names = model_dict["feature_names"]
norm_state = model_dict.get("normalizer_state", {})

# Load v2c dataset (OOS period)
v2c_path = ROOT / "artifacts/datasets/v2c_cs_regime/data.parquet"
df = pd.read_parquet(str(v2c_path))
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
OOS_START = pd.Timestamp("2025-01-01", tz="UTC")
df_oos = df[df.index >= OOS_START].copy()
print(f"OOS rows: {len(df_oos):,}")

# Get features and compute model scores
avail = [f for f in feature_names if f in df_oos.columns]
X = df_oos[avail].fillna(0.0)
if norm_state:
    try:
        from src.features.normalizer import FeatureNormalizer
        norm = FeatureNormalizer()
        norm.load_state(norm_state)
        X = norm.transform_known(X).fillna(0.0)
    except Exception as e:
        print(f"Normalizer warning: {e}")
scores = estimator.predict(X.to_numpy(dtype=float))
df_oos["model_score"] = scores

# Load NIFTY for excess return computation
nifty_pq = ROOT / "data/1d/1d/NIFTY.parquet"
nifty_close = None
if nifty_pq.exists():
    nf = pd.read_parquet(str(nifty_pq))
    if nf.index.tz is None:
        nf.index = nf.index.tz_localize("UTC")
    nifty_close = nf["close"]

print("\n=== IC ANALYSIS ACROSS DIFFERENT RETURN TYPES ===\n")

PARQUET_DIR = ROOT / "data/1d/1d"
HORIZON = 7

# Compute multiple return types per symbol per date
return_data = []
symbols = df_oos["symbol"].unique()

for sym in symbols:
    sym_mask = df_oos["symbol"] == sym
    sym_df = df_oos[sym_mask].copy()
    
    pq = PARQUET_DIR / f"{sym}.parquet"
    if not pq.exists():
        continue
    
    raw = pd.read_parquet(str(pq))
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    raw = raw[raw.index >= OOS_START]
    
    if len(raw) < HORIZON + 2:
        continue
    
    close = raw["close"].astype(float)
    open_ = raw["open"].astype(float) if "open" in raw.columns else close
    
    # Type A: close-to-close (what label was computed on)
    ret_ctc = close.shift(-HORIZON) / close - 1.0
    # Type B: next-open to close (what we actually trade)
    entry_open = open_.shift(-1)   # open[T+1]
    exit_close = close.shift(-HORIZON)  # close[T+7]
    ret_otc = exit_close / entry_open - 1.0
    
    # NIFTY excess
    if nifty_close is not None:
        nc = nifty_close.reindex(close.index, method="ffill")
        nifty_fwd = nc.shift(-HORIZON) / nc - 1.0
    else:
        nifty_fwd = pd.Series(0.0, index=close.index)
    
    exc_ctc = ret_ctc - nifty_fwd
    exc_otc = ret_otc - nifty_fwd
    
    # Merge with model scores
    sym_scores = sym_df["model_score"].reindex(close.index)
    
    temp = pd.DataFrame({
        "symbol": sym,
        "ret_ctc": ret_ctc,
        "ret_otc": ret_otc,
        "exc_ctc": exc_ctc,
        "exc_otc": exc_otc,
        "model_score": sym_scores,
        "nifty_fwd": nifty_fwd,
    }, index=close.index)
    return_data.append(temp)

if not return_data:
    print("No return data computed")
    sys.exit(1)

all_ret = pd.concat(return_data).dropna()
print(f"Valid rows for IC measurement: {len(all_ret):,}")

# Compute ICs
for ret_col, desc in [
    ("ret_ctc", "Close-to-close return (what label uses)"),
    ("ret_otc", "Open[T+1]-to-close[T+7] (what we TRADE)"),
    ("exc_ctc", "Excess return vs NIFTY (close-to-close)"),
    ("exc_otc", "Excess return vs NIFTY (open-to-close)"),
]:
    valid = all_ret[all_ret[ret_col].notna() & all_ret["model_score"].notna()]
    if len(valid) < 100:
        continue
    ic, pval = spearmanr(valid["model_score"], valid[ret_col])
    top20 = valid.groupby(valid.index).apply(
        lambda g: g.nlargest(max(1, len(g)//5), "model_score")[ret_col].mean()
    ) if False else None  # skip, too slow
    print(f"{desc}:")
    print(f"  IC = {ic:.4f}  (p={pval:.4f})  {'✓ POSITIVE SIGNAL' if ic > 0 else '✗ NO SIGNAL'}")

# Overnight gap analysis
print("\n=== OVERNIGHT GAP ANALYSIS ===")
if "ret_ctc" in all_ret.columns and "ret_otc" in all_ret.columns:
    gap = all_ret["ret_ctc"] - all_ret["ret_otc"]  # close-to-close minus next-open-to-close = gap effect
    print(f"Gap (close_T → open_T+1) contribution to return:")
    print(f"  Mean gap: {gap.mean()*100:.4f}%")
    
    # Is there a correlation between model score and gap?
    valid_gap = all_ret.dropna()
    if len(valid_gap) > 100:
        gap_ic, gap_pval = spearmanr(valid_gap["model_score"], valid_gap["ret_ctc"] - valid_gap["ret_otc"])
        print(f"  IC(model_score, overnight_gap): {gap_ic:.4f}  (p={gap_pval:.4f})")
        if gap_ic > 0.01:
            print(f"  ⚠ HIGH-SCORED STOCKS TEND TO HAVE POSITIVE GAP (already priced in at open)")
        elif gap_ic < -0.01:
            print(f"  ⚠ HIGH-SCORED STOCKS TEND TO HAVE NEGATIVE GAP (adverse selection at open)")
        else:
            print(f"  ✓ Gap is not correlated with model scores (no adverse selection)")

# Top-vs-bottom analysis
print("\n=== TOP 10% vs BOTTOM 10% ACTUAL RETURN ANALYSIS ===")
# For each date, compare top 10% and bottom 10% actual returns
ls_spread_ctc = []
ls_spread_otc = []
nifty_rets = []
dates = sorted(all_ret.index.unique())

for dt in dates:
    day = all_ret[all_ret.index == dt].dropna()
    if len(day) < 10:
        continue
    p90 = day["model_score"].quantile(0.90)
    p10 = day["model_score"].quantile(0.10)
    top = day[day["model_score"] >= p90]
    bot = day[day["model_score"] <= p10]
    if len(top) > 0 and len(bot) > 0:
        ls_spread_ctc.append(float(top["exc_ctc"].mean()) - float(bot["exc_ctc"].mean()))
        ls_spread_otc.append(float(top["exc_otc"].mean()) - float(bot["exc_otc"].mean()))
        nifty_rets.append(float(day["nifty_fwd"].mean()))

if ls_spread_ctc:
    s_ctc = np.array(ls_spread_ctc)
    s_otc = np.array(ls_spread_otc)
    print(f"Close-to-close L/S spread per 7-day period:")
    print(f"  Mean spread: {s_ctc.mean()*100:.3f}%  Win rate: {(s_ctc>0).mean():.1%}")
    print(f"Open-to-close L/S spread per 7-day period (ACTUAL TRADES):")
    print(f"  Mean spread: {s_otc.mean()*100:.3f}%  Win rate: {(s_otc>0).mean():.1%}")
    print(f"  Difference (gap impact): {(s_ctc.mean() - s_otc.mean())*100:.3f}%/period")
    print()
    if s_otc.mean() > 0:
        print(f"  ✓ POSITIVE L/S SPREAD ON ACTUAL TRADES: {s_otc.mean()*100:.3f}%/period")
        ann = s_otc.mean() * 52 * 100
        print(f"  Annualized (52 periods): {ann:.1f}%")
        cost = 7.26 / 100  # futures cost fraction
        net_ann = ann - cost * 52
        print(f"  Net after futures costs: {net_ann:.1f}%")
    else:
        print(f"  ✗ NEGATIVE L/S SPREAD ON ACTUAL TRADES")
        print(f"  This explains why the portfolio is unprofitable despite positive IC")
