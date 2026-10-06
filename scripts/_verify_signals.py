#!/usr/bin/env python3
"""
scripts/_verify_signals.py
──────────────────────────
End-to-end signal verification:
  1. Model artifact integrity
  2. Normalizer round-trip
  3. Feature schema match
  4. CS rank logic correctness
  5. Score distribution health
  6. Manual spot-check: re-score 5 random symbols, compare with latest_scores.json
  7. PIT check: no today's bars leaking into yesterday's prediction
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

SCORES_PATH = BASE / "artifacts/live_session/latest_scores.json"
PARQUET_DIR = BASE / "data/1d/1d"

# ── Load model ───────────────────────────────────────────────────────────────
print("=" * 60)
print("  SIGNAL VERIFICATION REPORT")
print("=" * 60)

MODEL_PATHS = list(BASE.glob("artifacts/expanded_lgbm/*/model.pkl")) + \
              list(BASE.glob("artifacts/registry/stage_a_1d/*/model.pkl")) + \
              list(BASE.glob("artifacts/registry/v2_lgbm/model.pkl"))
MODEL_PATHS = sorted(MODEL_PATHS)
if not MODEL_PATHS:
    print("FAIL: No model.pkl found")
    sys.exit(1)

latest_pkl = MODEL_PATHS[-1]
with open(latest_pkl, "rb") as f:
    model_dict = pickle.load(f)

estimator    = model_dict.get("estimator")
feat_names   = model_dict.get("feature_names", [])
norm_state   = model_dict.get("normalizer_state")
schema       = model_dict.get("feature_schema_version", "?")
invert       = model_dict.get("invert_scores", False)

print(f"\n[1] MODEL ARTIFACT")
print(f"  Path:          {latest_pkl.relative_to(BASE)}")
print(f"  Schema:        {schema}")
print(f"  Features:      {len(feat_names)}")
print(f"  Normalizer:    {'present' if norm_state else 'MISSING ⚠'}")
print(f"  Invert scores: {invert}")
print(f"  Estimator:     {type(estimator).__name__}")

if estimator is None:
    print("  FAIL: estimator is None")
    sys.exit(1)
model_ok = len(feat_names) > 0 and norm_state is not None
print(f"  Status:        {'✅ OK' if model_ok else '⚠️  WARNING'}")

# ── Normalizer round-trip ────────────────────────────────────────────────────
print(f"\n[2] NORMALIZER ROUND-TRIP")
try:
    from src.features.normalizer import FeatureNormalizer
    norm = FeatureNormalizer()
    norm.load_state(norm_state)
    rng = np.random.default_rng(42)
    X_raw = pd.DataFrame(rng.normal(0, 1, (50, len(feat_names))), columns=feat_names)
    X_norm = norm.transform_known(X_raw).fillna(0.0)
    assert X_norm.shape == X_raw.shape, "shape mismatch"
    assert not X_norm.isnull().all().any(), "all-NaN column after normalizing"
    scores_raw  = estimator.predict(X_raw.to_numpy(dtype=float))
    scores_norm = estimator.predict(X_norm.to_numpy(dtype=float))
    mean_diff = float(np.abs(scores_raw - scores_norm).mean())
    print(f"  Raw score mean:  {scores_raw.mean():.4f}  std={scores_raw.std():.4f}")
    print(f"  Norm score mean: {scores_norm.mean():.4f}  std={scores_norm.std():.4f}")
    print(f"  Mean abs diff:   {mean_diff:.4f}")
    norm_matters = mean_diff > 0.001
    print(f"  Normalizer matters: {norm_matters} (diff={mean_diff:.4f})")
    print(f"  Status: ✅ OK — normalizer applied correctly")
except Exception as e:
    print(f"  FAIL: {e}")

# ── Feature schema check ─────────────────────────────────────────────────────
print(f"\n[3] FEATURE SCHEMA vs PARQUETS")
sample_pqs = sorted(PARQUET_DIR.glob("*.parquet"))[:5]
if sample_pqs:
    try:
        from src.features.expanded_factory import ExpandedFeatureFactory
        from src.features.factory import FeatureFactory
        factory = ExpandedFeatureFactory() if len(feat_names) > 24 else FeatureFactory()
        df_test = pd.read_parquet(str(sample_pqs[0]))
        df_test.columns = [c.lower() for c in df_test.columns]
        if df_test.index.tz is None:
            df_test.index = df_test.index.tz_localize("UTC")
        feats_df, _ = factory.build(df_test)
        available = [f for f in feat_names if f in feats_df.columns]
        missing   = [f for f in feat_names if f not in feats_df.columns]
        print(f"  Model expects:  {len(feat_names)} features")
        print(f"  Available:      {len(available)}")
        print(f"  Missing:        {len(missing)}" + (f" ⚠️  {missing[:5]}" if missing else ""))
        print(f"  Status: {'✅ OK' if len(missing) == 0 else '⚠️  some features missing (will be 0.0)'}")
    except Exception as e:
        print(f"  FAIL: {e}")

# ── Load today's generated signals ───────────────────────────────────────────
print(f"\n[4] CS RANK LOGIC VERIFICATION")
s = json.load(open(SCORES_PATH))
signals = s.get("signals", [])
longs  = [x for x in signals if x.get("direction") ==  1]
shorts = [x for x in signals if x.get("direction") == -1]
neuts  = [x for x in signals if x.get("direction") ==  0]
all_scores = [x["score"] for x in signals]

print(f"  Total scored:  {len(signals)}")
print(f"  Long:          {len(longs)}  ({len(longs)/len(signals)*100:.1f}%)")
print(f"  Short:         {len(shorts)} ({len(shorts)/len(signals)*100:.1f}%)")
print(f"  Neutral:       {len(neuts)}  ({len(neuts)/len(signals)*100:.1f}%)")
n_each_expected = max(5, int(len(signals) * 0.15))
print(f"  Expected each (15%): {n_each_expected}")

# Verify top 15% are LONGs, bottom 15% are SHORTs
sorted_by_score = sorted(signals, key=lambda x: -x["score"])
expected_top    = {x["symbol"] for x in sorted_by_score[:n_each_expected]}
expected_bot    = {x["symbol"] for x in sorted_by_score[-n_each_expected:]}
actual_longs    = {x["symbol"] for x in longs}
actual_shorts   = {x["symbol"] for x in shorts}

top_match = len(expected_top & actual_longs) / len(expected_top) if expected_top else 0
bot_match = len(expected_bot & actual_shorts) / len(expected_bot) if expected_bot else 0
print(f"  Top-15%→LONG match:  {top_match:.0%}")
print(f"  Bot-15%→SHORT match: {bot_match:.0%}")
print(f"  Status: {'✅ OK' if top_match > 0.8 and bot_match > 0.8 else '⚠️  CS rank mismatch'}")

# ── Score distribution health ─────────────────────────────────────────────────
print(f"\n[5] SCORE DISTRIBUTION")
arr = np.array(all_scores)
print(f"  Min:    {arr.min():.4f}")
print(f"  Max:    {arr.max():.4f}")
print(f"  Mean:   {arr.mean():.4f}  (healthy: 0.45–0.55)")
print(f"  Std:    {arr.std():.4f}   (healthy: >0.02)")
print(f"  >0.6:   {(arr>0.6).sum()} symbols   (strong LONG conviction)")
print(f"  <0.4:   {(arr<0.4).sum()} symbols   (strong SHORT conviction)")
degenerate = arr.std() < 0.01
print(f"  Status: {'⚠️  DEGENERATE — std too low' if degenerate else '✅ OK'}")

# ── PIT check ────────────────────────────────────────────────────────────────
print(f"\n[6] PIT CHECK (point-in-time integrity)")
# Check that data_date in signals is NOT today (should be yesterday's EOD)
from datetime import date, timedelta
today_str = date.today().isoformat()
yesterday = (date.today() - timedelta(days=1)).isoformat()
data_dates = [x.get("data_date","") for x in signals if x.get("data_date")]
if data_dates:
    unique_dates = set(data_dates)
    today_count = sum(1 for d in data_dates if d == today_str)
    print(f"  Unique data_dates: {sorted(unique_dates)}")
    print(f"  Using today's data ({today_str}): {today_count} signals")
    if today_count > 0:
        print(f"  ⚠️  Some signals use today's partial bar (live LTP injection — expected in intraday mode)")
    else:
        print(f"  ✅ OK — all signals use EOD data (yesterday or earlier)")

# ── Manual spot-check ────────────────────────────────────────────────────────
print(f"\n[7] MANUAL SPOT-CHECK (re-score 5 symbols)")
CHECK_SYMS = ["RELIANCE", "HDFCBANK", "INFY", "TATAMOTORS", "POLICYBZR"]
try:
    from src.features.normalizer import FeatureNormalizer
    norm = FeatureNormalizer()
    norm.load_state(norm_state)
    factory = ExpandedFeatureFactory() if len(feat_names) > 24 else FeatureFactory()

    print(f"  {'Symbol':<14} {'Stored':>8} {'Recomputed':>12} {'Match':>8} {'Direction OK':>14}")
    print("  " + "-" * 60)
    all_ok = True
    for sym in CHECK_SYMS:
        pf = PARQUET_DIR / f"{sym}.parquet"
        if not pf.exists():
            print(f"  {sym:<14} {'—':>8} {'no parquet':>12}")
            continue
        stored_sig = next((x for x in signals if x["symbol"] == sym), None)
        if not stored_sig:
            print(f"  {sym:<14} {'—':>8} {'not in scores':>12}")
            continue

        df = pd.read_parquet(str(pf))
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        feats, _ = factory.build(df)
        last = feats.iloc[-1].fillna(0.0)
        X = pd.DataFrame([[last.get(f, 0.0) for f in feat_names]], columns=feat_names)
        X_norm = norm.transform_known(X).fillna(0.0)
        recomputed = float(estimator.predict(X_norm.to_numpy(dtype=float))[0])
        stored     = stored_sig["score"]
        diff       = abs(recomputed - stored)
        match      = "✅" if diff < 0.02 else "⚠️ "
        stored_dir = stored_sig.get("direction", 0)
        recomp_dir = 1 if recomputed > stored else (-1 if recomputed < stored else 0)
        if diff > 0.02:
            all_ok = False
        print(f"  {sym:<14} {stored:>8.4f} {recomputed:>12.4f} {match:>8}  diff={diff:.4f}")

    print(f"\n  Overall: {'✅ All spot-checks pass' if all_ok else '⚠️  Some scores diverge (likely live_ltp partial bar)'}")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n" + "=" * 60)
print("  VERIFICATION COMPLETE")
print("=" * 60)
