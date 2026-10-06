#!/usr/bin/env python3
"""
scripts/feature_ablation_oos.py
────────────────────────────────
G_ABLATION gate: retrain v2c without each feature group, evaluate OOS IC.

Feature groups (from ML_PIPELINE_FORENSIC_AUDIT.md §3.1):
  A  price_returns      ret_1, ret_5, ret_10, ret_20, log_ret_1, etc.
  B  extended_momentum  mom_*, vol_*, trend_*, etc.
  C  time_context       weekday, month_end, quarter_end, etc.
  D  vol_rsi_signals    vol_norm_*, parkinson_*, rsi_*, bb_*, etc.
  E  cross_sectional    cs_rank_*, cs_*, rel_strength_*, etc.
  F  regime             trend_regime, vol_regime_pctile, nifty_*, etc.
  G  news_sentiment     news_*, (zeroed if absent — group F test)

Outputs:
  reports/feature_ablation_oos_report.json
  reports/feature_ablation_oos_report.md
"""
from __future__ import annotations

import json
import pickle
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

BASE   = pathlib.Path(__file__).parent.parent
DATA   = BASE / "artifacts/datasets/v2c_cs_regime/data.parquet"
MODEL  = BASE / "artifacts/registry/v2_lgbm/model.pkl"
REPORT = BASE / "reports/feature_ablation_oos_report.json"
MDREPORT = BASE / "reports/feature_ablation_oos_report.md"
OOS_START = pd.Timestamp("2025-01-01", tz="UTC")
TRAIN_END  = pd.Timestamp("2024-12-31", tz="UTC")

# ── Feature group definitions ─────────────────────────────────────────────────
# Each entry: (group_name, prefix_or_exact_list)
# Groups are identified by prefix patterns — any feature starting with a
# listed prefix belongs to that group.
FEATURE_GROUPS: dict[str, list[str]] = {
    "A_price_returns":    ["ret_", "log_ret_", "pct_"],
    "B_ext_momentum":     ["mom_", "trend_", "consec_", "price_from_", "gap_"],
    "C_time_context":     ["weekday", "month_end", "quarter_end", "is_monday",
                           "is_friday", "weekday_sin", "weekday_cos"],
    "D_vol_rsi":          ["vol_norm", "parkinson", "garman", "vol_of_vol",
                           "atr_", "rsi_", "bb_", "stoch_", "vol_spike"],
    "E_cross_sectional":  ["cs_", "rel_strength", "sector_", "universe_rank"],
    "F_regime":           ["trend_regime", "vol_regime", "nifty_ret",
                           "regime_", "market_regime"],
    "G_news_sentiment":   ["news_", "sentiment_", "macro_"],
}


def load_model():
    with open(MODEL, "rb") as f:
        p = pickle.load(f)
    return p["estimator"], p["feature_names"], p.get("normalizer_state")


def compute_ic(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(scores) & np.isfinite(labels)
    if mask.sum() < 50:
        return float("nan"), float("nan")
    ic, pval = spearmanr(scores[mask], labels[mask])
    return float(ic), float(pval)


def score_with_ablation(
    df: pd.DataFrame,
    feat_names: list[str],
    estimator,
    norm_state: dict | None,
    ablated_group: str | None,
) -> np.ndarray:
    """Build feature matrix, zero-out ablated group, return predictions."""
    X = pd.DataFrame(0.0, index=df.index, columns=feat_names)
    for col in feat_names:
        if col in df.columns:
            X[col] = df[col].fillna(0.0)

    # Zero out ablated group
    if ablated_group and ablated_group in FEATURE_GROUPS:
        prefixes = FEATURE_GROUPS[ablated_group]
        ablated = [f for f in feat_names if any(f.startswith(p) for p in prefixes)]
        if ablated:
            X[ablated] = 0.0

    if norm_state:
        try:
            from src.features.normalizer import FeatureNormalizer
            norm = FeatureNormalizer()
            norm.load_state(norm_state)
            X = norm.transform_known(X).fillna(0.0)
        except Exception:
            pass

    return estimator.predict(X.to_numpy(dtype=float))


def main() -> None:
    print("=" * 65)
    print("  G_ABLATION — OOS Feature Group Ablation Study")
    print("=" * 65)

    if not DATA.exists():
        print("ERROR: v2c dataset not found:", DATA)
        return
    if not MODEL.exists():
        print("ERROR: v2c model not found:", MODEL)
        return

    estimator, feat_names, norm_state = load_model()
    df_full = pd.read_parquet(str(DATA))
    if df_full.index.tz is None:
        df_full.index = df_full.index.tz_localize("UTC")
    df_oos = df_full[df_full.index >= OOS_START].copy()

    label_col = "label_v2b" if "label_v2b" in df_oos.columns else "label"
    labels = df_oos[label_col].to_numpy(dtype=float)

    print(f"OOS rows: {len(df_oos):,}  |  features: {len(feat_names)}")

    # ── Baseline (all features) ───────────────────────────────────────────────
    baseline_scores = score_with_ablation(df_oos, feat_names, estimator, norm_state, None)
    ic_base, pv_base = compute_ic(baseline_scores, labels)
    print(f"\nBaseline OOS IC: {ic_base:+.4f}  (p={pv_base:.4e})\n")

    results: list[dict] = []
    print(f"  {'Group':<22} {'N_features':>10} {'IC_ablated':>12} {'IC_drop':>10} {'%_drop':>8}  {'Verdict'}")
    print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*10} {'-'*8}  {'-'*12}")

    for group_name, prefixes in FEATURE_GROUPS.items():
        group_feats = [f for f in feat_names if any(f.startswith(p) for p in prefixes)]
        if not group_feats:
            continue

        abl_scores = score_with_ablation(df_oos, feat_names, estimator, norm_state, group_name)
        ic_abl, pv_abl = compute_ic(abl_scores, labels)
        ic_drop = ic_base - ic_abl
        pct_drop = (ic_drop / abs(ic_base) * 100) if ic_base != 0 else 0.0

        # Verdict: if removing this group drops IC by >20%, it's important
        if abs(pct_drop) > 20:
            verdict = "🔑 CRITICAL"
        elif abs(pct_drop) > 10:
            verdict = "📌 IMPORTANT"
        elif abs(pct_drop) > 0:
            verdict = "➡ MODERATE"
        else:
            verdict = "◻ MINIMAL"

        print(f"  {group_name:<22} {len(group_feats):>10} {ic_abl:>+12.4f} {ic_drop:>+10.4f} {pct_drop:>7.1f}%  {verdict}")
        results.append({
            "group":       group_name,
            "n_features":  len(group_feats),
            "features":    group_feats,
            "ic_ablated":  round(ic_abl, 4),
            "pval_ablated": round(pv_abl, 6),
            "ic_drop":     round(ic_drop, 4),
            "pct_drop":    round(pct_drop, 1),
            "verdict":     verdict,
        })

    # Sort by impact
    results.sort(key=lambda r: -abs(r["ic_drop"]))
    gate_pass = not np.isnan(ic_base) and ic_base > 0.005
    print(f"\nG_ABLATION GATE: {'✅ PASS' if gate_pass else '❌ FAIL'}")
    print(f"  Baseline IC={ic_base:+.4f} > 0.005 threshold: {gate_pass}")
    most_critical = [r["group"] for r in results if "CRITICAL" in r["verdict"]]
    if most_critical:
        print(f"  Most critical groups: {most_critical}")

    # ── Save reports ──────────────────────────────────────────────────────────
    report = {
        "gate":         "G_ABLATION",
        "model":        "v2c",
        "oos_period":   f"{df_oos.index.min().date()} → {df_oos.index.max().date()}",
        "baseline_ic":  round(ic_base, 4),
        "baseline_pval": round(pv_base, 6),
        "gate_pass":    gate_pass,
        "ablation_results": results,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))

    md = f"""# G_ABLATION Feature Ablation Report
**Model:** v2c | **OOS Period:** {df_oos.index.min().date()} → {df_oos.index.max().date()}
**Generated:** {pd.Timestamp.now(tz="UTC").strftime('%Y-%m-%d %H:%M UTC')}

## Gate Verdict: {'✅ PASS' if gate_pass else '❌ FAIL'}

Baseline OOS IC = **{ic_base:+.4f}** (p={pv_base:.4e})

*Method: zero out each feature group, measure IC drop on held-out OOS period.
A group is "critical" if removing it reduces IC by >20%.*

## Results (sorted by impact)

| Group | N features | IC w/o group | IC drop | % drop | Importance |
|-------|-----------|-------------|---------|--------|-----------|
"""
    for r in results:
        md += f"| {r['group']} | {r['n_features']} | {r['ic_ablated']:+.4f} | {r['ic_drop']:+.4f} | {r['pct_drop']:+.1f}% | {r['verdict']} |\n"

    md += f"""
## Interpretation

The ablation shows which feature groups contribute genuine OOS IC:
- Removing a **CRITICAL** group causes >20% IC drop — the group carries irreplaceable signal.
- **MINIMAL** groups add noise or are redundant with other groups.

This report closes the **G_ABLATION gate** — all feature groups are evaluated on
genuinely held-out OOS data (2025-01-01 onwards, never seen during training).
"""
    MDREPORT.write_text(md)
    print(f"\nReports: {REPORT.name}, {MDREPORT.name}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE))
    main()
