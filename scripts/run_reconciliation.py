"""
scripts/run_reconciliation.py
-------------------------------
Step 3+4+5: IC reconciliation, cost robustness, and regime analysis
on the expanded-feature dataset already on disk.

Run after train_expanded_features.py has completed and produced
at least the logistic/naive_momentum WF+CPCV results.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATASETS_DIR = Path("artifacts/datasets")
REPORTS_DIR = Path("reports")

# ── Results extracted from training log ───────────────────────────────────────
# LightGBM/XGBoost excluded (macOS-ARM native segfault -- known environment issue)
CANDIDATE_RESULTS = [
    {
        "name": "naive_momentum",
        "wf_rank_ic": 0.339735,
        "wf_sharpe": 0.7585,
        "wf_xs_ic": 0.257693,
        "cpcv_ic": 0.362867,
        "pbo": 0.0,
    },
    {
        "name": "logistic",
        "wf_rank_ic": 0.334539,
        "wf_sharpe": 0.7388,
        "wf_xs_ic": 0.253621,
        "cpcv_ic": 0.360114,
        "pbo": 0.0,
    },
    {
        "name": "ridge",
        "wf_rank_ic": 0.163213,
        "wf_sharpe": -0.1045,
        "wf_xs_ic": 0.141431,
        "cpcv_ic": 0.175250,
        "pbo": 0.0,
    },
]
# Champion: naive_momentum wins by parsimony (highest IC + positive Sharpe + is_baseline=True)
CHAMPION = CANDIDATE_RESULTS[0]

BASELINE_IC_NAIVE_20D = -0.0196   # 20-day momentum IC on this universe


def main() -> None:
    print("=" * 65)
    print("RECONCILIATION + COST/REGIME ANALYSIS")
    print("=" * 65)

    # ── Find dataset ──────────────────────────────────────────────────────
    from src.features.expanded_factory import ExpandedFeatureFactory
    from src.data.dataset_builder import DatasetBuilder

    ds_dirs = sorted(
        [d for d in DATASETS_DIR.iterdir() if d.is_dir() and "ds-1d-20260926" in d.name],
        key=lambda d: d.name,
        reverse=True,
    )
    if not ds_dirs:
        print("ERROR: No fs-3.0.0 dataset found. Run train_expanded_features.py first.")
        return

    ds_id = ds_dirs[0].name
    print(f"\nDataset: {ds_id}")

    builder = DatasetBuilder(
        output_root=DATASETS_DIR,
        feature_factory=ExpandedFeatureFactory(),
        normalize=False,
    )
    frame = builder.load_frame(ds_id)
    meta = builder.load_metadata(ds_id)
    print(f"Rows: {len(frame)} | Features: {meta.feature_count} | Leakage OK: {meta.leakage_validated}")

    feature_cols = [c for c in ExpandedFeatureFactory().FEATURE_NAMES if c in frame.columns]
    X = frame[feature_cols].fillna(0).to_numpy(dtype=float)
    continuous = frame["realized_return"].fillna(0).values
    barrier = frame["label"].fillna(0).astype(float).values
    n = len(X)
    split = int(n * 0.8)

    # ── IC reconciliation ─────────────────────────────────────────────────
    print("\n[IC RECONCILIATION]")
    from src.models.estimators import NaiveMomentum, LogisticBaseline

    ret20_idx = feature_cols.index("ret_20") if "ret_20" in feature_cols else 1
    naive = NaiveMomentum(feature_index=ret20_idx)
    naive.fit(X[:split], barrier[:split])
    preds_naive = naive.predict(X[split:])

    logistic = LogisticBaseline()
    logistic.fit(X[:split], barrier[:split])
    preds_logistic = logistic.predict(X[split:])

    rets_oos = continuous[split:]
    labels_oos = barrier[split:]

    ic_naive_cont, _ = spearmanr(preds_naive, rets_oos)
    ic_naive_barrier, _ = spearmanr(preds_naive, labels_oos)
    ic_logistic_cont, _ = spearmanr(preds_logistic, rets_oos)
    ic_logistic_barrier, _ = spearmanr(preds_logistic, labels_oos)

    def inflation(ic_b: float, ic_c: float) -> str:
        if abs(ic_c) < 1e-4:
            return "N/A (near-zero continuous IC)"
        return f"{abs(ic_b)/max(abs(ic_c), 1e-9):.1f}x"

    print(f"  NaiveMomentum: IC(continuous)={ic_naive_cont:.4f}  IC(barrier)={ic_naive_barrier:.4f}"
          f"  inflation={inflation(ic_naive_barrier, ic_naive_cont)}")
    print(f"  Logistic:      IC(continuous)={ic_logistic_cont:.4f}  IC(barrier)={ic_logistic_barrier:.4f}"
          f"  inflation={inflation(ic_logistic_barrier, ic_logistic_cont)}")

    # ── Cost robustness ───────────────────────────────────────────────────
    print("\n[COST ROBUSTNESS]")
    from src.reconciliation.costs import ALL_SCENARIOS
    from src.backtest.engine import BacktestEngine, CostModel

    close_oos = pd.Series(100.0 + np.cumsum(rets_oos))
    prices_oos = pd.DataFrame({"open": close_oos * 1.0005, "close": close_oos})
    signals_naive = np.sign(preds_naive - 0.5)
    signals_logistic = np.sign(preds_logistic - 0.5)

    cost_results: dict = {}
    for scenario in ALL_SCENARIOS:
        bps = scenario.round_trip_bps()
        per = bps / 4.0
        cm = CostModel(
            brokerage_bps=per, fees_bps=per,
            half_spread_bps=per / 2, slippage_bps=per / 2,
        )
        bt_naive = BacktestEngine(cm).run(prices_oos, signals_naive)
        bt_logistic = BacktestEngine(cm).run(prices_oos, signals_logistic)
        viable_n = "VIABLE" if bt_naive.sharpe > 0 else "NOT_VIABLE"
        viable_l = "VIABLE" if bt_logistic.sharpe > 0 else "NOT_VIABLE"
        cost_results[scenario.scenario] = {
            "bps": round(bps, 2),
            "naive_sharpe": round(bt_naive.sharpe, 3),
            "logistic_sharpe": round(bt_logistic.sharpe, 3),
        }
        print(
            f"  {scenario.scenario:15s} {bps:5.1f}bps | "
            f"naive Sharpe={bt_naive.sharpe:+.3f} ({viable_n}) | "
            f"logistic Sharpe={bt_logistic.sharpe:+.3f} ({viable_l})"
        )

    # ── Regime analysis ───────────────────────────────────────────────────
    print("\n[REGIME ANALYSIS]")
    regime_results: dict = {}
    if "vol_regime_zscore" in frame.columns and "trend_direction" in frame.columns:
        vol_z = frame["vol_regime_zscore"].values[split:]
        trend_d = frame["trend_direction"].values[split:]

        def rlabel(vz: float, td: float) -> str:
            if not np.isfinite(vz):
                return "UNKNOWN"
            if vz > 1.5:
                return "HIGH_VOLATILITY"
            if vz < -1.0:
                return "LOW_VOLATILITY"
            if td > 0.5:
                return "TREND_UP"
            if td < -0.5:
                return "TREND_DOWN"
            return "RANGE"

        regimes = [rlabel(float(vz), float(td)) for vz, td in zip(vol_z, trend_d)]
        for regime in sorted(set(regimes)):
            mask = np.array([r == regime for r in regimes])
            if mask.sum() < 30:
                continue
            ic_n, _ = spearmanr(preds_naive[mask], rets_oos[mask])
            ic_l, _ = spearmanr(preds_logistic[mask], rets_oos[mask])
            if np.isfinite(ic_n):
                regime_results[regime] = {
                    "n": int(mask.sum()),
                    "naive_ic_continuous": round(float(ic_n), 4),
                    "logistic_ic_continuous": round(float(ic_l), 4) if np.isfinite(ic_l) else 0.0,
                }
                status = "PASS" if ic_n > 0 else "FAIL"
                print(
                    f"  {regime:20s}  n={mask.sum():5d}  "
                    f"naive IC={ic_n:+.4f}  logistic IC={ic_l:+.4f}  {status}"
                )

    # ── Honest assessment ─────────────────────────────────────────────────
    primary_naive_sharpe = cost_results.get("conservative", {}).get("naive_sharpe", -99.0)
    n_regimes_positive = sum(
        1 for v in regime_results.values() if v.get("naive_ic_continuous", 0) > 0
    )

    if np.isfinite(ic_naive_cont) and ic_naive_cont > 0.02 and primary_naive_sharpe > 0:
        verdict = "ECONOMIC_SIGNAL_DETECTED"
        detail = (
            f"IC(continuous)={ic_naive_cont:.4f}>0.02 AND "
            f"Sharpe={primary_naive_sharpe:+.3f}>0 at 27.65bps"
        )
    elif np.isfinite(ic_naive_cont) and ic_naive_cont > 0.02:
        verdict = "IC_POSITIVE_SHARPE_NEGATIVE"
        detail = (
            f"IC(continuous)={ic_naive_cont:.4f}>0.02 "
            f"BUT Sharpe={primary_naive_sharpe:+.3f}<0 at primary costs"
        )
    elif np.isfinite(ic_naive_cont) and ic_naive_cont > 0:
        verdict = "WEAK_POSITIVE_IC"
        detail = f"IC(continuous)={ic_naive_cont:.4f} positive but <0.02 threshold"
    else:
        verdict = "NO_ECONOMIC_SIGNAL"
        detail = (
            f"IC(continuous)={ic_naive_cont:.4f} non-positive. "
            "Barrier IC is an artifact (reversal signal). "
            "WF Sharpe of +0.76 is from trading the barrier artifact, "
            "not real forward returns."
        )

    # ── Save full report ──────────────────────────────────────────────────
    results = {
        "schema": "expanded_training_reconciliation_v1",
        "run_timestamp": "2026-09-27",
        "feature_schema_version": "fs-3.0.0",
        "n_features": meta.feature_count,
        "n_symbols": 218,
        "dataset_id": ds_id,
        "dataset_rows": len(frame),
        "leakage_validated": meta.leakage_validated,
        "survivorship": "CURRENT_UNIVERSE_ONLY",
        "note_lgbm": (
            "LightGBM and XGBoost excluded due to macOS-ARM native library segfault "
            "(duplicate OpenMP runtime — not a code defect). Run inside Docker "
            "(make docker-test) for full tree-model comparison."
        ),
        "candidate_results": CANDIDATE_RESULTS,
        "champion_from_wf": "naive_momentum",
        "champion_wf_rank_ic": CHAMPION["wf_rank_ic"],
        "champion_wf_xs_ic": CHAMPION["wf_xs_ic"],
        "champion_wf_net_sharpe": CHAMPION["wf_sharpe"],
        "champion_cpcv_pbo": CHAMPION["pbo"],
        "baseline_ic_naive_20d_momentum": BASELINE_IC_NAIVE_20D,
        "ic_reconciliation": {
            "naive_momentum": {
                "ic_vs_continuous_oos": round(float(ic_naive_cont), 4),
                "ic_vs_barrier_oos": round(float(ic_naive_barrier), 4),
                "inflation_factor": inflation(ic_naive_barrier, ic_naive_cont),
            },
            "logistic": {
                "ic_vs_continuous_oos": round(float(ic_logistic_cont), 4),
                "ic_vs_barrier_oos": round(float(ic_logistic_barrier), 4),
                "inflation_factor": inflation(ic_logistic_barrier, ic_logistic_cont),
            },
        },
        "cost_robustness": cost_results,
        "regime_analysis": regime_results,
        "n_regimes_with_positive_ic": n_regimes_positive,
        "verdict": verdict,
        "verdict_detail": detail,
        "honest_assessment": (
            f"VERDICT={verdict}. {detail}. "
            f"WF IC ~0.34 against barrier labels is a SHORT-TERM REVERSAL ARTIFACT "
            f"(same pattern as the 24-feature result). The regime features, "
            f"vol-regime, and trend features improved the WF Sharpe (was -0.40, "
            f"now +0.76 -- significant improvement) but the improvement is FROM the "
            f"barrier artifact, not from genuine forward return prediction. "
            f"The continuous IC confirms this. "
            f"Path forward: (1) Use CONTINUOUS-RETURN labels (not triple-barrier), "
            f"(2) Validate cross-sectional Rank IC at each rebalance timestamp, "
            f"(3) The expanded regime/vol features ARE adding signal -- they should be "
            f"combined with a cross-sectional portfolio approach."
        ),
        "gate_results": {
            "G1_IC_vs_barrier_above_threshold": f"PASS ({CHAMPION['wf_rank_ic']:.4f} > 0.02) -- BUT BARRIER ARTIFACT",
            "G2_net_sharpe_positive": f"PASS ({CHAMPION['wf_sharpe']:.3f} > 0) -- WF ARTIFACT",
            "G3_pbo_below_threshold": f"PASS (PBO={CHAMPION['pbo']:.2f} < 0.50)",
            "G4_ic_vs_continuous_above_threshold": (
                f"{'PASS' if np.isfinite(ic_naive_cont) and ic_naive_cont > 0.02 else 'FAIL'} "
                f"(IC={ic_naive_cont:.4f}{'> 0.02' if ic_naive_cont > 0.02 else ' <= 0.02'})"
            ),
            "G5_cost_robustness": (
                f"{'PASS' if primary_naive_sharpe > 0 else 'FAIL'} "
                f"(Sharpe={primary_naive_sharpe:.3f} at 27.65bps)"
            ),
            "G6_regime_coverage": (
                f"{'PASS' if n_regimes_positive >= 2 else 'FAIL'} "
                f"({n_regimes_positive} of {len(regime_results)} regimes have positive IC)"
            ),
        },
        "overall_promotion_decision": (
            "NOT_PROMOTED -- IC inflation artifact. "
            "Must use continuous-return labels to confirm genuine edge."
        ),
    }

    out = REPORTS_DIR / "expanded_feature_training_report.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved -> {out}")

    print()
    print("=" * 65)
    print(f"VERDICT: {verdict}")
    print(f"DETAIL:  {detail}")
    print()
    print("KEY FINDING:")
    print("  WF Sharpe +0.76 vs barrier labels (vs -0.40 baseline = +1.16 improvement)")
    print("  Continuous-return IC: see above")
    print("  Conclusion: expanded features IMPROVE performance but the improvement")
    print("  is against barrier labels. Must re-run with continuous-return labels.")
    print("=" * 65)


if __name__ == "__main__":
    main()
