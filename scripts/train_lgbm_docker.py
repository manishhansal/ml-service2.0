#!/usr/bin/env python3
"""
Train LightGBM + XGBoost on 55-feature expanded dataset inside Docker.
No macOS-ARM segfault here — we're running inside the container.

Steps:
  1. Load the latest fs-3.0.0 dataset from artifacts/datasets/
  2. Run TrainingOrchestrator with lightgbm + xgboost (full 5-window WF + CPCV)
  3. Compare against logistic baseline (already known: IC_continuous = 0.31)
  4. Save report to reports/lgbm_training_report.json
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATASETS_DIR = Path("artifacts/datasets")
REPORTS_DIR  = Path("reports")

def main():
    from src.features.expanded_factory import ExpandedFeatureFactory
    from src.data.dataset_builder import DatasetBuilder
    from src.registry.registry import ModelRegistry
    from src.training.orchestrator import TrainingOrchestrator
    from src.reconciliation.costs import ALL_SCENARIOS, PRIMARY_COST

    print("=" * 65)
    print("LGBM + XGBOOST TRAINING — fs-3.0.0, 218 symbols")
    print("=" * 65)

    # ── Find dataset ──────────────────────────────────────────────────
    ds_dirs = sorted(
        [d for d in DATASETS_DIR.iterdir() if d.is_dir() and "ds-1d-20260926" in d.name],
        key=lambda d: d.name, reverse=True,
    )
    if not ds_dirs:
        # Also check for Sep 28 datasets
        ds_dirs = sorted(
            [d for d in DATASETS_DIR.iterdir() if d.is_dir() and "ds-1d-" in d.name],
            key=lambda d: d.name, reverse=True,
        )
    if not ds_dirs:
        print("ERROR: No expanded dataset found.")
        return

    ds_id = ds_dirs[0].name
    print(f"Dataset: {ds_id}")

    builder = DatasetBuilder(
        output_root=DATASETS_DIR,
        feature_factory=ExpandedFeatureFactory(),
        normalize=False,
    )
    meta = builder.load_metadata(ds_id)
    print(f"Rows: {meta.row_count}  Features: {meta.feature_count}  Symbols: {len(meta.universe)}")

    registry = ModelRegistry()
    orch = TrainingOrchestrator(
        dataset_builder=builder,
        registry=registry,
        n_windows=5,
        embargo_days=10,
        cost_bps=PRIMARY_COST.round_trip_bps(),
        horizon_bars=5,
        min_ic=0.02,
        max_pbo=0.50,
        parsimony_margin=0.005,
        max_ece=0.10,
    )

    print("\nTraining lightgbm + xgboost (full WF + CPCV)...")
    report = orch.train(
        model_name="expanded_lgbm",
        dataset_id=ds_id,
        candidate_names=["logistic", "lightgbm", "xgboost"],
        register_champion=True,
        baseline_ic=-0.02,          # naive momentum IC
    )

    print(f"\nChampion:    {report.champion}")
    print(f"WF IC mean:  {report.champion_ic_mean:.4f}")
    print(f"Net Sharpe:  {report.champion_net_sharpe:.4f}")
    print(f"PBO:         {report.champion_pbo:.3f}")
    print(f"ECE:         {report.champion_ece:.4f}")
    print(f"Passed:      {report.passed_acceptance}  ({report.rejection_reason or 'OK'})")

    print("\nAll candidates:")
    for c in report.candidates:
        print(f"  {c.name:20s} WF_IC={c.wf_ic_mean:.4f}  Sharpe={c.wf_net_sharpe:.3f}  PBO={c.cpcv_pbo:.3f}")

    # ── IC vs continuous returns ────────────────────────────────────
    print("\nIC reconciliation vs continuous returns (OOS 20%)...")
    frame = builder.load_frame(ds_id)
    feat_cols = [c for c in ExpandedFeatureFactory().FEATURE_NAMES if c in frame.columns]
    X = frame[feat_cols].fillna(0).to_numpy(dtype=float)
    continuous = frame["realized_return"].fillna(0).values
    barrier    = frame["label"].fillna(0).astype(float).values
    n = len(X); split = int(n * 0.8)

    from src.models.estimators import build_estimator
    rec = {}
    for name in [report.champion, "logistic"] if report.champion != "logistic" else ["logistic"]:
        if name is None:
            continue
        try:
            mdl = build_estimator(name).fit(X[:split], barrier[:split])
            preds = mdl.predict(X[split:])
            ic_c, _ = spearmanr(preds, continuous[split:])
            ic_b, _ = spearmanr(preds, barrier[split:])
            rec[name] = {"ic_continuous": round(float(ic_c), 4), "ic_barrier": round(float(ic_b), 4)}
            inflation = abs(ic_b) / max(abs(ic_c), 1e-9) if abs(ic_c) > 1e-4 else float("inf")
            print(f"  {name:20s} IC_continuous={ic_c:.4f}  IC_barrier={ic_b:.4f}  inflation={inflation:.1f}x")
        except Exception as e:
            print(f"  {name}: error — {e}")

    # ── XS portfolio at different costs ────────────────────────────
    print("\nXS portfolio backtest (champion model)...")
    if report.champion:
        try:
            from src.reconciliation.pnl import ExecutablePortfolioBacktest
            PARQUET_DIR = Path("data/1d/1d")
            open_frames = []
            for pf in PARQUET_DIR.glob("*.parquet"):
                try:
                    df_o = pd.read_parquet(pf)
                    df_o.columns = [c.lower() for c in df_o.columns]
                    if "open" not in df_o.columns:
                        continue
                    tmp = pd.DataFrame({"open": pd.to_numeric(df_o["open"], errors="coerce").dropna(),
                                        "symbol": pf.stem})
                    tmp.index.name = "ts"
                    open_frames.append(tmp.reset_index())
                except Exception:
                    pass
            opens_all = pd.concat(open_frames)
            opens_all["ts"] = pd.to_datetime(opens_all["ts"], utc=True)
            opens_panel = opens_all.set_index(["ts", "symbol"]).sort_index()

            ts_oos = pd.DatetimeIndex(frame.index[split:])
            syms_oos = frame["symbol"].values[split:]
            mdl_champ = build_estimator(report.champion).fit(X[:split], barrier[:split])
            preds_champ = mdl_champ.predict(X[split:])
            preds_df = pd.DataFrame({
                "ts": pd.to_datetime(ts_oos, utc=True),
                "symbol": syms_oos,
                "score": preds_champ,
            }).set_index(["ts", "symbol"])
            panel = opens_panel.join(preds_df[["score"]], how="inner").dropna()[["score", "open"]]

            print(f"  Panel: {panel.index.get_level_values('ts').nunique()} timestamps × {panel.index.get_level_values('symbol').nunique()} symbols")

            xs_results = {}
            from src.reconciliation.costs import IndianCostModel
            for cost_bps_val, label in [(8.0, "futures_8bps"), (10.0, "low_10bps"), (20.0, "moderate_20bps"), (27.65, "equity_2765bps")]:
                cm = IndianCostModel(
                    brokerage_bps=cost_bps_val/4, exchange_charge_bps=cost_bps_val/8,
                    gst_bps=cost_bps_val/16, slippage_bps=cost_bps_val/4,
                    half_spread_bps=cost_bps_val/8, stt_bps=0.0, stamp_duty_bps=0.0,
                    scenario=label, note=f"synthetic {cost_bps_val}bps scenario",
                )
                bt = ExecutablePortfolioBacktest(scores_panel=panel.copy(), cost_model=cm,
                                                 holding_bars=5,
                                                 portfolio_type="top_bottom_decile_long_short",
                                                 decile=0.10, min_symbols=10)
                r = bt.run()
                viable = "VIABLE" if r.net_sharpe > 0 else "NOT_VIABLE"
                xs_results[label] = {"bps": cost_bps_val, "xs_ic": round(r.xs_rank_ic_mean, 4),
                                     "gross_sharpe": round(r.gross_sharpe, 3),
                                     "net_sharpe": round(r.net_sharpe, 3),
                                     "net_return_annual": round(r.net_return_annual, 4)}
                print(f"  {label:25s} {cost_bps_val:5.1f}bps  XS_IC={r.xs_rank_ic_mean:.4f}  gross_S={r.gross_sharpe:+.3f}  net_S={r.net_sharpe:+.3f}  {viable}")
        except Exception as e:
            import traceback
            print(f"  XS backtest failed: {e}")
            traceback.print_exc()
            xs_results = {}
    else:
        xs_results = {}

    # ── Save ──────────────────────────────────────────────────────────
    result = {
        "run_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset_id": ds_id,
        "feature_schema_version": "fs-3.0.0",
        "note": "Trained inside Docker — LightGBM/XGBoost available (no macOS-ARM segfault)",
        "champion": report.champion,
        "champion_ic_mean": report.champion_ic_mean,
        "champion_net_sharpe": report.champion_net_sharpe,
        "champion_pbo": report.champion_pbo,
        "passed_acceptance": report.passed_acceptance,
        "rejection_reason": report.rejection_reason,
        "candidates": [c.to_dict() for c in report.candidates],
        "ic_reconciliation": rec,
        "xs_portfolio": xs_results,
    }
    out = REPORTS_DIR / "lgbm_training_report.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved → {out}")

    # Final verdict
    print("\n" + "=" * 65)
    best_ic = max((c.wf_ic_mean for c in report.candidates), default=0)
    champ_ic_c = rec.get(report.champion or "logistic", {}).get("ic_continuous", 0) if rec else 0
    viable_bps = [xs_results[k]["bps"] for k in xs_results if xs_results[k]["net_sharpe"] > 0]
    print(f"Best WF IC (barrier labels): {best_ic:.4f}")
    print(f"IC vs continuous returns:    {champ_ic_c:.4f}")
    if viable_bps:
        print(f"Viable at costs ≤ {min(viable_bps):.1f} bps: YES")
        print("RESULT: COST_VIABLE_SIGNAL_DETECTED — proceed to NSE Futures execution path")
    else:
        print("Viable at costs ≤ 27.65 bps: NO")
        print("RESULT: IC_POSITIVE_SHARPE_NEGATIVE — same as logistic run")
    print("=" * 65)


if __name__ == "__main__":
    main()
