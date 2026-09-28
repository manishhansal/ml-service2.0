"""
scripts/train_expanded_features.py
-------------------------------------
Train ExpandedFeatureFactory (55 features, fs-3.0.0) on the 218 daily
parquet files already on disk.

Pipeline:
  1. Load all parquets from data/1d/1d/
  2. Build ExpandedFeatureFactory dataset (leakage-validate on raw features first)
  3. Run TrainingOrchestrator (5-window WF + CPCV + calibration + baselines)
  4. Write reconciliation report with continuous-return IC
  5. Write cost-robustness report at 0x / 1x / 1.5x / 2x / 3x cost
  6. Write regime report
  7. Save all results to reports/

Usage::
    PYTHONPATH=. .venv/bin/python scripts/train_expanded_features.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.dataset_builder import DatasetBuilder
from src.data.labels import LabelConfig
from src.features.expanded_factory import ExpandedFeatureFactory, EXPANDED_FEATURE_SCHEMA_VERSION
from src.reconciliation.costs import ALL_SCENARIOS, PRIMARY_COST
from src.registry.registry import ModelRegistry
from src.training.orchestrator import TrainingOrchestrator
from src.logging_config import get_logger

logger = get_logger(__name__)

PARQUET_DIR = Path("data/1d/1d")
REPORTS_DIR = Path("reports")
DATASETS_DIR = Path("artifacts/datasets")
REPORTS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# ── Naive baseline IC (NaiveMomentum on ret_20) ───────────────────────────────
# Pre-compute so the orchestrator can enforce the baseline comparison gate.
# We estimate this from the first 20 symbols to save time.
def estimate_naive_baseline_ic(ohlcv_by_symbol: dict, sample: int = 20) -> float:
    """Estimate naive 20-bar momentum IC as a baseline to beat."""
    from scipy.stats import spearmanr
    ics = []
    symbols = list(ohlcv_by_symbol.keys())[:sample]
    for sym in symbols:
        df = ohlcv_by_symbol[sym].copy()
        close = df["close"].astype(float)
        ret_20 = close.pct_change(20)
        fwd_ret = close.pct_change(5).shift(-6)  # 5-bar forward return, next-open approx
        aligned = pd.DataFrame({"signal": ret_20, "fwd": fwd_ret}).dropna()
        if len(aligned) < 50:
            continue
        ic, _ = spearmanr(aligned["signal"], aligned["fwd"])
        if np.isfinite(ic):
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else 0.0


def main() -> None:
    print("=" * 70)
    print("EXPANDED FEATURE TRAINING RUN — fs-3.0.0 (55 features)")
    print("=" * 70)

    # ── Step 1: Load parquets ──────────────────────────────────────────────
    parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))
    print(f"\n[1] Loading {len(parquet_files)} parquet files from {PARQUET_DIR}...")

    ohlcv_by_symbol: dict[str, pd.DataFrame] = {}
    required_cols = {"open", "high", "low", "close", "volume"}

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            # Normalise column names
            df.columns = [c.lower() for c in df.columns]
            if not required_cols.issubset(df.columns):
                continue
            if len(df) < 252:   # need at least 1 year
                continue
            # Ensure numeric OHLCV
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=list(required_cols))
            if len(df) < 252:
                continue
            sym = pf.stem
            ohlcv_by_symbol[sym] = df[list(required_cols)]
        except Exception as exc:
            logger.warning(f"Failed to load {pf.name}: {exc}")

    print(f"    Loaded {len(ohlcv_by_symbol)} symbols with ≥252 bars")
    sample_sym = next(iter(ohlcv_by_symbol))
    sample_df = ohlcv_by_symbol[sample_sym]
    print(f"    Sample: {sample_sym}  rows={len(sample_df)}  "
          f"range={sample_df.index.min().date()} → {sample_df.index.max().date()}")

    # ── Step 2: Estimate naive baseline IC ───────────────────────────────
    print("\n[2] Estimating naive 20-bar momentum baseline IC...")
    baseline_ic = estimate_naive_baseline_ic(ohlcv_by_symbol, sample=30)
    print(f"    Naive momentum IC (est.) = {baseline_ic:.4f}")

    # ── Step 3: Build dataset with ExpandedFeatureFactory ─────────────────
    print(f"\n[3] Building dataset with ExpandedFeatureFactory ({EXPANDED_FEATURE_SCHEMA_VERSION})...")
    label_config = LabelConfig(
        label_type="triple_barrier",
        horizon=5,
        execution_model="next_open",
        cost_bps=PRIMARY_COST.round_trip_bps(),
    )
    builder = DatasetBuilder(
        output_root=DATASETS_DIR,
        feature_factory=ExpandedFeatureFactory(),
        normalize=False,   # per-fold normalization in orchestrator
        run_leakage_validation=True,
    )
    try:
        meta = builder.build(
            ohlcv_by_symbol=ohlcv_by_symbol,
            label_config=label_config,
            timeframe="1d",
        )
    except Exception as exc:
        print(f"ERROR building dataset: {exc}")
        raise

    print(f"    Dataset ID:   {meta.dataset_id}")
    print(f"    Rows:         {meta.row_count}")
    print(f"    Features:     {meta.feature_count}")
    print(f"    Label quality: {meta.label_quality}")
    print(f"    Leakage OK:   {meta.leakage_validated}")
    print(f"    PIT status:   {meta.pit_status}")

    if not meta.leakage_validated:
        print("ABORT: leakage detected in dataset. Fix before training.")
        sys.exit(1)

    # ── Step 4: Train with TrainingOrchestrator ───────────────────────────
    print(f"\n[4] Training with TrainingOrchestrator (5-window WF + CPCV + calibration)...")
    print(f"    Baseline IC gate: champion must beat {baseline_ic:.4f} + 0.005 margin")
    print(f"    Cost scenario: {PRIMARY_COST.round_trip_bps():.2f} bps (Indian primary)")

    registry = ModelRegistry()
    orch = TrainingOrchestrator(
        dataset_builder=builder,
        registry=registry,
        n_windows=5,
        embargo_days=10,
        cost_bps=PRIMARY_COST.round_trip_bps(),
        horizon_bars=label_config.horizon,
        min_ic=0.02,
        max_pbo=0.5,
        parsimony_margin=0.005,
        max_ece=0.10,
    )

    report = orch.train(
        model_name=f"expanded_{EXPANDED_FEATURE_SCHEMA_VERSION.replace('.','_')}",
        dataset_id=meta.dataset_id,
        candidate_names=["logistic", "ridge", "naive_momentum", "lightgbm", "xgboost"],
        register_champion=True,
        baseline_ic=baseline_ic,
    )

    print(f"\n    Champion:      {report.champion}")
    print(f"    WF IC mean:    {report.champion_ic_mean:.4f}")
    print(f"    PBO:           {report.champion_pbo:.3f}")
    print(f"    Net Sharpe:    {report.champion_net_sharpe:.3f}  (horizon-corrected)")
    print(f"    ECE:           {report.champion_ece:.4f}")
    print(f"    Passed gates:  {report.passed_acceptance}")
    if not report.passed_acceptance:
        print(f"    Rejection:     {report.rejection_reason}")

    # ── Step 5: IC reconciliation vs continuous returns ───────────────────
    print("\n[5] IC reconciliation: barrier-IC vs continuous-return IC...")
    frame = builder.load_frame(meta.dataset_id)
    feature_cols = list(ExpandedFeatureFactory().FEATURE_NAMES)
    available_cols = [c for c in feature_cols if c in frame.columns]

    from scipy.stats import spearmanr
    continuous_rets = frame["realized_return"].fillna(0).values
    barrier_labels = frame["label"].fillna(0).astype(float).values

    # Use the champion's walk-forward split to get OOS predictions
    X = frame[available_cols].fillna(0).to_numpy(dtype=float)
    ts = pd.DatetimeIndex(frame.index)
    n = len(X)
    split = int(n * 0.8)
    # Simple single OOS split for reconciliation
    from src.models.estimators import build_estimator
    if report.champion:
        try:
            mdl = build_estimator(report.champion).fit(X[:split], barrier_labels[:split])
            preds_oos = mdl.predict(X[split:])
            rets_oos = continuous_rets[split:]
            labels_oos = barrier_labels[split:]
            ic_vs_continuous, _ = spearmanr(preds_oos, rets_oos)
            ic_vs_barrier, _ = spearmanr(preds_oos, labels_oos)
            print(f"    OOS IC vs continuous returns:  {ic_vs_continuous:.4f}  ← HONEST metric")
            print(f"    OOS IC vs barrier labels:       {ic_vs_barrier:.4f}  ← (artificially inflated if IC>>continuous)")
            print(f"    IC inflation factor:            {abs(ic_vs_barrier)/max(abs(ic_vs_continuous),1e-9):.2f}x")
        except Exception as exc:
            print(f"    Reconciliation failed: {exc}")
            ic_vs_continuous = 0.0
            ic_vs_barrier = 0.0
    else:
        ic_vs_continuous = 0.0
        ic_vs_barrier = 0.0

    # ── Step 6: Cost robustness analysis ──────────────────────────────────
    print("\n[6] Cost robustness analysis...")
    from src.backtest.engine import BacktestEngine, CostModel, cost_sensitivity_analysis

    # Build signal series from champion predictions on OOS set
    cost_results: dict[str, dict] = {}
    if report.champion and split < n:
        try:
            ohlcv_concat = pd.concat(list(ohlcv_by_symbol.values())).sort_index()
            close_oos = pd.Series(
                frame["close"].values[split:] if "close" in frame.columns else np.ones(n-split),
                index=ts[split:]
            )
            prices_oos = pd.DataFrame({
                "open": close_oos * 1.0005,
                "close": close_oos,
            })
            signals_oos = np.sign(preds_oos - 0.5)

            for scenario in ALL_SCENARIOS:
                bps = scenario.round_trip_bps()
                per = bps / 4.0
                cm = CostModel(
                    brokerage_bps=per, fees_bps=per,
                    half_spread_bps=per/2, slippage_bps=per/2
                )
                eng = BacktestEngine(cm)
                bt = eng.run(prices_oos, signals_oos)
                cost_results[f"{scenario.scenario}_{bps:.1f}bps"] = {
                    "bps": bps,
                    "scenario": scenario.scenario,
                    "net_return": round(bt.net_return, 6),
                    "sharpe": round(bt.sharpe, 4),
                    "n_trades": bt.n_trades,
                    "win_rate": round(bt.win_rate, 4),
                }
                viable = "VIABLE" if bt.sharpe > 0 else "NOT_VIABLE"
                print(f"    {scenario.scenario:15s} ({bps:5.1f}bps): "
                      f"Sharpe={bt.sharpe:+.3f}  net_ret={bt.net_return:+.4f}  {viable}")
        except Exception as exc:
            print(f"    Cost analysis failed: {exc}")
            cost_results = {}

    # ── Step 7: Regime analysis ───────────────────────────────────────────
    print("\n[7] Regime analysis...")
    regime_results: dict[str, dict] = {}
    if "vol_regime_zscore" in frame.columns and report.champion and split < n:
        try:
            vol_z_oos = frame["vol_regime_zscore"].values[split:]
            trend_d_oos = frame.get("trend_direction", pd.Series(np.zeros(n))).values[split:]

            def _regime_label(vz: float, td: float) -> str:
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

            regimes = [_regime_label(float(vz), float(td))
                       for vz, td in zip(vol_z_oos, trend_d_oos)]

            for regime in set(regimes):
                mask = np.array([r == regime for r in regimes])
                if mask.sum() < 20:
                    continue
                regime_rets = continuous_rets[split:][mask]
                regime_preds = preds_oos[mask]
                if len(regime_preds) < 20:
                    continue
                ic_r, _ = spearmanr(regime_preds, regime_rets)
                regime_results[regime] = {
                    "n": int(mask.sum()),
                    "ic": round(float(ic_r), 4) if np.isfinite(ic_r) else 0.0,
                    "mean_ret": round(float(np.mean(regime_rets)), 6),
                }
                status = "PASS" if (np.isfinite(ic_r) and ic_r > 0) else "FAIL"
                print(f"    {regime:20s}  n={mask.sum():4d}  IC={ic_r:+.4f}  {status}")
        except Exception as exc:
            print(f"    Regime analysis failed: {exc}")

    # ── Step 8: Save all results ──────────────────────────────────────────
    print("\n[8] Saving results...")

    results = {
        "schema": "expanded_training_run_v1",
        "feature_schema_version": EXPANDED_FEATURE_SCHEMA_VERSION,
        "n_features": meta.feature_count,
        "n_symbols": len(ohlcv_by_symbol),
        "dataset_id": meta.dataset_id,
        "dataset_rows": meta.row_count,
        "leakage_validated": meta.leakage_validated,
        "pit_status": meta.pit_status,
        "champion": report.champion,
        "champion_ic_mean": report.champion_ic_mean,
        "champion_pbo": report.champion_pbo,
        "champion_net_sharpe": report.champion_net_sharpe,
        "champion_ece": report.champion_ece,
        "passed_acceptance": report.passed_acceptance,
        "rejection_reason": report.rejection_reason,
        "champion_version": report.champion_version,
        "baseline_ic_naive_momentum": round(baseline_ic, 4),
        "ic_vs_continuous_returns_oos": round(float(ic_vs_continuous), 4),
        "ic_vs_barrier_labels_oos": round(float(ic_vs_barrier), 4),
        "ic_inflation_factor": round(
            abs(ic_vs_barrier) / max(abs(ic_vs_continuous), 1e-9), 2
        ) if abs(ic_vs_continuous) > 1e-4 else None,
        "cost_robustness": cost_results,
        "regime_analysis": regime_results,
        "candidates": [c.to_dict() for c in report.candidates],
        "label_config": {
            "label_type": label_config.label_type,
            "horizon": label_config.horizon,
            "execution_model": label_config.execution_model,
            "cost_bps": label_config.cost_bps,
        },
    }

    out_path = REPORTS_DIR / "expanded_feature_training_report.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"    Saved → {out_path}")

    # ── Step 9: Print final verdict ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPANDED FEATURE TRAINING — FINAL VERDICT")
    print("=" * 70)
    if report.passed_acceptance and ic_vs_continuous > 0.02:
        print("✓ ECONOMIC SIGNAL DETECTED")
        print(f"  Champion: {report.champion}  IC(continuous)={ic_vs_continuous:.4f}")
        any_cost_viable = any(
            v["sharpe"] > 0 for v in cost_results.values()
            if "primary" in v.get("scenario","")
        )
        if any_cost_viable:
            print("  Cost robustness: PASS — positive Sharpe at primary costs")
            print("  RECOMMENDATION: advance to walk-forward confirmation + paper promotion")
        else:
            print("  Cost robustness: FAIL — IC present but Sharpe not positive after costs")
            print("  RECOMMENDATION: investigate cost model / capacity assumptions")
    elif report.passed_acceptance:
        print("⚠ GATES PASSED but IC vs continuous returns is low:")
        print(f"  IC(continuous)={ic_vs_continuous:.4f} — potential barrier-IC artifact")
        print("  RECOMMENDATION: investigate IC inflation; do NOT promote")
    else:
        print("✗ NO VERIFIED EDGE")
        print(f"  Rejection reason: {report.rejection_reason}")
        print(f"  IC(continuous)={ic_vs_continuous:.4f}")
        print("  RECOMMENDATION: return to feature research")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
