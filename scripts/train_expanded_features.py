"""
scripts/train_expanded_features.py
-------------------------------------
Train ExpandedFeatureFactory (fs-4.0.0: 67 EOD + 8 intraday + 3 market = 78 features)
on all daily parquets in data/1d/1d/ plus 5-minute bars from data/5m/5m/.

Pipeline:
  1. Load all EOD parquets from data/1d/1d/ (optionally filtered by --start-date)
  2. Load NIFTY market-index parquet as market_index_ohlcv
  3. Build ExpandedFeatureFactory dataset (Group A–F EOD features)
     + Group G intraday features (from data/5m/5m/)
     + 3 market-return features (nifty_ret_1d/5d/20d)
  4. Leakage-validate on raw features
  5. Run TrainingOrchestrator (5-window WF + CPCV + calibration + baselines)
  6. Write reconciliation report with continuous-return IC
  7. Write cost-robustness report at 0x / 1x / 1.5x / 2x / 3x cost
  8. Write regime report
  9. Save all results to reports/

Usage::
    PYTHONPATH=. .venv/bin/python scripts/train_expanded_features.py
    PYTHONPATH=. .venv/bin/python scripts/train_expanded_features.py --start-date 2019-01-01
    PYTHONPATH=. .venv/bin/python scripts/train_expanded_features.py --no-intraday
    PYTHONPATH=. .venv/bin/python scripts/train_expanded_features.py --start-date 2020-01-01 --no-intraday
"""
from __future__ import annotations

import argparse
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
from src.features.families.intraday import INTRADAY_FEATURE_NAMES
from src.features.families.news import NEWS_FEATURE_NAMES
from src.features.families.options_iv import OPTIONS_FEATURE_NAMES
from src.reconciliation.costs import ALL_SCENARIOS, PRIMARY_COST
from src.registry.registry import ModelRegistry
from src.training.orchestrator import TrainingOrchestrator
from src.logging_config import get_logger

logger = get_logger(__name__)

PARQUET_DIR    = Path("data/1d/1d")
INTRADAY_DIR   = Path("data/5m/5m")
NEWS_DIR       = Path("data/news/1d")
OPTIONS_DIR    = Path("data/options/1d")
REPORTS_DIR    = Path("reports")
DATASETS_DIR   = Path("artifacts/datasets")
MARKET_SYMBOL  = "NIFTY"   # parquet used as market-regime features
MARKET_COLS    = ["nifty_ret_1d", "nifty_ret_5d", "nifty_ret_20d"]

REPORTS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


# ── Naive baseline IC (NaiveMomentum on ret_20) ───────────────────────────────
def estimate_naive_baseline_ic(ohlcv_by_symbol: dict, sample: int = 20) -> float:
    """Estimate naive 20-bar momentum IC as a baseline to beat."""
    from scipy.stats import spearmanr
    ics = []
    symbols = list(ohlcv_by_symbol.keys())[:sample]
    for sym in symbols:
        df = ohlcv_by_symbol[sym].copy()
        close = df["close"].astype(float)
        ret_20 = close.pct_change(20)
        fwd_ret = close.pct_change(5).shift(-6)
        aligned = pd.DataFrame({"signal": ret_20, "fwd": fwd_ret}).dropna()
        if len(aligned) < 50:
            continue
        ic, _ = spearmanr(aligned["signal"], aligned["fwd"])
        if np.isfinite(ic):
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else 0.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ExpandedFeatureFactory (fs-4.0.0)")
    p.add_argument(
        "--start-date", default=None,
        help="Filter training data to on/after this date (YYYY-MM-DD). "
             "Default: use all available data. Use '2019-01-01' for 7-year window.",
    )
    p.add_argument(
        "--no-intraday", action="store_true",
        help="Skip Group G intraday 5m features (use when data/5m/5m/ is empty).",
    )
    p.add_argument(
        "--no-market-features", action="store_true",
        help="Skip NIFTY market-return features.",
    )
    p.add_argument(
        "--no-news", action="store_true",
        help="Skip Group H SentinelPulse news features (use when data/news/1d/ is empty).",
    )
    p.add_argument(
        "--no-options", action="store_true",
        help="Skip Group I options/IV features (use when data/options/1d/ is empty).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    intraday_dir  = None if args.no_intraday else INTRADAY_DIR
    use_intraday  = (not args.no_intraday) and INTRADAY_DIR.exists() and any(INTRADAY_DIR.glob("*.parquet"))

    use_news = (
        not getattr(args, "no_news", False)
        and NEWS_DIR.exists()
        and any(NEWS_DIR.glob("*.parquet"))
    )
    news_dir = NEWS_DIR if use_news else None

    use_options = (
        not getattr(args, "no_options", False)
        and OPTIONS_DIR.exists()
        and any(OPTIONS_DIR.glob("*.parquet"))
    )
    options_dir = OPTIONS_DIR if use_options else None

    n_eod_features    = len(ExpandedFeatureFactory().FEATURE_NAMES)   # 67
    n_intraday        = len(INTRADAY_FEATURE_NAMES) if use_intraday else 0
    n_market          = len(MARKET_COLS) if not args.no_market_features else 0
    n_news            = len(NEWS_FEATURE_NAMES) if use_news else 0
    n_options         = len(OPTIONS_FEATURE_NAMES) if use_options else 0
    total_features    = n_eod_features + n_intraday + n_market + n_news + n_options

    print("=" * 70)
    print(f"EXPANDED FEATURE TRAINING RUN — {EXPANDED_FEATURE_SCHEMA_VERSION}")
    print(f"  EOD features:      {n_eod_features}  (Groups A–F)")
    print(f"  Intraday features: {n_intraday}  (Group G, 5m bars)")
    print(f"  Market features:   {n_market}  (NIFTY ret 1d/5d/20d)")
    print(f"  News features:     {n_news}  (Group H, SentinelPulse)")
    print(f"  Options features:  {n_options}  (Group I, PCR/IV/OI)")
    print(f"  Total features:    {total_features}")
    if args.start_date:
        print(f"  Training window:   {args.start_date} → today")
    print("=" * 70)

    # ── Step 1: Load EOD parquets ──────────────────────────────────────────
    parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))
    print(f"\n[1] Loading {len(parquet_files)} parquet files from {PARQUET_DIR}...")

    start_dt = pd.Timestamp(args.start_date) if args.start_date else None

    ohlcv_by_symbol: dict[str, pd.DataFrame] = {}
    required_cols = {"open", "high", "low", "close", "volume"}

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            df.columns = [c.lower() for c in df.columns]
            if not required_cols.issubset(df.columns):
                continue
            if len(df) < 252:
                continue
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=list(required_cols))
            if len(df) < 252:
                continue
            # ── 7-year window filter ──────────────────────────────────────
            if start_dt is not None:
                if df.index.tz is not None:
                    start_dt_tz = start_dt.tz_localize(df.index.tz) if start_dt.tzinfo is None else start_dt
                else:
                    start_dt_tz = start_dt.replace(tzinfo=None) if hasattr(start_dt, 'tzinfo') else start_dt
                df = df[df.index >= start_dt_tz]
                if len(df) < 252:
                    continue
            sym = pf.stem
            ohlcv_by_symbol[sym] = df[list(required_cols)]
        except Exception as exc:
            logger.warning(f"Failed to load {pf.name}: {exc}")

    print(f"    Loaded {len(ohlcv_by_symbol)} symbols with ≥252 bars")
    if ohlcv_by_symbol:
        sample_sym = next(iter(ohlcv_by_symbol))
        sample_df  = ohlcv_by_symbol[sample_sym]
        print(f"    Sample: {sample_sym}  rows={len(sample_df)}  "
              f"range={sample_df.index.min().date()} → {sample_df.index.max().date()}")

    # ── Step 2: Load NIFTY as market-index (for nifty_ret features) ───────
    market_index_ohlcv: pd.DataFrame | None = None
    if not args.no_market_features:
        nifty_pf = PARQUET_DIR / f"{MARKET_SYMBOL}.parquet"
        if nifty_pf.exists():
            try:
                nifty_df = pd.read_parquet(nifty_pf)
                nifty_df.columns = [c.lower() for c in nifty_df.columns]
                for col in required_cols:
                    if col in nifty_df.columns:
                        nifty_df[col] = pd.to_numeric(nifty_df[col], errors="coerce")
                market_index_ohlcv = nifty_df
                print(f"    NIFTY market index: {len(nifty_df)} bars "
                      f"({nifty_df.index.min().date()} → {nifty_df.index.max().date()})")
            except Exception as exc:
                logger.warning(f"Could not load NIFTY parquet: {exc}")
        else:
            print(f"    ⚠  NIFTY parquet not found at {nifty_pf} — skipping market features")

    # ── Step 3: Report intraday 5m coverage ───────────────────────────────
    if use_intraday:
        intraday_files = list(INTRADAY_DIR.glob("*.parquet"))
        print(f"\n[2] Intraday 5m data: {len(intraday_files)} symbols in {INTRADAY_DIR}")
        if intraday_files:
            sample_5m = pd.read_parquet(intraday_files[0])
            print(f"    Sample ({intraday_files[0].stem}): {len(sample_5m)} bars  "
                  f"range={sample_5m.index.min()} → {sample_5m.index.max()}")
    else:
        print(f"\n[2] Intraday features: DISABLED (run backfill_intraday.py first)")

    # ── Step 2b: News feature coverage ────────────────────────────────────
    if use_news:
        news_files = list(NEWS_DIR.glob("*.parquet"))
        print(f"\n[2b] News features: {len(news_files)} symbol parquets in {NEWS_DIR}")
        market_pf = NEWS_DIR / "MARKET.parquet"
        if market_pf.exists():
            mdf = pd.read_parquet(market_pf)
            print(f"    MARKET parquet: {len(mdf)} rows  "
                  f"range={mdf.index.min()} → {mdf.index.max()}")
    else:
        print(f"\n[2b] News features: DISABLED "
              f"(run backfill_news_features.py first)")

    # ── Step 4: Estimate naive baseline IC ────────────────────────────────
    print("\n[3] Estimating naive 20-bar momentum baseline IC...")
    baseline_ic = estimate_naive_baseline_ic(ohlcv_by_symbol, sample=30)
    print(f"    Naive momentum IC (est.) = {baseline_ic:.4f}")

    # ── Step 5: Build dataset ─────────────────────────────────────────────
    print(f"\n[4] Building dataset ({EXPANDED_FEATURE_SCHEMA_VERSION}, "
          f"{total_features} features)...")
    label_config = LabelConfig(
        label_type="triple_barrier",
        horizon=5,
        execution_model="next_open",
        cost_bps=PRIMARY_COST.round_trip_bps(),
    )
    builder = DatasetBuilder(
        output_root=DATASETS_DIR,
        feature_factory=ExpandedFeatureFactory(),
        normalize=False,
        run_leakage_validation=True,
        intraday_5m_dir=intraday_dir if use_intraday else None,
        market_index_ohlcv=market_index_ohlcv,
        news_features_dir=news_dir,
        news_source="SentinelPulse" if use_news else "DISABLED",
        options_features_dir=options_dir,
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

    print(f"    Dataset ID:    {meta.dataset_id}")
    print(f"    Rows:          {meta.row_count}")
    print(f"    Features:      {meta.feature_count}")
    print(f"    Schema:        {meta.feature_schema_version}")
    print(f"    Label quality: {meta.label_quality}")
    print(f"    Leakage OK:    {meta.leakage_validated}")
    print(f"    PIT status:    {meta.pit_status}")

    if not meta.leakage_validated:
        print("ABORT: leakage detected in dataset. Fix before training.")
        sys.exit(1)

    # ── Step 6: Train with TrainingOrchestrator ───────────────────────────
    print(f"\n[5] Training with TrainingOrchestrator (5-window WF + CPCV)...")
    print(f"    Baseline IC gate: must beat {baseline_ic:.4f} + 0.005 margin")
    print(f"    Cost scenario:    {PRIMARY_COST.round_trip_bps():.2f} bps")

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

    # ── Step 7: IC reconciliation vs continuous returns ───────────────────
    print("\n[6] IC reconciliation: barrier-IC vs continuous-return IC...")
    frame = builder.load_frame(meta.dataset_id)

    # All feature columns = EOD + intraday (if present) + market (if present) + news (if present) + options (if present)
    eod_feature_cols  = list(ExpandedFeatureFactory().FEATURE_NAMES)
    intraday_feat_cols = [c for c in INTRADAY_FEATURE_NAMES if c in frame.columns]
    market_feat_cols  = [c for c in MARKET_COLS if c in frame.columns]
    news_feat_cols    = [c for c in NEWS_FEATURE_NAMES if c in frame.columns]
    opts_feat_cols    = [c for c in OPTIONS_FEATURE_NAMES if c in frame.columns]
    feature_cols      = eod_feature_cols + intraday_feat_cols + market_feat_cols + news_feat_cols + opts_feat_cols
    available_cols    = [c for c in feature_cols if c in frame.columns]

    from scipy.stats import spearmanr
    continuous_rets = frame["realized_return"].fillna(0).values
    barrier_labels  = frame["label"].fillna(0).astype(float).values

    X = frame[available_cols].fillna(0).to_numpy(dtype=float)
    n = len(X)
    split = int(n * 0.8)

    ic_vs_continuous = 0.0
    ic_vs_barrier    = 0.0
    preds_oos: np.ndarray = np.array([])

    from src.models.estimators import build_estimator
    if report.champion:
        try:
            mdl = build_estimator(report.champion).fit(X[:split], barrier_labels[:split])
            preds_oos = mdl.predict(X[split:])
            rets_oos  = continuous_rets[split:]
            labels_oos = barrier_labels[split:]
            ic_vs_continuous, _ = spearmanr(preds_oos, rets_oos)
            ic_vs_barrier, _    = spearmanr(preds_oos, labels_oos)
            print(f"    OOS IC vs continuous returns:  {ic_vs_continuous:.4f}  ← HONEST metric")
            print(f"    OOS IC vs barrier labels:       {ic_vs_barrier:.4f}")
            print(f"    IC inflation factor:            "
                  f"{abs(ic_vs_barrier)/max(abs(ic_vs_continuous),1e-9):.2f}x")
        except Exception as exc:
            print(f"    Reconciliation failed: {exc}")

    # ── Step 8: Cost robustness ────────────────────────────────────────────
    print("\n[7] Cost robustness analysis...")
    from src.backtest.engine import BacktestEngine, CostModel, cost_sensitivity_analysis
    cost_results: dict[str, dict] = {}
    ts = pd.DatetimeIndex(frame.index)

    if report.champion and split < n and len(preds_oos) > 0:
        try:
            close_oos = pd.Series(
                frame["close"].values[split:] if "close" in frame.columns else np.ones(n - split),
                index=ts[split:],
            )
            prices_oos = pd.DataFrame({"open": close_oos * 1.0005, "close": close_oos})
            signals_oos = np.sign(preds_oos - 0.5)

            for scenario in ALL_SCENARIOS:
                bps = scenario.round_trip_bps()
                per = bps / 4.0
                cm  = CostModel(
                    brokerage_bps=per, fees_bps=per,
                    half_spread_bps=per / 2, slippage_bps=per / 2,
                )
                eng = BacktestEngine(cm)
                bt  = eng.run(prices_oos, signals_oos)
                cost_results[f"{scenario.scenario}_{bps:.1f}bps"] = {
                    "bps": bps,
                    "scenario": scenario.scenario,
                    "net_return": round(bt.net_return, 6),
                    "sharpe":     round(bt.sharpe, 4),
                    "n_trades":   bt.n_trades,
                    "win_rate":   round(bt.win_rate, 4),
                }
                viable = "VIABLE" if bt.sharpe > 0 else "NOT_VIABLE"
                print(f"    {scenario.scenario:15s} ({bps:5.1f}bps): "
                      f"Sharpe={bt.sharpe:+.3f}  net_ret={bt.net_return:+.4f}  {viable}")
        except Exception as exc:
            print(f"    Cost analysis failed: {exc}")

    # ── Step 9: Regime analysis ────────────────────────────────────────────
    print("\n[8] Regime analysis...")
    regime_results: dict[str, dict] = {}
    if "vol_regime_zscore" in frame.columns and report.champion and split < n and len(preds_oos) > 0:
        try:
            vol_z_oos  = frame["vol_regime_zscore"].values[split:]
            trend_d_oos = frame.get("trend_direction", pd.Series(np.zeros(n))).values[split:]

            def _regime_label(vz: float, td: float) -> str:
                if not np.isfinite(vz): return "UNKNOWN"
                if vz > 1.5:  return "HIGH_VOLATILITY"
                if vz < -1.0: return "LOW_VOLATILITY"
                if td > 0.5:  return "TREND_UP"
                if td < -0.5: return "TREND_DOWN"
                return "RANGE"

            regimes = [_regime_label(float(vz), float(td))
                       for vz, td in zip(vol_z_oos, trend_d_oos)]
            for regime in set(regimes):
                mask = np.array([r == regime for r in regimes])
                if mask.sum() < 20:
                    continue
                regime_rets  = continuous_rets[split:][mask]
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

    # ── Step 10: Feature importance (Group F / G coverage) ────────────────
    print("\n[9] Feature group importance analysis...")
    group_importance: dict[str, float] = {}
    if report.champion and hasattr(report, "champion_feature_importance"):
        fi = report.champion_feature_importance or {}
        from src.features.families.reversal import REVERSAL_FEATURE_NAMES
        group_f_imp = sum(fi.get(f, 0) for f in REVERSAL_FEATURE_NAMES)
        group_g_imp = sum(fi.get(f, 0) for f in INTRADAY_FEATURE_NAMES)
        group_a_imp = sum(fi.get(f, 0) for f in ["ret_3", "ret_60", "mom_accel_5",
                                                    "mom_accel_20", "ret_60_rel_vol"])
        total_imp   = sum(fi.values()) or 1.0
        group_importance = {
            "group_f_reversal_pct": round(group_f_imp / total_imp * 100, 1),
            "group_g_intraday_pct": round(group_g_imp / total_imp * 100, 1),
            "group_a_momentum_pct": round(group_a_imp / total_imp * 100, 1),
        }
        print(f"    Group F (reversal):  {group_importance['group_f_reversal_pct']:.1f}%")
        print(f"    Group G (intraday):  {group_importance['group_g_intraday_pct']:.1f}%")
        print(f"    Group A (momentum):  {group_importance['group_a_momentum_pct']:.1f}%")
        if group_importance["group_f_reversal_pct"] < 1.0:
            print("    ⚠  Group F importance < 1% — reversal features not contributing")
        if use_intraday and group_importance["group_g_intraday_pct"] < 1.0:
            print("    ⚠  Group G importance < 1% — intraday features not contributing")
    else:
        print("    Feature importance not available on this report object")

    # ── Step 11: Save results ──────────────────────────────────────────────
    print("\n[10] Saving results...")
    results = {
        "schema":                      "expanded_training_run_v2",
        "feature_schema_version":      EXPANDED_FEATURE_SCHEMA_VERSION,
        "n_eod_features":              n_eod_features,
        "n_intraday_features":         n_intraday,
        "n_market_features":           n_market,
        "n_news_features":             n_news,
        "n_features_total":            total_features,
        "n_symbols":                   len(ohlcv_by_symbol),
        "training_start_date":         args.start_date,
        "dataset_id":                  meta.dataset_id,
        "dataset_rows":                meta.row_count,
        "leakage_validated":           meta.leakage_validated,
        "pit_status":                  meta.pit_status,
        "champion":                    report.champion,
        "champion_ic_mean":            report.champion_ic_mean,
        "champion_pbo":                report.champion_pbo,
        "champion_net_sharpe":         report.champion_net_sharpe,
        "champion_ece":                report.champion_ece,
        "passed_acceptance":           report.passed_acceptance,
        "rejection_reason":            report.rejection_reason,
        "champion_version":            report.champion_version,
        "baseline_ic_naive_momentum":  round(baseline_ic, 4),
        "ic_vs_continuous_returns_oos": round(float(ic_vs_continuous), 4),
        "ic_vs_barrier_labels_oos":    round(float(ic_vs_barrier), 4),
        "ic_inflation_factor":         round(
            abs(ic_vs_barrier) / max(abs(ic_vs_continuous), 1e-9), 2
        ) if abs(float(ic_vs_continuous)) > 1e-4 else None,
        "group_importance":            group_importance,
        "cost_robustness":             cost_results,
        "regime_analysis":             regime_results,
        "candidates":                  [c.to_dict() for c in report.candidates],
        "label_config": {
            "label_type":       label_config.label_type,
            "horizon":          label_config.horizon,
            "execution_model":  label_config.execution_model,
            "cost_bps":         label_config.cost_bps,
        },
    }

    out_path = REPORTS_DIR / "expanded_feature_training_report.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"    Saved → {out_path}")

    # ── Step 12: Final verdict ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPANDED FEATURE TRAINING — FINAL VERDICT")
    print("=" * 70)
    if report.passed_acceptance and float(ic_vs_continuous) > 0.02:
        print("✓ ECONOMIC SIGNAL DETECTED")
        print(f"  Champion: {report.champion}  IC(continuous)={float(ic_vs_continuous):.4f}")
        any_cost_viable = any(
            v["sharpe"] > 0 for v in cost_results.values()
            if "primary" in v.get("scenario", "")
        )
        if any_cost_viable:
            print("  Cost robustness: PASS — positive Sharpe at primary costs")
            print("  RECOMMENDATION: advance to walk-forward confirmation + paper promotion")
        else:
            print("  Cost robustness: FAIL — IC present but Sharpe not positive after costs")
            print("  RECOMMENDATION: investigate cost model / capacity assumptions")
    elif report.passed_acceptance:
        print("⚠ GATES PASSED but IC vs continuous returns is low:")
        print(f"  IC(continuous)={float(ic_vs_continuous):.4f} — potential barrier-IC artifact")
        print("  RECOMMENDATION: investigate IC inflation; do NOT promote")
    else:
        print("✗ NO VERIFIED EDGE")
        print(f"  Rejection reason: {report.rejection_reason}")
        print(f"  IC(continuous)={float(ic_vs_continuous):.4f}")
        print("  RECOMMENDATION: return to feature research")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
