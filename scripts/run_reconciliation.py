"""
scripts/run_reconciliation.py — Canonical Prediction-to-P&L Reconciliation.

STEP 2 of the master mandate execution order (§64):
  "Reconcile 65-symbol prediction metrics against independent next-open P&L."
  "Identify exact divergence."

This script:
  1. Loads the frozen 65-symbol dataset and model artifact (no retraining)
  2. Loads the OHLCV cache from run_cross_sectional_research.py
  3. Runs the full reconciliation matrix (mandate §6)
  4. Runs all pre-registered baselines (mandate §14)
  5. Computes honest IC family (TS, XS, corrected) (mandate §23–§25)
  6. Executes the canonical 5-day next-open portfolio backtest (mandate §8)
  7. Runs cost sensitivity across all pre-registered scenarios (mandate §18)
  8. Updates the canonical certification report (mandate §61)
  9. Records the experiment in the research trial ledger (mandate §54)

Exit codes:
  0 : reconciliation complete (verdict may be NO_EDGE — that is valid)
  2 : cannot run (missing required artifacts)

Mandate compliance:
  §5  : do NOT retrain — use frozen predictions
  §6  : row-level comparison of ML eval vs economic eval
  §25 : never declare success unless complete evidence chain supports it
  §49 : if reconciliation == FAIL: STOP TRAINING, FIX DATA/TARGET/EVALUATION
  §64 STEP 2: reconcile before training anything new

Usage:
    PYTHONPATH=. python3 scripts/run_reconciliation.py [--out reports/reconciliation.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
        if _k and _k not in os.environ:
            os.environ[_k] = _v

# OpenMP safety on macOS
for _k, _v in {
    "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1",
    "OMP_MAX_ACTIVE_LEVELS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd

from src.reconciliation.costs import PRIMARY_COST, ALL_SCENARIOS
from src.reconciliation.matrix import PredictionToPnLReconciler
from src.reconciliation.targets import (
    TARGET_REGISTRATION_CERTIFICATE,
    compute_next_open_raw,
    compute_cross_sectional_rank,
)
from src.reconciliation.ic import (
    compute_timeseries_ic,
    compute_cross_sectional_ic,
    compute_overlapping_correction,
    compute_clustered_se,
)
from src.reconciliation.pnl import ExecutablePortfolioBacktest, build_scores_panel
from src.reconciliation.baselines import BaselineFamily, compare_model_to_baselines
from src.validation.ledger import ResearchTrialLedger


# ── Paths ─────────────────────────────────────────────────────────────────────

DATASET_PATH = Path("artifacts/datasets/ds-1d-20260925080802-73141694/data.parquet")
MODEL_PATH = Path("artifacts/registry/stage_a_1d/1.0.0-20260925080931531542/model.pkl")
OHLCV_CACHE = Path("artifacts/cross_sectional/ohlcv_cache")
LEDGER_PATH = Path("artifacts/ledger/RESEARCH_TRIAL_LEDGER.jsonl")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


RAW_OHLCV_DIR = Path("data/1d/1d")


def _load_ohlcv_raw(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load raw OHLCV data (with open prices) from the per-symbol ingestion cache.

    Uses data/1d/1d/<SYMBOL>.parquet — this is the authoritative data source
    with all OHLCV columns including 'open', which is required for next-open
    portfolio execution.
    """
    ohlcv: dict[str, pd.DataFrame] = {}
    missing = []
    for sym in symbols:
        p = RAW_OHLCV_DIR / f"{sym}.parquet"
        if not p.exists():
            missing.append(sym)
            continue
        try:
            df = pd.read_parquet(p)
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(df.columns):
                missing.append(f"{sym}(missing cols)")
                continue
            df.index = pd.to_datetime(df.index, utc=True)
            ohlcv[sym] = df[["open", "high", "low", "close", "volume"]].astype(float)
        except Exception as exc:
            missing.append(f"{sym}({exc})")

    print(f"[reconciliation] Loaded {len(ohlcv)} symbols from raw OHLCV cache "
          f"({len(missing)} missing).", flush=True)
    if missing[:5]:
        print(f"  Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
    return ohlcv


def _load_ohlcv_cache() -> dict[str, pd.DataFrame]:
    """
    Load OHLCV data with open prices for portfolio backtest.

    Primary: data/1d/1d/<SYMBOL>.parquet (raw ingestion, has open prices)
    Fallback: cross_sectional OHLCV cache (no open prices — portfolio disabled)
    """
    # Load 65-symbol universe from dataset metadata
    meta_path = Path("artifacts/datasets/ds-1d-20260925080802-73141694/metadata.json")
    if meta_path.exists():
        import json as _json
        meta = _json.loads(meta_path.read_text())
        symbols = meta.get("universe", [])
        print(f"[reconciliation] Loading OHLCV for {len(symbols)} universe symbols ...",
              flush=True)
        if RAW_OHLCV_DIR.exists():
            ohlcv = _load_ohlcv_raw(symbols)
            if len(ohlcv) >= 10:
                return ohlcv

    # Fallback: cross_sectional cache (no open prices — limited use)
    cache_files = list(OHLCV_CACHE.glob("universe_1d_*.parquet"))
    if not cache_files:
        print("[reconciliation] WARNING: No OHLCV data available. "
              "Economic portfolio evaluation will be skipped.", flush=True)
        return {}

    cache_file = sorted(cache_files)[-1]
    print(f"[reconciliation] Fallback: OHLCV cache {cache_file.name} (no open prices)", flush=True)
    try:
        flat = pd.read_parquet(cache_file)
        # This cache has 'time' as datetime, no 'open' column
        ohlcv: dict[str, pd.DataFrame] = {}
        for sym, grp in flat.groupby("symbol"):
            df = grp.set_index("time").sort_index()
            df.index = pd.to_datetime(df.index, utc=True)
            available = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            ohlcv[sym] = df[available].astype(float)
        print(f"[reconciliation] Fallback loaded {len(ohlcv)} symbols (open may be missing).",
              flush=True)
        return ohlcv
    except Exception as exc:
        print(f"[reconciliation] OHLCV fallback load failed: {exc}", flush=True)
        return {}


def _cost_sensitivity(
    panel_for_bt: pd.DataFrame,
    holding_bars: int,
) -> list[dict]:
    """Run cost sensitivity across all pre-registered scenarios."""
    results = []
    for cm in ALL_SCENARIOS:
        try:
            bt = ExecutablePortfolioBacktest(
                panel_for_bt,
                cost_model=cm,
                holding_bars=holding_bars,
                portfolio_type="top_bottom_decile_long_short",
            )
            r = bt.run()
            results.append({
                "scenario": cm.scenario,
                "round_trip_bps": cm.round_trip_bps(),
                "xs_rank_ic_mean": r.xs_rank_ic_mean,
                "net_sharpe": r.net_sharpe,
                "gross_sharpe": r.gross_sharpe,
                "net_return_annual": r.net_return_annual,
                "max_drawdown": r.max_drawdown,
                "cost_drag_annual": r.cost_drag_annual,
            })
            print(f"  [cost] {cm.scenario:15s} ({cm.round_trip_bps():.1f} bps): "
                  f"net_sharpe={r.net_sharpe:+.3f}", flush=True)
        except Exception as exc:
            results.append({
                "scenario": cm.scenario,
                "round_trip_bps": cm.round_trip_bps(),
                "error": str(exc),
            })
    return results


def _build_baseline_panel(
    pred_df: pd.DataFrame,
    ohlcv: dict[str, pd.DataFrame],
    horizon: int,
) -> pd.DataFrame:
    """Build a panel suitable for the baseline family evaluation."""
    frames = []
    for sym, df in ohlcv.items():
        if "open" not in df.columns:
            continue
        op = df["open"].astype(float).sort_index()
        entry = op.shift(-1)
        exit_ = op.shift(-(1 + horizon))
        true_ret = (exit_ - entry) / entry

        sym_preds = pred_df[pred_df["symbol"] == sym].copy()
        if sym_preds.empty:
            continue

        # Align true_ret
        sym_preds["true_ret"] = true_ret.reindex(sym_preds.index)
        # Add feature columns for regression baselines
        feature_cols = [
            "ret_1", "ret_5", "ret_10", "ret_20", "log_ret_1",
            "vol_5", "vol_10", "vol_20", "atr_14_pct", "rel_volume_20",
            "volume_zscore_20", "vwap_distance_pct", "rsi_14", "macd_hist",
            "stoch_k_14", "ema_5_20", "ema_10_50", "adx_14", "hl_range_pct",
            "close_position", "gap_pct", "bb_zscore_20", "skew_20", "kurt_20",
        ]
        sym_preds["symbol"] = sym
        sym_preds.index.name = "ts"
        frames.append(sym_preds)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames)
    panel.index.name = "ts"
    if "symbol" in panel.columns:
        panel = panel.reset_index().set_index(["ts", "symbol"]).sort_index()
    return panel


def _record_in_ledger(result_summary: dict, code_sha: str) -> None:
    """Append reconciliation result to the research trial ledger."""
    try:
        # Simple append to JSONL — don't depend on the full ledger module
        entry = {
            "experiment_id": result_summary.get("experiment_id", "recon-001"),
            "recorded_at": datetime.now(tz=UTC).isoformat(),
            "experiment_date": str(datetime.now(tz=UTC).date()),
            "code_sha": code_sha,
            "dataset_id": "ds-1d-20260925080802-73141694",
            "experiment_class": "RECONCILIATION",
            "hypothesis": (
                "65-symbol triple-barrier model: reconcile ML IC=0.486/Sharpe=5.47 "
                "against independent executable 5-day next-open portfolio P&L."
            ),
            "model": "lightgbm (frozen, no retraining)",
            "model_version": "1.0.0-20260925080931531542",
            "universe": "65-symbol F&O (CURRENT_UNIVERSE_ONLY)",
            "timeframe": "1d",
            "execution_convention": "next_open",
            "label": "triple_barrier_h5_±2%",
            "ic_mean": result_summary.get("xs_rank_ic_h5", None),
            "net_sharpe": result_summary.get("net_sharpe_ls", None),
            "selection_status": result_summary.get("lifecycle_state", "UNKNOWN"),
            "rejection_reason": (
                "RECONCILIATION" if result_summary.get("lifecycle_state") == "RESEARCH_READY"
                else None
            ),
            "pre_registered": True,
            "reason_for_experiment": (
                "Mandatory reconciliation per master mandate §5 PHASE 2. "
                "DO NOT TRAIN until this is complete."
            ),
            "notes": result_summary.get("honest_verdict", ""),
        }
        with open(LEDGER_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[reconciliation] Ledger entry written: {entry['experiment_id']}", flush=True)
    except Exception as exc:
        print(f"[reconciliation] WARNING: ledger write failed: {exc}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonical prediction-to-P&L reconciliation (mandate §5–§6)."
    )
    parser.add_argument(
        "--out", default="reports/reconciliation.json",
        help="Output report path (default: reports/reconciliation.json)"
    )
    parser.add_argument(
        "--holding-bars", type=int, default=5,
        help="Holding period in bars (default: 5, matches model training horizon)"
    )
    parser.add_argument(
        "--skip-portfolio", action="store_true",
        help="Skip portfolio backtest (faster; IC-only reconciliation)"
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    code_sha = _git_sha()
    print("=" * 70, flush=True)
    print("PREDICTION-TO-P&L RECONCILIATION", flush=True)
    print(f"  git SHA: {code_sha}", flush=True)
    print(f"  mandate: §5 PHASE 2, §6 EXACT RECONCILIATION MATRIX", flush=True)
    print(f"  primary cost: {PRIMARY_COST.round_trip_bps():.1f} bps round-trip", flush=True)
    print("=" * 70, flush=True)

    # ── Preflight checks ──────────────────────────────────────────────────
    missing = []
    if not DATASET_PATH.exists():
        missing.append(str(DATASET_PATH))
    if not MODEL_PATH.exists():
        missing.append(str(MODEL_PATH))

    if missing:
        print(f"[ERROR] Required artifacts missing: {missing}", flush=True)
        print("Cannot run reconciliation without frozen dataset and model.", flush=True)
        sys.exit(2)

    # ── Target pre-registration certificate ─────────────────────────────
    print(f"\n[1/7] Target pre-registration: {TARGET_REGISTRATION_CERTIFICATE['registration_hash'][:16]}...",
          flush=True)
    for t in TARGET_REGISTRATION_CERTIFICATE["all_targets"]:
        print(f"  → {t}", flush=True)

    # ── Load OHLCV cache ─────────────────────────────────────────────────
    print(f"\n[2/7] Loading OHLCV data ...", flush=True)
    ohlcv = {} if args.skip_portfolio else _load_ohlcv_cache()

    # ── Run reconciliation matrix ─────────────────────────────────────────
    print(f"\n[3/7] Running reconciliation matrix (frozen model, no retraining) ...",
          flush=True)

    reconciler = PredictionToPnLReconciler(
        dataset_path=DATASET_PATH,
        model_path=MODEL_PATH,
        ohlcv_by_symbol=ohlcv,
        cost_model=PRIMARY_COST,
        holding_bars=args.holding_bars,
        code_sha=code_sha,
    )
    matrix = reconciler.run()

    # ── Cost sensitivity ─────────────────────────────────────────────────
    cost_sensitivity: list[dict] = []
    if ohlcv and not args.skip_portfolio and matrix.economic_evaluation.n_rebalances > 0:
        print(f"\n[4/7] Cost sensitivity across {len(ALL_SCENARIOS)} scenarios ...",
              flush=True)
        try:
            # Rebuild scores panel from reconciler's predictions
            pred_df = reconciler._generate_predictions(
                reconciler._load_dataset(),
                reconciler._load_model()
            )
            scores_panel = reconciler._build_scores_panel_from_predictions(pred_df)
            cost_sensitivity = _cost_sensitivity(scores_panel, args.holding_bars)
        except Exception as exc:
            print(f"  [cost sensitivity] Failed: {exc}", flush=True)
    else:
        print(f"\n[4/7] Cost sensitivity: skipped (no portfolio or no OHLCV)", flush=True)

    # ── Baselines ─────────────────────────────────────────────────────────
    baseline_results_dicts: list[dict] = []
    baseline_comparison: dict = {}
    if ohlcv and not args.skip_portfolio:
        print(f"\n[5/7] Running baseline family ...", flush=True)
        try:
            pred_df = reconciler._generate_predictions(
                reconciler._load_dataset(),
                reconciler._load_model()
            )
            bl_panel = _build_baseline_panel(pred_df, ohlcv, args.holding_bars)
            if not bl_panel.empty:
                feature_cols = [
                    "ret_1", "ret_5", "ret_10", "ret_20", "log_ret_1",
                    "vol_5", "vol_10", "vol_20", "atr_14_pct", "rel_volume_20",
                    "volume_zscore_20", "vwap_distance_pct", "rsi_14", "macd_hist",
                    "stoch_k_14", "ema_5_20", "ema_10_50", "adx_14", "hl_range_pct",
                    "close_position", "gap_pct", "bb_zscore_20", "skew_20", "kurt_20",
                ]
                bl_family = BaselineFamily(
                    bl_panel,
                    feature_cols=feature_cols,
                    return_col="true_ret",
                    holding_bars=args.holding_bars,
                )
                bl_results = bl_family.run_all()
                baseline_results_dicts = [b.to_dict() for b in bl_results]

                model_ic = {"xs_rank_ic_mean": matrix.economic_evaluation.xs_rank_ic_h5}
                baseline_comparison = compare_model_to_baselines(
                    model_ic, bl_results, key="xs_rank_ic_mean"
                )
                print(f"  [baselines] Model XS IC = {model_ic['xs_rank_ic_mean']:+.4f}", flush=True)
                print(f"  [baselines] Best baseline IC = {baseline_comparison['baseline_max']:+.4f}", flush=True)
                print(f"  [baselines] {baseline_comparison['verdict']}", flush=True)
        except Exception as exc:
            print(f"  [baselines] Failed: {exc}", flush=True)

    # ── IC family summary ──────────────────────────────────────────────────
    print(f"\n[6/7] IC summary ...", flush=True)
    ic_summary = {
        "ml_ts_pearson_ic": matrix.ml_evaluation.ic_pearson,
        "ml_ts_rank_ic": matrix.ml_evaluation.ic_rank,
        "barrier_artifact_fraction": matrix.dataset.barrier_fraction,
        "ts_ic_vs_barrier_clamped": matrix.ml_evaluation.ic_vs_barrier_clamped,
        "ts_ic_inflation_factor": matrix.ml_evaluation.ic_inflation_from_barrier,
        "economic_xs_rank_ic_h5": matrix.economic_evaluation.xs_rank_ic_h5,
        "ts_ic_vs_continuous_returns": matrix.economic_evaluation.ts_ic_vs_continuous,
        "ml_sharpe_reported": matrix.ml_evaluation.net_sharpe_reported,
        "economic_net_sharpe_ls": matrix.economic_evaluation.net_sharpe_ls,
        "economic_gross_sharpe_ls": matrix.economic_evaluation.gross_sharpe_ls,
        "t_stat_raw": matrix.ml_evaluation.t_stat_raw,
        "t_stat_corrected_for_overlap": matrix.ml_evaluation.t_stat_corrected,
        "first_divergence": matrix.divergences[0].root_cause[:150] if matrix.divergences else "",
    }
    for k, v in ic_summary.items():
        print(f"  {k:45s}: {v}", flush=True)

    # ── Lifecycle determination ───────────────────────────────────────────
    print(f"\n[7/7] Lifecycle: {matrix.lifecycle_state}", flush=True)
    print(f"  Verdict: {matrix.honest_verdict[:200]}", flush=True)
    if matrix.remaining_blockers:
        print(f"  Blockers ({len(matrix.remaining_blockers)}):", flush=True)
        for b in matrix.remaining_blockers:
            print(f"    - {b}", flush=True)

    # ── Build final report ─────────────────────────────────────────────────
    report = {
        "report_type": "PREDICTION_TO_PNL_RECONCILIATION",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "git_sha": code_sha,
        "mandate": "§5 PHASE 2, §6 EXACT RECONCILIATION MATRIX",
        "primary_cost_scenario": PRIMARY_COST.to_dict(),
        "target_registration": TARGET_REGISTRATION_CERTIFICATE,
        "reconciliation_matrix": matrix.to_dict(),
        "ic_summary": ic_summary,
        "cost_sensitivity": cost_sensitivity,
        "baselines": baseline_results_dicts,
        "baseline_vs_model": baseline_comparison,
        "lifecycle_state": matrix.lifecycle_state,
        "certification_gates": matrix.certification_gate_results,
        "remaining_blockers": matrix.remaining_blockers,
        "honest_verdict": matrix.honest_verdict,
        "training_decision": (
            "DO_NOT_TRAIN — reconciliation shows IC/Sharpe contradictions that must be "
            "resolved first. See divergence chain for root causes."
            if matrix.lifecycle_state == "RESEARCH_READY"
            else "MAY_PROCEED_WITH_CAUTION — pass economic gates, pending robustness."
        ),
    }

    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[reconciliation] Report written: {out_path}", flush=True)

    # ── Update ledger ─────────────────────────────────────────────────────
    result_summary = {
        "experiment_id": f"recon-65sym-h{args.holding_bars}-{code_sha[:8]}",
        "xs_rank_ic_h5": matrix.economic_evaluation.xs_rank_ic_h5,
        "net_sharpe_ls": matrix.economic_evaluation.net_sharpe_ls,
        "lifecycle_state": matrix.lifecycle_state,
        "honest_verdict": matrix.honest_verdict[:300],
    }
    _record_in_ledger(result_summary, code_sha)

    # ── Exit code based on reconciliation result ──────────────────────────
    # Exit 0 always — a NO_EDGE result is a valid, honest outcome (mandate §48)
    # The caller interprets the lifecycle_state.
    print("\n" + "=" * 70, flush=True)
    print(f"RECONCILIATION COMPLETE: {matrix.lifecycle_state}", flush=True)
    print("=" * 70, flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
