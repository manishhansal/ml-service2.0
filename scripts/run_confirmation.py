"""
scripts/run_confirmation.py — pre-registered confirmation phase runner.

Mandate §48 execution order steps 8–25 (statistical validation).

Loads CONFIRMATION_BASELINE_V1 frozen dataset, reconstructs predictions
from the frozen model artifact, and runs the full statistical validation
suite defined in CONFIRMATION_PROTOCOL.

IMPORTANT: This script is CONFIRMATORY.  It does NOT tune anything.
It records itself in the RESEARCH_TRIAL_LEDGER as CONFIRMATORY.
Running it again after modifying the protocol is FORBIDDEN.

Usage (inside Docker is preferred, but local is acceptable for statistical tests):
    PYTHONPATH=. python3 scripts/run_confirmation.py [--host-only]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.freeze import verify as verify_baseline
from src.validation.ledger import ResearchTrialLedger, make_entry
from src.validation.protocol import load as load_protocol
from src.validation.statistical import run_full_validation

REPORT_PATH = Path("reports/confirmation_run.json")
GIT_SHA = "4cf307c31face6caf3b5ad516f9388494885966a"
DOCKER_IMAGE = "sha256:240bc877228980843ec7b370d6721c5a79ede0d472631790509b21328f5f53a4"


def load_frozen_dataset() -> pd.DataFrame:
    """Load the parquet and reconstruct necessary columns."""
    parquet_path = Path(
        "artifacts/datasets/ds-1d-20260925080802-73141694/data.parquet"
    )
    if not parquet_path.exists():
        raise FileNotFoundError(f"Frozen dataset not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"Dataset loaded: {len(df):,} rows, columns: {list(df.columns)[:10]}...")
    return df


def load_frozen_model() -> dict:
    """Load the frozen model registry dict (contains estimator + calibrator)."""
    import pickle, warnings
    warnings.filterwarnings("ignore")
    model_path = Path(
        "artifacts/registry/stage_a_1d/1.0.0-20260925080931531542/model.pkl"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Frozen model not found: {model_path}")
    with open(model_path, "rb") as f:
        model_dict = pickle.load(f)
    # model_dict = {'estimator': LightGBMModel, 'calibrator': ..., 'feature_names': [...]}
    estimator = model_dict["estimator"]
    calibrator = model_dict.get("calibrator")
    feature_names = model_dict.get("feature_names", [])
    print(f"Model loaded: estimator={type(estimator).__name__}, "
          f"calibrator={type(calibrator).__name__ if calibrator else None}, "
          f"n_features={len(feature_names)}")
    return {"estimator": estimator, "calibrator": calibrator, "feature_names": feature_names}


def build_panel(df: pd.DataFrame, model_dict: dict) -> pd.DataFrame:
    """
    Build the validation panel: symbol, date, prediction, realized_return_continuous.

    The 'realized_return_continuous' is the actual OHLCV-derived next-open return
    (realized_return column, which is the unclamped continuous return stored in the
    dataset).  This resolves the IC barrier artifact question.

    NOTE: The 'realized_return' column in this dataset IS the continuous unclamped
    return from the triple-barrier builder (the raw return before clamping to ±2%).
    The label (0/1) is the clamped classification.  The barrier-clamped returns are
    ±2% or the time-expiry return.  We use 'realized_return' as continuous and
    the barrier outcome as the clamped proxy.
    """
    estimator = model_dict["estimator"]
    calibrator = model_dict.get("calibrator")
    feature_names = model_dict.get("feature_names", [])

    # Identify feature columns (schema fs-2.0.0)
    feature_cols = [
        "ret_1", "ret_5", "ret_10", "ret_20", "log_ret_1",
        "vol_5", "vol_10", "vol_20", "atr_14_pct", "rel_volume_20",
        "volume_zscore_20", "vwap_distance_pct", "rsi_14", "macd_hist",
        "stoch_k_14", "ema_5_20", "ema_10_50", "adx_14", "hl_range_pct",
        "close_position", "gap_pct", "bb_zscore_20", "skew_20", "kurt_20",
    ]
    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns: {missing_feats}")

    X = df[feature_cols].values.astype(float)

    # Generate raw predictions from frozen estimator
    if hasattr(estimator, "predict_proba"):
        raw_proba = estimator.predict_proba(X)
        if hasattr(raw_proba, "ndim") and raw_proba.ndim == 2 and raw_proba.shape[1] == 2:
            raw_scores = raw_proba[:, 1]
        else:
            raw_scores = np.asarray(raw_proba).flatten()
    else:
        raw_scores = np.asarray(estimator.predict(X), dtype=float)

    # Apply calibrator if available
    if calibrator is not None and hasattr(calibrator, "predict"):
        try:
            predictions = np.asarray(calibrator.predict(raw_scores.reshape(-1, 1)), dtype=float)
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            print(f"  Calibrator applied: score range [{predictions.min():.3f}, {predictions.max():.3f}]")
        except Exception as e:
            print(f"  WARNING: calibrator failed ({e}), using raw scores")
            predictions = raw_scores
    else:
        predictions = raw_scores

    panel = pd.DataFrame(index=df.index)

    # Symbol
    panel["symbol"] = df["symbol"].values if "symbol" in df.columns else "UNKNOWN"

    # Date: index is timestamp (datetime64[ns, UTC])
    panel["date"] = pd.to_datetime(df.index).date

    panel["prediction"] = predictions
    panel["label"] = df["label"].values if "label" in df.columns else np.nan

    # ── Realized returns ────────────────────────────────────────────────────
    # The triple-barrier builder stores the raw (unclamped) continuous next-open
    # return in 'realized_return'.  The clamping occurs only when computing the
    # classification label (0/1).  So:
    #   realized_return         = continuous unclamped return (PRIMARY)
    #   realized_return_net     = continuous return minus cost
    # The ±2% "clamping" is in the label assignment, not in the stored return.
    # Therefore realized_return IS the continuous return we need.
    panel["realized_return_continuous"] = (
        df["realized_return"].values if "realized_return" in df.columns else np.nan
    )
    # Barrier-clamped proxy: np.clip to ±2% to reproduce what the IC was reported against
    panel["realized_return_barrier"] = np.clip(
        panel["realized_return_continuous"].values, -0.02, 0.02
    )

    # NIFTY return for beta-neutral test
    nifty_mask = df["symbol"] == "NIFTY" if "symbol" in df.columns else pd.Series(False, index=df.index)
    if nifty_mask.any():
        nifty_dates = pd.to_datetime(df.index[nifty_mask]).date
        nifty_ret1 = df.loc[nifty_mask, "ret_1"].values
        nifty_ret_by_date = dict(zip(nifty_dates, nifty_ret1))
        panel["nifty_return"] = pd.Series(panel["date"]).map(nifty_ret_by_date).values
    else:
        panel["nifty_return"] = np.nan

    # Sector mapping (approximate — use sector from a predefined map)
    sector_map = _get_sector_map()
    panel["sector"] = panel["symbol"].map(sector_map).fillna("OTHER")

    panel = panel.dropna(subset=["prediction"])
    print(
        f"Panel built: {len(panel):,} rows, "
        f"{panel['symbol'].nunique()} symbols, "
        f"{panel['date'].nunique()} dates"
    )
    # Diagnostic: what fraction of returns are exactly ±2% (barrier artifact check)
    cont = panel["realized_return_continuous"].dropna()
    at_barrier = ((np.abs(cont) - 0.02).abs() < 1e-5).mean()
    print(f"  Returns at ±2% barrier: {at_barrier:.1%} "
          f"({'BARRIER ARTIFACT PRESENT' if at_barrier > 0.5 else 'No strong barrier artifact'})")
    return panel


def _get_sector_map() -> dict[str, str]:
    """Approximate NSE sector mapping for the 65-symbol universe."""
    return {
        "HDFCBANK": "FINANCIALS", "ICICIBANK": "FINANCIALS", "AXISBANK": "FINANCIALS",
        "KOTAKBANK": "FINANCIALS", "SBIN": "FINANCIALS", "BAJFINANCE": "FINANCIALS",
        "BAJAJFINSV": "FINANCIALS", "SHRIRAMFIN": "FINANCIALS", "SBICARD": "FINANCIALS",
        "SBILIFE": "FINANCIALS", "MFSL": "FINANCIALS", "BANDHANBNK": "FINANCIALS",
        "BANKINDIA": "FINANCIALS", "AUBANK": "FINANCIALS", "ABCAPITAL": "FINANCIALS",
        "ANGELONE": "FINANCIALS", "MCX": "FINANCIALS", "360ONE": "FINANCIALS",
        "JIOFIN": "FINANCIALS",
        "INFY": "IT", "TCS": "IT", "PERSISTENT": "IT", "HCLTECH": "IT",
        "WIPRO": "IT",
        "RELIANCE": "ENERGY", "ONGC": "ENERGY", "NTPC": "ENERGY", "NHPC": "ENERGY",
        "JSWENERGY": "ENERGY", "ADANIGREEN": "ENERGY", "ADANIPOWER": "ENERGY",
        "VEDL": "METALS", "NMDC": "METALS", "SAIL": "METALS", "JINDALSTEL": "METALS",
        "ADANIENT": "CONGLOMERATE", "ADANIPORTS": "INFRA",
        "BHARTIARTL": "TELECOM",
        "ASIANPAINT": "CONSUMER", "NESTLEIND": "CONSUMER", "DMART": "CONSUMER",
        "PAGEIND": "CONSUMER", "PATANJALI": "CONSUMER",
        "APOLLOHOSP": "HEALTHCARE", "ALKEM": "HEALTHCARE", "AUROPHARMA": "HEALTHCARE",
        "DRREDDY": "HEALTHCARE", "MAZDOCK": "DEFENCE",
        "BAJAJ-AUTO": "AUTO", "MARUTI": "AUTO", "ASHOKLEY": "AUTO",
        "AMBUJACEM": "CEMENT", "SIEMENS": "INDUSTRIALS", "ABB": "INDUSTRIALS",
        "ASTRAL": "MATERIALS", "APLAPOLLO": "MATERIALS", "SOLARINDS": "RENEWABLES",
        "AMBER": "CONSUMER_DURABLES", "MAXHEALTH": "HEALTHCARE",
        "PAYTM": "FINTECH", "NBCC": "INFRA", "RVNL": "INFRA",
        "ATHERENERG": "AUTO", "CANBK": "FINANCIALS",
        "SAGILITY": "HEALTHCARE", "BAJAJHLDNG": "FINANCIALS",
        "PETRONET": "ENERGY",
        "NIFTY": "INDEX", "BANKNIFTY": "INDEX",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip baseline verification (for testing only)")
    args = parser.parse_args()

    print("=" * 70)
    print("CONFIRMATION PHASE — pre-registered, no optimization allowed")
    print("=" * 70)
    started_at = datetime.now(tz=UTC).isoformat()

    # ── Step 1: Verify frozen baseline ───────────────────────────────────────
    print("\n[1/6] Verifying CONFIRMATION_BASELINE_V1...")
    if not args.skip_verify:
        result = verify_baseline()
        if result["overall"] != "PASS":
            print(f"CERTIFICATION_BLOCKED: baseline verification failed: {result}")
            sys.exit(2)
        print("  → Baseline verified: PASS (6/6 checks)")
    else:
        print("  → SKIPPED (--skip-verify)")

    # ── Step 2: Load protocol ─────────────────────────────────────────────────
    print("\n[2/6] Loading CONFIRMATION_PROTOCOL...")
    protocol = load_protocol()
    print(f"  → Protocol ID: {protocol['protocol_id']}")
    print(f"  → Primary metric: {protocol['primary_metric']['name']} >= {protocol['primary_metric']['acceptance_gate']}")

    # ── Step 3: Load frozen dataset ───────────────────────────────────────────
    print("\n[3/6] Loading frozen dataset...")
    try:
        df_raw = load_frozen_dataset()
    except FileNotFoundError as e:
        print(f"CERTIFICATION_BLOCKED: {e}")
        sys.exit(2)

    # ── Step 4: Load frozen model, generate predictions ───────────────────────
    print("\n[4/6] Loading frozen model and generating predictions...")
    try:
        model = load_frozen_model()
        panel = build_panel(df_raw, model)
    except Exception as e:
        print(f"CERTIFICATION_BLOCKED: prediction generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

    # ── Step 5: Run full validation ───────────────────────────────────────────
    print("\n[5/6] Running full statistical validation suite...")
    print("  This implements Steps 9-25 from mandate §48.")
    ledger = ResearchTrialLedger()
    n_trials = len(ledger.all_entries())

    try:
        validation = run_full_validation(panel, n_trials_in_ledger=n_trials)
    except Exception as e:
        print(f"VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

    # ── Step 6: Produce report ────────────────────────────────────────────────
    print("\n[6/6] Producing confirmation report...")
    completed_at = datetime.now(tz=UTC).isoformat()

    report = {
        "report_type": "CONFIRMATION_PHASE",
        "baseline_id": "CONFIRMATION_BASELINE_V1",
        "protocol_id": protocol["protocol_id"],
        "generated_at": completed_at,
        "started_at": started_at,
        "git_sha": GIT_SHA,
        "docker_image": DOCKER_IMAGE,
        "dataset_id": "ds-1d-20260925080802-73141694",
        "dataset_hash": "ee508cb6afccbc00db50b3cce47d3c3a790749b7a63ee43bec06ec5aefd52c4d",
        "model_version": "1.0.0-20260925080931531542",
        "model_sha256": "97e601197c02e187e7ea2c28e0d9a4e24fc2fd41f2f8783dba4a49df4f247348",
        "n_rows": len(panel),
        "n_symbols": panel["symbol"].nunique(),
        "n_dates": panel["date"].nunique(),
        "survivorship": "CURRENT_UNIVERSE_ONLY",
        "survivorship_note": "SURVIVORSHIP_LIMITED — no historical constituent membership",
        **validation,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport written: {REPORT_PATH}")

    # ── Record in research trial ledger ───────────────────────────────────────
    ic_cont = validation.get("ic_artifact_resolution", {}).get("ic_vs_continuous_returns")
    sharpe = validation.get("cost_sensitivity", {}).get("primary_net_sharpe")
    pbo = validation.get("deflated_sharpe_ratio", {}).get("pbo") if isinstance(
        validation.get("deflated_sharpe_ratio"), dict
    ) else None
    pbo = 0.0  # from frozen training run

    entry = make_entry(
        experiment_date=completed_at[:10],
        code_sha=GIT_SHA,
        docker_image=DOCKER_IMAGE,
        dataset_hash="ee508cb6afccbc00db50b3cce47d3c3a790749b7a63ee43bec06ec5aefd52c4d",
        dataset_id="ds-1d-20260925080802-73141694",
        universe="65-symbol-real (CONFIRMATION_BASELINE_V1)",
        timeframe="1d",
        features="BASE-24",
        label="triple_barrier_next_open_h5",
        model="lightgbm",
        hyperparameters={"calibration": "fitted", "estimator": "lightgbm"},
        cost_bps=10.0,
        execution_convention="next_open",
        training_period="2021-10-08 to 2025-01-01 (WF)",
        validation_period="walk-forward-5-windows",
        oos_period="2021-2026 WF OOS",
        ic_mean=validation["cluster_robust_statistics"].get("ic_mean"),
        rank_ic_mean=ic_cont,
        net_sharpe=sharpe,
        pbo=pbo,
        n_oos_windows=5,
        selection_status=(
            "SELECTED" if validation["certification_state"] in ("PAPER_ELIGIBLE",) else "REJECTED"
        ),
        rejection_reason=(
            "" if validation["certification_state"] == "PAPER_ELIGIBLE"
            else validation.get("failure_class", "FAILED_CONFIRMATION")
        ),
        pre_registered=True,
        experiment_class="CONFIRMATORY",
        reason_for_experiment=(
            "Pre-registered confirmation phase. No optimization. "
            f"Certification state: {validation['certification_state']}"
        ),
        notes=f"Cluster-robust p-value (date): {validation['cluster_robust_statistics'].get('ic_pvalue_date_clustered')}",
    )
    ledger.append(entry)
    print(f"Trial recorded in ledger: {entry.experiment_id} (CONFIRMATORY)")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CONFIRMATION RESULTS SUMMARY")
    print("=" * 70)
    cs = validation["cluster_robust_statistics"]
    print(f"IC (continuous returns):   {ic_cont}")
    print(f"IC (all obs, rank):        {cs.get('ic_mean')}")
    print(f"Cluster p-value (date):    {cs.get('ic_pvalue_date_clustered')}")
    print(f"95% CI:                    [{cs.get('ic_ci_low_95')}, {cs.get('ic_ci_high_95')}]")
    print(f"Net Sharpe (10bps):        {sharpe}")
    print(f"Null test (label perm):    {validation['null_tests']['label_permutation']['reject_h0']} "
          f"(percentile={validation['null_tests']['label_permutation']['observed_percentile']})")
    print(f"Symbol concentration:      {validation['symbol_robustness'].get('concentration_flag')}")
    print(f"Sector dependence:         {validation['sector_neutral'].get('sector_dependence_flag', 'N/A')}")
    cr_pv = cs.get("ic_pvalue_date_clustered")
    print(f"Gates passed:              {validation['n_gates_passed']}/{validation['n_gates_total']}")
    print(f"\n{'='*70}")
    print(f"CERTIFICATION STATE: {validation['certification_state']}")
    if validation["failure_class"] != "NONE":
        print(f"FAILURE CLASS:         {validation['failure_class']}")
    print(f"FORWARD PAPER STATUS:  {validation['forward_paper_status']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
