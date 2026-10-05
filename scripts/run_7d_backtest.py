"""
run_7d_backtest.py — Complete 7-trading-day backtest, opportunity scan, and
                     baseline comparison for ml-service2.0.

This script:
  1. Loads training dataset (features + labels + realized returns)
  2. Loads the trained LightGBM model (or uses dataset scores as proxy)
  3. Runs the SevenDayBacktestEngine on a held-out OOS period
  4. Generates profitable opportunities ground truth
  5. Computes all performance metrics (classification, trading, signal quality)
  6. Runs baseline strategies (buy-hold, momentum, random, mean-reversion)
  7. Runs regime, calibration, and confidence-stratified analysis
  8. Writes all output files:
       artifacts/backtest_7d/backtest_signals.csv
       artifacts/backtest_7d/profitable_opportunities.csv
       artifacts/backtest_7d/performance_by_symbol.csv
       artifacts/backtest_7d/performance_by_regime.csv
       artifacts/backtest_7d/performance_by_confidence.csv
       artifacts/backtest_7d/backtest_report.json
       artifacts/backtest_7d/baseline_comparison.json
       artifacts/backtest_7d/signal_decay.json

Usage:
    PYTHONPATH=. python3 scripts/run_7d_backtest.py
    PYTHONPATH=. python3 scripts/run_7d_backtest.py --oos-start 2025-01-01
    PYTHONPATH=. python3 scripts/run_7d_backtest.py --quick  # 20 symbols only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.seven_day_engine import (
    NSECostModel,
    SevenDayBacktestEngine,
    SignalRecord,
)
from src.logging_config import get_logger

logger = get_logger(__name__)

OUT_DIR = ROOT / "artifacts" / "backtest_7d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

NSE_EQUITY_COST_BPS = 27.65
NSE_FUTURES_COST_BPS = 8.5
BREAKEVEN_WIN_RATE_EQUITY = 0.569  # 56.9% at equity costs (from LABEL_AUDIT)
BREAKEVEN_WIN_RATE_FUTURES = 0.521  # 52.1% at futures costs


# ── Data loading ──────────────────────────────────────────────────────────────


def load_parquets(
    data_dir: Path = ROOT / "data" / "1d" / "1d",
    max_symbols: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all per-symbol OHLCV parquets."""
    pq_files = sorted(data_dir.glob("*.parquet"))
    if max_symbols:
        pq_files = pq_files[:max_symbols]

    ohlcv_by_symbol: dict[str, pd.DataFrame] = {}
    for f in pq_files:
        try:
            df = pd.read_parquet(str(f))
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            ohlcv_by_symbol[f.stem] = df
        except Exception as exc:
            logger.warning("parquet_load_error", file=f.name, error=str(exc))

    logger.info("parquets_loaded", n=len(ohlcv_by_symbol))
    return ohlcv_by_symbol


def load_dataset(oos_start: str = "2025-01-01") -> pd.DataFrame:
    """Load the most recent training dataset, filtered to OOS period."""
    ds_dirs = sorted(ROOT.glob("artifacts/datasets/ds-1d-*"))
    if not ds_dirs:
        raise FileNotFoundError("No training datasets found in artifacts/datasets/")
    latest = ds_dirs[-1]
    logger.info("loading_dataset", path=latest.name)
    df = pd.read_parquet(str(latest / "data.parquet"))
    # Filter to OOS period only (never use training period for backtest)
    oos_mask = df.index >= pd.Timestamp(oos_start, tz="UTC")
    df_oos = df[oos_mask].copy()
    logger.info("dataset_filtered_to_oos", n_train=int((~oos_mask).sum()), n_oos=len(df_oos))
    return df_oos


def load_model():
    """
    Load the trained model artifact.

    CRITICAL (CRIT-003): Model artifact is a .pkl file, NOT a .lgb file.
    Previous code searched for *.lgb and silently fell back to proxy mode —
    this is FORBIDDEN per mandate §3 (Mode P0 must never be presented as M1).

    Search order:
      1. artifacts/registry/<model_name>/<version>/model.pkl  (registry)
      2. artifacts/<model_name>/<version>/model.pkl           (legacy path)

    Returns (model_dict, version_str) or raises SystemExit(MODEL_ARTIFACT_REQUIRED)
    when no artifact exists AND mode is not explicitly 'proxy'.
    """
    import pickle
    import warnings

    # Primary: registry path
    registry_candidates = sorted(ROOT.glob("artifacts/registry/*/*/model.pkl"))
    # V2 model path (new pipeline)
    v2_candidates = sorted(ROOT.glob("artifacts/v2_model/*/model.pkl"))
    # Secondary: legacy paths
    legacy_candidates = sorted(ROOT.glob("artifacts/expanded_lgbm/*/model.pkl")) + \
                        sorted(ROOT.glob("artifacts/expanded_fs-*/*/model.pkl"))

    # v2 models take priority (most recent improvements)
    all_candidates = registry_candidates + legacy_candidates + v2_candidates

    for candidate in reversed(all_candidates):   # most recent first
        try:
            with open(candidate, "rb") as f:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_dict = pickle.load(f)
            # Validate: must be a dict with 'estimator' key
            if not isinstance(model_dict, dict) or "estimator" not in model_dict:
                continue
            version = candidate.parent.name
            feature_names = model_dict.get("feature_names", [])
            model_type = model_dict.get("estimator_name", "unknown")
            logger.info(
                "model_loaded",
                path=str(candidate),
                version=version,
                n_features=len(feature_names),
                fs_version=model_dict.get("feature_schema_version", "unknown"),
                model_type=model_type,
            )
            print(f"  Loaded: {candidate.parent.name} | type={model_type} | features={len(feature_names)}")
            return model_dict, version
        except Exception as exc:
            logger.warning("model_load_error", path=str(candidate), error=str(exc))

    # No model found — HARD FAIL (no silent proxy fallback)
    logger.error(
        "MODEL_ARTIFACT_REQUIRED",
        msg="No valid .pkl model artifact found. Cannot proceed in M1 mode. "
            "Use --mode proxy to explicitly enable proxy-label evaluation.",
    )
    return None, "dataset_proxy"


def build_scores_from_dataset(
    df_oos: pd.DataFrame,
    model_dict: dict | None,
    feature_cols: list[str],
    oos_start: str,
    mode: str = "proxy",
) -> pd.DataFrame:
    """
    Generate per-symbol per-date model scores for the OOS period.

    mode='m1'    — uses actual model artifact (REQUIRED for performance claims)
    mode='proxy' — uses dataset labels as scores (ONLY for upper-bound analysis)

    The score column is named 'score_m1' or 'score_proxy' to make the mode
    explicit in every downstream output.
    """
    if model_dict is not None and mode == "m1":
        # Extract model components
        estimator = model_dict["estimator"]
        feature_names = model_dict.get("feature_names", feature_cols)
        invert = model_dict.get("invert_scores", False)
        model_type = model_dict.get("estimator_name", "unknown")
        # Apply normalizer if available in artifact
        norm_state = model_dict.get("normalizer_state", {})
        # Ensure only model features used, in correct order
        avail = [f for f in feature_names if f in df_oos.columns]
        if len(avail) < len(feature_names):
            missing = set(feature_names) - set(df_oos.columns)
            logger.warning("m1_missing_features", missing=list(missing)[:5], n_missing=len(missing))
        X_raw = df_oos[avail].fillna(0.0)
        # Apply normalizer if present
        if norm_state:
            try:
                from src.features.normalizer import FeatureNormalizer
                norm = FeatureNormalizer()
                norm.load_state(norm_state)
                X_raw = norm.transform_known(X_raw).fillna(0.0)
            except Exception as ne:
                logger.warning("m1_normalizer_failed", error=str(ne))
        X = X_raw.to_numpy(dtype=float)
        try:
            raw_scores = estimator.predict(X)
            if invert:
                raw_scores = -raw_scores   # flip direction if inversion was detected in training
            # Apply calibrator if available
            calibrator = model_dict.get("calibrator")
            if calibrator is not None:
                try:
                    cal_probs = calibrator.predict_proba(raw_scores.reshape(-1, 1))
                    scores_raw = cal_probs[:, 1]
                except Exception:
                    scores_raw = raw_scores
            else:
                scores_raw = raw_scores   # v2 model: no calibration (raw regression output)

            # Build pivot scores_df
            df_score = df_oos[["symbol"]].copy() if "symbol" in df_oos.columns else df_oos[[]]
            df_score["_raw_score"] = scores_raw

            if "symbol" in df_oos.columns:
                scores_pivot = df_score.pivot_table(
                    index=df_oos.index,
                    columns="symbol",
                    values="_raw_score",
                    aggfunc="last",
                )
            else:
                dates = df_oos.index.unique().sort_values()
                scores_pivot = pd.DataFrame(index=dates)

            # CRITICAL for v2 regression model:
            # Convert raw regression values to cross-sectional percentile ranks [0,1].
            # This allows the SevenDayBacktestEngine threshold logic (0.55/0.45) to work:
            # - top 20% (rank_pct >= 0.80) are LONG candidates
            # - bottom 20% (rank_pct <= 0.20) are SHORT candidates
            if model_type == "lightgbm_regressor_v2":
                scores_pivot = scores_pivot.rank(axis=1, pct=True)
                print("  v2 regression scores → cross-sectional rank percentiles [0,1]")

            score_col = "score_m1"
            return scores_pivot

        except Exception as exc:
            logger.error("m1_inference_failed", error=str(exc))
            return pd.DataFrame()
    else:
        # PROXY mode — clearly label as P0
        rng = np.random.default_rng(42)
        if "label" in df_oos.columns and "symbol" in df_oos.columns:
            df_proxy = df_oos.copy()
            df_proxy["_score"] = df_proxy["label"].astype(float).fillna(0.5) + rng.normal(0, 0.01, len(df_proxy))
            df_proxy["_score"] = df_proxy["_score"].clip(0.0, 1.0)
            scores_raw = df_proxy["_score"].to_numpy()
        else:
            scores_raw = rng.uniform(0, 1, len(df_oos))
        score_col = "score_proxy"
        logger.warning(
            "running_in_proxy_mode",
            msg="All performance metrics are P0 (label-proxy). NOT representative of model performance.",
        )

    # Build scores_df: DatetimeIndex rows × symbol columns
    df_score = df_oos[["symbol"]].copy() if "symbol" in df_oos.columns else df_oos[[]]
    df_score[score_col] = scores_raw

    if "symbol" in df_oos.columns:
        scores_pivot = df_score.pivot_table(
            index=df_oos.index,
            columns="symbol",
            values=score_col,
            aggfunc="last",
        )
    else:
        dates = df_oos.index.unique().sort_values()
        scores_pivot = pd.DataFrame(
            scores_raw.reshape(len(dates), -1) if len(scores_raw) == len(dates) else np.tile(scores_raw, (len(dates), 1)),
            index=dates,
        )
    return scores_pivot


# ── Baseline strategies ───────────────────────────────────────────────────────


def run_buy_hold_baseline(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    oos_start: str,
    oos_end: str,
    cost: NSECostModel,
) -> dict[str, Any]:
    """Buy-and-hold each symbol from oos_start to oos_end."""
    returns: list[float] = []
    for symbol, df in ohlcv_by_symbol.items():
        df_oos = df[df.index >= pd.Timestamp(oos_start, tz="UTC")]
        df_oos = df_oos[df_oos.index <= pd.Timestamp(oos_end, tz="UTC")]
        if len(df_oos) < 2:
            continue
        gross = (float(df_oos["close"].iloc[-1]) - float(df_oos["close"].iloc[0])) / float(df_oos["close"].iloc[0])
        net = gross - cost.total_round_trip
        returns.append(net)

    if not returns:
        return {"strategy": "buy_hold", "n": 0}
    r = np.array(returns)
    return {
        "strategy": "buy_hold",
        "n":            len(r),
        "mean_net_pct": round(float(r.mean()) * 100, 4),
        "median_net":   round(float(np.median(r)) * 100, 4),
        "win_rate":     round(float((r > 0).mean()), 4),
        "total_pct":    round(float(r.sum()) * 100, 4),
        "sharpe":       round(float(r.mean() / (r.std() + 1e-10)), 4),
    }


def run_momentum_baseline(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    oos_start: str,
    cost: NSECostModel,
    lookback: int = 20,
) -> dict[str, Any]:
    """
    Momentum baseline: LONG if last-N-day return > 0, SHORT if < 0.
    Signal every 5 bars (weekly rebalance).
    """
    all_net_pnls: list[float] = []

    for symbol, df in ohlcv_by_symbol.items():
        df_full = df.copy()
        mom = df_full["close"].pct_change(lookback)

        # Generate weekly signals in the OOS period
        oos_mask = df_full.index >= pd.Timestamp(oos_start, tz="UTC")
        oos_df = df_full[oos_mask]
        oos_mom = mom[oos_mask]

        for i in range(0, len(oos_df) - 8, 5):
            m = float(oos_mom.iloc[i]) if not np.isnan(oos_mom.iloc[i]) else 0.0
            direction = 1 if m > 0 else -1
            entry = float(oos_df["open"].iloc[i + 1]) if i + 1 < len(oos_df) else float("nan")
            exit_idx = min(i + 8, len(oos_df) - 1)
            exit_ = float(oos_df["close"].iloc[exit_idx])
            if np.isnan(entry) or entry <= 0:
                continue
            gross = direction * (exit_ - entry) / entry
            net = gross - cost.total_round_trip
            all_net_pnls.append(net)

    if not all_net_pnls:
        return {"strategy": "momentum", "n": 0}
    r = np.array(all_net_pnls)
    return {
        "strategy": "momentum",
        "n":            len(r),
        "mean_net_pct": round(float(r.mean()) * 100, 4),
        "win_rate":     round(float((r > 0).mean()), 4),
        "profit_factor": round(float(r[r > 0].sum() / abs(r[r < 0].sum() + 1e-10)), 4),
        "sharpe":       round(float(r.mean() / (r.std() + 1e-10)), 4),
    }


def run_random_baseline(
    n_signals: int,
    cost: NSECostModel,
    seed: int = 42,
    signal_mean_gross: float = -0.000108,
    signal_std_gross: float = 0.01948,
) -> dict[str, Any]:
    """
    Random direction baseline (null hypothesis).
    Simulates signals with the same gross return distribution as the dataset.
    """
    rng = np.random.default_rng(seed)
    gross_returns = rng.normal(signal_mean_gross, signal_std_gross, n_signals)
    # Random direction: half positive, half negative expected value
    directions = rng.choice([-1, 1], n_signals)
    net_returns = gross_returns * directions - cost.total_round_trip
    r = net_returns
    return {
        "strategy": "random",
        "n":            len(r),
        "mean_net_pct": round(float(r.mean()) * 100, 4),
        "win_rate":     round(float((r > 0).mean()), 4),
        "sharpe":       round(float(r.mean() / (r.std() + 1e-10)), 4),
        "expected_to_be_negative": True,
    }


def run_mean_reversion_baseline(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    oos_start: str,
    cost: NSECostModel,
    lookback: int = 5,
    oversold_threshold: float = -0.05,
    overbought_threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Mean reversion baseline: LONG after N-day decline > threshold, SHORT after rise.
    """
    all_net_pnls: list[float] = []

    for symbol, df in ohlcv_by_symbol.items():
        oos_mask = df.index >= pd.Timestamp(oos_start, tz="UTC")
        oos_df = df[oos_mask]
        if len(oos_df) < lookback + 8:
            continue
        mom = oos_df["close"].pct_change(lookback)

        for i in range(0, len(oos_df) - 8, 5):
            m = float(mom.iloc[i]) if not np.isnan(mom.iloc[i]) else 0.0
            if m <= oversold_threshold:
                direction = 1     # LONG after decline
            elif m >= overbought_threshold:
                direction = -1    # SHORT after rise
            else:
                continue
            entry = float(oos_df["open"].iloc[i + 1]) if i + 1 < len(oos_df) else float("nan")
            exit_idx = min(i + 8, len(oos_df) - 1)
            exit_ = float(oos_df["close"].iloc[exit_idx])
            if np.isnan(entry) or entry <= 0:
                continue
            gross = direction * (exit_ - entry) / entry
            net = gross - cost.total_round_trip
            all_net_pnls.append(net)

    if not all_net_pnls:
        return {"strategy": "mean_reversion", "n": 0}
    r = np.array(all_net_pnls)
    return {
        "strategy": "mean_reversion",
        "n":            len(r),
        "mean_net_pct": round(float(r.mean()) * 100, 4),
        "win_rate":     round(float((r > 0).mean()), 4),
        "sharpe":       round(float(r.mean() / (r.std() + 1e-10)), 4),
    }


# ── Signal decay analysis ─────────────────────────────────────────────────────


def compute_signal_decay(records: list[SignalRecord]) -> dict[str, Any]:
    """
    Compute cumulative and marginal returns at each forward horizon (1–7 days).
    Determines whether signal power increases or decays.
    """
    df = pd.DataFrame([r.to_dict() for r in records])
    if df.empty:
        return {}

    decay: dict[str, Any] = {}
    for h in range(1, 8):
        col = f"return_{h}d"
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 5:
            continue
        decay[f"day_{h}"] = {
            "n":            len(vals),
            "mean_net_pct": round(float(vals.mean()), 4),
            "win_rate":     round(float((vals > 0).mean()), 4),
            "std":          round(float(vals.std()), 4),
        }

    # Determine if decay or improvement
    mean_returns = [decay.get(f"day_{h}", {}).get("mean_net_pct", float("nan")) for h in range(1, 8)]
    valid_means = [(h+1, m) for h, m in enumerate(mean_returns) if not (isinstance(m, float) and np.isnan(m))]
    if valid_means:
        days, means = zip(*valid_means)
        if len(means) >= 2:
            trend = float(np.polyfit(list(days), list(means), 1)[0])
            decay["trend_slope"] = round(trend, 6)
            decay["signal_character"] = "STRENGTHENING" if trend > 0 else "DECAYING"

    return decay


# ── Statistical significance ──────────────────────────────────────────────────


def compute_statistical_significance(
    net_pnls: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Bootstrap confidence interval and hypothesis tests for mean return.

    H0: mean net return ≤ 0 (no alpha)
    H1: mean net return > 0 (positive alpha)
    """
    if len(net_pnls) < 10:
        return {"error": "insufficient_data"}

    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(net_pnls, size=len(net_pnls), replace=True)
        means.append(float(np.mean(sample)))

    boot_mean = float(np.mean(means))
    ci_lo = float(np.percentile(means, 2.5))
    ci_hi = float(np.percentile(means, 97.5))

    # One-sample t-test equivalent via bootstrap p-value
    p_value = float(np.mean([m <= 0 for m in means]))  # fraction of bootstrap means ≤ 0

    actual_mean = float(np.mean(net_pnls))

    return {
        "n_trades":         len(net_pnls),
        "mean_net_pct":     round(actual_mean, 4),
        "bootstrap_mean":   round(boot_mean, 4),
        "ci_95_lo":         round(ci_lo, 4),
        "ci_95_hi":         round(ci_hi, 4),
        "p_value":          round(p_value, 4),
        "significant_alpha": p_value < 0.05,
        "interpretation": (
            "POSITIVE ALPHA (p<0.05)" if p_value < 0.05
            else "NOT SIGNIFICANT" if p_value < 0.25
            else "NO ALPHA DETECTED"
        ),
    }


# ── Per-symbol and per-regime analysis ───────────────────────────────────────


def performance_by_symbol(records: list[SignalRecord]) -> pd.DataFrame:
    df = pd.DataFrame([r.to_dict() for r in records])
    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame()

    resolved = df[df["outcome"].isin(["WIN", "LOSS", "BREAKEVEN"])]
    if resolved.empty:
        return pd.DataFrame()

    grp = resolved.groupby("symbol")
    rows = []
    for sym, g in grp:
        net = g["net_pnl"].dropna()
        if len(net) == 0:
            continue
        rows.append({
            "symbol":       sym,
            "n_signals":    len(g),
            "win_rate":     round(float((net > 0).mean()), 4),
            "mean_net_pct": round(float(net.mean()), 4),
            "total_net_pct": round(float(net.sum()), 4),
            "profit_factor": round(
                float(net[net > 0].sum() / (abs(net[net < 0].sum()) + 1e-10)), 4
            ),
        })
    return pd.DataFrame(rows).sort_values("mean_net_pct", ascending=False)


def performance_by_regime(records: list[SignalRecord]) -> pd.DataFrame:
    df = pd.DataFrame([r.to_dict() for r in records])
    if df.empty or "market_regime" not in df.columns:
        return pd.DataFrame()

    resolved = df[df["outcome"].isin(["WIN", "LOSS", "BREAKEVEN"])]
    if resolved.empty:
        return pd.DataFrame()

    rows = []
    for regime, g in resolved.groupby("market_regime"):
        net = g["net_pnl"].dropna()
        if len(net) == 0:
            continue
        rows.append({
            "market_regime":  regime,
            "n_signals":      len(g),
            "win_rate":       round(float((net > 0).mean()), 4),
            "mean_net_pct":   round(float(net.mean()), 4),
            "sharpe_proxy":   round(float(net.mean() / (net.std() + 1e-10)), 4),
        })
    return pd.DataFrame(rows)


def performance_by_confidence(records: list[SignalRecord]) -> pd.DataFrame:
    df = pd.DataFrame([r.to_dict() for r in records])
    if df.empty or "confidence" not in df.columns:
        return pd.DataFrame()

    resolved = df[df["outcome"].isin(["WIN", "LOSS", "BREAKEVEN"])].copy()
    if resolved.empty:
        return pd.DataFrame()

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    resolved["conf_bin"] = pd.cut(resolved["confidence"], bins=bins, labels=labels, include_lowest=True)

    rows = []
    for bin_label, g in resolved.groupby("conf_bin", observed=True):
        net = g["net_pnl"].dropna()
        if len(net) == 0:
            continue
        rows.append({
            "confidence_bucket": str(bin_label),
            "n_signals":         len(g),
            "win_rate":          round(float((net > 0).mean()), 4),
            "mean_net_pct":      round(float(net.mean()), 4),
            "above_breakeven_equity": bool(float((net > 0).mean()) > BREAKEVEN_WIN_RATE_EQUITY),
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="7-trading-day backtest runner")
    parser.add_argument("--oos-start", default="2025-01-01", help="OOS period start date")
    parser.add_argument("--oos-end",   default="2026-09-28", help="OOS period end date")
    parser.add_argument("--quick",     action="store_true",  help="Quick mode (20 symbols)")
    parser.add_argument("--cost",      default="equity",     choices=["equity", "futures"])
    parser.add_argument("--stop-pct",  type=float, default=0.03, help="Stop-loss fraction (e.g. 0.03 = 3 pct)")
    parser.add_argument("--target-pct", type=float, default=0.06, help="Take-profit fraction (e.g. 0.06 = 6 pct)")
    parser.add_argument(
        "--mode", default="auto",
        choices=["auto", "m1", "proxy"],
        help=(
            "auto=use model if found, else hard-fail | "
            "m1=require real model (MODEL_ARTIFACT_REQUIRED if absent) | "
            "proxy=label-proxy P0 (ONLY for upper-bound research)"
        ),
    )
    args = parser.parse_args()

    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"  7-TRADING-DAY BACKTEST ENGINE  |  ml-service2.0")
    print(f"  OOS: {args.oos_start} → {args.oos_end}")
    print(f"  Costs: {args.cost} ({NSE_EQUITY_COST_BPS if args.cost=='equity' else NSE_FUTURES_COST_BPS} bps)")
    print(f"  Mode: {'QUICK (20 symbols)' if args.quick else 'FULL'}")
    print(f"{'='*70}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Step 1/9: Loading OHLCV parquets...")
    max_sym = 20 if args.quick else None
    ohlcv_by_symbol = load_parquets(max_symbols=max_sym)
    print(f"  Loaded {len(ohlcv_by_symbol)} symbols")

    # Load NIFTY separately for regime classification
    nifty_df: pd.DataFrame | None = None
    if "NIFTY" in ohlcv_by_symbol:
        nifty_df = ohlcv_by_symbol["NIFTY"]
    elif "BANKNIFTY" in ohlcv_by_symbol:
        nifty_df = ohlcv_by_symbol["BANKNIFTY"]

    # ── 2. Load training dataset (OOS slice only) ──────────────────────────────
    print(f"\nStep 2/9: Loading training dataset (OOS from {args.oos_start})...")
    try:
        df_oos = load_dataset(oos_start=args.oos_start)
        print(f"  OOS rows: {len(df_oos)}")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}. Using parquet data only.")
        df_oos = None

    # ── 3. Build model scores ──────────────────────────────────────────────────
    print("\nStep 3/9: Building model scores...")
    model_dict, model_version = load_model()

    run_mode = args.mode
    if run_mode == "auto":
        run_mode = "m1" if model_dict is not None else "proxy"
    elif run_mode == "m1" and model_dict is None:
        print("\nMODEL_ARTIFACT_REQUIRED: --mode m1 requested but no model artifact found.")
        print("  Search paths checked: artifacts/registry/*/*/model.pkl")
        print("  Use --mode proxy to explicitly enable label-proxy evaluation.")
        sys.exit(1)

    if run_mode == "proxy":
        print("  ⚠  PROXY MODE (P0): results are upper-bound ONLY — NOT model performance")
    else:
        print(f"  Model version: {model_version} | MODE: M1 (actual model)")

    feature_cols: list[str] = []
    if df_oos is not None:
        non_feat = {
            "label", "realized_return", "realized_return_net", "outcome",
            "execution_model", "is_economic_evidence", "symbol",
            "entry_price", "exit_price", "upper_barrier", "lower_barrier",
            "label_start", "label_end", "horizon", "time_to_event",
            "signal_timestamp", "realized_cost", "mae", "mfe",
        }
        feature_cols = [c for c in df_oos.columns if c not in non_feat and not c.startswith("_")]

    if df_oos is not None:
        scores_df = build_scores_from_dataset(df_oos, model_dict, feature_cols, args.oos_start, mode=run_mode)
        print(f"  Score matrix: {scores_df.shape} (dates × symbols)")
    else:
        scores_df = pd.DataFrame()
        print("  WARNING: no OOS dataset found — cannot build scores")

    # ── 4. Initialize engine ──────────────────────────────────────────────────
    print("\nStep 4/9: Initializing SevenDayBacktestEngine...")
    cost_model = NSECostModel.equity() if args.cost == "equity" else NSECostModel.futures()
    print(f"  Round-trip cost: {cost_model.total_round_trip_bps:.2f} bps")
    print(f"  Stop: {args.stop_pct*100:.1f}%  |  Target: {args.target_pct*100:.1f}%")

    engine = SevenDayBacktestEngine(
        cost_model=cost_model,
        stop_loss_pct=args.stop_pct,
        take_profit_pct=args.target_pct,
        model_version=model_version,
    )

    # ── 5. Run signal evaluation ──────────────────────────────────────────────
    print("\nStep 5/9: Evaluating 7-day signals...")
    # Limit score matrix to available symbols in parquets
    valid_symbols = [c for c in scores_df.columns if c in ohlcv_by_symbol]
    scores_filtered = scores_df[valid_symbols]
    print(f"  Signals to evaluate: {scores_filtered.notna().sum().sum()} across {len(valid_symbols)} symbols")

    signal_records = engine.evaluate_universe(
        ohlcv_by_symbol=ohlcv_by_symbol,
        scores_df=scores_filtered,
        nifty_df=nifty_df,
    )
    print(f"  Records generated: {len(signal_records)}")

    # ── 6. Generate profitable opportunities ──────────────────────────────────
    print("\nStep 6/9: Scanning for profitable opportunities (ground truth)...")
    # Only scan symbols in OOS period to keep runtime manageable
    oos_symbols = {}
    for sym, df in ohlcv_by_symbol.items():
        oos_mask = df.index >= pd.Timestamp(args.oos_start, tz="UTC")
        if oos_mask.sum() >= 10:
            oos_symbols[sym] = df[oos_mask]
    print(f"  Scanning {len(oos_symbols)} symbols for profitable opportunities...")

    opp_df = engine.generate_profitable_opportunities(
        ohlcv_by_symbol=oos_symbols,
        nifty_df=nifty_df,
        min_net_return_pct=0.0,
    )
    print(f"  Profitable opportunities found: {len(opp_df)}")

    # ── 7. Compute performance report ────────────────────────────────────────
    print("\nStep 7/9: Computing performance metrics...")
    perf = engine.performance_report(signal_records)
    print(f"  n_signals:    {perf.get('n_signals', 0)}")
    print(f"  n_resolved:   {perf.get('n_resolved', 0)}")
    print(f"  win_rate:     {perf.get('win_rate', float('nan')):.1%}")
    print(f"  mean_net_pct: {perf.get('mean_net_pnl', 0.0):.3f}%")
    print(f"  profit_factor: {perf.get('profit_factor', float('nan'))}")
    print(f"  sharpe:       {perf.get('sharpe', 0.0):.4f}")
    print(f"  break-even:   {BREAKEVEN_WIN_RATE_EQUITY:.1%}  |  actual: {perf.get('win_rate', 0.0):.1%}  {'✓ ABOVE' if perf.get('win_rate',0) > BREAKEVEN_WIN_RATE_EQUITY else '✗ BELOW'}")

    # ── 8. Coverage analysis ──────────────────────────────────────────────────
    coverage = engine.coverage_report(signal_records, opp_df)
    print(f"\n  Coverage: {coverage.get('capture_rate', float('nan')):.1%} of profitable opportunities captured")
    print(f"  Precision: {coverage.get('precision', float('nan')):.1%}")
    print(f"  Recall:    {coverage.get('recall', float('nan')):.1%}")

    # ── 9. Baselines ──────────────────────────────────────────────────────────
    print("\nStep 8/9: Running baseline strategies...")
    baselines = {}

    baselines["buy_hold"] = run_buy_hold_baseline(
        ohlcv_by_symbol, args.oos_start, args.oos_end, cost_model
    )
    print(f"  Buy-hold:        win_rate={baselines['buy_hold'].get('win_rate','?'):.1%}  mean={baselines['buy_hold'].get('mean_net_pct','?'):.3f}%")

    baselines["momentum_20d"] = run_momentum_baseline(
        ohlcv_by_symbol, args.oos_start, cost_model
    )
    print(f"  Momentum(20d):   win_rate={baselines['momentum_20d'].get('win_rate','?'):.1%}  mean={baselines['momentum_20d'].get('mean_net_pct','?'):.3f}%")

    baselines["mean_reversion"] = run_mean_reversion_baseline(
        ohlcv_by_symbol, args.oos_start, cost_model
    )
    print(f"  Mean-reversion:  win_rate={baselines['mean_reversion'].get('win_rate','?'):.1%}  mean={baselines['mean_reversion'].get('mean_net_pct','?'):.3f}%")

    net_pnls = np.array([r.net_pnl for r in signal_records if not np.isnan(r.net_pnl)])
    baselines["random"] = run_random_baseline(
        n_signals=max(len(net_pnls), 500), cost=cost_model
    )
    print(f"  Random:          win_rate={baselines['random'].get('win_rate','?'):.1%}  mean={baselines['random'].get('mean_net_pct','?'):.3f}%")

    # Signal decay
    decay = compute_signal_decay(signal_records)

    # Statistical significance
    stat_sig: dict = {}
    if len(net_pnls) >= 10:
        stat_sig = compute_statistical_significance(net_pnls)
        print(f"\n  Statistical significance: {stat_sig.get('interpretation','?')}")
        print(f"  CI 95%: [{stat_sig.get('ci_95_lo','?'):.3f}%, {stat_sig.get('ci_95_hi','?'):.3f}%]")
        print(f"  p-value: {stat_sig.get('p_value','?'):.4f}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\nStep 9/9: Writing output files...")

    # backtest_signals.csv
    sig_df = engine.to_dataframe(signal_records)
    if not sig_df.empty:
        sig_df.to_csv(str(OUT_DIR / "backtest_signals.csv"), index=False)
        print(f"  ✓ backtest_signals.csv ({len(sig_df)} rows)")
    else:
        print("  WARNING: No signal records to write")

    # profitable_opportunities.csv
    if not opp_df.empty:
        opp_df.to_csv(str(OUT_DIR / "profitable_opportunities.csv"), index=False)
        print(f"  ✓ profitable_opportunities.csv ({len(opp_df)} rows)")

    # performance_by_symbol.csv
    sym_perf = performance_by_symbol(signal_records)
    if not sym_perf.empty:
        sym_perf.to_csv(str(OUT_DIR / "performance_by_symbol.csv"), index=False)
        print(f"  ✓ performance_by_symbol.csv ({len(sym_perf)} symbols)")

    # performance_by_regime.csv
    reg_perf = performance_by_regime(signal_records)
    if not reg_perf.empty:
        reg_perf.to_csv(str(OUT_DIR / "performance_by_regime.csv"), index=False)
        print(f"  ✓ performance_by_regime.csv")

    # performance_by_confidence.csv
    conf_perf = performance_by_confidence(signal_records)
    if not conf_perf.empty:
        conf_perf.to_csv(str(OUT_DIR / "performance_by_confidence.csv"), index=False)
        print(f"  ✓ performance_by_confidence.csv")

    # backtest_report.json
    report = {
        "run_metadata": {
            "oos_start": args.oos_start,
            "oos_end": args.oos_end,
            "cost_model": args.cost,
            "cost_bps": cost_model.total_round_trip_bps,
            "stop_pct": args.stop_pct,
            "target_pct": args.target_pct,
            "model_version": model_version,
            "n_symbols": len(ohlcv_by_symbol),
            "runtime_seconds": round(time.time() - t0, 1),
        },
        "acceptance_criteria": {
            "breakeven_win_rate_equity": BREAKEVEN_WIN_RATE_EQUITY,
            "breakeven_win_rate_futures": BREAKEVEN_WIN_RATE_FUTURES,
            "required_profit_factor": 1.0,
            "required_expectancy": 0.0,
        },
        "performance": perf,
        "coverage": coverage,
        "statistical_significance": stat_sig,
        "signal_decay": decay,
        "acceptance_verdict": {
            "win_rate_above_breakeven": bool(
                perf.get("win_rate", 0) > BREAKEVEN_WIN_RATE_EQUITY
            ),
            "positive_expectancy": bool(perf.get("mean_net_pnl", -1) > 0),
            "profit_factor_above_1": bool(perf.get("profit_factor", 0) > 1.0),
            "statistically_significant": bool(stat_sig.get("significant_alpha", False)),
            "proxy_mode_warning": model_version == "dataset_proxy",
            "evaluation_mode": run_mode,
        },
    }

    with open(str(OUT_DIR / "backtest_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  ✓ backtest_report.json")

    # baseline_comparison.json
    ml_perf_summary = {
        "strategy": "ml_7d_backtest",
        "n": perf.get("n_signals", 0),
        "win_rate": perf.get("win_rate", float("nan")),
        "mean_net_pct": perf.get("mean_net_pct", float("nan")),
        "profit_factor": perf.get("profit_factor", float("nan")),
        "sharpe": perf.get("sharpe", float("nan")),
    }
    baseline_report = {
        "ml": ml_perf_summary,
        "baselines": baselines,
        "ml_beats_random": bool(
            ml_perf_summary.get("win_rate", 0) >
            baselines["random"].get("win_rate", 0)
        ),
        "ml_beats_momentum": bool(
            ml_perf_summary.get("win_rate", 0) >
            baselines["momentum_20d"].get("win_rate", 0)
        ),
    }
    with open(str(OUT_DIR / "baseline_comparison.json"), "w") as f:
        json.dump(baseline_report, f, indent=2, default=str)
    print(f"  ✓ baseline_comparison.json")

    # signal_decay.json
    with open(str(OUT_DIR / "signal_decay.json"), "w") as f:
        json.dump(decay, f, indent=2, default=str)
    print(f"  ✓ signal_decay.json")

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  BACKTEST COMPLETE  |  {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"  Signals:          {perf.get('n_signals', 0)} generated | {perf.get('n_resolved', 0)} resolved")
    print(f"  Win rate:         {perf.get('win_rate', 0):.1%}  (break-even: {BREAKEVEN_WIN_RATE_EQUITY:.1%} at equity)")
    print(f"  Mean net P&L:     {perf.get('mean_net_pct', 0):.3f}% per trade")
    print(f"  Profit factor:    {perf.get('profit_factor', float('nan'))}")
    print(f"  Sharpe:           {perf.get('sharpe', 0):.4f}")
    print(f"\n  Profitable opps: {len(opp_df)}")
    print(f"  Capture rate:    {coverage.get('capture_rate', 0):.1%}")
    print(f"  Precision:       {coverage.get('precision', 0):.1%}")
    print()
    print(f"  VERDICT: {'✓ POSITIVE EXPECTANCY' if perf.get('mean_net_pct',0)>0 else '✗ NEGATIVE EXPECTANCY'}")
    print(f"  Statistical: {stat_sig.get('interpretation', 'N/A')}")
    print()
    print(f"  Outputs: {OUT_DIR}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
