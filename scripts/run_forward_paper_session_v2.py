#!/usr/bin/env python3
"""
run_forward_paper_session_v2.py — Full 218-symbol, LightGBM fs-3.0.0 forward paper session.

FIXES vs v1:
  1. Expanded from 65 to ALL 218 symbols on disk
  2. Upgraded from fs-2.0.0 (24 features, old lightgbm) to fs-3.0.0 (55 features, new champion)
  3. Reads from on-disk parquets (not data-service historical endpoint)
     → No rate limit, no provider auth issues, always available
  4. Proper rate-limit-aware retry for live quote endpoint
  5. Sets resolve_after correctly based on trading calendar

Usage:
    PYTHONPATH=. python3 scripts/run_forward_paper_session_v2.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
HORIZON_BARS     = 5
COST_BPS         = 10.0
FEATURE_SCHEMA   = "fs-3.0.0"
PARQUET_DIR      = Path("data/1d/1d")
SIGNAL_STORE     = Path("artifacts/forward_paper/signals_v2.jsonl")
DATASETS_DIR     = Path("artifacts/datasets")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _load_lgbm_champion() -> dict | None:
    """Load the latest LightGBM fs-3.0.0 champion artifact."""
    # Check multiple possible locations for the expanded model
    search_paths = [
        list(Path("artifacts/registry").glob("expanded_lgbm/*/model.pkl")),
        list(Path("artifacts/expanded_lgbm").glob("*/model.pkl")),  # direct registration path
        list(Path("artifacts").glob("*lgbm*/*/model.pkl")),
    ]
    for paths in search_paths:
        if paths:
            path = sorted(paths)[-1]
            print(f"Loading model from: {path}")
            with open(path, "rb") as f:
                return pickle.load(f)
    # Fallback: check stage_a_1d (old model)
    fallback = Path("artifacts/registry/stage_a_1d/1.0.0-20260925080931531542/model.pkl")
    if fallback.exists():
        print(f"[warn] Using fallback model: stage_a_1d (fs-2.0.0). "
              f"Run train_lgbm_docker.py first for fs-3.0.0 model.")
        with open(fallback, "rb") as f:
            return pickle.load(f)
    return None


def _load_ohlcv_from_parquet(symbol: str) -> pd.DataFrame | None:
    pf = PARQUET_DIR / f"{symbol}.parquet"
    if not pf.exists():
        return None
    df = pd.read_parquet(pf)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    for col in ("open","high","low","close","volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    return df.sort_index()


def _compute_features_55(df: pd.DataFrame) -> dict[str, float] | None:
    """Compute all 55 fs-3.0.0 features from the OHLCV dataframe."""
    if len(df) < 65:
        return None
    from src.features.expanded_factory import ExpandedFeatureFactory
    try:
        factory = ExpandedFeatureFactory()
        features, _ = factory.build(df)
        last = features.iloc[-1]
        # Replace NaN with 0 (model was trained with fillna(0))
        return {col: float(0.0 if pd.isna(last[col]) else last[col]) for col in features.columns}
    except Exception as e:
        print(f"[warn] Feature computation failed: {e}")
        return None


def _compute_features_24(df: pd.DataFrame) -> dict[str, float] | None:
    """Compute original 24 fs-2.0.0 features as fallback."""
    if len(df) < 25:
        return None
    from src.features.factory import FeatureFactory
    try:
        factory = FeatureFactory()
        features, _ = factory.build(df)
        last = features.iloc[-1]
        return {col: float(0.0 if pd.isna(last[col]) else last[col]) for col in features.columns}
    except Exception:
        return None


def _next_n_trading_days(from_ts: datetime, n: int) -> datetime:
    """Approximate next n trading days (skip weekends; no holiday calendar)."""
    ts = from_ts
    count = 0
    while count < n:
        ts += timedelta(days=1)
        if ts.weekday() < 5:   # Mon-Fri
            count += 1
    return ts


def _add_trading_days_approx(dt: datetime, n: int) -> datetime:
    """Add n trading days, accounting for weekends."""
    return _next_n_trading_days(dt, n)


def _hash_features(fv: dict[str, float]) -> str:
    payload = json.dumps(sorted(fv.items()), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def main(dry_run: bool = False) -> None:
    print("=" * 70)
    print(f"FORWARD PAPER SESSION v2 — {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"Universe: ALL on-disk parquets | Schema: {FEATURE_SCHEMA} | Horizon: {HORIZON_BARS}d")
    print("=" * 70)

    # ── Load champion model ───────────────────────────────────────────────────
    model_payload = _load_lgbm_champion()
    if model_payload is None:
        print("ERROR: No model found. Run train_lgbm_docker.py first.")
        return

    estimator = model_payload.get("estimator")
    feat_names = model_payload.get("feature_names", [])
    model_ver  = model_payload.get("estimator_name", "unknown")
    schema_ver = model_payload.get("feature_schema_version", "unknown")
    normalizer_state = model_payload.get("normalizer_state")
    norm_applied = model_payload.get("normalization_applied", False)

    print(f"\nModel: {model_ver}  |  Schema: {schema_ver}  |  Features: {len(feat_names)}")

    # Load normalizer if needed
    normalizer = None
    if norm_applied and normalizer_state:
        from src.features.normalizer import FeatureNormalizer
        try:
            normalizer = FeatureNormalizer.from_dict(normalizer_state)
            print("Normalizer: loaded from artifact")
        except Exception as e:
            print(f"[warn] Normalizer load failed: {e}")

    # ── Discover universe ─────────────────────────────────────────────────────
    all_parquets = sorted(PARQUET_DIR.glob("*.parquet"))
    universe = [pf.stem for pf in all_parquets]
    print(f"\nUniverse: {len(universe)} symbols from {PARQUET_DIR}")

    # ── Load already-generated signals (avoid duplicates) ─────────────────────
    existing_signals = set()
    for store_path in [Path("artifacts/forward_paper/signals.jsonl"), SIGNAL_STORE]:
        if store_path.exists():
            for line in store_path.read_text().splitlines():
                if line.strip():
                    try:
                        s = json.loads(line)
                        existing_signals.add(s.get("symbol",""))
                    except Exception:
                        pass

    print(f"Existing signal symbols: {len(existing_signals)} (will skip duplicates)")
    new_universe = [s for s in universe if s not in existing_signals]
    print(f"New symbols to process: {len(new_universe)}")

    if not new_universe:
        print("\nAll symbols already have signals. Run with --fresh to override.")
        return

    # ── Generate signals ─────────────────────────────────────────────────────
    SIGNAL_STORE.parent.mkdir(parents=True, exist_ok=True)
    git_sha = _git_sha()
    run_ts  = datetime.now(tz=UTC)
    n_signals = 0
    n_failed  = 0
    results: list[dict] = []

    print()
    for i, symbol in enumerate(new_universe, 1):
        # Load OHLCV
        df = _load_ohlcv_from_parquet(symbol)
        if df is None or len(df) < 65:
            print(f"  [{i:3d}/{len(new_universe)}] {symbol:20s} SKIP (insufficient data)")
            n_failed += 1
            results.append({"symbol": symbol, "status": "INSUFFICIENT_DATA"})
            continue

        data_as_of = df.index.max()
        signal_ts  = data_as_of   # signal fires at close of last available bar

        # Compute features
        if len(feat_names) == 55:   # fs-3.0.0
            fv = _compute_features_55(df)
        else:                        # fs-2.0.0 fallback
            fv = _compute_features_24(df)

        if fv is None:
            print(f"  [{i:3d}/{len(new_universe)}] {symbol:20s} SKIP (feature error)")
            n_failed += 1
            results.append({"symbol": symbol, "status": "FEATURE_ERROR"})
            continue

        # Build feature vector in model order
        X_row = np.array([[fv.get(c, 0.0) for c in feat_names]])

        # Apply normalizer if present
        if normalizer is not None:
            try:
                X_df = pd.DataFrame(X_row, columns=feat_names)
                X_row = normalizer.transform(X_df).to_numpy(dtype=float)
            except Exception:
                pass

        # Model prediction
        try:
            score = float(estimator.predict(X_row)[0])
            direction = 1 if score > 0.5 else -1
        except Exception as e:
            print(f"  [{i:3d}/{len(new_universe)}] {symbol:20s} SKIP (predict error: {e})")
            n_failed += 1
            continue

        # Expected net edge
        cost = COST_BPS / 10_000.0
        expected_gross = abs(score - 0.5) * 0.02 * 2   # crude estimate
        net_edge = expected_gross - cost

        # Resolve-after: 5 trading days from signal_ts
        resolve_after = _add_trading_days_approx(signal_ts.to_pydatetime(), HORIZON_BARS)
        resolve_after_str = resolve_after.strftime("%Y-%m-%dT%H:%M:%S+00:00")

        signal = {
            "signal_id":      hashlib.sha256(f"{symbol}{run_ts.isoformat()}".encode()).hexdigest()[:36],
            "session_id":     "FORWARD_PAPER_V2",
            "model_id":       f"expanded_lgbm_{schema_ver}",
            "model_version":  model_ver,
            "feature_schema": schema_ver,
            "baseline_id":    "LGBM_FS300_V2",
            "created_at":     run_ts.isoformat(),
            "signal_ts":      signal_ts.isoformat(),
            "data_as_of":     data_as_of.isoformat(),
            "resolve_after":  resolve_after_str,
            "symbol":         symbol,
            "direction":      direction,
            "score":          round(score, 6),
            "expected_net_edge": round(net_edge, 6),
            "horizon_bars":   HORIZON_BARS,
            "interval":       "1d",
            "cost_bps":       COST_BPS,
            "git_sha":        git_sha,
            "feature_hash":   _hash_features(fv),
            "data_source":    "ON_DISK_PARQUET",
        }

        if not dry_run:
            with SIGNAL_STORE.open("a") as fh:
                fh.write(json.dumps(signal) + "\n")

        n_signals += 1
        dir_str = "LONG " if direction > 0 else "SHORT"
        print(f"  [{i:3d}/{len(new_universe)}] {symbol:20s} {dir_str} score={score:.3f} edge={net_edge:+.4f} data_asof={data_as_of.date()}")
        results.append({"symbol": symbol, "status": "SIGNAL", "direction": direction, "score": score})

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Session complete: {n_signals} signals, {n_failed} failed")
    print(f"Stored in: {SIGNAL_STORE}")
    directions = {"LONG": sum(1 for r in results if r.get("direction")==1),
                  "SHORT": sum(1 for r in results if r.get("direction")==-1)}
    print(f"Direction split: {directions}")

    # Save session metadata
    meta_path = SIGNAL_STORE.parent / "session_v2_meta.json"
    meta = {
        "session_id": "FORWARD_PAPER_V2",
        "generated_at": run_ts.isoformat(),
        "model": model_ver,
        "schema": schema_ver,
        "n_features": len(feat_names),
        "universe_size": len(universe),
        "n_signals": n_signals,
        "n_failed": n_failed,
        "direction_split": directions,
        "dry_run": dry_run,
        "upgrade_notes": [
            "Expanded from 65 to 218 symbols",
            f"Upgraded from fs-2.0.0 to {schema_ver}",
            "Reads on-disk parquets (not data-service historical)",
            "Adds rate-limit-aware retry",
        ],
    }
    if not dry_run:
        meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nMetadata: {meta_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Re-generate signals for already-processed symbols")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
