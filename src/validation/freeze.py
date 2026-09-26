"""
src.validation.freeze — CONFIRMATION_BASELINE_V1 manifest creation and verification.

Creates an immutable, hashed experiment manifest from the current frozen
candidate (65-symbol stage_a_1d dataset + lightgbm artifact).  Once created,
the manifest must NOT be modified.  Any run that modifies the manifest
automatically becomes EXPLORATORY_POST_HOC.

Mandate §1: freeze before changing anything.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path("artifacts/confirmation/CONFIRMATION_BASELINE_V1.json")
MANIFEST_HASH_PATH = Path("artifacts/confirmation/CONFIRMATION_BASELINE_V1.sha256")


# ── Frozen candidate definition (do NOT change) ───────────────────────────────

FROZEN_CANDIDATE: dict[str, Any] = {
    "baseline_id": "CONFIRMATION_BASELINE_V1",
    "created_at": "2026-09-25T08:14:29.617058+00:00",  # timestamp of the training run
    "frozen_at": None,  # filled at freeze time
    "git_sha": "4cf307c31face6caf3b5ad516f9388494885966a",

    # ── Model artifact ──────────────────────────────────────────────────────
    "model_name": "stage_a_1d",
    "model_version": "1.0.0-20260925080931531542",
    "model_stage": "challenger",
    "model_artifact_path": "artifacts/registry/stage_a_1d/1.0.0-20260925080931531542/model.pkl",
    "model_artifact_sha256": "97e601197c02e187e7ea2c28e0d9a4e24fc2fd41f2f8783dba4a49df4f247348",
    "model_artifact_sha256_file": "artifacts/registry/stage_a_1d/1.0.0-20260925080931531542/model.pkl.sha256",

    # ── Dataset ─────────────────────────────────────────────────────────────
    "dataset_id": "ds-1d-20260925080802-73141694",
    "dataset_sha256": "ee508cb6afccbc00db50b3cce47d3c3a790749b7a63ee43bec06ec5aefd52c4d",
    "dataset_parquet_sha256": "10f3e3253c06cbed5af46f210caf0ad6f1ef920be84d3cd41784a5ec6b2d0a31",
    "dataset_path": "artifacts/datasets/ds-1d-20260925080802-73141694/data.parquet",
    "dataset_metadata_path": "artifacts/datasets/ds-1d-20260925080802-73141694/metadata.json",
    "row_count": 127122,
    "feature_count": 24,

    # ── Feature schema ──────────────────────────────────────────────────────
    "feature_schema_version": "fs-2.0.0",
    "features": [
        "ret_1", "ret_5", "ret_10", "ret_20", "log_ret_1",
        "vol_5", "vol_10", "vol_20", "atr_14_pct", "rel_volume_20",
        "volume_zscore_20", "vwap_distance_pct", "rsi_14", "macd_hist",
        "stoch_k_14", "ema_5_20", "ema_10_50", "adx_14", "hl_range_pct",
        "close_position", "gap_pct", "bb_zscore_20", "skew_20", "kurt_20",
    ],

    # ── Label schema ────────────────────────────────────────────────────────
    "label_schema_version": "ls-2.0.0",
    "label_type": "triple_barrier",
    "label_horizon_bars": 5,
    "upper_barrier_pct": 0.02,
    "lower_barrier_pct": 0.02,
    "vol_window": 20,
    "vol_multiplier": 1.5,
    "cost_bps_label": 10.0,
    "return_threshold": 0.0,
    "execution_model": "next_open",
    "is_economic_evidence": True,

    # ── Universe ────────────────────────────────────────────────────────────
    "universe_count": 65,
    "universe_hash": "5fe2c15e8e6660c66f26c8a0d8606474654e50ac7fe55b6311371e96cb81782e",
    "survivorship": "CURRENT_UNIVERSE_ONLY",
    "survivorship_note": (
        "Universe is the current-live F&O equity set (single open effective_from). "
        "Historical constituent membership is unavailable. All results are "
        "SURVIVORSHIP_LIMITED and must not be represented as unbiased historical evidence."
    ),
    "symbols": [
        "360ONE", "ABB", "ABCAPITAL", "ADANIENSOL", "ADANIENT", "ADANIGREEN",
        "ADANIPORTS", "ADANIPOWER", "ALKEM", "AMBER", "AMBUJACEM", "ANGELONE",
        "APLAPOLLO", "APOLLOHOSP", "ASHOKLEY", "ASIANPAINT", "ASTRAL",
        "ATHERENERG", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO",
        "BAJAJFINSV", "BAJAJHLDNG", "BAJFINANCE", "BANDHANBNK", "BANKINDIA",
        "BANKNIFTY", "BHARTIARTL", "CANBK", "DMART", "DRREDDY", "HDFCBANK",
        "ICICIBANK", "INFY", "JINDALSTEL", "JIOFIN", "JSWENERGY", "KOTAKBANK",
        "MAXHEALTH", "MAZDOCK", "MCX", "MFSL", "NBCC", "NESTLEIND", "NHPC",
        "NIFTY", "NMDC", "NTPC", "PAGEIND", "PATANJALI", "PAYTM", "PERSISTENT",
        "PETRONET", "RVNL", "SAGILITY", "SAIL", "SBICARD", "SBILIFE", "SBIN",
        "SHRIRAMFIN", "SIEMENS", "SOLARINDS", "TCS", "VEDL",
    ],

    # ── Time periods ────────────────────────────────────────────────────────
    "data_start": "2021-10-08T03:45:00+00:00",
    "data_end": "2026-09-23T03:45:00+00:00",
    "training_period": "2021-10-08 to 2025-01-01 (approximate — walk-forward splits)",
    "oos_windows": 5,
    "embargo_days": 10,
    "timeframe": "1d",

    # ── Model hyperparameters (as trained) ───────────────────────────────────
    "estimator": "lightgbm",
    "calibration": "fitted (Platt/isotonic on held-out OOS tail)",
    "calibration_ece": 0.0,
    "calibration_brier": 0.208053,
    "walk_forward_ic_mean": 0.450154,  # NOTE: BARRIER ARTIFACT — see ic_artifact_note
    "walk_forward_ic_worst": 0.429871,
    "walk_forward_positive_fraction": 1.0,
    "walk_forward_net_sharpe": 3.8373,
    "cpcv_ic_mean": 0.414055,
    "cpcv_pbo": 0.0,

    # ── IC artifact warning (non-negotiable) ─────────────────────────────────
    "ic_artifact_note": (
        "The IC of 0.45 is a BARRIER ARTIFACT: 89.4% of triple-barrier realized "
        "returns are clamped at exactly ±2% (the barrier boundaries). The IC "
        "against these clamped returns is approximately equivalent to a "
        "classification accuracy IC, not a continuous-return IC. Against "
        "unclamped continuous next-open returns, the honest IC is ~0.29 (22-symbol "
        "calibration run). This must be verified on the 65-symbol dataset during "
        "the confirmation phase. The confirmation protocol requires IC measured "
        "against continuous (unclamped) next-open returns as the primary metric."
    ),

    # ── Cross-sectional contradiction ────────────────────────────────────────
    "cross_sectional_contradiction_note": (
        "The broad cross-sectional research (209 symbols, 439k rows, rank-IC primary) "
        "showed next-open rank IC ~0.02 and negative Sharpe at all cost levels. "
        "This is the OPPOSITE of the per-symbol walk-forward result. The "
        "confirmation phase must resolve this contradiction. Hypotheses: "
        "(1) the per-symbol model captures something the cross-sectional model misses; "
        "(2) the per-symbol IC is a feature-label correlation artifact from the "
        "barrier clamping; (3) the cross-sectional research used different features. "
        "The confirmation phase will test both using continuous returns on the same "
        "65-symbol universe."
    ),

    # ── Execution convention ─────────────────────────────────────────────────
    "execution_convention": "next_open (open[T+1] entry, open[T+1+h] exit)",
    "primary_cost_bps": 10.0,
    "cost_sensitivity_levels_bps": [5, 10, 15, 20, 30],

    # ── Infrastructure ───────────────────────────────────────────────────────
    "docker_image": "sha256:240bc877228980843ec7b370d6721c5a79ede0d472631790509b21328f5f53a4",
    "python_version": "3.11.16",
    "lightgbm_version": "4.5.0",
    "sklearn_version": "1.5.2",
    "market_source": "data-service2.0",
    "news_source": "DISABLED",

    # ── Random seeds ────────────────────────────────────────────────────────
    "random_seed_note": (
        "LightGBM uses random_state via its internal seed; the training orchestrator "
        "does not set a global numpy/random seed. Reproducibility is deterministic "
        "only within Docker on the same platform (Linux x86_64). "
        "The dataset hash provides the primary reproducibility anchor."
    ),

    # ── Calibrator path note ─────────────────────────────────────────────────
    "calibrator_path_note": (
        "The model pkl is a dict with keys: estimator, calibrator, feature_names. "
        "The calibrator is _IsotonicWrapper with predict_proba (not predict). "
        "IC reproduced via: est.predict_proba(X)[:,1] -> raw_proba. "
        "Calibrated: cal._model.predict(raw_proba). "
        "Uncalibrated: use raw_proba directly. "
        "IC difference between calibrated and uncalibrated: 1.49e-4 (within 1e-3 tolerance). "
        "Confirmation runner uses uncalibrated raw proba; independent repro uses calibrated. "
        "REPRODUCIBILITY: PASS (tolerance 1e-3)."
    ),

    # ── Forward paper ────────────────────────────────────────────────────────
    "forward_paper_status": "NOT_RUN",
    "forward_paper_note": (
        "No genuine signal-at-T / outcome-after-T pairs have accumulated. "
        "Infrastructure exists (src/analytics/forward_paper.py). "
        "Forward-paper is mandatory before shadow/production eligibility."
    ),

    # ── What this baseline must NOT be modified by ────────────────────────────
    "immutability_contract": (
        "This manifest is frozen. Any change constitutes EXPLORATORY_POST_HOC and "
        "must be tracked in the research trial ledger. The confirmation phase runs "
        "against this exact candidate. No hyperparameters, features, labels, "
        "universe, thresholds, or cost assumptions may be modified based on "
        "confirmation results without classifying the new run as exploratory."
    ),
}


def _hash_manifest(d: dict[str, Any]) -> str:
    payload = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def freeze() -> Path:
    """Write the frozen manifest and its hash.  Idempotent — safe to call twice."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest = dict(FROZEN_CANDIDATE)
    manifest["frozen_at"] = datetime.now(tz=UTC).isoformat()

    # If file already exists, verify the substantive content hasn't drifted.
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text())
        # Compare everything except frozen_at (timestamp changes each call)
        existing_core = {k: v for k, v in existing.items() if k != "frozen_at"}
        new_core = {k: v for k, v in manifest.items() if k != "frozen_at"}
        if existing_core == new_core:
            return MANIFEST_PATH  # already frozen — no-op
        raise RuntimeError(
            "MANIFEST DRIFT DETECTED: existing CONFIRMATION_BASELINE_V1.json "
            "differs from the current frozen candidate definition. "
            "Do NOT overwrite — investigate the source of change."
        )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))
    h = _hash_manifest(manifest)
    MANIFEST_HASH_PATH.write_text(h + "\n")
    print(f"CONFIRMATION_BASELINE_V1 frozen at {MANIFEST_PATH}")
    print(f"Manifest SHA256: {h}")
    return MANIFEST_PATH


def verify() -> dict[str, Any]:
    """Verify the frozen manifest against on-disk artifacts.

    Returns a dict with verification status for each checked component.
    Raises RuntimeError if the manifest file is missing.
    """
    if not MANIFEST_PATH.exists():
        raise RuntimeError(
            "CONFIRMATION_BASELINE_V1.json not found — run freeze() first."
        )

    manifest = json.loads(MANIFEST_PATH.read_text())
    result: dict[str, Any] = {
        "baseline_id": manifest["baseline_id"],
        "checks": {},
        "overall": "PASS",
    }

    def _check(name: str, passed: bool, detail: str) -> None:
        result["checks"][name] = {"passed": passed, "detail": detail}
        if not passed:
            result["overall"] = "FAIL"

    # 1. Model artifact SHA256
    model_path = Path(manifest["model_artifact_path"])
    if model_path.exists():
        actual_sha = subprocess.check_output(
            ["shasum", "-a", "256", str(model_path)]
        ).decode().split()[0]
        expected_sha = manifest["model_artifact_sha256"]
        _check(
            "model_sha256",
            actual_sha == expected_sha,
            f"expected={expected_sha} actual={actual_sha}",
        )
    else:
        _check("model_sha256", False, f"artifact missing: {model_path}")

    # 2. Dataset parquet SHA256
    ds_path = Path(manifest["dataset_path"])
    if ds_path.exists():
        actual_ds = subprocess.check_output(
            ["shasum", "-a", "256", str(ds_path)]
        ).decode().split()[0]
        expected_ds = manifest["dataset_parquet_sha256"]
        _check(
            "dataset_parquet_sha256",
            actual_ds == expected_ds,
            f"expected={expected_ds} actual={actual_ds}",
        )
    else:
        _check("dataset_parquet_sha256", False, f"parquet missing: {ds_path}")

    # 3. Dataset metadata hash
    meta_path = Path(manifest["dataset_metadata_path"])
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        _check(
            "dataset_metadata_hash",
            meta.get("dataset_hash") == manifest["dataset_sha256"],
            f"expected={manifest['dataset_sha256']} actual={meta.get('dataset_hash')}",
        )
    else:
        _check("dataset_metadata_hash", False, f"metadata missing: {meta_path}")

    # 4. Row count
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        _check(
            "row_count",
            meta.get("row_count") == manifest["row_count"],
            f"expected={manifest['row_count']} actual={meta.get('row_count')}",
        )

    # 5. Feature schema version
    if meta_path.exists():
        _check(
            "feature_schema_version",
            meta.get("feature_schema_version") == manifest["feature_schema_version"],
            f"expected={manifest['feature_schema_version']}",
        )

    # 6. Manifest self-hash
    if MANIFEST_HASH_PATH.exists():
        stored_hash = MANIFEST_HASH_PATH.read_text().strip()
        current_hash = _hash_manifest(manifest)
        _check(
            "manifest_self_hash",
            stored_hash == current_hash,
            f"stored={stored_hash} current={current_hash}",
        )
    else:
        _check("manifest_self_hash", False, "hash file missing")

    return result


if __name__ == "__main__":
    freeze()
    result = verify()
    print(json.dumps(result, indent=2))
