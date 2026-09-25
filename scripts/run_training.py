"""
scripts/run_training.py — Docker-based ML training pipeline (mandate §2, §55).

Stages:
  1. Training-readiness gate (STOP if NOT_READY)
  2. Ingest real OHLCV data from data-service2.0 (market-only, no news)
  3. Build validated dataset (next_open execution, PIT-safe)
  4. Train Stage A baseline: logistic
  5. Train Stage A advanced: LightGBM, XGBoost
  6. Champion selection (OOS IC + parsimony)
  7. Register artifacts
  8. Report results

Mandate compliance:
  §2  — all execution inside Docker
  §20 — next_open execution model enforced
  §35 — baseline first
  §36 — advanced only where appropriate
  §39 — acceptance requires IC > threshold AND PBO < threshold AND net_sharpe >= 0
  §42 — every artifact gets provenance metadata
  §43 — never auto-promote to production
  §54 — training STOPS if readiness gate fails
  §55 — train ONLY after readiness confirmed
  §65 — current_universe_only survivorship disclosed

Usage (inside Docker):
    python3 scripts/run_training.py \
        --timeframe 1d \
        --min-history 252 \
        --symbols NIFTY BANKNIFTY RELIANCE \
        --output-report /app/reports/training_run.json

All execution must happen inside Docker (mandate §2).
Never call this script directly on the host.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/app")

from src.clients.data_service import DataServiceClient, DataServiceUnavailableError
from src.clients.sentinel_pulse import SentinelPulseClient
from src.data.dataset_builder import DatasetBuilder
from src.data.ingestion import DataIngestionPipeline
from src.data.labels import LabelConfig
from src.data.readiness import TrainingReadinessGate
from src.logging_config import get_logger
from src.training.orchestrator import TrainingOrchestrator
from src.registry.registry import ModelRegistry

logger = get_logger(__name__)


def _docker_image() -> str:
    return os.environ.get(
        "DOCKER_IMAGE_DIGEST",
        os.environ.get("IMAGE_DIGEST", "unknown"),
    )


async def run_training(
    timeframe: str,
    symbols: list[str],
    min_history_days: int,
    output_report: Path,
) -> dict:
    """Full training pipeline. Stops on readiness failure (mandate §54)."""

    report: dict = {
        "phase": "data_hardening_docker_training_alphaforge_integration",
        "execution_environment": "DOCKER",
        "docker_image": _docker_image(),
        "python_version": f"{sys.version.split()[0]}",
        "started_at": datetime.now(tz=UTC).isoformat(),
        "timeframe": timeframe,
        "symbols_requested": symbols,
        "survivorship": "CURRENT_UNIVERSE_ONLY",
        "execution_model": "next_open",
    }

    # ── Step 1: Readiness gate ───────────────────────────────────────────────
    print("\n[1/7] Running training-readiness gate...", flush=True)

    data_client = DataServiceClient()
    await data_client.connect()
    sentinel_client = SentinelPulseClient()
    await sentinel_client.connect()

    gate = TrainingReadinessGate()
    readiness = await gate.check(
        data_client=data_client,
        sentinel_client=sentinel_client,
        universe=symbols,
        timeframe=timeframe,
        min_history_days=min_history_days,
        news_required=False,
    )
    report["training_readiness"] = readiness.to_dict()

    if not readiness.training_ready:
        msg = f"TRAINING BLOCKED — NOT_READY. Blockers: {readiness.blockers}"
        print(f"\nSTOP: {msg}", flush=True)
        report["training"] = {"status": "BLOCKED", "reason": readiness.blockers}
        report["promotion"] = {
            "status": "NOT_ELIGIBLE",
            "reason": "Training did not execute — readiness gate failed",
        }
        return report

    print(f"  -> {readiness.mode} (news: {readiness.news_status})", flush=True)

    # ── Step 2: Ingest real OHLCV ─────────────────────────────────────────────
    print(f"\n[2/7] Ingesting real OHLCV data for {len(symbols)} symbols @ {timeframe}...", flush=True)

    output_dir = Path("/app/data") / timeframe
    pipeline = DataIngestionPipeline(
        data_client=data_client,
        output_root=output_dir,
        max_concurrency=4,
        rate_per_sec=5.0,
    )
    ingest_result = await pipeline.ingest(
        symbols=symbols,
        interval=timeframe,
        exchange="NSE",
        resume=True,
    )

    print(
        f"  -> Ingested: {len(ingest_result.symbols_ingested)} symbols, "
        f"{ingest_result.total_rows} total bars, "
        f"failed: {len(ingest_result.symbols_failed)}",
        flush=True,
    )

    if ingest_result.symbols_failed:
        print(f"  -> Failed symbols: {ingest_result.symbols_failed[:10]}", flush=True)

    report["dataset"] = {
        "ingestion": ingest_result.to_dict(),
        "execution_model": "next_open",
        "market_source": "data-service2.0",
        "news_source": "DISABLED",
    }

    # ── Step 3: Build validated dataset ──────────────────────────────────────
    print("\n[3/7] Building validated training dataset...", flush=True)

    ohlcv_by_symbol: dict = {}
    for sym in ingest_result.symbols_ingested:
        try:
            df = pipeline.load_symbol(sym, interval=timeframe)
            if len(df) >= 60:
                ohlcv_by_symbol[sym] = df
            else:
                print(f"  -> Skip {sym}: only {len(df)} bars", flush=True)
        except FileNotFoundError:
            pass

    if len(ohlcv_by_symbol) < 3:
        msg = f"Insufficient symbols with data: {len(ohlcv_by_symbol)} (need >= 3)"
        print(f"\nSTOP: {msg}", flush=True)
        report["training"] = {"status": "BLOCKED", "reason": msg}
        report["promotion"] = {"status": "NOT_ELIGIBLE", "reason": msg}
        return report

    print(f"  -> {len(ohlcv_by_symbol)} symbols with sufficient data", flush=True)

    builder = DatasetBuilder(
        output_root=Path("/app/artifacts/datasets"),
        market_source="data-service2.0",
        news_source="DISABLED",
        docker_image=_docker_image(),
        survivorship="CURRENT_UNIVERSE_ONLY",
    )

    label_cfg = LabelConfig(
        label_type="triple_barrier",
        horizon=5,
        upper_barrier_pct=0.02,
        lower_barrier_pct=0.02,
        cost_bps=10.0,
        execution_model="next_open",  # mandate §20
    )

    try:
        meta = builder.build(
            ohlcv_by_symbol=ohlcv_by_symbol,
            label_config=label_cfg,
            timeframe=timeframe,
        )
    except ValueError as e:
        msg = f"Dataset build failed: {e}"
        print(f"\nSTOP: {msg}", flush=True)
        report["training"] = {"status": "BLOCKED", "reason": msg}
        report["promotion"] = {"status": "NOT_ELIGIBLE", "reason": msg}
        return report

    print(
        f"  -> Dataset: {meta.dataset_id}, {meta.row_count} rows, "
        f"leakage={meta.leakage_validated}, pit={meta.pit_status}",
        flush=True,
    )
    print(f"  -> execution_model: {meta.execution_model}", flush=True)
    print(f"  -> is_economic_evidence: {meta.is_economic_evidence}", flush=True)

    report["dataset"].update(meta.to_dict())

    # Block if leakage detected
    if not meta.leakage_validated:
        msg = "Catastrophic leakage detected in dataset — training blocked"
        print(f"\nSTOP: {msg}", flush=True)
        report["training"] = {"status": "BLOCKED", "reason": msg}
        report["promotion"] = {"status": "NOT_ELIGIBLE", "reason": msg}
        return report

    # ── Step 4-6: Train baseline first, then challengers ─────────────────────
    print("\n[4/7] Training Stage A — baseline (logistic)...", flush=True)

    registry = ModelRegistry(artifacts_path=Path("/app/artifacts/registry"))
    orchestrator = TrainingOrchestrator(
        dataset_builder=builder,
        registry=registry,
        n_windows=5,
        embargo_days=10,
        cost_bps=10.0,
        min_ic=0.02,
        max_pbo=0.5,
        parsimony_margin=0.005,
    )

    # Stage A: Baseline only first
    baseline_report = orchestrator.train(
        model_name=f"market_regime_{timeframe}",
        dataset_id=meta.dataset_id,
        candidate_names=["logistic"],
        register_champion=False,
    )
    print(
        f"  -> Baseline IC={baseline_report.champion_ic_mean:.4f}, "
        f"PBO={baseline_report.champion_pbo:.3f}, "
        f"Sharpe={baseline_report.champion_net_sharpe:.3f}, "
        f"Accepted={baseline_report.passed_acceptance}",
        flush=True,
    )

    print("\n[5/7] Training Stage A challengers (LightGBM, XGBoost)...", flush=True)
    full_report = orchestrator.train(
        model_name=f"market_regime_{timeframe}",
        dataset_id=meta.dataset_id,
        candidate_names=["logistic", "lightgbm", "xgboost"],
        register_champion=True,
    )
    print(
        f"  -> Champion={full_report.champion}, "
        f"IC={full_report.champion_ic_mean:.4f}, "
        f"PBO={full_report.champion_pbo:.3f}, "
        f"Sharpe={full_report.champion_net_sharpe:.3f}, "
        f"Accepted={full_report.passed_acceptance}",
        flush=True,
    )

    report["training"] = full_report.to_dict()

    # ── Step 7: Final lifecycle determination ────────────────────────────────
    print("\n[6/7] Determining lifecycle eligibility...", flush=True)

    if not full_report.passed_acceptance:
        lifecycle = "NO_ELIGIBLE_CHAMPION"
        lifecycle_reason = full_report.rejection_reason
    elif not meta.is_economic_evidence:
        lifecycle = "ECONOMICALLY_UNVIABLE"
        lifecycle_reason = "Labels use close_to_close execution model — not executable alpha"
    else:
        lifecycle = "RESEARCH_READY"
        lifecycle_reason = (
            "Model passed OOS IC/PBO/Sharpe gates on next_open labels. "
            "Requires forward-paper validation before shadow/production eligibility."
        )

    report["promotion"] = {
        "status": lifecycle,
        "reason": lifecycle_reason,
        "champion_version": full_report.champion_version,
        "passed_acceptance": full_report.passed_acceptance,
        "ic_mean": full_report.champion_ic_mean,
        "pbo": full_report.champion_pbo,
        "net_sharpe": full_report.champion_net_sharpe,
        "ece": full_report.champion_ece,
    }

    print(f"  -> Lifecycle: {lifecycle}", flush=True)
    print(f"  -> Reason: {lifecycle_reason}", flush=True)

    print("\n[7/7] Writing report...", flush=True)
    report["completed_at"] = datetime.now(tz=UTC).isoformat()

    await data_client.disconnect()
    await sentinel_client.disconnect()

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docker ML training pipeline")
    parser.add_argument("--timeframe", default="1d", choices=["1d", "15m", "5m", "1h"])
    parser.add_argument("--min-history", type=int, default=252)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "ICICIBANK",
                 "INFY", "TCS", "SBIN", "AXISBANK", "KOTAKBANK"],
        help="Symbols to train on (default: 10 core F&O symbols)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("/app/reports/training_run_docker.json"),
    )
    args = parser.parse_args()

    result = asyncio.run(
        run_training(
            timeframe=args.timeframe,
            symbols=args.symbols,
            min_history_days=args.min_history,
            output_report=args.output_report,
        )
    )

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nReport saved to: {args.output_report}", flush=True)
    print(
        f"\nFINAL RESULT: {result.get('promotion', {}).get('status', 'UNKNOWN')}",
        flush=True,
    )

    rc = 0 if result.get("promotion", {}).get("status") != "NOT_ELIGIBLE" else 1
    sys.exit(rc)
