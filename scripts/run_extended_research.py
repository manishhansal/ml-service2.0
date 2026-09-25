"""
scripts/run_extended_research.py — Extended research matrix (mandate §66).

Stages:
  A. Broad universe triple-barrier next_open (220 symbols)
  B. Broad universe excess-return labels next_open (vs NIFTY benchmark)
  C. Cross-sectional rank labels (rank percentile within universe at T)

Mandate compliance:
  §20 — next_open execution enforced for all economic evidence
  §35 — baseline first
  §36 — advanced challengers after
  §38 — news ablation deferred (no training samples in SentinelPulse DB yet)
  §39 — acceptance: IC > 0.02, PBO < 0.5, net_sharpe >= 0
  §54 — stop if readiness gate fails
  §64 — do not change thresholds to pass; negative results preserved
  §66 — staged matrix: A → B → C → final candidate selection
  §67 — register all experiments; report full matrix

No threshold manipulation.
No cherry-picking.
NO_ELIGIBLE_CHAMPION is a valid outcome.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/app")

from src.clients.data_service import DataServiceClient, DataServiceRateLimitedError
from src.clients.sentinel_pulse import SentinelPulseClient
from src.data.dataset_builder import DatasetBuilder
from src.data.ingestion import DataIngestionPipeline
from src.data.labels import LabelConfig, LabelFactory
from src.data.readiness import TrainingReadinessGate
from src.logging_config import get_logger
from src.registry.registry import ModelRegistry
from src.training.orchestrator import TrainingOrchestrator

import pandas as pd
import numpy as np

logger = get_logger(__name__)

DOCKER_IMAGE = os.environ.get("DOCKER_IMAGE_DIGEST", "unknown")


# ── Excess-return label builder ────────────────────────────────────────────────

def build_excess_return_labels(
    ohlcv: pd.DataFrame,
    benchmark_ohlcv: pd.DataFrame,
    horizon: int = 5,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Build labels as excess return vs benchmark (NIFTY) over horizon.

    entry = open[T+1], exit = open[T+1+horizon]
    gross_excess = stock_return - benchmark_return
    label = 1 if gross_excess > 0 else 0

    Mandate §20: uses next_open execution.
    """
    if "open" not in ohlcv.columns or "open" not in benchmark_ohlcv.columns:
        raise ValueError("Excess-return labels require 'open' column")

    stock_open = ohlcv["open"].astype(float)
    bench_open = benchmark_ohlcv["open"].astype(float).reindex(ohlcv.index, method="ffill")

    entry_stock = stock_open.shift(-1)
    exit_stock  = stock_open.shift(-(1 + horizon))
    entry_bench = bench_open.shift(-1)
    exit_bench  = bench_open.shift(-(1 + horizon))

    stock_ret = (exit_stock - entry_stock) / entry_stock
    bench_ret = (exit_bench - entry_bench) / entry_bench
    excess = stock_ret - bench_ret
    cost = cost_bps / 10_000.0

    out = pd.DataFrame(index=ohlcv.index)
    out["label_start"] = ohlcv.index
    out["signal_timestamp"] = ohlcv.index
    out["realized_return"] = excess
    out["realized_cost"] = cost
    out["realized_return_net"] = excess - cost
    out["execution_model"] = "next_open"
    out["is_economic_evidence"] = True
    out["label"] = (excess > 0).astype("Int64")
    out["outcome"] = np.where(excess > 0, "EXCESS_POSITIVE", "EXCESS_NEGATIVE")
    out.loc[excess.isna(), "label"] = pd.NA
    out.loc[excess.isna(), "outcome"] = "UNRESOLVED"
    return out


def build_crosssectional_rank_labels(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    horizon: int = 5,
    cost_bps: float = 10.0,
) -> dict[str, pd.DataFrame]:
    """Build cross-sectional rank labels: percentile rank of each stock's
    next_open forward return within the universe at each timestamp T.

    label = 1 if rank_pct >= 0.6 (top-40% outperformer)
    label = 0 if rank_pct <= 0.4 (bottom-40% underperformer)
    label = NA if rank_pct in (0.4, 0.6) — neutral zone excluded

    Mandate §18 (cross-sectional): normalization at timestamp T only.
    Mandate §20: next_open execution.
    """
    # Compute forward returns for every symbol at every timestamp
    all_returns: dict[str, pd.Series] = {}
    for sym, df in ohlcv_by_symbol.items():
        if "open" not in df.columns:
            continue
        op = df["open"].astype(float)
        fwd = (op.shift(-(1 + horizon)) - op.shift(-1)) / op.shift(-1)
        all_returns[sym] = fwd

    if not all_returns:
        return {}

    # Align all symbols on common timestamps
    ret_df = pd.DataFrame(all_returns)

    cost = cost_bps / 10_000.0
    result: dict[str, pd.DataFrame] = {}

    for sym in ret_df.columns:
        out = pd.DataFrame(index=ret_df.index)
        # Cross-sectional rank at each T (use all symbols with data at T)
        rank_pct = ret_df.rank(axis=1, pct=True)[sym]
        raw_ret = ret_df[sym]

        labels = pd.Series(pd.NA, index=ret_df.index, dtype="Int64")
        outcomes = pd.Series("NEUTRAL_ZONE", index=ret_df.index)

        top = rank_pct >= 0.6
        bot = rank_pct <= 0.4

        labels[top] = 1
        labels[bot] = 0
        labels[raw_ret.isna()] = pd.NA

        outcomes[top] = "OUTPERFORMER"
        outcomes[bot] = "UNDERPERFORMER"
        outcomes[raw_ret.isna()] = "UNRESOLVED"

        out["label_start"] = ret_df.index
        out["signal_timestamp"] = ret_df.index
        out["realized_return"] = raw_ret
        out["realized_return_net"] = raw_ret - cost
        out["realized_cost"] = cost
        out["rank_pct"] = rank_pct
        out["execution_model"] = "next_open"
        out["is_economic_evidence"] = True
        out["label"] = labels
        out["outcome"] = outcomes

        result[sym] = out

    return result


# ── Ingestion (with rate-limit patience) ──────────────────────────────────────

async def ingest_with_rate_limit_patience(
    pipeline: DataIngestionPipeline,
    symbols: list[str],
    interval: str,
    from_date: str | None,
    to_date: str | None,
    batch_size: int = 20,
    batch_sleep: float = 8.0,
) -> "IngestionResult":
    """Ingest in batches with sleep between batches to avoid data-service rate limits.

    data-service2.0: 100 req/60s. With 4 concurrent workers and rate_per_sec=5,
    we stay well below the limit but add explicit batch sleeps for large universes.
    """
    from src.data.ingestion import IngestionResult
    import asyncio

    aggregate = IngestionResult(output_dir=pipeline._root)

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        print(f"  Ingesting batch {i//batch_size + 1}/{(len(symbols)+batch_size-1)//batch_size}: "
              f"{batch[:3]}...{' (' + str(len(batch)) + ' symbols)'}",
              flush=True)

        result = await pipeline.ingest(
            symbols=batch,
            interval=interval,
            from_date=from_date,
            to_date=to_date,
            exchange="NSE",
            resume=True,
        )
        aggregate.symbols_ingested.extend(result.symbols_ingested)
        aggregate.symbols_failed.extend(result.symbols_failed)
        aggregate.reports.extend(result.reports)
        aggregate.total_rows += result.total_rows

        if i + batch_size < len(symbols):
            await asyncio.sleep(batch_sleep)

    return aggregate


# ── Dataset builder helpers ───────────────────────────────────────────────────

def load_ohlcv_with_min_bars(
    pipeline: DataIngestionPipeline,
    symbols_ingested: list[str],
    interval: str,
    min_bars: int = 252,
) -> dict[str, pd.DataFrame]:
    """Load ingested symbols, filter to those with sufficient history."""
    ohlcv_by_symbol: dict[str, pd.DataFrame] = {}
    skipped = 0
    for sym in symbols_ingested:
        try:
            df = pipeline.load_symbol(sym, interval=interval)
            if len(df) >= min_bars:
                ohlcv_by_symbol[sym] = df
            else:
                skipped += 1
        except FileNotFoundError:
            skipped += 1
    print(f"  Loaded {len(ohlcv_by_symbol)} symbols with ≥{min_bars} bars "
          f"({skipped} skipped)", flush=True)
    return ohlcv_by_symbol


# ── Stage A: Broad universe triple-barrier ────────────────────────────────────

async def run_stage_a(
    data_client: DataServiceClient,
    builder: DatasetBuilder,
    registry: ModelRegistry,
    symbols: list[str],
    interval: str,
    data_dir: Path,
) -> dict:
    """Stage A: triple-barrier next_open labels on broad universe."""
    print("\n[Stage A] Triple-barrier labels, broad universe...", flush=True)

    pipeline = DataIngestionPipeline(
        data_client=data_client,
        output_root=data_dir / interval,
        max_concurrency=3,
        rate_per_sec=4.0,
    )

    ingest = await ingest_with_rate_limit_patience(
        pipeline, symbols, interval,
        from_date="2019-01-01", to_date=None,
        batch_size=25, batch_sleep=10.0,
    )
    print(f"  Ingested {len(ingest.symbols_ingested)}/{len(symbols)} symbols, "
          f"{ingest.total_rows} total bars", flush=True)

    ohlcv = load_ohlcv_with_min_bars(pipeline, ingest.symbols_ingested, interval)
    if len(ohlcv) < 10:
        return {"stage": "A", "status": "BLOCKED", "reason": f"Only {len(ohlcv)} symbols with sufficient data"}

    label_cfg = LabelConfig(
        label_type="triple_barrier", horizon=5,
        upper_barrier_pct=0.02, lower_barrier_pct=0.02,
        cost_bps=10.0, execution_model="next_open",
    )

    try:
        meta = builder.build(ohlcv_by_symbol=ohlcv, label_config=label_cfg, timeframe=interval)
    except ValueError as e:
        return {"stage": "A", "status": "BLOCKED", "reason": str(e)}

    print(f"  Dataset: {meta.dataset_id}, {meta.row_count} rows, "
          f"{len(ohlcv)} symbols, pit={meta.pit_status}", flush=True)

    orch = TrainingOrchestrator(
        dataset_builder=builder, registry=registry,
        n_windows=5, embargo_days=10, cost_bps=10.0,
    )

    report = orch.train(
        model_name=f"stage_a_{interval}",
        dataset_id=meta.dataset_id,
        candidate_names=["logistic", "lightgbm", "xgboost"],
        register_champion=True,
    )

    print(f"  Champion={report.champion}, IC={report.champion_ic_mean:.4f}, "
          f"Sharpe={report.champion_net_sharpe:.3f}, Accepted={report.passed_acceptance}",
          flush=True)

    return {
        "stage": "A",
        "label_type": "triple_barrier",
        "execution_model": "next_open",
        "symbols": len(ohlcv),
        "rows": meta.row_count,
        "dataset_id": meta.dataset_id,
        "dataset_hash": meta.dataset_hash,
        "champion": report.champion,
        "ic_mean": report.champion_ic_mean,
        "pbo": report.champion_pbo,
        "net_sharpe": report.champion_net_sharpe,
        "passed_acceptance": report.passed_acceptance,
        "rejection_reason": report.rejection_reason,
        "candidates": [c.to_dict() for c in report.candidates],
        "is_economic_evidence": meta.is_economic_evidence,
    }


# ── Stage B: Excess-return labels ─────────────────────────────────────────────

async def run_stage_b(
    data_client: DataServiceClient,
    builder: DatasetBuilder,
    registry: ModelRegistry,
    symbols: list[str],
    interval: str,
    data_dir: Path,
) -> dict:
    """Stage B: excess-return labels vs NIFTY benchmark, broad universe."""
    print("\n[Stage B] Excess-return labels (vs NIFTY), broad universe...", flush=True)

    pipeline = DataIngestionPipeline(
        data_client=data_client,
        output_root=data_dir / interval,
        max_concurrency=3,
        rate_per_sec=4.0,
    )

    # Load existing ingested data (Stage A would have populated it)
    ohlcv = load_ohlcv_with_min_bars(pipeline, symbols, interval)
    if len(ohlcv) < 10:
        return {"stage": "B", "status": "BLOCKED", "reason": f"Only {len(ohlcv)} symbols with data"}

    # Load NIFTY as benchmark (needed for excess returns)
    try:
        nifty_df = pipeline.load_symbol("NIFTY", interval=interval)
    except FileNotFoundError:
        return {"stage": "B", "status": "BLOCKED", "reason": "NIFTY not ingested — required as benchmark"}

    # Build custom labels: excess return vs NIFTY
    from src.features.factory import FeatureFactory
    ff = FeatureFactory()

    frames: list[pd.DataFrame] = []
    for sym, df in ohlcv.items():
        if len(df) < 60:
            continue
        try:
            features, _ = ff.build(df)
            labels = build_excess_return_labels(df, nifty_df, horizon=5, cost_bps=10.0)
            merged = features.copy()
            merged["symbol"] = sym
            merged["label"] = labels["label"]
            merged["realized_return"] = labels["realized_return"]
            merged["realized_return_net"] = labels["realized_return_net"]
            merged["outcome"] = labels["outcome"]
            merged["execution_model"] = "next_open"
            merged["is_economic_evidence"] = True
            frames.append(merged)
        except Exception as e:
            logger.warning("stage_b_symbol_failed", symbol=sym, error=str(e))
            continue

    if not frames:
        return {"stage": "B", "status": "BLOCKED", "reason": "No symbols produced valid frames"}

    import pandas as pd
    import numpy as np
    import hashlib

    combined = pd.concat(frames).sort_index()
    feature_cols = ff.FEATURE_NAMES
    combined = combined.dropna(subset=["label"])
    combined = combined.dropna(subset=feature_cols)

    if combined.empty:
        return {"stage": "B", "status": "BLOCKED", "reason": "All rows dropped after NaN filtering"}

    print(f"  Dataset: {len(combined)} rows, {len(ohlcv)} symbols", flush=True)

    # Run walk-forward training directly (bypass DatasetBuilder for custom labels)
    from src.training.walk_forward import WalkForwardValidator
    from src.training.cpcv import CombinatorialPurgedCV
    from src.models.estimators import build_estimator, BASELINE_MODELS

    X = combined[feature_cols].to_numpy(dtype=float)
    y = combined["label"].to_numpy(dtype=float)
    returns = combined["realized_return"].fillna(0.0).to_numpy(dtype=float)
    ts = pd.DatetimeIndex(combined.index)

    wf = WalkForwardValidator(n_windows=5, embargo_days=10, cost_bps=10.0)
    cpcv = CombinatorialPurgedCV(n_groups=6, k_test_groups=2, embargo=10)

    candidates = []
    for name in ["logistic", "lightgbm", "xgboost"]:
        try:
            wf_r = wf.validate(X, y, returns, ts, lambda: build_estimator(name))
            cpcv_r = cpcv.run(X, y, returns, ts, lambda: build_estimator(name))
            candidates.append({
                "name": name,
                "wf_ic_mean": wf_r.ic_mean,
                "wf_net_sharpe": wf_r.net_sharpe_mean,
                "cpcv_pbo": cpcv_r.pbo,
                "accepted": (wf_r.ic_mean >= 0.02 and cpcv_r.pbo <= 0.5
                             and wf_r.net_sharpe_mean >= 0.0),
            })
            print(f"    {name}: IC={wf_r.ic_mean:.4f}, Sharpe={wf_r.net_sharpe_mean:.3f}, "
                  f"PBO={cpcv_r.pbo:.2f}", flush=True)
        except Exception as e:
            logger.warning("stage_b_candidate_failed", name=name, error=str(e))

    if not candidates:
        return {"stage": "B", "status": "BLOCKED", "reason": "No candidates trained"}

    best = max(candidates, key=lambda c: c["wf_ic_mean"])
    accepted = [c for c in candidates if c["accepted"]]

    return {
        "stage": "B",
        "label_type": "excess_return_vs_nifty",
        "execution_model": "next_open",
        "symbols": len(ohlcv),
        "rows": len(combined),
        "champion": best["name"],
        "ic_mean": best["wf_ic_mean"],
        "pbo": best["cpcv_pbo"],
        "net_sharpe": best["wf_net_sharpe"],
        "passed_acceptance": best["accepted"],
        "rejection_reason": "" if best["accepted"] else "IC_BELOW_THRESHOLD_OR_NEGATIVE_SHARPE",
        "candidates": candidates,
        "is_economic_evidence": True,
    }


# ── Stage C: Cross-sectional rank labels ──────────────────────────────────────

async def run_stage_c(
    data_client: DataServiceClient,
    builder: DatasetBuilder,
    registry: ModelRegistry,
    symbols: list[str],
    interval: str,
    data_dir: Path,
) -> dict:
    """Stage C: cross-sectional rank labels, broad universe.

    Label = 1 if stock is in top-40% of 5-day next_open returns within universe.
    Label = 0 if stock is in bottom-40%.
    Neutral zone (middle 20%) excluded.

    Mandate §18: cross-sectional normalization at T only, no future universe info.
    """
    print("\n[Stage C] Cross-sectional rank labels, broad universe...", flush=True)

    pipeline = DataIngestionPipeline(
        data_client=data_client,
        output_root=data_dir / interval,
        max_concurrency=3,
        rate_per_sec=4.0,
    )

    ohlcv = load_ohlcv_with_min_bars(pipeline, symbols, interval)
    if len(ohlcv) < 20:
        return {"stage": "C", "status": "BLOCKED",
                "reason": f"Only {len(ohlcv)} symbols — need ≥20 for meaningful cross-section"}

    print(f"  Building cross-sectional rank labels for {len(ohlcv)} symbols...", flush=True)

    cs_labels = build_crosssectional_rank_labels(ohlcv, horizon=5, cost_bps=10.0)

    from src.features.factory import FeatureFactory
    ff = FeatureFactory()

    frames = []
    for sym, df in ohlcv.items():
        if sym not in cs_labels or len(df) < 60:
            continue
        try:
            features, _ = ff.build(df)
            labels = cs_labels[sym]
            merged = features.copy()
            merged["symbol"] = sym
            merged["label"] = labels["label"]
            merged["realized_return"] = labels["realized_return"]
            merged["realized_return_net"] = labels["realized_return_net"]
            merged["outcome"] = labels["outcome"]
            merged["execution_model"] = "next_open"
            merged["is_economic_evidence"] = True
            frames.append(merged)
        except Exception as e:
            logger.warning("stage_c_symbol_failed", symbol=sym, error=str(e))

    if not frames:
        return {"stage": "C", "status": "BLOCKED", "reason": "No valid frames"}

    import pandas as pd
    combined = pd.concat(frames).sort_index()
    feature_cols = ff.FEATURE_NAMES
    combined = combined.dropna(subset=["label"])
    combined = combined.dropna(subset=feature_cols)

    if combined.empty:
        return {"stage": "C", "status": "BLOCKED", "reason": "All rows dropped after NaN filtering"}

    print(f"  Dataset: {len(combined)} rows, {len(ohlcv)} symbols", flush=True)
    print(f"  Label balance: {combined['label'].mean():.3f} positive rate", flush=True)

    from src.training.walk_forward import WalkForwardValidator
    from src.training.cpcv import CombinatorialPurgedCV
    from src.models.estimators import build_estimator

    X = combined[feature_cols].to_numpy(dtype=float)
    y = combined["label"].to_numpy(dtype=float)
    returns = combined["realized_return"].fillna(0.0).to_numpy(dtype=float)
    ts = pd.DatetimeIndex(combined.index)

    wf = WalkForwardValidator(n_windows=5, embargo_days=10, cost_bps=10.0)
    cpcv = CombinatorialPurgedCV(n_groups=6, k_test_groups=2, embargo=10)

    candidates = []
    for name in ["logistic", "lightgbm", "xgboost"]:
        try:
            wf_r = wf.validate(X, y, returns, ts, lambda: build_estimator(name))
            cpcv_r = cpcv.run(X, y, returns, ts, lambda: build_estimator(name))
            cand = {
                "name": name,
                "wf_ic_mean": wf_r.ic_mean,
                "wf_ic_worst": wf_r.ic_worst,
                "wf_positive_fraction": wf_r.positive_ic_fraction,
                "wf_net_sharpe": wf_r.net_sharpe_mean,
                "cpcv_pbo": cpcv_r.pbo,
                "accepted": (wf_r.ic_mean >= 0.02 and cpcv_r.pbo <= 0.5
                             and wf_r.net_sharpe_mean >= 0.0),
            }
            candidates.append(cand)
            print(f"    {name}: IC={wf_r.ic_mean:.4f}, Sharpe={wf_r.net_sharpe_mean:.3f}, "
                  f"PBO={cpcv_r.pbo:.2f}", flush=True)
        except Exception as e:
            logger.warning("stage_c_candidate_failed", name=name, error=str(e))

    if not candidates:
        return {"stage": "C", "status": "BLOCKED", "reason": "No candidates trained"}

    best = max(candidates, key=lambda c: c["wf_ic_mean"])

    return {
        "stage": "C",
        "label_type": "crosssectional_rank",
        "execution_model": "next_open",
        "symbols": len(ohlcv),
        "rows": len(combined),
        "champion": best["name"],
        "ic_mean": best["wf_ic_mean"],
        "pbo": best["cpcv_pbo"],
        "net_sharpe": best["wf_net_sharpe"],
        "passed_acceptance": best["accepted"],
        "rejection_reason": "" if best["accepted"] else "IC_BELOW_THRESHOLD_OR_NEGATIVE_SHARPE",
        "candidates": candidates,
        "is_economic_evidence": True,
    }


# ── Main orchestration ─────────────────────────────────────────────────────────

async def main(
    interval: str = "1d",
    output_report: Path = Path("/app/reports/extended_research.json"),
    max_symbols: int = 220,
) -> dict:
    report: dict = {
        "phase": "extended_research_matrix",
        "execution_environment": "DOCKER",
        "docker_image": DOCKER_IMAGE,
        "python_version": f"{sys.version.split()[0]}",
        "started_at": datetime.now(tz=UTC).isoformat(),
        "interval": interval,
        "max_symbols": max_symbols,
        "survivorship": "CURRENT_UNIVERSE_ONLY",
        "note": (
            "Extended research matrix: Stage A (triple-barrier broad), "
            "Stage B (excess-return vs NIFTY), "
            "Stage C (cross-sectional rank). "
            "Mandate §64: no threshold changes, negative results preserved."
        ),
    }

    data_client = DataServiceClient()
    await data_client.connect()

    sentinel_client = SentinelPulseClient()
    await sentinel_client.connect()

    # ── Readiness gate ──────────────────────────────────────────────────────────
    print("[0] Readiness gate...", flush=True)
    gate = TrainingReadinessGate()

    # Use a subset of symbols for the gate check (not all 220 to avoid rate limits)
    gate_symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "ICICIBANK"]
    readiness = await gate.check(
        data_client=data_client,
        sentinel_client=sentinel_client,
        universe=gate_symbols,
        timeframe=interval,
        min_history_days=252,
        news_required=False,
    )
    report["training_readiness"] = readiness.to_dict()

    if not readiness.training_ready:
        print(f"STOP: NOT_READY — {readiness.blockers}", flush=True)
        report["stages"] = []
        report["final"] = {"status": "NOT_RUN", "reason": str(readiness.blockers)}
        return report

    print(f"  -> {readiness.mode} (news: {readiness.news_status})", flush=True)

    # Wait for rate limit to settle after gate
    import asyncio as _a
    await _a.sleep(5.0)

    # ── Fetch full universe ────────────────────────────────────────────────────
    print("[1] Fetching F&O universe...", flush=True)
    try:
        universe_resp = await data_client.get_fno_universe()
        raw_syms = universe_resp.get("data", {}).get("symbols", []) or []
        all_symbols = [s if isinstance(s, str) else s for s in raw_syms[:max_symbols]]
        # Ensure NIFTY is in the list (needed as benchmark for Stage B)
        if "NIFTY" not in all_symbols:
            all_symbols = ["NIFTY"] + all_symbols
        print(f"  -> {len(all_symbols)} symbols", flush=True)
    except DataServiceRateLimitedError:
        await _a.sleep(15)
        try:
            universe_resp = await data_client.get_fno_universe()
            raw_syms = universe_resp.get("data", {}).get("symbols", []) or []
            all_symbols = raw_syms[:max_symbols]
        except Exception:
            all_symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "ICICIBANK",
                           "INFY", "TCS", "SBIN", "AXISBANK", "KOTAKBANK",
                           "TATAMOTORS", "MARUTI", "WIPRO", "LT", "ONGC",
                           "HINDUNILVR", "ITC", "BAJFINANCE", "BHARTIARTL", "M&M"]
            print(f"  -> Rate limited, using fallback {len(all_symbols)} symbols", flush=True)

    report["universe"] = {"symbols": all_symbols, "count": len(all_symbols)}

    # ── Shared infrastructure ──────────────────────────────────────────────────
    data_dir = Path("/app/data")
    builder = DatasetBuilder(
        output_root=Path("/app/artifacts/datasets"),
        market_source="data-service2.0",
        news_source="DISABLED",
        docker_image=DOCKER_IMAGE,
        survivorship="CURRENT_UNIVERSE_ONLY",
    )
    registry = ModelRegistry(artifacts_path=Path("/app/artifacts/registry"))

    # ── Stage A ────────────────────────────────────────────────────────────────
    stage_a = await run_stage_a(data_client, builder, registry, all_symbols, interval, data_dir)
    report["stage_a"] = stage_a

    await _a.sleep(5.0)

    # ── Stage B ────────────────────────────────────────────────────────────────
    stage_b = await run_stage_b(data_client, builder, registry, all_symbols, interval, data_dir)
    report["stage_b"] = stage_b

    await _a.sleep(5.0)

    # ── Stage C ────────────────────────────────────────────────────────────────
    stage_c = await run_stage_c(data_client, builder, registry, all_symbols, interval, data_dir)
    report["stage_c"] = stage_c

    await data_client.disconnect()
    await sentinel_client.disconnect()

    # ── Final summary ──────────────────────────────────────────────────────────
    stages = [stage_a, stage_b, stage_c]
    accepted = [s for s in stages if s.get("passed_acceptance")]
    best_stage = max(stages, key=lambda s: s.get("ic_mean", -999.0)) if stages else None

    if accepted:
        champion_stage = max(accepted, key=lambda s: s.get("ic_mean", 0))
        final_status = "ELIGIBLE_CANDIDATE"
        print(f"\n✓ Champion found in Stage {champion_stage['stage']}: "
              f"IC={champion_stage['ic_mean']:.4f}", flush=True)
    else:
        champion_stage = None
        final_status = "NO_ELIGIBLE_CHAMPION"
        print(f"\n✗ NO_ELIGIBLE_CHAMPION across all stages.", flush=True)
        if best_stage:
            print(f"  Best IC={best_stage.get('ic_mean', 0):.4f} in Stage {best_stage.get('stage')}", flush=True)

    report["final"] = {
        "status": final_status,
        "champion_stage": champion_stage["stage"] if champion_stage else None,
        "champion_ic": champion_stage["ic_mean"] if champion_stage else None,
        "best_ic_seen": best_stage.get("ic_mean") if best_stage else None,
        "best_stage": best_stage.get("stage") if best_stage else None,
        "stages_run": [s.get("stage") for s in stages],
        "stages_accepted": [s.get("stage") for s in accepted],
        "note": (
            "NO_ELIGIBLE_CHAMPION is a valid, honest outcome (mandate §64). "
            if not accepted else
            "Champion requires forward-paper validation before shadow/production eligibility."
        ),
    }
    report["completed_at"] = datetime.now(tz=UTC).isoformat()
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--max-symbols", type=int, default=220)
    parser.add_argument("--output", type=Path,
                        default=Path("/app/reports/extended_research.json"))
    args = parser.parse_args()

    result = asyncio.run(main(
        interval=args.interval,
        output_report=args.output,
        max_symbols=args.max_symbols,
    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nReport: {args.output}", flush=True)
    print(f"FINAL: {result.get('final', {}).get('status', 'UNKNOWN')}", flush=True)
