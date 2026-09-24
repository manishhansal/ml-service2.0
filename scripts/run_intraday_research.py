"""
scripts/run_intraday_research.py — real intraday alpha research (mandate §8, §12, §17, §29 Stage 3).

The binding negative baseline was established on DAILY bars only. This runner asks
one honest, previously-untested question on REAL intraday data pulled from
data-service2.0:

    Does a walk-forward, PIT-safe model produce out-of-sample IC on 15m
    (and optionally 5m) NSE bars that the daily configuration did not?

It reuses the SAME PIT-safe pipeline as the daily certification (DatasetBuilder →
feature factory → triple-barrier labels → TrainingOrchestrator walk-forward OOS IC
+ CPCV PBO), so the intraday result is directly comparable to the daily baseline
and cannot smuggle in a different (weaker) validation regime.

Honesty rules enforced here:
  - REAL DATA ONLY. If data-service2.0 is unavailable / returns no usable series,
    the run FAILS (exit 2). No synthetic fallback (mandate §43).
  - The proxy backtest is NOT used; only walk-forward OOS IC / CPCV PBO decide the
    research state (mandate §2.1).
  - The result is classified into an explicit research state (mandate §46), and a
    NEGATIVE result is a valid, reported outcome (mandate §47).

Usage:
    PYTHONPATH=. python3 scripts/run_intraday_research.py \
        [--symbols NIFTY,BANKNIFTY,RELIANCE] [--interval 15m] \
        [--horizon 6] [--out reports/intraday_research.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
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
os.environ.setdefault("ML_SERVICE_API_KEY", "intraday-research-key")
os.environ.setdefault("DATA_SERVICE_API_KEY", "intraday-research-key")
os.environ.setdefault("DATA_SERVICE_2_URL", "http://localhost:8200")
os.environ.setdefault("SENTINEL_PULSE_URL", "http://localhost:3001")

import pandas as pd

from src.data.dataset_builder import DatasetBuilder
from src.data.labels import LabelConfig
from src.registry.registry import ModelRegistry
from src.training.orchestrator import TrainingOrchestrator


class IntradayDataUnavailableError(RuntimeError):
    """Raised when real intraday data cannot be sourced (no synthetic fallback)."""


# Research-state taxonomy (mandate §46).
class ResearchState:
    NO_SIGNAL = "NO_SIGNAL"
    WEAK_SIGNAL = "WEAK_SIGNAL"
    STATISTICALLY_INTERESTING = "STATISTICALLY_INTERESTING"
    ECONOMICALLY_UNVIABLE = "ECONOMICALLY_UNVIABLE"
    COST_SURVIVING_RESEARCH_CANDIDATE = "COST_SURVIVING_RESEARCH_CANDIDATE"
    NOT_RUN = "NOT_RUN"


MIN_IC = 0.02          # same acceptance IC threshold as daily (mandate §50: not moved)
MAX_PBO = 0.5


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


async def _fetch_intraday(
    symbols: list[str], interval: str, days: int
) -> dict[str, pd.DataFrame]:
    from src.clients.data_service import DataServiceClient

    client = DataServiceClient()
    await client.connect()
    try:
        to_date = datetime.now(tz=UTC).date().isoformat()
        from_date = (datetime.now(tz=UTC) - timedelta(days=days)).date().isoformat()
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            bars = await client.get_historical_ohlcv(
                sym, interval=interval, from_date=from_date, to_date=to_date
            )
            if not bars or len(bars) < 300:
                continue
            df = pd.DataFrame(bars)
            df.columns = [c.lower() for c in df.columns]
            # Normalise the epoch 'time' column to a UTC DatetimeIndex.
            if "time" in df.columns:
                df.index = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            out[sym] = df
        return out
    finally:
        await client.disconnect()


def _classify(ic_mean: float, pbo: float, net_sharpe: float) -> str:
    """Classify a research result into an explicit state (mandate §46, §48).

    The economic dimension is decisive: a statistically interesting signal that
    does NOT survive realistic costs (net Sharpe < 0) is ECONOMICALLY_UNVIABLE,
    NOT a research-leading candidate. This prevents over-promoting a thin edge
    that turnover destroys (mandate §14, §19, §65-F).
    """
    if ic_mean < MIN_IC:
        return ResearchState.NO_SIGNAL
    if pbo > MAX_PBO:
        return ResearchState.WEAK_SIGNAL
    # Positive IC + acceptable PBO, but the cost gate is the final arbiter.
    if net_sharpe < 0.0:
        return ResearchState.ECONOMICALLY_UNVIABLE
    return ResearchState.COST_SURVIVING_RESEARCH_CANDIDATE


def run(symbols: list[str], interval: str, horizon: int, out_path: Path, days: int) -> dict:
    root = Path("./artifacts/intraday_research")
    root.mkdir(parents=True, exist_ok=True)

    ohlcv = asyncio.run(_fetch_intraday(symbols, interval, days))
    if not ohlcv:
        raise IntradayDataUnavailableError(
            f"data-service2.0 returned no usable {interval} series for {symbols}; "
            "no synthetic fallback (mandate §43)."
        )
    print(f"[intraday] interval={interval} symbols={list(ohlcv)} "
          f"bars={{ {', '.join(f'{k}:{len(v)}' for k, v in ohlcv.items())} }}")

    builder = DatasetBuilder(output_root=root / "datasets", run_leakage_validation=True)
    meta = builder.build(
        ohlcv,
        LabelConfig(label_type="triple_barrier", horizon=horizon, cost_bps=10),
        timeframe=interval,
    )
    print(f"[intraday] dataset rows={meta.row_count} pit={meta.pit_status} "
          f"leakage_validated={meta.leakage_validated}")

    registry = ModelRegistry(artifacts_path=root / "artifacts")
    # More OOS windows are affordable intraday given the larger sample.
    orch = TrainingOrchestrator(
        builder, registry, n_windows=6, embargo_days=1, cost_bps=10,
        min_ic=MIN_IC, max_pbo=MAX_PBO,
    )
    report = orch.train(
        f"intraday_{interval}", meta.dataset_id,
        candidate_names=["logistic", "lightgbm", "xgboost"],
        register_champion=False,   # research only; never auto-register a champion
    )

    state = _classify(
        report.champion_ic_mean, report.champion_pbo, report.champion_net_sharpe
    )
    evidence = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_sha(),
        "data_source_class": "HISTORICAL_REAL",
        "interval": interval,
        "label_horizon_bars": horizon,
        "dataset": {
            "dataset_id": meta.dataset_id,
            "dataset_hash": meta.dataset_hash,
            "rows": meta.row_count,
            "universe": meta.universe,
            "timeframe": meta.timeframe,
            "date_start": meta.date_start,
            "date_end": meta.date_end,
            "pit_status": meta.pit_status,
            "leakage_validated": meta.leakage_validated,
            "feature_schema_version": meta.feature_schema_version,
            "label_schema_version": meta.label_schema_version,
        },
        "training": report.to_dict(),
        "research_state": state,
        "acceptance": {
            "min_ic": MIN_IC, "max_pbo": MAX_PBO,
            "champion_ic_mean": report.champion_ic_mean,
            "champion_pbo": report.champion_pbo,
            "champion_net_sharpe": report.champion_net_sharpe,
            "passed_acceptance": report.passed_acceptance,
            "rejection_reason": report.rejection_reason,
        },
        "note": (
            "Real intraday walk-forward OOS research using the SAME PIT-safe "
            "pipeline as the daily certification. Proxy backtest NOT used; only "
            "walk-forward OOS IC / CPCV PBO decide the research state. A negative "
            "result (NO_SIGNAL) is a valid, reported outcome (mandate §47)."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evidence, indent=2, default=str))
    print(f"[intraday] evidence written to {out_path}")
    print(f"[intraday] champion={report.champion} ic_mean={report.champion_ic_mean} "
          f"pbo={report.champion_pbo} net_sharpe={report.champion_net_sharpe}")
    print(f"[intraday] RESEARCH_STATE={state}")
    return evidence


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="NIFTY,BANKNIFTY,RELIANCE")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--horizon", type=int, default=6, help="label horizon in bars")
    ap.add_argument("--days", type=int, default=900)
    ap.add_argument("--out", default="reports/intraday_research.json")
    args = ap.parse_args()
    try:
        run(args.symbols.split(","), args.interval, args.horizon, Path(args.out), args.days)
    except IntradayDataUnavailableError as exc:
        print(f"[intraday][FAIL] {exc}", file=sys.stderr)
        sys.exit(2)
