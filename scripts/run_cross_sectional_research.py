"""
scripts/run_cross_sectional_research.py — broad cross-sectional F&O alpha research
(mandate §17-§41, §68-§69, §72, §91 RUN A-G, §92, §107).

Answers the central unanswered question (§108):

    Does a broad, liquid, PIT-safe F&O equity universe reveal cross-sectional
    alpha that survives turnover, realistic costs, and OOS validation?

Pipeline (all REAL data from data-service2.0; no synthetic fallback, §97):

    fno-universe  →  per-symbol REAL OHLCV  →  cross-sectional panel
      →  labels {raw, excess, residual, rank}  ×  models {logistic, ridge,
         lightgbm, xgboost}  →  walk-forward per-timestamp cross-sectional IC
         + decile long-short / long-only portfolios + turnover + net Sharpe
      →  persisted OOS predictions  →  independent metric recomputation
      →  honest classification (§69)

Every experiment gets an experiment_id and is recorded (§38); failed / negative
experiments are NOT deleted and ALL are reported (§101). Thresholds are the SAME
as the daily/intraday certification (min_ic=0.02, max_pbo=0.5); NOT tuned to
pass (§67).

Usage:
    PYTHONPATH=. python3 scripts/run_cross_sectional_research.py \
        [--interval 1d] [--horizon 1] [--max-symbols 0] \
        [--labels raw,excess,residual,rank] \
        [--models logistic,ridge,lightgbm,xgboost] \
        [--days 1200] [--out reports/cross_sectional_research.json]

Exit 2 if the universe / data cannot be sourced (no synthetic fallback).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

# ── Env bootstrap (real creds win) + OpenMP safety on macOS ────────────────
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
os.environ.setdefault("DATA_SERVICE_2_URL", "http://localhost:8200")
os.environ.setdefault("ML_SERVICE_API_KEY", "xs-research-key")
os.environ.setdefault("DATA_SERVICE_API_KEY", "xs-research-key")
# Prevent duplicate-OpenMP-runtime segfaults (lightgbm/xgboost/sklearn/numba).
for _k, _v in {
    "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1",
    "OMP_MAX_ACTIVE_LEVELS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd

from src.analytics.cross_sectional import (
    CrossSectionalValidator,
    PanelConfig,
    build_cross_sectional_panel,
)
from src.analytics.independent_metrics import compare, pearson_ic, rank_ic
from src.features.factory import FeatureFactory
from src.models.estimators import build_estimator

MIN_IC = 0.02      # same acceptance IC as daily/intraday (mandate §67: not moved)
MAX_PBO = 0.5

# The TRAINING target per label family. The OOS SCORE is evaluated against the
# economically meaningful forward return (residual for pure alpha IC; raw for
# portfolio P&L), kept separate from the training target.
LABEL_TARGET = {
    "raw": "fwd_return_raw",
    "excess": "fwd_return_excess",
    "residual": "fwd_return_residual",
    "rank": "fwd_return_rank",
}
# Realized column used to SCORE each label (mandate §23: report cross-sectional
# alpha on the residual; portfolio P&L uses raw return separately).
LABEL_REALIZED = {
    "raw": "fwd_return_raw",
    "excess": "fwd_return_excess",
    "residual": "fwd_return_residual",
    "rank": "fwd_return_residual",
}

# Feature groups (mandate §20): start with BASE (the 24), then relative overlay.
BASE_FEATURES = list(FeatureFactory.FEATURE_NAMES)
XS_FEATURES = [
    "xs_ret5_demean", "xs_ret20_demean", "xs_ret5_rank", "xs_ret20_rank",
    "xs_relvol", "xs_ret5_z",
]


class CrossSectionalDataUnavailableError(RuntimeError):
    """Raised when real data cannot be sourced (no synthetic fallback, §97)."""


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


_OHLCV_CACHE_ROOT = Path("./artifacts/cross_sectional/ohlcv_cache")


def _cache_path(interval: str, days: int) -> Path:
    return _OHLCV_CACHE_ROOT / f"universe_{interval}_{days}d.parquet"


def _load_cache(interval: str, days: int) -> tuple[dict[str, pd.DataFrame], dict] | None:
    """Load a previously-fetched REAL universe snapshot from local cache.

    The cache stores ONLY real bars already retrieved from data-service2.0; it
    is a performance cache, never a synthetic substitute (§97). Invalidated by
    changing interval/days (encoded in the filename).
    """
    p = _cache_path(interval, days)
    meta_p = p.with_suffix(".meta.json")
    if not (p.exists() and meta_p.exists()):
        return None
    try:
        flat = pd.read_parquet(p)
        meta = json.loads(meta_p.read_text())
        out: dict[str, pd.DataFrame] = {}
        for sym, g in flat.groupby("symbol"):
            df = g.drop(columns=["symbol"]).set_index("ts").sort_index()
            df.index = pd.to_datetime(df.index, utc=True)
            out[sym] = df[["open", "high", "low", "close", "volume"]].astype(float)
        meta["from_cache"] = True
        return out, meta
    except Exception:
        return None


def _save_cache(interval: str, days: int, ohlcv: dict[str, pd.DataFrame], meta: dict) -> None:
    p = _cache_path(interval, days)
    p.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for sym, df in ohlcv.items():
        f = df.copy()
        f["symbol"] = sym
        f = f.reset_index().rename(columns={f.columns[0]: "ts"})
        frames.append(f)
    if frames:
        pd.concat(frames).to_parquet(p)
        p.with_suffix(".meta.json").write_text(json.dumps(meta, default=str))


async def _fetch_universe_ohlcv(
    interval: str, days: int, max_symbols: int, min_bars: int,
    use_cache: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict]:
    from src.clients.data_service import (
        DataServiceClient,
        DataServiceUnavailableError,
        LowDataConfidenceError,
        SignalEngineNotAllowedError,
    )

    if use_cache and not max_symbols:
        cached = _load_cache(interval, days)
        if cached is not None:
            return cached

    client = DataServiceClient()
    await client.connect()
    try:
        try:
            uni = await client.get_fno_universe()
        except DataServiceUnavailableError as exc:
            raise CrossSectionalDataUnavailableError(
                f"fno-universe unavailable: {exc}; no synthetic fallback (§97)."
            ) from exc

        data = uni.get("data", uni)
        symbols: list[str] = list(data.get("symbols", []))
        availability = data.get("availability") or data.get("status")
        meta_list = list(data.get("instrumentMetadata", []))
        if not symbols:
            raise CrossSectionalDataUnavailableError(
                f"fno-universe availability={availability} with 0 symbols (§9, §97)."
            )
        if max_symbols and max_symbols > 0:
            symbols = symbols[:max_symbols]

        to_date = datetime.now(tz=UTC).date().isoformat()
        from_date = (datetime.now(tz=UTC) - timedelta(days=days)).date().isoformat()

        out: dict[str, pd.DataFrame] = {}
        sem = asyncio.Semaphore(6)

        async def fetch(sym: str) -> None:
            async with sem:
                try:
                    bars = await client.get_historical_ohlcv(
                        sym, interval=interval, from_date=from_date, to_date=to_date
                    )
                except (SignalEngineNotAllowedError, LowDataConfidenceError,
                        DataServiceUnavailableError):
                    return
                except Exception:
                    return
                if not bars or len(bars) < min_bars:
                    return
                df = pd.DataFrame(bars)
                df.columns = [c.lower() for c in df.columns]
                if "time" not in df.columns:
                    return
                df.index = pd.to_datetime(df["time"], unit="s", utc=True)
                cols = ["open", "high", "low", "close", "volume"]
                if not set(cols).issubset(df.columns):
                    return
                df = df[cols].astype(float).sort_index()
                df = df[~df.index.duplicated(keep="last")]
                out[sym] = df

        await asyncio.gather(*(fetch(s) for s in symbols))

        universe_meta = {
            "availability": availability,
            "reported_symbols": len(data.get("symbols", [])),
            "requested_symbols": len(symbols),
            "fetched_symbols": len(out),
            "survivorship_classification": "CURRENT_UNIVERSE_ONLY",
            "survivorship_note": (
                "Membership is a current-universe bootstrap (single open "
                "effective_from); results are SURVIVORSHIP_LIMITED and must not "
                "be represented as unbiased historical evidence (§11, §54)."
            ),
            "from_cache": False,
        }
        if use_cache and not max_symbols and out:
            try:
                _save_cache(interval, days, out, universe_meta)
            except Exception:
                pass
        return out, universe_meta
    finally:
        await client.disconnect()


def _dataset_hash(panel: pd.DataFrame, cols: list[str]) -> str:
    arr = np.ascontiguousarray(panel[cols].fillna(0.0).to_numpy(dtype=float))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _independent_recompute(preds: pd.DataFrame) -> dict:
    """Recompute per-timestamp cross-sectional IC from persisted predictions,
    independently of the validator (mandate §37, §50, §79)."""
    if preds.empty:
        return {"n_predictions": 0, "note": "no predictions persisted"}
    per_ts_ic, per_ts_rank = [], []
    for _ts, grp in preds.groupby("ts"):
        if len(grp) < 5 or grp["score"].std() < 1e-12:
            continue
        per_ts_ic.append(pearson_ic(grp["score"].tolist(), grp["realized"].tolist()))
        per_ts_rank.append(rank_ic(grp["score"].tolist(), grp["realized"].tolist()))
    return {
        "n_predictions": int(len(preds)),
        "n_scored_timestamps": len(per_ts_ic),
        "independent_mean_xs_ic": round(float(np.mean(per_ts_ic)), 6) if per_ts_ic else 0.0,
        "independent_mean_xs_rank_ic": round(float(np.mean(per_ts_rank)), 6) if per_ts_rank else 0.0,
        "independent_positive_ts_fraction": (
            round(float(np.mean(np.array(per_ts_ic) > 0)), 4) if per_ts_ic else 0.0
        ),
    }


def _classify(rank_ic: float, positive_frac: float, net_sharpe_ls: float,
              net_sharpe_lo: float, outlier_dominated: bool) -> str:
    """Honest classification (mandate §69), gated on the robust RANK IC.

    The Pearson IC and net Sharpe are DELIBERATELY not used as the primary gate:
    on fat-tailed daily returns they are inflated by a handful of extreme moves.
    Cross-sectional RANK IC is the honest metric (§23, §40, §81). When the result
    is outlier-dominated, any positive Pearson/Sharpe is treated as an artifact.
    """
    if rank_ic < MIN_IC:
        return "NO_SIGNAL"
    # A positive, leakage-clean RANK IC above threshold is a real statistical
    # relationship. But when the result is outlier-dominated, the Pearson-based
    # net Sharpe / spread are NOT trustworthy economics, so the strongest claim
    # allowed is STATISTICALLY_INTERESTING — the tradeability question must be
    # settled by the real-OHLCV, next-open-execution backtest (§32), not by the
    # close-to-close Sharpe (which reflects bid-ask-bounce reversal).
    if outlier_dominated:
        return "STATISTICALLY_INTERESTING"
    best_net = max(net_sharpe_ls, net_sharpe_lo)
    if best_net <= 0.0:
        return "ECONOMICALLY_UNVIABLE"
    if positive_frac < 0.5:
        return "STATISTICALLY_INTERESTING"
    return "COST_SURVIVING_RESEARCH_CANDIDATE"


def run(args: argparse.Namespace) -> dict:
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]

    ohlcv, universe_meta = asyncio.run(
        _fetch_universe_ohlcv(args.interval, args.days, args.max_symbols, args.min_bars)
    )
    if len(ohlcv) < 10:
        raise CrossSectionalDataUnavailableError(
            f"only {len(ohlcv)} symbols with usable {args.interval} data; "
            "insufficient for cross-sectional research (§14, §97)."
        )
    print(f"[xs] fetched {len(ohlcv)} symbols @ {args.interval}")

    panel = build_cross_sectional_panel(
        ohlcv, PanelConfig(horizon=args.horizon, cost_bps=args.cost_bps)
    )
    n_ts = panel.index.get_level_values("ts").nunique()
    print(f"[xs] panel rows={len(panel)} timestamps={n_ts} "
          f"median_xs={int(panel.groupby(level='ts').size().median())}")

    feature_sets = {"BASE": BASE_FEATURES, "BASE+XS": BASE_FEATURES + XS_FEATURES}

    preds_root = Path("./artifacts/cross_sectional/predictions")
    preds_root.mkdir(parents=True, exist_ok=True)

    experiments: list[dict] = []
    n_hypotheses = 0

    for label in labels:
        target = LABEL_TARGET[label]
        realized = LABEL_REALIZED[label]
        for fs_name, fcols in feature_sets.items():
            for model in models:
                n_hypotheses += 1
                exp_id = f"xs-{args.interval}-{label}-{fs_name}-{model}-{uuid.uuid4().hex[:8]}"
                validator = CrossSectionalValidator(
                    n_windows=args.windows, embargo_bars=args.embargo,
                    cost_bps=args.cost_bps, decile=args.decile,
                )
                try:
                    result, preds = validator.evaluate(
                        panel, fcols, target, realized,
                        lambda m=model: build_estimator(m),
                        model, fs_name, args.horizon,
                    )
                except Exception as exc:  # record the failure, never hide it
                    experiments.append({
                        "experiment_id": exp_id, "label": label,
                        "feature_set": fs_name, "model": model,
                        "state": "ERROR", "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"[xs][ERR] {exp_id}: {exc}")
                    continue

                # Persist OOS predictions (mandate §50).
                pred_path = preds_root / f"{exp_id}.parquet"
                if not preds.empty:
                    try:
                        preds.to_parquet(pred_path)
                    except Exception:
                        preds.to_csv(pred_path.with_suffix(".csv"), index=False)

                indep = _independent_recompute(preds)
                # Independent-vs-reported agreement on the headline RANK IC (§79).
                agreement = compare(
                    result.mean_rank_ic, indep.get("independent_mean_xs_rank_ic", 0.0)
                )

                state = _classify(
                    result.mean_rank_ic, result.positive_window_fraction,
                    result.net_sharpe_ls, result.net_sharpe_long_only,
                    result.outlier_dominated,
                )
                exp = {
                    "experiment_id": exp_id,
                    "hypothesis": f"cross-sectional {label} return predictable via {model}/{fs_name}",
                    "data_source_class": "HISTORICAL_REAL",
                    "label": label, "feature_set": fs_name, "model": model,
                    "horizon": args.horizon, "cost_bps": args.cost_bps,
                    "result": result.to_dict(),
                    "independent_metrics": indep,
                    "independent_agreement": agreement,
                    "prediction_store": str(pred_path),
                    "state": state,
                }
                experiments.append(exp)
                print(f"[xs] {exp_id} RANK_IC={result.mean_rank_ic:+.4f} "
                      f"(pearson={result.mean_pearson_ic:+.4f} "
                      f"outlier_dom={result.outlier_dominated}) "
                      f"pos_frac={result.positive_window_fraction} "
                      f"median_spread={result.median_top_bottom_spread:+.5f} "
                      f"turn={result.mean_turnover} → {state}")

    # ── Overall summary (report ALL; do not cherry-pick, §101) ─────────────
    scored = [e for e in experiments if e.get("state") != "ERROR"]
    # Rank by the ROBUST rank IC, not the outlier-sensitive Pearson IC.
    best = max(scored, key=lambda e: e["result"]["mean_rank_ic"], default=None)
    # Overall = the strongest state achieved by ANY experiment, on the honest
    # ladder (do not cherry-pick a single run; report the best achieved state
    # but never above what the robust metrics support).
    _ladder = [
        "NO_SIGNAL", "STATISTICALLY_INTERESTING", "ECONOMICALLY_UNVIABLE",
        "COST_SURVIVING_RESEARCH_CANDIDATE",
    ]
    _rank = {s: i for i, s in enumerate(_ladder)}
    overall_state = "NO_SIGNAL"
    for e in scored:
        st = e.get("state", "NO_SIGNAL")
        if _rank.get(st, 0) > _rank.get(overall_state, 0):
            overall_state = st

    dataset_hash = _dataset_hash(panel, BASE_FEATURES + XS_FEATURES + ["fwd_return_raw"])
    dataset_id = f"xs-{args.interval}-{datetime.now(tz=UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

    evidence = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_sha(),
        "mandate": "§17-§41, §68-§69, §91 RUN A-G, §108",
        "data_source_class": "HISTORICAL_REAL",
        "interval": args.interval,
        "label_horizon_bars": args.horizon,
        "universe": universe_meta,
        "dataset": {
            "dataset_id": dataset_id,
            "dataset_hash": dataset_hash,
            "panel_rows": int(len(panel)),
            "n_timestamps": int(n_ts),
            "median_symbols_per_ts": int(panel.groupby(level="ts").size().median()),
            "date_start": str(panel.index.get_level_values("ts").min()),
            "date_end": str(panel.index.get_level_values("ts").max()),
            "feature_schema": {"base": BASE_FEATURES, "cross_sectional": XS_FEATURES},
            "label_families": labels,
            "pit_note": (
                "Features causal (bars<=T); labels strictly forward (T->T+h); "
                "beta estimated only from pre-T history (shifted). Walk-forward "
                "by date with embargo. Survivorship: CURRENT_UNIVERSE_ONLY."
            ),
        },
        "acceptance": {"min_ic": MIN_IC, "max_pbo": MAX_PBO,
                       "note": "same thresholds as daily/intraday; not tuned to pass (§67)"},
        "multiple_testing": {
            "n_hypotheses_tested": n_hypotheses,
            "note": "Each (label × feature_set × model) is one hypothesis; all "
                    "reported (§100, §101). Deflate significance accordingly.",
        },
        "experiments": experiments,
        "best_by_ic": (best["experiment_id"] if best else None),
        "overall_state": overall_state,
        "note": (
            "Genuine cross-sectional walk-forward research on a broad, real, "
            "PIT-safe F&O universe. Per-timestamp cross-sectional IC + decile "
            "long-short/long-only portfolios with turnover-charged net returns. "
            "A negative / economically-unviable result is a valid, reported "
            "outcome (§69, §90, §103). SURVIVORSHIP_LIMITED (§11)."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, default=str))
    print(f"[xs] evidence → {out}")
    print(f"[xs] OVERALL_STATE={overall_state} "
          f"(best_rank_ic={best['result']['mean_rank_ic'] if best else 'n/a'})")
    return evidence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-symbols", type=int, default=0, help="0 = all")
    ap.add_argument("--min-bars", type=int, default=400)
    ap.add_argument("--labels", default="raw,excess,residual,rank")
    ap.add_argument("--models", default="logistic,ridge,lightgbm,xgboost")
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--embargo", type=int, default=5)
    ap.add_argument("--decile", type=float, default=0.1)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--days", type=int, default=1200)
    ap.add_argument("--out", default="reports/cross_sectional_research.json")
    args = ap.parse_args()
    try:
        run(args)
    except CrossSectionalDataUnavailableError as exc:
        print(f"[xs][FAIL] {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
