"""
scripts/run_cross_sectional_backtest.py — decisive tradeability test + diagnostics
(mandate §25-§35, §51, §82-§85).

Takes the cross-sectional residual/rank signal that showed a positive, leakage-
clean RANK IC and answers the ECONOMIC question honestly:

    Does it survive REAL next-open execution and realistic costs, or is the
    close-to-close spread a bid-ask-bounce / reversal illusion?

Produces:
  - IC decay curve across forward horizons (§82).
  - Signal persistence (ranking autocorrelation) across lags (§26).
  - Real-OHLCV NEXT-OPEN backtest across holding periods, rebalance cadences,
    long-only vs long-short, hysteresis on/off (§27-§29, §32).
  - Cost sensitivity 5/10/15/20/30 bps + realistic base model (§31).
  - Turnover-vs-net-return frontier (§83) and honest final classification (§69).

Uses the local OHLCV cache populated by run_cross_sectional_research.py (real
data only; never synthetic). Exit 2 if no data.

Usage:
    PYTHONPATH=. python3 scripts/run_cross_sectional_backtest.py \
        [--label rank] [--model lightgbm] [--out reports/cross_sectional_backtest.json]
"""
from __future__ import annotations

import argparse
import asyncio
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
os.environ.setdefault("DATA_SERVICE_2_URL", "http://localhost:8200")
for _k, _v in {
    "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1",
    "OMP_MAX_ACTIVE_LEVELS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd

from scripts.run_cross_sectional_research import (
    LABEL_TARGET,
    XS_FEATURES,
    _fetch_universe_ohlcv,
    CrossSectionalDataUnavailableError,
)
from src.analytics.cross_sectional import PanelConfig, build_cross_sectional_panel
from src.analytics.cross_sectional_backtest import (
    CostModel,
    backtest_next_open,
    ic_decay_curve,
    signal_persistence,
)
from src.features.factory import FeatureFactory
from src.models.estimators import build_estimator

BASE = list(FeatureFactory.FEATURE_NAMES)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _fit_oos_scores(panel: pd.DataFrame, feature_cols: list[str], target: str,
                    train_frac: float, model: str) -> pd.Series:
    """Train on the first train_frac of timestamps, score the OOS remainder.

    A single expanding split (not the 6-window walk-forward) is sufficient here
    because we only need OOS SCORES to feed the execution backtest; the IC-level
    validation was already done in run_cross_sectional_research.py.
    """
    clean = panel.dropna(subset=feature_cols + [target]).copy()
    ts_all = np.array(sorted(clean.index.get_level_values("ts").unique()))
    split = int(len(ts_all) * train_frac)
    train_ts, test_ts = ts_all[:split], ts_all[split + 5:]  # 5-bar embargo
    tr = clean[clean.index.get_level_values("ts").isin(train_ts)]
    te = clean[clean.index.get_level_values("ts").isin(test_ts)]

    est = build_estimator(model)
    y = tr[target].to_numpy(float)
    if type(est).__name__.lower().startswith(("logistic", "lightgbm", "xgboost", "catboost")):
        y = (y > np.median(y)).astype(float)
    est.fit(tr[feature_cols].to_numpy(float), y)
    scores = est.predict(te[feature_cols].to_numpy(float))
    return pd.Series(scores, index=te.index, name="score")


def run(args: argparse.Namespace) -> dict:
    ohlcv, uni_meta = asyncio.run(
        _fetch_universe_ohlcv(args.interval, args.days, 0, args.min_bars)
    )
    print(f"[bt] symbols={len(ohlcv)} (from_cache={uni_meta.get('from_cache')})")
    panel = build_cross_sectional_panel(ohlcv, PanelConfig(horizon=1, cost_bps=10))

    feature_cols = BASE + XS_FEATURES
    target = LABEL_TARGET[args.label]
    scores = _fit_oos_scores(panel, feature_cols, target, args.train_frac, args.model)
    # Attach scores to the panel (OOS subset only) for decay/persistence.
    panel_oos = panel.loc[scores.index].copy()
    panel_oos["_score"] = scores

    # ── IC decay (§82) ─────────────────────────────────────────────────────
    decay = ic_decay_curve(panel_oos, "_score", horizons=[1, 2, 3, 5, 10, 20])
    print(f"[bt] IC decay: {{h: v['mean_rank_ic'] for ...}} = "
          + ", ".join(f"{h}:{v['mean_rank_ic']:+.4f}" for h, v in decay.items()))

    # ── Signal persistence (§26) ───────────────────────────────────────────
    persistence = signal_persistence(panel_oos, "_score", lags=[1, 2, 3, 5])
    print(f"[bt] persistence(rank autocorr): "
          + ", ".join(f"{k}:{v:+.3f}" for k, v in persistence.items()))

    base_cost = CostModel()

    # ── Holding-period sweep (§28) at rebalance=holding, LS + long-only ────
    holding_sweep: list[dict] = []
    for h in [1, 2, 3, 5]:
        for lo in (False, True):
            r = backtest_next_open(
                panel_oos, scores, holding_bars=h, rebalance_every=h,
                decile=args.decile, long_only=lo, cost_model=base_cost,
            )
            holding_sweep.append(r.to_dict() | {"variant": "long_only" if lo else "long_short"})
            print(f"[bt] hold={h} {'LO' if lo else 'LS'}: "
                  f"net_ann={r.net_return_annual:+.4f} net_sharpe={r.net_sharpe} "
                  f"gross_sharpe={r.gross_sharpe} turn={r.mean_daily_turnover} "
                  f"maxdd={r.max_drawdown}")

    # ── Hysteresis on/off (§27) at holding=1 LS ────────────────────────────
    hyst = {}
    for use in (False, True):
        r = backtest_next_open(
            panel_oos, scores, holding_bars=1, rebalance_every=1, decile=args.decile,
            long_only=False, cost_model=base_cost, use_hysteresis=use,
        )
        hyst["hysteresis_on" if use else "hysteresis_off"] = r.to_dict()

    # ── Cost sensitivity (§31) at holding=1 LS ─────────────────────────────
    cost_sens = {}
    for bps in [5, 10, 15, 20, 30]:
        cm = CostModel(
            brokerage_bps=0.0, exchange_bps=0.0, stt_bps=0.0, gst_bps=0.0,
            stamp_bps=0.0, slippage_bps=bps / 2.0, half_spread_bps=bps / 2.0,
        )
        r = backtest_next_open(panel_oos, scores, holding_bars=1, rebalance_every=1,
                               decile=args.decile, long_only=False, cost_model=cm)
        cost_sens[f"{bps}bps"] = {
            "round_trip_bps": r.cost_model_round_trip_bps,
            "net_return_annual": r.net_return_annual,
            "net_sharpe": r.net_sharpe,
        }
        print(f"[bt] cost {bps}bps: net_ann={r.net_return_annual:+.4f} sharpe={r.net_sharpe}")

    # ── Base realistic model, holding=1 LS + LO (the headline economics) ───
    base_ls = backtest_next_open(panel_oos, scores, holding_bars=1, rebalance_every=1,
                                 decile=args.decile, long_only=False, cost_model=base_cost)
    base_lo = backtest_next_open(panel_oos, scores, holding_bars=1, rebalance_every=1,
                                 decile=args.decile, long_only=True, cost_model=base_cost)

    # ── Honest classification (§69) — economics decide ─────────────────────
    decay_1 = decay.get(1, {}).get("mean_rank_ic", 0.0)
    all_net_sharpes = [base_ls.net_sharpe, base_lo.net_sharpe] + [
        h["net_sharpe"] for h in holding_sweep
    ]
    best_net_sharpe = max(all_net_sharpes)
    median_net_sharpe = float(np.median(all_net_sharpes))
    # Honest thresholds. A NEXT-OPEN net Sharpe must be MEANINGFULLY positive
    # (>0.5) AND the typical configuration must not be deeply negative for a
    # cost-surviving claim. A single barely-positive config amid uniformly
    # negative ones is noise, not an edge.
    if decay_1 < 0.02:
        state = "NO_SIGNAL"
    elif best_net_sharpe > 0.5 and median_net_sharpe > 0.0:
        state = "COST_SURVIVING_RESEARCH_CANDIDATE"
    elif best_net_sharpe > 0.5:
        state = "STATISTICALLY_INTERESTING"
    else:
        # Statistically detectable close-to-close signal that does NOT survive
        # realistic next-open execution — the classic reversal / bid-ask-bounce
        # artifact. This is the honest, expected outcome (§69, §90).
        state = "ECONOMICALLY_UNVIABLE"

    evidence = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_sha(),
        "mandate": "§25-§35, §51, §82-§85",
        "data_source_class": "HISTORICAL_REAL",
        "pnl_provenance": "REAL_HISTORICAL_OHLCV_NEXT_OPEN",
        "is_economic_evidence": True,
        "interval": args.interval,
        "label": args.label,
        "model": args.model,
        "feature_set": "BASE+XS",
        "universe": uni_meta,
        "execution_note": (
            "Signals formed at close[T]; positions ENTERED at open[T+1] and "
            "EXITED at open[T+1+holding]. No close-to-close fills, no "
            "reconstructed price path (§32, §33). This is the honest tradeability "
            "test that separates real alpha from close-to-close reversal / "
            "bid-ask-bounce illusion."
        ),
        "ic_decay": decay,
        "signal_persistence": persistence,
        "holding_sweep": holding_sweep,
        "hysteresis": hyst,
        "cost_sensitivity": cost_sens,
        "headline": {
            "long_short_hold1": base_ls.to_dict(),
            "long_only_hold1": base_lo.to_dict(),
            "next_open_ic_h1": decay_1,
            "best_net_sharpe": best_net_sharpe,
            "median_net_sharpe": round(median_net_sharpe, 4),
        },
        "state": state,
        "note": (
            "SURVIVORSHIP_LIMITED (§11). If the close-to-close rank IC is positive "
            "but the NEXT-OPEN net Sharpe is <=0, the signal is a non-tradeable "
            "microstructure/reversal artifact — a valid, reported outcome (§69, §90)."
        ),
    }
    # Trim per_rebalance detail from the big sweeps to keep the report readable;
    # keep a bounded sample for the headline runs (trade-level audit §52). The
    # full audit trail lives in the persisted prediction store.
    for h in evidence["holding_sweep"]:
        h.pop("per_rebalance", None)
    for v in evidence["hysteresis"].values():
        v.pop("per_rebalance", None)
    for k in ("long_short_hold1", "long_only_hold1"):
        pr = evidence["headline"][k].get("per_rebalance", [])
        evidence["headline"][k]["per_rebalance_sample"] = pr[:30]
        evidence["headline"][k]["per_rebalance_count"] = len(pr)
        evidence["headline"][k].pop("per_rebalance", None)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, default=str))
    print(f"[bt] evidence → {out}")
    print(f"[bt] IC(h=1)={decay_1:+.4f} base_LS_net_sharpe={base_ls.net_sharpe} "
          f"base_LO_net_sharpe={base_lo.net_sharpe} → STATE={state}")
    return evidence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--label", default="rank", choices=list(LABEL_TARGET))
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--decile", type=float, default=0.1)
    ap.add_argument("--min-bars", type=int, default=400)
    ap.add_argument("--days", type=int, default=1200)
    ap.add_argument("--out", default="reports/cross_sectional_backtest.json")
    args = ap.parse_args()
    try:
        run(args)
    except CrossSectionalDataUnavailableError as exc:
        print(f"[bt][FAIL] {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
