"""
scripts/run_holding_period_analysis.py
----------------------------------------
Test cross-sectional portfolio at holding periods h=5,10,20,40,60 bars.
At 27.65bps round-trip, a 5-bar hold has 88% annual turnover and
12.35% cost drag vs 2.45% gross return → not viable.
Longer holds reduce turnover, potentially unlocking the IC signal.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from src.features.expanded_factory import ExpandedFeatureFactory
from src.data.dataset_builder import DatasetBuilder
from src.reconciliation.costs import PRIMARY_COST
from src.reconciliation.pnl import ExecutablePortfolioBacktest
from src.models.estimators import LogisticBaseline

DATASETS_DIR = Path("artifacts/datasets")
PARQUET_DIR  = Path("data/1d/1d")
REPORTS_DIR  = Path("reports")


def build_panel() -> pd.DataFrame:
    builder = DatasetBuilder(
        output_root=DATASETS_DIR, feature_factory=ExpandedFeatureFactory(), normalize=False
    )
    ds_dirs = sorted(
        [d for d in DATASETS_DIR.iterdir() if d.is_dir() and "ds-1d-20260926" in d.name],
        key=lambda d: d.name, reverse=True,
    )
    ds_id = ds_dirs[0].name
    frame = builder.load_frame(ds_id)
    feature_cols = [c for c in ExpandedFeatureFactory().FEATURE_NAMES if c in frame.columns]

    X = frame[feature_cols].fillna(0).to_numpy(dtype=float)
    barrier = frame["label"].fillna(0).astype(float).values
    n = len(X); split = int(n * 0.8)

    mdl = LogisticBaseline()
    mdl.fit(X[:split], barrier[:split])
    preds_oos = mdl.predict(X[split:])
    ts_oos = pd.DatetimeIndex(frame.index[split:])
    syms_oos = frame["symbol"].values[split:]

    # Build open prices panel
    open_frames = []
    for pf in PARQUET_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(pf)
            df.columns = [c.lower() for c in df.columns]
            if "open" not in df.columns:
                continue
            s = pd.to_numeric(df["open"], errors="coerce").dropna()
            df_o = pd.DataFrame({"open": s, "symbol": pf.stem})
            df_o.index.name = "ts"
            open_frames.append(df_o.reset_index())
        except Exception:
            pass

    opens_all = pd.concat(open_frames)
    opens_all["ts"] = pd.to_datetime(opens_all["ts"], utc=True)
    opens_panel = opens_all.set_index(["ts", "symbol"]).sort_index()

    preds_df = pd.DataFrame({
        "ts":     pd.to_datetime(ts_oos, utc=True),
        "symbol": syms_oos,
        "score":  preds_oos,
    })
    preds_panel = preds_df.set_index(["ts", "symbol"])
    panel = opens_panel.join(preds_panel[["score"]], how="inner").dropna()[["score", "open"]]
    return panel


def main() -> None:
    print("=" * 70)
    print("HOLDING PERIOD ANALYSIS — Cross-sectional portfolio (logistic, fs-3.0.0)")
    print("=" * 70)
    print()

    panel = build_panel()
    n_ts  = panel.index.get_level_values("ts").nunique()
    n_sym = panel.index.get_level_values("symbol").nunique()
    print(f"Panel: {n_ts} timestamps × {n_sym} symbols\n")

    # ── Long-short decile ─────────────────────────────────────────────────
    print("Long-short decile (top 10% long / bottom 10% short) — conservative 27.65bps")
    print(f"{'h':>4}  {'XS_IC':>8}  {'gross_S':>8}  {'net_S':>8}  "
          f"{'cost_drag':>10}  {'turnover':>9}  {'verdict':>14}")
    print("-" * 70)

    ls_results: dict = {}
    for h in [5, 10, 20, 40, 60]:
        bt = ExecutablePortfolioBacktest(
            scores_panel=panel.copy(),
            cost_model=PRIMARY_COST,
            holding_bars=h,
            portfolio_type="top_bottom_decile_long_short",
            decile=0.10,
            min_symbols=10,
        )
        r = bt.run()
        verdict = "VIABLE" if r.net_sharpe > 0 else "NOT_VIABLE"
        ls_results[h] = {
            "xs_ic": r.xs_rank_ic_mean,
            "gross_sharpe": r.gross_sharpe,
            "net_sharpe": r.net_sharpe,
            "cost_drag_annual": r.cost_drag_annual,
            "mean_daily_turnover": r.mean_daily_turnover,
            "n_rebalances": r.n_rebalances,
            "gross_return_annual": r.gross_return_annual,
            "net_return_annual": r.net_return_annual,
        }
        print(
            f"{h:>4}  {r.xs_rank_ic_mean:>8.4f}  {r.gross_sharpe:>8.3f}  "
            f"{r.net_sharpe:>8.3f}  {r.cost_drag_annual:>10.4f}  "
            f"{r.mean_daily_turnover:>9.4f}  {verdict:>14}"
        )

    # ── Long-only quintile ────────────────────────────────────────────────
    print()
    print("Long-only top quintile (20%) — conservative 27.65bps")
    print(f"{'h':>4}  {'net_Sharpe':>10}  {'net_ann_ret':>12}  {'gross_ann_ret':>14}  {'verdict':>14}")
    print("-" * 60)

    lo_results: dict = {}
    for h in [5, 10, 20, 40, 60]:
        bt = ExecutablePortfolioBacktest(
            scores_panel=panel.copy(),
            cost_model=PRIMARY_COST,
            holding_bars=h,
            portfolio_type="top_quintile_long_only",
            decile=0.20,
            min_symbols=10,
        )
        r = bt.run()
        verdict = "VIABLE" if r.net_sharpe > 0 else "NOT_VIABLE"
        lo_results[h] = {
            "net_sharpe": r.net_sharpe,
            "gross_sharpe": r.gross_sharpe,
            "net_return_annual": r.net_return_annual,
            "gross_return_annual": r.gross_return_annual,
            "cost_drag_annual": r.cost_drag_annual,
            "n_rebalances": r.n_rebalances,
        }
        print(
            f"{h:>4}  {r.net_sharpe:>10.3f}  {r.net_return_annual:>12.4f}  "
            f"{r.gross_return_annual:>14.4f}  {verdict:>14}"
        )

    # ── Find minimum viable holding period ────────────────────────────────
    viable_ls = {h: v for h, v in ls_results.items() if v["net_sharpe"] > 0}
    viable_lo = {h: v for h, v in lo_results.items() if v["net_sharpe"] > 0}
    min_viable_ls = min(viable_ls.keys()) if viable_ls else None
    min_viable_lo = min(viable_lo.keys()) if viable_lo else None

    print()
    print("=" * 70)
    if min_viable_ls:
        v = ls_results[min_viable_ls]
        print(f"VIABLE at h={min_viable_ls}  (long-short)")
        print(f"  XS IC:         {v['xs_ic']:.4f}")
        print(f"  Net Sharpe:    {v['net_sharpe']:+.3f}")
        print(f"  Net ann ret:   {v['net_return_annual']:+.4f}")
        print(f"  Cost drag:     {v['cost_drag_annual']:.4f} ({v['cost_drag_annual']*100:.2f}%/yr)")
        print(f"  Turnover:      {v['mean_daily_turnover']:.4f}")
    elif min_viable_lo:
        v = lo_results[min_viable_lo]
        print(f"VIABLE at h={min_viable_lo}  (long-only)")
        print(f"  Net Sharpe:    {v['net_sharpe']:+.3f}")
        print(f"  Net ann ret:   {v['net_return_annual']:+.4f}")
    else:
        print("NOT_VIABLE at any tested holding period (5-60 bars)")
        # Find where breakeven is
        best_ls = max(ls_results.items(), key=lambda x: x[1]["net_sharpe"])
        print(f"  Best: h={best_ls[0]}, net_Sharpe={best_ls[1]['net_sharpe']:+.3f}")
        gross_best = best_ls[1]["gross_return_annual"]
        cost_best  = best_ls[1]["cost_drag_annual"]
        print(f"  Gross annual return: {gross_best:.4f}")
        print(f"  Cost drag:           {cost_best:.4f}")
        if cost_best > 0:
            breakeven_cost = gross_best
            print(f"  Need cost < {breakeven_cost*10000:.1f}bps to break even")
    print("=" * 70)

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        "schema": "holding_period_analysis_v1",
        "n_symbols": n_sym,
        "n_timestamps": n_ts,
        "long_short_decile": ls_results,
        "long_only_quintile": lo_results,
        "min_viable_holding_period_ls": min_viable_ls,
        "min_viable_holding_period_lo": min_viable_lo,
    }
    out = REPORTS_DIR / "holding_period_analysis.json"
    out.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
