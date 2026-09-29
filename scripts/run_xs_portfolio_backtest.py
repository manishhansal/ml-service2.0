"""
scripts/run_xs_portfolio_backtest.py
--------------------------------------
Run the CORRECT economic evaluation for a cross-sectional 218-symbol universe:
ExecutablePortfolioBacktest (signal at close[T] → enter open[T+1] → exit open[T+1+h]).

Why the single-symbol BacktestEngine showed negative Sharpe:
  The single-symbol engine trades signal[t] → position (long/short) every bar.
  With 394k rows / 218 symbols × 5-bar labels there are ~78k trades → turnover
  is 100% every bar → costs destroy everything.

The CORRECT approach for a cross-sectional model:
  1. At each rebalance date T, score all 218 symbols
  2. Long top decile (21 symbols), short bottom decile (21 symbols)
  3. Hold h=5 bars, then rebalance
  4. Turnover ≈ 20–40% per rebalance (not 100%)
  5. This is what institutional quant desks actually do

This script:
  (a) Builds the scores_panel from on-disk parquets + logistic OOS predictions
  (b) Runs ExecutablePortfolioBacktest at multiple cost scenarios
  (c) Saves BACKTEST_REPORT.md and cross_sectional_portfolio_backtest.json
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PARQUET_DIR  = Path("data/1d/1d")
DATASETS_DIR = Path("artifacts/datasets")
REPORTS_DIR  = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
from src.features.expanded_factory import ExpandedFeatureFactory
from src.data.dataset_builder import DatasetBuilder
from src.reconciliation.costs import ALL_SCENARIOS, PRIMARY_COST
from src.reconciliation.pnl import ExecutablePortfolioBacktest, build_scores_panel

def main() -> None:
    print("=" * 65)
    print("CROSS-SECTIONAL PORTFOLIO BACKTEST — 218 symbols / 55 features")
    print("=" * 65)

    # ── Find dataset ──────────────────────────────────────────────────────
    builder = DatasetBuilder(
        output_root=DATASETS_DIR,
        feature_factory=ExpandedFeatureFactory(),
        normalize=False,
    )
    ds_dirs = sorted(
        [d for d in DATASETS_DIR.iterdir() if d.is_dir() and "ds-1d-20260926" in d.name],
        key=lambda d: d.name, reverse=True,
    )
    if not ds_dirs:
        print("ERROR: No fs-3.0.0 dataset found. Run train_expanded_features.py first.")
        return

    ds_id = ds_dirs[0].name
    print(f"\nDataset: {ds_id}")
    frame = builder.load_frame(ds_id)
    feature_cols = [c for c in ExpandedFeatureFactory().FEATURE_NAMES if c in frame.columns]
    print(f"Rows: {len(frame)} | Features: {len(feature_cols)} | Symbols: {frame['symbol'].nunique()}")

    X = frame[feature_cols].fillna(0).to_numpy(dtype=float)
    continuous = frame["realized_return"].fillna(0).values
    barrier    = frame["label"].fillna(0).astype(float).values
    n = len(X); split = int(n * 0.8)

    # ── Fit logistic on train, score OOS ─────────────────────────────────
    print("\n[1] Fitting LogisticBaseline on train split (80%) → score OOS (20%)...")
    from src.models.estimators import LogisticBaseline
    mdl = LogisticBaseline()
    mdl.fit(X[:split], barrier[:split])
    preds_oos  = mdl.predict(X[split:])
    rets_oos   = continuous[split:]
    ts_oos     = pd.DatetimeIndex(frame.index[split:])
    syms_oos   = frame["symbol"].values[split:]

    ic_cont, _ = spearmanr(preds_oos, rets_oos)
    ic_barr, _ = spearmanr(preds_oos, barrier[split:])
    print(f"  IC(continuous, OOS): {ic_cont:.4f}  IC(barrier): {ic_barr:.4f}")

    # ── Build scores panel ────────────────────────────────────────────────
    print("\n[2] Building (ts, symbol) scores panel with open prices...")
    # Load open prices from parquets
    open_by_sym: dict[str, pd.Series] = {}
    for pf in PARQUET_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(pf)
            df.columns = [c.lower() for c in df.columns]
            if "open" in df.columns:
                s = pd.to_numeric(df["open"], errors="coerce").dropna()
                if len(s) > 0:
                    open_by_sym[pf.stem] = s
        except Exception:
            pass

    # Build predictions DataFrame aligned to OOS
    preds_df = pd.DataFrame({
        "ts":     ts_oos,
        "symbol": syms_oos,
        "score":  preds_oos,
    })

    # Build open panel (ts, symbol) from parquet data
    open_frames = []
    for sym, s in open_by_sym.items():
        df_o = pd.DataFrame({"open": s, "symbol": sym})
        df_o.index.name = "ts"
        open_frames.append(df_o.reset_index())
    if not open_frames:
        print("ERROR: Could not load open prices")
        return
    opens_all = pd.concat(open_frames)
    opens_all["ts"] = pd.to_datetime(opens_all["ts"], utc=True)
    opens_panel = opens_all.set_index(["ts", "symbol"]).sort_index()

    # Align predictions with opens
    preds_df["ts"] = pd.to_datetime(preds_df["ts"], utc=True)
    preds_panel = preds_df.set_index(["ts", "symbol"])
    panel = opens_panel.join(preds_panel[["score"]], how="inner").dropna()
    panel = panel[["score", "open"]]

    n_ts = panel.index.get_level_values("ts").nunique()
    n_sym = panel.index.get_level_values("symbol").nunique()
    print(f"  Panel: {len(panel)} rows | {n_ts} timestamps | {n_sym} symbols")

    if n_ts < 30 or n_sym < 10:
        print("  WARNING: Insufficient panel coverage for cross-sectional backtest")
        return

    # ── Run cross-sectional backtest across cost scenarios ────────────────
    print("\n[3] Running ExecutablePortfolioBacktest (decile long-short, h=5)...")
    results: dict[str, dict] = {}

    for scenario in ALL_SCENARIOS:
        bt = ExecutablePortfolioBacktest(
            scores_panel=panel.copy(),
            cost_model=scenario,
            holding_bars=5,
            portfolio_type="top_bottom_decile_long_short",
            decile=0.10,
            min_symbols=10,
        )
        r = bt.run()
        viable = "VIABLE" if r.net_sharpe > 0 else "NOT_VIABLE"
        results[scenario.scenario] = r.to_dict()
        print(
            f"  {scenario.scenario:15s} {r.round_trip_bps:.1f}bps | "
            f"XS_IC={r.xs_rank_ic_mean:.4f} | "
            f"gross_Sharpe={r.gross_sharpe:+.3f} | "
            f"net_Sharpe={r.net_sharpe:+.3f} | "
            f"n_rebal={r.n_rebalances} | {viable}"
        )

    # ── Also test long-only top quintile ─────────────────────────────────
    print("\n[4] Long-only top quintile (realistic for Indian F&O)...")
    long_results: dict[str, dict] = {}
    for scenario in ALL_SCENARIOS[:3]:   # conservative/moderate/aggressive only
        bt_lo = ExecutablePortfolioBacktest(
            scores_panel=panel.copy(),
            cost_model=scenario,
            holding_bars=5,
            portfolio_type="top_quintile_long_only",
            decile=0.20,
            min_symbols=10,
        )
        r_lo = bt_lo.run()
        viable = "VIABLE" if r_lo.net_sharpe > 0 else "NOT_VIABLE"
        long_results[scenario.scenario] = r_lo.to_dict()
        print(
            f"  {scenario.scenario:15s} {r_lo.round_trip_bps:.1f}bps | "
            f"net_Sharpe={r_lo.net_sharpe:+.3f} | "
            f"net_ret_ann={r_lo.net_return_annual:+.4f} | {viable}"
        )

    # ── Save report ───────────────────────────────────────────────────────
    primary = results.get("conservative", {})
    primary_long = long_results.get("conservative", {})

    output = {
        "schema": "xs_portfolio_backtest_v1",
        "run_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset_id": ds_id,
        "feature_schema_version": "fs-3.0.0",
        "n_symbols": n_sym,
        "n_timestamps": n_ts,
        "portfolio_type": "top_bottom_decile_long_short",
        "holding_bars": 5,
        "ic_continuous_oos": round(float(ic_cont), 4),
        "ic_barrier_oos": round(float(ic_barr), 4),
        "long_short_results": results,
        "long_only_results": long_results,
        "primary_verdict": {
            "xs_rank_ic": primary.get("xs_rank_ic_mean", 0),
            "gross_sharpe": primary.get("gross_sharpe", 0),
            "net_sharpe": primary.get("net_sharpe", 0),
            "net_return_annual": primary.get("net_return_annual", 0),
            "cost_drag_annual": primary.get("cost_drag_annual", 0),
            "is_viable": primary.get("net_sharpe", -99) > 0,
        },
    }

    out = REPORTS_DIR / "cross_sectional_portfolio_backtest_expanded.json"
    out.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved -> {out}")

    # ── Final verdict ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PORTFOLIO BACKTEST VERDICT")
    print("=" * 65)
    primary_sharpe = primary.get("net_sharpe", -99)
    primary_xs_ic  = primary.get("xs_rank_ic_mean", 0)
    primary_lo_sharpe = primary_long.get("net_sharpe", -99)

    print(f"  IC(continuous, OOS):    {ic_cont:.4f}")
    print(f"  XS Rank IC (portfolio): {primary_xs_ic:.4f}")
    print(f"  Long-short net Sharpe:  {primary_sharpe:+.3f}")
    print(f"  Long-only net Sharpe:   {primary_lo_sharpe:+.3f}")
    print()
    if primary_xs_ic > 0.02 and primary_sharpe > 0:
        print("RESULT: ECONOMIC_SIGNAL_DETECTED")
        print("  Cross-sectional long-short portfolio has POSITIVE net Sharpe")
        print("  RECOMMENDATION: advance to paper promotion evaluation")
    elif primary_xs_ic > 0.02 and primary_lo_sharpe > 0:
        print("RESULT: LONG_ONLY_VIABLE_LONG_SHORT_NOT_VIABLE")
        print("  Long-only top quintile has POSITIVE net Sharpe")
        print("  Shorting costs/constraints prevent long-short from working")
        print("  RECOMMENDATION: evaluate long-only strategy for paper promotion")
    elif primary_xs_ic > 0.02:
        print("RESULT: IC_POSITIVE_SHARPE_NEGATIVE")
        print(f"  XS IC={primary_xs_ic:.4f}>0.02 but net Sharpe negative")
        print("  Signal exists but costs are too high relative to expected return")
        print("  RECOMMENDATION: investigate capacity / larger position sizes / lower-cost execution")
    else:
        print("RESULT: NO_ECONOMIC_SIGNAL")
        print(f"  XS IC={primary_xs_ic:.4f} below 0.02 threshold")
    print("=" * 65)


if __name__ == "__main__":
    main()
