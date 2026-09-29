#!/usr/bin/env python3
"""
Concentrated portfolio backtest with LightGBM champion (fs-3.0.0).

Key finding from previous analysis:
  - Standard decile (10%, 21 positions): viable only at ≤5bps
  - Concentrated 5% (11 positions): viable up to 10bps → Sharpe +1.94

With LightGBM IC_continuous=0.3757 vs logistic 0.3088, we expect
a further improvement.

Tests: concentrated 5% at 5/8.5/10/15bps — the key NSE Futures range.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATASETS_DIR = Path("artifacts/datasets")
PARQUET_DIR  = Path("data/1d/1d")
REPORTS_DIR  = Path("reports")


def make_cm(target_bps: float, label: str):
    from src.reconciliation.costs import IndianCostModel
    per_side = target_bps / 2.0
    each = per_side / 5.0
    return IndianCostModel(
        brokerage_bps=each, exchange_charge_bps=each,
        gst_bps=each, slippage_bps=each, half_spread_bps=each,
        stt_bps=0.0, stamp_duty_bps=0.0,
        scenario=label, note=f"~{target_bps}bps round-trip",
    )


def build_panel_lgbm(builder, ds_id: str):
    from src.features.expanded_factory import ExpandedFeatureFactory
    from src.models.estimators import LightGBMModel

    frame = builder.load_frame(ds_id)
    feat_cols = [c for c in ExpandedFeatureFactory().FEATURE_NAMES if c in frame.columns]
    X = frame[feat_cols].fillna(0).to_numpy(dtype=float)
    continuous = frame["realized_return"].fillna(0).values
    barrier    = frame["label"].fillna(0).astype(float).values
    n = len(X); split = int(n * 0.8)

    print("Fitting LightGBM on 80% train split...")
    mdl = LightGBMModel().fit(X[:split], barrier[:split])
    preds_oos = mdl.predict(X[split:])
    ts_oos    = pd.DatetimeIndex(frame.index[split:])
    syms_oos  = frame["symbol"].values[split:]
    rets_oos  = continuous[split:]

    open_frames = []
    for pf in PARQUET_DIR.glob("*.parquet"):
        try:
            df_o = pd.read_parquet(pf)
            df_o.columns = [c.lower() for c in df_o.columns]
            if "open" not in df_o.columns:
                continue
            s = pd.to_numeric(df_o["open"], errors="coerce").dropna()
            tmp = pd.DataFrame({"open": s, "symbol": pf.stem})
            tmp.index.name = "ts"
            open_frames.append(tmp.reset_index())
        except Exception:
            pass

    opens = pd.concat(open_frames)
    opens["ts"] = pd.to_datetime(opens["ts"], utc=True)
    opens_panel = opens.set_index(["ts", "symbol"]).sort_index()
    preds_df = pd.DataFrame({
        "ts": pd.to_datetime(ts_oos, utc=True),
        "symbol": syms_oos, "score": preds_oos,
    }).set_index(["ts", "symbol"])
    panel = opens_panel.join(preds_df[["score"]], how="inner").dropna()[["score", "open"]]
    return panel, preds_oos, rets_oos


def run_bt(panel, cm, h, ptype, decile):
    from src.reconciliation.pnl import ExecutablePortfolioBacktest
    bt = ExecutablePortfolioBacktest(
        scores_panel=panel.copy(), cost_model=cm,
        holding_bars=h, portfolio_type=ptype,
        decile=decile, min_symbols=10,
    )
    return bt.run()


def main():
    from src.features.expanded_factory import ExpandedFeatureFactory
    from src.data.dataset_builder import DatasetBuilder

    print("=" * 70)
    print("LGBM CONCENTRATED PORTFOLIO BACKTEST — 218 symbols, fs-3.0.0")
    print("=" * 70)

    builder = DatasetBuilder(
        output_root=DATASETS_DIR,
        feature_factory=ExpandedFeatureFactory(), normalize=False,
    )
    ds_dirs = sorted(
        [d for d in DATASETS_DIR.iterdir() if d.is_dir() and "ds-1d-" in d.name],
        key=lambda d: d.name, reverse=True,
    )
    ds_id = ds_dirs[0].name
    print(f"Dataset: {ds_id}")

    panel, preds_oos, rets_oos = build_panel_lgbm(builder, ds_id)
    n_ts  = panel.index.get_level_values("ts").nunique()
    n_sym = panel.index.get_level_values("symbol").nunique()
    ic_c, _ = spearmanr(preds_oos, rets_oos)
    print(f"Panel: {n_ts} ts × {n_sym} syms  |  LightGBM IC_continuous = {ic_c:.4f}")

    results = {}

    # ── CONCENTRATED 5% LONG-SHORT ────────────────────────────────
    print()
    print("Concentrated 5% long-short (h=5):")
    print(f"  {'bps':7s}  {'XS_IC':7s}  {'gross_S':8s}  {'net_S':8s}  {'net_ann%':9s}  verdict")
    for bps in [5.0, 8.5, 10.0, 12.0, 15.0, 20.0, 27.65]:
        cm = make_cm(bps, f"{bps}bps")
        r = run_bt(panel, cm, 5, "top_bottom_decile_long_short", 0.05)
        v = "✓ VIABLE" if r.net_sharpe > 0 else "  NOT_VIABLE"
        results[f"conc_5pct_{bps}bps"] = {
            "bps": bps, "xs_ic": r.xs_rank_ic_mean,
            "gross_sharpe": r.gross_sharpe, "net_sharpe": r.net_sharpe,
            "net_ann": r.net_return_annual, "gross_ann": r.gross_return_annual,
            "cost_drag": r.cost_drag_annual,
        }
        print(f"  {bps:7.2f}  {r.xs_rank_ic_mean:7.4f}  {r.gross_sharpe:+8.3f}  "
              f"{r.net_sharpe:+8.3f}  {r.net_return_annual*100:+9.2f}%  {v}")

    # ── STANDARD DECILE ───────────────────────────────────────────
    print()
    print("Standard decile 10% long-short (h=5):")
    print(f"  {'bps':7s}  {'net_S':8s}  {'net_ann%':9s}  verdict")
    for bps in [5.0, 8.5, 10.0, 27.65]:
        cm = make_cm(bps, f"{bps}bps")
        r = run_bt(panel, cm, 5, "top_bottom_decile_long_short", 0.10)
        v = "✓ VIABLE" if r.net_sharpe > 0 else "  NOT_VIABLE"
        results[f"std_10pct_{bps}bps"] = {"bps": bps, "net_sharpe": r.net_sharpe, "net_ann": r.net_return_annual}
        print(f"  {bps:7.2f}  {r.net_sharpe:+8.3f}  {r.net_return_annual*100:+9.2f}%  {v}")

    # ── HOLDING PERIOD SWEEP AT 8.5bps ───────────────────────────
    print()
    print("Concentrated 5% holding period sweep at 8.5bps:")
    cm85 = make_cm(8.5, "8p5bps")
    hp = {}
    for h in [5, 10, 20]:
        r = run_bt(panel, cm85, h, "top_bottom_decile_long_short", 0.05)
        hp[h] = {"net_sharpe": r.net_sharpe, "xs_ic": r.xs_rank_ic_mean, "gross_sharpe": r.gross_sharpe}
        v = "✓" if r.net_sharpe > 0 else " "
        print(f"  h={h:2d}  XS_IC={r.xs_rank_ic_mean:.4f}  gross_S={r.gross_sharpe:+.3f}  net_S={r.net_sharpe:+.3f}  {v}")

    # ── Determine viability ─────────────────────────────────────
    viable = {k: v for k, v in results.items() if v.get("net_sharpe", -99) > 0}
    conc_viable = {k: v for k, v in viable.items() if "conc" in k}
    std_viable  = {k: v for k, v in viable.items() if "std" in k}

    # ── Save ──────────────────────────────────────────────────────
    output = {
        "model": "lightgbm",
        "ic_continuous_oos": round(float(ic_c), 4),
        "results": results,
        "holding_period_sweep_8p5bps": hp,
        "viable_configurations": list(viable.keys()),
        "min_viable_bps_concentrated": min(v["bps"] for v in conc_viable.values()) if conc_viable else None,
        "min_viable_bps_standard":    min(v["bps"] for v in std_viable.values()) if std_viable else None,
    }
    out = REPORTS_DIR / "lgbm_concentrated_backtest.json"
    out.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved → {out}")

    # ── Final verdict ─────────────────────────────────────────────
    print()
    print("=" * 70)
    if conc_viable:
        best_bps = min(v["bps"] for v in conc_viable.values())
        best_ns  = max(v.get("net_sharpe", 0) for v in conc_viable.values())
        best_ann = max(v.get("net_ann", 0) for v in conc_viable.values())
        print(f"✓ CONCENTRATED 5% VIABLE AT ≤{best_bps:.1f}bps")
        print(f"  Best net Sharpe: {best_ns:+.3f}  Best net annual: {best_ann*100:+.2f}%")
        print()
        print("CERTIFICATION GATE UPDATE:")
        print("  G6 (Cost robustness ≥1.5× costs): CHECK")
        max_viable_bps = max(v["bps"] for v in conc_viable.values())
        print(f"  Primary cost: {max_viable_bps:.1f}bps × 1.5 = {max_viable_bps*1.5:.1f}bps")
        if max_viable_bps * 1.5 in [v["bps"] for v in conc_viable.values()]:
            print(f"  G6: PASS — viable at {max_viable_bps*1.5:.1f}bps (1.5× primary)")
        else:
            print(f"  G6: FAIL — not viable at {max_viable_bps*1.5:.1f}bps (1.5× primary)")
    else:
        print("✗ NOT VIABLE at any cost (LightGBM concentrated)")
    print("=" * 70)


if __name__ == "__main__":
    main()
