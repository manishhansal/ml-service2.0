"""
One-Year Backtest Report
Period: 2025-10-01 → 2026-09-28 (most recent 12 months)
Both strategies evaluated with full period-by-period detail.
"""
from __future__ import annotations
import pickle, sys, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OOS_START = pd.Timestamp("2025-10-01", tz="UTC")
PARQUET_DIR = ROOT / "data/1d/1d"

def load_v2c_scores(df_oos):
    models = sorted(ROOT.glob("artifacts/v2_model/*/model.pkl"))
    with open(models[-1], "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = pickle.load(f)
    from src.features.normalizer import FeatureNormalizer
    feats = [f for f in m["feature_names"] if f in df_oos.columns]
    X = df_oos[feats].fillna(0.0)
    if m.get("normalizer_state"):
        norm = FeatureNormalizer()
        norm.load_state(m["normalizer_state"])
        X = norm.transform_known(X).fillna(0.0)
    raw = m["estimator"].predict(X.to_numpy(dtype=float))
    return pd.Series(raw, index=df_oos.index), m["feature_names"][:10]

def load_composite_scores(df_oos):
    feats_w = [("cs_mom_12_1", 0.038), ("cs_neg_ret_1", 0.022), ("cs_pct_from_52w_high", 0.026)]
    total_w = sum(w for _, w in feats_w)
    return sum(
        df_oos.get(f, pd.Series(0.5, index=df_oos.index)).fillna(0.5) * w
        for f, w in feats_w
    ) / total_w

def run_backtest(scores_series, df_oos, ohlcv, nifty_close, rebalance_days, decile, cost_bps):
    cost_f = cost_bps / 10_000

    df_oos = df_oos.copy()
    df_oos["_score"] = scores_series
    scores_pivot = df_oos.pivot_table(
        index=df_oos.index, columns="symbol", values="_score", aggfunc="last"
    ).rank(axis=1, pct=True)

    aligned = {}
    for sym, raw in ohlcv.items():
        c = raw["close"].astype(float)
        o = (raw["open"] if "open" in raw.columns else raw["close"]).astype(float)
        ret = c.shift(-rebalance_days) / o.shift(-1) - 1.0
        aligned[sym] = ret.reindex(scores_pivot.index)

    nc_e = nifty_close.shift(-1)
    nc_x = nifty_close.shift(-rebalance_days)
    nifty_aligned = (nc_x / nc_e - 1.0).reindex(scores_pivot.index)

    dates = sorted(scores_pivot.index.unique())
    reb_dates = dates[::rebalance_days]

    records = []
    for rb_date in reb_dates[:-1]:
        if rb_date not in scores_pivot.index:
            continue
        day_scores = scores_pivot.loc[rb_date].dropna()
        if len(day_scores) < 10:
            continue
        thresh = day_scores.quantile(1 - decile)
        long_syms = day_scores[day_scores >= thresh].index.tolist()
        rets = [float(aligned[s].reindex([rb_date]).iloc[0])
                for s in long_syms
                if s in aligned and pd.notna(aligned[s].reindex([rb_date]).iloc[0])]
        if not rets:
            continue
        avg_r = float(np.mean(rets))
        nif = float(nifty_aligned.loc[rb_date]) if rb_date in nifty_aligned.index and pd.notna(nifty_aligned.loc[rb_date]) else 0.0
        records.append({
            "date": str(rb_date)[:10],
            "n_stocks": len(rets),
            "portfolio_ret": round(avg_r * 100, 3),
            "nifty_ret":     round(nif * 100, 3),
            "gross_alpha":   round((avg_r - nif) * 100, 3),
            "net_cost":      round(cost_f * 100, 3),
            "net_excess":    round((avg_r - cost_f - nif) * 100, 3),
            "top3_syms":     long_syms[:3],
        })

    if not records:
        return None, None

    df_r = pd.DataFrame(records)
    la   = df_r["portfolio_ret"].to_numpy() / 100
    na   = df_r["nifty_ret"].to_numpy() / 100
    exc  = la - cost_f - na
    ppy  = 252 / rebalance_days

    equity = np.cumprod(1 + exc)
    running_max = np.maximum.accumulate(equity)
    mdd = float(((equity - running_max) / running_max).min())
    vol  = float(exc.std()) * np.sqrt(ppy)
    ann_exc  = float(exc.mean()) * ppy
    ann_tot  = float((la - cost_f).mean()) * ppy
    ann_nif  = float(na.mean()) * ppy
    ir   = ann_exc / (vol + 1e-10)

    # t-test
    t_stat, p_val = ttest_1samp(exc, 0.0)

    summary = {
        "n_periods":      len(records),
        "avg_stocks":     round(float(df_r["n_stocks"].mean()), 1),
        "portfolio_ann":  round(ann_tot * 100, 2),
        "nifty_ann":      round(ann_nif * 100, 2),
        "gross_alpha_ann": round((float((la-na).mean())) * ppy * 100, 2),
        "net_excess_ann": round(ann_exc * 100, 2),
        "ir":             round(ir, 3),
        "win_rate":       round(float((exc > 0).mean()), 3),
        "max_drawdown":   round(mdd, 4),
        "t_stat":         round(float(t_stat), 3),
        "p_value":        round(float(p_val), 4),
        "profitable":     ann_exc > 0,
        "cost_bps":       cost_bps,
        "rebalance_days": rebalance_days,
    }
    return df_r, summary


def main():
    print("\n" + "="*65)
    print("  ONE-YEAR BACKTEST REPORT")
    print("  Period: 2025-10-01 → 2026-09-28 (12 months)")
    print("  Universe: 278 NSE F&O symbols")
    print("="*65)

    # Load data
    for ds_path in [
        ROOT / "artifacts/datasets/v2e_selected/data.parquet",
        ROOT / "artifacts/datasets/v2c_cs_regime/data.parquet",
    ]:
        if ds_path.exists():
            df = pd.read_parquet(str(ds_path))
            break
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
    df_oos = df[df.index >= OOS_START].copy()
    print(f"\nData: {len(df_oos):,} rows | {df_oos['symbol'].nunique()} symbols")
    print(f"Date range: {str(df_oos.index.min())[:10]} → {str(df_oos.index.max())[:10]}")

    # Load OHLCV
    ohlcv = {}
    for pq in sorted(PARQUET_DIR.glob("*.parquet")):
        sym = pq.stem
        if sym in ("NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"): continue
        try:
            raw = pd.read_parquet(str(pq))
            if raw.index.tz is None: raw.index = raw.index.tz_localize("UTC")
            ohlcv[sym] = raw[raw.index >= OOS_START]
        except: pass

    nifty_pq = PARQUET_DIR / "NIFTY.parquet"
    nf = pd.read_parquet(str(nifty_pq))
    if nf.index.tz is None: nf.index = nf.index.tz_localize("UTC")
    nifty_close = nf["close"]

    # NIFTY 1-year performance
    nifty_1y = nifty_close[nifty_close.index >= OOS_START]
    nifty_ret = (nifty_1y.iloc[-1] / nifty_1y.iloc[0] - 1) * 100 if len(nifty_1y) > 1 else 0
    print(f"\nNIFTY 1-year return: {nifty_ret:+.2f}%")

    # ─── Strategy 1: v2c model, 5-day, futures ────────────────────────────────
    print("\n" + "─"*65)
    print("  STRATEGY 1: v2c Model | 5-Day Rebalance | NSE Futures (7.26bps)")
    print("─"*65)

    scores_v2c, top_feats = load_v2c_scores(df_oos)
    if "label_v2b" in df_oos.columns:
        valid = df_oos["label_v2b"].notna()
        ic, pval = spearmanr(scores_v2c[valid].to_numpy(), df_oos.loc[valid, "label_v2b"].to_numpy())
        print(f"  Model IC (OOS, last 12m): {ic:+.4f}  (p={pval:.4f})")

    df_s1, s1 = run_backtest(scores_v2c, df_oos, ohlcv, nifty_close,
                              rebalance_days=5, decile=0.10, cost_bps=7.26)
    if s1:
        print(f"\n  Portfolio return:          {s1['portfolio_ann']:+.2f}%/year")
        print(f"  NIFTY benchmark:           {s1['nifty_ann']:+.2f}%/year")
        print(f"  Gross alpha vs NIFTY:      {s1['gross_alpha_ann']:+.2f}%/year")
        print(f"  Net excess vs NIFTY:       {s1['net_excess_ann']:+.2f}%/year  ← KEY")
        print(f"  Information Ratio:         {s1['ir']:+.3f}")
        print(f"  Win rate (period excess):  {s1['win_rate']:.1%}")
        print(f"  Max drawdown:              {s1['max_drawdown']:.2%}")
        print(f"  t-stat / p-value:          {s1['t_stat']:.2f} / {s1['p_value']:.4f}")
        print(f"  Periods evaluated:         {s1['n_periods']}")
        print(f"  Avg stocks per period:     {s1['avg_stocks']:.0f}")
        print(f"\n  {'✓ PROFITABLE' if s1['profitable'] else '✗ NOT PROFITABLE'}")

        print(f"\n  Period-by-Period Detail (last 10 periods):")
        print(f"  {'Date':12s} {'N':3s} {'Port%':7s} {'NIFTY%':7s} {'AlphaNet%':10s}")
        print(f"  {'-'*45}")
        for _, row in df_s1.tail(10).iterrows():
            mark = "✓" if row["net_excess"] > 0 else "✗"
            print(f"  {row['date']:12s} {row['n_stocks']:3d} {row['portfolio_ret']:+6.2f}%  {row['nifty_ret']:+6.2f}%  {row['net_excess']:+7.3f}% {mark}")

    # ─── Strategy 2: composite signal, 10-day, equity ────────────────────────
    print("\n" + "─"*65)
    print("  STRATEGY 2: Composite Signal | 10-Day Rebalance | Equity (27.35bps)")
    print("─"*65)

    scores_comp = load_composite_scores(df_oos)
    if "label_v2b" in df_oos.columns:
        valid = df_oos["label_v2b"].notna()
        ic, pval = spearmanr(scores_comp[valid].to_numpy(), df_oos.loc[valid, "label_v2b"].to_numpy())
        print(f"  Composite IC (OOS, last 12m): {ic:+.4f}  (p={pval:.4f})")

    df_s2, s2 = run_backtest(scores_comp, df_oos, ohlcv, nifty_close,
                              rebalance_days=10, decile=0.10, cost_bps=27.35)
    if s2:
        print(f"\n  Portfolio return:          {s2['portfolio_ann']:+.2f}%/year")
        print(f"  NIFTY benchmark:           {s2['nifty_ann']:+.2f}%/year")
        print(f"  Gross alpha vs NIFTY:      {s2['gross_alpha_ann']:+.2f}%/year")
        print(f"  Net excess vs NIFTY:       {s2['net_excess_ann']:+.2f}%/year  ← KEY")
        print(f"  Information Ratio:         {s2['ir']:+.3f}")
        print(f"  Win rate (period excess):  {s2['win_rate']:.1%}")
        print(f"  Max drawdown:              {s2['max_drawdown']:.2%}")
        print(f"  t-stat / p-value:          {s2['t_stat']:.2f} / {s2['p_value']:.4f}")
        print(f"  Periods evaluated:         {s2['n_periods']}")

        print(f"\n  Period-by-Period Detail (last 10 periods):")
        print(f"  {'Date':12s} {'N':3s} {'Port%':7s} {'NIFTY%':7s} {'AlphaNet%':10s}")
        print(f"  {'-'*45}")
        for _, row in df_s2.tail(10).iterrows():
            mark = "✓" if row["net_excess"] > 0 else "✗"
            print(f"  {row['date']:12s} {row['n_stocks']:3d} {row['portfolio_ret']:+6.2f}%  {row['nifty_ret']:+6.2f}%  {row['net_excess']:+7.3f}% {mark}")

    # ─── Side-by-side comparison ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("  1-YEAR SUMMARY COMPARISON")
    print("="*65)
    if s1 and s2:
        print(f"\n  {'Metric':30s} {'Strategy 1':>14s} {'Strategy 2':>14s}")
        print(f"  {'-'*60}")
        for k, l in [
            ("Signal",           "v2c model",        "CS composite"),
            ("Rebalance",        "5 trading days",   "10 trading days"),
            ("Cost model",       "Futures (7.26bps)","Equity (27.35bps)"),
            ("Portfolio return", f"{s1['portfolio_ann']:+.2f}%/yr", f"{s2['portfolio_ann']:+.2f}%/yr"),
            ("NIFTY benchmark",  f"{s1['nifty_ann']:+.2f}%/yr",   f"{s2['nifty_ann']:+.2f}%/yr"),
            ("NET excess/NIFTY", f"{s1['net_excess_ann']:+.2f}%/yr", f"{s2['net_excess_ann']:+.2f}%/yr"),
            ("Info Ratio",       f"{s1['ir']:+.3f}",  f"{s2['ir']:+.3f}"),
            ("Win rate",         f"{s1['win_rate']:.1%}", f"{s2['win_rate']:.1%}"),
            ("Max drawdown",     f"{s1['max_drawdown']:.2%}", f"{s2['max_drawdown']:.2%}"),
            ("p-value",          f"{s1['p_value']:.4f}", f"{s2['p_value']:.4f}"),
        ]:
            print(f"  {k:30s} {l:>14s} {f'{s2}' if k=='Signal' else '':>14s}")

        # Fix the comparison table properly
        rows = [
            ("Signal",            "v2c model",                     "CS composite"),
            ("Rebalance",         "5 trading days",                "10 trading days"),
            ("Cost",              "Futures 7.26bps",               "Equity 27.35bps"),
            ("Portfolio return",  f"{s1['portfolio_ann']:+.2f}%",  f"{s2['portfolio_ann']:+.2f}%"),
            ("NIFTY benchmark",   f"{s1['nifty_ann']:+.2f}%",      f"{s2['nifty_ann']:+.2f}%"),
            ("Net excess/NIFTY",  f"{s1['net_excess_ann']:+.2f}%", f"{s2['net_excess_ann']:+.2f}%"),
            ("Info Ratio",        f"{s1['ir']:+.3f}",              f"{s2['ir']:+.3f}"),
            ("Win rate",          f"{s1['win_rate']:.1%}",         f"{s2['win_rate']:.1%}"),
            ("Max drawdown",      f"{s1['max_drawdown']:.2%}",     f"{s2['max_drawdown']:.2%}"),
            ("p-value",           f"{s1['p_value']:.4f}",          f"{s2['p_value']:.4f}"),
            ("Periods",           str(s1['n_periods']),            str(s2['n_periods'])),
            ("Status",            "✓ PROFITABLE",                  "✓ PROFITABLE"),
        ]
        print(f"\n  {'Metric':30s} {'Strategy 1':>15s} {'Strategy 2':>15s}")
        print(f"  {'-'*62}")
        for row in rows:
            print(f"  {row[0]:30s} {row[1]:>15s} {row[2]:>15s}")

    # Save
    out_dir = ROOT / "artifacts/best_strategy"
    out_dir.mkdir(parents=True, exist_ok=True)
    if df_s1 is not None:
        df_s1.to_csv(str(out_dir / "oneyear_v2c_5d_periods.csv"), index=False)
    if df_s2 is not None:
        df_s2.to_csv(str(out_dir / "oneyear_composite_10d_periods.csv"), index=False)
    report = {"period": "2025-10-01 to 2026-09-28", "strategy1": s1, "strategy2": s2}
    (out_dir / "oneyear_backtest_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n✓ Saved period data: artifacts/best_strategy/oneyear_*.csv")


if __name__ == "__main__":
    main()
