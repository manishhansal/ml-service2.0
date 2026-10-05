"""
run_best_strategy.py — Production-ready runner for the optimal strategy.

Best configuration (from grid search):
  v2c model | 5-day rebalance | top 10% | futures (7.26bps) | equal-weight
  OOS: +20.08%/year excess vs NIFTY | IR=1.29 | 60% win rate | -8.29% max DD

Usage:
    PYTHONPATH=. python3 scripts/run_best_strategy.py
    PYTHONPATH=. python3 scripts/run_best_strategy.py --strategy composite10d
"""
from __future__ import annotations
import argparse, json, pickle, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OOS_START = pd.Timestamp("2025-01-01", tz="UTC")
PARQUET_DIR = ROOT / "data/1d/1d"


def main():
    parser = argparse.ArgumentParser(description="Run optimal alpha strategy")
    parser.add_argument("--strategy", default="v2c_5d",
                        choices=["v2c_5d", "composite10d"],
                        help="v2c_5d=futures+5day (max alpha) | composite10d=equity+10day")
    parser.add_argument("--oos-start", default="2025-01-01")
    args = parser.parse_args()

    oos_start = pd.Timestamp(args.oos_start, tz="UTC")

    # Strategy parameters
    if args.strategy == "v2c_5d":
        signal_type = "v2c"
        rebalance_days = 5
        decile = 0.10
        cost_bps = 7.26
        cost_label = "NSE Futures (7.26bps)"
        print("="*65)
        print("  RUNNING: v2c model | 5-day | top 10% | futures | equal-wt")
        print("  Expected: +20% excess vs NIFTY/year | IR ≈ 1.3")
    else:
        signal_type = "composite"
        rebalance_days = 10
        decile = 0.10
        cost_bps = 27.35
        cost_label = "NSE Equity (27.35bps)"
        print("="*65)
        print("  RUNNING: composite signal | 10-day | top 10% | equity | equal-wt")
        print("  Expected: +8.6% excess vs NIFTY/year | IR ≈ 1.1")
    print("="*65)

    # Load dataset
    for ds_path in [
        ROOT / "artifacts/datasets/v2e_selected/data.parquet",
        ROOT / "artifacts/datasets/v2c_cs_regime/data.parquet",
    ]:
        if ds_path.exists():
            df = pd.read_parquet(str(ds_path))
            break
    else:
        raise FileNotFoundError("No dataset found")
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
    df_oos = df[df.index >= oos_start].copy()
    print(f"\nOOS data: {len(df_oos):,} rows | {df_oos['symbol'].nunique()} symbols")

    # Compute signal scores
    print(f"Computing {signal_type} scores...")
    if signal_type == "v2c":
        models = sorted(ROOT.glob("artifacts/v2_model/*/model.pkl"))
        if not models:
            raise FileNotFoundError("No v2 model found. Run: make v2-train")
        with open(models[-1], "rb") as f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = pickle.load(f)
        feats = [f for f in m["feature_names"] if f in df_oos.columns]
        from src.features.normalizer import FeatureNormalizer
        X = df_oos[feats].fillna(0.0)
        if m.get("normalizer_state"):
            norm = FeatureNormalizer()
            norm.load_state(m["normalizer_state"])
            X = norm.transform_known(X).fillna(0.0)
        raw_scores = m["estimator"].predict(X.to_numpy(dtype=float))
        score_series = pd.Series(raw_scores, index=df_oos.index)
        model_label = models[-1].parent.name
    else:
        feats_w = [("cs_mom_12_1", 0.038), ("cs_neg_ret_1", 0.022), ("cs_pct_from_52w_high", 0.026)]
        total_w = sum(w for _, w in feats_w)
        score_series = sum(
            df_oos.get(f, pd.Series(0.5, index=df_oos.index)).fillna(0.5) * w
            for f, w in feats_w
        ) / total_w
        model_label = "IC-weighted composite"

    # CS rank per date
    df_oos["_score"] = score_series
    scores_pivot = df_oos.pivot_table(
        index=df_oos.index, columns="symbol", values="_score", aggfunc="last"
    ).rank(axis=1, pct=True)
    print(f"  Model: {model_label}")

    # Report IC if label available
    if "label_v2b" in df_oos.columns:
        valid = df_oos["label_v2b"].notna()
        ic, pval = spearmanr(
            score_series[valid].to_numpy(),
            df_oos.loc[valid, "label_v2b"].to_numpy()
        )
        print(f"  OOS IC vs rank label: {ic:+.4f} (p={pval:.4f})")

    # Load OHLCV
    print("Loading OHLCV...")
    ohlcv = {}
    for pq in sorted(PARQUET_DIR.glob("*.parquet")):
        sym = pq.stem
        if sym in ("NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"):
            continue
        try:
            raw = pd.read_parquet(str(pq))
            if raw.index.tz is None:
                raw.index = raw.index.tz_localize("UTC")
            ohlcv[sym] = raw[raw.index >= oos_start]
        except:
            pass
    print(f"  {len(ohlcv)} symbols loaded")

    # NIFTY benchmark
    nifty_close = None
    if (PARQUET_DIR / "NIFTY.parquet").exists():
        nf = pd.read_parquet(str(PARQUET_DIR / "NIFTY.parquet"))
        if nf.index.tz is None:
            nf.index = nf.index.tz_localize("UTC")
        nifty_close = nf["close"]

    # Pre-compute aligned returns
    print("Pre-computing aligned returns...")
    aligned = {}
    for sym, raw in ohlcv.items():
        c = raw["close"].astype(float)
        o = (raw["open"] if "open" in raw.columns else raw["close"]).astype(float)
        ret = c.shift(-rebalance_days) / o.shift(-1) - 1.0
        aligned[sym] = ret.reindex(scores_pivot.index)

    nifty_aligned = None
    if nifty_close is not None:
        nifty_aligned = (nifty_close.shift(-rebalance_days) / nifty_close.shift(-1) - 1.0
                         ).reindex(scores_pivot.index)

    # Run backtest
    print(f"\nRunning {rebalance_days}-day rebalance backtest...")
    cost_f = cost_bps / 10_000
    dates = sorted(scores_pivot.index.unique())
    reb_dates = dates[::rebalance_days]

    long_rets, nifty_rets, period_info = [], [], []

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
        avg = float(np.mean(rets))
        long_rets.append(avg)

        nif = 0.0
        if nifty_aligned is not None and rb_date in nifty_aligned.index:
            nv = nifty_aligned.loc[rb_date]
            if pd.notna(nv):
                nif = float(nv)
        nifty_rets.append(nif)
        period_info.append({"date": str(rb_date)[:10], "n_long": len(rets),
                             "avg_ret": round(avg*100,3), "nifty_ret": round(nif*100,3)})

    la = np.array(long_rets)
    na = np.array(nifty_rets)
    ppy = 252 / rebalance_days
    net = la - cost_f
    exc = net - na
    ann_ret = float(net.mean()) * ppy
    ann_exc = float(exc.mean()) * ppy
    vol_exc = float(exc.std()) * np.sqrt(ppy)
    ir = ann_exc / (vol_exc + 1e-10)
    eq = np.cumprod(1 + exc)
    mdd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())

    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS")
    print(f"{'='*65}")
    print(f"  Periods evaluated:       {len(la)}")
    print(f"  Avg stocks per period:   {np.mean([p['n_long'] for p in period_info]):.0f}")
    print(f"  Cost model:              {cost_label}")
    print()
    print(f"  Portfolio return:        {ann_ret*100:+.2f}%/year")
    print(f"  NIFTY benchmark:         {float(na.mean())*ppy*100:+.2f}%/year")
    print(f"  GROSS alpha vs NIFTY:    {float((la-na).mean())*ppy*100:+.2f}%/year")
    print(f"  NET excess vs NIFTY:     {ann_exc*100:+.2f}%/year  ← KEY METRIC")
    print(f"  Information Ratio:       {ir:+.3f}")
    print(f"  Win rate (excess>NIFTY): {float((exc>0).mean()):.1%}")
    print(f"  Max drawdown:            {mdd:.2%}")
    print()
    if ann_exc > 0:
        print(f"  ✓ PROFITABLE: +{ann_exc*100:.2f}%/year excess vs NIFTY")
    else:
        print(f"  ✗ NOT PROFITABLE: {ann_exc*100:.2f}%/year vs NIFTY")
    print(f"{'='*65}")

    # Save results
    out_dir = ROOT / "artifacts/best_strategy"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "strategy": args.strategy,
        "signal": model_label,
        "rebalance_days": rebalance_days,
        "decile": decile,
        "cost_bps": cost_bps,
        "n_periods": len(la),
        "ann_total_return": round(ann_ret*100, 2),
        "ann_excess_return": round(ann_exc*100, 2),
        "information_ratio": round(ir, 3),
        "win_rate_excess": round(float((exc>0).mean()), 3),
        "max_drawdown": round(mdd, 4),
        "profitable": ann_exc > 0,
    }
    (out_dir / f"{args.strategy}_results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(period_info).to_csv(str(out_dir / f"{args.strategy}_periods.csv"), index=False)
    print(f"\n✓ Saved: artifacts/best_strategy/{args.strategy}_results.json")


if __name__ == "__main__":
    main()
