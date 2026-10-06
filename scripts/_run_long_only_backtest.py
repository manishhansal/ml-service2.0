"""
Long-only strategy backtest: go LONG the top decile predicted by v2c model.

Key insight: the v2c model's LONG signals work (+10.91%/year)
             but SHORT signals fail in bull markets.

This backtest measures:
  - Raw return of long portfolio
  - Excess return vs NIFTY (alpha)
  - Risk-adjusted performance (Sharpe, Information Ratio)
  - Performance at equity AND futures costs
"""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PARQUET_DIR = ROOT / "data/1d/1d"
OOS_START = pd.Timestamp("2025-01-01", tz="UTC")
REBALANCE_DAYS = 7   # weekly
DECILE = 0.10        # top 10% = LONG
COST_EQUITY_BPS = 27.35
COST_FUTURES_BPS = 7.26


def load_latest_v2_model():
    candidates = sorted(ROOT.glob("artifacts/v2_model/*/model.pkl"))
    if not candidates:
        raise FileNotFoundError("No v2 model found")
    latest = candidates[-1]
    print(f"Loading model: {latest.parent.name}")
    with open(latest, "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pickle.load(f)


def generate_cs_rank_scores(model_dict, df_oos):
    estimator = model_dict["estimator"]
    feature_names = model_dict["feature_names"]
    norm_state = model_dict.get("normalizer_state", {})
    avail = [f for f in feature_names if f in df_oos.columns]
    X = df_oos[avail].fillna(0.0)
    if norm_state:
        try:
            from src.features.normalizer import FeatureNormalizer
            norm = FeatureNormalizer()
            norm.load_state(norm_state)
            X = norm.transform_known(X).fillna(0.0)
        except Exception:
            pass
    raw = estimator.predict(X.to_numpy(dtype=float))
    df_s = df_oos[["symbol"]].copy()
    df_s["raw_score"] = raw
    pivot = df_s.pivot_table(index=df_oos.index, columns="symbol", values="raw_score", aggfunc="last")
    cs_rank = pivot.rank(axis=1, pct=True)
    return cs_rank


def main():
    print("="*65)
    print("  LONG-ONLY STRATEGY BACKTEST (v2c model, top decile)")
    print("="*65)

    model_dict = load_latest_v2_model()

    # Load dataset
    v2c_path = ROOT / "artifacts/datasets/v2c_cs_regime/data.parquet"
    df = pd.read_parquet(str(v2c_path))
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df_oos = df[df.index >= OOS_START].copy()
    print(f"OOS: {len(df_oos):,} rows | {df_oos['symbol'].nunique()} symbols")

    # Generate cross-sectional rank scores
    print("Generating model scores...")
    cs_rank = generate_cs_rank_scores(model_dict, df_oos)

    # Load OHLCV for return calculation
    print("Loading OHLCV...")
    symbol_data = {}
    for sym in cs_rank.columns:
        pq = PARQUET_DIR / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(str(pq))
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        symbol_data[sym] = raw

    # Load NIFTY benchmark
    nifty_pq = PARQUET_DIR / "NIFTY.parquet"
    nifty_data = None
    if nifty_pq.exists():
        nf = pd.read_parquet(str(nifty_pq))
        if nf.index.tz is None:
            nf.index = nf.index.tz_localize("UTC")
        nifty_data = nf["close"]

    # Pre-compute aligned returns (no ffill)
    print("Pre-computing aligned returns...")
    aligned_returns = {}
    for sym, raw in symbol_data.items():
        close = raw["close"].astype(float)
        open_ = (raw["open"] if "open" in raw.columns else raw["close"]).astype(float)
        entry = open_.shift(-1)
        exit_ = close.shift(-REBALANCE_DAYS)
        ret = exit_ / entry - 1.0
        aligned_returns[sym] = ret.reindex(cs_rank.index)

    nifty_aligned = None
    if nifty_data is not None:
        nc_entry = nifty_data.shift(-1)
        nc_exit  = nifty_data.shift(-REBALANCE_DAYS)
        nifty_aligned = (nc_exit / nc_entry - 1.0).reindex(cs_rank.index)

    # Run long-only backtest
    print(f"\nRunning backtest (top {DECILE:.0%} LONG, weekly rebalance)...")
    dates = sorted(cs_rank.index.unique())
    rebalance_dates = dates[::REBALANCE_DAYS]

    long_port_returns = []
    nifty_period_returns = []
    top_decile_sizes = []

    for rb_date in rebalance_dates[:-1]:
        if rb_date not in cs_rank.index:
            continue
        day_ranks = cs_rank.loc[rb_date].dropna()
        if len(day_ranks) < 10:
            continue

        threshold = day_ranks.quantile(1 - DECILE)
        long_syms = day_ranks[day_ranks >= threshold].index.tolist()
        top_decile_sizes.append(len(long_syms))

        long_rets = [float(aligned_returns[s].reindex([rb_date]).iloc[0])
                     for s in long_syms
                     if s in aligned_returns and pd.notna(aligned_returns[s].reindex([rb_date]).iloc[0])]

        if not long_rets:
            continue

        avg_long = float(np.mean(long_rets))
        long_port_returns.append(avg_long)

        if nifty_aligned is not None and rb_date in nifty_aligned.index:
            nf_val = nifty_aligned.loc[rb_date]
            nifty_period_returns.append(float(nf_val) if pd.notna(nf_val) else 0.0)
        else:
            nifty_period_returns.append(0.0)

    if not long_port_returns:
        print("No valid periods")
        return

    long_arr  = np.array(long_port_returns)
    nifty_arr = np.array(nifty_period_returns)

    periods_per_year = 252 / REBALANCE_DAYS

    print(f"\n{'='*65}")
    print(f"  LONG-ONLY PORTFOLIO RESULTS (GROSS, no cost)")
    print(f"{'='*65}")
    print(f"  Periods: {len(long_arr)}")
    print(f"  Avg stocks/period: {np.mean(top_decile_sizes):.0f}")
    print()
    print(f"  Portfolio:         {float(long_arr.mean())*periods_per_year*100:.2f}%/year")
    print(f"  NIFTY benchmark:   {float(nifty_arr.mean())*periods_per_year*100:.2f}%/year")
    print(f"  Excess vs NIFTY:   {float((long_arr - nifty_arr).mean())*periods_per_year*100:.2f}%/year")
    print()
    excess = long_arr - nifty_arr
    print(f"  Excess win rate:   {float((excess > 0).mean()):.1%}")
    print()

    for cost_label, cost_bps in [("Equity (27.35bps)", COST_EQUITY_BPS), ("Futures (7.26bps)", COST_FUTURES_BPS)]:
        cost_frac = cost_bps / 10_000
        net = long_arr - cost_frac
        net_excess = net - nifty_arr
        ann_net = float(net.mean()) * periods_per_year
        ann_exc = float(net_excess.mean()) * periods_per_year
        vol = float(net_excess.std()) * np.sqrt(periods_per_year)
        ir = ann_exc / (vol + 1e-10)
        max_dd = 0.0
        equity_curve = np.cumprod(1 + net_excess)
        running_max = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - running_max) / running_max
        max_dd = float(dd.min())

        print(f"  Net return [{cost_label}]: {ann_net*100:.2f}%/year")
        print(f"  Excess return net:      {ann_exc*100:.2f}%/year")
        print(f"  Info ratio (IR):        {ir:.3f}")
        print(f"  Win rate (excess > 0):  {float((net_excess > 0).mean()):.1%}")
        print(f"  Max drawdown:           {max_dd:.2%}")
        print(f"  {'✓ PROFITABLE (excess > 0)' if ann_exc > 0 else '✗ NOT PROFITABLE'}")
        print()

    # IC verification
    valid_mask = df_oos["label_v2b"].notna() if "label_v2b" in df_oos.columns else pd.Series(False, index=df_oos.index)
    if valid_mask.sum() > 100:
        avail = [f for f in model_dict["feature_names"] if f in df_oos.columns]
        X = df_oos.loc[valid_mask, avail].fillna(0.0).to_numpy(dtype=float)
        y = df_oos.loc[valid_mask, "label_v2b"].to_numpy(dtype=float)
        scores = model_dict["estimator"].predict(X)
        ic, pval = spearmanr(scores, y)
        print(f"  Model IC (rank label): {ic:.4f} (p={pval:.4f})")

    print("="*65)


if __name__ == "__main__":
    main()
