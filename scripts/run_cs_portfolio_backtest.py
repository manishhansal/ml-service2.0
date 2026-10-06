"""
run_cs_portfolio_backtest.py — Cross-sectional long-short portfolio backtest.

This is the CORRECT evaluation method for a ranking model with IC=0.020.

Strategy:
  Each rebalance day:
    - Rank all eligible symbols by v2 model score (predicted 7-day excess return)
    - LONG top DECILE (10%) of symbols
    - SHORT bottom DECILE (10%) of symbols
    - Hold for 7 trading days
    - Rebalance every 7 days (non-overlapping windows)
  
  This produces a market-neutral long-short portfolio that captures the
  cross-sectional spread between predicted winners and losers.

Expected performance with IC=0.020:
  Gross alpha per week ≈ IC × cross_sectional_std ≈ 0.020 × 2.35 = 0.047 vol-units
  = 0.047 × avg_vol ≈ 0.047 × 1.5% = 0.070%/week = 3.6%/year (gross)
  
  At FUTURES costs (7bps RT): net ≈ 3.6% - 0.4% = +3.2%/year
  At EQUITY costs (27bps RT): net ≈ 3.6% - 1.4% = +2.2%/year

Usage:
    PYTHONPATH=. python3 scripts/run_cs_portfolio_backtest.py
    PYTHONPATH=. python3 scripts/run_cs_portfolio_backtest.py --cost futures
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "artifacts" / "cs_portfolio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_EQUITY_BPS  = 27.35
COST_FUTURES_BPS = 7.26
DECILE = 0.10   # top/bottom 10% for long/short
REBALANCE_DAYS = 7  # rebalance every 7 trading days (=weekly)
OOS_START = "2025-01-01"


def load_v2_model():
    """Load the v2 regression model."""
    candidates = sorted(ROOT.glob("artifacts/v2_model/*/model.pkl"))
    if not candidates:
        raise FileNotFoundError("No v2 model found in artifacts/v2_model/")
    latest = candidates[-1]
    print(f"Loading model: {latest.parent.name}")
    with open(latest, "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pickle.load(f)


def generate_scores(model_dict: dict, df_oos: pd.DataFrame) -> pd.DataFrame:
    """Generate per-symbol per-date cross-sectional rank scores."""
    estimator    = model_dict["estimator"]
    feature_names = model_dict["feature_names"]
    norm_state   = model_dict.get("normalizer_state", {})
    invert       = model_dict.get("invert_scores", False)

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
    if invert:
        raw = -raw

    df_scores = df_oos[["symbol"]].copy()
    df_scores["raw_score"] = raw

    # Pivot to dates × symbols
    pivot = df_scores.pivot_table(index=df_oos.index, columns="symbol",
                                  values="raw_score", aggfunc="last")
    # Cross-sectional rank percentile [0,1] per day
    cs_rank = pivot.rank(axis=1, pct=True)
    return cs_rank, pivot


def run_cs_backtest(
    model_dict: dict,
    ohlcv_dir: Path,
    cost_bps: float = COST_EQUITY_BPS,
    decile: float = DECILE,
    rebalance_days: int = REBALANCE_DAYS,
    oos_start: str = OOS_START,
    min_symbols: int = 20,
) -> dict:
    """
    Run cross-sectional long-short portfolio backtest.
    Returns a dict with equity curve, per-period P&L, and summary metrics.
    """
    print(f"\nLoading OOS dataset...")
    ds_files = sorted(ROOT.glob("artifacts/datasets/ds-1d-*/data.parquet"))
    if not ds_files:
        raise FileNotFoundError("No dataset found")

    # Priority: v2c (has all new features + proper rank labels)
    for ds_candidate in [
        ROOT / "artifacts" / "datasets" / "v2c_cs_regime" / "data.parquet",
        ROOT / "artifacts" / "datasets" / "v2b_ranked" / "data.parquet",
        ROOT / "artifacts" / "datasets" / "v2_labeled" / "data.parquet",
    ]:
        if ds_candidate.exists():
            df_full = pd.read_parquet(str(ds_candidate))
            print(f"  Using dataset: {ds_candidate.parent.name}")
            break
    else:
        df_full = pd.read_parquet(str(ds_files[-1]))
        print(f"  Using latest dataset: {ds_files[-1].parent.name}")

    # Standardize label column name
    if "label_v2b" in df_full.columns and "label_v2" not in df_full.columns:
        df_full["label_v2"] = df_full["label_v2b"]

    if df_full.index.tz is None:
        df_full.index = df_full.index.tz_localize("UTC")
    df_oos = df_full[df_full.index >= pd.Timestamp(oos_start, tz="UTC")].copy()
    print(f"  OOS rows: {len(df_oos):,} | Symbols: {df_oos['symbol'].nunique()}")

    print("Generating model scores...")
    cs_rank, raw_scores = generate_scores(model_dict, df_oos)
    print(f"  Score matrix: {cs_rank.shape[0]} dates × {cs_rank.shape[1]} symbols")

    # Load OHLCV for actual return calculation using ALIGNED time-series approach
    print("Loading OHLCV for return calculation (aligned time-series method)...")
    # Store complete OHLCV per symbol for vectorized calculation
    ohlcv_indexed = {}   # symbol → DataFrame indexed by scoring date

    for sym in cs_rank.columns:
        pq = ohlcv_dir / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(str(pq))
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        ohlcv_indexed[sym] = raw

    print(f"  Loaded {len(ohlcv_indexed)} symbol parquets")

    # Load NIFTY for beta adjustment
    nifty_pq = ohlcv_dir / "NIFTY.parquet"
    nifty_close = None
    if nifty_pq.exists():
        nf = pd.read_parquet(str(nifty_pq))
        if nf.index.tz is None:
            nf.index = nf.index.tz_localize("UTC")
        nifty_close = nf["close"]

    print(f"\nRunning CS long-short backtest (aligned time-series method)...")
    print(f"  Decile: {decile:.0%} | Rebalance: every {rebalance_days} scoring days")
    print(f"  Cost: {cost_bps:.2f} bps round-trip")
    print(f"  Return: open[T+1]-to-close[T+7] (aligned to model signal date)")

    cost_frac = cost_bps / 10_000.0
    
    # Pre-compute aligned returns per symbol WITHOUT forward-filling (avoids stale close bug)
    print("  Pre-computing aligned returns per symbol (no ffill)...")
    aligned_returns = {}  # symbol → Series(scoring_date → return)

    for sym in cs_rank.columns:
        pq = ohlcv_dir / f"{sym}.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(str(pq))
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")

        # Work in symbol's OWN index (no reindex, no ffill — avoids stale close bug)
        close = raw["close"].astype(float)
        open_ = (raw["open"] if "open" in raw.columns else raw["close"]).astype(float)

        # Compute forward return: open[T+1] to close[T+rebalance_days]
        entry_price = open_.shift(-1)       # open of NEXT trading day
        exit_price  = close.shift(-rebalance_days)  # close of Nth trading day
        ret = exit_price / entry_price - 1.0

        # Reindex to scoring dates WITHOUT ffill — NaN if date not in symbol data
        ret_aligned = ret.reindex(cs_rank.index)
        aligned_returns[sym] = ret_aligned

    # NIFTY aligned returns (same approach)
    nifty_fwd_aligned = None
    if nifty_close is not None:
        nifty_close_data = nifty_close
        nc_entry = nifty_close_data.shift(-1)
        nc_exit  = nifty_close_data.shift(-rebalance_days)
        nifty_ret_raw = nc_exit / nc_entry - 1.0
        nifty_fwd_aligned = nifty_ret_raw.reindex(cs_rank.index)

    dates = sorted(cs_rank.index.unique())
    rebalance_dates = dates[::rebalance_days]

    portfolio_returns = []
    long_only_returns = []
    short_only_returns = []
    nifty_returns = []
    detailed_periods = []
    n_symbols_per_side_avg = []

    for i, rb_date in enumerate(rebalance_dates[:-1]):
        if rb_date not in cs_rank.index:
            continue
        day_ranks = cs_rank.loc[rb_date].dropna()
        if len(day_ranks) < max(min_symbols, int(1 / decile) + 1):
            continue

        threshold_long  = day_ranks.quantile(1 - decile)
        threshold_short = day_ranks.quantile(decile)

        long_syms  = day_ranks[day_ranks >= threshold_long].index.tolist()
        short_syms = day_ranks[day_ranks <= threshold_short].index.tolist()
        n_long  = len(long_syms)
        n_short = len(short_syms)
        n_symbols_per_side_avg.append((n_long + n_short) / 2)

        if n_long == 0 or n_short == 0:
            continue

        long_rets  = [float(aligned_returns[s].loc[rb_date])
                      for s in long_syms
                      if s in aligned_returns and pd.notna(aligned_returns[s].reindex([rb_date]).iloc[0])]
        short_rets = [float(aligned_returns[s].loc[rb_date])
                      for s in short_syms
                      if s in aligned_returns and pd.notna(aligned_returns[s].reindex([rb_date]).iloc[0])]

        if not long_rets or not short_rets:
            continue

        nifty_ret = 0.0
        if nifty_fwd_aligned is not None and rb_date in nifty_fwd_aligned.index:
            nf_val = nifty_fwd_aligned.loc[rb_date]
            if pd.notna(nf_val):
                nifty_ret = float(nf_val)

        avg_long  = float(np.mean(long_rets))
        avg_short = float(np.mean(short_rets))
        ls_return = (avg_long - avg_short) / 2 - cost_frac

        portfolio_returns.append(ls_return)
        long_only_returns.append(avg_long - cost_frac / 2)
        short_only_returns.append(-avg_short - cost_frac / 2)
        nifty_returns.append(nifty_ret)

        detailed_periods.append({
            "rebalance_date": str(rb_date)[:10],
            "n_long": n_long, "n_short": n_short,
            "avg_long_ret": round(avg_long * 100, 4),
            "avg_short_ret": round(avg_short * 100, 4),
            "ls_return": round(ls_return * 100, 4),
            "nifty_ret": round(nifty_ret * 100, 4),
        })

    if not portfolio_returns:
        print("  No valid periods computed")
        return {}

    pnl = np.array(portfolio_returns)
    long_pnl = np.array(long_only_returns)
    short_pnl = np.array(short_only_returns)

    # Annualization
    periods_per_year = 252 / rebalance_days
    ann_return = float(np.mean(pnl)) * periods_per_year
    ann_vol    = float(np.std(pnl)) * np.sqrt(periods_per_year)
    sharpe     = ann_return / (ann_vol + 1e-10)
    win_rate   = float((pnl > 0).mean())
    max_dd     = 0.0
    equity     = np.cumprod(1 + pnl)
    if len(equity) > 1:
        rolling_max = np.maximum.accumulate(equity)
        dd = (equity - rolling_max) / rolling_max
        max_dd = float(dd.min())

    # IC on held-out OOS data  
    ic_oos = 0.0
    if "label_v2" in df_oos.columns and "symbol" in df_oos.columns:
        raw_flat = raw_scores.stack().rename("raw_score").reset_index()
        raw_flat.columns = ["ts", "symbol", "raw_score"]
        raw_flat = raw_flat.set_index("ts")
        merged = df_oos[["symbol", "label_v2"]].join(raw_flat[["symbol", "raw_score"]], rsuffix="_r", how="inner")
        merged = merged[merged["symbol"] == merged["symbol_r"]]
        if len(merged) > 100:
            ic_oos, ic_pval = spearmanr(merged["raw_score"], merged["label_v2"])
            print(f"  Verified IC (raw score vs label_v2): {ic_oos:.4f} (p={ic_pval:.4f})")

    result = {
        "strategy":           "cross_sectional_long_short",
        "model":              "v2_lgbm_regressor",
        "oos_start":          oos_start,
        "cost_bps":           cost_bps,
        "cost_model":         "equity" if cost_bps > 15 else "futures",
        "decile":             decile,
        "rebalance_days":     rebalance_days,
        "n_periods":          len(pnl),
        "ic_oos":             round(float(ic_oos), 4),
        "ann_return_pct":     round(ann_return * 100, 4),
        "ann_vol_pct":        round(ann_vol * 100, 4),
        "sharpe":             round(sharpe, 4),
        "win_rate":           round(win_rate, 4),
        "max_drawdown":       round(max_dd, 4),
        "mean_period_return": round(float(pnl.mean()) * 100, 4),
        "std_period_return":  round(float(pnl.std()) * 100, 4),
        "avg_n_symbols":      round(float(np.mean(n_symbols_per_side_avg)), 1),
        "long_only_ann_ret":  round(float(long_pnl.mean()) * periods_per_year * 100, 4),
        "short_only_ann_ret": round(float(short_pnl.mean()) * periods_per_year * 100, 4),
        "profitable":         ann_return > 0 and sharpe > 0,
        "equity_curve":       equity.tolist(),
        "detailed_periods":   detailed_periods,
    }

    print(f"\n{'='*60}")
    print(f"  CROSS-SECTIONAL LONG-SHORT BACKTEST RESULTS")
    print(f"  Model: v2 LGBMRegressor | Cost: {cost_bps:.1f}bps {result['cost_model']}")
    print(f"{'='*60}")
    print(f"  Periods evaluated:    {len(pnl)}")
    print(f"  Avg symbols/side:     {result['avg_n_symbols']:.0f}")
    print(f"  Win rate:             {win_rate:.1%}")
    print(f"  Mean period return:   {result['mean_period_return']:.3f}%")
    print(f"  Ann. return:          {ann_return*100:.2f}%")
    print(f"  Ann. volatility:      {ann_vol*100:.2f}%")
    print(f"  Sharpe ratio:         {sharpe:.3f}")
    print(f"  Max drawdown:         {max_dd:.2%}")
    print(f"  Long-only ann ret:    {result['long_only_ann_ret']:.2f}%")
    print(f"  Short-only ann ret:   {result['short_only_ann_ret']:.2f}%")
    print()
    if result["profitable"]:
        print(f"  ✓ PROFITABLE: Ann return = {ann_return*100:.2f}%, Sharpe = {sharpe:.3f}")
    else:
        print(f"  ✗ NOT PROFITABLE: Ann return = {ann_return*100:.2f}%, Sharpe = {sharpe:.3f}")
    print(f"{'='*60}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Cross-sectional L/S portfolio backtest")
    parser.add_argument("--cost", default="equity", choices=["equity", "futures"])
    parser.add_argument("--decile", type=float, default=0.10)
    parser.add_argument("--rebalance", type=int, default=7)
    parser.add_argument("--oos-start", default=OOS_START)
    args = parser.parse_args()

    cost_bps = COST_EQUITY_BPS if args.cost == "equity" else COST_FUTURES_BPS
    model_dict = load_v2_model()
    ohlcv_dir = ROOT / "data" / "1d" / "1d"

    # Run at BOTH cost levels
    print("\n" + "="*60)
    print("  EQUITY COSTS (27.35 bps RT)")
    print("="*60)
    result_equity = run_cs_backtest(
        model_dict=model_dict,
        ohlcv_dir=ohlcv_dir,
        cost_bps=COST_EQUITY_BPS,
        decile=args.decile,
        rebalance_days=args.rebalance,
        oos_start=args.oos_start,
    )

    print("\n" + "="*60)
    print("  FUTURES COSTS (7.26 bps RT)")
    print("="*60)
    result_futures = run_cs_backtest(
        model_dict=model_dict,
        ohlcv_dir=ohlcv_dir,
        cost_bps=COST_FUTURES_BPS,
        decile=args.decile,
        rebalance_days=args.rebalance,
        oos_start=args.oos_start,
    )

    # Save
    for res, name in [(result_equity, "cs_equity"), (result_futures, "cs_futures")]:
        if res:
            out = OUT_DIR / f"{name}_backtest.json"
            detailed = res.pop("detailed_periods", [])
            equity_curve = res.pop("equity_curve", [])
            out.write_text(json.dumps(res, indent=2, default=str))
            if detailed:
                pd.DataFrame(detailed).to_csv(str(OUT_DIR / f"{name}_periods.csv"), index=False)
            if equity_curve:
                pd.Series(equity_curve).to_csv(str(OUT_DIR / f"{name}_equity_curve.csv"))
            print(f"✓ Saved: {out}")

    # Summary comparison
    print("\n" + "="*60)
    print("  SUMMARY COMPARISON")
    print("="*60)
    print(f"  {'Metric':30s} {'Equity':>12s} {'Futures':>12s}")
    print(f"  {'-'*56}")
    for key in ["ann_return_pct", "sharpe", "win_rate", "max_drawdown"]:
        eq_val  = result_equity.get(key, 0)
        fut_val = result_futures.get(key, 0)
        print(f"  {key:30s} {eq_val:>12.3f} {fut_val:>12.3f}")
    print(f"\n  Equity:  {'✓ PROFITABLE' if result_equity.get('profitable') else '✗ NOT PROFITABLE'}")
    print(f"  Futures: {'✓ PROFITABLE' if result_futures.get('profitable') else '✗ NOT PROFITABLE'}")


if __name__ == "__main__":
    main()
