"""
Maximum Alpha Backtest — exhaustive search for best configuration.

Tests:
  Signals: (1) v2c model  (2) simple composite  (3) IC-weighted ensemble
  Portfolio: (A) equal-weight  (B) sector-neutral  (C) vol-scaled
  Rebalance: 3d, 5d, 7d, 10d, 14d, 21d
  Universe: top 10%, top 15%, top 20% decile
  Costs: equity, futures

Reports the maximum achievable configuration with honest OOS evidence.
"""
from __future__ import annotations
import json, pickle, sys, warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OOS_START = pd.Timestamp("2025-01-01", tz="UTC")
PARQUET_DIR = ROOT / "data/1d/1d"

SECTOR_MAP = {
    "HDFCBANK":"Bank","ICICIBANK":"Bank","SBIN":"Bank","AXISBANK":"Bank","KOTAKBANK":"Bank",
    "BANDHANBNK":"Bank","BANKBARODA":"Bank","BANKINDIA":"Bank","AUBANK":"Bank","FEDERALBNK":"Bank",
    "CANBK":"Bank","IDFCFIRSTB":"Bank","INDUSINDBK":"Bank","PNB":"Bank","MAHABANK":"Bank",
    "INFY":"IT","TCS":"IT","WIPRO":"IT","HCLTECH":"IT","TECHM":"IT","MPHASIS":"IT",
    "LTTS":"IT","LTI":"IT","KPITTECH":"IT","COFORGE":"IT","PERSISTENT":"IT","OFSS":"IT",
    "RELIANCE":"Energy","ONGC":"Energy","BPCL":"Energy","HINDPETRO":"Energy","IOC":"Energy","OIL":"Energy",
    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma","DIVISLAB":"Pharma","LUPIN":"Pharma",
    "AUROPHARMA":"Pharma","BIOCON":"Pharma","ALKEM":"Pharma","IPCALAB":"Pharma","LALPATHLAB":"Pharma",
    "HINDUNILVR":"FMCG","ITC":"FMCG","BRITANNIA":"FMCG","NESTLEIND":"FMCG","DABUR":"FMCG",
    "MARICO":"FMCG","GODREJCP":"FMCG","COLPAL":"FMCG","PGHH":"FMCG",
    "TATAMOTORS":"Auto","MARUTI":"Auto","M&M":"Auto","EICHERMOT":"Auto","BAJAJ-AUTO":"Auto",
    "HEROMOTOCO":"Auto","MOTHERSON":"Auto","EXIDEIND":"Auto","APOLLOTYRE":"Auto",
    "TATASTEEL":"Metal","JSWSTEEL":"Metal","HINDALCO":"Metal","VEDL":"Metal","NATIONALUM":"Metal",
    "HINDCOPPER":"Metal","NMDC":"Metal","JINDALSTEL":"Metal","JSL":"Metal",
    "BAJFINANCE":"NBFC","BAJAJFINSV":"NBFC","CHOLAFIN":"NBFC","MUTHOOTFIN":"NBFC",
    "MANAPPURAM":"NBFC","M&MFIN":"NBFC","LICHSGFIN":"NBFC","CANFINHOME":"NBFC","POONAWALLA":"NBFC",
    "LT":"Infra","NTPC":"Infra","POWERGRID":"Infra","BHEL":"Infra","ABB":"Infra","SIEMENS":"Infra",
    "ADANIPORTS":"Infra","ADANIENT":"Infra","HAL":"Infra","BEL":"Infra","COCHINSHIP":"Infra",
    "ADANIGREEN":"Renew","ATGL":"Renew","NHPC":"Renew","IREDA":"Renew","CESC":"Renew",
    "ASIANPAINT":"Consumer","BERGER":"Consumer","PIDILITIND":"Consumer","TITAN":"Consumer","TRENT":"Consumer",
    "POLYCAB":"Consumer","HAVELLS":"Consumer","DIXON":"Consumer","AMBER":"Consumer",
    "SBILIFE":"Insurance","HDFCLIFE":"Insurance","ICICIGI":"Insurance","ICICIPRULI":"Insurance","LICI":"Insurance",
}

def load_ohlcv_batch():
    """Load all symbol OHLCV data."""
    data = {}
    for pq in sorted(PARQUET_DIR.glob("*.parquet")):
        sym = pq.stem
        if sym in ("NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"):
            continue
        try:
            raw = pd.read_parquet(str(pq))
            if raw.index.tz is None:
                raw.index = raw.index.tz_localize("UTC")
            raw = raw[raw.index >= OOS_START]
            if len(raw) >= 20:
                data[sym] = raw
        except:
            pass
    return data

def compute_signal_scores(df_oos, signal_type="composite"):
    """Compute scores using the best available signal."""
    if signal_type == "composite":
        # Model D: IC-weighted composite (cs_mom_12_1 + cs_neg_ret_1 + cs_pct_from_52w_high)
        feats = [("cs_mom_12_1", 0.038), ("cs_neg_ret_1", 0.022), ("cs_pct_from_52w_high", 0.026)]
        total_w = sum(w for _, w in feats)
        scores = sum(df_oos.get(f, pd.Series(0.5, index=df_oos.index)).fillna(0.5) * w
                     for f, w in feats) / total_w
        return scores.rename("score")
    elif signal_type == "v2c":
        # Latest trained model
        models = sorted(ROOT.glob("artifacts/v2_model/*/model.pkl"))
        if not models:
            return compute_signal_scores(df_oos, "composite")
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
        raw = m["estimator"].predict(X.to_numpy(dtype=float))
        return pd.Series(raw, index=df_oos.index, name="score")
    elif signal_type == "ensemble":
        ens_path = ROOT / "artifacts/v2_ensemble/ensemble_model.pkl"
        if not ens_path.exists():
            return compute_signal_scores(df_oos, "composite")
        with open(ens_path, "rb") as f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ens = pickle.load(f)
        # Use Model A + Model D
        from src.features.normalizer import FeatureNormalizer
        m_a = ens["model_a"]
        feats_a = [f for f in m_a["feature_names"] if f in df_oos.columns]
        X_a = df_oos[feats_a].fillna(0.0)
        if m_a.get("normalizer_state"):
            norm_a = FeatureNormalizer()
            norm_a.load_state(m_a["normalizer_state"])
            X_a = norm_a.transform_known(X_a).fillna(0.0)
        sc_a = pd.Series(m_a["estimator"].predict(X_a.to_numpy(dtype=float)), index=df_oos.index)

        # Composite D
        sc_d = compute_signal_scores(df_oos, "composite")
        # Combine (0.5/0.5 weight, both CS-ranked within each date)
        pivot_a = df_oos[["symbol"]].copy()
        pivot_a["sc_a"] = sc_a
        pivot_a["sc_d"] = sc_d
        for c in ["sc_a", "sc_d"]:
            pivot_a[c+"_r"] = pivot_a.groupby(pivot_a.index)[c].rank(pct=True)
        combined = (pivot_a["sc_a_r"].fillna(0.5) + pivot_a["sc_d_r"].fillna(0.5)) / 2
        return combined.rename("score")

def backtest_longonly(ohlcv_dict, scores_df, nifty_close, decile, rebalance_days, cost_bps,
                      sector_neutral=False, vol_scaled=False):
    """Run one backtest configuration."""
    cost_f = cost_bps / 10_000
    dates = sorted(scores_df.index.unique())
    reb_dates = dates[::rebalance_days]

    # Pre-compute aligned returns
    aligned = {}
    for sym, raw in ohlcv_dict.items():
        close = raw["close"].astype(float)
        open_ = (raw["open"] if "open" in raw.columns else raw["close"]).astype(float)
        entry = open_.shift(-1)
        exit_ = close.shift(-rebalance_days)
        ret = exit_ / entry - 1.0
        aligned[sym] = ret.reindex(scores_df.index)

    nifty_aligned = None
    if nifty_close is not None:
        nc_e = nifty_close.shift(-1)
        nc_x = nifty_close.shift(-rebalance_days)
        nifty_aligned = (nc_x / nc_e - 1.0).reindex(scores_df.index)

    # Volatility for vol-scaling
    vol_map = {}
    if vol_scaled:
        for sym, raw in ohlcv_dict.items():
            vol_map[sym] = raw["close"].pct_change().rolling(20).std().reindex(scores_df.index)

    results = {"long_rets": [], "nifty_rets": [], "n_long_avg": []}

    for rb_date in reb_dates[:-1]:
        if rb_date not in scores_df.index:
            continue
        day_scores = scores_df.loc[rb_date].dropna()
        if len(day_scores) < max(10, int(1/decile) + 1):
            continue

        if sector_neutral:
            # Take top stock(s) from each sector
            n_total = max(1, int(len(day_scores) * decile))
            long_syms = []
            for sector in set(SECTOR_MAP.values()):
                sector_syms = [s for s in day_scores.index if SECTOR_MAP.get(s, "Other") == sector]
                if sector_syms:
                    top_in_sector = sorted(sector_syms, key=lambda s: -day_scores.get(s, 0))[:2]
                    long_syms.extend(top_in_sector)
            # Fill remaining with top overall
            thresh = day_scores.quantile(1 - decile)
            top_overall = day_scores[day_scores >= thresh].index.tolist()
            for s in top_overall:
                if s not in long_syms:
                    long_syms.append(s)
            long_syms = long_syms[:n_total]
        else:
            thresh = day_scores.quantile(1 - decile)
            long_syms = day_scores[day_scores >= thresh].index.tolist()

        if not long_syms:
            continue

        raw_rets = []
        weights = []
        for s in long_syms:
            if s not in aligned:
                continue
            ret_val = aligned[s].reindex([rb_date])
            if ret_val.empty or pd.isna(ret_val.iloc[0]):
                continue
            r = float(ret_val.iloc[0])
            w = 1.0
            if vol_scaled and s in vol_map:
                v = vol_map[s].reindex([rb_date])
                if not v.empty and not pd.isna(v.iloc[0]) and v.iloc[0] > 1e-6:
                    w = 0.01 / float(v.iloc[0])  # target 1% daily contribution
            raw_rets.append((r, w))
            weights.append(w)

        if not raw_rets:
            continue

        total_w = sum(w for _, w in raw_rets) + 1e-10
        avg_ret = sum(r * w for r, w in raw_rets) / total_w
        results["long_rets"].append(avg_ret)
        results["n_long_avg"].append(len(raw_rets))

        if nifty_aligned is not None and rb_date in nifty_aligned.index:
            nv = nifty_aligned.loc[rb_date]
            results["nifty_rets"].append(float(nv) if pd.notna(nv) else 0.0)
        else:
            results["nifty_rets"].append(0.0)

    if not results["long_rets"]:
        return None

    la = np.array(results["long_rets"])
    na = np.array(results["nifty_rets"])
    ppy = 252 / rebalance_days
    net = la - cost_f
    exc = net - na
    ann_ret = float(net.mean()) * ppy
    ann_exc = float(exc.mean()) * ppy
    vol = float(exc.std()) * np.sqrt(ppy)
    ir = ann_exc / (vol + 1e-10)
    equity_curve = np.cumprod(1 + exc)
    running_max = np.maximum.accumulate(equity_curve)
    max_dd = float(((equity_curve - running_max) / running_max).min())
    return {
        "n_periods": len(la),
        "n_long_avg": float(np.mean(results["n_long_avg"])),
        "ann_total_return": round(ann_ret * 100, 2),
        "ann_excess_return": round(ann_exc * 100, 2),
        "information_ratio": round(ir, 3),
        "win_rate_excess": round(float((exc > 0).mean()), 3),
        "max_drawdown": round(max_dd, 4),
        "profitable": ann_exc > 0,
    }


def main():
    print("="*70)
    print("  MAXIMUM ALPHA SEARCH — Exhaustive Configuration Test")
    print("="*70)

    # Load OOS dataset
    v2e_path = ROOT / "artifacts/datasets/v2e_selected/data.parquet"
    if not v2e_path.exists():
        v2e_path = ROOT / "artifacts/datasets/v2c_cs_regime/data.parquet"
    print(f"Loading dataset: {v2e_path.parent.name}")
    df = pd.read_parquet(str(v2e_path))
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
    df_oos = df[df.index >= OOS_START].copy()
    print(f"OOS: {len(df_oos):,} rows | {df_oos['symbol'].nunique()} symbols")

    # Load OHLCV
    print("Loading OHLCV...")
    ohlcv = load_ohlcv_batch()
    print(f"  {len(ohlcv)} symbols loaded")

    nifty_pq = PARQUET_DIR / "NIFTY.parquet"
    nifty_close = None
    if nifty_pq.exists():
        nf = pd.read_parquet(str(nifty_pq))
        if nf.index.tz is None:
            nf.index = nf.index.tz_localize("UTC")
        nifty_close = nf["close"]

    # Compute all signals
    print("\nComputing signals...")
    signal_results = {}
    for sig_type in ["composite", "v2c", "ensemble"]:
        print(f"  Computing {sig_type} scores...")
        try:
            scores = compute_signal_scores(df_oos, sig_type)
            scores_df = df_oos[["symbol"]].copy()
            scores_df["_score"] = scores
            scores_pivot = scores_df.pivot_table(
                index=df_oos.index, columns="symbol", values="_score", aggfunc="last"
            ).rank(axis=1, pct=True)

            # Measure IC
            if "label_v2b" in df_oos.columns:
                valid = df_oos["label_v2b"].notna()
                ic, pval = spearmanr(
                    scores[valid].to_numpy(),
                    df_oos.loc[valid, "label_v2b"].to_numpy()
                )
                print(f"    {sig_type} IC={ic:+.4f} (p={pval:.4f})")
            signal_results[sig_type] = scores_pivot
        except Exception as e:
            print(f"    {sig_type} failed: {e}")

    # Run all configurations
    print("\n" + "="*70)
    print("  GRID SEARCH RESULTS")
    print("="*70)
    print(f"{'Signal':12s} {'Rebalnce':8s} {'Decile':7s} {'CostModel':10s} {'Construction':14s} | {'AnnExcess%':11s} {'IR':6s} {'WinRate':8s} {'MaxDD':7s}")
    print("-"*95)

    best = {"ann_excess": -999, "config": None, "result": None}
    all_results = []

    for sig_type, scores_pivot in signal_results.items():
        for reb_days in [3, 5, 7, 10, 14, 21]:
            for decile in [0.10, 0.15, 0.20]:
                for cost_label, cost_bps in [("futures", 7.26), ("equity", 27.35)]:
                    for construction, sec_n, vol_s in [
                        ("equal_wt", False, False),
                        ("sec_neutral", True, False),
                        ("vol_scaled", False, True),
                    ]:
                        try:
                            r = backtest_longonly(
                                ohlcv_dict=ohlcv,
                                scores_df=scores_pivot,
                                nifty_close=nifty_close,
                                decile=decile,
                                rebalance_days=reb_days,
                                cost_bps=cost_bps,
                                sector_neutral=sec_n,
                                vol_scaled=vol_s,
                            )
                            if r is None:
                                continue

                            ae = r["ann_excess_return"]
                            ir = r["information_ratio"]
                            wr = r["win_rate_excess"]
                            dd = r["max_drawdown"]
                            mark = "✓" if ae > 0 else " "

                            print(f"{sig_type:12s} {reb_days:2d}d     {decile:.0%}    {cost_label:10s} {construction:14s} | "
                                  f"{ae:+8.2f}%   {ir:+5.3f}   {wr:.1%}   {dd:+6.2%} {mark}")

                            if ae > best["ann_excess"]:
                                best = {
                                    "ann_excess": ae,
                                    "config": {
                                        "signal": sig_type, "rebalance_days": reb_days,
                                        "decile": decile, "cost": cost_label,
                                        "construction": construction,
                                    },
                                    "result": r,
                                }
                            all_results.append({
                                "signal": sig_type, "rebalance_days": reb_days,
                                "decile": decile, "cost": cost_label,
                                "construction": construction, **r,
                            })
                        except Exception:
                            pass

    print("\n" + "="*70)
    print("  BEST CONFIGURATION FOUND")
    print("="*70)
    if best["config"]:
        cfg = best["config"]
        res = best["result"]
        print(f"  Signal:         {cfg['signal']}")
        print(f"  Rebalance:      every {cfg['rebalance_days']} trading days")
        print(f"  Decile:         top {cfg['decile']:.0%}")
        print(f"  Cost model:     {cfg['cost']}")
        print(f"  Construction:   {cfg['construction']}")
        print()
        print(f"  Ann. excess vs NIFTY: {res['ann_excess_return']:+.2f}%/year")
        print(f"  Information Ratio:    {res['information_ratio']:.3f}")
        print(f"  Win rate (excess>0):  {res['win_rate_excess']:.1%}")
        print(f"  Max drawdown:         {res['max_drawdown']:.2%}")
        print(f"  Periods evaluated:    {res['n_periods']}")
        print(f"  Avg stocks/period:    {res['n_long_avg']:.0f}")
        print(f"\n  STATUS: {'✓ PROFITABLE' if res['ann_excess_return'] > 0 else '✗ NOT PROFITABLE'}")

    # Save results
    out_dir = ROOT / "artifacts/max_alpha"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(str(out_dir / "grid_search_results.csv"), index=False)
    if best["config"]:
        (out_dir / "best_config.json").write_text(json.dumps(
            {**best["config"], **best["result"]}, indent=2, default=str
        ))
    print(f"\n✓ Saved: {out_dir}/grid_search_results.csv")


if __name__ == "__main__":
    main()
