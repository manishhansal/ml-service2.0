#!/usr/bin/env python3
"""
Track live intraday P&L for all forward paper positions.
Called every 10 minutes to update the live session report.
"""
from __future__ import annotations
import json, time, urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path('/Users/manishkumar/Desktop/ml-service2.0')
PARQUET_DIR = BASE / 'data/1d/1d'

env = {k.strip(): v.strip() for line in (BASE/'.env').read_text().splitlines()
       if '=' in line and not line.strip().startswith('#')
       for k, _, v in [line.partition('=')]}
DATA_KEY = env.get('DATA_SERVICE_API_KEY', '')
DATA_URL = env.get('DATA_SERVICE_2_URL', 'http://localhost:8200')

COST_BPS = 27.65  # equity round-trip


def get_quote(sym: str) -> dict | None:
    url = f'{DATA_URL}/v1/india/quotes/{sym}'
    req = urllib.request.Request(url, headers={'X-API-KEY': DATA_KEY})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read().decode())
            dd = d.get('data', {}) or {}
            if dd.get('ltp') is not None:
                return dd
    except Exception:
        pass
    return None


def ist_now() -> datetime:
    return datetime.now(tz=timezone.utc) + timedelta(hours=5, minutes=30)


def main():
    now_ist = ist_now()
    print(f'\n=== LIVE P&L TRACKER — {now_ist.strftime("%H:%M IST")} ===')

    # Load all forward paper signals (v1 + v2)
    all_sigs = {}
    for sp in [BASE/'artifacts/forward_paper/signals.jsonl',
               BASE/'artifacts/forward_paper/signals_v2.jsonl']:
        if sp.exists():
            for line in sp.read_text().splitlines():
                if line.strip():
                    s = json.loads(line)
                    sym = s['symbol']
                    if sym not in all_sigs:
                        all_sigs[sym] = s

    print(f'Forward paper positions: {len(all_sigs)}')

    # Get live quotes for all FP symbols (batch - rate limit aware)
    live_quotes = {}
    batch_count = 0
    for sym in sorted(all_sigs.keys()):
        q = get_quote(sym)
        if q:
            live_quotes[sym] = q
        batch_count += 1
        if batch_count % 50 == 0:
            time.sleep(1)   # brief pause every 50 calls
        time.sleep(0.15)   # ~6.7 req/sec, well within 500/60s

    print(f'Live quotes obtained: {len(live_quotes)}/{len(all_sigs)}')

    # Calculate P&L for each position
    results = []
    for sym, sig in all_sigs.items():
        direction = sig.get('direction', 0)
        if direction == 0:
            continue

        # Get entry price from signal (v2 has no entry_price, use last parquet close as proxy)
        entry_price = sig.get('entry_price')
        if entry_price is None:
            # v2 signal: use open of next trading day after signal_ts
            pf = PARQUET_DIR / f'{sym}.parquet'
            if pf.exists():
                df = pd.read_parquet(pf)
                sig_ts = pd.Timestamp(sig.get('signal_ts', sig.get('created_at')))
                if sig_ts.tzinfo is None:
                    sig_ts = sig_ts.tz_localize('UTC')
                df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index
                future = df[df.index > sig_ts]
                if len(future) > 0:
                    entry_price = float(future['open'].iloc[0]) if 'open' in future.columns else None
                if entry_price is None and len(df) > 0:
                    entry_price = float(df['close'].iloc[-1])

        if entry_price is None or entry_price <= 0:
            continue

        ltp = live_quotes.get(sym, {}).get('ltp')
        if ltp is None:
            continue

        ltp = float(ltp)
        # DATA QUALITY FILTER: exclude clear instrument mismatches (e.g., TATAMOTORS-DVR vs regular)
        if entry_price > 0 and abs(ltp - entry_price) / entry_price > 0.50:
            continue  # >50% discrepancy = likely data mismatch, not signal performance
        gross_pnl_pct = direction * (ltp - entry_price) / entry_price * 100
        cost_pct = COST_BPS / 100
        net_pnl_pct = gross_pnl_pct - cost_pct
        change_pct = live_quotes[sym].get('changePct', 0) or 0

        results.append({
            'symbol': sym,
            'direction': direction,
            'dir_str': 'LONG' if direction > 0 else 'SHORT',
            'entry_price': round(entry_price, 2),
            'ltp': ltp,
            'gross_pnl_pct': round(gross_pnl_pct, 3),
            'net_pnl_pct': round(net_pnl_pct, 3),
            'change_pct_today': float(change_pct),
            'session': 'v1' if sig.get('session_id') != 'FORWARD_PAPER_V2' else 'v2',
        })

    if not results:
        print('No live P&L available')
        return {}

    df_res = pd.DataFrame(results)
    wins = df_res[df_res['net_pnl_pct'] > 0]
    losses = df_res[df_res['net_pnl_pct'] <= 0]

    mean_net = df_res['net_pnl_pct'].mean()
    win_rate = len(wins) / len(df_res) * 100

    print(f'\nLIVE P&L SUMMARY ({len(results)} positions):')
    print(f'  Mean net P&L:  {mean_net:+.3f}%')
    print(f'  Win rate:      {win_rate:.1f}%  ({len(wins)} wins / {len(losses)} losses)')
    print(f'  Best trade:    {df_res.loc[df_res["net_pnl_pct"].idxmax(), "symbol"]} {df_res["net_pnl_pct"].max():+.3f}%')
    print(f'  Worst trade:   {df_res.loc[df_res["net_pnl_pct"].idxmin(), "symbol"]} {df_res["net_pnl_pct"].min():+.3f}%')

    # Breakdown by direction
    long_pnl = df_res[df_res['direction'] == 1]['net_pnl_pct']
    short_pnl = df_res[df_res['direction'] == -1]['net_pnl_pct']
    if len(long_pnl) > 0:
        print(f'  LONG mean P&L:  {long_pnl.mean():+.3f}% (n={len(long_pnl)})')
    if len(short_pnl) > 0:
        print(f'  SHORT mean P&L: {short_pnl.mean():+.3f}% (n={len(short_pnl)})')

    # Top winners and losers
    print(f'\nTop 5 Winners:')
    for _, r in df_res.nlargest(5, 'net_pnl_pct').iterrows():
        print(f'  {r["dir_str"]} {r["symbol"]:15s} entry={r["entry_price"]:.2f} ltp={r["ltp"]:.2f} P&L={r["net_pnl_pct"]:+.3f}%')

    print(f'\nTop 5 Losers:')
    for _, r in df_res.nsmallest(5, 'net_pnl_pct').iterrows():
        print(f'  {r["dir_str"]} {r["symbol"]:15s} entry={r["entry_price"]:.2f} ltp={r["ltp"]:.2f} P&L={r["net_pnl_pct"]:+.3f}%')

    # Market alignment
    nifty_q = get_quote('NIFTY')
    nifty_chg = float(nifty_q.get('changePct', 0) or 0) if nifty_q else 0
    print(f'\nMarket Context:')
    print(f'  NIFTY: {nifty_q.get("ltp","?") if nifty_q else "?"} ({nifty_chg:+.2f}%)')
    print(f'  Model bias: {len(df_res[df_res["direction"]==-1])} SHORT, {len(df_res[df_res["direction"]==1])} LONG')
    alignment = 'ALIGNED' if (nifty_chg < 0 and len(short_pnl) > len(long_pnl)) else 'MIXED'
    print(f'  Signal alignment: {alignment}')

    # Score all 218 symbols with LightGBM
    print(f'\nScoring all 218 symbols with LightGBM fs-3.0.0...')
    all_parquets = sorted(PARQUET_DIR.glob('*.parquet'))
    universe = [p.stem for p in all_parquets]

    try:
        import pickle, numpy as np
        model_paths = sorted(Path('artifacts/expanded_lgbm').glob('*/model.pkl'))
        if not model_paths:
            model_paths = sorted(Path('artifacts/registry/stage_a_1d').glob('*/model.pkl'))
        payload = pickle.load(open(model_paths[-1], 'rb'))
        estimator = payload['estimator']
        feat_names = payload['feature_names']
        norm_state = payload.get('normalizer_state')

        normalizer = None
        if norm_state:
            from src.features.normalizer import FeatureNormalizer
            normalizer = FeatureNormalizer.from_dict(norm_state)

        scores_all = []
        for sym in universe:
            pf = PARQUET_DIR / f'{sym}.parquet'
            if not pf.exists(): continue
            try:
                import warnings; warnings.filterwarnings('ignore')
                df = pd.read_parquet(pf)
                df.columns = [c.lower() for c in df.columns]
                if df.index.tz is None: df.index = df.index.tz_localize('UTC')

                if len(feat_names) == 55:
                    from src.features.expanded_factory import ExpandedFeatureFactory
                    factory = ExpandedFeatureFactory()
                else:
                    from src.features.factory import FeatureFactory
                    factory = FeatureFactory()
                features, _ = factory.build(df)
                last = features.iloc[-1].fillna(0)
                X = np.array([[last.get(c, 0.0) for c in feat_names]])
                if normalizer:
                    X_df = pd.DataFrame(X, columns=feat_names)
                    X = normalizer.transform(X_df).to_numpy(dtype=float)
                score = float(estimator.predict(X)[0])
                scores_all.append({'symbol': sym, 'score': score,
                                   'direction': 1 if score > 0.5 else -1})
            except Exception:
                pass

        n_long = sum(1 for s in scores_all if s['direction'] == 1)
        n_short = sum(1 for s in scores_all if s['direction'] == -1)
        top_long = sorted([s for s in scores_all if s['direction'] == 1],
                          key=lambda x: x['score'], reverse=True)[:10]
        top_short = sorted([s for s in scores_all if s['direction'] == -1],
                           key=lambda x: x['score'])[:10]
        print(f'  Scored: {len(scores_all)} | LONG: {n_long} | SHORT: {n_short}')
    except Exception as e:
        print(f'  Scoring failed: {e}')
        scores_all, top_long, top_short = [], [], []

    # Save snapshot
    snapshot = {
        'timestamp': now_ist.isoformat(),
        'nifty_ltp': nifty_q.get('ltp') if nifty_q else None,
        'nifty_chg': nifty_chg,
        'n_positions': len(results),
        'mean_net_pnl_pct': round(float(mean_net), 4),
        'win_rate_pct': round(win_rate, 2),
        'n_wins': len(wins),
        'n_losses': len(losses),
        'long_mean_pnl': round(float(long_pnl.mean()), 4) if len(long_pnl) > 0 else 0,
        'short_mean_pnl': round(float(short_pnl.mean()), 4) if len(short_pnl) > 0 else 0,
        'all_positions': results,
        'model_scores_summary': {
            'n_scored': len(scores_all),
            'n_long': n_long,
            'n_short': n_short,
            'top_long': top_long[:5],
            'top_short': top_short[:5],
        },
    }

    snap_path = BASE / 'artifacts/live_session/pnl_snapshot.json'
    snap_path.write_text(json.dumps(snapshot, indent=2, default=str))
    print(f'\nSnapshot saved → {snap_path}')
    return snapshot


if __name__ == '__main__':
    main()
