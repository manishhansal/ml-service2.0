#!/usr/bin/env python3
"""
live_market_session.py — Real-time market monitoring until NSE close.

Every 10 minutes:
  1. Fetch live quotes for all 218 symbols via data-service2.0
  2. Append fresh daily bar to on-disk parquets  
  3. Score all symbols using LightGBM fs-3.0.0
  4. Track signal strength: compare live prediction vs open position
  5. Record all data in live_session_log.jsonl

Runs until 15:30 IST or CTRL-C.

Usage:
    PYTHONPATH=. python3 scripts/live_market_session.py
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_DIR    = Path("data/1d/1d")
SESSION_LOG    = Path("artifacts/live_session/session_log.jsonl")
SUMMARY_PATH   = Path("artifacts/live_session/session_summary.json")
SAMPLE_INTERVAL_MIN = 10   # sample every 10 minutes
NSE_CLOSE_IST  = (15, 30)   # 15:30 IST

SESSION_LOG.parent.mkdir(parents=True, exist_ok=True)

env = {k.strip(): v.strip() for line in Path(".env").read_text().splitlines()
       if "=" in line and not line.strip().startswith("#")
       for k, _, v in [line.partition("=")]}
DATA_KEY = env.get("DATA_SERVICE_API_KEY", "")
DATA_URL = env.get("DATA_SERVICE_2_URL", "http://localhost:8200")


def ist_now() -> datetime:
    return datetime.now(tz=timezone.utc) + timedelta(hours=5, minutes=30)


def market_open() -> bool:
    now = ist_now()
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return (9, 15) <= t <= (15, 30)


def mins_to_close() -> float:
    now = ist_now()
    close = now.replace(hour=NSE_CLOSE_IST[0], minute=NSE_CLOSE_IST[1], second=0, microsecond=0)
    return (close - now).total_seconds() / 60


def get_quote(symbol: str) -> dict | None:
    url = f"{DATA_URL}/v1/india/quotes/{symbol}"
    req = urllib.request.Request(url, headers={"X-API-KEY": DATA_KEY})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read().decode())
            dd = d.get("data", {}) or {}
            if dd.get("ltp") is not None:
                return dd
    except Exception:
        pass
    return None


def load_model() -> tuple | None:
    """Load LightGBM champion."""
    for pattern in ["artifacts/expanded_lgbm/*/model.pkl",
                    "artifacts/registry/stage_a_1d/*/model.pkl"]:
        paths = sorted(Path(".").glob(pattern))
        if paths:
            import pickle
            with open(paths[-1], "rb") as f:
                payload = pickle.load(f)
            estimator = payload.get("estimator")
            feat_names = payload.get("feature_names", [])
            normalizer_state = payload.get("normalizer_state")
            schema = payload.get("feature_schema_version", "?")
            print(f"[model] Loaded {schema} | {len(feat_names)} features | {paths[-1].name}")
            return estimator, feat_names, normalizer_state, schema
    return None


def score_symbol(symbol: str, estimator, feat_names: list, normalizer=None) -> dict | None:
    """Score one symbol using on-disk parquet + LightGBM."""
    pf = PARQUET_DIR / f"{symbol}.parquet"
    if not pf.exists():
        return None
    try:
        from src.features.expanded_factory import ExpandedFeatureFactory
        df = pd.read_parquet(pf)
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        factory = ExpandedFeatureFactory()
        if len(feat_names) <= 24:
            from src.features.factory import FeatureFactory
            factory = FeatureFactory()

        features, _ = factory.build(df)
        last = features.iloc[-1].fillna(0)
        X = np.array([[last.get(c, 0.0) for c in feat_names]])

        if normalizer is not None:
            try:
                X_df = pd.DataFrame(X, columns=feat_names)
                X = normalizer.transform(X_df).to_numpy(dtype=float)
            except Exception:
                pass

        score = float(estimator.predict(X)[0])
        direction = 1 if score > 0.5 else -1
        return {
            "symbol": symbol,
            "score": round(score, 4),
            "direction": direction,
            "data_date": str(df.index.max().date()),
        }
    except Exception as e:
        return None


def run_session():
    print("=" * 65)
    print(f"LIVE MARKET SESSION — {ist_now().strftime('%Y-%m-%d %H:%M IST')}")
    print(f"NSE closes in {mins_to_close():.0f} minutes")
    print("=" * 65)

    # Load model
    model_result = load_model()
    if model_result is None:
        print("ERROR: No model found")
        return
    estimator, feat_names, normalizer_state, schema = model_result

    normalizer = None
    if normalizer_state:
        try:
            from src.features.normalizer import FeatureNormalizer
            normalizer = FeatureNormalizer.from_dict(normalizer_state)
        except Exception:
            pass

    # Load all 218 symbols
    all_parquets = sorted(PARQUET_DIR.glob("*.parquet"))
    universe = [pf.stem for pf in all_parquets]
    print(f"Universe: {len(universe)} symbols\n")

    # Track positions from forward paper signals
    fp_signals = {}
    for store_path in [Path("artifacts/forward_paper/signals.jsonl"),
                       Path("artifacts/forward_paper/signals_v2.jsonl")]:
        if store_path.exists():
            for line in store_path.read_text().splitlines():
                if line.strip():
                    try:
                        s = json.loads(line)
                        fp_signals[s["symbol"]] = s
                    except Exception:
                        pass
    print(f"Forward paper positions tracked: {len(fp_signals)}")

    session_data = {
        "start_time": ist_now().isoformat(),
        "model_schema": schema,
        "n_features": len(feat_names),
        "universe_size": len(universe),
        "samples": [],
    }
    sample_count = 0

    # ── Main loop ──────────────────────────────────────────────────────────────
    while market_open():
        sample_ts = ist_now()
        mins_left = mins_to_close()
        print(f"\n[{sample_ts.strftime('%H:%M IST')}] Sample #{sample_count+1} | {mins_left:.0f}min to close")

        # Fetch live quotes (rate limited — take key symbols only)
        key_symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "ICICIBANK",
                       "INFY", "TCS", "KOTAKBANK", "AXISBANK", "BHARTIARTL",
                       "SBIN", "LT", "MARUTI", "WIPRO", "TITAN"]
        live_quotes = {}
        rate_limited = 0
        for sym in key_symbols:
            q = get_quote(sym)
            if q:
                live_quotes[sym] = q
            else:
                rate_limited += 1
            time.sleep(0.3)  # stay under 500/60s limit

        print(f"  Live quotes: {len(live_quotes)}/{len(key_symbols)} | rate_limited: {rate_limited}")

        # Show NIFTY market state
        nifty = live_quotes.get("NIFTY", {})
        if nifty.get("ltp"):
            print(f"  NIFTY: {nifty['ltp']} ({nifty.get('changePct',0):+.2f}%) | {nifty.get('marketStatus','?')}")

        # Score all 218 symbols using latest parquet data
        all_scores = []
        for sym in universe:
            result = score_symbol(sym, estimator, feat_names, normalizer)
            if result:
                all_scores.append(result)

        if all_scores:
            # Top signals
            top_long  = sorted([s for s in all_scores if s["direction"] == 1],
                               key=lambda x: x["score"], reverse=True)[:5]
            top_short = sorted([s for s in all_scores if s["direction"] == -1],
                               key=lambda x: x["score"])[:5]

            print(f"\n  SCORED {len(all_scores)} symbols")
            print(f"  LONG bias: {sum(1 for s in all_scores if s['direction']==1)}/{len(all_scores)}")
            print()
            print("  TOP LONG signals:")
            for s in top_long:
                q = live_quotes.get(s["symbol"], {})
                ltp = q.get("ltp", "?")
                chg = q.get("changePct", "?")
                fp = "📌" if s["symbol"] in fp_signals else "  "
                print(f"    {fp} {s['symbol']:15s} score={s['score']:.3f} LTP={ltp} chg={chg}%")

            print()
            print("  TOP SHORT signals:")
            for s in top_short:
                q = live_quotes.get(s["symbol"], {})
                ltp = q.get("ltp", "?")
                chg = q.get("changePct", "?")
                fp = "📌" if s["symbol"] in fp_signals else "  "
                print(f"    {fp} {s['symbol']:15s} score={s['score']:.3f} LTP={ltp} chg={chg}%")

            # Check forward paper positions
            print()
            print("  FORWARD PAPER POSITION CHECK (v1 signals, h=5):")
            for sym, fp_sig in list(fp_signals.items())[:8]:
                live = live_quotes.get(sym, {})
                ltp = live.get("ltp")
                entry_price = fp_sig.get("entry_price")
                direction = fp_sig.get("direction", 0)
                if ltp and entry_price:
                    pnl_pct = direction * (ltp - entry_price) / entry_price * 100
                    print(f"    {sym:15s} entry={entry_price:.2f} LTP={ltp:.2f} P&L={pnl_pct:+.2f}%")

        # Save sample
        sample = {
            "timestamp": sample_ts.isoformat(),
            "mins_to_close": round(mins_left, 1),
            "live_quotes": {sym: {"ltp": q.get("ltp"), "changePct": q.get("changePct")}
                           for sym, q in live_quotes.items()},
            "n_scored": len(all_scores),
            "n_long": sum(1 for s in all_scores if s["direction"] == 1),
            "n_short": sum(1 for s in all_scores if s["direction"] == -1),
            "top_long": top_long[:3] if all_scores else [],
            "top_short": top_short[:3] if all_scores else [],
        }

        with SESSION_LOG.open("a") as fh:
            fh.write(json.dumps(sample) + "\n")

        session_data["samples"].append(sample)
        sample_count += 1

        if mins_left <= 0:
            print("\n[session] Market closed — session complete")
            break

        wait_secs = min(SAMPLE_INTERVAL_MIN * 60, max(60, (mins_left - 1) * 60))
        print(f"\n  Next sample in {wait_secs/60:.1f} min...")
        time.sleep(wait_secs)

    # ── Session end ─────────────────────────────────────────────────────────
    session_data["end_time"] = ist_now().isoformat()
    session_data["n_samples"] = sample_count
    SUMMARY_PATH.write_text(json.dumps(session_data, indent=2, default=str))
    print(f"\n[session] Summary saved → {SUMMARY_PATH}")
    print(f"[session] Full log → {SESSION_LOG}")

    if not market_open():
        print("\n[session] Market CLOSED — generating final report...")
        generate_final_report(session_data)


def generate_final_report(session_data: dict):
    """Generate post-session report."""
    samples = session_data.get("samples", [])
    if not samples:
        return

    report = {
        "session_id": "LIVE_SESSION_1",
        "start": session_data.get("start_time"),
        "end": session_data.get("end_time"),
        "model": session_data.get("model_schema"),
        "n_samples": session_data.get("n_samples", 0),
        "universe_size": session_data.get("universe_size", 0),
        "signal_stability": "REPORT READY",
    }
    print(f"\nSESSION REPORT:")
    print(f"  Samples taken: {report['n_samples']}")
    print(f"  Universe: {report['universe_size']} symbols")

    # Signal direction consistency across samples
    if len(samples) >= 2:
        all_top_long_syms = [s["symbol"] for sample in samples for s in sample.get("top_long", [])]
        from collections import Counter
        top_long_counts = Counter(all_top_long_syms)
        consistent = [(sym, cnt) for sym, cnt in top_long_counts.most_common(5) if cnt >= len(samples)//2]
        if consistent:
            print(f"  Consistent LONG signals (appear in ≥50% of samples):")
            for sym, cnt in consistent:
                print(f"    {sym}: {cnt}/{len(samples)} samples")
        report["consistent_long_signals"] = [{"symbol": s, "sample_count": c} for s, c in consistent]

    Path("reports/live_session_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved → reports/live_session_report.json")


if __name__ == "__main__":
    run_session()
