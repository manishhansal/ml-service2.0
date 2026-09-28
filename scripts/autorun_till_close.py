#!/usr/bin/env python3
"""
autorun_till_close.py
=====================
Single unified automation that runs from NOW until NSE market close (15:30 IST).

Every 5 minutes:
  • Fetches live quotes for all 218 symbols
  • Scores all 218 with LightGBM fs-3.0.0
  • Calculates live P&L for all forward paper positions
  • Appends a structured JSON record to the session log
  • Prints a clean dashboard to stdout

At 15:30 IST (market close):
  1. Final quote snapshot
  2. Ingest Sep-28 close bars into all parquets
  3. Resolve all due forward paper signals
  4. Run SignalPromotionEngine on resolved outcomes
  5. Update LIVE_SESSION_REPORT.md with final numbers
  6. Print comprehensive end-of-day summary

Usage:
    PYTHONPATH=. python3 -W ignore scripts/autorun_till_close.py
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
import urllib.request
import urllib.error
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# Silence all debug/info from structlog/src modules BEFORE importing anything from src
import logging
logging.disable(logging.WARNING)
for name in ("structlog", "src", "root", ""):
    logging.getLogger(name).setLevel(logging.CRITICAL)
os.environ["LOG_LEVEL"] = "ERROR"

# Patch structlog to suppress debug/info
try:
    import structlog
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR),
    )
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# ── Phil-inspired integrations ────────────────────────────────────────────────
# Imported lazily inside functions to avoid startup cost; explicit imports here
# so linters and type checkers see them and to document the dependency surface.
try:
    from src.analytics.forecast_ledger import ForecastLedger
    from src.analytics.counterfactual_ledger import CounterfactualLedger
    from src.analytics.score_threshold_sweep import ScoreThresholdSweep
    from src.analytics.feature_weight_manager import FeatureWeightManager
    _PHIL_IMPORTS_OK = True
except Exception as _e:
    _PHIL_IMPORTS_OK = False
    print(f"[warn] Phil integrations not available: {_e}", file=sys.stderr)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent.parent
PARQUET_DIR  = BASE / "data/1d/1d"
SESSION_DIR  = BASE / "artifacts/live_session"
SESSION_LOG  = SESSION_DIR / "autorun_log.jsonl"
REPORT_PATH  = BASE / "reports/LIVE_SESSION_REPORT.md"
COST_BPS     = 27.65  # equity round-trip
SAMPLE_MINS  = 5      # sample every 5 minutes

SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ── Load env ──────────────────────────────────────────────────────────────────
env = {}
env_path = BASE / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
DATA_KEY = env.get("DATA_SERVICE_API_KEY", "")
DATA_URL = env.get("DATA_SERVICE_2_URL", "http://localhost:8200")

# ── Time helpers ──────────────────────────────────────────────────────────────
def ist_now() -> datetime:
    return datetime.now(tz=timezone.utc) + timedelta(hours=5, minutes=30)

def mins_to_close() -> float:
    now = ist_now()
    close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return (close - now).total_seconds() / 60

def market_open() -> bool:
    now = ist_now()
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return (9, 15) <= t <= (15, 31)

# ── Data fetching ─────────────────────────────────────────────────────────────
def get_quote(sym: str) -> dict | None:
    url = f"{DATA_URL}/v1/india/quotes/{sym}"
    req = urllib.request.Request(url, headers={"X-API-KEY": DATA_KEY})
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            d = json.loads(r.read().decode())
            dd = d.get("data", {}) or {}
            if dd.get("ltp") is not None:
                return dd
    except Exception:
        pass
    return None

def get_all_quotes(symbols: list[str]) -> dict[str, dict]:
    quotes = {}
    for sym in symbols:
        q = get_quote(sym)
        if q:
            quotes[sym] = q
        time.sleep(0.12)  # ~8 req/sec, within 500/60s limit
    return quotes

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    for pattern in ["artifacts/expanded_lgbm/*/model.pkl",
                     "artifacts/registry/stage_a_1d/*/model.pkl"]:
        paths = sorted(BASE.glob(pattern))
        if paths:
            with open(paths[-1], "rb") as f:
                p = pickle.load(f)
            return (p["estimator"], p["feature_names"],
                    p.get("normalizer_state"), p.get("feature_schema_version","?"))
    return None, [], None, "?"

def score_symbol(sym: str, estimator, feat_names, normalizer=None) -> dict | None:
    pf = PARQUET_DIR / f"{sym}.parquet"
    if not pf.exists():
        return None
    try:
        from src.features.expanded_factory import ExpandedFeatureFactory
        from src.features.factory import FeatureFactory
        df = pd.read_parquet(pf)
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        factory = ExpandedFeatureFactory() if len(feat_names) > 24 else FeatureFactory()
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
        return {"symbol": sym, "score": round(score, 4), "direction": 1 if score > 0.5 else -1,
                "data_date": str(df.index.max().date())}
    except Exception:
        return None

def score_all(estimator, feat_names, normalizer) -> list[dict]:
    results = []
    for pf in sorted(PARQUET_DIR.glob("*.parquet")):
        r = score_symbol(pf.stem, estimator, feat_names, normalizer)
        if r:
            results.append(r)
    return results

# ── P&L calculation ──────────────────────────────────────────────────────────
def load_fp_signals() -> dict[str, dict]:
    sigs = {}
    for sp in [BASE/"artifacts/forward_paper/signals.jsonl",
               BASE/"artifacts/forward_paper/signals_v2.jsonl"]:
        if sp.exists():
            for line in sp.read_text().splitlines():
                if line.strip():
                    try:
                        s = json.loads(line)
                        if s["symbol"] not in sigs:
                            sigs[s["symbol"]] = s
                    except Exception:
                        pass
    return sigs

# Load data quality flags
def load_excluded_symbols() -> set[str]:
    flag_path = BASE / "artifacts/data_quality_flags.json"
    if not flag_path.exists():
        return set()
    try:
        flags = json.loads(flag_path.read_text())
        return {f["symbol"] for f in flags}
    except Exception:
        return set()

def calc_pnl(fp_signals: dict, live_quotes: dict, excluded: set) -> dict:
    results = []
    for sym, sig in fp_signals.items():
        if sym in excluded:
            continue
        direction = sig.get("direction", 0)
        if direction == 0:
            continue
        # Get entry price
        entry = sig.get("entry_price")
        if entry is None:
            pf = PARQUET_DIR / f"{sym}.parquet"
            if pf.exists():
                df = pd.read_parquet(pf)
                sig_ts = pd.Timestamp(sig.get("signal_ts", sig.get("created_at", "")))
                if sig_ts.tzinfo is None:
                    sig_ts = sig_ts.tz_localize("UTC")
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                future = df[df.index > sig_ts]
                if len(future) > 0 and "open" in future.columns:
                    entry = float(future["open"].iloc[0])
                elif len(df) > 0:
                    entry = float(df["close"].iloc[-1])
        if not entry or entry <= 0:
            continue
        ltp_data = live_quotes.get(sym, {})
        ltp = ltp_data.get("ltp")
        if ltp is None:
            continue
        ltp = float(ltp)
        # Outlier filter: >50% mismatch = data issue
        if abs(ltp - entry) / max(entry, 1e-6) > 0.50:
            continue
        gross = direction * (ltp - entry) / entry * 100
        net   = gross - (COST_BPS / 100)
        results.append({
            "symbol": sym, "direction": direction,
            "entry": round(entry, 2), "ltp": round(ltp, 2),
            "gross_pct": round(gross, 3), "net_pct": round(net, 3),
            "chg_today_pct": float(ltp_data.get("changePct", 0) or 0),
        })
    if not results:
        return {"n": 0, "mean_net": 0, "win_rate": 0, "short_mean": 0, "long_mean": 0, "positions": []}
    arr = pd.DataFrame(results)
    wins = arr[arr["net_pct"] > 0]
    short_arr = arr[arr["direction"] == -1]
    long_arr  = arr[arr["direction"] == 1]
    return {
        "n": len(arr),
        "mean_net": round(float(arr["net_pct"].mean()), 4),
        "win_rate": round(len(wins) / len(arr) * 100, 1),
        "short_mean": round(float(short_arr["net_pct"].mean()), 4) if len(short_arr) else 0,
        "long_mean":  round(float(long_arr["net_pct"].mean()), 4) if len(long_arr) else 0,
        "best":  arr.loc[arr["net_pct"].idxmax(), ["symbol","net_pct"]].to_dict(),
        "worst": arr.loc[arr["net_pct"].idxmin(), ["symbol","net_pct"]].to_dict(),
        "n_short": len(short_arr), "n_long": len(long_arr),
        "positions": results,
    }

# ── Post-close actions ────────────────────────────────────────────────────────
def post_close_ingest():
    print("\n[close] Ingesting Sep 28 closing bars...")
    result = subprocess.run(
        ["python3", "-W", "ignore", str(BASE/"scripts/ingest_all_outdated.py")],
        capture_output=True, text=True, cwd=str(BASE),
        env={**os.environ, "PYTHONPATH": str(BASE)}, timeout=300,
    )
    out = result.stdout.strip()
    if out:
        print(out[-400:])
    return out

def post_close_resolve():
    print("\n[close] Resolving forward paper signals...")
    result = subprocess.run(
        ["python3", "-W", "ignore", str(BASE/"scripts/resolve_forward_paper.py")],
        capture_output=True, text=True, cwd=str(BASE),
        env={**os.environ, "PYTHONPATH": str(BASE)}, timeout=120,
    )
    out = result.stdout.strip()
    if out:
        print(out[-600:])
    return out

def post_close_promotion():
    print("\n[close] Running SignalPromotionEngine...")
    result = subprocess.run(
        ["python3", "-W", "ignore", str(BASE/"scripts/run_signal_promotion.py")],
        capture_output=True, text=True, cwd=str(BASE),
        env={**os.environ, "PYTHONPATH": str(BASE)}, timeout=60,
    )
    out = result.stdout.strip()
    if out:
        print(out[-400:])
    return out

# ── Dashboard printing ────────────────────────────────────────────────────────
def print_dashboard(sample_n: int, now: datetime, nifty: dict | None, scores: list,
                    pnl: dict, mins_left: float) -> None:
    n_long  = sum(1 for s in scores if s["direction"] == 1)
    n_short = sum(1 for s in scores if s["direction"] == -1)
    nifty_ltp = nifty.get("ltp","?") if nifty else "UNAVAILABLE"
    nifty_chg = nifty.get("changePct", 0) if nifty else 0

    bar_pnl  = "▓" * max(0, min(20, int(pnl.get("short_mean",0)*20))) if pnl["n"] > 0 else ""
    bar_long = "░" * max(0, min(20, int(abs(pnl.get("long_mean",0))*10))) if pnl["n"] > 0 else ""

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  LIVE SESSION #{sample_n:02d}  │  {now.strftime('%H:%M IST')}  │  {mins_left:.0f}min to close      ║
╠══════════════════════════════════════════════════════════════════════╣
║  MARKET     │ NIFTY {nifty_ltp} ({nifty_chg:+.2f}%)                             ║
║  SIGNALS    │ {len(scores)} scored │ LONG={n_long} SHORT={n_short} (bias={n_short/max(len(scores),1)*100:.0f}% SHORT)  ║
╠══════════════════════════════════════════════════════════════════════╣
║  LIVE P&L   │ {pnl['n']} positions tracked                                 ║
║  Mean net   │ {pnl.get('mean_net',0):+.3f}%                                          ║
║  Win rate   │ {pnl.get('win_rate',0):.1f}%                                             ║
║  SHORT avg  │ {pnl.get('short_mean',0):+.3f}% ({pnl.get('n_short',0)} positions) {bar_pnl}   ║
║  LONG avg   │ {pnl.get('long_mean',0):+.3f}% ({pnl.get('n_long',0)} positions) {bar_long}  ║""")
    if pnl["n"] > 0 and "best" in pnl:
        print(f"║  Best trade │ {pnl['best']['symbol']} {pnl['best']['net_pct']:+.3f}%                              ║")
        print(f"║  Worst      │ {pnl['worst']['symbol']} {pnl['worst']['net_pct']:+.3f}%                              ║")
    print(f"╚══════════════════════════════════════════════════════════════════════╝")

    # Top signals
    top_long  = sorted([s for s in scores if s["direction"] == 1],  key=lambda x: -x["score"])[:5]
    top_short = sorted([s for s in scores if s["direction"] == -1], key=lambda x:  x["score"])[:5]
    long_str  = ", ".join(f'{s["symbol"]}({s["score"]:.3f})' for s in top_long)
    short_str = ", ".join(f'{s["symbol"]}({s["score"]:.3f})' for s in top_short)
    print(f"\n  TOP LONG  (score->1.0):  {long_str}")
    print(f"  TOP SHORT (score->0.0):  {short_str}")


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"  AUTORUN TILL CLOSE  |  {ist_now().strftime('%Y-%m-%d %H:%M IST')}")
    print(f"  NSE closes in {mins_to_close():.0f} minutes (15:30 IST)")
    print(f"  Sampling every {SAMPLE_MINS} minutes")
    print("=" * 70)

    # Load model once
    estimator, feat_names, normalizer_state, schema = load_model()
    if estimator is None:
        print("ERROR: No model found. Check artifacts/expanded_lgbm/")
        sys.exit(1)

    normalizer = None
    if normalizer_state:
        from src.features.normalizer import FeatureNormalizer
        try:
            normalizer = FeatureNormalizer.from_dict(normalizer_state)
        except Exception:
            pass

    print(f"  Model: {schema} | {len(feat_names)} features | normalizer: {'yes' if normalizer else 'no'}")

    # ── Phil integration: initialise components ───────────────────────────────
    _forecast_ledger    = ForecastLedger()    if _PHIL_IMPORTS_OK else None
    _cfactual_ledger    = CounterfactualLedger() if _PHIL_IMPORTS_OK else None
    _weight_mgr         = FeatureWeightManager() if _PHIL_IMPORTS_OK else None
    session_date        = ist_now().strftime("%Y-%m-%d")
    if _PHIL_IMPORTS_OK:
        print(f"  Phil integrations: ForecastLedger ✓ | CounterfactualLedger ✓ | FeatureWeightManager ✓")

    # Load forward paper signals and exclusions
    fp_signals = load_fp_signals()
    excluded   = load_excluded_symbols()
    universe   = [pf.stem for pf in sorted(PARQUET_DIR.glob("*.parquet"))]
    key_syms   = ["NIFTY","BANKNIFTY","RELIANCE","HDFCBANK","ICICIBANK",
                  "INFY","TCS","KOTAKBANK","AXISBANK","BHARTIARTL",
                  "SBIN","LT","MARUTI","WIPRO","TITAN","NTPC","ONGC",
                  "BAJFINANCE","HINDUNILVR","ADANIENT"]

    print(f"  Universe: {len(universe)} | FP positions: {len(fp_signals)} | Excluded: {excluded}")
    print("  Running...\n")

    # Session state
    all_samples: list[dict] = []
    sample_n   = 0
    post_close_done = False

    while True:
        now     = ist_now()
        mins    = mins_to_close()

        # ── MARKET CLOSED & POST-CLOSE ─────────────────────────────────────
        if mins <= 0 and not post_close_done:
            print(f"\n{'='*70}")
            print(f"  MARKET CLOSED — {now.strftime('%H:%M IST')}  Running post-close actions...")
            print(f"{'='*70}")

            # Final quote snapshot
            print("\n[close] Final quote snapshot...")
            final_quotes = get_all_quotes(key_syms + list(fp_signals.keys())[:50])

            # Post-close actions
            ingest_out  = post_close_ingest()
            resolve_out = post_close_resolve()
            promo_out   = post_close_promotion()

            # Final P&L with fresh quotes
            final_scores = score_all(estimator, feat_names, normalizer)
            final_pnl    = calc_pnl(fp_signals, final_quotes, excluded)

            # ── Phil: resolve all 218 forecasts with final realized returns ──
            if _forecast_ledger is not None and final_pnl.get("positions"):
                realized_returns = {
                    p["symbol"]: p["net_pct"]
                    for p in final_pnl.get("positions", [])
                }
                brier_report = _forecast_ledger.resolve_session(
                    session_date=session_date,
                    realized_returns=realized_returns,
                )
                bd = brier_report.get("brier_delta", 0)
                print(f"\n[close] ForecastLedger: brier_delta={bd:+.6f} "
                      f"({'BEATING market' if bd < 0 else 'behind market'}) | "
                      f"n={brier_report.get('n_resolved', 0)} resolved")

            # ── Phil: score threshold sweep (update optimal threshold) ───────
            if _PHIL_IMPORTS_OK and final_pnl.get("positions"):
                sweep = ScoreThresholdSweep()
                resolved_outcomes = [
                    {"symbol": p["symbol"], "net_pct": p["net_pct"],
                     "direction": p["direction"]}
                    for p in final_pnl.get("positions", [])
                ]
                sweep_report = sweep.run(
                    scores=final_scores, resolved_outcomes=resolved_outcomes,
                    date=session_date,
                )
                sweep.save_report(
                    sweep_report,
                    BASE / "reports" / f"score_threshold_sweep_{session_date}.json",
                )
                if sweep_report.optimal_threshold > 0:
                    print(f"[close] ScoreThresholdSweep: optimal threshold={sweep_report.optimal_threshold:.2f} "
                          f"win_rate={sweep_report.optimal_win_rate:.1%}")
                    # Auto-update feature_weights.json with sweep result
                    if _weight_mgr is not None:
                        _weight_mgr.update_from_sweep(
                            optimal_threshold=sweep_report.optimal_threshold,
                            win_rate=sweep_report.optimal_win_rate,
                            evidence=sweep_report.reasoning,
                        )

            # ── Phil: grade any blocked signals (counterfactual ledger) ──────
            if _cfactual_ledger is not None and final_pnl.get("positions"):
                realized_for_cf = {
                    p["symbol"]: p["net_pct"] for p in final_pnl.get("positions", [])
                }
                cf_report = _cfactual_ledger.resolve(
                    session_date=session_date,
                    realized_returns=realized_for_cf,
                )
                if cf_report.get("n", 0) > 0:
                    print(f"[close] CounterfactualLedger: {cf_report['n']} blocked signals graded | "
                          f"verdict={cf_report.get('overall_verdict','N/A')}")

            # Final sample
            nifty_q = final_quotes.get("NIFTY", {})
            final_sample = {
                "timestamp":   now.isoformat(),
                "is_close":    True,
                "nifty_ltp":   nifty_q.get("ltp"),
                "nifty_chg":   nifty_q.get("changePct", 0),
                "n_scored":    len(final_scores),
                "n_long":      sum(1 for s in final_scores if s["direction"] == 1),
                "n_short":     sum(1 for s in final_scores if s["direction"] == -1),
                "pnl":         final_pnl,
            }
            all_samples.append(final_sample)
            with SESSION_LOG.open("a") as f:
                f.write(json.dumps(final_sample) + "\n")

            print_dashboard(sample_n + 1, now, nifty_q, final_scores, final_pnl, 0)

            # Summary
            print(f"\n{'='*70}")
            print("  END-OF-DAY SUMMARY")
            print(f"{'='*70}")
            print(f"  Session samples: {len(all_samples)}")
            print(f"  Market move: NIFTY {nifty_q.get('changePct',0):+.2f}%")
            print(f"  Final P&L: {final_pnl.get('mean_net',0):+.4f}% mean net | {final_pnl.get('win_rate',0):.1f}% win rate")
            print(f"  SHORT avg: {final_pnl.get('short_mean',0):+.4f}% | LONG avg: {final_pnl.get('long_mean',0):+.4f}%")
            n_pos = final_pnl.get("n_short", 0) + final_pnl.get("n_long", 0)
            if n_pos > 0:
                pos_side = "SHORT" if final_pnl.get("n_short",0) > final_pnl.get("n_long",0) else "LONG"
                print(f"  Model bias: {pos_side} ({final_pnl.get('n_short',0)} SHORT / {final_pnl.get('n_long',0)} LONG)")
            print(f"\n  All files saved:")
            print(f"    • {SESSION_LOG}")
            print(f"    • {REPORT_PATH}")
            print(f"    • artifacts/forward_paper/outcomes.jsonl")
            print(f"    • artifacts/live_session/session_summary.json")

            # Save session summary
            summary = {
                "session_date": now.strftime("%Y-%m-%d"),
                "n_samples": len(all_samples),
                "final_nifty_chg": nifty_q.get("changePct", 0),
                "final_pnl": final_pnl,
                "all_samples": all_samples,
            }
            (SESSION_DIR/"session_summary.json").write_text(json.dumps(summary, indent=2, default=str))

            # Update live report
            _append_close_to_report(now, nifty_q, final_pnl, len(all_samples))

            post_close_done = True
            print(f"\n  Done! Session complete. All logs saved.")
            break

        # ── MARKET CLOSED — no more work ──────────────────────────────────
        if mins <= 0 and post_close_done:
            break

        # ── MARKET OPEN — regular sample ──────────────────────────────────
        sample_n += 1

        # Fetch quotes
        print(f"[{now.strftime('%H:%M')}] Sample #{sample_n} | {mins:.0f}min left | Fetching {len(key_syms)} quotes...", end="", flush=True)
        live_quotes = get_all_quotes(key_syms)
        nifty_q = live_quotes.get("NIFTY", {})
        print(f" NIFTY={nifty_q.get('ltp','?')} ({nifty_q.get('changePct',0):+.2f}%)")

        # Score all 218
        print(f"[{now.strftime('%H:%M')}] Scoring 218 symbols...", end="", flush=True)
        scores = score_all(estimator, feat_names, normalizer)
        print(f" done ({len(scores)} scored)")

        # ── Phil: apply feature weights (sector-regime filter + threshold) ────
        nifty_chg_now = float(nifty_q.get("changePct", 0) or 0)
        if _weight_mgr is not None and scores:
            regime = _weight_mgr.detect_regime(nifty_chg=nifty_chg_now)
            scores, _wt_summary = _weight_mgr.apply(scores, regime=regime, nifty_chg=nifty_chg_now)
            n_filtered = _wt_summary.get("n_filtered", 0)
            if n_filtered > 0:
                print(f"[{now.strftime('%H:%M')}] Feature weights: regime={regime}, "
                      f"{n_filtered} signals filtered (sector_dim={_wt_summary.get('n_sector_dimmed',0)} "
                      f"threshold={_wt_summary.get('n_threshold',0)})")

        # ── Phil: log ALL 218 forecasts (not just 26 tracked positions) ───────
        if _forecast_ledger is not None:
            n_logged = _forecast_ledger.record_session(
                scores=scores, session_date=session_date,
                nifty_chg=nifty_chg_now, model_version=schema,
            )
            if sample_n == 1:  # only print on first sample
                print(f"[{now.strftime('%H:%M')}] ForecastLedger: {n_logged} forecasts logged")

        # P&L calculation
        all_fp_quotes = get_all_quotes(list(fp_signals.keys()))
        live_quotes.update(all_fp_quotes)
        pnl = calc_pnl(fp_signals, live_quotes, excluded)

        # Dashboard
        print_dashboard(sample_n, now, nifty_q, scores, pnl, mins)

        # Save sample
        sample = {
            "timestamp":  now.isoformat(),
            "sample_n":   sample_n,
            "nifty_ltp":  nifty_q.get("ltp"),
            "nifty_chg":  nifty_q.get("changePct", 0),
            "n_scored":   len(scores),
            "n_long":     sum(1 for s in scores if s["direction"] == 1),
            "n_short":    sum(1 for s in scores if s["direction"] == -1),
            "top_long":   sorted([s for s in scores if s["direction"]==1],  key=lambda x: -x["score"])[:5],
            "top_short":  sorted([s for s in scores if s["direction"]==-1], key=lambda x:  x["score"])[:5],
            "pnl":        pnl,
        }
        all_samples.append(sample)
        with SESSION_LOG.open("a") as f:
            f.write(json.dumps(sample) + "\n")

        # Wait until next sample
        sleep_secs = min(SAMPLE_MINS * 60, max(30, int(mins * 60) - 60))
        print(f"\n  Next sample in {sleep_secs//60}m{sleep_secs%60:02d}s  ({now.strftime('%H:%M')} → {(now+timedelta(seconds=sleep_secs)).strftime('%H:%M')} IST)")
        time.sleep(sleep_secs)

    print("\nAutorun complete.")


def _append_close_to_report(now: datetime, nifty_q: dict, pnl: dict, n_samples: int) -> None:
    """Append end-of-day close summary to LIVE_SESSION_REPORT.md."""
    try:
        existing = REPORT_PATH.read_text() if REPORT_PATH.exists() else ""
        close_section = f"""

---

## CLOSE-OF-DAY UPDATE — {now.strftime('%H:%M IST')}

### Market Close Summary

| Metric | Value |
|--------|-------|
| Session samples | {n_samples} |
| NIFTY close | {nifty_q.get('ltp','?')} ({nifty_q.get('changePct',0):+.2f}%) |
| SHORT mean P&L | {pnl.get('short_mean',0):+.4f}% |
| LONG mean P&L | {pnl.get('long_mean',0):+.4f}% |
| Mean net P&L | {pnl.get('mean_net',0):+.4f}% |
| Win rate | {pnl.get('win_rate',0):.1f}% |
| Positions tracked | {pnl.get('n',0)} |

*Report auto-updated at {now.strftime('%H:%M IST')} by autorun_till_close.py*
"""
        REPORT_PATH.write_text(existing.rstrip() + close_section)
    except Exception as e:
        print(f"[warn] Could not update report: {e}")


if __name__ == "__main__":
    main()
