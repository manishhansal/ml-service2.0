#!/usr/bin/env python3
"""
scripts/signal_tracker_v2.py
-------------------------------
Live intraday signal tracker: compares ML-service2.0 signals against
actual NSE market movers in real time.

Runs continuously during market hours, capturing:
  - ML signal direction + conviction grade each cycle
  - Actual LTP change from session open to current time
  - Whether the ML signal correctly predicted direction
  - Top 20 market movers (both actual winners ML may have missed)

Saves a structured JSONL log and prints a live comparison table every
5 minutes.  At market close, writes a session summary used for RCA.

Usage:
    PYTHONPATH=. python3 scripts/signal_tracker_v2.py
    PYTHONPATH=. python3 scripts/signal_tracker_v2.py --interval 5
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE        = Path(__file__).parent.parent
SESSION_DIR = BASE / "artifacts" / "live_session"
TRACKER_LOG = SESSION_DIR / "signal_tracker_v2.jsonl"

env = {}
env_path = BASE / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()

DATA_URL = env.get("DATA_SERVICE_2_URL", "http://localhost:8200")
DATA_KEY = env.get("DATA_SERVICE_API_KEY", "")

IST = timezone(timedelta(hours=5, minutes=30))


def ist_now() -> datetime:
    return datetime.now(tz=IST)


LIVE_QUOTES_PATH = SESSION_DIR / "live_quotes.json"


def load_autorun_quotes() -> dict[str, dict]:
    """Read the live_quotes.json that autorun writes after every sample.

    This avoids competing with autorun for the data-service rate limit.
    Returns a dict of {symbol: quote_dict} with valid ltp > 0 only.
    """
    if not LIVE_QUOTES_PATH.exists():
        return {}
    try:
        raw = json.loads(LIVE_QUOTES_PATH.read_text())
        quotes = raw.get("quotes", {}) or {}
        return {
            sym: q for sym, q in quotes.items()
            if isinstance(q, dict) and q.get("ltp") is not None and float(q.get("ltp", 0) or 0) > 0
        }
    except Exception:
        return {}


def get_ltp(symbols: list[str]) -> dict[str, dict]:
    """Fetch live LTP for symbols NOT covered by autorun's live_quotes.json.

    Rate: 0.15 s/req (~6.7 req/s) — stays within 500 req/60 s service limit.
    """
    result = {}
    for sym in symbols:
        url = f"{DATA_URL}/v1/india/quotes/{sym}"
        req = urllib.request.Request(url, headers={"X-API-KEY": DATA_KEY})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                d = json.loads(r.read().decode())
                dd = d.get("data", {}) or {}
                if dd.get("ltp") is not None:
                    result[sym] = dd
        except Exception:
            pass
        time.sleep(0.15)  # 6.7 req/s — safely under 500/60s limit
    return result


def get_all_quotes_batch(symbols: list[str], batch_size: int = 50) -> dict[str, dict]:
    """Fetch LTP for all symbols — uses per-symbol endpoint internally."""
    return get_ltp(symbols)


def load_ml_signals() -> dict[str, dict]:
    """Load latest ML signals from latest_scores.json."""
    scores_file = SESSION_DIR / "latest_scores.json"
    if not scores_file.exists():
        return {}
    try:
        d = json.loads(scores_file.read_text())
        signals = d.get("signals", [])
        # Include ALL scored symbols (direction=0 too) so we can detect
        # large-cap stocks that moved big but the model left neutral.
        return {
            s["symbol"]: {
                "score":        s.get("score", 0.5),
                "direction":    s.get("direction", 0),
                "conviction":   s.get("conviction", "D"),
                "data_date":    s.get("data_date", ""),
                "has_live_ltp": s.get("has_live_ltp", False),
                "ltp_override": s.get("ltp_override", ""),
                "sector_boost": s.get("sector_boost", ""),
                "stock_dampened": s.get("stock_dampened", ""),
            }
            for s in signals
        }
    except Exception:
        return {}


class SessionTracker:
    """Tracks ML signals vs actual moves throughout the day."""

    def __init__(self, session_date: str) -> None:
        self.session_date = session_date
        self.session_open: dict[str, float] = {}   # symbol → open LTP
        self.snapshots:    list[dict] = []
        SESSION_DIR.mkdir(parents=True, exist_ok=True)

    def capture(self, sample_n: int) -> dict:
        """Take one snapshot: fetch LTPs, compare to ML signals."""
        now = ist_now()

        # Load current ML signals
        ml_signals = load_ml_signals()
        active_syms = list(ml_signals.keys())

        # Key symbols always monitored for missed-mover detection
        key_syms = [
            "NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "HDFCBANK", "INFY",
            "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN", "WIPRO", "ITC",
            "BHARTIARTL", "HINDALCO", "JSWSTEEL", "LT", "NTPC", "ONGC",
            "TITAN", "BAJFINANCE", "MARUTI", "M&M", "BAJAJFINSV", "ADANIENT",
            "EICHERMOT", "ULTRACEMCO", "AMBUJACEM", "SUNPHARMA", "DRREDDY",
        ]

        # ── Primary: read live_quotes.json written by autorun ─────────────
        # This covers all symbols autorun fetched (~43 of 218) without using
        # any of the 500 req/60s rate-limit budget.
        all_quotes: dict[str, dict] = load_autorun_quotes()

        # ── Fallback: API calls only for key symbols not in autorun quotes ─
        # Typically just NIFTY/BANKNIFTY and a handful of key movers.
        missing_keys = [s for s in key_syms if s not in all_quotes]
        if missing_keys:
            api_quotes = get_ltp(missing_keys)   # 0.15 s/req — rate-safe
            all_quotes.update(api_quotes)

        # Also sample ~30 random symbols for missed-mover detection
        from pathlib import Path as _P
        universe = [p.stem for p in (_P("data/1d/1d")).glob("*.parquet")]
        import random; random.seed(int(now.timestamp()) // 300)  # stable per 5-min window
        already_have = set(all_quotes.keys())
        sample_extra = random.sample(
            [s for s in universe if s not in already_have],
            min(30, max(0, len(universe) - len(already_have)))
        )
        if sample_extra:
            extra_quotes = get_ltp(sample_extra)
            all_quotes.update(extra_quotes)

        # Record session opens on first sample
        if not self.session_open:
            for sym, q in all_quotes.items():
                ltp = q.get("ltp")
                op  = q.get("open") or q.get("prevClose") or ltp
                if ltp and op:
                    self.session_open[sym] = float(op)

        # ── Build signal comparison ────────────────────────────────────────
        signal_results: list[dict] = []
        neutral_missed: list[dict] = []   # direction=0 but stock moved ≥1.5%
        for sym, sig in ml_signals.items():
            q    = all_quotes.get(sym, {})
            ltp  = float(q.get("ltp", 0) or 0)
            if ltp <= 0:
                continue   # skip symbols with failed LTP fetch
            op   = self.session_open.get(sym) or float(q.get("open") or q.get("prevClose") or 0) or ltp
            if op <= 0:
                continue
            chg_pct = ((ltp - op) / op * 100)
            chg_today = float(q.get("changePct", chg_pct) or chg_pct)
            # Sanity check: reject data errors
            if abs(chg_today) > 30:
                chg_today = chg_pct
            if abs(chg_today) > 30:
                continue

            ml_dir  = sig["direction"]
            grade   = sig["conviction"]

            # direction=0 → model was neutral; track separately for gap analysis
            if ml_dir == 0:
                if abs(chg_today) >= 1.5:
                    neutral_missed.append({
                        "symbol":    sym,
                        "score":     sig["score"],
                        "chg_today": round(chg_today, 2),
                        "ltp":       round(ltp, 2),
                        "implied":   "LONG" if sig["score"] > 0.55 else ("SHORT" if sig["score"] < 0.45 else "FLAT"),
                    })
                continue   # exclude from accuracy computation

            correct = (ml_dir == 1 and chg_today > 0.1) or (ml_dir == -1 and chg_today < -0.1)
            neutral = abs(chg_today) <= 0.1

            signal_results.append({
                "symbol":       sym,
                "direction":    ml_dir,
                "label":        "LONG" if ml_dir == 1 else "SHORT",
                "grade":        grade,
                "score":        sig["score"],
                "has_live_ltp": sig.get("has_live_ltp", False),
                "ltp_override": sig.get("ltp_override", ""),
                "ltp":          round(ltp, 2),
                "chg_today":    round(chg_today, 2),
                "correct":      correct,
                "neutral":      neutral,
                "wrong":        not correct and not neutral,
            })

        # ── Top movers the model might have missed ────────────────────────
        movers: list[dict] = []
        for sym, q in all_quotes.items():
            ltp = float(q.get("ltp", 0) or 0)
            if ltp <= 0:
                continue   # skip symbols with failed/zero LTP fetch
            op  = self.session_open.get(sym) or float(q.get("open") or q.get("prevClose") or 0) or ltp
            if op <= 0:
                continue
            chg = ((ltp - op) / op * 100)
            chg_today = float(q.get("changePct", chg) or chg)
            # Sanity-check: changePct > 30% or < -30% is likely a data error
            if abs(chg_today) > 30:
                chg_today = chg  # fallback to calculated
            if abs(chg_today) > 30:
                continue  # still unreasonable — skip
            ml_has_signal = sym in ml_signals and ml_signals[sym]["direction"] != 0
            movers.append({
                "symbol":       sym,
                "chg_today":    round(chg_today, 2),
                "ltp":          round(ltp, 2),
                "ml_signal":    ml_signals.get(sym, {}).get("label", "NONE") if ml_has_signal else "NONE",
                "ml_direction": ml_signals.get(sym, {}).get("direction", 0) if ml_has_signal else 0,
                "missed":       not ml_has_signal and abs(chg_today) > 1.5,
            })

        top_winners = sorted([m for m in movers if m["chg_today"] > 0], key=lambda x: -x["chg_today"])[:10]
        top_losers  = sorted([m for m in movers if m["chg_today"] < 0], key=lambda x:  x["chg_today"])[:10]
        missed_winners = [m for m in top_winners if m["missed"]]
        missed_losers  = [m for m in top_losers  if m["missed"]]

        # ── Accuracy stats ─────────────────────────────────────────────────
        # Only count direction!=0 signals in accuracy. Split LONGs into
        # verified (has_live_ltp=True) and unverified for separate reporting.
        with_move = [s for s in signal_results if not s["neutral"]]
        n_correct = sum(1 for s in with_move if s["correct"])
        n_wrong   = sum(1 for s in with_move if s["wrong"])
        accuracy  = n_correct / len(with_move) if with_move else 0.0

        longs_all      = [s for s in signal_results if s["direction"] == 1]
        longs_verified = [s for s in longs_all if s.get("has_live_ltp")]
        shorts         = [s for s in signal_results if s["direction"] == -1]

        long_win  = (
            sum(1 for s in longs_verified if s["correct"]) / max(len(longs_verified), 1)
            if longs_verified else 0.0
        )
        short_win = sum(1 for s in shorts if s["correct"]) / max(len(shorts), 1)

        nifty_q   = all_quotes.get("NIFTY", {})
        nifty_ltp = float(nifty_q.get("ltp", 0) or 0)
        nifty_chg = float(nifty_q.get("changePct", 0) or 0)

        snapshot = {
            "timestamp":         now.isoformat(),
            "sample_n":          sample_n,
            "session_date":      self.session_date,
            "nifty_ltp":         nifty_ltp,
            "nifty_chg":         round(nifty_chg, 3),
            "n_ml_signals":      len(signal_results),
            "n_long":            len(longs_all),
            "n_long_verified":   len(longs_verified),
            "n_short":           len(shorts),
            "n_correct":         n_correct,
            "n_wrong":           n_wrong,
            "accuracy":          round(accuracy, 4),
            "long_win_rate":     round(long_win, 4),
            "short_win_rate":    round(short_win, 4),
            "missed_winners":    missed_winners[:5],
            "missed_losers":     missed_losers[:5],
            "neutral_missed":    sorted(neutral_missed, key=lambda x: abs(x["chg_today"]), reverse=True)[:10],
            "signal_results":    signal_results,
            "top_winners":       top_winners[:5],
            "top_losers":        top_losers[:5],
        }
        self.snapshots.append(snapshot)

        # Append to JSONL log
        with TRACKER_LOG.open("a") as f:
            f.write(json.dumps(snapshot) + "\n")

        return snapshot

    def print_dashboard(self, snap: dict) -> None:
        now_str = ist_now().strftime("%H:%M")
        nifty_c = snap["nifty_chg"]
        acc     = snap["accuracy"] * 100
        lwr     = snap["long_win_rate"] * 100
        swr     = snap["short_win_rate"] * 100
        n_lv    = snap.get("n_long_verified", snap["n_long"])

        print(f"\n{'='*72}")
        print(f"  📊 SIGNAL TRACKER v2  |  {now_str} IST  |  NIFTY {snap['nifty_ltp']} ({nifty_c:+.2f}%)")
        print(f"  Signals: {snap['n_ml_signals']} (L={snap['n_long']}[ver={n_lv}] S={snap['n_short']})")
        print(f"  Accuracy: {acc:.1f}%  |  LONG win (verified): {lwr:.1f}%  |  SHORT win: {swr:.1f}%")
        print(f"  Correct={snap['n_correct']}  Wrong={snap['n_wrong']}")

        # Wrong signals
        wrong = [s for s in snap["signal_results"] if s["wrong"]]
        if wrong:
            print(f"\n  ❌ WRONG SIGNALS ({len(wrong)}):")
            for s in sorted(wrong, key=lambda x: abs(x["chg_today"]), reverse=True)[:6]:
                arrow = "↑" if s["chg_today"] > 0 else "↓"
                ml_lbl = s["label"]
                print(f"    {s['symbol']:15s}  ML={ml_lbl:5s} {s['grade']}  actual={arrow}{abs(s['chg_today']):.1f}%  score={s['score']:.3f}")

        # Missed movers
        missed_w = snap["missed_winners"]
        missed_l = snap["missed_losers"]
        if missed_w or missed_l:
            print(f"\n  🎯 MISSED MOVERS (model had no signal):")
            for m in (missed_w + missed_l)[:6]:
                arrow = "↑" if m["chg_today"] > 0 else "↓"
                print(f"    {m['symbol']:15s}  actual={arrow}{abs(m['chg_today']):.1f}%  (no ML signal)")

        # Neutral-but-moving: model scored but left as direction=0
        nm = snap.get("neutral_missed", [])
        if nm:
            print(f"\n  ⚠️  MODEL NEUTRAL BUT MOVED ≥1.5% (direction=0, coverage gap):")
            for m in nm[:6]:
                arrow = "↑" if m["chg_today"] > 0 else "↓"
                implied = m.get("implied", "?")
                print(f"    {m['symbol']:15s}  actual={arrow}{abs(m['chg_today']):.1f}%  score={m['score']:.3f}  implied={implied}")

        # Correct signals
        correct = [s for s in snap["signal_results"] if s["correct"]]
        if correct:
            print(f"\n  ✅ CORRECT SIGNALS ({len(correct)}):")
            for s in sorted(correct, key=lambda x: abs(x["chg_today"]), reverse=True)[:6]:
                arrow = "↑" if s["chg_today"] > 0 else "↓"
                print(f"    {s['symbol']:15s}  ML={s['label']:5s} {s['grade']}  actual={arrow}{abs(s['chg_today']):.1f}%")

        print(f"{'='*72}")


def main(interval_mins: int = 5) -> None:
    now = ist_now()
    session_date = now.strftime("%Y-%m-%d")

    print(f"\n{'='*72}")
    print(f"  SIGNAL TRACKER V2  —  {session_date}")
    print(f"  Tracking ML signals vs actual NSE moves every {interval_mins} min")
    print(f"  Log: {TRACKER_LOG}")
    print(f"{'='*72}\n")

    tracker  = SessionTracker(session_date)
    sample_n = 0

    while True:
        now_ist = ist_now()
        t = (now_ist.hour, now_ist.minute)

        # Market hours: 9:15 AM – 3:30 PM IST
        if not ((9, 15) <= t <= (15, 30)) or now_ist.weekday() >= 5:
            if t > (15, 30):
                print(f"\n[{now_ist.strftime('%H:%M')}] Market closed. Final snapshot captured.")
                break
            print(f"[{now_ist.strftime('%H:%M')}] Market not open yet. Waiting...")
            time.sleep(60)
            continue

        sample_n += 1
        print(f"\n[{now_ist.strftime('%H:%M')}] Sample #{sample_n} ...", end="", flush=True)

        try:
            snap = tracker.capture(sample_n)
            print(f" done (acc={snap['accuracy']*100:.1f}%  L={snap['n_long']}/S={snap['n_short']})")
            tracker.print_dashboard(snap)
        except Exception as exc:
            print(f" ERROR: {exc}")

        time.sleep(interval_mins * 60)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=5)
    args = p.parse_args()
    main(args.interval)
