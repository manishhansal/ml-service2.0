# -*- coding: utf-8 -*-
"""Live log watcher — polls every 5s, prints new entries as they arrive."""
import json
import time
from pathlib import Path

AUTORUN  = Path("artifacts/live_session/autorun_log.jsonl")
TRACKER  = Path("artifacts/live_session/signal_tracker_v2.jsonl")
SEP = "-" * 72

def tail(path, n=1):
    """Return last n lines of a file."""
    lines = path.read_text().strip().split("\n")
    return lines[-n:]

def fmt_autorun(d):
    ts  = d["timestamp"][11:16]
    nch = d.get("nifty_chg", 0) or 0
    nl  = d.get("n_long", 0)
    ns  = d.get("n_short", 0)
    p   = d.get("pnl") or {}
    wr  = p.get("win_rate", 0) or 0
    sm  = p.get("short_mean", 0) or 0
    lm  = p.get("long_mean", 0) or 0
    mn  = p.get("mean_net", 0) or 0
    return (f"  [AUTORUN {ts}]  NIFTY={nch:+.2f}%  "
            f"L={nl} S={ns}  win={wr:.0f}%  "
            f"short={sm:+.2f}%  long={lm:+.2f}%  net={mn:+.3f}%")

def fmt_tracker(d):
    ts  = d["timestamp"][11:16]
    acc = d.get("accuracy", 0) * 100
    nc  = d.get("n_correct", 0)
    nw  = d.get("n_wrong", 0)
    ns  = d.get("n_ml_signals", 0)
    swr = d.get("short_win_rate", 0) * 100
    nch = d.get("nifty_chg", 0)
    wrongs = [s["symbol"] for s in d.get("signal_results", []) if s.get("wrong")]
    wrong_str = f"  wrong=[{', '.join(wrongs[:4])}]" if wrongs else ""
    return (f"  [TRACKER {ts}]  acc={acc:.0f}%  correct={nc} wrong={nw}  "
            f"signals={ns}  short_win={swr:.0f}%  NIFTY={nch:+.2f}%{wrong_str}")

print("=" * 72)
print("  LIVE LOG WATCHER  —  refreshes every 5s  |  Ctrl+C to stop")
print("=" * 72)

last_autorun  = None
last_tracker  = None

while True:
    try:
        # ── Autorun ────────────────────────────────────────────────────────────
        lines = tail(AUTORUN, 1)
        for line in lines:
            try:
                d = json.loads(line)
                ts = d["timestamp"]
                if ts != last_autorun:
                    last_autorun = ts
                    print(fmt_autorun(d))
            except Exception:
                pass

        # ── Tracker ────────────────────────────────────────────────────────────
        lines2 = tail(TRACKER, 1)
        for line in lines2:
            try:
                d = json.loads(line)
                ts = d["timestamp"]
                if ts != last_tracker:
                    last_tracker = ts
                    print(fmt_tracker(d))
                    # Print correct/wrong detail on tracker updates
                    correct = [s["symbol"] for s in d.get("signal_results", []) if s.get("correct")]
                    if correct:
                        print(f"           correct=[{', '.join(correct[:8])}]")
                    neutral = d.get("neutral_missed", [])
                    if neutral:
                        gaps = [f"{m['symbol']}({m['chg_today']:+.1f}%)" for m in neutral[:4]]
                        print(f"           coverage_gap=[{', '.join(gaps)}]")
                    print(SEP)
            except Exception:
                pass

        time.sleep(5)

    except KeyboardInterrupt:
        print("\nStopped.")
        break
    except Exception as e:
        time.sleep(5)
