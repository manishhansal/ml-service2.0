# -*- coding: utf-8 -*-
"""End-of-day comprehensive analysis: profitable signals vs ML signals."""
import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent.parent
SESSION = BASE / "artifacts/live_session"

# ── Load all data ─────────────────────────────────────────────────────────────
print("Loading session data...")

# Autorun log (all cycles)
autorun_samples = []
for line in (SESSION/"autorun_log.jsonl").read_text().strip().split("\n"):
    try: autorun_samples.append(json.loads(line))
    except: pass

# Tracker log (all accuracy samples)
tracker_samples = []
for line in (SESSION/"signal_tracker_v2.jsonl").read_text().strip().split("\n"):
    try: tracker_samples.append(json.loads(line))
    except: pass

# Latest scores (EOD)
latest = json.loads((SESSION/"latest_scores.json").read_text())

# Session summary (EOD resolved P&L)
summary = json.loads((SESSION/"session_summary.json").read_text())

# Score threshold sweep
sweep_path = BASE / "reports/score_threshold_sweep_2026-10-01.json"
sweep = json.loads(sweep_path.read_text()) if sweep_path.exists() else {}

# Symbol IC state
ic_state = json.loads((SESSION/"symbol_ic_state.json").read_text())

# Live quotes (EOD prices)
live_q = json.loads((SESSION/"live_quotes.json").read_text())
quotes = live_q.get("quotes", {})

SEP = "=" * 72

# ── 1. SESSION OVERVIEW ───────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  1. SESSION OVERVIEW  —  Oct 1, 2026")
print(SEP)
n_samples = len(autorun_samples)
last = autorun_samples[-1] if autorun_samples else {}
pnl = last.get("pnl", {})
print(f"  Autorun samples     : {n_samples}")
print(f"  Tracker samples     : {len(tracker_samples)}")
print(f"  NIFTY close         : {last.get('nifty_ltp','?')} ({last.get('nifty_chg',0):+.2f}%)")
print(f"  Win rate (final)    : {pnl.get('win_rate',0):.1f}%")
print(f"  Mean net (final)    : {pnl.get('mean_net',0):+.3f}%")
print(f"  SHORT avg P&L       : {pnl.get('short_mean',0):+.3f}%")
print(f"  LONG avg P&L        : {pnl.get('long_mean',0):+.3f}%")
print(f"  Positions tracked   : {pnl.get('n',0)}")

# ── 2. TRACKER ACCURACY OVER THE DAY ─────────────────────────────────────────
print(f"\n{SEP}")
print("  2. ACCURACY PROGRESSION (tracker, all samples)")
print(SEP)
print(f"  {'Time':6s}  {'Acc':6s}  {'L win':7s}  {'S win':7s}  {'Sig':5s}  {'Correct':8s}  NIFTY")
for s in tracker_samples:
    ts = s.get("timestamp", "")[:19]
    t  = ts[11:16]
    acc = s.get("accuracy", 0) * 100
    lwr = s.get("long_win_rate", 0) * 100
    swr = s.get("short_win_rate", 0) * 100
    ns  = s.get("n_ml_signals", 0)
    nc  = s.get("n_correct", 0)
    nw  = s.get("n_wrong", 0)
    nch = s.get("nifty_chg", 0)
    print(f"  {t:6s}  {acc:5.1f}%  {lwr:6.1f}%  {swr:6.1f}%  {ns:5d}  {nc:3d}/{nc+nw:3d}     {nch:+.2f}%")

# ── 3. ALL WRONG SIGNALS (full day) ──────────────────────────────────────────
print(f"\n{SEP}")
print("  3. ALL WRONG SIGNALS  (across entire session)")
print(SEP)
wrong_tally = defaultdict(list)
for s in tracker_samples:
    ts = s.get("timestamp","")[:19][11:16]
    for sig in s.get("signal_results", []):
        if sig.get("wrong"):
            sym = sig["symbol"]
            wrong_tally[sym].append({
                "time":    ts,
                "label":   sig.get("label"),
                "score":   sig.get("score", 0),
                "actual":  sig.get("chg_today", 0),
            })

print(f"  {'Symbol':14s}  {'Direction':9s}  {'Score':6s}  {'Times wrong':12s}  Actual moves")
for sym, wrongs in sorted(wrong_tally.items(), key=lambda x: -len(x[1])):
    dirs   = set(w["label"] for w in wrongs)
    scores = [w["score"] for w in wrongs]
    actuals = [w["actual"] for w in wrongs]
    times  = [w["time"] for w in wrongs]
    print(f"  {sym:14s}  {'/'.join(dirs):9s}  {sum(scores)/len(scores):.3f}  {len(wrongs):2d}x {times[0]}–{times[-1]}  "
          f"{min(actuals):+.1f}% to {max(actuals):+.1f}%")

# ── 4. ALL MISSED MOVERS (market moves with no ML signal) ─────────────────────
print(f"\n{SEP}")
print("  4. MISSED MOVERS  (moved ≥1.5% with no ML signal — full day)")
print(SEP)
missed_tally = defaultdict(list)
for s in tracker_samples:
    ts = s.get("timestamp","")[:19][11:16]
    for m in s.get("missed_winners", []) + s.get("missed_losers", []):
        sym = m["symbol"]
        missed_tally[sym].append({"time": ts, "chg": m["chg_today"]})
    # Also check neutral_missed
    for m in s.get("neutral_missed", []):
        sym = m["symbol"]
        missed_tally[sym].append({"time": ts, "chg": m["chg_today"], "neutral": True})

print(f"  {'Symbol':14s}  {'Miss count':10s}  {'Avg move':9s}  {'Max move':9s}  Type")
for sym, misses in sorted(missed_tally.items(), key=lambda x: -abs(sum(m["chg"] for m in x[1])/len(x[1]))):
    avg_chg = sum(m["chg"] for m in misses) / len(misses)
    max_chg = max(misses, key=lambda x: abs(x["chg"]))["chg"]
    is_neut = any(m.get("neutral") for m in misses)
    mtype   = "NEUTRAL" if is_neut else "NO SIGNAL"
    if abs(avg_chg) >= 1.0:
        print(f"  {sym:14s}  {len(misses):2d}x misses    {avg_chg:+.1f}%      {max_chg:+.1f}%      {mtype}")

# ── 5. EOD P&L — TOP WINNERS AND LOSERS ──────────────────────────────────────
print(f"\n{SEP}")
print("  5. EOD P&L — FINAL POSITION OUTCOMES")
print(SEP)
positions = summary.get("final_pnl", {}).get("positions", [])
if not positions:
    positions = pnl.get("positions", [])
positions_sorted = sorted(positions, key=lambda x: x.get("net_pct", 0) or 0, reverse=True)

print(f"  {'Symbol':14s}  {'Direction':9s}  {'Entry':8s}  {'Exit LTP':8s}  {'Net P&L':9s}")
print(f"  --- TOP 5 WINNERS ---")
for p in positions_sorted[:5]:
    print(f"  {p['symbol']:14s}  {p.get('direction',0):+d}→{'L' if p.get('direction',0)==1 else 'S'}  "
          f"  {p.get('entry',0):8.2f}  {p.get('ltp',0):8.2f}  {p.get('net_pct',0):+.3f}%")
print(f"  --- BOTTOM 5 LOSERS ---")
for p in positions_sorted[-5:]:
    print(f"  {p['symbol']:14s}  {p.get('direction',0):+d}→{'L' if p.get('direction',0)==1 else 'S'}  "
          f"  {p.get('entry',0):8.2f}  {p.get('ltp',0):8.2f}  {p.get('net_pct',0):+.3f}%")

# ── 6. MARKET TOP MOVERS vs ML UNIVERSE ──────────────────────────────────────
print(f"\n{SEP}")
print("  6. ACTUAL MARKET TOP MOVERS  vs  ML SIGNAL COVERAGE")
print(SEP)
all_quotes_list = []
for sym, q in quotes.items():
    if isinstance(q, dict) and q.get("changePct") is not None and q.get("ltp"):
        try:
            all_quotes_list.append({
                "symbol": sym,
                "ltp":    float(q.get("ltp", 0)),
                "chg":    float(q.get("changePct", 0)),
            })
        except: pass

all_quotes_list.sort(key=lambda x: x["chg"], reverse=True)
all_signals = {s["symbol"]: s for s in latest.get("signals", [])}

print(f"  TOP 15 GAINERS today:")
for q in all_quotes_list[:15]:
    sym = q["symbol"]
    sig = all_signals.get(sym)
    if sig:
        ml = "LONG ✓" if sig["direction"]==1 else ("SHORT ✗" if sig["direction"]==-1 else "NEUTRAL")
    else:
        ml = "NOT SCORED"
    print(f"  {sym:14s}  {q['chg']:+.2f}%   ML={ml}")

print(f"\n  TOP 15 LOSERS today:")
for q in all_quotes_list[-15:][::-1]:
    sym = q["symbol"]
    sig = all_signals.get(sym)
    if sig:
        ml = "SHORT ✓" if sig["direction"]==-1 else ("LONG ✗" if sig["direction"]==1 else "NEUTRAL")
    else:
        ml = "NOT SCORED"
    print(f"  {sym:14s}  {q['chg']:+.2f}%   ML={ml}")

# ── 7. SCORE THRESHOLD SWEEP RESULTS ─────────────────────────────────────────
print(f"\n{SEP}")
print("  7. SCORE THRESHOLD SWEEP  (post-session optimal)")
print(SEP)
if sweep:
    print(f"  Optimal threshold : {sweep.get('optimal_threshold', '?')}")
    print(f"  Optimal win rate  : {sweep.get('optimal_win_rate', 0)*100:.1f}%")
    print(f"  Current threshold : 0.10 (dynamic via weight manager)")
    thresh_table = sweep.get("threshold_table", [])
    if thresh_table:
        print(f"  {'Threshold':10s}  {'Win rate':10s}  {'N signals':10s}")
        for row in thresh_table[:8]:
            print(f"  {row.get('threshold',0):.2f}        {row.get('win_rate',0)*100:6.1f}%     {row.get('n_signals',0)}")
else:
    print("  No sweep data for today")

# ── 8. SYMBOL IC STATE ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  8. SYMBOL IC TRACKER  (chronically wrong symbols)")
print(SEP)
ic_syms = ic_state.get("symbols", {})
dead = [(sym, v) for sym, v in ic_syms.items() if v.get("dead")]
bad  = sorted([(sym, v) for sym, v in ic_syms.items()
               if v.get("rolling_ic", 0) < -0.03 and not v.get("dead")],
              key=lambda x: x[1].get("rolling_ic", 0))
print(f"  Dead (IC < -0.05, on cooldown): {len(dead)}")
for sym, v in dead[:5]:
    print(f"    {sym:12s}  IC={v.get('rolling_ic',0):+.4f}  trades={v.get('n_trades',0)}")
print(f"  Weak (IC < -0.03): {len(bad)}")
for sym, v in bad[:5]:
    print(f"    {sym:12s}  IC={v.get('rolling_ic',0):+.4f}  trades={v.get('n_trades',0)}")

# ── 9. GAP SUMMARY ────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  9. GAP SUMMARY  (actionable fixes for tomorrow)")
print(SEP)

# Quantify each gap
long_pnl = pnl.get("long_mean", 0)
short_pnl = pnl.get("short_mean", 0)
final_acc = tracker_samples[-1].get("accuracy", 0) * 100 if tracker_samples else 0

print(f"  A. LONG book avg P&L: {long_pnl:+.3f}%  (SHORTs: {short_pnl:+.3f}%)")
print(f"     Mid-cap LONGs (no live LTP) are pure beta exposure in down markets")
print()

# Most missed movers
top_missed = sorted(missed_tally.items(), key=lambda x: -abs(sum(m["chg"] for m in x[1])/len(x[1])))
print(f"  B. MOST MISSED profitable signals:")
for sym, misses in top_missed[:8]:
    avg_chg = sum(m["chg"] for m in misses) / len(misses)
    if abs(avg_chg) >= 1.5:
        print(f"     {sym:12s}  {avg_chg:+.1f}% avg move  {len(misses)}x missed")

print()
# Most wrong signals
print(f"  C. MOST WRONG signals (recurring):")
for sym, wrongs in sorted(wrong_tally.items(), key=lambda x: -len(x[1]))[:6]:
    dirs   = set(w["label"] for w in wrongs)
    avg_act = sum(w["actual"] for w in wrongs) / len(wrongs)
    print(f"     {sym:12s}  ML={'/'.join(dirs):5s}  actual={avg_act:+.1f}%  {len(wrongs)}x wrong")

print(f"\n  Final accuracy: {final_acc:.1f}% on {tracker_samples[-1].get('n_ml_signals',0)} signals")
print(f"  SHORT win rate: {tracker_samples[-1].get('short_win_rate',0)*100:.1f}%")
print(f"  LONG  win rate: {tracker_samples[-1].get('long_win_rate',0)*100:.1f}%")
print()
