# -*- coding: utf-8 -*-
"""Deep gap analysis for Oct 1 session."""
import json
from pathlib import Path

SESSION = Path("artifacts/live_session")
latest = json.loads((SESSION / "latest_scores.json").read_text())
sig_map = {s["symbol"]: s for s in latest["signals"]}
live_q = json.loads((SESSION / "live_quotes.json").read_text())["quotes"]
ic_state = json.loads((SESSION / "symbol_ic_state.json").read_text())
ic_syms = ic_state.get("symbols", {})

KEY_SYMS = {
    "NIFTY","BANKNIFTY","RELIANCE","HDFCBANK","ICICIBANK","INFY","TCS",
    "KOTAKBANK","AXISBANK","BHARTIARTL","SBIN","LT","MARUTI","WIPRO",
    "TITAN","NTPC","ONGC","BAJFINANCE","HINDUNILVR","ADANIENT",
}

MISSED = {
    "INFY": 39, "HDFCLIFE": 43, "M&M": 22, "EICHERMOT": 16,
    "ULTRACEMCO": 18, "KOTAKBANK": 17, "POWERGRID": 17, "MARUTI": 14,
    "GRASIM": 10, "TATASTEEL": 8, "ADANIPORTS": 6, "TCS": 8,
    "SBILIFE": 4, "JSWSTEEL": 3,
}

WRONG = {
    "HDFCBANK": 17, "INDUSINDBK": 12, "TATAMOTORS": 11, "BHARTIARTL": 11,
    "BRITANNIA": 9, "SUNPHARMA": 7, "SBIN": 6, "KOTAKBANK": 6, "CIPLA": 5,
    "INFY": 5, "HINDALCO": 4,
}

SEP = "=" * 70

print(SEP)
print("  MISSED MOVERS — live LTP & key_syms coverage")
print(SEP)
print(f"  {'Symbol':14s}  {'Missed':7s}  {'key_syms':9s}  {'live_ltp':9s}  EOD score  EOD dir")
for sym, cnt in sorted(MISSED.items(), key=lambda x: -x[1]):
    in_key = sym in KEY_SYMS
    q = live_q.get(sym, {})
    has_ltp = isinstance(q, dict) and q.get("ltp") is not None
    s = sig_map.get(sym, {})
    score = s.get("score", 0)
    direction = s.get("direction", "?")
    print(f"  {sym:14s}  {cnt:3d}x     {'YES' if in_key else 'no':9s}  {'YES' if has_ltp else 'no':9s}  {score:.3f}      {direction}")

print()
print(SEP)
print("  WRONG SIGNALS — IC state & root cause")
print(SEP)
print(f"  {'Symbol':14s}  {'Wrong':7s}  {'IC':8s}  {'Trades':7s}  {'Dead':5s}  EOD score  EOD dir")
for sym, cnt in sorted(WRONG.items(), key=lambda x: -x[1]):
    v = ic_syms.get(sym, {})
    ic = v.get("rolling_ic", None)
    n = v.get("n_trades", 0)
    dead = v.get("dead", False)
    s = sig_map.get(sym, {})
    score = s.get("score", 0)
    direction = s.get("direction", "?")
    ic_str = f"{ic:+.4f}" if ic is not None else "N/A"
    print(f"  {sym:14s}  {cnt:3d}x     {ic_str:8s}  {n:7d}  {'YES' if dead else 'no':5s}  {score:.3f}      {direction}")

print()
print(SEP)
print("  SCORE THRESHOLD SWEEP")
print(SEP)
sweep_path = Path("reports/score_threshold_sweep_2026-10-01.json")
if sweep_path.exists():
    sweep = json.loads(sweep_path.read_text())
    print(f"  Optimal threshold : {sweep.get('optimal_threshold')}")
    print(f"  Optimal win rate  : {sweep.get('optimal_win_rate', 0) * 100:.1f}%")
    print(f"  Current dynamic   : 0.05 – 0.15 (vol-adaptive)")
    tbl = sweep.get("threshold_table", [])
    print(f"  {'Threshold':10s}  {'Win rate':10s}  {'N signals':10s}")
    for row in tbl[:8]:
        print(f"  {row.get('threshold', 0):.2f}         {row.get('win_rate', 0) * 100:6.1f}%      {row.get('n_signals', 0):4d}")
else:
    print("  No sweep data")

print()
print(SEP)
print("  GAP SUMMARY — prioritized by impact")
print(SEP)
gaps = [
    ("CRITICAL", "HDFCBANK wrong SHORT 17x",
     "Bank stocks rising but model contracts-SHORT. BANK_SYMS dampening not firing enough"),
    ("CRITICAL", "INFY/HDFCLIFE missed 39x/43x",
     "IT/Insurance consistently up but model contrarian-neutral. IT UP boost needs lower threshold"),
    ("HIGH",     "M&M/EICHERMOT missed 22x/16x — sector boost intermittent",
     "Not in key_syms -> no guaranteed LTP -> sym_chg=0 -> boost skipped"),
    ("HIGH",     "INDUSINDBK wrong LONG 12x",
     "Persistent bad LONG. IC tracker should kill it, or add to excluded"),
    ("HIGH",     "LONG book avg -4.583%",
     "Mid-cap LONGs all beta exposure in down market. Need beta hedge or restrict to live-priced"),
    ("MEDIUM",   "TATAMOTORS wrong SHORT 11x",
     "Already excluded from FP but still scored. Should add to stock-level dampen exclusion"),
    ("MEDIUM",   "Threshold should be 0.15 (sweep-optimal)",
     "Current 0.10 lets too many borderline signals through (SBIN 0.532, BHARTIARTL 0.517)"),
    ("MEDIUM",   "GRASIM missed 10x — not in any sector",
     "Diversified conglomerate. Added to CEMENT_SYMS but needs reclassification"),
]
for severity, title, detail in gaps:
    print(f"  [{severity}] {title}")
    print(f"    -> {detail}")
    print()
