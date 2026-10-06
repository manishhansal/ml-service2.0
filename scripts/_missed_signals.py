#!/usr/bin/env python3
"""Print today's top missed/filtered signals."""
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).parent.parent
s = json.load(open(BASE / "artifacts/live_session/latest_scores.json"))
signals = s.get("signals", [])
missed   = [x for x in signals if x.get("direction") == 0]
missed_s = sorted(missed, key=lambda x: -abs(x["score"] - 0.5))

print(f"Filtered out: {len(missed)} of {len(signals)}")
print(f"NIFTY: {s.get('nifty_ltp')} ({s.get('nifty_chg', 0):+.2f}%)")
print()

print("TOP MISSED LONG (strong score, direction suppressed):")
print(f"  {'Symbol':<14} {'Score':>7}  Filter Reason")
print("  " + "-" * 55)
for sig in [x for x in missed_s if x["score"] > 0.5][:12]:
    r = sig.get("filter_reason") or "cs_neutral_band"
    print(f"  {sig['symbol']:<14} {sig['score']:>7.4f}  {r}")

print()
print("TOP MISSED SHORT (strong score, direction suppressed):")
print(f"  {'Symbol':<14} {'Score':>7}  Filter Reason")
print("  " + "-" * 55)
for sig in [x for x in missed_s if x["score"] <= 0.5][:12]:
    r = sig.get("filter_reason") or "cs_neutral_band"
    print(f"  {sig['symbol']:<14} {sig['score']:>7.4f}  {r}")

print()
print("FILTER BREAKDOWN (why signals were suppressed):")
reasons = Counter()
for sig in missed:
    r = sig.get("filter_reason") or "cs_neutral_band"
    key = r.split(":")[0]
    reasons[key] += 1
for reason, count in reasons.most_common(10):
    print(f"  {count:>4}  {reason}")
