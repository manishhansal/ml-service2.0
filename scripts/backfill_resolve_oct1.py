# -*- coding: utf-8 -*-
"""
Back-fill ForecastLedger resolution for 2026-10-01.
Three-tier approach to maximise resolved symbol count:
  Tier 1: FP positions — accurate P&L from session_summary
  Tier 2: Scored symbols with changePct from live_quotes.json
  Tier 3: Remaining open forecasts via signal_tracker changePct
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from src.analytics.forecast_ledger import ForecastLedger  # noqa: E402

SESSION_DATE = "2026-10-01"
COST_BPS = 27.65

latest  = json.loads((BASE / "artifacts/live_session/latest_scores.json").read_text())
lq_raw  = json.loads((BASE / "artifacts/live_session/live_quotes.json").read_text())
summary = json.loads((BASE / "artifacts/live_session/session_summary.json").read_text())

quotes  = lq_raw.get("quotes", {})
signals = latest.get("signals", [])
fp_pos  = summary.get("final_pnl", {}).get("positions", [])

# Build tracker changePct lookup
tracker_chg: dict[str, float] = {}
tk = BASE / "artifacts/live_session/signal_tracker_v2.jsonl"
if tk.exists():
    for line in tk.read_text().strip().split("\n"):
        try:
            d = json.loads(line)
            for sig in d.get("signal_results", []):
                s, c = sig.get("symbol"), sig.get("chg_today")
                if s and c is not None and abs(c) < 30:
                    tracker_chg[s] = c
            for m in d.get("top_winners", []) + d.get("top_losers", []):
                s, c = m.get("symbol"), m.get("chg_today")
                if s and c is not None and abs(c) < 30:
                    tracker_chg[s] = c
        except Exception:
            pass

realized: dict[str, float] = {}

# Tier 1
for p in fp_pos:
    s, n = p.get("symbol"), p.get("net_pct")
    if s and n is not None:
        realized[s] = n
print(f"Tier 1 (FP positions):  {len(realized)}")

# Tier 2
n2 = 0
for sig in signals:
    s = sig.get("symbol")
    if not s or s in realized:
        continue
    d = sig.get("direction", 0)
    if d == 0:
        continue
    q = quotes.get(s, {})
    c = q.get("changePct")
    if c is None:
        continue
    realized[s] = round(d * float(c) - COST_BPS / 100, 4)
    n2 += 1
print(f"Tier 2 (live_quotes):   {n2}")

# Tier 3: load open forecast directions and match via tracker
open_dirs: dict[str, int] = {}
for line in (BASE / "artifacts/forward_paper/forecasts.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    try:
        r = json.loads(line)
        if r.get("session_date") == SESSION_DATE and r.get("status") == "open":
            d = int(r.get("direction", 0))
            if d != 0:
                open_dirs[r["symbol"]] = d
    except Exception:
        pass

n3 = 0
for sym, direction in open_dirs.items():
    if sym in realized:
        continue
    c = tracker_chg.get(sym)
    if c is None:
        continue
    realized[sym] = round(direction * c - COST_BPS / 100, 4)
    n3 += 1
print(f"Tier 3 (tracker):       {n3}")
print(f"Total:                  {len(realized)}")

# Run resolution
ledger = ForecastLedger()
report = ledger.resolve_session(session_date=SESSION_DATE, realized_returns=realized)
n_res = report.get("n_resolved", 0)
bd    = report.get("brier_delta", 0)
wr    = report.get("win_rate", 0)
print(f"\nResolved: {n_res}  brier_delta={bd:+.6f}  win_rate={wr*100:.1f}%")

# Verify
by_status: dict[str, int] = {}
for line in (BASE / "artifacts/forward_paper/forecasts.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    try:
        r = json.loads(line)
        if r.get("session_date") == SESSION_DATE:
            st = r.get("status", "?")
            by_status[st] = by_status.get(st, 0) + 1
    except Exception:
        pass

print("\nOct-1 status distribution:")
for k, v in sorted(by_status.items(), key=lambda x: -x[1]):
    print(f"  {k:15s}: {v}")
print("\nDone — refresh AlphaForge History tab.")
