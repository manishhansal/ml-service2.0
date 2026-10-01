# -*- coding: utf-8 -*-
"""Check how many open Oct-1 symbols we can resolve from tracker data."""
import json
from pathlib import Path

BASE = Path(".")
COST_BPS = 27.65

# Build sym_moves from tracker (actual intraday moves with real changePct)
tracker_lines = Path("artifacts/live_session/signal_tracker_v2.jsonl").read_text().strip().split("\n")
sym_moves = {}
for line in tracker_lines:
    try:
        d = json.loads(line)
        for sig in d.get("signal_results", []):
            sym = sig.get("symbol")
            chg = sig.get("chg_today")
            if sym and chg is not None and abs(chg) < 30:
                sym_moves[sym] = chg
        for m in d.get("top_winners", []) + d.get("top_losers", []):
            sym = m.get("symbol")
            chg = m.get("chg_today")
            if sym and chg is not None and abs(chg) < 30:
                sym_moves[sym] = chg
    except Exception:
        pass

# Load open Oct-1 forecasts with their directions
open_records = {}
for line in Path("artifacts/forward_paper/forecasts.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    try:
        r = json.loads(line)
        if r.get("session_date") == "2026-10-01" and r.get("status") == "open":
            open_records[r["symbol"]] = r.get("direction", 0)
    except Exception:
        pass

print(f"Open symbols remaining: {len(open_records)}")
print(f"Symbols found in tracker: {len(sym_moves)}")

# Build realized_returns for symbols we can match
new_resolved = {}
for sym, direction in open_records.items():
    if sym not in sym_moves:
        continue
    if direction == 0:
        continue
    chg = sym_moves[sym]
    gross = direction * chg
    net = gross - (COST_BPS / 100)
    new_resolved[sym] = round(net, 4)

print(f"Can resolve from tracker: {len(new_resolved)}")
print(f"Sample: {list(new_resolved.items())[:8]}")
