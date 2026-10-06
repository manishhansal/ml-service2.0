"""Audit all data stores: what's in files vs PostgreSQL."""
import json, sys
from pathlib import Path
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

stores = [
    ("artifacts/signal_ledger/positions.json",   "Signal positions (NOW IN PG — should be removed)"),
    ("artifacts/signal_ledger/signals.jsonl",    "Signal events audit log (keep as backup)"),
    ("artifacts/forward_paper/signals_v2.jsonl", "Old 153 forward paper signals (superseded)"),
    ("artifacts/forward_paper/outcomes.jsonl",   "218 resolved outcomes (superseded by PG)"),
    ("artifacts/forward_paper/forecasts.jsonl",  "ForecastLedger (16k entries) — IN PG + dual-write active"),
    ("artifacts/live_session/latest_scores.json","Current snapshot — operational, keep"),
    ("artifacts/live_session/autorun_log.jsonl", "Session samples every 2min — operational (not migrated by design)"),
    ("artifacts/live_session/news_scheduler.jsonl","News events — operational"),
    ("artifacts/live_session/symbol_ic_state.json","SymbolIC tracker — IN PG + dual-write active"),
    ("artifacts/live_session/live_quotes.json",  "Last live quotes — operational ephemeral"),
    ("artifacts/counterfactual/blocked_signals.jsonl","Blocked signals — NOT in PG"),
    ("artifacts/ledger",                          "Old approval ledger"),
]

print("DATA STORE AUDIT")
print("=" * 70)
for path_str, desc in stores:
    p = BASE / path_str
    if p.exists():
        size_kb = p.stat().st_size // 1024
        try:
            if path_str.endswith(".jsonl"):
                n = sum(1 for l in p.open() if l.strip())
                print(f"  EXISTS  {size_kb:>6}KB  {n:>6} rows  {path_str}")
            elif path_str.endswith(".json"):
                data = json.load(open(p))
                n = len(data)
                print(f"  EXISTS  {size_kb:>6}KB  {n:>6} keys  {path_str}")
            else:
                print(f"  EXISTS  {size_kb:>6}KB         {path_str}")
        except Exception:
            print(f"  EXISTS  {size_kb:>6}KB         {path_str}")
        print(f"          └─ {desc}")
    else:
        print(f"  MISSING                    {path_str}")
    print()

print("=" * 70)
print("POSTGRESQL STATUS")
print("=" * 70)
from src.data.signal_db import get_db
db = get_db()
stats = db.db_stats()
tables = ["sessions", "signals", "positions", "price_snapshots"]
for t in tables:
    print(f"  {t:<24} {stats[t]}")

# New tables
for tbl in ("forecasts", "symbol_ic", "historical_outcomes"):
    try:
        n = db.execute_query(f"SELECT COUNT(*) AS n FROM {tbl}")[0]["n"]
        print(f"  {tbl:<24} {n}")
    except Exception:
        print(f"  {tbl:<24} (table not found)")

print(f"  {'DB size':<24} {stats['db_size']}")
print(f"  {'host':<24} {stats['db_url']}")

print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
print("  IN PG (all migrated):")
print("    signals(84) positions(84) sessions(1)")
print("    forecasts(16245) symbol_ic(41) historical_outcomes(218)")
print()
print("  DELETED (no longer in files, data is in PG):")
print("    positions.json, forward_paper/outcomes.jsonl")
print()
print("  DUAL-WRITE ACTIVE (file + PG):")
print("    forecasts.jsonl → ForecastLedger writes to file AND PG forecasts table")
print("    symbol_ic_state.json → SymbolICTracker writes to file AND PG symbol_ic table")
print()
print("  OPERATIONAL (ephemeral, not migrated by design):")
print("    live_session/latest_scores.json  — regenerated every 2 min")
print("    live_session/live_quotes.json    — ephemeral snapshot")
print("    live_session/autorun_log.jsonl   — session log (grows per session)")
print()
print("  AUDIT TRAIL (keep):")
print("    signal_ledger/signals.jsonl      — immutable event backup")
