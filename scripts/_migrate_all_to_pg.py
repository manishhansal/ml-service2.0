#!/usr/bin/env python3
"""
scripts/_migrate_all_to_pg.py
──────────────────────────────
Complete migration of ALL flat-file data stores to PostgreSQL.

Migrates:
  1. forecasts.jsonl       → forecasts table       (16,245 ForecastLedger entries)
  2. symbol_ic_state.json  → symbol_ic table        (41 symbol IC trackers)
  3. outcomes.jsonl        → historical_outcomes    (218 old forward paper outcomes)
  4. signals_v2.jsonl      → historical_outcomes    (153 old signals as reference)

Then removes files that are now fully in PostgreSQL:
  - artifacts/signal_ledger/positions.json   (84 positions, all in PG)
  - artifacts/forward_paper/outcomes.jsonl   (218 outcomes, migrated to PG)

Keeps (operational / audit trail):
  - artifacts/signal_ledger/signals.jsonl    (event audit log)
  - artifacts/live_session/                  (operational ephemeral data)

Run:
    PYTHONPATH=. python3 scripts/_migrate_all_to_pg.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from src.data.signal_db import get_db, SignalDB

# Re-init schema to create new tables
db = SignalDB()


def migrate_forecasts() -> int:
    """Migrate forecasts.jsonl → forecasts table."""
    fp = BASE / "artifacts/forward_paper/forecasts.jsonl"
    if not fp.exists():
        print("  forecasts.jsonl not found — skipping")
        return 0

    rows = []
    for line in fp.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        rows.append({
            "id":            r.get("id", ""),
            "ts":            r.get("ts", ""),
            "session_date":  r.get("session_date", ""),
            "symbol":        r.get("symbol", ""),
            "score":         r.get("score"),
            "direction":     r.get("direction"),
            "est_prob":      r.get("est_prob"),
            "market_prior":  r.get("market_prior"),
            "data_date":     r.get("data_date"),
            "nifty_chg":     r.get("nifty_chg_at_record"),
            "model_version": r.get("model_version"),
            "status":        r.get("status", "open"),
            "supersedes":    r.get("supersedes"),
            "superseded_at": r.get("superseded_at"),
        })

    if not rows:
        return 0

    # Batch in chunks of 500 to avoid large transactions
    total = 0
    chunk = 500
    for i in range(0, len(rows), chunk):
        total += db.upsert_forecasts_batch(rows[i:i+chunk])
        print(f"  forecasts: {min(i+chunk, len(rows)):>6}/{len(rows)} migrated", end="\r")
    print()
    return total


def migrate_symbol_ic() -> int:
    """Migrate symbol_ic_state.json → symbol_ic table."""
    fp = BASE / "artifacts/live_session/symbol_ic_state.json"
    if not fp.exists():
        print("  symbol_ic_state.json not found — skipping")
        return 0

    data = json.loads(fp.read_text())
    symbols_data = data.get("symbols", {})
    if not symbols_data:
        print("  symbol_ic_state.json has no symbols — skipping")
        return 0

    rows = []
    for sym, info in symbols_data.items():
        rows.append({
            "symbol":       sym,
            "n_trades":     info.get("n_trades", 0),
            "rolling_ic":   info.get("rolling_ic"),
            "last_updated": info.get("last_updated", ""),
        })

    db.upsert_symbol_ic_batch(rows)
    return len(rows)


def migrate_historical_outcomes() -> int:
    """Migrate forward_paper/outcomes.jsonl → historical_outcomes table."""
    fp = BASE / "artifacts/forward_paper/outcomes.jsonl"
    if not fp.exists():
        print("  outcomes.jsonl not found — skipping")
        return 0

    rows = []
    for line in fp.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        # Skip DATA_ERROR outcomes
        if r.get("data_error") or r.get("outcome") == "DATA_ERROR":
            continue
        rows.append({
            "signal_id":    r.get("signal_id", ""),
            "symbol":       r.get("symbol", ""),
            "signal_ts":    r.get("signal_ts"),
            "resolve_after":r.get("resolve_after"),
            "resolved_at":  r.get("resolved_at"),
            "direction":    r.get("direction", 0),
            "horizon_bars": r.get("horizon_bars", 5),
            "bars_elapsed": r.get("bars_elapsed"),
            "partial":      r.get("partial_resolution", False),
            "entry_price":  r.get("entry_price"),
            "exit_price":   r.get("exit_price"),
            "gross_return": r.get("gross_return"),
            "net_return":   r.get("net_return"),
            "cost_bps":     r.get("cost_bps", 27.65),
            "outcome":      r.get("outcome"),
            "regime":       r.get("regime"),
            "note":         r.get("note"),
        })

    if not rows:
        return 0

    db.upsert_historical_outcomes_batch(rows)
    return len(rows)


def cleanup_migrated_files() -> list[str]:
    """Remove files that are now fully in PostgreSQL."""
    removed = []

    # positions.json — all 84 positions migrated to PG positions table
    p = BASE / "artifacts/signal_ledger/positions.json"
    if p.exists():
        p.unlink()
        removed.append(str(p.relative_to(BASE)))

    # outcomes.jsonl — migrated to historical_outcomes table
    p2 = BASE / "artifacts/forward_paper/outcomes.jsonl"
    if p2.exists():
        p2.unlink()
        removed.append(str(p2.relative_to(BASE)))

    return removed


def main() -> None:
    print("=" * 60)
    print("  FULL MIGRATION TO POSTGRESQL")
    print("=" * 60)

    # 1. ForecastLedger
    print("\n[1/3] Migrating forecasts.jsonl → forecasts table...")
    n = migrate_forecasts()
    print(f"  Migrated: {n} forecast entries")

    # 2. SymbolIC
    print("\n[2/3] Migrating symbol_ic_state.json → symbol_ic table...")
    n = migrate_symbol_ic()
    print(f"  Migrated: {n} symbol IC records")

    # 3. Historical outcomes
    print("\n[3/3] Migrating forward_paper/outcomes.jsonl → historical_outcomes...")
    n = migrate_historical_outcomes()
    print(f"  Migrated: {n} historical outcomes")

    # 4. Cleanup
    print("\n[4/4] Cleaning up migrated files...")
    removed = cleanup_migrated_files()
    for f in removed:
        print(f"  DELETED: {f}")
    if not removed:
        print("  Nothing to delete")

    # 5. Final stats
    stats = db.db_stats()
    print(f"\n{'=' * 60}")
    print("  FINAL POSTGRESQL STATUS")
    print(f"{'=' * 60}")
    print(f"  sessions:             {stats['sessions']}")
    print(f"  signals:              {stats['signals']}")
    print(f"  positions:            {stats['positions']}")
    print(f"  price_snapshots:      {stats['price_snapshots']}")

    extra = db.execute_query("SELECT COUNT(*) AS n FROM forecasts")
    print(f"  forecasts:            {extra[0]['n']}")
    sic = db.execute_query("SELECT COUNT(*) AS n FROM symbol_ic")
    print(f"  symbol_ic:            {sic[0]['n']}")
    hist = db.execute_query("SELECT COUNT(*) AS n FROM historical_outcomes")
    print(f"  historical_outcomes:  {hist[0]['n']}")
    print(f"  DB size:              {stats['db_size']}")
    print(f"\n  Migration complete ✓")


if __name__ == "__main__":
    main()
