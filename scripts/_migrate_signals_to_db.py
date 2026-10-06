#!/usr/bin/env python3
"""
scripts/_migrate_signals_to_db.py
───────────────────────────────────
Migration: artifacts/signal_ledger/positions.json → PostgreSQL (ml_signals.db).

Run once after setting up ml-service-postgres:
    PYTHONPATH=. python3 scripts/_migrate_signals_to_db.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

POSITIONS_FILE = BASE / "artifacts/signal_ledger/positions.json"

from src.data.signal_db import SignalDB, DATABASE_URL

def main() -> None:
    print(f"Migrating to PostgreSQL: {DATABASE_URL.split('@')[-1]}")
    db = SignalDB()

    migrated_sessions = set()
    migrated_signals  = 0
    migrated_positions = 0

    # ── 1. Load from positions.json (most current state) ─────────────────────
    if POSITIONS_FILE.exists():
        positions = json.loads(POSITIONS_FILE.read_text())
        print(f"Found {len(positions)} positions in positions.json")

        for pos_key, pos in positions.items():
            session_date = pos.get("session_date", "")
            if not session_date:
                continue

            # Create session row if needed
            if session_date not in migrated_sessions:
                db.upsert_session(
                    session_date=session_date,
                    n_long=0, n_short=0,
                    nifty_ltp=pos.get("nifty_at_signal"),
                )
                migrated_sessions.add(session_date)

            # Signal row
            sig_row = {
                "signal_id":      pos.get("signal_id", pos_key[:16]),
                "session_date":   session_date,
                "symbol":         pos.get("symbol", ""),
                "generated_at":   pos.get("generated_at", ""),
                "data_date":      pos.get("data_date", ""),
                "direction":      pos.get("direction", 0),
                "score":          pos.get("score", 0.5),
                "conviction":     pos.get("conviction", ""),
                "rank":           pos.get("rank"),
                "cost_bps":       pos.get("cost_bps", 7.26),
                "nifty_at_signal": pos.get("nifty_at_signal"),
                "model_version":  None,
            }
            if db.insert_signal(sig_row):
                migrated_signals += 1

            # Position row
            pos_row = {
                "signal_id":       sig_row["signal_id"],
                "session_date":    session_date,
                "symbol":          pos.get("symbol", ""),
                "direction":       pos.get("direction", 0),
                "entry_time":      pos.get("entry_time", pos.get("generated_at", "")),
                "entry_price":     pos.get("entry_price"),
                "exit_time":       pos.get("exit_time"),
                "exit_price":      pos.get("exit_price"),
                "last_price":      pos.get("last_price"),
                "last_price_time": pos.get("last_price_time"),
                "unrealized_pct":  pos.get("unrealized_pct"),
                "status":          pos.get("status", "OPEN"),
                "settled_at":      pos.get("settled_at"),
                "final_return_pct": pos.get("final_return_pct"),
                "outcome":         pos.get("outcome"),
            }
            db.upsert_position(pos_row)
            migrated_positions += 1

        print(f"  Sessions created:  {len(migrated_sessions)}")
        print(f"  Signals migrated:  {migrated_signals}")
        print(f"  Positions migrated: {migrated_positions}")
    else:
        print("positions.json not found — skipping")

    # ── 2. Fix session stats (n_long, n_short, n_scored) ─────────────────────
    for session_date in migrated_sessions:
        with db._connect() as conn:
            with conn.cursor(cursor_factory=__import__('psycopg2.extras', fromlist=['RealDictCursor']).RealDictCursor) as cur:
                cur.execute("""
                    SELECT COUNT(*) AS n_scored,
                           SUM((direction=1)::int)  AS n_long,
                           SUM((direction=-1)::int) AS n_short
                    FROM signals WHERE session_date=%s
                """, (session_date,))
                stats = cur.fetchone()
                cur.execute("""
                    UPDATE sessions SET n_scored=%s, n_long=%s, n_short=%s
                    WHERE session_date=%s
                """, (stats["n_scored"], stats["n_long"], stats["n_short"], session_date))

    # ── 3. Print final stats ──────────────────────────────────────────────────
    stats = db.db_stats()
    print(f"\nPostgreSQL DB stats:")
    print(f"  sessions:        {stats['sessions']}")
    print(f"  signals:         {stats['signals']}")
    print(f"  positions:       {stats['positions']}")
    print(f"  price_snapshots: {stats['price_snapshots']}")
    print(f"  db size:         {stats['db_size']}")
    print(f"  host:            {stats['db_url']}")
    print(f"\nMigration complete ✓")

if __name__ == "__main__":
    main()
