#!/usr/bin/env python3
"""Remove duplicate positions — keep only the best-settled per (symbol, session_date)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from src.data.signal_db import get_db

db = get_db()
STATUS_PRIORITY = {'SETTLED_WIN': 0, 'SETTLED_LOSS': 1, 'EXPIRED': 2, 'OPEN': 3, 'DATA_ERROR': 4}

for session_date in ('2026-10-06', '2026-10-07'):
    rows = db.execute_query("""
        SELECT p.signal_id, p.symbol, p.status, p.final_return_pct
        FROM positions p JOIN signals s USING(signal_id)
        WHERE p.session_date=%s
        ORDER BY p.symbol, p.status
    """, (session_date,))

    # Keep best status per symbol
    best: dict = {}
    for r in rows:
        sym = r['symbol']
        rp  = STATUS_PRIORITY.get(r['status'], 9)
        if sym not in best or rp < STATUS_PRIORITY.get(best[sym]['status'], 9):
            best[sym] = r

    keep_ids   = {r['signal_id'] for r in best.values()}
    delete_ids = [r['signal_id'] for r in rows if r['signal_id'] not in keep_ids]

    if delete_ids:
        with db._connect() as conn:
            with conn.cursor() as cur:
                for sid in delete_ids:
                    cur.execute("DELETE FROM positions WHERE signal_id=%s", (sid,))
                    cur.execute("DELETE FROM signals WHERE signal_id=%s", (sid,))
        print(f"{session_date}: removed {len(delete_ids)} duplicates, kept {len(keep_ids)}")
    else:
        print(f"{session_date}: no duplicates found")

# Final state
rows = db.execute_query(
    "SELECT session_date, status, COUNT(*) n FROM positions GROUP BY session_date, status ORDER BY session_date, status"
)
print("\nPositions after dedup:")
total = 0
for r in rows:
    print(f"  {r['session_date']}  {r['status']:<16}  n={r['n']}")
    total += r['n']
print(f"Total: {total}")
