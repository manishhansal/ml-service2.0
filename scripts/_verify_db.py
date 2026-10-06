"""Verify SQLite signal database after migration."""
import sys
from pathlib import Path
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from src.data.signal_db import SignalDB

db = SignalDB()

print("SQLite Signal DB — Verification")
print("-" * 40)

# T1: structure
stats = db.db_stats()
assert stats['sessions'] == 1, f"Expected 1 session, got {stats['sessions']}"
assert stats['signals']  == 84
assert stats['positions'] == 84
print(f"T1 Structure: {stats['sessions']} sessions / {stats['signals']} signals / {stats['positions']} positions  OK")
print(f"   DB size: {stats['db_size_kb']} KB at {stats['db_path']}")

# T2: check session has all 84 positions (settled from earlier)
with db._connect() as conn:
    all_pos = conn.execute("SELECT COUNT(*) FROM positions WHERE session_date='2026-10-06'").fetchone()[0]
assert all_pos == 84, f"Expected 84, got {all_pos}"
p0_row = db.get_session_summary('2026-10-06')
print(f"T2 All positions: {all_pos}  OK")
print(f"   Status: wins={p0_row.get('wins')} losses={p0_row.get('losses')} open={p0_row.get('open')}")

# T3: session summary
s = db.get_session_summary('2026-10-06')
assert s['total'] == 84
print(f"T3 Session summary: total={s['total']} wins={s['wins']} losses={s['losses']}  OK")

# T4: settle + verify using a fresh test signal
import uuid
test_id = str(uuid.uuid4())[:16]
db.upsert_session('2099-01-01', n_long=1, n_short=0)
db.insert_signal({'signal_id': test_id, 'session_date': '2099-01-01', 'symbol': 'TEST',
    'generated_at': '2099-01-01T09:00:00Z', 'data_date': '2099-01-01', 'direction': 1,
    'score': 0.7, 'conviction': 'A', 'rank': 1, 'cost_bps': 7.26, 'nifty_at_signal': 22000.0, 'model_version': None})
db.upsert_position({'signal_id': test_id, 'session_date': '2099-01-01', 'symbol': 'TEST',
    'direction': 1, 'entry_time': '2099-01-01T09:30:00Z', 'entry_price': 100.0,
    'exit_time': None, 'exit_price': None, 'last_price': 100.0, 'last_price_time': None,
    'unrealized_pct': 0.0, 'status': 'OPEN', 'settled_at': None, 'final_return_pct': None, 'outcome': None})
test_open = db.get_open_positions('2099-01-01')
assert len(test_open) == 1
db.settle_position(test_id, {'exit_time': '2099-01-01T15:15:00Z', 'exit_price': 105.0,
    'status': 'SETTLED_WIN', 'settled_at': '2099-01-01T15:15:00Z', 'final_return_pct': 4.927, 'outcome': 'SETTLED_WIN'})
ts2 = db.get_session_summary('2099-01-01')
assert ts2['wins'] == 1 and ts2['open'] == 0
print(f"T4 Settle test: wins={ts2['wins']}, open={ts2['open']}  OK")
# cleanup
with db._connect() as conn:
    conn.execute("DELETE FROM positions WHERE session_date='2099-01-01'")
    conn.execute("DELETE FROM signals WHERE session_date='2099-01-01'")
    conn.execute("DELETE FROM sessions WHERE session_date='2099-01-01'")

# T5: historical performance query
hp = db.get_historical_performance(30)
print(f"T5 Historical query: {len(hp)} sessions  OK")

# T6: SQL direct query — top scored symbols
with db._connect() as conn:
    top = conn.execute("SELECT symbol, score, direction FROM signals ORDER BY score DESC LIMIT 5").fetchall()
print("T6 Top scored signals:")
for r in top:
    d = "LONG" if r['direction'] == 1 else "SHORT"
    print(f"   {r['symbol']:<14} {d}  score={r['score']:.4f}")
print("   Direct SQL query  OK")

print()
print("All 6 tests PASSED")
print()

# Demo useful queries
print("=== DEMO QUERIES ===")
print()
print("-- Win rate by session --")
with db._connect() as conn:
    rows = conn.execute("""
        SELECT session_date, n_long, n_short, n_scored FROM sessions
    """).fetchall()
for r in rows:
    print(f"  {r['session_date']}: scored={r['n_scored']} long={r['n_long']} short={r['n_short']}")
