"""End-to-end PostgreSQL signal DB verification — PYTHONPATH=. python3 scripts/_verify_db.py"""
import sys, uuid
from pathlib import Path
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from src.data.signal_db import SignalDB, DATABASE_URL

db = SignalDB()

print("PostgreSQL Signal DB — End-to-End Verification")
print(f"DSN: {DATABASE_URL.split('@')[-1]}")
print("-" * 50)

# T1: structure
stats = db.db_stats()
assert stats['sessions']  >= 1,  f"Expected >=1 sessions, got {stats['sessions']}"
assert stats['signals']   >= 84, f"Expected >=84 signals, got {stats['signals']}"
assert stats['positions'] >= 84, f"Expected >=84 positions, got {stats['positions']}"
print(f"T1  Structure: {stats['sessions']} sessions / {stats['signals']} signals / {stats['positions']} positions  OK")
print(f"    DB size: {stats['db_size']} at {stats['db_url']}")

# T2: session summary
s = db.get_session_summary('2026-10-06')
assert s is not None and s['total'] == 84
print(f"T2  Session 2026-10-06: total={s['total']} wins={s['wins']} losses={s['losses']}  OK")

# T3: historical performance
hp = db.get_historical_performance(30)
assert len(hp) >= 1
print(f"T3  Historical query: {len(hp)} session(s)  OK")

# T4: conviction stats
cv = db.get_conviction_stats()
print(f"T4  Conviction stats: {len(cv)} grades  OK")

# T5: symbol performance
sp = db.get_symbol_performance('HDFCBANK')
print(f"T5  Symbol performance (HDFCBANK): {len(sp)} trade(s)  OK")

# T6: upsert session + insert signal + position + settle (full lifecycle)
TEST_DATE = '2099-01-07'
TEST_SID  = str(uuid.uuid4())[:16]
db.upsert_session(TEST_DATE, n_long=1, n_short=0, nifty_ltp=22000.0, nifty_chg_pct=0.5)
db.insert_signal({'signal_id': TEST_SID, 'session_date': TEST_DATE, 'symbol': 'TESTXYZ',
    'generated_at': f'{TEST_DATE}T09:00:00Z', 'data_date': f'{TEST_DATE}',
    'direction': 1, 'score': 0.72, 'conviction': 'A', 'rank': 1,
    'cost_bps': 7.26, 'nifty_at_signal': 22000.0, 'model_version': None})
db.upsert_position({'signal_id': TEST_SID, 'session_date': TEST_DATE, 'symbol': 'TESTXYZ',
    'direction': 1, 'entry_time': f'{TEST_DATE}T09:30:00Z', 'entry_price': 100.0,
    'exit_time': None, 'exit_price': None, 'last_price': 100.0, 'last_price_time': None,
    'unrealized_pct': 0.0, 'status': 'OPEN', 'settled_at': None,
    'final_return_pct': None, 'outcome': None})
open_pos = db.get_open_positions(TEST_DATE)
assert len(open_pos) == 1 and open_pos[0]['symbol'] == 'TESTXYZ'
print(f"T6a Full lifecycle — open position captured  OK")

# MTM update
upd = db.update_prices_batch(TEST_DATE, [(105.0, 4.927, f'{TEST_DATE}T12:00:00Z', 'TESTXYZ', TEST_DATE)])
assert upd >= 1
open_refreshed = db.get_open_positions(TEST_DATE)
assert abs(open_refreshed[0]['last_price'] - 105.0) < 0.01
print(f"T6b MTM update: last_price={open_refreshed[0]['last_price']}  OK")

# Settle
db.settle_position(TEST_SID, {'exit_time': f'{TEST_DATE}T15:15:00Z', 'exit_price': 107.0,
    'status': 'SETTLED_WIN', 'settled_at': f'{TEST_DATE}T15:15:00Z',
    'final_return_pct': 6.927, 'outcome': 'SETTLED_WIN'})
ts2 = db.get_session_summary(TEST_DATE)
assert ts2['wins'] == 1 and ts2['open'] == 0
print(f"T6c Settlement: wins={ts2['wins']}, mean_ret={ts2['mean_return']}%  OK")

# Price snapshot
db.insert_snapshot_batch(TEST_DATE, [{'session_date': TEST_DATE, 'symbol': 'TESTXYZ',
    'snapshot_time': f'{TEST_DATE}T10:00:00Z', 'price': 102.0, 'unrealized_pct': 1.93}])
snap_count = db.execute_query("SELECT COUNT(*) AS n FROM price_snapshots WHERE session_date=%s", (TEST_DATE,))
assert snap_count[0]['n'] == 1
print(f"T6d Price snapshot: {snap_count[0]['n']} row  OK")

# Cleanup test data
with db._connect() as conn:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM price_snapshots WHERE session_date=%s", (TEST_DATE,))
        cur.execute("DELETE FROM positions WHERE session_date=%s",       (TEST_DATE,))
        cur.execute("DELETE FROM signals WHERE session_date=%s",         (TEST_DATE,))
        cur.execute("DELETE FROM sessions WHERE session_date=%s",        (TEST_DATE,))
print("T6e Cleanup test data  OK")

# T7: raw SQL passthrough
rows = db.execute_query("""
    SELECT p.symbol, p.direction, s.score, p.status
    FROM positions p JOIN signals s USING (signal_id)
    WHERE p.session_date=%s
    ORDER BY s.score DESC LIMIT 5
""", ('2026-10-06',))
assert len(rows) == 5
print(f"T7  Raw SQL: top 5 by score = {[r['symbol'] for r in rows]}  OK")

print()
print("All 7 tests PASSED  ✓  PostgreSQL end-to-end verified")
print()
print("=== DEMO: Signal summary from PostgreSQL ===")
s_final = db.get_session_summary('2026-10-06')
print(f"Session 2026-10-06:")
print(f"  Total:  {s_final['total']}  |  Open: {s_final['open']}")
print(f"  Wins:   {s_final['wins']}   |  Losses: {s_final['losses']}")
print(f"  Win%:   {s_final['win_pct'] or '—'}%")
print(f"  Mean:   {s_final['mean_return'] or '—'}%")
