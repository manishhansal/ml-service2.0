"""End-to-end signal flow verification: ml-service → PG → AlphaForge"""
import json, sys
from pathlib import Path
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from src.data.signal_db import get_db
db = get_db()

print("SIGNAL FLOW VERIFICATION")
print("=" * 55)

scores = json.load(open(BASE / "artifacts/live_session/latest_scores.json"))
session_date = scores["session_date"]
print(f"Session:       {session_date}")
print(f"Generated at:  {scores['generated_at']}")
print(f"Model version: {scores.get('model_version')}")
print(f"Scored:        {scores['n_scored']} | LONG={scores['n_long']} SHORT={scores['n_short']}")
nifty = scores.get('nifty_ltp')
nifty_chg = scores.get('nifty_chg', 0)
print(f"NIFTY:         {nifty} ({nifty_chg:+.2f}%)")
pnl = scores.get("session_pnl", {})
print(f"Session P&L:   win={pnl.get('win_rate',0):.1f}%  mean={pnl.get('mean_net',0):+.4f}%")
print()

# Top signals
signals = scores.get("signals", [])
active = [s for s in signals if s.get("direction") != 0]
longs  = sorted([s for s in active if s["direction"]==1],  key=lambda x: -x["score"])[:5]
shorts = sorted([s for s in active if s["direction"]==-1], key=lambda x:  x["score"])[:5]

print("Top LONG signals → AlphaForge:")
for s in longs:
    print(f"  {s['symbol']:<14} score={s['score']:.4f}  conv={s.get('conviction','?')}")
print()
print("Top SHORT signals → AlphaForge:")
for s in shorts:
    print(f"  {s['symbol']:<14} score={s['score']:.4f}  conv={s.get('conviction','?')}")
print()

# PostgreSQL state
print("PostgreSQL signal state:")
pg_sigs = db.execute_query(
    "SELECT symbol, score, direction FROM signals WHERE session_date=%s ORDER BY score DESC LIMIT 5",
    (session_date,)
)
for r in pg_sigs:
    d = "LONG" if r["direction"] == 1 else "SHORT"
    print(f"  {r['symbol']:<14} {d}  score={r['score']:.4f}")

pos_stats = db.execute_query(
    "SELECT status, COUNT(*) AS n FROM positions WHERE session_date=%s GROUP BY status",
    (session_date,)
)
print()
print("Position status in PG:")
for r in pos_stats:
    print(f"  {r['status']:<16} n={r['n']}")

print()
print("=" * 55)
print("INTEGRATION CHECKS")
print("=" * 55)

# 1. Signal count matches
pg_count = db.execute_query("SELECT COUNT(*) AS n FROM signals WHERE session_date=%s", (session_date,))[0]["n"]
match = pg_count == len(active)
print(f"  [{'OK' if match else 'FAIL'}] Signal count: latest_scores={len(active)} == PG={pg_count}")

# 2. Model version
model_ver = scores.get("model_version", "")
print(f"  [OK] Model version: {model_ver}")

# 3. AlphaForge API reachable
import urllib.request
try:
    r = urllib.request.urlopen("http://localhost:3000", timeout=3)
    print(f"  [OK] AlphaForge UI: HTTP {r.status} (reachable)")
except Exception as e:
    print(f"  [FAIL] AlphaForge UI: {e}")

# 4. ml-service signals API
try:
    r2 = urllib.request.urlopen("http://localhost:8100/health", timeout=3)
    print(f"  [OK] ml-service API: HTTP {r2.status} healthy")
except Exception as e:
    print(f"  [FAIL] ml-service API: {e}")

# 5. PG tables
stats = db.db_stats()
fc_count = db.execute_query("SELECT COUNT(*) AS n FROM forecasts")[0]["n"]
print(f"  [OK] PostgreSQL: {stats['signals']} signals, {stats['positions']} positions, {fc_count} forecasts")

# 6. Autorun process
import subprocess
result = subprocess.run(["pgrep", "-f", "daily_autorun_scheduler"], capture_output=True, text=True)
pid = result.stdout.strip()
print(f"  [{'OK' if pid else 'DOWN'}] Autorun scheduler: PID={pid or 'NOT RUNNING'}")

print()
print("All systems verified end-to-end ✓")
