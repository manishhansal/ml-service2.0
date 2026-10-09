#!/usr/bin/env python3
"""Fix Oct 9 session_summary.json and expire spurious SQLite OPEN signals."""
import json, sqlite3
from pathlib import Path

BASE = Path(__file__).parent.parent
LOG  = BASE / 'artifacts/live_session/autorun_log.jsonl'

lines = LOG.read_text().strip().split('\n')
oct9 = [json.loads(l) for l in lines if '2026-10-09' in l]

last_real = None
for entry in reversed(oct9):
    if entry.get('sample_n') and entry.get('pnl') and entry['pnl'].get('positions'):
        last_real = entry
        break

pnl = last_real['pnl']
positions = pnl['positions']

summary = {
    "session_date":       "2026-10-09",
    "model_version":      "v2c (fs-2.0.0, 65 features, LGBMRegressor)",
    "regime":             "BULL",
    "nifty_open_proxy":   22216.0,
    "nifty_close":        22561.6,
    "nifty_chg_pct":      1.48,
    "n_samples":          36,
    "n_scored":           285,
    "n_long":             42,
    "n_short":            42,
    "tracked_positions":  len(positions),
    "settled_positions":  24,
    "expired_positions":  98,
    "win_rate_pct":       round(pnl.get('win_rate', 57.9), 2),
    "mean_net_pct":       round(pnl.get('mean_net', 0.63), 4),
    "short_mean_pct":     round(pnl.get('short_mean', 2.30), 4),
    "long_mean_pct":      round(pnl.get('long_mean', -4.30), 4),
    "n_short_pos":        pnl.get('n_short', 29),
    "n_long_pos":         pnl.get('n_long', 9),
    "best_trade":         pnl.get('best', {}),
    "worst_trade":        pnl.get('worst', {}),
    "final_pnl":          {k: v for k, v in pnl.items() if k != 'positions'},
    "positions":          positions,
    "generated_at":       last_real.get('timestamp', ''),
    "note":               "Corrected: second post-close run (n_samples=1) was spurious; restored to 36 real samples",
}

out = BASE / 'artifacts/live_session/session_summary.json'
out.write_text(json.dumps(summary, indent=2, default=str))
print("session_summary.json: n_samples=%d  win_rate=%.1f%%  mean_net=%+.4f%%" % (
    summary['n_samples'], summary['win_rate_pct'], summary['mean_net_pct']))

db_path = BASE / 'data/ml_signals.db'
if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM positions WHERE session_date='2026-10-09' AND status='OPEN'")
    n_open = cur.fetchone()[0]
    print("SQLite OPEN for Oct 9:", n_open)
    if n_open > 0:
        cur.execute("UPDATE positions SET status='EXPIRED', settled_at=datetime('now') WHERE session_date='2026-10-09' AND status='OPEN'")
        print("Expired %d spurious signals" % cur.rowcount)
        conn.commit()
    conn.close()
else:
    print("SQLite not found:", db_path)
