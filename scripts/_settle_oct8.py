#!/usr/bin/env python3
"""
One-shot script: settle Oct 8 positions using final closing LTPs from autorun log.
Expires the remaining positions that had no live price coverage.
"""
from __future__ import annotations
import json
import psycopg2
from pathlib import Path

BASE = Path(__file__).parent.parent
LOG = BASE / "artifacts/live_session/autorun_log.jsonl"

# ── Find last sample with position P&L ──────────────────────────────────────
lines = LOG.read_text().strip().split("\n")
last = None
for line in reversed(lines):
    try:
        d = json.loads(line)
        if d.get("pnl") and d["pnl"].get("positions"):
            last = d
            break
    except Exception:
        pass

if not last:
    print("ERROR: no pnl data found in autorun_log.jsonl")
    raise SystemExit(1)

pnl = last["pnl"]
positions = pnl["positions"]
nifty_close = last.get("nifty_ltp", 22216.0)
nifty_chg = last.get("nifty_chg", -1.71)
sample_n = last.get("sample_n")
ts = last.get("timestamp", "")

print(f"Using sample {sample_n} at {ts[:16]}")
print(f"NIFTY close: {nifty_close} ({nifty_chg:+.2f}%)")
print(f"Positions to settle: {len(positions)}")

COST_BPS = 7.26
SESSION = "2026-10-08"

conn = psycopg2.connect("postgresql://mlservice:mlservice_dev@localhost:5445/mlservice")
conn.autocommit = False
cur = conn.cursor()

settled = 0
wins = 0
losses = 0
total_gross = 0.0
total_net = 0.0

for pos in positions:
    sym = pos["symbol"]
    exit_price = float(pos["ltp"])
    gross_pct = float(pos["gross_pct"])
    net_pct = float(pos["net_pct"])
    status = "SETTLED_WIN" if net_pct > 0 else "SETTLED_LOSS"
    if net_pct > 0:
        wins += 1
    else:
        losses += 1
    total_gross += gross_pct
    total_net += net_pct

    cur.execute(
        """
        UPDATE positions
        SET exit_price       = %s,
            gross_return_pct = %s,
            final_return_pct = %s,
            cost_bps         = %s,
            status           = %s,
            model_version    = 'v2c',
            settled_at       = NOW()
        WHERE symbol = %s
          AND session_date = %s
          AND status = 'OPEN'
        """,
        (exit_price, gross_pct, net_pct, COST_BPS, status, sym, SESSION),
    )
    if cur.rowcount > 0:
        settled += 1

# Expire remaining OPEN positions (no live price available)
cur.execute(
    """
    UPDATE positions
    SET status     = 'EXPIRED',
        model_version = 'v2c',
        settled_at = NOW()
    WHERE session_date = %s AND status = 'OPEN'
    """,
    (SESSION,),
)
expired = cur.rowcount

n = len(positions)
win_rate = wins / n * 100 if n > 0 else 0.0
mean_net = total_net / n if n > 0 else 0.0
mean_gross = total_gross / n if n > 0 else 0.0

# Update sessions record with full Oct 8 metrics
nifty_open = 22605.4  # prior day close ≈ today's open proxy
cur.execute(
    """
    UPDATE sessions
    SET nifty_open            = %s,
        nifty_close           = %s,
        nifty_chg_pct         = %s,
        regime                = 'BEAR',
        win_rate              = %s,
        mean_return_pct       = %s,
        model_version         = 'v2c',
        bull_suppressor_active = FALSE
    WHERE session_date = %s
    """,
    (nifty_open, nifty_close, nifty_chg, round(win_rate, 4), round(mean_net, 6), SESSION),
)

conn.commit()
conn.close()

print()
print("Settlement complete:")
print(f"  Settled:   {settled} positions  (wins={wins}  losses={losses})")
print(f"  Expired:   {expired} positions  (no live price coverage)")
print(f"  Win rate:  {win_rate:.1f}%")
print(f"  Mean net:  {mean_net:+.4f}%")
print(f"  Mean gross:{mean_gross:+.4f}%")
print(f"  Session:   BEAR  NIFTY {nifty_chg:+.2f}%  model=v2c")
