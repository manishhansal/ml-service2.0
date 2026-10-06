#!/usr/bin/env python3
"""
scripts/_close_open_signals.py
───────────────────────────────
Close all past OPEN signals in PostgreSQL using actual market returns.

Resolution logic:
  direction=1  (LONG):   WIN if changePct > 0, LOSS if changePct <= 0
  direction=-1 (SHORT):  WIN if changePct < 0, LOSS if changePct >= 0
  direction=0  (neutral): EXPIRED (no directional bet — not win/loss)

Outcome mapping → ForecastLedger status:
  WIN  → 'won'
  LOSS → 'lost'
  neutral → 'expired'

Run:
    PYTHONPATH=. python3 scripts/_close_open_signals.py
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from src.data.signal_db import get_db

db = get_db()
now_utc = datetime.now(tz=timezone.utc).isoformat()


def get_actual_returns() -> dict[str, float]:
    """Get today's actual changePct per symbol from live_quotes + parquet fallback."""
    import pandas as pd

    returns: dict[str, float] = {}

    # Primary: live_quotes.json (changePct at market close)
    lq_path = BASE / "artifacts/live_session/live_quotes.json"
    if lq_path.exists():
        lq = json.load(open(lq_path))
        for sym, q in lq.get("quotes", {}).items():
            if isinstance(q, dict) and q.get("changePct") is not None:
                returns[sym] = float(q["changePct"])

    # Fallback: parquet last close vs previous close
    pdir = BASE / "data/1d/1d"
    if pdir.exists():
        for pf in pdir.glob("*.parquet"):
            sym = pf.stem
            if sym in returns:
                continue
            try:
                df = pd.read_parquet(str(pf), columns=["close"])
                if len(df) >= 2:
                    ret = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100
                    returns[sym] = float(ret)
            except Exception:
                pass

    return returns


def close_open_forecasts(session_date: str, actual_returns: dict[str, float]) -> dict:
    """Close all open forecasts for session_date using actual returns."""
    open_forecasts = db.execute_query(
        "SELECT id, symbol, direction, score FROM forecasts WHERE status=%s AND session_date=%s",
        ("open", session_date)
    )

    if not open_forecasts:
        return {"session_date": session_date, "closed": 0, "won": 0, "lost": 0, "expired": 0}

    won = lost = expired = 0
    updates: list[tuple] = []

    for fc in open_forecasts:
        sym       = fc["symbol"]
        direction = fc["direction"]
        ret       = actual_returns.get(sym)

        if direction == 0 or ret is None:
            new_status = "expired"
            expired += 1
        elif direction == 1:
            new_status = "won" if ret > 0 else "lost"
            if new_status == "won":
                won += 1
            else:
                lost += 1
        else:  # direction == -1
            new_status = "won" if ret < 0 else "lost"
            if new_status == "won":
                won += 1
            else:
                lost += 1

        updates.append((new_status, now_utc, fc["id"]))

    # Batch update
    with db._connect() as conn:
        import psycopg2.extras
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, """
                UPDATE forecasts SET status=%s, superseded_at=%s WHERE id=%s
            """, updates)

    return {
        "session_date": session_date,
        "closed":       len(updates),
        "won":          won,
        "lost":         lost,
        "expired":      expired,
    }


def close_open_positions(actual_returns: dict[str, float]) -> dict:
    """Close any OPEN positions (should be 0, but safety net)."""
    open_pos = db.execute_query(
        "SELECT signal_id, session_date, symbol, direction, entry_price FROM positions WHERE status='OPEN'"
    )
    if not open_pos:
        return {"closed": 0}

    closed = won = lost = 0
    for pos in open_pos:
        sym    = pos["symbol"]
        ret    = actual_returns.get(sym)
        entry  = pos.get("entry_price")

        if ret is None or entry is None or entry <= 0:
            outcome = "EXPIRED"
            final_ret = None
        else:
            cost_frac = 7.26 / 10_000
            gross = pos["direction"] * (ret / 100)  # changePct to fraction
            net   = gross - cost_frac
            final_ret = round(net * 100, 4)
            outcome   = "SETTLED_WIN" if net > 0 else "SETTLED_LOSS"
            if outcome == "SETTLED_WIN":
                won += 1
            else:
                lost += 1
        closed += 1
        db.settle_position(pos["signal_id"], {
            "exit_time":       now_utc,
            "exit_price":      entry,  # best estimate: last known
            "status":          outcome,
            "settled_at":      now_utc,
            "final_return_pct": final_ret,
            "outcome":         outcome,
        })
    return {"closed": closed, "won": won, "lost": lost}


def main() -> None:
    print("=" * 55)
    print("  CLOSE ALL OPEN SIGNALS")
    print("=" * 55)

    print("\nLoading actual market returns...")
    actual_returns = get_actual_returns()
    print(f"  Returns loaded: {len(actual_returns)} symbols")

    # Find all sessions with open forecasts
    sessions = db.execute_query(
        "SELECT DISTINCT session_date FROM forecasts WHERE status='open' ORDER BY session_date"
    )
    print(f"\nSessions with open forecasts: {len(sessions)}")

    total_closed = total_won = total_lost = total_expired = 0
    for sess in sessions:
        sdate = sess["session_date"]
        result = close_open_forecasts(sdate, actual_returns)
        total_closed  += result["closed"]
        total_won     += result["won"]
        total_lost    += result["lost"]
        total_expired += result["expired"]
        print(f"\n  Session {sdate}:")
        print(f"    Closed:  {result['closed']}")
        print(f"    Won:     {result['won']}")
        print(f"    Lost:    {result['lost']}")
        print(f"    Expired: {result['expired']}")
        wr = result['won'] / (result['won'] + result['lost']) if (result['won'] + result['lost']) > 0 else None
        if wr is not None:
            print(f"    Win rate (dir signals): {wr:.0%}")

    # Close any open positions (safety net)
    print("\nChecking open positions (safety net)...")
    pos_result = close_open_positions(actual_returns)
    if pos_result["closed"]:
        print(f"  Closed {pos_result['closed']} open positions")
    else:
        print("  No open positions found (already settled)")

    # Final DB state
    print(f"\n{'=' * 55}")
    print("  FINAL STATE")
    print(f"{'=' * 55}")
    remaining = db.execute_query("SELECT COUNT(*) n FROM forecasts WHERE status='open'")
    open_pos  = db.execute_query("SELECT COUNT(*) n FROM positions WHERE status='OPEN'")
    print(f"  Open forecasts remaining: {remaining[0]['n']}")
    print(f"  Open positions remaining: {open_pos[0]['n']}")
    print(f"\n  Total forecasts closed: {total_closed}")
    print(f"    Won:     {total_won}")
    print(f"    Lost:    {total_lost}")
    print(f"    Expired: {total_expired}")
    wr = total_won / (total_won + total_lost) if (total_won + total_lost) > 0 else None
    if wr is not None:
        print(f"    Win rate (directional): {wr:.0%}")
    print(f"\n  All signals closed ✓")


if __name__ == "__main__":
    main()
