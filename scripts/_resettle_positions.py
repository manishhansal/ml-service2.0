#!/usr/bin/env python3
"""
scripts/_resettle_positions.py
────────────────────────────────
Re-settle Oct 6-7 positions with ACTUAL market returns.

Sources (in priority order):
  1. autorun_log.jsonl close entries  — FP-tracked positions: actual entry, ltp, net_pct
  2. live_quotes.json changePct        — Oct 7 non-FP signals: derive return from changePct
  3. EXPIRED                           — Oct 6 non-FP signals: no reliable Oct 6 price data

Run:
    PYTHONPATH=. python3 scripts/_resettle_positions.py
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from src.data.signal_db import get_db
db = get_db()
NOW = datetime.now(tz=timezone.utc).isoformat()
COST_FRAC = 7.26 / 10_000.0


# ── Load data sources ─────────────────────────────────────────────────────────

def load_fp_pnl_from_log(session_date: str) -> dict[str, dict]:
    """Load FP position P&L from autorun_log.jsonl close entry for a given date."""
    fp: dict[str, dict] = {}
    log_path = BASE / "artifacts/live_session/autorun_log.jsonl"
    if not log_path.exists():
        return fp
    for line in log_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if session_date not in e.get("timestamp", "") or not e.get("is_close"):
            continue
        for pos in e.get("pnl", {}).get("positions", []):
            sym = pos.get("symbol", "")
            if sym:
                fp[sym] = pos
    return fp


def load_session_summary_pnl() -> dict[str, dict]:
    """Load final P&L from session_summary.json (today's session)."""
    fp: dict[str, dict] = {}
    ss_path = BASE / "artifacts/live_session/session_summary.json"
    if not ss_path.exists():
        return fp
    ss = json.load(open(ss_path))
    for pos in ss.get("final_pnl", {}).get("positions", []):
        sym = pos.get("symbol", "")
        if sym:
            fp[sym] = pos
    return fp


def load_live_quotes_changepct() -> dict[str, float]:
    """Load today's changePct from live_quotes.json."""
    lq_path = BASE / "artifacts/live_session/live_quotes.json"
    if not lq_path.exists():
        return {}
    lq = json.load(open(lq_path))
    return {
        sym: float(q["changePct"])
        for sym, q in lq.get("quotes", {}).items()
        if isinstance(q, dict) and q.get("changePct") is not None
    }


# ── Re-settlement logic ───────────────────────────────────────────────────────

def resettle_session(session_date: str, fp_pnl: dict[str, dict],
                     changepct_map: dict[str, float], allow_changepct: bool) -> dict:
    """Re-settle all positions for session_date with actual returns."""
    positions = db.execute_query(
        "SELECT signal_id, symbol, direction, entry_price FROM positions WHERE session_date=%s",
        (session_date,)
    )

    settled = wins = losses = expired = corrected = 0

    for pos in positions:
        sym       = pos["symbol"]
        direction = pos["direction"]
        sig_id    = pos["signal_id"]

        net_ret   = None
        gross_ret = None
        entry_p   = pos.get("entry_price")
        exit_p    = None
        outcome   = "EXPIRED"

        # Priority 1: FP book actual P&L
        if sym in fp_pnl:
            fp = fp_pnl[sym]
            entry_p   = fp.get("entry") or entry_p
            exit_p    = fp.get("ltp")   or exit_p
            net_pct   = fp.get("net_pct")
            gross_pct = fp.get("gross_pct")
            if net_pct is not None:
                net_ret   = round(float(net_pct), 4)
                gross_ret = round(float(gross_pct), 4) if gross_pct else round(net_ret + COST_FRAC * 100, 4)
                outcome   = "SETTLED_WIN" if net_ret > 0 else "SETTLED_LOSS"
                corrected += 1

        # Priority 2: changePct from live_quotes (only if allowed = current-day quotes)
        elif allow_changepct and sym in changepct_map:
            chg = changepct_map[sym]
            if entry_p and entry_p > 0:
                exit_p    = round(entry_p * (1 + chg / 100), 2)
            gross_ret = round(direction * chg / 100 * 100, 4)  # direction * changePct
            net_ret   = round(gross_ret - COST_FRAC * 100, 4)
            outcome   = "SETTLED_WIN" if net_ret > 0 else "SETTLED_LOSS"
            corrected += 1

        # Status count
        if outcome == "SETTLED_WIN":
            wins += 1
        elif outcome == "SETTLED_LOSS":
            losses += 1
        else:
            expired += 1

        db.settle_position(sig_id, {
            "exit_time":       NOW,
            "exit_price":      exit_p,
            "status":          outcome,
            "settled_at":      NOW,
            "final_return_pct": net_ret,
            "gross_return_pct": gross_ret,
            "outcome":         outcome,
            "cost_bps":        COST_FRAC * 10_000,
        })
        settled += 1

    # Update session-level analytics
    total = wins + losses
    win_rate = round(wins / total, 4) if total else None
    mean_ret = None
    if total:
        rets = db.execute_query(
            "SELECT final_return_pct FROM positions WHERE session_date=%s AND status NOT IN ('OPEN','EXPIRED')",
            (session_date,)
        )
        valid = [r["final_return_pct"] for r in rets if r["final_return_pct"] is not None]
        mean_ret = round(sum(valid) / len(valid), 4) if valid else None

    db.upsert_session(session_date=session_date, win_rate=win_rate, mean_return_pct=mean_ret)

    return {"settled": settled, "wins": wins, "losses": losses, "expired": expired,
            "corrected": corrected, "win_rate": win_rate, "mean_ret": mean_ret}


def main() -> None:
    print("=" * 60)
    print("  RE-SETTLE POSITIONS WITH ACTUAL MARKET RETURNS")
    print("=" * 60)

    # Oct 7 live changePct (current live_quotes.json)
    oct7_changepct = load_live_quotes_changepct()
    print(f"\nLive quotes changePct loaded: {len(oct7_changepct)} symbols")

    # ── Oct 6 ────────────────────────────────────────────────────────────────
    print("\n[1/2] Re-settling 2026-10-06...")
    fp_oct6 = load_fp_pnl_from_log("2026-10-06")
    print(f"  FP positions from log: {len(fp_oct6)}")
    result6 = resettle_session("2026-10-06", fp_oct6, {}, allow_changepct=False)
    print(f"  Settled: {result6['settled']}  |  Corrected: {result6['corrected']}")
    print(f"  Wins: {result6['wins']}  |  Losses: {result6['losses']}  |  Expired: {result6['expired']}")
    if result6["win_rate"] is not None:
        print(f"  Win rate: {result6['win_rate']:.0%}  |  Mean net: {result6['mean_ret']:+.3f}%")

    # ── Oct 7 ────────────────────────────────────────────────────────────────
    print("\n[2/2] Re-settling 2026-10-07...")
    fp_oct7 = load_session_summary_pnl()
    print(f"  FP positions from session_summary: {len(fp_oct7)}")
    result7 = resettle_session("2026-10-07", fp_oct7, oct7_changepct, allow_changepct=True)
    print(f"  Settled: {result7['settled']}  |  Corrected: {result7['corrected']}")
    print(f"  Wins: {result7['wins']}  |  Losses: {result7['losses']}  |  Expired: {result7['expired']}")
    if result7["win_rate"] is not None:
        print(f"  Win rate: {result7['win_rate']:.0%}  |  Mean net: {result7['mean_ret']:+.3f}%")

    # ── Verify ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VERIFICATION")
    print("=" * 60)
    rows = db.execute_query("""
        SELECT session_date, status, COUNT(*) n,
               SUM((entry_price=exit_price OR exit_price IS NULL)::int) still_same,
               ROUND(AVG(final_return_pct)::numeric,3) avg_ret
        FROM positions WHERE session_date IN (%s,%s)
        GROUP BY session_date, status ORDER BY session_date, status
    """, ("2026-10-06", "2026-10-07"))
    for r in rows:
        same = r["still_same"] or 0
        print(f"  {r['session_date']}  {r['status']:<16} n={r['n']}  still_same={same}  avg={r['avg_ret']}")

    print("\nRe-settlement complete ✓")


if __name__ == "__main__":
    main()
