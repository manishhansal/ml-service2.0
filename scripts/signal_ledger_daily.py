#!/usr/bin/env python3
"""
scripts/signal_ledger_daily.py
───────────────────────────────
Post-close EOD runner for the Signal Lifecycle Ledger.

Runs once after market close each trading day:

  1. Load today's final signals from artifacts/live_session/latest_scores.json
  2. Load live quotes from artifacts/live_session/live_quotes.json
  3. Record any NEW signals (first appearance with direction != 0)
  4. Update mark-to-market on all OPEN positions
  5. Settle expired positions (resolve_after <= today)
  6. Write daily markdown report to reports/live/signal_ledger_{date}.md
  7. Print summary to stdout

Usage:
    PYTHONPATH=. python3 scripts/signal_ledger_daily.py
    PYTHONPATH=. python3 scripts/signal_ledger_daily.py --date 2026-10-06
    PYTHONPATH=. python3 scripts/signal_ledger_daily.py --status   # print status only
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

SESSION_DIR  = BASE / "artifacts" / "live_session"
REPORTS_DIR  = BASE / "reports" / "live"


def load_today(session_date: str | None = None):
    scores_path = SESSION_DIR / "latest_scores.json"
    quotes_path = SESSION_DIR / "live_quotes.json"

    if not scores_path.exists():
        print("ERROR: latest_scores.json not found — run after market close")
        sys.exit(1)

    scores = json.load(open(scores_path))
    quotes = {}
    if quotes_path.exists():
        raw = json.load(open(quotes_path))
        quotes = raw.get("quotes", {})

    if session_date is None:
        session_date = scores.get("session_date", datetime.now().strftime("%Y-%m-%d"))

    # Fill missing LTPs from parquet last close
    parquet_dir = BASE / "data/1d/1d"
    import pandas as pd
    for sig in scores.get("signals", []):
        sym = sig.get("symbol", "")
        if not sym:
            continue
        # Skip if we already have a live quote with ltp
        if isinstance(quotes.get(sym), dict) and quotes[sym].get("ltp"):
            continue
        pf = parquet_dir / f"{sym}.parquet"
        if pf.exists():
            try:
                df = pd.read_parquet(str(pf), columns=["close"])
                if len(df) > 0:
                    close_price = float(df["close"].iloc[-1])
                    if close_price > 0:
                        quotes.setdefault(sym, {})
                        if isinstance(quotes[sym], dict) and not quotes[sym].get("ltp"):
                            quotes[sym]["ltp"] = close_price
                            quotes[sym]["changePct"] = quotes[sym].get("changePct")
            except Exception:
                pass

    return scores, quotes, session_date


def main(session_date: str | None = None, status_only: bool = False) -> None:
    from src.analytics.signal_ledger import SignalLedger

    ledger = SignalLedger()

    if status_only:
        rpt = ledger.status_report()
        print(json.dumps(rpt, indent=2, default=str))
        return

    scores, quotes, session_date = load_today(session_date)

    signals      = scores.get("signals", [])
    generated_at = scores.get("generated_at", datetime.now(tz=timezone.utc).isoformat())
    nifty_ltp    = scores.get("nifty_ltp")
    nifty_chg    = scores.get("nifty_chg", 0)

    active_signals = [s for s in signals if s.get("direction", 0) != 0]
    print(f"\n{'=' * 60}")
    print(f"  SIGNAL LEDGER  |  {session_date}  |  NIFTY {nifty_chg:+.2f}%")
    print(f"{'=' * 60}")
    print(f"  Active signals: {len(active_signals)} (LONG={sum(1 for s in active_signals if s['direction']==1)}, "
          f"SHORT={sum(1 for s in active_signals if s['direction']==-1)})")
    print(f"  Live quotes:    {sum(1 for q in quotes.values() if isinstance(q,dict) and q.get('ltp'))} symbols")

    # 1. Settle today's positions at close
    settled = ledger.settle_session(session_date, final_quotes=quotes)
    if settled:
        wins   = [p for p in settled if p["status"] == "SETTLED_WIN"]
        losses = [p for p in settled if p["status"] == "SETTLED_LOSS"]
        exps   = [p for p in settled if p["status"] == "EXPIRED"]
        settled_rets = [p["final_return_pct"] for p in wins + losses if p.get("final_return_pct") is not None]
        mean_ret = sum(settled_rets) / len(settled_rets) if settled_rets else 0.0
        print(f"\n  SETTLED TODAY: {len(settled)} positions")
        print(f"    Wins:    {len(wins)}  |  Losses: {len(losses)}  |  Expired: {len(exps)}")
        print(f"    Mean net return: {mean_ret:+.2f}%")
        if wins:
            top_win = max(wins, key=lambda p: p.get("final_return_pct") or 0)
            print(f"    Best:   {top_win['symbol']} {top_win['direction']==1 and 'LONG' or 'SHORT'} {top_win.get('final_return_pct',0):+.2f}%  "
                  f"(entry ₹{top_win.get('entry_price','?'):.1f} → exit ₹{top_win.get('exit_price','?'):.1f})" if top_win.get("entry_price") else "")
        if losses:
            worst = min(losses, key=lambda p: p.get("final_return_pct") or 0)
            print(f"    Worst:  {worst['symbol']} {worst['direction']==1 and 'LONG' or 'SHORT'} {worst.get('final_return_pct',0):+.2f}%  "
                  f"(entry ₹{worst.get('entry_price','?'):.1f} → exit ₹{worst.get('exit_price','?'):.1f})" if worst.get("entry_price") else "")
    else:
        print(f"\n  No positions to settle today (run at post-close time).")

    # 2. Update mark-to-market on any remaining OPEN positions (safety net)
    updated = ledger.update_mark_to_market(live_quotes=quotes, today=session_date)
    if updated:
        print(f"  Mark-to-market: {updated} open positions updated (safety net)")

    # 3. Record signals if not already captured during intraday (fallback path)
    new_count = ledger.record_signals(
        signals=active_signals,
        session_date=session_date,
        nifty_ltp=nifty_ltp,
        generated_at=generated_at,
        live_quotes=quotes,
    )
    if new_count:
        print(f"  New signals recorded (fallback): {new_count}")

    # 4. Status report
    rpt = ledger.status_report()
    print(f"\n  LEDGER STATUS")
    print(f"    Open:         {rpt['open']}")
    print(f"    Settled wins: {rpt['settled_win']}")
    print(f"    Settled loss: {rpt['settled_loss']}")
    print(f"    Expired:      {rpt['expired']}")
    if rpt["win_rate"] is not None:
        print(f"    Win rate:     {rpt['win_rate']:.0%}")
    if rpt["unrealized_mean_pct"] is not None:
        print(f"    Unrealized:   {rpt['unrealized_mean_pct']:+.2f}% mean")

    # Top open positions
    open_pos = sorted(
        [p for p in ledger.open_positions() if p.get("unrealized_pct") is not None],
        key=lambda p: p.get("unrealized_pct", 0), reverse=True,
    )
    if open_pos:
        print(f"\n  TOP OPEN POSITIONS (by unrealized P&L):")
        print(f"  {'Symbol':<14} {'Dir':>6} {'Score':>7} {'Entry':>8} {'Last':>8} {'Unreal':>9} {'Expires'}")
        print("  " + "-" * 68)
        for p in open_pos[:10]:
            d = "LONG " if p["direction"]==1 else "SHORT"
            ep = f"{p['entry_price']:.1f}" if p.get("entry_price") else "—"
            lp = f"{p['last_price']:.1f}"  if p.get("last_price")  else "—"
            ur = f"{p['unrealized_pct']:+.2f}%" if p.get("unrealized_pct") is not None else "—"
            print(f"  {p['symbol']:<14} {d:>6} {p['score']:>7.4f} {ep:>8} {lp:>8} {ur:>9} {p['resolve_after']}")
        if len(open_pos) > 10:
            print(f"  ... and {len(open_pos)-10} more")

    # 5. Save markdown report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"signal_ledger_{session_date}.md"
    report_path.write_text(ledger.markdown_report(session_date))
    print(f"\n  Report: {report_path.relative_to(BASE)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Ledger daily EOD runner")
    parser.add_argument("--date",   default=None, help="Session date YYYY-MM-DD (default: today)")
    parser.add_argument("--status", action="store_true", help="Print current status only")
    args = parser.parse_args()
    main(session_date=args.date, status_only=args.status)
