"""
src.analytics.signal_ledger — Intraday Signal Lifecycle Tracker
================================================================

Tracks every signal from open (9:15 IST) to close (15:15 IST) on the
same trading day.

Lifecycle
---------
  OPEN         Signal recorded at first live quote after 9:15 IST
  SETTLED_WIN  Position closed profitably at 15:15 IST (net > 0)
  SETTLED_LOSS Position closed at a loss at 15:15 IST (net ≤ 0)

Flow (wired inside autorun_till_close.py)
-----------------------------------------
  sample_n == 1 (≈09:30 IST)
      → record_signals(scores, live_quotes)   # entry prices captured
  every sample (every 2 min)
      → update_mark_to_market(live_quotes)    # unrealized P&L updated
  post-close (15:30 IST)
      → settle_expired(session_date, final_quotes)  # WIN / LOSS recorded
      → write markdown report

Storage
-------
  artifacts/signal_ledger/signals.jsonl        append-only event log
  artifacts/signal_ledger/positions.json       mutable current state
  artifacts/signal_ledger/daily_reports/       per-session markdown files
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any

_BASE        = Path(__file__).parent.parent.parent
LEDGER_DIR   = _BASE / "artifacts" / "signal_ledger"
SIGNALS_FILE = LEDGER_DIR / "signals.jsonl"
POSITIONS_FILE = LEDGER_DIR / "positions.json"
REPORTS_DIR  = LEDGER_DIR / "daily_reports"

COST_BPS_FUTURES = 7.26   # round-trip (COST_MODEL_V2)
COST_BPS_EQUITY  = 27.65


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class SignalLedger:
    """
    Intraday signal lifecycle tracker.

    One position per (symbol, session_date).
    Entry = first live price after 9:15 IST on session_date.
    Exit  = live price at 15:15–15:30 IST on session_date.
    """

    def __init__(self, cost_bps: float = COST_BPS_FUTURES) -> None:
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self._cost_frac = cost_bps / 10_000.0
        self._positions: dict[str, dict[str, Any]] = self._load_positions()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_positions(self) -> dict[str, dict]:
        if POSITIONS_FILE.exists():
            try:
                return json.loads(POSITIONS_FILE.read_text())
            except Exception:
                return {}
        return {}

    def _save_positions(self) -> None:
        tmp = POSITIONS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._positions, indent=2, default=str))
        tmp.replace(POSITIONS_FILE)

    def _append_event(self, record: dict) -> None:
        SIGNALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with SIGNALS_FILE.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # ── Public API ────────────────────────────────────────────────────────────

    def record_signals(
        self,
        signals: list[dict],
        session_date: str,
        nifty_ltp: float | None,
        generated_at: str,
        live_quotes: dict | None = None,
    ) -> int:
        """
        Record signals at session open (first live quote ≈09:30 IST).

        Creates one OPEN position per active signal (direction != 0).
        Idempotent: re-calling with the same session_date is a no-op.

        Returns number of NEW positions created.
        """
        new_count = 0
        entry_time = _now_utc()

        for sig in signals:
            sym       = sig.get("symbol", "")
            direction = sig.get("direction", 0)
            if direction == 0 or not sym:
                continue

            pos_key = f"{sym}:{session_date}"
            if pos_key in self._positions:
                continue  # idempotent

            # Entry price from live quote
            entry_price: float | None = None
            if live_quotes and isinstance(live_quotes.get(sym), dict):
                ltp = live_quotes[sym].get("ltp")
                if ltp and float(ltp) > 0:
                    entry_price = float(ltp)

            record: dict[str, Any] = {
                "signal_id":       str(uuid.uuid4())[:16],
                "pos_key":         pos_key,
                "session_date":    session_date,
                "symbol":          sym,
                "generated_at":    generated_at,
                "entry_time":      entry_time,
                "data_date":       sig.get("data_date", ""),
                "direction":       direction,
                "score":           round(sig.get("score", 0.5), 4),
                "conviction":      sig.get("conviction", ""),
                "rank":            sig.get("rank"),
                "resolve_after":   session_date,   # same day — settles at post-close
                "entry_price":     entry_price,
                "last_price":      entry_price,
                "last_price_time": entry_time if entry_price else None,
                "exit_time":       None,
                "exit_price":      None,
                "unrealized_pct":  0.0 if entry_price else None,
                "status":          "OPEN",
                "settled_at":      None,
                "final_return_pct": None,
                "outcome":         None,
                "nifty_at_signal": nifty_ltp,
                "cost_bps":        self._cost_frac * 10_000,
            }

            self._positions[pos_key] = record
            self._append_event({**record, "_event": "OPEN"})
            new_count += 1

        if new_count > 0:
            self._save_positions()
        return new_count

    def update_mark_to_market(
        self,
        live_quotes: dict,
        today: str,
    ) -> int:
        """
        Update unrealized P&L for all OPEN positions of today.
        Called every sample (~every 2 minutes).
        Returns count of positions updated.
        """
        updated  = 0
        now_str  = _now_utc()

        for pos_key, pos in self._positions.items():
            if pos["status"] != "OPEN" or pos["session_date"] != today:
                continue
            sym = pos["symbol"]
            q   = live_quotes.get(sym, {})
            if not isinstance(q, dict):
                continue
            ltp = q.get("ltp")
            if not ltp or float(ltp) <= 0:
                continue

            current = float(ltp)
            pos["last_price"]      = current
            pos["last_price_time"] = now_str

            entry = pos.get("entry_price")
            if entry and entry > 0:
                gross = pos["direction"] * (current - entry) / entry
                net   = gross - self._cost_frac
                pos["unrealized_pct"] = round(net * 100, 4)
            updated += 1

        if updated > 0:
            self._save_positions()
        return updated

    def settle_session(
        self,
        session_date: str,
        final_quotes: dict | None = None,
    ) -> list[dict]:
        """
        Settle ALL OPEN positions for today at session close (15:15-15:30 IST).
        Uses final live price as exit price.
        Returns list of settled records.
        """
        settled: list[dict] = []
        exit_time = _now_utc()

        for pos_key, pos in self._positions.items():
            if pos["status"] != "OPEN" or pos["session_date"] != session_date:
                continue

            # Exit price: prefer final quote, fall back to last_price
            exit_price: float | None = pos.get("last_price")
            if final_quotes:
                sym = pos["symbol"]
                q   = final_quotes.get(sym, {})
                if isinstance(q, dict) and q.get("ltp") and float(q.get("ltp", 0)) > 0:
                    exit_price = float(q["ltp"])

            if exit_price and pos.get("entry_price") and pos["entry_price"] > 0:
                gross     = pos["direction"] * (exit_price - pos["entry_price"]) / pos["entry_price"]
                net       = gross - self._cost_frac
                final_ret = round(net * 100, 4)
                outcome   = "SETTLED_WIN" if net > 0 else "SETTLED_LOSS"
            else:
                final_ret = None
                outcome   = "EXPIRED"

            pos["status"]            = outcome
            pos["exit_time"]         = exit_time
            pos["exit_price"]        = exit_price
            pos["settled_at"]        = exit_time
            pos["final_return_pct"]  = final_ret
            pos["outcome"]           = outcome
            pos["unrealized_pct"]    = final_ret  # realized

            self._append_event({**pos, "_event": "SETTLED"})
            settled.append(pos)

        if settled:
            self._save_positions()
        return settled

    # keep settle_expired as alias for settle_session (backward-compat)
    def settle_expired(self, session_date: str, live_quotes: dict | None = None) -> list[dict]:
        return self.settle_session(session_date, final_quotes=live_quotes)

    def status_report(self) -> dict:
        open_pos  = [p for p in self._positions.values() if p["status"] == "OPEN"]
        wins      = [p for p in self._positions.values() if p["status"] == "SETTLED_WIN"]
        losses    = [p for p in self._positions.values() if p["status"] == "SETTLED_LOSS"]
        expired   = [p for p in self._positions.values() if p["status"] == "EXPIRED"]

        open_pnl   = [p["unrealized_pct"] for p in open_pos if p.get("unrealized_pct") is not None]
        settled_pnl= [p["final_return_pct"] for p in wins + losses if p.get("final_return_pct") is not None]

        open_sorted = sorted(
            [p for p in open_pos if p.get("unrealized_pct") is not None],
            key=lambda p: p["unrealized_pct"], reverse=True,
        )

        return {
            "total_positions":     len(self._positions),
            "open":                len(open_pos),
            "settled_win":         len(wins),
            "settled_loss":        len(losses),
            "expired":             len(expired),
            "win_rate":            len(wins) / (len(wins) + len(losses)) if (wins or losses) else None,
            "unrealized_mean_pct": round(sum(open_pnl)/len(open_pnl), 4) if open_pnl else None,
            "realized_mean_pct":   round(sum(settled_pnl)/len(settled_pnl), 4) if settled_pnl else None,
            "top_unrealized_wins":   [_ps(p) for p in open_sorted[:5]],
            "top_unrealized_losses": [_ps(p) for p in open_sorted[-5:][::-1]],
        }

    def open_positions(self) -> list[dict]:
        return [p for p in self._positions.values() if p["status"] == "OPEN"]

    def today_positions(self, session_date: str) -> list[dict]:
        return [p for p in self._positions.values() if p["session_date"] == session_date]

    def all_positions(self) -> list[dict]:
        return list(self._positions.values())

    def markdown_report(self, session_date: str) -> str:
        rpt  = self.status_report()
        now  = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        today_pos = self.today_positions(session_date)
        settled   = [p for p in today_pos if p["status"] != "OPEN"]

        lines = [
            f"# Signal Ledger — {session_date}",
            f"**Generated:** {now}  |  **Mode:** Intraday (09:15–15:15 IST)",
            "",
            "## Summary",
            f"| Status | Count | Mean Return |",
            f"|--------|-------|-------------|",
            f"| OPEN | {rpt['open']} | {rpt['unrealized_mean_pct']:+.2f}% unrealized |"
              if rpt['unrealized_mean_pct'] is not None else f"| OPEN | {rpt['open']} | — |",
            f"| SETTLED_WIN | {rpt['settled_win']} | — |",
            f"| SETTLED_LOSS | {rpt['settled_loss']} | — |",
            f"| Win rate | {rpt['win_rate']:.0%} | Realized: {rpt['realized_mean_pct']:+.2f}% |"
              if rpt['win_rate'] is not None and rpt['realized_mean_pct'] is not None
              else f"| Win rate | — | — |",
            "",
        ]

        if settled:
            lines += [
                f"## Settled Positions ({session_date})",
                "| Symbol | Dir | Score | Entry | Exit | Return | Outcome |",
                "|--------|-----|-------|-------|------|--------|---------|",
            ]
            for p in sorted(settled, key=lambda x: x.get("final_return_pct") or 0, reverse=True):
                d   = "LONG" if p["direction"] == 1 else "SHORT"
                ep  = f"₹{p['entry_price']:.2f}" if p.get("entry_price") else "—"
                xp  = f"₹{p['exit_price']:.2f}"  if p.get("exit_price")  else "—"
                ret = f"{p['final_return_pct']:+.2f}%" if p.get("final_return_pct") is not None else "—"
                lines.append(f"| {p['symbol']} | {d} | {p['score']:.4f} | {ep} | {xp} | {ret} | {p['outcome']} |")
            lines.append("")

        if rpt["top_unrealized_wins"]:
            lines += [
                "## Top Open (by unrealized P&L)",
                "| Symbol | Dir | Score | Entry | Last | Unrealized |",
                "|--------|-----|-------|-------|------|-----------|",
            ]
            for p in rpt["top_unrealized_wins"] + rpt["top_unrealized_losses"]:
                d  = "LONG" if p["direction"] == 1 else "SHORT"
                ep = f"₹{p['entry_price']:.2f}" if p.get("entry_price") else "—"
                lp = f"₹{p['last_price']:.2f}"  if p.get("last_price")  else "—"
                ur = f"{p['unrealized_pct']:+.2f}%" if p.get("unrealized_pct") is not None else "—"
                lines.append(f"| {p['symbol']} | {d} | {p['score']:.4f} | {ep} | {lp} | {ur} |")

        return "\n".join(lines)


def _ps(pos: dict) -> dict:
    return {k: pos.get(k) for k in
            ["symbol", "direction", "score", "conviction", "entry_price",
             "last_price", "unrealized_pct", "session_date"]}
