"""
src.analytics.signal_ledger — Intraday Signal Lifecycle Tracker
================================================================

Primary storage: SQLite via src.data.signal_db.SignalDB
  data/ml_signals.db  — sessions, signals, positions, price_snapshots tables
Audit trail:  artifacts/signal_ledger/signals.jsonl  (append-only backup)

Lifecycle (same-day intraday)
------------------------------
  OPEN         Entry price recorded at first live quote after 09:15 IST
  SETTLED_WIN  Closed at 15:15 IST with net positive return
  SETTLED_LOSS Closed at 15:15 IST with net negative return
  EXPIRED      Closed at 15:15 IST but entry price unavailable

Flow (wired in autorun_till_close.py)
--------------------------------------
  sample_n==1  (≈09:30 IST) → record_signals()       entry prices captured
  every sample (every 2 min) → update_mark_to_market() unrealized P&L live
  post-close (15:30 IST)    → settle_session()        WIN / LOSS + report
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_BASE         = Path(__file__).parent.parent.parent
LEDGER_DIR    = _BASE / "artifacts" / "signal_ledger"
SIGNALS_FILE  = LEDGER_DIR / "signals.jsonl"   # append-only audit backup
REPORTS_DIR   = LEDGER_DIR / "daily_reports"

COST_BPS_FUTURES = 7.26
COST_BPS_EQUITY  = 27.65


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class SignalLedger:
    """
    Intraday signal lifecycle tracker backed by SQLite.

    All data is stored in data/ml_signals.db (via SignalDB).
    JSONL audit trail kept in artifacts/signal_ledger/signals.jsonl.
    """

    def __init__(self, cost_bps: float = COST_BPS_FUTURES) -> None:
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self._cost_frac = cost_bps / 10_000.0

        # Primary: SQLite
        from src.data.signal_db import SignalDB
        self._db = SignalDB()

    # ── Audit trail ───────────────────────────────────────────────────────────

    def _append_audit(self, record: dict) -> None:
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
        Record active signals at session open (≈09:30 IST).
        Idempotent: re-calling with same session_date is a no-op.
        Returns number of NEW positions created.
        """
        entry_time  = _now_utc()
        active      = [s for s in signals if s.get("direction", 0) != 0 and s.get("symbol")]
        n_long      = sum(1 for s in active if s["direction"] == 1)
        n_short     = sum(1 for s in active if s["direction"] == -1)

        # Upsert session row
        self._db.upsert_session(
            session_date=session_date,
            n_scored=len(signals),
            n_long=n_long,
            n_short=n_short,
            nifty_ltp=nifty_ltp,
        )

        sig_rows: list[dict] = []
        pos_rows: list[dict] = []

        # Pre-check: get existing (symbol, session_date) pairs to skip them (true idempotency)
        existing = set(
            r["symbol"]
            for r in self._db.execute_query(
                "SELECT symbol FROM signals WHERE session_date=%s", (session_date,)
            )
        )

        for sig in active:
            sym   = sig["symbol"]
            if sym in existing:
                continue  # already recorded for this session — skip (true idempotent)

            sid   = str(uuid.uuid4())[:16]

            # Entry price from live quote
            entry_price: float | None = None
            if live_quotes and isinstance(live_quotes.get(sym), dict):
                ltp = live_quotes[sym].get("ltp")
                if ltp and float(ltp) > 0:
                    entry_price = float(ltp)

            sig_row: dict[str, Any] = {
                "signal_id":      sid,
                "session_date":   session_date,
                "symbol":         sym,
                "generated_at":   generated_at,
                "data_date":      sig.get("data_date", ""),
                "direction":      sig["direction"],
                "score":          round(sig.get("score", 0.5), 4),
                "conviction":     sig.get("conviction", ""),
                "rank":           sig.get("rank"),
                "cost_bps":       self._cost_frac * 10_000,
                "nifty_at_signal": nifty_ltp,
                "model_version":  None,
            }
            pos_row: dict[str, Any] = {
                "signal_id":       sid,
                "session_date":    session_date,
                "symbol":          sym,
                "direction":       sig["direction"],
                "entry_time":      entry_time,
                "entry_price":     entry_price,
                "exit_time":       None,
                "exit_price":      None,
                "last_price":      entry_price,
                "last_price_time": entry_time if entry_price else None,
                "unrealized_pct":  0.0 if entry_price else None,
                "status":          "OPEN",
                "settled_at":      None,
                "final_return_pct": None,
                "outcome":         None,
            }
            sig_rows.append(sig_row)
            pos_rows.append(pos_row)

        # Batch-insert signals (idempotent: INSERT OR IGNORE)
        new_sigs = self._db.insert_signals_batch(sig_rows)

        # Insert positions only for newly inserted signals
        inserted = 0
        for sig_row, pos_row in zip(sig_rows, pos_rows):
            if self._db.signal_exists(sig_row["signal_id"]):
                # signal was freshly inserted → insert position too
                self._db.upsert_position(pos_row)
                self._append_audit({**sig_row, **pos_row, "_event": "OPEN"})
                inserted += 1

        return new_sigs

    def update_mark_to_market(
        self,
        live_quotes: dict,
        today: str,
    ) -> int:
        """Update unrealized P&L for all OPEN positions. Called every 2 min."""
        now_str    = _now_utc()
        open_pos   = self._db.get_open_positions(today)
        if not open_pos:
            return 0

        price_updates: list[tuple] = []
        snap_rows: list[dict]      = []

        for pos in open_pos:
            sym = pos["symbol"]
            q   = live_quotes.get(sym, {})
            if not isinstance(q, dict):
                continue
            ltp = q.get("ltp")
            if not ltp or float(ltp) <= 0:
                continue

            current = float(ltp)
            unrealized: float | None = None
            entry = pos.get("entry_price")
            if entry and entry > 0:
                gross    = pos["direction"] * (current - entry) / entry
                unrealized = round((gross - self._cost_frac) * 100, 4)

            price_updates.append((current, unrealized, now_str, sym, today))
            snap_rows.append({
                "session_date":   today,
                "symbol":         sym,
                "snapshot_time":  now_str,
                "price":          current,
                "unrealized_pct": unrealized,
            })

        if price_updates:
            self._db.update_prices_batch(today, price_updates)
        if snap_rows:
            self._db.insert_snapshot_batch(today, snap_rows)

        return len(price_updates)

    def settle_session(
        self,
        session_date: str,
        final_quotes: dict | None = None,
    ) -> list[dict]:
        """Settle all OPEN positions at close (15:15-15:30 IST). Returns settled rows."""
        open_pos  = self._db.get_open_positions(session_date)
        exit_time = _now_utc()
        settled: list[dict] = []

        for pos in open_pos:
            sym = pos["symbol"]

            # Exit price: prefer final quote, fall back to last_price
            exit_price: float | None = pos.get("last_price")
            if final_quotes:
                q = final_quotes.get(sym, {})
                if isinstance(q, dict) and q.get("ltp") and float(q.get("ltp", 0)) > 0:
                    exit_price = float(q["ltp"])

            entry = pos.get("entry_price")
            if exit_price and entry and entry > 0:
                gross     = pos["direction"] * (exit_price - entry) / entry
                net       = gross - self._cost_frac
                final_ret = round(net * 100, 4)
                outcome   = "SETTLED_WIN" if net > 0 else "SETTLED_LOSS"
            else:
                final_ret = None
                outcome   = "EXPIRED"

            settle_data = {
                "exit_time":       exit_time,
                "exit_price":      exit_price,
                "status":          outcome,
                "settled_at":      exit_time,
                "final_return_pct": final_ret,
                "outcome":         outcome,
            }
            self._db.settle_position(pos["signal_id"], settle_data)
            self._append_audit({**pos, **settle_data, "_event": "SETTLED"})

            settled.append({**pos, **settle_data})

        # Update session summary
        if settled:
            wins   = sum(1 for p in settled if p["outcome"] == "SETTLED_WIN")
            losses = sum(1 for p in settled if p["outcome"] == "SETTLED_LOSS")
            rets   = [p["final_return_pct"] for p in settled if p.get("final_return_pct") is not None]
            # nifty_chg comes from the session row already set
            self._db.upsert_session(session_date=session_date, n_long=0, n_short=0)

        return settled

    # backward-compat alias
    def settle_expired(self, session_date: str, live_quotes: dict | None = None) -> list[dict]:
        return self.settle_session(session_date, final_quotes=live_quotes)

    # ── Reports / Queries ─────────────────────────────────────────────────────

    def status_report(self) -> dict:
        """Summary of all positions across all sessions."""
        from src.data.signal_db import SignalDB, DB_PATH
        db = self._db
        all_hist = db.get_historical_performance(days=365)
        wins   = sum(r["wins"] or 0 for r in all_hist)
        losses = sum(r["losses"] or 0 for r in all_hist)
        rets   = [r["mean_return_pct"] for r in all_hist if r.get("mean_return_pct") is not None]

        # Today's open
        from datetime import date
        today = date.today().isoformat()
        open_pos = db.get_open_positions(today)
        open_unrealized = [p["unrealized_pct"] for p in open_pos if p.get("unrealized_pct") is not None]

        return {
            "total_sessions":      len(all_hist),
            "total_settled":       wins + losses,
            "settled_win":         wins,
            "settled_loss":        losses,
            "expired":             0,   # not tracked at session level; always 0
            "win_rate":            wins / (wins + losses) if (wins + losses) else None,
            "realized_mean_pct":   round(sum(rets) / len(rets), 4) if rets else None,
            "open":                len(open_pos),
            "unrealized_mean_pct": round(sum(open_unrealized) / len(open_unrealized), 4) if open_unrealized else None,
            "db_stats":            db.db_stats(),
        }

    def open_positions(self) -> list[dict]:
        from datetime import date
        return self._db.get_open_positions(date.today().isoformat())

    def today_positions(self, session_date: str) -> list[dict]:
        return self._db.get_open_positions(session_date)

    def all_positions(self) -> list[dict]:
        """All positions (open + settled) for all sessions."""
        with self._db._connect() as conn:
            rows = conn.execute("""
                SELECT p.*, s.score, s.conviction, s.rank, s.generated_at,
                       s.data_date, s.nifty_at_signal
                FROM positions p JOIN signals s USING (signal_id)
                ORDER BY p.session_date DESC, s.score DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def markdown_report(self, session_date: str) -> str:
        summary = self._db.get_session_summary(session_date)
        conv    = self._db.get_conviction_stats()
        now     = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        wins   = summary.get("wins") or 0
        losses = summary.get("losses") or 0
        open_  = summary.get("open") or 0
        total  = summary.get("total") or 0
        win_pct = summary.get("win_pct")
        mean_r  = summary.get("mean_return")

        lines = [
            f"# Signal Ledger — {session_date}",
            f"**Generated:** {now}  |  **Mode:** Intraday (09:15–15:15 IST)",
            f"**Storage:** SQLite `data/ml_signals.db`",
            "",
            "## Session Summary",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total signals | {total} |",
            f"| OPEN | {open_} |",
            f"| SETTLED_WIN | {wins} |",
            f"| SETTLED_LOSS | {losses} |",
            f"| Win rate | {f'{win_pct:.1f}%' if win_pct else '—'} |",
            f"| Mean net return | {f'{mean_r:+.3f}%' if mean_r is not None else '—'} |",
            "",
        ]

        # Settled positions
        rows = self._db.execute_query("""
            SELECT p.symbol, p.direction, s.score, s.conviction,
                   p.entry_price, p.exit_price, p.final_return_pct, p.status
            FROM positions p JOIN signals s USING (signal_id)
            WHERE p.session_date=%s AND p.status!='OPEN'
            ORDER BY p.final_return_pct DESC
        """, (session_date,))

        if rows:
            lines += [
                "## Settled Positions",
                "| Symbol | Dir | Score | Conv | Entry | Exit | Return | Outcome |",
                "|--------|-----|-------|------|-------|------|--------|---------|",
            ]
            for r in rows:
                d   = "LONG" if r["direction"] == 1 else "SHORT"
                ep  = f"₹{r['entry_price']:.2f}" if r["entry_price"] else "—"
                xp  = f"₹{r['exit_price']:.2f}"  if r["exit_price"]  else "—"
                ret = f"{r['final_return_pct']:+.2f}%" if r["final_return_pct"] is not None else "—"
                mk  = "✅" if r["status"] == "SETTLED_WIN" else "❌"
                lines.append(
                    f"| {r['symbol']} | {d} | {r['score']:.4f} | {r['conviction'] or '—'} "
                    f"| {ep} | {xp} | {ret} | {mk} |"
                )
            lines.append("")

        if conv:
            lines += [
                "## Conviction Grade Performance (all sessions)",
                "| Grade | Trades | Win% | Mean Return |",
                "|-------|--------|------|-------------|",
            ]
            for c in conv:
                lines.append(
                    f"| {c['conviction']} | {c['trades']} | {c['win_pct']:.1f}% | {c['mean_return']:+.3f}% |"
                )

        return "\n".join(lines)
