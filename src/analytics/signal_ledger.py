"""
src.analytics.signal_ledger — Signal Lifecycle Tracking System
==============================================================

Fills the 4 gaps identified in the signal tracking audit (2026-10-06):

  GAP 1  No expiry timestamp per live signal
  GAP 2  No live status (OPEN / SUCCESS / FAILURE) in real-time
  GAP 3  No per-signal entry time (only 2-min snapshots)
  GAP 4  ForecastLedger never settling; coverage only 153/285 symbols

Signal Lifecycle:
  OPEN        Signal generated; position not yet expired
  SETTLED_WIN  Settled profitably (direction correct, return > cost)
  SETTLED_LOSS Settled at a loss (direction wrong or cost > move)
  EXPIRED     T+7 trading days elapsed; closed at last available price

Storage (all append-only / atomic writes):
  artifacts/signal_ledger/signals.jsonl     — immutable signal log
  artifacts/signal_ledger/positions.json    — mutable current state per position
  artifacts/signal_ledger/daily_reports/    — per-session markdown summaries

Usage:
    ledger = SignalLedger()
    ledger.record_signals(signals, session_date, nifty_ltp, generated_at)
    ledger.update_mark_to_market(live_quotes, session_date)
    ledger.settle_expired(session_date, live_quotes)
    report = ledger.status_report()
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# ── Storage paths ─────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent.parent.parent
LEDGER_DIR   = _BASE / "artifacts" / "signal_ledger"
SIGNALS_FILE = LEDGER_DIR / "signals.jsonl"          # append-only log
POSITIONS_FILE = LEDGER_DIR / "positions.json"        # mutable current state
REPORTS_DIR  = LEDGER_DIR / "daily_reports"

COST_BPS_FUTURES = 7.26        # round-trip cost for futures (COST_MODEL_V2)
COST_BPS_EQUITY  = 27.65
HORIZON_DAYS     = 7           # 7 trading days forward


def _trading_days_ahead(start: date, n: int) -> date:
    """Return the date that is exactly n NSE trading days after start.
    Uses weekday check only (no holiday lookup — good enough for 7-day window).
    """
    try:
        from src.validation.calendar import is_trading_day
    except Exception:
        is_trading_day = lambda d: d.weekday() < 5

    current = start
    days_counted = 0
    while days_counted < n:
        current += timedelta(days=1)
        if is_trading_day(current):
            days_counted += 1
    return current


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class SignalLedger:
    """
    Tracks the full lifecycle of every live signal from generation to settlement.

    Thread-safety: not thread-safe. Call from a single process (post-close runner).
    Atomic writes: uses tmp-file + rename for positions.json to prevent corruption.
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

    def _append_signal(self, record: dict) -> None:
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
        Record new signals from today's scoring session.

        For each active signal (direction != 0):
          - If this symbol+session_date combination is new → create position
          - If position already exists for today → skip (idempotent)

        Returns the number of NEW signals recorded.
        """
        new_count = 0
        session_dt = date.fromisoformat(session_date)
        resolve_date = _trading_days_ahead(session_dt, HORIZON_DAYS)

        for sig in signals:
            sym       = sig.get("symbol", "")
            direction = sig.get("direction", 0)
            if direction == 0 or not sym:
                continue

            # Position key: symbol + session date (one position per symbol per day)
            pos_key = f"{sym}:{session_date}"
            if pos_key in self._positions:
                continue  # already recorded for today

            # Entry price from live quote if available
            entry_price: float | None = None
            if live_quotes and isinstance(live_quotes.get(sym), dict):
                ltp = live_quotes[sym].get("ltp")
                if ltp and float(ltp) > 0:
                    entry_price = float(ltp)

            signal_id = str(uuid.uuid4())[:16]
            record: dict[str, Any] = {
                "signal_id":         signal_id,
                "pos_key":           pos_key,
                "session_date":      session_date,
                "symbol":            sym,
                "generated_at":      generated_at,
                "data_date":         sig.get("data_date", ""),
                "direction":         direction,
                "score":             round(sig.get("score", 0.5), 4),
                "conviction":        sig.get("conviction", ""),
                "rank":              sig.get("rank"),
                "resolve_after":     resolve_date.isoformat(),
                "horizon_days":      HORIZON_DAYS,
                "entry_price":       entry_price,
                "last_price":        entry_price,
                "last_price_date":   session_date if entry_price else None,
                "unrealized_pct":    0.0 if entry_price else None,
                "status":            "OPEN",
                "settled_at":        None,
                "final_return_pct":  None,
                "outcome":           None,
                "nifty_at_signal":   nifty_ltp,
                "cost_bps":          self._cost_frac * 10_000,
                "regime":            _infer_regime(nifty_ltp),
            }

            self._positions[pos_key] = record
            self._append_signal(record)
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
        Update unrealized P&L for all OPEN positions using current live prices.
        Returns count of positions updated.
        """
        updated = 0
        for pos_key, pos in self._positions.items():
            if pos["status"] != "OPEN":
                continue
            sym = pos["symbol"]
            q = live_quotes.get(sym, {})
            if not isinstance(q, dict):
                continue
            ltp = q.get("ltp")
            if not ltp or float(ltp) <= 0:
                continue

            current_price = float(ltp)
            pos["last_price"]      = current_price
            pos["last_price_date"] = today

            entry = pos.get("entry_price")
            if entry and entry > 0:
                direction = pos["direction"]
                gross     = direction * (current_price - entry) / entry
                net       = gross - self._cost_frac
                pos["unrealized_pct"] = round(net * 100, 4)
            updated += 1

        if updated > 0:
            self._save_positions()
        return updated

    def settle_expired(
        self,
        today: str,
        live_quotes: dict | None = None,
    ) -> list[dict]:
        """
        Settle all OPEN positions whose resolve_after date has passed.
        Uses last available price as exit price.
        Returns list of settled position records.
        """
        today_dt = date.fromisoformat(today)
        settled: list[dict] = []

        for pos_key, pos in self._positions.items():
            if pos["status"] != "OPEN":
                continue
            resolve_dt = date.fromisoformat(pos["resolve_after"])
            if today_dt < resolve_dt:
                continue

            # Get exit price: prefer live quote, fall back to last_price
            exit_price: float | None = pos.get("last_price")
            if live_quotes:
                sym = pos["symbol"]
                q   = live_quotes.get(sym, {})
                if isinstance(q, dict) and q.get("ltp") and float(q.get("ltp", 0)) > 0:
                    exit_price = float(q["ltp"])

            if exit_price and pos.get("entry_price") and pos["entry_price"] > 0:
                direction    = pos["direction"]
                gross        = direction * (exit_price - pos["entry_price"]) / pos["entry_price"]
                net          = gross - self._cost_frac
                final_ret    = round(net * 100, 4)
                outcome      = "SETTLED_WIN" if net > 0 else "SETTLED_LOSS"
            else:
                final_ret = None
                outcome   = "EXPIRED"

            pos["status"]           = outcome
            pos["settled_at"]       = _now_utc()
            pos["final_return_pct"] = final_ret
            pos["outcome"]          = outcome
            pos["last_price"]       = exit_price
            pos["last_price_date"]  = today
            pos["unrealized_pct"]   = final_ret  # final = realized

            # Append settlement record to signals.jsonl
            self._append_signal({**pos, "_event": "SETTLED"})
            settled.append(pos)

        if settled:
            self._save_positions()
        return settled

    def status_report(self) -> dict:
        """Return a summary dict of all positions by status."""
        open_pos      = [p for p in self._positions.values() if p["status"] == "OPEN"]
        wins          = [p for p in self._positions.values() if p["status"] == "SETTLED_WIN"]
        losses        = [p for p in self._positions.values() if p["status"] == "SETTLED_LOSS"]
        expired       = [p for p in self._positions.values() if p["status"] == "EXPIRED"]

        # Unrealized P&L on open positions
        open_with_pnl = [p for p in open_pos if p.get("unrealized_pct") is not None]
        unrealized_mean = (
            sum(p["unrealized_pct"] for p in open_with_pnl) / len(open_with_pnl)
            if open_with_pnl else None
        )

        # Realized P&L
        settled_rets = [p["final_return_pct"] for p in wins + losses
                        if p.get("final_return_pct") is not None]
        realized_mean = sum(settled_rets) / len(settled_rets) if settled_rets else None

        # Top unrealized winners/losers
        open_sorted = sorted(open_with_pnl, key=lambda p: p["unrealized_pct"] or 0, reverse=True)

        return {
            "total_positions":   len(self._positions),
            "open":              len(open_pos),
            "settled_win":       len(wins),
            "settled_loss":      len(losses),
            "expired":           len(expired),
            "win_rate":          len(wins) / (len(wins) + len(losses)) if wins or losses else None,
            "unrealized_mean_pct": round(unrealized_mean, 4) if unrealized_mean is not None else None,
            "realized_mean_pct": round(realized_mean, 4) if realized_mean is not None else None,
            "top_unrealized_wins":   [_pos_summary(p) for p in open_sorted[:5]],
            "top_unrealized_losses": [_pos_summary(p) for p in open_sorted[-5:][::-1]],
        }

    def open_positions(self) -> list[dict]:
        return [p for p in self._positions.values() if p["status"] == "OPEN"]

    def all_positions(self) -> list[dict]:
        return list(self._positions.values())

    def markdown_report(self, session_date: str) -> str:
        rpt = self.status_report()
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"# Signal Ledger Report — {session_date}",
            f"**Generated:** {now}  |  **Horizon:** {HORIZON_DAYS} trading days",
            "",
            "## Summary",
            f"| Status | Count | Mean Return |",
            f"|--------|-------|-------------|",
            f"| OPEN | {rpt['open']} | {rpt['unrealized_mean_pct']:+.2f}% unrealized |" if rpt['unrealized_mean_pct'] is not None else f"| OPEN | {rpt['open']} | — |",
            f"| SETTLED_WIN | {rpt['settled_win']} | {rpt['realized_mean_pct']:+.2f}% |" if rpt['realized_mean_pct'] is not None else f"| SETTLED_WIN | {rpt['settled_win']} | — |",
            f"| SETTLED_LOSS | {rpt['settled_loss']} | — |",
            f"| EXPIRED | {rpt['expired']} | — |",
            f"| **Win rate** | **{rpt['win_rate']:.0%}** | — |" if rpt['win_rate'] is not None else "| Win rate | — | — |",
            "",
        ]

        if rpt["top_unrealized_wins"]:
            lines += ["## Top Unrealized Wins (Open)", "| Symbol | Dir | Score | Entry | Last | Unrealized | Expires |", "|--------|-----|-------|-------|------|-----------|---------|"]
            for p in rpt["top_unrealized_wins"]:
                lines.append(f"| {p['symbol']} | {'LONG' if p['direction']==1 else 'SHORT'} | {p['score']:.4f} | {p.get('entry_price','—')} | {p.get('last_price','—')} | {p.get('unrealized_pct',0):+.2f}% | {p['resolve_after']} |")
            lines.append("")

        if rpt["top_unrealized_losses"]:
            lines += ["## Top Unrealized Losses (Open)", "| Symbol | Dir | Score | Entry | Last | Unrealized | Expires |", "|--------|-----|-------|-------|------|-----------|---------|"]
            for p in rpt["top_unrealized_losses"]:
                lines.append(f"| {p['symbol']} | {'LONG' if p['direction']==1 else 'SHORT'} | {p['score']:.4f} | {p.get('entry_price','—')} | {p.get('last_price','—')} | {p.get('unrealized_pct',0):+.2f}% | {p['resolve_after']} |")
            lines.append("")

        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_regime(nifty_ltp: float | None) -> str:
    """Infer regime from NIFTY LTP — placeholder; proper regime uses 20d return."""
    return "UNKNOWN"


def _pos_summary(pos: dict) -> dict:
    return {
        "symbol":         pos["symbol"],
        "direction":      pos["direction"],
        "score":          pos["score"],
        "conviction":     pos.get("conviction", ""),
        "entry_price":    pos.get("entry_price"),
        "last_price":     pos.get("last_price"),
        "unrealized_pct": pos.get("unrealized_pct"),
        "resolve_after":  pos["resolve_after"],
        "session_date":   pos["session_date"],
    }
