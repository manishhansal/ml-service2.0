"""
src/analytics/symbol_ic_tracker.py
--------------------------------------
Rolling per-symbol Information Coefficient (IC) tracker.

Motivation
-----------
The Sep-30 forensic analysis found 3 symbols destroyed session accuracy:
  ADANIENT  — near-neutral D-grade signal, -3% moves completely missed
  TCS       — SHORT while IT sector rallied +2% all day
  APOLLOHOSP — LONG during corporate earnings drop -8.8%

A global accuracy metric hides these symbol-level failures.  Tracking a
rolling 60-day IC per symbol lets autorun skip chronically wrong calls and
down-weight unreliable symbols rather than waiting for a full model retrain.

How it works
-------------
  1.  After each resolved forward-paper signal (actual outcome known),
      store (symbol, predicted_score, realized_return) in the tracker.
  2.  For each symbol maintain a rolling 60-trade window.
  3.  Compute Spearman rank IC on those 60 trades.
  4.  Symbols with rolling_ic < IC_FLOOR (default: -0.05) are flagged
      DEAD and suppressed in autorun for DEAD_COOLDOWN_DAYS (default: 5).
  5.  All state is persisted in artifacts/live_session/symbol_ic_state.json
      so it survives restarts.

Usage in autorun_till_close.py
-------------------------------
  tracker = SymbolICTracker()
  # After resolving forward-paper outcomes:
  tracker.record(symbol="TCS", score=0.72, realized_return=-0.031)
  # Before emitting signals:
  dead = tracker.dead_symbols()
  scores = [s for s in scores if s['symbol'] not in dead]
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.stats import spearmanr as _spearmanr
    def _ic(scores: list[float], returns: list[float]) -> float:
        if len(scores) < 10:
            return float("nan")
        ic, _ = _spearmanr(scores, returns)
        return float(ic) if np.isfinite(ic) else float("nan")
except ImportError:
    def _ic(scores: list[float], returns: list[float]) -> float:  # type: ignore[misc]
        return float("nan")

# ── Tunable parameters ────────────────────────────────────────────────────────
WINDOW_TRADES    = 60    # rolling window length (trades per symbol)
IC_FLOOR         = -0.05 # IC below this → symbol is DEAD for a while
DEAD_COOLDOWN_DAYS = 5   # days to suppress a DEAD symbol
MIN_TRADES_GATE  = 10    # minimum trades before IC is considered reliable
STATE_FILE       = Path("artifacts/live_session/symbol_ic_state.json")


class SymbolICTracker:
    """
    Rolling per-symbol IC tracker with DEAD-symbol suppression.

    Usage::

        tracker = SymbolICTracker()

        # Record an outcome (call after forward-paper resolution)
        tracker.record("RELIANCE", predicted_score=0.72, realized_return=0.014)

        # Check dead symbols before emitting signals
        dead = tracker.dead_symbols()

        # Get per-symbol IC summary for dashboard
        summary = tracker.summary()

        # IC for a specific symbol
        ic = tracker.symbol_ic("TCS")
    """

    def __init__(self, state_file: Path = STATE_FILE) -> None:
        self._path  = state_file
        self._state = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                pass
        return {"symbols": {}, "dead": {}}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._state, indent=2))

    # ── Core API ──────────────────────────────────────────────────────────────

    def record(
        self,
        symbol: str,
        predicted_score: float,
        realized_return: float,
        trade_date: str | None = None,
    ) -> None:
        """
        Record one resolved trade outcome.

        Args:
            symbol:            NSE trading symbol.
            predicted_score:   Model score (0–1; >0.5 = LONG prediction).
            realized_return:   Actual net return (positive = profit in predicted direction).
            trade_date:        ISO date string; defaults to today.
        """
        date = trade_date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        sym  = self._state["symbols"].setdefault(symbol, {"trades": []})

        sym["trades"].append({
            "date":   date,
            "score":  round(float(predicted_score), 4),
            "return": round(float(realized_return), 6),
        })
        # Keep only the last WINDOW_TRADES trades
        if len(sym["trades"]) > WINDOW_TRADES:
            sym["trades"] = sym["trades"][-WINDOW_TRADES:]

        # Recompute rolling IC
        rolling_ic = self._compute_ic(symbol)
        sym["rolling_ic"]   = round(rolling_ic, 4) if np.isfinite(rolling_ic) else None
        sym["n_trades"]      = len(sym["trades"])
        sym["last_updated"]  = date

        # Promote to DEAD if IC below floor with enough trades
        if (
            np.isfinite(rolling_ic)
            and rolling_ic < IC_FLOOR
            and sym["n_trades"] >= MIN_TRADES_GATE
        ):
            dead_until = (
                datetime.now(tz=timezone.utc) + timedelta(days=DEAD_COOLDOWN_DAYS)
            ).strftime("%Y-%m-%d")
            self._state["dead"][symbol] = {
                "ic_at_death": round(rolling_ic, 4),
                "dead_until":  dead_until,
                "n_trades":    sym["n_trades"],
            }

        self._save()

        # Dual-write: also persist to PostgreSQL symbol_ic table
        try:
            from src.data.signal_db import get_db as _get_db
            _get_db().upsert_symbol_ic_batch([{
                "symbol":       symbol,
                "n_trades":     sym["n_trades"],
                "rolling_ic":   sym.get("rolling_ic"),
                "last_updated": date,
            }])
        except Exception:
            pass  # PG unavailable — JSONL file is the fallback

    def record_batch(self, outcomes: list[dict]) -> None:
        """
        Record multiple outcomes at once.

        Each item: {"symbol": str, "score": float, "realized_return": float}
        """
        for o in outcomes:
            self.record(
                symbol          = o["symbol"],
                predicted_score = o["score"],
                realized_return = o["realized_return"],
                trade_date      = o.get("date"),
            )

    # ── Query API ─────────────────────────────────────────────────────────────

    def symbol_ic(self, symbol: str) -> float | None:
        """Return rolling IC for symbol, or None if < MIN_TRADES_GATE trades."""
        sym = self._state["symbols"].get(symbol)
        if not sym or sym.get("n_trades", 0) < MIN_TRADES_GATE:
            return None
        return sym.get("rolling_ic")

    def dead_symbols(self) -> set[str]:
        """Return the set of symbols currently suppressed (DEAD)."""
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        dead: set[str] = set()
        for sym, info in list(self._state["dead"].items()):
            if info.get("dead_until", "") >= today:
                dead.add(sym)
            else:
                del self._state["dead"][sym]   # cooldown expired — revive
        return dead

    def bottom_symbols(self, n: int = 10) -> list[dict]:
        """Return the n symbols with lowest rolling IC (worst performers)."""
        rows = []
        for sym, data in self._state["symbols"].items():
            ic = data.get("rolling_ic")
            if ic is not None and data.get("n_trades", 0) >= MIN_TRADES_GATE:
                rows.append({"symbol": sym, "ic": ic, "n_trades": data["n_trades"]})
        return sorted(rows, key=lambda x: x["ic"])[:n]

    def top_symbols(self, n: int = 10) -> list[dict]:
        """Return the n symbols with highest rolling IC (best performers)."""
        rows = []
        for sym, data in self._state["symbols"].items():
            ic = data.get("rolling_ic")
            if ic is not None and data.get("n_trades", 0) >= MIN_TRADES_GATE:
                rows.append({"symbol": sym, "ic": ic, "n_trades": data["n_trades"]})
        return sorted(rows, key=lambda x: -x["ic"])[:n]

    def summary(self) -> dict:
        """Full summary dict for dashboard / logging."""
        syms = self._state["symbols"]
        ics  = [v["rolling_ic"] for v in syms.values()
                if v.get("rolling_ic") is not None
                and v.get("n_trades", 0) >= MIN_TRADES_GATE]
        dead = self.dead_symbols()
        return {
            "n_symbols_tracked":  len(syms),
            "n_with_ic":          len(ics),
            "n_dead":             len(dead),
            "dead_symbols":       sorted(dead),
            "mean_ic":            round(float(np.mean(ics)), 4) if ics else None,
            "worst_ic":           round(float(np.min(ics)),  4) if ics else None,
            "best_ic":            round(float(np.max(ics)),  4) if ics else None,
            "bottom5":            self.bottom_symbols(5),
            "top5":               self.top_symbols(5),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_ic(self, symbol: str) -> float:
        trades = self._state["symbols"].get(symbol, {}).get("trades", [])
        if len(trades) < MIN_TRADES_GATE:
            return float("nan")
        scores  = [t["score"]  for t in trades]
        returns = [t["return"] for t in trades]
        return _ic(scores, returns)

    def apply_dead_filter(
        self,
        scores: list[dict],
        verbose: bool = True,
    ) -> tuple[list[dict], int]:
        """
        Filter dead symbols from a scored signal list.

        Args:
            scores:   List of signal dicts with "symbol" key.
            verbose:  Print suppressed symbols if any.

        Returns:
            (filtered_scores, n_suppressed)
        """
        dead     = self.dead_symbols()
        filtered = [s for s in scores if s["symbol"] not in dead]
        n_sup    = len(scores) - len(filtered)

        if n_sup > 0 and verbose:
            suppressed_syms = [s["symbol"] for s in scores if s["symbol"] in dead]
            dead_info = {
                sym: self._state["dead"].get(sym, {})
                for sym in suppressed_syms[:5]
            }
            print(f"  [SymbolIC] Suppressed {n_sup} dead symbols: "
                  f"{suppressed_syms[:5]}  "
                  f"(IC at death: {[v.get('ic_at_death') for v in dead_info.values()]})")

        return filtered, n_sup
