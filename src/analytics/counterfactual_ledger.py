"""
src.analytics.counterfactual_ledger — Phil-inspired counterfactual trade grader.

Problem
-------
The DrawdownManager and score threshold gates block some signals. Currently
ml-service2.0 never knows whether the blocked signals would have been profitable.
The gates might be too conservative (blocking good trades) or correctly
filtering losers — but without measurement, we cannot tell.

Solution (from Phil's core/counterfactual.py)
----------------------------------------------
Record every blocked/filtered signal with its reason.  When exit prices are
available, compute: "What would this trade have returned?"

This creates a mechanical answer to:
    "Did DrawdownManager CAUTION/DEFENSIVE/HALTED correctly block losers?"
    "Did the score threshold correctly filter low-conviction signals?"
    "Did the sector-regime dimmer correctly suppress defensive-sector shorts?"

If blocked signals consistently LOSE → gates are calibrated correctly.
If blocked signals consistently WIN  → gates need loosening.

Block reasons tracked:
    DRAWDOWN_CAUTION    — DrawdownManager CAUTION (75% sizing → some blocked)
    DRAWDOWN_DEFENSIVE  — DrawdownManager DEFENSIVE (50% sizing)
    DRAWDOWN_HALTED     — DrawdownManager HALTED (all blocked)
    THRESHOLD_FILTERED  — Score within neutral band (threshold sweep filter)
    SECTOR_REGIME_DIM   — Sector-regime dimmer suppressed signal
    SIGNAL_BELOW_MIN_IC — AlphaDecayDetector flagged low IC
    MANUALLY_EXCLUDED   — data quality flag (e.g. TATAMOTORS)

File:
    artifacts/counterfactual/blocked_signals.jsonl

Usage::

    ledger = CounterfactualLedger()

    # When DrawdownManager blocks a position:
    ledger.record_blocked(symbol="DRREDDY", direction=-1, score=0.42,
                          entry_price=1200.4, reason="THRESHOLD_FILTERED")

    # At end of day:
    report = ledger.resolve({"DRREDDY": -1.99, "MARUTI": -0.69, ...})
    # → reveals that THRESHOLD_FILTERED blocks correctly avoided losers
    #   OR that those were false positives

Requirements: Phil's counterfactual.py, mandate §4.
"""
from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)

LEDGER_PATH = Path("artifacts/counterfactual/blocked_signals.jsonl")

BLOCK_REASONS = {
    "DRAWDOWN_CAUTION",
    "DRAWDOWN_DEFENSIVE",
    "DRAWDOWN_HALTED",
    "THRESHOLD_FILTERED",
    "SECTOR_REGIME_DIM",
    "SIGNAL_BELOW_MIN_IC",
    "MANUALLY_EXCLUDED",
}


class CounterfactualLedger:
    """
    Records signals blocked by any gate and resolves them at end of day.

    Phil's system grades EVERY declined trade.  This does the same for
    NSE F&O signals, split by blocking reason so each gate is evaluated
    independently.

    A gate is adding value if its blocked signals have worse realized
    outcomes than the non-blocked book.  A gate is too conservative if
    its blocked signals consistently outperform.
    """

    def __init__(self, ledger_path: Path = LEDGER_PATH) -> None:
        self._path = Path(ledger_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Write ─────────────────────────────────────────────────────────────────

    def record_blocked(
        self,
        symbol: str,
        direction: int,
        score: float,
        entry_price: float,
        reason: str,
        session_date: str = "",
        context: dict | None = None,
    ) -> str:
        """
        Record a signal blocked by any gate.

        Args:
            symbol:       NSE F&O symbol.
            direction:    +1 (LONG) or -1 (SHORT).
            score:        LightGBM probability score [0,1].
            entry_price:  Price at time of blocking decision.
            reason:       One of BLOCK_REASONS.
            session_date: "YYYY-MM-DD".
            context:      Optional dict with gate-specific context.

        Returns:
            Record ID (12-char hex).
        """
        if reason not in BLOCK_REASONS:
            reason = "MANUALLY_EXCLUDED"   # safe default

        row = {
            "id":           uuid.uuid4().hex[:12],
            "ts":           datetime.now(tz=timezone.utc).isoformat(),
            "session_date": session_date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            "symbol":       symbol,
            "direction":    direction,
            "score":        round(score, 6),
            "entry_price":  round(entry_price, 4),
            "reason":       reason,
            "context":      context or {},
            "status":       "unresolved",
        }
        with self._path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        logger.debug(
            "counterfactual_blocked_recorded",
            symbol=symbol, reason=reason, score=round(score, 4),
        )
        return row["id"]

    def record_blocked_batch(
        self,
        blocked_signals: list[dict],
        session_date: str = "",
    ) -> int:
        """Record multiple blocked signals in one call."""
        ts  = datetime.now(tz=timezone.utc).isoformat()
        date = session_date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        written = 0
        with self._path.open("a") as fh:
            for s in blocked_signals:
                row = {
                    "id":           uuid.uuid4().hex[:12],
                    "ts":           ts,
                    "session_date": date,
                    "symbol":       s.get("symbol", ""),
                    "direction":    int(s.get("direction", 0)),
                    "score":        round(float(s.get("score", 0.5)), 6),
                    "entry_price":  round(float(s.get("entry_price", 0)), 4),
                    "reason":       s.get("reason", "MANUALLY_EXCLUDED"),
                    "context":      s.get("context", {}),
                    "status":       "unresolved",
                }
                fh.write(json.dumps(row) + "\n")
                written += 1
        logger.info(
            "counterfactual_batch_recorded",
            n=written, session_date=date,
        )
        return written

    # ── Resolve ───────────────────────────────────────────────────────────────

    def resolve(
        self,
        session_date: str,
        realized_returns: dict[str, float],
    ) -> dict[str, Any]:
        """
        At end of day: compute what each blocked signal would have returned.

        Args:
            session_date:     "YYYY-MM-DD"
            realized_returns: {symbol: net_pct} end-of-day net returns.

        Returns:
            CounterfactualReport dict with per-reason analysis.
        """
        rows = self._load_all()
        resolve_ts = datetime.now(tz=timezone.utc).isoformat()
        updated: list[dict] = []

        for row in rows:
            if row.get("session_date") != session_date:
                updated.append(row)
                continue
            if row.get("status") != "unresolved":
                updated.append(row)
                continue
            sym = row["symbol"]
            if sym not in realized_returns:
                updated.append(row)
                continue

            net_pct = float(realized_returns[sym])
            direction = int(row.get("direction", 0))
            # Net P&L for direction-aware position
            directional_pnl = direction * net_pct
            row["status"]           = "resolved"
            row["realized_net_pct"] = round(net_pct, 4)
            row["counterfactual_pnl"] = round(directional_pnl, 4)
            row["would_have_won"]   = directional_pnl > 0
            row["resolved_at"]      = resolve_ts
            updated.append(row)

        self._path.write_text("".join(json.dumps(r) + "\n" for r in updated))

        # Build report
        resolved = [r for r in updated
                    if r.get("session_date") == session_date
                    and r.get("status") == "resolved"]
        return self._build_report(resolved, session_date)

    # ── Report ────────────────────────────────────────────────────────────────

    def gate_value_report(
        self,
        session_date: str | None = None,
        last_n_sessions: int | None = None,
    ) -> dict[str, Any]:
        """
        Answer: "Did each blocking gate add or subtract value?"

        Returns per-reason analysis:
            - positive mean_counterfactual_pnl → gate blocked WINNERS (bad gate)
            - negative mean_counterfactual_pnl → gate blocked LOSERS  (good gate)
        """
        rows = self._load_all()
        resolved = [r for r in rows if r.get("status") == "resolved"]

        if session_date:
            resolved = [r for r in resolved if r.get("session_date") == session_date]
        elif last_n_sessions:
            dates = sorted({r["session_date"] for r in resolved}, reverse=True)
            keep_dates = set(dates[:last_n_sessions])
            resolved = [r for r in resolved if r.get("session_date") in keep_dates]

        return self._build_report(resolved, session_date or "all")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_all(self) -> list[dict]:
        if not self._path.exists():
            return []
        rows = []
        for line in self._path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def _build_report(self, resolved: list[dict], label: str) -> dict[str, Any]:
        if not resolved:
            return {"label": label, "n": 0, "verdict": "NO_DATA"}

        by_reason: dict[str, list[dict]] = defaultdict(list)
        for r in resolved:
            by_reason[r.get("reason", "UNKNOWN")].append(r)

        per_reason = {}
        verdicts = {}
        for reason, rs in sorted(by_reason.items()):
            pnls = [r.get("counterfactual_pnl", 0.0) for r in rs]
            wins  = sum(1 for p in pnls if p > 0)
            mean_ = float(np.mean(pnls))
            # Verdict: negative mean = gate correctly blocked losers
            verdict = "GATE_ADDING_VALUE" if mean_ < 0 else "GATE_TOO_CONSERVATIVE"
            per_reason[reason] = {
                "n":                     len(rs),
                "wins_if_not_blocked":   wins,
                "win_rate_if_not_blocked": round(wins / len(rs), 3),
                "mean_counterfactual_pnl": round(mean_, 4),
                "verdict":               verdict,
            }
            verdicts[reason] = verdict

        overall_pnls = [r.get("counterfactual_pnl", 0.0) for r in resolved]
        overall_verdict = "GATES_ADDING_VALUE" if float(np.mean(overall_pnls)) < 0 else "GATES_TOO_CONSERVATIVE"

        logger.info(
            "counterfactual_gate_report",
            label=label, n=len(resolved),
            overall=overall_verdict,
        )
        return {
            "label":            label,
            "n":                len(resolved),
            "overall_mean_pnl": round(float(np.mean(overall_pnls)), 4),
            "overall_verdict":  overall_verdict,
            "by_reason":        per_reason,
            "recommendation":   self._recommendation(verdicts),
        }

    def _recommendation(self, verdicts: dict[str, str]) -> str:
        """Generate a one-line recommendation based on gate verdicts."""
        too_conservative = [r for r, v in verdicts.items() if v == "GATE_TOO_CONSERVATIVE"]
        adding_value     = [r for r, v in verdicts.items() if v == "GATE_ADDING_VALUE"]
        parts = []
        if adding_value:
            parts.append(f"Gates correctly filtering: {', '.join(adding_value)}")
        if too_conservative:
            parts.append(f"Consider loosening: {', '.join(too_conservative)}")
        return " | ".join(parts) if parts else "Insufficient data"
