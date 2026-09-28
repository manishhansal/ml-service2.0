"""
src.analytics.forecast_ledger — Phil-inspired full-universe forecast logging.

Problem
-------
ml-service2.0 currently scores 218 NSE F&O symbols every 5 minutes but only
tracks P&L for the 26 forward-paper positions.  The remaining 192 signals are
discarded.  This creates a calibration desert:

    Current calibration data:  26 × 52 weeks = 1,352 data points / year
    With ForecastLedger:      218 × 252 days  = 54,936 data points / year  (40× more)

Solution (from Phil's core/forecast.py)
----------------------------------------
ForecastLedger records ALL 218 estimates after every scoring run — even for
symbols with no forward-paper position.  At end of day (or when called with
exit prices), it resolves every open forecast and computes:

    brier_delta = brier_agent - brier_market
                = (est_prob - realized)² - (market_prior - realized)²

Negative brier_delta means the model's estimate is a BETTER probability
estimate than the market's own price.  This is the honest measure of alpha.

One live forecast per symbol per day: a re-score of the same symbol on the same
day overwrites the live forecast (no flood of correlated rows).

Key files:
    artifacts/forward_paper/forecasts.jsonl  — append-only ledger of all forecasts
    artifacts/forward_paper/forecast_sweep.json — daily sweep report (optimal threshold)

Usage::

    ledger = ForecastLedger()

    # After every score_all() call in autorun_till_close.py:
    ledger.record_session(scores, session_date="2026-09-28", nifty_chg=-1.52)

    # At end of day with realized returns:
    ledger.resolve_session(
        session_date="2026-09-28",
        realized_returns={sym: pct for sym, pct in final_returns.items()},
    )

    # Brier delta report:
    report = ledger.brier_report(session_date="2026-09-28")

Requirements: Phil adaptation, mandate §4 (one canonical pipeline).
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

LEDGER_PATH   = Path("artifacts/forward_paper/forecasts.jsonl")
MARKET_PRIOR  = 0.50   # neutral probability — "market doesn't know" baseline


class ForecastLedger:
    """
    Phil's forecast.py adapted for NSE cross-sectional signals.

    Key differences from Phil:
    - No Polymarket API — market prior is fixed at 0.50 (neutral cross-section).
    - Forecast = LightGBM probability score (0.0 = strong SHORT, 1.0 = strong LONG).
    - Realized = 1.0 if net_pct > 0 (signal was directionally correct).
    - One record per (symbol, session_date) — re-scored same day overwrites.
    - No per-record stake; this is a calibration ledger, not a paper trading ledger.
    """

    def __init__(self, ledger_path: Path = LEDGER_PATH) -> None:
        self._path = Path(ledger_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Write ─────────────────────────────────────────────────────────────────

    def record_session(
        self,
        scores: list[dict],
        session_date: str,
        nifty_chg: float = 0.0,
        model_version: str = "",
    ) -> int:
        """
        Record ALL scored symbols as open forecasts for this session.

        One live record per (symbol, session_date).  A second call on the same
        date supersedes the previous record for that symbol (same as Phil's
        --supersede: the old row stays but status="superseded").

        Args:
            scores:        Output of autorun_till_close.score_all() — list of
                           {symbol, score, direction, data_date}.
            session_date:  "YYYY-MM-DD" trading date.
            nifty_chg:     NIFTY intraday change % at time of scoring.
            model_version: Model artifact version string.

        Returns:
            Number of records written.
        """
        existing = self._load_open(session_date)
        ts = datetime.now(tz=timezone.utc).isoformat()
        written = 0

        with self._path.open("a") as fh:
            for s in scores:
                sym   = s.get("symbol", "")
                score = float(s.get("score", MARKET_PRIOR))
                direction = int(s.get("direction", 0))
                if not sym:
                    continue

                # Supersede existing open record for this (symbol, date)
                old_id = existing.get(sym)
                if old_id:
                    self._mark_superseded(old_id, ts)

                row = {
                    "id":            uuid.uuid4().hex[:12],
                    "ts":            ts,
                    "session_date":  session_date,
                    "symbol":        sym,
                    "score":         round(score, 6),
                    "direction":     direction,
                    "est_prob":      round(score, 6),
                    "market_prior":  MARKET_PRIOR,
                    "data_date":     s.get("data_date", ""),
                    "nifty_chg_at_record": round(nifty_chg, 4),
                    "model_version": model_version,
                    "status":        "open",
                    "supersedes":    old_id,
                }
                fh.write(json.dumps(row) + "\n")
                written += 1

        logger.info(
            "forecast_ledger_session_recorded",
            session_date=session_date, n_written=written,
            n_superseded=len(existing),
        )
        return written

    def resolve_session(
        self,
        session_date: str,
        realized_returns: dict[str, float],
    ) -> dict[str, Any]:
        """
        Resolve all open forecasts for session_date using realized end-of-day returns.

        Realized = 1.0 if signal was directionally correct AND net return > 0.
        Also computes and returns brier_delta for the session.

        Args:
            session_date:     "YYYY-MM-DD"
            realized_returns: {symbol: net_pct} — net % return after costs.

        Returns:
            dict with brier_delta, win_rate, n_resolved, by_direction.
        """
        rows = self._load_all()
        resolve_ts = datetime.now(tz=timezone.utc).isoformat()
        updated: list[dict] = []
        resolved_count = 0

        for row in rows:
            if row.get("session_date") != session_date:
                updated.append(row)
                continue
            if row.get("status") != "open":
                updated.append(row)
                continue
            sym = row["symbol"]
            if sym not in realized_returns:
                updated.append(row)
                continue

            net_pct   = float(realized_returns[sym])
            direction = int(row.get("direction", 0))
            realized  = 1.0 if net_pct > 0 else 0.0
            # For SHORT positions, direction=-1: realized=1.0 means price fell = SHORT won
            # For LONG positions, direction=+1: realized=1.0 means price rose = LONG won
            # net_pct already accounts for direction, so net_pct > 0 = correct direction

            row["status"]         = "won" if net_pct > 0 else "lost"
            row["realized"]       = realized
            row["net_pct"]        = round(net_pct, 4)
            row["resolved_at"]    = resolve_ts
            row["brier_agent"]    = round((row["est_prob"] - realized) ** 2, 6)
            row["brier_market"]   = round((MARKET_PRIOR    - realized) ** 2, 6)
            row["brier_delta"]    = round(row["brier_agent"] - row["brier_market"], 6)
            updated.append(row)
            resolved_count += 1

        # Rewrite the ledger (compaction)
        self._path.write_text("".join(json.dumps(r) + "\n" for r in updated))

        # Compute session brier report
        resolved = [r for r in updated
                    if r.get("session_date") == session_date
                    and r.get("status") in ("won", "lost")]
        report = self._brier_stats(resolved, session_date)
        report["n_resolved"] = resolved_count

        logger.info(
            "forecast_ledger_session_resolved",
            session_date=session_date, n_resolved=resolved_count,
            brier_delta=round(report.get("brier_delta", 0), 6),
        )
        return report

    # ── Read ──────────────────────────────────────────────────────────────────

    def brier_report(
        self,
        session_date: str | None = None,
        last_n_sessions: int | None = None,
    ) -> dict[str, Any]:
        """
        Compute brier_delta report — the honest measure of alpha.

        Args:
            session_date:      Restrict to one date (or None for all).
            last_n_sessions:   Restrict to last N sessions.

        Returns:
            {brier_delta, win_rate, n, by_symbol, by_direction, ...}
        """
        rows = self._load_all()
        settled = [r for r in rows if r.get("status") in ("won", "lost")
                   and not r.get("superseded_by")]

        if session_date:
            settled = [r for r in settled if r.get("session_date") == session_date]
        elif last_n_sessions:
            dates = sorted({r["session_date"] for r in settled}, reverse=True)
            keep_dates = set(dates[:last_n_sessions])
            settled = [r for r in settled if r.get("session_date") in keep_dates]

        return self._brier_stats(settled, session_date or "all")

    def calibration_table(self, session_date: str | None = None) -> list[dict]:
        """
        Phil-style calibration table: est_prob buckets vs realized frequency.
        Each bucket covers a 0.10-wide band of estimated probability.
        """
        rows = self._load_all()
        settled = [r for r in rows if r.get("status") in ("won", "lost")
                   and not r.get("superseded_by")]
        if session_date:
            settled = [r for r in settled if r.get("session_date") == session_date]

        buckets: dict[int, list[dict]] = defaultdict(list)
        for r in settled:
            b = min(int(float(r.get("est_prob", 0.5)) * 10), 9)
            buckets[b].append(r)

        table = []
        for b in sorted(buckets):
            rs = buckets[b]
            realized_freq = float(np.mean([r.get("realized", 0) for r in rs]))
            table.append({
                "est_range": f"{b/10:.1f}–{(b+1)/10:.1f}",
                "n": len(rs),
                "realized_freq": round(realized_freq, 3),
                "expected_freq": round((b + 0.5) / 10, 3),
                "calibration_error": round(realized_freq - (b + 0.5) / 10, 3),
            })
        return table

    def status(self) -> dict[str, Any]:
        """Summary: total open, settled, by session date."""
        rows = self._load_all()
        by_date: dict[str, dict] = defaultdict(lambda: {"open": 0, "won": 0, "lost": 0})
        for r in rows:
            d = r.get("session_date", "unknown")
            s = r.get("status", "open")
            by_date[d][s] = by_date[d].get(s, 0) + 1
        return {
            "total_records": len(rows),
            "open":          sum(1 for r in rows if r.get("status") == "open"),
            "settled":       sum(1 for r in rows if r.get("status") in ("won","lost")),
            "by_session":    dict(sorted(by_date.items())),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

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

    def _load_open(self, session_date: str) -> dict[str, str]:
        """Return {symbol: id} for open records on this session_date."""
        rows = self._load_all()
        return {
            r["symbol"]: r["id"]
            for r in rows
            if r.get("session_date") == session_date
            and r.get("status") == "open"
            and not r.get("superseded_by")
        }

    def _mark_superseded(self, old_id: str, ts: str) -> None:
        """Mark an existing record as superseded (in-place rewrite)."""
        rows = self._load_all()
        changed = False
        for r in rows:
            if r.get("id") == old_id and r.get("status") == "open":
                r["status"]         = "superseded"
                r["superseded_at"]  = ts
                changed = True
                break
        if changed:
            self._path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def _brier_stats(self, rows: list[dict], label: str) -> dict[str, Any]:
        if not rows:
            return {"label": label, "n": 0, "brier_delta": 0.0, "win_rate": 0.0}

        n   = len(rows)
        ba  = float(np.mean([(float(r.get("est_prob", 0.5)) - float(r.get("realized", 0))) ** 2
                              for r in rows]))
        bm  = float(np.mean([(MARKET_PRIOR - float(r.get("realized", 0))) ** 2
                              for r in rows]))
        wins = sum(1 for r in rows if r.get("status") == "won")
        shorts = [r for r in rows if int(r.get("direction", 0)) == -1]
        longs  = [r for r in rows if int(r.get("direction", 0)) == 1]
        by_sym: dict[str, dict] = defaultdict(lambda: {"n": 0, "brier_delta": 0.0})
        for r in rows:
            sym = r.get("symbol", "?")
            by_sym[sym]["n"] += 1
            by_sym[sym]["brier_delta"] = round(
                by_sym[sym].get("brier_delta_sum", 0.0)
                + float(r.get("brier_delta", 0.0)), 6
            )
            by_sym[sym]["brier_delta_sum"] = by_sym[sym]["brier_delta"]

        # Top alpha symbols (most negative brier_delta = we beat market most)
        top_alpha = sorted(
            [(sym, d["brier_delta"] / d["n"]) for sym, d in by_sym.items() if d["n"] >= 3],
            key=lambda x: x[1],
        )[:10]

        return {
            "label":            label,
            "n":                n,
            "win_rate":         round(wins / n, 4),
            "brier_agent":      round(ba, 6),
            "brier_market":     round(bm, 6),
            "brier_delta":      round(ba - bm, 6),
            "beating_market":   ba < bm,
            "n_short":          len(shorts),
            "n_long":           len(longs),
            "short_win_rate":   round(sum(1 for r in shorts if r.get("status")=="won") / max(len(shorts),1), 4),
            "long_win_rate":    round(sum(1 for r in longs  if r.get("status")=="won") / max(len(longs),1),  4),
            "top_alpha_symbols": [{"symbol": s, "mean_brier_delta": round(d, 6)} for s, d in top_alpha],
        }
