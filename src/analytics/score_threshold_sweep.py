"""
src.analytics.score_threshold_sweep — Phil-inspired min-edge threshold optimiser.

Problem
-------
The current model uses a fixed score split at 0.50: score < 0.50 = SHORT,
score > 0.50 = LONG.  This admits low-conviction signals (scores like 0.46 or
0.54) that are near-random and erode the book's win rate.

Solution (from Phil's core/score.py ``threshold_sweep``)
---------------------------------------------------------
Sweep ``min_conviction`` levels — the minimum distance from 0.50 required to
open a position.  At each level, simulate a flat portfolio that only enters
symbols whose score is outside the neutral band [0.5 - t, 0.5 + t]:

    SHORT if score < (0.50 - t)   →  high bearish conviction
    LONG  if score > (0.50 + t)   →  high bullish conviction
    FLAT  otherwise               →  no position

Key outputs per threshold:
    n_signals    — how many of 218 symbols clear the bar
    win_rate     — realized win rate across resolved outcomes
    brier_delta  — are filtered estimates better than the market's own price?
    net_sharpe   — portfolio Sharpe after 8.5bps NSE Futures cost

Evidence base
-------------
2026-09-28 live session:
    • All 4 SHORT misses (DRREDDY, ASIANPAINT, AXISBANK, MARUTI) had
      scores in [0.35, 0.49] — low conviction.
    • All 16 SHORT winners had scores in [0.05, 0.32] — high conviction.
    • Threshold 0.15 (only enter when score < 0.35 or > 0.65) would have
      eliminated all 4 misses while keeping all 16 winners.

Usage::

    from src.analytics.score_threshold_sweep import ScoreThresholdSweep

    sweep = ScoreThresholdSweep()
    results = sweep.run(scores, resolved_outcomes)
    optimal = sweep.optimal_threshold(results)

    # Apply optimal threshold to live scores
    filtered = sweep.apply(scores, optimal.threshold)

Requirements: Phil phase (core/score.py threshold_sweep), mandate §4.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)

# Threshold levels to sweep (distance from 0.50 neutral).
# 0.00 = enter everything; 0.30 = only enter when score < 0.20 or > 0.80.
DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
)

NEUTRAL_SCORE = 0.50
NSE_FUTURES_BPS = 8.5          # primary execution cost
COST_PER_POSITION = NSE_FUTURES_BPS / 10_000.0


@dataclass
class ThresholdResult:
    """Metrics at one min-conviction threshold level."""

    threshold: float            # min distance from 0.50 to open a position
    long_entry: float           # LONG if score > long_entry
    short_entry: float          # SHORT if score < short_entry
    n_signals: int              # symbols that cleared the bar
    n_long: int
    n_short: int
    win_rate: float             # realized win rate from resolved outcomes
    mean_net_pct: float         # mean net return (%)
    short_win_rate: float       # SHORT-only win rate
    long_win_rate: float        # LONG-only win rate
    brier_delta: float          # agent brier - market brier (negative = we beat market)
    estimated_annual_sharpe: float  # rough annualised Sharpe estimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "long_entry": round(self.long_entry, 4),
            "short_entry": round(self.short_entry, 4),
            "n_signals": self.n_signals,
            "n_long": self.n_long,
            "n_short": self.n_short,
            "win_rate": round(self.win_rate, 4),
            "mean_net_pct": round(self.mean_net_pct, 4),
            "short_win_rate": round(self.short_win_rate, 4),
            "long_win_rate": round(self.long_win_rate, 4),
            "brier_delta": round(self.brier_delta, 6),
            "estimated_annual_sharpe": round(self.estimated_annual_sharpe, 4),
        }


@dataclass
class SweepReport:
    """Full sweep report across all thresholds."""

    date: str
    n_outcomes_used: int
    results: list[ThresholdResult] = field(default_factory=list)
    optimal_threshold: float = 0.00
    optimal_win_rate: float = 0.00
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "n_outcomes_used": self.n_outcomes_used,
            "results": [r.to_dict() for r in self.results],
            "optimal_threshold": self.optimal_threshold,
            "optimal_win_rate": round(self.optimal_win_rate, 4),
            "reasoning": self.reasoning,
        }


class ScoreThresholdSweep:
    """
    Sweeps minimum conviction thresholds over historical outcomes and identifies
    the level that maximises risk-adjusted returns with the highest sample size.

    Phil's ``threshold_sweep`` uses a fixed flat stake across all resolved
    forecasts.  This implementation adapts that for the NSE cross-sectional
    universe: instead of $ stake, it uses direction-adjusted % returns and
    estimates a Sharpe using observed mean and std of net returns.

    The sweep is applied AFTER scoring — it is a portfolio construction filter,
    not a feature or model change.  It requires no retraining.
    """

    def __init__(
        self,
        thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
        min_signals: int = 10,
        cost_bps: float = NSE_FUTURES_BPS,
    ) -> None:
        self.thresholds = thresholds
        self.min_signals = min_signals
        self.cost_per_pos = cost_bps / 10_000.0

    def _apply_threshold(
        self, scores: list[dict], t: float
    ) -> list[dict]:
        """Filter scores to only those outside the neutral band."""
        long_entry  = NEUTRAL_SCORE + t
        short_entry = NEUTRAL_SCORE - t
        kept = []
        for s in scores:
            score = s.get("score", NEUTRAL_SCORE)
            if score < short_entry or score > long_entry:
                kept.append(s)
        return kept

    def _compute_metrics(
        self,
        filtered_scores: list[dict],
        resolved: dict[str, dict],
    ) -> tuple[float, float, float, float, float]:
        """
        Compute win_rate, mean_net_pct, short_win_rate, long_win_rate, brier_delta
        for the filtered set.

        ``resolved`` is a dict of symbol → {net_pct, direction, realized}.
        """
        matched = [
            {**s, **resolved[s["symbol"]]}
            for s in filtered_scores
            if s["symbol"] in resolved
        ]
        if not matched:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        wins = [m for m in matched if m["net_pct"] > 0]
        shorts = [m for m in matched if m["direction"] == -1]
        longs  = [m for m in matched if m["direction"] == 1]
        short_wins = [m for m in shorts if m["net_pct"] > 0]
        long_wins  = [m for m in longs  if m["net_pct"] > 0]

        win_rate   = len(wins) / len(matched)
        mean_net   = float(np.mean([m["net_pct"] for m in matched]))
        s_win_rate = len(short_wins) / len(shorts) if shorts else 0.0
        l_win_rate = len(long_wins)  / len(longs)  if longs  else 0.0

        # Brier delta: is our score direction a better probability than 0.50 market prior?
        brier_agent  = float(np.mean(
            [(m["score"] - m.get("realized", 0.5)) ** 2 for m in matched]
        ))
        brier_market = float(np.mean(
            [(NEUTRAL_SCORE - m.get("realized", 0.5)) ** 2 for m in matched]
        ))
        brier_delta = brier_agent - brier_market

        return win_rate, mean_net, s_win_rate, l_win_rate, brier_delta

    def _estimate_sharpe(self, filtered_scores: list[dict], resolved: dict) -> float:
        """Rough annualised Sharpe from observed mean/std of net returns."""
        matched_returns = [
            resolved[s["symbol"]]["net_pct"]
            for s in filtered_scores
            if s["symbol"] in resolved
        ]
        if len(matched_returns) < 5:
            return 0.0
        arr = np.array(matched_returns)
        mean_, std_ = float(np.mean(arr)), float(np.std(arr))
        if std_ < 1e-9:
            return 0.0
        # Annualise: ~52 weekly rebalances, each gives the observed distribution
        return round(float(mean_ / std_ * np.sqrt(52)), 4)

    def run(
        self,
        scores: list[dict],
        resolved_outcomes: list[dict],
        date: str = "",
    ) -> SweepReport:
        """
        Run the full threshold sweep.

        Args:
            scores:             List of {symbol, score, direction} dicts from score_all().
            resolved_outcomes:  List of {symbol, net_pct, direction, realized} dicts.
                                ``realized`` = 1.0 if signal was correct, 0.0 otherwise.
                                ``net_pct``   = realized net return (%).
            date:               Session date string (for the report).

        Returns:
            SweepReport with results per threshold + optimal recommendation.
        """
        # Build resolved lookup
        resolved: dict[str, dict] = {}
        for o in resolved_outcomes:
            sym = o.get("symbol", "")
            if sym:
                # realized = 1.0 if direction was correct and net_pct > 0
                net = float(o.get("net_pct", 0.0))
                resolved[sym] = {
                    "net_pct":  net,
                    "direction": int(o.get("direction", 0)),
                    "realized":  1.0 if net > 0 else 0.0,
                }

        report = SweepReport(date=date, n_outcomes_used=len(resolved))

        for t in self.thresholds:
            filtered = self._apply_threshold(scores, t)
            n = len(filtered)
            n_long  = sum(1 for s in filtered if s["direction"] == 1)
            n_short = sum(1 for s in filtered if s["direction"] == -1)

            if n < self.min_signals:
                # Threshold is too strict — not enough signals
                result = ThresholdResult(
                    threshold=t,
                    long_entry=NEUTRAL_SCORE + t,
                    short_entry=NEUTRAL_SCORE - t,
                    n_signals=n, n_long=n_long, n_short=n_short,
                    win_rate=0.0, mean_net_pct=0.0,
                    short_win_rate=0.0, long_win_rate=0.0,
                    brier_delta=0.0, estimated_annual_sharpe=0.0,
                )
                report.results.append(result)
                continue

            win_rate, mean_net, s_win, l_win, brier = self._compute_metrics(
                filtered, resolved
            )
            sharpe = self._estimate_sharpe(filtered, resolved)

            result = ThresholdResult(
                threshold=t,
                long_entry=NEUTRAL_SCORE + t,
                short_entry=NEUTRAL_SCORE - t,
                n_signals=n, n_long=n_long, n_short=n_short,
                win_rate=win_rate, mean_net_pct=mean_net,
                short_win_rate=s_win, long_win_rate=l_win,
                brier_delta=brier,
                estimated_annual_sharpe=sharpe,
            )
            report.results.append(result)
            logger.debug(
                "score_threshold_sweep_result",
                threshold=t, n_signals=n, win_rate=round(win_rate, 3),
                brier_delta=round(brier, 6),
            )

        # Select optimal: highest brier_delta-adjusted win_rate with enough signals
        valid = [
            r for r in report.results
            if r.n_signals >= self.min_signals and r.win_rate > 0
        ]
        if valid:
            # Maximise: win_rate (primary) + small penalty for signal count reduction
            best = max(
                valid,
                key=lambda r: r.win_rate - 0.01 * (1 - r.n_signals / max(len(scores), 1)),
            )
            report.optimal_threshold = best.threshold
            report.optimal_win_rate  = best.win_rate
            report.reasoning = (
                f"threshold={best.threshold:.2f} → "
                f"win_rate={best.win_rate:.1%} | "
                f"n_signals={best.n_signals}/{len(scores)} | "
                f"brier_delta={best.brier_delta:+.4f} | "
                f"sharpe_est={best.estimated_annual_sharpe:+.3f}"
            )

        logger.info(
            "score_threshold_sweep_complete",
            n_thresholds=len(self.thresholds),
            optimal=report.optimal_threshold,
            optimal_win_rate=round(report.optimal_win_rate, 3),
        )
        return report

    def apply(
        self,
        scores: list[dict],
        threshold: float,
    ) -> list[dict]:
        """
        Apply a threshold to live scores, returning only high-conviction signals.
        Signals in the neutral band are replaced with direction=0 (FLAT/no trade).
        """
        long_entry  = NEUTRAL_SCORE + threshold
        short_entry = NEUTRAL_SCORE - threshold
        result = []
        for s in scores:
            score = s.get("score", NEUTRAL_SCORE)
            direction = 1 if score > long_entry else (-1 if score < short_entry else 0)
            result.append({**s, "direction": direction, "threshold_applied": threshold})
        n_active = sum(1 for r in result if r["direction"] != 0)
        logger.info(
            "score_threshold_applied",
            threshold=threshold, n_active=n_active, n_total=len(scores),
        )
        return result

    def save_report(self, report: SweepReport, path: Path) -> None:
        """Persist the sweep report as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("score_threshold_sweep_saved", path=str(path))
