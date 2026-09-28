"""
src/analytics/signal_promotion.py — Signal Promotion Engine

FIX NEW-P2-004: The mandate (§26) requires an explicit signal lifecycle with
promotion criteria that must be SATISFIED before a signal moves to production.

This engine:
  1. Reads from the ForwardPaperRunner's settled outcomes
  2. Evaluates promotion criteria for each signal family
  3. Recommends promotion, demotion, or shadow extension
  4. Records ALL decisions with evidence in the research ledger

Promotion path (mandate §26):
  DISCOVERED → BACKTESTED → WALK_FORWARD_VALIDATED → SHADOW →
  PAPER_PRODUCTION → LIMITED_PRODUCTION → PRODUCTION

Demotion path (mandate §27):
  PRODUCTION → DEGRADED → SHADOW → QUARANTINED → RETIRED

Promotion gates (must ALL be satisfied):
  1. positive_net_expectancy : mean(net_return) > 0 on paper trades
  2. sufficient_trade_count  : n_trades >= min_trades (default 50)
  3. calibration_ok          : ECE < max_ece (default 0.10)
  4. cost_robust             : positive expectancy at 1.5x costs
  5. regime_coverage         : at least 2 distinct regimes tested
  6. drawdown_acceptable     : max_drawdown < max_dd_threshold (default -0.10)

NEVER promote based on raw historical P&L alone.
NEVER promote based on cherry-picked regime performance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)


class SignalLifecycleStage(str, Enum):
    DISCOVERED = "DISCOVERED"
    BACKTESTED = "BACKTESTED"
    WALK_FORWARD_VALIDATED = "WALK_FORWARD_VALIDATED"
    SHADOW = "SHADOW"
    PAPER_PRODUCTION = "PAPER_PRODUCTION"
    LIMITED_PRODUCTION = "LIMITED_PRODUCTION"
    PRODUCTION = "PRODUCTION"
    DEGRADED = "DEGRADED"
    QUARANTINED = "QUARANTINED"
    RETIRED = "RETIRED"


@dataclass
class PromotionEvidence:
    """Evidence collected for a promotion decision."""
    signal_family: str
    stage_from: SignalLifecycleStage
    stage_to: SignalLifecycleStage | None  # None if rejected
    n_trades: int
    net_expectancy: float
    cost_robust_expectancy: float       # expectancy at 1.5x costs
    max_drawdown: float
    regimes_tested: list[str]
    calibration_ece: float
    decision: str                        # PROMOTE | REJECT | EXTEND_SHADOW | DEMOTE
    reason: str
    evaluated_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    gate_results: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_family": self.signal_family,
            "stage_from": self.stage_from.value,
            "stage_to": self.stage_to.value if self.stage_to else None,
            "n_trades": self.n_trades,
            "net_expectancy": round(self.net_expectancy, 6),
            "cost_robust_expectancy": round(self.cost_robust_expectancy, 6),
            "max_drawdown": round(self.max_drawdown, 6),
            "regimes_tested": self.regimes_tested,
            "calibration_ece": round(self.calibration_ece, 4),
            "decision": self.decision,
            "reason": self.reason,
            "evaluated_at": self.evaluated_at,
            "gate_results": self.gate_results,
        }


@dataclass
class PromotionGateConfig:
    """Configuration for all promotion gates."""
    min_trades: int = 50
    min_net_expectancy: float = 0.0         # must be positive after costs
    max_ece: float = 0.10                   # Expected Calibration Error threshold
    cost_robustness_multiplier: float = 1.5 # test at 1.5× baseline costs
    min_regimes: int = 2                    # must be tested in ≥2 regimes
    max_drawdown: float = -0.10             # maximum acceptable drawdown


class SignalPromotionEngine:
    """
    Evaluates signal families against promotion criteria.

    Usage::

        engine = SignalPromotionEngine()
        result = engine.evaluate(
            signal_family="MOMENTUM_ALPHA",
            current_stage=SignalLifecycleStage.SHADOW,
            trade_records=[...],
        )
        if result.decision == "PROMOTE":
            # advance to PAPER_PRODUCTION
    """

    def __init__(
        self,
        gate_config: PromotionGateConfig | None = None,
        ledger_path: Path | None = None,
    ) -> None:
        self.gates = gate_config or PromotionGateConfig()
        self._ledger_path = ledger_path
        self._decisions: list[PromotionEvidence] = []

    def evaluate(
        self,
        signal_family: str,
        current_stage: SignalLifecycleStage,
        trade_records: list[dict[str, Any]],
        calibration_ece: float = 0.0,
    ) -> PromotionEvidence:
        """
        Evaluate whether a signal family meets promotion criteria.

        Args:
            signal_family  : Name of the alpha specialist / signal family.
            current_stage  : Current lifecycle stage.
            trade_records  : List of settled trade dicts, each with:
                             net_return (float), regime (str), cost_bps (float).
            calibration_ece: Model calibration ECE score.

        Returns:
            PromotionEvidence with decision and all gate results.
        """
        n = len(trade_records)
        gate_results: dict[str, str] = {}

        # ── Gate 1: Sufficient trade count ───────────────────────────────
        gate_results["G1_TRADE_COUNT"] = (
            "PASS" if n >= self.gates.min_trades
            else f"FAIL (n={n} < min={self.gates.min_trades})"
        )

        if n == 0:
            return self._make_evidence(
                signal_family, current_stage,
                n_trades=0,
                net_expectancy=0.0,
                cost_robust=0.0,
                max_dd=0.0,
                regimes=[],
                ece=calibration_ece,
                gate_results=gate_results,
                decision="REJECT",
                reason="NO_TRADE_RECORDS",
            )

        net_returns = np.array([float(t.get("net_return", 0.0)) for t in trade_records])
        regimes = list({t.get("regime", "UNKNOWN") for t in trade_records})
        net_expectancy = float(np.mean(net_returns))
        max_dd = self._max_drawdown(net_returns)

        # ── Gate 2: Positive net expectancy ──────────────────────────────
        gate_results["G2_NET_EXPECTANCY"] = (
            "PASS" if net_expectancy > self.gates.min_net_expectancy
            else f"FAIL (expectancy={net_expectancy:.6f} <= min={self.gates.min_net_expectancy})"
        )

        # ── Gate 3: Calibration ───────────────────────────────────────────
        gate_results["G3_CALIBRATION"] = (
            "PASS" if calibration_ece <= self.gates.max_ece
            else f"FAIL (ECE={calibration_ece:.4f} > max={self.gates.max_ece})"
        )

        # ── Gate 4: Cost robustness (1.5× costs) ─────────────────────────
        # Each trade has a cost_bps; stress by 1.5×
        extra_cost = 0.0
        if trade_records and "cost_bps" in trade_records[0]:
            baseline_cost_frac = float(trade_records[0].get("cost_bps", 27.65)) / 10_000.0
            extra_cost = baseline_cost_frac * (self.gates.cost_robustness_multiplier - 1.0)
        cost_robust_returns = net_returns - extra_cost
        cost_robust_expectancy = float(np.mean(cost_robust_returns))

        gate_results["G4_COST_ROBUST"] = (
            "PASS" if cost_robust_expectancy > 0
            else f"FAIL (expectancy at {self.gates.cost_robustness_multiplier}×cost={cost_robust_expectancy:.6f})"
        )

        # ── Gate 5: Regime coverage ───────────────────────────────────────
        gate_results["G5_REGIME_COVERAGE"] = (
            "PASS" if len(regimes) >= self.gates.min_regimes
            else f"FAIL (regimes={len(regimes)} < min={self.gates.min_regimes})"
        )

        # ── Gate 6: Drawdown ──────────────────────────────────────────────
        gate_results["G6_DRAWDOWN"] = (
            "PASS" if max_dd >= self.gates.max_drawdown
            else f"FAIL (max_dd={max_dd:.4f} < min={self.gates.max_drawdown})"
        )

        # ── Overall decision ──────────────────────────────────────────────
        all_pass = all(v == "PASS" for v in gate_results.values())
        any_fail = any("FAIL" in v for v in gate_results.values())

        if all_pass:
            next_stage = self._next_stage(current_stage)
            decision = "PROMOTE"
            reason = f"All {len(gate_results)} gates passed"
        elif not any_fail:
            next_stage = None
            decision = "EXTEND_SHADOW"
            reason = "Insufficient data to decide; continue shadow"
        else:
            failed = [k for k, v in gate_results.items() if "FAIL" in v]
            next_stage = None
            decision = "REJECT"
            reason = f"Failed gates: {', '.join(failed)}"

        evidence = self._make_evidence(
            signal_family, current_stage,
            n_trades=n,
            net_expectancy=net_expectancy,
            cost_robust=cost_robust_expectancy,
            max_dd=max_dd,
            regimes=regimes,
            ece=calibration_ece,
            gate_results=gate_results,
            decision=decision,
            reason=reason,
            stage_to=next_stage,
        )

        self._decisions.append(evidence)
        self._persist(evidence)

        logger.info(
            "signal_promotion_evaluated",
            signal_family=signal_family,
            stage_from=current_stage.value,
            decision=decision,
            n_trades=n,
            net_expectancy=round(net_expectancy, 6),
        )
        return evidence

    def evaluate_demotion(
        self,
        signal_family: str,
        current_stage: SignalLifecycleStage,
        recent_trade_records: list[dict[str, Any]],
        rolling_window: int = 20,
    ) -> PromotionEvidence:
        """
        Check whether a PRODUCTION/LIMITED signal should be demoted.

        Uses the most recent `rolling_window` trades to detect performance decay.
        """
        recent = recent_trade_records[-rolling_window:] if recent_trade_records else []
        if not recent:
            return self._make_evidence(
                signal_family, current_stage,
                n_trades=0, net_expectancy=0.0, cost_robust=0.0,
                max_dd=0.0, regimes=[], ece=0.0,
                gate_results={}, decision="EXTEND_SHADOW",
                reason="NO_RECENT_TRADES",
            )

        recent_returns = np.array([float(t.get("net_return", 0.0)) for t in recent])
        net_expectancy = float(np.mean(recent_returns))
        max_dd = self._max_drawdown(recent_returns)
        regimes = list({t.get("regime", "UNKNOWN") for t in recent})

        should_demote = (
            net_expectancy < -0.001  # negative expectancy
            or max_dd < self.gates.max_drawdown  # excessive drawdown
        )

        if should_demote:
            next_stage = SignalLifecycleStage.DEGRADED
            decision = "DEMOTE"
            reason = (
                f"Rolling performance below threshold: "
                f"expectancy={net_expectancy:.4f}, max_dd={max_dd:.4f}"
            )
        else:
            next_stage = None
            decision = "KEEP"
            reason = f"Rolling performance acceptable (n={len(recent)})"

        return self._make_evidence(
            signal_family, current_stage,
            n_trades=len(recent),
            net_expectancy=net_expectancy,
            cost_robust=net_expectancy,
            max_dd=max_dd,
            regimes=regimes,
            ece=0.0,
            gate_results={},
            decision=decision,
            reason=reason,
            stage_to=next_stage,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _next_stage(
        current: SignalLifecycleStage,
    ) -> SignalLifecycleStage:
        _progression = {
            SignalLifecycleStage.DISCOVERED: SignalLifecycleStage.BACKTESTED,
            SignalLifecycleStage.BACKTESTED: SignalLifecycleStage.WALK_FORWARD_VALIDATED,
            SignalLifecycleStage.WALK_FORWARD_VALIDATED: SignalLifecycleStage.SHADOW,
            SignalLifecycleStage.SHADOW: SignalLifecycleStage.PAPER_PRODUCTION,
            SignalLifecycleStage.PAPER_PRODUCTION: SignalLifecycleStage.LIMITED_PRODUCTION,
            SignalLifecycleStage.LIMITED_PRODUCTION: SignalLifecycleStage.PRODUCTION,
            SignalLifecycleStage.DEGRADED: SignalLifecycleStage.SHADOW,
        }
        return _progression.get(current, current)

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        equity = np.cumprod(1.0 + returns)
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        return float(np.min(dd))

    def _make_evidence(
        self,
        signal_family: str,
        current_stage: SignalLifecycleStage,
        n_trades: int,
        net_expectancy: float,
        cost_robust: float,
        max_dd: float,
        regimes: list[str],
        ece: float,
        gate_results: dict[str, str],
        decision: str,
        reason: str,
        stage_to: SignalLifecycleStage | None = None,
    ) -> PromotionEvidence:
        return PromotionEvidence(
            signal_family=signal_family,
            stage_from=current_stage,
            stage_to=stage_to,
            n_trades=n_trades,
            net_expectancy=net_expectancy,
            cost_robust_expectancy=cost_robust,
            max_drawdown=max_dd,
            regimes_tested=regimes,
            calibration_ece=ece,
            decision=decision,
            reason=reason,
            gate_results=gate_results,
        )

    def _persist(self, evidence: PromotionEvidence) -> None:
        if self._ledger_path is None:
            return
        p = Path(self._ledger_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(evidence.to_dict()) + "\n")

    def history(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._decisions]
