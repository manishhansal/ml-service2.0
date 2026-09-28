"""
src/risk/drawdown_manager.py — Drawdown-Aware Behavior Engine

FIX NEW-P2-005: The mandate (§32) requires NORMAL/CAUTION/DEFENSIVE/HALTED states
based on rolling drawdown. Without this, the system continues trading at full
confidence during sustained losses — the "revenge trading" failure mode.

State machine:
  NORMAL      : drawdown < caution_threshold
  CAUTION     : caution_threshold <= drawdown < defensive_threshold
  DEFENSIVE   : defensive_threshold <= drawdown < halt_threshold
  HALTED      : drawdown >= halt_threshold (no new trades)

Effects by state:
  NORMAL    : full signal confidence multiplier (1.0), all alphas active
  CAUTION   : reduced multiplier (0.75), marginal alphas suppressed
  DEFENSIVE : reduced multiplier (0.50), only high-conviction alphas
  HALTED    : multiplier 0.0, no new signals

The DrawdownManager integrates with MetaDecisionEngine by providing a
confidence_multiplier that scales the final signal confidence.

Usage::

    manager = DrawdownManager()
    manager.update_pnl(daily_returns=[0.01, -0.02, -0.03, -0.02])
    state = manager.current_state
    multiplier = manager.confidence_multiplier
    # MetaDecisionEngine multiplies final confidence by this value
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)


class DrawdownState(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    DEFENSIVE = "DEFENSIVE"
    HALTED = "HALTED"


# Default thresholds — must be configurable, not magic numbers
# These are starting values based on institutional risk practice;
# they should be validated against the strategy's historical drawdown distribution.
DEFAULT_CAUTION_DD = 0.03     # 3% drawdown → CAUTION
DEFAULT_DEFENSIVE_DD = 0.07   # 7% drawdown → DEFENSIVE
DEFAULT_HALT_DD = 0.12        # 12% drawdown → HALTED

# Confidence multipliers per state
_STATE_MULTIPLIERS = {
    DrawdownState.NORMAL:    1.00,
    DrawdownState.CAUTION:   0.75,
    DrawdownState.DEFENSIVE: 0.50,
    DrawdownState.HALTED:    0.00,
}

# Minimum edge required per state (higher = more selective trading)
_STATE_MIN_EDGE = {
    DrawdownState.NORMAL:    0.0,
    DrawdownState.CAUTION:   0.001,   # needs at least 0.1% expected edge
    DrawdownState.DEFENSIVE: 0.003,   # needs at least 0.3% expected edge
    DrawdownState.HALTED:    math.inf, # no trade allowed
}


@dataclass
class DrawdownSnapshot:
    """Point-in-time drawdown state snapshot."""
    timestamp: str
    current_drawdown: float
    peak_equity: float
    current_equity: float
    state: DrawdownState
    confidence_multiplier: float
    min_edge_required: float
    state_duration_bars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "current_drawdown": round(self.current_drawdown, 6),
            "peak_equity": round(self.peak_equity, 6),
            "current_equity": round(self.current_equity, 6),
            "state": self.state.value,
            "confidence_multiplier": self.confidence_multiplier,
            "min_edge_required": (
                round(self.min_edge_required, 6)
                if self.min_edge_required != math.inf else "INF"
            ),
            "state_duration_bars": self.state_duration_bars,
        }


class DrawdownManager:
    """
    Tracks rolling drawdown and emits risk state adjustments.

    Parameters
    ----------
    caution_dd    : drawdown fraction triggering CAUTION (default 3%)
    defensive_dd  : drawdown fraction triggering DEFENSIVE (default 7%)
    halt_dd       : drawdown fraction triggering HALTED (default 12%)
    recovery_bars : consecutive bars above peak required to return to NORMAL

    All thresholds MUST be set before any capital deployment and documented
    with their validation methodology per mandate §63.
    """

    def __init__(
        self,
        caution_dd: float = DEFAULT_CAUTION_DD,
        defensive_dd: float = DEFAULT_DEFENSIVE_DD,
        halt_dd: float = DEFAULT_HALT_DD,
        recovery_bars: int = 5,
    ) -> None:
        if not (0 < caution_dd < defensive_dd < halt_dd < 1.0):
            raise ValueError(
                f"Drawdown thresholds must satisfy 0 < caution < defensive < halt < 1. "
                f"Got: caution={caution_dd}, defensive={defensive_dd}, halt={halt_dd}"
            )
        self.caution_dd = caution_dd
        self.defensive_dd = defensive_dd
        self.halt_dd = halt_dd
        self.recovery_bars = recovery_bars

        self._equity: float = 1.0
        self._peak_equity: float = 1.0
        self._state: DrawdownState = DrawdownState.NORMAL
        self._state_entered_at: int = 0  # bar index when current state was entered
        self._bar_count: int = 0
        self._bars_above_peak: int = 0
        self._history: deque[DrawdownSnapshot] = deque(maxlen=500)

    # ── Core update ────────────────────────────────────────────────────────

    def update(self, return_: float) -> DrawdownSnapshot:
        """
        Update equity with one bar's return and recompute state.

        Args:
            return_: realized return for this bar (fraction, e.g. -0.005)

        Returns:
            DrawdownSnapshot with current state.
        """
        self._bar_count += 1
        self._equity *= (1.0 + return_)
        self._equity = max(0.0001, self._equity)  # floor at near-zero

        # Update peak
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity
            self._bars_above_peak = 0  # reset recovery counter (we're at new peak)
        else:
            # Track consecutive bars at or above peak for recovery logic
            if abs(self._equity - self._peak_equity) < 1e-9:
                self._bars_above_peak += 1
            else:
                self._bars_above_peak = 0

        current_dd = (self._peak_equity - self._equity) / self._peak_equity
        old_state = self._state
        new_state = self._compute_state(current_dd)

        if new_state != old_state:
            self._state = new_state
            self._state_entered_at = self._bar_count
            logger.warning(
                "drawdown_state_transition",
                from_state=old_state.value,
                to_state=new_state.value,
                drawdown=round(current_dd, 4),
                equity=round(self._equity, 6),
                bar=self._bar_count,
            )

        snap = DrawdownSnapshot(
            timestamp=datetime.now(tz=UTC).isoformat(),
            current_drawdown=current_dd,
            peak_equity=self._peak_equity,
            current_equity=self._equity,
            state=self._state,
            confidence_multiplier=self.confidence_multiplier,
            min_edge_required=self.min_edge_required,
            state_duration_bars=self._bar_count - self._state_entered_at,
        )
        self._history.append(snap)
        return snap

    def update_pnl(self, daily_returns: list[float]) -> DrawdownSnapshot:
        """Convenience: update with a list of returns. Returns final snapshot."""
        snap = DrawdownSnapshot(
            timestamp=datetime.now(tz=UTC).isoformat(),
            current_drawdown=0.0, peak_equity=1.0, current_equity=1.0,
            state=DrawdownState.NORMAL, confidence_multiplier=1.0,
            min_edge_required=0.0, state_duration_bars=0,
        )
        for r in daily_returns:
            snap = self.update(r)
        return snap

    # ── State logic ────────────────────────────────────────────────────────

    def _compute_state(self, drawdown: float) -> DrawdownState:
        """Determine target state from drawdown. Recovery hysteresis: must hit
        recovery_bars above peak before returning from CAUTION/DEFENSIVE/HALTED."""
        if drawdown >= self.halt_dd:
            return DrawdownState.HALTED
        if drawdown >= self.defensive_dd:
            return DrawdownState.DEFENSIVE
        if drawdown >= self.caution_dd:
            return DrawdownState.CAUTION
        # In normal territory — but only recover fully if enough bars at/above peak
        if self._state == DrawdownState.NORMAL:
            return DrawdownState.NORMAL
        # Recovery: need recovery_bars consecutive bars at/above peak
        if self._bars_above_peak >= self.recovery_bars:
            return DrawdownState.NORMAL
        # Still recovering — stay in current degraded state
        return self._state

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def current_state(self) -> DrawdownState:
        return self._state

    @property
    def current_drawdown(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self._equity) / self._peak_equity

    @property
    def confidence_multiplier(self) -> float:
        """Multiply signal confidence by this factor in the current state."""
        return _STATE_MULTIPLIERS[self._state]

    @property
    def min_edge_required(self) -> float:
        """Minimum expected net edge required to generate a signal."""
        return _STATE_MIN_EDGE[self._state]

    @property
    def is_halted(self) -> bool:
        return self._state == DrawdownState.HALTED

    @property
    def allows_new_trades(self) -> bool:
        return self._state != DrawdownState.HALTED

    def summary(self) -> dict[str, Any]:
        return {
            "current_state": self._state.value,
            "current_drawdown": round(self.current_drawdown, 6),
            "peak_equity": round(self._peak_equity, 6),
            "current_equity": round(self._equity, 6),
            "confidence_multiplier": self.confidence_multiplier,
            "min_edge_required": (
                self.min_edge_required
                if self.min_edge_required != math.inf else "INF"
            ),
            "bars_above_peak": self._bars_above_peak,
            "state_duration_bars": self._bar_count - self._state_entered_at,
            "thresholds": {
                "caution_dd": self.caution_dd,
                "defensive_dd": self.defensive_dd,
                "halt_dd": self.halt_dd,
            },
        }
