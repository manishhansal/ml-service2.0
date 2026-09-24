"""
src.backtest.position_sizing — position sizing + portfolio risk limits (Phase L).

Implements fractional-Kelly sizing bounded by hard risk caps. Model confidence
can NEVER bypass the hard limits (Phase 74): the final size is the minimum of
the Kelly suggestion and every applicable cap.

Caps:
    - max single-position weight
    - max gross portfolio exposure
    - max sector concentration
    - volatility target scaling
    - drawdown throttle (reduce size as drawdown deepens)

Requirements: Phase L, Phase 74, Phase 75, 15_RISK_SPECIFICATION.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RiskLimits:
    """Hard portfolio risk limits — never exceeded regardless of confidence."""

    max_position_weight: float = 0.10     # 10% of capital per position
    max_gross_exposure: float = 1.0       # 100% gross
    max_sector_weight: float = 0.30       # 30% per sector
    kelly_fraction: float = 0.25          # quarter-Kelly (conservative)
    vol_target: float = 0.15              # annualized target vol
    max_drawdown_throttle: float = 0.20   # start throttling past 20% dd


def fractional_kelly(
    prob_win: float,
    win_return: float,
    loss_return: float,
    kelly_fraction: float = 0.25,
) -> float:
    """
    Fractional Kelly fraction for a binary bet.

    Kelly f* = p/|loss| - (1-p)/win  (edge/odds form). We use the standard
    f* = (b*p - q) / b where b = win/|loss| payoff ratio, q = 1-p.
    Clamped to [0, 1] then scaled by kelly_fraction. Negative edge → 0.
    """
    p = max(0.0, min(1.0, prob_win))
    q = 1.0 - p
    b = abs(win_return) / abs(loss_return) if loss_return != 0 else 0.0
    if b <= 0:
        return 0.0
    f_star = (b * p - q) / b
    f_star = max(0.0, min(1.0, f_star))
    return f_star * kelly_fraction


@dataclass
class SizingResult:
    """Result of a position-sizing decision with the binding constraint."""

    weight: float
    kelly_weight: float
    binding_constraint: str
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class PositionSizer:
    """
    Computes a position weight from a probabilistic edge, bounded by hard caps.

    Usage::

        sizer = PositionSizer(RiskLimits())
        result = sizer.size(
            prob_target=0.58, win_return=0.02, loss_return=-0.015,
            current_gross=0.4, current_sector_weight=0.1, current_drawdown=-0.05,
        )
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def size(
        self,
        prob_target: float,
        win_return: float,
        loss_return: float,
        current_gross: float = 0.0,
        current_sector_weight: float = 0.0,
        current_drawdown: float = 0.0,
        realized_vol: float | None = None,
    ) -> SizingResult:
        lim = self.limits
        reasons: list[str] = []

        # 1. Fractional Kelly base size.
        kelly_w = fractional_kelly(prob_target, win_return, loss_return, lim.kelly_fraction)
        weight = kelly_w
        binding = "KELLY"

        if kelly_w <= 0.0:
            return SizingResult(0.0, 0.0, "NO_EDGE", ["NEGATIVE_OR_ZERO_EDGE"])

        # 2. Volatility targeting (scale down if realized vol exceeds target).
        if realized_vol is not None and realized_vol > 0:
            vol_scalar = min(1.0, lim.vol_target / realized_vol)
            if vol_scalar < 1.0:
                weight *= vol_scalar
                binding = "VOL_TARGET"
                reasons.append("VOL_SCALED")

        # 3. Drawdown throttle (linear reduction as drawdown deepens).
        dd = abs(current_drawdown)
        if dd > 0 and lim.max_drawdown_throttle > 0:
            throttle = max(0.0, 1.0 - dd / lim.max_drawdown_throttle)
            if throttle < 1.0:
                weight *= throttle
                binding = "DRAWDOWN_THROTTLE"
                reasons.append("DRAWDOWN_THROTTLED")

        # 4. Single-position cap (HARD).
        if weight > lim.max_position_weight:
            weight = lim.max_position_weight
            binding = "MAX_POSITION_WEIGHT"
            reasons.append("POSITION_CAP")

        # 5. Gross exposure cap (HARD).
        headroom_gross = max(0.0, lim.max_gross_exposure - current_gross)
        if weight > headroom_gross:
            weight = headroom_gross
            binding = "MAX_GROSS_EXPOSURE"
            reasons.append("GROSS_CAP")

        # 6. Sector concentration cap (HARD).
        headroom_sector = max(0.0, lim.max_sector_weight - current_sector_weight)
        if weight > headroom_sector:
            weight = headroom_sector
            binding = "MAX_SECTOR_WEIGHT"
            reasons.append("SECTOR_CAP")

        return SizingResult(
            weight=round(max(0.0, weight), 6),
            kelly_weight=round(kelly_w, 6),
            binding_constraint=binding,
            reason_codes=reasons,
        )


@dataclass
class PortfolioRiskState:
    """Aggregate portfolio risk snapshot for gating new positions (Phase 75)."""

    n_positions: int = 0
    gross_exposure: float = 0.0
    sector_weights: dict[str, float] = field(default_factory=dict)
    current_drawdown: float = 0.0

    def can_add(self, limits: RiskLimits, max_positions: int = 10) -> tuple[bool, str]:
        if self.n_positions >= max_positions:
            return False, "MAX_POSITIONS_REACHED"
        if self.gross_exposure >= limits.max_gross_exposure:
            return False, "GROSS_EXPOSURE_FULL"
        if abs(self.current_drawdown) >= limits.max_drawdown_throttle * 2:
            return False, "DRAWDOWN_STOP"
        return True, "OK"
