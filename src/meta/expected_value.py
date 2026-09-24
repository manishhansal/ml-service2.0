"""
src.meta.expected_value — explicit expected-net-edge calculation (Phase 40).

The decision engine must reject any signal whose expected net edge is not
positive AFTER costs. Confidence alone is NOT sufficient to trade.

Expected net edge (per unit capital) for a directional trade:

    E[net] = P(target) * target_return
             - P(stop)  * |stop_return|
             - E[cost]
             - E[slippage]

where P(target) and P(stop) are CALIBRATED probabilities (not raw scores),
and P(neither) = 1 - P(target) - P(stop) is treated as a small residual
carry (approximated as time-decay ~ 0 for intraday horizons).

An uncertainty penalty (proportional to epistemic uncertainty) is subtracted
so that low-conviction edges are discounted.

Requirements: Phase 40, Phase 41, 16_DECISION_ENGINE_SPEC.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExpectedValueResult:
    """Expected-value breakdown for a candidate trade."""

    expected_gross_edge: float
    expected_cost: float
    expected_net_edge: float
    prob_target: float
    prob_stop: float
    prob_neither: float
    uncertainty_penalty: float
    is_viable: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def compute_expected_value(
    prob_target: float,
    prob_stop: float,
    target_return: float,
    stop_return: float,
    cost_bps: float = 10.0,
    slippage_bps: float = 2.0,
    uncertainty: float = 0.0,
    uncertainty_weight: float = 0.5,
    min_edge: float = 0.0,
) -> ExpectedValueResult:
    """
    Compute expected net edge and decide viability.

    Args:
        prob_target:   CALIBRATED P(target hit) in [0,1].
        prob_stop:     CALIBRATED P(stop hit) in [0,1].
        target_return: fractional return if target hits (e.g. 0.02).
        stop_return:   fractional loss if stop hits (negative, e.g. -0.015).
        cost_bps:      round-trip transaction cost in basis points.
        slippage_bps:  expected slippage in basis points.
        uncertainty:   epistemic uncertainty in [0,1] (discounts the edge).
        uncertainty_weight: how strongly uncertainty discounts the raw edge.
        min_edge:      minimum acceptable net edge (default 0 → must be positive).

    Returns:
        ExpectedValueResult with is_viable=True only when expected_net_edge > min_edge.
    """
    p_t = max(0.0, min(1.0, prob_target))
    p_s = max(0.0, min(1.0, prob_stop))
    # Enforce the probability invariant: P(target)+P(stop) <= 1.
    if p_t + p_s > 1.0:
        total = p_t + p_s
        p_t, p_s = p_t / total, p_s / total
    p_n = max(0.0, 1.0 - p_t - p_s)

    tgt = abs(target_return)
    stp = abs(stop_return)

    gross = p_t * tgt - p_s * stp
    cost = (cost_bps + slippage_bps) / 10_000.0
    penalty = uncertainty_weight * uncertainty * (tgt + stp) / 2.0

    net = gross - cost - penalty

    viable = net > min_edge
    reason = "POSITIVE_NET_EDGE" if viable else "INSUFFICIENT_EDGE"

    return ExpectedValueResult(
        expected_gross_edge=round(gross, 6),
        expected_cost=round(cost, 6),
        expected_net_edge=round(net, 6),
        prob_target=round(p_t, 4),
        prob_stop=round(p_s, 4),
        prob_neither=round(p_n, 4),
        uncertainty_penalty=round(penalty, 6),
        is_viable=viable,
        reason=reason,
    )
