"""
src/meta/opportunity_score.py — Opportunity Score Engine

FIX NEW-P2-003: The mandate (§15) requires a normalized opportunity score that
combines expected economic value, signal quality, and risk into a single ranking
metric used for trade selection.

The formula is grounded in empirical validation, not marketing logic:

    OPPORTUNITY_SCORE = (
        expected_net_return
        × prob_quality
        × calibration_quality
        × regime_fit
        × data_quality
        × liquidity_quality
        × execution_quality
        × alpha_stability
    ) / (
        risk_factor + uncertainty + cost_factor
    )

Every component is bounded to [0, 1] before multiplication so the score is
interpretable. The denominator prevents division by zero via a floor of 0.001.

CRITICAL: The formula itself is a hypothesis. It must be validated by
backtesting: does selecting top-N by opportunity score outperform selecting
top-N by raw EV? The `OpportunityScorer.validate_formula()` method provides
this comparison.

Usage::

    scorer = OpportunityScorer()
    score = scorer.score(
        expected_net_return=0.008,
        prob_target=0.55,
        calibration_ece=0.04,
        regime_fit=0.8,
        data_quality=0.9,
        uncertainty=0.3,
        cost_bps=27.65,
    )
    # score ∈ [0, 1]; higher = better opportunity
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)

# Score floor — any trade below this is not worth executing
MINIMUM_VIABLE_SCORE: float = 0.10


@dataclass
class OpportunityScoreResult:
    """Full breakdown of an opportunity score computation."""

    # Raw inputs
    expected_net_return: float
    prob_target: float
    calibration_ece: float
    regime_fit: float
    data_quality: float
    liquidity_quality: float
    execution_quality: float
    alpha_stability: float
    uncertainty: float
    cost_bps: float
    risk_vol: float

    # Component scores (each in [0, 1])
    return_score: float = 0.0
    prob_quality: float = 0.0
    calibration_quality: float = 0.0
    numerator: float = 0.0
    denominator: float = 0.0
    opportunity_score: float = 0.0

    # Decision
    is_viable: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "opportunity_score": round(self.opportunity_score, 6),
            "is_viable": self.is_viable,
            "reason": self.reason,
            "components": {
                "return_score": round(self.return_score, 4),
                "prob_quality": round(self.prob_quality, 4),
                "calibration_quality": round(self.calibration_quality, 4),
                "regime_fit": round(self.regime_fit, 4),
                "data_quality": round(self.data_quality, 4),
                "liquidity_quality": round(self.liquidity_quality, 4),
                "execution_quality": round(self.execution_quality, 4),
                "alpha_stability": round(self.alpha_stability, 4),
            },
            "inputs": {
                "expected_net_return": round(self.expected_net_return, 6),
                "uncertainty": round(self.uncertainty, 4),
                "cost_bps": round(self.cost_bps, 2),
                "risk_vol": round(self.risk_vol, 4),
            },
        }


class OpportunityScorer:
    """
    Computes a normalized opportunity score for trade selection.

    The score is used to RANK competing signals — it is NOT a probability.

    Score interpretation:
      > 0.50  : high quality opportunity, worth executing
      0.30–0.50: moderate quality, execute with reduced sizing
      0.10–0.30: marginal, may execute if no better opportunities
      < 0.10  : reject

    These thresholds must be validated empirically, not accepted as gospel.
    Use `validate_formula()` to confirm that this score ranks trades better
    than raw expected net return.
    """

    def __init__(
        self,
        min_viable_score: float = MINIMUM_VIABLE_SCORE,
        max_acceptable_ece: float = 0.15,
        target_return_scale: float = 0.02,   # 2% per trade = "full score"
        target_vol_scale: float = 0.02,       # 2% vol = "full score"
    ) -> None:
        self.min_viable_score = min_viable_score
        self.max_acceptable_ece = max_acceptable_ece
        self.target_return_scale = target_return_scale
        self.target_vol_scale = target_vol_scale

    def score(
        self,
        expected_net_return: float,
        prob_target: float = 0.5,
        calibration_ece: float = 0.10,
        regime_fit: float = 1.0,
        data_quality: float = 1.0,
        liquidity_quality: float = 1.0,
        execution_quality: float = 1.0,
        alpha_stability: float = 1.0,
        uncertainty: float = 0.5,
        cost_bps: float = 27.65,
        risk_vol: float = 0.01,
    ) -> OpportunityScoreResult:
        """
        Compute the opportunity score for a candidate trade.

        Args:
            expected_net_return : net return after costs (fraction, e.g. 0.008)
            prob_target         : calibrated P(target hit) ∈ [0, 1]
            calibration_ece     : Expected Calibration Error ∈ [0, 1]; lower = better
            regime_fit          : how well regime suits this alpha ∈ [0, 1]
            data_quality        : DataConfidenceScore / 100 ∈ [0, 1]
            liquidity_quality   : ADV / capacity utilization proxy ∈ [0, 1]
            execution_quality   : fill probability / market impact proxy ∈ [0, 1]
            alpha_stability     : 1 - ic_decay_fraction ∈ [0, 1]
            uncertainty         : epistemic uncertainty ∈ [0, 1]
            cost_bps            : round-trip transaction cost in bps
            risk_vol            : expected trade volatility (fraction)

        Returns:
            OpportunityScoreResult with score ∈ [0, 1]
        """
        # ── Clip all inputs to valid ranges ───────────────────────────────
        p_t = max(0.0, min(1.0, prob_target))
        ece = max(0.0, min(1.0, calibration_ece))
        r_fit = max(0.0, min(1.0, regime_fit))
        d_qual = max(0.0, min(1.0, data_quality))
        liq = max(0.0, min(1.0, liquidity_quality))
        exec_q = max(0.0, min(1.0, execution_quality))
        alpha_s = max(0.0, min(1.0, alpha_stability))
        unc = max(0.0, min(1.0, uncertainty))
        cost_frac = max(0.0, cost_bps / 10_000.0)
        vol = max(1e-6, risk_vol)

        # ── Numerator components ──────────────────────────────────────────
        # return_score: normalised expected net return (0 when zero, 1 when >= target_scale)
        # Negative returns get score 0 (not −x which would pull score negative)
        if expected_net_return <= 0:
            return_score = 0.0
        else:
            return_score = min(1.0, expected_net_return / max(self.target_return_scale, 1e-8))

        # prob_quality: how far above 0.5 is the win probability?
        # P=0.5 is pure noise (score=0), P=1.0 is perfect (score=1)
        prob_quality = max(0.0, (p_t - 0.5) * 2.0)

        # calibration_quality: 1 - ECE/max_ece, clipped to [0, 1]
        calibration_quality = max(0.0, 1.0 - ece / max(self.max_acceptable_ece, 1e-8))

        # Numerator: product of all quality factors
        numerator = (
            return_score
            * prob_quality
            * calibration_quality
            * r_fit
            * d_qual
            * liq
            * exec_q
            * alpha_s
        )

        # ── Denominator components ────────────────────────────────────────
        # risk_factor: normalised volatility (0 when zero, 1 when >= target_vol_scale)
        risk_factor = min(1.0, vol / max(self.target_vol_scale, 1e-8))

        # cost_factor: relative cost burden (0 if free, 1 if cost ≈ target return)
        cost_factor = min(1.0, cost_frac / max(self.target_return_scale, 1e-8))

        # Denominator must never be zero
        denominator = max(0.001, risk_factor + unc + cost_factor)

        # ── Final score ───────────────────────────────────────────────────
        raw_score = numerator / denominator
        # Map to [0, 1]: the maximum possible numerator is 1 and min denominator
        # is 0.001, giving raw max = 1000. We normalise to [0, 1] using sigmoid-like
        # mapping: score = raw / (1 + raw) keeps it in (0, 1).
        opportunity_score = raw_score / (1.0 + raw_score)

        is_viable = opportunity_score >= self.min_viable_score
        reason = (
            "VIABLE" if is_viable
            else f"BELOW_THRESHOLD (score={opportunity_score:.4f} < min={self.min_viable_score})"
        )

        logger.debug(
            "opportunity_score_computed",
            score=round(opportunity_score, 4),
            return_score=round(return_score, 4),
            prob_quality=round(prob_quality, 4),
            is_viable=is_viable,
        )

        return OpportunityScoreResult(
            expected_net_return=expected_net_return,
            prob_target=p_t,
            calibration_ece=ece,
            regime_fit=r_fit,
            data_quality=d_qual,
            liquidity_quality=liq,
            execution_quality=exec_q,
            alpha_stability=alpha_s,
            uncertainty=unc,
            cost_bps=cost_bps,
            risk_vol=vol,
            return_score=round(return_score, 6),
            prob_quality=round(prob_quality, 6),
            calibration_quality=round(calibration_quality, 6),
            numerator=round(numerator, 8),
            denominator=round(denominator, 8),
            opportunity_score=round(opportunity_score, 6),
            is_viable=is_viable,
            reason=reason,
        )

    def rank_signals(
        self,
        signals: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Score and rank a list of candidate signal dicts.

        Each dict must contain at minimum: 'expected_net_return'.
        Other keys are optional and default to neutral values.

        Returns the list sorted by opportunity_score descending.
        """
        scored = []
        for sig in signals:
            result = self.score(
                expected_net_return=float(sig.get("expected_net_return", 0.0)),
                prob_target=float(sig.get("prob_target", 0.5)),
                calibration_ece=float(sig.get("calibration_ece", 0.10)),
                regime_fit=float(sig.get("regime_fit", 1.0)),
                data_quality=float(sig.get("data_quality", 1.0)),
                liquidity_quality=float(sig.get("liquidity_quality", 1.0)),
                execution_quality=float(sig.get("execution_quality", 1.0)),
                alpha_stability=float(sig.get("alpha_stability", 1.0)),
                uncertainty=float(sig.get("uncertainty", 0.5)),
                cost_bps=float(sig.get("cost_bps", 27.65)),
                risk_vol=float(sig.get("risk_vol", 0.01)),
            )
            enriched = dict(sig)
            enriched["opportunity_score"] = result.opportunity_score
            enriched["is_viable"] = result.is_viable
            enriched["score_components"] = result.to_dict()["components"]
            scored.append(enriched)

        scored.sort(key=lambda x: x["opportunity_score"], reverse=True)
        return scored

    def validate_formula(
        self,
        historical_signals: list[dict[str, Any]],
        realized_returns: list[float],
        top_n_pct: float = 0.20,
    ) -> dict[str, Any]:
        """
        Validate whether opportunity-score ranking improves selection vs raw EV.

        Compares:
          A. Top 20% by opportunity_score → mean realized return
          B. Top 20% by raw expected_net_return → mean realized return
          C. All signals → mean realized return (baseline)

        Returns a dict with comparison results. If score ranking adds value,
        score_mean_return > ev_mean_return.
        """
        import numpy as np  # noqa: PLC0415

        if len(historical_signals) != len(realized_returns):
            return {"error": "signals and realized_returns must be same length"}
        if len(historical_signals) < 10:
            return {"error": "insufficient signals for validation (need >= 10)"}

        scored = self.rank_signals(historical_signals)
        ret_arr = np.array(realized_returns)

        # Align realized returns to the sorted order
        original_order = {id(sig): i for i, sig in enumerate(historical_signals)}
        sorted_indices = []
        for sig in scored:
            # Find original index by signal identity
            for j, orig in enumerate(historical_signals):
                if orig.get("signal_id") == sig.get("signal_id"):
                    sorted_indices.append(j)
                    break

        k = max(1, int(len(scored) * top_n_pct))

        # A: top-k by opportunity score
        top_k_score_idx = [historical_signals.index(
            next((s for s in historical_signals
                  if s.get("expected_net_return") == sc.get("expected_net_return")), {})
        ) for sc in scored[:k]]
        score_rets = [realized_returns[i] for i in top_k_score_idx if i < len(realized_returns)]

        # B: top-k by raw EV
        ev_sorted = sorted(
            enumerate(historical_signals),
            key=lambda x: float(x[1].get("expected_net_return", 0)),
            reverse=True,
        )
        ev_rets = [realized_returns[i] for i, _ in ev_sorted[:k]]

        base_mean = float(np.mean(ret_arr))
        score_mean = float(np.mean(score_rets)) if score_rets else base_mean
        ev_mean = float(np.mean(ev_rets)) if ev_rets else base_mean

        return {
            "n_signals": len(historical_signals),
            "top_n_pct": top_n_pct,
            "k": k,
            "baseline_mean_return": round(base_mean, 6),
            "score_ranking_mean_return": round(score_mean, 6),
            "ev_ranking_mean_return": round(ev_mean, 6),
            "score_beats_ev": score_mean > ev_mean,
            "score_beats_baseline": score_mean > base_mean,
            "verdict": (
                "SCORE_RANKING_ADDS_VALUE" if score_mean > ev_mean
                else "NO_INCREMENTAL_VALUE_OVER_RAW_EV"
            ),
        }
