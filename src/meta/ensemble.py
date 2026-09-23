"""
EnsembleWeighter — regime-aware IC-proportional weighting for the MetaDecisionEngine.

Design (Req 10.3):
- Each model's weight is proportional to its IC in the current regime.
- No single model weight < 0.05 or > 0.40.
- All weights sum to 1.0.
- Models with IC < 0.05 in the current regime receive the minimum weight 0.05.
- Models with UNAVAILABLE provenance receive weight 0.0 (excluded from ensemble).

Requirements: Req 10.3
"""
from __future__ import annotations

import numpy as np
from src.logging_config import get_logger

logger = get_logger(__name__)

MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.40


class EnsembleWeighter:
    """
    Computes regime-aware IC-proportional weights for ensemble voting.

    Usage::
        weighter = EnsembleWeighter()
        weights = weighter.compute_weights(
            model_ids=["regime", "ranker", "risk"],
            ic_scores={"regime": 0.04, "ranker": 0.06, "risk": 0.03},
            regime="bull",
        )
        # Returns dict mapping model_id -> weight (sum to 1.0)
    """

    def __init__(self) -> None:
        # Historical IC per model per regime — populated by register_ic()
        self._ic_registry: dict[str, dict[str, float]] = {}

    def register_ic(self, model_id: str, regime: str, ic: float) -> None:
        """Register the IC of a model in a specific regime."""
        if model_id not in self._ic_registry:
            self._ic_registry[model_id] = {}
        self._ic_registry[model_id][regime] = float(ic)

    def compute_weights(
        self,
        model_ids: list[str],
        ic_scores: dict[str, float] | None = None,
        regime: str = "sideways",
        available_mask: dict[str, bool] | None = None,
    ) -> dict[str, float]:
        """
        Compute IC-proportional weights with [0.05, 0.40] constraints.

        Args:
            model_ids:     List of model identifiers to weight.
            ic_scores:     Optional override IC scores. If None, uses registry.
            regime:        Current market regime string (used to look up registry ICs).
            available_mask: Optional map of model_id -> bool (False = UNAVAILABLE, gets 0 weight).

        Returns:
            dict mapping model_id -> weight. Available models sum to 1.0.
            Unavailable models get weight 0.0 and are excluded from the sum.
        """
        if not model_ids:
            return {}

        # Determine effective ICs
        effective_ics: dict[str, float] = {}
        for mid in model_ids:
            # Check if model is available
            if available_mask and not available_mask.get(mid, True):
                effective_ics[mid] = 0.0  # UNAVAILABLE → excluded
                continue

            # Use provided IC or look up from registry
            if ic_scores and mid in ic_scores:
                effective_ics[mid] = max(0.0, float(ic_scores[mid]))
            elif mid in self._ic_registry and regime in self._ic_registry[mid]:
                effective_ics[mid] = max(0.0, float(self._ic_registry[mid][regime]))
            else:
                # Default: minimum eligible IC
                effective_ics[mid] = MIN_WEIGHT  # will get min weight

        # Split available vs unavailable
        available_models = [mid for mid in model_ids if effective_ics.get(mid, 0) > 0]
        unavailable_models = [mid for mid in model_ids if effective_ics.get(mid, 0) == 0]

        weights: dict[str, float] = {mid: 0.0 for mid in unavailable_models}

        if not available_models:
            return weights

        # Apply IC < 0.05 → minimum weight rule
        raw_weights: dict[str, float] = {}
        for mid in available_models:
            ic = effective_ics[mid]
            if ic < 0.05:
                raw_weights[mid] = MIN_WEIGHT  # IC below 0.05 → floor weight
            else:
                raw_weights[mid] = ic

        # Project onto [MIN_WEIGHT, MAX_WEIGHT] constraints
        constrained = self._apply_constraints(raw_weights)

        weights.update(constrained)
        return weights

    def _apply_constraints(self, raw_weights: dict[str, float]) -> dict[str, float]:
        """
        Project weights onto [MIN_WEIGHT, MAX_WEIGHT] with sum = 1.0.

        When the number of models makes the MAX_WEIGHT constraint infeasible
        (e.g. 2 models cannot both be ≤ 0.40 and sum to 1.0), the MAX_WEIGHT
        cap is relaxed and only the MIN_WEIGHT floor is enforced.
        """
        n = len(raw_weights)
        if n == 0:
            return {}

        models = list(raw_weights.keys())
        w = np.array([raw_weights[m] for m in models], dtype=float)

        # MAX_WEIGHT cap is only feasible when n * MAX_WEIGHT >= 1.0
        # (i.e. it's possible for all models to be <= MAX_WEIGHT and still sum to 1).
        max_weight_feasible = n * MAX_WEIGHT >= 1.0 - 1e-9

        # Normalize initial weights
        total = w.sum()
        if total > 0:
            w = w / total
        else:
            w = np.full(n, 1.0 / n)

        if max_weight_feasible:
            w = self._redistribute_excess(w, n)
        else:
            # Only enforce MIN_WEIGHT floor; renormalize
            w = np.maximum(w, MIN_WEIGHT)
            total = w.sum()
            if total > 0:
                w = w / total

        return {m: float(w[i]) for i, m in enumerate(models)}

    @staticmethod
    def _redistribute_excess(w: np.ndarray, n: int) -> np.ndarray:
        """
        Project weights onto [MIN_WEIGHT, MAX_WEIGHT] with sum = 1.0.

        Uses an exact water-filling approach:
        1. Pin models that are at their bounds (MAX_WEIGHT ceiling or MIN_WEIGHT floor).
        2. Renormalize the remaining "free" models to fill the residual budget.
        3. Repeat until stable.
        """
        w = w.copy()

        for _ in range(n * 2 + 2):
            # Determine budget consumed by pinned models
            pinned_max = w >= MAX_WEIGHT - 1e-12
            pinned_min = w <= MIN_WEIGHT + 1e-12
            free = ~pinned_max & ~pinned_min

            # Clamp pinned models exactly
            w[pinned_max] = MAX_WEIGHT
            w[pinned_min] = MIN_WEIGHT

            # Budget left for free models
            budget = 1.0 - w[pinned_max].sum() - w[pinned_min].sum()

            if not np.any(free):
                # All models are pinned; force-adjust to make sum = 1.0
                # Distribute residual among models pinned at MAX_WEIGHT
                residual = 1.0 - w.sum()
                if np.any(pinned_max):
                    w[pinned_max] += residual / pinned_max.sum()
                    # Clamp again in case rounding pushes above
                    w = np.clip(w, MIN_WEIGHT, MAX_WEIGHT)
                break

            # Distribute budget proportionally to free models
            free_sum = w[free].sum()
            if free_sum > 0:
                w[free] = w[free] / free_sum * budget
            else:
                w[free] = budget / free.sum()

            # Check if all free models are still within bounds
            if np.all(w[free] >= MIN_WEIGHT - 1e-12) and np.all(w[free] <= MAX_WEIGHT + 1e-12):
                break

            # Some free models went out of bounds — they'll be pinned next iteration
            w = np.clip(w, MIN_WEIGHT, MAX_WEIGHT)

        # Final: ensure exact bounds and fix floating-point residual in sum
        w = np.clip(w, MIN_WEIGHT, MAX_WEIGHT)
        total = w.sum()
        if total > 0 and abs(total - 1.0) > 1e-12:
            # Adjust the weight farthest from its bound to absorb the residual
            residual = 1.0 - total
            if residual > 0:
                idx = int(np.argmin(w))  # model farthest from MAX, most room to grow
            else:
                idx = int(np.argmax(w))  # model farthest from MIN, most room to shrink
            w[idx] += residual
            w[idx] = min(max(float(w[idx]), MIN_WEIGHT), MAX_WEIGHT)

        return w
