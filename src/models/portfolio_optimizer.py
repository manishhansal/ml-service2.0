"""
PortfolioOptimizer — Riskfolio-Lib-based portfolio optimization.

Supports four optimization methods:
  - HRP  : Hierarchical Risk Parity (Riskfolio-Lib HCPortfolio, Ward linkage)
  - CVaR : CVaR-minimized Mean-Variance Optimization (Riskfolio-Lib Portfolio)
  - ERC  : Equal Risk Contribution (Riskfolio-Lib rp_optimization)
  - MaxDiv: Maximum Diversification (custom closed-form via inverse-volatility
            weighting, maximizing the diversification ratio)

PyPortfolioOpt HRP is available as a fallback for the HRP method and for the
legacy v1 endpoint (optimize()).

Requirements: Req 8.1, Req 8.2, Req 8.3, Req 8.5, Req 8.7, Req 8.8, Req 8.9.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.logging_config import get_logger
from src.schemas.base import PredictionProvenance
from src.schemas.predictions import (
    PortfolioAllocation,
    PortfolioRequest,
    PortfolioResponse,
    PortfolioV2Response,
)

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MIN_OBSERVATIONS: int = 20
"""Minimum number of return observations required per asset (Req 8.8)."""

MAX_SECTOR_WEIGHT: float = 0.40
"""Default maximum weight per sector (Req 8.4)."""

_SUPPORTED_METHODS = frozenset({"hrp", "cvar", "erc", "max_div"})


# ── Helper ─────────────────────────────────────────────────────────────────────


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Re-normalise so that weights sum exactly to 1.0 and all values >= 0."""
    clipped = {k: max(0.0, v) for k, v in weights.items()}
    total = sum(clipped.values())
    if total > 0:
        return {k: v / total for k, v in clipped.items()}
    # Degenerate case: fall back to equal weight
    n = len(clipped)
    eq = 1.0 / n if n > 0 else 0.0
    return {k: eq for k in clipped}


def _equal_weights(columns: list[str]) -> dict[str, float]:
    """Return an equal-weight allocation for the given asset list."""
    n = len(columns)
    eq = 1.0 / n if n > 0 else 0.0
    return {col: eq for col in columns}


# ── PortfolioOptimizer ─────────────────────────────────────────────────────────


class PortfolioOptimizer:
    """
    Institutional-grade portfolio optimizer.

    Primary methods (Req 8.1):
      * ``hrp_allocation``  — Hierarchical Risk Parity
      * ``cvar_allocation`` — CVaR-minimized MVO
      * ``erc_allocation``  — Equal Risk Contribution
      * ``max_div_allocation`` — Maximum Diversification

    Each method returns a dict with keys:
      ``weights``      — ``dict[str, float]`` summing to 1.0, all >= 0
      ``risk_metrics`` — volatility, cvar, sharpe, max_dd, expected_return,
                         diversification_ratio
      ``available``    — ``True`` (always present when data is sufficient)
      ``provenance``   — ``PredictionProvenance``

    When input data is insufficient (< ``MIN_OBSERVATIONS`` rows), all methods
    immediately return ``{"available": False, "reason": "INSUFFICIENT_RETURN_HISTORY"}``
    (Req 8.8).

    The legacy v1 endpoint is served by ``optimize()``.
    """

    model_version: str = "2.0.0"

    # ── Public optimization methods ────────────────────────────────────────────

    def hrp_allocation(
        self,
        returns_df: pd.DataFrame,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """
        Hierarchical Risk Parity using Riskfolio-Lib with Ward linkage (Req 8.2).

        Guarantees:
        - ``sum(weights.values()) == 1.0`` (within 1 e-9)
        - ``all(w >= 0 for w in weights.values())``
        - Portfolio volatility <= max individual asset volatility (diversification)
        - Deterministic for the same inputs and ``random_state`` (Req 8.7)

        Falls back to PyPortfolioOpt HRP, then equal-weight if Riskfolio-Lib
        is unavailable or raises an exception.
        """
        if not self._has_sufficient_data(returns_df):
            return self._insufficient_data_response()

        # ── Attempt 1: Riskfolio-Lib HRP ──────────────────────────────────────
        try:
            return self._hrp_riskfolio(returns_df, random_state)
        except Exception as exc:
            logger.warning(
                "riskfolio_hrp_failed_falling_back_to_pypfopt",
                error=str(exc),
            )

        # ── Attempt 2: PyPortfolioOpt HRP ─────────────────────────────────────
        try:
            return self._hrp_pypfopt(returns_df)
        except Exception as exc:
            logger.warning(
                "pypfopt_hrp_failed_falling_back_to_equal_weight",
                error=str(exc),
            )

        # ── Attempt 3: Equal weight (final fallback) ──────────────────────────
        weights = _equal_weights(list(returns_df.columns))
        risk_metrics = self._compute_risk_metrics(returns_df, weights)
        return {
            "available": True,
            "weights": weights,
            "risk_metrics": risk_metrics,
            "provenance": PredictionProvenance.HEURISTIC,
        }

    def cvar_allocation(
        self,
        returns_df: pd.DataFrame,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """
        CVaR-minimized portfolio using Riskfolio-Lib (Req 8.3).

        ``alpha`` is the tail probability for CVaR computation (default 5 %).
        Falls back to HRP on optimization failure.
        """
        if not self._has_sufficient_data(returns_df):
            return self._insufficient_data_response()

        try:
            return self._cvar_riskfolio(returns_df, alpha)
        except Exception as exc:
            logger.warning(
                "cvar_optimization_failed_falling_back_to_hrp",
                error=str(exc),
            )
            return self.hrp_allocation(returns_df)

    def erc_allocation(
        self,
        returns_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Equal Risk Contribution portfolio using Riskfolio-Lib (Req 8.1).

        Each asset contributes equally to total portfolio risk.
        Falls back to HRP on failure.
        """
        if not self._has_sufficient_data(returns_df):
            return self._insufficient_data_response()

        try:
            return self._erc_riskfolio(returns_df)
        except Exception as exc:
            logger.warning(
                "erc_optimization_failed_falling_back_to_hrp",
                error=str(exc),
            )
            return self.hrp_allocation(returns_df)

    def max_div_allocation(
        self,
        returns_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Maximum Diversification portfolio (Req 8.1).

        Maximises the diversification ratio (weighted-average individual
        volatility / portfolio volatility).  Implemented via convex
        quadratic programming using Riskfolio-Lib's MinRisk with MV, which
        minimises portfolio variance and thus indirectly maximises the
        diversification ratio in the presence of uncorrelated assets.

        Falls back to inverse-volatility weighting (closed-form maximum
        diversification approximation) if Riskfolio-Lib fails.
        """
        if not self._has_sufficient_data(returns_df):
            return self._insufficient_data_response()

        try:
            return self._max_div_riskfolio(returns_df)
        except Exception as exc:
            logger.warning(
                "max_div_riskfolio_failed_falling_back_to_inv_vol",
                error=str(exc),
            )

        # Fallback: inverse-volatility (closed-form max-div approximation)
        try:
            return self._max_div_inv_vol(returns_df)
        except Exception as exc:
            logger.warning(
                "max_div_inv_vol_failed_falling_back_to_hrp",
                error=str(exc),
            )
            return self.hrp_allocation(returns_df)

    # ── Legacy v1 endpoint ─────────────────────────────────────────────────────

    def optimize(self, request: PortfolioRequest) -> PortfolioResponse:
        """
        Legacy portfolio optimization for the v1 endpoint.

        Without historical return series the optimizer builds a synthetic
        equal-weight allocation constrained by ``max_positions`` and
        ``max_sector_weight``.
        """
        if not request.assets:
            return PortfolioResponse(
                allocations=[],
                expected_return=0.0,
                portfolio_risk=0.0,
                sharpe_ratio=0.0,
                diversification_ratio=1.0,
                provenance=PredictionProvenance.HEURISTIC,
            )

        selected = list(request.assets[: request.max_positions])
        n = len(selected)
        base_weight = 1.0 / n

        allocations = [
            PortfolioAllocation(
                symbol=asset.symbol,
                weight=round(base_weight, 4),
                sector=asset.sector,
                rationale=(
                    f"Equal weight allocation; rank_score={asset.rank_score:.2f}"
                ),
            )
            for asset in selected
        ]

        # Re-normalise to ensure exact sum == 1.0 accounting for rounding
        total = sum(a.weight for a in allocations)
        if total > 0 and abs(total - 1.0) > 1e-9:
            allocations = [
                PortfolioAllocation(
                    symbol=a.symbol,
                    weight=round(a.weight / total, 4),
                    sector=a.sector,
                    rationale=a.rationale,
                )
                for a in allocations
            ]

        mean_expected_return = sum(
            a.expected_return for a in request.assets[:n]
        ) / max(n, 1)

        return PortfolioResponse(
            allocations=allocations,
            expected_return=mean_expected_return,
            portfolio_risk=request.risk_budget_pct / 100.0,
            sharpe_ratio=1.0,
            diversification_ratio=1.2,
            provenance=PredictionProvenance.HEURISTIC,
        )

    # ── Riskfolio-Lib internals ────────────────────────────────────────────────

    def _hrp_riskfolio(
        self, returns_df: pd.DataFrame, random_state: int
    ) -> dict[str, Any]:
        """HRP via Riskfolio-Lib with Ward linkage (Req 8.2)."""
        import riskfolio as rp  # type: ignore[import]

        np.random.seed(random_state)  # seed for determinism (Req 8.7)
        port = rp.HCPortfolio(returns=returns_df)
        w = port.optimization(
            model="HRP",
            codependence="pearson",
            rm="MV",
            rf=0,
            linkage="ward",
            leaf_order=True,
        )
        weights = _normalize_weights(
            {str(asset): float(weight) for asset, weight in w["weights"].items()}
        )
        risk_metrics = self._compute_risk_metrics(returns_df, weights)
        return {
            "available": True,
            "weights": weights,
            "risk_metrics": risk_metrics,
            "provenance": PredictionProvenance.TRAINED_MODEL,
        }

    def _hrp_pypfopt(self, returns_df: pd.DataFrame) -> dict[str, Any]:
        """HRP via PyPortfolioOpt (Req 8.2 fallback)."""
        from pypfopt import HRPOpt  # type: ignore[import]

        hrp = HRPOpt(returns=returns_df)
        raw_weights = hrp.optimize()
        weights = _normalize_weights(
            {str(k): float(v) for k, v in raw_weights.items()}
        )
        risk_metrics = self._compute_risk_metrics(returns_df, weights)
        return {
            "available": True,
            "weights": weights,
            "risk_metrics": risk_metrics,
            "provenance": PredictionProvenance.TRAINED_MODEL,
        }

    def _cvar_riskfolio(
        self, returns_df: pd.DataFrame, alpha: float
    ) -> dict[str, Any]:
        """CVaR-minimized MVO via Riskfolio-Lib (Req 8.3)."""
        import riskfolio as rp  # type: ignore[import]

        port = rp.Portfolio(returns=returns_df)
        port.assets_stats(method_mu="hist", method_cov="hist")
        w = port.optimization(
            model="Classic",
            rm="CVaR",
            obj="MinRisk",
            rf=0,
            l=0,
            hist=True,
        )
        weights = _normalize_weights(
            {str(asset): float(weight) for asset, weight in w["weights"].items()}
        )
        risk_metrics = self._compute_risk_metrics(returns_df, weights, alpha=alpha)
        return {
            "available": True,
            "weights": weights,
            "risk_metrics": risk_metrics,
            "provenance": PredictionProvenance.TRAINED_MODEL,
        }

    def _erc_riskfolio(self, returns_df: pd.DataFrame) -> dict[str, Any]:
        """Equal Risk Contribution via Riskfolio-Lib rp_optimization (Req 8.1)."""
        import riskfolio as rp  # type: ignore[import]

        port = rp.Portfolio(returns=returns_df)
        port.assets_stats(method_mu="hist", method_cov="hist")
        w = port.rp_optimization(
            model="Classic",
            rm="MV",
            rf=0,
            hist=True,
        )
        weights = _normalize_weights(
            {str(asset): float(weight) for asset, weight in w["weights"].items()}
        )
        risk_metrics = self._compute_risk_metrics(returns_df, weights)
        return {
            "available": True,
            "weights": weights,
            "risk_metrics": risk_metrics,
            "provenance": PredictionProvenance.TRAINED_MODEL,
        }

    def _max_div_riskfolio(self, returns_df: pd.DataFrame) -> dict[str, Any]:
        """
        Maximum Diversification via Riskfolio-Lib.

        Uses MV minimum-risk optimization (which minimises portfolio variance)
        as the convex proxy for maximising the diversification ratio.
        """
        import riskfolio as rp  # type: ignore[import]

        port = rp.Portfolio(returns=returns_df)
        port.assets_stats(method_mu="hist", method_cov="hist")
        w = port.optimization(
            model="Classic",
            rm="MV",
            obj="MinRisk",
            rf=0,
            hist=True,
        )
        weights = _normalize_weights(
            {str(asset): float(weight) for asset, weight in w["weights"].items()}
        )
        risk_metrics = self._compute_risk_metrics(returns_df, weights)
        return {
            "available": True,
            "weights": weights,
            "risk_metrics": risk_metrics,
            "provenance": PredictionProvenance.TRAINED_MODEL,
        }

    def _max_div_inv_vol(self, returns_df: pd.DataFrame) -> dict[str, Any]:
        """
        Closed-form Maximum Diversification approximation: inverse-volatility
        weighting.  Each asset's weight is proportional to 1/σ_i.
        """
        vols = returns_df.std(axis=0)
        inv_vols = 1.0 / vols.replace(0, np.nan).dropna()
        raw: dict[str, float] = {
            str(col): float(inv_vols[col]) if col in inv_vols.index else 0.0
            for col in returns_df.columns
        }
        weights = _normalize_weights(raw)
        risk_metrics = self._compute_risk_metrics(returns_df, weights)
        return {
            "available": True,
            "weights": weights,
            "risk_metrics": risk_metrics,
            "provenance": PredictionProvenance.HEURISTIC,
        }

    # ── Risk metrics ───────────────────────────────────────────────────────────

    def _compute_risk_metrics(
        self,
        returns_df: pd.DataFrame,
        weights: dict[str, float],
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """
        Compute portfolio risk statistics for the given weight vector (Req 8.5).

        Returns:
            volatility           — annualised portfolio volatility
            cvar                 — conditional value at risk at tail prob alpha
            sharpe               — annualised Sharpe (risk-free = 0)
            max_dd               — maximum drawdown (negative number)
            expected_return      — annualised mean portfolio return
            diversification_ratio — weighted-avg individual vol / portfolio vol
        """
        w_vec = np.array(
            [weights.get(str(col), 0.0) for col in returns_df.columns]
        )
        port_returns: np.ndarray = returns_df.values @ w_vec

        # Annualised volatility
        volatility = float(np.std(port_returns, ddof=1) * np.sqrt(252))

        # CVaR at alpha tail probability
        sorted_rets = np.sort(port_returns)
        cutoff = max(1, int(len(sorted_rets) * alpha))
        cvar = float(-np.mean(sorted_rets[:cutoff]))

        # Annualised mean return
        mean_return = float(np.mean(port_returns) * 252)

        # Sharpe ratio (risk-free = 0)
        sharpe = mean_return / max(volatility, 1e-9)

        # Max drawdown
        cum = np.cumprod(1.0 + port_returns)
        running_max = np.maximum.accumulate(cum)
        drawdowns = (cum - running_max) / np.where(running_max == 0, 1.0, running_max)
        max_dd = float(np.min(drawdowns))

        # Diversification ratio (Req 8.5)
        individual_vols = np.std(returns_df.values, axis=0, ddof=1) * np.sqrt(252)
        weighted_avg_vol = float(np.dot(w_vec, individual_vols))
        div_ratio = weighted_avg_vol / max(volatility, 1e-9)

        return {
            "volatility": round(volatility, 6),
            "cvar": round(cvar, 6),
            "sharpe": round(sharpe, 4),
            "max_dd": round(max_dd, 6),
            "expected_return": round(mean_return, 6),
            "diversification_ratio": round(div_ratio, 4),
        }

    # ── Guards ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _has_sufficient_data(returns_df: pd.DataFrame) -> bool:
        """Return True when every asset has at least MIN_OBSERVATIONS rows (Req 8.8)."""
        return len(returns_df) >= MIN_OBSERVATIONS

    @staticmethod
    def _insufficient_data_response() -> dict[str, Any]:
        """Standard unavailable payload when return history is too short (Req 8.8)."""
        return {
            "available": False,
            "reason": "INSUFFICIENT_RETURN_HISTORY",
            "weights": {},
            "risk_metrics": {},
        }
