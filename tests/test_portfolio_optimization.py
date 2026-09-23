"""
test_portfolio_optimization.py

Property-based tests for PortfolioOptimizer.

Properties tested:
  Property 9a: sum(weights(HRP(R))) == 1.0 for all valid R
  Property 9b: all(w >= 0 for w in weights(HRP(R)))
  Property 9c: portfolio_volatility(HRP(R)) <= max(asset_volatilities(R)) [diversification]
  Property 10:  determinism — same inputs + random_state -> same weights

**Validates: Requirements 8.1, 8.2, 8.7, 8.8**
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as h_settings
from hypothesis import strategies as st

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def valid_returns_matrix(draw: st.DrawFn) -> pd.DataFrame:
    """Generate a valid returns DataFrame with 2–5 assets and 20–100 observations."""
    n_assets = draw(st.integers(min_value=2, max_value=5))
    n_obs = draw(st.integers(min_value=20, max_value=100))
    seed = draw(st.integers(min_value=0, max_value=9999))
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, size=(n_obs, n_assets))
    cols = [f"SYM{i}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=cols)


# ---------------------------------------------------------------------------
# Property 9a / 9b / 9c / 10 — HRP weight invariants
# ---------------------------------------------------------------------------


class TestHRPWeightInvariants:
    """Properties 9a, 9b, 9c, 10 — Validates: Requirements 8.1, 8.2, 8.7."""

    def test_hrp_weights_sum_to_one_fixed(self) -> None:
        """Fixed example: HRP weights sum to 1.0 for a 5-asset 252-day matrix."""
        from src.models.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            rng.normal(0, 0.01, size=(252, 5)), columns=list("ABCDE")
        )
        result = optimizer.hrp_allocation(returns, random_state=42)
        weights = result["weights"]
        assert abs(sum(weights.values()) - 1.0) < 1e-6, (
            f"Weights sum = {sum(weights.values())}"
        )

    def test_hrp_all_weights_non_negative_fixed(self) -> None:
        """Fixed example: All HRP weights >= 0."""
        from src.models.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            rng.normal(0, 0.01, size=(252, 5)), columns=list("ABCDE")
        )
        result = optimizer.hrp_allocation(returns, random_state=42)
        weights = result["weights"]
        assert all(w >= 0 for w in weights.values()), (
            f"Negative weight found: {weights}"
        )

    def test_hrp_diversification_property_fixed(self) -> None:
        """Property 9c: Portfolio volatility <= max individual asset volatility."""
        from src.models.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            rng.normal(0, 0.01, size=(252, 5)), columns=list("ABCDE")
        )
        result = optimizer.hrp_allocation(returns, random_state=42)
        portfolio_vol = result["risk_metrics"]["volatility"]
        asset_vols = returns.std(axis=0).values * np.sqrt(252)
        max_asset_vol = float(asset_vols.max())
        assert portfolio_vol <= max_asset_vol + 1e-6, (
            f"portfolio_vol={portfolio_vol:.6f} > max_asset_vol={max_asset_vol:.6f}"
        )

    def test_hrp_determinism(self) -> None:
        """Property 10: Same inputs + same seed produce identical weights."""
        from src.models.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()
        rng = np.random.default_rng(99)
        returns = pd.DataFrame(
            rng.normal(0, 0.01, size=(100, 3)), columns=["A", "B", "C"]
        )
        r1 = optimizer.hrp_allocation(returns, random_state=42)
        r2 = optimizer.hrp_allocation(returns, random_state=42)
        assert r1["weights"] == r2["weights"], "HRP is not deterministic"

    @given(returns_df=valid_returns_matrix())
    @h_settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=5000,
    )
    def test_hrp_weights_sum_to_one_hypothesis(self, returns_df: pd.DataFrame) -> None:
        """Property 9a (Hypothesis): sum(weights) == 1.0 for all valid inputs.

        **Validates: Requirements 8.1, 8.2**
        """
        from src.models.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()
        result = optimizer.hrp_allocation(returns_df, random_state=42)
        weights = result.get("weights", {})
        if not weights:
            return  # INSUFFICIENT_RETURN_HISTORY guard triggered
        assert abs(sum(weights.values()) - 1.0) < 1e-6, (
            f"weights sum to {sum(weights.values()):.9f}, expected 1.0"
        )

    @given(returns_df=valid_returns_matrix())
    @h_settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=5000,
    )
    def test_hrp_non_negative_weights_hypothesis(self, returns_df: pd.DataFrame) -> None:
        """Property 9b (Hypothesis): all(w >= 0) for all valid inputs.

        **Validates: Requirements 8.1, 8.2**
        """
        from src.models.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()
        result = optimizer.hrp_allocation(returns_df, random_state=42)
        weights = result.get("weights", {})
        if not weights:
            return
        neg = {k: v for k, v in weights.items() if v < -1e-9}
        assert not neg, f"Negative weights found: {neg}"


# ---------------------------------------------------------------------------
# Req 8.8 — Insufficient-data guard
# ---------------------------------------------------------------------------


class TestInsufficientDataGuard:
    """Req 8.8: INSUFFICIENT_RETURN_HISTORY returned when < MIN_OBSERVATIONS rows."""

    @pytest.mark.parametrize("n_rows", [1, 5, 10, 19])
    def test_returns_unavailable_when_fewer_than_20_obs(self, n_rows: int) -> None:
        """All methods must return available=False for histories shorter than 20 rows."""
        from src.models.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer()
        rng = np.random.default_rng(0)
        small_df = pd.DataFrame(
            rng.normal(0, 0.01, size=(n_rows, 3)), columns=["A", "B", "C"]
        )
        for method_name in ("hrp_allocation", "cvar_allocation", "erc_allocation", "max_div_allocation"):
            result = getattr(optimizer, method_name)(small_df)
            assert result.get("available") is False, (
                f"{method_name} with {n_rows} rows returned available=True"
            )
            assert result.get("reason") == "INSUFFICIENT_RETURN_HISTORY", (
                f"{method_name} returned unexpected reason: {result.get('reason')}"
            )

    def test_returns_available_at_boundary(self) -> None:
        """Exactly MIN_OBSERVATIONS rows (20) should not trigger the guard."""
        from src.models.portfolio_optimizer import MIN_OBSERVATIONS, PortfolioOptimizer

        optimizer = PortfolioOptimizer()
        rng = np.random.default_rng(7)
        df = pd.DataFrame(
            rng.normal(0, 0.01, size=(MIN_OBSERVATIONS, 3)), columns=["X", "Y", "Z"]
        )
        result = optimizer.hrp_allocation(df, random_state=42)
        # available may be True (success) or absent; it must NOT be False with INSUFFICIENT reason
        assert result.get("reason") != "INSUFFICIENT_RETURN_HISTORY", (
            "Boundary case of exactly MIN_OBSERVATIONS rows incorrectly rejected"
        )
