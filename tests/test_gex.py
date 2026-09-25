"""
Failing tests for the Dealer GEX (Gamma Exposure) Engine.

These tests are written BEFORE the implementation exists in ml-service/src/gex.py.
They will fail with ImportError until Task 5.2 (implementation) is complete.
This is the TDD red phase — all tests are expected to FAIL initially.

GEX formula (dealer sign convention):
  ce_gex  = gamma × ce_oi × lot_size × spot² × (-1)   # dealers short calls = negative
  pe_gex  = gamma × pe_oi × lot_size × spot² × (+1)   # dealers short puts  = positive
  net_gex = ce_gex + pe_gex
  aggregate_gex = Σ net_gex for all strikes

Gamma flip = strike where cumulative GEX crosses zero (sorted ascending by strike).
expected_move_pct > 0 for any valid chain snapshot.

Validates: Requirements 4.1, 4.2, 4.3, 4.6, 4.7
"""

from __future__ import annotations

import math
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Import-under-test
# This import WILL raise ImportError until src/gex.py is created.
# That is intentional — it confirms the TDD red phase.
# ---------------------------------------------------------------------------
from src.gex import LOT_SIZES, compute_gex  # noqa: E402  (fails until implemented)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NIFTY_SPOT = 23_000.0
NIFTY_LOT = 50   # canonical NSE lot size for NIFTY
BANKNIFTY_LOT = 15  # canonical NSE lot size for BANKNIFTY


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_row(
    strike: float,
    ce_gamma: float = 0.001,
    pe_gamma: float = 0.001,
    ce_oi: float = 10_000.0,
    pe_oi: float = 10_000.0,
) -> dict:
    """Return a minimal chain row compatible with compute_gex."""
    return {
        "strike": strike,
        "ce_gamma": ce_gamma,
        "pe_gamma": pe_gamma,
        "ce_oi": ce_oi,
        "pe_oi": pe_oi,
    }


def make_symmetric_chain(
    n_strikes: int = 10,
    spot: float = NIFTY_SPOT,
    step: float = 100.0,
    ce_gamma: float = 0.001,
    pe_gamma: float = 0.001,
    ce_oi: float = 10_000.0,
    pe_oi: float = 10_000.0,
) -> list[dict]:
    """
    Build a symmetric chain of *n_strikes* rows centred around *spot*.
    Strikes run from spot − (n//2)×step to spot + (n//2)×step.
    """
    half = n_strikes // 2
    return [
        make_row(
            strike=spot + (i - half) * step,
            ce_gamma=ce_gamma,
            pe_gamma=pe_gamma,
            ce_oi=ce_oi,
            pe_oi=pe_oi,
        )
        for i in range(n_strikes)
    ]


# ---------------------------------------------------------------------------
# Test 1 — GEX sign convention: positive CE-only gamma → negative aggregate
# ---------------------------------------------------------------------------

class TestGEXSignConvention:
    """
    Requirement 4.1: CE exposure adds with sign -1 (dealers short calls),
    PE exposure adds with sign +1 (dealers short puts).

    When all PE OI is zero, aggregate GEX is negative because CE GEX < 0.
    When all CE OI is zero, aggregate GEX is positive because PE GEX > 0.
    """

    def test_all_ce_oi_gives_negative_aggregate(self):
        """
        Given a chain where pe_oi == 0 for all strikes, the aggregate GEX
        must be strictly negative (dealers are net short calls → negative exposure).
        """
        chain = [
            make_row(strike=22900, ce_gamma=0.002, pe_gamma=0.002, ce_oi=50_000, pe_oi=0),
            make_row(strike=23000, ce_gamma=0.001, pe_gamma=0.001, ce_oi=45_000, pe_oi=0),
            make_row(strike=23100, ce_gamma=0.0015, pe_gamma=0.0015, ce_oi=40_000, pe_oi=0),
        ]
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)

        assert "aggregate_gex" in result, "result must contain 'aggregate_gex'"
        assert result["aggregate_gex"] < 0, (
            f"With all-CE OI and zero PE OI, aggregate_gex must be negative "
            f"(dealers short calls). Got {result['aggregate_gex']}"
        )

    def test_all_pe_oi_gives_positive_aggregate(self):
        """
        Given a chain where ce_oi == 0 for all strikes, the aggregate GEX
        must be strictly positive (dealers net short puts → positive exposure).
        """
        chain = [
            make_row(strike=22900, ce_gamma=0.002, pe_gamma=0.002, ce_oi=0, pe_oi=50_000),
            make_row(strike=23000, ce_gamma=0.001, pe_gamma=0.001, ce_oi=0, pe_oi=45_000),
            make_row(strike=23100, ce_gamma=0.0015, pe_gamma=0.0015, ce_oi=0, pe_oi=40_000),
        ]
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)

        assert result["aggregate_gex"] > 0, (
            f"With zero CE OI and all-PE OI, aggregate_gex must be positive "
            f"(dealers short puts). Got {result['aggregate_gex']}"
        )

    def test_per_strike_gex_sign_for_ce_only(self):
        """
        Each per-strike GEX value must be negative when pe_oi == 0.
        Validates the individual-strike sign convention, not just the sum.
        """
        chain = [
            make_row(strike=22800 + i * 100, ce_gamma=0.001, pe_gamma=0.001,
                     ce_oi=10_000, pe_oi=0)
            for i in range(5)
        ]
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)

        assert "gex_per_strike" in result, "result must contain 'gex_per_strike'"
        for gex_val in result["gex_per_strike"]:
            assert gex_val < 0, (
                f"Per-strike GEX must be negative when pe_oi == 0, got {gex_val}"
            )

    def test_per_strike_gex_sign_for_pe_only(self):
        """
        Each per-strike GEX value must be positive when ce_oi == 0.
        """
        chain = [
            make_row(strike=22800 + i * 100, ce_gamma=0.001, pe_gamma=0.001,
                     ce_oi=0, pe_oi=10_000)
            for i in range(5)
        ]
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)

        for gex_val in result["gex_per_strike"]:
            assert gex_val > 0, (
                f"Per-strike GEX must be positive when ce_oi == 0, got {gex_val}"
            )


# ---------------------------------------------------------------------------
# Test 2 — Gamma flip within [min_strike, max_strike]
# ---------------------------------------------------------------------------

class TestGammaFlipRange:
    """
    Requirement 4.2: the gamma flip strike must lie within the range
    [min_strike, max_strike] of the input chain.

    Design postcondition: `gammaFlip ∈ [min_strike, max_strike]`.
    """

    def test_gamma_flip_within_chain_range(self):
        """
        For a balanced chain the gamma flip must be within the chain's
        strike range — not extrapolated beyond the input data.
        """
        # Unequal CE/PE to ensure a real flip exists in the chain
        chain = [
            make_row(strike=22500, ce_gamma=0.003, pe_gamma=0.001, ce_oi=5_000, pe_oi=80_000),
            make_row(strike=22700, ce_gamma=0.002, pe_gamma=0.001, ce_oi=10_000, pe_oi=60_000),
            make_row(strike=22900, ce_gamma=0.001, pe_gamma=0.001, ce_oi=20_000, pe_oi=40_000),
            make_row(strike=23100, ce_gamma=0.001, pe_gamma=0.001, ce_oi=40_000, pe_oi=20_000),
            make_row(strike=23300, ce_gamma=0.001, pe_gamma=0.002, ce_oi=60_000, pe_oi=10_000),
            make_row(strike=23500, ce_gamma=0.001, pe_gamma=0.003, ce_oi=80_000, pe_oi=5_000),
        ]
        min_strike = min(r["strike"] for r in chain)
        max_strike = max(r["strike"] for r in chain)

        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)

        assert "gamma_flip" in result, "result must contain 'gamma_flip'"
        gamma_flip = result["gamma_flip"]

        assert min_strike <= gamma_flip <= max_strike, (
            f"gamma_flip {gamma_flip} must be within "
            f"[{min_strike}, {max_strike}]"
        )

    def test_gamma_flip_in_range_no_flip_defaults_to_boundary(self):
        """
        When cumulative GEX never crosses zero (all-CE chain), the gamma flip
        must still be within the chain's strike range — it should default to
        the first or last strike, not an out-of-range value.
        """
        chain = [
            make_row(strike=22900 + i * 100, ce_gamma=0.001, pe_gamma=0.001,
                     ce_oi=50_000, pe_oi=0)
            for i in range(10)
        ]
        min_strike = min(r["strike"] for r in chain)
        max_strike = max(r["strike"] for r in chain)

        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)
        gamma_flip = result["gamma_flip"]

        assert min_strike <= gamma_flip <= max_strike, (
            f"gamma_flip {gamma_flip} must be in [{min_strike}, {max_strike}] "
            f"even when no zero-crossing exists"
        )

    @pytest.mark.parametrize("n_strikes,step", [
        (10, 50.0),
        (20, 100.0),
        (30, 50.0),
    ])
    def test_gamma_flip_in_range_parametrised(self, n_strikes: int, step: float):
        """
        gamma_flip stays within chain bounds across different chain sizes and spacings.
        """
        chain = make_symmetric_chain(
            n_strikes=n_strikes,
            spot=NIFTY_SPOT,
            step=step,
            ce_gamma=0.0015,
            pe_gamma=0.0010,
            ce_oi=30_000,
            pe_oi=20_000,
        )
        min_strike = min(r["strike"] for r in chain)
        max_strike = max(r["strike"] for r in chain)

        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)

        assert min_strike <= result["gamma_flip"] <= max_strike, (
            f"gamma_flip out of bounds: {result['gamma_flip']} not in "
            f"[{min_strike}, {max_strike}]"
        )


# ---------------------------------------------------------------------------
# Test 3 — expected_move_pct > 0 for any valid chain snapshot
# ---------------------------------------------------------------------------

class TestExpectedMovePct:
    """
    Requirement 4.3 / design postcondition: expected_move_pct must be > 0
    for any valid chain snapshot with positive OI.
    """

    def test_expected_move_positive_for_balanced_chain(self):
        """
        A balanced chain with equal CE and PE OI must still yield a positive
        expected move percentage.
        """
        chain = make_symmetric_chain(n_strikes=12, spot=NIFTY_SPOT)
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)

        assert "expected_move_pct" in result, "result must contain 'expected_move_pct'"
        assert result["expected_move_pct"] > 0, (
            f"expected_move_pct must be > 0 for a valid chain, "
            f"got {result['expected_move_pct']}"
        )

    def test_expected_move_positive_ce_heavy_chain(self):
        """
        A CE-heavy chain (aggregate GEX < 0) must still give a positive expected move.
        """
        chain = [
            make_row(strike=22900 + i * 100, ce_oi=50_000, pe_oi=5_000)
            for i in range(15)
        ]
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)
        assert result["expected_move_pct"] > 0

    def test_expected_move_positive_pe_heavy_chain(self):
        """
        A PE-heavy chain (aggregate GEX > 0) must still give a positive expected move.
        """
        chain = [
            make_row(strike=22900 + i * 100, ce_oi=5_000, pe_oi=50_000)
            for i in range(15)
        ]
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)
        assert result["expected_move_pct"] > 0

    @given(
        ce_oi=st.floats(min_value=100.0, max_value=1_000_000.0,
                        allow_nan=False, allow_infinity=False),
        pe_oi=st.floats(min_value=100.0, max_value=1_000_000.0,
                        allow_nan=False, allow_infinity=False),
        gamma=st.floats(min_value=1e-6, max_value=0.1,
                        allow_nan=False, allow_infinity=False),
        spot=st.floats(min_value=1_000.0, max_value=100_000.0,
                       allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
    def test_expected_move_positive_property(
        self, ce_oi: float, pe_oi: float, gamma: float, spot: float
    ):
        """
        Property: expected_move_pct > 0 for any valid chain with positive OI.

        **Validates: Requirements 4.3**
        """
        chain = [
            make_row(strike=spot * (0.95 + 0.01 * i),
                     ce_gamma=gamma, pe_gamma=gamma,
                     ce_oi=ce_oi, pe_oi=pe_oi)
            for i in range(10)
        ]
        result = compute_gex(chain, spot=spot, lot_size=NIFTY_LOT)
        assert result["expected_move_pct"] > 0, (
            f"expected_move_pct must be > 0 for valid inputs, "
            f"got {result['expected_move_pct']} "
            f"(ce_oi={ce_oi}, pe_oi={pe_oi}, gamma={gamma}, spot={spot})"
        )


# ---------------------------------------------------------------------------
# Test 4 — Σ gex_per_strike == aggregate_gex to within 1e-6  (Property 6)
# ---------------------------------------------------------------------------

class TestAggregateConsistency:
    """
    Property 6 (GEX aggregate consistency):
    The aggregate GEX must equal the sum of all per-strike GEX values to
    within 1e-6 (floating-point rounding tolerance).

    Validates: Requirements 4.1
    **Validates: Requirements 4.1**
    """

    def test_sum_matches_aggregate_balanced_chain(self):
        """Σ gex_per_strike == aggregate_gex for a standard balanced chain."""
        chain = make_symmetric_chain(n_strikes=20, spot=NIFTY_SPOT)
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)

        per_strike_sum = sum(result["gex_per_strike"])
        assert abs(per_strike_sum - result["aggregate_gex"]) < 1e-6, (
            f"Σ gex_per_strike ({per_strike_sum}) != aggregate_gex "
            f"({result['aggregate_gex']}) by more than 1e-6"
        )

    def test_sum_matches_aggregate_ce_heavy(self):
        """Consistency holds for a CE-heavy (short-call dominant) chain."""
        chain = [
            make_row(strike=22900 + i * 100, ce_oi=80_000, pe_oi=5_000)
            for i in range(12)
        ]
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)
        per_strike_sum = sum(result["gex_per_strike"])

        assert abs(per_strike_sum - result["aggregate_gex"]) < 1e-6, (
            f"Aggregate inconsistency for CE-heavy chain: "
            f"sum={per_strike_sum}, aggregate={result['aggregate_gex']}"
        )

    def test_sum_matches_aggregate_pe_heavy(self):
        """Consistency holds for a PE-heavy (short-put dominant) chain."""
        chain = [
            make_row(strike=22900 + i * 100, ce_oi=5_000, pe_oi=80_000)
            for i in range(12)
        ]
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)
        per_strike_sum = sum(result["gex_per_strike"])

        assert abs(per_strike_sum - result["aggregate_gex"]) < 1e-6

    @given(
        n_strikes=st.integers(min_value=5, max_value=40),
        ce_oi=st.floats(min_value=100.0, max_value=500_000.0,
                        allow_nan=False, allow_infinity=False),
        pe_oi=st.floats(min_value=100.0, max_value=500_000.0,
                        allow_nan=False, allow_infinity=False),
        gamma=st.floats(min_value=1e-6, max_value=0.05,
                        allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_aggregate_consistency_property(
        self, n_strikes: int, ce_oi: float, pe_oi: float, gamma: float
    ):
        """
        Property 6: Σ gex_per_strike == aggregate_gex for any valid chain.

        **Validates: Requirements 4.1**
        """
        chain = [
            make_row(
                strike=NIFTY_SPOT * (0.9 + 0.02 * i),
                ce_gamma=gamma,
                pe_gamma=gamma,
                ce_oi=ce_oi,
                pe_oi=pe_oi,
            )
            for i in range(n_strikes)
        ]
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)
        per_strike_sum = sum(result["gex_per_strike"])

        assert abs(per_strike_sum - result["aggregate_gex"]) < 1e-6, (
            f"Aggregate inconsistency: sum={per_strike_sum}, "
            f"aggregate={result['aggregate_gex']} "
            f"(n={n_strikes}, ce_oi={ce_oi}, pe_oi={pe_oi}, gamma={gamma})"
        )

    def test_gex_per_strike_length_matches_chain(self):
        """
        The length of gex_per_strike must equal the number of rows in the
        input chain snapshot.
        """
        chain = make_symmetric_chain(n_strikes=15, spot=NIFTY_SPOT)
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)

        assert len(result["gex_per_strike"]) == len(chain), (
            f"gex_per_strike has {len(result['gex_per_strike'])} entries "
            f"but chain has {len(chain)} rows"
        )


# ---------------------------------------------------------------------------
# Test 5 — Lot sizes applied correctly (NIFTY=50, BANKNIFTY=15)
# ---------------------------------------------------------------------------

class TestLotSizes:
    """
    Requirement 4.1: per-strike GEX = gamma × OI × lot_size × spot².
    Providing a different lot_size must proportionally scale the output.
    The canonical NSE lot sizes must be exposed in LOT_SIZES.
    """

    def test_lot_sizes_dict_contains_nifty(self):
        """LOT_SIZES must expose NIFTY = 75  # post-SEBI Nov 2024."""
        assert "NIFTY" in LOT_SIZES, "LOT_SIZES dict must have a 'NIFTY' key"
        assert LOT_SIZES["NIFTY"] == 75, (
            f"LOT_SIZES['NIFTY'] must be 75 (post-Nov2024), got {LOT_SIZES['NIFTY']}"
        )

    def test_lot_sizes_dict_contains_banknifty(self):
        """LOT_SIZES must expose BANKNIFTY = 30  # post-SEBI Nov 2024."""
        assert "BANKNIFTY" in LOT_SIZES, "LOT_SIZES dict must have a 'BANKNIFTY' key"
        assert LOT_SIZES["BANKNIFTY"] == 30, (
            f"LOT_SIZES['BANKNIFTY'] must be 30 (post-Nov2024), got {LOT_SIZES['BANKNIFTY']}"
        )

    def test_lot_sizes_dict_contains_all_indices(self):
        """
        LOT_SIZES must contain all four canonical NSE index contracts
        (NIFTY=75, BANKNIFTY=30, FINNIFTY=65, MIDCPNIFTY=120).
        """
        expected = {"NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 65, "MIDCPNIFTY": 120}
        for symbol, lot in expected.items():
            assert symbol in LOT_SIZES, f"LOT_SIZES is missing '{symbol}'"
            assert LOT_SIZES[symbol] == lot, (
                f"LOT_SIZES['{symbol}'] expected {lot}, got {LOT_SIZES[symbol]}"
            )

    def test_nifty_vs_banknifty_lot_size_scaling(self):
        """
        The same chain run with NIFTY lot (50) vs BANKNIFTY lot (15) must
        produce GEX values that differ by a ratio of exactly 50/15.

        Note: must use an *asymmetric* chain so aggregate_gex != 0 (a
        symmetric chain with equal CE/PE OI yields aggregate_gex == 0,
        making the ratio undefined).
        """
        # CE-heavy chain ensures non-zero per-strike and aggregate GEX values
        chain = make_symmetric_chain(
            n_strikes=10, spot=NIFTY_SPOT, ce_oi=30_000, pe_oi=10_000
        )

        result_nifty = compute_gex(chain, spot=NIFTY_SPOT, lot_size=50)
        result_bnifty = compute_gex(chain, spot=NIFTY_SPOT, lot_size=15)

        ratio_agg = result_nifty["aggregate_gex"] / result_bnifty["aggregate_gex"]
        assert abs(ratio_agg - (50 / 15)) < 1e-9, (
            f"Expected ratio of aggregate GEX (NIFTY/BANKNIFTY) = "
            f"{50/15:.6f}, got {ratio_agg:.6f}"
        )

        for nifty_gex, bnifty_gex in zip(
            result_nifty["gex_per_strike"], result_bnifty["gex_per_strike"]
        ):
            ratio = nifty_gex / bnifty_gex
            assert abs(ratio - (50 / 15)) < 1e-9, (
                f"Per-strike GEX ratio mismatch: expected {50/15:.6f}, "
                f"got {ratio:.6f} (nifty={nifty_gex}, bnifty={bnifty_gex})"
            )

    def test_lot_size_linear_scaling(self):
        """
        Doubling the lot size must exactly double every per-strike GEX value
        and the aggregate GEX (lot_size is a linear multiplier in the formula).
        """
        chain = make_symmetric_chain(n_strikes=10, spot=NIFTY_SPOT)

        result_50 = compute_gex(chain, spot=NIFTY_SPOT, lot_size=50)
        result_100 = compute_gex(chain, spot=NIFTY_SPOT, lot_size=100)

        assert abs(result_100["aggregate_gex"] - 2 * result_50["aggregate_gex"]) < 1e-6, (
            "Doubling lot_size must double aggregate_gex"
        )
        for g50, g100 in zip(result_50["gex_per_strike"], result_100["gex_per_strike"]):
            assert abs(g100 - 2 * g50) < 1e-6, (
                f"Doubling lot_size must double per-strike GEX: "
                f"got {g100} vs 2×{g50}={2*g50}"
            )

    def test_formula_correctness_single_row(self):
        """
        Verify the per-strike GEX formula directly for a single-row chain.

        ce_gex = gamma × ce_oi × lot_size × spot² × (-1)
        pe_gex = gamma × pe_oi × lot_size × spot² × (+1)
        net    = ce_gex + pe_gex
        """
        gamma = 0.001
        ce_oi = 20_000.0
        pe_oi = 30_000.0
        spot = NIFTY_SPOT
        lot = NIFTY_LOT

        chain = [make_row(strike=spot, ce_gamma=gamma, pe_gamma=gamma,
                          ce_oi=ce_oi, pe_oi=pe_oi)]
        result = compute_gex(chain, spot=spot, lot_size=lot)

        expected_ce = gamma * ce_oi * lot * spot ** 2 * (-1)
        expected_pe = gamma * pe_oi * lot * spot ** 2 * (+1)
        expected_net = expected_ce + expected_pe

        assert abs(result["gex_per_strike"][0] - expected_net) < 1e-3, (
            f"Per-strike GEX formula mismatch: "
            f"expected {expected_net:.2f}, got {result['gex_per_strike'][0]:.2f}"
        )
        assert abs(result["aggregate_gex"] - expected_net) < 1e-3, (
            f"Aggregate GEX formula mismatch: "
            f"expected {expected_net:.2f}, got {result['aggregate_gex']:.2f}"
        )


# ---------------------------------------------------------------------------
# Test 6 — Result shape and required output keys
# ---------------------------------------------------------------------------

class TestResultShape:
    """
    The result dict from compute_gex must contain all required keys as per
    the GexResult model in the design document.
    """

    REQUIRED_KEYS = {
        "strikes",
        "gex_per_strike",
        "aggregate_gex",
        "gamma_flip",
        "expected_move_pct",
    }

    def test_result_contains_required_keys(self):
        """All required keys must be present in the return value."""
        chain = make_symmetric_chain(n_strikes=10, spot=NIFTY_SPOT)
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)

        missing = self.REQUIRED_KEYS - result.keys()
        assert not missing, f"compute_gex result is missing keys: {missing}"

    def test_strikes_list_matches_chain(self):
        """
        result['strikes'] must list the strike prices in the same order as the
        input chain rows.
        """
        chain = make_symmetric_chain(n_strikes=10, spot=NIFTY_SPOT)
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)

        expected_strikes = [row["strike"] for row in chain]
        assert result["strikes"] == expected_strikes, (
            f"result['strikes'] does not match input chain strikes. "
            f"Expected {expected_strikes}, got {result['strikes']}"
        )

    def test_gex_per_strike_is_list_of_floats(self):
        """gex_per_strike must be a list of numeric values."""
        chain = make_symmetric_chain(n_strikes=10, spot=NIFTY_SPOT)
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)

        assert isinstance(result["gex_per_strike"], list), (
            "gex_per_strike must be a list"
        )
        for val in result["gex_per_strike"]:
            assert isinstance(val, (int, float)), (
                f"gex_per_strike entry must be numeric, got {type(val)}"
            )

    def test_aggregate_gex_is_scalar(self):
        """aggregate_gex must be a scalar numeric value."""
        chain = make_symmetric_chain(n_strikes=10, spot=NIFTY_SPOT)
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)

        assert isinstance(result["aggregate_gex"], (int, float)), (
            f"aggregate_gex must be a scalar, got {type(result['aggregate_gex'])}"
        )

    def test_gamma_flip_is_scalar(self):
        """gamma_flip must be a scalar numeric value."""
        chain = make_symmetric_chain(n_strikes=10, spot=NIFTY_SPOT)
        result = compute_gex(chain, spot=NIFTY_SPOT, lot_size=NIFTY_LOT)

        assert isinstance(result["gamma_flip"], (int, float)), (
            f"gamma_flip must be a scalar, got {type(result['gamma_flip'])}"
        )
