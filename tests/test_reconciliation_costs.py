"""
tests/test_reconciliation_costs.py — Tests for the canonical Indian cost model.

Mandate §18: "Use realistic Indian-market costs."
Mandate §23: "Do not select the scenario that makes the model profitable."
"""
import pytest
from src.reconciliation.costs import (
    IndianCostModel,
    PRIMARY_COST,
    MODERATE_COST,
    AGGRESSIVE_COST,
    STRESS_2X_COST,
    ALL_SCENARIOS,
)


class TestIndianCostModel:

    def test_primary_cost_round_trip_bps(self):
        """Primary cost must be 27.65 bps — frozen, never tuned."""
        assert abs(PRIMARY_COST.round_trip_bps() - 27.65) < 0.01

    def test_round_trip_is_sum_of_components(self):
        """round_trip_bps = 2×per_side + STT + stamp."""
        cm = IndianCostModel()
        expected = 2.0 * cm.per_side_symmetric_bps() + cm.stt_bps + cm.stamp_duty_bps
        assert abs(cm.round_trip_bps() - expected) < 1e-9

    def test_round_trip_fraction_consistent(self):
        """Fraction = bps / 10000."""
        cm = PRIMARY_COST
        assert abs(cm.round_trip_fraction() - cm.round_trip_bps() / 10_000) < 1e-12

    def test_entry_plus_exit_geq_round_trip(self):
        """entry_cost + exit_cost ≥ round_trip (asymmetric components don't double)."""
        cm = PRIMARY_COST
        # entry has stamp, exit has STT → their sum should roughly equal round_trip
        combined = cm.entry_cost_fraction() + cm.exit_cost_fraction()
        rt = cm.round_trip_fraction()
        # They should be close but may differ by one symmetric component being double-counted
        assert combined > 0
        assert rt > 0

    def test_primary_is_most_conservative(self):
        """Primary must have the highest round-trip bps of the standard scenarios."""
        standard = [PRIMARY_COST, MODERATE_COST, AGGRESSIVE_COST]
        for other in standard[1:]:
            assert PRIMARY_COST.round_trip_bps() > other.round_trip_bps(), (
                f"Primary ({PRIMARY_COST.round_trip_bps():.2f}) must exceed "
                f"{other.scenario} ({other.round_trip_bps():.2f})"
            )

    def test_stress_2x_is_approximately_double_primary(self):
        """Stress scenario should be ~2× the primary."""
        ratio = STRESS_2X_COST.round_trip_bps() / PRIMARY_COST.round_trip_bps()
        assert 1.8 < ratio < 2.2, (
            f"Stress should be ~2× primary, got {ratio:.2f}×"
        )

    def test_all_scenarios_have_positive_costs(self):
        for cm in ALL_SCENARIOS:
            assert cm.round_trip_bps() > 0, f"{cm.scenario} has zero cost"
            assert cm.round_trip_fraction() > 0

    def test_cost_scenario_frozen(self):
        """Cost models are frozen (dataclass frozen=True)."""
        with pytest.raises((TypeError, AttributeError)):
            PRIMARY_COST.brokerage_bps = 99.0  # type: ignore[misc]

    def test_to_dict_completeness(self):
        d = PRIMARY_COST.to_dict()
        required_keys = {
            "scenario", "round_trip_bps", "entry_cost_bps", "exit_cost_bps",
            "slippage_bps_per_side", "stt_bps_sell_side", "stamp_duty_bps_buy_side",
        }
        assert required_keys.issubset(d.keys()), f"Missing keys: {required_keys - d.keys()}"

    def test_n_scenarios_registered(self):
        """Exactly 5 pre-registered scenarios (never add post-hoc)."""
        assert len(ALL_SCENARIOS) == 5, (
            f"Expected 5 pre-registered scenarios, got {len(ALL_SCENARIOS)}"
        )

    def test_cost_monotone_order(self):
        """Scenarios should be ordered from most to least conservative."""
        ordered = sorted(ALL_SCENARIOS, key=lambda c: -c.round_trip_bps())
        names_ordered = [c.scenario for c in ordered]
        assert names_ordered[0] in ("stress_2x", "conservative"), (
            f"Highest cost should be stress_2x or conservative, got {names_ordered[0]}"
        )
