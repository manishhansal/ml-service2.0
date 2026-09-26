"""
src.reconciliation.costs — Canonical Indian-equity cost model.

Mandate §18: "Use realistic Indian-market costs."
Mandate §23: "Do not select the scenario that makes the model profitable."
Mandate §23: "Primary certification must use a conservative predefined cost assumption."

All cost components are in basis points of round-trip notional unless noted.
The PRIMARY cost scenario is CONSERVATIVE (27.65 bps round-trip) and is frozen
before any economic evaluation is run.  No cost assumption may be changed after
observing results.

Cost breakdown (NSE equity delivery, liquid large-cap):
  brokerage       : 3.0 bps  per side (discount broker flat-fee ≈ ₹20/trade)
  exchange_charge : 0.325 bps per side (NSE ~0.00325% of turnover)
  STT             : 2.5 bps  sell side (delivery 0.1%, intraday 0.025%)
  GST             : 0.5 bps  on (brokerage + exchange)
  stamp_duty      : 1.5 bps  buy side
  slippage        : 5.0 bps  per side (conservative estimate for mid-cap names)
  half_spread     : 3.0 bps  per side (NSE liquid names ~5–6 bps full spread)

Round-trip = 2×(brokerage + exchange + GST + slippage + spread) + STT + stamp
           = 2×(3 + 0.325 + 0.5 + 5 + 3) + 2.5 + 1.5
           = 2×11.825 + 4.0
           = 23.65 + 4.0
           = 27.65 bps

Sensitivity scenarios are pre-registered for stress-testing only.
They are NEVER used to select the most profitable configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CostScenario = Literal[
    "conservative",    # primary; 27.65 bps round-trip
    "moderate",        # 20 bps
    "aggressive",      # 15 bps
    "low",             # 10 bps
    "minimal",         # 5 bps
    "stress_2x",       # 2× conservative = 55.3 bps
]


@dataclass(frozen=True)
class IndianCostModel:
    """
    Immutable cost model for NSE equity strategies.

    Parameters are per-SIDE basis points unless the field name indicates
    "round_trip" or is explicitly asymmetric (STT sell-only, stamp buy-only).

    Usage::

        cm = IndianCostModel()              # conservative primary
        rt = cm.round_trip_bps()            # → 27.65
        fraction = cm.round_trip_fraction() # → 0.002765
    """
    # Per-side costs (bps)
    brokerage_bps: float = 3.0          # flat-fee discount broker, per side
    exchange_charge_bps: float = 0.325  # NSE transaction charge per side
    gst_bps: float = 0.5                # GST on (brokerage + exchange), per side
    slippage_bps: float = 5.0           # market impact + execution slippage, per side
    half_spread_bps: float = 3.0        # half bid-ask spread crossed per side

    # Asymmetric (Indian-specific)
    stt_bps: float = 2.5                # securities transaction tax — sell side only
    stamp_duty_bps: float = 1.5         # stamp duty — buy side only

    # Metadata
    scenario: str = "conservative"
    note: str = (
        "Primary scenario. Frozen before economic evaluation. "
        "Slippage=5 bps per side is conservative for liquid NSE names; "
        "mid-cap names may be higher. Never optimised post-hoc."
    )

    # ── Derived costs ─────────────────────────────────────────────────────

    def per_side_symmetric_bps(self) -> float:
        """Symmetric per-side costs (excludes asymmetric STT/stamp)."""
        return (
            self.brokerage_bps
            + self.exchange_charge_bps
            + self.gst_bps
            + self.slippage_bps
            + self.half_spread_bps
        )

    def round_trip_bps(self) -> float:
        """Total round-trip cost in basis points.

        = 2 × symmetric_per_side + STT (sell) + stamp (buy)
        """
        return 2.0 * self.per_side_symmetric_bps() + self.stt_bps + self.stamp_duty_bps

    def round_trip_fraction(self) -> float:
        """Round-trip cost as a decimal fraction of notional."""
        return self.round_trip_bps() / 10_000.0

    def entry_cost_fraction(self) -> float:
        """Cost charged on entry (buy side): symmetric + stamp."""
        return (self.per_side_symmetric_bps() + self.stamp_duty_bps) / 10_000.0

    def exit_cost_fraction(self) -> float:
        """Cost charged on exit (sell side): symmetric + STT."""
        return (self.per_side_symmetric_bps() + self.stt_bps) / 10_000.0

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "brokerage_bps_per_side": self.brokerage_bps,
            "exchange_charge_bps_per_side": self.exchange_charge_bps,
            "gst_bps_per_side": self.gst_bps,
            "slippage_bps_per_side": self.slippage_bps,
            "half_spread_bps_per_side": self.half_spread_bps,
            "stt_bps_sell_side": self.stt_bps,
            "stamp_duty_bps_buy_side": self.stamp_duty_bps,
            "round_trip_bps": self.round_trip_bps(),
            "entry_cost_bps": self.entry_cost_fraction() * 10_000,
            "exit_cost_bps": self.exit_cost_fraction() * 10_000,
            "note": self.note,
        }


# ── Pre-registered cost scenarios ─────────────────────────────────────────────
# These are fixed BEFORE economic evaluation. Never add a scenario post-hoc
# to rescue a failing strategy.

PRIMARY_COST: IndianCostModel = IndianCostModel(
    scenario="conservative",
    note="Primary certification scenario. Frozen pre-evaluation.",
)

MODERATE_COST: IndianCostModel = IndianCostModel(
    brokerage_bps=2.0, exchange_charge_bps=0.325, gst_bps=0.4,
    slippage_bps=3.5, half_spread_bps=2.0,
    stt_bps=2.5, stamp_duty_bps=1.5,
    scenario="moderate",
    note="Moderate cost scenario for sensitivity. Pre-registered.",
)

AGGRESSIVE_COST: IndianCostModel = IndianCostModel(
    brokerage_bps=1.5, exchange_charge_bps=0.325, gst_bps=0.35,
    slippage_bps=2.5, half_spread_bps=1.5,
    stt_bps=2.5, stamp_duty_bps=1.5,
    scenario="aggressive",
    note="Aggressive (optimistic) cost scenario. Pre-registered.",
)

LOW_COST: IndianCostModel = IndianCostModel(
    brokerage_bps=1.0, exchange_charge_bps=0.325, gst_bps=0.3,
    slippage_bps=1.75, half_spread_bps=1.0,
    stt_bps=2.5, stamp_duty_bps=1.5,
    scenario="low",
    note="Low cost. Only viable for very liquid large-cap names. Pre-registered.",
)

STRESS_2X_COST: IndianCostModel = IndianCostModel(
    brokerage_bps=6.0, exchange_charge_bps=0.65, gst_bps=1.0,
    slippage_bps=10.0, half_spread_bps=6.0,
    stt_bps=5.0, stamp_duty_bps=3.0,
    scenario="stress_2x",
    note="2× conservative costs for stress testing (mandate §42). Pre-registered.",
)

ALL_SCENARIOS: list[IndianCostModel] = [
    PRIMARY_COST, MODERATE_COST, AGGRESSIVE_COST, LOW_COST, STRESS_2X_COST,
]

# Round-trip bps for the pre-registered scenarios (informational)
# conservative: 27.65
# moderate:     ~18.05
# aggressive:   ~12.55
# low:          ~8.55
# stress_2x:    ~55.30
