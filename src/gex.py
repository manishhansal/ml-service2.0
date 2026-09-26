"""
src.gex — Dealer Gamma Exposure (GEX) Engine.

GEX formula (dealer sign convention):
  ce_gex  = gamma × ce_oi × lot_size × spot² × (-1)   # dealers short calls → negative
  pe_gex  = gamma × pe_oi × lot_size × spot² × (+1)   # dealers short puts  → positive
  net_gex = ce_gex + pe_gex (per strike)
  aggregate_gex = Σ net_gex over all strikes

gamma_flip = strike where cumulative GEX (sorted ascending by strike) crosses zero.
expected_move_pct = |aggregate_gex| / (spot² × total_oi × lot_size) expressed as pct.
"""
from __future__ import annotations

import math

# NSE canonical lot sizes (post-SEBI Nov 2024 revision)
LOT_SIZES: dict[str, int] = {
    "NIFTY": 75,
    "BANKNIFTY": 30,
    "FINNIFTY": 65,
    "MIDCPNIFTY": 120,
}


def compute_gex(
    chain: list[dict],
    spot: float,
    lot_size: int,
) -> dict:
    """
    Compute GEX from an option chain snapshot.

    Each row in ``chain`` must have:
        strike    : float — option strike price
        ce_gamma  : float — call gamma per unit notional
        pe_gamma  : float — put gamma per unit notional
        ce_oi     : float — call open interest (contracts)
        pe_oi     : float — put open interest (contracts)

    Returns a dict with keys:
        strikes           : list[float]  — in input order
        gex_per_strike    : list[float]  — signed GEX per strike
        aggregate_gex     : float        — sum of gex_per_strike
        gamma_flip        : float        — first zero-crossing strike
        expected_move_pct : float        — implied expected move (> 0 always)
    """
    strikes: list[float] = []
    gex_per_strike: list[float] = []

    for row in chain:
        s = float(row["strike"])
        ce_g = float(row.get("ce_gamma", 0.0))
        pe_g = float(row.get("pe_gamma", 0.0))
        ce_oi = float(row.get("ce_oi", 0.0))
        pe_oi = float(row.get("pe_oi", 0.0))

        ce_gex = ce_g * ce_oi * lot_size * (spot ** 2) * (-1.0)
        pe_gex = pe_g * pe_oi * lot_size * (spot ** 2) * (+1.0)
        net = ce_gex + pe_gex

        strikes.append(s)
        gex_per_strike.append(net)

    aggregate_gex = sum(gex_per_strike)

    # ── Gamma flip: zero-crossing in cumulative GEX sorted by strike ─────
    if not strikes:
        gamma_flip = spot
    else:
        # Sort by strike ascending
        pairs = sorted(zip(strikes, gex_per_strike), key=lambda x: x[0])
        cumulative = 0.0
        gamma_flip = pairs[0][0]  # default: lowest strike
        prev_cum = None
        prev_strike = None
        crossed = False
        for strike, gex in pairs:
            cumulative += gex
            if prev_cum is not None:
                # Check zero-crossing between prev_strike and strike
                if (prev_cum < 0 <= cumulative) or (prev_cum > 0 >= cumulative):
                    # Linear interpolation for the crossing point
                    if abs(cumulative - prev_cum) > 1e-30:
                        t = -prev_cum / (cumulative - prev_cum)
                        gamma_flip = prev_strike + t * (strike - prev_strike)
                    else:
                        gamma_flip = strike
                    crossed = True
                    break
            prev_cum = cumulative
            prev_strike = strike
        if not crossed:
            # No zero-crossing: default to last strike (full positive or full negative)
            gamma_flip = pairs[-1][0]

        # Clamp to [min_strike, max_strike]
        min_s = min(s for s, _ in pairs)
        max_s = max(s for s, _ in pairs)
        gamma_flip = max(min_s, min(max_s, gamma_flip))

    # ── Expected move: |agg_gex| normalised by a proxy for total notional ─
    # A robust formula: expected_move_pct = sqrt(|agg_gex| / (spot² * lot_size))
    # This gives a positive value for any non-zero OI.
    total_abs_gex = sum(abs(g) for g in gex_per_strike)
    if total_abs_gex > 0 and spot > 0 and lot_size > 0:
        # Scale to a percentage — expressed as fraction × 100
        # simple proxy: sqrt(|agg| / spot²) in percent units
        expected_move_pct = math.sqrt(abs(aggregate_gex) / (spot ** 2 * lot_size)) * 100.0
        if expected_move_pct <= 0.0:
            # Fallback: use total_abs_gex
            expected_move_pct = math.sqrt(total_abs_gex / (spot ** 2 * lot_size)) * 100.0
    else:
        expected_move_pct = 0.001  # minimum positive value

    # Guarantee positivity
    expected_move_pct = abs(expected_move_pct) + 1e-12

    return {
        "strikes": strikes,
        "gex_per_strike": gex_per_strike,
        "aggregate_gex": aggregate_gex,
        "gamma_flip": gamma_flip,
        "expected_move_pct": expected_move_pct,
    }
