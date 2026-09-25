"""
F&O / Derivatives & Options Feature Family — Phase 3D.

Missing data policy (CRITICAL)
-------------------------------
Every function returns NaN / None when input data is absent.
No silent substitutions:
  - Missing PCR → NaN, not 0.0 or 1.0
  - Missing ATM IV → NaN, not 0.0
  - Missing OI → NaN, not 0.0
  - Missing IV history → NaN rank, not 50.0

Data availability note
-----------------------
Historical Indian F&O data (PCR, OI, IV history) is DATA_UNAVAILABLE
in the offline test environment.  All functions return NaN in that case.
The feature registry documents this; the quality gate reports it.

OI build-up methodology
-----------------------
The four-quadrant classification (long buildup, short buildup, etc.)
is the standard NSE derivatives interpretation.  It is explicitly versioned
(derivatives-v1) so any methodology change creates a new version.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


# ── PCR ───────────────────────────────────────────────────────────────────────

def compute_pcr_score(pcr: Optional[float]) -> Optional[float]:
    """
    Normalise PCR to [-1, 1].
    PCR > 1.3 → +1 (bullish: PE writers dominant)
    PCR < 0.7 → -1 (bearish: CE writers dominant)

    Returns None when pcr is None — NEVER returns 0.0 for missing data.
    """
    if pcr is None or not math.isfinite(pcr):
        return None
    return float(np.clip((pcr - 1.0) / 0.5, -1.0, 1.0))


def compute_pcr_change(
    current_pcr: Optional[float],
    prev_pcr: Optional[float],
) -> Optional[float]:
    """
    PCR change: positive = shift toward more PE writing (bullish).
    Returns None when either input is absent.
    """
    if current_pcr is None or prev_pcr is None:
        return None
    if not math.isfinite(current_pcr) or not math.isfinite(prev_pcr):
        return None
    return float(current_pcr - prev_pcr)


# ── OI build-up ───────────────────────────────────────────────────────────────

# Methodology version: derivatives-v1
# Changing these thresholds or classification rules requires derivatives-v2.
_OI_BUILDUP_VERSION = "derivatives-v1"


def compute_oi_buildup_score(
    price_change_pct: Optional[float],
    oi_change_pct: Optional[float],
) -> tuple[Optional[float], Optional[str]]:
    """
    OI build-up quadrant classification.

    Returns (score, kind) or (None, None) when data is absent.
    NEVER returns (0.0, ...) for missing data.

    Quadrant classification (methodology: derivatives-v1)
    -----------------------------------------------------
    Long  Buildup  : price↑ + OI↑  → +1.0
    Short Covering  : price↑ + OI↓  → +0.6
    Short Buildup   : price↓ + OI↑  → -1.0
    Long  Unwinding : price↓ + OI↓  → -0.6
    """
    if price_change_pct is None or oi_change_pct is None:
        return None, None
    if not math.isfinite(price_change_pct) or not math.isfinite(oi_change_pct):
        return None, None

    if price_change_pct > 0 and oi_change_pct > 0:
        return 1.0, "LONG_BUILDUP"
    elif price_change_pct > 0 and oi_change_pct <= 0:
        return 0.6, "SHORT_COVERING"
    elif price_change_pct <= 0 and oi_change_pct > 0:
        return -1.0, "SHORT_BUILDUP"
    else:
        return -0.6, "LONG_UNWINDING"


# ── IV Rank ───────────────────────────────────────────────────────────────────

def compute_iv_rank(
    current_iv: Optional[float],
    iv_history: Optional[list[float]],
    period: int = 252,
    min_history: int = 5,
) -> Optional[float]:
    """
    IV Rank: where current IV sits in its N-day range [0, 100].
    0 = at the lowest IV seen in the period.
    100 = at the highest.

    Returns None when history is insufficient (< min_history bars).
    NEVER returns 50.0 as a neutral substitute for missing history.

    Parameters
    ----------
    current_iv  : Current implied volatility (float).
    iv_history  : List of past IV observations (oldest first).
    period      : How many historical bars to use.
    min_history : Minimum required history. Fewer → None.
    """
    if current_iv is None or not math.isfinite(current_iv):
        return None
    if not iv_history or len(iv_history) < min_history:
        return None

    history = [v for v in iv_history[-period:] if v is not None and math.isfinite(v)]
    if len(history) < min_history:
        return None

    iv_min = min(history)
    iv_max = max(history)
    if iv_max <= iv_min:
        return None   # Constant IV — rank is indeterminate, not 50

    return float(((current_iv - iv_min) / (iv_max - iv_min)) * 100.0)


def compute_iv_rank_with_status(
    current_iv: Optional[float],
    iv_history: Optional[list[float]],
    period: int = 252,
    min_history: int = 5,
) -> dict:
    """
    IV Rank with explicit data-quality status.

    Returns
    -------
    {
        "iv_rank":   float | None,
        "status":    "OK" | "INSUFFICIENT_HISTORY" | "CONSTANT_IV" | "MISSING_IV",
        "n_history": int,
    }
    """
    if current_iv is None or not math.isfinite(current_iv):
        return {"iv_rank": None, "status": "MISSING_IV", "n_history": 0}

    hist = [v for v in (iv_history or []) if v is not None and math.isfinite(v)]
    n    = len(hist)
    if n < min_history:
        return {"iv_rank": None, "status": "INSUFFICIENT_HISTORY", "n_history": n}

    window  = hist[-period:]
    iv_min  = min(window)
    iv_max  = max(window)
    if iv_max <= iv_min:
        return {"iv_rank": None, "status": "CONSTANT_IV", "n_history": len(window)}

    rank = ((current_iv - iv_min) / (iv_max - iv_min)) * 100.0
    return {"iv_rank": round(float(rank), 2), "status": "OK", "n_history": len(window)}


def compute_iv_percentile(
    current_iv: Optional[float],
    iv_history: Optional[list[float]],
    period: int = 252,
    min_history: int = 5,
) -> Optional[float]:
    """
    IV Percentile: % of past IV observations below current IV.
    More robust than IV Rank for tail events.
    Returns None when history is insufficient.
    """
    if current_iv is None or not math.isfinite(current_iv):
        return None
    hist = [v for v in (iv_history or []) if v is not None and math.isfinite(v)]
    if len(hist) < min_history:
        return None
    window = hist[-period:]
    below  = sum(1 for v in window if v < current_iv)
    return float(below / len(window) * 100.0)


# ── OI wall proximity ─────────────────────────────────────────────────────────

def compute_max_pain_distance(
    spot: Optional[float],
    max_pain: Optional[float],
) -> Optional[float]:
    """
    % distance from spot to max-pain strike.
    Positive: max pain above spot (bullish pull).
    Negative: max pain below spot (bearish pull).
    Returns None when either input is absent.
    """
    if spot is None or max_pain is None:
        return None
    if not math.isfinite(spot) or not math.isfinite(max_pain):
        return None
    if spot <= 0:
        return None
    return float(((max_pain - spot) / spot) * 100.0)


def compute_oi_wall_proximity(
    spot: Optional[float],
    max_ce_oi_strike: Optional[float],
    max_pe_oi_strike: Optional[float],
) -> dict[str, Optional[float]]:
    """
    Distance from spot to CE/PE OI walls.

    Returns a dict with keys:
        ce_wall_distance_pct : % above spot to CE wall
        pe_wall_distance_pct : % below spot to PE wall
        oi_wall_score        : [-1,1]; +1 = sitting on PE floor (bullish)

    Any value is None when its input is absent.
    NEVER substitutes 0.0 for missing strikes.
    """
    result: dict[str, Optional[float]] = {
        "ce_wall_distance_pct": None,
        "pe_wall_distance_pct": None,
        "oi_wall_score":        None,
    }
    if spot is None or not math.isfinite(spot) or spot <= 0:
        return result

    if max_ce_oi_strike is not None and math.isfinite(max_ce_oi_strike):
        result["ce_wall_distance_pct"] = float((max_ce_oi_strike - spot) / spot * 100.0)

    if max_pe_oi_strike is not None and math.isfinite(max_pe_oi_strike):
        result["pe_wall_distance_pct"] = float((spot - max_pe_oi_strike) / spot * 100.0)

    ce_d = result["ce_wall_distance_pct"]
    pe_d = result["pe_wall_distance_pct"]
    if ce_d is not None and pe_d is not None:
        total = abs(ce_d) + abs(pe_d)
        if total > 0:
            result["oi_wall_score"] = float(np.clip((abs(ce_d) - abs(pe_d)) / total, -1.0, 1.0))

    return result


def compute_options_flow_features(
    total_ce_oi: Optional[float],
    total_pe_oi: Optional[float],
    total_ce_oi_change: Optional[float],
    total_pe_oi_change: Optional[float],
    atm_iv: Optional[float],
) -> dict[str, Optional[float]]:
    """
    Composite options flow features from chain-level aggregates.

    Returns explicit None for every field where inputs are absent.
    NEVER substitutes pcr_oi=1.0 or atm_iv=0.0 for missing data.
    """
    features: dict[str, Optional[float]] = {}

    # PCR from OI
    if (total_pe_oi is not None and total_ce_oi is not None
            and math.isfinite(total_pe_oi) and math.isfinite(total_ce_oi)
            and total_ce_oi > 0):
        features["pcr_oi"] = float(total_pe_oi / total_ce_oi)
    else:
        features["pcr_oi"] = None  # DATA_UNAVAILABLE — not 1.0

    # OI delta skew
    ce_chg = total_ce_oi_change
    pe_chg = total_pe_oi_change
    if ce_chg is not None and pe_chg is not None:
        features["oi_delta_skew"] = float(pe_chg - ce_chg)
        total_chg = abs(ce_chg) + abs(pe_chg)
        if total_chg > 0:
            features["oi_delta_skew_norm"] = float((pe_chg - ce_chg) / total_chg)
        else:
            features["oi_delta_skew_norm"] = 0.0  # zero change in both = neutral
    else:
        features["oi_delta_skew"]      = None
        features["oi_delta_skew_norm"] = None

    # ATM IV — 0% IV is impossible; None means DATA_UNAVAILABLE
    if atm_iv is not None and math.isfinite(atm_iv) and atm_iv > 0:
        features["atm_iv"] = float(atm_iv)
    else:
        features["atm_iv"] = None  # DATA_UNAVAILABLE — not 0.0

    return features


# ── Expiry features ───────────────────────────────────────────────────────────

def compute_expiry_features(
    days_to_weekly_expiry: Optional[int],
    days_to_monthly_expiry: Optional[int],
    is_expiry_day: Optional[bool] = None,
) -> dict[str, Optional[float]]:
    """
    Options expiry proximity features.

    Returns explicit None for any field where input is absent.
    NEVER substitutes default values (5, 20, False) for missing data.

    The expiry calendar must come from the historical contract store
    (Phase 3B), not from current-schedule assumptions.
    """
    features: dict[str, Optional[float]] = {}

    features["is_expiry_day"] = (
        1.0 if is_expiry_day is True else (0.0 if is_expiry_day is False else None)
    )
    features["days_to_weekly_expiry"]  = (
        float(days_to_weekly_expiry) if days_to_weekly_expiry is not None else None
    )
    features["days_to_monthly_expiry"] = (
        float(days_to_monthly_expiry) if days_to_monthly_expiry is not None else None
    )

    # Theta pressure: 1 / max(days, 0.5)
    features["weekly_theta_pressure"] = (
        1.0 / max(float(days_to_weekly_expiry), 0.5)
        if days_to_weekly_expiry is not None else None
    )
    features["monthly_theta_pressure"] = (
        1.0 / max(float(days_to_monthly_expiry), 0.5)
        if days_to_monthly_expiry is not None else None
    )

    return features


# ── VIX features ──────────────────────────────────────────────────────────────

def compute_vix_features(
    current_vix: Optional[float],
    vix_history: Optional[list[float]] = None,
    min_history: int = 5,
) -> dict[str, Optional[float]]:
    """
    India VIX regime and derived features.

    Returns explicit None for any field where data is absent.
    NEVER substitutes vix_level=15.0, vix_regime=1.0, or vix_percentile=50.0.
    """
    features: dict[str, Optional[float]] = {
        "vix_level":          None,
        "vix_regime":         None,
        "vix_percentile":     None,
        "vix_change_pct":     None,
        "vix_mean_reversion": None,
    }

    if current_vix is None or not math.isfinite(current_vix):
        return features  # All None — no fabrication

    features["vix_level"] = float(current_vix)

    # Regime thresholds calibrated for India VIX historical range
    if current_vix < 13:
        features["vix_regime"] = 0.0
    elif current_vix < 18:
        features["vix_regime"] = 1.0
    elif current_vix < 25:
        features["vix_regime"] = 2.0
    else:
        features["vix_regime"] = 3.0

    hist = [v for v in (vix_history or []) if v is not None and math.isfinite(v)]
    if len(hist) >= min_history:
        window = hist[-252:]
        features["vix_percentile"] = float(
            sum(1 for v in window if v < current_vix) / len(window) * 100.0
        )
        if len(hist) >= 2:
            features["vix_change_pct"] = float(
                (current_vix - hist[-2]) / hist[-2] * 100.0
            ) if hist[-2] > 0 else None
        vix_mean = float(np.mean(window))
        vix_std  = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
        if vix_std > 0:
            features["vix_mean_reversion"] = float(
                np.clip((current_vix - vix_mean) / vix_std, -3.0, 3.0)
            )

    return features
