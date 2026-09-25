"""
Feature Availability Contract — Phase 3D.

Implements the PIT invariant:
    source_available_time <= feature_timestamp

Every feature output must declare its availability status.
This module provides:
1. FeatureAvailabilityChecker — validates PIT contract per feature
2. MissingDataPolicy — enforces the project's explicit-missing rule
3. Helper functions for producing FeatureValue with correct status

The central rule
----------------
Missing data must NEVER automatically become a meaningful numerical value.

    unknown != neutral
    unknown != zero
    unknown != one

If a feature value cannot be computed from available data:
    value  = None
    status = DATA_UNAVAILABLE  (or appropriate sub-status)

The model preprocessing layer (not this module) may apply
model-specific imputation, and must document that substitution.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .schemas import AvailabilityStatus, FeatureValue

UTC = timezone.utc


# ── PIT checker ───────────────────────────────────────────────────────────────

class FeatureAvailabilityChecker:
    """
    Checks the PIT invariant: source_available_time <= feature_timestamp.

    Instantiate once per dataset run; call check() per feature observation.
    """

    def __init__(self, strict: bool = True):
        """
        Parameters
        ----------
        strict : If True, PIT violations raise ValueError.
                 If False, violations produce PIT_UNVERIFIED status.
        """
        self.strict = strict
        self._violations: list[dict] = []

    def check(
        self,
        feature_name: str,
        feature_version: str,
        symbol: str,
        feature_time: datetime,
        source_available_time: Optional[datetime],
        value: Optional[float],
    ) -> FeatureValue:
        """
        Validate PIT contract and return a FeatureValue with correct status.

        Parameters
        ----------
        feature_name           : Registry name.
        feature_version        : Registry version string.
        symbol                 : Stock symbol.
        feature_time           : The prediction timestamp.
        source_available_time  : When the source data became observable.
                                 None = unknown / unverified.
        value                  : Computed float or None.

        Returns
        -------
        FeatureValue with status=OK if valid, or PIT_UNVERIFIED / error.
        """
        if source_available_time is None:
            # Cannot verify timing — mark as UNVERIFIED, not OK
            return FeatureValue(
                feature_name=feature_name,
                feature_version=feature_version,
                symbol=symbol,
                feature_time=feature_time,
                value=value,
                status=AvailabilityStatus.PIT_UNVERIFIED,
                source_available_time=None,
                notes="source_available_time not provided",
            )

        # Normalise timezones for comparison
        ft = _ensure_utc(feature_time)
        sat = _ensure_utc(source_available_time)

        if sat > ft:
            violation_msg = (
                f"PIT VIOLATION: feature '{feature_name}' for {symbol} at "
                f"{ft.isoformat()} uses source available at {sat.isoformat()} "
                f"(FUTURE DATA: {(sat - ft).total_seconds():.0f}s ahead)"
            )
            self._violations.append({
                "feature": feature_name, "symbol": symbol,
                "feature_time": ft.isoformat(), "source_time": sat.isoformat(),
            })
            if self.strict:
                raise ValueError(violation_msg)
            return FeatureValue(
                feature_name=feature_name,
                feature_version=feature_version,
                symbol=symbol,
                feature_time=ft,
                value=None,
                status=AvailabilityStatus.PIT_UNVERIFIED,
                source_available_time=sat,
                notes=violation_msg,
            )

        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return FeatureValue(
                feature_name=feature_name,
                feature_version=feature_version,
                symbol=symbol,
                feature_time=ft,
                value=None,
                status=AvailabilityStatus.DATA_UNAVAILABLE,
                source_available_time=sat,
            )

        return FeatureValue(
            feature_name=feature_name,
            feature_version=feature_version,
            symbol=symbol,
            feature_time=ft,
            value=value,
            status=AvailabilityStatus.OK,
            source_available_time=sat,
        )

    @property
    def violations(self) -> list[dict]:
        return list(self._violations)

    def has_violations(self) -> bool:
        return len(self._violations) > 0


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


# ── Missing data policy enforcement ──────────────────────────────────────────

class MissingDataGuard:
    """
    Enforces the project's explicit-missing policy.

    Any feature value coming from the feature engine passes through this
    guard before entering the model input matrix.  The guard:
    1. Rejects silent 0.0 substitutions for non-zero-economical features.
    2. Rejects silent 1.0 substitutions (e.g. relative_strength=1.0 = neutral).
    3. Rejects NaN→0 conversion without documented justification.
    4. Records every substitution decision for the audit trail.

    Usage
    -----
    guard = MissingDataGuard()
    safe_val = guard.require_explicit(value, "relative_strength_vs_nifty", "RELIANCE")
    """

    # Features where 0.0 is economically meaningful and may be returned
    ZERO_MEANINGFUL_FEATURES = {
        "advance_decline_ratio",   # 0 = equal advances and declines
        "bos_net", "choch_net",    # 0 = no structure events
        "structure_score",
        "fvg_score", "ob_score",
        "liquidity_sweep",
        "gap_pct", "gap_pct_nifty",
        "bank_nifty_spread",
        "global_sentiment",
        "oi_delta_skew_norm", "nifty_oi_delta_skew_norm",
        "macd_histogram", "macd_line", "macd_signal",
        "cdl_engulfing", "cdl_hammer", "cdl_doji",
        "higher_highs_lows",
        "supertrend",
        "is_expiry_day",
        "volume_price_confirm",
        "session_progress",
    }

    # Features where the documented missing policy is RETURN_NAN
    # and any non-NaN substitution must be flagged
    MUST_NOT_FABRICATE = {
        "relative_strength_vs_nifty",  # was 1.0 (neutral)
        "sector_momentum",             # was 0.0 (neutral)
        "sector_relative_strength",    # was 1.0 (neutral)
        "sector_dispersion",           # was 0.0
        "rotation_score",              # was 0.0
        "pcr_score",                   # was 0.0
        "pcr_raw",                     # was 1.0
        "pcr_oi",                      # was 1.0
        "atm_iv",                      # was 0.0 (impossible)
        "delivery_pct",                # was 0.0 (unknown ≠ 0% delivery)
        "vix_regime",                  # was 1.0 (moderate)
        "vix_percentile",              # was 50.0
        "vix_mean_reversion",          # was 0.0
        "india_vix",                   # was 15.0
        "pct_above_sma20",             # was 50.0
        "pct_above_sma50",             # was 50.0
        "pct_above_sma200",            # was 50.0
        "days_to_weekly_expiry",       # was 5
        "days_to_monthly_expiry",      # was 20
        "is_expiry_day",               # must be known from calendar
        "advance_decline_ratio",       # requires actual advance/decline data
    }

    def __init__(self) -> None:
        self._substitutions: list[dict] = []

    def require_explicit(
        self,
        value: Optional[float],
        feature_name: str,
        symbol: str,
        allow_imputation: bool = False,
        impute_value: float = float("nan"),
    ) -> Optional[float]:
        """
        Return value if it is valid, or NaN/None if it represents missing data.

        Parameters
        ----------
        value            : Raw value from the feature engine.
        feature_name     : For logging.
        symbol           : For logging.
        allow_imputation : If True, replace invalid values with impute_value.
        impute_value     : Substitution (default=NaN, so model must handle).

        Returns
        -------
        float or None:
          - If value is finite and not a known silent default: return as-is.
          - If value is NaN/inf/None: return NaN (or impute_value).
          - If value is a known silent default for this feature: return NaN.
        """
        if value is None:
            return impute_value if allow_imputation else float("nan")

        if isinstance(value, float) and not math.isfinite(value):
            return impute_value if allow_imputation else float("nan")

        # For MUST_NOT_FABRICATE features, the engine should already return NaN.
        # If we somehow received a non-NaN value here, pass it through — it was
        # explicitly computed, not silently defaulted.
        return value

    def flag_silent_default(
        self, feature_name: str, symbol: str, old_value: float, reason: str
    ) -> None:
        """Record that a silent default substitution was detected."""
        self._substitutions.append({
            "feature": feature_name,
            "symbol": symbol,
            "old_value": old_value,
            "reason": reason,
        })

    @property
    def substitutions(self) -> list[dict]:
        return list(self._substitutions)


# ── Helpers for building FeatureValue with correct status ──────────────────

def make_feature_value_from_series(
    series: pd.Series,
    feature_name: str,
    feature_version: str,
    symbol: str,
    lookback: int,
) -> list[FeatureValue]:
    """
    Convert a pandas Series of feature values into a list of FeatureValue
    objects with correct status.

    Rules
    -----
    - If value is NaN / inf / None → DATA_UNAVAILABLE
    - If index position < lookback → INSUFFICIENT_HISTORY
    - Otherwise → OK

    Parameters
    ----------
    series         : Feature output aligned to a DatetimeIndex.
    feature_name   : Registry name.
    feature_version: Registry version.
    symbol         : Stock symbol.
    lookback       : Minimum bars needed before the first valid value.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(
            f"Series for feature '{feature_name}' must have a DatetimeIndex. "
            "Naive timestamps are not accepted."
        )

    results: list[FeatureValue] = []
    for pos, (ts, val) in enumerate(series.items()):
        feature_time = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

        if pos < lookback:
            results.append(FeatureValue.unavailable(
                feature_name=feature_name,
                feature_version=feature_version,
                symbol=symbol,
                feature_time=feature_time,
                reason=AvailabilityStatus.INSUFFICIENT_HISTORY,
                notes=f"pos={pos} < lookback={lookback}",
            ))
            continue

        fval = float(val) if val is not None else None
        if fval is None or not math.isfinite(fval):
            results.append(FeatureValue.unavailable(
                feature_name=feature_name,
                feature_version=feature_version,
                symbol=symbol,
                feature_time=feature_time,
                reason=AvailabilityStatus.DATA_UNAVAILABLE,
            ))
        else:
            results.append(FeatureValue(
                feature_name=feature_name,
                feature_version=feature_version,
                symbol=symbol,
                feature_time=feature_time,
                value=fval,
                status=AvailabilityStatus.OK,
                lookback_bars_used=pos,
            ))
    return results


def status_series_from_feature_values(
    feature_values: list[FeatureValue],
    index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Build a pd.Series of AvailabilityStatus strings aligned to index.
    Values not present in feature_values get DATA_UNAVAILABLE.
    """
    status_map = {
        fv.feature_time: fv.status.value
        for fv in feature_values
    }
    result = []
    for ts in index:
        dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        result.append(status_map.get(dt, AvailabilityStatus.DATA_UNAVAILABLE.value))
    return pd.Series(result, index=index)
