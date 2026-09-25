"""
Feature Governance Schemas — Phase 3D.

Defines the canonical data structures for feature specification,
availability status, and feature output with provenance.

Design principles
-----------------
1. Every feature has an explicit availability status.  Missing data is
   NEVER silently converted to a neutral numerical value (0.0, 1.0, 50.0).
2. Every feature carries its computation timestamp and source timestamp so
   the PIT invariant (source_available_time <= feature_timestamp) can be
   verified by the leakage validator.
3. Feature versions are immutable identifiers.  Changing formula, lookback,
   normalization, or missing-data semantics requires a new version string.
4. The FeatureSpec is the authoritative contract between the feature engine
   and model consumers.  Models declare which FeatureSpec they consume;
   the engine validates compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ── Enumerations ──────────────────────────────────────────────────────────────

class FeatureFamily(str, Enum):
    """Top-level economic family classification."""
    MOMENTUM          = "momentum"
    TREND             = "trend"
    MEAN_REVERSION    = "mean_reversion"
    VOLATILITY        = "volatility"
    VOLUME_LIQUIDITY  = "volume_liquidity"
    MARKET_STRUCTURE  = "market_structure"
    CROSS_SECTIONAL   = "cross_sectional"
    BREADTH           = "breadth"
    SECTOR            = "sector"
    DERIVATIVES       = "derivatives"
    OPTIONS_IV        = "options_iv"
    EXPIRY_CONTRACT   = "expiry_contract"
    MARKET_REGIME     = "market_regime"
    INTERMARKET       = "intermarket"
    MICROSTRUCTURE    = "microstructure"
    RISK_TAIL         = "risk_tail"
    TIME_OF_DAY       = "time_of_day"
    OVERNIGHT_GAP     = "overnight_gap"


class AvailabilityStatus(str, Enum):
    """
    Explicit availability classification for a feature value.

    These states must NEVER be collapsed to a single float without
    documenting the substitution.  The model preprocessing layer may
    apply family-specific imputation, but the feature engine must surface
    the correct status.
    """
    OK                   = "OK"           # value is valid and PIT-safe
    DATA_UNAVAILABLE     = "DATA_UNAVAILABLE"  # source data does not exist
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"  # lookback window not yet filled
    PROVIDER_MISSING     = "PROVIDER_MISSING"   # data provider returned nothing
    STALE                = "STALE"        # source is older than expected cadence
    PIT_UNVERIFIED       = "PIT_UNVERIFIED"  # cannot confirm source_available_time <= feature_time
    INVALID_INPUT        = "INVALID_INPUT"   # upstream data was malformed (zero price, NaN OHLCV)
    PROXY                = "PROXY"        # value is a best-available proxy, not the true measure


class PITSafety(str, Enum):
    """
    Point-in-time safety classification.
    """
    SAFE       = "SAFE"        # proven causal; source_available_time <= feature_time
    UNVERIFIED = "UNVERIFIED"  # timing contract not established
    UNSAFE     = "UNSAFE"      # future data confirmed to be used (BLOCKED)


class FeaturePromotion(str, Enum):
    """
    Feature lifecycle state.  Only PRODUCTION_CANDIDATE features may
    enter an official training dataset.
    """
    EXPERIMENTAL        = "EXPERIMENTAL"       # new, untested
    RESEARCH            = "RESEARCH"           # reviewed, no OOS evidence yet
    VALIDATED           = "VALIDATED"          # OOS IC computed and stable
    PRODUCTION_CANDIDATE = "PRODUCTION_CANDIDATE"  # all gates passed
    DEPRECATED          = "DEPRECATED"         # no longer used
    BLOCKED             = "BLOCKED"            # failed leakage or data test


class MissingPolicy(str, Enum):
    """
    Declared policy for handling missing source data.
    Must match the actual implementation in the feature function.
    """
    RETURN_NAN           = "RETURN_NAN"       # value = NaN, status = DATA_UNAVAILABLE
    RETURN_ZERO          = "RETURN_ZERO"      # only valid when 0 is economically meaningful
    RETURN_ONE           = "RETURN_ONE"       # only valid when 1.0 is economically meaningful
    RETURN_NEUTRAL       = "RETURN_NEUTRAL"   # explicit neutral substitution (must be documented)
    PROPAGATE_NAN        = "PROPAGATE_NAN"    # NaN flows through all derived features
    RAISE                = "RAISE"            # raise ValueError; caller must handle


class NormalizationPolicy(str, Enum):
    """
    Declared normalization applied to the raw feature value.
    """
    RAW                  = "RAW"           # no normalization
    PERCENT              = "PERCENT"       # expressed as %
    Z_SCORE_ROLLING      = "Z_SCORE_ROLLING"  # (x - rolling_mean) / rolling_std
    PERCENTILE_ROLLING   = "PERCENTILE_ROLLING"  # rolling empirical percentile [0,100]
    MINMAX_ROLLING       = "MINMAX_ROLLING"  # rolling min-max [0,1]
    CROSS_SECTIONAL_RANK = "CROSS_SECTIONAL_RANK"  # rank within eligible universe at t
    CROSS_SECTIONAL_ZSCORE = "CROSS_SECTIONAL_ZSCORE"  # z-score within universe at t
    CLIPPED              = "CLIPPED"       # clipped to fixed range
    BINARY               = "BINARY"        # {0, 1}
    SIGNED               = "SIGNED"        # {-1, 0, +1}


# ── Core specification ────────────────────────────────────────────────────────

@dataclass
class FeatureSpec:
    """
    Canonical specification for a single feature.

    This is the contract between the feature engine and all consumers
    (model training, validation, live inference).  Every field has a
    documented meaning; no field may be omitted or defaulted silently.

    Versioning rule
    ---------------
    Changing any of: formula_id, lookback, source, normalization_policy,
    missing_policy, or pit_safety requires bumping feature_version.
    The old version must be DEPRECATED, never silently overwritten.
    """
    # Identity
    feature_name:    str
    feature_version: str          # e.g. "momentum-v1", "volatility-v2"
    family:          FeatureFamily
    description:     str

    # Source
    source:           str         # e.g. "equity_ohlcv", "nse_derivatives", "india_vix"
    formula_id:       str         # short stable identifier for the formula
    required_columns: list[str]   # e.g. ["high", "low", "close"]
    lookback:         int         # minimum bars of source history required
    bar_frequency:    str         # e.g. "1D", "5min"

    # PIT contract
    causal:           bool        # True = only past data used in formula
    pit_safety:       PITSafety
    cross_sectional:  bool        # True = ranked/normalised across universe at t
    market_scope:     str         # e.g. "single_stock", "nifty500", "nifty50"

    # Policies
    missing_policy:       MissingPolicy
    normalization_policy: NormalizationPolicy

    # Lifecycle
    promotion:            FeaturePromotion = FeaturePromotion.EXPERIMENTAL
    deprecated_by:        Optional[str] = None   # version string of replacement
    deprecation_note:     str = ""

    # Optional metadata
    economic_rationale:   str = ""
    known_limitations:    str = ""
    proxy_for:            str = ""  # what real market quantity this proxies (if PROXY)
    tags:                 list[str] = field(default_factory=list)


# ── Feature output with provenance ───────────────────────────────────────────

@dataclass
class FeatureValue:
    """
    A single computed feature value with full provenance.

    This is what the feature engine returns per feature per observation.
    Models consume the `value` field; the governance layer validates the
    rest.  The `status` field is the authoritative signal for whether the
    value may be used or must be treated as missing.
    """
    feature_name:    str
    feature_version: str
    symbol:          str
    feature_time:    datetime   # the prediction timestamp this feature is for
    value:           Optional[float]          # None when status != OK
    status:          AvailabilityStatus

    # PIT provenance
    source_event_time:     Optional[datetime] = None  # when the source event occurred
    source_available_time: Optional[datetime] = None  # when source became observable

    # Diagnostics
    lookback_bars_used: Optional[int] = None
    notes:              str = ""

    def is_usable(self) -> bool:
        """Return True only when value is safe to use as a model input."""
        return (
            self.status == AvailabilityStatus.OK
            and self.value is not None
            and self._is_finite(self.value)
        )

    @staticmethod
    def _is_finite(v: float) -> bool:
        import math
        return math.isfinite(v)

    @classmethod
    def unavailable(
        cls,
        feature_name: str,
        feature_version: str,
        symbol: str,
        feature_time: datetime,
        reason: AvailabilityStatus = AvailabilityStatus.DATA_UNAVAILABLE,
        notes: str = "",
    ) -> "FeatureValue":
        """Factory for DATA_UNAVAILABLE / missing values.  Never returns 0."""
        return cls(
            feature_name=feature_name,
            feature_version=feature_version,
            symbol=symbol,
            feature_time=feature_time,
            value=None,
            status=reason,
            notes=notes,
        )


@dataclass
class FeatureRow:
    """
    Full feature vector for one symbol at one timestamp.

    Contains both the flat value dict (for model.predict) and the
    status dict (for governance/validation).  The two dicts share keys.

    Construction rule
    -----------------
    feature_values[k] may be NaN only if status_map[k] != OK.
    If status_map[k] == OK, feature_values[k] must be a finite float.
    """
    symbol:          str
    feature_time:    datetime
    feature_set_id:  str         # identifies which FeatureSpec set was used
    feature_version: str         # version of the feature set

    # Primary outputs
    feature_values: dict[str, Optional[float]]   # NaN-safe; None = unavailable
    status_map:     dict[str, AvailabilityStatus]

    # Provenance
    universe_version:    str = ""
    dataset_snapshot_id: str = ""
    git_commit:          str = ""

    def to_model_input(
        self,
        feature_names: list[str],
        impute_unavailable: bool = False,
        impute_value: float = float("nan"),
    ) -> list[float]:
        """
        Extract a model-ready list of floats in the declared feature order.

        Parameters
        ----------
        feature_names     : Ordered list from FeatureSetSpec.feature_names.
        impute_unavailable: If True, replace unavailable values with
                            `impute_value` (NaN by default).
                            The model's preprocessor must handle NaN.
        impute_value      : Substitution when status != OK and impute=True.

        Raises
        ------
        KeyError if a declared feature is not in this row's values.
        ValueError if status is OK but value is None or non-finite (bug).
        """
        import math
        result = []
        for fname in feature_names:
            if fname not in self.feature_values:
                raise KeyError(
                    f"Feature '{fname}' not found in FeatureRow for "
                    f"{self.symbol} at {self.feature_time}"
                )
            status = self.status_map.get(fname, AvailabilityStatus.DATA_UNAVAILABLE)
            val = self.feature_values[fname]

            if status == AvailabilityStatus.OK:
                if val is None or not math.isfinite(val):
                    raise ValueError(
                        f"Feature '{fname}' has status=OK but value={val!r} "
                        f"for {self.symbol} at {self.feature_time}"
                    )
                result.append(val)
            else:
                if impute_unavailable:
                    result.append(impute_value)
                else:
                    result.append(float("nan"))
        return result

    def availability_summary(self) -> dict[str, int]:
        """Return counts by AvailabilityStatus."""
        counts: dict[str, int] = {}
        for s in self.status_map.values():
            counts[s.value] = counts.get(s.value, 0) + 1
        return counts


# ── Feature set specification ─────────────────────────────────────────────────

@dataclass
class FeatureSetSpec:
    """
    Declares an ordered, versioned set of features consumed by one model.

    A model must explicitly declare its FeatureSetSpec.  The engine
    validates that every declared feature exists in the registry and that
    all required features have OK status before building the matrix.

    Versioning rule
    ---------------
    Adding, removing, or reordering features requires a new set_version.
    Models pin to a specific set_version; the training pipeline validates
    this at dataset-build time.
    """
    set_id:          str         # e.g. "RANKING_SET", "REGIME_SET"
    set_version:     str         # e.g. "v1", "v2"
    feature_names:   list[str]   # ordered; matches RANKING_FEATURES etc.
    required_families: list[FeatureFamily]

    # Minimum availability requirement to proceed with training
    min_ok_fraction:  float = 0.80   # at least 80% of features must be OK
    allow_imputation: bool = False   # if False, NaN in model input raises


# ── Leakage certification ─────────────────────────────────────────────────────

@dataclass
class LeakageCertification:
    """
    Result of the feature leakage certification sweep.

    Produced by FeatureLeakageValidator at the end of Phase 3D (and on
    every subsequent feature change).  Must be stored alongside the
    dataset snapshot.
    """
    certified_at:           datetime
    total_features:         int
    causal_features:        int
    non_causal_features:    int
    unverified_features:    int
    failed_features:        int
    blocked_features:       int
    future_dependency_paths: list[str]  # code paths that use future data
    pit_violations:         list[str]   # features where source_available > feature_time
    verdict:                str         # "PASS" | "FAIL" | "CONDITIONAL"
    notes:                  str = ""
