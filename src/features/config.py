"""
Feature Configuration — Phase 3D.

Central configuration for the feature engine.  All lookback windows,
thresholds, and version strings live here so that changing a parameter
is a single-source edit that also bumps the relevant version identifier.

Versioning contract
-------------------
Changing any default in a FeatureFamilyConfig produces a new
family_version string.  The feature_set_version is the hash of all
active family versions; it bumps automatically.

This module MUST NOT import from labels/, training/, or models/.
Features must remain independent of label outcomes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(frozen=True)
class MomentumConfig:
    """Configuration for the Momentum / Relative-Strength family."""
    family_version:  str = "momentum-v1"
    return_periods:  tuple[int, ...] = (1, 2, 3, 5, 10, 20, 60)
    roc_period:      int = 12
    rs_period:       int = 20     # relative strength vs benchmark
    trend_strength_period: int = 20
    breakout_lookback: int = 20
    distance_from_high_period: int = 252  # 52-week on daily bars
    distance_from_low_period:  int = 252
    hhhl_period: int = 5


@dataclass(frozen=True)
class TrendConfig:
    """Configuration for the Trend family (EMA stack, ADX, Supertrend)."""
    family_version: str = "trend-v1"
    ema_periods: tuple[int, ...] = (8, 13, 21, 55, 200)
    adx_period:  int = 14
    supertrend_period:     int = 10
    supertrend_multiplier: float = 3.0
    macd_fast:   int = 12
    macd_slow:   int = 26
    macd_signal: int = 9


@dataclass(frozen=True)
class MeanReversionConfig:
    """Configuration for Mean Reversion family."""
    family_version:  str = "mean_reversion-v1"
    bb_period:       int = 20
    bb_std:          float = 2.0
    rsi_period:      int = 14
    rsi_fast_period: int = 7
    stoch_period:    int = 14
    williams_period: int = 14
    cci_period:      int = 20
    zscore_period:   int = 20      # rolling z-score of price


@dataclass(frozen=True)
class VolatilityConfig:
    """Configuration for the Volatility family."""
    family_version:   str = "volatility-v1"
    atr_period:       int = 14
    atr_fast_period:  int = 5
    atr_slow_period:  int = 20
    realized_vol_period: int = 20  # close-to-close
    parkinson_period: int = 20     # Parkinson high-low estimator
    vol_percentile_windows: tuple[int, ...] = (20, 60, 120)
    annualization_factor: float = 252.0   # trading days per year


@dataclass(frozen=True)
class VolumeLiquidityConfig:
    """Configuration for Volume / Liquidity family."""
    family_version:      str = "volume_liquidity-v1"
    relative_vol_period: int = 20
    vol_trend_period:    int = 10
    vol_breakout_thresh: float = 1.5
    vwap_mode:           str = "rolling"  # "rolling" or "intraday"
    vwap_period:         int = 20         # used only in rolling mode
    obv_trend_period:    int = 20
    force_index_period:  int = 13
    vol_oscillator_fast: int = 5
    vol_oscillator_slow: int = 20
    mfi_period:          int = 14
    cmf_period:          int = 20
    amihud_period:       int = 20         # rolling window for Amihud proxy


@dataclass(frozen=True)
class MarketStructureConfig:
    """Configuration for Market Structure (BOS/CHOCH/FVG/OB)."""
    family_version:       str = "market_structure-v1"
    bos_choch_lookback:   int = 5    # swing lookback for BOS/CHOCH
    fvg_count_window:     int = 10   # rolling window for FVG counts
    ob_window:            int = 10   # rolling window for order block counts
    ob_impulse_atr_mult:  float = 1.5
    liq_sweep_lookback:   int = 20
    # Causal note: BOS/CHOCH use trailing swing levels only.
    # FVG detected at bar i using only bars [i-2, i-1, i].
    # No future confirmation required.


@dataclass(frozen=True)
class CrossSectionalConfig:
    """Configuration for Cross-Sectional features."""
    family_version:       str = "cross_sectional-v1"
    rank_universe:        str = "historical_eligible"  # never current_universe
    zscore_min_stocks:    int = 5    # minimum universe size for valid z-score
    rank_min_stocks:      int = 5
    cs_features:          tuple[str, ...] = (
        "return_20d", "return_5d", "atr_pct", "relative_volume",
        "rsi_14", "momentum_20d",
    )
    # Survivorship safety: universe must come from historical_universe(t),
    # not the current live F&O universe.


@dataclass(frozen=True)
class BreadthConfig:
    """Configuration for Market Breadth family."""
    family_version:    str = "breadth-v1"
    sma_periods:       tuple[int, ...] = (20, 50, 200)
    advance_decline_min_stocks: int = 10
    # Breadth must only count stocks eligible at t (historical universe).
    # Using today's universe for historical breadth = survivorship bias.


@dataclass(frozen=True)
class SectorConfig:
    """Configuration for Sector Relative Strength family."""
    family_version:      str = "sector-v1"
    sector_rs_period:    int = 20
    sector_momentum_period: int = 5
    sector_dispersion_period: int = 20
    # Sector membership must be point-in-time.
    # A company reclassified in 2025 must not alter its 2020 sector identity.
    india_sectors: tuple[str, ...] = (
        "Auto", "Bank", "FMCG", "IT", "Metal", "Pharma",
        "Realty", "Energy", "Infra", "Media", "PSU",
    )


@dataclass(frozen=True)
class DerivativesConfig:
    """Configuration for F&O / Derivatives family."""
    family_version:  str = "derivatives-v1"
    oi_zscore_period: int = 20
    oi_pct_period:    int = 20
    basis_period:     int = 5
    # OI and basis must use data available at prediction_time.
    # End-of-day OI must not be used for intraday predictions.


@dataclass(frozen=True)
class OptionsIVConfig:
    """Configuration for Options / IV family."""
    family_version:    str = "options_iv-v1"
    iv_rank_period:    int = 252   # 1 year of daily IV history
    iv_percentile_period: int = 252
    iv_min_history:    int = 5     # fewer than this → INSUFFICIENT_HISTORY, not NaN→50
    # If historical IV data does not exist: DATA_UNAVAILABLE, never fabricated.


@dataclass(frozen=True)
class ExpiryConfig:
    """Configuration for Expiry / Contract State family."""
    family_version:    str = "expiry-v1"
    # India NSE: Thursday weekly, last-Thursday-of-month monthly.
    # The actual calendar must come from the historical contract store
    # (Phase 3B), NOT from hardcoded rules.
    max_days_to_weekly:  int = 5
    max_days_to_monthly: int = 30


@dataclass(frozen=True)
class FeatureEngineConfig:
    """
    Master configuration for the entire feature engine.

    hash property is deterministic: same configuration → same hash.
    Any change to any sub-config bumps the hash, which should trigger
    a new dataset snapshot.
    """
    momentum:        MomentumConfig        = field(default_factory=MomentumConfig)
    trend:           TrendConfig           = field(default_factory=TrendConfig)
    mean_reversion:  MeanReversionConfig   = field(default_factory=MeanReversionConfig)
    volatility:      VolatilityConfig      = field(default_factory=VolatilityConfig)
    volume:          VolumeLiquidityConfig = field(default_factory=VolumeLiquidityConfig)
    market_structure: MarketStructureConfig = field(default_factory=MarketStructureConfig)
    cross_sectional: CrossSectionalConfig  = field(default_factory=CrossSectionalConfig)
    breadth:         BreadthConfig         = field(default_factory=BreadthConfig)
    sector:          SectorConfig          = field(default_factory=SectorConfig)
    derivatives:     DerivativesConfig     = field(default_factory=DerivativesConfig)
    options_iv:      OptionsIVConfig       = field(default_factory=OptionsIVConfig)
    expiry:          ExpiryConfig          = field(default_factory=ExpiryConfig)

    @property
    def hash(self) -> str:
        """
        Deterministic 16-char hex hash of the full configuration.
        Identical configs produce identical hashes across runs.
        """
        raw = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def feature_set_version(self) -> str:
        """
        Human-readable version string composed of all family versions.
        Changes when any family version changes.
        """
        parts = [
            self.momentum.family_version,
            self.trend.family_version,
            self.mean_reversion.family_version,
            self.volatility.family_version,
            self.volume.family_version,
            self.market_structure.family_version,
            self.cross_sectional.family_version,
            self.breadth.family_version,
            self.sector.family_version,
            self.derivatives.family_version,
            self.options_iv.family_version,
            self.expiry.family_version,
        ]
        return "|".join(parts)

    @classmethod
    def default(cls) -> "FeatureEngineConfig":
        """Return the canonical default configuration."""
        return cls()


# Module-level singleton — import and use directly
DEFAULT_CONFIG = FeatureEngineConfig.default()
