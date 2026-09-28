"""
src/features/expanded_factory.py — ExpandedFeatureFactory

FIX NEW-P0-002: The original FeatureFactory produces only 24 same-bar OHLCV
features. This expanded factory adds:

  GROUP A: Extended momentum (multi-period at 3/5/10/20/60 bars)
  GROUP B: Regime features (volatility regime, trend regime, gap regime)
  GROUP C: Time-context features (weekday, expiry proximity, month-end)
  GROUP D: Relative value (single-symbol vs its own rolling mean = z-score proxy)
  GROUP E: Advanced volatility (Parkinson, Garman-Klass estimators)

Cross-sectional features (relative value vs universe) are NOT computed here
because they require a panel of simultaneous symbols. They are computed in
`src/features/families/cross_sectional.py` and called by the DatasetBuilder
when a multi-symbol panel is available.

Every feature remains CAUSAL (no forward-looking shifts).
NaN is the correct value when lookback is insufficient — never zero-fill.

Schema version incremented to fs-3.0.0 to distinguish from the 24-feature set.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.factory import FeatureFactory, FeatureAvailability
from src.logging_config import get_logger

logger = get_logger(__name__)

EXPANDED_FEATURE_SCHEMA_VERSION = "fs-3.0.0"


class ExpandedFeatureFactory(FeatureFactory):
    """
    Extends FeatureFactory with regime, time-context, and extended momentum features.

    The original 24 features are kept intact; new groups are appended.
    This allows backward-compatible dataset construction: existing datasets
    (schema fs-2.0.0) can still load the base 24 features.

    Total features: 24 (base) + up to 42 (extended) = up to 66 features.
    Actual count depends on whether index has timezone info (time features).

    Usage::

        factory = ExpandedFeatureFactory()
        features, avail = factory.build(ohlcv_df)
        # features has 66 columns (or fewer if time features disabled)
    """

    # ── Extended feature names (appended after the 24 base features) ──────────
    EXTENDED_FEATURE_NAMES: list[str] = [
        # GROUP A: Extended momentum (causal multi-period returns)
        "ret_3",        # 3-bar return
        "ret_60",       # 60-bar return (quarterly momentum)
        "mom_accel_5",  # 5-bar momentum acceleration (ret_5 - ret_10)
        "mom_accel_20", # 20-bar momentum acceleration (ret_20 - ret_40)
        "ret_60_rel_vol",  # 60-bar return / trailing vol (risk-adjusted momentum)
        # GROUP B: Regime features
        "vol_regime_zscore",   # z-score of current vol vs long-run history
        "vol_regime_pctile",   # percentile rank of current vol (0–1)
        "vol_expanding",       # +1 vol expanding, -1 contracting
        "vol_ratio",           # short/long vol ratio
        "trend_strength",      # ADX (0–100; >25 = trending)
        "trend_direction",     # +1 uptrend, -1 downtrend (EMA sign)
        "ema_spread",          # (EMA20 - EMA60) / EMA60 — trend magnitude
        "trend_persistence",   # fraction of last 20 bars in EMA direction
        "gap_magnitude",       # |overnight gap| / prev close
        "gap_regime_rolling",  # rolling mean of gap magnitude
        "gap_direction",       # +1 gap up, -1 gap down
        # GROUP C: Time-context features
        "weekday",             # 0=Mon … 4=Fri
        "weekday_sin",         # cyclical weekday encoding
        "weekday_cos",
        "month_end_proximity", # calendar days until end of month
        "quarter_end",         # 1 if last 5 days of quarter
        "is_monday",
        "is_friday",
        # GROUP D: Relative value (self-comparison, causal)
        "price_zscore_60",     # (close - EMA60) / rolling_std_60 — mean reversion signal
        "price_zscore_20",     # (close - EMA20) / rolling_std_20
        "vol_norm_ret_5",      # ret_5 / vol_20 — volatility-normalised momentum
        "vol_norm_ret_20",     # ret_20 / vol_20
        # GROUP E: Advanced volatility estimators
        "parkinson_vol",       # Parkinson high-low range estimator (more efficient than close-close)
        "garman_klass_vol",    # Garman-Klass OHLC estimator
        "vol_of_vol_20",       # volatility of volatility (vol_5 rolling std over 20 bars)
        "atr_zscore",          # ATR z-score vs 60-bar ATR history
    ]

    @property
    def FEATURE_NAMES(self) -> list[str]:  # type: ignore[override]
        """All feature names: 24 base + extended."""
        return list(FeatureFactory.FEATURE_NAMES) + self.EXTENDED_FEATURE_NAMES

    def build(
        self, ohlcv: pd.DataFrame
    ) -> tuple[pd.DataFrame, FeatureAvailability]:
        """
        Build the full expanded feature matrix.

        1. Build base 24 features (super().build())
        2. Append extended features
        3. Replace inf with NaN
        4. Return aligned to declared FEATURE_NAMES schema
        """
        # Step 1: build base 24 features directly (bypass the FEATURE_NAMES property override)
        # We instantiate a plain FeatureFactory to avoid the overridden FEATURE_NAMES
        # property sending it to try to select extended columns that don't exist yet.
        _base_factory = FeatureFactory.__new__(FeatureFactory)
        base_features, _ = FeatureFactory.build(_base_factory, ohlcv)

        df = ohlcv.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)
        rets = close.pct_change()

        ext = pd.DataFrame(index=df.index)

        # ── GROUP A: Extended momentum ─────────────────────────────────────
        ext["ret_3"] = close.pct_change(3)
        ext["ret_60"] = close.pct_change(60)
        ext["mom_accel_5"] = close.pct_change(5) - close.pct_change(10)
        # 40-bar return — only available with enough history
        ext["mom_accel_20"] = close.pct_change(20) - close.pct_change(40)
        vol_20 = rets.rolling(20).std()
        ext["ret_60_rel_vol"] = close.pct_change(60) / (vol_20 * np.sqrt(60)).replace(0, np.nan)

        # ── GROUP B: Regime features ───────────────────────────────────────
        from src.features.families.regime import (  # noqa: PLC0415
            compute_volatility_regime,
            compute_trend_regime,
            compute_gap_regime,
        )
        vol_reg = compute_volatility_regime(close)
        for k, v in vol_reg.items():
            ext[k] = v

        trend_reg = compute_trend_regime(close, high, low)
        for k, v in trend_reg.items():
            ext[k] = v

        gap_reg = compute_gap_regime(open_, close)
        for k, v in gap_reg.items():
            ext[k] = v

        # ── GROUP C: Time-context features ─────────────────────────────────
        from src.features.families.time_context import compute_calendar_features  # noqa: PLC0415
        try:
            cal = compute_calendar_features(df.index)
            for k, v in cal.items():
                ext[k] = v
        except Exception as exc:
            # If timezone info is missing from the index, skip time features
            logger.warning(
                "expanded_factory_time_features_skipped",
                error=str(exc),
            )
            for k in ["weekday", "weekday_sin", "weekday_cos",
                       "month_end_proximity", "quarter_end", "is_monday", "is_friday"]:
                ext[k] = np.nan

        # ── GROUP D: Relative value (self-comparison) ──────────────────────
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema60 = close.ewm(span=60, adjust=False).mean()
        std20 = close.rolling(20).std()
        std60 = close.rolling(60).std()
        ext["price_zscore_20"] = (close - ema20) / std20.replace(0, np.nan)
        ext["price_zscore_60"] = (close - ema60) / std60.replace(0, np.nan)
        ext["vol_norm_ret_5"] = close.pct_change(5) / vol_20.replace(0, np.nan)
        ext["vol_norm_ret_20"] = close.pct_change(20) / vol_20.replace(0, np.nan)

        # ── GROUP E: Advanced volatility estimators ────────────────────────
        # Parkinson (HL range estimator): σ_P² = 1/(4ln2) * E[(ln(H/L))²]
        hl_log = np.log(high / low.replace(0, np.nan))
        parkinson_var = (hl_log ** 2) / (4 * np.log(2))
        ext["parkinson_vol"] = np.sqrt(parkinson_var.rolling(20).mean())

        # Garman-Klass (OHLC estimator — more efficient than Parkinson)
        ln_hl = np.log(high / low.replace(0, np.nan)) ** 2
        ln_co = np.log(close / open_.replace(0, np.nan)) ** 2
        gk_daily = 0.5 * ln_hl - (2 * np.log(2) - 1) * ln_co
        ext["garman_klass_vol"] = np.sqrt(gk_daily.rolling(20).mean())

        # Vol-of-vol: rolling std of 5-bar realized vol over 20 bars
        vol_5 = rets.rolling(5).std()
        ext["vol_of_vol_20"] = vol_5.rolling(20).std()

        # ATR z-score
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        atr_60_mean = atr_14.rolling(60).mean()
        atr_60_std = atr_14.rolling(60).std()
        ext["atr_zscore"] = (atr_14 - atr_60_mean) / atr_60_std.replace(0, np.nan)

        # ── Assemble and clean ─────────────────────────────────────────────
        ext = ext.replace([np.inf, -np.inf], np.nan)

        # Ensure all declared extended features are present (fill with NaN if not computed)
        for col in self.EXTENDED_FEATURE_NAMES:
            if col not in ext.columns:
                ext[col] = np.nan

        # Enforce declared extended schema order
        ext = ext[self.EXTENDED_FEATURE_NAMES]

        # Combine with base features
        combined = pd.concat([base_features, ext], axis=1)
        combined = combined.replace([np.inf, -np.inf], np.nan)

        availability = self._availability(combined)

        logger.debug(
            "expanded_factory_built",
            n_features=len(combined.columns),
            n_rows=len(combined),
            schema_version=EXPANDED_FEATURE_SCHEMA_VERSION,
        )
        return combined, availability

    def _availability(self, features: pd.DataFrame) -> FeatureAvailability:
        avail = FeatureAvailability(total_rows=len(features))
        for col in features.columns:
            avail.per_feature_missing[col] = int(features[col].isna().sum())
        return avail
