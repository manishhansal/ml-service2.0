"""
Feature Registry — Phase 3D.

Canonical registry of every feature in the AlphaForge feature engine.

Rules
-----
1. Every feature in RANKING_FEATURES / REGIME_FEATURES / RISK_FEATURES /
   STRATEGY_FEATURES must have a corresponding FeatureSpec in FEATURE_REGISTRY.
2. Deprecated features must have promotion=DEPRECATED and a non-empty
   deprecation_note.
3. The registry is append-only with respect to active feature names.
   Removing a feature requires adding it as DEPRECATED first.
4. Features may not be used in production training unless
   promotion == PRODUCTION_CANDIDATE.
5. All features in this registry are RESEARCH or EXPERIMENTAL until
   an OOS IC experiment confirms stability — this is Phase 3D's correct
   starting point.
"""

from __future__ import annotations

from .schemas import (
    AvailabilityStatus,
    FeatureFamily,
    FeaturePromotion,
    FeatureSetSpec,
    FeatureSpec,
    MissingPolicy,
    NormalizationPolicy,
    PITSafety,
)

# ── Helper ────────────────────────────────────────────────────────────────────

def _spec(
    name: str,
    version: str,
    family: FeatureFamily,
    description: str,
    source: str,
    formula_id: str,
    required_columns: list[str],
    lookback: int,
    causal: bool = True,
    pit_safety: PITSafety = PITSafety.SAFE,
    cross_sectional: bool = False,
    market_scope: str = "single_stock",
    missing_policy: MissingPolicy = MissingPolicy.RETURN_NAN,
    normalization_policy: NormalizationPolicy = NormalizationPolicy.RAW,
    promotion: FeaturePromotion = FeaturePromotion.RESEARCH,
    economic_rationale: str = "",
    known_limitations: str = "",
    proxy_for: str = "",
    tags: list[str] | None = None,
    bar_frequency: str = "1D",
    deprecated_by: str | None = None,
    deprecation_note: str = "",
) -> FeatureSpec:
    return FeatureSpec(
        feature_name=name,
        feature_version=version,
        family=family,
        description=description,
        source=source,
        formula_id=formula_id,
        required_columns=required_columns,
        lookback=lookback,
        bar_frequency=bar_frequency,
        causal=causal,
        pit_safety=pit_safety,
        cross_sectional=cross_sectional,
        market_scope=market_scope,
        missing_policy=missing_policy,
        normalization_policy=normalization_policy,
        promotion=promotion,
        economic_rationale=economic_rationale,
        known_limitations=known_limitations,
        proxy_for=proxy_for,
        tags=tags or [],
        deprecated_by=deprecated_by,
        deprecation_note=deprecation_note,
    )


# ── Registry ──────────────────────────────────────────────────────────────────

FEATURE_REGISTRY: dict[str, FeatureSpec] = {}

def _reg(spec: FeatureSpec) -> None:
    """Register a FeatureSpec. Duplicate names raise ValueError."""
    if spec.feature_name in FEATURE_REGISTRY:
        raise ValueError(
            f"Duplicate feature name in registry: '{spec.feature_name}'. "
            "Use a new name or deprecate the old one first."
        )
    FEATURE_REGISTRY[spec.feature_name] = spec


# ══════════════════════════════════════════════════════════════════════════════
# 1. MOMENTUM / RELATIVE STRENGTH
# ══════════════════════════════════════════════════════════════════════════════

for _p in [1, 2, 3, 5, 10, 20, 60]:
    _reg(_spec(
        name=f"return_{_p}d",
        version="momentum-v1",
        family=FeatureFamily.MOMENTUM,
        description=f"Close-to-close pct return over {_p} trading days.",
        source="equity_ohlcv",
        formula_id=f"pct_change_{_p}",
        required_columns=["close"],
        lookback=_p + 1,
        missing_policy=MissingPolicy.RETURN_NAN,
        normalization_policy=NormalizationPolicy.PERCENT,
        economic_rationale=(
            "Raw momentum signal. Persistent positive returns reflect "
            "trend-following behaviour and investor under-reaction."
        ),
        known_limitations="Highly auto-correlated across periods.",
        promotion=FeaturePromotion.RESEARCH,
    ))

_reg(_spec(
    name="rate_of_change_12",
    version="momentum-v1",
    family=FeatureFamily.MOMENTUM,
    description="12-bar price rate of change as a percentage.",
    source="equity_ohlcv",
    formula_id="roc_12",
    required_columns=["close"],
    lookback=13,
    normalization_policy=NormalizationPolicy.PERCENT,
    economic_rationale="ROC captures medium-term price velocity.",
))

_reg(_spec(
    name="trend_strength",
    version="momentum-v1",
    family=FeatureFamily.MOMENTUM,
    description="Linear-regression slope over 20 bars, normalised to [-1,1].",
    source="equity_ohlcv",
    formula_id="linreg_slope_norm_20",
    required_columns=["close"],
    lookback=20,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.CLIPPED,
    economic_rationale="Distinguishes persistent trend from single-day spike.",
))

_reg(_spec(
    name="breakout_score",
    version="momentum-v1",
    family=FeatureFamily.MOMENTUM,
    description="Breakout / breakdown score in [-1,1] with volume confirmation.",
    source="equity_ohlcv",
    formula_id="breakout_vol_confirmed_20",
    required_columns=["close", "high", "low", "volume"],
    lookback=21,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.CLIPPED,
))

_reg(_spec(
    name="gap_pct",
    version="momentum-v1",
    family=FeatureFamily.OVERNIGHT_GAP,
    description="Overnight gap as % of previous close.",
    source="equity_ohlcv",
    formula_id="gap_open_prev_close",
    required_columns=["open", "close"],
    lookback=2,
    normalization_policy=NormalizationPolicy.PERCENT,
    known_limitations="Gap direction only; fill ratio is future-dependent.",
))

_reg(_spec(
    name="distance_from_52w_high",
    version="momentum-v1",
    family=FeatureFamily.MOMENTUM,
    description="% distance of close from the 252-day high (negative or zero).",
    source="equity_ohlcv",
    formula_id="dist_from_252d_high",
    required_columns=["close"],
    lookback=252,
    normalization_policy=NormalizationPolicy.PERCENT,
    economic_rationale="52-week high proximity is a documented momentum predictor.",
))

_reg(_spec(
    name="distance_from_52w_low",
    version="momentum-v1",
    family=FeatureFamily.MOMENTUM,
    description="% distance of close from the 252-day low (positive or zero).",
    source="equity_ohlcv",
    formula_id="dist_from_252d_low",
    required_columns=["close"],
    lookback=252,
    normalization_policy=NormalizationPolicy.PERCENT,
))

_reg(_spec(
    name="higher_highs_lows",
    version="momentum-v1",
    family=FeatureFamily.TREND,
    description="Fraction of HH+HL minus LH+LL swings in last 5 bars, in [-1,1].",
    source="equity_ohlcv",
    formula_id="hhhl_5",
    required_columns=["high", "low"],
    lookback=6,
    normalization_policy=NormalizationPolicy.CLIPPED,
))

_reg(_spec(
    name="relative_strength_vs_nifty",
    version="momentum-v1",
    family=FeatureFamily.MOMENTUM,
    description="Mansfield RS: (1+stock_ret_20d)/(1+nifty_ret_20d).",
    source="equity_ohlcv,nifty_close",
    formula_id="mansfield_rs_20",
    required_columns=["close"],
    lookback=21,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    economic_rationale="Relative strength vs benchmark is a core momentum factor.",
    known_limitations=(
        "Returns NaN when NIFTY data absent. "
        "Previously defaulted to 1.0 (neutral) — FIXED in Phase 3D."
    ),
    tags=["cross_sectional_candidate"],
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="atr_expansion",
    version="volatility-v1",
    family=FeatureFamily.VOLATILITY,
    description="ATR(5) / ATR(20): >1 means volatility expanding.",
    source="equity_ohlcv",
    formula_id="atr_ratio_5_20",
    required_columns=["high", "low", "close"],
    lookback=21,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

# ══════════════════════════════════════════════════════════════════════════════
# 2. TREND
# ══════════════════════════════════════════════════════════════════════════════

_reg(_spec(
    name="ema_stack_score",
    version="trend-v1",
    family=FeatureFamily.TREND,
    description="EMA alignment score [-1,1]: +1=bull stack(8>13>21>55>200).",
    source="equity_ohlcv",
    formula_id="ema_stack_8_13_21_55_200",
    required_columns=["close"],
    lookback=200,
    normalization_policy=NormalizationPolicy.CLIPPED,
    tags=["talib_required"],
))

_reg(_spec(
    name="adx_14",
    version="trend-v1",
    family=FeatureFamily.TREND,
    description="Average Directional Index over 14 bars.",
    source="equity_ohlcv",
    formula_id="adx_14",
    required_columns=["high", "low", "close"],
    lookback=28,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

_reg(_spec(
    name="macd_line",
    version="trend-v1",
    family=FeatureFamily.TREND,
    description="MACD line (EMA12 - EMA26).",
    source="equity_ohlcv",
    formula_id="macd_12_26_9",
    required_columns=["close"],
    lookback=34,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

_reg(_spec(
    name="macd_signal",
    version="trend-v1",
    family=FeatureFamily.TREND,
    description="MACD 9-bar EMA signal line.",
    source="equity_ohlcv",
    formula_id="macd_12_26_9",
    required_columns=["close"],
    lookback=34,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

_reg(_spec(
    name="macd_histogram",
    version="trend-v1",
    family=FeatureFamily.TREND,
    description="MACD histogram (line - signal).",
    source="equity_ohlcv",
    formula_id="macd_12_26_9",
    required_columns=["close"],
    lookback=34,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

_reg(_spec(
    name="supertrend",
    version="trend-v1",
    family=FeatureFamily.TREND,
    description="Supertrend direction: +1 uptrend, -1 downtrend.",
    source="equity_ohlcv",
    formula_id="supertrend_10_3",
    required_columns=["high", "low", "close"],
    lookback=20,
    normalization_policy=NormalizationPolicy.SIGNED,
    tags=["talib_required"],
))

_reg(_spec(
    name="ht_trendline_dev",
    version="trend-v1",
    family=FeatureFamily.TREND,
    description="(close - HT_TRENDLINE) / HT_TRENDLINE. Hilbert Transform deviation.",
    source="equity_ohlcv",
    formula_id="ht_trendline_dev",
    required_columns=["close"],
    lookback=63,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

# ══════════════════════════════════════════════════════════════════════════════
# 3. MEAN REVERSION
# ══════════════════════════════════════════════════════════════════════════════

_reg(_spec(
    name="rsi_14",
    version="mean_reversion-v1",
    family=FeatureFamily.MEAN_REVERSION,
    description="Wilder RSI over 14 bars [0, 100].",
    source="equity_ohlcv",
    formula_id="rsi_14",
    required_columns=["close"],
    lookback=28,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

_reg(_spec(
    name="rsi_7",
    version="mean_reversion-v1",
    family=FeatureFamily.MEAN_REVERSION,
    description="Wilder RSI over 7 bars [0, 100]. Faster oscillator.",
    source="equity_ohlcv",
    formula_id="rsi_7",
    required_columns=["close"],
    lookback=14,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

_reg(_spec(
    name="bollinger_position",
    version="mean_reversion-v1",
    family=FeatureFamily.MEAN_REVERSION,
    description="Position within Bollinger Bands: 0=lower, 1=upper.",
    source="equity_ohlcv",
    formula_id="bb_position_20_2",
    required_columns=["close"],
    lookback=20,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.MINMAX_ROLLING,
    tags=["talib_required"],
))

_reg(_spec(
    name="stoch_rsi",
    version="mean_reversion-v1",
    family=FeatureFamily.MEAN_REVERSION,
    description="Stochastic RSI fastk [0, 100].",
    source="equity_ohlcv",
    formula_id="stoch_rsi_14_14",
    required_columns=["close"],
    lookback=42,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

_reg(_spec(
    name="williams_r",
    version="mean_reversion-v1",
    family=FeatureFamily.MEAN_REVERSION,
    description="Williams %R [-100, 0].",
    source="equity_ohlcv",
    formula_id="willr_14",
    required_columns=["high", "low", "close"],
    lookback=14,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

_reg(_spec(
    name="cci",
    version="mean_reversion-v1",
    family=FeatureFamily.MEAN_REVERSION,
    description="Commodity Channel Index (20 bars).",
    source="equity_ohlcv",
    formula_id="cci_20",
    required_columns=["high", "low", "close"],
    lookback=20,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

# ══════════════════════════════════════════════════════════════════════════════
# 4. VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════

_reg(_spec(
    name="atr_14",
    version="volatility-v1",
    family=FeatureFamily.VOLATILITY,
    description="Wilder ATR over 14 bars (absolute price units).",
    source="equity_ohlcv",
    formula_id="atr_14",
    required_columns=["high", "low", "close"],
    lookback=28,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

_reg(_spec(
    name="atr_pct",
    version="volatility-v1",
    family=FeatureFamily.VOLATILITY,
    description="ATR(14) as % of close price. Cross-stock comparable.",
    source="equity_ohlcv",
    formula_id="atr_pct_14",
    required_columns=["high", "low", "close"],
    lookback=28,
    normalization_policy=NormalizationPolicy.PERCENT,
    economic_rationale=(
        "Normalised volatility proxy. Comparable across price levels. "
        "Used for barrier sizing in Phase 3C labels."
    ),
    tags=["talib_required"],
))

_reg(_spec(
    name="realized_vol_20",
    version="volatility-v1",
    family=FeatureFamily.VOLATILITY,
    description="Close-to-close realised volatility over 20 days (annualised %).",
    source="equity_ohlcv",
    formula_id="realized_vol_cc_20",
    required_columns=["close"],
    lookback=21,
    normalization_policy=NormalizationPolicy.PERCENT,
    economic_rationale=(
        "Realised vol is the most direct measure of actual price uncertainty."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="parkinson_vol_20",
    version="volatility-v1",
    family=FeatureFamily.VOLATILITY,
    description="Parkinson (high-low) volatility estimator over 20 bars (annualised %).",
    source="equity_ohlcv",
    formula_id="parkinson_vol_20",
    required_columns=["high", "low"],
    lookback=20,
    normalization_policy=NormalizationPolicy.PERCENT,
    economic_rationale=(
        "Parkinson estimator is 5x more efficient than close-to-close "
        "for estimating intraday volatility from daily bars."
    ),
    known_limitations="Assumes no overnight gap; underestimates true vol on gap days.",
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="vol_regime",
    version="volatility-v1",
    family=FeatureFamily.VOLATILITY,
    description="Volatility regime: 0=low, 1=normal, 2=high, 3=extreme (rolling percentile).",
    source="equity_ohlcv",
    formula_id="vol_regime_percentile_60",
    required_columns=["high", "low", "close"],
    lookback=60,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    economic_rationale=(
        "Regime context. Many strategies have strongly regime-conditional performance."
    ),
    known_limitations=(
        "Uses rolling 60-bar reference — regime label at bar 60 has very short history."
    ),
))

# ══════════════════════════════════════════════════════════════════════════════
# 5. VOLUME / LIQUIDITY
# ══════════════════════════════════════════════════════════════════════════════

_reg(_spec(
    name="relative_volume",
    version="volume_liquidity-v1",
    family=FeatureFamily.VOLUME_LIQUIDITY,
    description="Today's volume / 20-day average volume.",
    source="equity_ohlcv",
    formula_id="rel_vol_20",
    required_columns=["volume"],
    lookback=20,
    normalization_policy=NormalizationPolicy.RAW,
))

_reg(_spec(
    name="volume_breakout",
    version="volume_liquidity-v1",
    family=FeatureFamily.VOLUME_LIQUIDITY,
    description="Binary: 1 if volume >= 1.5x 20-day avg.",
    source="equity_ohlcv",
    formula_id="vol_breakout_20_1.5",
    required_columns=["volume"],
    lookback=20,
    normalization_policy=NormalizationPolicy.BINARY,
))

_reg(_spec(
    name="volume_trend",
    version="volume_liquidity-v1",
    family=FeatureFamily.VOLUME_LIQUIDITY,
    description="Volume trend: recent 10-day avg / prior 10-day avg.",
    source="equity_ohlcv",
    formula_id="vol_trend_10",
    required_columns=["volume"],
    lookback=20,
    normalization_policy=NormalizationPolicy.RAW,
))

_reg(_spec(
    name="volume_price_confirm",
    version="volume_liquidity-v1",
    family=FeatureFamily.VOLUME_LIQUIDITY,
    description="Volume-price confirmation: +1 rising price+vol, -1 divergence.",
    source="equity_ohlcv",
    formula_id="vol_price_confirm_20",
    required_columns=["close", "volume"],
    lookback=20,
    normalization_policy=NormalizationPolicy.SIGNED,
))

_reg(_spec(
    name="vwap_distance_pct",
    version="volume_liquidity-v1",
    family=FeatureFamily.VOLUME_LIQUIDITY,
    description="% distance from rolling 20-bar VWAP. Causal rolling window.",
    source="equity_ohlcv",
    formula_id="vwap_distance_rolling_20",
    required_columns=["high", "low", "close", "volume"],
    lookback=20,
    normalization_policy=NormalizationPolicy.PERCENT,
    known_limitations=(
        "For daily bars: rolling-window VWAP, not session-reset VWAP. "
        "Session-reset VWAP requires sub-daily data."
    ),
))

_reg(_spec(
    name="volume_profile_score",
    version="volume_liquidity-v1",
    family=FeatureFamily.VOLUME_LIQUIDITY,
    description="Position relative to 20-bar volume-weighted price distribution, [-1,1].",
    source="equity_ohlcv",
    formula_id="vol_profile_score_20",
    required_columns=["close", "high", "low", "volume"],
    lookback=20,
    normalization_policy=NormalizationPolicy.CLIPPED,
))

_reg(_spec(
    name="obv_trend",
    version="volume_liquidity-v1",
    family=FeatureFamily.VOLUME_LIQUIDITY,
    description="OBV z-score vs 20-bar rolling mean/std.",
    source="equity_ohlcv",
    formula_id="obv_zscore_20",
    required_columns=["close", "volume"],
    lookback=20,
    normalization_policy=NormalizationPolicy.Z_SCORE_ROLLING,
    tags=["talib_required"],
))

_reg(_spec(
    name="obv_last",
    version="volume_liquidity-v1",
    family=FeatureFamily.VOLUME_LIQUIDITY,
    description="Raw On-Balance Volume (cumulative).",
    source="equity_ohlcv",
    formula_id="obv",
    required_columns=["close", "volume"],
    lookback=1,
    normalization_policy=NormalizationPolicy.RAW,
    known_limitations="Absolute OBV is not cross-stock comparable; use obv_trend instead.",
    tags=["talib_required"],
))

_reg(_spec(
    name="mfi",
    version="volume_liquidity-v1",
    family=FeatureFamily.VOLUME_LIQUIDITY,
    description="Money Flow Index [0, 100] — volume-weighted RSI.",
    source="equity_ohlcv",
    formula_id="mfi_14",
    required_columns=["high", "low", "close", "volume"],
    lookback=14,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

_reg(_spec(
    name="cmf",
    version="volume_liquidity-v1",
    family=FeatureFamily.VOLUME_LIQUIDITY,
    description="Chaikin Money Flow [-1, 1] over 20 bars.",
    source="equity_ohlcv",
    formula_id="cmf_20",
    required_columns=["high", "low", "close", "volume"],
    lookback=20,
    normalization_policy=NormalizationPolicy.CLIPPED,
))

_reg(_spec(
    name="force_index_norm",
    version="volume_liquidity-v1",
    family=FeatureFamily.VOLUME_LIQUIDITY,
    description="Elder Force Index (13-bar EMA), normalised by price*volume.",
    source="equity_ohlcv",
    formula_id="force_index_norm_13",
    required_columns=["close", "volume"],
    lookback=13,
    normalization_policy=NormalizationPolicy.RAW,
))

_reg(_spec(
    name="volume_oscillator",
    version="volume_liquidity-v1",
    family=FeatureFamily.VOLUME_LIQUIDITY,
    description="Volume Oscillator: (EMA5 - EMA20) / EMA20 × 100.",
    source="equity_ohlcv",
    formula_id="vol_osc_5_20",
    required_columns=["volume"],
    lookback=20,
    normalization_policy=NormalizationPolicy.PERCENT,
))

_reg(_spec(
    name="amihud_illiquidity",
    version="volume_liquidity-v1",
    family=FeatureFamily.VOLUME_LIQUIDITY,
    description="Amihud illiquidity proxy: |return| / dollar_volume, 20-day average.",
    source="equity_ohlcv",
    formula_id="amihud_proxy_20",
    required_columns=["close", "volume"],
    lookback=20,
    normalization_policy=NormalizationPolicy.RAW,
    economic_rationale=(
        "Amihud ratio is the most widely used microstructure proxy from OHLCV. "
        "High values indicate illiquid stocks where large trades move price more."
    ),
    proxy_for="true_price_impact",
    tags=["proxy", "microstructure"],
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="delivery_pct",
    version="volume_liquidity-v1",
    family=FeatureFamily.VOLUME_LIQUIDITY,
    description="Delivery volume as % of total volume (NSE-specific).",
    source="nse_bhavcopy",
    formula_id="delivery_pct",
    required_columns=["delivery_volume", "volume"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.PERCENT,
    economic_rationale=(
        "High delivery % signals conviction buying/selling vs intraday speculation."
    ),
    known_limitations=(
        "Previously defaulted to 0.0 when absent — FIXED in Phase 3D. "
        "Now returns NaN with DATA_UNAVAILABLE status."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

# ══════════════════════════════════════════════════════════════════════════════
# 6. MARKET STRUCTURE (BOS / CHOCH / FVG / Order Blocks)
# ══════════════════════════════════════════════════════════════════════════════

_reg(_spec(
    name="fvg_score",
    version="market_structure-v1",
    family=FeatureFamily.MARKET_STRUCTURE,
    description="Net FVG score [-1,1]: rolling 10-bar bullish minus bearish FVGs.",
    source="equity_ohlcv",
    formula_id="fvg_3bar_roll10",
    required_columns=["high", "low", "close"],
    lookback=12,
    normalization_policy=NormalizationPolicy.CLIPPED,
    economic_rationale=(
        "FVG at bar i is detected using only bars [i-2, i-1, i] — fully causal."
    ),
    known_limitations="FVG fill ratio is a future outcome; NOT used as a feature here.",
))

_reg(_spec(
    name="bullish_fvg_count",
    version="market_structure-v1",
    family=FeatureFamily.MARKET_STRUCTURE,
    description="Rolling 10-bar count of bullish FVGs.",
    source="equity_ohlcv",
    formula_id="fvg_bull_count_10",
    required_columns=["high", "low"],
    lookback=12,
    normalization_policy=NormalizationPolicy.RAW,
))

_reg(_spec(
    name="bearish_fvg_count",
    version="market_structure-v1",
    family=FeatureFamily.MARKET_STRUCTURE,
    description="Rolling 10-bar count of bearish FVGs.",
    source="equity_ohlcv",
    formula_id="fvg_bear_count_10",
    required_columns=["high", "low"],
    lookback=12,
    normalization_policy=NormalizationPolicy.RAW,
))

_reg(_spec(
    name="ob_score",
    version="market_structure-v1",
    family=FeatureFamily.MARKET_STRUCTURE,
    description="Net Order Block score [-1,1]: rolling 10-bar bullish minus bearish OBs.",
    source="equity_ohlcv",
    formula_id="ob_atr_impulse_roll10",
    required_columns=["open", "high", "low", "close"],
    lookback=25,
    normalization_policy=NormalizationPolicy.CLIPPED,
    known_limitations="OB mitigation (future price return to OB) is NOT used here.",
))

_reg(_spec(
    name="bos_net",
    version="market_structure-v1",
    family=FeatureFamily.MARKET_STRUCTURE,
    description="Rolling 10-bar net BOS events (bullish minus bearish).",
    source="equity_ohlcv",
    formula_id="bos_trailing_swing_roll10",
    required_columns=["high", "low", "close"],
    lookback=20,
    normalization_policy=NormalizationPolicy.RAW,
    economic_rationale="BOS uses trailing swing levels only — verified causal.",
))

_reg(_spec(
    name="choch_net",
    version="market_structure-v1",
    family=FeatureFamily.MARKET_STRUCTURE,
    description="Rolling 10-bar net CHOCH events (potential reversals).",
    source="equity_ohlcv",
    formula_id="choch_trailing_swing_roll10",
    required_columns=["high", "low", "close"],
    lookback=20,
    normalization_policy=NormalizationPolicy.RAW,
))

_reg(_spec(
    name="structure_score",
    version="market_structure-v1",
    family=FeatureFamily.MARKET_STRUCTURE,
    description="Combined structure score [-1,1]: 0.6*BOS + 0.4*CHOCH.",
    source="equity_ohlcv",
    formula_id="structure_score_bos_choch",
    required_columns=["high", "low", "close"],
    lookback=20,
    normalization_policy=NormalizationPolicy.CLIPPED,
))

_reg(_spec(
    name="liquidity_sweep",
    version="market_structure-v1",
    family=FeatureFamily.MARKET_STRUCTURE,
    description="Liquidity sweep score: +1 bullish sweep, -1 bearish sweep, 0 none.",
    source="equity_ohlcv",
    formula_id="liq_sweep_20",
    required_columns=["high", "low", "close"],
    lookback=21,
    normalization_policy=NormalizationPolicy.SIGNED,
))

_reg(_spec(
    name="bullish_ob_count",
    version="market_structure-v1",
    family=FeatureFamily.MARKET_STRUCTURE,
    description="Rolling 10-bar count of bullish order blocks.",
    source="equity_ohlcv",
    formula_id="ob_bull_count_10",
    required_columns=["open", "high", "low", "close"],
    lookback=25,
    normalization_policy=NormalizationPolicy.RAW,
))

_reg(_spec(
    name="bearish_ob_count",
    version="market_structure-v1",
    family=FeatureFamily.MARKET_STRUCTURE,
    description="Rolling 10-bar count of bearish order blocks.",
    source="equity_ohlcv",
    formula_id="ob_bear_count_10",
    required_columns=["open", "high", "low", "close"],
    lookback=25,
    normalization_policy=NormalizationPolicy.RAW,
))

# ══════════════════════════════════════════════════════════════════════════════
# 7. CROSS-SECTIONAL
# ══════════════════════════════════════════════════════════════════════════════

_reg(_spec(
    name="cs_return_20d_rank",
    version="cross_sectional-v1",
    family=FeatureFamily.CROSS_SECTIONAL,
    description="Cross-sectional percentile rank of return_20d within historical universe at t.",
    source="equity_ohlcv,historical_universe",
    formula_id="cs_pct_rank_return_20d",
    required_columns=["close"],
    lookback=21,
    cross_sectional=True,
    market_scope="nifty500_historical",
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.CROSS_SECTIONAL_RANK,
    economic_rationale=(
        "Cross-sectional rank removes market-wide bias. "
        "Prevents today's bull market from inflating every stock's score."
    ),
    known_limitations=(
        "Requires >= 5 eligible stocks at t. "
        "Uses historical_universe(t) — NOT current F&O universe."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="cs_return_5d_rank",
    version="cross_sectional-v1",
    family=FeatureFamily.CROSS_SECTIONAL,
    description="Cross-sectional percentile rank of return_5d within historical universe at t.",
    source="equity_ohlcv,historical_universe",
    formula_id="cs_pct_rank_return_5d",
    required_columns=["close"],
    lookback=6,
    cross_sectional=True,
    market_scope="nifty500_historical",
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.CROSS_SECTIONAL_RANK,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="cs_rs_nifty_rank",
    version="cross_sectional-v1",
    family=FeatureFamily.CROSS_SECTIONAL,
    description="Cross-sectional percentile rank of relative_strength_vs_nifty at t.",
    source="equity_ohlcv,nifty_close,historical_universe",
    formula_id="cs_pct_rank_rs_nifty",
    required_columns=["close"],
    lookback=21,
    cross_sectional=True,
    market_scope="nifty500_historical",
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.CROSS_SECTIONAL_RANK,
    promotion=FeaturePromotion.RESEARCH,
))

# ══════════════════════════════════════════════════════════════════════════════
# 8. BREADTH
# ══════════════════════════════════════════════════════════════════════════════

_reg(_spec(
    name="pct_above_sma20",
    version="breadth-v1",
    family=FeatureFamily.BREADTH,
    description="% of eligible stocks above their 20-day SMA at t.",
    source="equity_ohlcv,historical_universe",
    formula_id="breadth_pct_above_sma20",
    required_columns=["close"],
    lookback=20,
    cross_sectional=True,
    market_scope="nifty500_historical",
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.PERCENT,
    known_limitations=(
        "Previously defaulted to 50.0 when absent — FIXED in Phase 3D. "
        "Now returns NaN with DATA_UNAVAILABLE status."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="pct_above_sma50",
    version="breadth-v1",
    family=FeatureFamily.BREADTH,
    description="% of eligible stocks above their 50-day SMA at t.",
    source="equity_ohlcv,historical_universe",
    formula_id="breadth_pct_above_sma50",
    required_columns=["close"],
    lookback=50,
    cross_sectional=True,
    market_scope="nifty500_historical",
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.PERCENT,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="pct_above_sma200",
    version="breadth-v1",
    family=FeatureFamily.BREADTH,
    description="% of eligible stocks above their 200-day SMA at t.",
    source="equity_ohlcv,historical_universe",
    formula_id="breadth_pct_above_sma200",
    required_columns=["close"],
    lookback=200,
    cross_sectional=True,
    market_scope="nifty500_historical",
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.PERCENT,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="advance_decline_ratio",
    version="breadth-v1",
    family=FeatureFamily.BREADTH,
    description="(advances - declines) / (advances + declines) in [-1, 1].",
    source="nse_bhavcopy",
    formula_id="adv_dec_ratio",
    required_columns=["advances", "declines"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.CLIPPED,
    known_limitations="Requires NSE bhavcopy advance/decline data — DATA_UNAVAILABLE offline.",
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="market_breadth",
    version="breadth-v1",
    family=FeatureFamily.BREADTH,
    description="Composite breadth score [0, 100] passed from market data provider.",
    source="market_data_provider",
    formula_id="breadth_composite",
    required_columns=[],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    promotion=FeaturePromotion.RESEARCH,
))

# ══════════════════════════════════════════════════════════════════════════════
# 9. SECTOR
# ══════════════════════════════════════════════════════════════════════════════

_reg(_spec(
    name="sector_momentum",
    version="sector-v1",
    family=FeatureFamily.SECTOR,
    description="Average 5-day return of sector peers (PIT sector membership).",
    source="equity_ohlcv,sector_master",
    formula_id="sector_avg_return_5d",
    required_columns=["close"],
    lookback=6,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.PERCENT,
    known_limitations=(
        "Previously defaulted to 0.0 when sector_data absent — FIXED in Phase 3D. "
        "Now returns NaN. Sector membership must be point-in-time."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="sector_relative_strength",
    version="sector-v1",
    family=FeatureFamily.SECTOR,
    description="Stock 20d return relative to average sector peer 20d return.",
    source="equity_ohlcv,sector_master",
    formula_id="stock_vs_sector_return_20d",
    required_columns=["close"],
    lookback=21,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    known_limitations=(
        "Previously defaulted to 1.0 (neutral) when sector absent — FIXED. "
        "Sector membership must be PIT."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="sector_dispersion",
    version="sector-v1",
    family=FeatureFamily.SECTOR,
    description="Cross-sector return dispersion (std of sector returns at t).",
    source="equity_ohlcv,sector_master",
    formula_id="sector_return_dispersion",
    required_columns=["close"],
    lookback=6,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.PERCENT,
    economic_rationale=(
        "High dispersion = sector rotation active. "
        "Different information from breadth."
    ),
    known_limitations=(
        "Previously defaulted to 0.0 when absent — FIXED. "
        "Now returns NaN."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="rotation_score",
    version="sector-v1",
    family=FeatureFamily.SECTOR,
    description="Cyclical vs defensive sector return spread, clipped to [-5,5].",
    source="equity_ohlcv,sector_master",
    formula_id="sector_rotation_cyc_def",
    required_columns=["close"],
    lookback=6,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.CLIPPED,
    known_limitations=(
        "Previously defaulted to 0.0 — FIXED. "
        "Cyclical/defensive classification is India-specific."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

# ══════════════════════════════════════════════════════════════════════════════
# 10. DERIVATIVES / F&O
# ══════════════════════════════════════════════════════════════════════════════

_reg(_spec(
    name="pcr_score",
    version="derivatives-v1",
    family=FeatureFamily.DERIVATIVES,
    description="PCR normalised to [-1,1]: PCR>1.3 → +1, PCR<0.7 → -1.",
    source="nse_derivatives",
    formula_id="pcr_normalised",
    required_columns=["put_call_ratio"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.CLIPPED,
    economic_rationale="High PCR suggests market makers are net short puts → bullish.",
    known_limitations=(
        "Previously returned 0.0 for missing PCR — FIXED in Phase 3D. "
        "Now returns NaN with DATA_UNAVAILABLE status."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="pcr_raw",
    version="derivatives-v1",
    family=FeatureFamily.DERIVATIVES,
    description="Raw Put/Call Ratio (OI-based).",
    source="nse_derivatives",
    formula_id="pcr_oi_raw",
    required_columns=["put_call_ratio"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    known_limitations=(
        "Previously defaulted to 1.0 (neutral) — FIXED in Phase 3D. "
        "1.0 is a meaningful market state; absent PCR must be NaN."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="oi_buildup_score",
    version="derivatives-v1",
    family=FeatureFamily.DERIVATIVES,
    description="OI build-up quadrant score [-1,1]: long/short buildup/covering.",
    source="nse_derivatives",
    formula_id="oi_buildup_quadrant",
    required_columns=["price_change_pct", "oi_change_pct"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.CLIPPED,
    known_limitations=(
        "Classification assumes price↑+OI↑=long buildup. "
        "Methodology documented in derivatives.py."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="max_pain_distance_pct",
    version="derivatives-v1",
    family=FeatureFamily.DERIVATIVES,
    description="% distance from spot to max-pain strike.",
    source="nse_options",
    formula_id="max_pain_distance",
    required_columns=["spot", "max_pain_strike"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.PERCENT,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="ce_wall_distance_pct",
    version="derivatives-v1",
    family=FeatureFamily.DERIVATIVES,
    description="% distance from spot to highest CE OI strike (resistance wall).",
    source="nse_options",
    formula_id="ce_wall_distance",
    required_columns=["spot", "max_ce_oi_strike"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.PERCENT,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="pe_wall_distance_pct",
    version="derivatives-v1",
    family=FeatureFamily.DERIVATIVES,
    description="% distance from spot to highest PE OI strike (support wall).",
    source="nse_options",
    formula_id="pe_wall_distance",
    required_columns=["spot", "max_pe_oi_strike"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.PERCENT,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="oi_wall_score",
    version="derivatives-v1",
    family=FeatureFamily.DERIVATIVES,
    description="OI wall score [-1,1]: +1 when PE wall nearby (support), -1 when CE wall nearby.",
    source="nse_options",
    formula_id="oi_wall_score",
    required_columns=["spot", "max_ce_oi_strike", "max_pe_oi_strike"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.CLIPPED,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="oi_delta_skew_norm",
    version="derivatives-v1",
    family=FeatureFamily.DERIVATIVES,
    description="Normalised OI change skew: (PE_chg - CE_chg)/(|PE_chg|+|CE_chg|).",
    source="nse_derivatives",
    formula_id="oi_delta_skew_norm",
    required_columns=["total_ce_oi_change", "total_pe_oi_change"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.CLIPPED,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="pcr_oi",
    version="derivatives-v1",
    family=FeatureFamily.DERIVATIVES,
    description="Put/Call ratio from OI totals.",
    source="nse_derivatives",
    formula_id="pcr_from_oi_totals",
    required_columns=["total_ce_oi", "total_pe_oi"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    known_limitations=(
        "Previously defaulted to 1.0 when OI data absent — FIXED in Phase 3D."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

# ══════════════════════════════════════════════════════════════════════════════
# 11. OPTIONS / IV
# ══════════════════════════════════════════════════════════════════════════════

_reg(_spec(
    name="iv_rank",
    version="options_iv-v1",
    family=FeatureFamily.OPTIONS_IV,
    description="IV Rank [0,100]: where current IV sits in its 252-day range.",
    source="nse_options",
    formula_id="iv_rank_252",
    required_columns=["current_iv", "iv_history"],
    lookback=252,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    economic_rationale="IV Rank identifies cheap vs expensive options.",
    known_limitations=(
        "Returns NaN when < 5 history bars. "
        "Must NEVER be substituted with 50 silently."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="iv_rank_status",
    version="options_iv-v1",
    family=FeatureFamily.OPTIONS_IV,
    description="IV Rank data quality flag: 0=OK, 1=insufficient history.",
    source="nse_options",
    formula_id="iv_rank_status",
    required_columns=["current_iv", "iv_history"],
    lookback=252,
    missing_policy=MissingPolicy.RETURN_ZERO,
    normalization_policy=NormalizationPolicy.BINARY,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="atm_iv",
    version="options_iv-v1",
    family=FeatureFamily.OPTIONS_IV,
    description="At-the-money implied volatility.",
    source="nse_options",
    formula_id="atm_iv",
    required_columns=["atm_iv"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.PERCENT,
    known_limitations=(
        "Previously defaulted to 0.0 when absent — FIXED in Phase 3D. "
        "0% IV is impossible in reality."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

# ══════════════════════════════════════════════════════════════════════════════
# 12. EXPIRY / CONTRACT STATE
# ══════════════════════════════════════════════════════════════════════════════

_reg(_spec(
    name="is_expiry_day",
    version="expiry-v1",
    family=FeatureFamily.EXPIRY_CONTRACT,
    description="1 if today is NSE options expiry day, else 0.",
    source="nse_expiry_calendar",
    formula_id="is_expiry_day",
    required_columns=["expiry_calendar"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.BINARY,
    known_limitations=(
        "Calendar must be historical (not current schedule). "
        "India changed to Tuesday expiry Sep 2025 — historical dates use Thursday."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="days_to_weekly_expiry",
    version="expiry-v1",
    family=FeatureFamily.EXPIRY_CONTRACT,
    description="Calendar days to next weekly options expiry.",
    source="nse_expiry_calendar",
    formula_id="days_to_weekly_expiry",
    required_columns=["expiry_calendar"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    known_limitations=(
        "Previously defaulted to 5 when absent — FIXED in Phase 3D."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="days_to_monthly_expiry",
    version="expiry-v1",
    family=FeatureFamily.EXPIRY_CONTRACT,
    description="Calendar days to next monthly options expiry.",
    source="nse_expiry_calendar",
    formula_id="days_to_monthly_expiry",
    required_columns=["expiry_calendar"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="weekly_theta_pressure",
    version="expiry-v1",
    family=FeatureFamily.EXPIRY_CONTRACT,
    description="1 / max(days_to_weekly_expiry, 0.5) — theta acceleration proxy.",
    source="nse_expiry_calendar",
    formula_id="theta_pressure_weekly",
    required_columns=["expiry_calendar"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="monthly_theta_pressure",
    version="expiry-v1",
    family=FeatureFamily.EXPIRY_CONTRACT,
    description="1 / max(days_to_monthly_expiry, 0.5) — theta acceleration proxy.",
    source="nse_expiry_calendar",
    formula_id="theta_pressure_monthly",
    required_columns=["expiry_calendar"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    promotion=FeaturePromotion.RESEARCH,
))

# ══════════════════════════════════════════════════════════════════════════════
# 13. MARKET REGIME / INTERMARKET
# ══════════════════════════════════════════════════════════════════════════════

_reg(_spec(
    name="vix_regime",
    version="market_regime-v1",
    family=FeatureFamily.MARKET_REGIME,
    description="India VIX regime: 0=low(<13), 1=moderate(<18), 2=high(<25), 3=extreme.",
    source="india_vix",
    formula_id="vix_regime_thresholds",
    required_columns=["india_vix"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    known_limitations=(
        "Previously defaulted to 1.0 (moderate) when VIX absent — FIXED. "
        "Fixed thresholds are calibrated for India VIX levels historically."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="vix_percentile",
    version="market_regime-v1",
    family=FeatureFamily.MARKET_REGIME,
    description="% of historical VIX readings below current VIX (rolling 252-day).",
    source="india_vix",
    formula_id="vix_percentile_252",
    required_columns=["india_vix"],
    lookback=252,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.PERCENT,
    known_limitations=(
        "Previously defaulted to 50.0 when VIX history absent — FIXED."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="vix_mean_reversion",
    version="market_regime-v1",
    family=FeatureFamily.MARKET_REGIME,
    description="VIX z-score vs 252-day mean/std, clipped to [-3, 3].",
    source="india_vix",
    formula_id="vix_zscore_252",
    required_columns=["india_vix"],
    lookback=252,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.CLIPPED,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="nifty_change_pct",
    version="market_regime-v1",
    family=FeatureFamily.MARKET_REGIME,
    description="NIFTY 50 1-day pct change.",
    source="nifty_ohlcv",
    formula_id="nifty_pct_change_1d",
    required_columns=["close"],
    lookback=2,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.PERCENT,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="banknifty_change_pct",
    version="market_regime-v1",
    family=FeatureFamily.MARKET_REGIME,
    description="BANKNIFTY 1-day pct change.",
    source="banknifty_ohlcv",
    formula_id="banknifty_pct_change_1d",
    required_columns=["close"],
    lookback=2,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.PERCENT,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="bank_nifty_spread",
    version="market_regime-v1",
    family=FeatureFamily.MARKET_REGIME,
    description="BANKNIFTY change - NIFTY change (Bank outperformance = risk-on).",
    source="nifty_ohlcv,banknifty_ohlcv",
    formula_id="bank_nifty_spread",
    required_columns=["close"],
    lookback=2,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.PERCENT,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="global_sentiment",
    version="intermarket-v1",
    family=FeatureFamily.INTERMARKET,
    description="Composite global risk-on score: US futures + crude + USD/INR.",
    source="intermarket_data",
    formula_id="global_sentiment_composite",
    required_columns=["us_futures_change", "crude_change", "dollar_inr_change"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.CLIPPED,
    known_limitations=(
        "Previously substituted 0.0 for each missing component — FIXED. "
        "Now returns NaN when all components absent."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

# ══════════════════════════════════════════════════════════════════════════════
# 14. CANDLESTICK PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

_reg(_spec(
    name="cdl_engulfing",
    version="market_structure-v1",
    family=FeatureFamily.MARKET_STRUCTURE,
    description="Engulfing pattern: 1.0 if detected on current bar, else 0.",
    source="equity_ohlcv",
    formula_id="cdl_engulfing",
    required_columns=["open", "high", "low", "close"],
    lookback=2,
    normalization_policy=NormalizationPolicy.BINARY,
    tags=["talib_required"],
))

_reg(_spec(
    name="cdl_hammer",
    version="market_structure-v1",
    family=FeatureFamily.MARKET_STRUCTURE,
    description="Hammer pattern: 1.0 if detected, else 0.",
    source="equity_ohlcv",
    formula_id="cdl_hammer",
    required_columns=["open", "high", "low", "close"],
    lookback=2,
    normalization_policy=NormalizationPolicy.BINARY,
    tags=["talib_required"],
))

_reg(_spec(
    name="cdl_doji",
    version="market_structure-v1",
    family=FeatureFamily.MARKET_STRUCTURE,
    description="Doji pattern: 1.0 if detected, else 0.",
    source="equity_ohlcv",
    formula_id="cdl_doji",
    required_columns=["open", "high", "low", "close"],
    lookback=2,
    normalization_policy=NormalizationPolicy.BINARY,
    tags=["talib_required"],
))

# ══════════════════════════════════════════════════════════════════════════════
# 15. MICROSTRUCTURE PROXIES
# ══════════════════════════════════════════════════════════════════════════════

_reg(_spec(
    name="vpin_score",
    version="microstructure-v1",
    family=FeatureFamily.MICROSTRUCTURE,
    description="VPIN order-flow toxicity proxy (requires intraday bar data).",
    source="intraday_ohlcv",
    formula_id="vpin_tick_rule",
    required_columns=["open", "close", "volume"],
    lookback=50,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    pit_safety=PITSafety.SAFE,
    proxy_for="order_flow_imbalance",
    known_limitations=(
        "Tick-rule approximation. Requires intraday data. "
        "DATA_UNAVAILABLE when only daily bars exist."
    ),
    tags=["proxy", "microstructure", "intraday_required"],
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="nifty_atr_pct",
    version="volatility-v1",
    family=FeatureFamily.VOLATILITY,
    description="NIFTY ATR(14) as % of NIFTY close.",
    source="nifty_ohlcv",
    formula_id="nifty_atr_pct_14",
    required_columns=["high", "low", "close"],
    lookback=28,
    normalization_policy=NormalizationPolicy.PERCENT,
    tags=["talib_required"],
))

_reg(_spec(
    name="nifty_adx",
    version="trend-v1",
    family=FeatureFamily.TREND,
    description="NIFTY ADX(14).",
    source="nifty_ohlcv",
    formula_id="nifty_adx_14",
    required_columns=["high", "low", "close"],
    lookback=28,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

_reg(_spec(
    name="nifty_rsi",
    version="mean_reversion-v1",
    family=FeatureFamily.MEAN_REVERSION,
    description="NIFTY RSI(14).",
    source="nifty_ohlcv",
    formula_id="nifty_rsi_14",
    required_columns=["close"],
    lookback=28,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

_reg(_spec(
    name="nifty_macd_hist",
    version="trend-v1",
    family=FeatureFamily.TREND,
    description="NIFTY MACD histogram.",
    source="nifty_ohlcv",
    formula_id="nifty_macd_12_26_9",
    required_columns=["close"],
    lookback=34,
    normalization_policy=NormalizationPolicy.RAW,
    tags=["talib_required"],
))

_reg(_spec(
    name="gap_pct_nifty",
    version="momentum-v1",
    family=FeatureFamily.OVERNIGHT_GAP,
    description="NIFTY overnight gap % (open vs prev close).",
    source="nifty_ohlcv",
    formula_id="gap_nifty_open_prev_close",
    required_columns=["open", "close"],
    lookback=2,
    normalization_policy=NormalizationPolicy.PERCENT,
))

_reg(_spec(
    name="volume_ratio_market",
    version="volume_liquidity-v1",
    family=FeatureFamily.VOLUME_LIQUIDITY,
    description="NIFTY relative volume (today / 20-day avg).",
    source="nifty_ohlcv",
    formula_id="nifty_rel_vol_20",
    required_columns=["volume"],
    lookback=20,
    normalization_policy=NormalizationPolicy.RAW,
))

_reg(_spec(
    name="india_vix",
    version="market_regime-v1",
    family=FeatureFamily.MARKET_REGIME,
    description="India VIX level.",
    source="india_vix",
    formula_id="india_vix_raw",
    required_columns=["india_vix"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    known_limitations=(
        "Previously defaulted to 15.0 when absent — FIXED in Phase 3D."
    ),
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="put_call_ratio",
    version="derivatives-v1",
    family=FeatureFamily.DERIVATIVES,
    description="Raw NIFTY Put/Call OI ratio (pass-through from market data).",
    source="nse_derivatives",
    formula_id="nifty_pcr_raw",
    required_columns=["put_call_ratio"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="nifty_oi_delta_skew_norm",
    version="derivatives-v1",
    family=FeatureFamily.DERIVATIVES,
    description="Nifty total OI change skew (PE chg - CE chg), normalised.",
    source="nse_derivatives",
    formula_id="nifty_oi_skew_norm",
    required_columns=["total_ce_oi_change", "total_pe_oi_change"],
    lookback=1,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.CLIPPED,
    promotion=FeaturePromotion.RESEARCH,
))

_reg(_spec(
    name="session_progress",
    version="time_of_day-v1",
    family=FeatureFamily.TIME_OF_DAY,
    description="Session progress [0,1]: 0=open, 1=close. NSE: 09:15-15:30.",
    source="clock",
    formula_id="session_progress_nse",
    required_columns=["minutes_since_open"],
    lookback=0,
    normalization_policy=NormalizationPolicy.MINMAX_ROLLING,
))

# ══════════════════════════════════════════════════════════════════════════════
# 16. DEPRECATED FEATURES (former lv1 defaults)
# ══════════════════════════════════════════════════════════════════════════════
# These entries document features that had silent-default bugs.
# They remain in the registry as DEPRECATED so consumers can detect the
# old name and migrate.

_reg(_spec(
    name="market_breadth_score",
    version="breadth-v1-deprecated",
    family=FeatureFamily.BREADTH,
    description="DEPRECATED: market breadth score alias. Use pct_above_sma20.",
    source="market_data_provider",
    formula_id="breadth_composite",
    required_columns=[],
    lookback=1,
    promotion=FeaturePromotion.DEPRECATED,
    deprecated_by="pct_above_sma20",
    deprecation_note=(
        "Ambiguous name with silent 50.0 default. "
        "Replaced by explicit pct_above_sma20/50/200."
    ),
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
))

_reg(_spec(
    name="volatility_rank",
    version="volatility-v1-deprecated",
    family=FeatureFamily.VOLATILITY,
    description="DEPRECATED: unspecified volatility rank. Use vol_regime or realized_vol_20.",
    source="unknown",
    formula_id="unknown",
    required_columns=[],
    lookback=0,
    promotion=FeaturePromotion.DEPRECATED,
    deprecated_by="vol_regime",
    deprecation_note=(
        "Feature in STRATEGY_FEATURES list had no implementation in engineer.py. "
        "Silently became 0.0 via feats.get(f, 0.0). Replaced by vol_regime."
    ),
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
))

_reg(_spec(
    name="trend_alignment",
    version="trend-v1-deprecated",
    family=FeatureFamily.TREND,
    description="DEPRECATED: unspecified trend alignment. Use ema_stack_score.",
    source="unknown",
    formula_id="unknown",
    required_columns=[],
    lookback=0,
    promotion=FeaturePromotion.DEPRECATED,
    deprecated_by="ema_stack_score",
    deprecation_note=(
        "Feature in RISK_FEATURES list had no implementation in engineer.py. "
        "Silently became 0.0 via feats.get(f, 0.0)."
    ),
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
))

_reg(_spec(
    name="stop_distance_atr",
    version="risk-v1-deprecated",
    family=FeatureFamily.RISK_TAIL,
    description="DEPRECATED: stop distance in ATR units. Not computed in engineer.py.",
    source="unknown",
    formula_id="unknown",
    required_columns=[],
    lookback=0,
    promotion=FeaturePromotion.DEPRECATED,
    deprecated_by="atr_pct",
    deprecation_note=(
        "Feature in RISK_FEATURES had no computation; silently became 0.0. "
        "Stop distance depends on trade entry — not a feature of the stock itself."
    ),
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
))

_reg(_spec(
    name="target_distance_atr",
    version="risk-v1-deprecated",
    family=FeatureFamily.RISK_TAIL,
    description="DEPRECATED: target distance in ATR units. Not computed in engineer.py.",
    source="unknown",
    formula_id="unknown",
    required_columns=[],
    lookback=0,
    promotion=FeaturePromotion.DEPRECATED,
    deprecated_by="atr_pct",
    deprecation_note="Same issue as stop_distance_atr.",
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
))

_reg(_spec(
    name="risk_reward_ratio",
    version="risk-v1-deprecated",
    family=FeatureFamily.RISK_TAIL,
    description="DEPRECATED: risk/reward ratio. Not computed in engineer.py.",
    source="unknown",
    formula_id="unknown",
    required_columns=[],
    lookback=0,
    promotion=FeaturePromotion.DEPRECATED,
    deprecated_by="atr_pct",
    deprecation_note="Trade-specific — not a stock feature. Silently became 0.0.",
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
))

_reg(_spec(
    name="regime_encoded",
    version="market_regime-v1",
    family=FeatureFamily.MARKET_REGIME,
    description="Market regime encoded label (from MarketRegimeClassifier).",
    source="model_market_regime",
    formula_id="regime_model_output",
    required_columns=[],
    lookback=0,
    missing_policy=MissingPolicy.RETURN_NAN,
    normalization_policy=NormalizationPolicy.RAW,
    known_limitations=(
        "This is MODEL_PREDICTED_REGIME, not RAW_REGIME_FEATURE. "
        "Must not be confused with realized future regime labels."
    ),
    promotion=FeaturePromotion.RESEARCH,
))


# ── Lookup helpers ────────────────────────────────────────────────────────────

def get_feature_spec(feature_name: str) -> FeatureSpec:
    """Return FeatureSpec or raise KeyError."""
    if feature_name not in FEATURE_REGISTRY:
        raise KeyError(
            f"Feature '{feature_name}' not found in FEATURE_REGISTRY. "
            "Register it before use."
        )
    return FEATURE_REGISTRY[feature_name]


def list_active_features() -> list[str]:
    """Return names of all non-deprecated features."""
    return [
        name for name, spec in FEATURE_REGISTRY.items()
        if spec.promotion not in (FeaturePromotion.DEPRECATED, FeaturePromotion.BLOCKED)
    ]


def list_by_family(family: FeatureFamily) -> list[str]:
    """Return names of all active features in a given family."""
    return [
        name for name, spec in FEATURE_REGISTRY.items()
        if spec.family == family
        and spec.promotion not in (FeaturePromotion.DEPRECATED, FeaturePromotion.BLOCKED)
    ]


def list_deprecated_features() -> list[str]:
    return [
        name for name, spec in FEATURE_REGISTRY.items()
        if spec.promotion == FeaturePromotion.DEPRECATED
    ]


def list_blocked_features() -> list[str]:
    return [
        name for name, spec in FEATURE_REGISTRY.items()
        if spec.promotion == FeaturePromotion.BLOCKED
    ]


def list_talib_required() -> list[str]:
    """Return features that require talib to compute."""
    return [
        name for name, spec in FEATURE_REGISTRY.items()
        if "talib_required" in spec.tags
        and spec.promotion not in (FeaturePromotion.DEPRECATED, FeaturePromotion.BLOCKED)
    ]


def list_data_unavailable_features() -> list[str]:
    """Return features whose source is not present in the current environment."""
    UNAVAILABLE_SOURCES = {
        "nse_derivatives",
        "nse_options",
        "nse_bhavcopy",
        "nse_expiry_calendar",
        "intraday_ohlcv",
        "india_vix",
        "intermarket_data",
        "sector_master",
        "historical_universe",
        "model_market_regime",
    }
    return [
        name for name, spec in FEATURE_REGISTRY.items()
        if any(s in UNAVAILABLE_SOURCES for s in spec.source.split(","))
        and spec.promotion not in (FeaturePromotion.DEPRECATED, FeaturePromotion.BLOCKED)
    ]


def get_registration(feature_name: str) -> FeatureSpec:
    """Alias for get_feature_spec for API compatibility."""
    return get_feature_spec(feature_name)


# ── Feature set declarations ──────────────────────────────────────────────────
# These replace the bare lists in engineer.py.  Models import these specs
# rather than uncontrolled global lists.

from .schemas import FeatureFamily  # noqa: E402 (needed for FeatureSetSpec)

RANKING_FEATURE_SET = FeatureSetSpec(
    set_id="RANKING_SET",
    set_version="v1",
    feature_names=[
        "return_1d", "return_2d", "return_3d", "return_5d", "return_10d", "return_20d",
        "rsi_14", "rsi_7", "macd_histogram", "macd_line", "adx_14",
        "bollinger_position", "ema_stack_score", "stoch_rsi", "williams_r",
        "cci", "supertrend",
        "relative_volume", "volume_breakout", "volume_trend", "volume_price_confirm",
        "vwap_distance_pct", "volume_profile_score", "obv_trend",
        "mfi", "cmf", "force_index_norm", "volume_oscillator",
        "rate_of_change_12", "trend_strength", "atr_expansion",
        "breakout_score", "gap_pct", "distance_from_52w_high", "distance_from_52w_low",
        "higher_highs_lows",
        "pcr_score", "oi_buildup_score", "iv_rank", "max_pain_distance_pct",
        "ce_wall_distance_pct", "pe_wall_distance_pct", "oi_wall_score",
        "oi_delta_skew_norm", "atm_iv",
        "fvg_score", "ob_score", "structure_score", "liquidity_sweep",
        "relative_strength_vs_nifty", "sector_momentum", "sector_relative_strength",
        "regime_encoded", "vix_regime", "market_breadth",
        "atr_pct", "atr_14",
        "delivery_pct",
        "cdl_engulfing", "cdl_hammer", "cdl_doji",
        "ht_trendline_dev",
        "macd_signal",
        "obv_last",
    ],
    required_families=[
        FeatureFamily.MOMENTUM, FeatureFamily.TREND, FeatureFamily.MEAN_REVERSION,
        FeatureFamily.VOLATILITY, FeatureFamily.VOLUME_LIQUIDITY,
    ],
    min_ok_fraction=0.70,  # many derivatives features may be DATA_UNAVAILABLE
    allow_imputation=True,
)

REGIME_FEATURE_SET = FeatureSetSpec(
    set_id="REGIME_SET",
    set_version="v1",
    feature_names=[
        "nifty_change_pct", "banknifty_change_pct", "india_vix", "vix_regime",
        "vix_mean_reversion", "vix_percentile", "nifty_atr_pct", "nifty_adx",
        "nifty_rsi", "nifty_macd_hist", "advance_decline_ratio", "market_breadth",
        "pct_above_sma20", "pct_above_sma50", "pct_above_sma200",
        "sector_dispersion", "rotation_score", "volume_ratio_market",
        "gap_pct_nifty", "bank_nifty_spread", "global_sentiment",
        "put_call_ratio", "nifty_oi_delta_skew_norm",
        "is_expiry_day", "days_to_weekly_expiry", "session_progress",
        "vpin_score",
    ],
    required_families=[FeatureFamily.MARKET_REGIME, FeatureFamily.BREADTH],
    min_ok_fraction=0.60,
    allow_imputation=True,
)

STRATEGY_FEATURE_SET = FeatureSetSpec(
    set_id="STRATEGY_SET",
    set_version="v1",
    feature_names=[
        "regime_encoded", "rsi_14", "adx_14", "atr_pct", "relative_volume",
        "vwap_distance_pct", "bollinger_position", "trend_strength",
        "vol_regime", "session_progress", "ema_stack_score",
        "breakout_score", "volume_price_confirm", "macd_histogram",
        "pcr_score", "oi_buildup_score", "gap_pct",
    ],
    required_families=[FeatureFamily.MOMENTUM, FeatureFamily.TREND],
    min_ok_fraction=0.75,
    allow_imputation=True,
)

RISK_FEATURE_SET = FeatureSetSpec(
    set_id="RISK_SET",
    set_version="v1",
    feature_names=[
        "atr_pct", "vix_regime", "adx_14", "rsi_14", "relative_volume",
        "regime_encoded", "atr_14",
        "is_expiry_day", "bollinger_position",
        "volume_price_confirm", "breakout_score", "structure_score",
        "pcr_score", "oi_buildup_score", "session_progress",
    ],
    required_families=[FeatureFamily.VOLATILITY, FeatureFamily.MARKET_REGIME],
    min_ok_fraction=0.75,
    allow_imputation=True,
)
