"""
Feature Engineering Orchestrator.

Combines all feature modules into a unified feature vector for each stock.
Produces 150+ features from OHLCV + derivatives + macro data.

The output is a flat dictionary per stock that can be fed directly into
any of the downstream models (regime classifier, stock ranker, strategy
selector, risk predictor).
"""

import numpy as np
import pandas as pd

from .technical import (
    compute_adx,
    compute_atr,
    compute_bollinger_position,
    compute_cci,
    compute_cdl_doji,
    compute_cdl_engulfing,
    compute_cdl_hammer,
    compute_cmf,
    compute_ema_stack_score,
    compute_ht_trendline_dev,
    compute_macd,
    compute_mfi,
    compute_obv,
    compute_obv_trend,
    compute_rsi,
    compute_stochastic_rsi,
    compute_supertrend,
    compute_williams_r,
)
from .volume import (
    compute_accumulation_distribution,
    compute_force_index,
    compute_relative_volume,
    compute_volume_breakout,
    compute_volume_oscillator,
    compute_volume_price_confirmation,
    compute_volume_profile_score,
    compute_volume_trend,
    compute_vwap_distance_pct,
)
from .momentum import (
    compute_atr_expansion,
    compute_breakout_score,
    compute_distance_from_high,
    compute_distance_from_low,
    compute_gap_pct,
    compute_higher_highs_higher_lows,
    compute_rate_of_change,
    compute_relative_strength_vs_index,
    compute_returns,
    compute_trend_strength,
)
from .derivatives import (
    compute_iv_rank,
    compute_iv_rank_with_status,
    compute_iv_rank_safe,
    compute_max_pain_distance,
    compute_oi_buildup_score,
    compute_oi_wall_proximity,
    compute_options_flow_features,
    compute_pcr_score,
)
from .market_structure import (
    detect_bos_choch,
    detect_fair_value_gaps,
    detect_liquidity_sweeps,
    detect_order_blocks,
)
from .macro import (
    compute_advance_decline_ratio,
    compute_expiry_features,
    compute_intermarket_features,
    compute_time_features,
    compute_vix_features,
)


# ─── Feature Names (canonical ordering for all models) ────────────────────────

REGIME_FEATURES = [
    "nifty_change_pct", "banknifty_change_pct", "india_vix", "vix_regime",
    "vix_change_pct", "vix_percentile", "nifty_atr_pct", "nifty_adx",
    "nifty_rsi", "nifty_macd_hist", "advance_decline_ratio", "market_breadth",
    "pct_above_sma20", "pct_above_sma50", "pct_above_sma200",
    "sector_dispersion", "rotation_score", "volume_ratio_market",
    "gap_pct_nifty", "bank_nifty_spread", "global_sentiment",
    "put_call_ratio", "nifty_oi_delta_skew_norm", "vix_mean_reversion",
    "is_expiry_day", "days_to_weekly_expiry", "session_progress",
    "vpin_score",  # VPIN order-flow toxicity (Requirement 7.8)
]

RANKING_FEATURES = [
    # Price-based
    "return_1d", "return_2d", "return_3d", "return_5d", "return_10d", "return_20d",
    "rsi_14", "rsi_7", "macd_histogram", "macd_line", "adx_14",
    "bollinger_position", "ema_stack_score", "stoch_rsi", "williams_r",
    "cci", "supertrend",
    # Volume
    "relative_volume", "volume_breakout", "volume_trend", "volume_price_confirm",
    "vwap_distance_pct", "volume_profile_score", "obv_trend",
    "mfi", "cmf", "force_index_norm", "volume_oscillator",
    # Momentum
    "rate_of_change_12", "trend_strength", "atr_expansion",
    "breakout_score", "gap_pct", "distance_from_52w_high", "distance_from_52w_low",
    "higher_highs_lows",
    # Derivatives
    "pcr_score", "oi_buildup_score", "iv_rank", "max_pain_distance_pct",
    "ce_wall_distance_pct", "pe_wall_distance_pct", "oi_wall_score",
    "oi_delta_skew_norm", "atm_iv",
    # Market structure
    "fvg_score", "ob_score", "structure_score", "liquidity_sweep",
    # Relative strength — vs NIFTY AND vs sector peers (new)
    "relative_strength_vs_nifty", "sector_momentum", "sector_relative_strength",
    # Macro context (passed through from regime)
    "regime_encoded", "vix_regime", "market_breadth_score",
    # ATR
    "atr_pct", "atr_14",
    # Delivery quality gate (NSE-specific)
    "delivery_pct",
    # Candlestick patterns (Requirement 6.2)
    "cdl_engulfing", "cdl_hammer", "cdl_doji",
    # HT_TRENDLINE deviation (Requirement 6.2)
    "ht_trendline_dev",
    # Macro context supplement
    "macd_signal",
    "obv_last",
]

STRATEGY_FEATURES = [
    "regime_encoded", "rsi_14", "adx_14", "atr_pct", "relative_volume",
    "vwap_distance_pct", "bollinger_position", "trend_strength",
    "volatility_rank", "session_progress", "ema_stack_score",
    "breakout_score", "volume_price_confirm", "macd_histogram",
    "pcr_score", "oi_buildup_score", "gap_pct",
]

RISK_FEATURES = [
    "atr_pct", "vix_regime", "adx_14", "rsi_14", "relative_volume",
    "regime_encoded", "stop_distance_atr", "target_distance_atr",
    "risk_reward_ratio", "trend_alignment", "pcr_score", "oi_buildup_score",
    "session_progress", "is_expiry_day", "bollinger_position",
    "volume_price_confirm", "breakout_score", "structure_score",
]


def compute_stock_features(
    ohlcv: pd.DataFrame,
    index_close: pd.Series | None = None,
    sector_data: dict[str, pd.Series] | None = None,
    derivatives_data: dict | None = None,
    macro_context: dict | None = None,
) -> dict[str, float]:
    """
    Compute the full feature vector for a single stock.

    Args:
        ohlcv: DataFrame with columns [open, high, low, close, volume]
               and a DatetimeIndex. Should have at least 200 rows for
               all features to compute.
        index_close: NIFTY 50 close series for relative strength calc.
        sector_data: Dict of {symbol: close_series} for sector peers.
        derivatives_data: Dict with keys like 'pcr', 'oi_change_pct',
                         'price_change_pct', 'iv_history', 'current_iv',
                         'max_pain', 'max_ce_oi_strike', 'max_pe_oi_strike',
                         'total_ce_oi', 'total_pe_oi', 'total_ce_oi_change',
                         'total_pe_oi_change', 'atm_iv'.
        macro_context: Dict with regime/market-wide features already computed.

    Returns:
        Flat dict of feature_name → float value. NaN/None → 0.0.
    """
    features: dict[str, float] = {}

    o = ohlcv["open"]
    h = ohlcv["high"]
    l = ohlcv["low"]  # noqa: E741
    c = ohlcv["close"]
    v = ohlcv["volume"]

    # ─── Technical ────────────────────────────────────────────────────────
    rsi_14 = compute_rsi(c, 14)
    rsi_7 = compute_rsi(c, 7)
    macd_line, macd_signal, macd_hist = compute_macd(c)
    adx_14 = compute_adx(h, l, c, 14)
    atr_14 = compute_atr(h, l, c, 14)
    bb_pos = compute_bollinger_position(c, 20)
    ema_stack = compute_ema_stack_score(c)
    stoch_rsi = compute_stochastic_rsi(c, 14, 14)
    williams = compute_williams_r(h, l, c, 14)
    cci = compute_cci(h, l, c, 20)
    mfi = compute_mfi(h, l, c, v, 14)
    cmf = compute_cmf(h, l, c, v, 20)
    obv_t = compute_obv_trend(c, v, 20)
    supertrend = compute_supertrend(h, l, c, 10, 3.0)
    features["rsi_14"] = _last(rsi_14)
    features["rsi_7"] = _last(rsi_7)
    features["macd_line"] = _last(macd_line)
    features["macd_signal"] = _last(macd_signal)
    features["macd_histogram"] = _last(macd_hist)
    features["adx_14"] = _last(adx_14)
    features["atr_14"] = _last(atr_14)
    features["atr_pct"] = _last(atr_14) / _last(c) * 100 if _last(c) > 0 else 0.0
    features["bollinger_position"] = _last(bb_pos)
    features["ema_stack_score"] = _last(ema_stack)
    features["stoch_rsi"] = _last(stoch_rsi)
    features["williams_r"] = _last(williams)
    features["cci"] = _last(cci)
    features["mfi"] = _last(mfi)
    features["cmf"] = _last(cmf)
    features["obv_trend"] = _last(obv_t)
    features["obv_last"] = _last(compute_obv(c, v))
    features["supertrend"] = _last(supertrend)

    # ─── Candlestick patterns & HT_TRENDLINE (Requirement 6.2) ───────────
    cdl_engulfing = compute_cdl_engulfing(o, h, l, c)
    cdl_hammer = compute_cdl_hammer(o, h, l, c)
    cdl_doji = compute_cdl_doji(o, h, l, c)
    ht_dev = compute_ht_trendline_dev(c)

    features["cdl_engulfing"] = _last(cdl_engulfing)
    features["cdl_hammer"] = _last(cdl_hammer)
    features["cdl_doji"] = _last(cdl_doji)
    features["ht_trendline_dev"] = _last(ht_dev)

    # ─── Volume ───────────────────────────────────────────────────────────
    rel_vol = compute_relative_volume(v, 20)
    vol_breakout = compute_volume_breakout(v, 20, 1.5)
    vol_trend = compute_volume_trend(v, 10)
    vol_price_confirm = compute_volume_price_confirmation(c, v, 20)
    vwap_dist = compute_vwap_distance_pct(c, h, l, v)
    vol_profile = compute_volume_profile_score(c, v, h, l, 20)
    force_idx = compute_force_index(c, v, 13)
    vol_osc = compute_volume_oscillator(v, 5, 20)

    features["relative_volume"] = _last(rel_vol)
    features["volume_breakout"] = _last(vol_breakout)
    features["volume_trend"] = _last(vol_trend)
    features["volume_price_confirm"] = _last(vol_price_confirm)
    features["vwap_distance_pct"] = _last(vwap_dist)
    features["volume_profile_score"] = _last(vol_profile)
    features["obv_trend"] = _last(obv_t)
    features["force_index_norm"] = _last(force_idx) / (_last(c) * _last(v) + 1e-10)
    features["volume_oscillator"] = _last(vol_osc)

    # ─── Momentum ─────────────────────────────────────────────────────────
    returns = compute_returns(c, [1, 2, 3, 5, 10, 20, 60])
    for key, series in returns.items():
        features[key] = _last(series)

    roc = compute_rate_of_change(c, 12)
    trend_str = compute_trend_strength(c, 20)
    atr_exp = compute_atr_expansion(h, l, c, 5, 20)
    breakout = compute_breakout_score(c, h, l, v, 20)
    gap = compute_gap_pct(o, c.shift(1))
    dist_high = compute_distance_from_high(c, 252)
    dist_low = compute_distance_from_low(c, 252)
    hh_hl = compute_higher_highs_higher_lows(h, l, 5)

    features["rate_of_change_12"] = _last(roc)
    features["trend_strength"] = _last(trend_str)
    features["atr_expansion"] = _last(atr_exp)
    features["breakout_score"] = _last(breakout)
    features["gap_pct"] = _last(gap)
    features["distance_from_52w_high"] = _last(dist_high)
    features["distance_from_52w_low"] = _last(dist_low)
    features["higher_highs_lows"] = _last(hh_hl)

    # ─── Relative Strength ────────────────────────────────────────────────
    if index_close is not None and len(index_close) >= 20:
        rs = compute_relative_strength_vs_index(c, index_close, 20)
        features["relative_strength_vs_nifty"] = _last_or_nan(rs)
    else:
        # DATA_UNAVAILABLE — do NOT substitute 1.0 (neutral ratio)
        # 1.0 is a real market state (stock performing exactly in line with NIFTY)
        features["relative_strength_vs_nifty"] = float("nan")

    # Sector momentum (average 5-day return of sector peers)
    # Sector relative strength: stock RS vs average of sector peers over 20d
    if sector_data:
        sector_returns = []
        peer_closes = []
        for sym, peer_close in sector_data.items():
            if len(peer_close) > 5:
                ret = (peer_close.iloc[-1] - peer_close.iloc[-6]) / peer_close.iloc[-6] * 100
                sector_returns.append(ret)
            if len(peer_close) >= 20:
                peer_closes.append(peer_close)
        # Use NaN when no sector returns available — NOT 0.0
        features["sector_momentum"] = float(np.mean(sector_returns)) if sector_returns else float("nan")

        # Sector RS: 20d return of this stock vs average 20d return of sector
        if peer_closes and len(c) >= 21:
            stock_ret20 = (c.iloc[-1] - c.iloc[-21]) / c.iloc[-21]
            peer_rets = []
            for pc in peer_closes:
                if len(pc) >= 21:
                    peer_rets.append((pc.iloc[-1] - pc.iloc[-21]) / pc.iloc[-21])
            if peer_rets:
                avg_sector_ret = float(np.mean(peer_rets))
                features["sector_relative_strength"] = (
                    1.0 + float(stock_ret20) - avg_sector_ret
                )
            else:
                features["sector_relative_strength"] = float("nan")
        else:
            features["sector_relative_strength"] = float("nan")
    else:
        # DATA_UNAVAILABLE — NOT 0.0 / 1.0
        features["sector_momentum"]          = float("nan")
        features["sector_relative_strength"] = float("nan")

    # ─── Market Structure ─────────────────────────────────────────────────
    fvg = detect_fair_value_gaps(h, l, c)
    ob = detect_order_blocks(o, h, l, c)
    bos_choch = detect_bos_choch(h, l, c, 5)
    liq_sweep = detect_liquidity_sweeps(h, l, c, 20)

    features["fvg_score"] = _last(fvg["fvg_score"])
    features["bullish_fvg_count"] = _last(fvg["bullish_fvg_count"])
    features["bearish_fvg_count"] = _last(fvg["bearish_fvg_count"])
    features["ob_score"] = _last(ob["ob_score"])
    features["bullish_ob_count"] = _last(ob["bullish_ob_count"])
    features["bearish_ob_count"] = _last(ob["bearish_ob_count"])
    features["bos_net"] = _last(bos_choch["bos_net"])
    features["choch_net"] = _last(bos_choch["choch_net"])
    features["structure_score"] = _last(bos_choch["structure_score"])
    features["liquidity_sweep"] = _last(liq_sweep)

    # ─── Derivatives ──────────────────────────────────────────────────────
    deriv = derivatives_data or {}

    pcr = deriv.get("pcr")
    features["pcr_score"] = compute_pcr_score(pcr)
    # pcr_raw: None when no PCR data — NOT 1.0 (1.0 is a real PCR level)
    features["pcr_raw"] = float(pcr) if pcr is not None else float("nan")

    price_change = deriv.get("price_change_pct", features.get("return_1d", 0))
    oi_change = deriv.get("oi_change_pct", 0)
    oi_score, oi_kind = compute_oi_buildup_score(price_change, oi_change)
    features["oi_buildup_score"] = oi_score

    current_iv = deriv.get("current_iv")
    iv_history = deriv.get("iv_history", [])

    # Use compute_iv_rank_with_status so downstream code can distinguish
    # "rank = 50 because IV is at the median" from "rank = 50 because we
    # have no data".  When history is insufficient, iv_rank_status will be
    # "INSUFFICIENT_HISTORY" and iv_rank will be NaN — the NaN cleanup at
    # the end of this function will convert it to 0.0 (neutral).
    iv_rank_result = compute_iv_rank_with_status(
        current_iv or 20.0, iv_history if iv_history else []
    )
    features["iv_rank"] = (
        iv_rank_result["iv_rank"]
        if iv_rank_result["iv_rank"] is not None
        else float("nan")   # will be set to 0.0 by NaN cleanup below
    )
    features["iv_rank_status"] = 0.0 if iv_rank_result["status"] == "OK" else 1.0

    spot = _last(c)
    features["max_pain_distance_pct"] = compute_max_pain_distance(
        spot, deriv.get("max_pain")
    )

    oi_walls = compute_oi_wall_proximity(
        spot,
        deriv.get("max_ce_oi_strike"),
        deriv.get("max_pe_oi_strike"),
    )
    features.update(oi_walls)

    options_flow = compute_options_flow_features(
        deriv.get("total_ce_oi"),
        deriv.get("total_pe_oi"),
        deriv.get("total_ce_oi_change"),
        deriv.get("total_pe_oi_change"),
        deriv.get("atm_iv"),
    )
    features.update(options_flow)

    # ─── Macro Context (pass-through) ────────────────────────────────────
    if macro_context:
        for key, val in macro_context.items():
            if key not in features:
                features[key] = float(val) if val is not None else 0.0

    # ─── Delivery % (if provided) ────────────────────────────────────────
    # NaN when absent — 0% delivery is a real market state; unknown != 0%
    raw_delivery = deriv.get("delivery_pct")
    features["delivery_pct"] = float(raw_delivery) if raw_delivery is not None else float("nan")

    # ─── Clean up Inf values only; preserve NaN as DATA_UNAVAILABLE ──────
    # Phase 3D: NaN is the correct representation of unavailable data.
    # Callers that need model-ready inputs use FeatureRow.to_model_input()
    # which handles imputation with explicit documentation.
    # We only clean up inf/-inf here (which are always bugs).
    for key in list(features.keys()):
        val = features[key]
        if val is not None and isinstance(val, float) and np.isinf(val):
            features[key] = float("nan")

    return features


def compute_regime_features(
    nifty_ohlcv: pd.DataFrame,
    banknifty_ohlcv: pd.DataFrame | None = None,
    market_data: dict | None = None,
) -> dict[str, float]:
    """
    Compute the feature vector specifically for the Market Regime Classifier.

    Args:
        nifty_ohlcv: NIFTY 50 OHLCV data (at least 50 bars).
        banknifty_ohlcv: BANKNIFTY OHLCV data.
        market_data: Dict with keys like 'india_vix', 'vix_history',
                    'advances', 'declines', 'sector_returns',
                    'fii_net_cr', 'minutes_since_open', 'pcr',
                    'days_to_weekly_expiry', 'is_expiry_day'.
    """
    features: dict[str, float] = {}
    md = market_data or {}

    n_c = nifty_ohlcv["close"]
    n_h = nifty_ohlcv["high"]
    n_l = nifty_ohlcv["low"]

    # NIFTY features
    features["nifty_change_pct"] = float(n_c.pct_change().iloc[-1] * 100) if len(n_c) > 1 else 0.0
    atr = compute_atr(n_h, n_l, n_c, 14)
    features["nifty_atr_pct"] = _last(atr) / _last(n_c) * 100 if _last(n_c) > 0 else 0.0
    features["nifty_adx"] = _last(compute_adx(n_h, n_l, n_c, 14))
    features["nifty_rsi"] = _last(compute_rsi(n_c, 14))
    _, _, macd_hist = compute_macd(n_c)
    features["nifty_macd_hist"] = _last(macd_hist)
    features["gap_pct_nifty"] = _last(compute_gap_pct(nifty_ohlcv["open"], n_c.shift(1)))

    # BANKNIFTY features
    if banknifty_ohlcv is not None and len(banknifty_ohlcv) > 1:
        bn_c = banknifty_ohlcv["close"]
        features["banknifty_change_pct"] = float(bn_c.pct_change().iloc[-1] * 100)
        features["bank_nifty_spread"] = features["banknifty_change_pct"] - features["nifty_change_pct"]
    else:
        features["banknifty_change_pct"] = 0.0
        features["bank_nifty_spread"] = 0.0

    # VIX features
    vix_feats = compute_vix_features(
        md.get("india_vix"),
        md.get("vix_history"),
    )
    features.update(vix_feats)
    features["india_vix"] = vix_feats["vix_level"]

    # Market breadth — NaN when data absent (not 50.0)
    advances = md.get("advances", 0)
    declines = md.get("declines", 0)
    features["advance_decline_ratio"] = compute_advance_decline_ratio(advances, declines)
    features["market_breadth"] = (
        md["market_breadth"] if "market_breadth" in md and md["market_breadth"] is not None
        else float("nan")
    )
    features["pct_above_sma20"] = (
        md["pct_above_sma20"] if "pct_above_sma20" in md and md["pct_above_sma20"] is not None
        else float("nan")
    )
    features["pct_above_sma50"] = (
        md["pct_above_sma50"] if "pct_above_sma50" in md and md["pct_above_sma50"] is not None
        else float("nan")
    )
    features["pct_above_sma200"] = (
        md["pct_above_sma200"] if "pct_above_sma200" in md and md["pct_above_sma200"] is not None
        else float("nan")
    )

    # Sector rotation — NaN when data absent (not 0.0)
    sector_returns = md.get("sector_returns", {})
    if sector_returns:
        from .macro import compute_sector_rotation_score
        rotation = compute_sector_rotation_score(sector_returns)
        features["sector_dispersion"] = rotation.get("sector_dispersion") or float("nan")
        features["rotation_score"]    = rotation.get("rotation_score") or float("nan")
    else:
        features["sector_dispersion"] = float("nan")
        features["rotation_score"]    = float("nan")

    # Volume
    vol = nifty_ohlcv["volume"]
    features["volume_ratio_market"] = _last(compute_relative_volume(vol, 20))

    # Inter-market
    intermarket = compute_intermarket_features(
        features["nifty_change_pct"],
        features["banknifty_change_pct"],
        md.get("us_futures_change"),
        md.get("crude_change"),
        md.get("dollar_inr_change"),
    )
    features["global_sentiment"] = intermarket["global_sentiment"]

    # PCR
    features["put_call_ratio"] = float(md.get("pcr", 1.0))
    features["nifty_oi_delta_skew_norm"] = float(md.get("oi_delta_skew_norm", 0.0))

    # Time / Expiry
    time_feats = compute_time_features(md.get("minutes_since_open", 180))
    features["session_progress"] = time_feats["session_progress"]

    expiry_feats = compute_expiry_features(
        md.get("days_to_weekly_expiry"),
        md.get("days_to_monthly_expiry"),
        md.get("is_expiry_day", False),
    )
    features["is_expiry_day"] = expiry_feats["is_expiry_day"]
    features["days_to_weekly_expiry"] = expiry_feats["days_to_weekly_expiry"]

    # ─── Clean up Inf values only; preserve NaN as DATA_UNAVAILABLE ──────
    for key in list(features.keys()):
        val = features[key]
        if val is not None and isinstance(val, float) and np.isinf(val):
            features[key] = float("nan")

    return features


def _last(series: pd.Series) -> float:
    """Safely get the last value from a series as float; returns 0.0 for NaN/None."""
    if series is None or len(series) == 0:
        return 0.0
    val = series.iloc[-1]
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return 0.0
    return float(val)


def _last_or_nan(series: pd.Series) -> float:
    """
    Safely get the last value from a series.
    Returns NaN (not 0.0) when the last value is unavailable.
    Use this for features where 0.0 or other numerics have economic meaning.
    """
    if series is None or len(series) == 0:
        return float("nan")
    val = series.iloc[-1]
    if val is None or (isinstance(val, float) and np.isinf(val)):
        return float("nan")
    return float(val)
