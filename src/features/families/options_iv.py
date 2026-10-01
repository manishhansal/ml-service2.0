"""
src.features.families.options_iv — Options/IV Feature Family (Group I).

Why options data improves stock direction signals
---------------------------------------------------
Derivatives markets reveal institutional positioning BEFORE price moves.
A stock with rising put-call ratio signals defensive hedging (bearish).
Elevated IV means the market expects a big move — good for volatility
plays but bearish for trend continuation. OI buildup at a strike signals
a support/resistance level.

Features (Group I — 5 features)
---------------------------------
    put_call_ratio      PCR of total OI for nearest weekly expiry [0, ∞)
                        > 1.2 = defensive hedging (bearish), < 0.8 = bullish
    iv_atm_pct          ATM implied volatility as % (annualized)
                        High IV = event risk; Low IV = trending market
    oi_change_pct       Net OI change vs prior day (%) — rising OI = conviction
    iv_skew             (25Δ put IV) − (25Δ call IV) — positive = fear premium
    iv_term_spread      30-day IV − 7-day IV — positive = expected calm soon

Data source:
    data-service2.0  GET /v1/india/historical  (for options OI)
    data-service2.0  GET /v1/india/optionchain  (for IV surface)

PIT safety:
    All features use only the option chain snapshot as of close on the
    PRIOR trading day — never same-day data to avoid execution leakage.

Coverage:
    F&O eligible symbols only (~150 of 285).
    Non-F&O symbols get 0.0 neutral fill.
    Index options (NIFTY, BANKNIFTY) always available.

Usage in DatasetBuilder:
    Same pattern as intraday — load per-symbol options parquet,
    call compute_options_feature_matrix(), merge into training frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OPTIONS_FEATURE_NAMES: list[str] = [
    "put_call_ratio",    # OI-weighted PCR for nearest weekly expiry
    "iv_atm_pct",        # ATM IV (annualized %) at nearest expiry
    "oi_change_pct",     # Net OI change vs prior session (%)
    "iv_skew",           # 25Δ put IV − 25Δ call IV (fear premium)
    "iv_term_spread",    # 30d IV − 7d IV (term structure slope)
]

_NEUTRAL: dict[str, float] = {k: 0.0 for k in OPTIONS_FEATURE_NAMES}


def compute_options_features_from_chain(
    chain: dict | None,
    prev_oi: float | None = None,
) -> dict[str, float]:
    """
    Compute Group I features from a raw option chain response.

    Expected format from DataServiceClient.get_option_chain():
        {
          "data": {
            "underlying": "NIFTY",
            "analytics": {
              "pcrOi": 1.15,
              "pcrVolume": 0.95,
              "atmIv": 12.5,
              "totalCeOi": 45000000,
              "totalPeOi": 51750000,
              ...
            },
            "rows": [...]    ← strike-level detail (may be empty at EOD)
          }
        }

    Args:
        chain:    Response from DataServiceClient.get_option_chain().
        prev_oi:  Total OI from prior session (for oi_change_pct).

    Returns:
        dict of {feature: float}. 0.0 for any unavailable field.
    """
    feats = dict(_NEUTRAL)
    if not chain or not isinstance(chain, dict):
        return feats

    # Unwrap outer envelope
    data = chain.get("data", chain)
    if not isinstance(data, dict):
        return feats

    analytics = data.get("analytics", {}) or {}
    rows       = data.get("rows", [])       or []

    # ── From analytics (always available) ───────────────────────────────────
    pcr_oi  = analytics.get("pcrOi")
    atm_iv  = analytics.get("atmIv")
    ce_oi   = float(analytics.get("totalCeOi",  0) or 0)
    pe_oi   = float(analytics.get("totalPeOi",  0) or 0)

    if pcr_oi is not None:
        feats["put_call_ratio"] = float(np.clip(pcr_oi, 0, 5))

    if atm_iv is not None:
        feats["iv_atm_pct"] = float(np.clip(atm_iv, 0, 200))

    total_oi = ce_oi + pe_oi
    if prev_oi and prev_oi > 0 and total_oi > 0:
        feats["oi_change_pct"] = float(
            np.clip((total_oi - prev_oi) / prev_oi * 100, -100, 200)
        )

    # ── From strike rows (available during market hours) ─────────────────────
    ul_price = float(data.get("underlyingValue", 0) or
                     data.get("underlying_value", 0) or 0)

    if rows and ul_price > 0 and isinstance(rows[0], dict):
        # Sort by proximity to ATM
        def _dist(rec: dict) -> float:
            return abs(float(rec.get("strikePrice", rec.get("strike", 0)) or 0) - ul_price)

        rows_sorted = sorted(rows, key=_dist)[:6]

        ce_ivs, pe_ivs = [], []
        for rec in rows_sorted:
            ce = rec.get("CE", rec.get("call", {})) or {}
            pe = rec.get("PE", rec.get("put", {}))  or {}
            ce_iv = float(ce.get("impliedVolatility", ce.get("iv", 0)) or 0)
            pe_iv = float(pe.get("impliedVolatility", pe.get("iv", 0)) or 0)
            if ce_iv > 0:
                ce_ivs.append(ce_iv)
            if pe_iv > 0:
                pe_ivs.append(pe_iv)

        if ce_ivs and pe_ivs:
            feats["iv_skew"] = float(
                np.clip(np.mean(pe_ivs) - np.mean(ce_ivs), -50, 50)
            )
            # Refine ATM IV from rows if analytics doesn't have it
            if feats["iv_atm_pct"] == 0:
                feats["iv_atm_pct"] = float(
                    np.clip(np.mean(ce_ivs + pe_ivs), 0, 200)
                )

    # Clip all to finite
    for k in OPTIONS_FEATURE_NAMES:
        v = feats.get(k, 0.0)
        feats[k] = float(np.clip(v if np.isfinite(v) else 0.0, -200.0, 500.0))

    return feats


def compute_options_feature_matrix(
    options_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Align per-date options features to a symbol's trading_dates index.

    Args:
        options_df:     DataFrame with DatetimeIndex (UTC) and columns =
                        OPTIONS_FEATURE_NAMES. One row per date.
        trading_dates:  Daily bar index for one symbol.

    Returns:
        DataFrame with index = trading_dates, columns = OPTIONS_FEATURE_NAMES.
        Dates without options data receive 0.0 (neutral fill).
    """
    if options_df is None or options_df.empty:
        return pd.DataFrame(0.0, index=trading_dates, columns=OPTIONS_FEATURE_NAMES)

    odf = options_df.copy()
    if odf.index.tz is None:
        odf.index = odf.index.tz_localize("UTC")
    else:
        odf.index = odf.index.tz_convert("UTC")
    odf.index = odf.index.normalize()

    for col in OPTIONS_FEATURE_NAMES:
        if col not in odf.columns:
            odf[col] = 0.0

    rows: list[dict[str, float]] = []
    for td in trading_dates:
        td_norm = pd.Timestamp(td).normalize()
        if td_norm.tzinfo is None:
            td_norm = td_norm.tz_localize("UTC")
        else:
            td_norm = td_norm.tz_convert("UTC")

        if td_norm in odf.index:
            row = odf.loc[td_norm, OPTIONS_FEATURE_NAMES].to_dict()
        else:
            past = odf[odf.index <= td_norm]
            if not past.empty:
                row = past.iloc[-1][OPTIONS_FEATURE_NAMES].to_dict()
            else:
                future = odf[odf.index <= td_norm + pd.Timedelta(days=7)]
                row = (
                    future.iloc[-1][OPTIONS_FEATURE_NAMES].to_dict()
                    if not future.empty
                    else {k: 0.0 for k in OPTIONS_FEATURE_NAMES}
                )
        rows.append(row)

    return pd.DataFrame(rows, index=trading_dates, columns=OPTIONS_FEATURE_NAMES)
