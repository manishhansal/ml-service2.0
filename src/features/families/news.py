"""
src.features.families.news — News/NLP Feature Family (Group H).

Why news matters for F&O signals
----------------------------------
A pure price-momentum model is blind to the macro regime that drives returns:
  - When Saudi Arabia cuts oil supply → Indian energy stocks rally regardless of RSI
  - When RBI surprises with a rate hike → banking stocks drop across the board
  - When US Fed signals a hold → risk-on sentiment lifts mid-cap equities

These events cannot be predicted from OHLCV data alone.  SentinelPulse
provides NLP-processed news sentiment from the same data sources that
institutional desks monitor in real time.

Integration architecture
-------------------------
  SentinelPulse (port 3001)
      ↓ /api/v1/alphaforge/context/market       (market-level, 1 call/cycle)
      ↓ /api/v1/alphaforge/context/asset/NSE:X  (per-asset, NSE prefix required)
      ↓ /api/v1/ml/training/events               (event importance for training)
    → Group H features (6) → appended to feature vector

Feature definitions (Group H — 6 features)
--------------------------------------------
    news_market_sentiment    India market overall NLP sentiment   [-1, 1]
    news_macro_sentiment     Macro-economic news sentiment        [-1, 1]
    news_risk_sentiment      Risk/fear NLP sentiment              [-1, 1]
    news_regime_score        Numeric regime: +1 bull, 0 neutral, -1 bear
    news_asset_sentiment     Symbol-level NLP sentiment           [-1, 1]
    news_event_importance    Max importance of recent events      [0, 1]

PIT safety
-----------
All features are stored with the UTC publication timestamp.  The DatasetBuilder
fetches features dated BEFORE the trading date, preventing future news leakage.
For training rows without news data (pre-ingestion dates), features fill 0.0
(neutral).  The model learns to treat 0.0 as "no information" — not bearish.

Data coverage
--------------
SentinelPulse ingests India-specific news continuously from Sep 2026.
Historical backfill is NOT available for pre-ingestion dates.
Coverage grows organically as the system accumulates data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NEWS_FEATURE_NAMES: list[str] = [
    "news_market_sentiment",   # India market NLP sentiment [-1, +1]
    "news_macro_sentiment",    # Macro-economic NLP sentiment [-1, +1]
    "news_risk_sentiment",     # Risk/fear NLP sentiment [-1, +1]
    "news_regime_score",       # Regime: +1=bull 0=neutral -1=bear
    "news_asset_sentiment",    # Per-asset NLP sentiment [-1, +1]
    "news_event_importance",   # Max event importance in 24h window [0, 1]
]

# Neutral/default values — used for dates without ingested news
_NEUTRAL: dict[str, float] = {k: 0.0 for k in NEWS_FEATURE_NAMES}


def _direction_to_score(direction: str | None) -> float:
    """Convert a textual regime/direction label to a numeric score."""
    if not direction:
        return 0.0
    d = str(direction).lower()
    if d in ("bullish", "bull", "positive", "strong_bull"):
        return 1.0
    if d in ("bearish", "bear", "negative", "strong_bear"):
        return -1.0
    return 0.0


def news_features_from_context(
    market_ctx: dict | None,
    asset_ctx: dict | None,
    events: list[dict] | None,
) -> dict[str, float]:
    """
    Compute Group H features from raw SentinelPulse API responses.

    Args:
        market_ctx:  Response from /api/v1/alphaforge/context/market
        asset_ctx:   Response from /api/v1/alphaforge/context/asset/NSE:{symbol}
        events:      Response from /api/v1/ml/training/events (list)

    Returns:
        dict of {feature_name: float}.  All values are finite scalars in
        expected ranges.  0.0 for any unavailable field.
    """
    feats = dict(_NEUTRAL)

    # ── Market-level sentiment ──────────────────────────────────────────────
    if market_ctx and isinstance(market_ctx, dict):
        sent = market_ctx.get("overall_sentiment") or {}
        if isinstance(sent, dict):
            raw_mkt = sent.get("market")
            raw_mac = sent.get("macro")
            raw_rsk = sent.get("risk")
            feats["news_market_sentiment"] = float(raw_mkt) if raw_mkt is not None else 0.0
            feats["news_macro_sentiment"]  = float(raw_mac) if raw_mac is not None else 0.0
            feats["news_risk_sentiment"]   = float(raw_rsk) if raw_rsk is not None else 0.0

        regime = market_ctx.get("regime") or (sent.get("direction") if sent else None)
        feats["news_regime_score"] = _direction_to_score(regime)

    # ── Per-asset sentiment ────────────────────────────────────────────────
    if asset_ctx and isinstance(asset_ctx, dict):
        sent_summary = asset_ctx.get("sentiment_summary") or {}
        if isinstance(sent_summary, dict):
            raw_overall = sent_summary.get("overall")
            if raw_overall is not None:
                feats["news_asset_sentiment"] = float(raw_overall)
            else:
                # Fall back to direction-encoded score
                feats["news_asset_sentiment"] = _direction_to_score(
                    sent_summary.get("direction")
                )

    # ── Event importance ───────────────────────────────────────────────────
    if events and isinstance(events, list):
        importances = [
            float(e["importanceScore"])
            for e in events
            if isinstance(e, dict) and e.get("importanceScore") is not None
        ]
        feats["news_event_importance"] = float(max(importances)) if importances else 0.0

    # Clip all to [-1, 1] and ensure finite
    for k in NEWS_FEATURE_NAMES:
        v = feats.get(k, 0.0)
        feats[k] = float(np.clip(v if np.isfinite(v) else 0.0, -1.0, 1.0))

    return feats


def compute_news_feature_matrix(
    news_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Align per-date news features to a symbol's trading_dates index.

    Args:
        news_df:        DataFrame with DatetimeIndex (UTC) and columns =
                        NEWS_FEATURE_NAMES.  One row per date that has real data.
        trading_dates:  The daily bar index for one symbol.

    Returns:
        DataFrame with index = trading_dates, columns = NEWS_FEATURE_NAMES.
        Dates without news data receive 0.0 (neutral fill).
    """
    if news_df is None or news_df.empty:
        return pd.DataFrame(0.0, index=trading_dates, columns=NEWS_FEATURE_NAMES)

    # Normalise news_df index to UTC midnight
    ndf = news_df.copy()
    if ndf.index.tz is None:
        ndf.index = ndf.index.tz_localize("UTC")
    else:
        ndf.index = ndf.index.tz_convert("UTC")
    ndf.index = ndf.index.normalize()

    # Ensure all columns exist
    for col in NEWS_FEATURE_NAMES:
        if col not in ndf.columns:
            ndf[col] = 0.0

    # Align trading_dates → UTC midnight for lookup
    rows: list[dict[str, float]] = []
    for td in trading_dates:
        td_norm = pd.Timestamp(td).normalize()
        if td_norm.tzinfo is None:
            td_norm = td_norm.tz_localize("UTC")
        else:
            td_norm = td_norm.tz_convert("UTC")

        if td_norm in ndf.index:
            row = ndf.loc[td_norm, NEWS_FEATURE_NAMES].to_dict()
        else:
            # Forward-fill: use most recent past news before this date
            past = ndf[ndf.index <= td_norm]
            if not past.empty:
                row = past.iloc[-1][NEWS_FEATURE_NAMES].to_dict()
            else:
                # Back-fill: look up to 7 calendar days ahead.
                # This handles the common case where backfill ran the day AFTER
                # the last training date (e.g., backfill Sep 30 but data ends Sep 28).
                # News sentiment is stable over a few days so this is safe.
                future = ndf[ndf.index <= td_norm + pd.Timedelta(days=7)]
                if not future.empty:
                    row = future.iloc[-1][NEWS_FEATURE_NAMES].to_dict()
                else:
                    row = {k: 0.0 for k in NEWS_FEATURE_NAMES}
        rows.append(row)

    result = pd.DataFrame(rows, index=trading_dates, columns=NEWS_FEATURE_NAMES)
    return result
