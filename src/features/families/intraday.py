"""
src.features.families.intraday — Intraday Feature Family (Group G).

Why intraday data improves EOD predictions
-------------------------------------------
The EOD model sees only daily OHLCV bars.  This means two fundamentally
different days look identical:

  Day A: Stock opens +0.5%, climbs all day, closes +0.5% → accumulation signal
  Day B: Stock opens +2.5%, gives back all gains, closes +0.5% → distribution signal

From EOD bars both days show close_ret = +0.5%.  From 5m bars:
  Day A: monotonic intraday trend, close > VWAP, volume concentrated in afternoon
  Day B: downtrend after open, close < VWAP, volume spike at open then fading

The 5-minute intraday features in Group G capture these intraday dynamics
that are invisible to EOD features.  Each feature is computed PER DATE from
that day's 5m bars and merged into the daily row for model training.

Features (Group G — 8 features):
    intraday_morning_ret      Return from first bar to 11:00 AM (bullish/bearish open)
    intraday_close_vs_vwap    (close - VWAP) / close: +1 = close above VWAP (bullish)
    intraday_vol_profile      Volume in first 30 min / total daily volume (institutional activity)
    intraday_realized_vol     Realized vol from 5m log-returns (intraday risk measure)
    intraday_range_pct        (high - low) / open: intraday range as % of price
    intraday_last_hour_ret    Return in final hour 14:30→15:30 IST (smart-money activity)
    intraday_open_to_high_pct (high - open) / open: upper potential seen during day
    intraday_open_to_low_pct  (open - low) / open: downside seen during day (positive = lower low)

PIT safety:
    All features use only the bars from date D to characterise date D.
    No future-bar data is used.  At EOD scoring time all bars for the day
    are available (09:15→15:30 IST = 75 × 5-minute bars).

Partial-day handling (live scoring):
    If bars are incomplete (< 30 bars), features that require late-day data
    (intraday_last_hour_ret) are set to 0.0 (neutral).  This is safe —
    the model was trained on complete days; neutral fill at inference degrades
    gracefully rather than causing directional bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical intraday feature names — must match DatasetBuilder merge logic
INTRADAY_FEATURE_NAMES: list[str] = [
    "intraday_morning_ret",
    "intraday_close_vs_vwap",
    "intraday_vol_profile",
    "intraday_realized_vol",
    "intraday_range_pct",
    "intraday_last_hour_ret",
    "intraday_open_to_high_pct",
    "intraday_open_to_low_pct",
]

# NSE IST timezone offset from UTC (UTC+5:30)
_IST_OFFSET_HRS = 5.5

# NSE session bounds in IST (hour, minute)
_SESSION_OPEN  = (9, 15)
_SESSION_CLOSE = (15, 30)
_MORNING_END   = (11,  0)   # 12 × 5m bars from open
_LAST_HR_START = (14, 30)   # start of last hour


def _to_ist_hour_minute(ts: pd.Timestamp) -> tuple[int, int]:
    """Convert a UTC timestamp to IST (hour, minute)."""
    if ts.tzinfo is not None:
        # Already tz-aware — convert to IST
        ist = ts.tz_convert("Asia/Kolkata")
    else:
        # Assume UTC if no timezone
        ist = ts + pd.Timedelta(hours=_IST_OFFSET_HRS)
    return ist.hour, ist.minute


def compute_intraday_features_for_date(
    df_5m: pd.DataFrame,
    is_partial: bool = False,
) -> dict[str, float]:
    """Compute intraday Group G features from a single day's 5m bars.

    Args:
        df_5m:      DataFrame with columns [open, high, low, close, volume],
                    index = DatetimeIndex of bar timestamps (any tz).
                    Should contain ONLY bars from one trading date.
        is_partial: True when the day is incomplete (live scoring).
                    Late-day features (last_hour_ret) are set to 0.0.

    Returns:
        Dict of {feature_name: float}.  All values are finite scalars.
        On insufficient data (< 6 bars), all features returned as 0.0.
    """
    neutral: dict[str, float] = {k: 0.0 for k in INTRADAY_FEATURE_NAMES}

    if df_5m is None or len(df_5m) < 6:
        return neutral

    df = df_5m.copy()
    df.columns = [c.lower() for c in df.columns]

    # Ensure numeric
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < 6:
        return neutral

    close  = df["close"]
    open_  = df["open"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    volume = volume.fillna(0)

    day_open  = float(open_.iloc[0])
    day_close = float(close.iloc[-1])
    day_high  = float(high.max())
    day_low   = float(low.min())
    total_vol = float(volume.sum())

    if day_open <= 0:
        return neutral

    # ── Morning return (open → 11:00 AM IST) ─────────────────────────────────
    # Select bars where time ≤ 11:00 IST
    morning_bars = df[df.index.map(
        lambda ts: (h := _to_ist_hour_minute(ts)[0]) < 11
                   or (h == 11 and _to_ist_hour_minute(ts)[1] == 0)
    )]
    if len(morning_bars) >= 2:
        morning_close = float(morning_bars["close"].iloc[-1])
        morning_ret   = (morning_close - day_open) / day_open
    else:
        morning_ret = 0.0

    # ── VWAP ─────────────────────────────────────────────────────────────────
    # VWAP = sum(typical_price × volume) / sum(volume)
    typical_price = (high + low + close) / 3.0
    if total_vol > 0:
        vwap = float((typical_price * volume).sum() / total_vol)
    else:
        vwap = float(typical_price.mean())
    close_vs_vwap = (day_close - vwap) / day_close if day_close > 0 else 0.0

    # ── Volume profile — first 30 min (6 bars) / total ───────────────────────
    first_6_vol = float(volume.iloc[:6].sum())
    vol_profile = (first_6_vol / total_vol) if total_vol > 0 else 0.0
    vol_profile = float(np.clip(vol_profile, 0.0, 1.0))

    # ── Realized volatility from 5m log-returns ──────────────────────────────
    log_rets = np.log(close / close.shift(1)).dropna()
    realized_vol = float(log_rets.std() * np.sqrt(75)) if len(log_rets) >= 4 else 0.0  # annualise to daily

    # ── Intraday range ────────────────────────────────────────────────────────
    range_pct = (day_high - day_low) / day_open if day_open > 0 else 0.0

    # ── Last hour return (14:30–15:30 IST) ───────────────────────────────────
    if is_partial:
        last_hour_ret = 0.0
    else:
        last_hr_bars = df[df.index.map(
            lambda ts: (h := _to_ist_hour_minute(ts)[0]) >= 14
                       and (h > 14 or _to_ist_hour_minute(ts)[1] >= 30)
        )]
        if len(last_hr_bars) >= 2:
            lh_open  = float(last_hr_bars["open"].iloc[0])
            lh_close = float(last_hr_bars["close"].iloc[-1])
            last_hour_ret = (lh_close - lh_open) / lh_open if lh_open > 0 else 0.0
        else:
            last_hour_ret = 0.0

    # ── Open-to-high / open-to-low ────────────────────────────────────────────
    open_to_high_pct = (day_high - day_open) / day_open
    open_to_low_pct  = (day_open - day_low)  / day_open   # positive = lower low

    # ── Clip all values to [-5, 5] to suppress outliers ──────────────────────
    def _clip(x: float) -> float:
        return float(np.clip(x, -5.0, 5.0)) if np.isfinite(x) else 0.0

    return {
        "intraday_morning_ret":    _clip(morning_ret),
        "intraday_close_vs_vwap":  _clip(close_vs_vwap),
        "intraday_vol_profile":    float(np.clip(vol_profile, 0.0, 1.0)),
        "intraday_realized_vol":   _clip(realized_vol),
        "intraday_range_pct":      _clip(range_pct),
        "intraday_last_hour_ret":  _clip(last_hour_ret),
        "intraday_open_to_high_pct": _clip(open_to_high_pct),
        "intraday_open_to_low_pct":  _clip(open_to_low_pct),
    }


def compute_intraday_feature_matrix(
    df_5m_full: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute intraday features for every date in trading_dates.

    Args:
        df_5m_full:     Full 5m bar history for one symbol.  Index must be
                        DatetimeIndex.  Columns: open, high, low, close, volume.
        trading_dates:  The dates in the daily training dataset for this symbol.
                        Each date gets one row of intraday features.

    Returns:
        DataFrame with index = trading_dates (date-normalised to midnight UTC),
        columns = INTRADAY_FEATURE_NAMES.
        Dates without 5m bars get all-zero rows (neutral fill).
    """
    if df_5m_full is None or len(df_5m_full) == 0:
        result = pd.DataFrame(0.0, index=trading_dates, columns=INTRADAY_FEATURE_NAMES)
        return result

    df = df_5m_full.copy()
    df.columns = [c.lower() for c in df.columns]

    # Normalise index to UTC-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Group by calendar date (IST date = UTC date for most NSE sessions)
    # Convert to IST date for grouping
    try:
        df_ist = df.copy()
        df_ist.index = df.index.tz_convert("Asia/Kolkata")
        date_groups = df_ist.groupby(df_ist.index.date)
    except Exception:
        # Fallback: group by UTC date
        date_groups = df.groupby(df.index.date)

    # Pre-compute features per date
    features_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    for date, day_bars in date_groups:
        feats = compute_intraday_features_for_date(day_bars.reset_index(drop=False).set_index(day_bars.index))
        # Normalise date to midnight UTC for alignment
        ts = pd.Timestamp(date).normalize()
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        features_by_date[ts] = feats

    # Build output DataFrame aligned to trading_dates
    rows: list[dict[str, float]] = []
    norm_dates: list[pd.Timestamp] = []

    for td in trading_dates:
        # Normalise trading date to midnight UTC
        td_norm = pd.Timestamp(td).normalize()
        if td_norm.tzinfo is None:
            td_norm = td_norm.tz_localize("UTC")
        elif str(td_norm.tzinfo) != "UTC":
            td_norm = td_norm.tz_convert("UTC")

        feats = features_by_date.get(td_norm, {k: 0.0 for k in INTRADAY_FEATURE_NAMES})
        rows.append(feats)
        norm_dates.append(td)

    result = pd.DataFrame(rows, index=trading_dates, columns=INTRADAY_FEATURE_NAMES)
    return result
