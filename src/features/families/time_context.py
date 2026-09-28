"""
src/features/families/time_context.py — Time-Context Feature Family

Features that encode the temporal context of a bar:
  - weekday (Monday effect, Friday effect)
  - month-end / quarter-end proximity
  - NSE F&O expiry proximity (last Thursday of each month)
  - intraday session phase (for intraday bars)

These features are among the cheapest to compute and often among the
most predictive for short-horizon strategies due to calendar effects
in institutional rebalancing, F&O expiry rollovers, and fund flows.

FIX NEW-P0-002: Time context features were completely absent from the
existing FeatureFactory.

All features are strictly CAUSAL: they use only the bar's timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_calendar_features(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """
    Calendar-based time-context features for daily or intraday bars.

    Returns
    -------
    weekday             : 0=Monday … 4=Friday (float for model use)
    weekday_sin         : sin(2π × weekday / 5) — cyclical encoding
    weekday_cos         : cos(2π × weekday / 5)
    month_end_proximity : trading days until end of calendar month (0 = last day)
    quarter_end         : 1 if last 5 days of quarter, 0 otherwise
    is_monday           : 1 if Monday, 0 otherwise
    is_friday           : 1 if Friday, 0 otherwise
    """
    if index.tz is None:
        # Try to convert; if it fails, work with naive index
        try:
            index = index.tz_localize("UTC")
        except Exception:
            pass

    wday = pd.Series(index.dayofweek.astype(float), index=index)
    wday_sin = np.sin(2 * np.pi * wday / 5.0)
    wday_cos = np.cos(2 * np.pi * wday / 5.0)

    # Days until end of month (approximate using days in month)
    days_in_month = pd.Series(index.days_in_month, index=index).astype(float)
    day_of_month = pd.Series(index.day, index=index).astype(float)
    month_end_proximity = days_in_month - day_of_month

    # Quarter-end: last 5 calendar days of each quarter
    quarter_month_ends = {3: True, 6: True, 9: True, 12: True}
    is_quarter_month = pd.Series(
        [1.0 if m in quarter_month_ends else 0.0 for m in index.month],
        index=index,
    )
    quarter_end = (is_quarter_month * (month_end_proximity <= 5)).astype(float)

    is_monday = (wday == 0).astype(float)
    is_friday = (wday == 4).astype(float)

    return {
        "weekday": wday,
        "weekday_sin": wday_sin,
        "weekday_cos": wday_cos,
        "month_end_proximity": month_end_proximity,
        "quarter_end": quarter_end,
        "is_monday": is_monday,
        "is_friday": is_friday,
    }


def compute_expiry_proximity(
    index: pd.DatetimeIndex,
    lookback_days: int = 10,
) -> pd.Series:
    """
    NSE F&O expiry proximity — days until the next monthly expiry.

    NSE F&O contracts expire on the last Thursday of each month.
    This feature encodes proximity (0 = expiry day, 1 = 1 day before, etc.).
    High values indicate far from expiry; low values indicate expiry week.

    This is a valuable regime feature because:
    - Expiry week: high open interest rollovers, increased volatility
    - Post-expiry: fresh positions, lower OI
    - Pre-expiry: gamma / vega effects peak

    Returns
    -------
    pd.Series : trading days until next expiry (float, NaN if cannot compute)
    """
    if index.tz is None:
        try:
            index = index.tz_localize("UTC")
        except Exception:
            pass

    # Compute last Thursday of each month for the date range
    def _last_thursday_of_month(year: int, month: int) -> pd.Timestamp:
        # Last day of month
        last_day = pd.Timestamp(year=year, month=month, day=1, tz="UTC") + pd.offsets.MonthEnd(0)
        # Walk back to Thursday (weekday 3)
        while last_day.weekday() != 3:
            last_day -= pd.Timedelta(days=1)
        return last_day

    # Build expiry calendar for the range
    if len(index) == 0:
        return pd.Series(dtype=float)

    start = index.min()
    end = index.max()
    expiries: list[pd.Timestamp] = []
    yr, mo = start.year, start.month
    while True:
        exp = _last_thursday_of_month(yr, mo)
        expiries.append(exp)
        if exp >= end + pd.Timedelta(days=31):
            break
        mo += 1
        if mo > 12:
            mo = 1
            yr += 1

    expiry_arr = sorted(expiries)

    # For each bar, find the next expiry and compute business days until it
    proximity = []
    for ts in index:
        future = [e for e in expiry_arr if e >= ts]
        if future:
            nxt = future[0]
            # Approximate trading days: calendar days × (252/365)
            cal_days = (nxt - ts).days
            approx_td = int(round(cal_days * 252 / 365))
            proximity.append(float(approx_td))
        else:
            proximity.append(np.nan)

    return pd.Series(proximity, index=index, name="expiry_proximity")


def compute_intraday_session_features(
    index: pd.DatetimeIndex,
    session_open_hour: int = 9,
    session_open_min: int = 15,
    session_close_hour: int = 15,
    session_close_min: int = 30,
) -> dict[str, pd.Series]:
    """
    Intraday session-phase features for intraday bars.

    Returns
    -------
    session_progress   : 0.0 (open) → 1.0 (close) fraction through session
    is_opening_phase   : 1 if first 30 minutes of session
    is_closing_phase   : 1 if last 30 minutes of session
    minutes_from_open  : minutes elapsed since session open
    """
    if index.tz is None:
        try:
            index = index.tz_localize("UTC")
        except Exception:
            pass

    # Convert to IST for NSE session reference (UTC+5:30)
    try:
        ist_index = index.tz_convert("Asia/Kolkata")
    except Exception:
        ist_index = index  # fallback

    session_open_minutes = session_open_hour * 60 + session_open_min
    session_close_minutes = session_close_hour * 60 + session_close_min
    session_duration = session_close_minutes - session_open_minutes

    bar_minutes = pd.Series(
        ist_index.hour * 60 + ist_index.minute, index=index
    ).astype(float)

    minutes_from_open = (bar_minutes - session_open_minutes).clip(lower=0)
    session_progress = (minutes_from_open / session_duration).clip(0, 1)
    is_opening_phase = (minutes_from_open <= 30).astype(float)
    is_closing_phase = (
        (bar_minutes >= session_close_minutes - 30) &
        (bar_minutes <= session_close_minutes)
    ).astype(float)

    return {
        "session_progress": session_progress,
        "is_opening_phase": is_opening_phase,
        "is_closing_phase": is_closing_phase,
        "minutes_from_open": minutes_from_open,
    }
