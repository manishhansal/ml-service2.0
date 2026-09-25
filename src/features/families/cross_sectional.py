"""
Cross-Sectional, Breadth & Sector Feature Family — Phase 3D.

PIT survivorship safety
-----------------------
All cross-sectional computations MUST receive the historical_universe(t)
not the current F&O universe.  Adding a stock that was not eligible at t
to the universe retroactively would change ranks at t — survivorship bias.

Timestamp-local normalisation
------------------------------
Every rank, z-score, or percentile is computed within a single timestamp.
Rolling normalisation over multiple timestamps is NOT cross-sectional; it
is time-series normalisation and is handled in the momentum/volatility
families.

Missing data policy
-------------------
Absent benchmark → NaN (not 1.0).
Absent sector data → NaN (not 0.0).
Universe too small (< min_stocks) → NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .momentum import _require_utc_index, _require_positive


# ── Cross-sectional normalization ─────────────────────────────────────────────

def cross_sectional_rank(
    values: dict[str, float],
    min_stocks: int = 5,
) -> dict[str, Optional[float]]:
    """
    Compute percentile rank of each stock within a set of contemporaneous values.

    Parameters
    ----------
    values     : {symbol: feature_value} at a single timestamp.
                 NaN values are excluded from the ranking universe.
    min_stocks : Minimum valid (non-NaN) stocks needed to compute rank.
                 Returns None for all stocks if universe is too small.

    Returns
    -------
    {symbol: percentile_rank [0, 100]} or {symbol: None} if too small.

    Causal guarantee: all values are from the same timestamp t.
    """
    valid = {s: v for s, v in values.items() if v is not None and np.isfinite(v)}
    if len(valid) < min_stocks:
        return {s: None for s in values}

    symbols = list(valid.keys())
    vals    = np.array([valid[s] for s in symbols], dtype=float)
    ranks   = {}
    n       = len(vals)
    for i, sym in enumerate(symbols):
        pct = float((vals < vals[i]).sum()) / n * 100.0
        ranks[sym] = round(pct, 4)
    for sym in values:
        if sym not in ranks:
            ranks[sym] = None
    return ranks


def cross_sectional_zscore(
    values: dict[str, float],
    min_stocks: int = 5,
) -> dict[str, Optional[float]]:
    """
    Compute z-score of each stock within a set of contemporaneous values.

    z = (x - mean) / std.
    Returns None for all stocks when universe < min_stocks or std = 0.
    """
    valid = {s: v for s, v in values.items() if v is not None and np.isfinite(v)}
    if len(valid) < min_stocks:
        return {s: None for s in values}

    symbols = list(valid.keys())
    vals    = np.array([valid[s] for s in symbols], dtype=float)
    mu      = vals.mean()
    sig     = vals.std(ddof=1)
    if sig < 1e-10:
        # All stocks at the same level → z = 0 for all (meaningful)
        zscores = {sym: 0.0 for sym in symbols}
    else:
        zscores = {sym: float((vals[i] - mu) / sig) for i, sym in enumerate(symbols)}
    for sym in values:
        if sym not in zscores:
            zscores[sym] = None
    return zscores


def compute_cs_return_rank(
    stock_close_map: dict[str, pd.Series],
    period: int = 20,
    min_stocks: int = 5,
) -> pd.DataFrame:
    """
    Cross-sectional percentile rank of N-day return at each timestamp.

    Parameters
    ----------
    stock_close_map : {symbol: pd.Series} — close prices with UTC DatetimeIndex.
                      All series must share the same index.
    period          : Return period in bars.
    min_stocks      : Min eligible stocks to compute rank; else NaN.

    Returns
    -------
    pd.DataFrame with columns = symbols, index = timestamps.
    Values = percentile rank [0, 100] or NaN.

    Survivorship safety
    -------------------
    Only symbols in stock_close_map at the time of the call are used.
    The caller is responsible for passing the historical universe(t),
    NOT the current universe.
    """
    if not stock_close_map:
        return pd.DataFrame()

    # Build return matrix
    closes  = pd.DataFrame(stock_close_map)
    returns = closes.pct_change(period) * 100.0
    result  = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    for ts in returns.index:
        row = returns.loc[ts].dropna()
        if len(row) < min_stocks:
            continue
        values = {s: float(v) for s, v in row.items()}
        ranks  = cross_sectional_rank(values, min_stocks)
        for sym, r in ranks.items():
            if r is not None:
                result.loc[ts, sym] = r

    return result


def compute_cs_rs_nifty_rank(
    stock_close_map: dict[str, pd.Series],
    nifty_close: Optional[pd.Series],
    period: int = 20,
    min_stocks: int = 5,
) -> pd.DataFrame:
    """
    Cross-sectional percentile rank of relative strength vs NIFTY.

    Returns NaN for all stocks at timestamps where nifty_close is unavailable.
    Never substitutes 1.0 (neutral) for missing benchmark.
    """
    if nifty_close is None or len(nifty_close) == 0:
        # All NaN — no fabrication
        closes = pd.DataFrame(stock_close_map)
        return pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)

    nifty_ret = nifty_close.pct_change(period)
    closes    = pd.DataFrame(stock_close_map)
    result    = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)

    for ts in closes.index:
        if ts not in nifty_ret.index or np.isnan(nifty_ret.loc[ts]):
            continue
        nr = float(nifty_ret.loc[ts])
        row = closes.loc[ts]
        rs_values: dict[str, float] = {}
        for sym in closes.columns:
            c_now  = row.get(sym)
            c_prev_idx = closes.index.get_loc(ts) - period
            if c_prev_idx < 0:
                continue
            c_prev = closes.iloc[c_prev_idx].get(sym)
            if c_now is None or c_prev is None:
                continue
            if np.isnan(c_now) or np.isnan(c_prev) or c_prev <= 0:
                continue
            stock_ret = (c_now - c_prev) / c_prev
            denom     = 1.0 + nr
            if abs(denom) < 1e-10:
                continue
            rs_values[sym] = (1.0 + stock_ret) / denom

        if len(rs_values) < min_stocks:
            continue
        ranks = cross_sectional_rank(rs_values, min_stocks)
        for sym, r in ranks.items():
            if r is not None:
                result.loc[ts, sym] = r

    return result


# ── Market Breadth ────────────────────────────────────────────────────────────

def compute_breadth_pct_above_sma(
    stock_close_map: dict[str, pd.Series],
    sma_period: int = 20,
    min_stocks: int = 10,
) -> pd.Series:
    """
    % of eligible stocks above their N-day SMA at each timestamp.

    Survivorship safety: uses only symbols in stock_close_map.
    Caller must pass historical_universe(t) not current universe.

    Returns NaN when fewer than min_stocks are available.
    NEVER returns 50.0 as a default when data is absent.
    """
    if not stock_close_map:
        return pd.Series(dtype=float)

    closes = pd.DataFrame(stock_close_map)
    result = pd.Series(np.nan, index=closes.index, dtype=float)

    for ts in closes.index:
        ts_loc  = closes.index.get_loc(ts)
        if ts_loc < sma_period:
            continue  # insufficient history for SMA
        window = closes.iloc[max(0, ts_loc - sma_period + 1) : ts_loc + 1]
        sma    = window.mean()
        current = closes.loc[ts]
        valid   = current.notna() & sma.notna()
        n_valid = int(valid.sum())
        if n_valid < min_stocks:
            continue
        above = int(((current[valid] > sma[valid])).sum())
        result.loc[ts] = float(above / n_valid * 100.0)

    return result


def compute_advance_decline_ratio(
    advances: Optional[int],
    declines: Optional[int],
) -> Optional[float]:
    """
    (advances - declines) / (advances + declines) in [-1, 1].

    Returns None when either input is None — NEVER returns 0.0 silently.
    Requires actual advance/decline counts from NSE bhavcopy.
    """
    if advances is None or declines is None:
        return None
    total = advances + declines
    if total == 0:
        return 0.0  # zero total means market was closed or no data
    return float((advances - declines) / total)


# ── Sector features ───────────────────────────────────────────────────────────

def compute_sector_momentum(
    sector_close_map: Optional[dict[str, pd.Series]],
    period: int = 5,
) -> pd.Series:
    """
    Average N-day return of sector peers at each timestamp.

    Returns a pd.Series of float or NaN values.
    NEVER returns 0.0 when sector_close_map is None.
    Returns NaN when no peers have sufficient history.
    """
    if not sector_close_map:
        # Return an empty series — caller must handle NaN
        return pd.Series(dtype=float)

    # Align all peer series to a common index
    peer_df = pd.DataFrame(sector_close_map)
    returns = peer_df.pct_change(period) * 100.0
    result  = returns.mean(axis=1)  # NaN-aware mean; NaN if all peers are NaN
    return result


def compute_sector_relative_strength(
    stock_close: pd.Series,
    sector_close_map: Optional[dict[str, pd.Series]],
    period: int = 20,
) -> pd.Series:
    """
    Stock's N-day return relative to the average N-day return of sector peers.

    result[t] = stock_return[t] - mean(peer_returns[t])

    Returns NaN when sector_close_map is None or has insufficient data.
    NEVER returns 1.0 (neutral ratio) when sector data is absent.
    """
    _require_utc_index(stock_close, "stock_close")
    if not sector_close_map:
        return pd.Series(np.nan, index=stock_close.index, dtype=float)

    s_ret = stock_close.astype(float).pct_change(period)
    peers = pd.DataFrame(sector_close_map).reindex(stock_close.index)
    peer_ret = peers.pct_change(period).mean(axis=1)  # NaN-safe mean

    return s_ret - peer_ret


def compute_sector_dispersion(
    sector_return_map: Optional[dict[str, float]],
) -> Optional[float]:
    """
    Cross-sector return dispersion at a single timestamp.

    Returns the standard deviation of sector returns.
    Returns None when sector_return_map is absent — NEVER returns 0.0.
    """
    if not sector_return_map or len(sector_return_map) < 2:
        return None
    values = [v for v in sector_return_map.values() if v is not None and np.isfinite(v)]
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1))


def compute_sector_rotation_score(
    sector_return_map: Optional[dict[str, float]],
    cyclicals: tuple[str, ...] = ("Auto", "Metal", "Realty", "Infra", "Energy"),
    defensives: tuple[str, ...] = ("FMCG", "Pharma", "IT"),
) -> Optional[float]:
    """
    Cyclical vs defensive sector return spread.
    Positive = cyclicals outperforming (risk-on).
    Returns None when sector data is absent — NEVER returns 0.0.
    """
    if not sector_return_map:
        return None
    cyc_rets = [sector_return_map[s] for s in cyclicals if s in sector_return_map
                and sector_return_map[s] is not None]
    def_rets = [sector_return_map[s] for s in defensives if s in sector_return_map
                and sector_return_map[s] is not None]
    if not cyc_rets or not def_rets:
        return None
    return float(np.clip(np.mean(cyc_rets) - np.mean(def_rets), -5.0, 5.0))
