"""
src.labels.relative — Relative (excess-return) label V2.

generate_ranking_labels_v2_compat():
  Entry point used by data_pipeline.py generate_ranking_labels_v2().
  Returns a pd.Series of vol-adjusted excess return vs NIFTY.

  Formula per bar i:
    stock_fwd   = (stock.close[i+h] - stock.close[i]) / stock.close[i]
    nifty_fwd   = (nifty_close[i+h] - nifty_close[i]) / nifty_close[i]
    excess      = stock_fwd - nifty_fwd
    vol         = stock.close.pct_change().rolling(20).std().iloc[i]
    ra_label    = excess / vol  (clipped to [-5, 5])

  PIT-safe: only data up to and including bar i is used for vol.
  Label is NaN for the last `horizon` bars (no forward data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ranking_labels_v2_compat(
    stock_df: pd.DataFrame,
    nifty_close: pd.Series,
    horizon: int = 5,
) -> pd.Series:
    """
    Vol-adjusted excess return label vs NIFTY.
    Returns pd.Series aligned to stock_df.index, NaN for tail bars.
    """
    close = stock_df["close"].astype(float)
    nc = nifty_close.astype(float).reindex(close.index, method="ffill")

    stock_fwd = close.shift(-horizon) / close - 1.0
    nifty_fwd = nc.shift(-horizon) / nc - 1.0
    excess = stock_fwd - nifty_fwd

    # Causal 20-day realized vol (only uses past data)
    vol = close.pct_change().rolling(20).std()
    vol = vol.replace(0, np.nan)

    ra = excess / vol
    ra = ra.clip(-5.0, 5.0)

    # Tail bars must be NaN (no forward data)
    ra.iloc[-horizon:] = np.nan

    return ra
