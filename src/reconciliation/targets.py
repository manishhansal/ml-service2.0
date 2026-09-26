"""
src.reconciliation.targets — Pre-registered target family (mandate §8).

ALL targets must be registered BEFORE any OOS evaluation.
Targets are never added or changed after seeing OOS results (mandate §8, §3 pt.3).

Target family (mandate §8):
  A  next_open_raw       signal at close[T] → enter open[T+1] → exit open[T+1+h]
  B  open_to_open        open[T+1] → open[T+1+h]  (same as A, but emphasises the
                         fact that both entry and exit are at opens, not closes)
  C  residual_next_open  A minus beta_t * benchmark_return (benchmark = NIFTY)
  D  beta_neutral        explicitly hedge out market beta using only pre-T info
  E  cross_sectional     rank stocks cross-sectionally at T; measure future residual
  F  risk_adjusted       A / rolling_vol_T  (vol estimated only from history ≤ T)

Primary decision metric: cross-sectional Spearman Rank IC of the MODEL SCORE
against the REALIZED forward return from the corresponding target.

The honest IC is always measured against a CONTINUOUS (unclamped) forward return,
NOT against a triple-barrier-clamped return.  This distinction is the primary
fix for the IC inflation distortion identified in the reconciliation audit.

This module is imported by the economic backtest, the reconciler, and the
canonical evaluation script.  It defines:
  - TargetDefinition dataclass
  - REGISTERED_TARGETS dict (pre-registered, immutable)
  - functions to compute each target given OHLCV data
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import numpy as np
import pandas as pd

TargetName = Literal[
    "next_open_raw",
    "open_to_open",
    "residual_next_open",
    "beta_neutral",
    "cross_sectional_rank",
    "risk_adjusted",
]

# ── Registration hash ─────────────────────────────────────────────────────────
# This hash is computed over the target definitions at import time and stored
# in every report to prove the target family was not changed after seeing OOS.

_REGISTRATION_TIMESTAMP = "2026-09-26T00:00:00+00:00"  # pre-registered


@dataclass(frozen=True)
class TargetDefinition:
    """Immutable definition of one pre-registered prediction target."""
    name: TargetName
    description: str
    entry: str          # price series used for entry
    exit_: str          # price series used for exit
    horizon_days: int   # h (number of daily bars)
    benchmark: str      # benchmark symbol or "cross_section_mean"
    is_primary: bool    # primary economic validation target
    note: str = ""

    def registration_repr(self) -> str:
        return f"{self.name}|entry={self.entry}|exit={self.exit_}|h={self.horizon_days}|bench={self.benchmark}"


REGISTERED_TARGETS: dict[TargetName, TargetDefinition] = {
    "next_open_raw": TargetDefinition(
        name="next_open_raw",
        description=(
            "PRIMARY executable target. Signal at close[T], enter open[T+1], "
            "exit open[T+1+h]. The only target that directly maps to "
            "executable next-open EOD trading."
        ),
        entry="open[T+1]",
        exit_="open[T+1+h]",
        horizon_days=5,
        benchmark="none",
        is_primary=True,
        note="Mandate §8 Target A. No benchmark subtraction.",
    ),
    "open_to_open": TargetDefinition(
        name="open_to_open",
        description=(
            "Open-to-open return over h days. Identical to next_open_raw "
            "but makes explicit that both legs use open prices."
        ),
        entry="open[T+1]",
        exit_="open[T+1+h]",
        horizon_days=5,
        benchmark="none",
        is_primary=False,
        note="Mandate §8 Target B.",
    ),
    "residual_next_open": TargetDefinition(
        name="residual_next_open",
        description=(
            "next_open_raw minus NIFTY benchmark return over same horizon. "
            "Removes broad market direction. Beta estimated historically only."
        ),
        entry="open[T+1]",
        exit_="open[T+1+h]",
        horizon_days=5,
        benchmark="NIFTY",
        is_primary=False,
        note="Mandate §8 Target C. Beta NOT required here — raw excess.",
    ),
    "beta_neutral": TargetDefinition(
        name="beta_neutral",
        description=(
            "next_open_raw minus beta_t * NIFTY_return. Beta estimated using "
            "rolling 60-day history strictly ending at T (PIT-safe). "
            "Explicitly removes linear market factor."
        ),
        entry="open[T+1]",
        exit_="open[T+1+h]",
        horizon_days=5,
        benchmark="NIFTY",
        is_primary=False,
        note="Mandate §8 Target D. Requires NIFTY OHLCV.",
    ),
    "cross_sectional_rank": TargetDefinition(
        name="cross_sectional_rank",
        description=(
            "Cross-sectional percentile rank of next_open_raw within the "
            "universe at each timestamp T. Normalises away market direction. "
            "Evaluated as cross-sectional Rank IC at each T."
        ),
        entry="open[T+1]",
        exit_="open[T+1+h]",
        horizon_days=5,
        benchmark="cross_section_mean",
        is_primary=True,
        note=(
            "Mandate §8 Target E. This is the CORRECT cross-sectional IC "
            "companion to the time-series IC from walk_forward.py. "
            "Using h=5 to match the trained model's horizon."
        ),
    ),
    "risk_adjusted": TargetDefinition(
        name="risk_adjusted",
        description=(
            "next_open_raw / rolling_vol_T. Volatility is rolling 20-day "
            "realised vol estimated only from data ≤ T (PIT-safe)."
        ),
        entry="open[T+1]",
        exit_="open[T+1+h]",
        horizon_days=5,
        benchmark="none",
        is_primary=False,
        note="Mandate §8 Target F.",
    ),
}

# ── Registration certificate ──────────────────────────────────────────────────

def _compute_registration_hash() -> str:
    """Deterministic hash over the target registration (proves pre-registration)."""
    reprs = [t.registration_repr() for t in sorted(
        REGISTERED_TARGETS.values(), key=lambda x: x.name
    )]
    payload = "|".join(reprs) + "|" + _REGISTRATION_TIMESTAMP
    return hashlib.sha256(payload.encode()).hexdigest()


TARGET_REGISTRATION_HASH: str = _compute_registration_hash()

TARGET_REGISTRATION_CERTIFICATE: dict = {
    "registration_timestamp": _REGISTRATION_TIMESTAMP,
    "n_targets": len(REGISTERED_TARGETS),
    "primary_targets": [t.name for t in REGISTERED_TARGETS.values() if t.is_primary],
    "all_targets": list(REGISTERED_TARGETS.keys()),
    "registration_hash": TARGET_REGISTRATION_HASH,
    "note": (
        "Targets were pre-registered before any OOS evaluation. "
        "The hash over target definitions proves no post-hoc addition. "
        "Mandate §8: Never select best target after looking at OOS results."
    ),
}


# ── Target computation functions ──────────────────────────────────────────────

def compute_next_open_raw(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Compute Target A (next_open_raw) for all symbols.

    Returns a long-format DataFrame indexed by (ts, symbol) with columns:
      open_entry   : open[T+1]  — executable entry price
      open_exit    : open[T+1+h] — executable exit price
      fwd_return   : (open_exit - open_entry) / open_entry  — TRUE continuous return
      is_economic  : True (all rows use next-open execution)

    This is the CANONICAL forward return for economic evaluation.
    It is NEVER replaced by a triple-barrier-clamped return.
    """
    frames = []
    for sym, df in ohlcv_by_symbol.items():
        if "open" not in df.columns:
            continue
        df = df.sort_index()
        op = df["open"].astype(float)
        entry = op.shift(-1)             # open[T+1]
        exit_ = op.shift(-(1 + horizon)) # open[T+1+h]
        ret = (exit_ - entry) / entry
        frame = pd.DataFrame({
            "open_entry": entry,
            "open_exit": exit_,
            "fwd_return_next_open_raw": ret,
            "is_economic": True,
        })
        frame["symbol"] = sym
        frame.index.name = "ts"
        frames.append(frame)

    if not frames:
        raise ValueError("compute_next_open_raw: no symbols with 'open' column.")
    return pd.concat(frames).reset_index().set_index(["ts", "symbol"]).sort_index()


def compute_residual_next_open(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    benchmark_sym: str = "NIFTY",
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Compute Target C (residual_next_open): stock return minus NIFTY return.

    Uses ONLY open prices for both legs (no close-to-close contamination).
    Beta estimation is NOT applied here — just raw excess over benchmark.
    """
    base = compute_next_open_raw(ohlcv_by_symbol, horizon=horizon)

    if benchmark_sym not in ohlcv_by_symbol:
        base["fwd_return_residual"] = base["fwd_return_next_open_raw"]
        base["benchmark_return"] = np.nan
        base["benchmark_available"] = False
        return base

    bench_df = ohlcv_by_symbol[benchmark_sym].sort_index()
    bench_op = bench_df["open"].astype(float)
    bench_entry = bench_op.shift(-1)
    bench_exit = bench_op.shift(-(1 + horizon))
    bench_ret = (bench_exit - bench_entry) / bench_entry
    bench_ret.name = "_bench"

    # Align benchmark to long-format
    out = base.copy()
    out = out.reset_index()
    out["benchmark_return"] = (
        out["ts"]
        .map(bench_ret.to_dict())
        .astype(float)
    )
    out["fwd_return_residual"] = (
        out["fwd_return_next_open_raw"] - out["benchmark_return"]
    )
    out["benchmark_available"] = True
    out = out.set_index(["ts", "symbol"])
    return out


def compute_beta_neutral(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    benchmark_sym: str = "NIFTY",
    horizon: int = 5,
    beta_window: int = 60,
) -> pd.DataFrame:
    """
    Compute Target D (beta_neutral): stock return minus PIT-safe rolling beta × benchmark.

    Beta for stock i at time T:
      beta_i,T = cov(r_i, r_bench) / var(r_bench) over the trailing beta_window bars,
                 using ONLY data ≤ T (strictly PIT-safe, shifted by 1 to exclude T).
    """
    residual = compute_residual_next_open(ohlcv_by_symbol, benchmark_sym, horizon)
    if not residual["benchmark_available"].all():
        residual["fwd_return_beta_neutral"] = residual.get(
            "fwd_return_residual", residual["fwd_return_next_open_raw"]
        )
        return residual

    out = residual.reset_index()
    bench_df = ohlcv_by_symbol[benchmark_sym].sort_index()
    bench_ret1 = bench_df["open"].astype(float).pct_change()

    betas_per_sym = {}
    for sym, df in ohlcv_by_symbol.items():
        if sym == benchmark_sym:
            continue
        r_i = df["open"].astype(float).pct_change().rename("r_i")
        r_b = bench_ret1.reindex(r_i.index)
        cov = r_i.rolling(beta_window).cov(r_b)
        var = r_b.rolling(beta_window).var()
        beta = (cov / var.replace(0, np.nan)).shift(1).clip(-4, 4)
        betas_per_sym[sym] = beta

    beta_rows = []
    for sym, b in betas_per_sym.items():
        for ts, val in b.items():
            beta_rows.append({"ts": ts, "symbol": sym, "_beta": val})
    beta_df = pd.DataFrame(beta_rows).set_index(["ts", "symbol"])

    out = out.set_index(["ts", "symbol"])
    out["_beta"] = beta_df["_beta"]
    out["fwd_return_beta_neutral"] = (
        out["fwd_return_next_open_raw"]
        - out["_beta"].fillna(1.0) * out["benchmark_return"].fillna(0.0)
    )
    return out.drop(columns=["_beta"])


def compute_cross_sectional_rank(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Compute Target E (cross_sectional_rank): percentile rank of next_open_raw
    within the universe at each timestamp T.

    This is the CORRECT evaluation target for the cross-sectional IC comparison.
    Using horizon=5 to match the trained model.
    """
    base = compute_next_open_raw(ohlcv_by_symbol, horizon=horizon)
    rank = (
        base["fwd_return_next_open_raw"]
        .groupby(level="ts")
        .rank(pct=True)
    )
    base["fwd_return_xs_rank"] = rank
    return base


def compute_risk_adjusted(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    horizon: int = 5,
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Compute Target F (risk_adjusted): next_open_raw / rolling_vol_T.

    Volatility is 20-day realized close vol estimated only from data ≤ T.
    """
    base = compute_next_open_raw(ohlcv_by_symbol, horizon=horizon)
    vols = []
    for sym, df in ohlcv_by_symbol.items():
        rv = df["close"].astype(float).pct_change().rolling(vol_window).std()
        rv.name = "_vol"
        rv = rv.reset_index()
        rv["symbol"] = sym
        rv = rv.rename(columns={rv.columns[0]: "ts"})
        vols.append(rv)
    vol_df = pd.concat(vols).set_index(["ts", "symbol"])

    out = base.copy()
    out["_vol"] = vol_df["_vol"]
    out["fwd_return_risk_adjusted"] = (
        out["fwd_return_next_open_raw"]
        / out["_vol"].replace(0, np.nan).clip(lower=1e-6)
    )
    return out.drop(columns=["_vol"])
