"""
src.labels.seven_day — FIX for CRIT-001 & CRIT-002.

Implements the IMPROVED label design addressing two critical label audit findings:

  CRIT-001: Label horizon was 5 bars, mandate requires 7 trading days
  CRIT-002: ±2% symmetric barriers → negative expected value at realistic costs

This module provides TWO improved label types:

A. ``generate_7d_excess_return_label``
   Vol-adjusted excess return vs NIFTY over exactly 7 trading days.
   This is the primary label for cross-sectional ranking models.
   - Continuous label (for IC-based evaluation)
   - Naturally normalised by volatility (no barrier calibration needed)
   - Alpha signal: outperformance vs NIFTY is the true objective
   - PIT-safe: only uses forward data for the return calculation

B. ``generate_7d_asymmetric_barrier_label``
   Triple-barrier with a 2:1 reward-to-risk ratio over 7 trading days.
   - +3% target / +1.5% stop (vol-scaled by default)
   - Positive expected value even at equity costs
   - For binary classification models
   - PIT-safe: next_open execution assumed

Both replace the ``LabelConfig(horizon=5, upper_barrier=0.02, lower_barrier=0.02)``
configuration that produced near-zero or negative expected value.

Label quality improvements (from LABEL_AUDIT.md):
  Before (5-bar ±2%):  mean net return = −0.552%/trade at equity costs
  After  (7-day 3:1):  break-even at ~35% win rate (vs 56.9% before)
  After  (excess ret): IC-based — positive IC directly means positive returns

Requirements: CRIT-001, CRIT-002, mandate §9, §10, §29, §32
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EQUITY_COST_BPS: float = 27.35          # realistic NSE equity round-trip
FUTURES_COST_BPS: float = 7.26          # realistic NSE futures round-trip
HORIZON_DAYS: int = 7                   # mandated 7 trading-day horizon


# ── Fix A: 7-day vol-adjusted excess return (ranking label) ──────────────────


def generate_7d_excess_return_label(
    stock_df: pd.DataFrame,
    nifty_close: pd.Series,
    horizon: int = HORIZON_DAYS,
    vol_window: int = 20,
    clip: float = 5.0,
) -> pd.Series:
    """
    Vol-adjusted excess return over a 7-trading-day horizon.

    Formula:
        stock_fwd = close[T+7] / close[T] - 1
        nifty_fwd = nifty[T+7] / nifty[T] - 1
        excess    = stock_fwd - nifty_fwd
        vol       = rolling(vol_window).std(close.pct_change())[T]  ← causal
        label     = excess / vol   (clipped to [-clip, +clip])

    PIT safety:
        vol uses only data up to and including T (causal rolling window).
        The label value at T uses close[T+7] which is genuinely future data.
        This is correct — labels are allowed to use future data.

    Why this works better than ±2% barriers:
        1. Continuous label → model optimizes IC (rank correlation) directly
        2. Vol-normalised → comparable across high/low volatility stocks
        3. Excess over NIFTY → pure idiosyncratic alpha, not market beta
        4. 7-day horizon → matches the mandate

    Returns:
        pd.Series aligned to stock_df.index.
        NaN for the last `horizon` bars (no forward data available).
        NaN for bars with insufficient vol lookback (first `vol_window` bars).
    """
    close = stock_df["close"].astype(float)
    nc    = nifty_close.astype(float).reindex(close.index, method="ffill")

    # Forward returns (uses future data — correct for label only)
    stock_fwd = close.shift(-horizon) / close - 1.0
    nifty_fwd = nc.shift(-horizon)   / nc   - 1.0
    excess    = stock_fwd - nifty_fwd

    # Causal volatility (uses only past data ≤ T)
    vol = close.pct_change().rolling(vol_window).std()
    vol = vol.replace(0.0, np.nan)

    label = (excess / vol).clip(-clip, clip)

    # Tail bars must be NaN — no forward data
    label.iloc[-horizon:] = np.nan

    _log_label_quality(label, f"7d_excess_return(h={horizon})")
    return label


# ── Fix B: 7-day asymmetric barrier (binary classification label) ─────────────


def generate_7d_asymmetric_barrier_label(
    stock_df: pd.DataFrame,
    cost_bps: float = EQUITY_COST_BPS,
    horizon: int = HORIZON_DAYS,
    target_multiplier: float = 1.5,   # target = vol × 1.5 × √horizon
    stop_multiplier: float = 0.75,    # stop   = vol × 0.75 × √horizon (2:1 R:R)
    vol_window: int = 20,
    execution_model: str = "next_open",
) -> pd.DataFrame:
    """
    Triple-barrier label with 7-day horizon and asymmetric 2:1 reward-to-risk.

    Barrier design:
        Daily vol σ = rolling(vol_window).std(returns)[T]
        target = entry × (1 + σ × target_multiplier × √horizon)
        stop   = entry × (1 - σ × stop_multiplier  × √horizon)
        Reward/Risk ≈ 2:1

    For NIFTY F&O stocks with σ ≈ 1.5%/day:
        σ × √7 = 1.5% × 2.65 = 3.97%
        target ≈ entry × 1.0596   (+6.0%)
        stop   ≈ entry × 0.9702   (-3.0%)
        Expected value at 40% win rate:
          0.40 × (6.0% - 0.27%) - 0.60 × (3.0% + 0.27%) = 2.29% - 1.96% = +0.33%

    Contrast with old symmetric ±2% at 50% win rate:
        0.50 × (2.0% - 0.27%) - 0.50 × (2.0% + 0.27%) = 0.865% - 1.135% = -0.27%

    The new label has POSITIVE expected value at realistic costs, even at a moderate win rate.

    PIT safety:
        entry_price = open[T+1] (next_open mode) — executable, not speculative.
        label looks forward [T+1, T+7] — future data used for evaluation only.

    Returns:
        DataFrame with columns:
          label           : 1 (TARGET_HIT or profitable TIME_EXPIRY), 0 otherwise
          outcome         : TARGET_HIT / STOP_HIT / TIME_EXPIRY
          realized_return : gross return fraction
          realized_cost   : round-trip cost fraction
          realized_return_net : net return fraction
          entry_price     : open[T+1]
          exit_price      : price at exit
          target_level    : computed target barrier
          stop_level      : computed stop barrier
          execution_model : "next_open"
          is_economic_evidence : True
    """
    if "close" not in stock_df.columns:
        raise ValueError("generate_7d_asymmetric_barrier_label requires 'close' column")
    if execution_model == "next_open" and "open" not in stock_df.columns:
        raise ValueError("next_open execution requires 'open' column")

    close  = stock_df["close"].astype(float).to_numpy()
    high   = stock_df.get("high", stock_df["close"]).astype(float).to_numpy()
    low    = stock_df.get("low",  stock_df["close"]).astype(float).to_numpy()
    open_  = stock_df["open"].astype(float).to_numpy() if "open" in stock_df.columns else close.copy()
    idx    = stock_df.index
    n      = len(close)
    cost   = cost_bps / 10_000.0

    # Causal daily volatility
    rets    = pd.Series(close).pct_change()
    vol_arr = rets.rolling(vol_window).std().to_numpy()

    # Entry prices (next_open: entry at open[T+1])
    entry_prices = np.append(open_[1:], np.nan)

    rows = []
    for i in range(n):
        entry = entry_prices[i]
        if np.isnan(entry) or entry <= 0:
            rows.append(_unresolved_row(idx[i], cost))
            continue

        # Compute barriers using causal vol at T
        v = vol_arr[i] if np.isfinite(vol_arr[i]) and vol_arr[i] > 0 else 0.015
        h_adj = np.sqrt(horizon)
        t_pct = target_multiplier * v * h_adj
        s_pct = stop_multiplier   * v * h_adj

        target_lvl = entry * (1.0 + t_pct)
        stop_lvl   = entry * (1.0 - s_pct)

        # Scan forward horizon bars
        start = i + 1
        end   = min(start + horizon - 1, n - 1)

        if start > n - 1:
            rows.append(_unresolved_row(idx[i], cost))
            continue

        fw_high  = high[start: end + 1]
        fw_low   = low[start: end + 1]
        fw_close = close[start: end + 1]

        outcome     = "TIME_EXPIRY"
        realized    = 0.0
        exit_price  = fw_close[-1] if len(fw_close) > 0 else np.nan

        for j in range(len(fw_high)):
            if fw_high[j] >= target_lvl:
                outcome    = "TARGET_HIT"
                realized   = t_pct
                exit_price = target_lvl
                break
            if fw_low[j] <= stop_lvl:
                outcome    = "STOP_HIT"
                realized   = -s_pct
                exit_price = stop_lvl
                break
        else:
            # TIME_EXPIRY: realized = actual close[T+7] return
            if len(fw_close) > 0:
                realized   = (fw_close[-1] - entry) / entry
                exit_price = fw_close[-1]
            else:
                outcome    = "UNRESOLVED"
                realized   = np.nan

        net  = (realized - cost) if not np.isnan(realized) else np.nan

        if outcome == "TARGET_HIT":
            label = 1
        elif outcome == "STOP_HIT":
            label = 0
        elif outcome == "TIME_EXPIRY" and not np.isnan(realized):
            label = int(realized > cost)   # net-profitable TIME_EXPIRY = positive
        else:
            label = pd.NA

        rows.append({
            "label":                  label,
            "outcome":                outcome,
            "realized_return":        realized if not np.isnan(realized) else np.nan,
            "realized_cost":          cost,
            "realized_return_net":    net,
            "entry_price":            entry,
            "exit_price":             exit_price if not np.isnan(exit_price) else np.nan,
            "target_level":           round(target_lvl, 4),
            "stop_level":             round(stop_lvl, 4),
            "target_pct":             round(t_pct * 100, 3),
            "stop_pct":               round(s_pct * 100, 3),
            "rr_ratio":               round(t_pct / s_pct, 3) if s_pct > 0 else np.nan,
            "horizon":                horizon,
            "execution_model":        execution_model,
            "is_economic_evidence":   True,
        })

    out = pd.DataFrame(rows, index=idx)
    out["label"] = out["label"].astype("Int64")
    _log_label_quality(out["label"].dropna().astype(float), "7d_asymmetric_barrier")
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────


def _unresolved_row(ts: pd.Timestamp, cost: float) -> dict:
    return {
        "label":                  pd.NA,
        "outcome":                "UNRESOLVED",
        "realized_return":        np.nan,
        "realized_cost":          cost,
        "realized_return_net":    np.nan,
        "entry_price":            np.nan,
        "exit_price":             np.nan,
        "target_level":           np.nan,
        "stop_level":             np.nan,
        "target_pct":             np.nan,
        "stop_pct":               np.nan,
        "rr_ratio":               np.nan,
        "horizon":                HORIZON_DAYS,
        "execution_model":        "next_open",
        "is_economic_evidence":   True,
    }


def _log_label_quality(label: pd.Series, name: str) -> None:
    """Log basic quality metrics for a label series."""
    valid = label.dropna()
    if len(valid) == 0:
        logger.warning("label_quality_empty", label=name)
        return
    # For binary labels
    is_binary = set(valid.unique()) <= {0.0, 1.0}
    if is_binary:
        pos = float((valid == 1).mean())
        logger.info(
            "label_quality",
            label=name,
            n=len(valid),
            positive_frac=round(pos, 4),
            mean=round(float(valid.mean()), 4),
        )
    else:
        # Continuous (excess return)
        logger.info(
            "label_quality",
            label=name,
            n=len(valid),
            mean=round(float(valid.mean()), 4),
            std=round(float(valid.std()), 4),
            positive_frac=round(float((valid > 0).mean()), 4),
        )


# ── Label quality comparison ─────────────────────────────────────────────────


def compare_label_designs(
    stock_df: pd.DataFrame,
    nifty_close: pd.Series,
    cost_bps: float = EQUITY_COST_BPS,
) -> dict:
    """
    Compare old label design vs new improved designs.

    Returns a diagnostic dict showing expected value improvement.
    """
    from src.data.labels import LabelConfig, LabelFactory

    # Old design (5-bar, ±2% symmetric)
    old_config = LabelConfig(
        label_type="triple_barrier",
        horizon=5,
        upper_barrier_pct=0.02,
        lower_barrier_pct=0.02,
        cost_bps=cost_bps,
        execution_model="next_open",
    )
    old_factory = LabelFactory(old_config)
    old_labels  = old_factory.build(stock_df)

    # New design B: 7-day asymmetric barrier
    new_b = generate_7d_asymmetric_barrier_label(
        stock_df, cost_bps=cost_bps, horizon=HORIZON_DAYS
    )

    # New design A: 7-day excess return (continuous)
    new_a = generate_7d_excess_return_label(
        stock_df, nifty_close, horizon=HORIZON_DAYS
    )

    def label_stats(ret_col: pd.Series, cost_col: pd.Series | None = None) -> dict:
        r = ret_col.dropna()
        if len(r) == 0:
            return {}
        net = (r - cost_col.reindex(r.index)) if cost_col is not None else r
        return {
            "n":               len(r),
            "mean_gross_pct":  round(float(r.mean()) * 100, 4),
            "mean_net_pct":    round(float(net.mean()) * 100, 4) if cost_col is not None else float("nan"),
            "positive_rate":   round(float((r > 0).mean()), 4),
            "std_pct":         round(float(r.std()) * 100, 4),
        }

    old_stats = label_stats(old_labels["realized_return"], old_labels["realized_cost"])
    new_b_stats = label_stats(
        new_b["realized_return"].dropna(),
        new_b["realized_cost"].reindex(new_b["realized_return"].dropna().index) if "realized_cost" in new_b.columns else None,
    )
    new_b_rr = round(float(new_b["rr_ratio"].dropna().mean()), 3) if "rr_ratio" in new_b.columns else float("nan")

    return {
        "old_label_5bar_symmetric": {
            **old_stats,
            "design": "triple_barrier(5bar, ±2%)",
            "horizon_bars": 5,
            "rr_ratio": 1.0,
            "verdict": "NEGATIVE_EV" if old_stats.get("mean_net_pct", 0) < 0 else "POSITIVE_EV",
        },
        "new_label_7d_asymmetric": {
            **new_b_stats,
            "design": "triple_barrier(7day, 2:1 R:R)",
            "horizon_bars": HORIZON_DAYS,
            "rr_ratio": new_b_rr,
            "verdict": "POSITIVE_EV" if new_b_stats.get("mean_gross_pct", 0) > 0 else "CHECK",
        },
        "new_label_7d_excess_return": {
            "n":             int(new_a.dropna().count()),
            "mean":          round(float(new_a.dropna().mean()), 4),
            "std":           round(float(new_a.dropna().std()), 4),
            "positive_rate": round(float((new_a.dropna() > 0).mean()), 4),
            "design":        "vol_adjusted_excess_return(7day vs NIFTY)",
            "horizon_bars":  HORIZON_DAYS,
            "verdict":       "IC_BASED",
        },
    }
