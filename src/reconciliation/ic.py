"""
src.reconciliation.ic — Canonical IC computation family.

Mandate §23–§25: IC must be computed against continuous realized returns,
with clustered standard errors, and never against triple-barrier-clamped returns.

This module provides:
  1. CanonicalICResult         — all IC variants in one structure
  2. compute_timeseries_ic     — per-row Pearson/Spearman (what walk_forward.py does)
  3. compute_cross_sectional_ic— per-timestamp cross-sectional Rank IC (the honest one)
  4. compute_icir              — IC information ratio
  5. compute_overlapping_correction — corrects t-stats for overlapping h-day labels
  6. compute_clustered_se      — date- and symbol-clustered standard errors
  7. reconcile_ic_components   — produce the full reconciliation table

The key insight from the forensic audit:
  - walk_forward.py computes TIME-SERIES IC: corr(score_i_T, return_i_T) pooled
    across ALL (symbol, time) pairs.  This is a valid but DIFFERENT metric from
    the cross-sectional IC used to assess a cross-sectional portfolio.
  - The HONEST economic IC is the per-timestamp cross-sectional Rank IC of model
    scores across symbols vs their realized next-open forward returns at each T.
  - Additionally, for h=5 overlapping labels, the effective sample size is
    ~N/h, so t-stats must be deflated by sqrt(h) as a first-order correction.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)

TRADING_DAYS = 252


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ICWindowResult:
    """Cross-sectional IC for one walk-forward OOS window."""
    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_timestamps: int
    n_predictions: int
    pearson_ic: float           # outlier-sensitive; diagnostic only
    rank_ic: float              # PRIMARY: Spearman rank IC
    rank_ic_std: float          # std of per-timestamp rank IC values
    icir: float                 # rank_ic / rank_ic_std * sqrt(periods)
    positive_fraction: float    # fraction of timestamps with positive rank IC
    note: str = ""


@dataclass
class CanonicalICResult:
    """
    Complete IC evidence package.

    Distinguishes between:
      ts_ic_pearson      : time-series IC (what walk_forward.py reports)
      ts_ic_rank         : time-series Rank IC (spearman, pooled across rows)
      xs_ic_rank         : cross-sectional Rank IC per timestamp (primary)
      xs_ic_rank_h5      : cross-sectional Rank IC at h=5 (matches model horizon)
      honest_ic          : xs_ic_rank_h5 with overlapping correction
      barrier_artifact_fraction : fraction of returns that are exactly ±barrier
      effective_n        : N / h (overlapping correction)
      t_stat_raw         : t-stat from raw n (inflated by overlapping)
      t_stat_corrected   : t-stat corrected for overlap (divide by sqrt(h))
      date_clustered_se  : cluster-robust SE using date clusters
      symbol_clustered_se: cluster-robust SE using symbol clusters
    """
    # Sample info
    n_rows: int = 0
    n_symbols: int = 0
    n_dates: int = 0
    horizon_days: int = 5

    # Time-series IC (what the existing pipeline reports)
    ts_ic_pearson: float = 0.0       # Pearson IC of score vs realized_return (pooled rows)
    ts_ic_rank: float = 0.0          # Spearman IC of score vs realized_return (pooled rows)
    ts_ic_vs_continuous: float = 0.0 # Pearson IC vs TRUE continuous return (no barrier clip)
    ts_ic_vs_barrier: float = 0.0    # Pearson IC vs barrier-clamped return (what was reported)

    # Barrier artifact quantification
    barrier_artifact_fraction: float = 0.0   # fraction of rows at exactly ±barrier
    ic_inflation_factor: float = 0.0          # ts_ic_vs_barrier / ts_ic_vs_continuous

    # Cross-sectional IC (honest economic metric)
    xs_rank_ic_h1: float = 0.0       # 1-day cross-sectional rank IC
    xs_rank_ic_h5: float = 0.0       # 5-day cross-sectional rank IC (matches model)
    xs_rank_ic_std: float = 0.0      # std of per-timestamp xs rank IC values
    xs_icir: float = 0.0             # xs_rank_ic_h5 / xs_rank_ic_std * sqrt(252/5)
    xs_positive_fraction: float = 0.0 # fraction of timestamps with positive xs IC

    # Overlapping label correction
    effective_n: int = 0
    t_stat_raw: float = 0.0          # naive t-stat from n_rows
    t_stat_corrected: float = 0.0    # t-stat from n_rows/horizon (Newey-West style)
    overlap_correction_factor: float = 0.0  # sqrt(horizon)

    # Cluster-robust SE (date-clustered is the most conservative for panel data)
    date_clustered_se: float = 0.0
    date_clustered_tstat: float = 0.0
    symbol_clustered_se: float = 0.0

    # Per-window results
    windows: list[ICWindowResult] = field(default_factory=list)

    # Diagnosis
    diagnosis: str = ""
    first_divergence_point: str = ""   # where TS IC and XS IC diverge

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "windows"}
        d["windows"] = [w.__dict__ for w in self.windows]
        return d


# ── Core IC functions ──────────────────────────────────────────────────────────

def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r, _ = pearsonr(a, b)
    return float(r) if np.isfinite(r) else 0.0


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r, _ = spearmanr(a, b)
    return float(r) if np.isfinite(r) else 0.0


def compute_timeseries_ic(
    scores: np.ndarray,
    realized_returns: np.ndarray,
    barrier_clamped: np.ndarray | None = None,
    barrier_pct: float = 0.02,
) -> dict[str, float]:
    """
    Compute the time-series IC family.

    This is what walk_forward.py currently computes: a pooled correlation
    across ALL (symbol, time) rows.

    Returns:
      ts_ic_pearson        : Pearson corr(score, realized_return)
      ts_ic_rank           : Spearman corr(score, realized_return)
      ts_ic_vs_continuous  : same as ts_ic_pearson (realized_return passed as-is)
      ts_ic_vs_barrier     : Pearson corr(score, clip(realized, ±barrier))
      barrier_artifact_fraction : fraction of |realized_return| ≈ barrier
      ic_inflation_factor  : ts_ic_vs_barrier / max(ts_ic_pearson, 1e-9)
    """
    scores = np.asarray(scores, dtype=float)
    realized = np.asarray(realized_returns, dtype=float)

    mask = np.isfinite(scores) & np.isfinite(realized)
    s, r = scores[mask], realized[mask]

    pearson = _safe_pearson(s, r)
    spearman = _safe_spearman(s, r)

    # Barrier fraction
    barrier_frac = float(np.mean((np.abs(r) - barrier_pct).__abs__() < 1e-5))

    # IC vs barrier-clamped
    r_clipped = np.clip(r, -barrier_pct, barrier_pct)
    pearson_vs_barrier = _safe_pearson(s, r_clipped)

    inflation = (
        abs(pearson_vs_barrier) / max(abs(pearson), 1e-9)
        if abs(pearson) > 1e-9 else 0.0
    )

    return {
        "ts_ic_pearson": round(pearson, 6),
        "ts_ic_rank": round(spearman, 6),
        "ts_ic_vs_continuous": round(pearson, 6),   # realized IS continuous here
        "ts_ic_vs_barrier": round(pearson_vs_barrier, 6),
        "barrier_artifact_fraction": round(barrier_frac, 4),
        "ic_inflation_factor": round(inflation, 4),
    }


def compute_cross_sectional_ic(
    panel: pd.DataFrame,
    score_col: str,
    return_col: str,
    min_symbols: int = 5,
) -> dict[str, float]:
    """
    Compute per-timestamp cross-sectional Rank IC, then average.

    This is the HONEST economic IC: at each date T, rank N symbols by score,
    measure rank correlation vs their realized forward returns.

    Args:
        panel      : (ts, symbol)-indexed DataFrame with score and return columns
        score_col  : column name for model scores
        return_col : column name for CONTINUOUS realized forward returns
        min_symbols: minimum symbols at T for IC to be computed

    Returns dict with xs_rank_ic_mean, xs_rank_ic_std, xs_rank_ic_icir,
    xs_rank_ic_positive_fraction, xs_rank_ic_per_ts (list).
    """
    if panel.empty:
        return {"xs_rank_ic_mean": 0.0, "xs_rank_ic_std": 0.0,
                "xs_rank_ic_icir": 0.0, "xs_rank_ic_positive_fraction": 0.0,
                "n_timestamps": 0, "xs_rank_ic_per_ts": []}

    needed = {score_col, return_col}
    clean = panel.dropna(subset=list(needed))

    per_ts_ic: list[float] = []
    ts_index = clean.index.get_level_values("ts").unique()

    for ts in ts_index:
        grp = clean.loc[ts] if ts in clean.index.get_level_values("ts") else pd.DataFrame()
        if len(grp) < min_symbols:
            continue
        s = grp[score_col].to_numpy(float)
        r = grp[return_col].to_numpy(float)
        if np.std(s) < 1e-12 or np.std(r) < 1e-12:
            continue
        ic = _safe_spearman(s, r)
        if np.isfinite(ic):
            per_ts_ic.append(ic)

    if not per_ts_ic:
        return {"xs_rank_ic_mean": 0.0, "xs_rank_ic_std": 0.0,
                "xs_rank_ic_icir": 0.0, "xs_rank_ic_positive_fraction": 0.0,
                "n_timestamps": 0, "xs_rank_ic_per_ts": []}

    arr = np.array(per_ts_ic)
    mean_ic = float(np.mean(arr))
    std_ic = float(np.std(arr)) if len(arr) > 1 else 0.0
    icir = (mean_ic / std_ic * math.sqrt(TRADING_DAYS)) if std_ic > 1e-12 else 0.0
    pos_frac = float(np.mean(arr > 0))

    return {
        "xs_rank_ic_mean": round(mean_ic, 6),
        "xs_rank_ic_std": round(std_ic, 6),
        "xs_rank_ic_icir": round(icir, 4),
        "xs_rank_ic_positive_fraction": round(pos_frac, 4),
        "n_timestamps": len(per_ts_ic),
        "xs_rank_ic_per_ts": [round(x, 6) for x in per_ts_ic],
    }


def compute_overlapping_correction(
    ic: float,
    n_rows: int,
    horizon: int,
    ic_se_naive: float | None = None,
) -> dict[str, float]:
    """
    Correct IC t-statistics for overlapping h-day labels.

    For h-day forward returns on daily data, consecutive labels overlap on h-1
    bars.  A conservative first-order correction divides the naive t-stat by
    sqrt(h), which gives approximate Newey-West (lag h-1) standard errors.

    More precisely: effective n ≈ n_rows / h.

    Args:
        ic         : reported IC value
        n_rows     : raw number of observations
        horizon    : h (label look-forward in bars)
        ic_se_naive: naive SE = 1/sqrt(n_rows) if None

    Returns dict with effective_n, t_stat_raw, t_stat_corrected,
    overlap_correction_factor.
    """
    se_naive = ic_se_naive if ic_se_naive is not None else (1.0 / math.sqrt(n_rows))
    t_raw = ic / max(se_naive, 1e-12)

    n_effective = n_rows / horizon
    # Corrected SE uses the provided naive SE scaled by sqrt(horizon):
    # if naive SE reflects the raw n, corrected SE = naive_SE * sqrt(horizon)
    # so t_corrected = ic / (se_naive * sqrt(horizon)) = t_raw / sqrt(horizon)
    se_corrected = se_naive * math.sqrt(horizon)
    t_corrected = ic / max(se_corrected, 1e-12)
    correction = math.sqrt(horizon)

    return {
        "n_rows": n_rows,
        "horizon": horizon,
        "effective_n": int(n_effective),
        "overlap_correction_factor": correction,  # full precision — tests compare to math.sqrt
        "t_stat_raw": round(t_raw, 4),
        "t_stat_corrected": round(t_corrected, 4),
        "se_naive": round(se_naive, 6),
        "se_corrected": round(se_corrected, 6),
    }


def compute_clustered_se(
    panel: pd.DataFrame,
    score_col: str,
    return_col: str,
) -> dict[str, float]:
    """
    Date-clustered and symbol-clustered standard errors for time-series IC.

    Date-clustered SE: treats each date as one observation (conservative for
    cross-sectionally correlated returns).  Symbol-clustered SE: treats each
    symbol as one observation (conservative for serially correlated returns).

    Both are computed from the per-cluster mean IC.
    """
    required = {score_col, return_col, "ts", "symbol"}
    df = panel.copy()
    # Handle MultiIndex
    if hasattr(df.index, "names") and "ts" in df.index.names:
        df = df.reset_index()

    missing = required - set(df.columns)
    if missing:
        return {"error": f"Missing columns: {missing}"}

    df = df.dropna(subset=[score_col, return_col])

    # ── Date-clustered ─────────────────────────────────────────────────────
    date_ics = []
    for ts, grp in df.groupby("ts"):
        s = grp[score_col].to_numpy(float)
        r = grp[return_col].to_numpy(float)
        if len(s) < 3:
            continue
        ic = _safe_pearson(s, r)
        if np.isfinite(ic):
            date_ics.append(ic)

    date_ic_arr = np.array(date_ics)
    n_dates = len(date_ic_arr)
    date_mean_ic = float(np.mean(date_ic_arr)) if n_dates else 0.0
    date_se = float(np.std(date_ic_arr) / math.sqrt(n_dates)) if n_dates > 1 else 0.0
    date_tstat = date_mean_ic / max(date_se, 1e-12)

    # ── Symbol-clustered ───────────────────────────────────────────────────
    sym_ics = []
    for sym, grp in df.groupby("symbol"):
        s = grp[score_col].to_numpy(float)
        r = grp[return_col].to_numpy(float)
        if len(s) < 3:
            continue
        ic = _safe_pearson(s, r)
        if np.isfinite(ic):
            sym_ics.append(ic)

    sym_ic_arr = np.array(sym_ics)
    n_syms = len(sym_ic_arr)
    sym_mean_ic = float(np.mean(sym_ic_arr)) if n_syms else 0.0
    sym_se = float(np.std(sym_ic_arr) / math.sqrt(n_syms)) if n_syms > 1 else 0.0

    return {
        "ic_date_mean": round(date_mean_ic, 6),
        "date_clustered_se": round(date_se, 6),
        "date_clustered_tstat": round(date_tstat, 4),
        "n_date_clusters": n_dates,
        "ic_symbol_mean": round(sym_mean_ic, 6),
        "symbol_clustered_se": round(sym_se, 6),
        "n_symbol_clusters": n_syms,
        "note": (
            "Date-clustered SE is conservative for cross-sectionally correlated "
            "returns. Symbol-clustered SE is conservative for serial correlation."
        ),
    }
