"""
src.analytics.cross_sectional_backtest — real-OHLCV cross-sectional backtest,
IC-decay, persistence, turnover/holding/hysteresis (mandate §25-§35, §51, §82-§85).

The cross-sectional research (src/analytics/cross_sectional.py) reports the
statistical signal (rank IC) plus a CLOSE-TO-CLOSE portfolio spread. A
close-to-close spread is NOT valid economic evidence (§32, §33): at the daily
horizon the dominant cross-sectional effect is short-term reversal, most of
which is bid-ask bounce that vanishes once you must actually TRADE.

This module settles the tradeability question honestly:

  - Signals are formed at the CLOSE of day T (using only data <= T).
  - Positions are ENTERED at the OPEN of day T+1 (the next executable bar) and
    EXITED at the open of day T+1+holding. No close-to-close fills.
  - Real OHLCV open/close are used; no reconstructed price path (§32).
  - Per-trade costs (spread + slippage + brokerage + taxes) are charged (§30,§51).

Also provides:
  - ic_decay_curve: cross-sectional rank IC vs forward horizon (§82).
  - signal_persistence: autocorrelation of the cross-sectional ranking (§26).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Realistic Indian-equity cost model (mandate §30, §31, §51)
# ---------------------------------------------------------------------------


@dataclass
class CostModel:
    """Round-trip cost components in basis points of notional (per side unless
    noted). Defaults reflect a realistic liquid NSE cash-equity delivery/intraday
    blend; run sensitivity around these (§31)."""

    brokerage_bps: float = 3.0          # per side (discount broker ~ flat, approximated)
    exchange_bps: float = 0.325 / 100 * 100  # NSE txn charge ~0.00325% -> ~0.325 bps
    stt_bps: float = 2.5                # securities transaction tax (sell side, delivery 0.1%/... approx blended)
    gst_bps: float = 0.5                # GST on (brokerage+exchange)
    stamp_bps: float = 1.5             # stamp duty (buy side)
    slippage_bps: float = 5.0           # execution slippage per side
    half_spread_bps: float = 3.0        # half bid-ask spread crossed per side

    def per_side_bps(self) -> float:
        return (
            self.brokerage_bps + self.exchange_bps + self.gst_bps
            + self.slippage_bps + self.half_spread_bps
        )

    def round_trip_bps(self) -> float:
        # STT + stamp are asymmetric; approximate a full round trip.
        return 2.0 * self.per_side_bps() + self.stt_bps + self.stamp_bps


# ---------------------------------------------------------------------------
# IC decay curve (mandate §82)
# ---------------------------------------------------------------------------


def ic_decay_curve(
    panel: pd.DataFrame,
    score_col: str,
    horizons: list[int],
    close_col: str = "_close",
) -> dict[int, dict[str, float]]:
    """Cross-sectional rank IC of *score_col* vs forward return at each horizon.

    For each horizon h, computes the forward return (close[T+h]/close[T]-1) per
    symbol, then the per-timestamp Spearman rank IC of the score against that
    forward return, averaged over timestamps. Determines whether the signal is
    very short-lived (1-bar reversal) or persists.
    """
    out: dict[int, dict[str, float]] = {}
    close = panel[close_col]
    for h in horizons:
        fwd = close.groupby(level="symbol").transform(
            lambda s: s.shift(-h) / s - 1.0
        )
        tmp = pd.DataFrame({"score": panel[score_col], "fwd": fwd}).dropna()
        ics = []
        for _ts, g in tmp.groupby(level="ts"):
            if len(g) < 10 or g["score"].std() < 1e-12 or g["fwd"].std() < 1e-12:
                continue
            r = spearmanr(g["score"], g["fwd"]).correlation
            if np.isfinite(r):
                ics.append(r)
        out[h] = {
            "mean_rank_ic": round(float(np.mean(ics)), 6) if ics else 0.0,
            "n_timestamps": len(ics),
        }
    return out


# ---------------------------------------------------------------------------
# Signal persistence / ranking autocorrelation (mandate §26)
# ---------------------------------------------------------------------------


def signal_persistence(
    panel: pd.DataFrame, score_col: str, lags: list[int]
) -> dict[int, float]:
    """Mean cross-sectional rank autocorrelation of the score at each lag.

    Low persistence (autocorr decays fast) means frequent rebalancing is forced
    and turnover will be high; high persistence supports longer holding."""
    scores = panel[score_col]
    # rank within each timestamp
    ranks = scores.groupby(level="ts").rank(pct=True)
    wide = ranks.unstack("symbol").sort_index()
    out: dict[int, float] = {}
    for lag in lags:
        cors = []
        shifted = wide.shift(lag)
        for i in range(lag, len(wide)):
            a = wide.iloc[i]
            b = shifted.iloc[i]
            m = a.notna() & b.notna()
            if m.sum() < 10:
                continue
            r = spearmanr(a[m], b[m]).correlation
            if np.isfinite(r):
                cors.append(r)
        out[lag] = round(float(np.mean(cors)), 6) if cors else 0.0
    return out


# ---------------------------------------------------------------------------
# Real-OHLCV next-open-execution backtest (mandate §32, §33, §51)
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    execution: str
    holding_bars: int
    rebalance_every: int
    decile: float
    enter_threshold_pct: float
    exit_threshold_pct: float
    cost_model_round_trip_bps: float
    n_rebalances: int
    n_trades: int
    gross_return_annual: float
    net_return_annual: float
    gross_sharpe: float
    net_sharpe: float
    mean_daily_turnover: float
    avg_holding_bars: float
    max_drawdown: float
    hit_rate: float
    long_only: bool
    pnl_provenance: str = "REAL_HISTORICAL_OHLCV_NEXT_OPEN"
    is_economic_evidence: bool = True
    per_rebalance: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d


def backtest_next_open(
    panel: pd.DataFrame,
    oos_scores: pd.Series,
    *,
    open_col: str = "_open",
    close_col: str = "_close",
    holding_bars: int = 1,
    rebalance_every: int = 1,
    decile: float = 0.1,
    long_only: bool = False,
    cost_model: CostModel | None = None,
    enter_threshold_pct: float = 0.9,
    exit_threshold_pct: float = 0.8,
    use_hysteresis: bool = False,
    min_symbols: int = 20,
) -> BacktestResult:
    """Backtest a cross-sectional score with NEXT-OPEN execution on real OHLCV.

    Trade lifecycle (no close-to-close fill):
        signal formed at close[T]  ->  enter at open[T+1]  ->  exit at
        open[T+1+holding_bars]. Return of a long leg = open[T+1+h]/open[T+1]-1.

    Args:
        panel: (ts, symbol) panel containing open_col/close_col.
        oos_scores: (ts, symbol)-indexed Series of OOS model scores (only OOS
            timestamps; in-sample rows must be excluded by the caller).
        holding_bars: bars held after entry.
        rebalance_every: rebalance cadence in bars (§29).
        decile: top/bottom fraction selected.
        long_only: if True, long the top bucket only (no shorts).
        cost_model: realistic round-trip cost; charged on entry+exit turnover.
        use_hysteresis / enter/exit thresholds: reduce churn (§27).
        min_symbols: skip thin cross-sections.

    Returns:
        BacktestResult with REAL_HISTORICAL_OHLCV_NEXT_OPEN provenance.
    """
    cost = cost_model or CostModel()
    rt_cost = cost.round_trip_bps() / 10_000.0

    # Build wide open/close matrices aligned on the union timeline.
    open_w = panel[open_col].unstack("symbol").sort_index()
    close_w = panel[close_col].unstack("symbol").sort_index()
    score_w = oos_scores.unstack("symbol").reindex(index=open_w.index).sort_index()

    timeline = list(open_w.index)
    n = len(timeline)

    # Precompute per-timestamp rank of score (percentile).
    rank_w = score_w.rank(axis=1, pct=True)

    prev_long: set = set()
    prev_short: set = set()
    period_returns: list[float] = []
    turnovers: list[float] = []
    n_trades = 0
    holdings: list[int] = []
    per_rebalance: list[dict] = []

    t = 0
    # We form signal at index i (close[i]); enter at open[i+1]; exit at
    # open[i+1+holding]. Advance by rebalance_every.
    while t < n - (holding_bars + 1):
        entry_i = t + 1
        exit_i = t + 1 + holding_bars
        if exit_i >= n:
            break

        ranks = rank_w.iloc[t].dropna()
        if len(ranks) < min_symbols:
            t += rebalance_every
            continue

        k = max(1, int(round(len(ranks) * decile)))
        ranked = ranks.sort_values()
        bottom = set(ranked.index[:k])
        top = set(ranked.index[-k:])

        if use_hysteresis:
            # Keep previously-held names while they remain above the exit band.
            keep_long = {s for s in prev_long if ranks.get(s, 0.0) >= exit_threshold_pct}
            new_long = {s for s in top if ranks.get(s, 0.0) >= enter_threshold_pct}
            top = keep_long | new_long
            keep_short = {s for s in prev_short if ranks.get(s, 1.0) <= (1 - exit_threshold_pct)}
            new_short = {s for s in bottom if ranks.get(s, 1.0) <= (1 - enter_threshold_pct)}
            bottom = keep_short | new_short
            if not top:
                top = set(ranked.index[-k:])

        # Realised leg returns via NEXT-OPEN to NEXT-OPEN(+holding).
        o_entry = open_w.iloc[entry_i]
        o_exit = open_w.iloc[exit_i]

        def leg_ret(names: set) -> float:
            if not names:
                return 0.0
            r = []
            for s in names:
                pe, px = o_entry.get(s), o_exit.get(s)
                if pd.notna(pe) and pd.notna(px) and pe > 0:
                    r.append(px / pe - 1.0)
            return float(np.mean(r)) if r else 0.0

        long_ret = leg_ret(top)
        if long_only:
            gross = long_ret
            turn = _turnover(prev_long, top)
            prev_long = top
            traded = len(top.symmetric_difference(prev_long)) + len(top)
        else:
            short_ret = leg_ret(bottom)
            gross = long_ret - short_ret
            turn = 0.5 * (_turnover(prev_long, top) + _turnover(prev_short, bottom))
            prev_long, prev_short = top, bottom

        # Costs: charge round-trip cost proportional to turnover on entry+exit.
        leg_cost = rt_cost * turn
        net = gross - leg_cost

        period_returns.append(net)
        turnovers.append(turn)
        holdings.append(holding_bars)
        n_trades += len(top) + (0 if long_only else len(bottom))
        per_rebalance.append({
            "signal_ts": str(timeline[t]),
            "entry_ts": str(timeline[entry_i]),
            "exit_ts": str(timeline[exit_i]),
            "long_ret": round(long_ret, 6),
            "gross": round(gross, 6),
            "net": round(net, 6),
            "turnover": round(turn, 4),
            "n_long": len(top),
            "n_short": 0 if long_only else len(bottom),
        })
        t += rebalance_every

    arr = np.array(period_returns, dtype=float)
    periods_per_year = TRADING_DAYS / max(1, rebalance_every)
    gross_series = np.array([p["gross"] for p in per_rebalance], dtype=float)

    def _sharpe(x: np.ndarray) -> float:
        if len(x) < 2 or np.std(x) < 1e-12:
            return 0.0
        return round(float(np.mean(x) / np.std(x)) * math.sqrt(periods_per_year), 4)

    return BacktestResult(
        execution="NEXT_OPEN",
        holding_bars=holding_bars,
        rebalance_every=rebalance_every,
        decile=decile,
        enter_threshold_pct=enter_threshold_pct if use_hysteresis else 0.0,
        exit_threshold_pct=exit_threshold_pct if use_hysteresis else 0.0,
        cost_model_round_trip_bps=round(cost.round_trip_bps(), 3),
        n_rebalances=len(period_returns),
        n_trades=n_trades,
        gross_return_annual=round(float(np.mean(gross_series)) * periods_per_year, 6) if len(gross_series) else 0.0,
        net_return_annual=round(float(np.mean(arr)) * periods_per_year, 6) if len(arr) else 0.0,
        gross_sharpe=_sharpe(gross_series),
        net_sharpe=_sharpe(arr),
        mean_daily_turnover=round(float(np.mean(turnovers)), 4) if turnovers else 0.0,
        avg_holding_bars=round(float(np.mean(holdings)), 2) if holdings else 0.0,
        max_drawdown=_max_drawdown(arr),
        hit_rate=round(float(np.mean(arr > 0)), 4) if len(arr) else 0.0,
        long_only=long_only,
        per_rebalance=per_rebalance,
    )


def _turnover(prev: set, cur: set) -> float:
    if not cur:
        return 0.0
    if not prev:
        return 1.0
    return float(min(1.0, len(cur.symmetric_difference(prev)) / (2.0 * len(cur))))


def _max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    return round(float(np.min(dd)), 6)
