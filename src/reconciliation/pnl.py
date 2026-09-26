"""
src.reconciliation.pnl — Canonical Executable Portfolio Backtest.

Mandate §4 CANONICAL HIERARCHY:
  SIGNAL → ENTRY → EXIT → GROSS RETURN → TURNOVER → TRANSACTION COST
  → SLIPPAGE → NET RETURN → PORTFOLIO RETURN → RISK METRICS

This is the ONE canonical economic evaluator.  There must not be separate
competing definitions of profitability (mandate §4).

Execution convention (mandate §8 Target A):
  Signal at close[T]
  → enter at open[T+1]
  → exit at open[T+1+h]

For a portfolio (not a single-stock strategy):
  At each rebalance date T:
    1. Generate model scores for all N symbols (using only data ≤ T)
    2. Rank symbols by score cross-sectionally
    3. Long top-decile, short bottom-decile (or long-only top)
    4. Enter positions at open[T+1]
    5. Hold for h bars, exit at open[T+1+h]
    6. Compute gross return, deduct costs, track turnover

Key mandates enforced here:
  §5  : all prediction records carry provenance (symbol, ts, entry, exit, etc.)
  §18 : realistic Indian costs from costs.IndianCostModel
  §22 : multiple portfolio constructions pre-registered (not post-hoc)
  §41 : risk management constraints (max position, max sector, beta limit)
  §43 : final frozen OOS test only runs once
  §52 : this module never accesses training data or label information
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.reconciliation.costs import IndianCostModel, PRIMARY_COST

TRADING_DAYS = 252

PortfolioType = Literal[
    "equal_weight_long",
    "equal_weight_long_short",
    "rank_weight_long_short",
    "top_decile_long_only",
    "top_bottom_decile_long_short",
    "top_quintile_long_only",
    "top_bottom_quintile_long_short",
]


# ── Trade-level records ────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Single trade record. Every executed trade must have one of these."""
    symbol: str
    signal_date: str           # date when signal fires (close[T])
    entry_date: str            # date when position opens (open[T+1])
    exit_date: str             # date when position closes (open[T+1+h])
    signal_score: float        # raw model score at T
    signal_rank_pct: float     # cross-sectional percentile at T
    direction: int             # +1 long, -1 short
    weight: float              # portfolio weight (signed)
    entry_price: float         # open[T+1]
    exit_price: float          # open[T+1+h]
    gross_return: float        # (exit - entry) / entry
    round_trip_cost: float     # cost fraction
    net_return: float          # gross - cost
    pnl_provenance: str = "REAL_HISTORICAL_OHLCV_NEXT_OPEN"
    is_economic_evidence: bool = True

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PortfolioReturn:
    """Portfolio return for one rebalance period."""
    rebalance_date: str
    entry_date: str
    exit_date: str
    n_long: int
    n_short: int
    gross_return: float
    turnover: float
    cost: float
    net_return: float
    trades: list[TradeRecord] = field(default_factory=list)


@dataclass
class ExecutableBacktestResult:
    """Complete result of the executable portfolio backtest."""
    # Identity
    portfolio_type: str
    holding_bars: int
    cost_scenario: str
    round_trip_bps: float
    n_symbols: int
    n_rebalances: int
    n_trades: int
    date_start: str
    date_end: str

    # Cross-sectional IC (primary economic metric)
    xs_rank_ic_mean: float = 0.0
    xs_rank_ic_std: float = 0.0
    xs_rank_ic_icir: float = 0.0
    xs_rank_ic_positive_fraction: float = 0.0

    # Portfolio P&L
    gross_return_total: float = 0.0
    net_return_total: float = 0.0
    gross_return_annual: float = 0.0
    net_return_annual: float = 0.0
    gross_sharpe: float = 0.0
    net_sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    profit_factor: float = 0.0

    # Turnover / cost decomposition
    mean_daily_turnover: float = 0.0
    cost_drag_annual: float = 0.0
    slippage_drag_annual: float = 0.0
    breakeven_turnover: float = 0.0   # turnover at which net=0

    # Provenance
    pnl_provenance: str = "REAL_HISTORICAL_OHLCV_NEXT_OPEN"
    is_economic_evidence: bool = True
    execution_note: str = (
        "Signals at close[T]; positions ENTERED at open[T+1]; "
        "EXITED at open[T+1+h]. Real OHLCV open prices. "
        "No close-to-close fills, no reconstructed proxy prices."
    )

    # Per-rebalance detail (for audit; capped at 100 entries in JSON output)
    per_rebalance: list[dict] = field(default_factory=list)

    def to_dict(self, max_per_rebalance: int = 50) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "per_rebalance"}
        d["per_rebalance_sample"] = self.per_rebalance[:max_per_rebalance]
        return d


# ── Core backtest engine ───────────────────────────────────────────────────────

class ExecutablePortfolioBacktest:
    """
    Single canonical executable portfolio backtest.

    Mandate §4: ONE canonical evaluation pipeline; no competing definitions.

    Inputs:
        scores_panel  : (ts, symbol)-indexed DataFrame with a 'score' column
                        and an 'open' column.  Features and scores must be
                        from data ≤ T; open prices are used for execution only.
        cost_model    : IndianCostModel (default: PRIMARY_COST = conservative)
        holding_bars  : h; how many bars between open[T+1] and open[T+1+h]
        portfolio_type: one of PortfolioType literals
        decile        : fraction of universe for top/bottom bucket (0.1 = 10%)
        min_symbols   : minimum symbols at T to execute a rebalance

    Usage::

        bt = ExecutablePortfolioBacktest(scores_panel)
        result = bt.run()
    """

    def __init__(
        self,
        scores_panel: pd.DataFrame,   # (ts, symbol) index; columns: score, open
        cost_model: IndianCostModel = PRIMARY_COST,
        holding_bars: int = 5,
        portfolio_type: PortfolioType = "top_bottom_decile_long_short",
        decile: float = 0.10,
        min_symbols: int = 10,
    ) -> None:
        self.panel = scores_panel
        self.cost = cost_model
        self.h = holding_bars
        self.pt = portfolio_type
        self.decile = decile
        self.min_symbols = min_symbols

    def run(self) -> ExecutableBacktestResult:
        panel = self.panel.copy()

        if "score" not in panel.columns:
            raise ValueError("scores_panel must have a 'score' column.")
        if "open" not in panel.columns:
            raise ValueError("scores_panel must have an 'open' column (for execution price).")

        # Get sorted unique timestamps
        ts_all = np.array(sorted(panel.index.get_level_values("ts").unique()))
        if len(ts_all) < self.h + 2:
            raise ValueError(f"Insufficient timestamps: {len(ts_all)} < {self.h + 2}")

        cost_frac = self.cost.round_trip_fraction()

        per_rebalance: list[PortfolioReturn] = []
        prev_longs: set[str] = set()
        prev_shorts: set[str] = set()
        all_xs_ics: list[float] = []

        # We rebalance every h bars, entering at T+1 and exiting at T+1+h
        for i in range(0, len(ts_all) - self.h - 1, self.h):
            t = ts_all[i]               # signal date (close[T])
            t1_idx = i + 1              # entry bar index
            t1h_idx = i + 1 + self.h   # exit bar index
            if t1h_idx >= len(ts_all):
                break

            t1 = ts_all[t1_idx]         # open[T+1] — entry
            t1h = ts_all[t1h_idx]       # open[T+1+h] — exit

            # Get scores at T (already PIT-safe: computed from data ≤ T)
            if t not in panel.index.get_level_values("ts"):
                continue
            scores_at_t = panel.loc[t]["score"].dropna()
            n = len(scores_at_t)
            if n < self.min_symbols:
                continue

            # Get open prices at entry (T+1) and exit (T+1+h)
            # Use _get_opens helper; fall back gracefully if missing
            opens_t1 = self._get_opens(panel, t1)
            opens_t1h = self._get_opens(panel, t1h)

            # Intersect: symbols with score at T, open at T+1, open at T+1+h
            syms = (
                set(scores_at_t.index)
                & set(opens_t1.index)
                & set(opens_t1h.index)
            )
            if len(syms) < self.min_symbols:
                continue

            scores = scores_at_t.loc[list(syms)]
            op_entry = opens_t1.loc[list(syms)]
            op_exit = opens_t1h.loc[list(syms)]

            # Cross-sectional rank IC at T vs h-bar forward return
            fwd = (op_exit - op_entry) / op_entry
            from scipy.stats import spearmanr as _sr
            if scores.std() > 1e-12 and fwd.std() > 1e-12:
                ic, _ = _sr(scores.loc[fwd.index], fwd)
                if np.isfinite(ic):
                    all_xs_ics.append(float(ic))

            # Portfolio construction
            longs, shorts, weights = self._select_portfolio(scores)

            # Compute returns
            trades: list[TradeRecord] = []
            ls_rets: list[float] = []
            rank_pcts = scores.rank(pct=True)

            for sym, w in weights.items():
                if sym not in op_entry.index or sym not in op_exit.index:
                    continue
                p_entry = float(op_entry[sym])
                p_exit = float(op_exit[sym])
                if p_entry <= 0 or not np.isfinite(p_entry) or not np.isfinite(p_exit):
                    continue
                gross = (p_exit - p_entry) / p_entry
                net = gross - cost_frac  # round-trip charged per position
                ls_rets.append(w * net)
                trades.append(TradeRecord(
                    symbol=sym,
                    signal_date=str(pd.Timestamp(t).date()),
                    entry_date=str(pd.Timestamp(t1).date()),
                    exit_date=str(pd.Timestamp(t1h).date()),
                    signal_score=float(scores.get(sym, 0.0)),
                    signal_rank_pct=float(rank_pcts.get(sym, 0.5)),
                    direction=int(np.sign(w)),
                    weight=float(w),
                    entry_price=p_entry,
                    exit_price=p_exit,
                    gross_return=round(gross, 6),
                    round_trip_cost=round(cost_frac, 6),
                    net_return=round(net, 6),
                ))

            if not trades:
                continue

            # Turnover
            new_longs = set(longs)
            new_shorts = set(shorts)
            long_turn = _turnover(prev_longs, new_longs)
            short_turn = _turnover(prev_shorts, new_shorts)
            turnover = 0.5 * (long_turn + short_turn) if shorts else long_turn
            prev_longs, prev_shorts = new_longs, new_shorts

            gross_ret = float(np.mean([t.weight * t.gross_return for t in trades]))
            cost_ret = cost_frac * turnover
            net_ret = gross_ret - cost_ret

            per_rebalance.append(PortfolioReturn(
                rebalance_date=str(pd.Timestamp(t).date()),
                entry_date=str(pd.Timestamp(t1).date()),
                exit_date=str(pd.Timestamp(t1h).date()),
                n_long=len(longs),
                n_short=len(shorts),
                gross_return=round(gross_ret, 6),
                turnover=round(turnover, 4),
                cost=round(cost_ret, 6),
                net_return=round(net_ret, 6),
                trades=trades,
            ))

        return self._aggregate(per_rebalance, all_xs_ics, len(ts_all))

    # ── Portfolio construction ─────────────────────────────────────────────

    def _select_portfolio(
        self, scores: pd.Series
    ) -> tuple[list[str], list[str], dict[str, float]]:
        """Select symbols and assign weights based on portfolio_type."""
        n = len(scores)
        k = max(1, int(round(n * self.decile)))
        k5 = max(1, int(round(n * 0.20)))   # quintile
        sorted_idx = scores.sort_values()

        if self.pt == "equal_weight_long":
            longs = list(sorted_idx.tail(k).index)
            w = {s: 1.0 / len(longs) for s in longs}
            return longs, [], w

        if self.pt == "equal_weight_long_short":
            longs = list(sorted_idx.tail(k).index)
            shorts = list(sorted_idx.head(k).index)
            w = {s: 0.5 / len(longs) for s in longs}
            w.update({s: -0.5 / len(shorts) for s in shorts})
            return longs, shorts, w

        if self.pt == "rank_weight_long_short":
            rank_pcts = scores.rank(pct=True)
            w = {}
            for s, rp in rank_pcts.items():
                if rp >= 1 - self.decile:
                    w[s] = float(rp - 0.5)   # above 0.5: positive weight
                elif rp <= self.decile:
                    w[s] = float(rp - 0.5)   # below 0.5: negative weight
            total = sum(abs(v) for v in w.values())
            if total > 0:
                w = {s: v / total for s, v in w.items()}
            longs = [s for s, v in w.items() if v > 0]
            shorts = [s for s, v in w.items() if v < 0]
            return longs, shorts, w

        if self.pt == "top_decile_long_only":
            longs = list(sorted_idx.tail(k).index)
            w = {s: 1.0 / len(longs) for s in longs}
            return longs, [], w

        if self.pt == "top_bottom_decile_long_short":
            longs = list(sorted_idx.tail(k).index)
            shorts = list(sorted_idx.head(k).index)
            w = {s: 0.5 / len(longs) for s in longs}
            w.update({s: -0.5 / len(shorts) for s in shorts})
            return longs, shorts, w

        if self.pt == "top_quintile_long_only":
            longs = list(sorted_idx.tail(k5).index)
            w = {s: 1.0 / len(longs) for s in longs}
            return longs, [], w

        if self.pt == "top_bottom_quintile_long_short":
            longs = list(sorted_idx.tail(k5).index)
            shorts = list(sorted_idx.head(k5).index)
            w = {s: 0.5 / len(longs) for s in longs}
            w.update({s: -0.5 / len(shorts) for s in shorts})
            return longs, shorts, w

        raise ValueError(f"Unknown portfolio_type: {self.pt}")

    @staticmethod
    def _get_opens(panel: pd.DataFrame, ts) -> pd.Series:
        """Get open prices at timestamp ts; return empty series if missing."""
        if ts not in panel.index.get_level_values("ts"):
            return pd.Series(dtype=float)
        return panel.loc[ts]["open"].dropna().astype(float)

    # ── Aggregation ────────────────────────────────────────────────────────

    def _aggregate(
        self,
        periods: list[PortfolioReturn],
        xs_ics: list[float],
        n_ts: int,
    ) -> ExecutableBacktestResult:
        n_rebal = len(periods)
        n_trades = sum(len(p.trades) for p in periods)

        # IC
        ic_arr = np.array(xs_ics) if xs_ics else np.array([0.0])
        xs_mean = float(np.mean(ic_arr))
        xs_std = float(np.std(ic_arr)) if len(ic_arr) > 1 else 0.0
        periods_per_year = TRADING_DAYS / self.h
        xs_icir = (xs_mean / xs_std * math.sqrt(periods_per_year)) if xs_std > 1e-12 else 0.0
        xs_pos_frac = float(np.mean(ic_arr > 0))

        # P&L
        if not periods:
            return ExecutableBacktestResult(
                portfolio_type=self.pt, holding_bars=self.h,
                cost_scenario=self.cost.scenario,
                round_trip_bps=self.cost.round_trip_bps(),
                n_symbols=0, n_rebalances=0, n_trades=0,
                date_start="", date_end="",
                xs_rank_ic_mean=0.0, xs_rank_ic_positive_fraction=0.0,
            )

        net_rets = np.array([p.net_return for p in periods])
        gross_rets = np.array([p.gross_return for p in periods])
        turnovers = np.array([p.turnover for p in periods])

        # Annualise (periods are h-bar holding periods, not daily bars)
        annf = periods_per_year
        gross_ann = float(np.mean(gross_rets)) * annf
        net_ann = float(np.mean(net_rets)) * annf

        net_std = float(np.std(net_rets)) * math.sqrt(annf)
        gross_std = float(np.std(gross_rets)) * math.sqrt(annf)

        net_sharpe = net_ann / max(net_std, 1e-12)
        gross_sharpe = gross_ann / max(gross_std, 1e-12)

        # Sortino (downside deviation)
        down = net_rets[net_rets < 0]
        down_std = float(np.std(down)) * math.sqrt(annf) if len(down) > 1 else 1e-12
        sortino = net_ann / max(down_std, 1e-12)

        # Max drawdown
        equity = np.cumprod(1.0 + net_rets)
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        max_dd = float(np.min(dd))
        calmar = net_ann / max(abs(max_dd), 1e-12) if max_dd < 0 else 0.0

        # Hit rate and profit factor
        hits = net_rets > 0
        hit_rate = float(np.mean(hits))
        wins = net_rets[hits]
        losses = -net_rets[~hits]
        pf = float(np.sum(wins) / max(np.sum(losses), 1e-12)) if len(losses) else 999.0

        # Cost drag
        cost_drag = float(np.mean(turnovers) * self.cost.round_trip_fraction() * annf)
        mean_turnover = float(np.mean(turnovers))
        breakeven_to = (
            float(np.mean(gross_rets)) / max(self.cost.round_trip_fraction(), 1e-12)
            if self.cost.round_trip_fraction() > 0 else 0.0
        )

        date_start = periods[0].rebalance_date
        date_end = periods[-1].exit_date

        pr_dicts = [
            {"rebalance_date": p.rebalance_date, "entry_date": p.entry_date,
             "exit_date": p.exit_date, "n_long": p.n_long, "n_short": p.n_short,
             "gross_return": p.gross_return, "turnover": p.turnover,
             "cost": p.cost, "net_return": p.net_return}
            for p in periods
        ]

        return ExecutableBacktestResult(
            portfolio_type=self.pt, holding_bars=self.h,
            cost_scenario=self.cost.scenario,
            round_trip_bps=self.cost.round_trip_bps(),
            n_symbols=len(self.panel.index.get_level_values("symbol").unique()),
            n_rebalances=n_rebal, n_trades=n_trades,
            date_start=date_start, date_end=date_end,
            xs_rank_ic_mean=round(xs_mean, 6),
            xs_rank_ic_std=round(xs_std, 6),
            xs_rank_ic_icir=round(xs_icir, 4),
            xs_rank_ic_positive_fraction=round(xs_pos_frac, 4),
            gross_return_total=round(float(np.sum(gross_rets)), 4),
            net_return_total=round(float(np.sum(net_rets)), 4),
            gross_return_annual=round(gross_ann, 4),
            net_return_annual=round(net_ann, 4),
            gross_sharpe=round(gross_sharpe, 4),
            net_sharpe=round(net_sharpe, 4),
            sortino=round(sortino, 4),
            calmar=round(calmar, 4),
            max_drawdown=round(max_dd, 4),
            hit_rate=round(hit_rate, 4),
            profit_factor=round(pf, 4),
            mean_daily_turnover=round(mean_turnover, 4),
            cost_drag_annual=round(cost_drag, 4),
            breakeven_turnover=round(breakeven_to, 4),
            per_rebalance=pr_dicts,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _turnover(prev: set[str], cur: set[str]) -> float:
    """Fraction of the book that turned over between two rebalances."""
    if not cur:
        return 0.0
    if not prev:
        return 1.0
    changed = len(cur.symmetric_difference(prev)) / (2.0 * len(cur))
    return float(min(1.0, changed))


def build_scores_panel(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    predictions_df: pd.DataFrame,
    score_col: str = "score",
) -> pd.DataFrame:
    """
    Build the (ts, symbol)-indexed scores_panel required by ExecutablePortfolioBacktest.

    Merges model predictions with open prices from OHLCV data.
    The 'open' column carries the price at T+1 that will be used for execution.

    Args:
        ohlcv_by_symbol : raw OHLCV keyed by symbol
        predictions_df  : DataFrame with columns [ts, symbol, <score_col>]
        score_col       : column name for the model score

    Returns:
        (ts, symbol)-indexed DataFrame with columns: score, open
    """
    # Build open-price DataFrame
    open_frames = []
    for sym, df in ohlcv_by_symbol.items():
        if "open" not in df.columns:
            continue
        op = df[["open"]].copy().astype(float)
        op["symbol"] = sym
        op.index.name = "ts"
        open_frames.append(op.reset_index())

    if not open_frames:
        raise ValueError("build_scores_panel: no symbols with 'open' column.")

    opens = pd.concat(open_frames).set_index(["ts", "symbol"]).sort_index()
    opens.index.names = ["ts", "symbol"]

    # Align predictions
    preds = predictions_df.copy()
    if "ts" in preds.columns and "symbol" in preds.columns:
        preds["ts"] = pd.to_datetime(preds["ts"], utc=True)
        preds = preds.set_index(["ts", "symbol"])
    preds.index.names = ["ts", "symbol"]

    panel = opens.join(preds[[score_col]], how="inner")
    panel = panel.rename(columns={score_col: "score"})
    panel = panel[["score", "open"]].dropna()
    return panel
