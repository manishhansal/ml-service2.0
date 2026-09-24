"""
src.backtest.engine — cost-aware event-driven backtest engine (Phase K).

Simulates trading a single-instrument signal series with realistic frictions:
    - spread (half-spread paid on entry and exit)
    - slippage (proportional to volatility / configurable bps)
    - brokerage + statutory fees (round-trip bps)
    - execution latency (signal executes at the NEXT bar's open, never the
      signal-bar close — no "signal price = execution price" cheating)
    - position limits (max gross exposure)

Never assumes signal price == execution price. Entry fills at next-bar open
plus half-spread plus slippage; exits symmetric.

Produces a full profitability report (Phase 34): gross/net return, Sharpe,
Sortino, Calmar, max drawdown, profit factor, win rate, turnover, trade count,
average win/loss, VaR/CVaR, exposure.

Requirements: Phase K, Phase 32, Phase 33, Phase 34, 11_BACKTESTING_SPEC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252


@dataclass
class CostModel:
    """Configurable transaction-cost model (Phase 33)."""

    brokerage_bps: float = 3.0       # round-trip brokerage
    fees_bps: float = 2.0            # statutory taxes/fees round-trip
    half_spread_bps: float = 2.5     # half bid-ask spread paid each side
    slippage_bps: float = 2.0        # market-impact / slippage each side

    @property
    def entry_cost_bps(self) -> float:
        return self.half_spread_bps + self.slippage_bps

    @property
    def exit_cost_bps(self) -> float:
        return self.half_spread_bps + self.slippage_bps

    @property
    def round_trip_fees_bps(self) -> float:
        return self.brokerage_bps + self.fees_bps

    def total_round_trip_bps(self) -> float:
        return self.entry_cost_bps + self.exit_cost_bps + self.round_trip_fees_bps


@dataclass
class Trade:
    """A single completed round-trip trade."""

    entry_index: int
    exit_index: int
    direction: int           # +1 long, -1 short
    entry_price: float
    exit_price: float
    gross_return: float
    cost: float
    net_return: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class BacktestReport:
    """Full profitability report."""

    n_trades: int = 0
    gross_return: float = 0.0
    net_return: float = 0.0
    ann_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    turnover: float = 0.0
    exposure: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    cost_bps_round_trip: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("equity_curve", "trades")}
        d["n_trades"] = self.n_trades
        return d


class BacktestEngine:
    """
    Event-driven cost-aware backtest.

    Usage::

        engine = BacktestEngine(CostModel())
        report = engine.run(prices, signals)

    ``prices`` — DataFrame with at least an 'open' and 'close' column, indexed
                 by timestamp.
    ``signals`` — array-like of target positions in {-1, 0, +1} aligned to prices.
                  Signal at bar t is ACTED ON at bar t+1's open (latency=1 bar).
    """

    def __init__(self, cost_model: CostModel | None = None, max_gross_exposure: float = 1.0) -> None:
        self.cost = cost_model or CostModel()
        self.max_gross_exposure = max_gross_exposure

    def run(
        self,
        prices: pd.DataFrame,
        signals: np.ndarray | list[int],
    ) -> BacktestReport:
        if "open" not in prices.columns or "close" not in prices.columns:
            raise ValueError("prices must contain 'open' and 'close' columns.")

        opens = prices["open"].to_numpy(dtype=float)
        closes = prices["close"].to_numpy(dtype=float)
        sig = np.asarray(signals, dtype=float)
        sig = np.clip(sig, -self.max_gross_exposure, self.max_gross_exposure)
        n = len(opens)
        if len(sig) != n:
            raise ValueError("signals length must equal number of price bars.")

        entry_c = self.cost.entry_cost_bps / 10_000.0
        exit_c = self.cost.exit_cost_bps / 10_000.0
        fees = self.cost.round_trip_fees_bps / 10_000.0

        trades: list[Trade] = []
        position = 0.0
        entry_price = 0.0
        entry_idx = 0
        bar_returns: list[float] = []  # per-bar net portfolio return
        exposure_bars = 0

        # Latency: signal at t executes at t+1 open. Iterate to n-1.
        for t in range(n - 1):
            target = sig[t]              # decision made using data up to bar t
            exec_price = opens[t + 1]    # filled at next bar's open

            # Mark-to-market the current bar's contribution while holding.
            if position != 0.0 and t > 0:
                bar_ret = position * (opens[t + 1] - opens[t]) / opens[t]
                bar_returns.append(bar_ret)
                exposure_bars += 1
            else:
                bar_returns.append(0.0)

            if target != position:
                # Close existing position (if any).
                if position != 0.0:
                    gross = position * (exec_price - entry_price) / entry_price
                    cost = entry_c + exit_c + fees
                    net = gross - cost
                    trades.append(Trade(
                        entry_index=entry_idx, exit_index=t + 1,
                        direction=int(np.sign(position)),
                        entry_price=entry_price, exit_price=exec_price,
                        gross_return=gross, cost=cost, net_return=net,
                    ))
                # Open new position (if target != 0).
                if target != 0.0:
                    position = target
                    entry_price = exec_price
                    entry_idx = t + 1
                else:
                    position = 0.0

        # Close any residual position at the last close.
        if position != 0.0:
            exec_price = closes[-1]
            gross = position * (exec_price - entry_price) / entry_price
            cost = entry_c + exit_c + fees
            trades.append(Trade(
                entry_index=entry_idx, exit_index=n - 1,
                direction=int(np.sign(position)),
                entry_price=entry_price, exit_price=exec_price,
                gross_return=gross, cost=cost, net_return=gross - cost,
            ))

        return self._report(trades, bar_returns, n)

    # ── Reporting ──────────────────────────────────────────────────────────

    def _report(self, trades: list[Trade], bar_returns: list[float], n_bars: int) -> BacktestReport:
        report = BacktestReport(
            n_trades=len(trades),
            cost_bps_round_trip=self.cost.total_round_trip_bps(),
            trades=trades,
        )
        if not trades:
            report.equity_curve = [1.0]
            return report

        net_rets = np.array([t.net_return for t in trades])
        gross_rets = np.array([t.gross_return for t in trades])

        report.gross_return = float(np.sum(gross_rets))
        report.net_return = float(np.sum(net_rets))

        # Equity curve from per-bar net portfolio returns (approx, cost applied at trades).
        bar_arr = np.array(bar_returns) if bar_returns else np.array([0.0])
        # Subtract trade costs at exit bars.
        cost_series = np.zeros(max(n_bars, 1))
        for t in trades:
            if t.exit_index < len(cost_series):
                cost_series[t.exit_index] -= t.cost
        combined = bar_arr[: len(cost_series)] + cost_series[: len(bar_arr)] if len(bar_arr) else cost_series
        equity = np.cumprod(1.0 + combined)
        report.equity_curve = equity.tolist()

        # Sharpe / Sortino on per-bar net returns.
        mean = float(np.mean(combined))
        std = float(np.std(combined))
        downside = combined[combined < 0]
        dstd = float(np.std(downside)) if len(downside) else 0.0
        report.sharpe = round(mean / std * np.sqrt(TRADING_DAYS), 4) if std > 1e-12 else 0.0
        report.sortino = round(mean / dstd * np.sqrt(TRADING_DAYS), 4) if dstd > 1e-12 else 0.0
        report.max_drawdown = round(self._max_drawdown(equity), 6)
        report.ann_return = round(mean * TRADING_DAYS, 6)
        report.calmar = (
            round(report.ann_return / abs(report.max_drawdown), 4)
            if report.max_drawdown < 0 else 0.0
        )

        wins = net_rets[net_rets > 0]
        losses = net_rets[net_rets < 0]
        report.win_rate = round(float(len(wins) / len(net_rets)), 4)
        report.avg_win = round(float(np.mean(wins)), 6) if len(wins) else 0.0
        report.avg_loss = round(float(np.mean(losses)), 6) if len(losses) else 0.0
        gross_profit = float(np.sum(wins))
        gross_loss = abs(float(np.sum(losses)))
        report.profit_factor = round(gross_profit / gross_loss, 4) if gross_loss > 1e-12 else 0.0

        report.turnover = round(float(len(trades)) / max(n_bars, 1), 4)
        report.exposure = round(float(np.mean(np.abs(bar_arr) > 0)), 4) if len(bar_arr) else 0.0

        # VaR / CVaR at 95% on trade net returns.
        if len(net_rets) >= 5:
            report.var_95 = round(float(np.percentile(net_rets, 5)), 6)
            tail = net_rets[net_rets <= report.var_95]
            report.cvar_95 = round(float(np.mean(tail)), 6) if len(tail) else report.var_95

        logger.info(
            "backtest_complete",
            n_trades=report.n_trades,
            net_return=report.net_return,
            sharpe=report.sharpe,
            cost_bps=report.cost_bps_round_trip,
        )
        return report

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        if len(equity) == 0:
            return 0.0
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        return float(np.min(dd))


def cost_sensitivity_analysis(
    prices: pd.DataFrame,
    signals: np.ndarray,
    bps_levels: tuple[float, ...] = (5.0, 10.0, 20.0),
) -> dict[str, dict[str, float]]:
    """
    Run the backtest across multiple total round-trip cost levels (Phase 33).

    Splits each bps level evenly across the four cost components so the total
    round-trip equals the requested level. Returns {level: report_dict}.
    """
    results: dict[str, dict[str, float]] = {}
    for total_bps in bps_levels:
        per = total_bps / 4.0
        cost = CostModel(
            brokerage_bps=per, fees_bps=per, half_spread_bps=per / 2, slippage_bps=per / 2
        )
        engine = BacktestEngine(cost)
        report = engine.run(prices, signals)
        results[f"{total_bps:.0f}bps"] = report.to_dict()
    return results
