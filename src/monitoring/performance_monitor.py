"""
src.monitoring.performance_monitor — Trade outcome monitoring and model health.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TradeOutcome:
    model_name: str
    strategy: str = ""
    symbol: str = ""
    predicted_prob: float = 0.5
    actual_outcome: float = 0.0
    pnl_pct: float = 0.0
    confidence: float = 0.5
    action: str = "BUY"


@dataclass
class PerformanceSnapshot:
    model_name: str
    n_trades: int
    brier_score: float
    accuracy: float
    ece: float
    win_rate: float
    trading_expectancy: float
    sharpe: float
    max_drawdown: float
    degraded: bool


@dataclass
class StrategyDecayMetrics:
    model_name: str
    strategy: str
    rolling_sharpe: float
    is_decaying: bool


@dataclass
class PerformanceThresholds:
    brier_critical: float = 0.25
    accuracy_floor: float = 0.50
    win_rate_floor: float = 0.40
    expectancy_floor: float = -0.005


def _brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def _accuracy(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    preds = (probs >= threshold).astype(float)
    return float(np.mean(preds == labels))


def _trading_expectancy(
    pnls: np.ndarray,
) -> tuple[float, float, float, float]:
    if len(pnls) == 0:
        return 0.0, 0.0, 0.0, 0.0
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / len(pnls)
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 0.0
    exp = wr * avg_win - (1 - wr) * avg_loss
    return float(exp), float(wr), avg_win, avg_loss


def _sharpe_ratio(pnls: np.ndarray, periods: int = 252) -> float:
    if len(pnls) < 2:
        return 0.0
    mu = np.mean(pnls)
    sigma = np.std(pnls, ddof=1)
    if sigma < 1e-12:
        # constant returns: positive → +inf proxy, negative → -inf proxy
        return 999.0 if mu > 0 else (-999.0 if mu < 0 else 0.0)
    return float(mu / sigma * math.sqrt(periods))


def _max_drawdown(pnls: np.ndarray) -> float:
    if len(pnls) == 0:
        return 0.0
    equity = np.cumprod(1.0 + pnls)
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    return float(np.min(dd))


class PerformanceMonitor:
    def __init__(
        self,
        alert_system=None,
        model_registry=None,
        window_size: int = 100,
        min_samples: int = 20,
        thresholds: PerformanceThresholds | None = None,
    ) -> None:
        self._alerts = alert_system
        self._registry = model_registry
        self._window = window_size
        self._min = min_samples
        self._thresholds = thresholds or PerformanceThresholds()
        # model -> list[TradeOutcome]
        self._outcomes: dict[str, list[TradeOutcome]] = defaultdict(list)
        # model -> strategy -> list[float] pnl
        self._strategy_pnls: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def record_outcome(self, outcome: TradeOutcome) -> None:
        buf = self._outcomes[outcome.model_name]
        buf.append(outcome)
        if len(buf) > self._window:
            self._outcomes[outcome.model_name] = buf[-self._window:]
        self._strategy_pnls[outcome.model_name][outcome.strategy].append(outcome.pnl_pct)

    def record_outcomes_batch(self, outcomes: list[TradeOutcome]) -> PerformanceSnapshot | None:
        for o in outcomes:
            self.record_outcome(o)
        if not outcomes:
            return None
        model = outcomes[0].model_name
        return self._compute_snapshot(model)

    def _compute_snapshot(self, model_name: str) -> PerformanceSnapshot | None:
        buf = self._outcomes.get(model_name, [])
        if len(buf) < self._min:
            return None
        probs = np.array([o.predicted_prob for o in buf])
        labels = np.array([o.actual_outcome for o in buf])
        pnls = np.array([o.pnl_pct for o in buf])

        bs = _brier_score(probs, labels)
        acc = _accuracy(probs, labels)
        ece_val = _ece(probs, labels)
        exp, wr, _, _ = _trading_expectancy(pnls)
        sharpe = _sharpe_ratio(pnls)
        mdd = _max_drawdown(pnls)

        degraded = (
            bs > self._thresholds.brier_critical
            or acc < self._thresholds.accuracy_floor
            or wr < self._thresholds.win_rate_floor
        )

        snap = PerformanceSnapshot(
            model_name=model_name,
            n_trades=len(buf),
            brier_score=bs,
            accuracy=acc,
            ece=ece_val,
            win_rate=wr,
            trading_expectancy=exp,
            sharpe=sharpe,
            max_drawdown=mdd,
            degraded=degraded,
        )

        if degraded and self._registry is not None:
            self._registry.set_degraded(
                model_name,
                reasons=[f"brier={bs:.3f} acc={acc:.3f} wr={wr:.3f}"]
            )

        return snap

    def get_strategy_decay(
        self, model_name: str, strategy: str
    ) -> StrategyDecayMetrics | None:
        pnls = self._strategy_pnls.get(model_name, {}).get(strategy, [])
        if len(pnls) < 10:
            return None
        arr = np.array(pnls, dtype=float)
        sharpe = _sharpe_ratio(arr)
        return StrategyDecayMetrics(
            model_name=model_name,
            strategy=strategy,
            rolling_sharpe=sharpe,
            is_decaying=sharpe < 0,
        )
