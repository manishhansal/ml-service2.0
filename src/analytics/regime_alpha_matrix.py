"""
src/analytics/regime_alpha_matrix.py — Regime × Alpha Performance Matrix

FIX NEW-P1-006: The mandate (§69) requires tracking each alpha specialist's
performance per market regime. This module maintains the matrix:

    regime × alpha_family → {ic_mean, ic_std, expectancy, sharpe, drawdown,
                              sample_count, confidence_interval}

The matrix is updated from:
  1. Walk-forward validation results (per-window regime labels)
  2. Paper trade outcomes (post-resolution)
  3. Historical backtest results

Usage::

    matrix = RegimeAlphaMatrix()
    matrix.update(
        alpha_name="momentum",
        regime="TREND_UP",
        ic=0.045,
        net_return=0.012,
    )
    report = matrix.get_performance("momentum", "TREND_UP")
    full = matrix.to_dataframe()

The matrix answers:
  "Which alpha works in which regime?"
  "When should we prefer mean-reversion vs momentum?"
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)

# ── Regime labels (aligned with mandate §13) ──────────────────────────────────
VALID_REGIMES = frozenset({
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
    "HIGH_VOLATILITY",
    "LOW_VOLATILITY",
    "PANIC",
    "BREAKOUT",
    "MEAN_REVERTING",
    "LIQUIDITY_STRESS",
    "UNCERTAIN",
})

# ── Alpha family names (aligned with mandate §12) ──────────────────────────────
VALID_ALPHA_FAMILIES = frozenset({
    "TREND_ALPHA",
    "MEAN_REVERSION_ALPHA",
    "BREAKOUT_ALPHA",
    "MOMENTUM_ALPHA",
    "VOLATILITY_ALPHA",
    "EVENT_ALPHA",
    "NEWS_ALPHA",
    "RELATIVE_VALUE_ALPHA",
    "MICROSTRUCTURE_ALPHA",
    "MARKET_REGIME_ALPHA",
    "ALL",  # aggregate across all alphas
})

# Minimum observations to compute meaningful statistics
MIN_OBSERVATIONS = 10


@dataclass
class RegimeAlphaCell:
    """Statistics for one (alpha, regime) combination."""

    alpha_name: str
    regime: str
    # Accumulated observations
    ic_values: list[float] = field(default_factory=list)
    net_returns: list[float] = field(default_factory=list)
    # Summary statistics (computed on demand)
    _ic_mean: float = 0.0
    _ic_std: float = 0.0
    _expectancy: float = 0.0
    _sharpe: float = 0.0
    _max_drawdown: float = 0.0
    _sample_count: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())

    def add_observation(self, ic: float | None = None, net_return: float | None = None) -> None:
        """Add one IC observation and/or one net-return observation."""
        if ic is not None and math.isfinite(ic):
            self.ic_values.append(ic)
        if net_return is not None and math.isfinite(net_return):
            self.net_returns.append(net_return)
        self.last_updated = datetime.now(tz=UTC).isoformat()
        self._recompute()

    def _recompute(self) -> None:
        """Recompute summary statistics from accumulated data."""
        self._sample_count = max(len(self.ic_values), len(self.net_returns))

        if self.ic_values:
            arr = np.array(self.ic_values)
            self._ic_mean = float(np.mean(arr))
            self._ic_std = float(np.std(arr)) if len(arr) > 1 else 0.0
        else:
            self._ic_mean = 0.0
            self._ic_std = 0.0

        if self.net_returns:
            rets = np.array(self.net_returns)
            self._expectancy = float(np.mean(rets))
            std = float(np.std(rets))
            self._sharpe = (
                self._expectancy / std * math.sqrt(252)
                if std > 1e-12 else 0.0
            )
            # Rolling max drawdown
            equity = np.cumprod(1.0 + rets)
            running_max = np.maximum.accumulate(equity)
            dd = (equity - running_max) / running_max
            self._max_drawdown = float(np.min(dd)) if len(dd) else 0.0
        else:
            self._expectancy = 0.0
            self._sharpe = 0.0
            self._max_drawdown = 0.0

    @property
    def ic_mean(self) -> float:
        return self._ic_mean

    @property
    def ic_std(self) -> float:
        return self._ic_std

    @property
    def expectancy(self) -> float:
        return self._expectancy

    @property
    def sharpe(self) -> float:
        return self._sharpe

    @property
    def max_drawdown(self) -> float:
        return self._max_drawdown

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def has_sufficient_data(self) -> bool:
        return self._sample_count >= MIN_OBSERVATIONS

    def ic_confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """95% CI for IC using t-distribution. Returns (-1, 1) when insufficient data."""
        n = len(self.ic_values)
        if n < 3:
            return (-1.0, 1.0)
        if self._ic_std < 1e-12:
            # Zero variance — return a symmetric interval based on expected IC precision
            return (-1.0, 1.0)
        from scipy import stats as _stats  # noqa: PLC0415
        se = self._ic_std / math.sqrt(n)
        alpha = 1.0 - confidence
        t_crit = float(_stats.t.ppf(1.0 - alpha / 2, df=n - 1))
        lo = self._ic_mean - t_crit * se
        hi = self._ic_mean + t_crit * se
        return (round(lo, 6), round(hi, 6))

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.ic_confidence_interval()
        return {
            "alpha_name": self.alpha_name,
            "regime": self.regime,
            "sample_count": self._sample_count,
            "has_sufficient_data": self.has_sufficient_data,
            "ic_mean": round(self._ic_mean, 6),
            "ic_std": round(self._ic_std, 6),
            "ic_ci_95_lo": lo,
            "ic_ci_95_hi": hi,
            "expectancy": round(self._expectancy, 6),
            "sharpe": round(self._sharpe, 4),
            "max_drawdown": round(self._max_drawdown, 6),
            "last_updated": self.last_updated,
        }


class RegimeAlphaMatrix:
    """
    Tracks and reports alpha performance per market regime.

    The matrix answers the question:
    "Which alpha family has genuine conditional edge in which regime?"

    It is the engine's empirical knowledge base and drives:
    - Alpha specialist selection in the MetaDecisionEngine
    - Regime-conditional ensemble weights in EnsembleWeighter
    - Automatic demotion of alpha specialists that lose their edge in a regime

    Usage::

        matrix = RegimeAlphaMatrix()
        matrix.update("MOMENTUM_ALPHA", "TREND_UP", ic=0.04, net_return=0.008)
        best = matrix.best_alpha_for_regime("TREND_UP")
        df = matrix.to_dataframe()
        matrix.save("artifacts/regime_alpha_matrix.json")
    """

    def __init__(self) -> None:
        # (alpha_name, regime) → RegimeAlphaCell
        self._cells: dict[tuple[str, str], RegimeAlphaCell] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        alpha_name: str,
        regime: str,
        ic: float | None = None,
        net_return: float | None = None,
    ) -> None:
        """Add one observation for the (alpha, regime) cell."""
        # Normalise
        alpha = alpha_name.upper()
        reg = regime.upper()

        # Warn on unknown values but don't reject — alpha/regime discovery
        if alpha not in VALID_ALPHA_FAMILIES:
            logger.debug("regime_alpha_matrix_unknown_alpha", alpha=alpha)
        if reg not in VALID_REGIMES:
            logger.debug("regime_alpha_matrix_unknown_regime", regime=reg)

        key = (alpha, reg)
        if key not in self._cells:
            self._cells[key] = RegimeAlphaCell(alpha_name=alpha, regime=reg)

        self._cells[key].add_observation(ic=ic, net_return=net_return)

        # Also update the ALL aggregate
        all_key = (alpha, "ALL")
        if all_key not in self._cells:
            self._cells[all_key] = RegimeAlphaCell(alpha_name=alpha, regime="ALL")
        self._cells[all_key].add_observation(ic=ic, net_return=net_return)

        logger.debug(
            "regime_alpha_matrix_updated",
            alpha=alpha,
            regime=reg,
            ic=ic,
            net_return=net_return,
        )

    def get_performance(self, alpha_name: str, regime: str) -> RegimeAlphaCell | None:
        """Return the cell for (alpha, regime), or None if no data."""
        key = (alpha_name.upper(), regime.upper())
        return self._cells.get(key)

    def best_alpha_for_regime(
        self,
        regime: str,
        min_samples: int = MIN_OBSERVATIONS,
    ) -> str | None:
        """
        Return the alpha family with the highest IC in the given regime.

        Only considers cells with >= min_samples observations.
        Returns None if no alpha has sufficient data.
        """
        reg = regime.upper()
        candidates = [
            cell for (alpha, r), cell in self._cells.items()
            if r == reg and cell.sample_count >= min_samples
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c.ic_mean)
        return best.alpha_name if best.ic_mean > 0 else None

    def regime_ranking(self, regime: str) -> list[dict[str, Any]]:
        """Return all alphas for a regime sorted by IC (descending)."""
        reg = regime.upper()
        cells = [
            cell for (alpha, r), cell in self._cells.items()
            if r == reg
        ]
        cells.sort(key=lambda c: c.ic_mean, reverse=True)
        return [c.to_dict() for c in cells]

    def alpha_ranking(self, alpha_name: str) -> list[dict[str, Any]]:
        """Return all regimes for an alpha sorted by IC (descending)."""
        alpha = alpha_name.upper()
        cells = [
            cell for (a, reg), cell in self._cells.items()
            if a == alpha
        ]
        cells.sort(key=lambda c: c.ic_mean, reverse=True)
        return [c.to_dict() for c in cells]

    def to_dataframe(self) -> "pd.DataFrame":
        """Return the full matrix as a DataFrame."""
        import pandas as pd  # noqa: PLC0415
        rows = [cell.to_dict() for cell in self._cells.values()]
        if not rows:
            return pd.DataFrame(columns=[
                "alpha_name", "regime", "sample_count", "ic_mean",
                "ic_std", "expectancy", "sharpe", "max_drawdown",
            ])
        return pd.DataFrame(rows)

    def summary(self) -> dict[str, Any]:
        """High-level summary: how many cells, which regimes/alphas covered."""
        regimes_covered = list({r for (_, r) in self._cells if r != "ALL"})
        alphas_covered = list({a for (a, _) in self._cells})
        cells_with_data = sum(
            1 for c in self._cells.values() if c.has_sufficient_data
        )
        return {
            "total_cells": len(self._cells),
            "cells_with_sufficient_data": cells_with_data,
            "regimes_covered": sorted(regimes_covered),
            "alphas_covered": sorted(alphas_covered),
            "min_observations_threshold": MIN_OBSERVATIONS,
        }

    def save(self, path: str | Path) -> None:
        """Persist the matrix to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": "1.0",
            "saved_at": datetime.now(tz=UTC).isoformat(),
            "cells": [cell.to_dict() for cell in self._cells.values()],
            # Store raw values for re-loading
            "_raw": {
                f"{alpha}|{reg}": {
                    "ic_values": cell.ic_values,
                    "net_returns": cell.net_returns,
                }
                for (alpha, reg), cell in self._cells.items()
            },
        }
        p.write_text(json.dumps(data, indent=2))
        logger.info("regime_alpha_matrix_saved", path=str(p), cells=len(self._cells))

    @classmethod
    def load(cls, path: str | Path) -> "RegimeAlphaMatrix":
        """Load a previously saved matrix."""
        p = Path(path)
        if not p.exists():
            logger.warning("regime_alpha_matrix_load_not_found", path=str(p))
            return cls()
        data = json.loads(p.read_text())
        matrix = cls()
        raw = data.get("_raw", {})
        for key, vals in raw.items():
            alpha, reg = key.split("|", 1)
            for ic, nr in zip(
                vals.get("ic_values", []),
                vals.get("net_returns", []) + [None] * len(vals.get("ic_values", [])),
            ):
                matrix.update(alpha, reg, ic=ic)
            for nr in vals.get("net_returns", []):
                matrix.update(alpha, reg, net_return=nr)
        logger.info("regime_alpha_matrix_loaded", path=str(p), cells=len(matrix._cells))
        return matrix
