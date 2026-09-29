"""
src.labels.multi_horizon — Multi-horizon label pipeline (P2-002 fix).

Problem (P2-002)
----------------
The current system trains exclusively on 5-bar horizon labels.  This
hard-wires the signal's temporal edge to a single time scale, potentially
missing stronger alpha at other horizons and preventing the construction of
ensemble or horizon-blended models.

Solution
--------
``MultiHorizonLabelFactory`` builds vol-adjusted excess-return labels for
multiple horizons simultaneously [1, 3, 5, 10, 21 bars].  The output is a
``MultiHorizonDataset`` that:
- Stores one label column per horizon in the feature DataFrame.
- Exposes the primary horizon label as ``label`` (compatible with existing
  training pipelines).
- Annotates metadata with the full horizon set so the training orchestrator
  can train horizon-specific or ensemble models.
- Maintains strict PIT (point-in-time) safety: no future close prices leak.

Institutional motivation
------------------------
- 1-bar:  intraday momentum capture; high turnover but high IC in trending markets
- 3-bar:  short-term mean-reversion after news events
- 5-bar:  primary operational horizon (matches current live session)
- 10-bar: intermediate trend; lower turnover, still momentum-driven
- 21-bar: monthly strategic; best Sharpe at lowest execution frequency

Rolling IC across horizons provides an IC term structure — a key diagnostic
for understanding whether the signal has momentum or mean-reversion character.

Usage::

    factory = MultiHorizonLabelFactory(horizons=[1, 3, 5, 10, 21])
    dataset = factory.build(stock_df, nifty_close)
    df = dataset.to_frame()    # one label column per horizon

Requirements: NEW-P2-002, mandate §3, Phase H.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.labels.relative import generate_ranking_labels_v2_compat
from src.logging_config import get_logger

logger = get_logger(__name__)

# Default horizon set — institutional convention.
# Covers: intraday momentum (1-3), primary (5), intermediate (10), monthly (21).
DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 21)


@dataclass
class MultiHorizonDataset:
    """
    Container for a set of label series across multiple horizons.

    Attributes:
        symbol:           Symbol this dataset was built for.
        horizons:         Sorted list of horizon bars used.
        primary_horizon:  The horizon that becomes the ``label`` column.
        labels:           Dict mapping horizon → pd.Series of vol-adj labels.
        index:            DatetimeIndex aligned to the input DataFrame.
        ic_term_structure: Dict mapping horizon → in-sample rank IC (diagnostic).
    """

    symbol: str
    horizons: list[int]
    primary_horizon: int
    labels: dict[int, pd.Series]
    index: pd.DatetimeIndex
    ic_term_structure: dict[int, float] = field(default_factory=dict)

    def primary_label(self) -> pd.Series:
        """Return the primary-horizon label series (used for training)."""
        return self.labels[self.primary_horizon]

    def to_frame(self) -> pd.DataFrame:
        """
        Return a DataFrame with one label column per horizon.

        Columns: ``label`` (primary), ``label_h1``, ``label_h3``, ...
        """
        dfs = {f"label_h{h}": s for h, s in sorted(self.labels.items())}
        dfs["label"] = self.labels[self.primary_horizon]
        return pd.DataFrame(dfs, index=self.index)

    def rank_ic(self, h: int) -> float:
        """Spearman rank IC for a specific horizon (requires ic_term_structure)."""
        return self.ic_term_structure.get(h, float("nan"))


class MultiHorizonLabelFactory:
    """
    Builds vol-adjusted excess-return labels across multiple time horizons.

    This is a thin wrapper around ``generate_ranking_labels_v2_compat``
    that calls it for each horizon and assembles the results into a
    ``MultiHorizonDataset``.

    PIT safety: the underlying label generator uses only data up to and
    including bar i for the volatility normalizer.  The forward return uses
    close[i+h] which is genuinely future data — this is the label definition,
    not leakage.  The last ``max(horizons)`` bars are always NaN.

    Args:
        horizons:        List of forward horizons (in bars) to compute.
                         Default: [1, 3, 5, 10, 21].
        primary_horizon: Horizon used as the main training label.
                         Default: 5 (matches the live forward paper).
        compute_ic:      If True, compute in-sample rank IC per horizon
                         to populate the IC term structure (diagnostic only —
                         in-sample IC is NOT used for model selection).
    """

    def __init__(
        self,
        horizons: list[int] | tuple[int, ...] = DEFAULT_HORIZONS,
        primary_horizon: int = 5,
        compute_ic: bool = True,
    ) -> None:
        horizons = sorted(horizons)
        if primary_horizon not in horizons:
            raise ValueError(
                f"primary_horizon={primary_horizon} must be in horizons={horizons}"
            )
        self.horizons = horizons
        self.primary_horizon = primary_horizon
        self.compute_ic = compute_ic

    def build(
        self,
        stock_df: pd.DataFrame,
        nifty_close: pd.Series,
        symbol: str = "UNKNOWN",
    ) -> MultiHorizonDataset:
        """
        Build multi-horizon labels for a single symbol.

        Args:
            stock_df:    OHLCV DataFrame indexed by date (requires 'close').
            nifty_close: NIFTY index close series (same date index).
            symbol:      Symbol name for logging / metadata.

        Returns:
            MultiHorizonDataset with labels for all horizons.
        """
        labels: dict[int, pd.Series] = {}
        ic_ts: dict[int, float] = {}

        for h in self.horizons:
            series = generate_ranking_labels_v2_compat(
                stock_df=stock_df,
                nifty_close=nifty_close,
                horizon=h,
            )
            labels[h] = series

            # Compute in-sample rank IC (diagnostic only — not for selection)
            if self.compute_ic:
                valid = series.dropna()
                if len(valid) >= 30:
                    # Use the primary-horizon lagged return as the "score"
                    # (self-IC measures autocorrelation as a sanity check)
                    lag_ret = stock_df["close"].pct_change().shift(1).reindex(valid.index)
                    mask = lag_ret.notna() & valid.notna()
                    if mask.sum() >= 20:
                        rank_score = lag_ret[mask].rank()
                        rank_label = valid[mask].rank()
                        n = mask.sum()
                        d = rank_score - rank_label
                        ic_ts[h] = round(float(1.0 - 6.0 * (d**2).sum() / (n * (n**2 - 1))), 4)

        dataset = MultiHorizonDataset(
            symbol=symbol,
            horizons=self.horizons,
            primary_horizon=self.primary_horizon,
            labels=labels,
            index=stock_df.index,
            ic_term_structure=ic_ts,
        )
        logger.debug(
            "multi_horizon_labels_built",
            symbol=symbol,
            horizons=self.horizons,
            primary_horizon=self.primary_horizon,
            n_valid_primary=int(labels[self.primary_horizon].notna().sum()),
        )
        return dataset

    def build_universe(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        nifty_close: pd.Series,
    ) -> dict[str, MultiHorizonDataset]:
        """
        Build multi-horizon labels for all symbols in a universe dict.

        Returns:
            Dict mapping symbol → MultiHorizonDataset.
        """
        result: dict[str, MultiHorizonDataset] = {}
        for symbol, df in ohlcv_by_symbol.items():
            if len(df) < max(self.horizons) + 30:
                logger.warning(
                    "multi_horizon_skip_symbol",
                    symbol=symbol,
                    rows=len(df),
                    required=max(self.horizons) + 30,
                )
                continue
            result[symbol] = self.build(df, nifty_close, symbol=symbol)
        logger.info(
            "multi_horizon_universe_built",
            n_symbols=len(result),
            horizons=self.horizons,
        )
        return result

    def ic_term_structure_summary(
        self, datasets: dict[str, MultiHorizonDataset]
    ) -> dict[str, Any]:
        """
        Aggregate IC term structure across the full universe.

        Returns a summary dict with mean IC and std IC per horizon —
        useful for identifying which horizon has the strongest signal.
        """
        ic_by_horizon: dict[int, list[float]] = {h: [] for h in self.horizons}
        for ds in datasets.values():
            for h, ic in ds.ic_term_structure.items():
                if not np.isnan(ic):
                    ic_by_horizon[h].append(ic)

        summary: dict[str, Any] = {}
        best_h, best_ic = self.primary_horizon, 0.0
        for h in self.horizons:
            vals = ic_by_horizon[h]
            mean_ic = float(np.mean(vals)) if vals else float("nan")
            std_ic  = float(np.std(vals))  if vals else float("nan")
            summary[f"h{h}"] = {
                "horizon_bars": h,
                "mean_ic":      round(mean_ic, 4),
                "std_ic":       round(std_ic,  4),
                "n_symbols":    len(vals),
            }
            if not np.isnan(mean_ic) and abs(mean_ic) > abs(best_ic):
                best_h, best_ic = h, mean_ic

        summary["best_horizon"] = best_h
        summary["best_ic"]      = round(best_ic, 4)
        logger.info(
            "ic_term_structure_summary",
            best_horizon=best_h,
            best_ic=round(best_ic, 4),
        )
        return summary
