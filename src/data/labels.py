"""
src.data.labels — LabelFactory (P0-002).

Constructs PIT-safe supervised-learning labels from a clean OHLCV DataFrame.

Supported label types
----------------------
1. Fixed-horizon return       (fixed_horizon)
2. Triple-barrier             (triple_barrier)      — López de Prado
3. Volatility-adjusted barrier(vol_adjusted_barrier) — barriers scaled by realized vol
4. Meta-label                 (meta_label)          — was the primary signal correct?

EXECUTION CONTRACT (mandate §20, §40)
--------------------------------------
Labels must be aligned with the ACTUAL execution assumption. Two modes are
supported:

  close_to_close   — entry at close[T], exit at close[T+horizon]
                     WARNING: this is NOT executable alpha. A close-to-close
                     signal cannot be entered at the same bar's close without
                     look-ahead. Use ONLY for research diagnostics.

  next_open        — signal at close[T], entry at open[T+1], exit at future open
                     This is the canonical EXECUTABLE mode for EOD strategies.
                     The label measures open[T+1+h] - open[T+1].

The DatasetBuilder MUST use execution_model="next_open" for any dataset
that claims economic viability (mandate §20).

Every label row carries full provenance so that a downstream dataset can be
audited and validated for leakage:
    label_start, label_end, horizon, upper_barrier, lower_barrier,
    outcome, realized_return, realized_cost, mae, mfe, time_to_event,
    execution_model, entry_price, exit_price, signal_timestamp

PIT safety
----------
A label at time t looks FORWARD over [t+1, t+horizon] (next-open mode) or
[t, t+horizon] (close-to-close mode, research only). The label is therefore
only knowable at t+horizon. The dataset builder must ensure that the feature
vector for row t uses ONLY data <= t, and that validation splits purge/embargo
the horizon so a label's forward window never overlaps a validation fold.

Costs
-----
Labels can incorporate a round-trip transaction cost (bps) so that
``realized_return_net`` reflects realistic post-cost outcomes.

Requirements: P0-002, 08_LABEL_SPECIFICATION, mandate §20, §40.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

LabelType = Literal[
    "fixed_horizon", "triple_barrier", "vol_adjusted_barrier", "meta_label"
]

ExecutionModel = Literal["next_open", "close_to_close"]


@dataclass
class LabelConfig:
    """Configuration for label construction.

    execution_model controls which price is used for entry:
      "next_open"      — executable: signal at close[T], enter at open[T+1]
      "close_to_close" — research diagnostic only: enter at close[T]

    Default is "next_open" (mandate §20). Use "close_to_close" ONLY for
    explicit research diagnostics where you understand it is not executable.
    """

    label_type: LabelType = "triple_barrier"
    horizon: int = 5                    # bars to look forward
    upper_barrier_pct: float = 0.02     # +2% target (fixed-barrier mode)
    lower_barrier_pct: float = 0.02     # -2% stop
    vol_window: int = 20                # window for realized-vol scaling
    vol_multiplier: float = 1.5         # barrier = vol_multiplier * realized_vol
    cost_bps: float = 10.0              # round-trip transaction cost (basis points)
    return_threshold: float = 0.0       # fixed-horizon: > threshold → class 1
    execution_model: ExecutionModel = "next_open"  # MUST be next_open for economic validity


class LabelFactory:
    """
    Builds PIT-safe labels from a clean OHLCV DataFrame indexed by timestamp.

    EXECUTION MODEL (mandate §20, §40):
    - execution_model="next_open" (default): signal at close[T], enter at open[T+1].
      The label measures returns from open[T+1] forward. This is the ONLY mode
      that produces economically valid labels for EOD strategies.
    - execution_model="close_to_close": research diagnostic only.
      INVARIANT: any dataset using close_to_close is marked is_economic_evidence=False.

    Usage::

        factory = LabelFactory(LabelConfig(label_type="triple_barrier", horizon=5))
        labels = factory.build(ohlcv_df)   # DataFrame aligned to ohlcv_df.index
    """

    def __init__(self, config: LabelConfig | None = None) -> None:
        self.config = config or LabelConfig()

    @property
    def execution_model(self) -> ExecutionModel:
        return self.config.execution_model

    @property
    def is_economic_evidence(self) -> bool:
        """True only when execution_model=next_open (mandate §20, §40)."""
        return self.config.execution_model == "next_open"

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Dispatch to the configured label constructor.

        Adds ``execution_model`` and ``is_economic_evidence`` columns to every
        output row so the dataset builder and backtest can enforce the mandate
        §20 invariant: close_to_close labels are NEVER used for champion selection.
        """
        if "close" not in ohlcv.columns:
            raise ValueError("LabelFactory requires a 'close' column.")

        # INVARIANT check: warn loudly if using close_to_close
        if self.config.execution_model == "close_to_close":
            logger.warning(
                "label_factory_close_to_close_mode",
                warning=(
                    "execution_model=close_to_close produces research-only labels. "
                    "is_economic_evidence=False. This dataset MUST NOT be used for "
                    "champion selection or OOS economic certification (mandate §20, §40)."
                ),
            )

        lt = self.config.label_type
        if lt == "fixed_horizon":
            out = self._fixed_horizon(ohlcv)
        elif lt == "triple_barrier":
            out = self._triple_barrier(ohlcv, vol_adjusted=False)
        elif lt == "vol_adjusted_barrier":
            out = self._triple_barrier(ohlcv, vol_adjusted=True)
        elif lt == "meta_label":
            raise ValueError("meta_label requires build_meta_labels(primary_side=...)")
        else:
            raise ValueError(f"Unknown label_type: {lt}")

        # Attach execution provenance to every row (mandate §20)
        out["execution_model"] = self.config.execution_model
        out["is_economic_evidence"] = self.is_economic_evidence
        return out

    def build_meta_labels(
        self,
        ohlcv: pd.DataFrame,
        primary_side: pd.Series,
    ) -> pd.DataFrame:
        """
        Meta-labels: given a primary model's directional call (+1 long / -1 short
        / 0 flat), label whether taking that side would have been PROFITABLE net
        of costs over the horizon. Output ``label`` = 1 (take) / 0 (pass).
        Execution model is inherited from self.config.
        """
        base = self._triple_barrier(ohlcv, vol_adjusted=True)
        side = primary_side.reindex(base.index).fillna(0)
        # Net directional return if we followed the primary side.
        directional_net = side * base["realized_return"] - base["realized_cost"]
        out = base.copy()
        out["primary_side"] = side
        out["label"] = (directional_net > 0).astype(int)
        out["meta_return_net"] = directional_net
        out["execution_model"] = self.config.execution_model
        out["is_economic_evidence"] = self.is_economic_evidence
        return out

    # ── Fixed-horizon ─────────────────────────────────────────────────────────

    def _fixed_horizon(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        close = ohlcv["close"].astype(float)

        if cfg.execution_model == "next_open":
            # Executable: entry at open[T+1], exit at open[T+1+horizon]
            # Requires an "open" column; raise explicitly if absent.
            if "open" not in ohlcv.columns:
                raise ValueError(
                    "LabelFactory: execution_model=next_open requires an 'open' column "
                    "for entry at open[T+1]. Provide OHLCV data with open prices."
                )
            open_ = ohlcv["open"].astype(float)
            entry_price = open_.shift(-1)            # open[T+1] — our entry
            exit_price = open_.shift(-(1 + cfg.horizon))  # open[T+1+h] — our exit
            realized = (exit_price - entry_price) / entry_price
            signal_col = "close"
        else:
            # close_to_close: research diagnostic only (not executable)
            entry_price = close
            exit_price = close.shift(-cfg.horizon)
            realized = (exit_price - entry_price) / entry_price
            signal_col = "close"

        cost = cfg.cost_bps / 10_000.0
        net = realized - cost

        out = pd.DataFrame(index=ohlcv.index)
        out["label_start"] = ohlcv.index
        out["signal_timestamp"] = ohlcv.index       # close[T] — when signal fires
        out["entry_price"] = entry_price
        out["exit_price"] = exit_price
        out["label_end"] = ohlcv.index.to_series().shift(-(1 + cfg.horizon) if cfg.execution_model == "next_open" else -cfg.horizon).values
        out["horizon"] = cfg.horizon
        out["realized_return"] = realized
        out["realized_cost"] = cost
        out["realized_return_net"] = net
        out["label"] = (realized > cfg.return_threshold).astype("Int64")
        out["outcome"] = np.where(
            realized > cfg.return_threshold, "UP",
            np.where(realized < -cfg.return_threshold, "DOWN", "FLAT"),
        )
        out.loc[realized.isna(), "label"] = pd.NA
        out.loc[realized.isna(), "outcome"] = "UNRESOLVED"
        self._log_summary(out)
        return out

    # ── Triple-barrier ──────────────────────────────────────────────────────

    def _triple_barrier(self, ohlcv: pd.DataFrame, vol_adjusted: bool) -> pd.DataFrame:
        cfg = self.config
        close = ohlcv["close"].astype(float).to_numpy()
        high = ohlcv.get("high", ohlcv["close"]).astype(float).to_numpy()
        low = ohlcv.get("low", ohlcv["close"]).astype(float).to_numpy()
        idx = ohlcv.index
        n = len(close)
        cost = cfg.cost_bps / 10_000.0

        # Determine entry prices based on execution model (mandate §20).
        if cfg.execution_model == "next_open":
            if "open" not in ohlcv.columns:
                raise ValueError(
                    "LabelFactory: execution_model=next_open requires an 'open' column. "
                    "Provide OHLCV data with open prices."
                )
            open_arr = ohlcv["open"].astype(float).to_numpy()
            # entry_prices[i] = open[i+1]  (we enter at the next bar's open)
            # For the last bar, no entry is possible → NaN
            entry_prices = np.append(open_arr[1:], np.nan)
        else:
            # close_to_close: entry at close[T] (research only)
            entry_prices = close.copy()

        if vol_adjusted:
            rets = pd.Series(close).pct_change()
            vol = rets.rolling(cfg.vol_window).std().to_numpy()
        else:
            vol = None

        rows = []
        for i in range(n):
            entry = entry_prices[i]
            if np.isnan(entry):
                # No entry possible (last bar in next_open mode)
                rows.append({
                    "label_start": idx[i],
                    "signal_timestamp": idx[i],
                    "entry_price": np.nan,
                    "exit_price": np.nan,
                    "label_end": pd.NaT,
                    "horizon": cfg.horizon,
                    "upper_barrier": np.nan,
                    "lower_barrier": np.nan,
                    "outcome": "UNRESOLVED",
                    "realized_return": np.nan,
                    "realized_cost": cost,
                    "realized_return_net": np.nan,
                    "mae": np.nan,
                    "mfe": np.nan,
                    "time_to_event": 0,
                    "label": pd.NA,
                })
                continue

            if vol_adjusted:
                v = vol[i] if vol is not None and not np.isnan(vol[i]) else None
                if v is None or v <= 0:
                    up_pct = cfg.upper_barrier_pct
                    dn_pct = cfg.lower_barrier_pct
                else:
                    up_pct = cfg.vol_multiplier * v
                    dn_pct = cfg.vol_multiplier * v
            else:
                up_pct = cfg.upper_barrier_pct
                dn_pct = cfg.lower_barrier_pct

            upper = entry * (1.0 + up_pct)
            lower = entry * (1.0 - dn_pct)

            # For next_open mode: the forward window starts at bar i+1 (entry bar)
            # For close_to_close: forward window starts at bar i+1 (next close)
            start_idx = i + 1
            end = min(start_idx + cfg.horizon - 1, n - 1)
            outcome = "TIME_EXPIRY"
            hit_idx = end
            realized = 0.0

            path_high = high[start_idx: end + 1] if start_idx <= end else np.array([])
            path_low = low[start_idx: end + 1] if start_idx <= end else np.array([])
            path_close = close[start_idx: end + 1] if start_idx <= end else np.array([])

            # Walk forward to find the first barrier touched.
            for j in range(len(path_close)):
                if path_high[j] >= upper:
                    outcome = "TARGET_HIT"
                    hit_idx = start_idx + j
                    realized = up_pct
                    break
                if path_low[j] <= lower:
                    outcome = "STOP_HIT"
                    hit_idx = start_idx + j
                    realized = -dn_pct
                    break
            else:
                if len(path_close) > 0:
                    realized = (path_close[-1] - entry) / entry
                    hit_idx = end
                else:
                    realized = np.nan
                    outcome = "UNRESOLVED"

            # MAE / MFE over the realized window.
            if len(path_low) > 0:
                mae = (np.min(path_low) - entry) / entry
                mfe = (np.max(path_high) - entry) / entry
            else:
                mae = np.nan
                mfe = np.nan

            label = pd.NA
            if outcome == "TARGET_HIT":
                label = 1
            elif outcome == "STOP_HIT":
                label = 0
            elif outcome == "TIME_EXPIRY" and not np.isnan(realized):
                label = int(realized > 0)

            exit_price_val = path_close[-1] if len(path_close) > 0 and not np.isnan(realized) else np.nan

            rows.append({
                "label_start": idx[i],
                "signal_timestamp": idx[i],
                "entry_price": entry,
                "exit_price": exit_price_val,
                "label_end": idx[hit_idx] if not np.isnan(realized) else pd.NaT,
                "horizon": cfg.horizon,
                "upper_barrier": upper,
                "lower_barrier": lower,
                "outcome": outcome,
                "realized_return": realized,
                "realized_cost": cost,
                "realized_return_net": (realized - cost) if not np.isnan(realized) else np.nan,
                "mae": mae,
                "mfe": mfe,
                "time_to_event": hit_idx - i,
                "label": label,
            })

        out = pd.DataFrame(rows, index=idx)
        out["label"] = out["label"].astype("Int64")
        self._log_summary(out)
        return out

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @staticmethod
    def _log_summary(out: pd.DataFrame) -> None:
        labels = out["label"].dropna()
        if len(labels) == 0:
            logger.warning("label_factory_no_resolved_labels")
            return
        pos = int((labels == 1).sum())
        logger.info(
            "label_factory_built",
            n=len(labels),
            positive=pos,
            positive_frac=round(pos / len(labels), 4),
        )


# ── Label quality diagnostics (Phase 17) ───────────────────────────────────────


def label_quality_report(labels: pd.DataFrame) -> dict[str, float | int | dict]:
    """
    Compute label-quality diagnostics used by 08_LABEL_SPECIFICATION:
      - class balance
      - label autocorrelation (lag-1) — high values indicate overlapping-label leakage
      - resolved fraction
      - outcome distribution
    """
    resolved = labels["label"].dropna()
    n = len(resolved)
    if n == 0:
        return {"n_resolved": 0, "warning": "NO_RESOLVED_LABELS"}

    pos = int((resolved == 1).sum())
    class_balance = pos / n

    # Lag-1 autocorrelation of the (0/1) label series.
    series = resolved.astype(float)
    if series.std() > 1e-12 and n > 2:
        autocorr = float(series.autocorr(lag=1))
    else:
        autocorr = 0.0

    outcome_dist: dict[str, int] = {}
    if "outcome" in labels.columns:
        outcome_dist = labels["outcome"].value_counts().to_dict()

    return {
        "n_resolved": n,
        "n_total": len(labels),
        "resolved_fraction": round(n / len(labels), 4),
        "class_balance_positive": round(class_balance, 4),
        "label_autocorrelation_lag1": round(autocorr, 4),
        "outcome_distribution": {str(k): int(v) for k, v in outcome_dist.items()},
        "mean_realized_return": round(float(labels["realized_return"].dropna().mean()), 6)
        if "realized_return" in labels.columns else 0.0,
    }
