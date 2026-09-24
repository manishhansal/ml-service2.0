"""
src.data.labels — LabelFactory (P0-002).

Constructs PIT-safe supervised-learning labels from a clean OHLCV DataFrame.

Supported label types
----------------------
1. Fixed-horizon return       (fixed_horizon)
2. Triple-barrier             (triple_barrier)      — López de Prado
3. Volatility-adjusted barrier(vol_adjusted_barrier) — barriers scaled by realized vol
4. Meta-label                 (meta_label)          — was the primary signal correct?

Every label row carries full provenance so that a downstream dataset can be
audited and validated for leakage:
    label_start, label_end, horizon, upper_barrier, lower_barrier,
    outcome, realized_return, realized_cost, mae, mfe, time_to_event.

PIT safety
----------
A label at time t looks FORWARD over [t+1, t+horizon]. The label is therefore
only knowable at t+horizon. The dataset builder must ensure that the feature
vector for row t uses ONLY data <= t, and that validation splits purge/embargo
the horizon so a label's forward window never overlaps a validation fold.

Costs
-----
Labels can incorporate a round-trip transaction cost (bps) so that
``realized_return_net`` reflects realistic post-cost outcomes.

Requirements: P0-002, 08_LABEL_SPECIFICATION.
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


@dataclass
class LabelConfig:
    """Configuration for label construction."""

    label_type: LabelType = "triple_barrier"
    horizon: int = 5                    # bars to look forward
    upper_barrier_pct: float = 0.02     # +2% target (fixed-barrier mode)
    lower_barrier_pct: float = 0.02     # -2% stop
    vol_window: int = 20                # window for realized-vol scaling
    vol_multiplier: float = 1.5         # barrier = vol_multiplier * realized_vol
    cost_bps: float = 10.0              # round-trip transaction cost (basis points)
    return_threshold: float = 0.0       # fixed-horizon: > threshold → class 1


class LabelFactory:
    """
    Builds PIT-safe labels from a clean OHLCV DataFrame indexed by timestamp.

    Usage::

        factory = LabelFactory(LabelConfig(label_type="triple_barrier", horizon=5))
        labels = factory.build(ohlcv_df)   # DataFrame aligned to ohlcv_df.index
    """

    def __init__(self, config: LabelConfig | None = None) -> None:
        self.config = config or LabelConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Dispatch to the configured label constructor."""
        if "close" not in ohlcv.columns:
            raise ValueError("LabelFactory requires a 'close' column.")
        lt = self.config.label_type
        if lt == "fixed_horizon":
            return self._fixed_horizon(ohlcv)
        if lt == "triple_barrier":
            return self._triple_barrier(ohlcv, vol_adjusted=False)
        if lt == "vol_adjusted_barrier":
            return self._triple_barrier(ohlcv, vol_adjusted=True)
        if lt == "meta_label":
            raise ValueError("meta_label requires build_meta_labels(primary_side=...)")
        raise ValueError(f"Unknown label_type: {lt}")

    def build_meta_labels(
        self,
        ohlcv: pd.DataFrame,
        primary_side: pd.Series,
    ) -> pd.DataFrame:
        """
        Meta-labels: given a primary model's directional call (+1 long / -1 short
        / 0 flat), label whether taking that side would have been PROFITABLE net
        of costs over the horizon. Output ``label`` = 1 (take) / 0 (pass).
        """
        base = self._triple_barrier(ohlcv, vol_adjusted=True)
        side = primary_side.reindex(base.index).fillna(0)
        # Net directional return if we followed the primary side.
        directional_net = side * base["realized_return"] - base["realized_cost"]
        out = base.copy()
        out["primary_side"] = side
        out["label"] = (directional_net > 0).astype(int)
        out["meta_return_net"] = directional_net
        return out

    # ── Fixed-horizon ─────────────────────────────────────────────────────────

    def _fixed_horizon(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        close = ohlcv["close"].astype(float)
        fwd = close.shift(-cfg.horizon)
        realized = (fwd - close) / close
        cost = cfg.cost_bps / 10_000.0
        net = realized - cost

        out = pd.DataFrame(index=ohlcv.index)
        out["label_start"] = ohlcv.index
        out["label_end"] = ohlcv.index.to_series().shift(-cfg.horizon).values
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

        if vol_adjusted:
            rets = pd.Series(close).pct_change()
            vol = rets.rolling(cfg.vol_window).std().to_numpy()
        else:
            vol = None

        rows = []
        for i in range(n):
            entry = close[i]
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

            end = min(i + cfg.horizon, n - 1)
            outcome = "TIME_EXPIRY"
            hit_idx = end
            realized = 0.0

            path_high = high[i + 1 : end + 1] if i + 1 <= end else np.array([])
            path_low = low[i + 1 : end + 1] if i + 1 <= end else np.array([])
            path_close = close[i + 1 : end + 1] if i + 1 <= end else np.array([])

            # Walk forward to find the first barrier touched.
            for j in range(len(path_close)):
                if path_high[j] >= upper:
                    outcome = "TARGET_HIT"
                    hit_idx = i + 1 + j
                    realized = up_pct
                    break
                if path_low[j] <= lower:
                    outcome = "STOP_HIT"
                    hit_idx = i + 1 + j
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

            rows.append({
                "label_start": idx[i],
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
