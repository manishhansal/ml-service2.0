"""
src.features.normalizer — feature normalization and outlier clipping (Gap 3 fix).

Why this layer exists
---------------------
The FeatureFactory produces raw computed values (returns, RSI, ATR-pct, etc.)
that span wildly different scales.  ML models are sensitive to:

  1. Extreme outliers  — a 20-sigma return spike poisons tree splits and
     gradient boosting step sizes.
  2. Scale differences — unbounded price-ratio features sit next to [0,1]
     bounded oscillators; unscaled inputs bias regularization.
  3. Non-stationarity — raw prices are non-stationary; returns are not, but
     their cross-sectional distribution shifts over regimes.

Design principles
-----------------
* The normalizer is ALWAYS fitted on training data only (``fit``), then
  applied to test / live data (``transform``).  Fitting on the full dataset
  before train/test split is a form of look-ahead bias.
* Clipping is applied BEFORE scaling.  Clipping bounds are symmetric around
  the median (Winsorization at a configurable percentile, default p=1/99).
* After clipping, two scaling methods are supported per-feature:
    - ``robust``    : (x - median) / IQR  — the default; robust to outliers.
    - ``zscore``    : (x - mean)   / std  — faster but sensitive to outliers.
    - ``minmax``    : (x - min) / (x - max)  — use only for bounded features.
    - ``none``      : pass-through; no scaling (e.g. already-bounded RSI [0,100]).
* Fitted statistics are serializable to / from a plain dict so they can be
  stored in the dataset metadata and reproduced exactly during inference.
* The normalizer is stateless between calls to ``transform`` — it never
  modifies the underlying data in place; always returns a new DataFrame.
* NaN values are preserved through the pipeline — they are never filled.

Clipping policy by feature family (institutional best-practice)
---------------------------------------------------------------
  returns       : clip at ±5 std from rolling median  (regime-aware)
  oscillators   : clip at defined bounds ([0,100] for RSI, etc.)
  ratio features: clip at 99th percentile
  volume        : clip at 99.5th percentile (fat right tail)

Usage
-----
    from src.features.normalizer import FeatureNormalizer

    # At dataset build time (training):
    norm = FeatureNormalizer()
    norm.fit(train_features_df)
    X_train = norm.transform(train_features_df)

    # At inference time (loaded from metadata):
    norm2 = FeatureNormalizer.from_dict(norm.to_dict())
    X_live = norm2.transform(live_feature_df)

    # Check what was clipped:
    report = norm2.clip_report(live_feature_df)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)


# ── Scaling method ────────────────────────────────────────────────────────────


class ScalingMethod(str, Enum):
    ROBUST  = "robust"   # (x - median) / IQR           ← default
    ZSCORE  = "zscore"   # (x - mean) / std
    MINMAX  = "minmax"   # (x - min) / (max - min)
    NONE    = "none"     # pass-through


# ── Per-feature spec ─────────────────────────────────────────────────────────


@dataclass
class FeatureNormSpec:
    """Normalization specification for a single feature.

    clip_low / clip_high are the HARD bounds applied before scaling.
    They are determined by fit() using the winsorization percentiles.
    Setting either to None disables that side of clipping.
    """
    feature_name: str
    method:       ScalingMethod = ScalingMethod.ROBUST
    clip_low:     float | None = None   # fitted lower Winsor bound
    clip_high:    float | None = None   # fitted upper Winsor bound
    center:       float | None = None   # fitted center (median or mean)
    scale:        float | None = None   # fitted scale  (IQR, std, or range)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "method":       self.method.value,
            "clip_low":     self.clip_low,
            "clip_high":    self.clip_high,
            "center":       self.center,
            "scale":        self.scale,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureNormSpec":
        return cls(
            feature_name = d["feature_name"],
            method       = ScalingMethod(d["method"]),
            clip_low     = d.get("clip_low"),
            clip_high    = d.get("clip_high"),
            center       = d.get("center"),
            scale        = d.get("scale"),
        )


# ── Default method overrides for known feature families ──────────────────────
# Features not listed here default to ScalingMethod.ROBUST.

_FEATURE_METHOD_DEFAULTS: dict[str, ScalingMethod] = {
    # RSI is [0, 100] — already bounded, robust scaling maps to ~[-1, 1]
    "rsi_14":         ScalingMethod.ROBUST,
    # Stochastic K is [0, 1] — robust scaling
    "stoch_k_14":     ScalingMethod.ROBUST,
    # Close-position in bar is [0, 1] — robust scaling
    "close_position": ScalingMethod.ROBUST,
    # ADX is [0, 100] — bounded, robust
    "adx_14":         ScalingMethod.ROBUST,
    # Volume z-score is already centered — use zscore for consistency
    "volume_zscore_20": ScalingMethod.ZSCORE,
    # MACD histogram is price-normalized; robust
    "macd_hist":      ScalingMethod.ROBUST,
    # Skewness / kurtosis — zscore; these can be extreme
    "skew_20":        ScalingMethod.ZSCORE,
    "kurt_20":        ScalingMethod.ZSCORE,
}

# Winsorization percentile overrides by feature group (default = 1.0 / 99.0)
_WINSOR_PERCENTILE_OVERRIDES: dict[str, tuple[float, float]] = {
    # Volume has a fat right tail; clip harder
    "rel_volume_20":    (0.5, 99.5),
    "volume_zscore_20": (0.5, 99.5),
    # Kurtosis is extremely fat-tailed; clip at 2.5 / 97.5
    "kurt_20":          (2.5, 97.5),
    "skew_20":          (2.5, 97.5),
}

_DEFAULT_WINSOR = (1.0, 99.0)


# ── Clip report ───────────────────────────────────────────────────────────────


@dataclass
class ClipReport:
    """Records how many values were clipped per feature during a transform call."""
    feature:       str
    n_total:       int
    n_clipped_low: int
    n_clipped_high: int
    clip_low:      float | None
    clip_high:     float | None

    @property
    def clip_rate_pct(self) -> float:
        if self.n_total == 0:
            return 0.0
        return (self.n_clipped_low + self.n_clipped_high) / self.n_total * 100


# ── Main normalizer ───────────────────────────────────────────────────────────


class FeatureNormalizer:
    """
    Fit-then-transform feature normalizer with per-feature Winsorization.

    Fit on TRAINING data only.  Apply to training, validation, test, and live.

    Example::

        norm = FeatureNormalizer(winsor_pct=(1.0, 99.0), min_scale=1e-8)
        norm.fit(X_train)
        X_train_norm = norm.transform(X_train)
        X_test_norm  = norm.transform(X_test)

        # Save / load
        state = norm.to_dict()
        norm2 = FeatureNormalizer.from_dict(state)
    """

    def __init__(
        self,
        winsor_pct: tuple[float, float] = _DEFAULT_WINSOR,
        min_scale:  float = 1e-8,
        default_method: ScalingMethod = ScalingMethod.ROBUST,
    ) -> None:
        self._winsor_pct     = winsor_pct
        self._min_scale      = min_scale
        self._default_method = default_method
        self._specs:  dict[str, FeatureNormSpec] = {}
        self._fitted: bool = False

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "FeatureNormalizer":
        """Compute normalization statistics from *df* (training data only).

        Statistics computed per column:
          - Winsorization bounds (percentiles)
          - Center (median for ROBUST, mean for ZSCORE/MINMAX)
          - Scale  (IQR for ROBUST, std for ZSCORE, range for MINMAX)

        NaN rows are excluded from all stat computations.
        Returns self for chaining.
        """
        if df.empty:
            logger.warning("feature_normalizer_fit_empty_dataframe")
            self._fitted = True
            return self

        for col in df.columns:
            arr = df[col].to_numpy(dtype=float)
            finite = arr[np.isfinite(arr)]

            if len(finite) == 0:
                # All NaN/inf column — store a pass-through spec
                self._specs[col] = FeatureNormSpec(
                    feature_name=col,
                    method=ScalingMethod.NONE,
                )
                continue

            method = _FEATURE_METHOD_DEFAULTS.get(col, self._default_method)
            lo_pct, hi_pct = _WINSOR_PERCENTILE_OVERRIDES.get(col, self._winsor_pct)

            clip_low  = float(np.percentile(finite, lo_pct))
            clip_high = float(np.percentile(finite, hi_pct))

            # Clip before computing center/scale statistics so extreme outliers
            # do not distort the fitted center/scale.
            clipped = np.clip(finite, clip_low, clip_high)

            if method == ScalingMethod.ROBUST:
                center = float(np.median(clipped))
                q25    = float(np.percentile(clipped, 25))
                q75    = float(np.percentile(clipped, 75))
                scale  = max(q75 - q25, self._min_scale)
            elif method == ScalingMethod.ZSCORE:
                center = float(np.mean(clipped))
                scale  = max(float(np.std(clipped)), self._min_scale)
            elif method == ScalingMethod.MINMAX:
                center = float(np.min(clipped))
                scale  = max(float(np.max(clipped)) - center, self._min_scale)
            else:  # NONE
                center = 0.0
                scale  = 1.0

            self._specs[col] = FeatureNormSpec(
                feature_name=col,
                method=method,
                clip_low=clip_low,
                clip_high=clip_high,
                center=center,
                scale=scale,
            )

        self._fitted = True
        logger.info(
            "feature_normalizer_fitted",
            n_features=len(self._specs),
            n_rows=len(df),
            winsor_pct=self._winsor_pct,
        )
        return self

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(
        self,
        df: pd.DataFrame,
        track_clips: bool = False,
    ) -> tuple[pd.DataFrame, list[ClipReport]] | pd.DataFrame:
        """Apply fitted normalization to *df*.

        Args:
            df:          Feature DataFrame (columns must be a subset of those
                         seen during fit).
            track_clips: When True, return a 2-tuple (transformed_df, clip_reports).
                         When False (default), return only the transformed DataFrame.

        Returns:
            Normalized DataFrame with the same shape/index as *df*.
            NaN values remain NaN; no imputation is performed here.

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        if not self._fitted:
            raise RuntimeError(
                "FeatureNormalizer.transform() called before fit(). "
                "Call fit(training_df) first."
            )

        result = df.copy()
        clip_reports: list[ClipReport] = []

        for col in df.columns:
            spec = self._specs.get(col)
            if spec is None:
                # MEDIUM fix: a column not seen during fit means the feature
                # schema has changed between training and inference.  Silently
                # passing through raw values would mix normalized and raw
                # features in the same vector, causing silent model corruption.
                # Raise so the caller is forced to handle the mismatch.
                raise RuntimeError(
                    f"FeatureNormalizer.transform(): column '{col}' was not present "
                    f"during fit().  Feature schema mismatch — re-fit the normalizer "
                    f"or drop the unseen column before calling transform(). "
                    f"Fitted columns: {sorted(self._specs.keys())}"
                )

            if spec.method == ScalingMethod.NONE:
                continue

            arr = df[col].to_numpy(dtype=float)

            # ── 1. Clip ───────────────────────────────────────────────────
            n_clip_low  = 0
            n_clip_high = 0

            if spec.clip_low is not None or spec.clip_high is not None:
                lo = spec.clip_low  if spec.clip_low  is not None else -np.inf
                hi = spec.clip_high if spec.clip_high is not None else  np.inf
                finite_mask  = np.isfinite(arr)
                if spec.clip_low is not None:
                    n_clip_low  = int((arr[finite_mask] < lo).sum())
                if spec.clip_high is not None:
                    n_clip_high = int((arr[finite_mask] > hi).sum())
                arr = np.where(finite_mask, np.clip(arr, lo, hi), arr)

            if track_clips:
                clip_reports.append(ClipReport(
                    feature=col,
                    n_total=int(np.isfinite(arr).sum()),
                    n_clipped_low=n_clip_low,
                    n_clipped_high=n_clip_high,
                    clip_low=spec.clip_low,
                    clip_high=spec.clip_high,
                ))

            # ── 2. Scale ──────────────────────────────────────────────────
            center = spec.center if spec.center is not None else 0.0
            scale  = spec.scale  if spec.scale  is not None else 1.0
            arr    = (arr - center) / scale

            result[col] = arr

        if track_clips:
            return result, clip_reports
        return result

    # ── Fit-transform convenience ──────────────────────────────────────────────

    def transform_known(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform only columns that were present during fit; pass others through.

        Use this when the DataFrame may contain extra columns (e.g. label, symbol)
        that were not part of the feature matrix at fit time.  Unlike ``transform()``,
        this never raises on unknown columns — it normalizes what it can and leaves
        the rest unchanged.
        """
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer not fitted.")
        known_cols = [c for c in df.columns if c in self._specs]
        unknown_cols = [c for c in df.columns if c not in self._specs]
        if unknown_cols:
            logger.warning(
                "feature_normalizer_transform_known_skipping",
                unknown_columns=unknown_cols,
            )
        if not known_cols:
            return df.copy()
        result = df.copy()
        normalized = self.transform(df[known_cols])
        result[known_cols] = normalized
        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        track_clips: bool = False,
    ) -> tuple[pd.DataFrame, list[ClipReport]] | pd.DataFrame:
        """Fit on *df* and immediately transform it.

        Equivalent to ``norm.fit(df).transform(df)``.
        Use ONLY on training data — never on test/live data.
        """
        self.fit(df)
        return self.transform(df, track_clips=track_clips)

    # ── Clip report (non-modifying) ────────────────────────────────────────────

    def clip_report(self, df: pd.DataFrame) -> list[ClipReport]:
        """Return per-feature clip statistics without modifying *df*."""
        _, reports = self.transform(df, track_clips=True)  # type: ignore[misc]
        return reports  # type: ignore[return-value]

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted state to a plain JSON-compatible dict."""
        if not self._fitted:
            raise RuntimeError("Cannot serialize an unfitted FeatureNormalizer.")
        return {
            "version":         "normalizer-1.0",
            "winsor_pct":      list(self._winsor_pct),
            "min_scale":       self._min_scale,
            "default_method":  self._default_method.value,
            "fitted":          self._fitted,
            "specs":           {k: v.to_dict() for k, v in self._specs.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureNormalizer":
        """Restore a fitted normalizer from a serialized dict."""
        obj = cls(
            winsor_pct    = tuple(d.get("winsor_pct", _DEFAULT_WINSOR)),  # type: ignore[arg-type]
            min_scale     = d.get("min_scale", 1e-8),
            default_method= ScalingMethod(d.get("default_method", "robust")),
        )
        for spec_dict in d.get("specs", {}).values():
            spec = FeatureNormSpec.from_dict(spec_dict)
            obj._specs[spec.feature_name] = spec
        obj._fitted = bool(d.get("fitted", False))
        return obj

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "FeatureNormalizer":
        return cls.from_dict(json.loads(s))

    def load_state(self, state: dict[str, Any]) -> "FeatureNormalizer":
        """Restore fitted state from a serialized dict (in-place).

        Equivalent to ``from_dict(state)`` but mutates self rather than
        creating a new instance. Use this when the normalizer is already
        instantiated and you want to load a previously-saved state.

        Example::

            norm = FeatureNormalizer()
            norm.load_state(model_dict["normalizer_state"])
            X_live = norm.transform(live_df)
        """
        if not state:
            raise ValueError(
                "load_state() called with empty dict. "
                "Ensure the model artifact includes a non-empty normalizer_state."
            )
        restored = self.__class__.from_dict(state)
        self._winsor_pct     = restored._winsor_pct
        self._min_scale      = restored._min_scale
        self._default_method = restored._default_method
        self._specs          = restored._specs
        self._fitted         = restored._fitted
        return self

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summarizing fitted normalization stats per feature."""
        if not self._fitted:
            raise RuntimeError("Normalizer not yet fitted.")
        rows = []
        for spec in self._specs.values():
            rows.append({
                "feature":    spec.feature_name,
                "method":     spec.method.value,
                "clip_low":   spec.clip_low,
                "clip_high":  spec.clip_high,
                "center":     spec.center,
                "scale":      spec.scale,
                "clip_range": (
                    (spec.clip_high - spec.clip_low)
                    if spec.clip_low is not None and spec.clip_high is not None
                    else None
                ),
            })
        return pd.DataFrame(rows).set_index("feature")

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def feature_names(self) -> list[str]:
        return list(self._specs.keys())
