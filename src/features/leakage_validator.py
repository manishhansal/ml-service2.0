"""
LeakageValidator — forward-looking correlation detector.

Scans a feature matrix for look-ahead bias by computing the Pearson correlation
between each feature and future returns over look-ahead windows of 1 to 22
trading days.

Design contract (Req 2.5):
- Raises PITViolationError when |correlation| > LEAK_THRESHOLD (0.05) for any feature
  and any look-ahead window.
- Logs the feature name, look-ahead window, and correlation value on every violation.
- Skips correlation computation when fewer than MIN_ALIGNED_SAMPLES (30) aligned
  non-NaN rows exist for a given window.
- Returns None on clean validation (no violations).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

LEAK_THRESHOLD: float = 0.05          # |correlation| must not exceed this (Req 2.5)
MIN_ALIGNED_SAMPLES: int = 30         # skip if fewer aligned samples
MIN_LOOK_AHEAD_DAYS: int = 1          # minimum look-ahead window (days)
MAX_LOOK_AHEAD_DAYS: int = 22         # maximum look-ahead window (22 trading days ≈ 1 month)


# ── Exception ────────────────────────────────────────────────────────────────

class PITViolationError(Exception):
    """Raised when a feature exhibits a forward-looking correlation above the threshold.

    Attributes:
        feature_name:    Name of the leaky feature.
        look_ahead_days: The look-ahead window in trading days.
        correlation:     The actual Pearson correlation value.
    """

    def __init__(
        self,
        feature_name: str,
        look_ahead_days: int,
        correlation: float,
    ) -> None:
        self.feature_name = feature_name
        self.look_ahead_days = look_ahead_days
        self.correlation = correlation
        super().__init__(
            f"PIT violation: feature '{feature_name}' has look-ahead correlation "
            f"{correlation:.4f} (|r|={abs(correlation):.4f} > threshold={LEAK_THRESHOLD}) "
            f"at window={look_ahead_days} trading days"
        )


# ── Validator ─────────────────────────────────────────────────────────────────

class LeakageValidator:
    """
    Validates that no feature in a training dataset has forward-looking correlation
    with future returns above the LEAK_THRESHOLD.

    Usage::

        validator = LeakageValidator()
        # Raises PITViolationError if leakage is detected; returns None if clean.
        validator.validate(feature_matrix, label_vector)
    """

    def __init__(
        self,
        threshold: float = LEAK_THRESHOLD,
        min_aligned_samples: int = MIN_ALIGNED_SAMPLES,
        look_ahead_range: tuple[int, int] = (MIN_LOOK_AHEAD_DAYS, MAX_LOOK_AHEAD_DAYS),
    ) -> None:
        """
        Args:
            threshold:           Maximum allowed |Pearson correlation| between any feature
                                 and future returns. Default: 0.05.
            min_aligned_samples: Minimum number of non-NaN aligned rows required to
                                 compute the correlation. Skip the window if below this.
                                 Default: 30.
            look_ahead_range:    (min_days, max_days) look-ahead window range (inclusive).
                                 Default: (1, 22).
        """
        self.threshold = threshold
        self.min_aligned_samples = min_aligned_samples
        self.min_window, self.max_window = look_ahead_range

    def validate(
        self,
        feature_matrix: pd.DataFrame,
        label_vector: pd.Series,
    ) -> None:
        """
        Check all features in feature_matrix for forward-looking correlation with label_vector.

        Algorithm:
        1. For each look-ahead window w from min_window to max_window:
           a. Shift label_vector by -w (so label_vector.shift(-w)[t] = return at t+w)
           b. For each feature column in feature_matrix:
              i. Align feature and shifted label on non-NaN rows
              ii. If aligned row count < min_aligned_samples: skip this window/feature pair
              iii. Compute Pearson correlation between feature and shifted label
              iv. If |correlation| > threshold: log + raise PITViolationError
        2. Return None if no violations found.

        Args:
            feature_matrix: DataFrame where each column is a feature. Index should be
                            a time series (DatetimeIndex or integer index).
            label_vector:   Series of realized returns (or any label).

        Raises:
            PITViolationError: On the first feature/window pair that exceeds the threshold.

        Returns:
            None if no leakage is detected.
        """
        if feature_matrix.empty or label_vector.empty:
            logger.info("leakage_validator_skipped_empty_input")
            return None

        feature_names = list(feature_matrix.columns)
        n_features = len(feature_names)

        logger.info(
            "leakage_validation_started",
            n_features=n_features,
            n_rows=len(feature_matrix),
            look_ahead_range=f"{self.min_window}-{self.max_window}",
            threshold=self.threshold,
        )

        for window in range(self.min_window, self.max_window + 1):
            # Shift the label vector to represent future returns
            future_returns = label_vector.shift(-window)

            for feature_name in feature_names:
                feature_series = feature_matrix[feature_name]

                # Align on non-NaN values
                aligned = pd.concat(
                    [feature_series, future_returns],
                    axis=1,
                    keys=["feature", "future_return"],
                ).dropna()

                if len(aligned) < self.min_aligned_samples:
                    # Not enough data to compute a reliable correlation — skip
                    continue

                # Compute Pearson correlation
                feature_vals = aligned["feature"].to_numpy(dtype=float)
                label_vals = aligned["future_return"].to_numpy(dtype=float)

                # Guard against zero-variance features (constant series → correlation undefined)
                if np.std(feature_vals) < 1e-10 or np.std(label_vals) < 1e-10:
                    continue

                correlation = float(np.corrcoef(feature_vals, label_vals)[0, 1])

                if not math.isfinite(correlation):
                    continue

                abs_corr = abs(correlation)

                if abs_corr > self.threshold:
                    logger.warning(
                        "pit_violation_detected",
                        feature_name=feature_name,
                        look_ahead_days=window,
                        correlation=round(correlation, 6),
                        abs_correlation=round(abs_corr, 6),
                        threshold=self.threshold,
                    )
                    raise PITViolationError(
                        feature_name=feature_name,
                        look_ahead_days=window,
                        correlation=correlation,
                    )

        logger.info(
            "leakage_validation_passed",
            n_features=n_features,
            n_windows_checked=self.max_window - self.min_window + 1,
        )
        return None


# ── LookAheadGuard — Phase 3 implementation ────────────────────────────────────

from datetime import datetime
from src.core.exceptions import PointInTimeViolationError  # noqa: E402


class LookAheadGuard:
    """
    Timestamp-level Point-In-Time guard for feature vector assembly.

    Enforces the strict rule: every source datum used to build a feature
    vector at time T must have a ``source_ts`` that is STRICTLY LESS THAN
    the ``pit_boundary``.  Any datum with ``source_ts >= pit_boundary``
    constitutes look-ahead bias and must be rejected.

    Usage::

        guard = LookAheadGuard()
        # Clean: source_ts < pit_boundary → returns None
        guard.check("close", past_ts, pit_boundary)

        # Violation: source_ts >= pit_boundary → raises PointInTimeViolationError
        guard.check("india_vix", future_ts, pit_boundary)

    Requirements: Req 2.6, Req 2.7
    """

    def check(
        self,
        feature_name: str,
        source_ts: datetime,
        pit_boundary: datetime,
    ) -> None:
        """
        Assert that *source_ts* is strictly before *pit_boundary*.

        Args:
            feature_name:  Name of the feature or data field being checked.
                           Included in the exception message for diagnostics.
            source_ts:     UTC timestamp of the source data point.
            pit_boundary:  The PIT boundary — the timestamp at which the
                           feature vector is being assembled.  All source
                           data must predate this point.

        Returns:
            ``None`` when ``source_ts < pit_boundary`` (no violation).

        Raises:
            PointInTimeViolationError: When ``source_ts >= pit_boundary``.
                The exception carries ``feature_name``, ``source_ts_iso``,
                and ``pit_boundary_iso`` for structured audit logging.
        """
        # Normalise both timestamps to the same tzinfo representation so
        # comparison is always valid regardless of tz-aware/naive mismatch.
        # If one is tz-aware and the other naive we use total_seconds() delta.
        try:
            violation = source_ts >= pit_boundary
        except TypeError:
            # Mixed aware/naive — convert both to UTC epoch for comparison
            import calendar

            def _to_epoch(dt: datetime) -> float:
                if dt.tzinfo is not None:
                    return dt.timestamp()
                return calendar.timegm(dt.timetuple()) + dt.microsecond / 1e6

            violation = _to_epoch(source_ts) >= _to_epoch(pit_boundary)

        if violation:
            raise PointInTimeViolationError(
                feature_name=feature_name,
                source_ts_iso=source_ts.isoformat(),
                pit_boundary_iso=pit_boundary.isoformat(),
            )

        return None
