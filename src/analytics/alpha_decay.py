"""
src/analytics/alpha_decay.py — Alpha Decay Detection

FIX NEW-P1-007: The mandate (§36) requires per-alpha health scores with rolling
IC decay detection. This module provides:

  1. AlphaHealthTracker — maintains rolling IC history per alpha
  2. AlphaDecayDetector — detects decay by comparing rolling IC to a peak/baseline
  3. AlphaHalfLifeEstimator — estimates the half-life of an alpha's IC decay

Decay triggers:
  - Rolling IC drops > 50% from peak sustained over min_decay_bars bars → DEGRADE
  - Rolling IC drops below zero for min_decay_bars consecutive bars → QUARANTINE
  - IC trend (linear fit slope) is significantly negative → WARN

Integration with SelfLearningLoop:
  When AlphaDecayDetector returns DEGRADE or QUARANTINE, it should trigger
  challenger training via SelfLearningLoop.

Usage::

    tracker = AlphaHealthTracker("MOMENTUM_ALPHA")
    tracker.add_ic(0.042)
    tracker.add_ic(0.038)
    tracker.add_ic(0.012)  # decay
    detector = AlphaDecayDetector()
    state = detector.evaluate(tracker)
    # state.health_state → "HEALTHY" | "WARN" | "DEGRADE" | "QUARANTINE"
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)


class AlphaHealthState(str, Enum):
    """Health states for an alpha specialist."""
    HEALTHY = "HEALTHY"         # IC stable and positive
    WARN = "WARN"               # IC declining but not yet degraded
    DEGRADE = "DEGRADE"         # IC has lost > 50% of peak
    QUARANTINE = "QUARANTINE"   # IC consistently below zero
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # not enough observations yet


@dataclass
class AlphaHealthRecord:
    """A point-in-time IC observation for one alpha."""
    timestamp: str
    ic: float
    regime: str = "UNKNOWN"


@dataclass
class AlphaDecayResult:
    """Result of one decay evaluation."""
    alpha_name: str
    health_state: AlphaHealthState
    rolling_ic_mean: float
    peak_ic: float
    ic_decay_fraction: float   # (peak_ic - rolling_ic) / |peak_ic|; positive = decay
    consecutive_negative: int
    ic_trend_slope: float      # per-observation slope of linear IC trend
    reason: str
    recommended_action: str    # MONITOR | WARN_OPERATOR | TRAIN_CHALLENGER | QUARANTINE
    evaluated_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    n_observations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_name": self.alpha_name,
            "health_state": self.health_state.value,
            "rolling_ic_mean": round(self.rolling_ic_mean, 6),
            "peak_ic": round(self.peak_ic, 6),
            "ic_decay_fraction": round(self.ic_decay_fraction, 4),
            "consecutive_negative": self.consecutive_negative,
            "ic_trend_slope": round(self.ic_trend_slope, 8),
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "evaluated_at": self.evaluated_at,
            "n_observations": self.n_observations,
        }

    @property
    def should_train_challenger(self) -> bool:
        return self.recommended_action in ("TRAIN_CHALLENGER",)

    @property
    def should_quarantine(self) -> bool:
        return self.health_state == AlphaHealthState.QUARANTINE


class AlphaHealthTracker:
    """
    Maintains a rolling history of IC observations for one alpha specialist.

    Observations should be added at each model evaluation cycle (daily or
    per walk-forward window). The tracker maintains a rolling window of the
    most recent `window_size` observations.

    Usage::

        tracker = AlphaHealthTracker("MOMENTUM_ALPHA", window_size=60)
        tracker.add_ic(ic=0.042, regime="TREND_UP")
        tracker.add_ic(ic=0.038)
        history = tracker.recent_ic(20)
    """

    def __init__(
        self,
        alpha_name: str,
        window_size: int = 60,
    ) -> None:
        self.alpha_name = alpha_name
        self.window_size = window_size
        self._history: deque[AlphaHealthRecord] = deque(maxlen=window_size)
        self._peak_ic: float = 0.0  # running peak of rolling IC means
        self._all_time_max_ic: float = -np.inf

    def add_ic(self, ic: float, regime: str = "UNKNOWN") -> None:
        """Add one IC observation."""
        if not math.isfinite(ic):
            return
        record = AlphaHealthRecord(
            timestamp=datetime.now(tz=UTC).isoformat(),
            ic=ic,
            regime=regime,
        )
        self._history.append(record)
        if ic > self._all_time_max_ic:
            self._all_time_max_ic = ic

    def recent_ic(self, n: int | None = None) -> list[float]:
        """Return the most recent n IC values (or all if n is None)."""
        history = list(self._history)
        if n is not None:
            history = history[-n:]
        return [r.ic for r in history]

    def rolling_ic_mean(self, window: int | None = None) -> float:
        """Mean IC over the most recent `window` observations."""
        ics = self.recent_ic(window)
        if not ics:
            return 0.0
        return float(np.mean(ics))

    def update_peak(self, window: int = 20) -> None:
        """Update the peak IC across the full observation history.

        Computes the rolling mean at every position in the history and
        keeps the maximum — this ensures the peak captures the best sustained
        performance period, not just the most recent window.
        """
        all_ics_records = list(self._history)
        if not all_ics_records:
            return
        for end in range(1, len(all_ics_records) + 1):
            w = min(window, end)
            seg = [r.ic for r in all_ics_records[end - w : end]]
            rolling = float(np.mean(seg)) if seg else 0.0
            if rolling > self._peak_ic:
                self._peak_ic = rolling

    @property
    def peak_ic(self) -> float:
        return self._peak_ic

    @property
    def n_observations(self) -> int:
        return len(self._history)


class AlphaDecayDetector:
    """
    Evaluates alpha health state and recommends actions based on IC decay.

    Decay thresholds (all configurable):
      decay_threshold  : IC decay fraction above which DEGRADE is triggered (default 0.50)
      negative_bars    : consecutive negative IC bars to trigger QUARANTINE (default 5)
      min_observations : minimum IC observations before evaluation (default 10)

    Recommended action mapping:
      HEALTHY              → MONITOR
      WARN                 → WARN_OPERATOR
      DEGRADE              → TRAIN_CHALLENGER
      QUARANTINE           → QUARANTINE (stop signals, train challenger)
    """

    def __init__(
        self,
        decay_threshold: float = 0.50,
        negative_bars: int = 5,
        min_observations: int = 10,
        warn_slope_threshold: float = -0.002,
    ) -> None:
        self.decay_threshold = decay_threshold
        self.negative_bars = negative_bars
        self.min_observations = min_observations
        self.warn_slope_threshold = warn_slope_threshold

    def evaluate(self, tracker: AlphaHealthTracker) -> AlphaDecayResult:
        """
        Evaluate the health state of an alpha based on its IC history.

        Algorithm:
        1. If insufficient data → INSUFFICIENT_DATA
        2. Compute rolling IC mean (recent 20 observations)
        3. Update peak (most recent rolling high-water mark)
        4. Count consecutive negative IC bars
        5. Compute IC trend slope (linear regression of IC over time)
        6. Apply rules:
           a. consecutive_negative >= negative_bars → QUARANTINE
           b. ic_decay_fraction >= decay_threshold → DEGRADE
           c. ic_trend_slope < warn_slope_threshold → WARN
           d. Otherwise → HEALTHY
        """
        alpha_name = tracker.alpha_name
        n = tracker.n_observations

        if n < self.min_observations:
            return AlphaDecayResult(
                alpha_name=alpha_name,
                health_state=AlphaHealthState.INSUFFICIENT_DATA,
                rolling_ic_mean=0.0,
                peak_ic=tracker.peak_ic,
                ic_decay_fraction=0.0,
                consecutive_negative=0,
                ic_trend_slope=0.0,
                reason=f"Only {n} observations (min={self.min_observations})",
                recommended_action="MONITOR",
                n_observations=n,
            )

        tracker.update_peak(window=min(20, n))
        rolling_mean = tracker.rolling_ic_mean(window=min(20, n))
        peak = tracker.peak_ic

        # (1) Consecutive negative IC
        recent = tracker.recent_ic(self.negative_bars)
        consecutive_negative = sum(1 for ic in recent if ic < 0)

        # (2) Decay fraction
        if abs(peak) > 1e-12:
            ic_decay_fraction = (peak - rolling_mean) / abs(peak)
        else:
            ic_decay_fraction = 0.0

        # (3) Trend slope via linear regression
        all_ics = tracker.recent_ic()
        if len(all_ics) >= 3:
            x = np.arange(len(all_ics), dtype=float)
            x_c = x - x.mean()
            y = np.array(all_ics)
            slope = float(np.dot(x_c, y) / (np.dot(x_c, x_c) + 1e-12))
        else:
            slope = 0.0

        # Apply rules
        if consecutive_negative >= self.negative_bars:
            state = AlphaHealthState.QUARANTINE
            reason = (
                f"IC negative for {consecutive_negative} consecutive observations "
                f"(threshold={self.negative_bars})"
            )
            action = "QUARANTINE"
        elif ic_decay_fraction >= self.decay_threshold:
            state = AlphaHealthState.DEGRADE
            reason = (
                f"IC decayed {ic_decay_fraction:.1%} from peak "
                f"(peak={peak:.4f}, current={rolling_mean:.4f}, "
                f"threshold={self.decay_threshold:.1%})"
            )
            action = "TRAIN_CHALLENGER"
        elif slope < self.warn_slope_threshold:
            state = AlphaHealthState.WARN
            reason = (
                f"IC trend slope is negative ({slope:.6f} < {self.warn_slope_threshold})"
            )
            action = "WARN_OPERATOR"
        else:
            state = AlphaHealthState.HEALTHY
            reason = (
                f"IC stable: rolling_mean={rolling_mean:.4f}, "
                f"peak={peak:.4f}, decay={ic_decay_fraction:.1%}"
            )
            action = "MONITOR"

        result = AlphaDecayResult(
            alpha_name=alpha_name,
            health_state=state,
            rolling_ic_mean=rolling_mean,
            peak_ic=peak,
            ic_decay_fraction=ic_decay_fraction,
            consecutive_negative=consecutive_negative,
            ic_trend_slope=slope,
            reason=reason,
            recommended_action=action,
            n_observations=n,
        )

        if state in (AlphaHealthState.DEGRADE, AlphaHealthState.QUARANTINE):
            logger.warning(
                "alpha_decay_detected",
                alpha=alpha_name,
                state=state.value,
                rolling_ic=rolling_mean,
                peak_ic=peak,
                decay_fraction=ic_decay_fraction,
                reason=reason,
            )
        else:
            logger.debug(
                "alpha_health_evaluated",
                alpha=alpha_name,
                state=state.value,
                rolling_ic=rolling_mean,
            )

        return result

    def evaluate_all(
        self, trackers: dict[str, AlphaHealthTracker]
    ) -> dict[str, AlphaDecayResult]:
        """Evaluate all alphas and return {alpha_name: result}."""
        return {name: self.evaluate(tracker) for name, tracker in trackers.items()}


class AlphaHalfLifeEstimator:
    """
    Estimates the half-life of IC decay using exponential fit.

    IC(t) ≈ IC_0 × exp(-λt), where λ = ln(2) / half_life.

    A shorter half-life means the alpha decays faster and needs more
    frequent retraining or position sizing reduction.

    Usage::

        estimator = AlphaHalfLifeEstimator()
        half_life_days = estimator.estimate(ic_series)
    """

    def estimate(self, ic_series: list[float]) -> float | None:
        """
        Estimate the IC half-life in observation units.

        Returns None when there is insufficient data or IC is non-positive.
        """
        ics = np.array([ic for ic in ic_series if math.isfinite(ic)])
        if len(ics) < 10:
            return None
        if np.all(ics <= 0):
            return None  # fully decayed — cannot estimate half-life

        # Autocorrelation decay: estimate AR(1) coefficient
        ic_c = ics - np.mean(ics)
        if np.sum(ic_c ** 2) < 1e-12:
            return None
        rho = float(np.dot(ic_c[:-1], ic_c[1:]) / np.sum(ic_c[:-1] ** 2))
        rho = max(-0.999, min(0.999, rho))

        if rho <= 0:
            return 1.0  # essentially white noise (half-life ≈ 1 period)

        # half_life = -ln(2) / ln(rho)
        half_life = -math.log(2) / math.log(rho)
        return max(1.0, round(half_life, 2))
