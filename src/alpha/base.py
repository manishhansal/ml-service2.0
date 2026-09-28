"""
src/alpha/base.py — AlphaSpecialist base class and registry

FIX NEW-P2-001: The mandate (§12) requires independent alpha specialists, each
responsible for one economic hypothesis. Every specialist:

  1. Has a single well-defined alpha hypothesis (e.g. "momentum persists")
  2. Maintains its own feature set aligned to that hypothesis
  3. Produces a standardized AlphaSignal output
  4. Tracks its own IC history via the RegimeAlphaMatrix
  5. Can be independently enabled/disabled/quarantined

The MetaDecisionEngine aggregates signals from all active specialists.
This replaces the single-model-does-everything architecture.

Mandate §12 specialist list:
  TREND_ALPHA, MEAN_REVERSION_ALPHA, BREAKOUT_ALPHA, MOMENTUM_ALPHA,
  VOLATILITY_ALPHA, EVENT_ALPHA, NEWS_ALPHA, RELATIVE_VALUE_ALPHA,
  MICROSTRUCTURE_ALPHA, MARKET_REGIME_ALPHA
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)


class AlphaStatus(str, Enum):
    """Lifecycle state of an alpha specialist."""
    ACTIVE = "ACTIVE"
    SHADOW = "SHADOW"        # running but not used in live signals
    DEGRADED = "DEGRADED"    # IC decaying; use with reduced weight
    QUARANTINED = "QUARANTINED"  # disabled; not used
    RETIRED = "RETIRED"      # permanently disabled


@dataclass
class AlphaSignal:
    """
    Standardised output from one alpha specialist (mandate §12).

    Every field has economic meaning — there are no raw model scores here.
    """
    alpha_name: str
    direction: int                   # +1 long, -1 short, 0 flat/no-trade
    expected_return: float           # expected gross return (fraction)
    confidence: float                # probability estimate in [0, 1]
    expected_holding_period: int     # bars
    risk_estimate: float             # expected volatility of the trade
    alpha_score: float               # composite score for ranking [-1, 1]
    regime_fit: float                # how well current regime suits this alpha [0, 1]
    feature_stability: float         # stability of input features [0, 1]
    reason: str                      # brief human-readable rationale
    timestamp: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    symbol: str = ""
    regime: str = "UNKNOWN"

    # Optional: IC metadata when backed by a trained model
    model_ic: float | None = None
    model_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_name": self.alpha_name,
            "direction": self.direction,
            "expected_return": round(self.expected_return, 6),
            "confidence": round(self.confidence, 4),
            "expected_holding_period": self.expected_holding_period,
            "risk_estimate": round(self.risk_estimate, 6),
            "alpha_score": round(self.alpha_score, 4),
            "regime_fit": round(self.regime_fit, 4),
            "feature_stability": round(self.feature_stability, 4),
            "reason": self.reason,
            "symbol": self.symbol,
            "regime": self.regime,
            "timestamp": self.timestamp,
        }

    @property
    def is_directional(self) -> bool:
        return self.direction != 0

    @property
    def expected_net_return(self) -> float:
        """Gross expected return minus a 28bps round-trip cost assumption."""
        cost = 28.0 / 10_000.0
        return self.expected_return - cost

    @property
    def has_positive_edge(self) -> bool:
        return self.expected_net_return > 0 and self.direction != 0


class AlphaSpecialist(ABC):
    """
    Abstract base for all alpha specialists.

    Subclasses implement:
      generate(features, symbol, regime) → AlphaSignal | None

    The generate() method must be:
      - STATELESS with respect to the call (no side effects)
      - DETERMINISTIC for the same inputs
      - FAST (< 10ms for a single symbol)
      - PIT-SAFE (must not use future information)
    """

    name: str = "BASE_ALPHA"
    default_holding_period: int = 5   # bars
    status: AlphaStatus = AlphaStatus.ACTIVE

    @abstractmethod
    def generate(
        self,
        features: pd.Series | dict[str, float],
        symbol: str = "",
        regime: str = "UNKNOWN",
    ) -> AlphaSignal | None:
        """
        Generate an alpha signal from a feature vector.

        Args:
            features : Feature vector (one bar's worth of features).
            symbol   : Instrument identifier.
            regime   : Current market regime label.

        Returns:
            AlphaSignal if a trade opportunity exists, None otherwise.
            Returning None is the CORRECT behavior when there is no edge.
        """
        ...

    def is_active(self) -> bool:
        return self.status in (AlphaStatus.ACTIVE, AlphaStatus.SHADOW)

    def regime_compatibility(self, regime: str) -> float:
        """
        How compatible is this alpha with the given regime? [0, 1].

        Default: 1.0 (compatible with all regimes).
        Subclasses override to be regime-selective.
        """
        return 1.0

    def _make_signal(
        self,
        direction: int,
        expected_return: float,
        confidence: float,
        regime: str,
        symbol: str,
        reason: str,
        regime_fit: float | None = None,
        feature_stability: float = 1.0,
    ) -> AlphaSignal:
        """Helper to construct an AlphaSignal with computed fields."""
        rf = regime_fit if regime_fit is not None else self.regime_compatibility(regime)
        # Risk estimate: simple volatility proxy = expected_return / 2 (RR=2)
        risk_est = abs(expected_return) / 2.0 if abs(expected_return) > 0 else 0.01

        # Alpha score: signed confidence × regime_fit × feature_stability
        alpha_score = float(direction) * confidence * rf * feature_stability

        return AlphaSignal(
            alpha_name=self.name,
            direction=direction,
            expected_return=expected_return,
            confidence=max(0.0, min(1.0, confidence)),
            expected_holding_period=self.default_holding_period,
            risk_estimate=risk_est,
            alpha_score=max(-1.0, min(1.0, alpha_score)),
            regime_fit=rf,
            feature_stability=feature_stability,
            reason=reason,
            symbol=symbol,
            regime=regime,
        )


# ── Concrete alpha specialists ────────────────────────────────────────────────

class MomentumAlpha(AlphaSpecialist):
    """
    MOMENTUM_ALPHA: Price momentum persists over 5–20 bar horizons.

    Hypothesis: stocks with strong recent returns continue to outperform
    over a short forward window. Uses multi-period return features.

    Best regimes: TREND_UP, TREND_DOWN, BREAKOUT
    Worst regimes: MEAN_REVERTING, RANGE, PANIC
    """

    name = "MOMENTUM_ALPHA"
    default_holding_period = 5

    GOOD_REGIMES = frozenset({"TREND_UP", "TREND_DOWN", "BREAKOUT"})
    BAD_REGIMES = frozenset({"MEAN_REVERTING", "RANGE", "PANIC", "LIQUIDITY_STRESS"})

    def regime_compatibility(self, regime: str) -> float:
        r = regime.upper()
        if r in self.GOOD_REGIMES:
            return 0.9
        if r in self.BAD_REGIMES:
            return 0.1
        return 0.5  # UNKNOWN / UNCERTAIN

    def generate(
        self,
        features: pd.Series | dict[str, float],
        symbol: str = "",
        regime: str = "UNKNOWN",
    ) -> AlphaSignal | None:
        f = features if isinstance(features, dict) else features.to_dict()

        # Key momentum features
        ret_5 = float(f.get("ret_5", 0.0) or 0.0)
        ret_20 = float(f.get("ret_20", 0.0) or 0.0)
        ret_60 = float(f.get("ret_60", 0.0) or 0.0)
        trend_dir = float(f.get("trend_direction", 0.0) or 0.0)
        trend_str = float(f.get("trend_strength", 0.0) or 0.0)
        vol_ratio = float(f.get("vol_ratio", 1.0) or 1.0)

        # Check for NaN / missing data
        if any(math.isnan(v) for v in [ret_5, ret_20]):
            return None

        # Regime filter
        regime_fit = self.regime_compatibility(regime)
        if regime_fit < 0.2:
            return None  # this alpha is not appropriate for this regime

        # Signal logic: strong 5-day + 20-day momentum alignment
        mom_score = 0.4 * ret_5 + 0.4 * ret_20 + 0.2 * ret_60
        trend_boost = 1.0 + 0.3 * (trend_str / 50.0) * abs(trend_dir)

        if abs(mom_score) < 0.005:  # below minimum threshold
            return None

        direction = int(np.sign(mom_score))
        raw_confidence = min(0.85, 0.50 + abs(mom_score) * 10.0)
        expected_return = abs(mom_score) * trend_boost * 0.5  # conservative scaling

        # Penalise in volatile regimes (momentum is less reliable)
        if vol_ratio > 1.5:
            expected_return *= 0.7
            raw_confidence *= 0.85

        feature_stability = 1.0 if not math.isnan(ret_60) else 0.7

        return self._make_signal(
            direction=direction,
            expected_return=expected_return,
            confidence=raw_confidence,
            regime=regime,
            symbol=symbol,
            reason=(
                f"mom_score={mom_score:.4f} "
                f"ret5={ret_5:.4f} ret20={ret_20:.4f} "
                f"trend_str={trend_str:.1f}"
            ),
            regime_fit=regime_fit,
            feature_stability=feature_stability,
        )


class MeanReversionAlpha(AlphaSpecialist):
    """
    MEAN_REVERSION_ALPHA: Overextended prices revert to their mean.

    Hypothesis: stocks that have moved significantly from their rolling mean
    tend to revert. Uses Bollinger band z-score and RSI.

    Best regimes: RANGE, MEAN_REVERTING, LOW_VOLATILITY
    Worst regimes: TREND_UP, TREND_DOWN, BREAKOUT, PANIC
    """

    name = "MEAN_REVERSION_ALPHA"
    default_holding_period = 3

    GOOD_REGIMES = frozenset({"RANGE", "MEAN_REVERTING", "LOW_VOLATILITY"})
    BAD_REGIMES = frozenset({"TREND_UP", "TREND_DOWN", "BREAKOUT", "PANIC"})

    def regime_compatibility(self, regime: str) -> float:
        r = regime.upper()
        if r in self.GOOD_REGIMES:
            return 0.9
        if r in self.BAD_REGIMES:
            return 0.1
        return 0.4

    def generate(
        self,
        features: pd.Series | dict[str, float],
        symbol: str = "",
        regime: str = "UNKNOWN",
    ) -> AlphaSignal | None:
        f = features if isinstance(features, dict) else features.to_dict()

        bb_zscore = float(f.get("bb_zscore_20", 0.0) or 0.0)
        rsi = float(f.get("rsi_14", 50.0) or 50.0)
        price_zscore_20 = float(f.get("price_zscore_20", 0.0) or 0.0)
        vol_regime = float(f.get("vol_regime_zscore", 0.0) or 0.0)

        if any(math.isnan(v) for v in [bb_zscore, rsi]):
            return None

        regime_fit = self.regime_compatibility(regime)
        if regime_fit < 0.2:
            return None

        # In high-volatility regime, mean reversion is unreliable
        if vol_regime > 2.0:
            return None

        # Mean reversion signal: price significantly above/below BB band
        # and RSI confirms (overbought/oversold)
        if abs(bb_zscore) < 1.5:  # not sufficiently extended
            return None

        # Direction: fade the move (opposite to displacement)
        direction = -int(np.sign(bb_zscore))

        # Strength proportional to displacement
        displacement = abs(bb_zscore)
        rsi_confirmation = abs(rsi - 50.0) / 50.0  # 0=neutral, 1=extreme

        # Both indicators must agree
        rsi_direction = 1 if rsi < 50 else -1  # oversold → long, overbought → short
        if rsi_direction != direction:
            rsi_confirmation *= 0.3  # conflicting signal, reduce conviction

        raw_confidence = min(0.80, 0.45 + displacement * 0.1 + rsi_confirmation * 0.15)
        expected_return = min(0.025, displacement * 0.005)

        feature_stability = 1.0 if not math.isnan(price_zscore_20) else 0.8

        return self._make_signal(
            direction=direction,
            expected_return=expected_return,
            confidence=raw_confidence,
            regime=regime,
            symbol=symbol,
            reason=(
                f"bb_zscore={bb_zscore:.2f} rsi={rsi:.1f} "
                f"displacement={displacement:.2f}"
            ),
            regime_fit=regime_fit,
            feature_stability=feature_stability,
        )


class BreakoutAlpha(AlphaSpecialist):
    """
    BREAKOUT_ALPHA: Consolidation followed by expansion.

    Hypothesis: when volatility is low (consolidation) and then price breaks
    with high volume, the move tends to continue.

    Best regimes: BREAKOUT, HIGH_VOLATILITY (after low)
    Worst regimes: RANGE (pre-breakout consolidation without trigger)
    """

    name = "BREAKOUT_ALPHA"
    default_holding_period = 7

    GOOD_REGIMES = frozenset({"BREAKOUT", "HIGH_VOLATILITY"})
    BAD_REGIMES = frozenset({"RANGE", "MEAN_REVERTING", "LIQUIDITY_STRESS"})

    def regime_compatibility(self, regime: str) -> float:
        r = regime.upper()
        if r in self.GOOD_REGIMES:
            return 0.85
        if r in self.BAD_REGIMES:
            return 0.15
        return 0.45

    def generate(
        self,
        features: pd.Series | dict[str, float],
        symbol: str = "",
        regime: str = "UNKNOWN",
    ) -> AlphaSignal | None:
        f = features if isinstance(features, dict) else features.to_dict()

        vol_expanding = float(f.get("vol_expanding", 0.0) or 0.0)
        vol_ratio = float(f.get("vol_ratio", 1.0) or 1.0)
        rel_vol = float(f.get("rel_volume_20", 1.0) or 1.0)
        ret_1 = float(f.get("ret_1", 0.0) or 0.0)
        gap_magnitude = float(f.get("gap_magnitude", 0.0) or 0.0)
        ema_spread = float(f.get("ema_spread", 0.0) or 0.0)

        if any(math.isnan(v) for v in [ret_1, rel_vol]):
            return None

        regime_fit = self.regime_compatibility(regime)
        if regime_fit < 0.2:
            return None

        # Breakout conditions:
        # 1. Volatility expanding
        # 2. Volume spike (relative volume > 1.5)
        # 3. Strong single-bar return
        if vol_expanding <= 0:
            return None
        if rel_vol < 1.3:
            return None
        if abs(ret_1) < 0.005:
            return None

        direction = int(np.sign(ret_1))
        vol_confirmation = min(2.0, vol_ratio)
        vol_score = (vol_confirmation - 1.0) / 1.0  # 0 if ratio=1, 1 if ratio=2
        raw_confidence = min(0.78, 0.45 + vol_score * 0.2 + min(rel_vol - 1.3, 1.0) * 0.1)
        expected_return = abs(ret_1) * 0.4 + abs(ema_spread) * 0.3

        return self._make_signal(
            direction=direction,
            expected_return=min(0.03, expected_return),
            confidence=raw_confidence,
            regime=regime,
            symbol=symbol,
            reason=(
                f"vol_expanding={vol_expanding:.0f} "
                f"rel_vol={rel_vol:.2f} ret1={ret_1:.4f}"
            ),
            regime_fit=regime_fit,
        )


class VolatilityAlpha(AlphaSpecialist):
    """
    VOLATILITY_ALPHA: Volatility mean-reversion and regime transitions.

    Hypothesis: realized volatility is mean-reverting at medium horizons.
    High volatility regimes tend to revert; low volatility regimes can expand.

    This alpha predicts VOLATILITY direction, not price direction directly.
    It generates a directional price signal based on the expected vol regime change
    (e.g., vol contraction after a spike often accompanies price stabilisation).

    Best regimes: HIGH_VOLATILITY (bet on contraction), LOW_VOLATILITY (bet on expansion)
    """

    name = "VOLATILITY_ALPHA"
    default_holding_period = 10

    def regime_compatibility(self, regime: str) -> float:
        r = regime.upper()
        if r in ("HIGH_VOLATILITY", "PANIC"):
            return 0.85  # vol contraction signal
        if r in ("LOW_VOLATILITY",):
            return 0.70  # vol expansion signal
        return 0.30

    def generate(
        self,
        features: pd.Series | dict[str, float],
        symbol: str = "",
        regime: str = "UNKNOWN",
    ) -> AlphaSignal | None:
        f = features if isinstance(features, dict) else features.to_dict()

        vol_regime_zscore = float(f.get("vol_regime_zscore", 0.0) or 0.0)
        vol_regime_pctile = float(f.get("vol_regime_pctile", 0.5) or 0.5)
        vol_of_vol = float(f.get("vol_of_vol_20", 0.0) or 0.0)
        vol_20 = float(f.get("vol_20", 0.0) or 0.0)
        parkinson = float(f.get("parkinson_vol", 0.0) or 0.0)

        if any(math.isnan(v) for v in [vol_regime_zscore, vol_regime_pctile]):
            return None

        regime_fit = self.regime_compatibility(regime)
        if regime_fit < 0.3:
            return None

        # Extreme vol regime: bet on mean reversion
        if abs(vol_regime_zscore) < 1.5:
            return None

        # High vol → expect contraction → slight bullish (less uncertainty premium)
        # Low vol → expect expansion → slight bearish (building risk premium)
        direction = -int(np.sign(vol_regime_zscore))  # fade extreme vol

        displacement = abs(vol_regime_zscore) - 1.5
        raw_confidence = min(0.70, 0.45 + displacement * 0.05)
        expected_return = min(0.015, displacement * 0.003)

        return self._make_signal(
            direction=direction,
            expected_return=expected_return,
            confidence=raw_confidence,
            regime=regime,
            symbol=symbol,
            reason=(
                f"vol_regime_z={vol_regime_zscore:.2f} "
                f"vol_pctile={vol_regime_pctile:.2f}"
            ),
            regime_fit=regime_fit,
        )


# ── Alpha Registry ─────────────────────────────────────────────────────────────

class AlphaSpecialistRegistry:
    """
    Registry of all active alpha specialists.

    Usage::

        registry = AlphaSpecialistRegistry.default()
        signals = registry.generate_all(features, symbol="NIFTY", regime="TREND_UP")
        ranked = registry.rank_signals(signals)
    """

    def __init__(self) -> None:
        self._specialists: dict[str, AlphaSpecialist] = {}

    def register(self, specialist: AlphaSpecialist) -> None:
        self._specialists[specialist.name] = specialist
        logger.debug("alpha_specialist_registered", name=specialist.name)

    def get(self, name: str) -> AlphaSpecialist | None:
        return self._specialists.get(name)

    def set_status(self, name: str, status: AlphaStatus) -> None:
        spec = self._specialists.get(name)
        if spec:
            old = spec.status
            spec.status = status
            logger.info(
                "alpha_specialist_status_changed",
                name=name,
                from_status=old.value,
                to_status=status.value,
            )

    def generate_all(
        self,
        features: pd.Series | dict[str, float],
        symbol: str = "",
        regime: str = "UNKNOWN",
    ) -> list[AlphaSignal]:
        """Run all active specialists and collect their signals."""
        signals: list[AlphaSignal] = []
        for name, spec in self._specialists.items():
            if not spec.is_active():
                continue
            try:
                sig = spec.generate(features, symbol=symbol, regime=regime)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                logger.warning(
                    "alpha_specialist_generate_failed",
                    name=name,
                    error=str(exc),
                )
        return signals

    def rank_signals(self, signals: list[AlphaSignal]) -> list[AlphaSignal]:
        """Sort signals by alpha_score (descending absolute value)."""
        return sorted(signals, key=lambda s: abs(s.alpha_score), reverse=True)

    def summary(self) -> dict[str, Any]:
        return {
            name: {
                "status": spec.status.value,
                "holding_period": spec.default_holding_period,
            }
            for name, spec in self._specialists.items()
        }

    @classmethod
    def default(cls) -> "AlphaSpecialistRegistry":
        """Create the default registry with all built-in specialists."""
        registry = cls()
        registry.register(MomentumAlpha())
        registry.register(MeanReversionAlpha())
        registry.register(BreakoutAlpha())
        registry.register(VolatilityAlpha())
        return registry
