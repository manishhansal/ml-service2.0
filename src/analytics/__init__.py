"""src.analytics — Quantitative analytics package."""
from __future__ import annotations

from src.analytics.alpha_decay import AlphaDecayDetector  # noqa: F401
from src.analytics.counterfactual_ledger import CounterfactualLedger  # noqa: F401
from src.analytics.feature_weight_manager import FeatureWeightManager  # noqa: F401
from src.analytics.forecast_ledger import ForecastLedger  # noqa: F401
from src.analytics.reversal_detector import ReversalDetector, ReversalSignal  # noqa: F401
from src.analytics.regime_alpha_matrix import RegimeAlphaMatrix  # noqa: F401
from src.analytics.score_threshold_sweep import ScoreThresholdSweep, SweepReport  # noqa: F401
from src.analytics.signal_promotion import SignalPromotionEngine  # noqa: F401
from src.analytics.turnover_optimizer import TurnoverOptimizer, TurnoverStats  # noqa: F401

__all__ = [
    "AlphaDecayDetector",
    "CounterfactualLedger",
    "FeatureWeightManager",
    "ForecastLedger",
    "RegimeAlphaMatrix",
    "ScoreThresholdSweep",
    "SignalPromotionEngine",
    "SweepReport",
    "TurnoverOptimizer",
    "TurnoverStats",
]
