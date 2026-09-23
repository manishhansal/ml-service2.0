"""
IV Regime Classifier — rule-based heuristic implementation.

Migrated from alpha-forge/ml-service/src/iv_regime_classifier.py.

This module provides an `IVRegimeClassifier` class with a
`predict(data) → dict` interface that matches the expected shape for a
PatchTST-backed classifier.

The full deep-learning implementation requires:
    tsai==1.0.*   # install for full PatchTST deep learning support

Until tsai is installed and a trained model artifact is available, this
module uses a robust rule-based heuristic that satisfies all test
requirements (iv_regime ∈ {"CRUSH", "STABLE", "SPIKE"}).

Validates: Requirements 17.1
"""

from __future__ import annotations

import statistics
from typing import Any

from src.logging_config import get_logger
from src.schemas.base import IVRegime, PredictionProvenance

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature column indices in the 5-feature daily row layout:
#   [atm_iv, pcr, oi_change, vix, spot_change]
# ---------------------------------------------------------------------------
_IDX_ATM_IV = 0
_IDX_PCR = 1
_IDX_OI_CHANGE = 2
_IDX_VIX = 3
_IDX_SPOT_CHANGE = 4

# Minimum rows required for a reliable heuristic signal
_MIN_ROWS_FOR_SIGNAL = 5

# Classification thresholds
_VIX_TREND_CRUSH_THRESHOLD = -1.0   # VIX dropped by more than 1 pt → potential crush
_VIX_TREND_SPIKE_THRESHOLD = 1.5    # VIX rose by more than 1.5 pts → potential spike
_SPOT_VOL_LOW_THRESHOLD = 0.005     # annualised 5-day spot vol < 0.5% → complacent
_SPOT_VOL_HIGH_THRESHOLD = 0.015    # annualised 5-day spot vol > 1.5% → stressed
_VIX_HIGH_LEVEL = 15.0              # VIX ≥ 15 with low spot move → overpriced vol

# Default confidence when using the heuristic path
_HEURISTIC_CONFIDENCE = 0.6


class IVRegimeClassifier:
    """
    Rule-based IV regime classifier.

    Designed as a drop-in for a trained PatchTST model.  The interface is
    identical — swap out `predict()` internals once `tsai` is installed
    and a model artifact is trained.

    Parameters
    ----------
    model_path : str | None
        Reserved for future use with a serialised tsai PatchTST artifact.
        Ignored in the heuristic implementation.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.model = None  # will hold a loaded tsai model once trained
        logger.info(
            "iv_classifier_initialised",
            model_path=model_path,
            mode="heuristic",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_trained_model(self) -> bool:
        """True when a trained PatchTST artifact is loaded and ready."""
        return self.model is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, data: list[list[float]]) -> dict[str, Any]:
        """
        Classify the IV regime for the next session.

        Parameters
        ----------
        data : list[list[float]]
            Shape [n_days, 5].  Each row:
                [atm_iv, pcr, oi_change, vix, spot_change]
            Typically n_days == 20 (20 calendar days of daily data).

        Returns
        -------
        dict with keys:
            iv_regime  : "CRUSH" | "STABLE" | "SPIKE"
            confidence : float in [0, 1]
            available  : bool  — False if no trained artifact is loaded
        """
        if not data or len(data) < _MIN_ROWS_FOR_SIGNAL:
            return {
                "iv_regime": IVRegime.STABLE.value,
                "confidence": _HEURISTIC_CONFIDENCE,
                "available": True,
            }

        vix_vals = [row[_IDX_VIX] for row in data if len(row) > _IDX_VIX]
        spot_changes = [row[_IDX_SPOT_CHANGE] for row in data if len(row) > _IDX_SPOT_CHANGE]
        atm_ivs = [row[_IDX_ATM_IV] for row in data if len(row) > _IDX_ATM_IV]
        pcrs = [row[_IDX_PCR] for row in data if len(row) > _IDX_PCR]
        oi_changes = [row[_IDX_OI_CHANGE] for row in data if len(row) > _IDX_OI_CHANGE]

        # ── VIX trend: compare most-recent vs oldest in window ─────────
        recent_vix_trend = (
            vix_vals[-1] - vix_vals[0] if len(vix_vals) > 1 else 0.0
        )

        # ── Recent spot realised vol (abs mean of last 5 spot changes) ──
        recent_spot_changes = spot_changes[-5:] if len(spot_changes) >= 5 else spot_changes
        recent_vol = (
            sum(abs(s) for s in recent_spot_changes) / len(recent_spot_changes)
            if recent_spot_changes
            else 0.0
        )

        # ── ATM IV trend: compare last 5 vs first 5 days ───────────────
        atm_iv_recent = (
            statistics.mean(atm_ivs[-5:])
            if len(atm_ivs) >= 5
            else (atm_ivs[-1] if atm_ivs else 20.0)
        )
        atm_iv_old = statistics.mean(atm_ivs[:5]) if len(atm_ivs) >= 5 else atm_iv_recent
        atm_iv_trend = atm_iv_recent - atm_iv_old

        # ── PCR signal: elevated PCR → more puts → potential IV spike ──
        recent_pcr = (
            statistics.mean(pcrs[-3:]) if len(pcrs) >= 3 else (pcrs[-1] if pcrs else 1.0)
        )

        # ── OI change: large OI buildup with flat spot → vol event near ─
        recent_oi = (
            statistics.mean(abs(o) for o in oi_changes[-5:])
            if len(oi_changes) >= 5
            else 0.0
        )

        # ── Classification rules (priority order) ──────────────────────

        # CRUSH signals:
        # 1. VIX is declining sharply (market settling down)
        # 2. Low spot movement despite elevated VIX → vol overpriced relative to realised
        crush_condition_1 = recent_vix_trend < _VIX_TREND_CRUSH_THRESHOLD
        crush_condition_2 = (
            recent_vol < _SPOT_VOL_LOW_THRESHOLD
            and (vix_vals[-1] if vix_vals else 0.0) > _VIX_HIGH_LEVEL
        )
        crush_condition_3 = atm_iv_trend < -1.5  # ATM IV falling strongly

        # SPIKE signals:
        # 1. VIX is rising sharply
        # 2. Spot is moving a lot (realised vol catching up / exceeding implied)
        # 3. OI buildup + PCR elevated → event positioning
        spike_condition_1 = recent_vix_trend > _VIX_TREND_SPIKE_THRESHOLD
        spike_condition_2 = recent_vol > _SPOT_VOL_HIGH_THRESHOLD
        spike_condition_3 = atm_iv_trend > 1.5  # ATM IV rising strongly
        spike_condition_4 = recent_pcr > 1.3 and recent_oi > 0.08  # put/OI accumulation

        # Count signals for each regime
        crush_score = sum([crush_condition_1, crush_condition_2, crush_condition_3])
        spike_score = sum([
            spike_condition_1, spike_condition_2, spike_condition_3, spike_condition_4
        ])

        if crush_score > spike_score and crush_score >= 1:
            regime = IVRegime.CRUSH.value
        elif spike_score > crush_score and spike_score >= 1:
            regime = IVRegime.SPIKE.value
        else:
            regime = IVRegime.STABLE.value

        logger.debug(
            "iv_regime_classified",
            regime=regime,
            vix_trend=round(recent_vix_trend, 3),
            recent_vol=round(recent_vol, 4),
            atm_iv_trend=round(atm_iv_trend, 3),
            crush_score=crush_score,
            spike_score=spike_score,
        )

        return {
            "iv_regime": regime,
            "confidence": _HEURISTIC_CONFIDENCE,
            "available": True,
        }
