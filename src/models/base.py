"""
src.models.base — abstract base class for all ml-service2.0 model wrappers.

Every concrete model (RegimeClassifier, StockRanker, PriceForecaster, etc.)
must inherit from ``BaseMLModel`` and implement the abstract methods defined
here.  This contract:

  1. Enforces a consistent predict / fit / score interface across model families.
  2. Makes the model registry generic — it stores ``BaseMLModel`` instances.
  3. Enables uniform provenance tracking and lifecycle state management.

Phase 2 TDD mandate
-------------------
All abstract methods raise ``NotImplementedError``.  Tests in
``tests/test_data_contracts.py`` and ``tests/test_meta_decision.py``
drive the expected behaviour; Phase 3 provides the concrete implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from src.schemas.base import ModelLifecycleStage, PredictionProvenance


class BaseMLModel(ABC):
    """Abstract base for all ml-service2.0 model wrappers.

    Subclasses must implement: ``predict``, ``fit``, ``score``, and ``health_check``.
    The lifecycle stage, model ID, and provenance are set by the model registry
    after promotion gates are passed.

    Attributes:
        model_id:       Unique registry identifier (e.g. ``"price_forecaster_v3"``).
        lifecycle_stage: Current promotion stage (default: ``HYPOTHESIS``).
        trained_at:     UTC timestamp of the most recent ``fit()`` call.
        provenance:     Provenance emitted by ``predict()`` calls.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.lifecycle_stage: ModelLifecycleStage = ModelLifecycleStage.HYPOTHESIS
        self.trained_at: datetime | None = None
        self.provenance: PredictionProvenance = PredictionProvenance.UNAVAILABLE

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Run inference on a feature dict and return a prediction dict.

        Args:
            features: Feature dict (typically the ``model_dump()`` of a
                      ``FeatureVector``).

        Returns:
            Prediction dict containing at minimum:
              ``action``     — one of BUY / SELL / WAIT / NO_TRADE
              ``confidence`` — float in [0, 1]
              ``provenance`` — ``PredictionProvenance`` value
              ``model_id``   — this model's identifier

        Raises:
            NotImplementedError: Until Phase 3 implementation.
            ModelNotReadyError:  If the model has not been fitted.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.predict() is not yet implemented. "
            "Phase 2 TDD stub — implement in Phase 3."
        )

    @abstractmethod
    def fit(self, X: Any, y: Any, **kwargs: Any) -> None:
        """Fit the model on training data.

        Args:
            X:       Feature matrix (pandas DataFrame or numpy array).
            y:       Target vector.
            **kwargs: Model-specific training arguments (e.g. ``eval_set``).

        Raises:
            NotImplementedError: Until Phase 3 implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.fit() is not yet implemented. "
            "Phase 2 TDD stub — implement in Phase 3."
        )

    @abstractmethod
    def score(self, X: Any, y: Any) -> dict[str, float]:
        """Evaluate the model on a hold-out set.

        Args:
            X: Feature matrix.
            y: Ground-truth labels / returns.

        Returns:
            Metrics dict — must contain at minimum ``{"ic": float, "sharpe": float}``.

        Raises:
            NotImplementedError: Until Phase 3 implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.score() is not yet implemented. "
            "Phase 2 TDD stub — implement in Phase 3."
        )

    @abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` if the model is loaded and ready for inference.

        Raises:
            NotImplementedError: Until Phase 3 implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.health_check() is not yet implemented. "
            "Phase 2 TDD stub — implement in Phase 3."
        )

    # ── Concrete helpers ──────────────────────────────────────────────────────

    @property
    def is_production_ready(self) -> bool:
        """``True`` when lifecycle stage is PRODUCTION and provenance is TRAINED_MODEL."""
        return (
            self.lifecycle_stage == ModelLifecycleStage.PRODUCTION
            and self.provenance == PredictionProvenance.TRAINED_MODEL
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"model_id={self.model_id!r}, "
            f"stage={self.lifecycle_stage.value}, "
            f"provenance={self.provenance.value})"
        )
