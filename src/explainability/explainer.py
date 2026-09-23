"""
ModelExplainer — SHAP-based feature attribution for every model prediction.

Provides:
- TreeExplainer for XGBoost, LightGBM, CatBoost models
- KernelExplainer for neural network models
- In-process LRU cache for explainer instances (max 100 entries, no TTL)
- Failure-safe: on SHAP failure, returns contributions=0.0 (no 500 errors)
- Rule attribution for HEURISTIC predictions

Requirements: Req 11.1, Req 11.2, Req 11.3, Req 11.4, Req 11.6
"""
from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any

import structlog

from src.schemas.base import PredictionProvenance
from src.schemas.predictions import ExplainResponse, FeatureContribution

logger = structlog.get_logger(__name__)


# ── Lightweight in-process LRU cache (no external dependencies) ───────────────
# We re-implement the essential subset of LRUCache here to avoid importing
# src.cache.redis_cache, which eagerly instantiates the Settings singleton and
# requires environment variables (ML_SERVICE_API_KEY, DATA_SERVICE_API_KEY).
# The full RedisCache / LRUCache from that module are still used for all other
# caching needs; this copy is exclusively for SHAP explainer objects.


class _ExplainerLRUCache:
    """Thread-safe in-process LRU cache for SHAP explainer instances.

    - max_size: 100 entries (evicts LRU on overflow)
    - No TTL: entries persist until evicted or :meth:`clear` is called
    """

    def __init__(self, max_size: int = 100) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
                return
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

# Type names recognised as tree-based models — use shap.TreeExplainer.
# LightGBM models can appear either as high-level sklearn wrappers or as the
# native `Booster` type; both are captured here.
TREE_MODEL_TYPES: frozenset[str] = frozenset(
    {
        # XGBoost
        "XGBClassifier",
        "XGBRegressor",
        "XGBRanker",
        # LightGBM
        "LGBMClassifier",
        "LGBMRegressor",
        "LGBMRanker",
        "Booster",  # native lgb.Booster
        # CatBoost
        "CatBoostClassifier",
        "CatBoostRegressor",
        "CatBoostRanker",
    }
)


class ModelExplainer:
    """SHAP-based feature attribution for ml-service2.0 models.

    Explainer instances are cached in an LRU cache (max 100 entries, no TTL)
    so that large SHAP background datasets are not re-computed on every
    prediction call.  The cache is invalidated automatically when a model is
    re-registered via :meth:`register_model`.

    Usage::

        explainer = ModelExplainer()
        explainer.register_model("market_regime", xgb_model, feature_names)

        response = explainer.explain(
            model_name="market_regime",
            features={"india_vix": 18.5, "nifty_adx": 28.0},
            prediction="bull",
            provenance=PredictionProvenance.TRAINED_MODEL,
        )
    """

    def __init__(self) -> None:
        # SHAP explainer instances: model_name → shap.Explainer (or subclass)
        self._cache: _ExplainerLRUCache = _ExplainerLRUCache(max_size=100)
        # Registered models: model_name → (model_object, ordered feature names)
        self._models: dict[str, tuple[Any, list[str]]] = {}
        # Historical mean prediction per model — used as base_value on failure
        self._mean_predictions: dict[str, float] = {}

    # ── Model registration ────────────────────────────────────────────────────

    def register_model(
        self,
        model_name: str,
        model: Any,
        feature_names: list[str],
        mean_prediction: float = 0.0,
    ) -> None:
        """Register a model for SHAP explanation.

        Must be called before :meth:`explain` for ``TRAINED_MODEL`` provenance.
        Re-registering an existing model invalidates the cached explainer so
        the next call re-creates it with the new model object.

        Args:
            model_name:       Unique identifier (e.g. ``"market_regime"``).
            model:            Trained model object (XGBoost, LightGBM, …).
            feature_names:    Ordered list of feature names that the model
                              expects.  The order must match the column order
                              used during training.
            mean_prediction:  Historical mean prediction value — used as the
                              ``base_value`` field when SHAP computation fails.
        """
        self._models[model_name] = (model, list(feature_names))
        self._mean_predictions[model_name] = mean_prediction
        # Invalidate any stale explainer (model may have been updated)
        self._cache.delete(model_name)
        logger.info(
            "model_registered_for_shap",
            model_name=model_name,
            n_features=len(feature_names),
        )

    # ── Public explain interface ──────────────────────────────────────────────

    def explain(
        self,
        model_name: str,
        features: dict[str, float],
        prediction: str,
        top_k: int = 10,
        provenance: PredictionProvenance = PredictionProvenance.TRAINED_MODEL,
    ) -> ExplainResponse:
        """Generate feature attribution for a prediction.

        Dispatch table:
        - ``HEURISTIC`` → rule attribution (magnitude-based, no SHAP)
        - ``UNAVAILABLE`` / ``INSUFFICIENT_EVIDENCE`` → empty contributions
        - ``TRAINED_MODEL`` → SHAP-based attribution; falls back to empty on
          any failure without raising (Req 11.6)

        Args:
            model_name:  Model identifier (must be registered for TRAINED_MODEL
                         provenance).
            features:    Input feature dict used for the prediction.
            prediction:  The prediction that was made (e.g. ``"bull"``).
            top_k:       Maximum number of feature contributions to return,
                         ordered by |SHAP value| descending.
            provenance:  Provenance of the prediction being explained.

        Returns:
            :class:`~src.schemas.predictions.ExplainResponse` — never raises.
        """
        if provenance == PredictionProvenance.HEURISTIC:
            return self._heuristic_attribution(model_name, features, prediction, top_k)

        if provenance in (
            PredictionProvenance.UNAVAILABLE,
            PredictionProvenance.INSUFFICIENT_EVIDENCE,
        ):
            return self._empty_response(model_name, prediction)

        # TRAINED_MODEL — attempt SHAP
        model_entry = self._models.get(model_name)
        if model_entry is None:
            logger.warning(
                "shap_model_not_registered",
                model_name=model_name,
                prediction=prediction,
            )
            return self._empty_response(model_name, prediction)

        model, feature_names = model_entry
        try:
            return self._compute_shap(
                model_name, model, feature_names, features, prediction, top_k
            )
        except Exception as exc:
            logger.error(
                "shap_computation_failed",
                model_name=model_name,
                prediction=prediction,
                error=str(exc),
                exc_info=True,
            )
            return self._empty_response(model_name, prediction)

    # ── Internal SHAP helpers ─────────────────────────────────────────────────

    def _get_or_create_explainer(self, model_name: str, model: Any) -> Any | None:
        """Return a cached SHAP explainer, or create and cache a new one.

        Tries :class:`shap.TreeExplainer` first (fast, exact for tree models);
        falls back to :class:`shap.KernelExplainer` for neural networks and
        any model that TreeExplainer rejects.

        Returns ``None`` if SHAP is not installed or explainer creation fails
        for both strategies (the caller will then return an empty response).
        """
        cached = self._cache.get(model_name)
        if cached is not None:
            return cached

        try:
            import shap  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("shap_not_installed", model_name=model_name)
            return None

        model_type_name = type(model).__name__

        # ── Try TreeExplainer first ───────────────────────────────────────────
        if model_type_name in TREE_MODEL_TYPES:
            try:
                explainer = shap.TreeExplainer(model)
                self._cache.set(model_name, explainer)
                logger.debug(
                    "shap_tree_explainer_created",
                    model_name=model_name,
                    model_type=model_type_name,
                )
                return explainer
            except Exception as exc:
                logger.warning(
                    "shap_tree_explainer_failed_falling_back_to_kernel",
                    model_name=model_name,
                    error=str(exc),
                )

        # ── Fall back to KernelExplainer (model-agnostic) ─────────────────────
        try:
            import numpy as np  # type: ignore[import-untyped]

            _, feature_names = self._models.get(model_name, (None, []))
            n_features = len(feature_names) if feature_names else 10

            # Use a single all-zeros background sample to keep overhead low.
            background = np.zeros((1, n_features))

            # Prefer model.predict_proba for classifiers; fall back to predict.
            predict_fn = (
                model.predict_proba
                if hasattr(model, "predict_proba")
                else model.predict
            )
            explainer = shap.KernelExplainer(predict_fn, background)
            self._cache.set(model_name, explainer)
            logger.debug(
                "shap_kernel_explainer_created",
                model_name=model_name,
                model_type=model_type_name,
            )
            return explainer
        except Exception as exc:
            logger.warning(
                "shap_kernel_explainer_failed",
                model_name=model_name,
                error=str(exc),
            )
            return None

    def _compute_shap(
        self,
        model_name: str,
        model: Any,
        feature_names: list[str],
        features: dict[str, float],
        prediction: str,
        top_k: int,
    ) -> ExplainResponse:
        """Compute SHAP values and assemble an :class:`ExplainResponse`.

        May raise — the caller (:meth:`explain`) wraps this in a try/except
        and returns an empty response on any failure.
        """
        import numpy as np  # type: ignore[import-untyped]

        explainer = self._get_or_create_explainer(model_name, model)
        if explainer is None:
            return self._empty_response(model_name, prediction)

        # Build the feature vector in the exact order the model expects.
        feature_vector = np.array(
            [[features.get(name, 0.0) for name in feature_names]],
            dtype=float,
        )

        # Compute SHAP values — may return an ndarray or a list of ndarrays
        # (multi-class classifiers).
        shap_values = explainer.shap_values(feature_vector)

        # Multi-class: take the first output class for simplicity.
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # shap_values is (n_samples, n_features); squeeze the sample dimension.
        shap_array: Any
        if hasattr(shap_values, "ndim") and shap_values.ndim == 2:
            shap_array = shap_values[0]
        else:
            shap_array = shap_values

        # ── Build contributions list ───────────────────────────────────────────
        raw: list[tuple[str, float, float]] = []
        for i, name in enumerate(feature_names):
            raw_value = features.get(name, 0.0)
            shap_val = float(shap_array[i]) if i < len(shap_array) else 0.0
            raw.append((name, raw_value, shap_val))

        # Sort by |SHAP| descending and keep top_k entries.
        raw.sort(key=lambda x: abs(x[2]), reverse=True)
        top = raw[:top_k]

        contributions = [
            FeatureContribution(
                feature=name,
                value=float(raw_val),
                contribution=float(shap_val),
                direction="positive" if shap_val >= 0 else "negative",
            )
            for name, raw_val, shap_val in top
        ]

        total_positive = sum(c.contribution for c in contributions if c.contribution > 0)
        total_negative = sum(c.contribution for c in contributions if c.contribution < 0)
        top_drivers = [c.feature for c in contributions[:5]]

        # Resolve base_value — expected_value may be a scalar or a list/array
        # (multi-class returns one expected value per class).
        base_value: float
        raw_base = getattr(explainer, "expected_value", None)
        if raw_base is None:
            base_value = self._mean_predictions.get(model_name, 0.0)
        elif isinstance(raw_base, (list, tuple)):
            base_value = float(raw_base[0])
        elif hasattr(raw_base, "__len__"):
            # numpy array
            base_value = float(raw_base.flat[0])
        else:
            base_value = float(raw_base)

        return ExplainResponse(
            model=model_name,
            prediction=prediction,
            base_value=base_value,
            contributions=contributions,
            total_positive=total_positive,
            total_negative=total_negative,
            top_drivers=top_drivers,
        )

    # ── Fallback response builders ─────────────────────────────────────────────

    def _heuristic_attribution(
        self,
        model_name: str,
        features: dict[str, float],
        prediction: str,
        top_k: int,
    ) -> ExplainResponse:
        """Rule attribution for HEURISTIC predictions.

        Since no trained model is involved, there are no SHAP values.  Instead,
        we surface the features with the largest absolute magnitude as the
        "most influential" inputs — i.e. the rule conditions that fired most
        strongly — and use each feature's own value as its contribution score.
        """
        sorted_features = sorted(
            features.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_k]

        contributions = [
            FeatureContribution(
                feature=name,
                value=float(val),
                contribution=float(val),  # rule-based: contribution ≡ value
                direction="positive" if val >= 0 else "negative",
            )
            for name, val in sorted_features
        ]

        total_positive = sum(c.contribution for c in contributions if c.contribution > 0)
        total_negative = sum(c.contribution for c in contributions if c.contribution < 0)
        top_drivers = [c.feature for c in contributions[:5]]

        return ExplainResponse(
            model=model_name,
            prediction=prediction,
            base_value=self._mean_predictions.get(model_name, 0.0),
            contributions=contributions,
            total_positive=total_positive,
            total_negative=total_negative,
            top_drivers=top_drivers,
        )

    def _empty_response(self, model_name: str, prediction: str) -> ExplainResponse:
        """Return a zero-contribution :class:`ExplainResponse` (failure-safe).

        Used when SHAP is unavailable, the model is not registered, or
        provenance indicates no model was used.  Never raises.
        """
        return ExplainResponse(
            model=model_name,
            prediction=prediction,
            base_value=self._mean_predictions.get(model_name, 0.0),
            contributions=[],
            total_positive=0.0,
            total_negative=0.0,
            top_drivers=[],
        )

    # ── Cache management ──────────────────────────────────────────────────────

    def clear_cache(self) -> None:
        """Evict all cached SHAP explainer instances.

        Call after reloading model artifacts so the next :meth:`explain` call
        re-creates explainers for the updated models.
        """
        self._cache.clear()
        logger.info("shap_explainer_cache_cleared")
