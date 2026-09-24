"""Tests for the BaseMLModel abstract contract (src/models/base.py).

Meaningful coverage: this base class defines the interface every model wrapper
implements and the `is_production_ready` safety property that gates capital
allocation. We verify the contract holds rather than padding line counts.
"""
from __future__ import annotations

import pytest

from src.models.base import BaseMLModel
from src.schemas.base import ModelLifecycleStage, PredictionProvenance


class _Concrete(BaseMLModel):
    """A minimal concrete model that satisfies the abstract interface."""

    def predict(self, features):
        return {"action": "WAIT", "confidence": 0.0,
                "provenance": self.provenance, "model_id": self.model_id}

    def fit(self, X, y, **kwargs):
        self.trained_at = None

    def score(self, X, y):
        return {"ic": 0.0, "sharpe": 0.0}

    def health_check(self):
        return True


class _StubOnly(BaseMLModel):
    """A subclass that defers to the base NotImplementedError stubs."""

    def predict(self, features):
        return super().predict(features)

    def fit(self, X, y, **kwargs):
        return super().fit(X, y)

    def score(self, X, y):
        return super().score(X, y)

    def health_check(self):
        return super().health_check()


class TestAbstractContract:
    def test_cannot_instantiate_base_directly(self):
        with pytest.raises(TypeError):
            BaseMLModel("x")  # type: ignore[abstract]

    def test_concrete_defaults(self):
        m = _Concrete("regime_v1")
        assert m.model_id == "regime_v1"
        assert m.lifecycle_stage == ModelLifecycleStage.HYPOTHESIS
        assert m.trained_at is None
        assert m.provenance == PredictionProvenance.UNAVAILABLE

    def test_concrete_predict_fit_score_health(self):
        m = _Concrete("regime_v1")
        m.fit([[0.0]], [0])
        assert m.score([[0.0]], [0]) == {"ic": 0.0, "sharpe": 0.0}
        assert m.health_check() is True
        out = m.predict({"f": 1.0})
        assert out["action"] == "WAIT"
        assert out["model_id"] == "regime_v1"


class TestNotImplementedStubs:
    def test_predict_stub_raises(self):
        m = _StubOnly("s")
        with pytest.raises(NotImplementedError):
            m.predict({})

    def test_fit_stub_raises(self):
        m = _StubOnly("s")
        with pytest.raises(NotImplementedError):
            m.fit(None, None)

    def test_score_stub_raises(self):
        m = _StubOnly("s")
        with pytest.raises(NotImplementedError):
            m.score(None, None)

    def test_health_check_stub_raises(self):
        m = _StubOnly("s")
        with pytest.raises(NotImplementedError):
            m.health_check()


class TestProductionReadyProperty:
    def test_not_ready_by_default(self):
        assert _Concrete("s").is_production_ready is False

    def test_requires_both_stage_and_provenance(self):
        m = _Concrete("s")
        m.lifecycle_stage = ModelLifecycleStage.PRODUCTION
        # Provenance still UNAVAILABLE -> not ready.
        assert m.is_production_ready is False
        m.provenance = PredictionProvenance.TRAINED_MODEL
        assert m.is_production_ready is True

    def test_trained_provenance_without_production_stage_not_ready(self):
        m = _Concrete("s")
        m.provenance = PredictionProvenance.TRAINED_MODEL
        assert m.is_production_ready is False

    def test_repr_contains_id_stage_provenance(self):
        r = repr(_Concrete("abc"))
        assert "abc" in r
        assert "hypothesis" in r.lower()
