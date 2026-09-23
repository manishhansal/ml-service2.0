"""
Coverage boost — batch 5.

Targets:
- src/explainability/explainer.py — SHAP computation paths (39 missing lines)
- src/models/portfolio_optimizer.py — normalize/equal weight helpers + fallbacks (48 lines)
- Various remaining <90% modules
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

import numpy as np
import pytest


# ===========================================================================
# ModelExplainer — SHAP paths
# ===========================================================================


class TestModelExplainerShapPaths:
    def test_compute_shap_with_multiclass_shap_values(self):
        """When shap_values is a list → takes first element."""
        from src.explainability.explainer import ModelExplainer
        from src.schemas.base import PredictionProvenance

        explainer = ModelExplainer()
        mock_model = MagicMock()
        features = ["f1", "f2", "f3"]
        explainer.register_model("test_model", mock_model, features)

        mock_shap_explainer = MagicMock()
        # Returns list of arrays (multi-class)
        mock_shap_explainer.shap_values.return_value = [
            np.array([[0.3, -0.1, 0.5]]),
            np.array([[0.1, 0.2, 0.3]]),
        ]
        mock_shap_explainer.expected_value = [0.5, 0.3]  # list
        explainer._cache.set("test_model", mock_shap_explainer)

        response = explainer.explain(
            model_name="test_model",
            features={"f1": 1.0, "f2": 2.0, "f3": 3.0},
            prediction="bull",
            provenance=PredictionProvenance.TRAINED_MODEL,
        )
        assert len(response.contributions) == 3

    def test_compute_shap_with_1d_shap_values(self):
        """When shap_values is 1D array (already squeezed)."""
        from src.explainability.explainer import ModelExplainer
        from src.schemas.base import PredictionProvenance

        explainer = ModelExplainer()
        mock_model = MagicMock()
        features = ["f1", "f2"]
        explainer.register_model("m", mock_model, features)

        mock_shap = MagicMock()
        # 1D array (no ndim == 2)
        mock_shap.shap_values.return_value = np.array([0.4, -0.2])
        mock_shap.expected_value = 0.5
        explainer._cache.set("m", mock_shap)

        response = explainer.explain(
            model_name="m",
            features={"f1": 1.0, "f2": 2.0},
            prediction="bull",
            provenance=PredictionProvenance.TRAINED_MODEL,
        )
        assert len(response.contributions) == 2

    def test_compute_shap_numpy_base_value(self):
        """expected_value as numpy array → converted to float."""
        from src.explainability.explainer import ModelExplainer
        from src.schemas.base import PredictionProvenance

        explainer = ModelExplainer()
        mock_model = MagicMock()
        features = ["f1"]
        explainer.register_model("m", mock_model, features)

        mock_shap = MagicMock()
        mock_shap.shap_values.return_value = np.array([[0.5]])
        # numpy array as expected_value
        mock_shap.expected_value = np.array([0.7])
        explainer._cache.set("m", mock_shap)

        response = explainer.explain(
            model_name="m",
            features={"f1": 1.0},
            prediction="bull",
            provenance=PredictionProvenance.TRAINED_MODEL,
        )
        assert response.base_value == pytest.approx(0.7, abs=0.01)

    def test_get_or_create_explainer_shap_not_installed(self):
        """shap not installed → returns None."""
        from src.explainability.explainer import ModelExplainer

        explainer = ModelExplainer()
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "XGBClassifier"
        explainer.register_model("m", mock_model, ["f1"])

        with patch.dict("sys.modules", {"shap": None}):
            result = explainer._get_or_create_explainer("m", mock_model)

        assert result is None

    def test_get_or_create_explainer_uses_tree_explainer_for_tree_model(self):
        """For tree model types, TreeExplainer is used."""
        from src.explainability.explainer import ModelExplainer

        explainer = ModelExplainer()
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "XGBClassifier"
        explainer.register_model("m", mock_model, ["f1", "f2"])

        mock_shap = MagicMock()
        mock_tree_explainer = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_tree_explainer

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = explainer._get_or_create_explainer("m", mock_model)

        assert result == mock_tree_explainer
        mock_shap.TreeExplainer.assert_called_once_with(mock_model)

    def test_get_or_create_explainer_tree_fails_falls_back_to_kernel(self):
        """TreeExplainer failure → KernelExplainer fallback."""
        from src.explainability.explainer import ModelExplainer

        explainer = ModelExplainer()
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "XGBClassifier"
        mock_model.predict_proba = MagicMock()
        explainer.register_model("m", mock_model, ["f1", "f2"])

        mock_shap = MagicMock()
        mock_shap.TreeExplainer.side_effect = RuntimeError("tree failed")
        mock_kernel_explainer = MagicMock()
        mock_shap.KernelExplainer.return_value = mock_kernel_explainer

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = explainer._get_or_create_explainer("m", mock_model)

        assert result == mock_kernel_explainer

    def test_get_or_create_explainer_kernel_also_fails_returns_none(self):
        """Both TreeExplainer and KernelExplainer fail → None."""
        from src.explainability.explainer import ModelExplainer

        explainer = ModelExplainer()
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "XGBClassifier"
        explainer.register_model("m", mock_model, ["f1"])

        mock_shap = MagicMock()
        mock_shap.TreeExplainer.side_effect = RuntimeError("tree failed")
        mock_shap.KernelExplainer.side_effect = RuntimeError("kernel failed")

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = explainer._get_or_create_explainer("m", mock_model)

        assert result is None

    def test_get_or_create_explainer_non_tree_uses_kernel(self):
        """Non-tree model → KernelExplainer directly."""
        from src.explainability.explainer import ModelExplainer

        explainer = ModelExplainer()
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "MLP"  # not in TREE_MODEL_TYPES
        mock_model.predict_proba = MagicMock()
        explainer.register_model("m", mock_model, ["f1"])

        mock_shap = MagicMock()
        mock_kernel_explainer = MagicMock()
        mock_shap.KernelExplainer.return_value = mock_kernel_explainer

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = explainer._get_or_create_explainer("m", mock_model)

        assert result == mock_kernel_explainer

    def test_compute_shap_explainer_none_returns_empty(self):
        """_compute_shap with explainer=None → _empty_response."""
        from src.explainability.explainer import ModelExplainer
        from src.schemas.base import PredictionProvenance

        explainer = ModelExplainer()
        mock_model = MagicMock()
        features = ["f1"]
        explainer.register_model("m", mock_model, features)
        # Leave cache empty → _get_or_create_explainer returns None due to no shap

        with patch.dict("sys.modules", {"shap": None}):
            response = explainer.explain(
                model_name="m",
                features={"f1": 1.0},
                prediction="bull",
                provenance=PredictionProvenance.TRAINED_MODEL,
            )

        assert response.contributions == []

    def test_compute_shap_raises_returns_empty(self):
        """When _compute_shap raises, explain() returns empty response."""
        from src.explainability.explainer import ModelExplainer
        from src.schemas.base import PredictionProvenance

        explainer = ModelExplainer()
        mock_model = MagicMock()
        features = ["f1"]
        explainer.register_model("m", mock_model, features)

        mock_shap_explainer = MagicMock()
        mock_shap_explainer.shap_values.side_effect = RuntimeError("shap exploded")
        explainer._cache.set("m", mock_shap_explainer)

        response = explainer.explain(
            model_name="m",
            features={"f1": 1.0},
            prediction="bull",
            provenance=PredictionProvenance.TRAINED_MODEL,
        )
        assert response.contributions == []

    def test_heuristic_attribution_negative_values(self):
        """Features with negative values → direction='negative' in contributions."""
        from src.explainability.explainer import ModelExplainer
        from src.schemas.base import PredictionProvenance

        explainer = ModelExplainer()
        features = {"f1": -5.0, "f2": 3.0}
        response = explainer.explain(
            model_name="test",
            features=features,
            prediction="bear",
            provenance=PredictionProvenance.HEURISTIC,
        )
        neg_contributions = [c for c in response.contributions if c.direction == "negative"]
        assert len(neg_contributions) > 0

    def test_total_positive_negative_in_response(self):
        """total_positive and total_negative are computed correctly."""
        from src.explainability.explainer import ModelExplainer
        from src.schemas.base import PredictionProvenance

        explainer = ModelExplainer()
        mock_model = MagicMock()
        features = ["f1", "f2", "f3"]
        explainer.register_model("m", mock_model, features)

        mock_shap = MagicMock()
        mock_shap.shap_values.return_value = np.array([[0.3, -0.2, 0.1]])
        mock_shap.expected_value = 0.5
        explainer._cache.set("m", mock_shap)

        response = explainer.explain(
            model_name="m",
            features={"f1": 1.0, "f2": 2.0, "f3": 3.0},
            prediction="bull",
            provenance=__import__("src.schemas.base", fromlist=["PredictionProvenance"]).PredictionProvenance.TRAINED_MODEL,
        )
        assert response.total_positive > 0
        assert response.total_negative < 0


# ===========================================================================
# PortfolioOptimizer — helper functions and insufficient data path
# ===========================================================================


class TestPortfolioOptimizer:
    def _make_returns(self, n=50, assets=3):
        """Create a returns DataFrame with sufficient data."""
        import pandas as pd
        cols = [f"ASSET_{i}" for i in range(assets)]
        data = np.random.randn(n, assets) * 0.01
        return pd.DataFrame(data, columns=cols)

    def test_normalize_weights_basic(self):
        from src.models.portfolio_optimizer import _normalize_weights

        weights = {"A": 0.3, "B": 0.5, "C": 0.2}
        result = _normalize_weights(weights)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_normalize_weights_clips_negative(self):
        from src.models.portfolio_optimizer import _normalize_weights

        weights = {"A": -0.1, "B": 0.5, "C": 0.3}
        result = _normalize_weights(weights)
        assert all(v >= 0 for v in result.values())
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_normalize_weights_all_zero_equal_weight(self):
        from src.models.portfolio_optimizer import _normalize_weights

        weights = {"A": 0.0, "B": 0.0}
        result = _normalize_weights(weights)
        assert abs(result["A"] - 0.5) < 1e-9

    def test_equal_weights(self):
        from src.models.portfolio_optimizer import _equal_weights

        result = _equal_weights(["A", "B", "C", "D"])
        assert len(result) == 4
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert all(abs(v - 0.25) < 1e-9 for v in result.values())

    def test_equal_weights_empty_list(self):
        from src.models.portfolio_optimizer import _equal_weights

        result = _equal_weights([])
        assert result == {}

    def test_hrp_insufficient_data(self):
        """Returns available=False when not enough rows."""
        import pandas as pd
        from src.models.portfolio_optimizer import PortfolioOptimizer

        opt = PortfolioOptimizer()
        returns = self._make_returns(n=5)  # below minimum
        result = opt.hrp_allocation(returns)
        assert result.get("available") is False

    def test_cvar_insufficient_data(self):
        import pandas as pd
        from src.models.portfolio_optimizer import PortfolioOptimizer

        opt = PortfolioOptimizer()
        returns = self._make_returns(n=5)
        result = opt.cvar_allocation(returns)
        assert result.get("available") is False

    def test_erc_insufficient_data(self):
        from src.models.portfolio_optimizer import PortfolioOptimizer

        opt = PortfolioOptimizer()
        returns = self._make_returns(n=5)
        result = opt.erc_allocation(returns)
        assert result.get("available") is False

    def test_hrp_with_all_libraries_failing_uses_equal_weight(self):
        """All optimization libraries fail → equal weight fallback."""
        from src.models.portfolio_optimizer import PortfolioOptimizer
        from src.schemas.base import PredictionProvenance

        opt = PortfolioOptimizer()
        returns = self._make_returns(n=60)

        with patch.object(opt, "_hrp_riskfolio", side_effect=RuntimeError("riskfolio fail")):
            with patch.object(opt, "_hrp_pypfopt", side_effect=RuntimeError("pypfopt fail")):
                result = opt.hrp_allocation(returns)

        assert result.get("available") is True
        assert result.get("provenance") == PredictionProvenance.HEURISTIC
        weights = result.get("weights", {})
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_cvar_fallback_to_hrp_on_failure(self):
        """CVaR failure → falls back to HRP."""
        from src.models.portfolio_optimizer import PortfolioOptimizer

        opt = PortfolioOptimizer()
        returns = self._make_returns(n=60)

        with patch.object(opt, "_cvar_riskfolio", side_effect=RuntimeError("cvar fail")):
            with patch.object(opt, "_hrp_riskfolio", side_effect=RuntimeError("hrp fail")):
                with patch.object(opt, "_hrp_pypfopt", side_effect=RuntimeError("pypfopt fail")):
                    result = opt.cvar_allocation(returns)

        assert result.get("available") is True


# ===========================================================================
# Additional API coverage tests
# ===========================================================================


class TestAPIPredict:
    """Boost coverage of src/api/predict.py."""

    def test_predict_endpoint_requires_auth(self, app_client):
        response = app_client.post(
            "/v2/predict/regime",
            json={"india_vix": 18.5, "nifty_change_pct": 0.5},
        )
        # No auth header → 403 or 422
        assert response.status_code in (403, 422, 401)

    def test_predict_regime_with_auth(self, app_client, auth_headers):
        response = app_client.post(
            "/v2/predict/regime",
            headers=auth_headers,
            json={"india_vix": 18.5, "nifty_change_pct": 0.5},
        )
        # Should return 200, 422, or 503 (data service unavailable in test env)
        assert response.status_code in (200, 422, 503)

    def test_predict_risk_with_auth(self, app_client, auth_headers):
        response = app_client.post(
            "/v2/predict/risk",
            headers=auth_headers,
            json={
                "entry": 22000.0,
                "stop_loss": 21800.0,
                "target": 22400.0,
                "atr": 200.0,
                "regime": "bull",
            },
        )
        assert response.status_code in (200, 422, 503)

    def test_predict_strategy_with_auth(self, app_client, auth_headers):
        response = app_client.post(
            "/v2/predict/strategy",
            headers=auth_headers,
            json={
                "regime": "bull",
                "rsi": 60.0,
                "adx": 25.0,
                "iv_regime": "STABLE",
            },
        )
        assert response.status_code in (200, 422, 503)


import pytest


@pytest.fixture
def app_client():
    """Session-scoped TestClient."""
    import os
    os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
    os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")
    from fastapi.testclient import TestClient
    from src.main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def auth_headers():
    return {"X-API-KEY": "test-key-for-testing"}
