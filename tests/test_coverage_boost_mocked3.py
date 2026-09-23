"""
Final coverage boost tests — batch 3.

Targets:
- src/cache/redis_cache.py (LRUCache portion — 52%)
- src/explainability/explainer.py (33%)
- src/audit/logger.py (85%)
- src/monitoring/drift_monitor.py (89% — last few lines)
- src/models/risk_predictor.py (78%)
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from threading import Thread
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

import pytest


# ===========================================================================
# LRUCache (in-process, no Redis needed)
# ===========================================================================


class TestLRUCache:
    def test_get_returns_none_on_miss(self):
        from src.cache.redis_cache import LRUCache

        cache = LRUCache(max_size=10)
        assert cache.get("missing_key") is None

    def test_set_and_get(self):
        from src.cache.redis_cache import LRUCache

        cache = LRUCache(max_size=10)
        cache.set("key1", {"data": 42})
        result = cache.get("key1")
        assert result == {"data": 42}

    def test_ttl_expiry_returns_none(self):
        from src.cache.redis_cache import LRUCache

        cache = LRUCache(max_size=10, ttl_seconds=1)
        cache.set("key1", "value")
        # Value should be present immediately
        assert cache.get("key1") == "value"
        # Simulate TTL elapsed
        with patch("time.monotonic", return_value=time.monotonic() + 2.0):
            assert cache.get("key1") is None

    def test_lru_eviction_when_full(self):
        from src.cache.redis_cache import LRUCache

        cache = LRUCache(max_size=3)
        cache.set("k1", 1)
        cache.set("k2", 2)
        cache.set("k3", 3)
        # Access k1 to make it recently used
        cache.get("k1")
        # Add k4 — should evict k2 (LRU)
        cache.set("k4", 4)
        assert cache.get("k2") is None
        assert cache.get("k1") == 1
        assert cache.get("k4") == 4

    def test_update_existing_key(self):
        from src.cache.redis_cache import LRUCache

        cache = LRUCache(max_size=5)
        cache.set("key", "old_value")
        cache.set("key", "new_value")
        assert cache.get("key") == "new_value"
        assert len(cache) == 1

    def test_delete_existing_key(self):
        from src.cache.redis_cache import LRUCache

        cache = LRUCache(max_size=5)
        cache.set("key", "value")
        cache.delete("key")
        assert cache.get("key") is None

    def test_delete_nonexistent_key_no_error(self):
        from src.cache.redis_cache import LRUCache

        cache = LRUCache(max_size=5)
        # Should not raise
        cache.delete("nonexistent")

    def test_clear_empties_cache(self):
        from src.cache.redis_cache import LRUCache

        cache = LRUCache(max_size=5)
        cache.set("k1", 1)
        cache.set("k2", 2)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("k1") is None

    def test_len_tracks_size(self):
        from src.cache.redis_cache import LRUCache

        cache = LRUCache(max_size=10)
        assert len(cache) == 0
        cache.set("k1", 1)
        assert len(cache) == 1
        cache.set("k2", 2)
        assert len(cache) == 2

    def test_thread_safety_concurrent_sets(self):
        """Multiple threads setting the same key should not crash."""
        from src.cache.redis_cache import LRUCache

        cache = LRUCache(max_size=100)
        errors = []

        def worker(i):
            try:
                for j in range(10):
                    cache.set(f"key_{i}_{j}", j)
                    cache.get(f"key_{i}_{j}")
            except Exception as e:
                errors.append(e)

        threads = [Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_no_ttl_never_expires(self):
        from src.cache.redis_cache import LRUCache

        cache = LRUCache(max_size=10, ttl_seconds=None)
        cache.set("key", "val")
        with patch("time.monotonic", return_value=time.monotonic() + 10000.0):
            assert cache.get("key") == "val"


# ===========================================================================
# RedisCache — test with mocked Redis client
# ===========================================================================


class TestRedisCacheWithMock:
    def test_get_returns_none_when_not_connected(self):
        async def _run():
            from src.cache.redis_cache import RedisCache

            cache = RedisCache()
            result = await cache.get("key")
            assert result is None

        asyncio.run(_run())

    def test_set_skips_when_not_connected(self):
        async def _run():
            from src.cache.redis_cache import RedisCache

            cache = RedisCache()
            # Should not raise
            await cache.set("key", {"val": 1})

        asyncio.run(_run())

    def test_delete_skips_when_not_connected(self):
        async def _run():
            from src.cache.redis_cache import RedisCache

            cache = RedisCache()
            await cache.delete("key")

        asyncio.run(_run())

    def test_exists_returns_false_when_not_connected(self):
        async def _run():
            from src.cache.redis_cache import RedisCache

            cache = RedisCache()
            result = await cache.exists("key")
            assert result is False

        asyncio.run(_run())

    def test_get_with_redis_connected(self):
        async def _run():
            from src.cache.redis_cache import RedisCache

            cache = RedisCache()
            mock_client = AsyncMock()
            mock_client.get.return_value = '{"data": 42}'
            cache._client = mock_client

            result = await cache.get("key")
            assert result == {"data": 42}

        asyncio.run(_run())

    def test_get_redis_error_returns_none(self):
        async def _run():
            from src.cache.redis_cache import RedisCache

            cache = RedisCache()
            mock_client = AsyncMock()
            mock_client.get.side_effect = RuntimeError("redis error")
            cache._client = mock_client

            result = await cache.get("key")
            assert result is None

        asyncio.run(_run())

    def test_set_with_ttl(self):
        async def _run():
            from src.cache.redis_cache import RedisCache

            cache = RedisCache()
            mock_client = AsyncMock()
            cache._client = mock_client

            await cache.set("key", {"val": 1}, ttl=60)
            mock_client.set.assert_called_once()

        asyncio.run(_run())

    def test_set_without_ttl(self):
        async def _run():
            from src.cache.redis_cache import RedisCache

            cache = RedisCache()
            mock_client = AsyncMock()
            cache._client = mock_client

            await cache.set("key", {"val": 1}, ttl=None)
            mock_client.set.assert_called_once()

        asyncio.run(_run())

    def test_exists_with_key_present(self):
        async def _run():
            from src.cache.redis_cache import RedisCache

            cache = RedisCache()
            mock_client = AsyncMock()
            mock_client.exists.return_value = 1
            cache._client = mock_client

            result = await cache.exists("key")
            assert result is True

        asyncio.run(_run())

    def test_delete_with_redis_error_no_raise(self):
        async def _run():
            from src.cache.redis_cache import RedisCache

            cache = RedisCache()
            mock_client = AsyncMock()
            mock_client.delete.side_effect = RuntimeError("redis error")
            cache._client = mock_client

            await cache.delete("key")  # Should not raise

        asyncio.run(_run())

    def test_get_feature_and_set_feature(self):
        async def _run():
            from src.cache.redis_cache import RedisCache

            cache = RedisCache()
            mock_client = AsyncMock()
            mock_client.get.return_value = '{"feature": 1.0}'
            cache._client = mock_client

            result = await cache.get_feature("NIFTY", "20240101")
            assert result == {"feature": 1.0}

            await cache.set_feature("NIFTY", "20240101", {"feature": 1.0})
            mock_client.set.assert_called_once()

        asyncio.run(_run())

    def test_get_and_set_news(self):
        async def _run():
            from src.cache.redis_cache import RedisCache

            cache = RedisCache()
            mock_client = AsyncMock()
            mock_client.get.return_value = '{"news": "positive"}'
            cache._client = mock_client

            await cache.set_news("NIFTY", {"news": "positive"})
            result = await cache.get_news("NIFTY")
            assert result == {"news": "positive"}

        asyncio.run(_run())

    def test_typed_helpers_llm_signal_drift(self):
        async def _run():
            from src.cache.redis_cache import RedisCache

            cache = RedisCache()
            mock_client = AsyncMock()
            mock_client.get.return_value = '{"ok": true}'
            cache._client = mock_client

            await cache.set_llm_news("NIFTY", {"ok": True})
            await cache.set_signal("NIFTY", {"action": "BUY"})
            await cache.set_model_status({"loaded": True})
            await cache.set_drift_latest({"psi": 0.1})

            await cache.get_llm_news("NIFTY")
            await cache.get_signal("NIFTY")
            await cache.get_model_status()
            await cache.get_drift_latest()

        asyncio.run(_run())

    def test_disconnect_when_connected(self):
        async def _run():
            from src.cache.redis_cache import RedisCache

            cache = RedisCache()
            mock_client = AsyncMock()
            cache._client = mock_client

            await cache.disconnect()
            assert cache._client is None
            mock_client.aclose.assert_called_once()

        asyncio.run(_run())

    def test_connect_idempotent(self):
        async def _run():
            from src.cache.redis_cache import RedisCache

            cache = RedisCache(url="redis://localhost:6379")
            mock_client = AsyncMock()
            cache._client = mock_client

            # Second connect call should be a no-op
            await cache.connect()
            assert cache._client == mock_client

        asyncio.run(_run())


# ===========================================================================
# ModelExplainer
# ===========================================================================


class TestModelExplainer:
    def test_explain_heuristic_provenance(self):
        from src.explainability.explainer import ModelExplainer
        from src.schemas.base import PredictionProvenance

        explainer = ModelExplainer()
        response = explainer.explain(
            model_name="test",
            features={"india_vix": 18.5, "nifty_change": 0.5, "adx": 25.0},
            prediction="bull",
            provenance=PredictionProvenance.HEURISTIC,
        )
        assert response.model == "test"
        assert response.prediction == "bull"
        assert len(response.contributions) > 0

    def test_explain_unavailable_provenance(self):
        from src.explainability.explainer import ModelExplainer
        from src.schemas.base import PredictionProvenance

        explainer = ModelExplainer()
        response = explainer.explain(
            model_name="test",
            features={"x": 1.0},
            prediction="bull",
            provenance=PredictionProvenance.UNAVAILABLE,
        )
        assert response.contributions == []

    def test_explain_insufficient_evidence_provenance(self):
        from src.explainability.explainer import ModelExplainer
        from src.schemas.base import PredictionProvenance

        explainer = ModelExplainer()
        response = explainer.explain(
            model_name="test",
            features={"x": 1.0},
            prediction="bull",
            provenance=PredictionProvenance.INSUFFICIENT_EVIDENCE,
        )
        assert response.contributions == []

    def test_explain_trained_model_not_registered_returns_empty(self):
        from src.explainability.explainer import ModelExplainer
        from src.schemas.base import PredictionProvenance

        explainer = ModelExplainer()
        response = explainer.explain(
            model_name="unregistered_model",
            features={"x": 1.0},
            prediction="bull",
            provenance=PredictionProvenance.TRAINED_MODEL,
        )
        assert response.contributions == []

    def test_register_model_invalidates_cache(self):
        from src.explainability.explainer import ModelExplainer

        explainer = ModelExplainer()
        mock_model = MagicMock()
        features = ["f1", "f2", "f3"]

        explainer.register_model("test_model", mock_model, features)
        assert "test_model" in explainer._models

        # Re-register should update
        mock_model2 = MagicMock()
        explainer.register_model("test_model", mock_model2, features)
        assert explainer._models["test_model"][0] is mock_model2

    def test_clear_cache_empties_shap_cache(self):
        from src.explainability.explainer import ModelExplainer

        explainer = ModelExplainer()
        explainer._cache.set("some_model", MagicMock())
        explainer.clear_cache()
        assert len(explainer._cache) == 0

    def test_heuristic_attribution_top_k(self):
        from src.explainability.explainer import ModelExplainer
        from src.schemas.base import PredictionProvenance

        explainer = ModelExplainer()
        features = {f"feat_{i}": float(i) for i in range(20)}
        response = explainer.explain(
            model_name="test",
            features=features,
            prediction="bull",
            top_k=5,
            provenance=PredictionProvenance.HEURISTIC,
        )
        assert len(response.contributions) <= 5

    def test_explain_with_mock_shap(self):
        """Mock SHAP computation to test TRAINED_MODEL path."""
        from src.explainability.explainer import ModelExplainer
        from src.schemas.base import PredictionProvenance
        import numpy as np

        explainer = ModelExplainer()
        mock_model = MagicMock()
        features = ["f1", "f2", "f3"]
        explainer.register_model("test_model", mock_model, features)

        # Mock SHAP explainer
        mock_shap_explainer = MagicMock()
        mock_shap_explainer.shap_values.return_value = np.array([[0.3, -0.1, 0.5]])
        mock_shap_explainer.expected_value = 0.5

        explainer._cache.set("test_model", mock_shap_explainer)

        response = explainer.explain(
            model_name="test_model",
            features={"f1": 1.0, "f2": 2.0, "f3": 3.0},
            prediction="bull",
            provenance=PredictionProvenance.TRAINED_MODEL,
        )
        assert len(response.contributions) == 3
        assert response.base_value == 0.5

    def test_empty_response_builder(self):
        from src.explainability.explainer import ModelExplainer

        explainer = ModelExplainer()
        response = explainer._empty_response("model", "prediction")
        assert response.contributions == []
        assert response.model == "model"
        assert response.prediction == "prediction"

    def test_explainer_lru_cache_operations(self):
        """Test _ExplainerLRUCache directly."""
        from src.explainability.explainer import _ExplainerLRUCache

        cache = _ExplainerLRUCache(max_size=3)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        assert cache.get("k1") == "v1"
        assert len(cache) == 2
        cache.delete("k1")
        assert cache.get("k1") is None
        cache.clear()
        assert len(cache) == 0

    def test_explainer_lru_eviction(self):
        """Test LRU eviction when max_size reached."""
        from src.explainability.explainer import _ExplainerLRUCache

        cache = _ExplainerLRUCache(max_size=2)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.get("k1")  # make k1 recently used
        cache.set("k3", "v3")  # should evict k2
        assert cache.get("k2") is None
        assert cache.get("k1") == "v1"


# ===========================================================================
# AuditLogger — missing coverage lines
# ===========================================================================


class TestAuditLoggerExtra:
    def test_check_append_only_with_empty_file(self, tmp_path):
        """Empty log file → no violation."""
        from src.audit.logger import AuditLogger

        log_file = tmp_path / "audit.jsonl"
        log_file.write_text("", encoding="utf-8")

        audit = AuditLogger(log_path=log_file)
        # Should not raise
        audit._check_append_only()

    def test_check_append_only_with_blank_lines(self, tmp_path):
        """Blank lines in log → no error (they're skipped)."""
        from src.audit.logger import AuditLogger

        log_file = tmp_path / "audit.jsonl"
        log_file.write_text("\n\n\n", encoding="utf-8")

        audit = AuditLogger(log_path=log_file)
        audit._check_append_only()

    def test_log_online_update(self, tmp_path):
        from src.audit.logger import AuditLogger

        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        audit.log_online_update(
            model_name="test",
            prior_version="1.0.0",
            new_version="1.0.0-online-20240101-01",
            ic_delta=0.01,
            consecutive_update_count=1,
        )
        # Should write an entry
        assert (tmp_path / "audit.jsonl").exists()

    def test_log_online_update_rejected(self, tmp_path):
        from src.audit.logger import AuditLogger

        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        audit.log_online_update_rejected(
            model_name="test",
            prior_version="1.0.0",
            candidate_version="1.0.0-online-20240101-01",
            reason="new_ic_below_prior",
            prior_ic=0.05,
            candidate_ic=0.02,
        )
        assert (tmp_path / "audit.jsonl").exists()

    def test_check_append_only_detects_tamper(self, tmp_path):
        """Tampered log entry → AuditLogViolation."""
        from src.audit.logger import AuditLogger, AuditLogViolation
        import json

        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        # Write a valid entry
        audit.log_training_run(
            run_id="r1",
            model_name="m",
            model_version="v1",
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T01:00:00Z",
            dataset_hash="a" * 64,
            training_date_range=("2024-01-01", "2024-01-01"),
            validation_date_range=("2024-01-01", "2024-01-01"),
            hyperparameters={},
            ic_per_fold=[0.05],
            ic_mean=0.05,
            sharpe_net=1.0,
            max_drawdown=-0.01,
            pbo=0.2,
            gate_results={"IC": "PASS"},
            outcome="PASSED",
        )

        # Tamper with the log by overwriting with modified content
        log_path = tmp_path / "audit.jsonl"
        original = log_path.read_text()
        entry = json.loads(original.strip())
        entry["ic_mean"] = 0.99  # Tamper
        entry_hash = entry.pop("entry_hash")  # Remove hash
        entry["entry_hash"] = entry_hash  # Keep original hash (mismatched now)
        tampered = json.dumps(entry) + "\n"
        log_path.write_text(tampered)

        with pytest.raises(AuditLogViolation):
            audit._check_append_only()

    def test_log_promotion_decision(self, tmp_path):
        from src.audit.logger import AuditLogger

        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        audit.log_promotion_decision(
            challenger_id="v2.0.0",
            champion_id="v1.0.0",
            outcome="PROMOTE",
            gate_results={"IC": "PASS", "SHARPE": "PASS"},
            approval_policy="AUTOMATIC",
        )
        assert (tmp_path / "audit.jsonl").exists()
