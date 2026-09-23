"""
test_performance.py

Performance and concurrency baseline tests.

Requirements: Req 16.8, Phase 13 integration testing
"""
from __future__ import annotations

import os
import time
import concurrent.futures

import pytest

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

HEADERS = {"X-API-KEY": "test-key-for-testing"}


@pytest.fixture(scope="module")
def perf_client():
    from fastapi.testclient import TestClient
    from src.main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestConcurrentRequests:
    """Verify service handles multiple concurrent requests without degradation."""

    def test_health_endpoint_concurrent(self, perf_client):
        """10 concurrent health checks should all succeed."""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(perf_client.get, "/health") for _ in range(10)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result().status_code)
        assert all(s == 200 for s in results), (
            f"Not all health checks succeeded: {results}"
        )

    def test_training_endpoint_concurrent(self, perf_client):
        """5 concurrent training run requests should all return 200."""
        payload = {
            "model_name": "market_regime",
            "feature_version": "latest",
            "start_date": "2023-01-01",
            "end_date": "2024-01-01",
        }
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    perf_client.post,
                    "/training/run",
                    json=payload,
                    headers=HEADERS,
                )
                for _ in range(5)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result().status_code)
        assert all(s == 200 for s in results), (
            f"Not all training requests succeeded: {results}"
        )

    def test_meta_decide_endpoint_concurrent(self, perf_client):
        """5 concurrent meta/decide requests should all succeed."""
        payload = {"symbol": "NIFTY", "regime": "bull"}
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    perf_client.post,
                    "/v2/meta/decide",
                    json=payload,
                    headers=HEADERS,
                )
                for _ in range(5)
            ]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result().status_code)
        assert all(s == 200 for s in results), (
            f"Not all meta/decide requests succeeded: {results}"
        )


class TestStubEndpointLatency:
    """Stub endpoints should respond within generous bounds."""

    def test_health_latency_under_100ms(self, perf_client):
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            perf_client.get("/health")
            latencies.append((time.perf_counter() - start) * 1000)
        p95 = sorted(latencies)[int(0.95 * len(latencies))]
        assert p95 < 100, f"Health p95={p95:.1f}ms exceeds 100ms"

    def test_meta_decide_latency_under_500ms(self, perf_client):
        """Meta decide should respond within 500ms p95."""
        payload = {"symbol": "NIFTY", "regime": "bull"}
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            perf_client.post("/v2/meta/decide", json=payload, headers=HEADERS)
            latencies.append((time.perf_counter() - start) * 1000)
        p95 = sorted(latencies)[int(0.95 * len(latencies))]
        assert p95 < 500, f"Meta/decide p95={p95:.1f}ms exceeds 500ms"
