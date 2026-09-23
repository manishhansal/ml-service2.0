"""
test_e2e_training.py

End-to-end training flow integration tests.
Tests: POST /training/run → GET /training/status/{run_id}

Requirements: Phase 13 integration testing
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

HEADERS = {"X-API-KEY": "test-key-for-testing"}


@pytest.fixture(scope="module")
def training_client():
    from fastapi.testclient import TestClient
    from src.main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestTrainingFlow:
    _CONFIG = {
        "model_name": "market_regime",
        "feature_version": "latest",
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
    }

    def test_training_run_returns_200(self, training_client):
        r = training_client.post("/training/run", json=self._CONFIG, headers=HEADERS)
        assert r.status_code == 200

    def test_training_run_returns_run_id(self, training_client):
        r = training_client.post("/training/run", json=self._CONFIG, headers=HEADERS)
        if r.status_code == 200:
            body = r.json()
            assert "run_id" in body
            assert len(body["run_id"]) > 0

    def test_training_run_returns_model_name(self, training_client):
        r = training_client.post("/training/run", json=self._CONFIG, headers=HEADERS)
        if r.status_code == 200:
            body = r.json()
            assert body.get("model_name") == "market_regime"

    def test_training_status_returns_queued(self, training_client):
        r = training_client.post("/training/run", json=self._CONFIG, headers=HEADERS)
        if r.status_code == 200:
            run_id = r.json()["run_id"]
            status_r = training_client.get(f"/training/status/{run_id}", headers=HEADERS)
            assert status_r.status_code == 200
            assert status_r.json()["status"] == "QUEUED"

    def test_training_status_unknown_run_returns_404(self, training_client):
        r = training_client.get("/training/status/nonexistent-run-id", headers=HEADERS)
        assert r.status_code == 404

    def test_training_status_contains_run_id(self, training_client):
        r = training_client.post("/training/run", json=self._CONFIG, headers=HEADERS)
        if r.status_code == 200:
            run_id = r.json()["run_id"]
            status_r = training_client.get(f"/training/status/{run_id}", headers=HEADERS)
            if status_r.status_code == 200:
                assert status_r.json().get("run_id") == run_id

    def test_multiple_training_runs_have_distinct_run_ids(self, training_client):
        r1 = training_client.post("/training/run", json=self._CONFIG, headers=HEADERS)
        r2 = training_client.post("/training/run", json=self._CONFIG, headers=HEADERS)
        if r1.status_code == 200 and r2.status_code == 200:
            assert r1.json()["run_id"] != r2.json()["run_id"]
