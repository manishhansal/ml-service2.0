from __future__ import annotations

import os

import pytest

# Set required env vars before any test imports src modules
os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")


@pytest.fixture(scope="session")
def app_client():
    """Session-scoped TestClient for integration tests within the same process."""
    from fastapi.testclient import TestClient

    os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
    os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")
    from src.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def auth_headers():
    """Standard auth headers for API calls."""
    return {"X-API-KEY": "test-key-for-testing"}
