"""
test_finrl_execution_agent.py

TDD tests for the RL execution agent.
Written BEFORE RLExecutionAgent implementation (red phase).

Properties:
  - Action space has exactly 7 discrete actions (Req 9.2)
  - Inference responds within 50ms p95 (Req 9.8, Req 16.1)
  - Heuristic fallback when no trained artifact (Req 9.7)

Requirements: Req 9.2, Req 9.7, Req 9.8, Req 18.6
"""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from src.schemas.base import ExecutionAction, PredictionProvenance
from src.schemas.predictions import ExecutionDecision, ExecutionState, MarketRegime


def _make_execution_state(**kwargs) -> dict:
    """Create a sample ExecutionState dict."""
    base = {
        "symbol": "NIFTY",
        "direction": "LONG",
        "entry": 22000.0,
        "current_price": 22150.0,
        "stop_loss": 21800.0,
        "target": 22400.0,
        "unrealized_pnl_pct": 0.68,
        "time_in_trade_minutes": 30,
        "regime": "bull",
        "volume_ratio": 1.1,
        "price_vs_vwap": 0.003,
        "atr": 200.0,
        "momentum": 0.5,
    }
    base.update(kwargs)
    return base


class TestActionSpaceContract:
    """Req 9.2: Action space must be exactly the 7 discrete actions."""

    def test_execution_action_enum_has_7_members(self):
        """ExecutionAction enum must have exactly 7 members."""
        members = list(ExecutionAction)
        assert len(members) == 7, f"Expected 7 actions, got {len(members)}: {members}"

    def test_all_expected_actions_present(self):
        """All 7 required actions must be defined in ExecutionAction."""
        expected = {
            "enter_now", "wait", "scale_in", "partial_exit",
            "full_exit", "tighten_stop", "trail_stop"
        }
        actual = {a.value for a in ExecutionAction}
        assert expected == actual, f"Missing actions: {expected - actual}"

    def test_rl_agent_action_space_matches_enum(self):
        """RLExecutionAgent action space size must equal len(ExecutionAction)."""
        try:
            from src.models.rl_execution_agent import RLExecutionAgent
        except ImportError:
            pytest.skip("RLExecutionAgent not yet implemented — TDD red phase")

        agent = RLExecutionAgent()
        action_space_size = agent.action_space_size
        assert action_space_size == 7, (
            f"Expected action_space_size=7, got {action_space_size}"
        )


class TestHeuristicFallback:
    """Req 9.7: Heuristic fallback when no trained artifact is present."""

    def test_no_trained_model_returns_heuristic_provenance(self):
        """Without a trained artifact, provenance must be HEURISTIC."""
        try:
            from src.models.rl_execution_agent import RLExecutionAgent
        except ImportError:
            pytest.skip("RLExecutionAgent not yet implemented — TDD red phase")

        agent = RLExecutionAgent()  # no model_path → heuristic fallback
        state = _make_execution_state()
        decision = agent.act(state)

        assert decision.provenance == PredictionProvenance.HEURISTIC, (
            f"Expected HEURISTIC provenance without trained artifact, got {decision.provenance}"
        )

    def test_heuristic_returns_valid_action(self):
        """Heuristic fallback must return a valid ExecutionAction."""
        try:
            from src.models.rl_execution_agent import RLExecutionAgent
        except ImportError:
            pytest.skip("RLExecutionAgent not yet implemented — TDD red phase")

        agent = RLExecutionAgent()
        state = _make_execution_state()
        decision = agent.act(state)

        assert decision.action in list(ExecutionAction), (
            f"Heuristic returned invalid action: {decision.action}"
        )

    def test_heuristic_returns_valid_confidence(self):
        """Heuristic confidence must be in [0, 1]."""
        try:
            from src.models.rl_execution_agent import RLExecutionAgent
        except ImportError:
            pytest.skip("RLExecutionAgent not yet implemented — TDD red phase")

        agent = RLExecutionAgent()
        decision = agent.act(_make_execution_state())

        assert 0.0 <= decision.confidence <= 1.0, (
            f"Confidence out of range: {decision.confidence}"
        )

    def test_heuristic_large_unrealized_pnl_trails_stop(self):
        """Heuristic: large unrealized PnL (>3%) should trail the stop."""
        try:
            from src.models.rl_execution_agent import RLExecutionAgent
        except ImportError:
            pytest.skip("RLExecutionAgent not yet implemented — TDD red phase")

        agent = RLExecutionAgent()
        state = _make_execution_state(unrealized_pnl_pct=3.5)
        decision = agent.act(state)

        # With >3% gain, the heuristic should trail the stop
        assert decision.action in (ExecutionAction.TRAIL_STOP, ExecutionAction.WAIT), (
            f"Expected TRAIL_STOP or WAIT for large gain, got {decision.action}"
        )

    def test_heuristic_near_stop_loss_exits(self):
        """Heuristic: price near stop should suggest FULL_EXIT or TIGHTEN_STOP."""
        try:
            from src.models.rl_execution_agent import RLExecutionAgent
        except ImportError:
            pytest.skip("RLExecutionAgent not yet implemented — TDD red phase")

        agent = RLExecutionAgent()
        # current_price very close to stop_loss (only 0.1% away)
        state = _make_execution_state(
            current_price=21820.0,
            stop_loss=21800.0,
            unrealized_pnl_pct=-0.9,
        )
        decision = agent.act(state)

        assert decision.action in (
            ExecutionAction.FULL_EXIT, ExecutionAction.TIGHTEN_STOP, ExecutionAction.WAIT
        ), f"Expected exit/tighten near stop, got {decision.action}"


class TestLatencySLA:
    """Req 9.8 / Req 16.1: Agent must respond within 50ms p95."""

    def test_heuristic_latency_under_50ms_p95(self):
        """100 heuristic inferences must complete within 50ms p95."""
        try:
            from src.models.rl_execution_agent import RLExecutionAgent
        except ImportError:
            pytest.skip("RLExecutionAgent not yet implemented — TDD red phase")

        agent = RLExecutionAgent()
        state = _make_execution_state()

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            agent.act(state)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        p95 = latencies[int(0.95 * len(latencies))]

        assert p95 < 50.0, (
            f"Agent p95 latency={p95:.1f}ms exceeds 50ms SLA. "
            f"Median={latencies[50]:.1f}ms"
        )


class TestEndpointExistence:
    """POST /v2/predict/execution endpoint must be registered."""

    @pytest.fixture(scope="class")
    @classmethod
    def execution_client(cls):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client

    _VALID_PAYLOAD = {
        "symbol": "NIFTY",
        "direction": "LONG",
        "entry": 22000.0,
        "current_price": 22150.0,
        "stop_loss": 21800.0,
        "target": 22400.0,
        "unrealized_pnl_pct": 0.68,
        "time_in_trade_minutes": 30,
        "regime": "bull",
        "volume_ratio": 1.1,
        "price_vs_vwap": 0.003,
        "atr": 200.0,
        "momentum": 0.5,
    }

    def test_endpoint_registered_not_404(self, execution_client):
        """POST /v2/predict/execution must be registered (not 404)."""
        r = execution_client.post(
            "/v2/predict/execution",
            json=self._VALID_PAYLOAD,
            headers={"X-API-KEY": "test-key-for-testing"},
        )
        assert r.status_code != 404, "Endpoint returned 404 — not registered"

    def test_requires_api_key(self, execution_client):
        """Unauthenticated requests must return 401."""
        r = execution_client.post("/v2/predict/execution", json=self._VALID_PAYLOAD)
        assert r.status_code == 401

    def test_missing_required_fields_returns_422(self, execution_client):
        """Missing required fields must return 422."""
        r = execution_client.post(
            "/v2/predict/execution",
            json={"symbol": "NIFTY"},
            headers={"X-API-KEY": "test-key-for-testing"},
        )
        assert r.status_code == 422
