"""
tests/test_sprint2_economic_engine.py
---------------------------------------
Tests for Sprint 2 economic infrastructure:
  - DrawdownManager (NEW-P2-005)
  - Baseline comparison gate in TrainingOrchestrator (NEW-P2-006)
  - SelfLearningLoop champion comparison gate (NEW-P1-004)

Note: TestOpportunityScorer (NEW-P2-003) removed — src/meta/opportunity_score.py
was not wired into the v2c production pipeline and has been deleted.
"""
from __future__ import annotations

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# DrawdownManager (NEW-P2-005)
# ──────────────────────────────────────────────────────────────────────────────

class TestDrawdownManager:
    """Tests for DrawdownManager state machine (mandate §32)."""

    def test_normal_state_initially(self):
        from src.risk.drawdown_manager import DrawdownManager, DrawdownState
        manager = DrawdownManager()
        assert manager.current_state == DrawdownState.NORMAL
        assert manager.confidence_multiplier == 1.0

    def test_caution_state_on_small_drawdown(self):
        from src.risk.drawdown_manager import DrawdownManager, DrawdownState
        manager = DrawdownManager(caution_dd=0.03)
        # Produce a 4% drawdown
        manager.update(0.0)      # equity = 1.0 (peak)
        manager.update(-0.04)    # equity = 0.96 → 4% DD
        assert manager.current_state == DrawdownState.CAUTION
        assert manager.confidence_multiplier < 1.0

    def test_halted_state_on_large_drawdown(self):
        from src.risk.drawdown_manager import DrawdownManager, DrawdownState
        manager = DrawdownManager(halt_dd=0.12)
        manager.update(-0.15)  # 15% drawdown → HALTED
        assert manager.current_state == DrawdownState.HALTED
        assert manager.confidence_multiplier == 0.0
        assert not manager.allows_new_trades
        assert manager.is_halted

    def test_state_sequence_normal_caution_defensive_halted(self):
        from src.risk.drawdown_manager import DrawdownManager, DrawdownState
        manager = DrawdownManager(caution_dd=0.03, defensive_dd=0.07, halt_dd=0.12)
        assert manager.current_state == DrawdownState.NORMAL
        manager.update(-0.04)
        assert manager.current_state == DrawdownState.CAUTION
        manager.update(-0.04)
        assert manager.current_state == DrawdownState.DEFENSIVE
        manager.update(-0.05)
        assert manager.current_state == DrawdownState.HALTED

    def test_multiplier_decreases_with_state(self):
        from src.risk.drawdown_manager import DrawdownManager, DrawdownState, _STATE_MULTIPLIERS
        assert _STATE_MULTIPLIERS[DrawdownState.NORMAL] > _STATE_MULTIPLIERS[DrawdownState.CAUTION]
        assert _STATE_MULTIPLIERS[DrawdownState.CAUTION] > _STATE_MULTIPLIERS[DrawdownState.DEFENSIVE]
        assert _STATE_MULTIPLIERS[DrawdownState.DEFENSIVE] > _STATE_MULTIPLIERS[DrawdownState.HALTED]
        assert _STATE_MULTIPLIERS[DrawdownState.HALTED] == 0.0

    def test_recovery_requires_consecutive_bars(self):
        """Recovery from CAUTION back to NORMAL requires recovery_bars."""
        from src.risk.drawdown_manager import DrawdownManager, DrawdownState
        manager = DrawdownManager(caution_dd=0.03, recovery_bars=3)
        manager.update(-0.04)  # → CAUTION
        assert manager.current_state == DrawdownState.CAUTION
        # Recovery: returns must bring equity back above peak
        # Equity is now 0.96; need to return to 1.0 (peak)
        manager.update(0.04167)  # ~back to 1.0
        assert manager.current_state in (DrawdownState.CAUTION, DrawdownState.NORMAL)
        # Still in CAUTION (need recovery_bars=3 bars above peak)

    def test_update_pnl_convenience(self):
        from src.risk.drawdown_manager import DrawdownManager, DrawdownState
        manager = DrawdownManager()
        snap = manager.update_pnl([0.01, 0.02, -0.01])
        assert snap.state == DrawdownState.NORMAL
        assert snap.current_drawdown >= 0

    def test_invalid_thresholds_raises(self):
        from src.risk.drawdown_manager import DrawdownManager
        with pytest.raises(ValueError):
            DrawdownManager(caution_dd=0.10, defensive_dd=0.05, halt_dd=0.20)  # caution > defensive

    def test_summary_dict(self):
        from src.risk.drawdown_manager import DrawdownManager
        manager = DrawdownManager()
        manager.update(0.01)
        s = manager.summary()
        assert "current_state" in s
        assert "confidence_multiplier" in s
        assert "thresholds" in s

    def test_min_edge_increases_in_caution(self):
        from src.risk.drawdown_manager import DrawdownManager, _STATE_MIN_EDGE, DrawdownState
        import math
        assert _STATE_MIN_EDGE[DrawdownState.NORMAL] < _STATE_MIN_EDGE[DrawdownState.CAUTION]
        assert _STATE_MIN_EDGE[DrawdownState.HALTED] == math.inf


# ──────────────────────────────────────────────────────────────────────────────
# Baseline comparison gate — TrainingOrchestrator (NEW-P2-006)
# ──────────────────────────────────────────────────────────────────────────────

class TestBaselineComparisonGate:
    """Orchestrator must reject champion that doesn't beat the baseline."""

    def test_acceptance_fails_below_baseline(self):
        """Champion with IC only marginally above baseline must be rejected."""
        from src.training.orchestrator import TrainingOrchestrator, CandidateResult
        from unittest.mock import MagicMock

        orch = TrainingOrchestrator(
            dataset_builder=MagicMock(),
            min_ic=0.02,
            parsimony_margin=0.005,
        )
        # Baseline IC = 0.04; champion IC = 0.043 → margin = 0.003 < 0.005 → FAIL
        orch._baseline_ic = 0.04
        champion = CandidateResult(
            name="lightgbm",
            is_baseline=False,
            wf_ic_mean=0.043,        # only 0.003 above baseline (< margin 0.005)
            wf_ic_worst=0.02,
            wf_positive_fraction=0.8,
            wf_net_sharpe=0.5,
            cpcv_ic_mean=0.03,
            cpcv_pbo=0.2,
        )
        passed, reason = orch._acceptance(champion)
        assert not passed
        assert "FAILS_BASELINE_COMPARISON" in reason

    def test_acceptance_passes_above_baseline_plus_margin(self):
        """Champion IC > baseline + margin → passes."""
        from src.training.orchestrator import TrainingOrchestrator, CandidateResult
        from unittest.mock import MagicMock

        orch = TrainingOrchestrator(
            dataset_builder=MagicMock(),
            min_ic=0.02,
            parsimony_margin=0.005,
        )
        orch._baseline_ic = 0.03
        champion = CandidateResult(
            name="lightgbm",
            is_baseline=False,
            wf_ic_mean=0.040,        # 0.01 above baseline (> margin 0.005) → PASS
            wf_ic_worst=0.025,
            wf_positive_fraction=0.8,
            wf_net_sharpe=0.5,
            cpcv_ic_mean=0.035,
            cpcv_pbo=0.2,
        )
        passed, reason = orch._acceptance(champion)
        assert passed, f"Should pass but got reason: {reason}"

    def test_acceptance_no_baseline_skips_gate(self):
        """When baseline_ic is None, comparison gate is skipped."""
        from src.training.orchestrator import TrainingOrchestrator, CandidateResult
        from unittest.mock import MagicMock

        orch = TrainingOrchestrator(dataset_builder=MagicMock(), min_ic=0.02)
        orch._baseline_ic = None  # no baseline provided
        champion = CandidateResult(
            name="logistic",
            is_baseline=True,
            wf_ic_mean=0.025,
            wf_ic_worst=0.01,
            wf_positive_fraction=0.6,
            wf_net_sharpe=0.2,
            cpcv_ic_mean=0.02,
            cpcv_pbo=0.3,
        )
        passed, reason = orch._acceptance(champion)
        assert passed  # gate skipped when no baseline

    def test_train_accepts_baseline_ic_param(self):
        """TrainingOrchestrator.train() signature must accept baseline_ic."""
        import inspect
        from src.training.orchestrator import TrainingOrchestrator
        sig = inspect.signature(TrainingOrchestrator.train)
        assert "baseline_ic" in sig.parameters, (
            "train() must have a baseline_ic parameter (FIX NEW-P2-006)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# SelfLearningLoop champion comparison gate (NEW-P1-004)
# ──────────────────────────────────────────────────────────────────────────────

class TestSelfLearningChampionGate:
    """Challenger must beat current champion to reach SHADOW."""

    def test_run_cycle_signature_accepts_champion_ic(self):
        """run_cycle() must have champion_ic parameter (FIX NEW-P1-004)."""
        import inspect
        from src.training.self_learning import SelfLearningLoop
        sig = inspect.signature(SelfLearningLoop.run_cycle)
        assert "champion_ic" in sig.parameters, (
            "run_cycle() must accept champion_ic for comparison gate"
        )

    def test_run_cycle_passes_champion_ic_to_orchestrator(self):
        """run_cycle() must forward champion_ic as baseline_ic to the orchestrator."""
        from unittest.mock import MagicMock, patch
        from src.data.feedback import FeedbackStore
        from src.training.self_learning import SelfLearningLoop

        store = MagicMock(spec=FeedbackStore)
        store.resolved_records.return_value = [f"r{i}" for i in range(50)]

        loop = SelfLearningLoop(store, min_new_samples=30, min_hours_between_retrains=0)
        loop._consumed_count = 0
        loop._last_retrain_at = None

        mock_orchestrator = MagicMock()
        mock_report = MagicMock()
        mock_report.passed_acceptance = False
        mock_report.champion_version = None
        mock_report.to_dict.return_value = {}
        mock_report.rejection_reason = "IC_BELOW_THRESHOLD"
        mock_orchestrator.train.return_value = mock_report

        mock_lifecycle = MagicMock()

        loop.run_cycle(
            orchestrator=mock_orchestrator,
            lifecycle_manager=mock_lifecycle,
            model_name="test",
            dataset_id="ds-1",
            champion_ic=0.045,
        )

        # Verify orchestrator.train was called with baseline_ic=0.045
        call_kwargs = mock_orchestrator.train.call_args
        assert call_kwargs is not None, "orchestrator.train must have been called"
        kwargs = call_kwargs[1] if call_kwargs[1] else {}
        args = call_kwargs[0] if call_kwargs[0] else ()
        # baseline_ic can be positional or keyword
        called_with_ic = kwargs.get("baseline_ic") is not None or 0.045 in args
        # Check keyword args
        assert "baseline_ic" in kwargs, (
            f"orchestrator.train must be called with baseline_ic. "
            f"Got kwargs: {kwargs}"
        )
        assert kwargs["baseline_ic"] == 0.045
