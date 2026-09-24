"""
test_lifecycle.py — Phase N/O champion/challenger/shadow + rollback tests.
"""
from __future__ import annotations

import pytest

from src.registry.lifecycle import (
    ChampionChallengerManager,
    ModelNotAllowedToTradeError,
    ShadowComparison,
)
from src.registry.registry import ModelRegistry
from src.schemas.base import PromotionOutcome


def _mgr(tmp_path):
    reg = ModelRegistry(artifacts_path=tmp_path / "artifacts")
    return ChampionChallengerManager(registry=reg, state_path=tmp_path / "roles.json")


def _passing_metrics(ic=0.05):
    return {
        "model_name": "market_regime", "version": "1.0.0-x",
        "ic_mean": ic, "brier_score": 0.18, "max_drawdown": 0.10,
        "oos_count": 120, "look_ahead_validated": True,
        "evidence_level": "LEVEL_A", "final_oos_used_for_selection": False,
        "ic_decay_status": "OK", "feature_drift_severity": "LOW",
        "net_return_validation": 0.03, "turnover_validation": 0.5,
    }


def test_challenger_to_shadow_flow(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.register_challenger("market_regime", "1.0.0-a")
    assert mgr.get_roles("market_regime").challenger_version == "1.0.0-a"
    mgr.promote_to_shadow("market_regime")
    assert mgr.get_roles("market_regime").shadow_version == "1.0.0-a"


def test_shadow_cannot_allocate_capital(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.register_challenger("market_regime", "1.0.0-a")
    mgr.promote_to_shadow("market_regime")
    # Shadow is not champion → cannot allocate capital.
    with pytest.raises(ModelNotAllowedToTradeError):
        mgr.assert_can_allocate_capital("market_regime", "1.0.0-a")


def test_unvalidated_challenger_never_becomes_champion(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.register_challenger("market_regime", "1.0.0-a")
    mgr.promote_to_shadow("market_regime")
    # Failing metrics (negative net return) → REJECTED, not promoted.
    bad = _passing_metrics()
    bad["net_return_validation"] = -0.01
    outcome = mgr.promote_shadow_to_champion("market_regime", bad, approval_token="tok123456789")
    assert outcome == PromotionOutcome.REJECTED
    assert mgr.get_roles("market_regime").champion_version is None


def test_promotion_requires_approval_token(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.register_challenger("market_regime", "1.0.0-a")
    mgr.promote_to_shadow("market_regime")
    # Gates pass but no approval token → BLOCKED, not champion yet.
    outcome = mgr.promote_shadow_to_champion("market_regime", _passing_metrics(), approval_token=None)
    assert outcome == PromotionOutcome.BLOCKED
    assert mgr.get_roles("market_regime").champion_version is None


def test_full_promotion_with_approval(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.register_challenger("market_regime", "1.0.0-a")
    mgr.promote_to_shadow("market_regime")
    outcome = mgr.promote_shadow_to_champion(
        "market_regime", _passing_metrics(), approval_token="approval-token-abcdef"
    )
    assert outcome == PromotionOutcome.PROMOTE
    role = mgr.get_roles("market_regime")
    assert role.champion_version == "1.0.0-a"
    # Champion CAN allocate capital.
    mgr.assert_can_allocate_capital("market_regime", "1.0.0-a")


def test_rollback_restores_previous_champion(tmp_path):
    mgr = _mgr(tmp_path)
    # First champion.
    mgr.register_challenger("market_regime", "1.0.0-a")
    mgr.promote_to_shadow("market_regime")
    mgr.promote_shadow_to_champion("market_regime", _passing_metrics(), approval_token="tok-aaaaaa")
    # Second champion.
    mgr.register_challenger("market_regime", "2.0.0-b")
    mgr.promote_to_shadow("market_regime")
    mgr.promote_shadow_to_champion("market_regime", _passing_metrics(ic=0.06), approval_token="tok-bbbbbb")
    assert mgr.get_roles("market_regime").champion_version == "2.0.0-b"
    # Rollback → previous champion restored.
    restored = mgr.rollback("market_regime", reason="DRIFT_CRITICAL")
    assert restored == "1.0.0-a"
    assert mgr.get_roles("market_regime").champion_version == "1.0.0-a"


def test_rollback_no_previous_returns_none(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.register_challenger("market_regime", "1.0.0-a")
    mgr.promote_to_shadow("market_regime")
    mgr.promote_shadow_to_champion("market_regime", _passing_metrics(), approval_token="tok-cccccc")
    # No previous champion before the first one.
    assert mgr.rollback("market_regime") is None


def test_state_persists_across_instances(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.register_challenger("market_regime", "1.0.0-a")
    mgr.promote_to_shadow("market_regime")
    mgr.promote_shadow_to_champion("market_regime", _passing_metrics(), approval_token="tok-dddddd")
    # New manager instance reloads persisted role state.
    reg = ModelRegistry(artifacts_path=tmp_path / "artifacts")
    mgr2 = ChampionChallengerManager(registry=reg, state_path=tmp_path / "roles.json")
    assert mgr2.get_roles("market_regime").champion_version == "1.0.0-a"


def test_shadow_comparison_recorded(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.register_challenger("market_regime", "1.0.0-a")
    mgr.promote_to_shadow("market_regime")
    cmp = ShadowComparison(
        n_observations=100, shadow_ic=0.05, champion_ic=0.03,
        shadow_hit_rate=0.55, champion_hit_rate=0.52, shadow_beats_champion=True,
    )
    result = mgr.evaluate_shadow("market_regime", cmp)
    assert result.shadow_beats_champion is True
    history = mgr.get_roles("market_regime").history
    assert any(h["event"] == "shadow_evaluated" for h in history)


# ── Explicit no-eligible-model states (mandate §12) ────────────────────────

def test_no_eligible_champion_is_explicit(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.record_no_eligible_champion("market_regime", "IC_BELOW_THRESHOLD")
    role = mgr.get_roles("market_regime")
    assert role.champion_version is None
    assert role.champion_status == "NO_ELIGIBLE_CHAMPION"
    assert role.champion_status_reason == "IC_BELOW_THRESHOLD"


def test_no_eligible_shadow_is_explicit(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.record_no_eligible_shadow("market_regime", "NEGATIVE_NET_SHARPE")
    role = mgr.get_roles("market_regime")
    assert role.shadow_version is None
    assert role.shadow_status == "NO_ELIGIBLE_SHADOW"
    assert role.shadow_status_reason == "NEGATIVE_NET_SHARPE"


def test_default_status_is_not_evaluated(tmp_path):
    mgr = _mgr(tmp_path)
    role = mgr.get_roles("brand_new")
    assert role.champion_status == "NOT_EVALUATED"
    assert role.shadow_status == "NOT_EVALUATED"


def test_no_eligible_states_persist_and_reload(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.record_no_eligible_champion("market_regime", "IC_BELOW_THRESHOLD")
    reg = ModelRegistry(artifacts_path=tmp_path / "artifacts")
    mgr2 = ChampionChallengerManager(registry=reg, state_path=tmp_path / "roles.json")
    role = mgr2.get_roles("market_regime")
    assert role.champion_status == "NO_ELIGIBLE_CHAMPION"
    assert role.champion_status_reason == "IC_BELOW_THRESHOLD"


def test_shadow_promotion_sets_eligible_status(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.register_challenger("market_regime", "1.0.0-a")
    mgr.promote_to_shadow("market_regime")
    assert mgr.get_roles("market_regime").shadow_status == "ELIGIBLE"
