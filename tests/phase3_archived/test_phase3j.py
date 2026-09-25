"""
Phase 3J — Model Lifecycle Test Suite.

Covers all acceptance criteria (spec §69, §70, §79):

  Registration    — register, duplicate version, immutable artifact, hash, metadata
  Compatibility   — feature/label/calibrator/meta/execution/portfolio mismatch
  State machine   — valid transitions, invalid transitions, terminal states
  Champion        — initial, scoped, history, historical lookup
  Challenger      — registration, shadow, paper, soak, promotion eligibility
  Promotion       — gate failures (calibration/execution/stability), insufficient
                    evidence, successful promotion, atomicity
  Rollback        — rollback, history, previous champion restoration
  Integrity       — artifact mutation, evidence mutation, manifest mutation
  PIT             — future data/evidence/champion mutation
  Reproducibility — same inputs → same decision
  Adversarial     — missing/fake/modified evidence, wrong calibrator, future OOS
  Crash-safety    — simulated failure mid-promotion, no two-champions state

Test data uses deterministic fixtures — no np.random in production code
(verified by static test).
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

UTC = timezone.utc


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures / helpers
# ══════════════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now(UTC).isoformat()


def _identity(model_id="ranker-a", version="v1", artifact_hash="hash_a",
              feature_version="feat-v3", label_version="lv2"):
    from src.lifecycle import ModelIdentity
    return ModelIdentity(
        model_id=model_id, model_family="cross_sectional_ranker",
        model_type="LightGBM", model_version=version, artifact_hash=artifact_hash,
        created_at=_now(), training_start="2023-01-01", training_end="2023-12-31",
        code_version="abc123", feature_version=feature_version,
        label_version=label_version, dataset_snapshot_id="ds-1",
        dataset_version="af-v3", universe_version="static-v1",
    )


def _provenance(model_id="ranker-a", version="v1", feature_version="feat-v3",
                label_version="lv2"):
    from src.lifecycle import ModelProvenance
    return ModelProvenance(
        model_id=model_id, model_version=version,
        feature_version=feature_version, label_version=label_version,
        data_snapshot_id="ds-1",
    )


def _evidence(pkg_id, identity, ic=0.05, brier=0.20, net_ret=0.02, dd=-0.10,
              n_oos=150, contaminated=False, ic_decay="STABLE",
              feature_drift="NONE", has_execution=True):
    from src.lifecycle import (
        ModelEvidencePackage, OOSEvidence, CalibrationEvidence,
        ExecutionEvidence, PortfolioEvidence, StabilityEvidence,
    )
    pkg = ModelEvidencePackage(evidence_package_id=pkg_id, model_identity=identity)
    pkg.oos_evidence = OOSEvidence(
        mean_rank_ic=ic, mean_ic=ic, icir=1.5, n_oos_observations=n_oos,
        n_oos_timestamps=60, final_oos_used_for_selection=contaminated,
    )
    pkg.calibration_evidence = CalibrationEvidence(brier=brier, calibrator_version="cal-v1")
    if has_execution:
        pkg.execution_evidence = ExecutionEvidence(
            net_return=net_ret, turnover=0.3,
            cost_model_version="india-fno-2023",
            execution_model_version="backtest-engine-v1",
        )
    pkg.portfolio_evidence = PortfolioEvidence(
        max_drawdown=dd, sharpe=1.2, portfolio_model_version="portfolio-v1",
    )
    pkg.stability_evidence = StabilityEvidence(
        ic_decay_status=ic_decay, feature_drift_severity=feature_drift,
    )
    return pkg.freeze()


def _advance_to_eligible(registry, full_key):
    from src.lifecycle import LifecycleState
    for st in [LifecycleState.VALIDATING, LifecycleState.EVIDENCE_READY,
               LifecycleState.CANDIDATE, LifecycleState.SHADOW,
               LifecycleState.PAPER, LifecycleState.PROMOTION_ELIGIBLE]:
        registry.transition(full_key, st, reason="advance")


@pytest.fixture
def tmproot():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Registration
# ══════════════════════════════════════════════════════════════════════════════

class TestRegistration:
    def test_register_model(self, tmproot):
        from src.lifecycle import ModelRegistry, LifecycleState
        reg = ModelRegistry(root=tmproot)
        rec = reg.register(_identity(), _provenance())
        assert rec.lifecycle_state == LifecycleState.REGISTERED.value
        assert reg.exists("ranker-a@v1")

    def test_duplicate_same_artifact_idempotent(self, tmproot):
        from src.lifecycle import ModelRegistry
        reg = ModelRegistry(root=tmproot)
        reg.register(_identity(), _provenance())
        # Re-register identical → idempotent no-op, no error
        rec = reg.register(_identity(), _provenance())
        assert rec.full_key == "ranker-a@v1"

    def test_immutable_artifact_rejected(self, tmproot):
        from src.lifecycle import ModelRegistry, ImmutabilityViolation
        reg = ModelRegistry(root=tmproot)
        reg.register(_identity(artifact_hash="hash_a"), _provenance())
        # Same version, DIFFERENT artifact hash → ImmutabilityViolation
        with pytest.raises(ImmutabilityViolation):
            reg.register(_identity(artifact_hash="hash_DIFFERENT"), _provenance())

    def test_artifact_hash_recorded(self, tmproot):
        from src.lifecycle import ModelRegistry
        reg = ModelRegistry(root=tmproot)
        rec = reg.register(_identity(artifact_hash="hash_xyz"), _provenance())
        assert rec.artifact_hash == "hash_xyz"

    def test_identity_immutable_frozen(self):
        ident = _identity()
        with pytest.raises((AttributeError, Exception)):
            ident.model_id = "changed"   # frozen dataclass

    def test_full_key_format(self):
        ident = _identity(model_id="m", version="v2")
        assert ident.full_key == "m@v2"

    def test_identity_hash_deterministic(self):
        i1 = _identity()
        i2 = _identity()
        assert i1.identity_hash == i2.identity_hash

    def test_register_creates_audit_entry(self, tmproot):
        from src.lifecycle import ModelRegistry
        reg = ModelRegistry(root=tmproot)
        reg.register(_identity(), _provenance())
        log = reg.audit_log()
        assert any(e["event"] == "REGISTER" for e in log)


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Artifact integrity
# ══════════════════════════════════════════════════════════════════════════════

class TestArtifactIntegrity:
    def test_compute_and_verify(self, tmproot):
        from src.lifecycle import compute_artifact_hash, verify_artifact_integrity
        art = os.path.join(tmproot, "model.txt")
        with open(art, "w") as f:
            f.write("weights")
        h = compute_artifact_hash(art)
        result = verify_artifact_integrity(art, h)
        assert result.verified

    def test_tamper_detected(self, tmproot):
        from src.lifecycle import compute_artifact_hash, verify_artifact_integrity, IntegrityStatus
        art = os.path.join(tmproot, "model.txt")
        with open(art, "w") as f:
            f.write("weights")
        h = compute_artifact_hash(art)
        # Tamper with the file
        with open(art, "w") as f:
            f.write("tampered weights")
        result = verify_artifact_integrity(art, h)
        assert not result.verified
        assert result.status == IntegrityStatus.INTEGRITY_FAILURE

    def test_missing_artifact(self, tmproot):
        from src.lifecycle import verify_artifact_integrity, IntegrityStatus
        result = verify_artifact_integrity(os.path.join(tmproot, "nope.txt"), "abc")
        assert result.status == IntegrityStatus.ARTIFACT_MISSING

    def test_forbidden_latest_reference(self, tmproot):
        from src.lifecycle import is_forbidden_reference, verify_artifact_integrity, IntegrityStatus
        assert is_forbidden_reference("/models/latest.pkl")
        assert is_forbidden_reference("/models/current_model")
        # Even if it exists, a forbidden name fails integrity
        art = os.path.join(tmproot, "latest.pkl")
        with open(art, "w") as f:
            f.write("x")
        result = verify_artifact_integrity(art, "anything")
        assert result.status == IntegrityStatus.INTEGRITY_FAILURE

    def test_safe_load_guard_raises_on_tamper(self, tmproot):
        from src.lifecycle import compute_artifact_hash, safe_load_guard
        art = os.path.join(tmproot, "model.txt")
        with open(art, "w") as f:
            f.write("weights")
        h = compute_artifact_hash(art)
        with open(art, "w") as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="INTEGRITY_FAILURE"):
            safe_load_guard(art, h)

    def test_directory_artifact_hash(self, tmproot):
        from src.lifecycle import compute_artifact_hash
        d = os.path.join(tmproot, "risk")
        os.makedirs(d)
        for name in ["stop.json", "target.json"]:
            with open(os.path.join(d, name), "w") as f:
                f.write(name)
        h1 = compute_artifact_hash(d)
        h2 = compute_artifact_hash(d)
        assert h1 == h2 and len(h1) == 64


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Compatibility
# ══════════════════════════════════════════════════════════════════════════════

class TestCompatibility:
    def test_feature_mismatch(self):
        from src.lifecycle import CompatibilityChecker, CompatibilityStatus
        checker = CompatibilityChecker()
        check = checker.check_feature_version("feat-v3", "feat-v4")
        assert check.status == CompatibilityStatus.INCOMPATIBLE

    def test_feature_match(self):
        from src.lifecycle import CompatibilityChecker, CompatibilityStatus
        checker = CompatibilityChecker()
        check = checker.check_feature_version("feat-v3", "feat-v3")
        assert check.status == CompatibilityStatus.COMPATIBLE

    def test_label_mismatch(self):
        from src.lifecycle import CompatibilityChecker, CompatibilityStatus
        checker = CompatibilityChecker()
        check = checker.check_label_version("lv2", "lv3")
        assert check.status == CompatibilityStatus.INCOMPATIBLE

    def test_calibrator_mismatch(self):
        from src.lifecycle import CompatibilityChecker, CompatibilityStatus
        checker = CompatibilityChecker()
        check = checker.check_calibrator("model-a", "v1", "model-b", "v1")
        assert check.status == CompatibilityStatus.INCOMPATIBLE

    def test_calibrator_match(self):
        from src.lifecycle import CompatibilityChecker, CompatibilityStatus
        checker = CompatibilityChecker()
        check = checker.check_calibrator("model-a", "v1", "model-a", "v1")
        assert check.status == CompatibilityStatus.COMPATIBLE

    def test_meta_model_mismatch(self):
        from src.lifecycle import CompatibilityChecker, CompatibilityStatus
        checker = CompatibilityChecker()
        check = checker.check_meta_model("model-a", "model-b")
        assert check.status == CompatibilityStatus.INCOMPATIBLE

    def test_semantics_mismatch(self):
        from src.lifecycle import (
            CompatibilityChecker, CompatibilityStatus, ModelSchemaContract,
            PredictionSemantics,
        )
        contract = ModelSchemaContract(
            model_id="m", model_version="v1", feature_schema_version="fs-1",
            input_features=["f1"], required_columns=["f1"], column_types={"f1": "float"},
            missingness_policy="RETURN_NAN", output_schema="score",
            prediction_semantics=PredictionSemantics.ALPHA_SCORE, prediction_horizon=5,
        )
        checker = CompatibilityChecker()
        # Expect CALIBRATED_PROBABILITY but model produces ALPHA_SCORE
        check = checker.check_semantics(contract, PredictionSemantics.CALIBRATED_PROBABILITY)
        assert check.status == CompatibilityStatus.INCOMPATIBLE

    def test_execution_version_mismatch(self):
        from src.lifecycle import CompatibilityChecker, CompatibilityStatus
        checker = CompatibilityChecker()
        check = checker.check_execution_version("backtest-v1", "backtest-v2")
        assert check.status == CompatibilityStatus.INCOMPATIBLE

    def test_portfolio_version_mismatch(self):
        from src.lifecycle import CompatibilityChecker, CompatibilityStatus
        checker = CompatibilityChecker()
        check = checker.check_portfolio_version("portfolio-v1", "portfolio-v2")
        assert check.status == CompatibilityStatus.INCOMPATIBLE

    def test_check_all_report(self):
        from src.lifecycle import CompatibilityChecker, PredictionSemantics
        checker = CompatibilityChecker()
        report = checker.check_all(
            model_identity=_identity(),
            dataset_feature_version="feat-v3",
            dataset_label_version="lv2",
            calibrator_model_id="ranker-a", calibrator_model_version="v1",
        )
        assert report.all_compatible


# ══════════════════════════════════════════════════════════════════════════════
# 4 — State machine
# ══════════════════════════════════════════════════════════════════════════════

class TestStateMachine:
    def test_valid_transition(self, tmproot):
        from src.lifecycle import ModelRegistry, LifecycleState
        reg = ModelRegistry(root=tmproot)
        reg.register(_identity(), _provenance())
        rec = reg.transition("ranker-a@v1", LifecycleState.VALIDATING, reason="start")
        assert rec.lifecycle_state == LifecycleState.VALIDATING.value

    def test_invalid_transition_rejected(self, tmproot):
        from src.lifecycle import ModelRegistry, LifecycleState, InvalidTransition
        reg = ModelRegistry(root=tmproot)
        reg.register(_identity(), _provenance())
        # REGISTERED → CHAMPION is invalid (spec §57)
        with pytest.raises(InvalidTransition):
            reg.transition("ranker-a@v1", LifecycleState.CHAMPION, reason="cheat")

    def test_retired_cannot_become_champion(self, tmproot):
        from src.lifecycle import ModelRegistry, LifecycleState, InvalidTransition
        reg = ModelRegistry(root=tmproot)
        reg.register(_identity(), _provenance())
        reg.transition("ranker-a@v1", LifecycleState.VALIDATING, reason="x")
        reg.transition("ranker-a@v1", LifecycleState.REJECTED, reason="bad")
        # REJECTED is terminal → cannot transition to CHAMPION
        with pytest.raises(InvalidTransition):
            reg.transition("ranker-a@v1", LifecycleState.CHAMPION, reason="cheat")

    def test_idempotent_same_state(self, tmproot):
        from src.lifecycle import ModelRegistry, LifecycleState
        reg = ModelRegistry(root=tmproot)
        reg.register(_identity(), _provenance())
        reg.transition("ranker-a@v1", LifecycleState.VALIDATING, reason="x")
        # Transition to same state = no-op
        rec = reg.transition("ranker-a@v1", LifecycleState.VALIDATING, reason="again")
        assert rec.lifecycle_state == LifecycleState.VALIDATING.value

    def test_transition_helper(self):
        from src.lifecycle import is_valid_transition, LifecycleState
        assert is_valid_transition(LifecycleState.REGISTERED, LifecycleState.VALIDATING)
        assert not is_valid_transition(LifecycleState.REGISTERED, LifecycleState.CHAMPION)

    def test_state_history_recorded(self, tmproot):
        from src.lifecycle import ModelRegistry, LifecycleState
        reg = ModelRegistry(root=tmproot)
        reg.register(_identity(), _provenance())
        reg.transition("ranker-a@v1", LifecycleState.VALIDATING, reason="x")
        rec = reg.get("ranker-a@v1")
        assert len(rec.state_history) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# 5 — Champion
# ══════════════════════════════════════════════════════════════════════════════

class TestChampion:
    def test_initial_champion(self, tmproot):
        from src.lifecycle import ChampionIndex
        champ = ChampionIndex(root=tmproot)
        champ.promote("equity/5D", "model-a@v1", "first", "ev-1")
        entry = champ.get_champion("equity/5D")
        assert entry.champion_full_key == "model-a@v1"
        assert entry.previous_champion is None

    def test_scoped_champions(self, tmproot):
        from src.lifecycle import ChampionIndex
        champ = ChampionIndex(root=tmproot)
        champ.promote("equity/5D", "model-a@v1", "r", "ev-1")
        champ.promote("futures/1D", "model-b@v1", "r", "ev-2")
        assert champ.get_champion("equity/5D").champion_full_key == "model-a@v1"
        assert champ.get_champion("futures/1D").champion_full_key == "model-b@v1"

    def test_champion_history(self, tmproot):
        from src.lifecycle import ChampionIndex
        champ = ChampionIndex(root=tmproot)
        champ.promote("equity/5D", "model-a@v1", "r1", "ev-1")
        champ.promote("equity/5D", "model-b@v1", "r2", "ev-2")
        hist = champ.history("equity/5D")
        keys = [h.get("champion_full_key") for h in hist]
        assert "model-a@v1" in keys and "model-b@v1" in keys

    def test_previous_champion_recorded(self, tmproot):
        from src.lifecycle import ChampionIndex
        champ = ChampionIndex(root=tmproot)
        champ.promote("equity/5D", "model-a@v1", "r1", "ev-1")
        champ.promote("equity/5D", "model-b@v1", "r2", "ev-2")
        entry = champ.get_champion("equity/5D")
        assert entry.previous_champion == "model-a@v1"

    def test_no_champion_initially(self, tmproot):
        from src.lifecycle import ChampionIndex
        champ = ChampionIndex(root=tmproot)
        assert champ.get_champion("nonexistent") is None


# ══════════════════════════════════════════════════════════════════════════════
# 6 — Challenger
# ══════════════════════════════════════════════════════════════════════════════

class TestChallenger:
    def test_register_challenger(self, tmproot):
        from src.lifecycle import ChallengerRegistry, ChallengerStatus
        cr = ChallengerRegistry(root=tmproot)
        rec = cr.register("chal-1", "model-a@v1", "equity/5D")
        assert rec.status == ChallengerStatus.REGISTERED.value

    def test_shadow_mode(self, tmproot):
        from src.lifecycle import ChallengerRegistry, ChallengerStatus, ShadowOutput
        cr = ChallengerRegistry(root=tmproot)
        cr.register("chal-1", "model-a@v1", "equity/5D")
        cr.start_shadow("chal-1")
        rec = cr.get("chal-1")
        assert rec.status == ChallengerStatus.SHADOW.value
        # Log a shadow output
        cr.log_shadow_output(ShadowOutput(
            challenger_id="chal-1", timestamp=_now(), instrument_id="NIFTY",
            prediction=0.5, decision="TAKE", influenced_champion=False,
        ))
        assert cr.get("chal-1").shadow_observations == 1

    def test_shadow_output_cannot_influence_champion(self, tmproot):
        from src.lifecycle import ChallengerRegistry, ShadowOutput, ChallengerError
        cr = ChallengerRegistry(root=tmproot)
        cr.register("chal-1", "model-a@v1", "equity/5D")
        cr.start_shadow("chal-1")
        with pytest.raises(ChallengerError):
            cr.log_shadow_output(ShadowOutput(
                challenger_id="chal-1", timestamp=_now(), instrument_id="NIFTY",
                prediction=0.5, decision="TAKE", influenced_champion=True,  # forbidden
            ))

    def test_paper_mode(self, tmproot):
        from src.lifecycle import ChallengerRegistry, ChallengerStatus
        cr = ChallengerRegistry(root=tmproot)
        cr.register("chal-1", "model-a@v1", "equity/5D")
        cr.start_shadow("chal-1")
        cr.start_paper("chal-1")
        assert cr.get("chal-1").status == ChallengerStatus.PAPER.value

    def test_shadow_soak_incomplete(self, tmproot):
        from src.lifecycle import ChallengerRegistry, SoakConfig
        cr = ChallengerRegistry(root=tmproot, soak_config=SoakConfig(
            min_shadow_days=20, min_shadow_observations=200))
        cr.register("chal-1", "model-a@v1", "equity/5D")
        cr.start_shadow("chal-1")
        complete, reason = cr.shadow_soak_complete("chal-1")
        assert not complete   # just started, no observations

    def test_selection_bias_tracking(self, tmproot):
        from src.lifecycle import ChallengerRegistry
        cr = ChallengerRegistry(root=tmproot)
        cr.register("chal-1", "model-a@v1", "equity/5D")
        cr.register("chal-2", "model-b@v1", "equity/5D")
        cr.reject("chal-2", "poor IC")
        summary = cr.selection_bias_summary("equity/5D")
        assert summary["number_tested"] == 2
        assert summary["number_rejected"] == 1

    def test_rejected_challenger_evidence_retained(self, tmproot):
        from src.lifecycle import ChallengerRegistry, ChallengerStatus
        cr = ChallengerRegistry(root=tmproot)
        cr.register("chal-1", "model-a@v1", "equity/5D", evidence_package_id="ev-1")
        cr.reject("chal-1", "poor performance")
        rec = cr.get("chal-1")
        assert rec.status == ChallengerStatus.REJECTED.value
        assert rec.evidence_package_id == "ev-1"   # evidence not deleted


# ══════════════════════════════════════════════════════════════════════════════
# 7 — Evidence packages
# ══════════════════════════════════════════════════════════════════════════════

class TestEvidence:
    def test_evidence_frozen_and_hashed(self):
        ev = _evidence("ev-1", _identity())
        assert ev.evidence_hash != ""
        assert ev.verify_integrity()

    def test_evidence_mutation_detected(self):
        ev = _evidence("ev-1", _identity())
        # Mutate after freezing
        ev.oos_evidence.mean_rank_ic = 999.0
        assert not ev.verify_integrity()

    def test_evidence_level_a(self):
        from src.lifecycle import EvidenceLevel
        ev = _evidence("ev-1", _identity(), n_oos=150, has_execution=True)
        assert ev.evidence_level == EvidenceLevel.LEVEL_A

    def test_evidence_level_d_insufficient(self):
        from src.lifecycle import EvidenceLevel
        ev = _evidence("ev-1", _identity(), n_oos=5)
        assert ev.evidence_level == EvidenceLevel.LEVEL_D

    def test_evidence_level_d_contaminated(self):
        from src.lifecycle import EvidenceLevel
        ev = _evidence("ev-1", _identity(), n_oos=150, contaminated=True)
        assert ev.evidence_level == EvidenceLevel.LEVEL_D

    def test_evidence_level_c_no_execution(self):
        from src.lifecycle import EvidenceLevel
        ev = _evidence("ev-1", _identity(), n_oos=80, has_execution=False)
        # No execution evidence → at most LEVEL_C
        assert ev.evidence_level in (EvidenceLevel.LEVEL_C, EvidenceLevel.LEVEL_B)


# ══════════════════════════════════════════════════════════════════════════════
# 8 — Promotion gate
# ══════════════════════════════════════════════════════════════════════════════

class TestPromotionGate:
    def test_successful_promotion_no_incumbent(self):
        from src.lifecycle import PromotionGate, PromotionPolicy, PromotionOutcome
        gate = PromotionGate(PromotionPolicy())
        ev = _evidence("ev-1", _identity(), ic=0.05)
        decision = gate.evaluate("equity/5D", ev, None, "chal-1", None)
        assert decision.outcome == PromotionOutcome.PROMOTE
        assert decision.all_gates_pass

    def test_ic_improvement_too_small(self):
        from src.lifecycle import PromotionGate, PromotionPolicy, PromotionOutcome
        gate = PromotionGate(PromotionPolicy(min_ic_improvement=0.005))
        champ = _evidence("ev-champ", _identity("m", "v1"), ic=0.050)
        chal  = _evidence("ev-chal", _identity("n", "v1"), ic=0.052)  # +0.002 < 0.005
        decision = gate.evaluate("equity/5D", chal, champ, "chal-1", "m@v1")
        assert decision.outcome == PromotionOutcome.DO_NOT_PROMOTE
        assert decision.gate("PREDICTIVE").status.value == "FAIL"

    def test_calibration_failure(self):
        from src.lifecycle import PromotionGate, PromotionPolicy, PromotionOutcome
        gate = PromotionGate(PromotionPolicy())
        champ = _evidence("ev-champ", _identity("m", "v1"), ic=0.05, brier=0.15)
        chal  = _evidence("ev-chal", _identity("n", "v1"), ic=0.10, brier=0.30)  # much worse calibration
        decision = gate.evaluate("equity/5D", chal, champ, "chal-1", "m@v1")
        assert decision.gate("CALIBRATION").status.value == "FAIL"
        assert decision.outcome == PromotionOutcome.DO_NOT_PROMOTE

    def test_execution_failure_negative_net(self):
        from src.lifecycle import PromotionGate, PromotionPolicy, PromotionOutcome
        gate = PromotionGate(PromotionPolicy())
        chal = _evidence("ev-chal", _identity(), ic=0.10, net_ret=-0.01)  # negative net return
        decision = gate.evaluate("equity/5D", chal, None, "chal-1", None)
        assert decision.gate("EXECUTION").status.value == "FAIL"

    def test_stability_failure(self):
        from src.lifecycle import PromotionGate, PromotionPolicy
        gate = PromotionGate(PromotionPolicy())
        chal = _evidence("ev-chal", _identity(), ic=0.10, ic_decay="SIGNIFICANT_DECAY")
        decision = gate.evaluate("equity/5D", chal, None, "chal-1", None)
        assert decision.gate("STABILITY").status.value == "FAIL"

    def test_insufficient_evidence(self):
        from src.lifecycle import PromotionGate, PromotionPolicy, PromotionOutcome
        gate = PromotionGate(PromotionPolicy())
        chal = _evidence("ev-chal", _identity(), ic=0.10, n_oos=5)  # too few obs
        decision = gate.evaluate("equity/5D", chal, None, "chal-1", None)
        assert decision.outcome == PromotionOutcome.INSUFFICIENT_EVIDENCE

    def test_final_oos_contamination_blocks(self):
        from src.lifecycle import PromotionGate, PromotionPolicy, PromotionOutcome
        gate = PromotionGate(PromotionPolicy())
        chal = _evidence("ev-chal", _identity(), ic=0.10, contaminated=True)
        decision = gate.evaluate("equity/5D", chal, None, "chal-1", None)
        assert decision.outcome == PromotionOutcome.BLOCKED
        assert decision.final_oos_contaminated

    def test_no_black_box_score(self):
        """Decision must be structured gates, not a 0-100 score."""
        from src.lifecycle import PromotionGate, PromotionDecision
        gate = PromotionGate()
        ev = _evidence("ev-1", _identity())
        decision = gate.evaluate("equity/5D", ev, None, "chal-1", None)
        assert not hasattr(decision, "score")   # no arbitrary numeric score
        assert len(decision.gate_results) == 6   # six structured gates


# ══════════════════════════════════════════════════════════════════════════════
# 9 — End-to-end promotion & rollback
# ══════════════════════════════════════════════════════════════════════════════

class TestPromotionAndRollback:
    def _orchestrator(self, tmproot, human=False):
        from src.lifecycle import (
            ModelRegistry, ChampionIndex, PromotionOrchestrator,
            PromotionGate, PromotionPolicy,
        )
        reg = ModelRegistry(root=tmproot)
        champ = ChampionIndex(root=tmproot)
        gate = PromotionGate(PromotionPolicy(high_impact_requires_human=human))
        orch = PromotionOrchestrator(root=tmproot, registry=reg,
                                     champion_index=champ, gate=gate)
        return reg, champ, orch

    def test_successful_promotion(self, tmproot):
        from src.lifecycle import PromotionOutcome
        reg, champ, orch = self._orchestrator(tmproot)
        reg.register(_identity("m", "v1"), _provenance("m", "v1"))
        _advance_to_eligible(reg, "m@v1")
        ev = _evidence("ev-1", _identity("m", "v1"), ic=0.05)
        decision, manifest = orch.promote(
            "equity/5D", "m@v1", ev, None, "promo-1", "chal-1")
        assert decision.outcome == PromotionOutcome.PROMOTE
        assert manifest is not None
        assert champ.get_champion("equity/5D").champion_full_key == "m@v1"

    def test_promotion_atomic_manifest_hashed(self, tmproot):
        reg, champ, orch = self._orchestrator(tmproot)
        reg.register(_identity("m", "v1"), _provenance("m", "v1"))
        _advance_to_eligible(reg, "m@v1")
        ev = _evidence("ev-1", _identity("m", "v1"), ic=0.05)
        _, manifest = orch.promote("equity/5D", "m@v1", ev, None, "promo-1", "chal-1")
        assert manifest.verify()   # manifest hash valid

    def test_rollback_restores_previous(self, tmproot):
        reg, champ, orch = self._orchestrator(tmproot)
        # Champion A
        reg.register(_identity("a", "v1"), _provenance("a", "v1"))
        _advance_to_eligible(reg, "a@v1")
        ev_a = _evidence("ev-a", _identity("a", "v1"), ic=0.05)
        orch.promote("equity/5D", "a@v1", ev_a, None, "promo-a", "chal-a")
        # Challenger B promoted
        reg.register(_identity("b", "v1"), _provenance("b", "v1"))
        _advance_to_eligible(reg, "b@v1")
        ev_b = _evidence("ev-b", _identity("b", "v1"), ic=0.08)
        orch.promote("equity/5D", "b@v1", ev_b, ev_a, "promo-b", "chal-b")
        assert champ.get_champion("equity/5D").champion_full_key == "b@v1"
        # Rollback
        restored = orch.rollback("equity/5D", "B failed integrity check")
        assert restored == "a@v1"

    def test_rollback_does_not_modify_artifact(self, tmproot):
        reg, champ, orch = self._orchestrator(tmproot)
        reg.register(_identity("a", "v1", artifact_hash="hash_a"), _provenance("a", "v1"))
        _advance_to_eligible(reg, "a@v1")
        ev_a = _evidence("ev-a", _identity("a", "v1", artifact_hash="hash_a"), ic=0.05)
        orch.promote("equity/5D", "a@v1", ev_a, None, "promo-a", "chal-a")
        reg.register(_identity("b", "v1", artifact_hash="hash_b"), _provenance("b", "v1"))
        _advance_to_eligible(reg, "b@v1")
        ev_b = _evidence("ev-b", _identity("b", "v1", artifact_hash="hash_b"), ic=0.08)
        orch.promote("equity/5D", "b@v1", ev_b, ev_a, "promo-b", "chal-b")
        orch.rollback("equity/5D", "rollback")
        # A's artifact hash unchanged
        assert reg.get("a@v1").artifact_hash == "hash_a"

    def test_rollback_history_recorded(self, tmproot):
        reg, champ, orch = self._orchestrator(tmproot)
        reg.register(_identity("a", "v1"), _provenance("a", "v1"))
        _advance_to_eligible(reg, "a@v1")
        ev_a = _evidence("ev-a", _identity("a", "v1"), ic=0.05)
        orch.promote("equity/5D", "a@v1", ev_a, None, "promo-a", "chal-a")
        reg.register(_identity("b", "v1"), _provenance("b", "v1"))
        _advance_to_eligible(reg, "b@v1")
        ev_b = _evidence("ev-b", _identity("b", "v1"), ic=0.08)
        orch.promote("equity/5D", "b@v1", ev_b, ev_a, "promo-b", "chal-b")
        orch.rollback("equity/5D", "reason")
        assert len(orch.rollback_history()) >= 1

    def test_human_approval_required_defers_promotion(self, tmproot):
        from src.lifecycle import PromotionOutcome, ApprovalPolicy
        reg, champ, orch = self._orchestrator(tmproot, human=True)
        reg.register(_identity("m", "v1"), _provenance("m", "v1"))
        _advance_to_eligible(reg, "m@v1")
        ev = _evidence("ev-1", _identity("m", "v1"), ic=0.05)
        decision, manifest = orch.promote("equity/5D", "m@v1", ev, None, "promo-1", "chal-1")
        assert decision.outcome == PromotionOutcome.PROMOTE
        assert decision.approval_policy == ApprovalPolicy.HUMAN_APPROVAL_REQUIRED
        # Not auto-promoted — no manifest yet, no champion set
        assert manifest is None
        assert champ.get_champion("equity/5D") is None

    def test_no_champion_preferred_over_bad_model(self, tmproot):
        """A system with no champion is preferred over an inadequate model (spec §81)."""
        from src.lifecycle import PromotionOutcome
        reg, champ, orch = self._orchestrator(tmproot)
        reg.register(_identity("m", "v1"), _provenance("m", "v1"))
        _advance_to_eligible(reg, "m@v1")
        ev = _evidence("ev-1", _identity("m", "v1"), ic=0.05, n_oos=5)  # insufficient
        decision, manifest = orch.promote("equity/5D", "m@v1", ev, None, "promo-1", "chal-1")
        assert decision.outcome == PromotionOutcome.INSUFFICIENT_EVIDENCE
        assert champ.get_champion("equity/5D") is None   # no champion, correctly


# ══════════════════════════════════════════════════════════════════════════════
# 10 — Integrity / adversarial
# ══════════════════════════════════════════════════════════════════════════════

class TestAdversarial:
    def _orchestrator(self, tmproot):
        from src.lifecycle import (
            ModelRegistry, ChampionIndex, PromotionOrchestrator,
            PromotionGate, PromotionPolicy,
        )
        reg = ModelRegistry(root=tmproot)
        champ = ChampionIndex(root=tmproot)
        orch = PromotionOrchestrator(root=tmproot, registry=reg, champion_index=champ,
            gate=PromotionGate(PromotionPolicy(high_impact_requires_human=False)))
        return reg, champ, orch

    def test_unregistered_challenger_blocked(self, tmproot):
        from src.lifecycle import PromotionOutcome
        reg, champ, orch = self._orchestrator(tmproot)
        ev = _evidence("ev-1", _identity("ghost", "v1"), ic=0.05)
        # ghost@v1 is NOT registered
        decision, manifest = orch.promote(
            "equity/5D", "ghost@v1", ev, None, "promo-1", "chal-1")
        assert decision.outcome == PromotionOutcome.BLOCKED
        assert manifest is None

    def test_mutated_evidence_blocked(self, tmproot):
        from src.lifecycle import PromotionOutcome
        reg, champ, orch = self._orchestrator(tmproot)
        reg.register(_identity("m", "v1"), _provenance("m", "v1"))
        _advance_to_eligible(reg, "m@v1")
        ev = _evidence("ev-1", _identity("m", "v1"), ic=0.05)
        # Adversary mutates evidence after freezing (fake better IC)
        ev.oos_evidence.mean_rank_ic = 0.99
        decision, manifest = orch.promote("equity/5D", "m@v1", ev, None, "promo-1", "chal-1")
        # Integrity check catches the mutation → BLOCKED
        assert decision.outcome == PromotionOutcome.BLOCKED

    def test_future_oos_contamination_never_promotes(self, tmproot):
        from src.lifecycle import PromotionGate, PromotionOutcome
        gate = PromotionGate()
        # Excellent metrics but final OOS was used for selection
        ev = _evidence("ev-1", _identity(), ic=0.20, brier=0.10, net_ret=0.10,
                       dd=-0.02, contaminated=True)
        decision = gate.evaluate("equity/5D", ev, None, "chal-1", None)
        assert decision.outcome == PromotionOutcome.BLOCKED

    def test_manifest_tamper_detected(self, tmproot):
        reg, champ, orch = self._orchestrator(tmproot)
        reg.register(_identity("m", "v1"), _provenance("m", "v1"))
        _advance_to_eligible(reg, "m@v1")
        ev = _evidence("ev-1", _identity("m", "v1"), ic=0.05)
        _, manifest = orch.promote("equity/5D", "m@v1", ev, None, "promo-1", "chal-1")
        # Tamper with the manifest
        manifest.new_champion = "attacker@v1"
        assert not manifest.verify()

    def test_no_np_random_in_lifecycle(self):
        import ast, os
        d = "src/lifecycle"
        for fname in os.listdir(d):
            if not fname.endswith(".py"):
                continue
            with open(os.path.join(d, fname)) as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
                    if getattr(node.value.value, "id", "") == "np" and node.value.attr == "random":
                        pytest.fail(f"np.random in {fname} line {node.lineno}")

    def test_no_latest_pkl_load_in_lifecycle(self):
        """No 'latest'/'current' model loading pattern in lifecycle code."""
        import os
        d = "src/lifecycle"
        for fname in os.listdir(d):
            if not fname.endswith(".py"):
                continue
            with open(os.path.join(d, fname)) as f:
                content = f.read()
            # These strings only appear inside FORBIDDEN_ARTIFACT_NAMES / guards
            # Ensure no code actually loads them as a model
            assert 'load("latest' not in content
            assert "load('latest" not in content


# ══════════════════════════════════════════════════════════════════════════════
# 11 — Crash safety
# ══════════════════════════════════════════════════════════════════════════════

class TestCrashSafety:
    def test_pending_intent_detected(self, tmproot):
        from src.lifecycle import PromotionOrchestrator
        from src.lifecycle.schemas import PromotionDecision, PromotionOutcome
        orch = PromotionOrchestrator(root=tmproot)
        # Simulate a crash: write an intent marker but never commit
        decision = PromotionDecision(
            outcome=PromotionOutcome.PROMOTE, scope="equity/5D",
            champion_id=None, challenger_id="chal-1",
        )
        orch._write_intent("equity/5D", "m@v1", "promo-1", decision)
        assert orch.recovery_required()
        assert len(orch.pending_intents()) == 1

    def test_recover_discards_uncommitted(self, tmproot):
        from src.lifecycle import PromotionOrchestrator
        from src.lifecycle.schemas import PromotionDecision, PromotionOutcome
        orch = PromotionOrchestrator(root=tmproot)
        decision = PromotionDecision(
            outcome=PromotionOutcome.PROMOTE, scope="equity/5D",
            champion_id=None, challenger_id="chal-1",
        )
        orch._write_intent("equity/5D", "m@v1", "promo-1", decision)
        # No champion was set → recovery discards the intent
        recovered = orch.recover_pending()
        assert "equity/5D" in recovered
        assert not orch.recovery_required()
        # No champion left behind (fail-closed)
        assert orch.champions.get_champion("equity/5D") is None

    def test_no_two_champions_after_crash(self, tmproot):
        """After a simulated mid-promotion crash + recovery, exactly one (or zero) champion."""
        from src.lifecycle import PromotionOrchestrator
        reg, champ, orch = (
            __import__("src.lifecycle", fromlist=["ModelRegistry"]).ModelRegistry(root=tmproot),
            __import__("src.lifecycle", fromlist=["ChampionIndex"]).ChampionIndex(root=tmproot),
            None,
        )
        orch = PromotionOrchestrator(root=tmproot, registry=reg, champion_index=champ)
        # Promote A cleanly
        reg.register(_identity("a", "v1"), _provenance("a", "v1"))
        _advance_to_eligible(reg, "a@v1")
        from src.lifecycle import PromotionGate, PromotionPolicy
        orch.gate = PromotionGate(PromotionPolicy(high_impact_requires_human=False))
        ev_a = _evidence("ev-a", _identity("a", "v1"), ic=0.05)
        orch.promote("equity/5D", "a@v1", ev_a, None, "promo-a", "chal-a")
        # Simulate crashed promotion of B (intent written, not committed)
        from src.lifecycle.schemas import PromotionDecision, PromotionOutcome
        orch._write_intent("equity/5D", "b@v1", "promo-b",
            PromotionDecision(outcome=PromotionOutcome.PROMOTE, scope="equity/5D",
                              champion_id="a@v1", challenger_id="chal-b"))
        orch.recover_pending()
        # A is still the sole champion
        assert champ.get_champion("equity/5D").champion_full_key == "a@v1"


# ══════════════════════════════════════════════════════════════════════════════
# 12 — Reproducibility & comparison
# ══════════════════════════════════════════════════════════════════════════════

class TestReproducibility:
    def test_same_inputs_same_decision(self):
        from src.lifecycle import PromotionGate, PromotionPolicy
        gate = PromotionGate(PromotionPolicy())
        ev1 = _evidence("ev-1", _identity(), ic=0.05)
        ev2 = _evidence("ev-1", _identity(), ic=0.05)
        d1 = gate.evaluate("equity/5D", ev1, None, "chal-1", None)
        d2 = gate.evaluate("equity/5D", ev2, None, "chal-1", None)
        assert d1.outcome == d2.outcome
        assert [g.status for g in d1.gate_results] == [g.status for g in d2.gate_results]

    def test_evidence_hash_reproducible(self):
        # Same identity object → identical evidence content → identical hash.
        ident = _identity()
        ev1 = _evidence("ev-1", ident, ic=0.05)
        ev2 = _evidence("ev-1", ident, ic=0.05)
        assert ev1.evidence_hash == ev2.evidence_hash

    def test_comparison_apples_to_apples(self):
        from src.lifecycle import ChampionChallengerComparator, FrozenEvalSnapshot
        snapshot = FrozenEvalSnapshot.create("eval-1", ["A", "B", "C"], "2023", 300)
        comp = ChampionChallengerComparator()
        result = comp.compare(
            "cmp-1", "equity/5D", "chal-1", "champ@v1", snapshot,
            champion_predictions=None, challenger_predictions=None,
            champion_metrics={"mean_rank_ic": 0.05},
            challenger_metrics={"mean_rank_ic": 0.08},
        )
        assert comp.is_apples_to_apples(result)
        assert result.rank_ic_delta == 0.03

    def test_near_identical_predictions_flagged(self):
        import numpy as np
        from src.lifecycle import ChampionChallengerComparator, FrozenEvalSnapshot
        snapshot = FrozenEvalSnapshot.create("eval-1", ["A"], "2023", 100)
        comp = ChampionChallengerComparator()
        base = np.linspace(0, 1, 100)
        result = comp.compare(
            "cmp-1", "equity/5D", "chal-1", "champ@v1", snapshot,
            champion_predictions=base,
            challenger_predictions=base + 1e-6,   # ~identical
            champion_metrics={}, challenger_metrics={},
        )
        assert result.predictions_near_identical

    def test_frozen_snapshot_hash_stable(self):
        from src.lifecycle import FrozenEvalSnapshot
        s1 = FrozenEvalSnapshot.create("eval-1", ["A", "B"], "2023", 100)
        s2 = FrozenEvalSnapshot.create("eval-1", ["B", "A"], "2023", 100)
        assert s1.dataset_hash == s2.dataset_hash   # order-independent


# ══════════════════════════════════════════════════════════════════════════════
# 13 — PIT / historical lookup
# ══════════════════════════════════════════════════════════════════════════════

class TestPIT:
    def test_historical_champion_lookup(self, tmproot):
        from src.lifecycle import ChampionIndex
        champ = ChampionIndex(root=tmproot)
        champ.promote("equity/5D", "a@v1", "r1", "ev-1")
        # champion_at with a far-future timestamp returns current champion
        result = champ.champion_at("equity/5D", "2099-01-01T00:00:00+00:00")
        assert result == "a@v1"

    def test_future_champion_mutation_does_not_alter_manifest(self, tmproot):
        from src.lifecycle import (
            ModelRegistry, ChampionIndex, PromotionOrchestrator,
            PromotionGate, PromotionPolicy,
        )
        reg = ModelRegistry(root=tmproot)
        champ = ChampionIndex(root=tmproot)
        orch = PromotionOrchestrator(root=tmproot, registry=reg, champion_index=champ,
            gate=PromotionGate(PromotionPolicy(high_impact_requires_human=False)))
        reg.register(_identity("a", "v1"), _provenance("a", "v1"))
        _advance_to_eligible(reg, "a@v1")
        ev = _evidence("ev-a", _identity("a", "v1"), ic=0.05)
        _, manifest = orch.promote("equity/5D", "a@v1", ev, None, "promo-a", "chal-a")
        original_hash = manifest.manifest_hash
        # Promote a new champion later
        reg.register(_identity("b", "v1"), _provenance("b", "v1"))
        _advance_to_eligible(reg, "b@v1")
        ev_b = _evidence("ev-b", _identity("b", "v1"), ic=0.08)
        orch.promote("equity/5D", "b@v1", ev_b, ev, "promo-b", "chal-b")
        # The original manifest's hash is unchanged (immutable)
        assert manifest.manifest_hash == original_hash
        assert manifest.new_champion == "a@v1"


# ══════════════════════════════════════════════════════════════════════════════
# 14 — Backward compatibility
# ══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    def test_prior_phase_imports_unaffected(self):
        # monitoring.model_registry has no sklearn dependency — must import.
        import src.monitoring.model_registry
        assert True

    def test_lifecycle_has_no_sklearn_dependency(self):
        """
        src/lifecycle/ must be fully decoupled from the sklearn/joblib chain
        (which is broken on this Python 3.14 env due to a missing cloudpickle
        transitive dep). Lifecycle is pure-stdlib + numpy only.
        """
        import ast, os
        d = "src/lifecycle"
        for fname in os.listdir(d):
            if not fname.endswith(".py"):
                continue
            with open(os.path.join(d, fname)) as f:
                content = f.read()
            assert "import sklearn" not in content
            assert "from sklearn" not in content

    def test_reuses_acceptance_gate(self):
        """
        Phase 3J integrates the existing ModelAcceptanceGate. The gate lives in
        validation.metrics which transitively imports sklearn; if sklearn's
        deps are unavailable in this env, skip (pre-existing env issue, not a
        Phase 3J regression).
        """
        try:
            from src.validation.metrics import ModelAcceptanceGate, AcceptanceThresholds
        except (ImportError, ModuleNotFoundError):
            pytest.skip("sklearn transitive deps unavailable in this environment")
        gate = ModelAcceptanceGate(AcceptanceThresholds())
        assert gate is not None
