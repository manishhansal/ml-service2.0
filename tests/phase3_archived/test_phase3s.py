"""
Phase 3S — Research Factory & Experimentation Governance test suite (spec §57, §58).

Deterministic: fixed seeds, tmp_path, no network / no credentials. Covers every
research-factory surface plus test-the-tests (§58) and security (§55). These tests
exercise the ADDITIVE `src.research` package only; no prior-phase behaviour is
changed.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.research as R


SEED = 20260906


def _rng(offset: int = 0):
    return np.random.default_rng(SEED + offset)


# ══════════════════════════════════════════════════════════════════════════════
# §3-§7 Hypothesis + pre-registration
# ══════════════════════════════════════════════════════════════════════════════

class TestHypothesis:
    def _good(self, hid="H1"):
        return R.Hypothesis(
            hypothesis_id=hid, hypothesis_type=R.HypothesisType.RANKING,
            statement="Sector-neutral relative strength improves next-5-day rank IC after costs",
            expected_mechanism="underreaction to sector-relative moves",
            expected_direction=R.ExpectedDirection.POSITIVE,
            target="next-5-day rank IC", horizon="5 days", universe="NIFTY100",
            regime_assumptions="trending", economic_rationale="momentum underreaction",
            success_criteria="rank IC > 0 OOS after costs",
            failure_criteria="rank IC <= 0 OOS or erased by costs")

    def test_falsifiable_hypothesis_accepted(self):
        h = self._good()
        assert h.hypothesis_hash
        assert h.hypothesis_type == R.HypothesisType.RANKING

    def test_vague_hypothesis_rejected(self):
        with pytest.raises(R.HypothesisError):
            R.Hypothesis(
                hypothesis_id="H2", hypothesis_type=R.HypothesisType.MODEL,
                statement="improve the model", expected_mechanism="",
                expected_direction=R.ExpectedDirection.POSITIVE, target="", horizon="5d",
                universe="X", regime_assumptions="", economic_rationale="",
                success_criteria="", failure_criteria="")

    def test_registry_dup_rejected(self, tmp_path):
        reg = R.HypothesisRegistry(tmp_path)
        reg.register(self._good())
        with pytest.raises(R.HypothesisError):
            reg.register(self._good())

    def test_preregistration_freeze_and_verify(self, tmp_path):
        reg = R.HypothesisRegistry(tmp_path)
        reg.register(self._good())
        pr = R.PreRegistration(prereg_id="P1", hypothesis_id="H1",
                               experiment_design="WF 5-fold", primary_metric="rank_ic",
                               secondary_metrics=["turnover"], validation_windows=["2019-2021"])
        reg.pre_register(pr)
        assert pr.frozen and pr.verify()

    def test_prereg_single_primary_metric(self):
        with pytest.raises(R.PreRegistrationError):
            R.PreRegistration(prereg_id="P", hypothesis_id="H1", experiment_design="d",
                              primary_metric="rank_ic", secondary_metrics=["rank_ic"])

    def test_prereg_requires_registered_hypothesis(self, tmp_path):
        reg = R.HypothesisRegistry(tmp_path)
        pr = R.PreRegistration(prereg_id="P1", hypothesis_id="UNKNOWN",
                               experiment_design="d", primary_metric="rank_ic")
        with pytest.raises(R.PreRegistrationError):
            reg.pre_register(pr)


# ══════════════════════════════════════════════════════════════════════════════
# §3, §4, §44 Manifest identity, immutability, lineage
# ══════════════════════════════════════════════════════════════════════════════

class TestManifest:
    def _m(self, eid, **kw):
        base = dict(experiment_id=eid, hypothesis_id="H1", experiment_type="RANKING",
                    dataset_hash="DS1", feature_version="F1", random_seed=7)
        base.update(kw)
        return R.ExperimentManifest(**base)

    def test_identical_inputs_same_hash(self):
        assert self._m("E1").same_identity_as(self._m("E2"))

    def test_changed_dataset_changes_hash(self):
        assert not self._m("E1").same_identity_as(self._m("E3", dataset_hash="DS2"))

    def test_changed_model_changes_hash(self):
        assert not self._m("E1").same_identity_as(self._m("E4", model_config_hash="M2"))

    def test_registry_append_and_current(self, tmp_path):
        reg = R.ResearchExperimentRegistry(tmp_path)
        reg.register(self._m("E1"))
        reg.update_status("E1", R.ExperimentStatus.RUNNING)
        reg.update_status("E1", R.ExperimentStatus.COMPLETED)
        assert reg.current("E1")["status"] == "COMPLETED"

    def test_terminal_experiment_immutable(self, tmp_path):
        reg = R.ResearchExperimentRegistry(tmp_path)
        reg.register(self._m("E1"))
        reg.update_status("E1", R.ExperimentStatus.RUNNING)
        reg.update_status("E1", R.ExperimentStatus.FAILED)
        with pytest.raises(R.ExperimentImmutabilityError):
            reg.update_status("E1", R.ExperimentStatus.RUNNING)

    def test_duplicate_experiment_rejected(self, tmp_path):
        reg = R.ResearchExperimentRegistry(tmp_path)
        reg.register(self._m("E1"))
        with pytest.raises(R.ExperimentImmutabilityError):
            reg.register(self._m("E1"))

    def test_lineage_chain(self, tmp_path):
        reg = R.ResearchExperimentRegistry(tmp_path)
        reg.register(self._m("E1"))
        reg.register(self._m("E2", parent_experiment_id="E1"))
        reg.register(self._m("E3", parent_experiment_id="E2"))
        assert reg.lineage("E3") == ["E1", "E2"]


# ══════════════════════════════════════════════════════════════════════════════
# §8, §9 Metric registry
# ══════════════════════════════════════════════════════════════════════════════

class TestMetrics:
    def test_unknown_metric_rejected(self):
        with pytest.raises(R.MetricError):
            R.DEFAULT_METRIC_REGISTRY.get("does_not_exist")

    def test_primary_metric_type_mismatch(self):
        with pytest.raises(R.MetricError):
            R.DEFAULT_METRIC_REGISTRY.validate_primary("brier", "RANKING")

    def test_direction_aware_improvement(self):
        reg = R.DEFAULT_METRIC_REGISTRY
        assert reg.get("rank_ic").is_improvement(0.05, 0.02)      # higher better
        assert reg.get("brier").is_improvement(0.10, 0.20)        # lower better


# ══════════════════════════════════════════════════════════════════════════════
# §10, §11 Data snapshot + universe
# ══════════════════════════════════════════════════════════════════════════════

class TestDataSnapshot:
    def _uni(self, pit=True):
        return R.Universe(universe_id="N100_PIT",
                          construction=R.UniverseConstruction.INDEX_MEMBERSHIP,
                          effective_date="2020-01-01", point_in_time=pit)

    def _snap(self, dataset_id="ds1", pit=True):
        return R.build_research_snapshot(
            dataset_id=dataset_id, providers=["angel"], universe=self._uni(pit),
            date_start="2019-01-01", date_end="2020-12-31", schema_version="s1",
            transformation_version="t1", corporate_action_version="ca1",
            feature_version="f1")

    def test_immutable_snapshot_ok(self):
        assert self._snap().snapshot_hash

    def test_mutable_dataset_rejected(self):
        with pytest.raises(R.DataSnapshotError):
            self._snap(dataset_id="latest")

    def test_non_pit_universe_rejected(self):
        with pytest.raises(R.DataSnapshotError):
            R.ResearchDataSnapshot(dataset_id="ds", identity=self._snap().identity,
                                   universe=self._uni(pit=False), pit_universe=False)


# ══════════════════════════════════════════════════════════════════════════════
# §12-§15 Feature registry + leakage guards
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatures:
    def _prov(self):
        return R.FeatureProvenance(source_data="adj_ohlcv", transformation="20d z",
                                   parameters=(20,), availability_time="close_t")

    def test_documented_feature(self):
        f = R.ResearchFeature(feature_id="F1", name="rs20",
                              category=R.FeatureCategory.MOMENTUM, formula="z(close/close.shift(20))",
                              availability_timestamp="close_t", min_history=20,
                              dependencies=("adj_ohlcv",), missing_behavior="nan",
                              provenance=self._prov())
        reg = R.ResearchFeatureRegistry()
        reg.register(f)
        assert reg.is_documented("F1")

    def test_undocumented_feature_rejected(self):
        with pytest.raises(R.FeatureError):
            R.ResearchFeature(feature_id="", name="x", category=R.FeatureCategory.PRICE,
                              formula="", availability_timestamp="", min_history=1,
                              dependencies=(), missing_behavior="", provenance=self._prov())

    def test_clean_leakage_audit(self):
        rep = R.audit_feature_leakage(
            feature_times=np.array([1, 2, 3, 4, 5]),
            label_event_start=np.array([2, 3, 4, 5, 6]),
            scaler_fit_indices=np.array([0, 1, 2]), oos_indices=np.array([8, 9]))
        assert rep.passed

    def test_future_feature_invalidates(self):
        with pytest.raises(R.LeakageInvalidation):
            R.audit_feature_leakage(feature_times=np.array([1, 2, 9]),
                                    label_event_start=np.array([2, 3, 4]))

    def test_normalization_leakage_invalidates(self):
        with pytest.raises(R.LeakageInvalidation):
            R.check_normalization_causal(train_indices=np.array([0, 1, 2, 3]),
                                         eval_indices=np.array([4, 5]),
                                         scaler_fit_indices=np.array([0, 1, 4]))


# ══════════════════════════════════════════════════════════════════════════════
# §16, §17 Validation + tiers
# ══════════════════════════════════════════════════════════════════════════════

class TestValidation:
    def test_research_split_sealed(self):
        split = R.build_research_split(100, oos_fraction=0.2, val_fraction=0.25)
        assert split.is_sealed()
        assert len(set(split.search_indices().tolist()) & set(split.oos_indices().tolist())) == 0

    def test_sealed_oos_allows_final_eval(self):
        sealed = R.SealedOOS(np.arange(80, 100))
        out = sealed.reveal("final_evaluation")
        assert len(out) == 20

    def test_sealed_oos_forbids_hpo(self):
        sealed = R.SealedOOS(np.arange(80, 100))
        with pytest.raises(R.OOSAccessError):
            sealed.reveal("hpo")
        assert sealed.contamination_check().contaminated

    def test_walk_forward_folds(self):
        folds = R.walk_forward_folds(300, train_bars=100, val_bars=30, test_bars=30)
        assert len(folds) >= 1

    def test_tier_classification(self):
        assert R.classify_tier(R.TierEvidence()) == R.ExperimentTier.TIER_A
        full = R.TierEvidence(pit_safe=True, walk_forward=True, cost_aware=True,
                              statistically_evaluated=True, independent_oos=True,
                              robustness_tested=True, multiple_testing_adjusted=True,
                              economic_rationale=True, full_evidence_package=True,
                              independent_validation=True, paper_shadow_compatible=True)
        assert R.classify_tier(full) == R.ExperimentTier.TIER_D
        assert R.is_promotable(R.ExperimentTier.TIER_D)
        assert not R.is_promotable(R.ExperimentTier.TIER_A)


# ══════════════════════════════════════════════════════════════════════════════
# §18, §19 Baseline + ablation
# ══════════════════════════════════════════════════════════════════════════════

class TestBaselineAblation:
    def test_baseline_required(self):
        with pytest.raises(R.BaselineError):
            R.require_baseline([])

    def test_beats_random_baseline(self):
        bl = R.BaselineResult(baseline_type=R.BaselineType.RANDOM_SIGNAL,
                              metric_name="rank_ic", value=0.0, n_observations=100)
        cmp = R.compare_to_baseline(metric_name="rank_ic", candidate_value=0.04, baseline=bl)
        assert cmp.beats_baseline

    def test_ablation_contributions(self):
        cases = [R.AblationCase(kind=R.AblationKind.FEATURE, component="F1",
                                metric_name="rank_ic", metric_value=0.02)]
        rep = R.build_ablation_report(metric_name="rank_ic", full_model_value=0.05,
                                      baseline_value=0.0, ablation_cases=cases)
        assert rep.contributions[0].contribution == pytest.approx(0.03)
        assert rep.full_vs_baseline == pytest.approx(0.05)


# ══════════════════════════════════════════════════════════════════════════════
# §20, §21 Incremental alpha + correlated signal
# ══════════════════════════════════════════════════════════════════════════════

class TestIncremental:
    def test_insufficient_evidence(self):
        r = R.incremental_alpha_test(gross_improvement=0.05, net_improvement=0.04,
                                     oos_improvement=0.03, n_observations=10)
        assert r.verdict == R.IncrementalVerdict.INSUFFICIENT_EVIDENCE

    def test_no_net_alpha(self):
        r = R.incremental_alpha_test(gross_improvement=0.05, net_improvement=-0.01,
                                     oos_improvement=0.03, n_observations=100)
        assert r.verdict == R.IncrementalVerdict.NO_NET_ALPHA

    def test_no_oos_alpha(self):
        r = R.incremental_alpha_test(gross_improvement=0.05, net_improvement=0.02,
                                     oos_improvement=-0.01, n_observations=100)
        assert r.verdict == R.IncrementalVerdict.NO_OOS_ALPHA

    def test_incremental_alpha(self):
        r = R.incremental_alpha_test(gross_improvement=0.05, net_improvement=0.02,
                                     oos_improvement=0.015, n_observations=100)
        assert r.verdict == R.IncrementalVerdict.INCREMENTAL_ALPHA

    def test_redundant_signal(self):
        rng = _rng(1)
        base = rng.standard_normal(300)
        realized = 0.4 * base + rng.standard_normal(300) * 0.5
        redundant = 2.5 * base - 1.0   # exact linear fn of base
        res = R.audit_correlated_signal(redundant, {"s1": base}, realized)
        assert res.relationship == R.SignalRelationship.REDUNDANT

    def test_incremental_signal(self):
        rng = _rng(2)
        base = rng.standard_normal(300)
        indep = rng.standard_normal(300)
        new = base + 2.0 * indep
        realized = 0.3 * base + 0.5 * indep + rng.standard_normal(300) * 0.1
        res = R.audit_correlated_signal(new, {"s1": base}, realized)
        assert res.relationship == R.SignalRelationship.INCREMENTAL

    def test_cross_sectional_and_time_series(self):
        rng = _rng(3)
        scores = [rng.standard_normal(20) for _ in range(40)]
        realized = [s * 0.2 + rng.standard_normal(20) for s in scores]
        csr = R.cross_sectional_diagnostics(scores, realized, n_buckets=5)
        assert csr.n_cross_sections == 40
        d = np.sign(rng.standard_normal(100))
        r = d * 0.01 + rng.standard_normal(100) * 0.02
        tsr = R.time_series_diagnostics(d, r, per_trade_cost=0.005)
        assert tsr.cost_adjusted_expectancy <= tsr.expectancy


# ══════════════════════════════════════════════════════════════════════════════
# §29, §30, §35 Statistics + multiple testing
# ══════════════════════════════════════════════════════════════════════════════

class TestStatistics:
    def test_multiple_testing_tracks_trials(self):
        pvals = [0.001, 0.02, 0.04]
        few = R.correct_multiple_testing(pvals, n_trials=3, method=R.CorrectionMethod.HOLM)
        many = R.correct_multiple_testing(pvals, n_trials=500, method=R.CorrectionMethod.HOLM)
        assert many.n_trials == 500
        assert many.n_significant < few.n_significant

    def test_bonferroni_strict_after_many_trials(self):
        res = R.correct_multiple_testing([0.001], n_trials=1000,
                                         method=R.CorrectionMethod.BONFERRONI)
        assert res.n_significant == 0

    def test_deflated_sharpe_deflates_with_trials(self):
        rng = _rng(4)
        marginal = rng.standard_normal(120) * 0.01 + 0.0008
        few = R.deflated_sharpe_ratio(marginal, n_trials=1)
        many = R.deflated_sharpe_ratio(marginal, n_trials=2000)
        assert many.expected_max_sharpe > few.expected_max_sharpe
        assert many.deflated_psr <= few.deflated_psr

    def test_pbo_genuine_low_noise_higher(self):
        rng = _rng(5)
        T, N = 40, 6
        genuine = rng.standard_normal((T, N)) * 0.01
        genuine[:, 0] += 0.02
        pbo_g = R.probability_backtest_overfitting(genuine, n_splits=6)
        noise = rng.standard_normal((T, N)) * 0.01
        pbo_n = R.probability_backtest_overfitting(noise, n_splits=6)
        assert pbo_g.pbo <= pbo_n.pbo
        assert not pbo_g.overfit

    def test_reality_check(self):
        rng = _rng(6)
        excess = rng.standard_normal((200, 20)) * 0.01
        excess[:, 3] += 0.004
        rc = R.whites_reality_check(excess, n_bootstrap=300)
        assert rc.reject_null
        allnoise = rng.standard_normal((200, 20)) * 0.01
        rc2 = R.whites_reality_check(allnoise, n_bootstrap=300)
        assert not rc2.reject_null


# ══════════════════════════════════════════════════════════════════════════════
# §32-§37, §45, §46 Controls
# ══════════════════════════════════════════════════════════════════════════════

def _lstsq_fit_predict(Xtr, ytr, Xoos):
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    beta = np.linalg.lstsq(A, ytr, rcond=None)[0]
    return np.column_stack([np.ones(len(Xoos)), Xoos]) @ beta


class TestControls:
    def test_placebo_collapse(self):
        # Multi-feature model: shuffling TRAIN labels destroys the feature-weighting
        # pattern, so the OOS prediction decorrelates and rank IC collapses. (A
        # single-feature linear model would keep |IC| under permutation because
        # rank IC is invariant to the coefficient — hence the multi-feature setup.)
        rng = np.random.default_rng(101)
        n, k = 400, 8
        w = np.array([0.5, -0.4, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0])
        Xtr = rng.standard_normal((n, k)); ytr = Xtr @ w + rng.standard_normal(n) * 0.3
        Xoos = rng.standard_normal((n, k)); yoos = Xoos @ w + rng.standard_normal(n) * 0.3
        pl = R.placebo_test(_lstsq_fit_predict, Xtr, ytr, Xoos, yoos, seed=1)
        assert pl.passed

    def test_negative_control(self):
        assert R.negative_control_test({"r1": 0.5, "r2": 0.3, "noise": 0.05}, "noise").passed
        assert not R.negative_control_test({"r1": 0.1, "noise": 0.9}, "noise").passed

    def test_multi_seed_stable_vs_fragile(self):
        stable = R.multi_seed_evaluate({0: 0.05, 1: 0.048, 2: 0.052, 3: 0.049, 4: 0.051})
        fragile = R.multi_seed_evaluate({0: 0.001, 1: 0.0, 2: 0.002, 3: 0.001, 4: 0.5})
        assert not stable.single_seed_dependent
        assert fragile.single_seed_dependent

    def test_degrees_of_freedom(self):
        dof = R.DegreesOfFreedomReport(choices={"universe": "N100", "horizon": "5d"})
        assert dof.n_documented == 2
        assert "features" in dof.undocumented

    def test_data_snooping_lineage(self, tmp_path):
        reg = R.ResearchExperimentRegistry(tmp_path)
        reg.register(R.ExperimentManifest(experiment_id="E1", hypothesis_id="H",
                     experiment_type="RANKING", dataset_hash="DS1", feature_version="F1"))
        reg.register(R.ExperimentManifest(experiment_id="E2", hypothesis_id="H",
                     experiment_type="RANKING", dataset_hash="DS1", feature_version="F1",
                     parent_experiment_id="E1"))
        ds = R.data_snooping_report(reg, "E2")
        assert ds.reused_dataset == 1
        assert ds.parent_chain == ["E1"]

    def test_hpo_governance_blocks_oos_touch(self):
        R.HPORunRecord(search_space={}, n_trials=1, seeds=[0], best_config={},
                       selection_metric="rank_ic", validation_score=0.04,
                       touched_oos=False).validate()
        with pytest.raises(R.HPOGovernanceError):
            R.HPORunRecord(search_space={}, n_trials=1, seeds=[0], best_config={},
                           selection_metric="rank_ic", validation_score=0.9,
                           touched_oos=True).validate()


# ══════════════════════════════════════════════════════════════════════════════
# §40 Status / §38 Comparison / §39 Leaderboard
# ══════════════════════════════════════════════════════════════════════════════

class TestStatusComparisonLeaderboard:
    def test_no_success_status(self):
        assert not any(s.value == "SUCCESS" for s in R.ResearchStatus)

    def test_comparison_invalid_on_mismatch(self):
        a = {"universe_id": "N100", "period": "P", "label_version": "L", "cost_config_hash": "c",
             "execution_config_hash": "e", "benchmark": "NIFTY"}
        assert R.check_comparable(a, dict(a)).is_valid
        b = dict(a); b["universe_id"] = "N500"
        assert R.check_comparable(a, b).status == R.ComparabilityStatus.COMPARISON_INVALID

    def test_leaderboard_not_ranked_by_return_only(self):
        assert "statistical_confidence" in R.LEADERBOARD_COLUMNS
        assert "multiple_testing_status" in R.LEADERBOARD_COLUMNS
        lb = R.Leaderboard()
        lb.add(R.LeaderboardRow("E1", "rank_ic", 0.03, R.LeaderboardClass.EXPERIMENTAL, net_return=0.5))
        lb.add(R.LeaderboardRow("E2", "rank_ic", 0.05, R.LeaderboardClass.VALIDATED, net_return=0.1))
        grouped = lb.grouped()
        assert len(grouped["EXPERIMENTAL"]) == 1 and len(grouped["VALIDATED"]) == 1


# ══════════════════════════════════════════════════════════════════════════════
# §51 Gates
# ══════════════════════════════════════════════════════════════════════════════

def _all_pass_conditions():
    return R.GateConditions(
        pit_safe=True, reproducible_snapshot=True, split_correct=True, no_leakage=True,
        baseline_defined=True, beats_baseline=True, oos_evaluated=True, oos_contaminated=False,
        net_economics_evaluated=True, survives_costs=True, temporal_stable=True,
        regime_stable=True, cross_sectional_stable=True, uncertainty_reported=True,
        multiple_testing_adjusted=True, sufficient_sample=True, reproduced=True,
        evidence_package_complete=True, evidence_immutable=True)


class TestGates:
    def test_default_conditions_fail_closed(self):
        rep = R.evaluate_research_gates(R.GateConditions())
        assert not rep.all_pass
        assert len(rep.failed_gates()) == len(R.RESEARCH_GATE_ORDER)

    def test_all_pass_eligible(self):
        rep = R.evaluate_research_gates(_all_pass_conditions())
        assert rep.all_pass and rep.challenger_eligible

    def test_one_failure_blocks(self):
        cond = _all_pass_conditions()
        cond.oos_contaminated = True   # contaminate OOS
        rep = R.evaluate_research_gates(cond)
        assert not rep.challenger_eligible
        assert "OOS" in rep.failed_gates()


# ══════════════════════════════════════════════════════════════════════════════
# §42, §43 Artifact + reproduction
# ══════════════════════════════════════════════════════════════════════════════

class TestArtifact:
    def _freeze(self, root, eid="EXP1"):
        art = R.ExperimentArtifact(root=root, experiment_id=eid)
        art.freeze(manifest={"experiment_id": eid},
                   sections={"metrics": {"rank_ic": 0.05}, "predictions": {"p": [0.1, 0.2]}})
        return art

    def test_freeze_and_integrity(self, tmp_path):
        art = self._freeze(tmp_path)
        ok, mismatched = art.verify_integrity()
        assert ok and not mismatched

    def test_no_overwrite(self, tmp_path):
        art = self._freeze(tmp_path)
        with pytest.raises(R.ArtifactError):
            art.freeze(manifest={"experiment_id": "EXP1"}, sections={})

    def test_reproduction_success(self, tmp_path):
        art = self._freeze(tmp_path)
        res = R.reproduce(art, lambda s: {"metrics": s["metrics"], "predictions": s["predictions"]})
        assert res.reproduced

    def test_reproduction_failure(self, tmp_path):
        art = self._freeze(tmp_path)
        res = R.reproduce(art, lambda s: {"metrics": {"rank_ic": 0.99}})
        assert res.status == R.ReproductionStatus.REPRODUCTION_FAILURE
        assert "metrics" in res.mismatched


# ══════════════════════════════════════════════════════════════════════════════
# §41, §52, §56 Factory + promotion boundary + isolation
# ══════════════════════════════════════════════════════════════════════════════

class TestFactoryBoundary:
    def _tier_d(self):
        return R.classify_tier(R.TierEvidence(
            pit_safe=True, walk_forward=True, cost_aware=True, statistically_evaluated=True,
            independent_oos=True, robustness_tested=True, multiple_testing_adjusted=True,
            economic_rationale=True, full_evidence_package=True, independent_validation=True,
            paper_shadow_compatible=True))

    def test_recommend_challenger_candidate_only(self):
        fac = R.ResearchFactory()
        dec = fac.recommend(experiment_id="E1",
                            gate_report=R.evaluate_research_gates(_all_pass_conditions()),
                            tier=self._tier_d())
        assert dec.outcome == R.RecommendationOutcome.CHALLENGER_CANDIDATE
        assert dec.requires_human_review and not dec.promoted

    def test_tier_below_d_not_recommended(self):
        fac = R.ResearchFactory()
        tier_b = R.classify_tier(R.TierEvidence(pit_safe=True, walk_forward=True,
                                                cost_aware=True, statistically_evaluated=True))
        dec = fac.recommend(experiment_id="E1",
                            gate_report=R.evaluate_research_gates(_all_pass_conditions()),
                            tier=tier_b)
        assert dec.outcome == R.RecommendationOutcome.DO_NOT_RECOMMEND

    def test_failed_gates_not_recommended(self):
        fac = R.ResearchFactory()
        dec = fac.recommend(experiment_id="E1",
                            gate_report=R.evaluate_research_gates(R.GateConditions()),
                            tier=self._tier_d())
        assert dec.outcome == R.RecommendationOutcome.DO_NOT_RECOMMEND

    def test_factory_has_no_promote_method(self):
        fac = R.ResearchFactory()
        assert not hasattr(fac, "promote")
        assert not hasattr(fac, "deploy")
        assert not hasattr(fac, "retrain")
        assert not hasattr(fac, "recalibrate")

    def test_promotion_boundary_invariant(self):
        with pytest.raises(R.PromotionBoundaryError):
            R.RecommendationDecision(
                outcome=R.RecommendationOutcome.CHALLENGER_CANDIDATE, experiment_id="X",
                tier=R.ExperimentTier.TIER_D, gates_all_pass=True,
                requires_human_review=True, promoted=True)

    def test_isolation_guard(self):
        before = R.IsolationGuard(champion_fingerprint="c1", challenger_fingerprint="ch1",
                                  paper_session_fingerprint="p1",
                                  production_config_fingerprint="pc1",
                                  historical_evidence_fingerprint="h1")
        same = R.IsolationGuard(champion_fingerprint="c1", challenger_fingerprint="ch1",
                                paper_session_fingerprint="p1",
                                production_config_fingerprint="pc1",
                                historical_evidence_fingerprint="h1")
        changed = R.IsolationGuard(champion_fingerprint="c2", challenger_fingerprint="ch1",
                                   paper_session_fingerprint="p1",
                                   production_config_fingerprint="pc1",
                                   historical_evidence_fingerprint="h1")
        assert before.verify_unchanged(same)[0]
        ok, domains = before.verify_unchanged(changed)
        assert not ok and "champion" in domains


# ══════════════════════════════════════════════════════════════════════════════
# §58 Test-the-tests — deliberately introduce failures; the guards must catch them
# ══════════════════════════════════════════════════════════════════════════════

class TestTheTests:
    def test_future_feature_detected(self):
        rep = R.audit_feature_leakage(feature_times=np.array([1, 2, 9]),
                                      label_event_start=np.array([2, 3, 4]),
                                      raise_on_violation=False)
        assert not rep.passed
        assert any("FUTURE_FEATURE" in v for v in rep.violations)

    def test_oos_contamination_detected(self):
        sealed = R.SealedOOS(np.arange(50, 60))
        with pytest.raises(R.OOSAccessError):
            sealed.reveal("feature_selection")
        assert sealed.contamination_check().contaminated

    def test_changed_config_changes_experiment_hash(self):
        m1 = R.ExperimentManifest(experiment_id="A", hypothesis_id="H",
                                  experiment_type="RANKING", training_config_hash="t1")
        m2 = R.ExperimentManifest(experiment_id="B", hypothesis_id="H",
                                  experiment_type="RANKING", training_config_hash="t2")
        assert m1.experiment_hash != m2.experiment_hash

    def test_changed_dataset_changes_hash(self):
        m1 = R.ExperimentManifest(experiment_id="A", hypothesis_id="H",
                                  experiment_type="RANKING", dataset_hash="D1")
        m2 = R.ExperimentManifest(experiment_id="B", hypothesis_id="H",
                                  experiment_type="RANKING", dataset_hash="D2")
        assert m1.experiment_hash != m2.experiment_hash

    def test_altered_metric_detected_by_reproduction(self, tmp_path):
        art = R.ExperimentArtifact(root=tmp_path, experiment_id="E")
        art.freeze(manifest={"experiment_id": "E"}, sections={"metrics": {"rank_ic": 0.05}})
        res = R.reproduce(art, lambda s: {"metrics": {"rank_ic": 0.06}})
        assert not res.reproduced

    def test_modified_artifact_detected(self, tmp_path):
        from src.lifecycle._storage import atomic_write_json
        art = R.ExperimentArtifact(root=tmp_path, experiment_id="E")
        art.freeze(manifest={"experiment_id": "E"}, sections={"metrics": {"rank_ic": 0.05}})
        # tamper with a frozen section on disk
        atomic_write_json(art.dir / "metrics.json", {"rank_ic": 0.99})
        ok, mismatched = art.verify_integrity()
        assert not ok and "metrics" in mismatched

    def test_incorrect_pvalue_caught_by_correction(self):
        # a single p<0.05 that is really the best of 1000 trials must NOT be significant
        res = R.correct_multiple_testing([0.03], n_trials=1000,
                                         method=R.CorrectionMethod.HOLM)
        assert res.n_significant == 0

    def test_duplicate_experiment_detected(self, tmp_path):
        reg = R.ResearchExperimentRegistry(tmp_path)
        m = R.ExperimentManifest(experiment_id="E1", hypothesis_id="H", experiment_type="RANKING")
        reg.register(m)
        with pytest.raises(R.ExperimentImmutabilityError):
            reg.register(m)

    def test_overwritten_experiment_detected(self, tmp_path):
        art = R.ExperimentArtifact(root=tmp_path, experiment_id="E")
        art.freeze(manifest={"experiment_id": "E"}, sections={"metrics": {"x": 1}})
        with pytest.raises(R.ArtifactError):
            art.freeze(manifest={"experiment_id": "E"}, sections={"metrics": {"x": 2}})


# ══════════════════════════════════════════════════════════════════════════════
# §55 Security — no secrets in artifacts
# ══════════════════════════════════════════════════════════════════════════════

class TestSecurity:
    def test_artifact_rejects_secret_in_section(self, tmp_path):
        art = R.ExperimentArtifact(root=tmp_path, experiment_id="SEC1")
        with pytest.raises(R.ArtifactSecretLeak):
            art.freeze(manifest={"experiment_id": "SEC1"},
                       sections={"configuration": {"api_key": "SUPERSECRETVALUE123"}})

    def test_artifact_rejects_secret_in_manifest(self, tmp_path):
        art = R.ExperimentArtifact(root=tmp_path, experiment_id="SEC2")
        with pytest.raises(R.ArtifactSecretLeak):
            art.freeze(manifest={"experiment_id": "SEC2", "access_token": "abcd1234token"},
                       sections={"metrics": {"rank_ic": 0.05}})
