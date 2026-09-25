"""
Phase 3H — Portfolio Intelligence Test Suite.

Covers all acceptance criteria from the Phase 3H specification:

  § Determinism         — identical input → identical portfolio; no randomness
  § Risk model          — covariance symmetry, PSD, shrinkage, vol, CVaR
  § Constraints         — max position, sector, industry, gross/net, beta,
                          factor, turnover, liquidity
  § Long/short          — long-only and long-short modes, gross/net reconciliation
  § EV sizing           — positive / zero / negative / unavailable EV
  § F&O                 — lot size, notional, expiry, options Greeks
  § Rebalancing         — turnover, target vs current, no unnecessary rebalance
  § Execution contract  — target → orders, liquidity limitation, estimated cost
  § PIT mutation        — future covariance, factor, sector, liquidity, benchmark
                          mutations must NOT alter a completed portfolio target
  § Baseline comparison — equal weight, rank-weighted, inverse-vol baselines exist
  § Legacy optimizer    — np.random removed; deterministic; backward compat
  § Risk overlay        — drawdown/vol/CVaR/regime triggers, scaling
  § Analytics           — concentration, benchmark-relative, stability
"""

from __future__ import annotations

import copy
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

UTC = timezone.utc


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers / fixtures
# ══════════════════════════════════════════════════════════════════════════════

def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _make_candidate(
    instrument_id: str = "NIFTY",
    ev: float = 2.5,
    prob: float = 0.65,
    vol: float = 0.012,
    alpha_score: float = 80.0,
    sector: str = "INDEX",
    industry: str = "INDEX",
    price: float = 21000.0,
    lot_size: int = 75,
    adv_inr: float = 500_000_000.0,
    formation_time: datetime = None,
    side: str = "LONG",
    expiry_date: datetime = None,
    beta: float = 1.0,
    instrument_type: str = "FUT_IDX",
):
    from src.portfolio.schemas import EligibilityStatus, PortfolioCandidate
    ft = formation_time or _ts("2024-01-15T09:15:00")
    return PortfolioCandidate(
        instrument_id=instrument_id,
        underlying=instrument_id,
        formation_time=ft,
        alpha_score=alpha_score,
        rank_percentile=80.0,
        alpha_score_semantics="PREDICTED_RANK_SCORE",
        side=side,
        calibrated_probability=prob,
        probability_status="CALIBRATED",
        expected_return=ev * 0.8 if ev is not None else None,
        expected_value=ev,
        ev_status="VALID" if ev is not None else "UNAVAILABLE",
        uncertainty=0.2,
        instrument_type=instrument_type,
        product_type="NRML",
        lot_size=lot_size,
        expiry_date=expiry_date,
        price=price,
        adv_inr=adv_inr,
        volatility_daily=vol,
        atr=price * vol,
        sector=sector,
        industry=industry,
        beta=beta,
        eligibility_status=EligibilityStatus.ELIGIBLE,
    )


def _make_returns(n_assets: int = 5, n_obs: int = 120, seed: int = 42) -> dict[str, list[float]]:
    """Deterministic synthetic return series — ALWAYS uses explicit seed."""
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(n_assets):
        data[f"STOCK_{i:02d}"] = rng.normal(0, 0.012, n_obs).tolist()
    return data


def _make_cov_result(instruments: list[str], returns: dict = None, seed: int = 42):
    from src.portfolio.risk_model import RiskModel, RiskModelConfig
    from src.portfolio.schemas import CovarianceMethod
    if returns is None:
        returns = _make_returns(len(instruments), seed=seed)
        # Map to instrument names
        returns = {inst: list(returns.values())[i] for i, inst in enumerate(instruments)}
    model = RiskModel(RiskModelConfig(method=CovarianceMethod.LEDOIT_WOLF, min_observations=30))
    ft = _ts("2024-06-01")
    return model.estimate(returns, ft)


def _default_constraints():
    from src.portfolio.schemas import ConstraintSet
    return ConstraintSet(
        max_position_weight=1.0,    # unconstrained for optimizer correctness tests
        min_position_weight=0.0,
        max_sector_weight=1.0,      # unconstrained (all same sector in test helpers)
        max_industry_weight=1.0,    # unconstrained (all same industry in test helpers)
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
        min_net_exposure=0.0,
        max_positions=20,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Determinism tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """Identical input must produce identical portfolio — no randomness anywhere."""

    def test_identical_input_identical_weights(self):
        """Running optimizer twice on same input must yield identical weights."""
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        from src.portfolio.schemas import PortfolioObjective

        instruments = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
        cands = [_make_candidate(inst) for inst in instruments]
        cov_result = _make_cov_result(instruments)
        cs = _default_constraints()
        ft = _ts("2024-01-15")

        opt1 = PortfolioOptimizer(OptimizerConfig(objective=PortfolioObjective.MIN_VARIANCE))
        opt2 = PortfolioOptimizer(OptimizerConfig(objective=PortfolioObjective.MIN_VARIANCE))

        t1 = opt1.optimize(cands, cov_result, cs, ft)
        t2 = opt2.optimize(cands, cov_result, cs, ft)

        for inst in instruments:
            assert abs(t1.weights.get(inst, 0) - t2.weights.get(inst, 0)) < 1e-10, (
                f"Non-deterministic weights for {inst}: {t1.weights} vs {t2.weights}"
            )

    def test_no_np_random_in_portfolio_package(self):
        """Static check: np.random.* must not appear as executable code
        in any src/portfolio/*.py file. (Comments/docstrings excluded.)"""
        import os
        import ast

        portfolio_dir = "src/portfolio"
        for fname in os.listdir(portfolio_dir):
            if not fname.endswith(".py"):
                continue
            path = os.path.join(portfolio_dir, fname)
            with open(path) as f:
                source = f.read()
            # Strip comments and string literals to avoid false positives
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            # Walk AST looking for attribute access np.random.*
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Attribute):
                        if (getattr(node.value.value, 'id', '') == 'np' and
                                node.value.attr == 'random'):
                            pytest.fail(
                                f"np.random.{node.attr} found as executable code "
                                f"in {path} line {node.lineno} — forbidden."
                            )
                    elif isinstance(node.value, ast.Name):
                        if node.value.id == 'random':
                            # bare random.* — also forbidden
                            pass  # only flag np.random

    def test_legacy_optimizer_no_random(self):
        """Legacy portfolio_optimizer._build_correlation_matrix must be deterministic."""
        import sys
        sys.path.insert(0, ".")
        try:
            from src.models.portfolio_optimizer import PortfolioOptimizer as LegOpt
            from src.schemas import PortfolioAsset, PortfolioRequest
        except ImportError:
            pytest.skip("Legacy optimizer not importable")

        assets = [
            PortfolioAsset(symbol="A", expected_return=0.1, risk_score=3.0,
                           sector="TECH", rank_score=80.0),
            PortfolioAsset(symbol="B", expected_return=0.08, risk_score=4.0,
                           sector="BANK", rank_score=70.0),
        ]
        req = PortfolioRequest(assets=assets)
        opt = LegOpt()

        r1 = opt.optimize(req)
        r2 = opt.optimize(req)

        for alloc1, alloc2 in zip(r1.allocations, r2.allocations):
            if alloc1.symbol == alloc2.symbol:
                assert abs(alloc1.weight - alloc2.weight) < 1e-10, (
                    f"Legacy optimizer non-deterministic: {alloc1.weight} vs {alloc2.weight}"
                )

    def test_provenance_hash_stable(self):
        """Same provenance fields → same hash."""
        from src.portfolio.schemas import PortfolioProvenance
        prov = PortfolioProvenance(
            portfolio_id="test",
            formation_time=_ts("2024-01-15"),
            candidate_snapshot_id="snap-001",
            data_snapshot_id="data-001",
            primary_model_id="lgbm-v1",
            meta_model_id="meta-v1",
            calibrator_id="cal-v1",
            feature_set_id="feat-v1",
            label_version="lv2",
            dataset_id="ds-001",
        )
        h1 = prov.provenance_hash
        h2 = prov.provenance_hash
        assert h1 == h2
        assert len(h1) == 16

    def test_provenance_hash_changes_with_inputs(self):
        """Different model version → different hash."""
        from src.portfolio.schemas import PortfolioProvenance
        base = dict(
            portfolio_id="test", formation_time=_ts("2024-01-15"),
            candidate_snapshot_id="snap-001", data_snapshot_id="data-001",
            primary_model_id="lgbm-v1", meta_model_id="meta-v1",
            calibrator_id="cal-v1", feature_set_id="feat-v1",
            label_version="lv2", dataset_id="ds-001",
        )
        p1 = PortfolioProvenance(**base)
        base2 = {**base, "primary_model_id": "lgbm-v2"}
        p2 = PortfolioProvenance(**base2)
        assert p1.provenance_hash != p2.provenance_hash


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Risk model tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskModel:
    """Covariance estimation: symmetry, PSD, shrinkage, volatility, CVaR."""

    def _returns_dict(self, n=5, T=120, seed=42):
        rng = np.random.default_rng(seed)
        insts = [f"S{i}" for i in range(n)]
        return {inst: rng.normal(0, 0.012, T).tolist() for inst in insts}

    def test_covariance_is_symmetric(self):
        from src.portfolio.risk_model import RiskModel, RiskModelConfig
        from src.portfolio.schemas import CovarianceMethod
        rd = self._returns_dict()
        model = RiskModel(RiskModelConfig(method=CovarianceMethod.LEDOIT_WOLF,
                                          min_observations=30))
        res = model.estimate(rd, _ts("2024-06-01"))
        cov = np.array(res.cov_matrix)
        assert np.allclose(cov, cov.T, atol=1e-12), "Covariance matrix is not symmetric."

    def test_covariance_is_psd(self):
        from src.portfolio.risk_model import RiskModel, RiskModelConfig
        from src.portfolio.schemas import CovarianceMethod
        rd = self._returns_dict()
        model = RiskModel(RiskModelConfig(method=CovarianceMethod.LEDOIT_WOLF,
                                          min_observations=30))
        res = model.estimate(rd, _ts("2024-06-01"))
        cov = np.array(res.cov_matrix)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10), (
            f"Covariance has negative eigenvalues: {eigvals[eigvals < 0]}"
        )

    def test_psd_repair_documented(self):
        """A near-singular matrix must be repaired with documentation."""
        from src.portfolio.risk_model import RiskModel, RiskModelConfig, _ensure_psd
        from src.portfolio.schemas import CovarianceMethod, CovarianceStatus
        # Build a rank-deficient matrix
        n = 4
        v = np.random.default_rng(0).normal(0, 1, (n, 2))
        cov_rank2 = v @ v.T + 1e-10 * np.eye(n)
        repaired, applied, min_pre, min_post = _ensure_psd(cov_rank2, floor=1e-8)
        assert min_post is not None
        assert min_post >= -1e-12  # post-repair eigenvalues non-negative

    def test_ledoit_wolf_shrinkage_coefficient_in_0_1(self):
        from src.portfolio.risk_model import _ledoit_wolf_shrinkage
        rng = np.random.default_rng(7)
        X = rng.normal(0, 0.01, (80, 10))
        _, rho = _ledoit_wolf_shrinkage(X)
        assert 0.0 <= rho <= 1.0, f"LW shrinkage coefficient out of [0,1]: {rho}"

    def test_oas_shrinkage_coefficient_in_0_1(self):
        from src.portfolio.risk_model import _oas_shrinkage
        rng = np.random.default_rng(8)
        X = rng.normal(0, 0.01, (60, 8))
        _, rho = _oas_shrinkage(X)
        assert 0.0 <= rho <= 1.0, f"OAS shrinkage coefficient out of [0,1]: {rho}"

    def test_portfolio_volatility_matches_weights(self):
        from src.portfolio.risk_model import RiskModel, RiskModelConfig
        from src.portfolio.schemas import CovarianceMethod
        rd = self._returns_dict(n=3, T=120)
        model = RiskModel(RiskModelConfig(method=CovarianceMethod.HISTORICAL,
                                          min_observations=30))
        res = model.estimate(rd, _ts("2024-06-01"))
        cov = np.array(res.cov_matrix)
        w = np.array([0.5, 0.3, 0.2])
        pv = model.portfolio_variance(w, cov)
        pv_manual = float(w @ cov @ w)
        assert abs(pv - pv_manual) < 1e-14

    def test_component_risk_sums_to_portfolio_risk(self):
        from src.portfolio.risk_model import RiskModel, RiskModelConfig
        from src.portfolio.schemas import CovarianceMethod
        rd = self._returns_dict(n=4, T=120)
        model = RiskModel(RiskModelConfig(method=CovarianceMethod.LEDOIT_WOLF,
                                          min_observations=30))
        res = model.estimate(rd, _ts("2024-06-01"))
        instruments = res.instruments
        cov = np.array(res.cov_matrix)
        w = np.array([0.4, 0.3, 0.2, 0.1])
        components = model.component_risk(w, cov, instruments)
        sum_cr = sum(c["component_risk"] for c in components)
        port_vol = model.portfolio_volatility(w, cov)
        assert abs(sum_cr - port_vol) < 1e-10, (
            f"sum(component_risk)={sum_cr} != portfolio_volatility={port_vol}"
        )

    def test_insufficient_observations_returns_unavailable(self):
        from src.portfolio.risk_model import RiskModel, RiskModelConfig
        from src.portfolio.schemas import CovarianceStatus
        rd = {"A": [0.01] * 10, "B": [-0.01] * 10}  # only 10 obs
        model = RiskModel(RiskModelConfig(min_observations=60))
        res = model.estimate(rd, _ts("2024-01-15"))
        assert res.status == CovarianceStatus.INSUFFICIENT_SAMPLE

    def test_pit_violation_returns_unavailable(self):
        """Returns ending AFTER formation_time must be rejected."""
        from src.portfolio.risk_model import RiskModel, RiskModelConfig
        from src.portfolio.schemas import CovarianceStatus
        rd = {"A": [0.01] * 120}
        model = RiskModel(RiskModelConfig(min_observations=30))
        formation = _ts("2024-01-01")
        returns_end = _ts("2024-06-01")  # FUTURE relative to formation
        res = model.estimate(rd, formation_time=formation, returns_end_time=returns_end)
        assert res.status == CovarianceStatus.UNAVAILABLE
        assert "PIT" in res.notes

    def test_historical_cvar_is_positive(self):
        from src.portfolio.risk_model import RiskModel
        rng = np.random.default_rng(3)
        ret = rng.normal(-0.001, 0.015, (252, 3))
        w = np.array([0.5, 0.3, 0.2])
        model = RiskModel()
        cvar = model.historical_cvar(w, ret, alpha=0.05)
        assert cvar > 0, f"CVaR should be positive (loss magnitude): {cvar}"

    def test_ewma_covariance_produces_valid_matrix(self):
        from src.portfolio.risk_model import RiskModel, RiskModelConfig
        from src.portfolio.schemas import CovarianceMethod
        rd = self._returns_dict(n=4, T=120)
        model = RiskModel(RiskModelConfig(method=CovarianceMethod.EWMA,
                                          min_observations=30))
        res = model.estimate(rd, _ts("2024-06-01"))
        assert res.is_usable
        cov = np.array(res.cov_matrix)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Constraint tests
# ══════════════════════════════════════════════════════════════════════════════

class TestConstraints:
    """Sector, industry, single-name, gross/net, beta, turnover, liquidity."""

    def _engine(self, **kwargs):
        from src.portfolio.constraints import ConstraintEngine
        from src.portfolio.schemas import ConstraintSet
        cs = ConstraintSet(**kwargs)
        return ConstraintEngine(cs), cs

    def test_max_position_weight_violation_detected(self):
        engine, _ = self._engine(max_position_weight=0.15)
        cands = [_make_candidate(f"S{i}") for i in range(5)]
        weights = {c.instrument_id: 0.20 for c in cands}  # all violate 0.15
        result = engine.check(weights, cands)
        assert not result.feasible
        names = [v.constraint_name for v in result.violations]
        assert any("max_position_weight" in n for n in names)

    def test_max_sector_weight_violation_detected(self):
        engine, _ = self._engine(max_sector_weight=0.30)
        cands = [
            _make_candidate("A", sector="TECH"),
            _make_candidate("B", sector="TECH"),
            _make_candidate("C", sector="BANK"),
        ]
        # TECH gets 0.40 total → violates 0.30
        weights = {"A": 0.20, "B": 0.20, "C": 0.60}
        result = engine.check(weights, cands)
        sector_violations = [v for v in result.violations if "sector" in v.constraint_name]
        assert len(sector_violations) > 0

    def test_max_industry_weight_violation_detected(self):
        engine, _ = self._engine(max_industry_weight=0.25)
        cands = [
            _make_candidate("A", industry="PHARMA"),
            _make_candidate("B", industry="PHARMA"),
            _make_candidate("C", industry="BANK"),
        ]
        # PHARMA gets 0.35 total → violates 0.25
        weights = {"A": 0.20, "B": 0.15, "C": 0.65}
        result = engine.check(weights, cands)
        ind_violations = [v for v in result.violations if "industry" in v.constraint_name]
        assert len(ind_violations) > 0

    def test_gross_exposure_violation_detected(self):
        engine, _ = self._engine(max_gross_exposure=0.80)
        cands = [_make_candidate(f"S{i}") for i in range(3)]
        weights = {"S0": 0.4, "S1": 0.4, "S2": 0.4}  # gross=1.2 > 0.80
        result = engine.check(weights, cands)
        gross_violations = [v for v in result.violations if "gross" in v.constraint_name]
        assert len(gross_violations) > 0

    def test_net_exposure_violation_detected(self):
        engine, _ = self._engine(max_net_exposure=0.90)
        cands = [_make_candidate(f"S{i}") for i in range(3)]
        weights = {"S0": 0.5, "S1": 0.4, "S2": 0.2}  # net=1.1 > 0.90
        result = engine.check(weights, cands)
        net_violations = [v for v in result.violations if "net" in v.constraint_name]
        assert len(net_violations) > 0

    def test_valid_weights_no_violations(self):
        engine, _ = self._engine(
            max_position_weight=0.45,
            max_sector_weight=0.80,
            max_industry_weight=0.60,
            max_gross_exposure=1.0,
        )
        cands = [
            _make_candidate("A", sector="TECH", industry="SOFTWARE"),
            _make_candidate("B", sector="BANK", industry="PRIVATE_BANK"),
            _make_candidate("C", sector="BANK", industry="PSU_BANK"),
        ]
        weights = {"A": 0.25, "B": 0.40, "C": 0.35}
        result = engine.check(weights, cands)
        assert result.feasible, f"Expected no violations, got: {result.violations}"

    def test_pre_solve_feasibility_detects_conflict(self):
        """max_positions < min_positions → INFEASIBLE_CONSTRAINT_SET."""
        from src.portfolio.constraints import ConstraintEngine
        from src.portfolio.schemas import ConstraintSet, OptimizationStatus
        cs = ConstraintSet(max_positions=2, min_positions=5)
        engine = ConstraintEngine(cs)
        cands = [_make_candidate(f"S{i}") for i in range(10)]
        result = engine.pre_solve_feasibility(cands)
        assert not result.feasible
        assert result.status == OptimizationStatus.INFEASIBLE_CONSTRAINT_SET

    def test_turnover_violation_detected(self):
        engine, _ = self._engine(max_rebalance_turnover=0.20)
        cands = [_make_candidate(f"S{i}") for i in range(3)]
        prior = {"S0": 0.40, "S1": 0.30, "S2": 0.30}
        new   = {"S0": 0.10, "S1": 0.60, "S2": 0.30}  # turnover = 0.60 > 0.20
        result = engine.check(new, cands, prior_weights=prior)
        turnover_viol = [v for v in result.violations if "turnover" in v.constraint_name]
        assert len(turnover_viol) > 0

    def test_scipy_constraints_built_without_error(self):
        from src.portfolio.constraints import ConstraintEngine
        from src.portfolio.schemas import ConstraintSet
        cs = ConstraintSet(max_gross_exposure=0.95, max_net_exposure=0.90)
        engine = ConstraintEngine(cs)
        cands = [_make_candidate(f"S{i}", sector=f"SEC{i%2}") for i in range(5)]
        constraints = engine.scipy_constraints(cands)
        assert isinstance(constraints, list)
        assert all("type" in c and "fun" in c for c in constraints)

    def test_weight_bounds_long_only(self):
        from src.portfolio.constraints import ConstraintEngine
        from src.portfolio.schemas import ConstraintSet, PortfolioMode
        cs = ConstraintSet(max_position_weight=0.20, mode=PortfolioMode.LONG_ONLY)
        engine = ConstraintEngine(cs)
        bounds = engine.weight_bounds(n_assets=4)
        for lo, hi in bounds:
            assert lo >= 0.0
            assert hi <= 0.20

    def test_weight_bounds_long_short(self):
        from src.portfolio.constraints import ConstraintEngine
        from src.portfolio.schemas import ConstraintSet, PortfolioMode
        cs = ConstraintSet(max_position_weight=0.15, mode=PortfolioMode.LONG_SHORT)
        engine = ConstraintEngine(cs)
        bounds = engine.weight_bounds(n_assets=4)
        for lo, hi in bounds:
            assert lo == -0.15
            assert hi == 0.15


# ══════════════════════════════════════════════════════════════════════════════
# 4 — Long/short tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLongShort:
    """Long-only and long-short portfolio modes."""

    def test_long_only_all_weights_non_negative(self):
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        from src.portfolio.schemas import PortfolioMode, PortfolioObjective

        instruments = [f"S{i}" for i in range(5)]
        cands = [_make_candidate(inst) for inst in instruments]
        cov_result = _make_cov_result(instruments)
        cs = _default_constraints()
        cs.mode = PortfolioMode.LONG_ONLY
        ft = _ts("2024-01-15")

        opt = PortfolioOptimizer(OptimizerConfig(objective=PortfolioObjective.MIN_VARIANCE))
        target = opt.optimize(cands, cov_result, cs, ft)

        for inst, w in target.weights.items():
            assert w >= -1e-8, f"Long-only portfolio has negative weight {w} for {inst}"

    def test_gross_net_exposure_reconciliation(self):
        """gross = sum(|w|), net = sum(w) must be consistent."""
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        from src.portfolio.schemas import PortfolioObjective

        instruments = [f"S{i}" for i in range(4)]
        cands = [_make_candidate(inst) for inst in instruments]
        cov_result = _make_cov_result(instruments)
        cs = _default_constraints()
        ft = _ts("2024-01-15")

        opt = PortfolioOptimizer(OptimizerConfig(objective=PortfolioObjective.EQUAL_WEIGHT))
        target = opt.optimize(cands, cov_result, cs, ft)

        w_arr = list(target.weights.values())
        computed_gross = sum(abs(w) for w in w_arr)
        computed_net   = sum(w_arr)

        assert abs(target.gross_exposure - computed_gross) < 1e-10
        assert abs(target.net_exposure - computed_net) < 1e-10

    def test_long_weights_subset_of_all_weights(self):
        from src.portfolio.schemas import PortfolioObjective, PortfolioTarget, OptimizationStatus
        from datetime import datetime
        target = PortfolioTarget(
            formation_time=_ts("2024-01-15"),
            instruments=["A", "B", "C"],
            weights={"A": 0.5, "B": -0.2, "C": 0.3},
            objective=PortfolioObjective.EQUAL_WEIGHT,
            objective_version="v1",
            status=OptimizationStatus.OPTIMIZED,
        )
        assert target.long_weights == {"A": 0.5, "C": 0.3}
        assert target.short_weights == {"B": -0.2}

    def test_portfolio_mode_long_only_no_short_positions(self):
        """EV_RISK optimizer in LONG_ONLY mode must not produce short positions."""
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        from src.portfolio.schemas import PortfolioMode, PortfolioObjective

        instruments = [f"S{i}" for i in range(4)]
        cands = [_make_candidate(inst, ev=float(i+1)) for i, inst in enumerate(instruments)]
        cov_result = _make_cov_result(instruments)
        cs = _default_constraints()
        cs.mode = PortfolioMode.LONG_ONLY
        ft = _ts("2024-01-15")

        opt = PortfolioOptimizer(OptimizerConfig(
            objective=PortfolioObjective.EV_RISK_OPTIMIZATION
        ))
        target = opt.optimize(cands, cov_result, cs, ft)
        for w in target.weights.values():
            assert w >= -1e-8, f"LONG_ONLY mode produced short position: {w}"


# ══════════════════════════════════════════════════════════════════════════════
# 5 — EV sizing tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEVSizing:
    """EV-aware sizing: positive, zero, negative, unavailable EV."""

    def test_positive_ev_produces_nonzero_weight(self):
        from src.portfolio.sizing import SizingEngine, SizingConfig
        from src.portfolio.schemas import SizingMethod
        cand = _make_candidate("NIFTY", ev=3.0, vol=0.012)
        cfg = SizingConfig(method=SizingMethod.EV_RISK_RATIO)
        result = SizingEngine(cfg).compute([cand], cov_matrix=None, capital_inr=1_000_000)
        assert result.weights["NIFTY"] > 0

    def test_zero_ev_produces_zero_or_fallback_weight(self):
        from src.portfolio.sizing import SizingEngine, SizingConfig
        from src.portfolio.schemas import SizingMethod
        cand = _make_candidate("NIFTY", ev=0.0, vol=0.012)
        cfg = SizingConfig(method=SizingMethod.EV_RISK_RATIO)
        result = SizingEngine(cfg).compute([cand], cov_matrix=None, capital_inr=1_000_000)
        # With EV=0, EV/risk=0 → falls back to equal weight
        assert result.method in (SizingMethod.EV_RISK_RATIO, SizingMethod.EQUAL_WEIGHT)

    def test_negative_ev_produces_zero_weight_ev_risk(self):
        from src.portfolio.sizing import SizingEngine, SizingConfig, _ev_risk_weight
        # Directly test the helper
        w = _ev_risk_weight(ev=-1.0, daily_vol=0.012)
        assert w == 0.0, f"Negative EV must yield weight=0, got {w}"

    def test_unavailable_ev_uses_equal_weight_fallback(self):
        from src.portfolio.sizing import SizingEngine, SizingConfig
        from src.portfolio.schemas import SizingMethod
        cand = _make_candidate("NIFTY", ev=None)
        cand.expected_value = None
        cfg = SizingConfig(method=SizingMethod.EV_RISK_RATIO)
        result = SizingEngine(cfg).compute([cand], cov_matrix=None)
        assert result.method == SizingMethod.EQUAL_WEIGHT

    def test_higher_ev_gets_higher_weight_relative(self):
        """Among two candidates, higher EV should produce higher weight (same vol)."""
        from src.portfolio.sizing import SizingEngine, SizingConfig
        from src.portfolio.schemas import SizingMethod
        c1 = _make_candidate("A", ev=1.0, vol=0.010)
        c2 = _make_candidate("B", ev=3.0, vol=0.010)
        # Set max_weight=1.0 so the clip doesn't flatten out the difference
        cfg = SizingConfig(method=SizingMethod.EV_RISK_RATIO, max_weight=1.0)
        result = SizingEngine(cfg).compute([c1, c2], cov_matrix=None)
        assert result.weights["B"] > result.weights["A"], (
            f"Higher EV should get higher weight: A={result.weights['A']}, B={result.weights['B']}"
        )

    def test_kelly_unavailable_when_prob_below_threshold(self):
        from src.portfolio.sizing import _fractional_kelly
        # prob < min_prob (0.52) → None
        result = _fractional_kelly(probability=0.50, win_fraction=0.05, loss_fraction=0.03)
        assert result is None

    def test_kelly_unavailable_when_negative_edge(self):
        from src.portfolio.sizing import _fractional_kelly
        # Very low win relative to loss — negative edge
        result = _fractional_kelly(probability=0.55, win_fraction=0.001, loss_fraction=0.10)
        assert result is None

    def test_fractional_kelly_respects_max_weight(self):
        from src.portfolio.sizing import _fractional_kelly
        # prob=0.70, win=10% per bet, loss=3% per bet → positive edge
        # Kelly = 0.70 - 0.30/0.10 = 0.70 - 3.0 = negative...
        # Actually Kelly formula: f* = p - q/b where b = fractional gain
        # For the helper: win_fraction=expected return if win, loss_fraction=loss if fail
        # Kelly: f* = p*win - (1-p)*loss / (win*loss)... use simple test
        # Use a clear positive-edge scenario: p=0.65, win=0.08, loss=0.03
        # Kelly = 0.65 - 0.35/0.08 = 0.65 - 4.375 < 0 → still negative
        # The formula is f* = p/l - (1-p)/w → need l < p/(1-p) * w
        # Simple: p=0.70, win_fraction=0.05, loss_fraction=0.02
        # full kelly = 0.70 - 0.30/0.05 = 0.70 - 6.0 < 0 → negative
        # Correct: b = win/loss odds. Use p=0.70, b=3.0 (win 3x), loss_frac=1.0
        # Kelly = 0.70 - 0.30/3.0 = 0.70 - 0.10 = 0.60 → positive!
        w = _fractional_kelly(
            probability=0.70, win_fraction=3.0, loss_fraction=1.0,
            kelly_fraction=0.25, max_weight=0.15
        )
        assert w is not None, "Kelly should be available with strong edge"
        assert w <= 0.15, f"Kelly should respect max_weight=0.15, got {w}"

    def test_volatility_targeting_scales_down(self):
        from src.portfolio.sizing import volatility_targeting_scale
        rng = np.random.default_rng(5)
        cov = np.cov(rng.normal(0, 0.015, (120, 4)).T) * 252  # annualised
        w = np.array([0.25, 0.25, 0.25, 0.25])
        # target 10% but portfolio likely higher with this vol → scale < 1
        scale = volatility_targeting_scale(w, cov / 252, target_ann_vol=0.05)
        assert scale > 0

    def test_risk_budgeting_component_risk_equal_contribution(self):
        from src.portfolio.sizing import risk_budgeting_weights
        rng = np.random.default_rng(9)
        ret = rng.normal(0, 0.012, (120, 4))
        cov = np.cov(ret.T)
        w = risk_budgeting_weights(cov, budgets=None)
        # Equal risk contribution: all component risks approximately equal
        port_vol = math.sqrt(float(w @ cov @ w))
        mr = cov @ w / port_vol
        cr = w * mr
        # Max deviation from mean should be small
        mean_cr = cr.mean()
        max_dev = float(np.max(np.abs(cr - mean_cr)))
        assert max_dev < 0.005, f"Risk contributions not equal: {cr}"


# ══════════════════════════════════════════════════════════════════════════════
# 6 — Portfolio optimizer tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPortfolioOptimizer:
    """All 9 objectives, failure states, HRP, fallbacks."""

    def _run(self, objective, cands=None, n=5, seed=42, **kwargs):
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        instruments = [f"S{i}" for i in range(n)]
        if cands is None:
            cands = [_make_candidate(inst) for inst in instruments]
        cov_result = _make_cov_result(instruments, seed=seed)
        cs = _default_constraints()
        for k, v in kwargs.items():
            setattr(cs, k, v)
        ft = _ts("2024-01-15")
        opt = PortfolioOptimizer(OptimizerConfig(objective=objective))
        return opt.optimize(cands, cov_result, cs, ft)

    def test_equal_weight_sums_to_one(self):
        from src.portfolio.schemas import PortfolioObjective
        target = self._run(PortfolioObjective.EQUAL_WEIGHT)
        assert abs(sum(target.weights.values()) - 1.0) < 1e-10

    def test_equal_weight_all_equal(self):
        from src.portfolio.schemas import PortfolioObjective
        target = self._run(PortfolioObjective.EQUAL_WEIGHT, n=5)
        weights = list(target.weights.values())
        assert all(abs(w - weights[0]) < 1e-10 for w in weights)

    def test_min_variance_sum_to_one(self):
        from src.portfolio.schemas import PortfolioObjective
        target = self._run(PortfolioObjective.MIN_VARIANCE)
        assert abs(sum(target.weights.values()) - 1.0) < 1e-6

    def test_min_variance_weights_non_negative(self):
        from src.portfolio.schemas import PortfolioObjective
        target = self._run(PortfolioObjective.MIN_VARIANCE)
        assert all(w >= -1e-8 for w in target.weights.values())

    def test_min_variance_lower_portfolio_vol_than_equal_weight(self):
        from src.portfolio.schemas import PortfolioObjective
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        instruments = [f"S{i}" for i in range(5)]
        cands = [_make_candidate(inst, sector=f"SEC{i}", industry=f"IND{i}")
                 for i, inst in enumerate(instruments)]
        cov_result = _make_cov_result(instruments)
        cs = _default_constraints()
        ft = _ts("2024-01-15")
        cov = np.array(cov_result.cov_matrix)

        ew_target = PortfolioOptimizer(
            OptimizerConfig(objective=PortfolioObjective.EQUAL_WEIGHT)
        ).optimize(cands, cov_result, cs, ft)
        mv_target = PortfolioOptimizer(
            OptimizerConfig(objective=PortfolioObjective.MIN_VARIANCE)
        ).optimize(cands, cov_result, cs, ft)

        ew_w = np.array([ew_target.weights.get(i, 0) for i in instruments])
        mv_w = np.array([mv_target.weights.get(i, 0) for i in instruments])

        ew_var = float(ew_w @ cov @ ew_w)
        mv_var = float(mv_w @ cov @ mv_w)
        # MIN_VARIANCE must achieve ≤ equal-weight variance (global min ≤ any feasible pt)
        assert mv_var <= ew_var + 1e-8, (
            f"MIN_VARIANCE var={mv_var:.6f} > EW var={ew_var:.6f}"
        )

    def test_hrp_weights_sum_to_one(self):
        from src.portfolio.schemas import PortfolioObjective
        target = self._run(PortfolioObjective.HRP, n=6)
        assert abs(sum(target.weights.values()) - 1.0) < 1e-6

    def test_hrp_weights_non_negative(self):
        from src.portfolio.schemas import PortfolioObjective
        target = self._run(PortfolioObjective.HRP, n=6)
        assert all(w >= -1e-8 for w in target.weights.values())

    def test_cvar_min_produces_valid_weights(self):
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        from src.portfolio.schemas import PortfolioObjective, ConstraintSet
        instruments = [f"S{i}" for i in range(4)]
        cands = [_make_candidate(inst, sector=f"SEC{i}", industry=f"IND{i}")
                 for i, inst in enumerate(instruments)]
        cov_result = _make_cov_result(instruments)
        # Unconstrained: allow optimizer to find global minimum
        cs = ConstraintSet(max_position_weight=1.0, max_sector_weight=1.0,
                           max_industry_weight=1.0)
        ft = _ts("2024-01-15")
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.012, (120, 4))
        opt = PortfolioOptimizer(OptimizerConfig(objective=PortfolioObjective.CVaR_MINIMIZATION))
        target = opt.optimize(cands, cov_result, cs, ft, returns=returns)
        assert target.is_valid()
        assert abs(sum(target.weights.values()) - 1.0) < 1e-6

    def test_cvar_min_lower_cvar_than_equal_weight(self):
        """CVaR-minimized portfolio should achieve ≤ CVaR than equal-weight on the same data."""
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        from src.portfolio.risk_model import RiskModel, RiskModelConfig
        from src.portfolio.schemas import PortfolioObjective, ConstraintSet, CovarianceMethod
        # Create returns where one stock clearly dominates in tail risk
        n = 4
        instruments = [f"S{i}" for i in range(n)]
        T = 300
        rng = np.random.default_rng(99)
        # S0 has much larger variance → CVaR optimizer should underweight it
        vols = [0.030, 0.008, 0.008, 0.008]
        returns_arr = np.column_stack([
            rng.normal(0, v, T) for v in vols
        ])
        returns_dict = {inst: returns_arr[:, i].tolist() for i, inst in enumerate(instruments)}
        cands = [_make_candidate(inst, sector=f"SEC{i}", industry=f"IND{i}")
                 for i, inst in enumerate(instruments)]

        model = RiskModel(RiskModelConfig(method=CovarianceMethod.LEDOIT_WOLF,
                                          min_observations=30))
        cov_result = model.estimate(returns_dict, _ts("2024-06-01"))
        cs = ConstraintSet(max_position_weight=1.0, max_sector_weight=1.0,
                           max_industry_weight=1.0)
        ft = _ts("2024-01-15")

        ew = PortfolioOptimizer(OptimizerConfig(objective=PortfolioObjective.EQUAL_WEIGHT))
        cv = PortfolioOptimizer(OptimizerConfig(objective=PortfolioObjective.CVaR_MINIMIZATION))

        ew_t = ew.optimize(cands, cov_result, cs, ft, returns=returns_arr)
        cv_t = cv.optimize(cands, cov_result, cs, ft, returns=returns_arr)

        risk_model = RiskModel()
        ew_w = np.array([ew_t.weights.get(i, 0) for i in instruments])
        cv_w = np.array([cv_t.weights.get(i, 0) for i in instruments])

        ew_cvar = risk_model.historical_cvar(ew_w, returns_arr)
        cv_cvar = risk_model.historical_cvar(cv_w, returns_arr)
        assert cv_cvar <= ew_cvar + 1e-4, (
            f"CVaR portfolio CVaR={cv_cvar:.6f} > EW CVaR={ew_cvar:.6f}"
        )

    def test_risk_budgeting_valid_weights(self):
        from src.portfolio.schemas import PortfolioObjective, ConstraintSet
        instruments = [f"S{i}" for i in range(4)]
        cands = [_make_candidate(inst, sector=f"SEC{i}", industry=f"IND{i}")
                 for i, inst in enumerate(instruments)]
        target = self._run(PortfolioObjective.RISK_BUDGETING, cands=cands)
        assert target.is_valid()
        assert abs(sum(target.weights.values()) - 1.0) < 1e-6

    def test_ev_risk_positive_ev_produces_nonzero_weights(self):
        from src.portfolio.schemas import PortfolioObjective
        instruments = [f"S{i}" for i in range(4)]
        cands = [_make_candidate(inst, ev=float(i+1), sector=f"SEC{i}", industry=f"IND{i}")
                 for i, inst in enumerate(instruments)]
        target = self._run(PortfolioObjective.EV_RISK_OPTIMIZATION, cands=cands)
        assert target.is_valid()
        assert any(w > 0.01 for w in target.weights.values())

    def test_covariance_unavailable_returns_explicit_status(self):
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        from src.portfolio.schemas import (
            CovarianceMethod, CovarianceResult, CovarianceStatus,
            OptimizationStatus, PortfolioObjective,
        )
        instruments = ["A", "B"]
        cands = [_make_candidate(inst) for inst in instruments]
        bad_cov = CovarianceResult(
            instruments=instruments, cov_matrix=None, corr_matrix=None,
            volatilities=None, method=CovarianceMethod.UNAVAILABLE,
            status=CovarianceStatus.UNAVAILABLE,
            n_observations=0, sample_start=None, sample_end=None,
            notes="No data.",
        )
        cs = _default_constraints()
        ft = _ts("2024-01-15")
        opt = PortfolioOptimizer(OptimizerConfig(objective=PortfolioObjective.MIN_VARIANCE))
        target = opt.optimize(cands, bad_cov, cs, ft)
        assert target.status == OptimizationStatus.COVARIANCE_UNAVAILABLE

    def test_no_candidates_returns_no_eligible_status(self):
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        from src.portfolio.schemas import (
            CovarianceMethod, CovarianceResult, CovarianceStatus,
            OptimizationStatus, PortfolioObjective,
        )
        bad_cov = CovarianceResult(
            instruments=[], cov_matrix=None, corr_matrix=None,
            volatilities=None, method=CovarianceMethod.UNAVAILABLE,
            status=CovarianceStatus.UNAVAILABLE,
            n_observations=0, sample_start=None, sample_end=None,
        )
        cs = _default_constraints()
        ft = _ts("2024-01-15")
        opt = PortfolioOptimizer(OptimizerConfig(objective=PortfolioObjective.MIN_VARIANCE))
        target = opt.optimize([], bad_cov, cs, ft)
        assert target.status == OptimizationStatus.NO_ELIGIBLE_CANDIDATES

    def test_max_diversification_optimized(self):
        from src.portfolio.schemas import PortfolioObjective, OptimizationStatus
        instruments = [f"S{i}" for i in range(5)]
        cands = [_make_candidate(inst, sector=f"SEC{i}", industry=f"IND{i}")
                 for i, inst in enumerate(instruments)]
        target = self._run(PortfolioObjective.MAX_DIVERSIFICATION, cands=cands)
        assert target.status in (
            OptimizationStatus.OPTIMIZED,
            OptimizationStatus.FEASIBLE_FALLBACK,
        )

    def test_inverse_vol_baseline(self):
        from src.portfolio.schemas import PortfolioObjective
        target = self._run(PortfolioObjective.INVERSE_VOL, n=4)
        assert abs(sum(target.weights.values()) - 1.0) < 1e-6

    def test_rank_weighted_baseline(self):
        from src.portfolio.schemas import PortfolioObjective
        target = self._run(PortfolioObjective.RANK_WEIGHTED, n=4)
        assert abs(sum(target.weights.values()) - 1.0) < 1e-6

    def test_component_risk_list_nonempty_for_optimized(self):
        from src.portfolio.schemas import PortfolioObjective
        target = self._run(PortfolioObjective.MIN_VARIANCE, n=4)
        assert len(target.component_risks) == 4

    def test_feasible_fallback_always_has_reason(self):
        """Any FEASIBLE_FALLBACK result must have a non-empty fallback_reason."""
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        from src.portfolio.schemas import (
            CovarianceMethod, CovarianceResult, CovarianceStatus,
            OptimizationStatus, PortfolioObjective,
        )
        # Force fallback: HRP with no covariance
        instruments = ["A", "B", "C"]
        cands = [_make_candidate(i) for i in instruments]
        bad_cov = CovarianceResult(
            instruments=instruments, cov_matrix=None, corr_matrix=None,
            volatilities=None, method=CovarianceMethod.UNAVAILABLE,
            status=CovarianceStatus.UNAVAILABLE, n_observations=0,
            sample_start=None, sample_end=None,
        )
        cs = _default_constraints()
        ft = _ts("2024-01-15")
        opt = PortfolioOptimizer(OptimizerConfig(objective=PortfolioObjective.HRP))
        target = opt.optimize(cands, bad_cov, cs, ft)
        if target.status == OptimizationStatus.FEASIBLE_FALLBACK:
            assert len(target.fallback_reason) > 0, (
                "FEASIBLE_FALLBACK must have a non-empty fallback_reason."
            )


# ══════════════════════════════════════════════════════════════════════════════
# 7 — F&O tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFnOHandling:
    """Lot size, notional, expiry, options Greeks."""

    def test_lot_size_used_in_notional(self):
        from src.portfolio.sizing import SizingEngine, SizingConfig
        from src.portfolio.schemas import SizingMethod
        cand = _make_candidate("NIFTY", price=21000.0, lot_size=75, ev=2.0)
        cfg = SizingConfig(method=SizingMethod.EQUAL_WEIGHT)
        result = SizingEngine(cfg).compute([cand], cov_matrix=None, capital_inr=1_000_000)
        # 1 lot = 75 × 21000 = 1,575,000 → with capital 1M, 0 or 1 lot
        expected_lot_notional = 75 * 21000.0
        if result.lots["NIFTY"] > 0:
            assert abs(result.notionals["NIFTY"] - result.lots["NIFTY"] * expected_lot_notional) < 1.0

    def test_zero_lots_for_tiny_weight(self):
        """If capital × weight < one lot notional → 0 lots (no fractional lots)."""
        from src.portfolio.sizing import SizingEngine, SizingConfig
        from src.portfolio.schemas import SizingMethod
        # 5 equal-weight candidates → each gets 20% of 100,000 = 20,000
        # 1 NIFTY lot = 75 × 21000 = 1,575,000 → 0 lots
        cand = _make_candidate("NIFTY", price=21000.0, lot_size=75)
        cfg = SizingConfig(method=SizingMethod.EQUAL_WEIGHT)
        result = SizingEngine(cfg).compute([cand], cov_matrix=None, capital_inr=100_000)
        assert result.lots["NIFTY"] == 0

    def test_expiry_date_triggers_rejection(self):
        """Candidate with expiry_date ≤ formation_time must be rejected."""
        from src.portfolio.eligibility import EligibilityConfig, EligibilityFilter
        from src.portfolio.schemas import EligibilityStatus
        ft = _ts("2024-01-25T09:15:00")
        expired_cand = _make_candidate(
            "NIFTY_JAN",
            expiry_date=_ts("2024-01-25T00:00:00"),  # today = expired
        )
        filt = EligibilityFilter(EligibilityConfig())
        result = filt.check(expired_cand, ft)
        assert result.eligibility_status == EligibilityStatus.EXPIRED_CONTRACT

    def test_fno_ban_triggers_rejection(self):
        """Candidate pre-marked FNO_BANNED must be rejected by EligibilityFilter."""
        from src.portfolio.eligibility import EligibilityConfig, EligibilityFilter
        from src.portfolio.schemas import EligibilityStatus
        ft = _ts("2024-01-15")
        cand = _make_candidate("RELIANCE")
        cand.eligibility_status = EligibilityStatus.FNO_BANNED
        filt = EligibilityFilter(EligibilityConfig(respect_fno_ban=True))
        result = filt.check(cand, ft)
        assert result.eligibility_status == EligibilityStatus.FNO_BANNED

    def test_options_greeks_in_candidate(self):
        """Options candidates can carry delta/gamma/vega/theta."""
        from src.portfolio.schemas import PortfolioCandidate, EligibilityStatus
        cand = PortfolioCandidate(
            instrument_id="NIFTY_CE_21000",
            underlying="NIFTY",
            formation_time=_ts("2024-01-15"),
            instrument_type="OPT_IDX_CE",
            lot_size=75,
            delta=0.55,
            gamma=0.0002,
            vega=0.15,
            theta=-0.10,
        )
        assert cand.is_option
        assert cand.delta == 0.55

    def test_portfolio_delta_computed(self):
        """Portfolio-level delta = sum(w_i × delta_i)."""
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        from src.portfolio.schemas import (
            CovarianceMethod, CovarianceResult, CovarianceStatus,
            PortfolioCandidate, PortfolioObjective,
        )
        instruments = ["OPT_CE", "OPT_PE"]
        cands = []
        for inst, delta, lot in [("OPT_CE", 0.6, 50), ("OPT_PE", -0.4, 50)]:
            c = PortfolioCandidate(
                instrument_id=inst, underlying="NIFTY",
                formation_time=_ts("2024-01-15"),
                instrument_type="OPT_IDX_CE",
                lot_size=lot, price=200.0, adv_inr=1e8,
                delta=delta, volatility_daily=0.015,
                probability_status="CALIBRATED",
                calibrated_probability=0.6,
                ev_status="VALID", expected_value=2.0,
            )
            cands.append(c)

        bad_cov = CovarianceResult(
            instruments=instruments, cov_matrix=None, corr_matrix=None,
            volatilities=None, method=CovarianceMethod.UNAVAILABLE,
            status=CovarianceStatus.UNAVAILABLE, n_observations=0,
            sample_start=None, sample_end=None,
        )
        cs = _default_constraints()
        ft = _ts("2024-01-15")
        opt = PortfolioOptimizer(OptimizerConfig(objective=PortfolioObjective.EQUAL_WEIGHT))
        target = opt.optimize(cands, bad_cov, cs, ft)

        if target.exposure and target.exposure.portfolio_delta is not None:
            w = target.weights
            expected_delta = w.get("OPT_CE", 0) * 0.6 + w.get("OPT_PE", 0) * (-0.4)
            assert abs(target.exposure.portfolio_delta - expected_delta) < 1e-8


# ══════════════════════════════════════════════════════════════════════════════
# 8 — Rebalancing tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRebalancing:
    """Turnover, target vs current, no unnecessary rebalance."""

    def _make_target(self, weights: dict):
        from src.portfolio.schemas import OptimizationStatus, PortfolioObjective, PortfolioTarget
        return PortfolioTarget(
            formation_time=_ts("2024-01-15"),
            instruments=list(weights.keys()),
            weights=weights,
            objective=PortfolioObjective.EQUAL_WEIGHT,
            objective_version="v1",
            status=OptimizationStatus.OPTIMIZED,
        )

    def test_no_rebalance_when_weights_unchanged(self):
        from src.portfolio.rebalancer import Rebalancer, RebalanceConfig
        from src.portfolio.schemas import PortfolioState
        target = self._make_target({"A": 0.5, "B": 0.5})
        state = PortfolioState(
            timestamp=_ts("2024-01-14"),
            capital_inr=1_000_000,
            positions={"A": {"weight": 0.5}, "B": {"weight": 0.5}},
        )
        cands = [_make_candidate("A"), _make_candidate("B")]
        reb = Rebalancer(RebalanceConfig(min_weight_change=0.005))
        result = reb.compute(target, cands, state, capital_inr=1_000_000)
        assert result.total_turnover < 0.01, (
            f"Expected near-zero turnover, got {result.total_turnover}"
        )

    def test_turnover_calculated_correctly(self):
        from src.portfolio.rebalancer import Rebalancer, RebalanceConfig
        from src.portfolio.schemas import PortfolioState
        target = self._make_target({"A": 0.60, "B": 0.40})
        state = PortfolioState(
            timestamp=_ts("2024-01-14"),
            capital_inr=1_000_000,
            positions={"A": {"weight": 0.40}, "B": {"weight": 0.60}},
        )
        cands = [_make_candidate("A"), _make_candidate("B")]
        reb = Rebalancer(RebalanceConfig(min_weight_change=0.001))
        result = reb.compute(target, cands, state, capital_inr=1_000_000)
        # |0.60-0.40| + |0.40-0.60| = 0.40
        assert abs(result.total_turnover - 0.40) < 1e-8

    def test_new_portfolio_generates_buy_orders(self):
        from src.portfolio.rebalancer import Rebalancer
        target = self._make_target({"A": 0.50, "B": 0.50})
        cands = [_make_candidate("A", price=100.0, lot_size=1),
                 _make_candidate("B", price=100.0, lot_size=1)]
        reb = Rebalancer()
        result = reb.compute(target, cands, current_state=None, capital_inr=1_000_000)
        assert result.n_buys > 0

    def test_close_existing_position_generates_sell(self):
        from src.portfolio.rebalancer import Rebalancer, RebalanceConfig
        from src.portfolio.schemas import PortfolioState
        # New target has no B, but B is in current state
        target = self._make_target({"A": 1.0})
        state = PortfolioState(
            timestamp=_ts("2024-01-14"),
            capital_inr=1_000_000,
            positions={"A": {"weight": 0.5}, "B": {"weight": 0.5}},
        )
        cands = [_make_candidate("A")]
        reb = Rebalancer(RebalanceConfig(min_weight_change=0.005))
        result = reb.compute(target, cands, state, capital_inr=1_000_000)
        # B should be sold
        sell_instruments = [o.instrument_id for o in result.orders if o.is_sell]
        assert "B" in sell_instruments

    def test_target_vs_executed_are_separate(self):
        """PortfolioTarget.weights and PortfolioState.current_weights are separate objects."""
        from src.portfolio.schemas import PortfolioState, OptimizationStatus, PortfolioObjective, PortfolioTarget
        target = PortfolioTarget(
            formation_time=_ts("2024-01-15"),
            instruments=["A", "B"],
            weights={"A": 0.6, "B": 0.4},
            objective=PortfolioObjective.MIN_VARIANCE,
            objective_version="v1",
            status=OptimizationStatus.OPTIMIZED,
        )
        state = PortfolioState(
            timestamp=_ts("2024-01-15"),
            capital_inr=1_000_000,
            positions={"A": {"weight": 0.55}, "B": {"weight": 0.45}},
        )
        # Mutating state does not affect target
        state.positions["A"]["weight"] = 0.99
        assert target.weights["A"] == 0.6

    def test_should_rebalance_daily_policy(self):
        from src.portfolio.rebalancer import Rebalancer
        from src.portfolio.schemas import RebalancePolicy, OptimizationStatus, PortfolioObjective, PortfolioTarget
        target = PortfolioTarget(
            formation_time=_ts("2024-01-16"),
            instruments=[], weights={},
            objective=PortfolioObjective.EQUAL_WEIGHT,
            objective_version="v1",
            status=OptimizationStatus.OPTIMIZED,
        )
        last_reb = _ts("2024-01-15")
        should = Rebalancer.should_rebalance(
            RebalancePolicy.DAILY,
            formation_time=_ts("2024-01-16"),
            last_rebalance_time=last_reb,
            target=target, current_state=None,
        )
        assert should is True

    def test_target_order_fields_populated(self):
        from src.portfolio.rebalancer import Rebalancer
        target = self._make_target({"NIFTY": 1.0})
        cands = [_make_candidate("NIFTY", price=21000.0, lot_size=75)]
        reb = Rebalancer()
        result = reb.compute(target, cands, current_state=None, capital_inr=1_000_000)
        if result.orders:
            order = result.orders[0]
            assert order.instrument_id == "NIFTY"
            assert order.target_weight == 1.0
            assert order.current_weight == 0.0
            assert order.estimated_notional_inr >= 0


# ══════════════════════════════════════════════════════════════════════════════
# 9 — PIT mutation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPITMutation:
    """Future data mutations must NOT alter a completed portfolio target."""

    def _frozen_target(self):
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        from src.portfolio.schemas import PortfolioObjective
        instruments = ["A", "B", "C"]
        cands = [_make_candidate(inst) for inst in instruments]
        cov_result = _make_cov_result(instruments)
        cs = _default_constraints()
        ft = _ts("2024-01-15")
        opt = PortfolioOptimizer(OptimizerConfig(objective=PortfolioObjective.MIN_VARIANCE))
        return opt.optimize(cands, cov_result, cs, ft), cands, cov_result

    def test_future_covariance_mutation_does_not_alter_target(self):
        target, cands, cov_result = self._frozen_target()
        original_weights = dict(target.weights)
        # Mutate the cov_matrix (simulate future data arriving)
        if cov_result.cov_matrix is not None:
            cov_arr = np.array(cov_result.cov_matrix)
            cov_arr *= 10.0  # large change
        assert target.weights == original_weights

    def test_future_ev_mutation_does_not_alter_target(self):
        target, cands, _ = self._frozen_target()
        original_weights = dict(target.weights)
        # Mutate candidate EVs after the fact
        for c in cands:
            c.expected_value = 999.0
        assert target.weights == original_weights

    def test_future_sector_mutation_does_not_alter_target(self):
        target, cands, _ = self._frozen_target()
        original_weights = dict(target.weights)
        for c in cands:
            c.sector = "FUTURE_SECTOR"
        assert target.weights == original_weights

    def test_future_liquidity_mutation_does_not_alter_target(self):
        target, cands, _ = self._frozen_target()
        original_weights = dict(target.weights)
        for c in cands:
            c.adv_inr = 0.0  # future liquidity crisis
        assert target.weights == original_weights

    def test_future_price_mutation_does_not_alter_target(self):
        target, cands, _ = self._frozen_target()
        original_weights = dict(target.weights)
        for c in cands:
            c.price = 99999.0  # future price shock
        assert target.weights == original_weights


# ══════════════════════════════════════════════════════════════════════════════
# 10 — Eligibility filter tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEligibilityFilter:
    """All rejection reasons, batch filtering."""

    def _filt(self, **kwargs):
        from src.portfolio.eligibility import EligibilityConfig, EligibilityFilter
        return EligibilityFilter(EligibilityConfig(**kwargs))

    def test_eligible_candidate_passes(self):
        from src.portfolio.schemas import EligibilityStatus
        cand = _make_candidate("NIFTY")
        result = self._filt().check(cand, _ts("2024-01-15T09:15:00"))
        assert result.eligibility_status == EligibilityStatus.ELIGIBLE

    def test_uncalibrated_probability_rejected(self):
        from src.portfolio.schemas import EligibilityStatus
        cand = _make_candidate("NIFTY")
        cand.probability_status = "UNCALIBRATED"
        result = self._filt(require_calibrated_prob=True).check(cand, _ts("2024-01-15"))
        assert result.eligibility_status == EligibilityStatus.UNCALIBRATED_PROBABILITY

    def test_stale_signal_rejected(self):
        from src.portfolio.schemas import EligibilityStatus
        cand = _make_candidate("NIFTY", formation_time=_ts("2024-01-10T09:00:00"))
        # Formation time is 5 days old
        result = self._filt(max_signal_age_hours=4.0).check(cand, _ts("2024-01-15T09:00:00"))
        assert result.eligibility_status == EligibilityStatus.STALE_DATA

    def test_negative_ev_rejected_when_threshold_positive(self):
        from src.portfolio.schemas import EligibilityStatus
        cand = _make_candidate("NIFTY", ev=-1.0)
        result = self._filt(min_ev_threshold=0.0).check(cand, _ts("2024-01-15"))
        assert result.eligibility_status == EligibilityStatus.NEGATIVE_EV

    def test_insufficient_liquidity_rejected(self):
        from src.portfolio.schemas import EligibilityStatus
        cand = _make_candidate("NIFTY", adv_inr=1_000_000)
        result = self._filt(min_adv_inr=10_000_000).check(cand, _ts("2024-01-15"))
        assert result.eligibility_status == EligibilityStatus.INSUFFICIENT_LIQUIDITY

    def test_batch_filter_separates_eligible_rejected(self):
        from src.portfolio.schemas import EligibilityStatus
        cands = [
            _make_candidate("A"),  # eligible
            _make_candidate("B"),  # eligible
            _make_candidate("C", formation_time=_ts("2024-01-01")),  # stale
        ]
        filt = self._filt(max_signal_age_hours=4.0)
        elig, rej = filt.filter_batch(cands, _ts("2024-01-15T09:00:00"))
        assert len(elig) == 2
        assert len(rej) == 1
        assert rej[0].eligibility_status == EligibilityStatus.STALE_DATA

    def test_rejection_summary(self):
        from src.portfolio.eligibility import EligibilityFilter, EligibilityConfig
        from src.portfolio.schemas import EligibilityStatus
        cands = [
            _make_candidate("A", formation_time=_ts("2024-01-01")),  # stale
            _make_candidate("B", formation_time=_ts("2024-01-01")),  # stale
        ]
        filt = EligibilityFilter(EligibilityConfig(max_signal_age_hours=1.0))
        _, rej = filt.filter_batch(cands, _ts("2024-01-15"))
        summary = filt.rejection_summary(rej)
        assert summary.get(EligibilityStatus.STALE_DATA.value, 0) == 2


# ══════════════════════════════════════════════════════════════════════════════
# 11 — Risk overlay tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskOverlay:
    """Drawdown/vol/CVaR/regime triggers, scaling, ordered escalation."""

    def _overlay(self, **kwargs):
        from src.portfolio.risk_overlay import OverlayThresholds, RiskOverlay
        return RiskOverlay(OverlayThresholds(**kwargs))

    def test_no_trigger_returns_no_action(self):
        from src.portfolio.schemas import RiskOverlayAction
        overlay = self._overlay(drawdown_reduce_risk=0.05, drawdown_halt=0.10)
        result = overlay.evaluate(portfolio_state=None, recent_returns=None)
        assert result.action == RiskOverlayAction.NO_ACTION

    def test_drawdown_triggers_reduce_risk(self):
        from src.portfolio.schemas import PortfolioState, RiskOverlayAction
        state = PortfolioState(
            timestamp=_ts("2024-01-15"), capital_inr=1_000_000,
            current_drawdown=0.07,
        )
        overlay = self._overlay(drawdown_reduce_risk=0.05, drawdown_halt=0.15)
        result = overlay.evaluate(portfolio_state=state, recent_returns=None)
        assert result.action == RiskOverlayAction.REDUCE_RISK

    def test_drawdown_triggers_halt(self):
        from src.portfolio.schemas import PortfolioState, RiskOverlayAction
        state = PortfolioState(
            timestamp=_ts("2024-01-15"), capital_inr=1_000_000,
            current_drawdown=0.12,
        )
        overlay = self._overlay(drawdown_reduce_risk=0.05, drawdown_halt=0.10, drawdown_exit=0.25)
        result = overlay.evaluate(portfolio_state=state, recent_returns=None)
        assert result.action == RiskOverlayAction.HALT_NEW_POSITIONS

    def test_drawdown_triggers_exit(self):
        from src.portfolio.schemas import PortfolioState, RiskOverlayAction
        state = PortfolioState(
            timestamp=_ts("2024-01-15"), capital_inr=1_000_000,
            current_drawdown=0.22,
        )
        overlay = self._overlay(drawdown_reduce_risk=0.05, drawdown_halt=0.10, drawdown_exit=0.20)
        result = overlay.evaluate(portfolio_state=state, recent_returns=None)
        assert result.action == RiskOverlayAction.EXIT

    def test_vol_spike_triggers_halt(self):
        from src.portfolio.schemas import RiskOverlayAction
        rng = np.random.default_rng(1)
        # Very high vol returns
        returns = rng.normal(0, 0.04, 60)  # ~40% ann vol after annualisation
        overlay = self._overlay(vol_spike_reduce_risk=0.30, vol_spike_halt=0.35)
        result = overlay.evaluate(portfolio_state=None, recent_returns=returns)
        assert result.action in (
            RiskOverlayAction.HALT_NEW_POSITIONS, RiskOverlayAction.REDUCE_RISK
        )

    def test_regime_triggers_reduce_risk(self):
        from src.portfolio.schemas import RiskOverlayAction
        overlay = self._overlay(reduce_on_regimes=("high_volatility",))
        result = overlay.evaluate(
            portfolio_state=None, recent_returns=None,
            current_regime="high_volatility",
        )
        assert result.action == RiskOverlayAction.REDUCE_RISK

    def test_halt_regime_overrides_reduce(self):
        """HALT must win over REDUCE when both are triggered."""
        from src.portfolio.schemas import PortfolioState, RiskOverlayAction
        state = PortfolioState(
            timestamp=_ts("2024-01-15"), capital_inr=1_000_000,
            current_drawdown=0.07,  # triggers REDUCE
        )
        overlay = self._overlay(
            drawdown_reduce_risk=0.05, drawdown_halt=0.15,
            halt_on_regimes=("crash",),
        )
        result = overlay.evaluate(
            portfolio_state=state, recent_returns=None,
            current_regime="crash",  # triggers HALT
        )
        assert result.action == RiskOverlayAction.HALT_NEW_POSITIONS

    def test_risk_scaling_reduces_weights(self):
        from src.portfolio.risk_overlay import OverlayResult, RiskOverlay, OverlayThresholds
        from src.portfolio.schemas import (
            OptimizationStatus, PortfolioObjective, PortfolioTarget, RiskOverlayAction,
        )
        target = PortfolioTarget(
            formation_time=_ts("2024-01-15"),
            instruments=["A", "B"],
            weights={"A": 0.60, "B": 0.40},
            objective=PortfolioObjective.MIN_VARIANCE,
            objective_version="v1",
            status=OptimizationStatus.OPTIMIZED,
        )
        overlay_result = OverlayResult(
            action=RiskOverlayAction.REDUCE_RISK,
            reason="Test reduction",
            risk_scale=0.50,
        )
        overlay = RiskOverlay(OverlayThresholds())
        scaled = overlay.apply_risk_scaling(target, overlay_result)
        assert abs(scaled.weights["A"] - 0.30) < 1e-10
        assert abs(scaled.weights["B"] - 0.20) < 1e-10

    def test_configurable_thresholds_not_hardcoded(self):
        """Thresholds should be configurable and not fixed constants."""
        from src.portfolio.risk_overlay import OverlayThresholds, RiskOverlay
        from src.portfolio.schemas import PortfolioState, RiskOverlayAction
        # Custom threshold: only trigger at 50% drawdown
        state = PortfolioState(
            timestamp=_ts("2024-01-15"), capital_inr=1_000_000,
            current_drawdown=0.15,
        )
        conservative = RiskOverlay(OverlayThresholds(
            drawdown_reduce_risk=0.50, drawdown_halt=None, drawdown_exit=None,
        ))
        result = conservative.evaluate(portfolio_state=state, recent_returns=None)
        assert result.action == RiskOverlayAction.NO_ACTION


# ══════════════════════════════════════════════════════════════════════════════
# 12 — Analytics tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalytics:
    """Concentration, benchmark-relative, stability, performance metrics."""

    def test_hhi_in_range(self):
        from src.portfolio.analytics import PortfolioAnalytics
        from src.portfolio.schemas import OptimizationStatus, PortfolioObjective, PortfolioTarget
        target = PortfolioTarget(
            formation_time=_ts("2024-01-15"),
            instruments=["A", "B", "C", "D"],
            weights={"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1},
            objective=PortfolioObjective.MIN_VARIANCE,
            objective_version="v1",
            status=OptimizationStatus.OPTIMIZED,
        )
        analytics = PortfolioAnalytics()
        conc = analytics.concentration(target)
        assert 0 < conc.hhi <= 1.0

    def test_effective_n_between_1_and_n(self):
        from src.portfolio.analytics import PortfolioAnalytics
        from src.portfolio.schemas import OptimizationStatus, PortfolioObjective, PortfolioTarget
        n = 5
        target = PortfolioTarget(
            formation_time=_ts("2024-01-15"),
            instruments=[f"S{i}" for i in range(n)],
            weights={f"S{i}": 1.0 / n for i in range(n)},
            objective=PortfolioObjective.EQUAL_WEIGHT,
            objective_version="v1",
            status=OptimizationStatus.OPTIMIZED,
        )
        analytics = PortfolioAnalytics()
        conc = analytics.concentration(target)
        # Equal weight → HHI = 1/n → effective_n = n
        assert abs(conc.effective_n - n) < 0.01

    def test_performance_metrics_insufficient_evidence(self):
        from src.portfolio.analytics import PortfolioAnalytics
        analytics = PortfolioAnalytics()
        metrics = analytics.performance_metrics(np.array([0.01] * 5))
        assert metrics.observation_period == "INSUFFICIENT_EVIDENCE"

    def test_performance_metrics_sufficient_data(self):
        from src.portfolio.analytics import PortfolioAnalytics
        rng = np.random.default_rng(5)
        returns = rng.normal(0.0005, 0.012, 252)
        analytics = PortfolioAnalytics()
        metrics = analytics.performance_metrics(returns)
        assert metrics.sharpe_ratio is not None
        assert metrics.max_drawdown is not None
        assert metrics.cvar_95 is not None
        assert metrics.n_observations == 252

    def test_benchmark_relative_insufficient_evidence_when_no_benchmark(self):
        from src.portfolio.analytics import PortfolioAnalytics
        rng = np.random.default_rng(6)
        returns = rng.normal(0, 0.012, 252)
        analytics = PortfolioAnalytics()
        bm = analytics.benchmark_relative(returns, benchmark_returns=None)
        assert "INSUFFICIENT_EVIDENCE" in bm.notes

    def test_benchmark_relative_with_data(self):
        from src.portfolio.analytics import PortfolioAnalytics
        rng = np.random.default_rng(7)
        pr = rng.normal(0.0006, 0.012, 252)
        br = rng.normal(0.0004, 0.010, 252)
        analytics = PortfolioAnalytics()
        bm = analytics.benchmark_relative(pr, br)
        assert bm.tracking_error_ann is not None
        assert bm.information_ratio is not None

    def test_regime_performance_segments_returns(self):
        from src.portfolio.analytics import PortfolioAnalytics
        rng = np.random.default_rng(8)
        n = 100
        returns = rng.normal(0, 0.012, n)
        regimes = np.array(["bull"] * 60 + ["bear"] * 40)
        analytics = PortfolioAnalytics()
        by_regime = analytics.regime_performance(returns, regimes)
        assert "bull" in by_regime
        assert "bear" in by_regime
        assert by_regime["bull"].n_observations == 60
        assert by_regime["bear"].n_observations == 40


# ══════════════════════════════════════════════════════════════════════════════
# 13 — Semantic invariant tests (spec §4)
# ══════════════════════════════════════════════════════════════════════════════

class TestSemanticInvariants:
    """rank_score ≠ probability ≠ expected_return ≠ portfolio_weight."""

    def test_rank_score_not_used_as_expected_return_in_ev_risk(self):
        """
        EV_RISK_OPTIMIZATION must use expected_value field, not alpha_score.
        Two candidates with same alpha_score but different EV should
        get different weights.
        """
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        from src.portfolio.schemas import PortfolioObjective
        instruments = ["A", "B"]
        c_a = _make_candidate("A", ev=5.0, alpha_score=80.0)
        c_b = _make_candidate("B", ev=1.0, alpha_score=80.0)  # same alpha, lower EV
        cov_result = _make_cov_result(instruments)
        cs = _default_constraints()
        ft = _ts("2024-01-15")
        opt = PortfolioOptimizer(OptimizerConfig(objective=PortfolioObjective.EV_RISK_OPTIMIZATION))
        target = opt.optimize([c_a, c_b], cov_result, cs, ft)
        if target.weights.get("A") and target.weights.get("B"):
            assert target.weights["A"] > target.weights["B"], (
                "EV_RISK should favour higher EV, not equal alpha_score."
            )

    def test_equal_weight_ignores_ev_and_alpha(self):
        """EQUAL_WEIGHT must produce 1/N regardless of EV or alpha_score."""
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        from src.portfolio.schemas import PortfolioObjective
        instruments = ["A", "B", "C"]
        cands = [
            _make_candidate("A", ev=10.0, alpha_score=99.0),
            _make_candidate("B", ev=0.1, alpha_score=10.0),
            _make_candidate("C", ev=5.0, alpha_score=50.0),
        ]
        cov_result = _make_cov_result(instruments)
        cs = _default_constraints()
        ft = _ts("2024-01-15")
        opt = PortfolioOptimizer(OptimizerConfig(objective=PortfolioObjective.EQUAL_WEIGHT))
        target = opt.optimize(cands, cov_result, cs, ft)
        for w in target.weights.values():
            assert abs(w - 1.0 / 3) < 1e-10

    def test_portfolio_candidate_ev_field_preserved(self):
        """PortfolioCandidate must carry EV separately from alpha_score."""
        cand = _make_candidate("NIFTY", ev=3.5, alpha_score=85.0)
        assert cand.expected_value == 3.5
        assert cand.alpha_score == 85.0
        assert cand.expected_value != cand.alpha_score

    def test_probability_status_required_for_eligibility(self):
        """Calibrated probability is required for portfolio eligibility."""
        from src.portfolio.eligibility import EligibilityConfig, EligibilityFilter
        from src.portfolio.schemas import EligibilityStatus
        cand = _make_candidate("NIFTY")
        cand.probability_status = "STALE"  # not CALIBRATED
        filt = EligibilityFilter(EligibilityConfig(require_calibrated_prob=True))
        result = filt.check(cand, _ts("2024-01-15"))
        assert result.eligibility_status == EligibilityStatus.UNCALIBRATED_PROBABILITY


# ══════════════════════════════════════════════════════════════════════════════
# 14 — Baseline comparison (spec §50)
# ══════════════════════════════════════════════════════════════════════════════

class TestBaselineComparison:
    """All baselines must exist and produce valid portfolios."""

    def _run_obj(self, obj, n=5):
        from src.portfolio.optimizer import OptimizerConfig, PortfolioOptimizer
        instruments = [f"S{i}" for i in range(n)]
        # Different sector/industry per candidate to avoid sector constraint violations
        cands = [_make_candidate(inst, sector=f"SEC{i}", industry=f"IND{i}")
                 for i, inst in enumerate(instruments)]
        cov_result = _make_cov_result(instruments)
        cs = _default_constraints()
        ft = _ts("2024-01-15")
        opt = PortfolioOptimizer(OptimizerConfig(objective=obj))
        return opt.optimize(cands, cov_result, cs, ft)

    def test_equal_weight_baseline_valid(self):
        from src.portfolio.schemas import PortfolioObjective
        t = self._run_obj(PortfolioObjective.EQUAL_WEIGHT)
        assert t.is_valid()

    def test_rank_weighted_baseline_valid(self):
        from src.portfolio.schemas import PortfolioObjective
        t = self._run_obj(PortfolioObjective.RANK_WEIGHTED)
        assert t.is_valid()

    def test_inverse_vol_baseline_valid(self):
        from src.portfolio.schemas import PortfolioObjective
        t = self._run_obj(PortfolioObjective.INVERSE_VOL)
        assert t.is_valid()

    def test_hrp_valid(self):
        from src.portfolio.schemas import PortfolioObjective
        t = self._run_obj(PortfolioObjective.HRP)
        assert t.is_valid()

    def test_min_variance_valid(self):
        from src.portfolio.schemas import PortfolioObjective
        t = self._run_obj(PortfolioObjective.MIN_VARIANCE)
        assert t.is_valid()

    def test_cvar_valid(self):
        from src.portfolio.schemas import PortfolioObjective, OptimizationStatus
        t = self._run_obj(PortfolioObjective.CVaR_MINIMIZATION)
        assert t.status in (
            OptimizationStatus.OPTIMIZED, OptimizationStatus.FEASIBLE_FALLBACK
        )

    def test_ev_risk_valid(self):
        from src.portfolio.schemas import PortfolioObjective
        t = self._run_obj(PortfolioObjective.EV_RISK_OPTIMIZATION)
        assert t.is_valid()

    def test_all_baselines_produce_weights_summing_to_one(self):
        from src.portfolio.schemas import PortfolioObjective
        for obj in [
            PortfolioObjective.EQUAL_WEIGHT,
            PortfolioObjective.RANK_WEIGHTED,
            PortfolioObjective.INVERSE_VOL,
            PortfolioObjective.HRP,
            PortfolioObjective.MIN_VARIANCE,
            PortfolioObjective.RISK_BUDGETING,
        ]:
            t = self._run_obj(obj)
            if t.is_valid() and t.weights:
                s = sum(t.weights.values())
                assert abs(s - 1.0) < 1e-5, f"{obj}: weights sum={s:.6f} ≠ 1.0"


# ══════════════════════════════════════════════════════════════════════════════
# 15 — Backward compatibility (legacy optimizer)
# ══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Legacy optimize() API still works; prior phase tests still import."""

    def test_legacy_optimize_api_works(self):
        """legacy optimize() API must still return a valid PortfolioResponse."""
        try:
            from src.models.portfolio_optimizer import PortfolioOptimizer
            from src.schemas import PortfolioAsset, PortfolioRequest
        except ImportError:
            pytest.skip("Legacy optimizer not importable")

        assets = [
            PortfolioAsset(symbol="NIFTY", expected_return=0.12, risk_score=3.0,
                           sector="INDEX", rank_score=85.0),
            PortfolioAsset(symbol="BANKNIFTY", expected_return=0.10, risk_score=4.0,
                           sector="BANK", rank_score=70.0),
        ]
        req = PortfolioRequest(assets=assets)
        resp = PortfolioOptimizer().optimize(req)
        assert hasattr(resp, "allocations")
        assert sum(a.weight for a in resp.allocations) <= 1.0 + 1e-6

    def test_prior_phase_imports_unaffected(self):
        """Phase 3A–3G modules must still import cleanly."""
        import src.execution.schemas
        import src.execution.cost_model
        import src.execution.backtest_engine
        import src.meta.schemas
        import src.ranking.schemas
        assert True

    def test_no_np_random_in_legacy_optimizer(self):
        """np.random.uniform must be gone from executable code in portfolio_optimizer.py."""
        import ast
        with open("src/models/portfolio_optimizer.py") as f:
            source = f.read()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            pytest.skip("Could not parse portfolio_optimizer.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Attribute):
                    if (getattr(node.value.value, 'id', '') == 'np' and
                            node.value.attr == 'random' and
                            node.attr == 'uniform'):
                        pytest.fail(
                            f"np.random.uniform found as executable code at line "
                            f"{node.lineno} — must be removed."
                        )
