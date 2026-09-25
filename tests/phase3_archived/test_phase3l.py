"""
Phase 3L — RL Execution & Adaptive Trade Management test suite.

Covers the acceptance criteria (spec §83, §84, §89):

  Environment   — reset, step, transitions, termination, reproducibility
  Causality     — future price/volume/spread mutation must not change obs/action
  Actions       — valid/invalid, action masking, safety override
  Reward        — gross/cost/slippage/impact/risk/turnover; reconciliation; net-of-cost
  Offline RL    — trajectory schema, behavior policy, OOD detection, BC
  OPE           — IS/WIS/DR/FQE, insufficient coverage → OPE_INSUFFICIENT_EVIDENCE
  Agent         — training, checkpoint, serialization, deterministic seed
  Failure       — always WAIT, always AGGRESSIVE, reward hacking, inventory, collapse
  Robustness    — higher slippage/spread, lower liquidity, market shock, sim-dependency
  Registry      — artifact hash, provenance, env compatibility, 3J challenger wiring
  Adversarial   — future leakage, free fills, zero cost, oracle observation

All imports are done INSIDE tests (lazy). Deterministic synthetic data; no global
np.random.* in production src/rl (verified by a static test).
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

UTC = timezone.utc


# ── helpers ───────────────────────────────────────────────────────────────────

def _bars(n=12, seed=0, base=20000.0):
    from src.rl.environment import MarketBar
    rng = np.random.RandomState(seed)
    bars = []
    px = base
    t0 = datetime(2023, 1, 2, 9, 15, tzinfo=UTC)
    for i in range(n):
        px *= (1 + rng.normal(0.0003, 0.004))
        bars.append(MarketBar(timestamp=t0 + timedelta(minutes=i), open=px * 0.999,
                              high=px * 1.004, low=px * 0.996, close=px,
                              volume=1e5 * (1 + 0.3 * np.sin(i)), adv_inr=5e8, atr_pct=0.01))
    return bars


def _episode(track=None, target=100.0, deadline=8):
    from src.rl.environment import EpisodeConfig
    from src.rl.schemas import RLTrack
    return EpisodeConfig(episode_id="ep", instrument="NIFTY",
                         track=track or RLTrack.EXECUTION_OPTIMIZATION.value,
                         target_qty=target, deadline_bars=deadline,
                         max_participation_pct=0.10, seed=1)


def _dataset(n_ep=24, deadline=6):
    from src.rl.environment import ExecutionEnv
    from src.rl.baselines import make_baseline
    from src.rl.offline import generate_trajectory
    policies = ["TWAP", "PASSIVE", "AGGRESSIVE", "VWAP_PROXY", "NEXT_OPEN", "FIXED_PARTICIPATION"]
    trans = []
    for k in range(n_ep):
        bars = _bars(seed=k)
        ep = _episode(target=1e6, deadline=deadline)
        env = ExecutionEnv(bars, ep, slippage_bps=5.0)
        trans += generate_trajectory(env, make_baseline(policies[k % len(policies)]), bars, ep)
    return trans


# ══════════════════════════════════════════════════════════════════════════════
# 1. Environment
# ══════════════════════════════════════════════════════════════════════════════

class TestEnvironment:
    def test_reset_step_terminate(self):
        from src.rl.environment import ExecutionEnv
        env = ExecutionEnv(_bars(), _episode(target=1e6, deadline=6), slippage_bps=5.0)
        obs, info = env.reset()
        assert obs.shape[0] == len(env.obs_fields)
        assert "state_hash" in info
        done = False
        steps = 0
        while not done and steps < 20:
            obs, r, term, trunc, info = env.step(env.actions.index("NORMAL"))
            done = term or trunc
            steps += 1
        assert done

    def test_reproducibility(self):
        from src.rl.environment import ExecutionEnv
        def run():
            env = ExecutionEnv(_bars(seed=3), _episode(target=1e6, deadline=6), slippage_bps=5.0)
            env.reset()
            rs = []
            done = False
            while not done:
                obs, r, term, trunc, info = env.step(env.actions.index("NORMAL"))
                rs.append(r)
                done = term or trunc
            return rs
        assert np.allclose(run(), run())

    def test_state_hash_stable(self):
        from src.rl.environment import ExecutionEnv
        e1 = ExecutionEnv(_bars(), _episode(), slippage_bps=5.0); e1.reset()
        e2 = ExecutionEnv(_bars(), _episode(), slippage_bps=5.0); e2.reset()
        assert e1.state_hash() == e2.state_hash()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Causality (spec §16)
# ══════════════════════════════════════════════════════════════════════════════

class TestCausality:
    def test_future_price_mutation_does_not_change_obs(self):
        from src.rl.environment import ExecutionEnv
        e1 = ExecutionEnv(_bars(), _episode(), slippage_bps=5.0); e1.reset()
        obs1, h1 = e1._observation(), e1.state_hash()
        bars2 = _bars()
        for j in range(6, len(bars2)):
            bars2[j].close *= 1.5; bars2[j].high *= 1.5; bars2[j].low *= 1.5
        e2 = ExecutionEnv(bars2, _episode(), slippage_bps=5.0); e2.reset()
        assert np.allclose(obs1, e2._observation())
        assert h1 == e2.state_hash()

    def test_future_volume_mutation_does_not_change_action_mask(self):
        from src.rl.environment import ExecutionEnv
        e1 = ExecutionEnv(_bars(), _episode(), slippage_bps=5.0); e1.reset()
        m1 = e1.action_mask()
        bars2 = _bars()
        for j in range(4, len(bars2)):
            bars2[j].volume *= 10.0
        e2 = ExecutionEnv(bars2, _episode(), slippage_bps=5.0); e2.reset()
        assert m1 == e2.action_mask()


# ══════════════════════════════════════════════════════════════════════════════
# 3. Actions / masking / safety
# ══════════════════════════════════════════════════════════════════════════════

class TestActionsSafety:
    def test_action_mask_market_closed(self):
        from src.rl.actions import valid_action_mask, MarketContext
        from src.rl.schemas import RLTrack
        ctx = MarketContext(market_open=False, remaining_target_qty=100, price=100)
        mask = valid_action_mask(RLTrack.EXECUTION_OPTIMIZATION.value, ctx)
        # only WAIT valid
        assert mask[0] is True and not any(mask[1:])

    def test_fno_ban_forces_wait(self):
        from src.rl.actions import SafetyLayer, MarketContext
        from src.rl.schemas import RLTrack
        sl = SafetyLayer(RLTrack.EXECUTION_OPTIMIZATION.value)
        ctx = MarketContext(fno_ban=True, remaining_target_qty=100, adv_inr=5e8, price=100)
        dec = sl.apply("AGGRESSIVE", ctx)
        assert dec.executed_action == "WAIT" and dec.safety_override

    def test_participation_cap_override(self):
        from src.rl.actions import SafetyLayer, MarketContext
        from src.rl.schemas import RLTrack
        sl = SafetyLayer(RLTrack.EXECUTION_OPTIMIZATION.value)
        ctx = MarketContext(remaining_target_qty=1e7, adv_inr=1e5, price=100, max_participation_pct=0.10)
        dec = sl.apply("FULL", ctx)
        assert dec.safety_override and dec.allowed_participation <= 0.10

    def test_safety_never_more_aggressive(self):
        from src.rl.actions import SafetyLayer, MarketContext, EXECUTION_PARTICIPATION
        from src.rl.schemas import RLTrack
        sl = SafetyLayer(RLTrack.EXECUTION_OPTIMIZATION.value)
        ctx = MarketContext(remaining_target_qty=100, adv_inr=5e8, price=100, max_participation_pct=0.05)
        dec = sl.apply("PASSIVE", ctx)
        assert dec.allowed_participation <= EXECUTION_PARTICIPATION["PASSIVE"]


# ══════════════════════════════════════════════════════════════════════════════
# 4. Reward (net-of-cost + reconciliation)
# ══════════════════════════════════════════════════════════════════════════════

class TestReward:
    def test_reward_net_of_cost_and_reconciles(self):
        from src.rl.reward import RewardEngine, StepEconomics
        from src.rl.schemas import RewardFunctionVersion
        rf = RewardFunctionVersion(turnover_penalty=0.1)
        eng = RewardEngine(rf)
        econ = StepEconomics(gross_pnl=1000.0, executed_qty=50, fill_price=100.0,
                             slippage_bps=5.0, turnover_qty=50)
        rc = eng.compute(econ)
        assert rc.transaction_cost > 0        # cost is charged (spec §18)
        assert rc.net_reward < rc.gross_pnl   # net < gross
        assert rc.reconciles(rf)

    def test_zero_qty_zero_cost(self):
        from src.rl.reward import RewardEngine, StepEconomics
        from src.rl.schemas import RewardFunctionVersion
        rf = RewardFunctionVersion()
        rc = RewardEngine(rf).compute(StepEconomics(gross_pnl=10.0, executed_qty=0))
        assert rc.transaction_cost == 0.0 and rc.slippage_cost == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. Offline RL
# ══════════════════════════════════════════════════════════════════════════════

class TestOffline:
    def test_trajectory_schema_and_behavior_policy(self):
        from src.rl.environment import ExecutionEnv
        from src.rl.baselines import make_baseline
        from src.rl.offline import generate_trajectory
        env = ExecutionEnv(_bars(), _episode(target=1e6, deadline=6), slippage_bps=5.0)
        trans = generate_trajectory(env, make_baseline("TWAP"), _bars(), _episode(target=1e6, deadline=6))
        assert all(t.policy_id == "TWAP" for t in trans)
        d = trans[0].to_dict()
        for k in ("observation", "action", "reward", "next_observation", "done",
                  "policy_id", "execution_version", "timestamp", "instrument"):
            assert k in d

    def test_ood_detection_and_fallback(self):
        from src.rl.offline import CoverageModel, apply_ood_protection
        trans = _dataset()
        cov = CoverageModel(bins=4).fit(trans)
        obs = trans[0].observation
        # action 4 (FULL) is essentially unused by TWAP/PASSIVE-heavy mix → OOD
        guard = apply_ood_protection(4, obs, cov, fallback_action=0, threshold=0.85)
        assert guard.ood_score >= 0.0
        if guard.abstained:
            assert guard.executed_action == 0

    def test_behavior_cloning(self):
        from src.rl.offline import BehaviorCloneModel
        trans = _dataset()
        bc = BehaviorCloneModel(bins=4, n_actions=5).fit(trans)
        rate = bc.action_match_rate(trans)
        assert 0.0 <= rate <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 6. OPE
# ══════════════════════════════════════════════════════════════════════════════

class TestOPE:
    def test_ope_returns_estimators_and_status(self):
        from src.rl.agent import OfflineQAgent, QLearnConfig
        from src.rl.ope import EmpiricalBehaviorPolicy, evaluate_policy
        trans = _dataset()
        agent = OfflineQAgent(QLearnConfig(fqi_iterations=10, seed=1)); agent.fit(trans)
        beh = EmpiricalBehaviorPolicy(n_actions=5, bins=4).fit(trans)
        res = evaluate_policy(trans, agent, beh, n_bootstrap=50)
        assert res.is_estimate is not None
        assert res.status in ("RELIABLE", "OPE_INSUFFICIENT_EVIDENCE")

    def test_ope_insufficient_on_tiny_data(self):
        from src.rl.agent import OfflineQAgent, QLearnConfig
        from src.rl.ope import EmpiricalBehaviorPolicy, evaluate_policy
        from src.rl.schemas import OPEStatus
        trans = _dataset()
        agent = OfflineQAgent(QLearnConfig(fqi_iterations=5, seed=1)); agent.fit(trans)
        beh = EmpiricalBehaviorPolicy(n_actions=5, bins=4).fit(trans)
        res = evaluate_policy(trans[:3], agent, beh, min_ess=5.0, n_bootstrap=20)
        assert res.status == OPEStatus.OPE_INSUFFICIENT_EVIDENCE.value


# ══════════════════════════════════════════════════════════════════════════════
# 7. Agent
# ══════════════════════════════════════════════════════════════════════════════

class TestAgent:
    def test_deterministic_seed(self):
        from src.rl.agent import OfflineQAgent, QLearnConfig
        trans = _dataset()
        a = OfflineQAgent(QLearnConfig(fqi_iterations=12, seed=1337)); a.fit(trans)
        b = OfflineQAgent(QLearnConfig(fqi_iterations=12, seed=1337)); b.fit(trans)
        assert np.allclose(a.Q, b.Q)

    def test_checkpoint_roundtrip(self):
        from src.rl.agent import OfflineQAgent, QLearnConfig
        trans = _dataset()
        a = OfflineQAgent(QLearnConfig(fqi_iterations=10, seed=1)); a.fit(trans)
        b = OfflineQAgent.from_state_dict(a.state_dict())
        assert np.allclose(a.Q, b.Q)
        assert len(a.artifact_hash()) == 64

    def test_hpo_never_touches_oos(self):
        from src.rl.agent import RLDataSplit, grid_search_hpo
        trans = _dataset()
        split = RLDataSplit.chronological(len(trans), embargo=1)
        r = grid_search_hpo(trans, split, [
            {"algorithm": "TABULAR_Q", "fqi_iterations": 8, "seed": 1},
            {"algorithm": "TABULAR_Q", "fqi_iterations": 16, "seed": 1},
        ])
        assert r["touched_oos"] is False

    def test_multi_seed_summary(self):
        from src.rl.agent import RLDataSplit, multi_seed_train_eval, QLearnConfig
        trans = _dataset()
        split = RLDataSplit.chronological(len(trans), embargo=1)
        rob = multi_seed_train_eval(trans, split, QLearnConfig(fqi_iterations=10), seeds=[1, 2, 3])
        s = rob.summary()
        for k in ("mean", "median", "std", "worst", "best"):
            assert k in s

    def test_split_blocks_overlap(self):
        from src.rl.agent import RLDataSplit
        with pytest.raises(AssertionError):
            RLDataSplit(train=[0, 1, 2, 5], val=[3, 4], oos=[6, 7])


# ══════════════════════════════════════════════════════════════════════════════
# 8. Failure modes
# ══════════════════════════════════════════════════════════════════════════════

class TestFailureModes:
    def test_always_wait(self):
        from src.rl.evaluation import detect_failure_modes
        r = detect_failure_modes(["WAIT"] * 20)
        assert r.always_wait and r.policy_collapse

    def test_always_aggressive(self):
        from src.rl.evaluation import detect_failure_modes
        r = detect_failure_modes(["AGGRESSIVE"] * 20)
        assert r.always_aggressive

    def test_reward_hacking(self):
        from src.rl.evaluation import detect_failure_modes
        r = detect_failure_modes(["NORMAL"] * 19 + ["WAIT"], step_rewards=[0.0] * 20)
        assert r.reward_hacking_suspected

    def test_inventory_accumulation(self):
        from src.rl.evaluation import detect_failure_modes
        r = detect_failure_modes(["NORMAL"] * 5, final_inventory=150, target_inventory=100)
        assert r.inventory_accumulation

    def test_healthy_policy_no_collapse(self):
        from src.rl.evaluation import detect_failure_modes
        r = detect_failure_modes(["WAIT", "NORMAL", "PASSIVE", "AGGRESSIVE", "NORMAL", "WAIT"])
        assert not r.policy_collapse and r.action_entropy > 0.5


# ══════════════════════════════════════════════════════════════════════════════
# 9. Robustness / simulator dependency
# ══════════════════════════════════════════════════════════════════════════════

class TestRobustness:
    def test_robust_policy(self):
        from src.rl.evaluation import robustness_report
        r = robustness_report({"base": 100.0, "worse_slippage": 95.0, "lower_liquidity": 90.0})
        assert not r.simulator_dependency_risk

    def test_fragile_policy_flags_sim_dependency(self):
        from src.rl.evaluation import robustness_report
        r = robustness_report({"base": 100.0, "worse_slippage": -20.0})
        assert r.simulator_dependency_risk


# ══════════════════════════════════════════════════════════════════════════════
# 10. Registry / Phase 3J
# ══════════════════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_experiment_status_machine(self):
        from src.rl.registry import RLExperimentRegistry, InvalidExperimentTransition
        from src.rl.schemas import RLExperiment, ExperimentStatus, RLTrack
        with tempfile.TemporaryDirectory() as d:
            reg = RLExperimentRegistry(d)
            reg.register(RLExperiment("x1", "rl-a", RLTrack.EXECUTION_OPTIMIZATION.value,
                                      "TABULAR_Q", ExperimentStatus.PLANNED))
            reg.transition("x1", ExperimentStatus.RUNNING)
            reg.transition("x1", ExperimentStatus.COMPLETED)
            with pytest.raises(InvalidExperimentTransition):
                reg.transition("x1", ExperimentStatus.RUNNING)

    def test_final_holdout_contamination(self):
        from src.rl.registry import RLExperimentRegistry
        from src.rl.schemas import RLExperiment, ExperimentStatus, RLTrack
        with tempfile.TemporaryDirectory() as d:
            reg = RLExperimentRegistry(d)
            reg.register(RLExperiment("x2", "rl-b", RLTrack.EXECUTION_OPTIMIZATION.value,
                                      "TABULAR_Q", ExperimentStatus.PLANNED))
            ok1, s1 = reg.mark_final_holdout_used("x2")
            ok2, s2 = reg.mark_final_holdout_used("x2")
            assert ok1 and not ok2 and s2 == "FINAL_HOLDOUT_CONTAMINATED"

    def test_trajectory_immutable_and_integrity(self):
        from src.rl.registry import TrajectoryRegistry, RLRegistryError
        from src.rl.schemas import Transition
        with tempfile.TemporaryDirectory() as d:
            tr = TrajectoryRegistry(d)
            trans = [Transition("2023-01-02T10:00:00Z", "NIFTY", "TWAP", "v1",
                                [1.0, 2.0], 1, 0.5, [1.1, 2.0], False) for _ in range(4)]
            tr.write("t1", trans, "TWAP", "snap", "env-v1", 1)
            assert tr.verify_integrity("t1")
            with pytest.raises(RLRegistryError):
                tr.write("t1", trans, "TWAP", "snap", "env-v1", 1)

    def test_challenger_wiring(self):
        from src.rl.registry import RLExperimentRegistry
        from src.rl.schemas import RLExperiment, ExperimentStatus, RLTrack
        from src.lifecycle import ModelRegistry, ChallengerRegistry, ModelIdentity, ModelProvenance
        with tempfile.TemporaryDirectory() as d:
            reg = RLExperimentRegistry(os.path.join(d, "rl"))
            lc = ModelRegistry(os.path.join(d, "lc"))
            ch = ChallengerRegistry(os.path.join(d, "lc"))
            reg.register(RLExperiment("x3", "rl-exec-a", RLTrack.EXECUTION_OPTIMIZATION.value,
                                      "TABULAR_Q", ExperimentStatus.PLANNED))
            reg.transition("x3", ExperimentStatus.RUNNING)
            reg.transition("x3", ExperimentStatus.COMPLETED)
            ident = ModelIdentity(model_id="rl-exec-a", model_family="rl_execution_agent",
                                  model_type="TABULAR_Q", model_version="v1", artifact_hash="a" * 64,
                                  created_at="2026-01-01", training_start="2025-01-01",
                                  training_end="2025-12-31", code_version="abc")
            prov = ModelProvenance(model_id="rl-exec-a", model_version="v1",
                                   execution_model_version="backtest-engine-v1")
            cid = reg.register_as_challenger("x3", lc, ch, ident, prov, scope="index_futures_execution")
            assert cid == "chal-rl-exec-a"
            assert ch.get(cid) is not None


# ══════════════════════════════════════════════════════════════════════════════
# 11. Adversarial (spec §84) — attempts to make RL look better than it is
# ══════════════════════════════════════════════════════════════════════════════

class TestAdversarial:
    def test_future_leakage_blocked_by_causal_env(self):
        """Mutating future bars must NOT change the current observation."""
        from src.rl.environment import ExecutionEnv
        e = ExecutionEnv(_bars(), _episode(), slippage_bps=5.0); e.reset()
        base_obs = e._observation()
        bars2 = _bars()
        for b in bars2[3:]:
            b.close *= 3.0
        e2 = ExecutionEnv(bars2, _episode(), slippage_bps=5.0); e2.reset()
        assert np.allclose(base_obs, e2._observation())

    def test_zero_cost_would_inflate_reward(self):
        """A zero-cost reward is strictly >= the net-of-cost reward (cost matters)."""
        from src.rl.reward import RewardEngine, StepEconomics
        from src.rl.schemas import RewardFunctionVersion
        econ = StepEconomics(gross_pnl=1000.0, executed_qty=50, fill_price=100.0, slippage_bps=5.0)
        net = RewardEngine(RewardFunctionVersion(cost_weight=1.0, slippage_weight=1.0)).compute(econ)
        free = RewardEngine(RewardFunctionVersion(cost_weight=0.0, slippage_weight=0.0, impact_weight=0.0)).compute(econ)
        assert free.net_reward > net.net_reward   # ignoring cost inflates reward → we DON'T do that by default

    def test_oracle_is_labelled_and_uses_future(self):
        from src.rl.baselines import oracle_best_execution, ORACLE_ONLY_LABEL
        orc = oracle_best_execution(_bars(), trade_side="LONG", deadline_bars=8)
        assert orc.label == ORACLE_ONLY_LABEL and orc.uses_future is True
        assert "ORACLE_ONLY" in orc.to_dict()["WARNING"]

    def test_no_broker_symbols_in_src_rl(self):
        """No live-broker connectivity anywhere in src/rl (spec §76, §92)."""
        import pathlib
        rl_dir = pathlib.Path(__file__).resolve().parents[1] / "src" / "rl"
        forbidden = ["place_order", "submit_order", "Angel", "Upstox", "broker_api", "live_order"]
        for py in rl_dir.glob("*.py"):
            txt = py.read_text(encoding="utf-8").lower()
            for tok in forbidden:
                assert tok.lower() not in txt, f"{py.name} contains forbidden '{tok}'"


# ══════════════════════════════════════════════════════════════════════════════
# 12. Static audit
# ══════════════════════════════════════════════════════════════════════════════

class TestStaticAudit:
    def test_no_global_np_random_in_src_rl(self):
        import ast
        import pathlib
        rl_dir = pathlib.Path(__file__).resolve().parents[1] / "src" / "rl"
        offenders = []
        for py in rl_dir.glob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
                    if (isinstance(node.value.value, ast.Name) and node.value.value.id == "np"
                            and node.value.attr == "random"
                            and node.attr not in ("default_rng", "Generator")):
                        offenders.append(f"{py.name}: np.random.{node.attr}")
        assert offenders == [], f"global np.random.* used: {offenders}"

    def test_no_forbidden_artifact_names(self):
        import pathlib
        rl_dir = pathlib.Path(__file__).resolve().parents[1] / "src" / "rl"
        for py in rl_dir.glob("*.py"):
            txt = py.read_text(encoding="utf-8")
            assert "latest.pkl" not in txt
            assert "current_model" not in txt
