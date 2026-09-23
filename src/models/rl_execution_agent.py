"""
RLExecutionAgent — FinRL-X based RL execution timing agent.

Design:
- Uses a trained FinRL-X PPO or SAC model artifact when available.
- Falls back to a deterministic rule-based policy (PredictionProvenance.HEURISTIC)
  when no trained artifact is loaded.
- 7-action discrete action space: ENTER_NOW, WAIT, SCALE_IN, PARTIAL_EXIT,
  FULL_EXIT, TIGHTEN_STOP, TRAIL_STOP.
- Observation space: 10 features from ExecutionState.
- Responds within 50ms at p95.
- Supports offline policy evaluation (OPE) using importance sampling.

Requirements: Req 9.1, Req 9.2, Req 9.3, Req 9.7, Req 9.8
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.logging_config import get_logger
from src.schemas.base import ExecutionAction, PredictionProvenance
from src.schemas.predictions import ExecutionDecision

logger = get_logger(__name__)

# ── Action space ──────────────────────────────────────────────────────────────

ALL_ACTIONS = list(ExecutionAction)
ACTION_SPACE_SIZE = len(ALL_ACTIONS)  # must be 7

# ── Observation feature names (in order) ─────────────────────────────────────

OBSERVATION_FEATURES = [
    "unrealized_pnl_pct",
    "time_in_trade_minutes",
    "regime_encoded",
    "volume_ratio",
    "price_vs_vwap",
    "atr_norm",           # atr / entry
    "momentum",
    "iv_regime_encoded",  # 0=CRUSH 1=STABLE 2=SPIKE
    "news_impact_score",
    "current_risk_score",
]

REGIME_ENCODING = {
    "strong_bull": 5, "bull": 4, "sideways": 3,
    "volatile": 2, "bear": 1, "crash": 0,
}


class RLExecutionAgent:
    """
    RL execution timing agent (FinRL-X PPO/SAC with rule-based fallback).

    Without a trained artifact: uses a deterministic heuristic policy.
    With a trained artifact: uses FinRL-X model inference.

    Usage::

        agent = RLExecutionAgent()
        decision = agent.act({
            "symbol": "NIFTY", "direction": "LONG",
            "entry": 22000.0, "current_price": 22150.0,
            "stop_loss": 21800.0, "target": 22400.0,
            "unrealized_pnl_pct": 0.68, "time_in_trade_minutes": 30,
            "regime": "bull", "volume_ratio": 1.1,
            "price_vs_vwap": 0.003, "atr": 200.0, "momentum": 0.5,
        })
        print(decision.action, decision.confidence, decision.provenance)
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self._model: Any = None
        self._model_version: str = "heuristic-v1"

        if model_path is not None and model_path.exists():
            self._load_model(model_path)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def action_space_size(self) -> int:
        """Number of discrete actions = 7."""
        return ACTION_SPACE_SIZE

    @property
    def has_trained_model(self) -> bool:
        """True when a FinRL-X model artifact is loaded."""
        return self._model is not None

    @property
    def model_version(self) -> str:
        return self._model_version

    # ── Public interface ───────────────────────────────────────────────────────

    def act(self, state: dict[str, Any]) -> ExecutionDecision:
        """
        Select an execution action for the current trade state.

        Args:
            state: dict with ExecutionState fields.

        Returns:
            ExecutionDecision with action, confidence, provenance.
        """
        if self._model is not None:
            return self._model_act(state)
        return self._heuristic_act(state)

    # ── Rule-based heuristic policy (Task 35.2) ────────────────────────────────

    def _heuristic_act(self, state: dict[str, Any]) -> ExecutionDecision:
        """
        Rule-based execution policy.

        Priority order:
        1. unrealized_pnl_pct > 3.0  → TRAIL_STOP (lock in large profits)
        2. unrealized_pnl_pct > 1.5  → TRAIL_STOP (profitable, protect gains)
        3. price very close to stop (<0.5% of entry) with negative PnL → FULL_EXIT
        4. price near stop (<1% of entry) with negative PnL → TIGHTEN_STOP
        5. time_in_trade_minutes > 300 (near session end) → PARTIAL_EXIT
        6. Strong momentum + positive PnL + above VWAP → SCALE_IN
        7. Default → WAIT
        """
        pnl_pct = float(state.get("unrealized_pnl_pct", 0))
        time_in_trade = int(state.get("time_in_trade_minutes", 0))
        momentum = float(state.get("momentum", 0))
        price_vs_vwap = float(state.get("price_vs_vwap", 0))
        current_price = float(state.get("current_price", 0))
        stop_loss = float(state.get("stop_loss", 0))
        entry = float(state.get("entry", current_price or 1))

        # Compute stop proximity as fraction of entry price
        stop_distance_pct = abs(current_price - stop_loss) / entry if entry > 0 else 1.0

        # Rule priority
        if pnl_pct > 3.0:
            action = ExecutionAction.TRAIL_STOP
            confidence = 0.85
            rationale = f"Large gain ({pnl_pct:.1f}%) — trailing stop to lock profits"

        elif pnl_pct > 1.5:
            action = ExecutionAction.TRAIL_STOP
            confidence = 0.72
            rationale = f"Profitable trade ({pnl_pct:.1f}%) — trailing stop activated"

        elif stop_distance_pct < 0.005 and pnl_pct < -0.5:
            # Very close to stop with meaningful negative PnL → exit immediately
            action = ExecutionAction.FULL_EXIT
            confidence = 0.78
            rationale = (
                f"Price {stop_distance_pct * 100:.2f}% from stop — "
                "exiting to avoid slippage"
            )

        elif stop_distance_pct < 0.01 and pnl_pct < 0:
            # Approaching stop loss with negative PnL → tighten stop
            action = ExecutionAction.TIGHTEN_STOP
            confidence = 0.68
            rationale = "Price approaching stop — tightening stop loss"

        elif time_in_trade > 300:
            # Near end of trading session
            action = ExecutionAction.PARTIAL_EXIT
            confidence = 0.65
            rationale = f"Session near close ({time_in_trade} min) — partial exit"

        elif momentum > 0.5 and pnl_pct > 0.5 and price_vs_vwap > 0:
            action = ExecutionAction.SCALE_IN
            confidence = 0.60
            rationale = "Strong momentum + positive trend — scaling in"

        else:
            action = ExecutionAction.WAIT
            confidence = 0.55
            rationale = "No strong signal — waiting for confirmation"

        return ExecutionDecision(
            action=action,
            confidence=confidence,
            new_stop_loss=None,
            exit_pct=None,
            rationale=rationale,
            provenance=PredictionProvenance.HEURISTIC,
        )

    # ── FinRL model inference ──────────────────────────────────────────────────

    def _model_act(self, state: dict[str, Any]) -> ExecutionDecision:
        """FinRL-X model inference. Falls back to heuristic on any error."""
        try:
            obs = self._state_to_observation(state)
            action_idx, _states = self._model.predict(obs, deterministic=True)
            action = ALL_ACTIONS[int(action_idx) % ACTION_SPACE_SIZE]
            confidence = 0.75  # SB3 doesn't expose action probs without rollout

            return ExecutionDecision(
                action=action,
                confidence=confidence,
                new_stop_loss=None,
                exit_pct=None,
                rationale=f"RL policy: {action.value}",
                provenance=PredictionProvenance.TRAINED_MODEL,
            )
        except Exception as exc:
            logger.warning("rl_agent_inference_failed", error=str(exc))
            return self._heuristic_act(state)

    def _state_to_observation(self, state: dict[str, Any]) -> "Any":
        """Convert state dict to a numpy observation vector (shape [1, 10])."""
        import numpy as np  # type: ignore[import-untyped]

        regime_str = state.get("regime", "sideways")
        if hasattr(regime_str, "value"):
            regime_str = regime_str.value
        regime_enc = float(REGIME_ENCODING.get(str(regime_str), 3))

        entry = float(state.get("entry", 1) or 1)
        atr = float(state.get("atr", 0) or 0)

        iv_regime_map = {"CRUSH": 0, "STABLE": 1, "SPIKE": 2}
        iv_raw = state.get("iv_regime", "STABLE")
        if hasattr(iv_raw, "value"):
            iv_raw = iv_raw.value
        iv_enc = float(iv_regime_map.get(str(iv_raw) if iv_raw else "STABLE", 1))

        obs = [
            float(state.get("unrealized_pnl_pct", 0)),
            float(state.get("time_in_trade_minutes", 0)) / 375.0,  # normalise [0,1]
            regime_enc / 5.0,
            float(state.get("volume_ratio", 1)) - 1.0,
            float(state.get("price_vs_vwap", 0)),
            (atr / entry) if entry > 0 else 0.0,
            float(state.get("momentum", 0)),
            iv_enc / 2.0,
            float(state.get("news_impact_score", 0)),
            float(state.get("current_risk_score", 0)) / 10.0,
        ]
        return np.array([obs], dtype=float)

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self, model_path: Path) -> None:
        """
        Load a FinRL-X trained PPO or SAC artifact from *model_path*.

        Silently falls back to heuristic mode if stable_baselines3 is not
        installed or the artifact cannot be loaded.
        """
        try:
            from stable_baselines3 import PPO, SAC  # type: ignore[import-untyped]

            # Try PPO first, then SAC
            for cls in (PPO, SAC):
                try:
                    self._model = cls.load(str(model_path))
                    self._model_version = model_path.stem
                    logger.info(
                        "rl_agent_model_loaded",
                        algo=cls.__name__,
                        path=str(model_path),
                    )
                    return
                except Exception:
                    continue

            logger.warning(
                "rl_agent_load_failed",
                path=str(model_path),
                error="Neither PPO nor SAC could load the artifact",
            )
        except ImportError:
            logger.warning(
                "stable_baselines3_not_installed",
                message="Using heuristic fallback",
            )
