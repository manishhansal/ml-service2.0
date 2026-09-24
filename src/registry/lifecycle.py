"""
src.registry.lifecycle — champion / challenger / shadow lifecycle manager (Phase N/O).

Manages the model lifecycle flow required by the certification:

    challenger  (validated OOS candidate, cannot allocate capital)
        -> shadow      (receives real inference traffic, cannot allocate capital)
        -> compare     (shadow vs champion on identical live traffic)
        -> approval    (six-gate promotion + human approval token)
        -> champion    (production, allocates capital)

Rollback restores the last known-good champion on degradation / drift / failure.

This sits ON TOP of ModelRegistry (immutable artifact store + checksums) and
ModelPromotion (six-gate evaluation). It does NOT create a competing source of
truth: the registry remains canonical for artifacts; this manager tracks the
per-model role assignments (champion/challenger/shadow) and rollback history.

Requirements: Phase N, Phase O, Phase 45, Phase 46, 13_MODEL_REGISTRY_SPEC.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.logging_config import get_logger
from src.registry.promotion import ModelPromotion
from src.registry.registry import ModelRegistry
from src.schemas.base import PromotionOutcome

logger = get_logger(__name__)


@dataclass
class ShadowComparison:
    """Result of comparing a shadow model against the champion on live traffic."""

    n_observations: int
    shadow_ic: float
    champion_ic: float
    shadow_hit_rate: float
    champion_hit_rate: float
    shadow_beats_champion: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RoleState:
    """Role assignments for a single model family."""

    champion_version: str | None = None
    challenger_version: str | None = None
    shadow_version: str | None = None
    previous_champion_version: str | None = None  # for rollback
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class ModelNotAllowedToTradeError(Exception):
    """Raised when a non-champion model is asked to allocate capital."""


class ChampionChallengerManager:
    """
    Orchestrates champion/challenger/shadow roles and rollback for each model.

    Usage::

        mgr = ChampionChallengerManager(registry)
        mgr.register_challenger("market_regime", "1.0.0-...")
        mgr.promote_to_shadow("market_regime")
        cmp = mgr.evaluate_shadow("market_regime", comparison)
        decision = mgr.promote_shadow_to_champion("market_regime", challenger_metrics)
        mgr.rollback("market_regime")  # restore previous champion
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        promotion: ModelPromotion | None = None,
        state_path: Path | None = None,
    ) -> None:
        self._registry = registry or ModelRegistry()
        self._promotion = promotion or ModelPromotion()
        self._state_path = state_path or (self._registry._root / "_roles.json")
        self._roles: dict[str, RoleState] = self._load_state()

    # ── State persistence ─────────────────────────────────────────────────

    def _load_state(self) -> dict[str, RoleState]:
        if self._state_path.exists():
            try:
                raw = json.loads(self._state_path.read_text())
                return {k: RoleState(**v) for k, v in raw.items()}
            except Exception:
                return {}
        return {}

    def _save_state(self) -> None:
        data = {k: v.to_dict() for k, v in self._roles.items()}
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._state_path)

    def _role(self, model_name: str) -> RoleState:
        return self._roles.setdefault(model_name, RoleState())

    def _record(self, model_name: str, event: str, **details: Any) -> None:
        entry = {
            "event": event,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            **details,
        }
        self._role(model_name).history.append(entry)
        logger.info("lifecycle_event", model_name=model_name, lifecycle_stage=event, **details)

    # ── Lifecycle transitions ──────────────────────────────────────────────

    def register_challenger(self, model_name: str, version: str) -> None:
        """Assign a validated candidate as the challenger (cannot trade)."""
        role = self._role(model_name)
        role.challenger_version = version
        self._record(model_name, "challenger_registered", version=version)
        self._save_state()

    def promote_to_shadow(self, model_name: str) -> None:
        """Move the challenger into shadow (receives traffic, cannot allocate capital)."""
        role = self._role(model_name)
        if role.challenger_version is None:
            raise ValueError(f"No challenger registered for {model_name}.")
        role.shadow_version = role.challenger_version
        self._record(model_name, "promoted_to_shadow", version=role.shadow_version)
        self._save_state()

    def evaluate_shadow(
        self, model_name: str, comparison: ShadowComparison
    ) -> ShadowComparison:
        """Record a shadow-vs-champion comparison on identical live traffic."""
        self._record(
            model_name, "shadow_evaluated",
            shadow_ic=comparison.shadow_ic,
            champion_ic=comparison.champion_ic,
            beats=comparison.shadow_beats_champion,
        )
        self._save_state()
        return comparison

    def promote_shadow_to_champion(
        self,
        model_name: str,
        challenger_metrics: dict[str, Any],
        champion_metrics: dict[str, Any] | None = None,
        approval_token: str | None = None,
    ) -> PromotionOutcome:
        """
        Run the six-gate promotion. On PROMOTE (with approval token), the shadow
        becomes champion; the old champion is retained for rollback.

        A challenger/shadow NEVER becomes champion without passing all gates.
        """
        role = self._role(model_name)
        if role.shadow_version is None:
            raise ValueError(f"No shadow model for {model_name} to promote.")

        decision = self._promotion.evaluate_all_gates(challenger_metrics, champion_metrics)

        if decision.outcome != PromotionOutcome.PROMOTE:
            self._record(
                model_name, "promotion_rejected",
                outcome=decision.outcome.value,
                blocked_gates=decision.blocked_gates,
            )
            self._save_state()
            return decision.outcome

        # Gates passed — require an approval token for the final capital-allocating step.
        if not approval_token:
            self._record(model_name, "promotion_awaiting_approval",
                        version=role.shadow_version)
            self._save_state()
            return PromotionOutcome.BLOCKED

        # Promote: shadow → champion, old champion retained for rollback.
        role.previous_champion_version = role.champion_version
        role.champion_version = role.shadow_version
        role.shadow_version = None
        role.challenger_version = None
        self._record(
            model_name, "promoted_to_champion",
            new_champion=role.champion_version,
            previous_champion=role.previous_champion_version,
            approval_token=approval_token[:12] + "...",
        )
        self._save_state()
        return PromotionOutcome.PROMOTE

    # ── Rollback (Phase 46) ─────────────────────────────────────────────────

    def rollback(self, model_name: str, reason: str = "MANUAL") -> str | None:
        """
        Restore the last known-good champion. Returns the restored version or
        None if there is no previous champion to restore.
        """
        role = self._role(model_name)
        if role.previous_champion_version is None:
            self._record(model_name, "rollback_failed_no_previous", reason=reason)
            self._save_state()
            return None
        failed = role.champion_version
        role.champion_version = role.previous_champion_version
        role.previous_champion_version = None
        self._record(
            model_name, "rolled_back",
            restored=role.champion_version, failed=failed, reason=reason,
        )
        self._save_state()
        return role.champion_version

    # ── Capital-allocation guard (Phase 45) ─────────────────────────────────

    def assert_can_allocate_capital(self, model_name: str, version: str) -> None:
        """
        Raise unless *version* is the current CHAMPION for *model_name*.
        Challenger/shadow models can never allocate capital.
        """
        role = self._role(model_name)
        if role.champion_version != version:
            raise ModelNotAllowedToTradeError(
                f"Model {model_name} v{version} is not the champion "
                f"(champion={role.champion_version}); it may not allocate capital."
            )

    def get_roles(self, model_name: str) -> RoleState:
        return self._role(model_name)
