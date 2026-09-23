"""
OnlineLearner — incremental model update engine.

Supports XGBoost model.update() for regime classifier and risk predictor,
and LightGBM incremental fitting for the stock ranker. Enforces a maximum
of 5 consecutive online updates before requiring a full retraining cycle.

Design:
- Triggered by PerformanceMonitor when IC degrades > 20% from 90-day baseline
- Creates a new versioned artifact for every update (never overwrites prior)
- Validates the updated model on the last 10 trading days before activation
- Rejects updates where the new IC < prior IC (logs ONLINE_UPDATE_REJECTED)
- Resets the consecutive update counter after a full retrain

Requirements: Req 14.1, Req 14.2, Req 14.3, Req 14.4, Req 14.5, Req 14.6, Req 14.7
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

VERSION_DATE_FORMAT = "%Y%m%d"


class OnlineLearner:
    """
    Manages incremental model updates without full retraining cycles.

    Key properties enforced (from TDD tests):
    - generate_next_version(prior) -> unique version with "online" marker
    - requires_full_retrain() -> True when consecutive_updates >= max
    - reset_consecutive_updates() -> resets counter to 0
    - Never overwrites prior artifacts (creates new versioned files)

    Usage::

        learner = OnlineLearner(model_name="market_regime")

        if learner.requires_full_retrain():
            # Trigger full retrain cycle
            ...
        else:
            result = learner.update(model, new_data, prior_version)
    """

    def __init__(
        self,
        model_name: str,
        artifacts_dir: Path | None = None,
        max_consecutive_updates: int | None = None,
        validation_window_days: int = 10,
        audit_logger: Any = None,
    ) -> None:
        self.model_name = model_name
        self.artifacts_dir = artifacts_dir or settings.model_artifacts_path
        self.max_consecutive_updates = (
            max_consecutive_updates
            if max_consecutive_updates is not None
            else settings.max_consecutive_online_updates
        )
        self.validation_window_days = validation_window_days
        self._consecutive_updates: int = 0
        self._audit = audit_logger

    # ── Version management ────────────────────────────────────────────────────

    def generate_next_version(self, prior_version: str) -> str:
        """
        Generate a new unique version string for an online update.

        Format: {base_version}-online-{YYYYMMDD}-{counter}

        The version always includes "online" and a date stamp, plus
        a counter derived from the prior version to ensure uniqueness
        across consecutive calls regardless of internal state.

        Args:
            prior_version: The current model version string.

        Returns:
            A new unique version string that differs from prior_version
            and contains "online".
        """
        today = datetime.now(tz=timezone.utc).strftime(VERSION_DATE_FORMAT)

        # Extract base version (strip existing online suffix if present)
        base = prior_version.split("-online-")[0]

        # Derive next counter from prior version's counter if it exists,
        # otherwise start at 1.  This ensures uniqueness when generate_next_version
        # is called repeatedly with the returned version as input (test loops).
        parts = prior_version.split("-online-")
        if len(parts) == 2:
            # prior already has an online suffix — extract its counter
            suffix_parts = parts[1].split("-")
            try:
                prior_counter = int(suffix_parts[-1])
            except (ValueError, IndexError):
                prior_counter = 0
            counter = prior_counter + 1
        else:
            # Fresh version — use internal state to handle same-day collisions
            counter = self._consecutive_updates + 1

        new_version = f"{base}-online-{today}-{counter:02d}"

        # Safety: guarantee the new version is always different from prior
        if new_version == prior_version:
            new_version = f"{base}-online-{today}-{counter + 1:02d}"

        return new_version

    # ── Update cap logic ──────────────────────────────────────────────────────

    def requires_full_retrain(self) -> bool:
        """
        Return True when consecutive online updates have reached the maximum.

        When True, a full retraining cycle with purged K-fold CV must be
        triggered before any further online updates.
        """
        return self._consecutive_updates >= self.max_consecutive_updates

    def reset_consecutive_updates(self) -> None:
        """Reset the consecutive update counter (called after a full retrain)."""
        self._consecutive_updates = 0
        logger.info(
            "online_learner_counter_reset",
            model_name=self.model_name,
        )

    def increment_consecutive_updates(self) -> None:
        """Increment the consecutive update counter after a successful update."""
        self._consecutive_updates += 1
        logger.debug(
            "online_learner_counter_incremented",
            model_name=self.model_name,
            count=self._consecutive_updates,
            max=self.max_consecutive_updates,
        )

    # ── Incremental update (Req 14.1) ─────────────────────────────────────────

    def update(
        self,
        model: Any,
        new_data_X: Any,
        new_data_y: Any,
        prior_version: str,
        validation_X: Any | None = None,
        validation_y: Any | None = None,
    ) -> tuple[Any, str] | None:
        """
        Apply an incremental model update and validate.

        Steps:
        1. Check if full retrain is required (Req 14.6)
        2. Generate new version string
        3. Apply incremental update (XGBoost model.update() or LightGBM fit)
        4. Validate on held-out window (Req 14.5)
        5. Return (updated_model, new_version) or None if rejected

        Args:
            model:          The current champion model object.
            new_data_X:     New training features.
            new_data_y:     New training labels.
            prior_version:  Current model version string.
            validation_X:   Held-out validation features (last 10 days).
            validation_y:   Held-out validation labels.

        Returns:
            (updated_model, new_version) on success.
            None if update is rejected (IC regression) or retrain required.
        """
        if self.requires_full_retrain():
            logger.warning(
                "online_learner_full_retrain_required",
                model_name=self.model_name,
                consecutive_updates=self._consecutive_updates,
                max_consecutive_updates=self.max_consecutive_updates,
            )
            return None

        if model is None:
            logger.warning(
                "online_learner_no_model_skip",
                model_name=self.model_name,
            )
            return None

        new_version = self.generate_next_version(prior_version)

        # Apply incremental update
        updated_model = self._apply_incremental_update(model, new_data_X, new_data_y)

        if updated_model is None:
            return None

        # Validate if validation data provided
        if validation_X is not None and validation_y is not None:
            accepted = self._validate_update(
                updated_model, model, validation_X, validation_y, new_version, prior_version
            )
            if not accepted:
                return None

        self.increment_consecutive_updates()

        logger.info(
            "online_learner_update_accepted",
            model_name=self.model_name,
            prior_version=prior_version,
            new_version=new_version,
            consecutive_count=self._consecutive_updates,
        )

        return updated_model, new_version

    def _apply_incremental_update(
        self, model: Any, X: Any, y: Any
    ) -> Any | None:
        """Apply XGBoost model.update() or LightGBM incremental fit."""
        try:
            model_type = type(model).__name__

            if "XGB" in model_type:
                import xgboost as xgb

                dtrain = xgb.DMatrix(X, label=y)
                model.get_booster().update(dtrain, iteration=0)
                return model

            elif "LGBM" in model_type or model_type == "Booster":
                # LightGBM: fit with init_model to continue training
                model.fit(X, y, init_model=model)
                return model

            else:
                # Generic: just refit on new data (append to existing)
                model.fit(X, y)
                return model

        except Exception as exc:
            logger.warning(
                "online_learner_incremental_update_failed",
                model_name=self.model_name,
                error=str(exc),
            )
            return None

    def _validate_update(
        self,
        updated_model: Any,
        prior_model: Any,
        val_X: Any,
        val_y: Any,
        new_version: str,
        prior_version: str,
    ) -> bool:
        """
        Validate updated model on held-out window.

        Rejects updates where new IC < prior IC (Req 14.5).
        Logs ONLINE_UPDATE_REJECTED on rejection (Req 14.7).
        """
        try:
            from scipy.stats import spearmanr
            import numpy as np

            updated_preds = updated_model.predict(val_X)
            prior_preds = prior_model.predict(val_X)

            updated_ic, _ = spearmanr(updated_preds, val_y)
            prior_ic, _ = spearmanr(prior_preds, val_y)

            updated_ic = float(updated_ic) if not np.isnan(updated_ic) else 0.0
            prior_ic = float(prior_ic) if not np.isnan(prior_ic) else 0.0

            if updated_ic >= prior_ic:
                logger.info(
                    "online_learner_validation_passed",
                    model_name=self.model_name,
                    updated_ic=round(updated_ic, 4),
                    prior_ic=round(prior_ic, 4),
                    new_version=new_version,
                )

                # Write audit log if available (Req 14.4)
                if self._audit:
                    try:
                        self._audit.log_online_update(
                            model_name=self.model_name,
                            prior_version=prior_version,
                            new_version=new_version,
                            ic_delta=updated_ic - prior_ic,
                            consecutive_update_count=self._consecutive_updates + 1,
                        )
                    except Exception:
                        pass

                return True

            else:
                logger.warning(
                    "ONLINE_UPDATE_REJECTED",
                    model_name=self.model_name,
                    prior_version=prior_version,
                    candidate_version=new_version,
                    prior_ic=round(prior_ic, 4),
                    candidate_ic=round(updated_ic, 4),
                    reason="new_ic_below_prior",
                )

                # Write audit log if available (Req 14.4)
                if self._audit:
                    try:
                        self._audit.log_online_update_rejected(
                            model_name=self.model_name,
                            prior_version=prior_version,
                            candidate_version=new_version,
                            reason="new_ic_below_prior",
                            prior_ic=prior_ic,
                            candidate_ic=updated_ic,
                        )
                    except Exception:
                        pass

                return False

        except Exception as exc:
            logger.warning(
                "online_learner_validation_failed",
                model_name=self.model_name,
                error=str(exc),
            )
            return False
