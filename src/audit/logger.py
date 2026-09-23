"""
Append-only, immutable audit logger for ml-service2.0.

Writes structured JSON Lines to ``Settings.audit_log_path`` (default: ``./audit.jsonl``).

Every log entry contains:
- ``entry_id``    — UUID4 string
- ``written_at``  — UTC ISO-8601 timestamp
- ``event_type``  — one of: training_run | promotion_decision | online_update |
                    online_update_rejected
- ``service``     — "ml-service2.0"
- ``version``     — "2.0.0"
- all domain-specific fields passed by the caller
- ``entry_hash``  — SHA-256 hex digest of the JSON string above (tamper-detection)

Design guarantees:
- Each entry is written atomically with ``os.fsync`` to prevent partial writes.
- ``_check_append_only()`` recomputes the hash for every line at startup and
  raises ``AuditLogViolation`` on the first mismatch.
- Modification or deletion of any prior entry raises ``AuditLogViolation``.
- ``IOError`` on write raises ``AuditLogWriteFailure`` and logs to stderr.
- Thread-safe via an internal ``threading.Lock``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


# ── Custom exceptions ─────────────────────────────────────────────────────────


class AuditLogViolation(Exception):
    """Raised when an attempt is made to modify or delete an existing audit log entry."""


class AuditLogWriteFailure(Exception):
    """Raised when an audit log entry cannot be written to disk."""


# ── AuditLogger ───────────────────────────────────────────────────────────────


class AuditLogger:
    """
    Append-only, immutable audit logger using JSON Lines format.

    Usage::

        audit = AuditLogger()
        audit.log_training_run(run_id="abc-123", model_name="market_regime", ...)
        audit._check_append_only()   # called at FastAPI lifespan startup
    """

    # Service metadata embedded in every entry
    _SERVICE = "ml-service2.0"
    _VERSION = "2.0.0"

    def __init__(self, log_path: Path | None = None) -> None:
        self._path = log_path or settings.audit_log_path
        self._lock = Lock()
        # Ensure the parent directory exists before the first write
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Integrity check ───────────────────────────────────────────────────────

    def _check_append_only(self) -> None:
        """
        Verify no existing entry has been tampered with.

        Algorithm:
        1. Open the log file for reading (skip if file does not exist yet).
        2. For each non-empty line, parse JSON.
        3. Pop the ``entry_hash`` field.
        4. Re-serialize the remaining dict (same ``sort_keys=True`` / no whitespace
           convention used in ``_write_entry``).
        5. Recompute SHA-256 of the re-serialized string.
        6. Compare against the popped ``entry_hash``; raise ``AuditLogViolation``
           on the first mismatch.

        Called automatically at FastAPI lifespan startup.
        Raises:
            AuditLogViolation: when any entry's hash does not match its content.
        """
        if not self._path.exists():
            return  # no log yet — nothing to check

        with self._path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue  # skip blank lines

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AuditLogViolation(
                        f"Audit log line {line_number} is not valid JSON: {exc}"
                    ) from exc

                stored_hash = entry.pop("entry_hash", None)
                if stored_hash is None:
                    raise AuditLogViolation(
                        f"Audit log line {line_number} is missing 'entry_hash' — "
                        "entry may have been tampered with."
                    )

                # Re-serialize with the same convention used during write
                recomputed_body = json.dumps(entry, sort_keys=True, separators=(",", ":"))
                recomputed_hash = hashlib.sha256(recomputed_body.encode("utf-8")).hexdigest()

                if recomputed_hash != stored_hash:
                    model_name = entry.get("model_name", "<unknown>")
                    raise AuditLogViolation(
                        f"Audit log entry at line {line_number} (model={model_name!r}) "
                        f"has been tampered with. "
                        f"Expected hash {recomputed_hash!r}, stored {stored_hash!r}."
                    )

        logger.info(
            "audit_log_integrity_verified",
            path=str(self._path),
        )

    # ── Internal write primitive ──────────────────────────────────────────────

    def _write_entry(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Write a single audit entry as a JSON Line.

        Entry structure::

            {
              "entry_id":   "<uuid4>",
              "written_at": "<utc-iso8601>",
              "event_type": "<event_type>",
              "service":    "ml-service2.0",
              "version":    "2.0.0",
              ... <all keys from *data*>,
              "entry_hash": "<sha256 of the above JSON>"
            }

        The hash is computed *after* all other fields are assembled so that it
        covers the complete entry content.

        Args:
            event_type: Semantic event name (e.g. ``"training_run"``).
            data: Domain-specific fields to include in the entry.

        Raises:
            AuditLogWriteFailure: if an ``IOError`` occurs while writing.
        """
        # Build the base entry (everything except the hash)
        entry: dict[str, Any] = {
            "entry_id": str(uuid4()),
            "written_at": datetime.now(tz=timezone.utc).isoformat(),
            "event_type": event_type,
            "service": self._SERVICE,
            "version": self._VERSION,
        }
        entry.update(data)

        # Serialize without hash to produce the hash input
        body = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        entry_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

        # Re-insert the hash and produce the final line
        entry["entry_hash"] = entry_hash
        final_line = json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"

        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(final_line)
                    fh.flush()
                    os.fsync(fh.fileno())
            except IOError as exc:
                msg = (
                    f"[ml-service2.0] CRITICAL: audit log write failure "
                    f"(event_type={event_type!r}, path={self._path!r}): {exc}\n"
                )
                sys.stderr.write(msg)
                sys.stderr.flush()
                logger.critical(
                    "audit_log_write_failure",
                    event_type=event_type,
                    path=str(self._path),
                    error=str(exc),
                )
                raise AuditLogWriteFailure(str(exc)) from exc

    # ── Public logging methods ─────────────────────────────────────────────────

    def log_training_run(
        self,
        run_id: str,
        model_name: str,
        model_version: str,
        started_at: str,
        completed_at: str,
        dataset_hash: str,
        training_date_range: tuple[str, str],
        validation_date_range: tuple[str, str],
        hyperparameters: dict[str, Any],
        ic_per_fold: list[float],
        ic_mean: float,
        sharpe_net: float,
        max_drawdown: float,
        pbo: float,
        gate_results: dict[str, str],
        outcome: str,
        approval_token: str | None = None,
        reviewer_identity: str | None = None,
    ) -> None:
        """Log a completed training run to the immutable audit log.

        Args:
            run_id: Unique identifier for this training run.
            model_name: Name of the model family being trained.
            model_version: Version string of the trained artifact.
            started_at: ISO-8601 UTC timestamp when training started.
            completed_at: ISO-8601 UTC timestamp when training completed.
            dataset_hash: SHA-256 hash of the training dataset.
            training_date_range: ``(start_date, end_date)`` ISO-8601 strings.
            validation_date_range: ``(start_date, end_date)`` ISO-8601 strings.
            hyperparameters: HPO-selected hyperparameter dict.
            ic_per_fold: Per-fold Spearman IC values.
            ic_mean: Mean IC across all folds.
            sharpe_net: Net Sharpe after 10 bp transaction costs.
            max_drawdown: Maximum drawdown fraction.
            pbo: Probability of backtest overfitting (CPCV).
            gate_results: Mapping of gate name → ``GateResult`` string value.
            outcome: ``PromotionOutcome`` string value.
            approval_token: Human-approval token when outcome is PROMOTE.
            reviewer_identity: Reviewer username/ID when human-approved.
        """
        data: dict[str, Any] = {
            "run_id": run_id,
            "model_name": model_name,
            "model_version": model_version,
            "started_at": started_at,
            "completed_at": completed_at,
            "dataset_hash": dataset_hash,
            "training_date_range": list(training_date_range),
            "validation_date_range": list(validation_date_range),
            "hyperparameters": hyperparameters,
            "ic_per_fold": ic_per_fold,
            "ic_mean": ic_mean,
            "sharpe_net": sharpe_net,
            "max_drawdown": max_drawdown,
            "pbo": pbo,
            "gate_results": gate_results,
            "outcome": outcome,
        }
        if approval_token is not None:
            data["approval_token"] = approval_token
        if reviewer_identity is not None:
            data["reviewer_identity"] = reviewer_identity

        self._write_entry("training_run", data)
        logger.info(
            "audit_training_run_logged",
            run_id=run_id,
            model_name=model_name,
            model_version=model_version,
            outcome=outcome,
        )

    def log_promotion_decision(
        self,
        challenger_id: str,
        champion_id: str | None,
        outcome: str,
        gate_results: dict[str, str],
        approval_policy: str,
        reviewer_identity: str | None = None,
        blocked_gates: list[str] | None = None,
    ) -> None:
        """Log a promotion gate decision (PROMOTE, REJECTED, BLOCKED).

        Args:
            challenger_id: Version string of the challenger artifact.
            champion_id: Version string of the current champion, or ``None``
                if no champion exists yet.
            outcome: ``PromotionOutcome`` string value.
            gate_results: Mapping of gate name → ``GateResult`` string value
                for all six gates.
            approval_policy: ``"HUMAN_APPROVAL_REQUIRED"`` or ``"AUTOMATIC"``.
            reviewer_identity: Reviewer username/ID when human-approved.
            blocked_gates: List of gate names that returned INSUFFICIENT_EVIDENCE.
        """
        data: dict[str, Any] = {
            "challenger_id": challenger_id,
            "champion_id": champion_id,
            "outcome": outcome,
            "gate_results": gate_results,
            "approval_policy": approval_policy,
            "blocked_gates": blocked_gates or [],
        }
        if reviewer_identity is not None:
            data["reviewer_identity"] = reviewer_identity

        self._write_entry("promotion_decision", data)
        logger.info(
            "audit_promotion_decision_logged",
            challenger_id=challenger_id,
            champion_id=champion_id,
            outcome=outcome,
        )

    def log_online_update(
        self,
        model_name: str,
        prior_version: str,
        new_version: str,
        ic_delta: float,
        consecutive_update_count: int,
    ) -> None:
        """Log a successful online learning update.

        Args:
            model_name: Name of the model family updated.
            prior_version: Version string of the artifact before the update.
            new_version: Version string of the newly created artifact.
            ic_delta: Change in IC (``new_ic - prior_ic``) from the validation window.
            consecutive_update_count: Number of consecutive online updates applied
                to this artifact (including this one).
        """
        data: dict[str, Any] = {
            "model_name": model_name,
            "prior_version": prior_version,
            "new_version": new_version,
            "ic_delta": ic_delta,
            "consecutive_update_count": consecutive_update_count,
        }
        self._write_entry("online_update", data)
        logger.info(
            "audit_online_update_logged",
            model_name=model_name,
            prior_version=prior_version,
            new_version=new_version,
            ic_delta=ic_delta,
        )

    def log_online_update_rejected(
        self,
        model_name: str,
        prior_version: str,
        candidate_version: str,
        reason: str,
        prior_ic: float,
        candidate_ic: float,
    ) -> None:
        """Log a rejected online learning update (new IC < prior IC).

        Args:
            model_name: Name of the model family.
            prior_version: Version string of the current champion artifact.
            candidate_version: Version string of the candidate artifact that
                was evaluated and discarded.
            reason: Human-readable rejection reason (e.g. ``"new_ic_below_prior"``).
            prior_ic: IC of the current champion on the validation window.
            candidate_ic: IC of the candidate on the same validation window.
        """
        data: dict[str, Any] = {
            "model_name": model_name,
            "prior_version": prior_version,
            "candidate_version": candidate_version,
            "reason": reason,
            "prior_ic": prior_ic,
            "candidate_ic": candidate_ic,
        }
        self._write_entry("online_update_rejected", data)
        logger.warning(
            "audit_online_update_rejected_logged",
            model_name=model_name,
            prior_version=prior_version,
            candidate_version=candidate_version,
            reason=reason,
            prior_ic=prior_ic,
            candidate_ic=candidate_ic,
        )
