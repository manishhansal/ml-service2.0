"""
src.explainability.decision_trace — persistable decision trace + reconstruction (Phase S).

Every signal must be reconstructable AFTER THE FACT from persisted evidence
(Phase 77, Phase 105). A DecisionTrace captures the complete provenance chain:

    input snapshot (symbol, timestamps, data confidence)
    feature snapshot + hash
    models used + raw outputs
    calibrated outputs
    ensemble result
    regime
    expected value
    risk (P(stop)/P(target))
    execution assumptions
    final decision + reason codes
    abstention reason

Traces are written to an append-only JSONL store keyed by signal_id and can be
reconstructed into a human-readable explanation answering: why signal? what
data? what features? what models? what probabilities? what EV? what risk? why
trade / why abstain?

Requirements: Phase S, Phase 77, Phase 105, 16_DECISION_ENGINE_SPEC.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)

# Actions that would deploy (paper or live) capital and therefore require a
# complete, reconstructable evidence chain before they may be persisted.
TRADEABLE_ACTIONS: frozenset[str] = frozenset({"BUY", "SELL"})

# Probability floor below which a "target" / "stop" probability is treated as
# degenerate (i.e. the RiskPredictor never actually produced it). A trained
# model that emits a directional trade must supply meaningful barrier probs.
_PROB_EPS = 1e-9


def _parse_iso(value: Any) -> datetime | None:
    """Parse an ISO-8601 string (or accept a datetime) into an aware datetime.

    Returns None on failure. Naive datetimes are assumed UTC so PIT ordering
    comparisons never raise on tz-mismatch.
    """
    if value is None:
        return None
    dt: datetime | None = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


@dataclass
class DecisionTrace:
    """Complete, reconstructable record of a single trading decision."""

    signal_id: str
    symbol: str
    prediction_timestamp: str
    # PIT provenance
    feature_as_of: str | None = None
    data_as_of: str | None = None
    news_as_of: str | None = None
    expires_at: str | None = None
    # Data quality
    data_confidence_score: int = 0
    # Feature snapshot
    feature_snapshot: dict[str, float] = field(default_factory=dict)
    feature_hash: str = ""
    feature_schema_version: str = ""
    # Model layer
    models_used: list[str] = field(default_factory=list)
    model_versions: dict[str, str] = field(default_factory=dict)
    raw_model_outputs: dict[str, float] = field(default_factory=dict)
    calibrated_outputs: dict[str, float] = field(default_factory=dict)
    # Ensemble / regime
    ensemble_score: float = 0.0
    agreement_ratio: float = 0.0
    regime: str = ""
    # Expected value / risk
    expected_net_edge: float | None = None
    prob_target_hit: float = 0.0
    prob_stop_hit: float = 0.0
    suggested_position_size_pct: float = 0.0
    # Execution assumptions
    execution_assumptions: dict[str, Any] = field(default_factory=dict)
    # Final decision
    action: str = "NO_TRADE"
    confidence: float = 0.0
    provenance: str = "unavailable"
    reason_codes: list[str] = field(default_factory=list)
    abstention: bool = False
    abstention_reason: str = ""
    # Reproducibility
    dataset_version: str = ""
    calibration_version: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def compute_feature_hash(features: dict[str, float]) -> str:
        serialized = json.dumps(features, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def explain(self) -> str:
        """Produce a human-readable reconstruction of why this decision was made."""
        lines = [
            f"Signal {self.signal_id} for {self.symbol} @ {self.prediction_timestamp}",
            f"  Decision: {self.action} (confidence={self.confidence:.2f}, provenance={self.provenance})",
            f"  Data confidence: {self.data_confidence_score}/100; feature_as_of={self.feature_as_of}",
            f"  Regime: {self.regime}; agreement_ratio={self.agreement_ratio:.2f}",
            f"  Models: {', '.join(self.models_used) or 'none'}",
        ]
        if self.calibrated_outputs:
            probs = ", ".join(f"{k}={v:.3f}" for k, v in self.calibrated_outputs.items())
            lines.append(f"  Calibrated outputs: {probs}")
        if self.expected_net_edge is not None:
            lines.append(
                f"  Expected net edge: {self.expected_net_edge:.4f} "
                f"(P_target={self.prob_target_hit:.2f}, P_stop={self.prob_stop_hit:.2f})"
            )
        if self.action == "NO_TRADE":
            lines.append(f"  Abstained: {self.abstention_reason or ', '.join(self.reason_codes)}")
        else:
            lines.append(f"  Position size: {self.suggested_position_size_pct:.2f}% of capital")
        lines.append(f"  Reason codes: {', '.join(self.reason_codes) or 'none'}")
        lines.append(f"  Feature hash: {self.feature_hash[:16]}...")
        return "\n".join(lines)

    # ── Evidence-chain integrity (mandate §7, §8, §9, §10) ──────────────────
    def integrity_violations(self) -> list[str]:
        """Return structured violation codes for a *tradeable* decision.

        A tradeable action (BUY/SELL) must carry a complete, reconstructable
        evidence chain. Missing point-in-time timestamps, degenerate barrier
        probabilities, an absent expected-net-edge, or an empty reason-code
        list all indicate the decision was NOT genuinely produced by the
        decision engine and MUST NOT be trusted (see mandate §7–§10).

        NO_TRADE / WAIT decisions are non-directional and are exempt from the
        probability/edge checks, but a persisted decision still needs a
        prediction_timestamp and at least one reason code to be auditable.
        """
        violations: list[str] = []
        tradeable = self.action in TRADEABLE_ACTIONS

        # Every persisted decision must be timestamped and give a reason.
        if not self.prediction_timestamp:
            violations.append("MISSING_PREDICTION_TIMESTAMP")
        if not self.reason_codes:
            violations.append("MISSING_REASON_CODES")

        if not tradeable:
            return violations

        # §7 — mandatory PIT chain for a directional (capital-deploying) signal.
        if not self.feature_as_of:
            violations.append("MISSING_FEATURE_AS_OF")
        if not self.data_as_of:
            violations.append("MISSING_DATA_AS_OF")

        # §7 — PIT ordering: nothing may be dated after the prediction.
        for label, ts in (
            ("FEATURE", self.feature_as_of),
            ("DATA", self.data_as_of),
            ("NEWS", self.news_as_of),
        ):
            if ts and self.prediction_timestamp and _parse_iso(ts) is not None \
                    and _parse_iso(self.prediction_timestamp) is not None \
                    and _parse_iso(ts) > _parse_iso(self.prediction_timestamp):
                violations.append(f"PIT_ORDER_VIOLATION_{label}_AFTER_PREDICTION")

        # §6 — a directional signal requires known data confidence (>0).
        if self.data_confidence_score <= 0:
            violations.append("MISSING_DATA_CONFIDENCE")

        # §9 — barrier probabilities must be real and self-consistent.
        pt, ps = self.prob_target_hit, self.prob_stop_hit
        if pt <= _PROB_EPS and ps <= _PROB_EPS:
            violations.append("DEGENERATE_BARRIER_PROBABILITIES")
        if not (0.0 <= pt <= 1.0) or not (0.0 <= ps <= 1.0):
            violations.append("BARRIER_PROBABILITY_OUT_OF_RANGE")
        if pt + ps > 1.0 + 1e-6:
            violations.append("BARRIER_PROBABILITY_SUM_EXCEEDS_ONE")

        # §10 — a directional trade must carry a POSITIVE expected net edge.
        # E[net] <= 0 ⇒ NO_TRADE (unless a documented, independently-validated
        # alternative policy applies — none does here).
        if self.expected_net_edge is None:
            violations.append("MISSING_EXPECTED_NET_EDGE")
        elif self.expected_net_edge <= 0.0:
            violations.append("NON_POSITIVE_EXPECTED_NET_EDGE")

        return violations

    def is_evidence_complete(self) -> bool:
        return not self.integrity_violations()

    def enforce_integrity(self) -> DecisionTrace:
        """Return a trace safe to persist.

        If a tradeable action fails the evidence-chain checks, it is
        DOWNGRADED to NO_TRADE with an ``EVIDENCE_CHAIN_INCOMPLETE`` reason
        code plus the specific violation codes — never silently trusted
        (mandate §7–§10: missing mandatory evidence ⇒ NO_TRADE). Non-violating
        traces are returned unchanged.
        """
        violations = self.integrity_violations()
        if not violations:
            return self
        if self.action in TRADEABLE_ACTIONS:
            downgraded_reasons = ["EVIDENCE_CHAIN_INCOMPLETE", *violations]
            return replace(
                self,
                action="NO_TRADE",
                abstention=True,
                abstention_reason="EVIDENCE_CHAIN_INCOMPLETE",
                suggested_position_size_pct=0.0,
                reason_codes=[*self.reason_codes, *downgraded_reasons]
                if self.reason_codes else downgraded_reasons,
            )
        # Non-tradeable but still missing a reason/timestamp: annotate, keep action.
        merged = list(self.reason_codes)
        for v in violations:
            if v not in merged:
                merged.append(v)
        return replace(self, reason_codes=merged)


class DecisionTraceStore:
    """Append-only store for decision traces; supports exact reconstruction."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, trace: DecisionTrace) -> DecisionTrace:
        """Persist a decision trace, enforcing evidence-chain integrity.

        A tradeable (BUY/SELL) trace with an incomplete evidence chain is
        downgraded to NO_TRADE before persistence (mandate §7–§10) so the
        append-only store can never contain a capital-deploying decision that
        cannot be fully reconstructed. Returns the (possibly downgraded) trace
        that was actually written.
        """
        original_action = trace.action
        trace = trace.enforce_integrity()
        if trace.action != original_action:
            logger.warning(
                "decision_trace_downgraded",
                signal_id=trace.signal_id,
                original_action=original_action,
                enforced_action=trace.action,
                reason="EVIDENCE_CHAIN_INCOMPLETE",
                violations=trace.reason_codes,
            )
        if not trace.feature_hash and trace.feature_snapshot:
            trace.feature_hash = DecisionTrace.compute_feature_hash(trace.feature_snapshot)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace.to_dict()) + "\n")
        logger.info("decision_trace_recorded", signal_id=trace.signal_id, action=trace.action)
        return trace

    def get(self, signal_id: str) -> DecisionTrace | None:
        if not self._path.exists():
            return None
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("signal_id") == signal_id:
                    return DecisionTrace(**data)
        return None

    def reconstruct(self, signal_id: str) -> str:
        """Reconstruct a human-readable explanation from persisted evidence."""
        trace = self.get(signal_id)
        if trace is None:
            return f"No decision trace found for signal_id={signal_id}"
        return trace.explain()

    def all_traces(self) -> list[DecisionTrace]:
        if not self._path.exists():
            return []
        out: list[DecisionTrace] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(DecisionTrace(**json.loads(line)))
        return out


def trace_from_meta_output(
    meta_output: Any,
    feature_snapshot: dict[str, float],
    regime: str = "",
    raw_outputs: dict[str, float] | None = None,
    calibrated_outputs: dict[str, float] | None = None,
    execution_assumptions: dict[str, Any] | None = None,
) -> DecisionTrace:
    """Build a DecisionTrace from a MetaOutput + supporting context."""
    def _iso(dt: Any) -> str | None:
        return dt.isoformat() if hasattr(dt, "isoformat") else (dt if isinstance(dt, str) else None)

    abstention_reason = ""
    if meta_output.abstention and meta_output.reason_codes:
        abstention_reason = meta_output.reason_codes[0]

    return DecisionTrace(
        signal_id=meta_output.signal_id,
        symbol=meta_output.symbol,
        prediction_timestamp=_iso(meta_output.prediction_timestamp) or "",
        feature_as_of=_iso(meta_output.feature_as_of),
        data_as_of=_iso(meta_output.data_as_of),
        news_as_of=_iso(meta_output.news_as_of),
        expires_at=_iso(meta_output.expires_at),
        data_confidence_score=meta_output.data_confidence_score,
        feature_snapshot=feature_snapshot,
        feature_hash=DecisionTrace.compute_feature_hash(feature_snapshot),
        feature_schema_version=meta_output.feature_schema_version,
        models_used=list(meta_output.contributing_models),
        model_versions=dict(meta_output.model_versions),
        raw_model_outputs=raw_outputs or {},
        calibrated_outputs=calibrated_outputs or {},
        ensemble_score=meta_output.ensemble_score,
        agreement_ratio=meta_output.agreement_ratio,
        regime=regime,
        expected_net_edge=meta_output.expected_net_edge,
        prob_target_hit=meta_output.prob_target_hit,
        prob_stop_hit=meta_output.prob_stop_hit,
        suggested_position_size_pct=meta_output.suggested_position_size_pct,
        execution_assumptions=execution_assumptions or {},
        action=meta_output.action,
        confidence=meta_output.confidence,
        provenance=meta_output.provenance.value if hasattr(meta_output.provenance, "value") else str(meta_output.provenance),
        reason_codes=list(meta_output.reason_codes),
        abstention=meta_output.abstention,
        abstention_reason=abstention_reason,
        dataset_version=meta_output.dataset_version,
        calibration_version=meta_output.calibration_version,
    )
