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
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)


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


class DecisionTraceStore:
    """Append-only store for decision traces; supports exact reconstruction."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, trace: DecisionTrace) -> None:
        if not trace.feature_hash and trace.feature_snapshot:
            trace.feature_hash = DecisionTrace.compute_feature_hash(trace.feature_snapshot)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace.to_dict()) + "\n")
        logger.info("decision_trace_recorded", signal_id=trace.signal_id, action=trace.action)

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
