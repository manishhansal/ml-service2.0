"""
src/meta — Meta Decision Engine components for ml-service2.0.

Public exports
--------------
MetaDecisionEngine   Final Go/No-Go trading decision engine (Req 10.1–10.12).
AbstentionPolicy     Five-condition Go/No-Go guard for the meta engine.
AbstentionContext    Input dataclass for AbstentionPolicy.
AbstentionResult     Output dataclass from AbstentionPolicy.
CalibrationLayer     Per-model Platt/isotonic calibration with ECE-based selection.
ConfidenceDecomposer Five-component confidence decomposition for MetaOutput.
LLMNewsReasoner      FinBERT/FinGPT news sentiment reasoning for the meta engine.
"""
from src.meta.abstention import AbstentionContext, AbstentionPolicy, AbstentionResult
from src.meta.calibration import CalibrationLayer, ConfidenceDecomposer
from src.meta.engine import MetaDecisionEngine
from src.meta.llm_reasoner import LLMNewsReasoner

__all__ = [
    "MetaDecisionEngine",
    "AbstentionPolicy",
    "AbstentionContext",
    "AbstentionResult",
    "CalibrationLayer",
    "ConfidenceDecomposer",
    "LLMNewsReasoner",
]
