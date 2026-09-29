"""src.labels — Label V2 factory (relative, triple-barrier, multi-horizon)."""
from __future__ import annotations

from src.labels.multi_horizon import (  # noqa: F401
    MultiHorizonDataset,
    MultiHorizonLabelFactory,
)
from src.labels.relative import generate_ranking_labels_v2_compat  # noqa: F401
from src.labels.triple_barrier import generate_risk_labels_v2  # noqa: F401

__all__ = [
    "MultiHorizonDataset",
    "MultiHorizonLabelFactory",
    "generate_ranking_labels_v2_compat",
    "generate_risk_labels_v2",
]
