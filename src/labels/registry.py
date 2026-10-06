"""
src.labels.registry — Label registration system.

Maps label_id strings to their family, config, and deprecation metadata.
Used by ``data_pipeline.generate_labels()`` to route label generation requests
without hard-coding label types in the pipeline.

RC-005 / Task 5 fix: this module was missing, causing a ModuleNotFoundError
when ``generate_labels()`` was called.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.labels.schemas import LabelFamily


@dataclass
class LabelRegistration:
    """Metadata for a registered label type."""

    label_id:         str
    label_family:     LabelFamily
    config:           dict[str, Any] = field(default_factory=dict)
    deprecated:       bool           = False
    deprecation_note: str            = ""


# ── Built-in registry ─────────────────────────────────────────────────────────

_REGISTRY: dict[str, LabelRegistration] = {

    # ── Triple-barrier labels ────────────────────────────────────────────────
    "TRIPLE_BARRIER_V2_DAILY": LabelRegistration(
        label_id="TRIPLE_BARRIER_V2_DAILY",
        label_family=LabelFamily.TRIPLE_BARRIER,
        config={
            "horizon":       5,
            "upper_barrier": 0.02,
            "lower_barrier": 0.02,
            "cost_bps":      27.65,
            "execution_model": "next_open",
        },
    ),
    "TRIPLE_BARRIER_FUTURES": LabelRegistration(
        label_id="TRIPLE_BARRIER_FUTURES",
        label_family=LabelFamily.TRIPLE_BARRIER,
        config={
            "horizon":       5,
            "upper_barrier": 0.02,
            "lower_barrier": 0.02,
            "cost_bps":      8.5,
            "execution_model": "next_open",
        },
    ),

    # ── Excess-return labels ─────────────────────────────────────────────────
    "EXCESS_RETURN_7D": LabelRegistration(
        label_id="EXCESS_RETURN_7D",
        label_family=LabelFamily.EXCESS_RETURN,
        config={
            "horizon":     7,
            "vol_window":  20,
            "clip":        5.0,
            "cost_bps":    0.0,   # continuous label — cost deducted at eval
        },
    ),
    "EXCESS_RETURN_5D": LabelRegistration(
        label_id="EXCESS_RETURN_5D",
        label_family=LabelFamily.EXCESS_RETURN,
        config={"horizon": 5, "vol_window": 20, "clip": 5.0, "cost_bps": 0.0},
    ),

    # ── Fixed-horizon return ─────────────────────────────────────────────────
    "FIXED_RETURN_5D": LabelRegistration(
        label_id="FIXED_RETURN_5D",
        label_family=LabelFamily.FIXED_RETURN,
        config={"horizon": 5, "cost_bps": 27.65},
    ),

    # ── 7-day asymmetric barrier (used by v2c training, Task 5) ─────────────
    "SEVEN_DAY_BARRIER": LabelRegistration(
        label_id="SEVEN_DAY_BARRIER",
        label_family=LabelFamily.SEVEN_DAY_BARRIER,
        config={
            "horizon":       7,
            "upper_barrier": 0.06,   # 6% target (2:1 R:R)
            "lower_barrier": 0.03,   # 3% stop
            "cost_bps":      8.5,    # futures
            "execution_model": "next_open",
        },
    ),

    # ── 7-day continuous excess return (v2c primary label) ───────────────────
    "SEVEN_DAY_EXCESS": LabelRegistration(
        label_id="SEVEN_DAY_EXCESS",
        label_family=LabelFamily.SEVEN_DAY_EXCESS,
        config={"horizon": 7, "vol_window": 20, "clip": 5.0},
    ),
    # Alias used in v2c documentation
    "CS_RANK_7D": LabelRegistration(
        label_id="CS_RANK_7D",
        label_family=LabelFamily.CS_RANK,
        config={"horizon": 7, "vol_window": 20, "clip": 5.0},
    ),
}


def get_label_registration(label_id: str) -> LabelRegistration:
    """Return the registration for *label_id*.

    Raises
    ------
    KeyError
        If *label_id* is not in the registry.  Callers should catch this and
        present a helpful error message listing available IDs.
    """
    if label_id not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"Unknown label_id '{label_id}'. "
            f"Available: {available}"
        )
    return _REGISTRY[label_id]


def register_label(reg: LabelRegistration) -> None:
    """Add or overwrite a registration at runtime (e.g. from experiment configs)."""
    _REGISTRY[reg.label_id] = reg


def list_label_ids() -> list[str]:
    """Return all registered label IDs sorted alphabetically."""
    return sorted(_REGISTRY)
