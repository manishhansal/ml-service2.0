"""
src.labels.schemas — Label family enum and shared data structures.

RC-005 / Task 5 fix: this module was missing, causing a ModuleNotFoundError
when ``generate_labels()`` in data_pipeline.py was called with any label_id.
"""
from __future__ import annotations

from enum import Enum


class LabelFamily(str, Enum):
    """Broad category of label design.  Used by the registry to route
    ``generate_labels()`` to the correct generator function.

    Values
    ------
    TRIPLE_BARRIER
        Classic Marcos López de Prado triple-barrier labels (binary:
        TARGET_HIT=1, STOP_HIT=0).
    EXCESS_RETURN
        Continuous vol-adjusted excess return vs benchmark (e.g. NIFTY).
        Good for regression / IC optimisation.
    FIXED_RETURN
        Simple fixed-horizon forward return (binary or continuous).
    SEVEN_DAY_BARRIER
        Asymmetric 7-trading-day triple barrier with 2:1 R:R
        (``src.labels.seven_day.generate_7d_asymmetric_barrier_label``).
    SEVEN_DAY_EXCESS
        Continuous 7-trading-day vol-adjusted excess return over NIFTY
        (``src.labels.seven_day.generate_7d_excess_return_label``).
        This is the label used to train the v2c model.
    CS_RANK
        Alias for SEVEN_DAY_EXCESS — kept for backward compatibility with
        internal documentation that calls the v2c target a "CS rank label".
    """

    TRIPLE_BARRIER    = "TRIPLE_BARRIER"
    EXCESS_RETURN     = "EXCESS_RETURN"
    FIXED_RETURN      = "FIXED_RETURN"
    SEVEN_DAY_BARRIER = "SEVEN_DAY_BARRIER"
    SEVEN_DAY_EXCESS  = "SEVEN_DAY_EXCESS"
    CS_RANK           = "CS_RANK"          # alias → SEVEN_DAY_EXCESS
