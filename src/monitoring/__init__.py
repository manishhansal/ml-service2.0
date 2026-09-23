"""
src.monitoring — drift monitoring package.

Exports:
    DriftMonitor: PSI-based feature drift detector with Evidently AI and NannyML support.
"""
from __future__ import annotations

from src.monitoring.drift_monitor import DriftMonitor

__all__ = ["DriftMonitor"]
