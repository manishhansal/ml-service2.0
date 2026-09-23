"""
Explainability package for ml-service2.0.

Exports:
    ModelExplainer — SHAP-based feature attribution (TreeExplainer / KernelExplainer)
                     with LRU caching and failure-safe fallback.
"""
from __future__ import annotations

from src.explainability.explainer import ModelExplainer

__all__ = ["ModelExplainer"]
