"""
Training package for ml-service2.0.

Exports:
  HyperparameterOptimizer  — Optuna-backed HPO with minimum trial enforcement.
  suggest_params_from_space — Helper to suggest params from a search space dict.
  TrainingPipeline          — End-to-end training orchestration with purged K-fold CV.
  TrainingResult            — Mutable result object populated by TrainingPipeline.
  TrainingAbortedError      — Raised on PIT violation or data integrity failure.
  MLflowTracker             — MLflow experiment tracking for training runs (Req 3.8).
  OnlineLearner             — Incremental model update engine (Req 14.1–14.7).
"""
from __future__ import annotations

from src.training.hpo import HyperparameterOptimizer, suggest_params_from_space
from src.training.mlflow_tracker import MLflowTracker
from src.training.online_learner import OnlineLearner
from src.training.pipeline import TrainingAbortedError, TrainingPipeline, TrainingResult

__all__ = [
    "HyperparameterOptimizer",
    "suggest_params_from_space",
    "TrainingPipeline",
    "TrainingResult",
    "TrainingAbortedError",
    "MLflowTracker",
    "OnlineLearner",
]
