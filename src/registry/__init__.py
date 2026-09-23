"""
Registry package for ml-service2.0.

Exports the ModelRegistry, ModelPromotion, ApprovalTokenManager,
ApprovalTokenValidator, and their exception classes for use across the service.
"""
from __future__ import annotations

from src.registry.approval_token import (
    ApprovalTokenError,
    ApprovalTokenManager,
    ApprovalTokenValidator,
    InvalidApprovalTokenError,
)
from src.registry.promotion import ModelPromotion
from src.registry.registry import (
    ArtifactAlreadyExistsError,
    ArtifactIntegrityFailure,
    ModelRegistry,
)

__all__ = [
    "ModelRegistry",
    "ArtifactIntegrityFailure",
    "ArtifactAlreadyExistsError",
    "ModelPromotion",
    "ApprovalTokenManager",
    "ApprovalTokenValidator",
    "ApprovalTokenError",
    "InvalidApprovalTokenError",
]
