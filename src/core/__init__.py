"""
src.core — cross-cutting concerns: configuration, exceptions, and logging.

Public re-exports for convenient single-import access:

    from src.core import settings, MLServiceError, PointInTimeViolationError
"""
from src.core.config import CoreSettings, settings
from src.core.exceptions import (
    DataContractError,
    LookAheadBiasError,
    MLServiceError,
    ModelNotReadyError,
    PointInTimeViolationError,
    ProvenanceError,
    SchemaValidationError,
)

__all__ = [
    # Config
    "CoreSettings",
    "settings",
    # Exceptions
    "MLServiceError",
    "DataContractError",
    "SchemaValidationError",
    "PointInTimeViolationError",
    "LookAheadBiasError",
    "ModelNotReadyError",
    "ProvenanceError",
]
