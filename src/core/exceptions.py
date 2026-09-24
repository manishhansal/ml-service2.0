"""
src.core.exceptions — canonical exception hierarchy for ml-service2.0.

All service-specific exceptions inherit from ``MLServiceError`` so callers
can catch the entire family with a single ``except MLServiceError`` clause.

Hierarchy
---------
MLServiceError
├── DataContractError           # incoming data violates its Pydantic contract
│   └── SchemaValidationError   # Pydantic V2 validation failure (field-level detail)
├── PointInTimeViolationError   # source data timestamp >= PIT boundary
│   └── LookAheadBiasError      # feature matrix shows forward-looking correlation
├── ModelNotReadyError          # model is not in a state eligible for inference
└── ProvenanceError             # provenance downgrade violates live-capital policy

Phase 2 — these classes define the *interface contract* (name, attributes,
inheritance).  The ``raise NotImplementedError`` stubs inside ``__init__``
are intentional TDD red-phase markers: tests that instantiate these exceptions
will fail until the stubs are replaced with real implementations in Phase 3.

EXCEPTION: the ``__init__`` bodies here ARE implemented (exceptions need to
be instantiable for ``pytest.raises`` to work in the test suite).  Only
higher-level service methods are left as ``NotImplementedError`` stubs.
"""
from __future__ import annotations


# ── Root ─────────────────────────────────────────────────────────────────────


class MLServiceError(Exception):
    """Base class for all ml-service2.0 domain exceptions.

    Attributes:
        message: Human-readable error description.
        code:    Optional short error code for structured logging (e.g. ``"PIT_001"``).
    """

    def __init__(self, message: str, code: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __repr__(self) -> str:
        return f"{type(self).__name__}(message={self.message!r}, code={self.code!r})"


# ── Data Contract ─────────────────────────────────────────────────────────────


class DataContractError(MLServiceError):
    """Raised when incoming data from data-service2.0 or SentinelPulse violates its contract.

    This is the parent for all schema-related failures.  Catch this to handle
    *any* upstream data-quality issue regardless of its exact cause.

    Attributes:
        source:  Name of the upstream service that produced bad data
                 (e.g. ``"data-service2.0"``, ``"SentinelPulse"``).
        payload: Optional raw payload excerpt for debugging (truncated to 500 chars).
    """

    def __init__(
        self,
        message: str,
        source: str = "",
        payload: str = "",
        code: str = "DATA_CONTRACT_001",
    ) -> None:
        super().__init__(message, code=code)
        self.source = source
        self.payload = payload[:500] if payload else ""


class SchemaValidationError(DataContractError):
    """Pydantic V2 validation failure with field-level detail.

    Raised when a raw dict/JSON response from an upstream service fails
    Pydantic model validation, providing structured access to the field
    name and the invalid value that triggered the failure.

    Attributes:
        field:         Dotted field path that failed (e.g. ``"metadata.quality.score"``).
        invalid_value: The value that triggered the validation failure.
    """

    def __init__(
        self,
        message: str,
        field: str = "",
        invalid_value: object = None,
        source: str = "",
        code: str = "SCHEMA_001",
    ) -> None:
        super().__init__(message, source=source, code=code)
        self.field = field
        self.invalid_value = invalid_value


# ── Point-In-Time ─────────────────────────────────────────────────────────────


class PointInTimeViolationError(MLServiceError):
    """Raised when a source datum's timestamp is >= the PIT boundary.

    The PIT (Point-In-Time) contract requires that every value used to
    construct a feature vector at time T was *observable* strictly before T.
    Any data with ``source_timestamp >= pit_boundary`` constitutes a violation
    that, if undetected, would introduce look-ahead bias into backtests.

    Attributes:
        feature_name:   Identifier of the feature or data field with the violation.
        source_ts_iso:  ISO-8601 string of the offending source timestamp.
        pit_boundary_iso: ISO-8601 string of the PIT boundary (the request timestamp).
    """

    def __init__(
        self,
        feature_name: str,
        source_ts_iso: str,
        pit_boundary_iso: str,
        code: str = "PIT_001",
    ) -> None:
        message = (
            f"PIT violation on '{feature_name}': source_ts={source_ts_iso} "
            f">= pit_boundary={pit_boundary_iso}"
        )
        super().__init__(message, code=code)
        self.feature_name = feature_name
        self.source_ts_iso = source_ts_iso
        self.pit_boundary_iso = pit_boundary_iso


# ── Stale market data ───────────────────────────────────────────────────────


class StaleDataError(MLServiceError):
    """Raised when the freshest available market datum is too old to trade on.

    Distinct from a PIT violation (data from the *future*): stale data is data
    from too far in the *past*. Trading on stale quotes risks acting on a price
    that no longer reflects the market. The maximum allowed staleness is
    timeframe-specific (a 5m strategy tolerates far less staleness than a daily
    one) — see :class:`src.features.stale_guard.StaleDataGuard` (mandate §62).

    Attributes:
        timeframe:        The strategy timeframe (e.g. ``"5m"``, ``"1d"``).
        age_seconds:      Age of the freshest datum, in seconds.
        max_age_seconds:  The maximum allowed staleness for this timeframe.
    """

    def __init__(
        self,
        timeframe: str,
        age_seconds: float,
        max_age_seconds: float,
        code: str = "STALE_001",
    ) -> None:
        message = (
            f"STALE_MARKET_DATA: freshest datum for timeframe '{timeframe}' is "
            f"{age_seconds:.1f}s old (max allowed {max_age_seconds:.1f}s). NO_TRADE."
        )
        super().__init__(message, code=code)
        self.timeframe = timeframe
        self.age_seconds = age_seconds
        self.max_age_seconds = max_age_seconds


class LookAheadBiasError(PointInTimeViolationError):
    """Forward-looking correlation detected in a feature matrix.

    Specialisation of :exc:`PointInTimeViolationError` for statistical
    look-ahead bias: when the Pearson correlation between a feature and
    future returns exceeds ``LEAK_THRESHOLD`` (0.05), the
    ``LeakageValidator`` raises this exception.

    Attributes:
        look_ahead_days: The look-ahead window (in trading days) at which the
                         correlation was detected.
        correlation:     The actual Pearson correlation value (signed).
    """

    def __init__(
        self,
        feature_name: str,
        look_ahead_days: int,
        correlation: float,
        code: str = "PIT_002",
    ) -> None:
        message = (
            f"Look-ahead bias on '{feature_name}': |r|={abs(correlation):.4f} "
            f"at window={look_ahead_days} days (threshold=0.05)"
        )
        # Source TS not meaningful for statistical violations — use placeholders.
        super().__init__(
            feature_name=feature_name,
            source_ts_iso="N/A",
            pit_boundary_iso="N/A",
            code=code,
        )
        # Override the parent message
        self.args = (message,)
        self.message = message
        self.look_ahead_days = look_ahead_days
        self.correlation = correlation


# ── Model Lifecycle ───────────────────────────────────────────────────────────


class ModelNotReadyError(MLServiceError):
    """Raised when a model is requested for inference but is not in a ready state.

    This covers: model file not found, model not promoted to PRODUCTION,
    model failed health check, or model requires retraining.

    Attributes:
        model_id:       Registry identifier of the model.
        current_stage:  The model's current lifecycle stage (e.g. ``"challenger"``).
    """

    def __init__(
        self,
        model_id: str,
        current_stage: str = "",
        code: str = "MODEL_001",
    ) -> None:
        message = (
            f"Model '{model_id}' is not ready for inference "
            f"(current stage: '{current_stage or 'unknown'}')"
        )
        super().__init__(message, code=code)
        self.model_id = model_id
        self.current_stage = current_stage


# ── Provenance ────────────────────────────────────────────────────────────────


class ProvenanceError(MLServiceError):
    """Raised when a provenance downgrade would violate the live-capital policy.

    In ``VALIDATED_ML_ONLY`` deployment mode, any prediction with provenance
    other than ``TRAINED_MODEL`` must be blocked from reaching live capital.
    This exception signals that the provenance contract has been violated
    at the API boundary.

    Attributes:
        required_provenance:  The provenance required for the current deployment mode.
        actual_provenance:    The provenance present on the prediction.
    """

    def __init__(
        self,
        required_provenance: str,
        actual_provenance: str,
        code: str = "PROV_001",
    ) -> None:
        message = (
            f"Provenance policy violation: required={required_provenance!r}, "
            f"actual={actual_provenance!r}"
        )
        super().__init__(message, code=code)
        self.required_provenance = required_provenance
        self.actual_provenance = actual_provenance
