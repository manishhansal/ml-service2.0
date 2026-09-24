"""
src.meta.signal — signal executability + expiry + contract validation (Phase T).

Implements signal expiry (Phase 43): a signal must NOT be executable after its
``expires_at`` timestamp. Provides an executability check with strict boundary
semantics and a contract validator for the ML -> AlphaForge signal contract
(Phase 42, Phase 93).

Requirements: Phase T, Phase 42, Phase 43, 14_SIGNAL_CONTRACT.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum

from src.logging_config import get_logger
from src.schemas.meta import MetaOutput

logger = get_logger(__name__)

# Fields the ML -> AlphaForge contract requires on every executable signal.
REQUIRED_CONTRACT_FIELDS = [
    "signal_id",
    "symbol",
    "action",
    "confidence",
    "prediction_timestamp",
    "expires_at",
    "provenance",
]


class SignalExecutability(str, Enum):
    EXECUTABLE = "EXECUTABLE"
    EXPIRED = "EXPIRED"
    NOT_YET_VALID = "NOT_YET_VALID"
    NON_DIRECTIONAL = "NON_DIRECTIONAL"
    NOT_LIVE_ELIGIBLE = "NOT_LIVE_ELIGIBLE"
    MISSING_TIMESTAMPS = "MISSING_TIMESTAMPS"


def is_executable(
    signal: MetaOutput,
    now: datetime | None = None,
    require_live_eligible: bool = False,
) -> SignalExecutability:
    """
    Determine whether *signal* may be executed at *now*.

    Rules:
      - NO_TRADE / WAIT are non-directional → NON_DIRECTIONAL (never executed).
      - prediction_timestamp and expires_at must be present → else MISSING_TIMESTAMPS.
      - now < prediction_timestamp → NOT_YET_VALID (future signal).
      - now >= expires_at → EXPIRED (strict: at exact expiry it is EXPIRED).
      - require_live_eligible and provenance not TRAINED_MODEL → NOT_LIVE_ELIGIBLE.
      - otherwise → EXECUTABLE.
    """
    now = now or datetime.now(tz=UTC)

    if signal.action in ("NO_TRADE", "WAIT"):
        return SignalExecutability.NON_DIRECTIONAL

    if signal.prediction_timestamp is None or signal.expires_at is None:
        return SignalExecutability.MISSING_TIMESTAMPS

    if now < signal.prediction_timestamp:
        return SignalExecutability.NOT_YET_VALID

    # Strict expiry boundary: at exactly expires_at the signal is expired.
    if now >= signal.expires_at:
        return SignalExecutability.EXPIRED

    if require_live_eligible and not signal.provenance.is_live_eligible:
        return SignalExecutability.NOT_LIVE_ELIGIBLE

    return SignalExecutability.EXECUTABLE


def attach_expiry(
    signal: MetaOutput,
    prediction_timestamp: datetime,
    ttl_seconds: int = 900,
    feature_as_of: datetime | None = None,
    data_as_of: datetime | None = None,
    news_as_of: datetime | None = None,
) -> MetaOutput:
    """
    Return a copy of *signal* with the PIT timestamp chain + expiry populated.

    Enforces the PIT invariant: feature_as_of/news_as_of must be <=
    prediction_timestamp (raises ValueError otherwise).
    """
    if feature_as_of is not None and feature_as_of > prediction_timestamp:
        raise ValueError("feature_as_of must be <= prediction_timestamp")
    if news_as_of is not None and news_as_of > prediction_timestamp:
        raise ValueError("news_as_of must be <= prediction_timestamp")

    return signal.model_copy(update={
        "prediction_timestamp": prediction_timestamp,
        "feature_as_of": feature_as_of,
        "data_as_of": data_as_of,
        "news_as_of": news_as_of,
        "expires_at": prediction_timestamp + timedelta(seconds=ttl_seconds),
    })


def validate_signal_contract(signal: MetaOutput) -> tuple[bool, list[str]]:
    """
    Validate that *signal* satisfies the ML -> AlphaForge contract (Phase 42).

    Returns (valid, missing_or_invalid_field_reasons).
    """
    problems: list[str] = []
    d = signal.model_dump()

    for f in REQUIRED_CONTRACT_FIELDS:
        val = d.get(f)
        if val is None or (isinstance(val, str) and val == ""):
            problems.append(f"MISSING_{f.upper()}")

    # PIT invariant.
    if signal.prediction_timestamp and signal.feature_as_of:
        if signal.feature_as_of > signal.prediction_timestamp:
            problems.append("FEATURE_AS_OF_AFTER_PREDICTION")
    if signal.prediction_timestamp and signal.news_as_of:
        if signal.news_as_of > signal.prediction_timestamp:
            problems.append("NEWS_AS_OF_AFTER_PREDICTION")
    if signal.prediction_timestamp and signal.expires_at:
        if signal.expires_at <= signal.prediction_timestamp:
            problems.append("EXPIRES_AT_NOT_AFTER_PREDICTION")

    # Directional signals must carry an expected edge for auditability.
    if signal.action in ("BUY", "SELL") and signal.expected_net_edge is None:
        problems.append("MISSING_EXPECTED_NET_EDGE")

    return (len(problems) == 0, problems)
