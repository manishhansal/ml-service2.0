"""
src.data.contracts — Pydantic V2 wire-format contracts for upstream data sources.

These models define the EXACT JSON shapes that ml-service2.0 expects to receive
from its two upstream services:

  1. data-service2.0  — market data (OHLCV, live quotes, quality metadata)
  2. SentinelPulse    — NLP news intelligence (sentiment, impact, regime)

Design principles
-----------------
- ``strict=True`` on all models prevents silent type coercion (``"1"`` → ``1``).
- ``frozen=True`` makes instances immutable — safe for caching and hashing.
- All ``datetime`` fields must arrive as UTC-aware ISO-8601 strings; the
  ``AwareDatetime`` type annotation enforces this at parse time.
- Optional upstream fields use ``None`` defaults so partial responses do not
  crash the pipeline — missing optional data degrades gracefully.
- Required fields have NO default: a missing required field raises
  ``ValidationError`` immediately, which ``DataServiceClient`` converts to
  ``DataContractError``.

Phase 2 TDD mandate
-------------------
These models are the *interface contract*.  The test suite in
``tests/test_data_contracts.py`` drives their exact shape via Hypothesis
property-based tests.  Any change to a field name, type, or validator here
must have a corresponding failing test first.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.functional_validators import AfterValidator

from src.schemas.base import ImpactDirection


# ── Wire-format base ──────────────────────────────────────────────────────────
# Data-contract models use frozen=True (immutable) but NOT strict=True.
# strict=True would prevent JSON string → datetime coercion, which is the
# normal behaviour when parsing HTTP response payloads via model_validate().
# The BaseSchema in src/schemas/base.py uses strict=True for internal
# domain objects where coercion is undesirable; wire-format models need
# standard Pydantic V2 coercion semantics.


class _WireSchema(BaseModel):
    """Base for all wire-format contract models.

    ``frozen=True``  — instances are immutable (safe for caching, hashing).
    ``strict=False`` — string-to-datetime coercion is allowed (JSON parsing).
    ``populate_by_name=True`` — accept both alias and field name.
    """

    model_config = ConfigDict(frozen=True, strict=False, populate_by_name=True)


# Re-export BaseSchema name so the module is consistent
BaseSchema = _WireSchema  # type: ignore[misc,assignment]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _require_utc(dt: datetime) -> datetime:
    """Validator: reject naive datetimes — all timestamps must be UTC-aware."""
    if dt.tzinfo is None:
        raise ValueError(
            f"Timestamp must be UTC-aware (got naive datetime: {dt.isoformat()})"
        )
    return dt


# Annotated type that rejects naive datetimes.
AwareDatetime = Annotated[datetime, AfterValidator(_require_utc)]


# ── data-service2.0 contracts ─────────────────────────────────────────────────


class OHLCVBar(_WireSchema):
    """A single OHLCV bar returned by ``GET /v1/india/history/{symbol}``.

    All price fields are in INR.  ``volume`` is in shares (not crores).
    ``timestamp`` must be UTC-aware.
    """

    timestamp: AwareDatetime
    open: float = Field(gt=0.0, description="Opening price in INR.")
    high: float = Field(gt=0.0, description="Session high in INR.")
    low: float = Field(gt=0.0, description="Session low in INR.")
    close: float = Field(gt=0.0, description="Closing / last traded price in INR.")
    volume: float = Field(ge=0.0, description="Total traded volume in shares.")
    vwap: float | None = Field(default=None, gt=0.0, description="Volume-weighted average price.")

    @field_validator("high")
    @classmethod
    def _high_gte_low(cls, v: float, info: object) -> float:
        """high must be >= low; validated after both fields are parsed."""
        # ``info.data`` is available after Pydantic has parsed ``low``.
        data = getattr(info, "data", {})
        low = data.get("low")
        if low is not None and v < low:
            raise ValueError(f"high ({v}) must be >= low ({low})")
        return v

    @field_validator("close")
    @classmethod
    def _close_between_low_and_high(cls, v: float, info: object) -> float:
        """close must lie between low and high (inclusive)."""
        data = getattr(info, "data", {})
        low = data.get("low")
        high = data.get("high")
        if low is not None and high is not None:
            if not (low <= v <= high):
                raise ValueError(
                    f"close ({v}) must be between low ({low}) and high ({high})"
                )
        return v


class DataQualityMetadata(_WireSchema):
    """Quality gate metadata embedded inside every data-service2.0 response.

    ``score``               — DataConfidenceScore [0, 100].  Values < 70 trigger
                              ``PredictionProvenance.INSUFFICIENT_EVIDENCE``.
    ``signal_engine_allowed`` — When False, the signal engine is administratively
                              disabled for this instrument; provenance → UNAVAILABLE.
    ``data_as_of``          — UTC timestamp of the most recent constituent data point.
                              Must be strictly before the PIT boundary.
    """

    score: int = Field(ge=0, le=100, description="DataConfidenceScore [0, 100].")
    signal_engine_allowed: bool = Field(
        default=True,
        description="False when the signal engine is disabled for this instrument.",
    )
    data_as_of: AwareDatetime = Field(
        description="UTC timestamp of the most-recent constituent data point."
    )


class DataServiceResponse(_WireSchema):
    """Top-level envelope returned by data-service2.0 live-quote endpoints.

    ``GET /v1/india/quotes/{symbol}`` and ``GET /v1/india/quotes/batch``
    both wrap their payload in this envelope.

    Required fields
    ---------------
    - ``instrument_id``  — e.g. ``"NSE:NIFTY:IDX"``
    - ``symbol``         — e.g. ``"NIFTY"``
    - ``close``          — last traded price (required; pipeline gates on its presence)
    - ``quality``        — quality metadata block (required for PIT + gate checks)

    Optional fields
    ---------------
    Everything else degrades gracefully to ``None`` when the upstream service
    does not provide it.
    """

    # ── Required ──────────────────────────────────────────────────────────────
    instrument_id: str = Field(min_length=1, description="Exchange:Symbol:Type triplet.")
    symbol: str = Field(min_length=1, max_length=50, description="NSE/NFO instrument symbol.")
    close: float = Field(gt=0.0, description="Last traded price in INR.")
    quality: DataQualityMetadata

    # ── Optional OHLCV ────────────────────────────────────────────────────────
    open: float | None = Field(default=None, gt=0.0)
    high: float | None = Field(default=None, gt=0.0)
    low: float | None = Field(default=None, gt=0.0)
    volume: float | None = Field(default=None, ge=0.0)
    vwap: float | None = Field(default=None, gt=0.0)

    # ── Optional derivatives ──────────────────────────────────────────────────
    india_vix: float | None = Field(default=None, ge=0.0)
    put_call_ratio: float | None = Field(default=None, ge=0.0)
    change_pct: float | None = None
    previous_close: float | None = Field(default=None, gt=0.0)

    @field_validator("symbol")
    @classmethod
    def _symbol_uppercase(cls, v: str) -> str:
        """Normalise symbol to upper-case (data-service2.0 may return mixed case)."""
        return v.upper()


# ── SentinelPulse contracts ───────────────────────────────────────────────────


class SentinelSentimentBreakdown(_WireSchema):
    """Five-axis sentiment breakdown from SentinelPulse NLP inference.

    All values are in the range [-1.0, 1.0]:
      -1.0 = maximally bearish / risky
       0.0 = neutral
      +1.0 = maximally bullish / safe

    ``overall`` is the weighted composite of the other four axes.
    """

    overall: float = Field(ge=-1.0, le=1.0)
    market: float = Field(ge=-1.0, le=1.0)
    company: float = Field(ge=-1.0, le=1.0)
    macro: float = Field(ge=-1.0, le=1.0)
    risk: float = Field(ge=-1.0, le=1.0)


class SentinelNewsContext(_WireSchema):
    """Full news context payload from ``GET /api/v1/alphaforge/news-context/{instrument}``.

    Required fields
    ---------------
    - ``instrument``        — must match the requested symbol
    - ``news_impact_score`` — [0, 1]: 0 = no impact, 1 = maximum market-moving impact
    - ``impact_direction``  — BULLISH | BEARISH | NEUTRAL
    - ``impact_confidence`` — [0, 1]: model confidence in ``impact_direction``
    - ``sentiment``         — five-axis breakdown (see ``SentinelSentimentBreakdown``)
    - ``as_of``             — UTC timestamp of the most-recent news item considered

    Optional
    --------
    - ``market_regime``     — NLP-derived regime string (e.g. ``"RISK_ON"``)
    - ``event_tags``        — list of event tags (e.g. ``["RBI_POLICY", "EARNINGS"]``)
    """

    # ── Required ──────────────────────────────────────────────────────────────
    instrument: str = Field(min_length=1, max_length=50)
    news_impact_score: float = Field(ge=0.0, le=1.0)
    impact_direction: ImpactDirection
    impact_confidence: float = Field(ge=0.0, le=1.0)
    sentiment: SentinelSentimentBreakdown
    as_of: AwareDatetime

    # ── Optional ──────────────────────────────────────────────────────────────
    market_regime: str | None = None
    event_tags: list[str] = Field(default_factory=list)

    @field_validator("instrument")
    @classmethod
    def _instrument_uppercase(cls, v: str) -> str:
        return v.upper()

    @field_validator("as_of")
    @classmethod
    def _as_of_not_in_future(cls, v: datetime) -> datetime:
        """Reject ``as_of`` timestamps more than 60 seconds in the future.

        SentinelPulse runs on the same infrastructure as ml-service2.0; a
        future timestamp is almost certainly a clock-skew bug or test data error.
        """
        from datetime import timezone

        now = datetime.now(tz=timezone.utc)
        delta = (v - now).total_seconds()
        if delta > 60:
            raise ValueError(
                f"SentinelPulse as_of timestamp {v.isoformat()} is "
                f"{delta:.0f}s in the future — likely a PIT violation."
            )
        return v


class SentinelPulseResponse(_WireSchema):
    """Top-level wrapper for SentinelPulse ML feature endpoint responses.

    ``GET /api/v1/ml/features/asset/{assetId}`` returns this envelope.
    The ``context`` field contains the full ``SentinelNewsContext`` payload.
    ``request_id`` is present on all successful responses and used for
    distributed tracing between SentinelPulse and ml-service2.0.
    """

    request_id: str = Field(min_length=1, description="SentinelPulse request trace ID.")
    context: SentinelNewsContext
    processing_time_ms: float = Field(ge=0.0, default=0.0)
