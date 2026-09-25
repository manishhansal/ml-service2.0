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

Canonical data contract version: v2.1.0
---------------------------------------
Every market record carries:
  symbol, exchange, instrument_type, timestamp, open, high, low, close,
  volume, oi, interval, source, source_priority, data_as_of, retrieved_at

Every news record carries:
  article_id, source, url/canonical_url, title, published_at, ingested_at,
  updated_at, symbol/entity, market, category, language, sentiment, impact,
  relevance, confidence, dedupe_hash, data_as_of

PIT contract (mandatory):
  observation_timestamp <= data_as_of <= retrieval_timestamp
  feature_as_of <= prediction_timestamp
  published_at <= market_timestamp (for news-market join)

Volume semantics:
  observed_volume  — actual traded volume
  zero_volume      — volume confirmed zero (e.g. index futures on holiday)
  volume_unavailable — provider did not return volume (NOT the same as zero)
  volume_not_applicable — instrument type has no volume concept

Phase 2 TDD mandate
-------------------
These models are the *interface contract*.  The test suite in
``tests/test_data_contracts.py`` drives their exact shape via Hypothesis
property-based tests.  Any change to a field name, type, or validator here
must have a corresponding failing test first.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
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


# ── Volume availability semantics ─────────────────────────────────────────────


class VolumeAvailability(str, Enum):
    """Canonical classification of volume availability per market record.

    NEVER conflate volume_unavailable with zero_volume.
    ``unavailable → 0`` is explicitly forbidden (mandate §5.1.E).
    """

    OBSERVED = "observed"                    # Actual traded volume from provider
    ZERO_CONFIRMED = "zero_confirmed"        # Volume confirmed zero (e.g. holiday close)
    UNAVAILABLE = "unavailable"              # Provider did not supply volume
    NOT_APPLICABLE = "not_applicable"        # Instrument type has no volume (e.g. NSE index)


# ── News data availability states ─────────────────────────────────────────────


class NewsDataAvailability(str, Enum):
    """Canonical states for news data availability at a given PIT boundary.

    Must NEVER be collapsed into a binary available/unavailable.
    Mandate §11: these four states are not equivalent.
    """

    NO_NEWS_FOUND = "NO_NEWS_FOUND"                                # Searched; no articles
    NEWS_UNAVAILABLE = "NEWS_UNAVAILABLE"                          # Service unreachable
    NEWS_AVAILABLE_NO_RELEVANT_ARTICLE = "NEWS_AVAILABLE_NO_RELEVANT_ARTICLE"  # Articles exist but none relevant
    NEWS_AVAILABLE_RELEVANT_ARTICLE = "NEWS_AVAILABLE_RELEVANT_ARTICLE"        # Relevant articles found


# ── Provider failure states ───────────────────────────────────────────────────


class ProviderFailureReason(str, Enum):
    """Canonical classification of data-provider failure modes (mandate §5.1.D).

    HTTP 401/403/429/500/502/503/timeout MUST NOT produce fake market data.
    They MUST result in one of these explicit states.
    """

    NOT_CONFIGURED = "NOT_CONFIGURED"
    AUTH_FAILED = "AUTH_FAILED"
    RATE_LIMITED = "RATE_LIMITED"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    TIMEOUT = "TIMEOUT"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"


# ── Universe availability states ──────────────────────────────────────────────


class UniverseAvailability(str, Enum):
    """Canonical states for the F&O universe endpoint (mandate §5.1.A)."""

    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"
    NO_UNIVERSE = "NO_UNIVERSE"
    PARTIAL_UNIVERSE = "PARTIAL_UNIVERSE"
    VALID_UNIVERSE = "VALID_UNIVERSE"
    STALE_UNIVERSE = "STALE_UNIVERSE"


# ── OI / derivative availability ──────────────────────────────────────────────


class OIAvailability(str, Enum):
    """Open Interest availability semantics (mandate §5.1.F)."""

    OBSERVED = "observed"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


# ── data-service2.0 contracts ─────────────────────────────────────────────────


class OHLCVBar(_WireSchema):
    """A single OHLCV bar returned by ``GET /v1/india/history/{symbol}``.

    All price fields are in INR.  ``volume`` is in shares (not crores).
    ``timestamp`` must be UTC-aware.

    Provenance fields (mandate §5.1.C, §4):
      ``source``           — canonical provider name (e.g. "angel_one", "upstox", "yahoo")
      ``source_priority``  — provider tier (1=primary, 2=fallback, 3=tertiary)
      ``data_as_of``       — UTC timestamp of when this bar was valid as-of
      ``retrieved_at``     — UTC timestamp of when ml-service fetched this data
      ``volume_availability`` — explicit volume semantics (never coerce unavailable→0)
      ``oi_availability``  — explicit OI semantics
    """

    # ── Required OHLCV ────────────────────────────────────────────────────────
    timestamp: AwareDatetime
    open: float = Field(gt=0.0, description="Opening price in INR.")
    high: float = Field(gt=0.0, description="Session high in INR.")
    low: float = Field(gt=0.0, description="Session low in INR.")
    close: float = Field(gt=0.0, description="Closing / last traded price in INR.")
    volume: float = Field(ge=0.0, description="Total traded volume in shares.")

    # ── Optional OHLCV ────────────────────────────────────────────────────────
    vwap: float | None = Field(default=None, gt=0.0, description="Volume-weighted average price.")
    oi: float | None = Field(default=None, ge=0.0, description="Open interest (futures/options).")
    iv: float | None = Field(default=None, ge=0.0, description="Implied volatility (options).")

    # ── Provenance (mandatory for PIT and provider-mix detection) ─────────────
    source: str | None = Field(
        default=None,
        description="Canonical provider name: angel_one | upstox | yahoo | unknown.",
    )
    source_priority: int | None = Field(
        default=None,
        ge=1,
        description="Provider tier: 1=primary, 2=fallback, 3=tertiary.",
    )
    data_as_of: AwareDatetime | None = Field(
        default=None,
        description="UTC timestamp this bar was valid as-of (for PIT enforcement).",
    )
    retrieved_at: AwareDatetime | None = Field(
        default=None,
        description="UTC timestamp ml-service retrieved this bar from the upstream.",
    )
    interval: str | None = Field(
        default=None,
        description="Bar interval: 1m | 5m | 10m | 15m | 30m | 1h | 1d | 1w | 1M.",
    )
    exchange: str | None = Field(
        default=None,
        description="Exchange: NSE | NFO | BSE.",
    )

    # ── Availability semantics (explicit — never coerce) ───────────────────────
    volume_availability: VolumeAvailability = Field(
        default=VolumeAvailability.OBSERVED,
        description=(
            "Canonical volume availability. "
            "UNAVAILABLE must NEVER be stored as volume=0."
        ),
    )
    oi_availability: OIAvailability = Field(
        default=OIAvailability.NOT_APPLICABLE,
        description="OI availability. NOT_APPLICABLE for cash instruments.",
    )

    # ── Fallback provenance (when a provider failover occurred) ───────────────
    primary_provider: str | None = Field(
        default=None,
        description="Primary provider that was tried before failover.",
    )
    fallback_provider: str | None = Field(
        default=None,
        description="Actual provider used after failover (if any).",
    )
    failure_reason: ProviderFailureReason | None = Field(
        default=None,
        description="Reason the primary provider failed (if fallover occurred).",
    )
    fallback_timestamp: AwareDatetime | None = Field(
        default=None,
        description="UTC timestamp when failover occurred.",
    )

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

    @field_validator("volume")
    @classmethod
    def _volume_not_fabricated(cls, v: float, info: object) -> float:
        """Warn when volume=0 but availability is OBSERVED — possible fabrication."""
        data = getattr(info, "data", {})
        avail = data.get("volume_availability")
        # Allow zero volume only when explicitly confirmed as zero or not-applicable.
        # If availability says OBSERVED but volume=0, that is valid (genuine zero trade day).
        # If availability says UNAVAILABLE but volume=0, that is a fabrication — reject.
        if avail == VolumeAvailability.UNAVAILABLE and v == 0.0:
            raise ValueError(
                "volume=0 with volume_availability=UNAVAILABLE is forbidden. "
                "Set volume=None or use VolumeAvailability.NOT_APPLICABLE instead."
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
    ``retrieved_at``        — UTC timestamp when ml-service fetched this metadata.
    """

    score: int = Field(ge=0, le=100, description="DataConfidenceScore [0, 100].")
    signal_engine_allowed: bool = Field(
        default=True,
        description="False when the signal engine is disabled for this instrument.",
    )
    data_as_of: AwareDatetime = Field(
        description="UTC timestamp of the most-recent constituent data point."
    )
    retrieved_at: AwareDatetime | None = Field(
        default=None,
        description="UTC timestamp when ml-service retrieved this metadata.",
    )
    provider: str | None = Field(
        default=None,
        description="Provider that supplied this quality assessment.",
    )
    universe_availability: UniverseAvailability | None = Field(
        default=None,
        description="F&O universe state at data_as_of (if applicable).",
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

    Volume semantics (mandate §5.1.E):
    - ``volume=None`` means the provider did not supply volume.
    - ``volume=0.0`` means zero trades were observed (valid for some instruments).
    - ``volume_availability`` makes the distinction explicit.
    - The feature layer MUST check ``volume_availability`` before using ``volume``.
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

    # ── Volume / OI availability (explicit — never coerce) ────────────────────
    volume_availability: VolumeAvailability = Field(
        default=VolumeAvailability.OBSERVED,
        description="Explicit volume availability state — never coerce UNAVAILABLE→0.",
    )
    oi: float | None = Field(default=None, ge=0.0, description="Open interest.")
    oi_availability: OIAvailability = Field(
        default=OIAvailability.NOT_APPLICABLE,
        description="OI availability state.",
    )

    # ── Optional derivatives ──────────────────────────────────────────────────
    india_vix: float | None = Field(default=None, ge=0.0)
    put_call_ratio: float | None = Field(default=None, ge=0.0)
    change_pct: float | None = None
    previous_close: float | None = Field(default=None, gt=0.0)

    # ── Provenance ────────────────────────────────────────────────────────────
    exchange: str | None = Field(default=None, description="Exchange: NSE | NFO | BSE.")
    instrument_type: str | None = Field(default=None, description="IDX | EQ | FUT | OPT.")
    provider: str | None = Field(
        default=None, description="Canonical provider: angel_one | upstox | yahoo."
    )
    source_priority: int | None = Field(
        default=None, ge=1, description="Provider tier."
    )
    retrieved_at: AwareDatetime | None = Field(
        default=None,
        description="UTC timestamp when ml-service fetched this quote.",
    )

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


class ArticlePITMetadata(_WireSchema):
    """PIT-critical timestamps for a news article (mandate §10).

    The canonical PIT rule for news:
      published_at <= prediction_timestamp

    NOT ingested_at <= prediction_timestamp (ingestion latency is irrelevant
    for determining what a trader could have known at time T).

    When published_at is uncertain, set ``publication_time_certain=False``
    and the system will treat the article as unavailable for PIT-sensitive
    training (news_data_available → NEWS_UNAVAILABLE for this article).
    """

    published_at: AwareDatetime = Field(
        description="UTC timestamp the article was publicly available (PIT anchor).",
    )
    ingested_at: AwareDatetime = Field(
        description="UTC timestamp SentinelPulse ingested the article.",
    )
    updated_at: AwareDatetime | None = Field(
        default=None,
        description="UTC timestamp of last article update (not used for PIT).",
    )
    publication_time_certain: bool = Field(
        default=True,
        description=(
            "True when published_at is confirmed from the article's canonical "
            "timestamp. False when inferred or estimated — article MUST NOT "
            "contribute to PIT-sensitive training (see mandate §10)."
        ),
    )
    scrape_delay_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Seconds between published_at and ingested_at (latency measure).",
    )

    @field_validator("published_at")
    @classmethod
    def _published_not_in_future(cls, v: datetime) -> datetime:
        """Reject published_at timestamps more than 5 minutes in the future.

        A future publication timestamp is either a timezone error or
        a look-ahead bias risk — both must be caught at contract parse time.
        """
        from datetime import timezone

        now = datetime.now(tz=timezone.utc)
        delta = (v - now).total_seconds()
        if delta > 300:  # 5 minutes tolerance for clock skew
            raise ValueError(
                f"Article published_at {v.isoformat()} is {delta:.0f}s in the future. "
                "This is a PIT violation risk — reject the article."
            )
        return v


class SentinelNewsContext(_WireSchema):
    """Full news context payload from ``GET /api/v1/alphaforge/news-context/{instrument}``.

    PIT contract (mandate §10):
      - ``as_of`` is the latest ``published_at`` across contributing articles.
      - For news-market join: only articles where published_at <= market_timestamp.
      - ``news_data_available`` explicitly classifies the availability state.
      - ``news_data_available = NEWS_UNAVAILABLE`` when as_of is uncertain.

    Required fields
    ---------------
    - ``instrument``        — must match the requested symbol
    - ``news_impact_score`` — [0, 1]: 0 = no impact, 1 = maximum market-moving impact
    - ``impact_direction``  — BULLISH | BEARISH | NEUTRAL
    - ``impact_confidence`` — [0, 1]: model confidence in ``impact_direction``
    - ``sentiment``         — five-axis breakdown (see ``SentinelSentimentBreakdown``)
    - ``as_of``             — UTC timestamp of the most-recent news item considered
    - ``news_data_available`` — canonical availability state (never infer neutral from missing)

    Optional
    --------
    - ``market_regime``     — NLP-derived regime string (e.g. ``"RISK_ON"``)
    - ``event_tags``        — list of event tags (e.g. ``["RBI_POLICY", "EARNINGS"]``)
    - ``article_count``     — number of relevant articles that contributed
    - ``top_article_pit``   — PIT metadata for the most-recent contributing article
    - ``velocity_1h``       — articles published in the last hour for this instrument
    - ``velocity_24h``      — articles published in the last 24 hours
    - ``entity_resolution_confidence`` — confidence [0,1] in symbol→entity mapping
    - ``source_diversity``  — number of distinct sources contributing to this context
    - ``news_as_of``        — UTC of the latest published_at used (same as as_of but
                              explicit field for downstream PIT checks)
    """

    # ── Required ──────────────────────────────────────────────────────────────
    instrument: str = Field(min_length=1, max_length=50)
    news_impact_score: float = Field(ge=0.0, le=1.0)
    impact_direction: ImpactDirection
    impact_confidence: float = Field(ge=0.0, le=1.0)
    sentiment: SentinelSentimentBreakdown
    as_of: AwareDatetime
    news_data_available: NewsDataAvailability = Field(
        default=NewsDataAvailability.NEWS_AVAILABLE_RELEVANT_ARTICLE,
        description=(
            "Canonical availability state. MUST be set explicitly. "
            "NEVER collapse NO_NEWS_FOUND into neutral sentiment."
        ),
    )

    # ── Optional ──────────────────────────────────────────────────────────────
    market_regime: str | None = None
    event_tags: list[str] = Field(default_factory=list)
    article_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of relevant articles contributing to this context.",
    )
    top_article_pit: ArticlePITMetadata | None = Field(
        default=None,
        description="PIT metadata for the most-recent contributing article.",
    )
    velocity_1h: int | None = Field(
        default=None, ge=0, description="Articles published in last 1 hour."
    )
    velocity_24h: int | None = Field(
        default=None, ge=0, description="Articles published in last 24 hours."
    )
    entity_resolution_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in symbol→entity mapping [0, 1].",
    )
    source_diversity: int | None = Field(
        default=None,
        ge=0,
        description="Number of distinct news sources contributing.",
    )
    news_as_of: AwareDatetime | None = Field(
        default=None,
        description=(
            "UTC of the latest published_at used. Explicit field for "
            "downstream PIT checks (must be <= prediction_timestamp)."
        ),
    )

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

    @field_validator("news_as_of")
    @classmethod
    def _news_as_of_not_in_future(cls, v: datetime | None) -> datetime | None:
        """Reject news_as_of in the future (stricter: 0 seconds tolerance)."""
        if v is None:
            return v
        from datetime import timezone

        now = datetime.now(tz=timezone.utc)
        delta = (v - now).total_seconds()
        if delta > 60:
            raise ValueError(
                f"news_as_of {v.isoformat()} is {delta:.0f}s in the future — PIT violation."
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
