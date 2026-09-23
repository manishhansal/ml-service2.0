"""
tests/test_data_contracts.py — TDD: Phase 2 data contract validation tests.

These tests define and enforce the EXACT wire-format contracts between
ml-service2.0 and its upstream services:
  - data-service2.0  (OHLCV bars, live quotes, quality metadata)
  - SentinelPulse    (news impact vector, sentiment breakdown)

TDD mandate
-----------
All tests MUST pass against the Pydantic model stubs in
``src/data/contracts.py``.  Because the models are fully defined (they are
interface contracts, not logic stubs), these tests are expected to PASS
once the contracts file exists.  Any test that currently fails indicates
a contract gap that must be fixed before Phase 3.

Test categories
---------------
1. Schema acceptance   — valid payloads parse without errors.
2. Schema rejection    — malformed payloads raise ``ValidationError``.
3. PIT enforcement     — future ``as_of`` timestamps are rejected.
4. Field boundary      — numeric fields respect ``ge/le`` constraints.
5. Hypothesis property — Hypothesis-generated malformed data is always rejected.

Markers: @pytest.mark.unit, @pytest.mark.tdd
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as h_settings
from hypothesis import strategies as st
from pydantic import ValidationError

# Bootstrap env before src imports
os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from src.data.contracts import (
    DataQualityMetadata,
    DataServiceResponse,
    OHLCVBar,
    SentinelNewsContext,
    SentinelPulseResponse,
    SentinelSentimentBreakdown,
)
from src.schemas.base import ImpactDirection


# ── Helpers ───────────────────────────────────────────────────────────────────

_NOW_UTC = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)
_PAST_UTC = _NOW_UTC - timedelta(minutes=15)
_FUTURE_UTC = _NOW_UTC + timedelta(minutes=30)


def _valid_ohlcv_bar(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "timestamp": _PAST_UTC.isoformat(),
        "open": 22050.0,
        "high": 22250.0,
        "low": 21980.0,
        "close": 22150.0,
        "volume": 1_234_567.0,
        "vwap": 22110.0,
    }
    base.update(overrides)
    return base


def _valid_quality_metadata(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "score": 88,
        "signal_engine_allowed": True,
        "data_as_of": _PAST_UTC.isoformat(),
    }
    base.update(overrides)
    return base


def _valid_data_service_response(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "instrument_id": "NSE:NIFTY:IDX",
        "symbol": "NIFTY",
        "close": 22150.35,
        "open": 22050.10,
        "high": 22250.80,
        "low": 21980.45,
        "volume": 1_234_567.0,
        "vwap": 22110.50,
        "quality": _valid_quality_metadata(),
    }
    base.update(overrides)
    return base


def _valid_sentiment(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "overall": 0.43,
        "market": 0.38,
        "company": 0.51,
        "macro": -0.05,
        "risk": -0.09,
    }
    base.update(overrides)
    return base


def _valid_sentinel_context(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "instrument": "NIFTY",
        "as_of": _PAST_UTC.isoformat(),
        "news_impact_score": 0.512,
        "impact_direction": "BULLISH",
        "impact_confidence": 0.73,
        "sentiment": _valid_sentiment(),
        "market_regime": "RISK_ON",
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OHLCVBar — acceptance
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.tdd
class TestOHLCVBarAcceptance:
    """Valid OHLCV bars must parse without errors."""

    def test_typical_nifty_bar_parses(self) -> None:
        """A realistic NIFTY daily bar parses successfully."""
        bar = OHLCVBar.model_validate(_valid_ohlcv_bar())
        assert bar.close == 22150.0
        assert bar.volume == 1_234_567.0

    def test_vwap_is_optional(self) -> None:
        """``vwap`` is optional — parsing succeeds when absent."""
        data = _valid_ohlcv_bar()
        data.pop("vwap")
        bar = OHLCVBar.model_validate(data)
        assert bar.vwap is None

    def test_timestamp_is_utc_aware(self) -> None:
        """Parsed ``timestamp`` must be UTC-aware."""
        bar = OHLCVBar.model_validate(_valid_ohlcv_bar())
        assert bar.timestamp.tzinfo is not None

    def test_high_equals_close_is_valid(self) -> None:
        """When high == close, the bar is valid (e.g. strong up-day that closed at high)."""
        bar = OHLCVBar.model_validate(
            _valid_ohlcv_bar(open=22000.0, high=22200.0, low=21900.0, close=22200.0)
        )
        assert bar.close == bar.high

    def test_low_equals_close_is_valid(self) -> None:
        """When low == close, the bar is valid (e.g. strong down-day that closed at low)."""
        bar = OHLCVBar.model_validate(
            _valid_ohlcv_bar(open=22200.0, high=22300.0, low=22000.0, close=22000.0)
        )
        assert bar.close == bar.low


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OHLCVBar — rejection
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.tdd
class TestOHLCVBarRejection:
    """Malformed OHLCV bars must raise ValidationError."""

    def test_rejects_naive_timestamp(self) -> None:
        """A naive (timezone-unaware) timestamp must be rejected."""
        data = _valid_ohlcv_bar(timestamp="2024-01-15T09:15:00")  # no tz
        with pytest.raises(ValidationError, match="UTC-aware"):
            OHLCVBar.model_validate(data)

    def test_rejects_zero_close(self) -> None:
        """close must be > 0; zero is invalid."""
        data = _valid_ohlcv_bar(close=0.0)
        with pytest.raises(ValidationError):
            OHLCVBar.model_validate(data)

    def test_rejects_negative_volume(self) -> None:
        """volume must be >= 0; negative is invalid."""
        data = _valid_ohlcv_bar(volume=-100.0)
        with pytest.raises(ValidationError):
            OHLCVBar.model_validate(data)

    def test_rejects_high_less_than_low(self) -> None:
        """high < low is physically impossible — must be rejected."""
        data = _valid_ohlcv_bar(open=22100.0, high=21900.0, low=22200.0, close=22050.0)
        with pytest.raises(ValidationError):
            OHLCVBar.model_validate(data)

    def test_rejects_close_above_high(self) -> None:
        """close > high violates OHLCV definition — must be rejected."""
        data = _valid_ohlcv_bar(open=22000.0, high=22200.0, low=21900.0, close=22500.0)
        with pytest.raises(ValidationError):
            OHLCVBar.model_validate(data)

    def test_rejects_close_below_low(self) -> None:
        """close < low violates OHLCV definition — must be rejected."""
        data = _valid_ohlcv_bar(open=22100.0, high=22300.0, low=22000.0, close=21500.0)
        with pytest.raises(ValidationError):
            OHLCVBar.model_validate(data)

    def test_rejects_missing_close(self) -> None:
        """close is required — missing field must raise ValidationError."""
        data = _valid_ohlcv_bar()
        data.pop("close")
        with pytest.raises(ValidationError):
            OHLCVBar.model_validate(data)

    def test_rejects_string_close(self) -> None:
        """Non-numeric string that cannot be coerced to float must be rejected."""
        data = _valid_ohlcv_bar(close="not-a-number")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            OHLCVBar.model_validate(data)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DataQualityMetadata — boundaries
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.tdd
class TestDataQualityMetadata:
    """DataQualityMetadata field constraints."""

    def test_score_100_is_valid(self) -> None:
        meta = DataQualityMetadata.model_validate(_valid_quality_metadata(score=100))
        assert meta.score == 100

    def test_score_0_is_valid(self) -> None:
        meta = DataQualityMetadata.model_validate(_valid_quality_metadata(score=0))
        assert meta.score == 0

    def test_score_101_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DataQualityMetadata.model_validate(_valid_quality_metadata(score=101))

    def test_score_minus_1_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DataQualityMetadata.model_validate(_valid_quality_metadata(score=-1))

    def test_signal_engine_allowed_defaults_true(self) -> None:
        """signal_engine_allowed defaults to True when omitted."""
        data = _valid_quality_metadata()
        data.pop("signal_engine_allowed")
        meta = DataQualityMetadata.model_validate(data)
        assert meta.signal_engine_allowed is True

    def test_naive_data_as_of_rejected(self) -> None:
        """Naive data_as_of timestamp must be rejected."""
        with pytest.raises(ValidationError, match="UTC-aware"):
            DataQualityMetadata.model_validate(
                _valid_quality_metadata(data_as_of="2024-01-15T09:15:00")
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DataServiceResponse — acceptance & rejection
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.tdd
class TestDataServiceResponseAcceptance:
    """Valid DataServiceResponse payloads parse correctly."""

    def test_full_payload_parses(self) -> None:
        resp = DataServiceResponse.model_validate(_valid_data_service_response())
        assert resp.symbol == "NIFTY"
        assert resp.close == 22150.35
        assert resp.quality.score == 88

    def test_symbol_normalised_to_uppercase(self) -> None:
        """Symbol 'nifty' must be normalised to 'NIFTY'."""
        resp = DataServiceResponse.model_validate(
            _valid_data_service_response(symbol="nifty")
        )
        assert resp.symbol == "NIFTY"

    def test_optional_fields_absent_is_ok(self) -> None:
        """Optional OHLCV fields may be absent."""
        data = _valid_data_service_response()
        for field in ("open", "high", "low", "volume", "vwap", "india_vix", "put_call_ratio"):
            data.pop(field, None)
        resp = DataServiceResponse.model_validate(data)
        assert resp.open is None
        assert resp.india_vix is None


@pytest.mark.unit
@pytest.mark.tdd
class TestDataServiceResponseRejection:
    """Invalid DataServiceResponse payloads must raise ValidationError."""

    def test_missing_instrument_id_rejected(self) -> None:
        data = _valid_data_service_response()
        data.pop("instrument_id")
        with pytest.raises(ValidationError):
            DataServiceResponse.model_validate(data)

    def test_missing_quality_block_rejected(self) -> None:
        data = _valid_data_service_response()
        data.pop("quality")
        with pytest.raises(ValidationError):
            DataServiceResponse.model_validate(data)

    def test_zero_close_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DataServiceResponse.model_validate(_valid_data_service_response(close=0.0))

    def test_negative_close_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DataServiceResponse.model_validate(_valid_data_service_response(close=-100.0))

    def test_empty_symbol_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DataServiceResponse.model_validate(_valid_data_service_response(symbol=""))

    def test_negative_india_vix_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DataServiceResponse.model_validate(
                _valid_data_service_response(india_vix=-1.0)
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SentinelSentimentBreakdown — boundaries
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.tdd
class TestSentinelSentimentBreakdown:
    """Sentiment scores must lie in [-1.0, 1.0]."""

    def test_max_bullish_parses(self) -> None:
        sent = SentinelSentimentBreakdown.model_validate(
            {k: 1.0 for k in ("overall", "market", "company", "macro", "risk")}
        )
        assert sent.overall == 1.0

    def test_max_bearish_parses(self) -> None:
        sent = SentinelSentimentBreakdown.model_validate(
            {k: -1.0 for k in ("overall", "market", "company", "macro", "risk")}
        )
        assert sent.overall == -1.0

    def test_score_above_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SentinelSentimentBreakdown.model_validate(
                _valid_sentiment(overall=1.01)
            )

    def test_score_below_minus_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SentinelSentimentBreakdown.model_validate(
                _valid_sentiment(market=-1.001)
            )

    def test_string_score_rejected(self) -> None:
        """Non-numeric string that cannot be coerced to int must be rejected."""
        with pytest.raises(ValidationError):
            SentinelSentimentBreakdown.model_validate(
                _valid_sentiment(overall="not-a-number")  # type: ignore[arg-type]
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SentinelNewsContext — acceptance & rejection
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.tdd
class TestSentinelNewsContextAcceptance:
    """Valid SentinelPulse payloads parse correctly."""

    def test_bullish_context_parses(self) -> None:
        ctx = SentinelNewsContext.model_validate(_valid_sentinel_context())
        assert ctx.impact_direction == ImpactDirection.BULLISH
        assert ctx.news_impact_score == 0.512
        assert ctx.instrument == "NIFTY"

    def test_instrument_normalised_to_uppercase(self) -> None:
        ctx = SentinelNewsContext.model_validate(_valid_sentinel_context(instrument="nifty"))
        assert ctx.instrument == "NIFTY"

    def test_event_tags_default_to_empty_list(self) -> None:
        data = _valid_sentinel_context()
        data.pop("event_tags", None)
        ctx = SentinelNewsContext.model_validate(data)
        assert ctx.event_tags == []

    def test_market_regime_optional(self) -> None:
        data = _valid_sentinel_context()
        data.pop("market_regime")
        ctx = SentinelNewsContext.model_validate(data)
        assert ctx.market_regime is None

    def test_all_impact_directions_parse(self) -> None:
        """All three ImpactDirection values must parse without error."""
        for direction in ("BULLISH", "BEARISH", "NEUTRAL"):
            ctx = SentinelNewsContext.model_validate(
                _valid_sentinel_context(impact_direction=direction)
            )
            assert ctx.impact_direction.value == direction


@pytest.mark.unit
@pytest.mark.tdd
class TestSentinelNewsContextRejection:
    """Invalid SentinelPulse payloads must raise ValidationError."""

    def test_missing_news_impact_score_rejected(self) -> None:
        data = _valid_sentinel_context()
        data.pop("news_impact_score")
        with pytest.raises(ValidationError):
            SentinelNewsContext.model_validate(data)

    def test_impact_score_above_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SentinelNewsContext.model_validate(
                _valid_sentinel_context(news_impact_score=1.01)
            )

    def test_impact_score_below_0_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SentinelNewsContext.model_validate(
                _valid_sentinel_context(news_impact_score=-0.001)
            )

    def test_invalid_impact_direction_rejected(self) -> None:
        """Invalid direction string must raise ValidationError."""
        with pytest.raises(ValidationError):
            SentinelNewsContext.model_validate(
                _valid_sentinel_context(impact_direction="MILDLY_BULLISH")
            )

    def test_future_as_of_timestamp_rejected(self) -> None:
        """as_of more than 60 seconds in the future must be rejected (PIT guard)."""
        far_future = (datetime.now(tz=timezone.utc) + timedelta(hours=2)).isoformat()
        with pytest.raises(ValidationError, match="future"):
            SentinelNewsContext.model_validate(
                _valid_sentinel_context(as_of=far_future)
            )

    def test_naive_as_of_rejected(self) -> None:
        """Naive (timezone-unaware) as_of timestamp must be rejected."""
        with pytest.raises(ValidationError, match="UTC-aware"):
            SentinelNewsContext.model_validate(
                _valid_sentinel_context(as_of="2024-01-15T09:15:00")
            )

    def test_missing_sentiment_block_rejected(self) -> None:
        data = _valid_sentinel_context()
        data.pop("sentiment")
        with pytest.raises(ValidationError):
            SentinelNewsContext.model_validate(data)

    def test_empty_instrument_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SentinelNewsContext.model_validate(
                _valid_sentinel_context(instrument="")
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SentinelPulseResponse envelope
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.tdd
class TestSentinelPulseResponse:
    """Top-level SentinelPulse response envelope."""

    def test_full_response_parses(self) -> None:
        resp = SentinelPulseResponse.model_validate({
            "request_id": "req-abc-123",
            "context": _valid_sentinel_context(),
            "processing_time_ms": 45.2,
        })
        assert resp.request_id == "req-abc-123"
        assert resp.context.instrument == "NIFTY"

    def test_empty_request_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SentinelPulseResponse.model_validate({
                "request_id": "",
                "context": _valid_sentinel_context(),
            })

    def test_negative_processing_time_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SentinelPulseResponse.model_validate({
                "request_id": "req-123",
                "context": _valid_sentinel_context(),
                "processing_time_ms": -1.0,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Hypothesis property-based tests
# ═══════════════════════════════════════════════════════════════════════════════


# Strategies for generating invalid field values
_st_bad_score = st.one_of(
    st.integers(max_value=-1),
    st.integers(min_value=101),
    # Non-numeric strings that cannot be coerced to int
    st.from_regex(r"[a-zA-Z][a-zA-Z0-9 ]{0,9}", fullmatch=True),
    st.none(),
)

_st_bad_price = st.one_of(
    st.floats(max_value=0.0, allow_nan=False),
    # Non-numeric strings (cannot be coerced to float)
    st.from_regex(r"[a-zA-Z][a-zA-Z0-9 ]{0,9}", fullmatch=True),
    st.none(),
)

_st_bad_sentiment_score = st.one_of(
    st.floats(min_value=1.001, max_value=100.0, allow_nan=False),
    st.floats(max_value=-1.001, allow_nan=False),
    # Non-numeric strings
    st.from_regex(r"[a-zA-Z][a-zA-Z0-9 ]{0,9}", fullmatch=True),
)

_st_bad_impact_score = st.one_of(
    st.floats(min_value=1.001, max_value=10.0, allow_nan=False),
    st.floats(max_value=-0.001, allow_nan=False),
    # Non-numeric strings
    st.from_regex(r"[a-zA-Z][a-zA-Z0-9 ]{0,9}", fullmatch=True),
)


@pytest.mark.unit
@pytest.mark.tdd
class TestHypothesisContracts:
    """Hypothesis property-based tests that ANY malformed value is rejected."""

    @given(bad_score=_st_bad_score)
    @h_settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_data_quality_score_rejects_all_invalid_values(self, bad_score: Any) -> None:
        """DataQualityMetadata.score must reject any value outside [0, 100]."""
        with pytest.raises((ValidationError, TypeError)):
            DataQualityMetadata.model_validate(
                _valid_quality_metadata(score=bad_score)
            )

    @given(bad_price=_st_bad_price)
    @h_settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_ohlcv_close_rejects_all_non_positive(self, bad_price: Any) -> None:
        """OHLCVBar.close must reject zero, negative, and non-numeric values."""
        with pytest.raises((ValidationError, TypeError)):
            OHLCVBar.model_validate(_valid_ohlcv_bar(close=bad_price))

    @given(bad_score=_st_bad_sentiment_score)
    @h_settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_sentiment_overall_rejects_out_of_range(self, bad_score: Any) -> None:
        """SentinelSentimentBreakdown.overall must reject values outside [-1, 1]."""
        with pytest.raises((ValidationError, TypeError)):
            SentinelSentimentBreakdown.model_validate(
                _valid_sentiment(overall=bad_score)
            )

    @given(bad_impact=_st_bad_impact_score)
    @h_settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_news_impact_score_rejects_out_of_range(self, bad_impact: Any) -> None:
        """SentinelNewsContext.news_impact_score must reject values outside [0, 1]."""
        with pytest.raises((ValidationError, TypeError)):
            SentinelNewsContext.model_validate(
                _valid_sentinel_context(news_impact_score=bad_impact)
            )

    @given(
        future_offset_seconds=st.integers(min_value=61, max_value=7200)
    )
    @h_settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_sentinel_future_timestamp_always_rejected(
        self, future_offset_seconds: int
    ) -> None:
        """Any as_of timestamp more than 60s in the future must ALWAYS be rejected."""
        future_ts = (
            datetime.now(tz=timezone.utc) + timedelta(seconds=future_offset_seconds)
        ).isoformat()
        with pytest.raises(ValidationError):
            SentinelNewsContext.model_validate(
                _valid_sentinel_context(as_of=future_ts)
            )

    @given(
        missing_field=st.sampled_from([
            "instrument_id", "symbol", "close", "quality"
        ])
    )
    @h_settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_data_service_response_rejects_any_missing_required_field(
        self, missing_field: str
    ) -> None:
        """DataServiceResponse must reject payloads missing any required field."""
        data = _valid_data_service_response()
        data.pop(missing_field, None)
        with pytest.raises(ValidationError):
            DataServiceResponse.model_validate(data)
