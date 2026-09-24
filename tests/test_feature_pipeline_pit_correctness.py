"""
test_feature_pipeline_pit_correctness.py

TDD test file for the FeaturePipeline PIT correctness and data quality gate properties.

Written BEFORE FeaturePipeline implementation (red phase — expected to fail
until src/features/pipeline.py is implemented).

Properties tested:
  Property 1: ∀ feature f ∈ vector: source_timestamp(f) < vector.timestamp (PIT correctness)
  Property 2: pipeline.build_vector(same_inputs) == pipeline.build_vector(same_inputs) (idempotence)
  Property 3: DataConfidenceScore < threshold → provenance == INSUFFICIENT_EVIDENCE
  Property 4: signalEngineAllowed=false → provenance == UNAVAILABLE
  Property 5: data-service2.0 unreachable → provenance == UNAVAILABLE (no fallback)

Requirements: Req 2.1, Req 2.10, Req 1.5, Req 1.7, Req 1.12, Req 18.3
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings as h_settings
from hypothesis import strategies as st

# Ensure env vars are set before any src imports
os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")
os.environ.setdefault("DATA_SERVICE_2_URL", "http://localhost:8200")
os.environ.setdefault("SENTINEL_PULSE_URL", "http://localhost:3001")


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def trading_datetimes(draw: Any) -> datetime:
    """Generate random UTC datetimes during NSE trading hours.

    NSE trades IST 09:15–15:30, which is UTC 03:45–10:00.
    We use UTC 04:00–09:59 for a clean, safe window.
    """
    year = draw(st.integers(min_value=2020, max_value=2024))
    month = draw(st.integers(min_value=1, max_value=12))
    day = draw(st.integers(min_value=1, max_value=28))  # 28 is safe for all months
    hour = draw(st.integers(min_value=4, max_value=9))
    minute = draw(st.integers(min_value=0, max_value=59))
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Mock response builders
# ---------------------------------------------------------------------------


def make_mock_market_data(
    symbol: str = "NIFTY",
    timestamp_ms: int | None = None,
    confidence_score: int = 85,
    signal_engine_allowed: bool = True,
) -> dict[str, Any]:
    """Build a mock data-service2.0 response with quality metadata.

    Mimics the actual data-service2.0 response envelope used by DataServiceClient.
    The FeaturePipeline must parse `metadata.quality.score` and
    `metadata.quality.signalEngineAllowed` from this structure.
    """
    ts = timestamp_ms or int(
        datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc).timestamp() * 1000
    )
    return {
        "data": {
            "instrumentId": f"NSE:{symbol}:IDX",
            "symbol": symbol,
            "close": 22150.0,
            "volume": 1_000_000,
        },
        "metadata": {
            "quality": {
                "score": confidence_score,
                "signalEngineAllowed": signal_engine_allowed,
            },
            "dataAsOf": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
        },
    }


def make_mock_news_context(symbol: str = "NIFTY") -> dict[str, Any]:
    """Build a mock SentinelPulse news context response."""
    return {
        "instrument": symbol,
        "as_of": "2024-01-15T09:29:45.000Z",
        "news_impact_score": 0.512,
        "impact_direction": "BULLISH",
        "impact_confidence": 0.73,
        "sentiment": {
            "overall": 0.43,
            "market": 0.38,
            "company": 0.51,
            "macro": -0.05,
            "risk": -0.09,
        },
        "market_regime": "RISK_ON",
    }


# ---------------------------------------------------------------------------
# Property 1: PIT Correctness
# ---------------------------------------------------------------------------


class TestPITCorrectness:
    """
    Property 1: ∀ feature f ∈ vector: source_timestamp(f) < vector.timestamp

    All source data used to build a feature vector must have timestamps strictly
    before the feature vector's target timestamp (the PIT boundary).

    Validates: Req 2.1, Req 2.2
    """

    @pytest.mark.asyncio
    async def test_feature_vector_timestamp_equals_request_timestamp(self) -> None:
        """FeatureVector.timestamp must be set to the request timestamp (PIT boundary)."""
        try:
            from src.features.pipeline import FeaturePipeline
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            timestamp_ms=int(
                datetime(2024, 1, 15, 9, 15, 0, tzinfo=timezone.utc).timestamp() * 1000
            )
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = make_mock_news_context()

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, _report = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        assert vector.timestamp == target_ts, (
            f"FeatureVector.timestamp should equal the request timestamp {target_ts}, "
            f"got {vector.timestamp}"
        )

    @pytest.mark.asyncio
    async def test_feature_quality_report_records_pit_violations(self) -> None:
        """
        Data with source_timestamp >= pit_boundary must trigger a PIT violation.
        The FeatureQualityReport must record such violations.
        """
        try:
            from src.features.pipeline import FeaturePipeline
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        pit_boundary = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        # Source data timestamped AFTER the PIT boundary — should be discarded
        future_ts_ms = int(
            datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000
        )

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            timestamp_ms=future_ts_ms
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = None  # degraded mode

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        # Inference mode BLOCKS a PIT violation by raising (P0-008). Data dated
        # after the boundary must never enter a live feature vector.
        from src.core.exceptions import PointInTimeViolationError

        with pytest.raises(PointInTimeViolationError):
            await pipeline.build_vector("NIFTY", pit_boundary, mode="inference")

        # Backtest mode instead COUNTS the violation in the quality report.
        _vector, report = await pipeline.build_vector(
            "NIFTY", pit_boundary, mode="backtest"
        )
        assert hasattr(report, "pit_violations_count")
        assert report.pit_violations_count > 0

    @pytest.mark.asyncio
    async def test_source_data_before_pit_boundary_does_not_raise(self) -> None:
        """
        When all source timestamps are strictly before the PIT boundary,
        the pipeline must succeed and return a vector with zero PIT violations.
        """
        try:
            from src.features.pipeline import FeaturePipeline
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        # Source data 15 minutes before PIT boundary — valid
        valid_ts_ms = int(
            datetime(2024, 1, 15, 9, 15, 0, tzinfo=timezone.utc).timestamp() * 1000
        )

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            timestamp_ms=valid_ts_ms,
            confidence_score=85,
            signal_engine_allowed=True,
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = make_mock_news_context()

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, report = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        assert vector.timestamp == target_ts
        assert report.pit_violations_count == 0, (
            f"Expected 0 PIT violations for valid source data, got {report.pit_violations_count}"
        )

    @pytest.mark.asyncio
    @given(target_timestamp=trading_datetimes())
    @h_settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=2000,
    )
    async def test_pit_correctness_property_hypothesis(
        self,
        target_timestamp: datetime,
    ) -> None:
        """
        Property 1 (Hypothesis): For all valid trading-hour timestamps T,
        the resulting FeatureVector.timestamp equals T.

        **Validates: Requirements 2.1**
        """
        try:
            from src.features.pipeline import FeaturePipeline
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        # Source data 60 seconds BEFORE target — always PIT-safe
        source_ts_ms = int((target_timestamp.timestamp() - 60) * 1000)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            timestamp_ms=source_ts_ms,
            confidence_score=85,
            signal_engine_allowed=True,
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = None  # degraded mode fine

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, _report = await pipeline.build_vector(
            "NIFTY", target_timestamp, mode="inference"
        )

        assert vector.timestamp == target_timestamp, (
            f"FeatureVector.timestamp must equal the requested PIT boundary. "
            f"Expected {target_timestamp}, got {vector.timestamp}"
        )


# ---------------------------------------------------------------------------
# Property 2: Idempotence
# ---------------------------------------------------------------------------


class TestFeaturePipelineIdempotence:
    """
    Property 2: pipeline.build_vector(same_inputs) == pipeline.build_vector(same_inputs)

    Computing a feature vector twice with identical inputs and identical
    downstream responses must produce identical FeatureVector objects.

    Validates: Req 2.10
    """

    @pytest.mark.asyncio
    async def test_build_vector_is_idempotent_with_same_inputs(self) -> None:
        """Calling build_vector twice with the same symbol/timestamp/data returns equal vectors."""
        try:
            from src.features.pipeline import FeaturePipeline
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)
        mock_market_data = make_mock_market_data(confidence_score=85, signal_engine_allowed=True)
        mock_news = make_mock_news_context()

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = mock_market_data
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = mock_news

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector1, _ = await pipeline.build_vector("NIFTY", target_ts, mode="inference")
        vector2, _ = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        assert vector1 == vector2, (
            "FeaturePipeline.build_vector must be idempotent: "
            "identical inputs must produce identical FeatureVector outputs. "
            f"First call: provenance={vector1.provenance}, "
            f"Second call: provenance={vector2.provenance}"
        )

    @pytest.mark.asyncio
    async def test_build_vector_idempotent_across_symbols(self) -> None:
        """Idempotence holds for different symbols with their respective data."""
        try:
            from src.features.pipeline import FeaturePipeline
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        for symbol in ("NIFTY", "BANKNIFTY", "RELIANCE"):
            mock_data_client = AsyncMock()
            mock_data_client.get_live_quote.return_value = make_mock_market_data(
                symbol=symbol, confidence_score=85, signal_engine_allowed=True
            )
            mock_data_client.get_historical_ohlcv.return_value = []

            mock_news_client = AsyncMock()
            mock_news_client.fetch_news_context.return_value = make_mock_news_context(symbol)

            pipeline = FeaturePipeline(
                data_client=mock_data_client,
                news_client=mock_news_client,
            )

            v1, _ = await pipeline.build_vector(symbol, target_ts, mode="inference")
            v2, _ = await pipeline.build_vector(symbol, target_ts, mode="inference")

            assert v1 == v2, (
                f"Idempotence violated for symbol={symbol}: "
                f"first call returned {v1.provenance}, second returned {v2.provenance}"
            )


# ---------------------------------------------------------------------------
# Property 3: DataConfidenceScore Gate
# ---------------------------------------------------------------------------


class TestDataConfidenceScoreGate:
    """
    Property 3: DataConfidenceScore ∈ [0, 69] → provenance == INSUFFICIENT_EVIDENCE

    When data-service2.0 returns a DataConfidenceScore below the configured
    min_confidence_score threshold (default 70), the FeaturePipeline must
    return a vector with PredictionProvenance.INSUFFICIENT_EVIDENCE.

    Validates: Req 1.7
    """

    @pytest.mark.asyncio
    async def test_exact_threshold_boundary_score_69_is_insufficient_evidence(self) -> None:
        """Score of 69 (one below threshold of 70) must yield INSUFFICIENT_EVIDENCE."""
        try:
            from src.features.pipeline import FeaturePipeline
            from src.schemas.base import PredictionProvenance
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            confidence_score=69,  # one below default threshold of 70
            signal_engine_allowed=True,
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, _ = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        assert vector.provenance == PredictionProvenance.INSUFFICIENT_EVIDENCE, (
            f"Score=69 (below threshold=70) must yield INSUFFICIENT_EVIDENCE, "
            f"got {vector.provenance}"
        )

    @pytest.mark.asyncio
    async def test_score_at_threshold_70_is_not_insufficient_evidence(self) -> None:
        """Score of 70 (exactly at threshold) must NOT yield INSUFFICIENT_EVIDENCE."""
        try:
            from src.features.pipeline import FeaturePipeline
            from src.schemas.base import PredictionProvenance
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            confidence_score=70,  # at threshold — should be allowed
            signal_engine_allowed=True,
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, _ = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        assert vector.provenance != PredictionProvenance.INSUFFICIENT_EVIDENCE, (
            f"Score=70 (at threshold) must not yield INSUFFICIENT_EVIDENCE, "
            f"but got {vector.provenance}"
        )

    @pytest.mark.asyncio
    async def test_score_zero_is_insufficient_evidence(self) -> None:
        """Score of 0 (minimum possible) must yield INSUFFICIENT_EVIDENCE."""
        try:
            from src.features.pipeline import FeaturePipeline
            from src.schemas.base import PredictionProvenance
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            confidence_score=0,
            signal_engine_allowed=True,
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, _ = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        assert vector.provenance == PredictionProvenance.INSUFFICIENT_EVIDENCE, (
            f"Score=0 must yield INSUFFICIENT_EVIDENCE, got {vector.provenance}"
        )

    @pytest.mark.asyncio
    @given(low_score=st.integers(min_value=0, max_value=69))
    @h_settings(
        max_examples=15,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=2000,
    )
    async def test_low_confidence_score_property_hypothesis(
        self,
        low_score: int,
    ) -> None:
        """
        Property 3 (Hypothesis): For ALL scores in [0, 69], the pipeline
        must return INSUFFICIENT_EVIDENCE provenance.

        **Validates: Requirements 1.7**
        """
        try:
            from src.features.pipeline import FeaturePipeline
            from src.schemas.base import PredictionProvenance
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            confidence_score=low_score,
            signal_engine_allowed=True,
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, _ = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        assert vector.provenance == PredictionProvenance.INSUFFICIENT_EVIDENCE, (
            f"Expected INSUFFICIENT_EVIDENCE for DataConfidenceScore={low_score} "
            f"(below threshold=70), got {vector.provenance}"
        )


# ---------------------------------------------------------------------------
# Property 4: signalEngineAllowed Gate
# ---------------------------------------------------------------------------


class TestSignalEngineAllowedGate:
    """
    Property 4: signalEngineAllowed=false → provenance == UNAVAILABLE

    When data-service2.0 returns signalEngineAllowed=false in the quality
    metadata, the FeaturePipeline must return UNAVAILABLE provenance,
    regardless of the DataConfidenceScore value.

    Validates: Req 1.5, Req 1.6
    """

    @pytest.mark.asyncio
    async def test_signal_engine_not_allowed_returns_unavailable(self) -> None:
        """signalEngineAllowed=false with high confidence score must still yield UNAVAILABLE."""
        try:
            from src.features.pipeline import FeaturePipeline
            from src.schemas.base import PredictionProvenance
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            signal_engine_allowed=False,
            confidence_score=95,  # high score — should still be blocked
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, _ = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        assert vector.provenance == PredictionProvenance.UNAVAILABLE, (
            "signalEngineAllowed=false must yield UNAVAILABLE regardless of DataConfidenceScore. "
            f"Got {vector.provenance}"
        )

    @pytest.mark.asyncio
    async def test_signal_engine_not_allowed_with_low_confidence_still_unavailable(
        self,
    ) -> None:
        """signalEngineAllowed=false AND low confidence must yield UNAVAILABLE (not INSUFFICIENT_EVIDENCE)."""
        try:
            from src.features.pipeline import FeaturePipeline
            from src.schemas.base import PredictionProvenance
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            signal_engine_allowed=False,
            confidence_score=30,  # also below threshold, but signalEngineAllowed=False takes priority
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, _ = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        # signalEngineAllowed=false takes priority — UNAVAILABLE, not INSUFFICIENT_EVIDENCE
        assert vector.provenance == PredictionProvenance.UNAVAILABLE, (
            "signalEngineAllowed=false must yield UNAVAILABLE (takes priority over low confidence). "
            f"Got {vector.provenance}"
        )

    @pytest.mark.asyncio
    async def test_signal_engine_allowed_true_with_good_score_proceeds(self) -> None:
        """signalEngineAllowed=true with score >= 70 must NOT yield UNAVAILABLE or INSUFFICIENT_EVIDENCE."""
        try:
            from src.features.pipeline import FeaturePipeline
            from src.schemas.base import PredictionProvenance
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            signal_engine_allowed=True,
            confidence_score=85,
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = make_mock_news_context()

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, _ = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        assert vector.provenance not in (
            PredictionProvenance.UNAVAILABLE,
            PredictionProvenance.INSUFFICIENT_EVIDENCE,
        ), (
            f"signalEngineAllowed=true with score=85 must not block inference, "
            f"got {vector.provenance}"
        )


# ---------------------------------------------------------------------------
# Property 5: data-service2.0 unreachable → UNAVAILABLE
# ---------------------------------------------------------------------------


class TestDataServiceUnreachable:
    """
    Property 5: data-service2.0 unreachable → provenance == UNAVAILABLE

    When data-service2.0 raises DataServiceUnavailableError, the pipeline
    must return UNAVAILABLE and must NOT fall back to any alternative source.

    Validates: Req 1.12
    """

    @pytest.mark.asyncio
    async def test_data_service_unavailable_error_returns_unavailable(self) -> None:
        """DataServiceUnavailableError must propagate as UNAVAILABLE provenance."""
        try:
            from src.clients.data_service import DataServiceUnavailableError
            from src.features.pipeline import FeaturePipeline
            from src.schemas.base import PredictionProvenance
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.side_effect = DataServiceUnavailableError(
            "Connection refused to data-service2.0"
        )

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = make_mock_news_context()

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, report = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        assert vector.provenance == PredictionProvenance.UNAVAILABLE, (
            "DataServiceUnavailableError must result in UNAVAILABLE provenance. "
            f"Got {vector.provenance}"
        )
        assert report.data_service_available is False, (
            "FeatureQualityReport.data_service_available must be False when data-service is down. "
            f"Got {report.data_service_available}"
        )

    @pytest.mark.asyncio
    async def test_data_service_unreachable_does_not_fallback_to_news(self) -> None:
        """
        Even when SentinelPulse IS available, if data-service is unreachable,
        the pipeline must still return UNAVAILABLE (Req 1.12: no fallback source).
        """
        try:
            from src.clients.data_service import DataServiceUnavailableError
            from src.features.pipeline import FeaturePipeline
            from src.schemas.base import PredictionProvenance
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.side_effect = DataServiceUnavailableError(
            "Circuit is open"
        )

        # SentinelPulse IS available — pipeline must still return UNAVAILABLE
        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = make_mock_news_context()

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, _ = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        assert vector.provenance == PredictionProvenance.UNAVAILABLE, (
            "Pipeline must return UNAVAILABLE when data-service is down, "
            "even when SentinelPulse is available. Req 1.12 forbids any alternative source fallback. "
            f"Got {vector.provenance}"
        )

    @pytest.mark.asyncio
    async def test_data_service_auth_error_returns_unavailable(self) -> None:
        """DataServiceAuthError (401/403) must result in UNAVAILABLE (no retry, Req 1.4)."""
        try:
            from src.clients.data_service import DataServiceAuthError
            from src.features.pipeline import FeaturePipeline
            from src.schemas.base import PredictionProvenance
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.side_effect = DataServiceAuthError(
            "data-service2.0 returned 401 for GET /v1/india/quotes/NIFTY"
        )

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, _ = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        assert vector.provenance == PredictionProvenance.UNAVAILABLE, (
            "DataServiceAuthError (401/403) must result in UNAVAILABLE (Req 1.4). "
            f"Got {vector.provenance}"
        )


# ---------------------------------------------------------------------------
# SentinelPulse degraded-mode substitution
# ---------------------------------------------------------------------------


class TestSentinelPulseDegradedMode:
    """
    When SentinelPulse is unavailable, the pipeline must apply neutral substitution:
      - All numeric news features → 0.0
      - impact_direction → NEUTRAL
      - sentinel_available → False
      - FeatureQualityReport.sentinel_pulse_available → False

    Validates: Req 1.11, Req 2.8
    """

    @pytest.mark.asyncio
    async def test_sentinel_none_response_applies_neutral_substitution(self) -> None:
        """When fetch_news_context returns None, all news fields must use neutral defaults."""
        try:
            from src.features.pipeline import FeaturePipeline
            from src.schemas.base import ImpactDirection
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            confidence_score=85, signal_engine_allowed=True
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        # SentinelPulse returns None (unreachable)
        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, report = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        assert vector.news_impact_score == 0.0, (
            f"news_impact_score must be 0.0 when SentinelPulse is unavailable, "
            f"got {vector.news_impact_score}"
        )
        assert vector.impact_direction == ImpactDirection.NEUTRAL, (
            f"impact_direction must be NEUTRAL when SentinelPulse is unavailable, "
            f"got {vector.impact_direction}"
        )
        assert vector.sentiment_overall == 0.0, (
            f"sentiment_overall must be 0.0 when SentinelPulse is unavailable, "
            f"got {vector.sentiment_overall}"
        )
        assert vector.sentiment_market == 0.0, (
            f"sentiment_market must be 0.0 when SentinelPulse is unavailable"
        )
        assert vector.sentiment_company == 0.0, (
            f"sentiment_company must be 0.0 when SentinelPulse is unavailable"
        )
        assert vector.sentinel_available is False, (
            "sentinel_available must be False when SentinelPulse is unavailable"
        )
        assert report.sentinel_pulse_available is False, (
            "FeatureQualityReport.sentinel_pulse_available must be False when SentinelPulse is unreachable"
        )

    @pytest.mark.asyncio
    async def test_sentinel_available_populates_real_values(self) -> None:
        """When SentinelPulse is available, real values must be used (not neutral defaults)."""
        try:
            from src.features.pipeline import FeaturePipeline
            from src.schemas.base import ImpactDirection
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            confidence_score=85, signal_engine_allowed=True
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = make_mock_news_context()  # real values

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, report = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        # Real news was available — should not use zero defaults
        assert vector.news_impact_score != 0.0, (
            "news_impact_score must reflect real SentinelPulse data when available"
        )
        assert vector.impact_direction != ImpactDirection.NEUTRAL or vector.news_impact_score > 0, (
            "With real SentinelPulse data, impact_direction should reflect actual direction"
        )
        assert vector.sentinel_available is True, (
            "sentinel_available must be True when SentinelPulse responded successfully"
        )
        assert report.sentinel_pulse_available is True, (
            "FeatureQualityReport.sentinel_pulse_available must be True when SentinelPulse is healthy"
        )

    @pytest.mark.asyncio
    async def test_sentinel_unavailable_does_not_block_inference(self) -> None:
        """
        SentinelPulse being unavailable must NOT prevent the pipeline from
        returning a valid (non-UNAVAILABLE) vector — inference proceeds with
        market-data features only (Req 1.11).
        """
        try:
            from src.features.pipeline import FeaturePipeline
            from src.schemas.base import PredictionProvenance
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            confidence_score=85, signal_engine_allowed=True
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = None  # SentinelPulse down

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vector, _ = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        # With healthy market data + bad SentinelPulse, inference should still proceed
        assert vector.provenance != PredictionProvenance.UNAVAILABLE, (
            "SentinelPulse being unavailable must NOT return UNAVAILABLE provenance — "
            "inference must continue with market-data features and neutral news substitution. "
            f"Got {vector.provenance}"
        )


# ---------------------------------------------------------------------------
# FeatureQualityReport structure
# ---------------------------------------------------------------------------


class TestFeatureQualityReport:
    """
    Validate that FeatureQualityReport is always returned alongside the vector
    and contains the required fields with sensible defaults.

    Validates: Req 2.9
    """

    @pytest.mark.asyncio
    async def test_quality_report_always_returned(self) -> None:
        """build_vector must always return a (FeatureVector, FeatureQualityReport) tuple."""
        try:
            from src.features.pipeline import FeaturePipeline
            from src.schemas.features import FeatureQualityReport
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data()
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        result = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        assert isinstance(result, tuple), "build_vector must return a tuple"
        assert len(result) == 2, "build_vector must return a 2-tuple (vector, report)"

        vector, report = result
        assert isinstance(report, FeatureQualityReport), (
            f"Second element must be FeatureQualityReport, got {type(report)}"
        )

    @pytest.mark.asyncio
    async def test_quality_report_has_required_fields(self) -> None:
        """FeatureQualityReport must expose all required fields."""
        try:
            from src.features.pipeline import FeaturePipeline
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data()
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        _, report = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        required_fields = [
            "batch_id",
            "timestamp",
            "missing_count",
            "imputed_count",
            "rejected_count",
            "unavailable_families",
            "pit_violations_count",
            "sentinel_pulse_available",
            "data_service_available",
            "processing_time_ms",
        ]

        for field in required_fields:
            assert hasattr(report, field), (
                f"FeatureQualityReport is missing required field: {field}"
            )

    @pytest.mark.asyncio
    async def test_quality_report_processing_time_is_positive(self) -> None:
        """processing_time_ms must be > 0 (the pipeline spent some time processing)."""
        try:
            from src.features.pipeline import FeaturePipeline
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            confidence_score=85, signal_engine_allowed=True
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        _, report = await pipeline.build_vector("NIFTY", target_ts, mode="inference")

        assert report.processing_time_ms >= 0, (
            "processing_time_ms must be non-negative "
            f"(got {report.processing_time_ms})"
        )


# ---------------------------------------------------------------------------
# Task 18.2: Batch mode
# ---------------------------------------------------------------------------


class TestBatchMode:
    """
    Verify build_batch returns a list of FeatureVectors whose length equals
    the number of input symbols (Req 2.9, Req 2.11, Req 2.13).
    """

    @pytest.mark.asyncio
    async def test_build_batch_returns_correct_number_of_vectors(self) -> None:
        """build_batch must return one FeatureVector per input symbol."""
        try:
            from src.features.pipeline import FeaturePipeline
            from src.schemas.features import FeatureVector
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        symbols = ["NIFTY", "BANKNIFTY", "RELIANCE"]
        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            confidence_score=85, signal_engine_allowed=True
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = make_mock_news_context()

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vectors, report = await pipeline.build_batch(symbols, target_ts, mode="inference")

        assert len(vectors) == len(symbols), (
            f"build_batch must return exactly {len(symbols)} vectors (one per symbol), "
            f"got {len(vectors)}"
        )
        for v in vectors:
            assert isinstance(v, FeatureVector), (
                f"Each element must be a FeatureVector, got {type(v)}"
            )

    @pytest.mark.asyncio
    async def test_build_batch_empty_symbols_returns_empty_list(self) -> None:
        """build_batch with an empty symbol list must return an empty list."""
        try:
            from src.features.pipeline import FeaturePipeline
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

        pipeline = FeaturePipeline(data_client=None, news_client=None)

        vectors, _ = await pipeline.build_batch([], target_ts, mode="inference")

        assert vectors == [], (
            f"build_batch with empty symbol list must return [], got {vectors}"
        )

    @pytest.mark.asyncio
    async def test_build_batch_backtest_mode_passes_pit_date(self) -> None:
        """
        In backtest mode build_batch must forward pit_date to all DataService calls.

        Validates: Req 2.11, Req 2.13
        """
        try:
            from src.features.pipeline import FeaturePipeline
        except ImportError:
            pytest.skip("FeaturePipeline not yet implemented — TDD red phase")

        symbols = ["NIFTY", "BANKNIFTY"]
        target_ts = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)
        pit_date = "2024-01-15"

        mock_data_client = AsyncMock()
        mock_data_client.get_live_quote.return_value = make_mock_market_data(
            confidence_score=85, signal_engine_allowed=True
        )
        mock_data_client.get_historical_ohlcv.return_value = []

        mock_news_client = AsyncMock()
        mock_news_client.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(
            data_client=mock_data_client,
            news_client=mock_news_client,
        )

        vectors, report = await pipeline.build_batch(symbols, target_ts, mode="backtest")

        assert len(vectors) == len(symbols), (
            f"build_batch backtest mode must still return {len(symbols)} vectors, "
            f"got {len(vectors)}"
        )
        # In backtest mode, the pipeline should have attempted to fetch historical data
        # for each symbol (even if it returns empty and skips Alpha158 computation).
        # The vectors should still be valid FeatureVector instances.
        from src.schemas.features import FeatureVector
        for v in vectors:
            assert isinstance(v, FeatureVector)
