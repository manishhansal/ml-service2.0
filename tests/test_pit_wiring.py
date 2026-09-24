"""
test_pit_wiring.py — Phase A regression tests.

Verifies:
1. MetaOutput carries the full PIT timestamp chain (P0-005).
2. MetaDecisionEngine threads real data_quality into abstention (P0-004).
3. LookAheadGuard is invoked by FeaturePipeline and blocks future data in
   inference mode (P0-008).
4. FeatureVector carries data_as_of / news_as_of provenance.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core.exceptions import PointInTimeViolationError
from src.features.pipeline import FeaturePipeline
from src.meta.engine import MetaDecisionEngine
from src.schemas.base import PredictionProvenance
from src.schemas.meta import FeedbackRecord, MetaOutput, OutcomeResolution

UTC = timezone.utc


# ─── P0-005: PIT timestamp chain on MetaOutput ──────────────────────────────


def test_metaoutput_has_pit_timestamp_fields():
    ts = datetime(2025, 1, 2, 9, 30, tzinfo=UTC)
    feat = datetime(2025, 1, 2, 9, 29, tzinfo=UTC)
    out = MetaOutput(
        action="BUY",
        confidence=0.6,
        uncertainty=0.4,
        agreement=0.6,
        agreement_ratio=0.6,
        prediction_timestamp=ts,
        feature_as_of=feat,
        data_as_of=feat,
        news_as_of=feat,
        expires_at=ts + timedelta(minutes=15),
    )
    assert out.prediction_timestamp == ts
    assert out.feature_as_of <= out.prediction_timestamp
    assert out.news_as_of <= out.prediction_timestamp
    assert out.expires_at > out.prediction_timestamp
    assert out.signal_id  # auto-generated


def test_metaoutput_signal_id_is_unique():
    a = MetaOutput(action="WAIT", confidence=0.0, uncertainty=1.0,
                   agreement=0.0, agreement_ratio=0.0)
    b = MetaOutput(action="WAIT", confidence=0.0, uncertainty=1.0,
                   agreement=0.0, agreement_ratio=0.0)
    assert a.signal_id != b.signal_id


# ─── P0-004: real data_quality threaded into abstention ─────────────────────


def _three_buy_outputs():
    return [
        {"model_id": "m1", "action": "BUY", "confidence": 0.8, "direction": 1,
         "provenance": PredictionProvenance.TRAINED_MODEL.value},
        {"model_id": "m2", "action": "BUY", "confidence": 0.75, "direction": 1,
         "provenance": PredictionProvenance.TRAINED_MODEL.value},
        {"model_id": "m3", "action": "BUY", "confidence": 0.7, "direction": 1,
         "provenance": PredictionProvenance.TRAINED_MODEL.value},
    ]


def test_low_data_quality_forces_abstention():
    engine = MetaDecisionEngine()
    out = engine.decide(
        model_outputs=_three_buy_outputs(),
        symbol="NIFTY",
        regime="bull",
        data_quality=0.30,  # below the 0.6 abstention threshold
    )
    assert out.action == "NO_TRADE"
    assert out.abstention is True
    assert "LOW_DATA_QUALITY" in out.reason_codes
    assert out.data_confidence_score == 30


def test_high_data_quality_allows_trade():
    engine = MetaDecisionEngine()
    out = engine.decide(
        model_outputs=_three_buy_outputs(),
        symbol="NIFTY",
        regime="bull",
        data_quality=0.95,
    )
    assert "LOW_DATA_QUALITY" not in out.reason_codes
    assert out.data_confidence_score == 95


def test_data_quality_default_backward_compatible():
    # Legacy callers that pass no data_quality still work (defaults to 1.0).
    engine = MetaDecisionEngine()
    out = engine.decide(model_outputs=_three_buy_outputs(), symbol="NIFTY", regime="bull")
    assert "LOW_DATA_QUALITY" not in out.reason_codes


# ─── P0-008: LookAheadGuard wired into FeaturePipeline ──────────────────────


class _FutureDataClient:
    """Data client that returns data dated AFTER the PIT boundary."""

    async def get_live_quote(self, symbol: str):
        future = datetime(2025, 6, 1, tzinfo=UTC)
        return {
            "data": {"close": 100.0},
            "metadata": {
                "quality": {"score": 90, "signalEngineAllowed": True},
                "dataAsOf": future.isoformat(),
            },
        }

    async def get_historical_ohlcv(self, *a, **k):
        return []


@pytest.mark.asyncio
async def test_inference_mode_blocks_future_data():
    pipeline = FeaturePipeline(data_client=_FutureDataClient(), news_client=None)
    boundary = datetime(2025, 1, 2, tzinfo=UTC)
    with pytest.raises(PointInTimeViolationError):
        await pipeline.build_vector("NIFTY", boundary, mode="inference")


@pytest.mark.asyncio
async def test_backtest_mode_counts_but_does_not_block():
    pipeline = FeaturePipeline(data_client=_FutureDataClient(), news_client=None)
    boundary = datetime(2025, 1, 2, tzinfo=UTC)
    vector, report = await pipeline.build_vector("NIFTY", boundary, mode="backtest")
    assert report.pit_violations_count >= 1
    assert vector.pit_validated is False


class _PastDataClient:
    async def get_live_quote(self, symbol: str):
        past = datetime(2025, 1, 1, tzinfo=UTC)
        return {
            "data": {"close": 100.0},
            "metadata": {
                "quality": {"score": 90, "signalEngineAllowed": True},
                "dataAsOf": past.isoformat(),
            },
        }

    async def get_historical_ohlcv(self, *a, **k):
        return []


@pytest.mark.asyncio
async def test_past_data_passes_pit_and_records_data_as_of():
    pipeline = FeaturePipeline(data_client=_PastDataClient(), news_client=None)
    boundary = datetime(2025, 1, 2, tzinfo=UTC)
    vector, report = await pipeline.build_vector("NIFTY", boundary, mode="inference")
    assert report.pit_violations_count == 0
    assert vector.data_as_of == datetime(2025, 1, 1, tzinfo=UTC)
    assert vector.pit_validated is True


# ─── Feedback / outcome schemas exist and validate ──────────────────────────


def test_feedback_record_schema():
    rec = FeedbackRecord(
        signal_id="sig-1",
        symbol="NIFTY",
        prediction_timestamp=datetime(2025, 1, 2, tzinfo=UTC),
        action="BUY",
        entry_price=100.0,
        exit_price=101.2,
        realized_return=0.012,
        exit_reason="TARGET_HIT",
    )
    assert rec.exit_reason == "TARGET_HIT"
    assert rec.realized_return == pytest.approx(0.012)


def test_outcome_resolution_schema():
    res = OutcomeResolution(
        signal_id="sig-1",
        symbol="NIFTY",
        resolved=True,
        exit_reason="STOP_HIT",
        entry_price=100.0,
        exit_price=98.5,
        realized_return=-0.015,
        mae=-0.02,
        mfe=0.005,
    )
    assert res.resolved is True
    assert res.exit_reason == "STOP_HIT"
