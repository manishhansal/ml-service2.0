"""
src.data — wire-format contracts for upstream data sources.

Re-exports for convenient single-import access:

    from src.data.contracts import OHLCVBar, DataServiceResponse, SentinelPulseResponse
"""
from src.data.contracts import (
    DataQualityMetadata,
    DataServiceResponse,
    OHLCVBar,
    SentinelNewsContext,
    SentinelPulseResponse,
    SentinelSentimentBreakdown,
)

__all__ = [
    "OHLCVBar",
    "DataQualityMetadata",
    "DataServiceResponse",
    "SentinelSentimentBreakdown",
    "SentinelNewsContext",
    "SentinelPulseResponse",
]
