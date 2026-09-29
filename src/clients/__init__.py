"""
src.clients — External service clients package.

Active clients:
    - DataServiceClient  — REST/HTTP to data-service2.0 (primary market data)
    - SentinelPulseClient — REST/HTTP to SentinelPulse (news/sentiment)

Deprecated (P3-007):
    - GRPCMarketDataClient — gRPC streaming client. Marked dead code.
      The gRPC transport was never deployed in production; all live data flows
      through data-service2.0 REST APIs. The protobuf stub files
      (market_data_pb2.py, market_data_pb2_grpc.py) and grpc_client.py are
      retained for reference but STUB_AVAILABLE is False in all current
      environments. These will be removed in the next major cleanup cycle.
"""
from __future__ import annotations

from src.clients.sentinel_pulse import SentinelPulseClient  # noqa: F401

STUB_AVAILABLE: bool

try:
    from src.clients.market_data_pb2 import (  # noqa: F401
        FeatureRequest,
        FeatureResponse,
        OHLCVBar,
        TickStreamRequest,
        TickStreamResponse,
    )
    from src.clients.market_data_pb2_grpc import (  # noqa: F401
        MarketDataFeederStub,
    )
    STUB_AVAILABLE = True
except Exception:  # pragma: no cover  # noqa: BLE001
    # grpcio not installed, or stubs not yet generated — degrade gracefully.
    # This is the expected state in all current deployments (P3-007: gRPC dead code).
    STUB_AVAILABLE = False

from src.clients.data_service import (  # noqa: F401
    DataServiceAuthError,
    DataServiceClient,
    DataServiceUnavailableError,
    LowDataConfidenceError,
    SignalEngineNotAllowedError,
)

# P3-007: GRPCMarketDataClient is dead code — never deployed in production.
# Import is guarded so the package loads cleanly without grpcio.
try:
    from src.clients.grpc_client import (  # noqa: F401  # pragma: no cover
        GRPCMarketDataClient,
        GRPCStreamError,
    )
except Exception:  # pragma: no cover  # noqa: BLE001
    pass  # grpcio not installed — expected in all current environments

__all__ = [
    "STUB_AVAILABLE",
    "SentinelPulseClient",
    # REST client
    "DataServiceClient",
    # REST client exceptions
    "DataServiceAuthError",
    "DataServiceUnavailableError",
    "LowDataConfidenceError",
    "SignalEngineNotAllowedError",
]
