# AUTO-GENERATED — do not edit manually
# This module exposes the gRPC client stubs generated from protos/market_data.proto
# by grpcio-tools.  Run `bash protos/generate_stubs.sh` from the repo root to
# regenerate after editing the proto file.
#
# STUB_AVAILABLE is True when grpcio is installed and the generated protobuf
# classes loaded successfully.  Code that needs gRPC functionality should guard
# on this flag and fall back to the REST client when it is False.
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
    # grpcio not installed, or stubs not yet generated — degrade gracefully
    STUB_AVAILABLE = False

from src.clients.data_service import (  # noqa: F401
    DataServiceAuthError,
    DataServiceClient,
    DataServiceUnavailableError,
    LowDataConfidenceError,
    SignalEngineNotAllowedError,
)
from src.clients.grpc_client import (  # noqa: F401
    GRPCMarketDataClient,
    GRPCStreamError,
)

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
    # gRPC streaming client
    "GRPCMarketDataClient",
    "GRPCStreamError",
]
