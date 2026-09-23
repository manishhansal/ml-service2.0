#!/usr/bin/env bash
# Generate Python gRPC stubs from market_data.proto
# Run from the repository root: bash protos/generate_stubs.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_FILE="$SCRIPT_DIR/market_data.proto"
OUT_DIR="$REPO_ROOT/src/clients"

if [[ ! -f "$PROTO_FILE" ]]; then
    echo "ERROR: $PROTO_FILE not found. Cannot generate stubs." >&2
    exit 1
fi

# Prefer python3 if python is not found
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null)}"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: No Python interpreter found. Install python3 and grpcio-tools." >&2
    exit 1
fi

echo "Using Python: $PYTHON_BIN"
echo "Generating gRPC stubs from $PROTO_FILE → $OUT_DIR"

"$PYTHON_BIN" -m grpc_tools.protoc \
    --proto_path="$SCRIPT_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    --pyi_out="$OUT_DIR" \
    "$PROTO_FILE"

# Patch the generated _grpc.py to use a package-relative import so the stub
# works when imported as src.clients.market_data_pb2_grpc (the normal case).
# The raw grpc_tools output writes a bare `import market_data_pb2` which fails
# unless the CWD happens to be src/clients.
GRPC_STUB="$OUT_DIR/market_data_pb2_grpc.py"
if [[ -f "$GRPC_STUB" ]]; then
    # Replace the bare import with a try/except that handles both contexts
    sed -i.bak \
        's/^import market_data_pb2 as market__data__pb2$/try:\n    from src.clients import market_data_pb2 as market__data__pb2  # type: ignore[import]\nexcept ImportError:\n    import market_data_pb2 as market__data__pb2  # type: ignore[import]  # noqa: F401/' \
        "$GRPC_STUB"
    rm -f "${GRPC_STUB}.bak"
    echo "Patched import in $GRPC_STUB"
fi

echo "Stubs written to $OUT_DIR:"
ls -1 "$OUT_DIR"/market_data_pb2*.py 2>/dev/null || true
echo "Done."
