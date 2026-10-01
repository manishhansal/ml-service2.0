#!/usr/bin/env bash
# train_after_backfill.sh
# Waits until the 5m backfill is ≥275/289 symbols complete, then
# immediately launches train_expanded_features.py with full intraday features.
# Usage: bash scripts/train_after_backfill.sh
set -e
cd "$(dirname "$0")/.."

INTRADAY_DIR="data/5m/5m"
TARGET_COUNT=275

echo "Waiting for 5m backfill to reach ${TARGET_COUNT} symbols..."
while true; do
    count=$(ls ${INTRADAY_DIR}/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    echo "  $(date +%H:%M:%S)  5m parquets: ${count}/${TARGET_COUNT}"
    if [ "${count}" -ge "${TARGET_COUNT}" ]; then
        echo "  Threshold reached. Starting training in 10 seconds..."
        sleep 10
        break
    fi
    sleep 30
done

echo ""
echo "========================================"
echo "  LAUNCHING FULL INTRADAY TRAINING"
echo "========================================"
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/train_expanded_features.py 2>&1
