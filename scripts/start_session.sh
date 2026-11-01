#!/bin/bash
# start_session.sh — Start the daily trading session: refresh data, run autorun.
#
# Run this once each morning before 09:15 IST:
#   ./scripts/start_session.sh
#
# What it does:
#   1. Backfills yesterday's parquet data (if Upstox CDN has it)
#   2. Starts autorun_till_close.py (runs until 15:30 IST)

set -e
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python"
IST_OFFSET="+05:30"

echo "======================================================"
echo "  AlphaForge Daily Session — $(date '+%Y-%m-%d %H:%M IST')"
echo "======================================================"
echo ""

# ── Step 1: Backfill yesterday's close (if not already done) ──────────────
YESTERDAY=$(python3 -c "
from datetime import datetime, timezone, timedelta
IST = timezone(timedelta(hours=5, minutes=30))
now = datetime.now(IST)
# Walk back to last trading day
d = now.date() - timedelta(days=1)
while d.weekday() >= 5:
    d -= timedelta(days=1)
print(d)
")
echo "Checking parquet data for $YESTERDAY..."
LATEST=$(${PYTHON} -c "
import pandas as pd
from pathlib import Path
from datetime import timezone, timedelta
IST = timezone(timedelta(hours=5, minutes=30))
pf = Path('data/1d/1d/NIFTY.parquet')
df = pd.read_parquet(pf, columns=['close'])
if df.index.tz is None: df.index = df.index.tz_localize('UTC')
print(df.index[-1].astimezone(IST).date())
" 2>/dev/null)

if [ "$LATEST" != "$YESTERDAY" ]; then
    echo "Fetching $YESTERDAY close data..."
    ${PYTHON} scripts/refresh_parquets.py --date "$YESTERDAY" 2>&1 | grep -E "Updated|No-data|Error"
    # Clear SMC scanner cache so it uses fresh data at open
    REDIS=$(docker ps --format "{{.Names}}" | grep ml-service20-redis 2>/dev/null)
    if [ -n "$REDIS" ]; then
        docker exec "$REDIS" redis-cli DEL smc:active smc:last_scan_meta >/dev/null 2>&1
        echo "SMC scanner cache cleared"
    fi
else
    echo "Data already at $LATEST — no backfill needed"
fi
echo ""

# ── Step 2: Start autorun ──────────────────────────────────────────────────
echo "Starting autorun_till_close.py..."
echo "  Market: 09:15 → 15:30 IST"
echo "  Logs:   artifacts/live_session/autorun_log.jsonl"
echo ""
PYTHONPATH=. ${PYTHON} scripts/autorun_till_close.py
