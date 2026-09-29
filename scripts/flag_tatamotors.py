#!/usr/bin/env python3
import json
from pathlib import Path

issue = {
    "symbol": "TATAMOTORS",
    "finding": "DATA_QUALITY_MISMATCH",
    "hist_close_last": 295.0,
    "live_ltp": 961.3,
    "discrepancy_pct": 226.0,
    "probable_cause": "Upstox historical data contains TATAMOTORS-DVR prices (295-303) while Angel One live quotes regular TATAMOTORS (961). Instrument mapping mismatch.",
    "action": "EXCLUDE from forward paper P&L and model validation",
    "confirmed_2026_09_28": True,
}
flag_path = Path('/Users/manishkumar/Desktop/ml-service2.0/artifacts/data_quality_flags.json')
flags = []
if flag_path.exists():
    flags = json.loads(flag_path.read_text())
# Remove duplicates
flags = [f for f in flags if f.get("symbol") != "TATAMOTORS"]
flags.append(issue)
flag_path.write_text(json.dumps(flags, indent=2))
print(f"Data quality flag created: {flag_path}")
print(json.dumps(issue, indent=2))
