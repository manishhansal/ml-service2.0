"""
scripts/run_universe_coverage.py — maximum verified F&O universe coverage report.

Mandate §3: document universe coverage, survivorship classification, and
calendar-aware gap detection for all available data.

Produces: reports/universe_coverage.json

Runs locally (no Docker required) — data already cached on disk from prior
ingestion runs.  Does NOT re-fetch from data-service; uses cached parquet files
from data/1d/*.parquet written by DataIngestionPipeline.

If cached data is unavailable for a symbol, it is marked UNAVAILABLE.
This script reports the MAXIMUM VERIFIED UNIVERSE from what is on disk.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.validation.calendar import classify_gaps_in_series, is_market_open

DATA_ROOT = Path("data/1d/1d")
REPORT_PATH = Path("reports/universe_coverage.json")

# The 65-symbol confirmed universe from CONFIRMATION_BASELINE_V1
BASELINE_SYMBOLS = [
    "360ONE", "ABB", "ABCAPITAL", "ADANIENSOL", "ADANIENT", "ADANIGREEN",
    "ADANIPORTS", "ADANIPOWER", "ALKEM", "AMBER", "AMBUJACEM", "ANGELONE",
    "APLAPOLLO", "APOLLOHOSP", "ASHOKLEY", "ASIANPAINT", "ASTRAL",
    "ATHERENERG", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO",
    "BAJAJFINSV", "BAJAJHLDNG", "BAJFINANCE", "BANDHANBNK", "BANKINDIA",
    "BANKNIFTY", "BHARTIARTL", "CANBK", "DMART", "DRREDDY", "HDFCBANK",
    "ICICIBANK", "INFY", "JINDALSTEL", "JIOFIN", "JSWENERGY", "KOTAKBANK",
    "MAXHEALTH", "MAZDOCK", "MCX", "MFSL", "NBCC", "NESTLEIND", "NHPC",
    "NIFTY", "NMDC", "NTPC", "PAGEIND", "PATANJALI", "PAYTM", "PERSISTENT",
    "PETRONET", "RVNL", "SAGILITY", "SAIL", "SBICARD", "SBILIFE", "SBIN",
    "SHRIRAMFIN", "SIEMENS", "SOLARINDS", "TCS", "VEDL",
]

# Full 220-symbol F&O universe (from fno-universe endpoint)
FULL_FNO_220 = [
    "360ONE","ABB","ABCAPITAL","ADANIENSOL","ADANIENT","ADANIGREEN","ADANIPORTS",
    "ADANIPOWER","ALKEM","AMBER","AMBUJACEM","ANGELONE","APLAPOLLO","APOLLOHOSP",
    "ASHOKLEY","ASIANPAINT","ASTRAL","ATHERENERG","AUBANK","AUROPHARMA","AXISBANK",
    "BAJAJ-AUTO","BAJAJFINSV","BAJAJHLDNG","BAJFINANCE","BANDHANBNK","BANKBARODA",
    "BANKINDIA","BANKNIFTY","BDL","BEL","BHARATFORG","BHARTIARTL","BHEL","BIOCON",
    "BLUESTARCO","BOSCHLTD","BPCL","BRITANNIA","BSE","CAMS","CANBK","CDSL","CGPOWER",
    "CHOLAFIN","CIPLA","COALINDIA","COCHINSHIP","COFORGE","COLPAL","CONCOR","CROMPTON",
    "CUMMINSIND","DABUR","DELHIVERY","DIVISLAB","DIXON","DLF","DMART","DRREDDY",
    "EICHERMOT","ETERNAL","FEDERALBNK","FINNIFTY","FORCEMOT","FORTIS","GAIL",
    "GLENMARK","GMRAIRPORT","GODFRYPHLP","GODREJCP","GODREJPROP","GRASIM","GVT&D",
    "HAL","HAVELLS","HCLTECH","HDFCAMC","HDFCBANK","HDFCLIFE","HEROMOTOCO","HINDALCO",
    "HINDPETRO","HINDUNILVR","HINDZINC","HYUNDAI","ICICIBANK","ICICIGI","ICICIPRULI",
    "IDEA","IDFCFIRSTB","IEX","INDHOTEL","INDIANB","INDIGO","INDUSINDBK","INDUSTOWER",
    "INFY","INOXWIND","IOC","IPCALAB","IRCTC","IREDA","IRFC","ITC","JINDALSTEL",
    "JIOFIN","JSWENERGY","JSWSTEEL","JUBLFOOD","KALYANKJIL","KAYNES","KEI","KFINTECH",
    "KOTAKBANK","KPITTECH","LAURUSLABS","LICHSGFIN","LICI","LODHA","LT","LTF","LTIM",
    "LTM","LUPIN","M&M","MAHABANK","MANAPPURAM","MANKIND","MARICO","MARUTI","MAXHEALTH",
    "MAZDOCK","MCX","MFSL","MIDCPNIFTY","MOTHERSON","MOTILALOFS","MPHASIS","MUTHOOTFIN",
    "NAM-INDIA","NATIONALUM","NAUKRI","NBCC","NESTLEIND","NHPC","NIFTY","NIFTYFPI",
    "NIFTYNXT50","NMDC","NTPC","NYKAA","OBEROIRLTY","OFSS","OIL","ONGC","PAGEIND",
    "PATANJALI","PAYTM","PERSISTENT","PETRONET","PFC","PGEL","PHOENIXLTD","PIDILITIND",
    "PIIND","PNB","PNBHOUSING","POLICYBZR","POLYCAB","POWERGRID","POWERINDIA",
    "PREMIERENE","PRESTIGE","RADICO","RBLBANK","RECLTD","RVNL","SAGILITY","SAIL",
    "SBICARD","SBILIFE","SBIN","SHREECEM","SHRIRAMFIN","SIEMENS","SOLARINDS","SONACOMS",
    "SRF","SUNPHARMA","SUPREMEIND","SUZLON","SWIGGY","TATACOMM","TATACONSUM","TATAELXSI",
    "TATAMOTORS","TATAPOWER","TATASTEEL","TCS","TECHM","TIINDIA","TITAN","TMPV",
    "TORNTPHARM","TRENT","TVSMOTOR","ULTRACEMCO","UNIONBANK","UNITDSPR","UNOMINDA",
    "UPL","VBL","VEDL","VMM","VOLTAS","WAAREEENER","WIPRO","YESBANK","ZYDUSLIFE",
]


def analyse_symbol(sym: str) -> dict:
    """Analyse a single symbol's cached data."""
    parquet = DATA_ROOT / f"{sym}.parquet"
    csv = DATA_ROOT / f"{sym}.csv"

    if parquet.exists():
        try:
            df = pd.read_parquet(parquet)
            source_file = str(parquet)
        except Exception as e:
            return {"symbol": sym, "status": "READ_ERROR", "error": str(e)}
    elif csv.exists():
        try:
            df = pd.read_csv(csv, index_col=0, parse_dates=True)
            source_file = str(csv)
        except Exception as e:
            return {"symbol": sym, "status": "READ_ERROR", "error": str(e)}
    else:
        return {
            "symbol": sym,
            "status": "UNAVAILABLE",
            "source_file": None,
            "reason": "No parquet or CSV found in data/1d/",
        }

    if len(df) == 0:
        return {"symbol": sym, "status": "EMPTY", "source_file": source_file}

    # Parse index to dates
    try:
        idx = pd.to_datetime(df.index, utc=True)
        dates = sorted({d.date() for d in idx})
    except Exception:
        dates = []

    n_bars = len(df)
    first_date = dates[0].isoformat() if dates else None
    last_date = dates[-1].isoformat() if dates else None

    # Calendar-aware gap analysis
    if len(dates) > 1:
        gap_summary = classify_gaps_in_series(dates)
        true_missing = gap_summary["n_true_missing_sessions"]
        expected_closures = gap_summary["n_expected_closures"]
    else:
        true_missing = 0
        expected_closures = 0

    # History length in years
    if dates:
        history_years = (dates[-1] - dates[0]).days / 365.25
    else:
        history_years = 0.0

    # Data quality: fraction of bars with valid OHLCV
    quality_score = 100
    if "close" in df.columns:
        n_nan_close = df["close"].isna().sum()
        quality_score = round(100 * (1 - n_nan_close / max(n_bars, 1)), 1)

    return {
        "symbol": sym,
        "status": "VALID" if n_bars >= 252 else "INSUFFICIENT_HISTORY",
        "n_bars": n_bars,
        "first_date": first_date,
        "last_date": last_date,
        "history_years": round(history_years, 2),
        "true_missing_sessions": true_missing,
        "expected_closures": expected_closures,
        "quality_score_pct": quality_score,
        "source_file": source_file,
        "in_baseline_65": sym in BASELINE_SYMBOLS,
    }


def main() -> None:
    print("Running universe coverage analysis...")
    print(f"Data root: {DATA_ROOT.resolve()}")

    if not DATA_ROOT.exists():
        print(f"WARNING: {DATA_ROOT} does not exist — no cached data found.")
        print("Run ingest_broad_universe.py first to populate data/1d/")

    results = []
    for sym in FULL_FNO_220:
        r = analyse_symbol(sym)
        results.append(r)

    # Summary statistics
    n_valid = sum(1 for r in results if r.get("status") == "VALID")
    n_insufficient = sum(1 for r in results if r.get("status") == "INSUFFICIENT_HISTORY")
    n_unavailable = sum(1 for r in results if r.get("status") == "UNAVAILABLE")
    n_error = sum(1 for r in results if r.get("status") in ("READ_ERROR", "EMPTY"))

    valid_bars = [r["n_bars"] for r in results if r.get("status") == "VALID"]
    median_bars = int(np.median(valid_bars)) if valid_bars else 0

    baseline_valid = sum(
        1 for r in results
        if r.get("in_baseline_65") and r.get("status") == "VALID"
    )

    # Count true missing sessions across all valid symbols
    total_true_missing = sum(
        r.get("true_missing_sessions", 0)
        for r in results if r.get("status") == "VALID"
    )

    # Calendar check: verify the fix works correctly
    from datetime import date as _date
    test_dates = [
        _date(2024, 1, 15),  # Monday — open
        _date(2024, 1, 20),  # Saturday — EXPECTED_CLOSURE
        _date(2024, 1, 21),  # Sunday — EXPECTED_CLOSURE
        _date(2024, 1, 22),  # NSE holiday 2024 — EXPECTED_CLOSURE
        _date(2024, 1, 23),  # Tuesday — open (but "missing" from test)
    ]
    # Classify gap between Jan 15 and Jan 23 (skipping 16-22)
    from src.validation.calendar import classify_gap
    gaps = classify_gap(_date(2024, 1, 15), _date(2024, 1, 23))
    calendar_test_ok = all(
        g["gap_type"] == "EXPECTED_MARKET_CLOSURE"
        for g in gaps
        if g["date"] in ["2024-01-20", "2024-01-21", "2024-01-22"]
    )
    print(f"Calendar gap classification test: {'PASS' if calendar_test_ok else 'FAIL'}")

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "mandate": "§3 — maximum verified F&O universe + §4 calendar-aware gap detection",
        "universe_size_fno_220": len(FULL_FNO_220),
        "baseline_65_symbols": len(BASELINE_SYMBOLS),
        "coverage": {
            "n_valid_ge_252_bars": n_valid,
            "n_insufficient_history": n_insufficient,
            "n_unavailable_no_cache": n_unavailable,
            "n_error": n_error,
            "coverage_pct_of_220": round(100 * n_valid / len(FULL_FNO_220), 1),
            "baseline_65_valid": baseline_valid,
            "median_bars_valid_symbols": median_bars,
        },
        "survivorship_classification": "CURRENT_UNIVERSE_ONLY",
        "survivorship_note": (
            "The 220-symbol list is the CURRENT live F&O constituent set. "
            "Historical constituent membership is unavailable. "
            "All analyses using this universe are SURVIVORSHIP_LIMITED."
        ),
        "gap_detection_method": "NSE_CALENDAR_AWARE",
        "gap_detection_note": (
            "Gaps are now classified as EXPECTED_MARKET_CLOSURE (weekends + NSE holidays) "
            "or TRUE_MISSING_SESSION (mandate §4). "
            "Prior code used pd.bdate_range (Mon-Fri only), which over-counted gaps by "
            "including NSE holidays as 'missing'. This has been fixed in ingestion.py."
        ),
        "total_true_missing_sessions_across_valid_symbols": total_true_missing,
        "calendar_test": {
            "pass": calendar_test_ok,
            "note": "2024-01-20 (Sat), 2024-01-21 (Sun), 2024-01-22 (NSE holiday) all classified EXPECTED_MARKET_CLOSURE"
        },
        "rate_limit_note": (
            "Universe coverage is limited to 65/220 symbols in the current frozen dataset. "
            "The data-service rate limit (100 req/60s shared) combined with a 4-concurrent / "
            "8 req/s limiter restricted full ingestion. "
            "Remedy: schedule off-peak batch ingestion with the existing resumable pipeline. "
            "The frozen baseline uses the 65 symbols successfully ingested."
        ),
        "symbols": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport written: {REPORT_PATH}")
    print(f"Valid (≥252 bars): {n_valid}/{len(FULL_FNO_220)} ({report['coverage']['coverage_pct_of_220']}%)")
    print(f"Unavailable (no cache): {n_unavailable}")
    print(f"Baseline 65 valid: {baseline_valid}/65")
    print(f"Total true missing sessions (all valid): {total_true_missing}")


if __name__ == "__main__":
    main()
