"""
refresh_oct9.py — Fetch NSE F&O Oct 9, 2026 (Friday) closing data from Upstox 
and append to parquet files. Run on Sunday before Monday market open.

Usage: python scripts/refresh_oct9.py
"""
import gzip, json, time, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / "data" / "1d" / "1d"
IST = timezone(timedelta(hours=5, minutes=30))

# Read Upstox access token from data-service .env
env_path = Path("/Users/manishkumar/Desktop/data-service2.0/.env")
env = {}
for line in env_path.read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()

ACCESS_TOKEN = env.get("UPSTOX_ACCESS_TOKEN", "")
if not ACCESS_TOKEN:
    raise SystemExit("UPSTOX_ACCESS_TOKEN not found in data-service .env")

# Build symbol → instrument_key mapping from Upstox instruments master
INSTRUMENTS_GZ = Path("/tmp/nse_instruments.json.gz")
if not INSTRUMENTS_GZ.exists():
    print("Downloading NSE instruments master...")
    urllib.request.urlretrieve(
        "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
        INSTRUMENTS_GZ
    )

with gzip.open(INSTRUMENTS_GZ) as f:
    instruments = json.load(f)

# Map trading_symbol → instrument_key for EQ and BE (book entry) types
sym_map = {}
for inst in instruments:
    if inst.get("segment") == "NSE_EQ" and inst.get("instrument_type") in ("EQ", "BE"):
        sym_map[inst["trading_symbol"]] = inst["instrument_key"]

# Manual overrides for symbols with different naming
MANUAL_MAP = {
    "TATAMOTORS": "NSE_EQ|INE155A01022",  # TMPV = Tata Motors passenger vehicles
    "HFCL":       "NSE_EQ|INE548A01028",
    "LTI":        "NSE_EQ|INE214T01019",  # LTIMindtree (was LTI before merger)
    # Indices — skip (no EQ instrument key)
    "NIFTY":      None,
    "BANKNIFTY":  None,
    "FINNIFTY":   None,
    "MIDCPNIFTY": None,
}
sym_map.update({k: v for k, v in MANUAL_MAP.items() if v})

def fetch_candle(instrument_key: str, date_str: str) -> dict | None:
    """Fetch 1d candle for instrument_key on date_str (YYYY-MM-DD IST)."""
    # Upstox historical: /v2/historical-candle/{key}/day/{to}/{from}
    # NOTE: safe='' encodes the pipe as %7C — required by Upstox CDN.
    # safe='|' leaves the pipe unencoded which causes the request to hang.
    url = (
        f"https://api.upstox.com/v2/historical-candle/"
        f"{urllib.parse.quote(instrument_key, safe='')}"
        f"/day/{date_str}/{date_str}"
    )
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",  # Upstox CDN blocks default Python UA
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        candles = data.get("data", {}).get("candles", [])
        if not candles:
            return None
        # candle = [timestamp, open, high, low, close, volume, oi]
        c = candles[0]
        ts = pd.Timestamp(c[0])  # already IST: "2026-10-09T00:00:00+05:30"
        return {
            "timestamp_ist": ts,
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
            "provider": "upstox",
        }
    except Exception as e:
        # Uncomment to debug: print(f"    fetch_candle error for {instrument_key}: {e}")
        return None


TARGET_DATE = "2026-10-09"
parquets = sorted(DATA_DIR.glob("*.parquet"))
print(f"Refreshing {len(parquets)} symbols with {TARGET_DATE} data...\n")

updated = skipped = failed = already_have = 0

for pf in parquets:
    sym = pf.stem
    
    # Skip indices
    if sym in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"):
        skipped += 1
        continue
    
    ikey = sym_map.get(sym)
    if not ikey:
        print(f"  [SKIP] {sym}: no instrument key")
        skipped += 1
        continue
    
    # Check if we already have Oct 9 data
    df_existing = pd.read_parquet(pf).drop(columns=["provider"], errors="ignore")
    if df_existing.index.tz is None:
        df_existing.index = df_existing.index.tz_localize("UTC")
    last_ist = df_existing.index[-1].astimezone(IST).date()
    if str(last_ist) >= TARGET_DATE:
        already_have += 1
        continue
    
    # Fetch Oct 9
    row = fetch_candle(ikey, TARGET_DATE)
    
    if row is None:
        # Oct 9 might not have traded (holiday or illiquid)
        failed += 1
        continue
    
    # Build new row with UTC timestamp (18:30 UTC = IST midnight next day convention)
    # Store as UTC like the existing data
    ts_utc = row["timestamp_ist"].astimezone(timezone.utc)
    new_row = pd.DataFrame([{
        "open":     row["open"],
        "high":     row["high"],
        "low":      row["low"],
        "close":    row["close"],
        "volume":   row["volume"],
        "provider": row["provider"],
    }], index=pd.DatetimeIndex([ts_utc], tz="UTC"))
    
    # Append and save
    df_full = pd.read_parquet(pf)
    if df_full.index.tz is None:
        df_full.index = df_full.index.tz_localize("UTC")
    
    # Deduplicate by index
    df_combined = pd.concat([df_full, new_row])
    df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
    df_combined = df_combined.sort_index()
    df_combined.to_parquet(pf)
    
    updated += 1
    print(f"  [+] {sym}: Oct 9 close={row['close']:.2f}  vol={row['volume']:.0f}")
    
    time.sleep(0.15)  # respect rate limits

print(f"\n{'='*55}")
print(f"  Updated  : {updated} symbols")
print(f"  Already  : {already_have} symbols had Oct 9 data")
print(f"  Skipped  : {skipped} symbols (indices/no key)")
print(f"  No data  : {failed} symbols (holiday/illiquid)")
print(f"{'='*55}")
