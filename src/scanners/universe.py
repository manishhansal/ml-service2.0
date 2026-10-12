"""
universe.py — NSE/BSE F&O trading universe for SMC scanners.

Instruments scanned:
  - NSE F&O indices: NIFTY 50, BANK NIFTY, FIN NIFTY, MIDCAP NIFTY
  - BSE index      : SENSEX
  - All NSE F&O permitted stocks (current quarterly revision)
"""
from __future__ import annotations

# ── Indices (NSE F&O + BSE Sensex) ────────────────────────────────────────────
NSE_INDICES: frozenset[str] = frozenset({
    "NIFTY",       # NSE Nifty 50
    "BANKNIFTY",   # NSE Bank Nifty
    "FINNIFTY",    # NSE Nifty Financial Services
    "MIDCPNIFTY",  # NSE Nifty MidCap Select
    "SENSEX",      # BSE Sensex (30-stock index)
})

# ── NSE F&O permitted stocks ──────────────────────────────────────────────────
# Source: NSE fno-universe endpoint (FULL_FNO_220, run_universe_coverage.py).
# Includes all stocks currently in the F&O segment.
# Exclusions:
#   TMPV  — Tata Motors DVR/passenger-vehicles subdivision; not a standalone F&O contract
#   LTI   — Legacy ticker for LTIMindtree (merged into LTM); parquet kept for history
#   NIFTYFPI / NIFTYNXT50 — broader index-level instruments, not single stocks
NSE_FO_STOCKS: frozenset[str] = frozenset({
    "360ONE", "ABB", "ABCAPITAL", "ADANIENSOL", "ADANIENT", "ADANIGREEN",
    "ADANIPORTS", "ADANIPOWER", "ALKEM", "AMBER", "AMBUJACEM", "ANGELONE",
    "APLAPOLLO", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT",
    "ASTRAL", "ATHERENERG", "ATGL", "ATUL", "AUBANK", "AUROPHARMA",
    "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJAJHLDNG", "BAJFINANCE",
    "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "BANKBARODA", "BANKINDIA",
    "BATAINDIA", "BDL", "BEL", "BERGEPAINT", "BHARATFORG", "BHARTIARTL",
    "BHEL", "BIOCON", "BLUESTARCO", "BOSCHLTD", "BPCL", "BRITANNIA",
    "BSE", "BSOFT", "CAMS", "CANBK", "CANFINHOME", "CDSL", "CESC",
    "CGPOWER", "CHAMBLFERT", "CHOLAFIN", "CIPLA", "COALINDIA", "COCHINSHIP",
    "COFORGE", "COLPAL", "CONCOR", "COROMANDEL", "CROMPTON", "CUB",
    "CUMMINSIND", "CYIENT", "DABUR", "DALBHARAT", "DEEPAKNTR", "DELHIVERY",
    "DELTACORP", "DIVISLAB", "DIXON", "DLF", "DMART", "DRREDDY",
    "EICHERMOT", "ESCORTS", "ETERNAL", "EXIDEIND", "FEDERALBNK", "FORCEMOT",
    "FORTIS", "FSL", "GAIL", "GLENMARK", "GMRAIRPORT", "GNFC",
    "GODFRYPHLP", "GODREJCP", "GODREJPROP", "GRANULES", "GRASIM", "GVT&D",
    "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HFCL", "HINDALCO", "HINDCOPPER", "HINDPETRO",
    "HINDUNILVR", "HINDZINC", "HONAUT", "HUDCO", "HYUNDAI", "ICICIBANK",
    "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX", "IGL", "IIFL",
    "INDHOTEL", "INDIACEM", "INDIAMART", "INDIANB", "INDIGO", "INDUSINDBK",
    "INDUSTOWER", "INFY", "INOXWIND", "INTELLECT", "IOC", "IPCALAB",
    "IRB", "IRCTC", "IREDA", "IRFC", "ITC", "JINDALSTEL", "JIOFIN",
    "JKCEMENT", "JSL", "JSWENERGY", "JSWSTEEL", "JUBLFOOD", "KALYANKJIL",
    "KAYNES", "KEI", "KFINTECH", "KOTAKBANK", "KPITTECH", "LALPATHLAB",
    "LAURUSLABS", "LICHSGFIN", "LICI", "LODHA", "LT", "LTF", "LTM",
    "LTTS", "LUPIN", "M&M", "M&MFIN", "MAHABANK", "MANAPPURAM",
    "MANKIND", "MARICO", "MARUTI", "MAXHEALTH", "MAZDOCK", "MCX",
    "METROPOLIS", "MFSL", "MGL", "MOTHERSON", "MOTILALOFS", "MPHASIS",
    "MRF", "MUTHOOTFIN", "NAM-INDIA", "NATIONALUM", "NAUKRI", "NAVINFLUOR",
    "NBCC", "NCC", "NESTLEIND", "NHPC", "NMDC", "NTPC", "NUVAMA",
    "NYKAA", "OBEROIRLTY", "OFSS", "OIL", "ONGC", "PAGEIND", "PATANJALI",
    "PAYTM", "PERSISTENT", "PETRONET", "PFC", "PFIZER", "PGEL",
    "PHOENIXLTD", "PIDILITIND", "PIIND", "PNB", "PNBHOUSING", "POLICYBZR",
    "POLYCAB", "POONAWALLA", "POWERGRID", "POWERINDIA", "PPLPHARMA",
    "PREMIERENE", "PRESTIGE", "PVRINOX", "RADICO", "RAIN", "RAMCOCEM",
    "RBLBANK", "RECLTD", "RELIANCE", "RVNL", "SAGILITY", "SAIL",
    "SAMMAANCAP", "SBICARD", "SBILIFE", "SBIN", "SHREECEM", "SHRIRAMFIN",
    "SIEMENS", "SJVN", "SOLARINDS", "SONACOMS", "SRF", "STAR", "SUNPHARMA",
    "SUNTV", "SUPREMEIND", "SUZLON", "SWIGGY", "SYNGENE", "TATACHEM",
    "TATACOMM", "TATACONSUM", "TATAELXSI", "TATAMOTORS", "TATAPOWER",
    "TATASTEEL", "TATATECH", "TCS", "TECHM", "TIINDIA", "TITAGARH",
    "TITAN", "TORNTPHARM", "TORNTPOWER", "TRENT", "TVSMOTOR", "UBL",
    "ULTRACEMCO", "UNIONBANK", "UNITDSPR", "UNOMINDA", "UPL", "VBL",
    "VEDL", "VMM", "VOLTAS", "WAAREEENER", "WHIRLPOOL", "WIPRO",
    "YESBANK", "ZEEL", "ZYDUSLIFE",
})

# ── Combined F&O universe (indices + stocks) ──────────────────────────────────
FO_UNIVERSE: frozenset[str] = NSE_INDICES | NSE_FO_STOCKS


def is_fo_symbol(symbol: str) -> bool:
    """Return True if *symbol* is a tradeable NSE index or F&O stock."""
    return symbol in FO_UNIVERSE
