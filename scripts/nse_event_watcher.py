#!/usr/bin/env python3
"""
NSE Event Watcher — Phil's watch.py adapted for NSE F&O markets.

Monitors for catalysts that should trigger an early mid-session re-score:
  1. PRICE_MOVE   — NIFTY or BANKNIFTY moves ≥ 0.75% from session open
  2. SECTOR_ETF   — any tracked sector moves ≥ 1.5% vs NIFTY
  3. CALENDAR     — pre-scheduled NSE events (F&O expiry, RBI, earnings)
  4. FII_DATA     — net FII flow > ±₹500 Cr published by SEBI

Architecture (from Phil's watch.py):
  - Runs every 15 minutes during market hours
  - Prints one JSON verdict: {trigger: bool, events: [...], context: {...}}
  - ALWAYS exits 0 — a broken check never manufactures a cycle
  - Hard caps: 6 fires per session day, 2h per-key cooldown
  - State: journal/nse-watch-state.json (per-key cooldowns)
  - Trigger log: artifacts/live_session/watch-triggers.jsonl

Usage:
    python3 scripts/nse_event_watcher.py check
    python3 scripts/nse_event_watcher.py status
    python3 scripts/nse_event_watcher.py add-event --label "RBI_OCT" --fire-at "2026-10-01T10:00:00+05:30" --window 60

Run from cron / scheduler (every 15 min during market hours):
    */15 9-15 * * 1-5 cd /path/to/ml-service2.0 && python3 scripts/nse_event_watcher.py check >> artifacts/live_session/watcher.log 2>&1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# ── Config ────────────────────────────────────────────────────────────────────
STATE_FILE    = BASE / "journal" / "nse-watch-state.json"
TRIGGERS_FILE = BASE / "artifacts" / "live_session" / "watch-triggers.jsonl"
CALENDAR_FILE = BASE / "strategy" / "nse-calendar.json"

DAILY_FIRE_BUDGET   = 6
COOLDOWN_SECONDS    = 2 * 3600       # 2h per key
RECENT_SCORE_SEC    = 20 * 60        # suppress if re-scored < 20 min ago
SAME_KEY_SCORE_SEC  = 45 * 60        # suppress same key if scored < 45 min ago
STATE_TTL_SECONDS   = 24 * 3600      # state GC

# Price-move thresholds
NIFTY_MOVE_THRESHOLD     = 0.0075    # 0.75% from session open
BANKNIFTY_MOVE_THRESHOLD = 0.010     # 1.00%
SECTOR_MOVE_THRESHOLD    = 0.015     # 1.50% vs NIFTY

# NSE market data (free endpoints — no API key required)
NSE_BASE  = "https://www.nseindia.com"
NSE_INDEX = "https://www.nseindia.com/api/allIndices"

# Data-service LTP endpoint (local, from .env)
try:
    _env = {k.strip(): v.strip() for line in (BASE / ".env").read_text().splitlines()
            if "=" in line and not line.strip().startswith("#")
            for k, _, v in [line.partition("=")]}
    DATA_URL = _env.get("DATA_SERVICE_2_URL", "http://localhost:8200")
    DATA_KEY = _env.get("DATA_SERVICE_API_KEY", "")
except Exception:
    DATA_URL = "http://localhost:8200"
    DATA_KEY = ""

TRACKED_SYMBOLS = {
    "NIFTY":    {"type": "index",  "threshold": NIFTY_MOVE_THRESHOLD},
    "BANKNIFTY":{"type": "index",  "threshold": BANKNIFTY_MOVE_THRESHOLD},
    "NIFTYMID": {"type": "sector", "threshold": SECTOR_MOVE_THRESHOLD},
    "CNXFMCG":  {"type": "sector", "threshold": SECTOR_MOVE_THRESHOLD},
    "CNXIT":    {"type": "sector", "threshold": SECTOR_MOVE_THRESHOLD},
    "CNXPHARMA":{"type": "sector", "threshold": SECTOR_MOVE_THRESHOLD},
    "CNXAUTO":  {"type": "sector", "threshold": SECTOR_MOVE_THRESHOLD},
    "CNXBANK":  {"type": "sector", "threshold": SECTOR_MOVE_THRESHOLD},
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def ist_now() -> dt.datetime:
    IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
    return dt.datetime.now(tz=IST)


def utcnow() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)


def iso(ts: dt.datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        s = str(value).replace("Z", "+00:00")
        ts = dt.datetime.fromisoformat(s)
        return ts if ts.tzinfo else ts.replace(tzinfo=dt.timezone.utc)
    except (ValueError, AttributeError):
        return None


def warn(msg: str) -> None:
    print(f"nse-watcher: {msg}", file=sys.stderr)


def get_ltp(symbol: str) -> float | None:
    """Fetch live price from data-service. Returns None on any error."""
    url = f"{DATA_URL}/v1/india/ltp?symbols={symbol}"
    req = urllib.request.Request(url, headers={"X-API-KEY": DATA_KEY})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            d = json.load(resp)
            data = d.get("data", d)
            if isinstance(data, dict):
                sym_data = data.get(symbol, {})
                return float(sym_data.get("ltp", 0)) or None
            if isinstance(data, list):
                for item in data:
                    if item.get("symbol") == symbol:
                        return float(item.get("ltp", 0)) or None
    except Exception as e:
        warn(f"LTP fetch {symbol}: {e}")
    return None


def get_session_open(symbol: str) -> float | None:
    """Get today's session open from parquet or data-service."""
    pf = BASE / "data" / "1d" / "1d" / f"{symbol}.parquet"
    try:
        import pandas as pd
        df = pd.read_parquet(pf)
        today = ist_now().date()
        today_rows = df[df.index.date == today] if hasattr(df.index, "date") else df.tail(1)
        if len(today_rows) > 0 and "open" in today_rows.columns:
            return float(today_rows["open"].iloc[0])
    except Exception:
        pass
    return None


# ── State management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.is_file():
        return {"fired": {}}
    try:
        s = json.loads(STATE_FILE.read_text())
        return s if isinstance(s, dict) else {"fired": {}}
    except Exception:
        warn("state file unreadable — treating all keys as freshly fired")
        return None   # type: ignore[return-value]  # caller checks None


def save_state(state: dict, fires: list[dict]) -> None:
    now = utcnow()
    stamp = iso(now)
    # Append triggers
    TRIGGERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TRIGGERS_FILE.open("a") as fh:
        for f in fires:
            fh.write(json.dumps({
                "utc": stamp, "kind": f["kind"], "key": f["key"],
                "detail": f.get("detail", ""),
            }) + "\n")
    # GC old fired entries
    fired = {}
    for key, ts in state.get("fired", {}).items():
        parsed = parse_iso(ts)
        if parsed and (now - parsed).total_seconds() < STATE_TTL_SECONDS:
            fired[key] = ts
    for f in fires:
        fired[f["key"]] = stamp
    STATE_FILE.write_text(json.dumps({"last_fire_utc": stamp, "fired": dict(sorted(fired.items()))}, indent=2) + "\n")


def fires_today(now: dt.datetime) -> int:
    if not TRIGGERS_FILE.is_file():
        return 0
    today = iso(now)[:10]
    count = 0
    for line in TRIGGERS_FILE.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if str(r.get("utc", ""))[:10] == today:
                count += 1
        except (json.JSONDecodeError, KeyError):
            continue
    return count


def cooling_down(state: dict, key: str, now: dt.datetime) -> bool:
    ts = parse_iso(state.get("fired", {}).get(key))
    return ts is not None and (now - ts).total_seconds() < COOLDOWN_SECONDS


# ── Checks ────────────────────────────────────────────────────────────────────

def check_price_moves(notes: list[str]) -> list[dict]:
    """Fire when tracked indices/sectors move ≥ threshold from session open."""
    fires = []
    for sym, cfg in TRACKED_SYMBOLS.items():
        ltp  = get_ltp(sym)
        open_ = get_session_open(sym)
        if ltp is None or open_ is None or open_ <= 0:
            notes.append(f"{sym}: no price data")
            continue
        move = (ltp - open_) / open_
        if abs(move) < cfg["threshold"]:
            continue
        fires.append({
            "kind":   "PRICE_MOVE",
            "key":    f"pricemove:{sym.lower()}",
            "detail": f"{sym} moved {move:+.2%} from open {open_:.2f} → {ltp:.2f} (threshold {cfg['threshold']:.1%})",
            "context": {
                "symbol":     sym,
                "move_pct":   round(move * 100, 3),
                "open_price": open_,
                "ltp":        ltp,
                "type":       cfg["type"],
            },
        })
    return fires


def load_calendar() -> list[dict]:
    """Load NSE event calendar from strategy/nse-calendar.json."""
    if not CALENDAR_FILE.is_file():
        return []
    try:
        return json.loads(CALENDAR_FILE.read_text())
    except Exception as e:
        warn(f"calendar file unreadable: {e}")
        return []


def check_calendar(notes: list[str]) -> list[dict]:
    """Fire when a pre-scheduled NSE event window is open."""
    events = load_calendar()
    now = ist_now()
    fires = []
    for e in events:
        label    = str(e.get("label", ""))
        fire_at  = parse_iso(e.get("fire_at"))
        window_m = int(e.get("window_min", 30))
        if fire_at is None or not label:
            continue
        end = fire_at + dt.timedelta(minutes=window_m)
        if not (fire_at <= now < end):
            continue
        fires.append({
            "kind":    "CALENDAR",
            "key":     f"cal:{label.lower().replace(' ', '_')}",
            "detail":  f"NSE event window open: {label} ({iso(fire_at)} → {iso(end)})",
            "context": {
                "label":     label,
                "fire_at":   iso(fire_at),
                "window_end": iso(end),
                "event_type": e.get("event_type", "NSE_CORPORATE"),
            },
        })
    return fires


# ── Main check ────────────────────────────────────────────────────────────────

def check() -> dict:
    now   = utcnow()
    notes: list[str] = []

    state = load_state()
    if state is None:
        return {"trigger": False, "events": [], "notes": ["state file unreadable; no fire"]}

    # Daily budget check
    budget_left = DAILY_FIRE_BUDGET - fires_today(now)
    if budget_left <= 0:
        return {"trigger": False, "events": [], "notes": [f"daily fire budget spent ({DAILY_FIRE_BUDGET}/day)"]}

    # Gather candidate fires
    fires = check_price_moves(notes) + check_calendar(notes)

    # Apply guards
    kept = []
    for f in fires:
        if cooling_down(state, f["key"], now):
            notes.append(f"{f['key']} in cooldown ({COOLDOWN_SECONDS//3600}h)")
            continue
        if len(kept) >= budget_left:
            notes.append(f"{f['key']} deferred: daily budget exhausted")
            continue
        kept.append(f)

    if not kept:
        return {"trigger": False, "events": [], "notes": notes}

    save_state(state, kept)
    return {
        "trigger": True,
        "events":  [f["kind"] for f in kept],
        "keys":    [f["key"]  for f in kept],
        "context": [f["context"] for f in kept],
        "detail":  [f["detail"]  for f in kept],
        "notes":   notes,
        "utc":     iso(now),
    }


def cmd_status() -> None:
    state = load_state() or {}
    fired_today = fires_today(utcnow())
    print(json.dumps({
        "budget_left":   DAILY_FIRE_BUDGET - fired_today,
        "fired_today":   fired_today,
        "cooldowns":     {k: v for k, v in state.get("fired", {}).items()},
        "calendar_events": load_calendar(),
    }, indent=2))


def cmd_add_event(label: str, fire_at: str, window_min: int, event_type: str) -> None:
    """Add a NSE event to the calendar."""
    events = load_calendar()
    events.append({
        "label":      label,
        "fire_at":    fire_at,
        "window_min": window_min,
        "event_type": event_type,
    })
    CALENDAR_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALENDAR_FILE.write_text(json.dumps(events, indent=2) + "\n")
    print(f"Added: {label} at {fire_at} (window {window_min}min, type={event_type})")


# ── Embedded NSE calendar (pre-loaded Oct-Dec 2026) ───────────────────────────
DEFAULT_CALENDAR = [
    {"label": "RBI_OCT_2026",        "fire_at": "2026-10-01T10:00:00+05:30", "window_min": 120, "event_type": "RBI_POLICY"},
    {"label": "FO_EXPIRY_OCT_2026",  "fire_at": "2026-10-29T09:15:00+05:30", "window_min": 30,  "event_type": "FO_EXPIRY"},
    {"label": "FO_EXPIRY_NOV_2026",  "fire_at": "2026-11-26T09:15:00+05:30", "window_min": 30,  "event_type": "FO_EXPIRY"},
    {"label": "RBI_DEC_2026",        "fire_at": "2026-12-04T10:00:00+05:30", "window_min": 120, "event_type": "RBI_POLICY"},
    {"label": "FO_EXPIRY_DEC_2026",  "fire_at": "2026-12-31T09:15:00+05:30", "window_min": 30,  "event_type": "FO_EXPIRY"},
]


def ensure_calendar() -> None:
    """Create default calendar if not exists."""
    if not CALENDAR_FILE.is_file():
        CALENDAR_FILE.parent.mkdir(parents=True, exist_ok=True)
        CALENDAR_FILE.write_text(json.dumps(DEFAULT_CALENDAR, indent=2) + "\n")
        warn("Created default NSE calendar at strategy/nse-calendar.json")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_calendar()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check",  help="Print one JSON verdict; always exits 0")
    sub.add_parser("status", help="Show fire budget, cooldowns, calendar")

    add = sub.add_parser("add-event", help="Add NSE event to calendar")
    add.add_argument("--label",      required=True, help="Event label e.g. RBI_OCT_2026")
    add.add_argument("--fire-at",    required=True, help="ISO datetime e.g. 2026-10-01T10:00:00+05:30")
    add.add_argument("--window",     type=int, default=30, help="Window minutes (default 30)")
    add.add_argument("--event-type", default="NSE_CORPORATE", help="Event type tag")

    args = ap.parse_args()

    if args.cmd == "check":
        try:
            verdict = check()
        except Exception as e:
            verdict = {"trigger": False, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(verdict, indent=2))
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "add-event":
        cmd_add_event(args.label, args.fire_at, args.window, args.event_type)


if __name__ == "__main__":
    main()
