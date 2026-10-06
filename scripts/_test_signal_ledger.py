"""Quick unit tests for SignalLedger — run with PYTHONPATH=. python3 scripts/_test_signal_ledger.py"""
import sys, json, copy
from pathlib import Path
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from src.analytics.signal_ledger import SignalLedger, _trading_days_ahead
from datetime import date

print("Signal Ledger Unit Tests")
print("-" * 40)

# T1
d = _trading_days_ahead(date(2026, 10, 6), 7)
assert str(d) == "2026-10-15", f"Expected 2026-10-15 got {d}"
print(f"T1 T+7 from Oct 6 = {d}  OK")

# T2
ledger = SignalLedger()
rpt = ledger.status_report()
assert rpt["total_positions"] == 84, f"Expected 84 got {rpt['total_positions']}"
assert rpt["open"] == 84
print(f"T2 Loaded {rpt['total_positions']} positions  OK")

# T3: idempotent
s = json.load(open(BASE / "artifacts/live_session/latest_scores.json"))
active = [x for x in s["signals"] if x.get("direction") != 0]
new = ledger.record_signals(active, "2026-10-06", None, s["generated_at"])
assert new == 0, f"Expected 0 new on re-run, got {new}"
print("T3 Idempotent (0 new on re-run)  OK")

# T4: settle WIN
test_pos = copy.deepcopy(list(ledger._positions.values())[0])
test_pos.update({"pos_key": "TEST_WIN:test", "symbol": "TEST_WIN",
                 "resolve_after": "2026-10-01", "entry_price": 100.0,
                 "last_price": 100.0, "status": "OPEN", "direction": 1})
ledger._positions["TEST_WIN:test"] = test_pos
settled = ledger.settle_expired("2026-10-06", live_quotes={"TEST_WIN": {"ltp": 105.0}})
found = next((p for p in settled if p["symbol"] == "TEST_WIN"), None)
assert found and found["status"] == "SETTLED_WIN", f"Got {found}"
print(f"T4 Settle WIN: {found['final_return_pct']:+.3f}%  OK")

# T5: settle LOSS — create from scratch, not from already-settled test_pos
test_pos2 = {"pos_key": "TEST_LOSS:test", "symbol": "TEST_LOSS",
             "resolve_after": "2026-10-01", "entry_price": 100.0,
             "last_price": 100.0, "status": "OPEN", "direction": 1,
             "session_date": "2026-10-01", "score": 0.4, "horizon_days": 7,
             "generated_at": "2026-10-01T04:00:00+00:00", "conviction": "C",
             "cost_bps": 7.26, "nifty_at_signal": 22000, "regime": "UNKNOWN",
             "data_date": "2026-09-30", "rank": None, "settled_at": None,
             "final_return_pct": None, "outcome": None, "last_price_date": None,
             "unrealized_pct": 0.0, "signal_id": "test-loss-001"}
ledger._positions["TEST_LOSS:test"] = test_pos2
settled2 = ledger.settle_expired("2026-10-06", live_quotes={"TEST_LOSS": {"ltp": 98.0}})
found2 = next((p for p in settled2 if p["symbol"] == "TEST_LOSS"), None)
assert found2 and found2["status"] == "SETTLED_LOSS", f"Got {found2}"
print(f"T5 Settle LOSS: {found2['final_return_pct']:+.3f}%  OK")

# T6: mark-to-market
sample_sym = list(ledger._positions.keys())[0].split(":")[0]
updated = ledger.update_mark_to_market({sample_sym: {"ltp": 9999.0}}, "2026-10-06")
assert updated >= 1
print(f"T6 MTM updated {updated} positions  OK")

# Cleanup test positions
for k in ["TEST_WIN:test", "TEST_LOSS:test"]:
    ledger._positions.pop(k, None)
ledger._save_positions()

print()
print("All 6 tests PASSED")
