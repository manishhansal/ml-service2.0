"""Quick unit tests for SignalLedger (intraday mode)."""
import sys, json, copy
from pathlib import Path
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from src.analytics.signal_ledger import SignalLedger

print("Signal Ledger Unit Tests (intraday)")
print("-" * 40)

# T1: resolve_after == session_date (same-day intraday)
ledger0 = SignalLedger()
sample_sig = [{"symbol": "TESTAAA", "direction": 1, "score": 0.7,
               "conviction": "A", "rank": 1, "data_date": "2026-10-06"}]
ledger0.record_signals(sample_sig, "2099-01-01", None, "2099-01-01T09:00:00Z",
                       live_quotes={"TESTAAA": {"ltp": 500.0}})
test_p = ledger0._positions.get("TESTAAA:2099-01-01", {})
assert test_p.get("resolve_after") == "2099-01-01", f"Expected same-day, got {test_p.get('resolve_after')}"
print(f"T1 resolve_after == session_date  OK")
# clean up T1 test position before loading the real ledger
ledger0._positions.pop("TESTAAA:2099-01-01", None)
ledger0._save_positions()

# T2: load real positions
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

# T4: settle WIN — use future test date to avoid touching real positions
TEST_DATE = "2099-01-05"
test_win = {"pos_key": f"TEST_WIN:{TEST_DATE}", "symbol": "TEST_WIN",
            "session_date": TEST_DATE, "entry_price": 100.0,
            "last_price": 100.0, "status": "OPEN", "direction": 1,
            "resolve_after": TEST_DATE, "score": 0.7, "conviction": "A",
            "generated_at": f"{TEST_DATE}T09:30:00Z", "entry_time": f"{TEST_DATE}T09:30:00Z",
            "exit_time": None, "exit_price": None, "unrealized_pct": 0.0,
            "settled_at": None, "final_return_pct": None, "outcome": None,
            "signal_id": "test-win-001", "data_date": "2099-01-04",
            "rank": 1, "nifty_at_signal": 22717.0, "cost_bps": 7.26,
            "last_price_time": None}
ledger._positions[f"TEST_WIN:{TEST_DATE}"] = test_win
settled = ledger.settle_session(TEST_DATE, final_quotes={"TEST_WIN": {"ltp": 105.0}})
found = next((p for p in settled if p["symbol"] == "TEST_WIN"), None)
assert found and found["status"] == "SETTLED_WIN", f"Got {found}"
print(f"T4 Settle WIN: {found['final_return_pct']:+.3f}%  OK")

# T5: settle LOSS
test_loss = copy.deepcopy(test_win)
test_loss.update({"pos_key": f"TEST_LOSS:{TEST_DATE}", "symbol": "TEST_LOSS",
                  "status": "OPEN", "signal_id": "test-loss-001"})
ledger._positions[f"TEST_LOSS:{TEST_DATE}"] = test_loss
settled2 = ledger.settle_session(TEST_DATE, final_quotes={"TEST_LOSS": {"ltp": 97.0}})
found2 = next((p for p in settled2 if p["symbol"] == "TEST_LOSS"), None)
assert found2 and found2["status"] == "SETTLED_LOSS", f"Got {found2}"
print(f"T5 Settle LOSS: {found2['final_return_pct']:+.3f}%  OK")

# T6: MTM update on real OPEN positions (2026-10-06 are still OPEN)
open_syms = [k.split(":")[0] for k, v in ledger._positions.items()
             if v.get("status") == "OPEN" and v.get("session_date") == "2026-10-06"]
assert len(open_syms) >= 42, f"Expected >=42 open today, got {len(open_syms)}"
sym0 = open_syms[0]
updated = ledger.update_mark_to_market({sym0: {"ltp": 9999.0}}, "2026-10-06")
assert updated >= 1
print(f"T6 MTM updated {updated} positions  OK")

# T7: today_positions filter
today = ledger.today_positions("2026-10-06")
other = ledger.today_positions("2025-01-01")
assert len(today) >= 42, f"Expected >=42 today, got {len(today)}"
assert len(other) == 0
print(f"T7 today_positions: {len(today)} today, {len(other)} other day  OK")

# Cleanup
for k in list(ledger._positions.keys()):
    if "TEST_WIN" in k or "TEST_LOSS" in k or "TESTAAA" in k:
        ledger._positions.pop(k, None)
ledger._save_positions()

print()
print("All 7 tests PASSED")
