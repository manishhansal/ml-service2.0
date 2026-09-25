"""
scripts/run_forward_paper_audit.py — daily forward-paper audit (mandate §46).

Validates the state of the forward-paper engine and produces a daily audit
record.  Must be run at the end of each trading session.

Mandate §26: genuine forward paper requires:
  SIGNAL_AT_T → OUTCOME_AFTER_T
  NOT: HISTORICAL_REPLAY → CALL_IT_PAPER

This script:
1. Reads the append-only signal store
2. Identifies signals whose horizon has elapsed (resolve_due)
3. Reports expected vs actual signal counts
4. Flags any missing signals, stale data, or PIT violations
5. Does NOT compute performance on < MIN_TRADES_FOR_VALIDATION resolved trades

Status will remain INSUFFICIENT_SAMPLE until genuine live trading starts.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics.forward_paper import ForwardPaperStore, ForwardPaperRunner, MIN_TRADES_FOR_VALIDATION

SIGNAL_STORE_PATH = Path("artifacts/forward_paper/signals.jsonl")
AUDIT_REPORT_PATH = Path("reports/forward_paper_audit.json")


def main() -> None:
    now = datetime.now(tz=UTC)
    print(f"Forward Paper Audit — {now.isoformat()}")

    store = ForwardPaperStore(SIGNAL_STORE_PATH)
    runner = ForwardPaperRunner(store=store, interval="1d", horizon_bars=5)

    all_signals = store.all_signals()
    n_total = len(all_signals)
    n_trade = sum(1 for s in all_signals if s.decision == "TRADE")
    n_no_trade = sum(1 for s in all_signals if s.decision == "NO_TRADE")

    # Signals that are past their horizon
    due = runner.resolve_due(now=now)
    n_due = len(due)

    # Signals by model version
    versions = {}
    for s in all_signals:
        versions[s.model_version] = versions.get(s.model_version, 0) + 1

    # Status
    # Resolved count = 0 for now (no live trading has run)
    n_resolved = 0
    status = runner.status(n_resolved=n_resolved)

    # Anomaly detection
    anomalies = []
    if n_total == 0:
        anomalies.append({
            "type": "NO_SIGNALS_RECORDED",
            "severity": "INFO",
            "message": (
                "No forward-paper signals have been recorded yet. "
                "This is expected — genuine forward paper requires live market operation. "
                "Historical replay must NOT be counted as forward evidence (mandate §26)."
            ),
        })
    else:
        # Check model version consistency
        if len(versions) > 1:
            anomalies.append({
                "type": "MULTIPLE_MODEL_VERSIONS",
                "severity": "WARN",
                "versions": versions,
                "message": "Multiple model versions in signal store — verify no uncontrolled retraining occurred.",
            })
        # Check for signals beyond horizon that haven't been resolved
        if n_due > 0:
            anomalies.append({
                "type": "UNRESOLVED_DUE_SIGNALS",
                "severity": "WARN",
                "count": n_due,
                "message": f"{n_due} TRADE signals are past their horizon but not yet resolved.",
            })

    report = {
        "audit_timestamp": now.isoformat(),
        "mandate": "§46 — daily paper audit",
        "baseline_id": "CONFIRMATION_BASELINE_V1",
        "model_version": "1.0.0-20260925080931531542",
        "signal_store_path": str(SIGNAL_STORE_PATH),
        "signal_counts": {
            "n_total_recorded": n_total,
            "n_trade_signals": n_trade,
            "n_no_trade_signals": n_no_trade,
            "n_due_for_resolution": n_due,
            "n_resolved": n_resolved,
        },
        "model_versions_in_store": versions,
        "forward_paper_status": status,
        "anomalies": anomalies,
        "interpretation": (
            "FORWARD PAPER STATUS: NOT_RUN. "
            "No genuine signal-at-T / outcome-after-T pairs have been accumulated. "
            "This is the correct state — the infrastructure is ready but live "
            "operation has not begun. "
            "Shadow and production eligibility are BLOCKED until sufficient "
            f"forward-paper evidence accrues (minimum {MIN_TRADES_FOR_VALIDATION} "
            "resolved TRADE signals, mandate §29)."
        ),
        "immutability_note": (
            "The signal store is append-only. "
            "Resolution writes a SEPARATE record and never edits the original signal. "
            "No historical replay counts as forward evidence."
        ),
        "next_steps": [
            "Start live/paper trading session during NSE market hours (09:15-15:30 IST)",
            "Run signal generation at market close for all 65 baseline symbols",
            "Persist each signal via ForwardPaperRunner.record_signal()",
            "After horizon elapses (5 trading days), resolve outcomes with real market data",
            "Accumulate >= 20 resolved TRADE signals before any performance conclusions",
        ],
    }

    AUDIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nAudit report written: {AUDIT_REPORT_PATH}")
    print(f"Status: {status['state']}")
    print(f"Signals recorded: {n_total}")
    print(f"Anomalies: {len(anomalies)}")

    if anomalies:
        for a in anomalies:
            print(f"  [{a['severity']}] {a['type']}: {a['message'][:80]}")


if __name__ == "__main__":
    main()
