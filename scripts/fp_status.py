#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime, timezone

BASE = Path('/Users/manishkumar/Desktop/ml-service2.0')
sigs_v1 = [json.loads(l) for l in (BASE/'artifacts/forward_paper/signals.jsonl').read_text().splitlines() if l.strip()]
v2_path = BASE/'artifacts/forward_paper/signals_v2.jsonl'
sigs_v2 = [json.loads(l) for l in v2_path.read_text().splitlines() if l.strip()] if v2_path.exists() else []
all_sigs = sigs_v1 + sigs_v2

out_path = BASE/'artifacts/forward_paper/outcomes.jsonl'
outs = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()] if out_path.exists() else []
resolved = {o['signal_id'] for o in outs}
now = datetime.now(tz=timezone.utc)
due = [s for s in all_sigs if datetime.fromisoformat(s['resolve_after']) <= now and s['signal_id'] not in resolved]
pending = [s for s in all_sigs if datetime.fromisoformat(s['resolve_after']) > now]

print(f'UTC: {now.strftime("%Y-%m-%d %H:%M")}')
print(f'Total signals: {len(all_sigs)} (v1:{len(sigs_v1)} + v2:{len(sigs_v2)})')
print(f'Resolved: {len(resolved)} | Due: {len(due)} | Pending: {len(pending)}')
if pending:
    nxt = min(datetime.fromisoformat(s["resolve_after"]) for s in pending)
    diff = (nxt - now).total_seconds() / 3600
    print(f'Next: {nxt.strftime("%Y-%m-%d %H:%M UTC")} ({diff:.1f}h away)')
print()
# Direction breakdown v2
if sigs_v2:
    long_v2 = sum(1 for s in sigs_v2 if s.get('direction') == 1)
    short_v2 = sum(1 for s in sigs_v2 if s.get('direction') == -1)
    print(f'V2 direction: LONG={long_v2} SHORT={short_v2} ({long_v2/(len(sigs_v2))*100:.0f}% long)')
