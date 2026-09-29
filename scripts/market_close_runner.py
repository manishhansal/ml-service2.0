#!/usr/bin/env python3
"""
Runs until 15:30 IST market close, then:
  1. Collects P&L snapshots every 10 min
  2. At close: ingests Sep 28 bars
  3. Runs forward paper resolution
  4. Updates LIVE_SESSION_REPORT.md
"""
from __future__ import annotations
import json, subprocess, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE = Path('/Users/manishkumar/Desktop/ml-service2.0')


def ist_now() -> datetime:
    return datetime.now(tz=timezone.utc) + timedelta(hours=5, minutes=30)


def mins_to_close() -> float:
    now = ist_now()
    close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return (close - now).total_seconds() / 60


def market_open() -> bool:
    now = ist_now()
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return (9, 15) <= t <= (15, 30)


def run_pnl_snapshot():
    """Run P&L tracker and return snapshot."""
    result = subprocess.run(
        ['python3', '-W', 'ignore', str(BASE/'scripts/track_live_pnl.py')],
        capture_output=True, text=True, cwd=str(BASE),
        env={**__import__('os').environ, 'PYTHONPATH': str(BASE)},
    )
    stdout = result.stdout
    # Extract key metrics
    metrics = {}
    for line in stdout.splitlines():
        if 'Mean net P&L:' in line:
            metrics['mean_net_pnl'] = line.split()[-1]
        if 'Win rate:' in line:
            metrics['win_rate'] = line.split()[2]
        if 'SHORT mean P&L:' in line:
            metrics['short_pnl'] = line.split()[-1]
        if 'LONG mean P&L:' in line:
            metrics['long_pnl'] = line.split()[-1]
    return metrics, stdout


def ingest_fresh_data():
    """Ingest Sep 28 closing bars."""
    print('[runner] Ingesting Sep 28 closing bars...')
    result = subprocess.run(
        ['python3', '-W', 'ignore', str(BASE/'scripts/ingest_all_outdated.py')],
        capture_output=True, text=True, cwd=str(BASE),
        env={**__import__('os').environ, 'PYTHONPATH': str(BASE)},
    )
    print(result.stdout[-500:] if result.stdout else 'No output')


def resolve_signals():
    """Resolve forward paper signals."""
    print('[runner] Resolving forward paper signals...')
    result = subprocess.run(
        ['python3', '-W', 'ignore', str(BASE/'scripts/resolve_forward_paper.py')],
        capture_output=True, text=True, cwd=str(BASE),
        env={**__import__('os').environ, 'PYTHONPATH': str(BASE)},
    )
    print(result.stdout[-800:] if result.stdout else 'No output')
    return result.stdout


def main():
    print(f'[runner] Market close runner started — {ist_now().strftime("%H:%M IST")}')
    print(f'[runner] Market close in {mins_to_close():.0f} minutes')

    snapshots = []
    sample_log = BASE / 'artifacts/live_session/session_log.jsonl'
    last_pnl_time = None

    while True:
        now = ist_now()
        mins_left = mins_to_close()

        # Take P&L snapshot every 10 minutes
        if last_pnl_time is None or (now - last_pnl_time).total_seconds() >= 600:
            print(f'\n[runner] {now.strftime("%H:%M IST")} | {mins_left:.0f}min to close')
            metrics, output = run_pnl_snapshot()
            snapshots.append({
                'time': now.strftime('%H:%M IST'),
                'mins_left': round(mins_left, 1),
                **metrics,
            })
            print(f'  P&L: mean={metrics.get("mean_net_pnl","?")} win={metrics.get("win_rate","?")} short={metrics.get("short_pnl","?")}')
            last_pnl_time = now

        if mins_left <= 0:
            print('\n[runner] MARKET CLOSED — 15:30 IST')
            break

        # Sleep until next check (min of 10min or time to close)
        sleep_secs = min(600, max(30, int((mins_left - 0.5) * 60)))
        if sleep_secs > 0:
            time.sleep(sleep_secs)

    # === POST-CLOSE ACTIONS ===
    print('\n[runner] === POST-CLOSE ACTIONS ===')
    time.sleep(300)  # 5 min buffer for prices to settle

    # 1. Ingest Sep 28 bars
    ingest_fresh_data()

    # 2. Resolve signals
    resolution_output = resolve_signals()

    # 3. Final P&L snapshot
    print('\n[runner] Final P&L snapshot...')
    metrics, output = run_pnl_snapshot()
    snapshots.append({
        'time': 'CLOSE',
        'mins_left': 0,
        'is_close': True,
        **metrics,
    })

    # 4. Save runner summary
    summary = {
        'session_date': '2026-09-28',
        'close_time': '15:30 IST',
        'n_snapshots': len(snapshots),
        'snapshots': snapshots,
        'final_metrics': metrics,
        'resolution_output': resolution_output[-500:] if resolution_output else '',
    }
    (BASE/'artifacts/live_session/runner_summary.json').write_text(
        json.dumps(summary, indent=2, default=str))

    print(f'\n[runner] Complete. Snapshots: {len(snapshots)}')
    print(f'Final mean P&L: {metrics.get("mean_net_pnl","?")}')
    print(f'Final win rate: {metrics.get("win_rate","?")}')


if __name__ == '__main__':
    main()
