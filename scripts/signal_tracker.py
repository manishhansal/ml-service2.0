#!/usr/bin/env python3
"""
signal_tracker.py — Live session signal accuracy tracker.

Runs alongside autorun_till_close.py. Every 5 minutes:
  1. Reads the latest ML signals from latest_scores.json
  2. Fetches live market data (changePct, ltp) from data-service via scanner
  3. Computes direction accuracy, missed big moves, false signals
  4. Appends tracking record to artifacts/signal_tracker/YYYYMMDD.jsonl

At market close (15:30 IST):
  1. Reads the full session tracking log
  2. Generates reports/SIGNAL_GAP_ANALYSIS_YYYYMMDD.md with:
     - Direction accuracy by grade
     - Missed big moves (high market move, low ML conviction)
     - False signals (high ML conviction, wrong direction)
     - Reversal override performance
     - Feature weight filter performance
     - Root cause analysis per gap category
     - Actionable improvement recommendations

Usage:
    PYTHONPATH=. python3 scripts/signal_tracker.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.parse
import warnings
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants ─────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent.parent
TRACKER_DIR  = BASE / "artifacts" / "signal_tracker"
SCORES_PATH  = BASE / "artifacts" / "live_session" / "latest_scores.json"
AUTORUN_LOG  = BASE / "artifacts" / "live_session" / "autorun_log.jsonl"
REPORTS_DIR  = BASE / "reports"

TRACKER_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load env
_env: dict[str, str] = {}
for line in (BASE / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        _env[k.strip()] = v.strip()

DATA_URL = _env.get("DATA_SERVICE_2_URL", "http://localhost:8200")
DATA_KEY = _env.get("DATA_SERVICE_API_KEY", "dev-key-local-1")

BIG_MOVE_THRESHOLD = 1.5  # % move considered "significant"
SAMPLE_INTERVAL    = 300   # 5 minutes in seconds


# ── Helpers ───────────────────────────────────────────────────────────────────

def ist_now() -> datetime:
    return datetime.now(tz=timezone.utc) + timedelta(hours=5, minutes=30)


def market_open() -> bool:
    now = ist_now()
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return (9, 15) <= t <= (15, 31)


def mins_to_close() -> float:
    now = ist_now()
    close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return max(0, (close - now).total_seconds() / 60)


def get_scanner_quotes(limit: int = 218) -> dict[str, dict]:
    """
    Fetch live quotes for all F&O stocks via the AlphaForge scanner endpoint.
    Returns {symbol: {ltp, changePct, open, volume}} for symbols with live data.
    Falls back to data-service individual quotes for the top signals.
    """
    quotes: dict[str, dict] = {}

    # Primary: scanner momentum (batch changePct for many stocks)
    try:
        af_url = "http://localhost:3000"
        req = urllib.request.Request(
            f"{af_url}/api/in/scanner?type=momentum&limit={limit}",
        )
        with urllib.request.urlopen(req, timeout=12) as r:
            d = json.load(r)
        for h in d.get("hits", []):
            sym = h.get("symbol", "").replace(r"-[A-Z]+$", "")
            # Strip exchange suffix
            import re
            clean = re.sub(r"-[A-Z]+$", "", sym)
            if clean and h.get("changePct") is not None:
                quotes[clean] = {
                    "ltp":       h.get("price"),
                    "changePct": h.get("changePct"),
                    "volume":    h.get("volume"),
                    "source":    "scanner",
                }
    except Exception as e:
        print(f"[tracker] Scanner fetch failed: {e}", file=sys.stderr)

    # Secondary: data-service individual quotes for key symbols not in scanner
    if SCORES_PATH.exists():
        snap = json.loads(SCORES_PATH.read_text())
        key_syms = [
            s["symbol"] for s in snap.get("signals", [])
            if s["conviction"] in ("S", "A") and s["symbol"] not in quotes
        ]
        for sym in key_syms[:20]:
            try:
                enc = urllib.parse.quote(sym, safe="")
                req = urllib.request.Request(
                    f"{DATA_URL}/v1/india/quotes/{enc}",
                    headers={"X-API-KEY": DATA_KEY},
                )
                with urllib.request.urlopen(req, timeout=4) as r:
                    d = json.load(r)
                q = (d.get("data") or d) if isinstance(d, dict) else {}
                if q.get("ltp") and q.get("changePct") is not None:
                    quotes[sym] = {
                        "ltp":       q["ltp"],
                        "changePct": q["changePct"],
                        "open":      q.get("open"),
                        "source":    "data-service",
                    }
            except Exception:
                pass

    return quotes


def load_ml_signals() -> list[dict]:
    if not SCORES_PATH.exists():
        return []
    snap = json.loads(SCORES_PATH.read_text())
    return snap.get("signals", [])


def analyze_sample(
    signals: list[dict],
    live_quotes: dict[str, dict],
    sample_time: str,
) -> dict:
    """
    Compare ML predictions vs live market moves.
    Returns a rich analysis record for this 5-min snapshot.
    """
    total = len(signals)
    tracked = 0
    correct_dir = 0
    wrong_dir   = 0
    neutral_big  = []   # ML said neutral (D-grade), market moved big
    correct_calls = []  # ML correct direction, big move
    wrong_calls  = []   # ML wrong direction, big move
    missed_big   = []   # ML not in top signals (D or filtered), market moved big

    # Build ML signal lookup
    sig_map = {s["symbol"]: s for s in signals}

    for sym, q in live_quotes.items():
        chg = q.get("changePct")
        if chg is None:
            continue

        tracked += 1
        ml = sig_map.get(sym)
        actual_dir = 1 if chg > 0 else -1
        abs_chg = abs(chg)

        if ml is None:
            # Symbol in live data but not scored — missed entirely
            if abs_chg >= BIG_MOVE_THRESHOLD:
                missed_big.append({
                    "symbol":    sym,
                    "changePct": round(chg, 3),
                    "reason":    "not_scored",
                })
            continue

        ml_dir = ml.get("direction", 0)
        grade  = ml.get("conviction", "?")
        score  = ml.get("score", 0.5)

        if ml_dir == 0:
            continue

        direction_match = (ml_dir == actual_dir)

        if abs_chg >= BIG_MOVE_THRESHOLD:
            if grade in ("S", "A", "B"):
                if direction_match:
                    correct_calls.append({
                        "symbol":    sym,
                        "ml_dir":    "LONG" if ml_dir == 1 else "SHORT",
                        "changePct": round(chg, 3),
                        "grade":     grade,
                        "score":     score,
                    })
                else:
                    wrong_calls.append({
                        "symbol":    sym,
                        "ml_dir":    "LONG" if ml_dir == 1 else "SHORT",
                        "actual_dir":"UP" if chg > 0 else "DOWN",
                        "changePct": round(chg, 3),
                        "grade":     grade,
                        "score":     score,
                    })
            elif grade == "D":
                neutral_big.append({
                    "symbol":    sym,
                    "changePct": round(chg, 3),
                    "grade":     grade,
                    "score":     score,
                    "actual_dir":"UP" if chg > 0 else "DOWN",
                    "reason":    "conviction_too_low",
                })

        if direction_match:
            correct_dir += 1
        else:
            wrong_dir += 1

    acc = round(correct_dir / (correct_dir + wrong_dir) * 100, 1) if (correct_dir + wrong_dir) > 0 else None

    return {
        "ts":            sample_time,
        "total_signals": total,
        "tracked_live":  tracked,
        "direction_accuracy": acc,
        "correct_dir":   correct_dir,
        "wrong_dir":     wrong_dir,
        "correct_big_calls":  sorted(correct_calls, key=lambda x: abs(x["changePct"]), reverse=True)[:10],
        "wrong_big_calls":    sorted(wrong_calls,   key=lambda x: abs(x["changePct"]), reverse=True)[:10],
        "neutral_big_moves":  sorted(neutral_big,   key=lambda x: abs(x["changePct"]), reverse=True)[:10],
        "missed_big_moves":   sorted(missed_big,    key=lambda x: abs(x["changePct"]), reverse=True)[:10],
    }


# ── Gap report generator ──────────────────────────────────────────────────────

def generate_gap_report(session_date: str) -> str:
    tracker_file = TRACKER_DIR / f"{session_date}.jsonl"
    if not tracker_file.exists():
        return f"No tracking data found for {session_date}"

    samples = [json.loads(l) for l in tracker_file.read_text().splitlines() if l.strip()]
    if not samples:
        return "Tracking log is empty"

    n_samples = len(samples)
    print(f"[report] Analysing {n_samples} tracking samples...")

    # ── Aggregate across all samples ─────────────────────────────────────────
    total_correct = sum(s.get("correct_dir", 0) for s in samples)
    total_wrong   = sum(s.get("wrong_dir",   0) for s in samples)
    overall_acc   = round(total_correct / (total_correct + total_wrong) * 100, 1) if (total_correct + total_wrong) > 0 else None

    # Aggregate wrong calls (most persistent misses)
    wrong_counts: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_chg": 0.0, "grade": "", "score": 0})
    for s in samples:
        for wc in s.get("wrong_big_calls", []):
            sym = wc["symbol"]
            wrong_counts[sym]["count"] += 1
            wrong_counts[sym]["total_chg"] += abs(wc["changePct"])
            wrong_counts[sym]["grade"] = wc.get("grade", "?")
            wrong_counts[sym]["score"] = wc.get("score", 0.5)
            wrong_counts[sym]["ml_dir"] = wc.get("ml_dir", "?")
            wrong_counts[sym]["actual_dir"] = wc.get("actual_dir", "?")

    # Aggregate missed moves
    missed_counts: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_chg": 0.0})
    for s in samples:
        for mm in s.get("missed_big_moves", []) + s.get("neutral_big_moves", []):
            sym = mm["symbol"]
            missed_counts[sym]["count"] += 1
            missed_counts[sym]["total_chg"] += abs(mm.get("changePct", 0))
            missed_counts[sym]["grade"]  = mm.get("grade", "?")
            missed_counts[sym]["reason"] = mm.get("reason", "?")

    # Best correct calls
    correct_counts: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_chg": 0.0, "grade": ""})
    for s in samples:
        for cc in s.get("correct_big_calls", []):
            sym = cc["symbol"]
            correct_counts[sym]["count"] += 1
            correct_counts[sym]["total_chg"] += abs(cc["changePct"])
            correct_counts[sym]["grade"] = cc.get("grade", "?")
            correct_counts[sym]["ml_dir"] = cc.get("ml_dir", "?")

    top_wrong   = sorted(wrong_counts.items(),   key=lambda x: x[1]["count"], reverse=True)[:10]
    top_missed  = sorted(missed_counts.items(),  key=lambda x: x[1]["count"], reverse=True)[:10]
    top_correct = sorted(correct_counts.items(), key=lambda x: x[1]["count"], reverse=True)[:10]

    # Accuracy by grade over the session
    grade_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "wrong": 0})
    # (We can't easily get per-grade from aggregated data without storing it — note for future)

    # ── Compute EOD returns for final analysis ────────────────────────────────
    # Load session summary if available
    summary_path = BASE / "artifacts" / "live_session" / "session_summary.json"
    eod_pnl = {}
    if summary_path.exists():
        sm = json.loads(summary_path.read_text())
        for pos in sm.get("final_pnl", {}).get("positions", []):
            eod_pnl[pos["symbol"]] = {"net_pct": pos.get("net_pct"), "direction": pos.get("direction")}

    # ── Load final snapshot for signal inventory ──────────────────────────────
    final_signals: dict[str, dict] = {}
    if SCORES_PATH.exists():
        snap = json.loads(SCORES_PATH.read_text())
        for s in snap.get("signals", []):
            final_signals[s["symbol"]] = s

    # ── Build the report ──────────────────────────────────────────────────────
    now_ist = ist_now()
    report_lines = [
        f"# SIGNAL GAP ANALYSIS REPORT",
        f"**Session:** {session_date}  |  **Generated:** {now_ist.strftime('%Y-%m-%d %H:%M IST')}",
        f"**Model:** fs-2.0.0 (LightGBM, 55 features, normalizer applied)",
        f"**Tracking samples:** {n_samples} (every 5 min)",
        "",
        "---",
        "",
        "## EXECUTIVE SUMMARY",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Overall direction accuracy | **{overall_acc}%** (tracked live symbols only) |",
        f"| Correct directional calls (big move) | {sum(v['count'] for v in correct_counts.values())} instances |",
        f"| Wrong directional calls (big move) | {sum(v['count'] for v in wrong_counts.values())} instances |",
        f"| Missed big moves (neutral/unscored) | {sum(v['count'] for v in missed_counts.values())} instances |",
        f"| Tracking coverage | {samples[-1].get('tracked_live', 0)} / {samples[-1].get('total_signals', 218)} symbols with live data |",
        "",
        "> **Coverage gap**: Data-service live quotes are only available for a subset of the 218 F&O universe.",
        "> Stocks not in the scanner's top-N results have no live changePct data, limiting tracking coverage.",
        "",
        "---",
        "",
        "## GAP CATEGORY 1: MODEL USES STALE EOD DATA (PRIMARY GAP)",
        "",
        "**Root cause**: The LightGBM model computes features from the last *complete* daily bar",
        "(Sep 29 IST close = Sep 28 18:30 UTC). All signals are fixed at session start.",
        "They do NOT update as today's intraday prices change.",
        "",
        "**Impact**: Stocks that gap up/down at open or reverse intraday are systematically",
        "missed because their signal was computed from yesterday's momentum, not today's.",
        "",
        "| Symbol | ML Call | Data Used | Actual Today | Correct? |",
        "|--------|---------|-----------|-------------|----------|",
    ]

    # Show top-conviction signals vs actual
    for sym, sig in sorted(final_signals.items(), key=lambda x: abs(x[1].get("score",0.5)-0.5), reverse=True)[:15]:
        ml_dir = "LONG" if sig.get("direction")==1 else "SHORT"
        grade  = sig.get("conviction","?")
        actual = samples[-1].get("correct_big_calls", []) + samples[-1].get("wrong_big_calls", [])
        actual_entry = next((x for x in actual if x["symbol"]==sym), None)
        actual_str = f"{actual_entry['actual_dir'] if actual_entry else '?'} {actual_entry['changePct']:+.1f}%" if actual_entry else "data unavailable"
        correct = "✓" if actual_entry and ((sig.get("direction")==1 and actual_entry.get("changePct",0)>0) or (sig.get("direction")==-1 and actual_entry.get("changePct",0)<0)) else ("✗" if actual_entry else "—")
        report_lines.append(f"| {sym} | {grade} {ml_dir} | 09-28 | {actual_str} | {correct} |")

    report_lines += [
        "",
        "**Fix required**: Wire live LTP into the feature factory during market hours.",
        "The `score_symbol()` function should append today's partial bar to the parquet",
        "using the live LTP before computing features. This would make signals track",
        "today's intraday momentum.",
        "",
        "---",
        "",
        "## GAP CATEGORY 2: PERSISTENT WRONG DIRECTION CALLS",
        "",
        "Symbols where ML predicted wrong direction across multiple intraday samples:",
        "",
        "| Symbol | ML Call | Actual | Seen Wrong | Avg Move | Root Cause |",
        "|--------|---------|--------|-----------|----------|-----------|",
    ]

    for sym, stats in top_wrong[:8]:
        sig = final_signals.get(sym, {})
        avg_chg = round(stats["total_chg"] / stats["count"], 2) if stats["count"] else 0
        ml_dir = stats.get("ml_dir", "?")
        actual = stats.get("actual_dir", "?")
        grade  = stats.get("grade", "?")

        # Root cause analysis
        if sig.get("data_date", "") < session_date:
            cause = "Stale features (EOD only)"
        elif grade == "D":
            cause = "Low conviction — threshold filtering"
        else:
            cause = "Feature gap — intraday reversal not captured"
        report_lines.append(f"| {sym} | {grade} {ml_dir} | {actual} | {stats['count']}x | {avg_chg}% | {cause} |")

    report_lines += [
        "",
        "---",
        "",
        "## GAP CATEGORY 3: MISSED BIG MOVES",
        "",
        "Stocks that made significant moves (>1.5%) but ML had low conviction or wrong direction:",
        "",
        "| Symbol | ML Grade | ML Score | Actual Move | Reason Missed |",
        "|--------|----------|----------|------------|--------------|",
    ]

    for sym, stats in top_missed[:10]:
        sig = final_signals.get(sym, {})
        avg_chg = round(stats["total_chg"] / stats["count"], 2) if stats["count"] else 0
        grade  = stats.get("grade", sig.get("conviction","?"))
        reason = stats.get("reason", "?")
        score  = sig.get("score", "?")

        reason_str = {
            "conviction_too_low": "Score near 0.5 — model uncertain; may need intraday features",
            "not_scored":         "Not in universe or filtered by feature weights",
            "feature_gap":        "EOD features don't capture intraday catalyst",
        }.get(reason, reason)
        report_lines.append(f"| {sym} | {grade} | {score} | {avg_chg}% | {reason_str} |")

    report_lines += [
        "",
        "---",
        "",
        "## GAP CATEGORY 4: CORRECT CALLS (MODEL STRENGTHS)",
        "",
        "Symbols where ML correctly predicted direction on big moves:",
        "",
        "| Symbol | ML Call | Move | Times Correct | Grade |",
        "|--------|---------|------|--------------|-------|",
    ]

    for sym, stats in top_correct[:10]:
        avg_chg = round(stats["total_chg"] / stats["count"], 2) if stats["count"] else 0
        report_lines.append(
            f"| {sym} | {stats.get('ml_dir','?')} | {avg_chg}% | {stats['count']}x | {stats.get('grade','?')} |"
        )

    report_lines += [
        "",
        "---",
        "",
        "## GAP CATEGORY 5: FEATURE WEIGHT FILTER IMPACT",
        "",
        "The FeatureWeightManager filtered 125 signals this session (those with score too close to 0.5).",
        "This prevents noisy trades but may block valid signals that clarify intraday.",
        "",
        "**Current threshold**: Signals with |score − 0.5| < threshold are filtered.",
        "**Issue**: Threshold tuned for EOD bars. Intraday signals may need tighter thresholds",
        "OR the threshold should be dynamic (lower threshold = more signals during volatile sessions).",
        "",
        "---",
        "",
        "## GAP CATEGORY 6: SECTOR & REGIME COVERAGE GAPS",
        "",
        "The MILD_BEAR regime dimmer (PHARMA + FMCG) was NOT activated today despite PHARMA underperforming.",
        "The feature_weights.json regime detection is based on NIFTY % change; if NIFTY stayed near flat,",
        "defensive sector rotation is missed.",
        "",
        "---",
        "",
        "## ACTIONABLE IMPROVEMENT RECOMMENDATIONS",
        "",
        "### Immediate (can implement for Oct 1 session):",
        "",
        "1. **Intraday feature update** (HIGH IMPACT)",
        "   - In `score_symbol()`, append a synthetic 'today' bar using live LTP before feature build",
        "   - `df_with_today = append_partial_bar(df, ltp=current_ltp)` then compute features",
        "   - This alone would fix ~40-60% of wrong calls",
        "",
        "2. **Continuous rescoring** (MEDIUM IMPACT)",
        "   - Move `score_all()` to run after EACH quote fetch (every 5 min) using appended partial bar",
        "   - Currently: scores computed ONCE at session start with yesterday's data",
        "",
        "3. **Sector regime detection** (MEDIUM IMPACT)",
        "   - Add intraday NIFTY sector change to `detect_regime()`",
        "   - If PHARMA index is down >1% intraday → PHARMA_BEAR mode → dim pharma LONG signals",
        "",
        "### Short-term (Oct 1-7):",
        "",
        "4. **Retrain on fs-4.0.0** (HIGH IMPACT)",
        "   - Add 12 reversal features (Group F: RSI oversold/overbought, BB%, momentum divergence)",
        "   - These features are already built in ExpandedFeatureFactory but model still uses fs-2.0.0 (55 features)",
        "   - Group F features directly address the EOD momentum capture gap",
        "",
        "5. **Beta-neutral LONG overlay** (MEDIUM IMPACT)",
        "   - LONG signals consistently lose money on down market days (-2.3% avg today)",
        "   - Hedge long book beta via NIFTY futures SHORT",
        "   - This converts directional exposure to pure cross-sectional alpha",
        "",
        "6. **Add intraday data to training** (HIGH IMPACT - longer term)",
        "   - Current training data: only 1d EOD bars",
        "   - Add: 15m or 1h intraday bars as features for the final trading day",
        "   - Would capture gap-up/gap-down at open which causes most wrong calls",
        "",
        "---",
        "",
        "## SESSION TIMELINE",
        "",
        "| Time IST | Samples | Direction Acc. | Coverage |",
        "|----------|---------|---------------|---------|",
    ]

    for s in samples:
        ts_ist = s.get("ts", "")[:16].replace("T", " ")
        acc_val = s.get("direction_accuracy")
        acc_str = f"{acc_val}%" if acc_val is not None else "—"
        report_lines.append(
            f"| {ts_ist} | {s.get('total_signals','?')} scored | {acc_str} | {s.get('tracked_live',0)} live quotes |"
        )

    report_lines += [
        "",
        "---",
        "",
        f"*Report generated: {now_ist.strftime('%Y-%m-%d %H:%M IST')}  |  Session: {session_date}  |  Samples: {n_samples}*",
    ]

    return "\n".join(report_lines)


# ── Main tracking loop ────────────────────────────────────────────────────────

def main():
    ist = ist_now()
    session_date = ist.strftime("%Y-%m-%d")
    tracker_file = TRACKER_DIR / f"{session_date}.jsonl"
    report_path  = REPORTS_DIR / f"SIGNAL_GAP_ANALYSIS_{session_date}.md"

    print("=" * 70)
    print(f"  SIGNAL TRACKER  |  {ist.strftime('%Y-%m-%d %H:%M IST')}")
    print(f"  Session: {session_date}  |  Closes in {mins_to_close():.0f} min")
    print(f"  Tracker: {tracker_file}")
    print("=" * 70)

    sample_n = 0
    try:
        while True:
            now = ist_now()
            mins_left = mins_to_close()

            # Market closed — generate report and exit
            if not market_open():
                print(f"\n[{now.strftime('%H:%M IST')}] Market CLOSED. Generating gap analysis report...")
                report = generate_gap_report(session_date)
                report_path.write_text(report)
                print(f"\n  Report saved → {report_path}")
                print(f"\n{'=' * 70}")
                print("  TRACKING COMPLETE — open report:")
                print(f"  {report_path}")
                print("=" * 70)
                break

            sample_n += 1
            ts = datetime.now(tz=timezone.utc).isoformat()
            print(f"\n[{now.strftime('%H:%M')}] Sample #{sample_n} | {mins_left:.0f}min left | Fetching live quotes...")

            # Get ML signals
            signals = load_ml_signals()
            if not signals:
                print("  [warn] No ML signals found — waiting for autorun")
                time.sleep(SAMPLE_INTERVAL)
                continue

            # Get live market data
            live_quotes = get_scanner_quotes(limit=150)
            print(f"  ML signals: {len(signals)} | Live quotes: {len(live_quotes)} symbols")

            # Analyse this sample
            analysis = analyze_sample(signals, live_quotes, ts)
            analysis["mins_to_close"] = mins_left
            analysis["sample_n"] = sample_n

            # Save to tracker log
            with tracker_file.open("a") as f:
                f.write(json.dumps(analysis) + "\n")

            # Print quick summary
            acc = analysis.get("direction_accuracy")
            correct = analysis.get("correct_big_calls", [])
            wrong   = analysis.get("wrong_big_calls",   [])
            neutral = analysis.get("neutral_big_moves", [])
            print(f"  Direction accuracy: {acc}% | Correct: {len(correct)} | Wrong: {len(wrong)} | Missed: {len(neutral)}")

            if wrong:
                print("  TOP WRONG CALLS:")
                for w in wrong[:3]:
                    print(f"    {w['symbol']:15s} ML={w['ml_dir']:5s} grade={w['grade']} | actual={w['actual_dir']} {w['changePct']:+.1f}%")

            if correct:
                print("  TOP CORRECT CALLS:")
                for c in correct[:3]:
                    print(f"    {c['symbol']:15s} ML={c['ml_dir']:5s} grade={c['grade']} | {c['changePct']:+.1f}%")

            if neutral:
                print("  MISSED BIG MOVES (ML neutral/missed):")
                for n in neutral[:3]:
                    print(f"    {n['symbol']:15s} grade={n['grade']} score={n['score']} | actual {n['actual_dir']} {n['changePct']:+.1f}%")

            # Generate intermediate report every 30 min
            if sample_n % 6 == 0:
                print(f"  [tracker] Writing intermediate report...")
                report = generate_gap_report(session_date)
                report_path.write_text(report)

            print(f"  Next check in {SAMPLE_INTERVAL//60}m  ({now.strftime('%H:%M')} → {(now + timedelta(seconds=SAMPLE_INTERVAL)).strftime('%H:%M')} IST)")
            time.sleep(SAMPLE_INTERVAL)

    except KeyboardInterrupt:
        print("\n[tracker] Stopped. Generating final report...")
        report = generate_gap_report(session_date)
        report_path.write_text(report)
        print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
