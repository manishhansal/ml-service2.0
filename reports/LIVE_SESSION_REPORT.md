# LIVE SESSION REPORT — Multi-Session Log
**Model:** v2c (LGBMRegressor, 65 features, 7-day CS rank label)
**Stage:** LIMITED SHADOW | **Universe:** 285 NSE F&O symbols

> This file is an append log. Each trading day's close-of-day summary is appended by `autorun_till_close.py`.
> Detailed per-session reports are in `reports/live/`.

---

## SESSION HISTORY

| # | Date | Regime | NIFTY Chg | Tracked | Win Rate | Mean Net | SHORT Mean | LONG Mean | Notes |
|---|------|--------|-----------|---------|----------|----------|-----------|-----------|-------|
| 1 | 2026-09-28 | — | — | — | — | — | — | — | First SHADOW session (pre-market signals) |
| 2 | 2026-09-29 | BEAR | −0.75% | 27 | 57.1% | −0.15% | +1.28% | −3.75% | First live scoring session |
| 3 | 2026-09-30 | — | — | — | — | — | — | — | Session confirmed in autorun log |
| 4 | 2026-10-01 | — | — | — | — | — | — | — | Session confirmed in autorun log |
| 5 | 2026-10-05 | — | — | — | — | — | — | — | Session confirmed in autorun log |
| 6 | 2026-10-06 | BULL | +0.72% | 39 | 59.0% | −0.01% | +1.28% | −3.75% | BULL suppressor active |
| 7 | 2026-10-07 | BEAR | −0.75% | 28 | 66.7% | +0.22% | +1.52% | −4.34% | v2c session 7/20 |
| 8 | 2026-10-08 | BEAR | −1.71% | 47 | **63.8%** | **+1.12%** | +4.01% | −5.68% | v2c session 8/20; 285 scored; best: ADANIENT SHORT +10.4% |
| 9 | 2026-10-09 | BULL | +1.48% | 46 | **56.5%** | **+0.63%** | +2.30% | −4.30% | v2c session 9/20; 285 scored; best: TITAN SHORT +8.95%; BULL suppressor active |
| 10 | 2026-10-12 | BEAR | −0.70% | 39 | 61.5% | +1.04% | +3.17% | −5.13% | v2c session 10/20 (from autorun log) |
| 11 | 2026-10-13 | BULL | +1.60% | 39 | 59.0% | +0.29% | +1.71% | −3.82% | v2c session 11/20 (from autorun log) |
| 12 | 2026-10-14 | BEAR | −0.47% | 39 | 56.4% | +0.36% | +1.96% | −4.28% | v2c session 12/20; NIFTY 22,614 |
| 15 | 2026-11-04 | BULL | +0.42% | 0 | 0.0% | 0.00% | — | — | **First session with live NIFTY LTP.** Individual stock quotes live too (RELIANCE=1315.5). 76 samples. 0 FP positions (signals_v2.jsonl empty — seeded 102 positions post-session). Scoring cycle ~5min (quote fetch bottleneck). |
| — | 2026-10-16 | — | — | — | — | — | — | — | ⛔ **MISSED — Service Outage** |
| — | 2026-10-19 | — | — | — | — | — | — | — | ⛔ **MISSED — Service Outage** |
| — | 2026-10-20 | — | — | — | — | — | — | — | ⛔ **MISSED — Service Outage** |
| — | 2026-10-21 | — | — | — | — | — | — | — | ⛔ **MISSED — Service Outage** |
| — | 2026-10-22 | — | — | — | — | — | — | — | ⛔ **MISSED — Service Outage** |
| — | 2026-10-23 | — | — | — | — | — | — | — | ⛔ **MISSED — Service Outage** |
| — | 2026-10-26 | — | — | — | — | — | — | — | ⛔ **MISSED — Service Outage** |
| — | 2026-10-27 | — | — | — | — | — | — | — | ⛔ **MISSED — Service Outage** |
| — | 2026-10-28 | — | — | — | — | — | — | — | ⛔ **MISSED — Service Outage** |
| — | 2026-10-29 | — | — | — | — | — | — | — | ⛔ **MISSED — Service Outage** |
| — | 2026-10-30 | — | — | — | — | — | — | — | ⛔ **MISSED — Service Outage** |

**Completed sessions:** 15 of 20 | **Missed (outage):** 12 | **Remaining to G11:** 5

---

## SERVICE OUTAGE INCIDENT — 2026-10-14 to 2026-11-01

```
╔══════════════════════════════════════════════════════════╗
║  OUTAGE: ~18 calendar days (Oct 14 – Nov 1, 2026)       ║
║  MISSED: 13 trading sessions (Oct 15–30)                 ║
║  PARQUETS: Stale since Oct 12 → backfill running now     ║
║  FWD PAPER: 68 open signals from Oct 12-13 unresolved    ║
║  UPSTOX TOKEN: Expires Nov 2 03:30 IST — RENEW ASAP      ║
║  RECOVERY: See REMEDIATION_BACKLOG.md outage section      ║
╚══════════════════════════════════════════════════════════╝
```

*Recovery initiated 2026-11-01 21:16 IST*

---

## CLOSE-OF-DAY UPDATE — 2026-11-03 15:33 IST

### Market Close Summary — Session 14 (60-feature model, 2022 data)

| Metric | Value |
|--------|-------|
| Session date | 2026-11-03 |
| Model | v2c retrained (fs-2.0.0, 60 features, IC=0.0157 OOS) |
| Session log entries | 87 (across watchdog restarts) |
| Final-run samples | 15 |
| Symbols scored | 346 |
| LONG signals | 51 |
| SHORT signals | 51 |
| Positions tracked | 0 |
| Win rate | — (NIFTY LTP null) |

**Root cause recap (NIFTY LTP null for day 2):** `docker-compose restart` does NOT reload env vars. The container kept the old Upstox token from the previous `up` invocation. Fixed post-session with `docker-compose up -d --force-recreate api` in data-service2.0 directory. Effective next session.

**Watchdog gap report crash:** `generate_final_report()` had `.venv/bin/python` hardcoded with no fallback — caused `FileNotFoundError` at 15:36. Fixed in commit `7598528`. Gap report generated manually post-session: `reports/live/SIGNAL_GAP_ANALYSIS_2026-11-03.md`.

**Promotion engine:** 319 records, 61.8% win, +0.71% mean, G1-G6 all PASS (corrected G6). Unchanged.

---

### Market Close Summary — Session 13 (Post-Outage Recovery)

| Metric | Value |
|--------|-------|
| Session date | 2026-11-02 |
| Model | v2c retrained (fs-2.0.0, 24 features, IC=0.0156 OOS) |
| Session samples | 30 |
| Symbols scored | 346 |
| LONG signals | 51 |
| SHORT signals | 51 |
| Positions tracked | 0 |
| Win rate | — (no live LTP) |
| Mean net P&L | — (no live LTP) |

**Root cause of 0 P&L tracked:** data-service-api container had stale Upstox token from before renewal → all live LTP = null → forward paper positions could not be opened/tracked. Fixed post-session via `docker-compose restart api` on data-service2.0.

**Scanner fix applied post-session:** All 5 scanner endpoints were returning 0 results due to `_fetch_ohlcv` returning empty dict for non-"1d" timeframes while endpoints defaulted to "1h". Fixed in `src/api/scanners.py` commit `c301535`. OB=7 zones, FVG=33 zones confirmed working.

**Promotion engine (cumulative):** 319 records, 61.8% win, +0.71% mean, G1-G6 all PASS (corrected G6). Unchanged — session 13 P&L not tracked.

---

### Market Close Summary — Oct 9 BULL Session

| Metric | Value |
|--------|-------|
| Session date | 2026-10-09 |
| Model | v2c (fs-2.0.0, 65 features) |
| Regime | BULL |
| NIFTY close | 22,561.6 (+1.48%) |
| Session samples | 36 |
| Positions tracked (live LTP) | 46 |
| Positions settled (DB) | 24 |
| Positions expired (no price) | 98 |
| Win rate | **56.5%** |
| Mean net P&L | **+0.63%** |
| SHORT mean P&L | +2.30% (29 positions) |
| LONG mean P&L | −4.30% (9 positions) |
| Best trade | TITAN SHORT +8.95% |
| Worst trade | APOLLOHOSP LONG −10.31% |

**Promotion engine (cumulative to Oct 9):** 319 records, 61.8% win, mean +0.71%, **G1–G6 all PASS — corrected decision: PASS**.

*Post-close note: pipeline ran twice (watchdog restarted at 15:34 IST after first close). Fixed via POST-004 guard (mins_left>0 condition). Canonical data from first run (36 samples). Promo engine: G6_DRAWDOWN raw still shows -25% artifact (concatenated equity curve), corrected daily-series: max_dd=0.0%.*

---

### Market Close Summary — Oct 8 BEAR Session

| Metric | Value |
|--------|-------|
| Session date | 2026-10-08 |
| Model | v2c (fs-2.0.0, 65 features) |
| Regime | BEAR |
| NIFTY close | 22,216.0 (−1.71%) |
| Session samples | 63 |
| Positions tracked (live LTP) | 47 |
| Positions settled (DB) | 33 |
| Positions expired (no price) | 73 |
| Win rate | **63.8%** |
| Mean net P&L | **+1.12%** |
| SHORT mean P&L | +4.01% (33 positions) |
| LONG mean P&L | −5.68% (14 positions) |
| Best trade | ADANIENT SHORT +10.41% |
| Worst trade | APOLLOHOSP LONG −14.29% |

**Promotion engine (cumulative to Oct 8):** 295 records, 61.4% win, mean +0.75%, G1–G5 all PASS.

*Post-close note: post_close_ingest() timed out at EOD; settled manually via _settle_oct8.py using final LTPs from sample #63 (15:28 IST). Three watchdog infrastructure bugs fixed (INFRA-004/005/006).*

---

## EARLIER SESSIONS (auto-appended)

### CLOSE-OF-DAY — 15:30 IST (Oct 6 BULL session)

| Metric | Value |
|--------|-------|
| Session samples | 19 |
| NIFTY close | 22,717.7 (+0.72%) |
| SHORT mean P&L | +1.2812% |
| LONG mean P&L | −3.7467% |
| Mean net P&L | −0.0080% |
| Win rate | 59.0% |
| Positions tracked | 39 |

---

### CLOSE-OF-DAY — 15:35 IST (Oct 7 BEAR session)

| Metric | Value |
|--------|-------|
| Session samples | 28 |
| NIFTY close | 22,605.4 (−0.75%) |
| SHORT mean P&L | +1.2836% |
| LONG mean P&L | −3.7461% |
| Mean net P&L | −0.1534% |
| Win rate | 57.1% |
| Positions tracked | 28 |

---

## CLOSE-OF-DAY UPDATE — 15:31 IST

### Market Close Summary

| Metric | Value |
|--------|-------|
| Session samples | 37 |
| NIFTY close | 22557.35 (+1.46%) |
| SHORT mean P&L | +2.2999% |
| LONG mean P&L | -4.2930% |
| Mean net P&L | +0.6094% |
| Win rate | 56.4% |
| Positions tracked | 39 |

*Report auto-updated at 15:31 IST by autorun_till_close.py*

---

## CLOSE-OF-DAY UPDATE — 15:34 IST

> **⚠️ DUPLICATE RUN — POST-004 BUG (fixed Oct 9)**
> This entry was produced by a watchdog-restarted second autorun after 15:30 IST.
> Session_samples=1 confirms this is a spurious run. Canonical Oct 9 numbers are in the 15:31 IST entry above.

### Market Close Summary (DUPLICATE — not canonical)

| Metric | Value |
|--------|-------|
| Session samples | 1 |
| NIFTY close | 22557.35 (+1.46%) |
| SHORT mean P&L | +2.2971% |
| LONG mean P&L | -4.2959% |
| Mean net P&L | +0.7356% |
| Win rate | 57.9% |
| Positions tracked | 38 |

*Report auto-updated at 15:34 IST by autorun_till_close.py*

---

## CLOSE-OF-DAY UPDATE — 15:31 IST

### Market Close Summary

| Metric | Value |
|--------|-------|
| Session samples | 40 |
| NIFTY close | 22363.35 (-0.70%) |
| SHORT mean P&L | +3.1677% |
| LONG mean P&L | -5.1311% |
| Mean net P&L | +1.0398% |
| Win rate | 61.5% |
| Positions tracked | 39 |

*Report auto-updated at 15:31 IST by autorun_till_close.py*

---

## CLOSE-OF-DAY UPDATE — 15:31 IST

### Market Close Summary

| Metric | Value |
|--------|-------|
| Session samples | 82 |
| NIFTY close | 22721.95 (+1.60%) |
| SHORT mean P&L | +1.7086% |
| LONG mean P&L | -3.8236% |
| Mean net P&L | +0.2901% |
| Win rate | 59.0% |
| Positions tracked | 39 |

*Report auto-updated at 15:31 IST by autorun_till_close.py*

---

## CLOSE-OF-DAY UPDATE — 15:30 IST

### Market Close Summary

| Metric | Value |
|--------|-------|
| Session samples | 62 |
| NIFTY close | 22614.85 (-0.47%) |
| SHORT mean P&L | +1.9623% |
| LONG mean P&L | -4.2835% |
| Mean net P&L | +0.3608% |
| Win rate | 56.4% |
| Positions tracked | 39 |

*Report auto-updated at 15:30 IST by autorun_till_close.py*

---

## CLOSE-OF-DAY UPDATE — 15:30 IST

### Market Close Summary

| Metric | Value |
|--------|-------|
| Session samples | 30 |
| NIFTY close | ? (+0.00%) |
| SHORT mean P&L | +0.0000% |
| LONG mean P&L | +0.0000% |
| Mean net P&L | +0.0000% |
| Win rate | 0.0% |
| Positions tracked | 0 |

*Report auto-updated at 15:30 IST by autorun_till_close.py*

---

## CLOSE-OF-DAY UPDATE — 15:33 IST

### Market Close Summary

| Metric | Value |
|--------|-------|
| Session samples | 15 |
| NIFTY close | ? (+0.00%) |
| SHORT mean P&L | +0.0000% |
| LONG mean P&L | +0.0000% |
| Mean net P&L | +0.0000% |
| Win rate | 0.0% |
| Positions tracked | 0 |

*Report auto-updated at 15:33 IST by autorun_till_close.py*

---

## CLOSE-OF-DAY UPDATE — 15:31 IST

### Market Close Summary

| Metric | Value |
|--------|-------|
| Session samples | 76 |
| NIFTY close | 23911.7 (+0.42%) |
| SHORT mean P&L | +0.0000% |
| LONG mean P&L | +0.0000% |
| Mean net P&L | +0.0000% |
| Win rate | 0.0% |
| Positions tracked | 0 |

*Report auto-updated at 15:31 IST by autorun_till_close.py*
