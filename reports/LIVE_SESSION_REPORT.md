# LIVE SESSION REPORT — Multi-Session Log
**Model:** v2c (LGBMRegressor, 65 features, 7-day CS rank label)
**Stage:** LIMITED SHADOW | **Universe:** 285 NSE F&O symbols

> This file is an append log. Each trading day's close-of-day summary is appended by `autorun_till_close.py`.
> Detailed per-session reports are in `reports/live/`.

---

## SESSION HISTORY

| Date | Regime | NIFTY Chg | Tracked | Win Rate | Mean Net | SHORT Mean | LONG Mean | Notes |
|------|--------|-----------|---------|----------|----------|-----------|-----------|-------|
| 2026-09-29 | BEAR | −0.75% | 27 | 57.1% | −0.15% | +1.28% | −3.75% | First SHADOW session |
| 2026-10-06 | BULL | +0.72% | 39 | 59.0% | −0.01% | +1.28% | −3.75% | BULL suppressor active |
| 2026-10-07 | BEAR | −0.75% | 28 | 66.7% | +0.22% | +1.52% | −4.34% | v2c session 7/20 |
| 2026-10-08 | BEAR | −1.71% | 47 | **63.8%** | **+1.12%** | +4.01% | −5.68% | v2c session 8/20; 285 scored; best: ADANIENT SHORT +10.4% |

---

## CLOSE-OF-DAY UPDATE — 2026-10-08 15:40 IST

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
