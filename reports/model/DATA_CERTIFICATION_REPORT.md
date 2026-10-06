# DATA CERTIFICATION REPORT
**AlphaForge ml-service2.0 — Data Pipeline Certification**
**Date:** 2026-09-29 (post-close) | **Status:** CERTIFIED | **Revision:** v4.0

---

## Data Sources

| Source | Role | Status |
|--------|------|--------|
| data-service2.0 (port 8200) | Sole market data authority | ✓ VERIFIED |
| SentinelPulse (port 3001) | News/NLP context | ✓ VERIFIED |
| On-disk parquets (`data/1d/1d/`) | Primary training + scoring source | ✓ VERIFIED |

---

## Universe Coverage

| Metric | Value |
|--------|-------|
| NSE F&O universe | 218 / ~220 (99%) |
| Parquet data range | 2021-09-20 → 2026-09-25 |
| Total rows | ~394,638 (5yr daily) |
| Sep 28-29 bars | **Not yet synced** — ingest Oct 1 |
| TATAMOTORS DVR flag | Excluded (226% price mismatch) |

**Sep 29 session note:** Data still uses Sep 23-24 parquets for scoring. Live LTP quotes flowed correctly from data-service via `/v1/india/quotes/{symbol}` endpoint (restored after restart). Angel One re-authenticated with TOTP at startup.

---

## PIT Compliance

| Contract | Status |
|----------|--------|
| Features causal at bar i | ✓ 0 INVALID in CI |
| Sep 29 scoring used Sep 23-24 data | ✓ Correct |
| Forward paper entry prices | ✓ next_open after signal_ts |

---

## Action Required

```bash
make ingest          # Oct 1: refresh Sep 28-29-30 bars
make universe-coverage  # Verify all 218 updated
```

*Generated: 2026-09-29 | Universe: 218/220 NSE F&O*

---

## Data Sources

| Source | Role | Status |
|--------|------|--------|
| data-service2.0 (port 8200) | Sole market data authority | ✓ VERIFIED — exclusive path enforced |
| SentinelPulse (port 3001) | News/NLP context | ✓ VERIFIED — degraded-mode fallback |
| On-disk parquets (`data/1d/1d/`) | Primary training data source | ✓ VERIFIED — 218 symbols, 5yr daily |
| Direct provider access | FORBIDDEN | ✓ VERIFIED — no direct calls in codebase |
| Yahoo Finance / external scrapers | FORBIDDEN | ✓ VERIFIED — not present |

---

## Universe

| Metric | Value |
|--------|-------|
| NSE F&O universe (expected) | ~220 symbols |
| Symbols on disk | **218** |
| Coverage | **99%** |
| Date range | 2021-09-20 → 2026-09-25 |
| Total parquet rows | ~394,638 |
| Symbols with Sep 23-25 data | 182 (83%) |
| Symbols at Sep 22 | 36 (17%) — pending Oct 1 refresh |
| Sep 28 bars in data-service | Not yet (Upstox sync pending Oct 1) |

---

## Data Quality Flags

| Symbol | Issue | Status |
|--------|-------|--------|
| TATAMOTORS | Upstox historical DVR price (~295) vs Angel One regular (~961) — 226% mismatch | **FLAGGED** — excluded from model |

*One symbol excluded. All 217 remaining symbols are clean.*

---

## PIT Compliance

All data access is point-in-time safe:

| Contract | Status |
|----------|--------|
| Features use only data ≤ bar i | ✓ Verified by LeakageValidator |
| Predictions use only data ≤ T | ✓ LookAheadGuard wired |
| Forward paper signals use T−1 close | ✓ `signal_ts = close[Sep 22-25]`, `entry = open[Sep 23+]` |
| Live session uses prior-close data | ✓ Sep 23-24 data → Sep 28 scoring |
| close-to-close mode marked non-economic | ✓ `is_economic_evidence=False` |

---

## Data Service Configuration

| Parameter | Value |
|-----------|-------|
| Port | 8200 |
| Rate limit | 500/60s (fixed from 100/60s) |
| Auth | Angel One JWT via Redis |
| Instrument master | 36,173 instruments loaded |
| Parquet fallback | Active — DataServiceClient falls back to on-disk |

---

## Data Actions Required (Oct 1)

```bash
# Ingest Sep 28-29-30 close bars (Upstox historical syncs within 24h of close)
PYTHONPATH=. python3 scripts/fast_ingest.py

# Fix TATAMOTORS instrument token (DQ-001)
# Map TATAMOTORS EQ token (not DVR) in instrument_master
```

*Generated: 2026-09-28 | Universe: 218/220 NSE F&O | Parquets: data/1d/1d/*
