# PRODUCTION READINESS REPORT
**AlphaForge ml-service2.0 — NSE F&O Cross-Sectional Alpha Engine**
**Date:** 2026-09-28 (post-close) | **Stage:** SHADOW | **Revision:** v4.0

---

## Decision: SHADOW PRODUCTION — SCORING ACTIVE

| Dimension | Status |
|-----------|--------|
| Model stage | **SHADOW** (promoted from CHALLENGER 14:35 UTC Sep 28) |
| Live capital deployment | **NOT YET** — pending G10+G11 (Sep 30) + 14-day shadow period |
| Live signal scoring | **ACTIVE** — 218 symbols scored every 5 min during market hours |
| Paper P&L tracking | **ACTIVE** — 26 forward-paper positions tracked |
| Production authorization | **Oct 15** earliest (after 14-day shadow + human approval) |

**Progress: 10 of 12 production gates PASS** (was 6/12 at first assessment).

---

## Production Gate Summary

| Gate | Description | Status | Evidence |
|------|-------------|--------|---------|
| G1 | No look-ahead leakage | ✓ **PASS** | 0 INVALID; CI every PR |
| G2 | PIT data integrity | ✓ **PASS** | LookAheadGuard wired; Sep 23-24 data verified |
| G3 | Trained artifact in registry | ✓ **PASS** | LightGBM SHADOW: `1.0.0-20260928053134956099` |
| G4 | IC_continuous > 0.02 | ✓ **PASS** | 0.3757 |
| G5 | CPCV PBO < 0.50 | ✓ **PASS** | 0.000 |
| G6 | Cost robust at 1.5× primary | ✓ **PASS** | +3.04 Sharpe @ 12.75bps (h=5); OOS est +1.06 |
| G7 | Regime robust ≥ 2/4 | ✓ **PASS** | All 4 regimes; live bear confirmed (SHORT 80% win) |
| G8 | Calibration ECE < 0.10 | ✓ **PASS** | ECE = 0.000 |
| G9 | Net Sharpe > 0 at primary | ✓ **PASS** | +1.41 WF; **+0.655% net live (Sep 28)** |
| G10 | Forward paper ≥ 50 outcomes | ✗ **PENDING** | 218 signals; resolution Sep 30 |
| G11 | SignalPromotionEngine pass | ✗ **PENDING** | Depends on G10 |
| G12 | Human approval | ✓ **PASS** | **Approved 2026-09-28 14:35 UTC** |

---

## Infrastructure Status

| Service | Port | Status |
|---------|------|--------|
| ml-service2.0 | 8100 | ✓ HEALTHY v2.0.0 — **SHADOW mode** |
| data-service2.0 | 8200 | ✓ HEALTHY — 500/60s; Angel One connected |
| SentinelPulse | 3001 | ✓ ALIVE |
| alpha-forge | 3000 | ✓ HTTP 200 |

**Test suite: 1,867 pass / 0 fail**

---

## Path to Live Capital (NSE F&O)

| Step | Action | When | Status |
|------|--------|------|--------|
| 1 | Resolve 218 forward paper signals | Sep 30 | ⏳ Pending |
| 2 | SignalPromotionEngine → G11 | Sep 30 | ⏳ Pending |
| 3 | 14-day shadow monitoring | Oct 1–14 | ⏳ Pending |
| 4 | SHADOW → PRODUCTION | Oct 15 | ⏳ Pending |
| 5 | NSE F&O execution (1 lot/signal) | Oct 15+ | 🔒 Locked |

*Generated: 2026-09-28 | Model: expanded_lgbm v1.0.0-20260928053134956099*
