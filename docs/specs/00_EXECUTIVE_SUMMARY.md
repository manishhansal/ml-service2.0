# Executive Summary
**AlphaForge ml-service2.0 — NSE F&O Cross-Sectional Alpha Engine**
**Last updated: 2026-10-09 (post-close)**
**Status: LIMITED SHADOW — long-only NSE futures authorized (v2c model)**

---

## Current State

The system has completed the v2c model transition and 9 live trading sessions. The original v1 model was retired Oct 1 (OOS IC = −0.001, zero predictive power). The v2c model demonstrates consistent cross-sectional alpha in BEAR/SIDEWAYS regimes.

| Dimension | Status |
|-----------|--------|
| Model stage | **LIMITED SHADOW** (long-only futures since Oct 1) |
| Active model | **v2c — LGBMRegressor, 65 features, 7-day CS rank label** |
| Live signal scoring | **ACTIVE** — 285 NSE F&O symbols, every 5 min |
| Live capital | **NOT YET** — paper P&L only |
| Production target | **~Nov 15, 2026** (after 20 sessions + G11 review) |
| Signal storage | **PostgreSQL (TimescaleDB)** at port 5445 |

---

## Key Numbers

| Metric | Value |
|--------|-------|
| IC (OOS, XS-Spearman) | **+0.040** (p<0.0001, n=116k) |
| Bootstrap CPCV PBO | **≈0.48** (B=1,000 resamples — healthy) |
| Annualized P&L (OOS) | **+17.07%/yr** absolute; +18.06% excess vs NIFTY; IR=1.374 |
| Live sessions | **9/20** (Oct 1–9) |
| Promotion engine | **319 records, 61.8% win, mean +0.71%** |
| Gates | **18 PASS / 3 PARTIAL / 0 FAIL** (21-gate v2c framework) |
| Issues closed | **60/64 (94%)** |
| Test suite | **1,891 pass / 0 fail** |

---

## Changelog

| Date | Phase | Key Outcome |
|------|-------|-------------|
| Sep 22 | Forensic audit | 32 findings; P0-P3 remediation backlog created |
| Sep 23 | Sprint 1: core fixes | Leakage, Sharpe horizon, normalization order, DriftDetector |
| Sep 23 | Sprint 2: expansion | ExpandedFeatureFactory (55 features, fs-3.0.0), regime features |
| Sep 24 | Sprint 3: LightGBM v1 | IC=0.376 (continuous) — later found to be OOS IC = −0.001 |
| Sep 28 | Live session 1 (v1) | G12 approved; SHORT book running |
| Sep 29 | Live session 2 (v1) | SHORT +0.945% net |
| Oct 1 | **v1 model retired** | OOS IC = −0.001; zero predictive power confirmed |
| Oct 1 | **v2c model promoted** | LGBMRegressor, 65 features, 7-day CS rank label; LIMITED SHADOW |
| Oct 1 | PostgreSQL migration | Signal storage moved from flat files → TimescaleDB (port 5445) |
| Oct 1 | Persistent scheduler | `daily_autorun_scheduler.py` + `session_watchdog.py` wired |
| Oct 1–3 | Live sessions 1–3 (v2c) | BEAR regime — BEAR IC=+0.034 confirmed |
| Oct 5 | DQ-001 resolved | TATAMOTORS token collision fixed; symbol re-enabled |
| Oct 5 | Multiple gates closed | V2C-P1-001, V2C-P1-002, V2C-P2-001, 3 P0 issues closed |
| Oct 6 | **BULL suppressor** | BULL IC=−0.017 (p=0.025) → multiplier=0.3 added to feature_weights.json |
| Oct 6–9 | Live sessions 4–9 (v2c) | Mixed BEAR/BULL; promotion engine 319 records, 61.8% win |
| Oct 9 | INFRA bugs fixed | POST-004 (watchdog over-restart), INFRA-008 (NIFTY None 78% → 16%) |

---

## Decision Framework

```
Is the v2c model production-ready?

1. Is there genuine alpha (IC > 0)?
   YES → IC = +0.040 XS-Spearman (p<0.0001, n=116k) — genuine cross-sectional alpha

2. Does it survive realistic costs (G_STRESS)?
   YES → Futures profitable at 1× and 1.5× cost (7.26bps, 10.89bps)

3. Has it been confirmed live?
   YES → 9/20 sessions: 61.8% win rate, mean +0.71%, G1–G6 all PASS
         BEAR sessions (6×): BEAR IC=+0.034 confirmed
         BULL sessions (3×): LONG weakness confirmed → suppressor active

4. Is the infrastructure production-ready?
   YES — 18/21 gates PASS; PostgreSQL signal storage active; process management stable

5. Is capital deployment authorized?
   NOT YET → Needs 20 live sessions (11 more), v2d retrain (B_ext_momentum drop),
              PIT F&O eligibility DB, and G11 formal review
   EARLIEST: ~Nov 15, 2026
```

---

## Production Path

```
LIMITED SHADOW (now)
  ↓ ~Nov 3: 20 sessions complete → G_REGIME review
  ↓ Oct 6–13: v2d retrain (drop B_ext_momentum) → G_ABLATION full PASS
  ↓ Oct 8–14: Build PIT F&O eligibility DB → G_UNIVERSE PASS
  ↓ ~Nov 3: G11 formal promotion review (if 20-session win rate ≥50%)
  ↓ ~Nov 15: SHADOW → PRODUCTION (if G11 + G_REGIME + G_ABLATION pass)
  ↓ Nov 15+: NSE F&O live execution (long-only futures initially, 1 lot/signal)
```

Remaining 3 partial gates:
- **G_REGIME**: 11 more live sessions needed (9/20); BULL suppressor active
- **G_ABLATION**: Full retrain ablation (v2d) — drop B_ext_momentum → +39.8% IC expected
- **G_UNIVERSE**: PIT F&O eligibility DB (survivorship bias <5% impact estimated)

---

*v2c model: LGBMRegressor | Artifact: v2-20261001184632 | Stage: LIMITED SHADOW*
*v1 model: Archived (CHALLENGER) — IC = −0.001 OOS, zero predictive power*
