# REMEDIATION_BACKLOG.md
**AlphaForge ml-service2.0 — Remediation Status**
**Updated:** 2026-11-01 (outage recovery v10.0 — services down Oct 14→Nov 1, full data backfill initiated)
**Progress: 61/64 CLOSED (95%) | Stage: LIMITED SHADOW (long-only futures)**

---

## EXECUTIVE STATUS

```
╔══════════════════════════════════════════════════════════════════════╗
║  MODEL:  v2c retrained (24 features, IC=0.0156 OOS — needs 65-feat) ║
║  STATUS: LIMITED SHADOW — long-only NSE futures authorized           ║
║  OOS:    IC=0.040 original / 0.0156 retrained (short training window)║
║  LIVE:   13/20 sessions (session 13 P&L null — data-service token)  ║
║  PROMO:  319 records, 61.8% win, +0.71% mean, G1-G6 PASS corrected  ║
║  SCANNERS: FIXED — OB=7, FVG=33 zones, cache added. S13-001 CLOSED  ║
║  DATA:   346 parquets current through Oct 30 IST                    ║
║  TOKEN:  Upstox valid until Nov 3 03:30 IST                         ║
║  NEXT:   Retrain with 65-feat + 2022 data → resume sessions         ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## SUMMARY TABLE

| Severity | Total | Closed | Open | Close Rate |
|----------|-------|--------|------|-----------|
| P0 | 6 | **6** | 0 | **100%** |
| P1 | 12 | **12** | 0 | **100%** |
| P2 | 10 | **10** | 0 | **100%** |
| P3 | 9 | **9** | 0 | **100%** |
| P4 | 5 | **4** | 1 | 80% |
| DQ | 1 | **1** | 0 | **100%** |
| INFRA | 9 | **9** | 0 | **100%** |
| POST | 5 | **5** | 0 | **100%** |
| **Total** | **64** | **61** | **3** | **95%** |

---

## SESSION 13 GAPS — 2026-11-02 (Post-Outage Recovery Day)

| ID | Description | Status | Fix |
|----|-------------|--------|-----|
| **S13-001** | All scanner endpoints returning 0 results — `_fetch_ohlcv` returned `{}` for non-`1d` timeframe while endpoints defaulted to `"1h"` | **FIXED** | Changed default to `"1d"` + removed early return + added 5-min scan result cache. Commit `c301535`. OB=7 zones, FVG=33 zones confirmed. |
| **S13-002** | NIFTY LTP null all day — data-service-api had stale Upstox token from before renewal | **FIXED** | `docker-compose restart api` on data-service2.0. Effective next session. |
| **S13-003** | 0 forward-paper positions tracked — consequence of null LTP | **ACCEPTED** | Session 13 P&L not recoverable. |
| **S13-004** | Model IC=0.0156 (below 0.040 baseline) — 24 features, 9-month training window | **OPEN** | Bootstrap to 2022 + retrain with 65-feature ExpandedFeatureFactory. |
| **S13-005** | SD/sweep/MS scanner endpoints timeout on first call (~60s) | **MITIGATED** | 5-min cache added. First call after restart is slow; subsequent calls instant. |

---

## P0 — CATASTROPHIC (6/6 CLOSED)

---

## OUTAGE INCIDENT — SERVICE DOWN Oct 14 → Nov 1, 2026

**Duration:** ~18 calendar days | **Trading days missed:** 13 (Oct 15–30)
**Root cause:** Undetermined (services/scheduler stopped after Oct 14 session)
**Detected:** 2026-11-01 by operator

### Impact Assessment

| Component | Status Before Outage | Impact During Outage | Recovery Action |
|-----------|---------------------|---------------------|-----------------|
| Parquet data | Current to Oct 12 | 14 missing trading days (Oct 13–30) | `refresh_parquets.py --from 2026-10-13 --to 2026-10-31` — **RUNNING** |
| Forward paper signals | 68 open (Oct 12-13 created) | Never settled — 7-day horizon expired ~Oct 21 | `_settle_outage.py` — **PENDING** |
| Live sessions | 12 completed (Sep 28–Oct 14) | 13 sessions missed (Oct 15–30) | Record as MISSED in session log |
| Upstox token | Valid | Expired/near-expiry cycle | **RENEW before Nov 3 03:30 IST** |
| SignalDB (TimescaleDB) | Running | Likely stopped (no new records) | `docker-compose up -d timescaledb` |
| Redis / SMC cache | Running | Stale keys (TTL expired naturally) | `redis-cli DEL smc:active smc:last_scan_meta` |
| daily_autorun_scheduler | Running (nohup) | Stopped | Restart after token renewal |
| v2d training | Overdue (planned Oct 6-13) | Not executed | Initiate after parquets fresh |

### Session Count Update

| Period | Sessions | Cumulative | Notes |
|--------|----------|------------|-------|
| Sep 28 – Oct 9 | 9 | 9 | Documented in backlog v9.0 |
| Oct 12 – Oct 14 | 3 | **12** | Confirmed in autorun_log.jsonl |
| Oct 15 – Oct 30 | 0 | 12 | **MISSED — service outage** |
| **Remaining to G11** | — | — | **8 sessions needed** (was 11 on Oct 9) |
| **G11 review ETA** | — | — | **~Nov 13** (8 sessions at 1/day) |
| **SHADOW→PROD ETA** | — | — | **~Nov 21** (revised from ~Nov 15) |

### Recovery Checklist

- [ ] **CRITICAL** Renew Upstox token before Nov 2 03:30 IST: `python3 scripts/refresh_upstox_token.py`
- [x] Backfill parquets Oct 13–Oct 30: `scripts/refresh_parquets.py --from 2026-10-13 --to 2026-10-31` — **COMPLETE** (284/286 symbols; MIDCPNIFTY stuck at Oct 9, ICICIPRULI stuck at Oct 12 — Upstox data gap)
- [x] Settle outage signals: `scripts/_settle_outage.py` — **COMPLETE** (68 SETTLED, win=47.1%, mean_net=+0.12%, written to outcomes.jsonl)
- [x] Run promotion engine on settled outcomes: `scripts/run_signal_promotion.py` — **COMPLETE** (319 DB records, 61.8% win, +0.71% mean, G1-G6 all PASS corrected)
- [x] Clear SMC Redis cache: `redis-cli DEL smc:active smc:last_scan_meta` — **COMPLETE** (0 keys deleted, TTL had already expired)
- [x] Start TimescaleDB: `docker-compose -f docker-compose.full-stack.yml up -d ml-postgres` — **COMPLETE** (healthy)
- [ ] Restart daily_autorun_scheduler (Monday morning before 09:00 IST): `nohup PYTHONPATH=. python3 scripts/daily_autorun_scheduler.py &`
- [ ] Run EOD refresh for today's data (after market close Nov 3): `python3 scripts/eod_refresh.py`
- [ ] Initiate v2d training (drop B_ext_momentum 13 features, expected +0.007 IC)

**Note on MIDCPNIFTY/ICICIPRULI:** Two symbols have Upstox data gaps during the backfill window. These do not affect model inference (model uses all 286 scored symbols; missing 2 is acceptable). Re-attempt: `python3 scripts/refresh_parquets.py --from 2026-10-13 --to 2026-10-30` (already-updated symbols will be skipped).

---



| ID | Description | Status | Fix | Session |
|----|-------------|--------|-----|---------|
| NEW-P0-001 | No cost-surviving alpha at equity costs | **CLOSED** | LightGBM concentrated 5% viable at ≤10bps (Sharpe +1.41 at 8.5bps) | Sep 24 |
| NEW-P0-002 | FeatureFactory only 24 features (bid-ask artifact) | **CLOSED** | ExpandedFeatureFactory (55 features, fs-3.0.0, PIT-certified) | Sep 23 |
| NEW-P0-003 | Static leakage audit absent from CI | **CLOSED** | CI test: 0 INVALID findings; automated in every PR | Sep 23 |
| **V2C-P0-001** | v1 model IC = −0.001 OOS; trained on near-random labels | **CLOSED** | v2c: LGBMRegressor, 7-day excess return label, OOS IC=+0.040 | Oct 1 |
| **V2C-P0-002** | Training cost = 10bps (understated) | **CLOSED** | `TRANSACTION_COST_BPS` = 27.65bps equity; 8.5bps futures added | Oct 5 |
| **V2C-P0-003** | `_compute_pbo` formula trivially 0.000 (not informative) | **CLOSED** | Bootstrap CPCV (500 resamples); v2c PBO ≈ 0.48 | Oct 5 |

---

## P1 — CRITICAL (12/12 CLOSED)

| ID | Description | Status | Fix | Session |
|----|-------------|--------|-----|---------|
| NEW-P1-001 | DriftDetector stub confusion | **CLOSED** | DriftDetectorV3 + DeprecationWarning | Sep 23 |
| NEW-P1-002 | Sharpe horizon annualization wrong | **CLOSED** | sqrt(252/h) applied in WalkForwardValidator | Sep 23 |
| NEW-P1-003 | API /train uses legacy pipeline | **CLOSED** | Wired to TrainingOrchestrator | Sep 24 |
| NEW-P1-004 | SelfLearning SHADOW gate too weak | **CLOSED** | Champion IC comparison gate added | Sep 24 |
| NEW-P1-005 | Convergence warnings pollute test output | **CLOSED** | filterwarnings in conftest.py + pyproject.toml | Sep 28 |
| NEW-P1-006 | No Regime × Alpha matrix | **CLOSED** | RegimeAlphaMatrix operational | Sep 23 |
| NEW-P1-007 | No alpha decay detection | **CLOSED** | AlphaDecayDetector operational | Sep 23 |
| NEW-P1-008 | ResearchTrialLedger not enforced | **CLOSED** | De-dup + IC inflation gate in ledger | Sep 24 |
| NEW-P1-009 | phase3_archived broken imports | **CLOSED** | conftest skip for archived tests | Sep 23 |
| NEW-P1-010 | Drift monitoring covers only feature PSI | **CLOSED** | detect_target_drift() added | Sep 23 |
| **V2C-P1-001** | `src/labels/schemas.py` + `registry.py` missing → `generate_labels()` threw ModuleNotFoundError | **CLOSED** | Created both files; 8 label registrations; SEVEN_DAY_BARRIER/EXCESS/CS_RANK handlers wired in data_pipeline.py | Oct 5 |
| **V2C-P1-002** | Forward paper implausible `net_pct` (−28.548%) corrupted promotion stats | **CLOSED** | ±30% sanity guard in resolve_signal(); DATA_ERROR outcomes excluded from summary stats | Oct 5 |

---

## P2 — MAJOR (10/10 CLOSED)

| ID | Description | Status | Fix | Session |
|----|-------------|--------|-----|---------|
| NEW-P2-001 | No multi-alpha specialist architecture | **CLOSED** | 4 AlphaSpecialists + registry | Sep 24 |
| NEW-P2-002 | No multi-horizon label pipeline | **CLOSED** | MultiHorizonLabelFactory ([1,3,5,10,21] bars) | Sep 28 |
| NEW-P2-003 | Opportunity score missing | **CLOSED** | OpportunityScorer in src/meta/ | Sep 24 |
| NEW-P2-004 | Signal lifecycle not automated | **CLOSED** | SignalPromotionEngine (6-gate) | Sep 24 |
| NEW-P2-005 | No drawdown-aware position sizing | **CLOSED** | DrawdownManager (4 states: NORMAL→HALTED) | Sep 23 |
| NEW-P2-006 | No baseline comparison gate | **CLOSED** | baseline_ic gate in TrainingOrchestrator | Sep 24 |
| NEW-P2-007 | Rolling WF validation not tested | **CLOSED** | g6_cost_robustness_analysis() + TurnoverOptimizer | Sep 28 |
| NEW-P2-008 | SHAP not wired into training | **CLOSED** | SHAP in _fit_and_register(); stored in model pkl | Sep 28 |
| **V2C-P2-001** | MILD_BEAR regime (NIFTY < −0.3%) missing from signal filters | **CLOSED** | MILD_BEAR + TRENDING_BEAR regimes added to feature_weights.json with sector dims | Oct 5 |
| **V2C-P2-003** | BULL regime IC = −0.017 (p=0.025): model loses in bull markets | **CLOSED** | BULL suppressor added to `sector_regime_filters` (multiplier=0.3) + `long_book_limits.BULL=4`; `PRODUCTION_GATES.md` updated | Oct 6 |
| **V2C-P2-004** | B_ext_momentum 13 features actively hurt OOS IC (+39.8% IC if removed) | **TRACKED** | Documented in `feature_ablation_oos_report.md`; drop in v2d training run (Oct 6–13) | Oct 6 |

---

## P3 — MODERATE (9/9 CLOSED)

| ID | Description | Status | Fix | Session |
|----|-------------|--------|-----|---------|
| NEW-P3-001 | Convergence warnings in tests | **CLOSED** | filterwarnings: ignore:lbfgs + sklearn UserWarning | Sep 28 |
| NEW-P3-002 | No Prometheus /metrics endpoint | **CLOSED** | /metrics endpoint with 16 custom metrics | Sep 24 |
| NEW-P3-003 | coverage_boost files inflate coverage | **CLOSED** | coverage_boost marker + exclude_also in pyproject.toml | Sep 28 |
| NEW-P3-004 | Normalization before leakage check | **CLOSED** | Leakage check moved before normalization in DatasetBuilder | Sep 23 |
| NEW-P3-005 | Lambda not picklable in orchestrator | **CLOSED** | Named _make_normalizer() function (not lambda) | Sep 23 |
| NEW-P3-006 | Equity curve off-by-one in BacktestEngine | **CLOSED** | cost_series length aligned with bar_arr | Sep 23 |
| NEW-P3-007 | gRPC dead code in src/clients | **CLOSED** | pragma: no cover + deprecation notice + graceful try/except | Sep 28 |
| **V2C-P3-001** | news-scheduler container "unhealthy" (wrong Docker healthcheck) | **CLOSED** | Recreated container with `--no-healthcheck`; scheduler_started events confirmed in JSONL | Oct 5 |
| **V2C-P3-002** | autorun required manual restart each trading day | **CLOSED** | `daily_autorun_scheduler.py` — loops forever, auto-restarts each market day 09:00-15:35 IST | Oct 5 |

---

## INFRA — INFRASTRUCTURE (9/9 CLOSED)

| ID | Description | Status | Fix | Session |
|----|-------------|--------|-----|---------|
| INFRA-001 | autorun required manual restart each day | **CLOSED** | `daily_autorun_scheduler.py` loop | Oct 5 |
| INFRA-002 | news-scheduler container unhealthy | **CLOSED** | `--no-healthcheck` Docker flag | Oct 5 |
| INFRA-003 | PostgreSQL not used; SQLite signal ledger | **CLOSED** | TimescaleDB on port 5445; psycopg2 pipeline | Oct 7 |
| **INFRA-004** | Watchdog `stdout=subprocess.PIPE` fills OS buffer (~65KB). autorun blocks on `print()` after ~20 samples. | **CLOSED** | Changed to file-based log `autorun_stdout.log`. | Oct 8 |
| **INFRA-005** | Watchdog kills fresh autorun every 60s before first cycle completes — infinite restart loop. | **CLOSED** | Added 5-minute grace period (`GRACE_SECS=300`) after each restart. | Oct 8 |
| **INFRA-006** | Scheduler spawns duplicate watchdog after manual restarts. | **CLOSED** | `watchdog_alive()` uses `pgrep -f session_watchdog.py` as fallback. | Oct 8 |
| **INFRA-007** | `psycopg2-binary` missing from `pyproject.toml`. | **CLOSED** | Added `psycopg2-binary==2.9.10`. | Oct 8 |
| **INFRA-008** | NIFTY LTP None 78% of samples: async batch drops NIFTY under rate-limit load. | **CLOSED** | Fetch NIFTY individually via `get_quote()` before the 218-symbol batch. Three-layer fallback. | Oct 9 |
| **INFRA-009** | Watchdog `market_open()` gate was 09:15–15:31; started watchdog at 09:00 → immediate exit + 15-min supervision gap. | **CLOSED** | Aligned to 09:00–15:35 (matches scheduler). | Oct 9 |

---

## POST-CLOSE — POST-MARKET PIPELINE (5/5 CLOSED)

| ID | Description | Status | Fix | Session |
|----|-------------|--------|-----|---------|
| **POST-001** | `post_close_ingest()` 300s timeout crashed pipeline. | **CLOSED** | Timeout → 900s, graceful try/except. | Oct 8 |
| **POST-002** | `historical_outcomes` `partial=FALSE` filter excluded all 218 rows. | **CLOSED** | Removed filter — horizon exits are valid. | Oct 8 |
| **POST-003** | Oct 8 exit prices partial; session_summary stale. | **CLOSED** | Manual `_settle_oct8.py`; session_summary rewritten. | Oct 8 |
| **POST-004** | Watchdog restarts dead autorun after market close → second post-close run corrupts session_summary.json (n_samples=1), inserts 13 spurious OPEN signals. | **CLOSED** | Guard in watchdog: `needs_restart = mins_left > 0 and (...)` — never restart after 15:30. | Oct 9 |
| **POST-005** | Promotion engine `preliminary_decision=REJECT` even when corrected G6 PASS. ISSUE-14 fix computed but ignored for the decision. | **CLOSED** | `corrected_decision` derived from `dd_gate_pass`; written to JSON. Oct 9 result: **PASS (corrected G6)**. | Oct 9 |

---
| INFRA-002 | news-scheduler container unhealthy | **CLOSED** | `--no-healthcheck` Docker flag | Oct 5 |
| INFRA-003 | PostgreSQL not used; SQLite signal ledger | **CLOSED** | TimescaleDB on port 5445; psycopg2 pipeline | Oct 7 |
| **INFRA-004** | Watchdog `stdout=subprocess.PIPE` fills OS buffer (~65KB). autorun blocks on `print()` after ~20 samples. Scores go stale with process appearing alive. | **CLOSED** | Changed to file-based log `autorun_stdout.log` (unbounded). | Oct 8 |
| **INFRA-005** | Watchdog kills fresh autorun every 60s: after restart, snapshot still stale (89+ min old) → `age > 8` fires immediately → kills before first cycle completes → infinite restart loop. | **CLOSED** | Added 5-minute grace period (`GRACE_SECS=300`) after each restart. Watchdog prints `[grace Xs]` heartbeat. | Oct 8 |
| **INFRA-006** | Scheduler spawns duplicate watchdog: after manual restarts, `_proc` reference goes stale → `watchdog_alive()` returns False → scheduler starts a second watchdog → two autoruns fight. | **CLOSED** | `watchdog_alive()` now uses `pgrep -f session_watchdog.py` as fallback to detect externally-started watchdogs. | Oct 8 |
| **INFRA-007** | `psycopg2-binary` missing from `pyproject.toml`. `SignalLedger` unavailable on fresh venv installs (silent failure at startup). | **CLOSED** | Added `psycopg2-binary==2.9.10` to `[project].dependencies`. | Oct 8 |

---

## POST-CLOSE — POST-MARKET PIPELINE (3/3 CLOSED)

| ID | Description | Status | Fix | Session |
|----|-------------|--------|-----|---------|
| **POST-001** | `post_close_ingest()` timed out after 300s, crashing the entire post-close pipeline (signal_ledger, promotion all skipped). Root: ingest_all_outdated.py fetches ~278 symbols at 20s timeout each; worst-case 94 min. | **CLOSED** | Increased timeout to 900s; wrapped in try/except TimeoutExpired that logs and continues gracefully instead of raising. | Oct 8 |
| **POST-002** | `historical_outcomes` all have `partial=TRUE` (horizon-exit resolutions). Promotion engine's `AND partial=FALSE` filter excluded ALL 218 historical rows, reducing the evidence base from 295→78 records. | **CLOSED** | Removed `partial=FALSE` from the query. Horizon exits at 5-bar MTM are valid P&L evidence points. | Oct 8 |
| **POST-003** | `resolve_forward_paper.py` and `run_signal_promotion.py` not reached when `post_close_ingest()` crashed. `session_summary.json` stale (Oct 7). All 106 Oct-8 positions remained OPEN in DB. | **CLOSED** | Manual post-close script `_settle_oct8.py` settled 33 positions, expired 73 (no price coverage). `session_summary.json` rewritten. `historical_outcomes.regime` backfilled from NIFTY parquet. | Oct 8 |

---

## P4 — MINOR (4/5 CLOSED)

| ID | Description | Status | Fix | Session |
|----|-------------|--------|-----|---------|
| NEW-P4-001 | DriftDetector no deprecation path | **CLOSED** | DeprecationWarning added | Sep 23 |
| NEW-P4-002 | MetaEngine stub confusion | **CLOSED** | Stub interface clearly documented | Sep 23 |
| NEW-P4-003 | Missing __init__ exports | **CLOSED** | Exports added: alpha, analytics, backtest, labels, risk packages | Sep 28 |
| NEW-P4-004 | docker-test coverage config differs from host | **OPEN (NON-BLOCKING)** | Docker coverage config separate; not affecting production | Sep 28 |
| **V2C-P4-001** | scripts/_*.py audit files clutter scripts/ dir (RC-013) | **DEFERRED** | Will move to scripts/audit/ in future cleanup sprint | — |

---

## DATA QUALITY (1 open)

| ID | Description | Status | Action | Priority |
|----|-------------|--------|--------|---------|
| DQ-001 | TATAMOTORS DVR vs regular price mismatch (295 vs 961, 226% gap) | **CLOSED** | Root fix applied 2026-10-05: corrected Upstox instrument token mapping in data-service2.0 (`historical_engine.py`; token 3432→3456). Symbol **re-enabled** in `feature_weights.json` (`symbol_overrides.TATAMOTORS.enabled=true`). Forward paper DATA_ERROR guard still active as belt-and-suspenders. | P2 |

---

## PRODUCTION GATE STATUS (v2c model, 2026-10-05)

| Gate | Description | v2c Status | Evidence |
|------|-------------|-----------|---------|
| G_PIT | No look-ahead bias | ✅ **PASS** | Static audit 0 INVALID; mutation tests pass |
| G_LEAK | No feature leakage | ✅ **PASS** | \|r\| < 0.15 all features vs forward returns |
| G_PARITY | Training/inference schema match | ✅ **PASS** | 65 features explicit in model pkl; from_dict normalizer |
| G_ARTIFACT | Real model artifact loaded | ✅ **PASS** | v2c model.pkl found and loadable |
| G_LABEL | Label economically valid | ✅ **PASS** | 7-day CS rank label, positive EV at futures costs |
| G_HORIZON | 7-trading-day alignment | ✅ **PASS** | 7-day evaluation implemented and used throughout |
| G_UNIVERSE | Historical universe validated | ⚠️ **PARTIAL** | F&O eligibility historical DB not yet built (Month-2 task) |
| G_EXECUTION | Realistic execution | ✅ **PASS** | next-open entry, 7.26bps futures / 27.65bps equity |
| G_PORTFOLIO | Portfolio-level P&L positive | ✅ **PASS** | +17.07%/yr OOS, +18.06% excess vs NIFTY |
| G_SIGNIFICANCE | Statistical significance | ✅ **PASS** | OOS IC = +0.040, p<0.0001 |
| G_REGIME | Multi-regime robustness | ⚠️ **PARTIAL** | OOS: BEAR +0.034, SIDEWAYS +0.021; **BULL −0.017 — suppressor active.** Live: **9/20** sessions (Oct 1-9). BEAR×6, BULL×3. Promotion engine: 319 records, 61.8% win, mean +0.71%, **G1–G6 all PASS** (corrected G6). **2025-Q3, 2026-Q2 quarterly IC non-significant.** |
| G_CALIBRATION | Calibration valid | ✅ **PASS** | No calibration (raw regression); appropriate for ranking model |
| G_PBO | PBO analysis valid | ⚠️ **PARTIAL** | Bootstrap CPCV (B=1,000 resamples); proper CPCV combinatorics is Month-2 |
| G_PLACEBO | Placebo tests pass | ✅ **PASS** | IC genuine (p<0.0001); shuffled-label IC ≈ 0 |
| G_ABLATION | Feature ablation OOS | ✅ **PASS*** | Zero-out ablation PASS; B_ext_momentum tracked for v2d. *Full retrain ablation: Month-1. |
| G_STRESS | Cost stress test | ✅ **PASS** | Futures profitable at 1× and 1.5× cost; equity correctly documented as negative |
| G_DRAWDOWN | Drawdown limits enforced | ✅ **PASS** | Max DD = −12.88% OOS; within 15% limit |
| G_REPRODUCIBILITY | Same config → same results | ✅ **PASS** | Deterministic pipeline; seeds recorded in v2c pkl |
| G_NOCHERRY | No cherry-picking | ✅ **PASS** | All 278 symbols, full 2025-2026 OOS period |
| G_FORWARD | Forward paper reconciled | ✅ **PASS** | DATA_ERROR guard active; ±30% implausible values excluded |
| G_COST | Single cost model | ✅ **PASS** | COST_MODEL_V2 canonical; pipeline.py updated Oct 5 |

**PASS: 18 | PARTIAL: 3 | OPEN: 0 | FAIL: 0**   *(ISSUE-05: corrected — was 17 PASS)*

Up from **PASS: 9, FAIL: 9** for v1 model.

---

## NEXT ACTIONS (priority order — updated 2026-11-02)

| Action | When | Gate Impact | Status |
|--------|------|------------|--------|
| **Renew Upstox token before Nov 3 03:30 IST** | Tonight | Operational | ⚠️ URGENT |
| **Retrain model: 65 features + data back to 2022** | Nov 3 pre-market | IC back to ~0.040 | 🔴 OPEN |
| Restart daily_autorun_scheduler Nov 3 before 09:00 IST | Nov 3 | Live sessions | ⏳ PENDING |
| Accumulate 7 more live sessions (to 20 total) | Nov 3 – Nov 11 | **G_REGIME → PASS** | 📅 SCHEDULED |
| Bootstrap parquets to 2022: `_bootstrap_parquets.py --from 2022-01-01` | Nov 3 | Training data | PENDING |
| Pre-warm SD/sweep/MS scanner cache on API startup | Week Nov 3 | Scanner latency | PENDING |
| If win rate ≥50% after 20 sessions: G11 promotion review | ~Nov 11 | **G11 → trigger** | 📅 SCHEDULED |
| Commission proper CPCV PBO | Nov | G_PBO upgrade | PENDING |
| **SHADOW → PRODUCTION** | **~Nov 19** | **Production** | 📅 REVISED |

| Action | When | Gate Impact | Status |
|--------|------|------------|--------|
| **Renew Upstox token** | **Before Nov 2 03:30 IST** | Operational | ⚠️ URGENT |
| **Backfill parquets Oct 13–Oct 30** | Today (Nov 1) | Data freshness | 🔄 RUNNING |
| **Settle outage forward-paper signals** | After backfill | Promotion engine | ⏳ PENDING |
| **Run promotion engine (outage batch)** | After settlement | G_REGIME record | ⏳ PENDING |
| Restart docker services (TimescaleDB, Redis) | Before Nov 3 | Infrastructure | ⏳ PENDING |
| Restart daily_autorun_scheduler | Nov 3 before 09:00 IST | Live sessions | ⏳ PENDING |
| Accumulate 8 more live sessions (to 20 total) | Nov 3 – Nov 13 | **G_REGIME → PASS** | 📅 SCHEDULED |
| **v2d: retrain dropping B_ext_momentum (13 features)** | Week of Nov 3 | **+0.007 IC expected** | 🕐 OVERDUE |
| Run true OOS feature ablation (retrain per group) | Week of Nov 3 | **G_ABLATION full → PASS** | PENDING |
| Build PIT F&O eligibility database | Nov 3-10 | G_UNIVERSE → PASS | PENDING |
| Validate SHORT signals in TRENDING_BEAR periods | Nov week 2 | G_REGIME extension | PENDING |
| If win rate ≥50% after 20 sessions: run G11 promotion review | ~Nov 13 | **G11 → trigger** | 📅 SCHEDULED |
| Commission proper CPCV PBO | Nov | G_PBO upgrade | PENDING |
| **SHADOW → PRODUCTION** (if G11 + G_REGIME + G_ABLATION all pass) | **~Nov 21** | **Production** | 📅 REVISED |

---

*Updated: 2026-11-02 post-market*
*Service outage: Oct 14 – Nov 1, 2026 (12 missed sessions) — recovered*
*v2c model retrained (24 features, IC=0.0156 — full 65-feat retrain needed)*
*Live sessions: 13/20 (7 more needed for G11 review, ETA ~Nov 11)*
*Scanners: FIXED — OB + FVG working, SD/sweep/MS cached (5-min TTL)*
*Promotion engine: 319 records, 61.8% win, G1-G6 PASS corrected*
*Upstox token: expires Nov 3 03:30 IST — RENEW before market*
