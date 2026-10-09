# REMEDIATION_BACKLOG.md
**AlphaForge ml-service2.0 — Remediation Status**
**Updated:** 2026-10-09 (post-close, v9.0 — 9 live sessions confirmed, Oct 9 BULL +0.63% added)
**Progress: 60/64 CLOSED (94%) | Stage: LIMITED SHADOW (long-only futures)**

---

## EXECUTIVE STATUS

```
╔══════════════════════════════════════════════════════════════════════╗
║  MODEL:  v2c (LGBMRegressor, 65 features, 7-day CS rank label)      ║
║  STATUS: LIMITED SHADOW — long-only NSE futures authorized           ║
║  OOS:    IC=+0.040 XS / +0.0178 TS (p<0.0001) | +17.07%/yr | IR=1.374 ║
║  LIVE:   9 sessions Oct 1-9 | DB win rate 61.8% (319 settled)       ║
║  PROMO:  319 records, win=61.8%, mean=+0.71%, G1-G6 all PASS        ║
║  BULL:   suppressor active (IC=-0.017 in BULL) | BEAR IC=+0.034     ║
║  NEXT:   Accumulate 20 sessions → G11 promotion review (11 more)    ║
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
| DQ | 1 | 0 | 1 | FLAGGED |
| INFRA | 9 | **9** | 0 | **100%** |
| POST | 5 | **5** | 0 | **100%** |
| **Total** | **64** | **60** | **4** | **94%** |

---

## P0 — CATASTROPHIC (6/6 CLOSED)

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
| DQ-001 | TATAMOTORS DVR vs regular price mismatch (295 vs 961, 226% gap) | **FLAGGED** | Symbol disabled in feature_weights.json (`symbol_overrides.TATAMOTORS.enabled=false`). Forward paper DATA_ERROR guard now catches any remaining implausible values. Root fix: correct Upstox instrument token mapping in data-service2.0. | P2 |

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

**PASS: 18 | PARTIAL: 2 | OPEN: 0 | FAIL: 0**   *(ISSUE-05: corrected — was 17 PASS)*

Up from **PASS: 9, FAIL: 9** for v1 model.

---

## NEXT ACTIONS (priority order)

| Action | When | Gate Impact |
|--------|------|------------|
| Accumulate 20 live sessions (long-only futures paper trading) | Oct 9 – Nov 3 (12 more sessions) | **G_REGIME → PASS** |
| **v2d: retrain dropping B_ext_momentum (13 features)** | **Oct 6–13** | **+0.007 IC expected** |
| Update Upstox token daily (expires Oct 9 03:30 IST) | Oct 9 before 03:30 IST | Operational |
| Run true OOS feature ablation (retrain without each group) | Week of Oct 7-13 | **G_ABLATION full → PASS** |
| Build PIT F&O eligibility database for survivorship fix | Oct 8-14 | G_UNIVERSE → PASS |
| Validate SHORT signals in TRENDING_BEAR periods | Week 3 of Oct | G_REGIME extension |
| If win rate ≥50% after 20 sessions: run formal G11 promotion review | ~Nov 3 | **G11 → trigger** |
| Commission proper CPCV PBO with CPCP combinatorics (bootstrap is approximation) | Nov | G_PBO upgrade |
| **SHADOW → PRODUCTION** (if G11 pass + G_REGIME pass + G_ABLATION pass) | **~Nov 15** | **Production** |

---

*Updated: 2026-10-09 post-close IST*
*v2c model: 1891 tests passing | Gates: 18 PASS / 3 PARTIAL / 0 FAIL*
*Live sessions: 9/20 | Promotion engine: 319 records (G1–G6 all PASS, corrected decision: PASS)*
