# ml-service2.0
**AlphaForge — NSE F&O Cross-Sectional Alpha Engine**
**Stage: LIMITED SHADOW** | **Port: 8100** | **Model: v2c (LGBMRegressor, 65 features)** | **Tests: 1,891 ✓**

---

## What this is

A standalone, production-grade quantitative ML microservice that generates daily cross-sectional alpha signals for all 285 NSE F&O symbols. The active model (v2c) scores every symbol using 65 PIT-certified features and a 7-day cross-sectional rank excess-return label. It has been running in **LIMITED SHADOW** mode (long-only NSE futures paper trading) since 2026-10-01.

**Live evidence (Oct 1–9, 2026):** 9 live sessions completed. Promotion engine: 319 settled records, **61.8% win rate**, mean return **+0.71%**. BEAR sessions (6×) confirm genuine SHORT alpha. BULL sessions show structural LONG weakness — BULL suppressor active.

**Production path:** ~20 sessions by Nov 3, G11 promotion review. Target SHADOW→PRODUCTION: **~Nov 15, 2026** (subject to G_REGIME, G_ABLATION, G_UNIVERSE clearing).

---

## Architecture

```
alpha-forge (3000)  ←─── REST + WebSocket ───┐
                                              │
                          ml-service2.0  ─────┘  (8100)
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
data-service2.0          SentinelPulse     PostgreSQL (TimescaleDB)
   (8200)                   (3001)              (5445)
(live quotes)          (news/sentiment)    (signal storage)
```

Data flows exclusively through data-service2.0 (never direct provider calls). News/sentiment through SentinelPulse. Signals persisted to PostgreSQL (TimescaleDB) at port 5445.

---

## Current State

| Dimension | Value |
|-----------|-------|
| **Model** | v2c — LGBMRegressor, 65 features (fs-2.0.0), 7-day CS rank label |
| **Stage** | **LIMITED SHADOW** — long-only NSE futures authorized |
| **Universe** | **285 symbols** (live sessions) |
| **IC (OOS)** | +0.040 XS-Spearman (p<0.0001, n=116k); +0.0178 TS |
| **Annualized P&L** | +17.07%/yr absolute, +18.06% excess vs NIFTY, IR=1.374 |
| **PBO** | ≈0.48 (bootstrap CPCV, B=1,000 — healthy) |
| **BULL suppressor** | Active — multiplier=0.3 (BULL IC=−0.017, p=0.025) |
| **Live sessions** | **9/20** (Oct 1–9) — win rate 61.8%, mean +0.71% (319 settled) |
| **Gates** | **18 PASS / 3 PARTIAL / 0 FAIL** (21-gate v2c framework) |
| **Issues closed** | 60/64 (94%) — all P0/P1/P2/P3 complete |
| **Tests** | **1,891 pass / 0 fail** |
| **Signal storage** | **PostgreSQL (TimescaleDB)** at port 5445 (`ml-service-postgres`) |
| **Artifact** | `artifacts/v2_model/v2-20261001184632/model.pkl` |

---

## Quick Start

```bash
# Prerequisites: Docker, Python 3.11+

# 1. Start all infrastructure services
cd ../data-service2.0 && docker-compose up -d
cd ../ml-service2.0
docker-compose -f docker-compose.full-stack.yml up -d

# 2. Run the daily scheduler (persistent — keeps live session going all market day)
PYTHONPATH=. python3 scripts/daily_autorun_scheduler.py

# 3. OR run a single session manually (scores 285 symbols every 5 min until 15:30 IST)
PYTHONPATH=. python3 scripts/autorun_till_close.py

# 4. Run tests
python3 -m pytest tests/ --no-cov -q
```

---

## Key Commands

```bash
# Training (v2d)
PYTHONPATH=. python3 scripts/run_lgbm_conc_backtest.py

# Signal promotion
PYTHONPATH=. python3 scripts/resolve_forward_paper.py
PYTHONPATH=. python3 scripts/run_signal_promotion.py

# Daily signal ledger report
PYTHONPATH=. python3 scripts/signal_ledger_daily.py

# Service health
curl http://localhost:8100/health
curl http://localhost:8200/health
```

---

## Production Gate Status (v2c model)

| | Gate | Status | Evidence |
|--|------|--------|---------|
| G_PIT | No look-ahead bias | ✅ | Static audit 0 INVALID; mutation tests pass |
| G_LEAK | No feature leakage | ✅ | \|r\| < 0.15 all features vs forward returns |
| G_PARITY | Training/inference schema match | ✅ | 65 features in model pkl; `from_dict()` normalizer |
| G_ARTIFACT | Real model artifact loaded | ✅ | v2c model.pkl found and loadable |
| G_LABEL | Label economically valid | ✅ | 7-day CS rank label; positive EV at futures costs |
| G_HORIZON | 7-trading-day alignment | ✅ | Implemented throughout |
| G_UNIVERSE | Historical universe validated | ⚠️ PARTIAL | PIT F&O eligibility DB is Month-2 task |
| G_EXECUTION | Realistic execution | ✅ | next-open entry; 7.26bps futures |
| G_PORTFOLIO | Portfolio P&L positive | ✅ | +17.07%/yr OOS, IR=1.374 |
| G_SIGNIFICANCE | Statistical significance | ✅ | IC=+0.040, p<0.0001, n=116k |
| G_REGIME | Multi-regime robustness | ⚠️ PARTIAL | BEAR IC=+0.034; BULL IC=−0.017 (suppressor active); 9/20 sessions |
| G_CALIBRATION | Calibration valid | ✅ | Raw regression (no calibration — correct for ranker) |
| G_PBO | PBO analysis valid | ⚠️ PARTIAL | Bootstrap CPCV (B=1,000); full CPCV is Month-2 |
| G_PLACEBO | Placebo tests pass | ✅ | IC genuine; shuffled-label IC ≈ 0 |
| G_ABLATION | Feature ablation OOS | ✅ | Zero-out ablation PASS; B_ext_momentum tracked for v2d |
| G_STRESS | Cost stress test | ✅ | Futures profitable at 1× and 1.5× cost |
| G_DRAWDOWN | Drawdown limits | ✅ | Max DD = −12.88% OOS; within 15% limit |
| G_REPRODUCIBILITY | Deterministic results | ✅ | Seeds recorded in model pkl |
| G_NOCHERRY | No cherry-picking | ✅ | All 278 symbols, full 2025–2026 OOS |
| G_FORWARD | Forward paper reconciled | ✅ | DATA_ERROR guard (±30%) active |
| G_COST | Single cost model | ✅ | COST_MODEL_V2 canonical |

**PASS: 18 | PARTIAL: 3 | FAIL: 0**

---

## Signal Processing Integrations

Seven components active in the live session pipeline:

| Component | File | Purpose |
|-----------|------|---------|
| `ScoreThresholdSweep` | `src/analytics/score_threshold_sweep.py` | Optimal min-conviction threshold (current: 0.15) |
| `ForecastLedger` | `src/analytics/forecast_ledger.py` | Logs all 285 scores → calibration data |
| `CounterfactualLedger` | `src/analytics/counterfactual_ledger.py` | Grades blocked signals → tunes risk gates |
| `FeatureWeightManager` | `src/analytics/feature_weight_manager.py` | Agent-editable sector-regime filters |
| `ReversalDetector` | `src/analytics/reversal_detector.py` | RSI + BB reversal override |
| `SymbolICTracker` | `src/analytics/symbol_ic_tracker.py` | Per-symbol IC tracking |
| `SignalPromotionEngine` | `src/analytics/signal_promotion.py` | 6-gate PROMOTE/REJECT/DEMOTE lifecycle |

---

## BULL Suppressor

Active since 2026-10-06. When NIFTY 20-day return > +3%:
- Signal conviction multiplied by **0.3** (70% suppression) across all sectors
- Max LONG book capped at **4 positions**
- SHORT signals not authorized in BULL regime

Evidence: `strategy/feature_weights.json` → `sector_regime_filters.BULL.ALL.multiplier=0.3`

---

## Self-Improvement Capabilities

| Component | Decision Made Autonomously |
|-----------|--------------------------|
| `DrawdownManager` | 4-state risk control (NORMAL→CAUTION→DEFENSIVE→HALTED) |
| `AlphaDecayDetector` | IC monitoring → quarantine + retrain trigger |
| `DriftDetectorV3` | Feature PSI drift → MONITOR/RETRAIN/ROLLBACK |
| `SignalPromotionEngine` | 6-gate lifecycle (PROMOTE/REJECT/DEMOTE) |
| `SelfLearningLoop` | Outcome-driven retraining (5 triggers + rate limiting) |
| `AlphaSpecialistRegistry` | 4 independent alpha hypotheses → ensemble |

---

## Repository Layout

```
src/
├── alpha/              AlphaSpecialist registry (4 hypotheses)
├── analytics/          ScoreThresholdSweep, ForecastLedger, CounterfactualLedger,
│                       SignalLedger, SymbolICTracker, ReversalDetector,
│                       TurnoverOptimizer, FeatureWeightManager, RegimeAlphaMatrix
├── api/                FastAPI routers: predict, meta, analytics, explain, signals,
│                       training, monitoring  (all under /v2/ prefix)
├── backtest/           BacktestEngine, SevenDayBacktestEngine
├── data/               DatasetBuilder, SignalDB (PostgreSQL backend), FeedbackStore
├── features/           ExpandedFeatureFactory (v1/compat), v2c feature families
│                       (momentum, volatility, volume, cross_sectional, derivatives,
│                        market_structure, news)
├── labels/             MultiHorizonLabelFactory [1,3,5,10,21], CS rank label
├── meta/               MetaDecisionEngine, OpportunityScorer
├── models/             LightGBM, XGBoost, logistic estimators
├── monitoring/         DriftDetectorV3, Prometheus metrics (16 custom)
├── registry/           ModelRegistry (CHALLENGER→SHADOW→PRODUCTION lifecycle)
├── risk/               DrawdownManager (4 states)
├── streaming/          SignalStreamer — WebSocket fan-out (WS /v2/stream/signals)
├── training/           TrainingOrchestrator, WalkForwardValidator, CPCV, SelfLearningLoop
└── validation/         LeakageValidator, ResearchTrialLedger

scripts/
├── autorun_till_close.py       Main live session loop (5-min samples, 285 symbols)
├── daily_autorun_scheduler.py  Persistent scheduler — starts/stops watchdog on market days
├── session_watchdog.py         Process monitor — restarts autorun if stale >10 min
├── run_signal_promotion.py     SignalPromotionEngine runner
├── signal_ledger_daily.py      Daily signal ledger markdown report generator
├── resolve_forward_paper.py    Forward paper resolution
└── fast_ingest.py              Bulk parquet ingestion

strategy/
└── feature_weights.json        Agent-editable sector-regime signal filters

artifacts/
├── v2_model/                   v2c LGBMRegressor model artifacts (LIMITED SHADOW stage)
├── approvals/                  G12 approval records
├── live_session/               latest_scores.json (WebSocket feed source)
└── signal_ledger/              signals.jsonl (JSONL audit backup)

reports/
├── LIVE_SESSION_REPORT.md       Multi-session log (Sep 29 – Oct 9)
├── forensic/
│   ├── PRODUCTION_GATES.md      Authoritative gate status (21 gates, updated Oct 9)
│   ├── ROOT_CAUSE_REGISTER.md   13 root causes (10 fixed, 2 partial, 1 deferred)
│   └── FINAL_PRODUCTION_RECOMMENDATION.md
├── gates/
│   ├── regime_validation_report.md   G_REGIME analysis (updated Oct 7)
│   └── signal_promotion_preliminary.json  Promotion engine output (319 records)
└── live/                        Daily signal ledger reports + gap analyses (Oct 6-9)

docs/
├── architecture/                01_CURRENT_ARCHITECTURE.md, ARCHITECTURE.md (ADR)
├── specs/                       00_EXECUTIVE_SUMMARY.md + 26 design specs
├── operations/                  API_REFERENCE.md, DEPLOYMENT.md, MODEL_OPERATIONS.md
└── audit/                       Forensic audit trail (Sep 22 cycle)
```

---

## Services

| Service | Port | Purpose |
|---------|------|---------|
| ml-service2.0 | 8100 | ML predictions, training, shadow scoring (this service) |
| data-service2.0 | 8200 | Market data (Angel One, Upstox, instrument master) |
| SentinelPulse | 3001 | News/sentiment context |
| alpha-forge | 3000 | Signal consumer / portfolio construction |
| PostgreSQL (TimescaleDB) | 5445 | Signal storage (`ml-service-postgres` container) |

---

## Signal Storage (PostgreSQL)

Signals are persisted to PostgreSQL (TimescaleDB) — not flat files. Connection: `postgresql://mlservice:mlservice_dev@localhost:5445/mlservice` (overridable via `ML_DATABASE_URL` env var).

Tables: `sessions`, `signals`, `positions`, `price_snapshots`, `forecasts`, `symbol_ic`, `historical_outcomes`.

```sql
-- Today's open positions
SELECT * FROM positions WHERE session_date='2026-10-09' AND status='OPEN';

-- Rolling win rate
SELECT session_date,
       SUM((status='SETTLED_WIN')::int) wins,
       COUNT(*) total,
       ROUND(AVG(final_return_pct)::numeric,3) mean_ret
FROM positions WHERE status!='OPEN'
GROUP BY session_date ORDER BY session_date DESC;
```

---

## Deployment Path (Limited Shadow → Production)

```
LIMITED SHADOW (now — long-only futures)
  ↓ ~Nov 3: 20 live sessions complete → G_REGIME upgrade
  ↓ Oct 6–13: v2d retrain (drop B_ext_momentum — +39.8% IC expected) → G_ABLATION full
  ↓ Oct 8–14: PIT F&O eligibility DB → G_UNIVERSE full
  ↓ ~Nov 3: G11 promotion review (if win rate ≥50%)
  ↓ ~Nov 15: LIMITED SHADOW → SHADOW → PRODUCTION
```

Demotion triggers (auto-rollback to CHALLENGER): 3-day IC < 0, 5+ consecutive paper-loss days, drift HIGH, drawdown > 5%.

---

*Model: v2c LGBMRegressor | Artifact: v2-20261001184632 | Stage: LIMITED SHADOW*
*Updated: 2026-10-09 post-close IST*
