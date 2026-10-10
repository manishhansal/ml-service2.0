# Current Architecture
**AlphaForge ml-service2.0 — As-Built Documentation**
**Date: 2026-10-09 (post-close, v2c model)**
**Status: LIMITED SHADOW — long-only NSE futures authorized**

---

## 1. Service Overview

ml-service2.0 is a standalone Python/FastAPI microservice (port 8100) that serves as the ML intelligence layer for AlphaForge. It consumes market data from data-service2.0, generates cross-sectional alpha signals for 285 NSE F&O symbols, and serves predictions to alpha-forge.

**Technology stack:**
- Python 3.11+, FastAPI 0.115+, uvicorn
- LightGBM (LGBMRegressor), scikit-learn, pandas, numpy
- Docker (production), pytest (1,891 tests)
- MLflow (experiment tracking), Redis (model cache)
- PostgreSQL (TimescaleDB) at port 5445 for signal storage
- Prometheus `/metrics` endpoint (16 custom metrics)

---

## 2. Data Flow

```
NSE F&O Market Data (285 symbols)
        │
        ▼
data-service2.0 (8200)          SentinelPulse (3001)
   Angel One JWT                  news/sentiment
   Upstox historical              (degraded-mode fallback)
   Instrument master (~34,500)
        │                               │
        ▼                               ▼
   DataServiceClient          SentinelPulseClient
        │                               │
        └───────────────┬───────────────┘
                        ▼
              v2c feature engineering (65 features, fs-2.0.0 base)
                        │
                        ▼
              v2c LGBMRegressor (7-day CS rank excess-return label)
                        │
                        ▼
              FeatureWeightManager (sector-regime filters, BULL suppressor)
                        │
                        ▼
              SignalLedger → PostgreSQL (TimescaleDB) at 5445
                        │
                        ▼
              latest_scores.json → WebSocket broadcast → alpha-forge (3000)
```

---

## 3. Active Model — v2c (LGBMRegressor)

| Attribute | Value |
|-----------|-------|
| Model type | LGBMRegressor |
| Feature set | fs-2.0.0 base + cross-sectional rank features — **65 features** |
| Label | 7-day cross-sectional rank excess return (vol-adjusted) |
| Universe | 278 symbols (OOS training) / 285 symbols (live sessions) |
| OOS IC | +0.040 XS-Spearman (p<0.0001, n=116k) |
| Annualized P&L | +17.07%/yr absolute; +18.06% excess vs NIFTY; IR=1.374 |
| Bootstrap CPCV PBO | ≈0.48 (B=1,000 resamples) — healthy |
| Artifact | `artifacts/v2_model/v2-20261001184632/model.pkl` |
| Stage | **LIMITED SHADOW** — long-only NSE futures authorized since Oct 1 |

**v1 model (LightGBM fs-3.0.0, 55 features, 218 symbols, IC=−0.001 OOS) is archived.** It had zero OOS predictive power and was replaced by v2c on Oct 1.

---

## 4. Model Lifecycle

```
HYPOTHESIS → BACKTEST → CHALLENGER → SHADOW → LIMITED SHADOW → PRODUCTION
                              ↑
                    TrainingOrchestrator
                    (walk-forward + bootstrap CPCV)

v1 model:   SHADOW (Sep 28 14:35 UTC) → CHALLENGER (Oct 1, IC = −0.001 OOS → unviable)
v2c model:  CHALLENGER (Oct 1) → LIMITED SHADOW (Oct 1, after gate review)
```

---

## 5. Key Components

### Signal Generation (v2c)
| Component | File | Purpose |
|-----------|------|---------|
| v2c feature families | `src/features/families/` | momentum, volatility, volume, cross_sectional, derivatives, market_structure, news |
| `LGBMRegressor` | `src/models/estimators.py` | v2c champion (IC=+0.040 OOS) |
| `ScoringPipeline` | `scripts/autorun_till_close.py` | 285 symbols × 5 min; NIFTY pre-batch fetch |
| `FeatureWeightManager` | `src/analytics/feature_weight_manager.py` | Sector-regime filters + BULL suppressor (multiplier=0.3) |
| `ReversalDetector` | `src/analytics/reversal_detector.py` | RSI + Bollinger Band reversal override |
| `SymbolICTracker` | `src/analytics/symbol_ic_tracker.py` | Per-symbol IC tracking |

### Signal Lifecycle
| Component | File | Purpose |
|-----------|------|---------|
| `SignalLedger` | `src/analytics/signal_ledger.py` | Intraday lifecycle (OPEN→SETTLED/EXPIRED) |
| `SignalDB` | `src/data/signal_db.py` | PostgreSQL backend (TimescaleDB, port 5445) |
| `SignalPromotionEngine` | `src/analytics/signal_promotion.py` | 6-gate PROMOTE/REJECT/DEMOTE (G1–G6) |
| `ScoreThresholdSweep` | `src/analytics/score_threshold_sweep.py` | Optimal threshold (current: 0.15) |
| `ForecastLedger` | `src/analytics/forecast_ledger.py` | 285-symbol score logging (brier_delta) |
| `CounterfactualLedger` | `src/analytics/counterfactual_ledger.py` | Grade blocked signals |

### Self-Improvement
| Component | File | Autonomous Decision |
|-----------|------|-------------------|
| `SelfLearningLoop` | `src/training/self_learning.py` | When to retrain (5 triggers) |
| `AlphaDecayDetector` | `src/analytics/alpha_decay.py` | IC decay → quarantine/retrain |
| `DriftDetectorV3` | `src/monitoring/drift.py` | Feature PSI → action recommendation |
| `DrawdownManager` | `src/risk/drawdown_manager.py` | 4-state risk (NORMAL→HALTED) |

### Training Pipeline
| Component | File | Purpose |
|-----------|------|---------|
| `TrainingOrchestrator` | `src/training/orchestrator.py` | End-to-end train/validate/register |
| `WalkForwardValidator` | `src/training/walk_forward.py` | 5-fold OOS validation |
| `CombinatorialPurgedCV` | `src/training/cpcv.py` | Bootstrap CPCV (B=1,000 resamples) |
| `DatasetBuilder` | `src/data/dataset_builder.py` | Frozen, leakage-validated datasets |
| `MultiHorizonLabelFactory` | `src/labels/multi_horizon.py` | [1,3,5,10,21]-bar + CS rank labels |

---

## 6. FastAPI Application (`src/main.py`)

**Port:** 8100, uvicorn

**Middleware stack** (outermost to innermost):
1. CORS
2. X-API-KEY authentication (all paths except `/health`, `/metrics`, `/docs`, `/openapi.json`, `/redoc`)
3. X-Request-ID tracing

**Route registration** (all `/v2/` prefix except training/monitoring):
```
POST   /v2/predict/single          Score a single symbol
POST   /v2/predict/batch           Score up to 50 symbols
GET    /v2/predict/regime          Regime classification
GET    /v2/predict/rankings        Full symbol rankings
POST   /v2/meta/decide             MetaDecisionEngine decision
GET    /v2/analytics/sweep         Score threshold sweep
GET    /v2/analytics/ledger        Signal ledger query
GET    /v2/explain/{symbol}        SHAP explainability
GET    /v2/signals/latest          Latest LightGBM scoring snapshot (file-based)
GET    /v2/signals/history         ForecastLedger records per session date
GET    /v2/signals/autorun-log     Last N tick samples
WS     /v2/stream/signals          Real-time WebSocket signal stream (auth via ?api_key=)
POST   /training/run               Async training (returns run_id)
POST   /training/run/sync          Synchronous training
GET    /training/status/{run_id}   Poll training status
GET    /health                     Service health (no auth required)
GET    /metrics                    Prometheus metrics (no auth required)
GET    /drift/check                Feature drift status
GET    /drift/targets              Target drift status
```

**Lifespan:**
1. Configures structlog
2. Verifies audit log integrity — refuses startup if tampered
3. Connects to Redis
4. Starts file-watcher coroutine: polls `artifacts/live_session/latest_scores.json` every 2 s, broadcasts updates to all WebSocket clients

---

## 7. Live Session Pipeline

**Entry point:** `scripts/autorun_till_close.py`

Every 5 minutes from 09:15 to 15:30 IST:
1. Fetches NIFTY LTP (dedicated pre-batch call — avoids rate-limit starvation, INFRA-008)
2. Fetches live LTPs for 285 symbols from data-service2.0
3. Scores with v2c model + 65 features + `FeatureWeightManager`
4. Applies BULL suppressor: multiplier=0.3 when NIFTY 20d return > +3%
5. Top 15% rank → LONG, bottom 15% → SHORT, middle → HOLD
6. Calls `SignalLedger.record_signals()` → PostgreSQL (first sample)
7. Calls `SignalLedger.update_mark_to_market()` → PostgreSQL (every sample)
8. Logs `ForecastLedger`, `CounterfactualLedger`, `ScoreThresholdSweep`, `SymbolICTracker`
9. Writes `artifacts/live_session/latest_scores.json` (WebSocket feed source)

Post-close (15:30 IST): resolves signals, runs `SignalPromotionEngine`, updates `LIVE_SESSION_REPORT.md`.

---

## 8. Process Management

**`scripts/daily_autorun_scheduler.py`** (new since Oct 1):
- Loops forever; starts `session_watchdog.py` on market days 09:00–15:35 IST
- Uses `pgrep` fallback to detect externally-started watchdogs (INFRA-006)
- Prevents duplicate watchdog processes

**`scripts/session_watchdog.py`** (updated Oct 1–9):
- Monitors `latest_scores.json` age; restarts autorun if stale >10 min
- 300s grace period after each restart (prevents immediate restart loop — INFRA-005)
- File-based stdout logging to avoid OS pipe buffer fill (INFRA-004)
- Never restarts after 15:30 IST (POST-004)

---

## 9. Signal Storage — PostgreSQL (TimescaleDB)

**Critical change since main merge (Oct 1):** Signal backend migrated from flat files → PostgreSQL.

```
Connection: postgresql://mlservice:mlservice_dev@localhost:5445/mlservice
Container:  ml-service-postgres (docker-compose.full-stack.yml)
Driver:     psycopg2-binary==2.9.10 (ThreadedConnectionPool, min=1, max=10)
```

**Tables:**
- `sessions` — one row per trading day (with `bull_suppressor_active` field)
- `signals` — immutable signal log (generated_at, score, direction, regime)
- `positions` — mutable intraday state (entry/exit price, WIN/LOSS/EXPIRED)
- `price_snapshots` — mark-to-market history every ~2 min
- `forecasts` — ForecastLedger entries (all 285 scores per sample)
- `symbol_ic` — per-symbol IC tracking
- `historical_outcomes` — resolved signal outcomes with attribution

---

## 10. BULL Suppressor (added Oct 6)

When NIFTY 20-day return > +3% (confirmed BULL regime):
- `strategy/feature_weights.json` → `sector_regime_filters.BULL.ALL.multiplier=0.3`
- All signal convictions multiplied by 0.3 (70% suppression)
- `long_book_limits.BULL.max_positions=4` (hard cap on LONG book)
- SHORT signals not authorized

Evidence: BULL OOS IC = −0.017 (p=0.025, statistically significant negative alpha). The model is a BEAR/SIDEWAYS detector.

---

## 11. Deployment

```bash
# Full stack (ml-service2.0 + PostgreSQL + Redis + MLflow)
docker-compose -f docker-compose.full-stack.yml up -d

# Start data dependencies
cd ../data-service2.0 && docker-compose up -d

# Persistent live session (recommended for production)
PYTHONPATH=. python3 scripts/daily_autorun_scheduler.py

# One-off session
PYTHONPATH=. python3 scripts/autorun_till_close.py

# Tests
python3 -m pytest tests/ --no-cov -q
# → 1,891 passed, 0 failed
```

---

## 12. Key Files & Artifacts

```
artifacts/
├── v2_model/v2-20261001184632/
│   ├── model.pkl          # v2c LGBMRegressor (LIMITED SHADOW)
│   └── metadata.json      # OOS IC=+0.040, PBO=0.48, all gate metrics
├── approvals/
│   └── G12-20260928143505.json    # human approval record
├── live_session/
│   └── latest_scores.json         # WebSocket feed source (updated every ~5 min)
└── signal_ledger/
    └── signals.jsonl              # JSONL audit backup

strategy/
└── feature_weights.json   # agent-editable sector-regime filters (BULL suppressor active)

reports/
├── LIVE_SESSION_REPORT.md                        # Sessions Sep 29 – Oct 9
├── forensic/PRODUCTION_GATES.md                  # Authoritative 21-gate status (Oct 9)
├── forensic/ROOT_CAUSE_REGISTER.md               # 13 root causes (10/2/1)
└── gates/regime_validation_report.md             # G_REGIME analysis (Oct 7)
```

*Architecture as of 2026-10-09 | Next review: ~2026-11-03 (20-session G_REGIME review)*
