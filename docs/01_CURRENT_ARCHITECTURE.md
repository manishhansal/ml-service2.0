# Current Architecture
**AlphaForge ml-service2.0 — As-Built Documentation**
**Date: 2026-09-28 (post-close)**
**Status: SHADOW production — scoring live, paper P&L tracking active**

---

## 1. Service Overview

ml-service2.0 is a standalone Python/FastAPI microservice (port 8100) that is the ML intelligence layer for AlphaForge. It consumes market data from data-service2.0, generates cross-sectional alpha signals for 218 NSE F&O symbols, and serves predictions to alpha-forge.

**Technology stack:**
- Python 3.11+, FastAPI 0.115+, uvicorn
- LightGBM, scikit-learn, pandas, numpy
- Docker (production), pytest (1,867 tests)
- MLflow (experiment tracking), Redis (model cache)
- Prometheus `/metrics` endpoint (16 custom metrics)

---

## 2. Data Flow

```
NSE F&O Market Data
        │
        ▼
data-service2.0 (8200)          SentinelPulse (3001)
   Angel One JWT                  news/sentiment
   Upstox historical              (degraded-mode fallback)
   Instrument master (36,173)
        │                               │
        ▼                               ▼
   DataServiceClient          SentinelPulseClient
        │                               │
        └───────────────┬───────────────┘
                        ▼
              data/1d/1d/*.parquet  (primary)
              (on-disk, 218 symbols, 5yr daily)
                        │
                        ▼
             ExpandedFeatureFactory
             (55 features, fs-3.0.0)
                        │
                        ▼
              DatasetBuilder → LeakageValidator → LightGBM training
                        │
                        ▼
              ModelRegistry (CHALLENGER→SHADOW→PRODUCTION)
                        │
                        ▼
              ScoringPipeline → 218 signals/5min → alpha-forge (3000)
```

---

## 3. Model Lifecycle

```
HYPOTHESIS → BACKTEST → CHALLENGER → SHADOW → APPROVED → PRODUCTION
                              ↑                    ↑
                    TrainingOrchestrator      G12 Human Approval
                    (walk-forward + CPCV)     (Sep 28 granted)

Current: SHADOW (since Sep 28 14:35 UTC)
```

Promotion gates (G1–G12) gate each stage transition. No auto-promotion to PRODUCTION without human approval.

---

## 4. Key Components

### Signal Generation
| Component | File | Purpose |
|-----------|------|---------|
| `ExpandedFeatureFactory` | `src/features/expanded_factory.py` | 55 PIT-certified features (fs-3.0.0) |
| `LightGBM estimator` | `src/models/estimators.py` | Champion model (IC=0.376) |
| `ScoringPipeline` | `scripts/autorun_till_close.py` | 218 symbols × 5 min |
| `FeatureWeightManager` | `src/analytics/feature_weight_manager.py` | Runtime sector-regime filters |

### Self-Improvement
| Component | File | Autonomous Decision |
|-----------|------|-------------------|
| `SelfLearningLoop` | `src/training/self_learning.py` | When to retrain (5 triggers) |
| `AlphaDecayDetector` | `src/analytics/alpha_decay.py` | IC decay → quarantine/retrain |
| `DriftDetectorV3` | `src/monitoring/drift.py` | Feature PSI → action recommendation |
| `SignalPromotionEngine` | `src/analytics/signal_promotion.py` | 6-gate PROMOTE/REJECT/DEMOTE |
| `DrawdownManager` | `src/risk/drawdown_manager.py` | 4-state risk (NORMAL→HALTED) |

### Phil Integrations (Sep 28)
| Component | File | Purpose |
|-----------|------|---------|
| `ScoreThresholdSweep` | `src/analytics/score_threshold_sweep.py` | Optimal entry threshold |
| `ForecastLedger` | `src/analytics/forecast_ledger.py` | 218-symbol brier_delta logging |
| `CounterfactualLedger` | `src/analytics/counterfactual_ledger.py` | Grade blocked signals |
| `NSEEventWatcher` | `scripts/nse_event_watcher.py` | Intraday catalyst detection |
| `FeatureWeightManager` | `src/analytics/feature_weight_manager.py` | Agent-editable filters |

### Training Pipeline
| Component | File | Purpose |
|-----------|------|---------|
| `TrainingOrchestrator` | `src/training/orchestrator.py` | End-to-end train/validate/register |
| `WalkForwardValidator` | `src/training/walk_forward.py` | 5-fold OOS validation |
| `CombinatorialPurgedCV` | `src/training/cpcv.py` | PBO estimation (15 paths) |
| `DatasetBuilder` | `src/data/dataset_builder.py` | Frozen, leakage-validated datasets |
| `MultiHorizonLabelFactory` | `src/labels/multi_horizon.py` | [1,3,5,10,21]-bar labels |

---

## 5. API Endpoints

### Prediction
- `POST /predict` — score a single symbol
- `POST /predict/batch` — score up to 50 symbols

### Training
- `POST /training/run` — async training (returns run_id)
- `POST /training/run/sync` — synchronous training
- `GET /training/status/{run_id}` — poll training status
- `GET /training/result/{run_id}` — get full training report

### Monitoring
- `GET /health` — service health
- `GET /metrics` — Prometheus metrics (16 custom)
- `GET /drift/check` — feature drift status
- `GET /drift/targets` — target drift status

---

## 6. Shadow Mode Configuration

**Active since: Sep 28 14:35 UTC**

```json
// artifacts/shadow_config.json
{
  "mode": "shadow",
  "model_version": "1.0.0-20260928053134956099",
  "monitoring": {
    "score_every_bars": 1,
    "drift_check_interval": 5,
    "alert_on_ic_drop": 0.05,
    "alert_on_pnl_drawdown": 0.05
  },
  "shadow_rules": {
    "emit_signals": true,
    "execute_trades": false,
    "paper_pnl": true
  }
}
```

Demotion triggers (auto-rollback):
- 3-day rolling IC < 0
- 5+ consecutive paper-loss days
- Drift severity = HIGH
- Paper drawdown > 5%

---

## 7. Deployment

```bash
# Start all services
docker-compose -f docker-compose.yml up -d  # ml-service2.0
cd ../data-service2.0 && docker-compose up -d

# Live session (market hours 9:15–15:30 IST)
PYTHONPATH=. python3 scripts/autorun_till_close.py

# Event watcher (run every 15 min via cron)
PYTHONPATH=. python3 scripts/nse_event_watcher.py check

# Tests
python3 -m pytest tests/ --no-cov -q
# → 1,867 passed, 13 skipped, 0 failed
```

---

## 8. Key Files & Artifacts

```
artifacts/
├── expanded_lgbm/1.0.0-20260928053134956099/
│   ├── model.pkl          # LightGBM champion (SHADOW)
│   ├── metadata.json      # stage=shadow, all gate metrics
│   └── model.pkl.sha256   # integrity hash
├── approvals/
│   └── G12-20260928143505.json    # human approval record
├── shadow_config.json     # shadow monitoring configuration
├── promotion_audit.jsonl  # append-only promotion log
├── forward_paper/
│   ├── signals.jsonl      # 65 v1 signals
│   ├── signals_v2.jsonl   # 153 v2 signals
│   ├── forecasts.jsonl    # Phil ForecastLedger (all 218)
│   └── outcomes.jsonl     # resolved outcomes
└── live_session/
    ├── autorun_log.jsonl  # 23 samples Sep 28
    └── session_summary.json

strategy/
└── feature_weights.json   # agent-editable sector-regime filters

reports/                   # all current (updated Sep 28)
docs/                      # architectural specs (01–27)
```

*Architecture as of 2026-09-28 | Next review: 2026-10-15 (production promotion)*
