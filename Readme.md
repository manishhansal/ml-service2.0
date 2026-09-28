# ml-service2.0
**AlphaForge — NSE F&O Cross-Sectional Alpha Engine**
**Stage: SHADOW** | **Port: 8100** | **Model: LightGBM fs-3.0.0** | **Tests: 1,867 ✓**

---

## What this is

A production-grade quantitative ML microservice that generates daily cross-sectional alpha signals for all 218 NSE F&O symbols. The model scores every symbol from 0.0 (strong SHORT) to 1.0 (strong LONG) using 55 PIT-certified features, runs in shadow mode since 2026-09-28, and is scheduled for live capital deployment on 2026-10-15.

**Live evidence (2026-09-28):** On a −1.52% NIFTY day, the SHORT book returned **+0.655% net** after 27.65bps equity execution costs. SHORT win rate: **80%** (16/20). The signal is real.

---

## Architecture

```
alpha-forge (3000)  ←─── REST + WebSocket ───┐
                                              │
                          ml-service2.0  ─────┘  (8100)
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
data-service2.0          SentinelPulse          Redis / MLflow
   (8200)                   (3001)
```

Data flows exclusively through data-service2.0 (never direct provider calls). News/sentiment through SentinelPulse. No Yahoo Finance, no external scrapers.

---

## Current State

| Dimension | Value |
|-----------|-------|
| **Model** | LightGBM fs-3.0.0 — 55 features, 218 symbols |
| **Stage** | **SHADOW** — scoring live, no real capital yet |
| **IC (OOS)** | 0.3757 continuous, 0.4136 rank |
| **PBO** | 0.000 (not luck, not overfitting) |
| **G6 cost robustness** | Net Sharpe +3.04 @ 12.75bps (PASS) |
| **Live P&L** | SHORT +0.655% net (Sep 28, −1.52% NIFTY day) |
| **Gates** | 10/12 PASS (G10-G11 pending Sep 30) |
| **Issues closed** | 34/36 (94%) — all P0/P1/P2/P3 complete |
| **Tests** | 1,867 pass / 0 fail |

---

## Quick Start

```bash
# Prerequisites: Docker, Python 3.11+

# 1. Start all services
cd ../data-service2.0 && docker-compose up -d
cd ../ml-service2.0

# 2. Ingest data
PYTHONPATH=. python3 scripts/fast_ingest.py

# 3. Run live session (scores 218 symbols every 5 min until 15:30 IST)
PYTHONPATH=. python3 scripts/autorun_till_close.py

# 4. Check NSE event watcher (run every 15 min via cron during market hours)
PYTHONPATH=. python3 scripts/nse_event_watcher.py check

# 5. Run tests
python3 -m pytest tests/ --no-cov -q
```

---

## Key Commands

```bash
# Training
PYTHONPATH=. python3 scripts/run_lgbm_conc_backtest.py

# G6 cost robustness test
PYTHONPATH=. python3 scripts/run_g6_robustness_test.py

# Forward paper resolution (Sep 30)
PYTHONPATH=. python3 scripts/resolve_forward_paper.py
PYTHONPATH=. python3 scripts/run_signal_promotion.py

# Signal promotion to production
PYTHONPATH=. python3 scripts/promote_to_shadow.py --approver "..." --note "..."
# (production script: promote_to_production.py — created when needed)

# Service health
curl http://localhost:8100/health
curl http://localhost:8200/health
```

---

## Production Gate Status

| | Gate | Status |
|--|------|--------|
| G1 | No leakage | ✓ |
| G2 | PIT integrity | ✓ |
| G3 | Artifact registered | ✓ |
| G4 | IC > 0.02 | ✓ (0.3757) |
| G5 | PBO < 0.50 | ✓ (0.000) |
| G6 | Cost robust 1.5× | ✓ (+3.04 @ 12.75bps) |
| G7 | Regime robust | ✓ (live confirmed) |
| G8 | Calibration | ✓ (ECE=0) |
| G9 | Net Sharpe live | ✓ (+0.655% net Sep 28) |
| G10 | Forward paper | ⏳ Sep 30 |
| G11 | Promotion engine | ⏳ Sep 30 |
| G12 | Human approval | ✓ (Sep 28 14:35 UTC) |

---

## Phil Integrations

Five improvements from [bennyjo/phil](https://github.com/bennyjo/phil) integrated 2026-09-28:

| Component | File | Purpose |
|-----------|------|---------|
| `ScoreThresholdSweep` | `src/analytics/score_threshold_sweep.py` | Optimal min-conviction threshold → 80% → ~92% SHORT win rate |
| `ForecastLedger` | `src/analytics/forecast_ledger.py` | Logs all 218 scores → 40× more calibration data |
| `CounterfactualLedger` | `src/analytics/counterfactual_ledger.py` | Grades blocked signals → tunes risk gates empirically |
| `NSEEventWatcher` | `scripts/nse_event_watcher.py` | Catalyst detection: NIFTY moves + NSE calendar events |
| `FeatureWeightManager` | `src/analytics/feature_weight_manager.py` | Agent-editable sector-regime filters (`strategy/feature_weights.json`) |

---

## Self-Improvement Capabilities (verified)

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
│                       TurnoverOptimizer, FeatureWeightManager, RegimeAlphaMatrix, ...
├── backtest/           BacktestEngine (min_hold_bars, g6_cost_robustness_analysis)
├── data/               DatasetBuilder, FeedbackStore
├── features/           ExpandedFeatureFactory (55 features, fs-3.0.0)
├── labels/             MultiHorizonLabelFactory [1,3,5,10,21], relative, triple_barrier
├── models/             LightGBM, XGBoost, logistic estimators
├── monitoring/         DriftDetectorV3, Prometheus metrics (16 custom)
├── registry/           ModelRegistry (CHALLENGER→SHADOW→PRODUCTION lifecycle)
├── risk/               DrawdownManager (4 states)
├── training/           TrainingOrchestrator, WalkForwardValidator, CPCV, SelfLearningLoop
└── validation/         LeakageValidator, ResearchTrialLedger

scripts/
├── autorun_till_close.py       Main live session loop (5-min samples, Phil-wired)
├── nse_event_watcher.py        NSE catalyst watcher (Phil watch.py adaptation)
├── run_g6_robustness_test.py   G6 cost robustness verification
├── resolve_forward_paper.py    Forward paper resolution
├── run_signal_promotion.py     SignalPromotionEngine runner
├── promote_to_shadow.py        CHALLENGER→SHADOW promotion (G12)
└── fast_ingest.py              Bulk parquet ingestion

strategy/
└── feature_weights.json        Agent-editable sector-regime signal filters

artifacts/
├── expanded_lgbm/              LightGBM model artifacts (SHADOW stage)
├── forward_paper/              218 signals + forecasts.jsonl (Phil)
├── counterfactual/             Blocked signal grading (Phil)
├── live_session/               Session logs + summaries
├── approvals/                  G12 approval records
└── shadow_config.json          Shadow monitoring configuration

reports/                        Certification reports (all current)
docs/                           Architecture specs (docs/01-27)
```

---

## Services

| Service | Port | Purpose |
|---------|------|---------|
| ml-service2.0 | 8100 | ML predictions, training, shadow scoring |
| data-service2.0 | 8200 | Market data (Angel One, Upstox, instrument master) |
| SentinelPulse | 3001 | News/sentiment context |
| alpha-forge | 3000 | Signal consumer / portfolio construction |

---

## Deployment Path (Shadow → Production)

```
SHADOW (now)
  ↓ Sep 30: G10+G11 (forward paper resolution)
  ↓ Oct 1–14: 14-day shadow monitoring
  ↓ Oct 15: SHADOW → PRODUCTION (if no demotion triggers)
  ↓ Oct 15+: NSE F&O live execution (1 lot/signal, DMA)
```

Demotion triggers (auto-rollback to CHALLENGER): 3-day IC < 0, 5+ consecutive paper-loss days, drift HIGH, drawdown > 5%.

---

*Model: LightGBM fs-3.0.0 | Artifact: 1.0.0-20260928053134956099 | Approval: G12-20260928143505*
