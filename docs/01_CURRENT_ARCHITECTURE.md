# Current Architecture
**ml-service2.0 — As-Built Architecture Documentation**

*Date: 2026-09-24*
*Status: Documents the existing implementation, not the target state*

---

## 1. Service Overview

ml-service2.0 is a standalone Python/FastAPI microservice running on port 8100. It is the ML intelligence layer for AlphaForge, consuming market data from data-service2.0 and news intelligence from SentinelPulse, and serving predictions to alpha-forge.

**Technology stack:**
- Python 3.11+
- FastAPI 0.115.6 (async)
- Pydantic V2 (strict mode)
- Redis (async, in-process LRU fallback)
- MLflow (experiment tracking — currently unconfigured)
- Structlog (structured JSON logging)
- Uvicorn (ASGI)

---

## 2. Module Map

```
src/
├── main.py                    # FastAPI app, middleware, lifespan
├── config.py                  # Pydantic Settings (env vars)
├── logging_config.py          # Structlog configuration
│
├── api/                       # HTTP route handlers
│   ├── predict.py             # POST /v2/predict/{regime,rankings,strategy,risk,...}
│   ├── meta.py                # POST /v2/meta/decide
│   ├── analytics.py           # POST /v2/analytics/{greeks,vol-surface,gex,vpin}
│   ├── explain.py             # POST /v2/explain/{regime,ranker,strategy,risk}
│   ├── training.py            # POST /train/{start,status}
│   └── monitoring.py          # GET /monitoring/{health,alerts,drift}
│
├── features/                  # Feature construction
│   ├── pipeline.py            # FeaturePipeline (fetches + assembles FeatureVector)
│   ├── qlib_engine.py         # QlibFeatureEngine (Alpha158/Alpha360)
│   └── leakage_validator.py   # LeakageValidator, LookAheadGuard, PITViolationError
│
├── models/                    # Model implementations (all heuristic currently)
│   ├── base.py                # BaseMLModel ABC (orphaned — no concrete subclasses)
│   ├── regime_classifier.py   # RegimeClassifier (XGBoost when trained)
│   ├── stock_ranker.py        # StockRanker (LightGBM LTR when trained)
│   ├── strategy_selector.py   # StrategySelector (CatBoost when trained)
│   ├── risk_predictor.py      # RiskPredictor (XGBoost when trained)
│   ├── price_forecaster.py    # PriceForecaster (TFT placeholder — heuristic)
│   ├── iv_regime_classifier.py# IVRegimeClassifier (rule-based)
│   ├── rl_execution_agent.py  # RLExecutionAgent (FinRL PPO when trained)
│   └── portfolio_optimizer.py # PortfolioOptimizer (Riskfolio-Lib)
│
├── training/                  # Training infrastructure
│   ├── pipeline.py            # TrainingPipeline (PurgedKFold + gates + audit)
│   ├── purged_kfold.py        # PurgedKFoldSplitter (embargo + purge)
│   ├── hpo.py                 # HyperparameterOptimizer (Optuna)
│   ├── online_learner.py      # OnlineLearner (incremental updates)
│   └── mlflow_tracker.py      # MLflowTracker wrapper
│
├── meta/                      # MetaDecisionEngine components
│   ├── engine.py              # MetaDecisionEngine (14-step algorithm)
│   ├── ensemble.py            # EnsembleWeighter (IC-proportional)
│   ├── calibration.py         # CalibrationLayer (Platt/Isotonic)
│   ├── abstention.py          # AbstentionPolicy (5 conditions)
│   ├── llm_reasoner.py        # LLMNewsReasoner (FinBERT)
│   └── calibration.py        # ConfidenceDecomposer
│
├── meta_engine/               # Phase 2/3 wrapper
│   └── engine.py              # MetaEngine (stub), MetaEngineV3 (wrapper)
│
├── registry/                  # Model lifecycle
│   ├── registry.py            # ModelRegistry (file-based, SHA256)
│   ├── promotion.py           # ModelPromotion (6 gates)
│   └── approval_token.py      # Time-limited approval token
│
├── monitoring/                # Drift detection
│   ├── drift.py               # DriftDetector, DriftDetectorV3 (PSI)
│   └── drift_monitor.py       # DriftMonitor (Evidently/NannyML wrappers)
│
├── analytics/                 # Financial analytics
│   ├── greeks.py              # Black-Scholes/Black-76 Greeks
│   ├── gex.py                 # Dealer GEX computation
│   ├── vpin.py                # Volume-synchronised PIN
│   └── vol_surface.py         # IV surface (SVI parameterisation)
│
├── clients/                   # External service clients
│   ├── data_service.py        # DataServiceClient (circuit breaker, quality gates)
│   ├── sentinel_pulse.py      # SentinelPulseClient (LRU cache, retry)
│   ├── grpc_client.py         # gRPC streaming client (unused)
│   └── market_data_pb2*.py    # Generated proto stubs
│
├── explainability/            # SHAP explainability
│   └── explainer.py           # ModelExplainer (TreeExplainer/KernelExplainer)
│
├── audit/                     # Immutable audit log
│   └── logger.py              # AuditLogger (JSONL, SHA256 chain)
│
├── cache/                     # Caching
│   └── redis_cache.py         # RedisCache (async), LRUCache (in-process)
│
├── streaming/                 # WebSocket
│   └── streamer.py            # SignalStreamer (broadcast manager)
│
├── schemas/                   # Pydantic V2 schemas
│   ├── base.py                # Enums, BaseSchema
│   ├── predictions.py         # All prediction request/response schemas
│   ├── features.py            # FeatureVector schema
│   ├── meta.py                # MetaOutput, GoNoGoDecision, NewsSignal
│   ├── monitoring.py          # DriftAlert, MonitoringHealth
│   ├── registry.py            # ModelArtifact, PromotionDecision, GateEvaluation
│   └── streaming.py           # WebSocket message schemas
│
└── data/
    └── contracts.py           # OHLCVBar, DataServiceResponse, SentinelNewsContext
```

---

## 3. Request Flow: Live Inference

```
AlphaForge
    │
    │ POST /v2/meta/decide
    │ {symbol, regime, model_outputs[], news_context}
    │ X-API-KEY: ****
    │
    ▼
api_key_middleware → request_id_middleware
    │
    ▼
meta.router (src/api/meta.py)
    │
    ├── 1. Validate request schema (Pydantic V2)
    │
    ├── 2. MetaDecisionEngine.decide(model_outputs, symbol, regime, news_context)
    │       │
    │       ├── Partition: available vs UNAVAILABLE model outputs
    │       ├── Absorbing identity: all UNAVAILABLE → NO_TRADE
    │       ├── Quorum guard: < 3 available → NO_TRADE
    │       ├── CalibrationLayer.calibrate(model_id, raw_score) [currently identity]
    │       ├── EnsembleWeighter.compute_weights(model_ids, regime) [currently equal]
    │       ├── Compute directional agreement_ratio
    │       ├── Compute weighted ensemble score
    │       ├── AbstentionPolicy.check(agreement, data_quality=1.0, confidence, ...)
    │       ├── LLMNewsReasoner.reason_sync(news_context) [optional FinBERT]
    │       ├── Determine final action: BUY/SELL/WAIT/NO_TRADE
    │       ├── Compute final_confidence = mean_confidence × agreement_ratio
    │       ├── Determine weakest-link provenance
    │       ├── Build XAI explainability block
    │       └── Return MetaOutput (frozen Pydantic model)
    │
    ├── 3. Convert MetaOutput → API response
    │
    └── 4. Return JSON response (HTTP 200)
```

---

## 4. Request Flow: Model Training

```
POST /train/start
    │
    ├── Validate request (model_name, hyperparameters, n_splits, ...)
    │
    ├── TrainingPipeline.run_training(
    │       model_name, X, y, timestamps, model_factory, search_space
    │   )
    │       │
    │       ├── Compute dataset SHA256 hash
    │       ├── HyperparameterOptimizer.optimize() [if search_space provided]
    │       ├── PurgedKFoldSplitter.split() [n_splits folds, embargo_days]
    │       │       ├── Sort by timestamp
    │       │       ├── Apply embargo mask
    │       │       └── Apply purge mask
    │       ├── Per fold: fit model_factory(params), predict, compute Spearman IC
    │       ├── Compute mean IC, net Sharpe (−10bps tx cost), max drawdown, PBO
    │       ├── Apply acceptance gates:
    │       │       ├── IC gate: mean IC >= min_ic (0.02)
    │       │       ├── Sharpe gate: net Sharpe >= 0.0
    │       │       └── PBO gate: PBO <= max_pbo (0.5)
    │       ├── Advance lifecycle to CHALLENGER if PASSED
    │       └── Write AuditLogger entry (JSONL)
    │
    └── Return TrainingResult (run_id, outcome, metrics)
```

---

## 5. Model Promotion Flow

```
ModelPromotion.evaluate_all_gates(challenger, champion)
    │
    ├── DATA gate
    │   ├── FAIL: final_oos_used_for_selection=True → FINAL_OOS_CONTAMINATED
    │   ├── FAIL: evidence_level in {LEVEL_C, LEVEL_D}
    │   ├── FAIL: look_ahead_validated=False
    │   └── INSUFFICIENT_EVIDENCE: oos_count < 60
    │
    ├── PREDICTIVE gate
    │   ├── No champion: challenger IC must be > 0.0
    │   └── With champion: challenger IC must exceed champion IC by margin
    │
    ├── CALIBRATION gate
    │   ├── No champion: Brier score must be < 0.25
    │   └── With champion: Brier score tolerance = 0.01 worse allowed
    │
    ├── EXECUTION gate
    │   ├── No champion: net_return_validation must be >= 0
    │   └── With champion: turnover <= champion × 1.20
    │
    ├── RISK gate
    │   ├── No champion: max_drawdown <= 0.20
    │   └── With champion: max_drawdown tolerance = 2pp worse allowed
    │
    └── STABILITY gate
        ├── FAIL: ic_decay_status in {FAILED, SIGNIFICANT_DECAY}
        └── FAIL: feature_drift_severity in {HIGH, CRITICAL}
    │
    ├── OUTCOME: PROMOTE (all PASS) → requires human ApprovalToken
    ├── OUTCOME: REJECTED (any FAIL)
    ├── OUTCOME: BLOCKED (any INSUFFICIENT_EVIDENCE)
    └── OUTCOME: FINAL_OOS_CONTAMINATED (permanent)
```

---

## 6. Data Quality Gate Flow

```
FeaturePipeline.build_vector(symbol, timestamp)
    │
    ├── DataServiceClient.get_historical_bars(symbol, interval)
    │   │
    │   ├── Circuit breaker check (open if ≥3 failures in 60s)
    │   ├── HTTP GET with X-API-KEY
    │   ├── Check signalEngineAllowed → SignalEngineNotAllowedError
    │   ├── Check DataConfidenceScore → LowDataConfidenceError
    │   └── Parse OHLCVBar list
    │
    ├── SentinelPulseClient.fetch_news_context(symbol)
    │   │
    │   ├── LRU cache check (500 entries, 90s TTL)
    │   ├── HTTP GET with Bearer token
    │   ├── Validate mandatory fields
    │   └── Return SentinelNewsContext or None
    │
    ├── QlibFeatureEngine.compute_alpha158(ohlcv_df, symbol)
    │   └── ~15 Qlib-style factors (native implementation)
    │
    └── Return FeatureVector with provenance set based on data quality
```

---

## 7. API Surface

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | None | Service health |
| POST | `/v2/predict/regime` | API Key | Market regime classification |
| POST | `/v2/predict/rankings` | API Key | Stock ranking |
| POST | `/v2/predict/strategy` | API Key | Strategy selection |
| POST | `/v2/predict/risk` | API Key | Risk estimation |
| POST | `/v2/predict/execution` | API Key | RL execution timing |
| POST | `/v2/predict/portfolio` | API Key | Portfolio optimization (legacy) |
| POST | `/v2/predict/portfolio-v2` | API Key | Portfolio optimization (Riskfolio-Lib) |
| POST | `/v2/predict/price-regime` | API Key | 1h ahead price regime (TFT) |
| POST | `/v2/predict/iv-regime` | API Key | IV regime classification |
| POST | `/v2/meta/decide` | API Key | MetaDecisionEngine Go/No-Go |
| POST | `/v2/analytics/greeks` | API Key | Option chain Greeks |
| GET  | `/v2/analytics/vol-surface` | API Key | Volatility surface |
| POST | `/v2/analytics/vol-surface` | API Key | Compute vol surface |
| POST | `/v2/analytics/gex` | API Key | Dealer GEX |
| POST | `/v2/analytics/vpin` | API Key | VPIN |
| POST | `/v2/explain/{model}` | API Key | SHAP explanation |
| GET  | `/v2/models/status` | API Key | Model load status |
| GET  | `/v2/models/registry` | API Key | Full model registry |
| POST | `/train/start` | API Key | Start training run |
| GET  | `/train/status/{run_id}` | API Key | Training run status |
| GET  | `/monitoring/health` | API Key | Monitoring health |
| GET  | `/monitoring/alerts` | API Key | Active drift alerts |
| WS   | `/v2/stream/signals` | Query param api_key | Real-time signal stream |

---

## 8. Configuration Surface

Key settings (from `src/config.py` Settings class):

| Setting | Default | Description |
|---|---|---|
| `deployment_mode` | `research` | research/paper/shadow/validated_ml_only |
| `ml_service_api_key` | Required | API key for authentication |
| `data_service_2_url` | `http://localhost:8200` | DataService URL |
| `data_service_api_key` | Required | DataService auth key |
| `sentinel_pulse_url` | `http://localhost:3001` | SentinelPulse URL |
| `sentinel_pulse_api_key` | Required | SentinelPulse auth key |
| `redis_url` | `redis://localhost:6379` | Redis connection |
| `min_confidence_score` | 60 | DataConfidenceScore threshold |
| `embargo_period_days` | 10 | Default embargo for PurgedKFold |
| `model_acceptance_min_ic` | 0.02 | Minimum IC for model acceptance |
| `model_acceptance_max_pbo` | 0.5 | Maximum PBO for model acceptance |
| `optuna_n_trials` | 50 | Minimum HPO trials |
| `llm_model_name` | `ProsusAI/finbert` | FinBERT model name |
| `llm_inference_timeout_ms` | 120 | LLM inference timeout |
| `news_context_cache_ttl` | 90 | SentinelPulse cache TTL (seconds) |
| `cpcv_n_paths` | 10 | CPCV paths for PBO estimation |

---

*End of Current Architecture*
