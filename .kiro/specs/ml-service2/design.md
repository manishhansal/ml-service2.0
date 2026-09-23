# Design Document: ml-service2.0

## Overview

ml-service2.0 is a standalone, institutional-grade Python 3.11+ FastAPI microservice that replaces and vastly upgrades the legacy ML service embedded in alpha-forge. It serves as the dedicated AI/ML inference, training, and monitoring layer for the AlphaForge trading platform.

The service runs on **port 8100** and sits at the intersection of three external systems:

- **data-service2.0** (port 8200) — the sole authority for all market data (NSE/NFO, OHLCV, live ticks, option chains, F&O universe), accessed via REST and gRPC streaming.
- **SentinelPulse** (port 3001) — the sole source of NLP/news intelligence (PIT-correct sentiment, event impact scores, market regime context), accessed via REST.
- **alpha-forge** (port 3000) — the primary consumer of ml-service2.0 predictions, receiving responses via REST and streaming signals via WebSocket.

### Design Goals

1. **PIT Correctness** — No feature used in training or inference may contain information unavailable at prediction time.
2. **Evidence-Gated Execution** — Every prediction carries a `PredictionProvenance` label; only `TRAINED_MODEL` provenance is eligible for live capital deployment.
3. **Self-Learning** — Models adapt to regime shifts via online learning and are governed by a six-gate promotion pipeline before any artifact becomes champion.
4. **Full Observability** — Every prediction, training run, and promotion decision is recorded in an append-only audit log with structured logs via `structlog`.
5. **Explainability by Default** — Every model response carries SHAP attributions; the Meta-Decision Engine includes a natural-language XAI rationale.
6. **Strict Typing** — Python 3.11+ with `from __future__ import annotations`, Pydantic V2 `ConfigDict(strict=True)`, `mypy --strict` zero errors.

---

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           ml-service2.0  (port 8100)                                    │
│                                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────┐    │
│  │  FastAPI Application Layer (src/api/)                                          │    │
│  │  /v2/predict/*  /v2/meta/*  /v2/analytics/*  /v2/stream/signals (WS)           │    │
│  │  /health  /models/status  /monitoring/*  /training/*  /features/quality         │    │
│  └──────────────────────────────────┬──────────────────────────┬───────────────────┘    │
│                 Request              │                          │ WS Push                │
│                                      ▼                          ▼                       │
│  ┌───────────────────────────┐  ┌─────────────────────────────────────────────────┐    │
│  │  Feature Pipeline          │  │  Signal Streamer (src/streaming/)               │    │
│  │  (src/features/)           │  │  WebSocket manager, MetaOutput broadcaster      │    │
│  │  FeaturePipeline           │  └─────────────────────────────────────────────────┘    │
│  │  QlibFeatureEngine         │                                                         │
│  │  LeakageValidator          │                                                         │
│  └──────────────┬─────────────┘                                                         │
│                 │ Feature Vectors                                                        │
│                 ▼                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                      Model Layer (src/models/)                                   │   │
│  │  RegimeClassifier  StockRanker  StrategySelector  RiskPredictor                  │   │
│  │  PortfolioOptimizer  RLExecutionAgent  PriceForecaster  IVRegimeClassifier        │   │
│  └──────────────────────────────┬───────────────────────────────────────────────────┘   │
│                                  │ Model Outputs                                         │
│                                  ▼                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │               Meta Decision Engine (src/meta/)                                   │   │
│  │  EnsembleWeighter  CalibrationLayer  AbstentionPolicy                            │   │
│  │  LLMNewsReasoner (FinGPT/FinBERT via LangChain)                                  │   │
│  │  ConfidenceDecomposer  MetaOutput assembler                                      │   │
│  └──────────────────────────────┬───────────────────────────────────────────────────┘   │
│                                  │ MetaOutput + SHAP                                    │
│                                  ▼                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │  Explainability Layer (src/explainability/)                                      │   │
│  │  SHAP TreeExplainer (XGBoost/LightGBM/CatBoost)  KernelExplainer (NN)            │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │  Training & Lifecycle Layer                                                      │   │
│  │  TrainingPipeline (src/training/)   ModelRegistry (src/registry/)                │   │
│  │  OnlineLearner (src/training/)      ModelPromotion (src/registry/)                │   │
│  └────────────────────────────────────────────┬─────────────────────────────────────┘   │
│                                               │                                          │
│  ┌──────────────────────────────────────────┐ │                                          │
│  │  Monitoring Layer (src/monitoring/)      │ │                                          │
│  │  DriftMonitor (Evidently AI + NannyML)   │ │                                          │
│  │  PerformanceMonitor  AlertSystem         │ │                                          │
│  └──────────────────┬─────────────────────--┘ │                                          │
│                     │                          │                                          │
│  ┌──────────────────▼──────────────────────────▼──────────────────────────────────────┐  │
│  │  Infrastructure Layer                                                               │  │
│  │  Redis Cache (feature:60s TTL, news:90s TTL, signal cache)                         │  │
│  │  MLflow (experiment tracking, model registry, artifact storage)                    │  │
│  │  Artifact Store (SHA-256 checksums, immutable)                                     │  │
│  │  AuditLogger (JSON Lines, append-only, structlog)                                  │  │
│  └─────────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
        │  REST + gRPC                    │  REST                      │  REST + WebSocket
        ▼                                 ▼                            ▼
┌──────────────────┐          ┌──────────────────────┐      ┌─────────────────────┐
│  data-service2.0 │          │    SentinelPulse      │      │    alpha-forge      │
│  port 8200       │          │    port 3001          │      │    port 3000        │
│  Market data     │          │    NLP/news sole      │      │    Primary consumer │
│  sole authority  │          │    authority          │      │                     │
└──────────────────┘          └──────────────────────┘      └─────────────────────┘
```

### Internal Message Flows

**Inference Path:**
```
alpha-forge POST /v2/predict/* or /v2/meta/decide
  → API Router validates X-API-KEY, assigns X-Request-ID
  → FeaturePipeline.build_vector(symbol, timestamp, mode=INFERENCE)
      → Redis cache check (key=feature:{symbol}:{bucket}, TTL=60s)
      → DataServiceClient.fetch_features(symbol) [REST + optional gRPC streaming]
      → SentinelPulseClient.fetch_news_context(symbol) [REST, LRU TTL=90s]
      → QlibFeatureEngine.compute_alpha158() → 158 cross-sectional factors
      → FeatureQualityReport assembled (missing/imputed/rejected counts)
  → Model inference (e.g. RegimeClassifier.predict(feature_vector))
  → ModelExplainer.explain(model, features, prediction) → SHAP contributions
  → MetaDecisionEngine.decide(all_model_outputs, news_context)
      → CalibrationLayer.calibrate() [Platt / isotonic per model]
      → EnsembleWeighter.weight() [regime-aware, IC-proportional]
      → AbstentionPolicy.check() → NO_TRADE if agreement_ratio < 0.5 etc.
      → LLMNewsReasoner.reason() [FinGPT/FinBERT, 120ms timeout]
      → ConfidenceDecomposer.decompose()
      → MetaOutput assembled
  → AuditLogger.log_prediction(request_id, symbol, meta_output, latency_ms)
  → Response with PredictionProvenance + SHAP + natural-language rationale
```

**Training Path:**
```
POST /training/run  →  TrainingPipeline.run(config)
  → FeaturePipeline.build_training_dataset(pit_date, symbols)
  → LeakageValidator.validate(feature_matrix, labels)
      → Abort with TrainingAbortedError if |correlation| > 0.05 for any look-ahead window
  → PurgedKFoldSplitter.split(X, y, timestamps, embargo_days=10)
  → For each fold: QlibFeatureEngine.compute_alpha158/360()
  → Optuna HPO: 50+ trials, objective = mean Spearman IC across held-out folds
  → CPCV: 10+ overlapping test paths, compute PBO
  → ModelAcceptanceGate: IC >= 0.02 AND Sharpe >= 0.0 AND PBO <= 0.5
  → MLflow.log_run(hyperparams, IC_per_fold, Sharpe, max_dd, PBO, dataset_sha256)
  → AuditLogger.log_training_run(run_id, start, end, dataset_hash, gates)
  → ModelPromotion.evaluate_gates(challenger, champion)
      → DATA / PREDICTIVE / CALIBRATION / EXECUTION / RISK / STABILITY gates
      → If all PASS → PromotionDecision(PROMOTE, HUMAN_APPROVAL_REQUIRED)
      → If any FAIL → PromotionDecision(REJECTED, reason)
      → await approvalToken from authorized reviewer
  → ModelRegistry.register_champion(artifact, sha256_checksum)
```

**Monitoring Path (scheduled 00:00 UTC daily):**
```
DriftMonitor.run_daily()
  → Fetch 7-day rolling production feature distribution
  → Evidently AI DataDriftPreset(current=7d_window, reference=training_distribution)
  → PSI computation: 10 equal-frequency bins per feature
      → 0.2 < PSI <= 0.25 → DRIFT_ALERT severity=MEDIUM
      → PSI > 0.25 → DRIFT_ALERT severity=HIGH → trigger OnlineLearner
  → NannyML CBPE → estimated IC + ECE per model (without ground-truth labels)
      → estimated IC < 0.015 → PERFORMANCE_DEGRADATION_ALERT → block model
  → All alerts emitted via structlog + AlertSystem → GET /monitoring/alerts
```

**Online Learning Path:**
```
PerformanceMonitor.check() detects IC degradation > 20% over rolling 30d window
  → OnlineLearner.initiate_update(model, last_60_trading_days)
      → Guard: model must be TRAINED_MODEL provenance
      → Guard: consecutive_online_updates < 5 (else require full retrain)
      → XGBoost model.update() / LightGBM incremental fit on new data
      → Version new artifact: {version}-online-{YYYY-MM-DD}
      → Validate on most recent 10 trading days (held-out)
          → If new IC < prior IC → discard, log ONLINE_UPDATE_REJECTED
          → If new IC >= prior IC → register artifact, increment counter
      → Continue serving from existing champion during update window
      → AuditLogger.log_online_update(run_id, prior_version, new_version, ic_delta)
```

---

## Components and Interfaces

### 1. FeaturePipeline (`src/features/pipeline.py`)

**Purpose:** PIT-correct assembly of feature vectors from data-service2.0 and SentinelPulse.

**Responsibilities:**
- Fetch market features from data-service2.0 with 10s timeout per call
- Fetch news context from SentinelPulse with 10s timeout per call
- Enforce PIT correctness: all source timestamps must be strictly before the feature vector's target timestamp
- Apply degraded-mode substitution when SentinelPulse is unavailable (zero numeric, NEUTRAL directional)
- Produce `FeatureQualityReport` for every batch
- Support backtest mode with `?pit_date=YYYY-MM-DD` query parameter
- Process 500 symbols within 60 seconds under normal latency conditions
- Check `signalEngineAllowed` and `DataConfidenceScore` from data-service2.0 before proceeding

**Interface:**
```python
class FeaturePipeline:
    async def build_vector(
        self,
        symbol: str,
        timestamp: datetime,
        mode: Literal["inference", "backtest"],
        pit_date: date | None = None,
    ) -> tuple[FeatureVector, FeatureQualityReport]: ...

    async def build_batch(
        self,
        symbols: list[str],
        timestamp: datetime,
        mode: Literal["inference", "backtest"],
    ) -> tuple[list[FeatureVector], FeatureQualityReport]: ...
```

**Dependencies:** `DataServiceClient`, `SentinelPulseClient`, `QlibFeatureEngine`, `RedisCache`

---

### 2. QlibFeatureEngine (`src/features/qlib_engine.py`)

**Purpose:** Compute Microsoft Qlib's Alpha158/Alpha360 cross-sectional equity factors.

**Responsibilities:**
- Compute 158 Alpha158 factors for each symbol using OHLCV data from data-service2.0
- Optionally compute 360-dimensional Alpha360 higher-order factors (config-gated)
- Guarantee idempotence: same input OHLCV data produces identical factor output
- Guarantee all factor outputs are finite (no NaN/Inf) for non-missing data
- Guarantee factor values are within documented ranges

**Interface:**
```python
class QlibFeatureEngine:
    def compute_alpha158(self, ohlcv_df: pd.DataFrame, symbol: str) -> dict[str, float]: ...
    def compute_alpha360(self, ohlcv_df: pd.DataFrame, symbol: str) -> dict[str, float]: ...
```

---

### 3. LeakageValidator (`src/features/leakage_validator.py`)

**Purpose:** Detect forward-looking (look-ahead) correlations in the feature matrix before training.

**Responsibilities:**
- Compute Pearson correlation between each feature and realized returns over look-ahead windows 1–22 trading days
- Raise `PITViolationError` if any correlation exceeds 0.05 in absolute value
- Log feature name, look-ahead window, and correlation value on violation

**Interface:**
```python
class LeakageValidator:
    def validate(
        self,
        feature_matrix: pd.DataFrame,
        label_vector: pd.Series,
    ) -> None:  # raises PITViolationError on violation
        ...
```

---

### 4. TrainingPipeline (`src/training/pipeline.py`)

**Purpose:** End-to-end model training with Qlib purged K-fold CV, CPCV, Optuna HPO, and MLflow tracking.

**Responsibilities:**
- Implement double-ensembled purged K-fold CV using `RollingPurgedKFold` with configurable embargo (default 10 days, minimum 5 days)
- Purge all training samples whose label window overlaps any validation sample's feature window
- Compute IC (Spearman), net Sharpe after 10bp transaction costs, and PBO via CPCV (10+ paths) per fold
- Gate model acceptance: IC >= 0.02, Sharpe >= 0.0, PBO <= 0.5
- Run Optuna HPO with 50+ trials (objective = mean Spearman IC)
- Log all experiments to MLflow (hyperparameters, IC per fold, Sharpe, max_dd, PBO, dataset SHA-256)
- Maintain append-only training audit log in JSON Lines format
- Enforce model SDLC: `HYPOTHESIS → BACKTEST → CHALLENGER → SHADOW → APPROVED → PRODUCTION`

**Interface:**
```python
class TrainingPipeline:
    async def run(self, config: TrainingConfig) -> TrainingRun: ...
    async def get_status(self, run_id: str) -> TrainingRunStatus: ...
```

---

### 5. ModelRegistry (`src/registry/registry.py`)

**Purpose:** Champion/challenger artifact management with SHA-256 integrity validation.

**Responsibilities:**
- Store model artifacts with SHA-256 checksums; reject artifacts with mismatched checksums at startup
- Maintain champion and challenger artifacts per model family
- Expose model status and version for `/models/status` and `/models/registry`
- Raise `ARTIFACT_INTEGRITY_FAILURE` and fall back to heuristic if artifact is corrupted
- Never overwrite an existing artifact (immutable artifact store)

**Interface:**
```python
class ModelRegistry:
    def register(self, artifact: ModelArtifact) -> None: ...
    def get_champion(self, model_name: str) -> ModelArtifact | None: ...
    def list_registry(self) -> list[ModelArtifact]: ...
```

---

### 6. RegimeClassifier (`src/models/regime_classifier.py`)

**Purpose:** Classify market into one of six regimes: `strong_bull`, `bull`, `sideways`, `volatile`, `bear`, `crash`.

**Responsibilities:**
- Use trained XGBoost artifact when available; fall back to rule-based heuristic (set `HEURISTIC`)
- Incorporate NIFTY/BANKNIFTY price features, India VIX, market breadth, FII/DII flows, PCR, SentinelPulse market_regime, and Qlib Alpha158 breadth factors (all with max 5-minute data age)
- Respond within 50ms at p95
- Attach top-10 SHAP features to every response
- Emit `regime_transition_event` log on regime change
- Trigger incremental update when IC degrades > 20% from 90-day baseline

---

### 7. StockRanker (`src/models/stock_ranker.py`)

**Purpose:** Produce outperformance score [0,100] and rank for each symbol relative to the current regime.

**Responsibilities:**
- Use trained LightGBM artifact when available; fall back to score-based heuristic
- Incorporate Qlib Alpha360 factors, SentinelPulse `news_impact_score` and `impact_direction`, delivery %, OI build-up, IV rank, sector RS vs NIFTY
- Respond for 200-symbol batch within 200ms at p95
- Return SHAP-based top-5 factor decomposition per symbol
- Support regime-conditioned ranking with regime-specific weight tables
- Idempotence: same feature vector always produces same rank

---

### 8. StrategySelector (`src/models/strategy_selector.py`)

**Purpose:** Recommend one of eight trading strategies with calibrated probabilities.

**Strategies:** `breakout`, `momentum`, `trend_following`, `mean_reversion`, `vwap_bounce`, `range_trading`, `scalping`, `volatility_breakout`

**Responsibilities:**
- Use trained CatBoost artifact when available; fall back to rule-based heuristic
- Incorporate IV regime classification (`CRUSH` | `STABLE` | `SPIKE`) as mandatory feature
- Return calibrated probabilities across all 8 strategies using isotonic regression
- Respond within 50ms at p95
- Include at least 2 alternatives when top-strategy confidence < 0.40

---

### 9. RiskPredictor (`src/models/risk_predictor.py`)

**Purpose:** Estimate per-trade risk before entry with calibrated confidence.

**Outputs:** `prob_stop_hit` ∈ [0,1], `prob_target_hit` ∈ [0,1], `expected_drawdown_pct`, `suggested_position_size_pct`, `risk_score` ∈ [0,10], SHAP factor breakdown

**Responsibilities:**
- Use XGBoost ensemble of 3 models when available; fall back to rule-based
- Derive `stop_distance_atr`, `target_distance_atr`, `risk_reward_ratio` from trade geometry
- Incorporate `regime_encoded`, `vix_regime`, `atr_pct`, `oi_buildup_score`, `news_impact_score`, `sentiment.risk`
- Enforce invariant: `prob_stop_hit + prob_target_hit <= 1.0` (log CALIBRATION_VIOLATION before clamping)
- When `risk_score > 7.0` and `DeploymentMode == VALIDATED_ML_ONLY` → append `HIGH_RISK_BLOCKED`
- Respond within 50ms at p95

---

### 10. PortfolioOptimizer (`src/models/portfolio_optimizer.py`)

**Purpose:** Institutional-grade portfolio optimization using Riskfolio-Lib.

**Methods:** HRP (Hierarchical Risk Parity), CVaR-minimized MVO, Equal Risk Contribution (ERC), Maximum Diversification

**Responsibilities:**
- HRP: Riskfolio-Lib `HRPOpt` with Ward linkage clustering; weights sum to 1.0, all >= 0
- CVaR: Riskfolio-Lib `RiskFolio` with configurable tail probability alpha (default 0.05)
- Enforce max sector weight constraint (default 40%), re-run with binding constraint if violated
- Return: expected return, portfolio volatility, Sharpe ratio, CVaR, max drawdown estimate, diversification ratio
- Return `available: false` with `INSUFFICIENT_RETURN_HISTORY` when fewer than 20 observations per asset
- Respond within 500ms at p95 for up to 50 assets
- Determinism: same inputs + same random seed produce identical weights

---

### 11. RLExecutionAgent (`src/models/rl_execution_agent.py`)

**Purpose:** FinRL deep reinforcement learning agent for execution timing decisions.

**Action space:** `ENTER_NOW`, `WAIT`, `SCALE_IN`, `PARTIAL_EXIT`, `FULL_EXIT`, `TIGHTEN_STOP`, `TRAIL_STOP`

**Responsibilities:**
- Use FinRL-X PPO or SAC trained artifact when available; fall back to rule-based policy (set `HEURISTIC`)
- Observation space: unrealized P&L %, time in trade (min), regime, volume ratio, price vs VWAP, ATR, momentum, IV regime, `news_impact_score`, current risk score
- Reward function: maximize Sharpe; penalize stop hits, holding at session close, turnover > 10 round-trips/session
- Train exclusively on data-service2.0 PIT-correct backtest mode; never on live/shadow data
- Use FinRL `StockTradingEnv` extended with NSE session calendar (375 min/session)
- Respond within 50ms at p95
- Support offline policy evaluation (OPE) with importance sampling on last 63 trading days
- Reject if OPE estimated Sharpe < 0.0

---

### 12. MetaDecisionEngine (`src/meta/engine.py`)

**Purpose:** LLM-augmented arbiter that combines all model outputs into a final Go/No-Go trading decision.

**Output fields:** `action` (BUY|SELL|WAIT|NO_TRADE), `confidence` ∈ [0,1], `uncertainty`, `agreement`, `reason_codes`, `contributing_models`, `ensemble_score`, `decomposition`, `abstention`, `explainability`

**Responsibilities:**
- Calibrate each model's raw score using Platt scaling or isotonic regression (lower ECE wins; reject if ECE > 0.15)
- Apply regime-aware ensemble weights proportional to IC in current regime (min 0.05, max 0.40, sum to 1.0)
- Compute `agreement_ratio` as fraction of non-UNAVAILABLE models matching plurality direction
- Activate `AbstentionPolicy` → NO_TRADE when: `agreement_ratio < 0.5`, `data_quality < 0.6`, `mean_confidence < 0.35`, `prob_stop_hit > 0.65`, or fewer than 3 available models
- LLM override: when `news_impact_score > 0.7` AND conflicts with ensemble → NO_TRADE with `NEWS_CONFLICT_OVERRIDE`
- Integrate FinGPT/FinBERT via LangChain for `news_sentiment_signal` with 120ms timeout
- Cache LLM result 60s; on timeout use cached result (`NEWS_CACHE_FALLBACK`) or neutral (`NEWS_TIMEOUT`)
- `VALIDATED_ML_ONLY` mode: convert all HEURISTIC model contributions to direction=0
- Respond within 150ms at p95 with cached news context
- Absorbing identity: all models UNAVAILABLE → NO_TRADE with `UNAVAILABLE` provenance

---

### 13. DriftMonitor (`src/monitoring/drift_monitor.py`)

**Purpose:** Automated daily drift detection using Evidently AI and NannyML.

**Responsibilities:**
- Run at 00:00 UTC daily: Evidently AI `DataDriftPreset` on every model's feature distribution (7-day rolling vs training reference)
- Compute PSI using 10 equal-frequency bins: `PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)`
- PSI > 0.2 and <= 0.25 → `DRIFT_ALERT` severity MEDIUM
- PSI > 0.25 → `DRIFT_ALERT` severity HIGH + trigger `OnlineLearner` if enabled
- NannyML CBPE for estimated IC + ECE on unlabeled production data
- estimated IC < 0.015 → `PERFORMANCE_DEGRADATION_ALERT` → block model from MetaDecisionEngine
- Expose `GET /monitoring/drift` and `GET /monitoring/performance`
- Include recommended action in alert payload: PSI <= 0.25 → MONITOR; PSI > 0.25, learning not triggered → RETRAIN; PSI > 0.25, learning triggered, IC still < 0.015 → ROLLBACK
- `POST /monitoring/performance/{model_name}/clear` to manually clear blocks

---

### 14. OnlineLearner (`src/training/online_learner.py`)

**Purpose:** Incremental model updates without full retraining cycles.

**Responsibilities:**
- Triggered by PerformanceMonitor when IC degrades > 20% from 90-day baseline over rolling 30 days
- Support XGBoost `model.update()` for regime classifier and risk predictor; LightGBM incremental fit for stock ranker
- Only apply to `TRAINED_MODEL` provenance; heuristics are not eligible
- Create new versioned artifact (e.g. `v1.0.3-online-2025-07-15`); never overwrite prior artifacts
- Validate on last 10 trading days before activation; discard and log `ONLINE_UPDATE_REJECTED` if new IC < prior IC
- Maximum 5 consecutive incremental updates before requiring full retrain
- Continue serving from existing champion during update window

---

### 15. ModelExplainer (`src/explainability/explainer.py`)

**Purpose:** SHAP-based feature attribution for every model prediction.

**Responsibilities:**
- Use `shap.TreeExplainer` for tree-based models (XGBoost, LightGBM, CatBoost)
- Use `shap.KernelExplainer` for neural network models (PriceForecaster, IVRegimeClassifier)
- Cache SHAP explainer instances in-process (LRU, per model)
- Return top-10 features by absolute SHAP value with direction and raw value
- For HEURISTIC predictions: return rule attribution (which rule conditions fired and their contributions)
- On SHAP failure: log error, return all contributions = 0.0 and `base_value` = historical mean (no 500 errors)
- Expose `POST /v2/explain/{model_name}`

---

### 16. SignalStreamer (`src/streaming/streamer.py`)

**Purpose:** WebSocket broadcast of real-time MetaOutput signals to alpha-forge.

**Responsibilities:**
- Manage WebSocket connections at `WS /v2/stream/signals`
- Emit a `SignalEvent` containing `MetaOutput` for each symbol on every new MetaDecisionEngine output
- Handle client disconnection gracefully (close stream without errors)
- Support multiple concurrent WebSocket clients

---

### 17. AuditLogger (`src/audit/logger.py`)

**Purpose:** Append-only, immutable JSON Lines audit log.

**Responsibilities:**
- Record every training run, promotion decision, online learning update, and approvalToken issuance
- Never allow modification or deletion of prior entries; raise `AuditLogViolation` on any such attempt
- Each entry includes: `run_id`, `timestamp`, `event_type`, `model_name`, `model_version`, `gate_results`, `outcome`, `reviewer_identity` (when applicable), `dataset_hash`
- structlog-formatted for compatibility with log aggregation pipelines

---

### 18. ModelPromotion (`src/registry/promotion.py`)

**Purpose:** Enforce the six-gate champion/challenger promotion pipeline.

**Gates:**
1. **DATA** — OOS observation count >= 60 (one per ticker-date pair), `look_ahead_validated`, evidence level >= LEVEL_B, `final_oos_used_for_selection` must be false
2. **PREDICTIVE** — Challenger Rank IC > Champion IC + 0.005 (with champion); > 0 (without)
3. **CALIBRATION** — Challenger Brier score <= Champion Brier + 0.01 (with champion); < 0.25 (without)
4. **EXECUTION** — Challenger net return > 0 on held-out partition; if champion exists, turnover increase <= 20%
5. **RISK** — Challenger max drawdown <= Champion max drawdown + 2pp (with champion); <= 20% (without)
6. **STABILITY** — IC decay not FAILED/SIGNIFICANT_DECAY; feature drift not HIGH/CRITICAL

**Responsibilities:**
- Enforce linear lifecycle: `HYPOTHESIS → BACKTEST → CHALLENGER → SHADOW → APPROVED → PRODUCTION`
- Require `approvalToken` (issued by authorized reviewer, valid 24h) when outcome is PROMOTE
- Record every decision in the append-only audit log
- Mark challengers `FINAL_OOS_CONTAMINATED` permanently if final OOS data was read during gate evaluation
- INSUFFICIENT_EVIDENCE on any gate → BLOCKED status, record in audit log


---

## Data Models

All schemas use Pydantic V2 with `model_config = ConfigDict(strict=True)` and `from __future__ import annotations`.

### Core Enums

```python
from __future__ import annotations
from enum import Enum

class PredictionProvenance(str, Enum):
    TRAINED_MODEL         = "trained_model"
    HEURISTIC             = "heuristic"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNAVAILABLE           = "unavailable"

    @property
    def is_live_eligible(self) -> bool:
        return self == PredictionProvenance.TRAINED_MODEL

class DeploymentMode(str, Enum):
    RESEARCH          = "research"
    PAPER             = "paper"
    SHADOW            = "shadow"
    VALIDATED_ML_ONLY = "validated_ml_only"

class MarketRegime(str, Enum):
    STRONG_BULL = "strong_bull"
    BULL        = "bull"
    SIDEWAYS    = "sideways"
    VOLATILE    = "volatile"
    BEAR        = "bear"
    CRASH       = "crash"

class TradingStrategy(str, Enum):
    BREAKOUT            = "breakout"
    MOMENTUM            = "momentum"
    TREND_FOLLOWING     = "trend_following"
    MEAN_REVERSION      = "mean_reversion"
    VWAP_BOUNCE         = "vwap_bounce"
    RANGE_TRADING       = "range_trading"
    SCALPING            = "scalping"
    VOLATILITY_BREAKOUT = "volatility_breakout"

class ExecutionAction(str, Enum):
    ENTER_NOW    = "enter_now"
    WAIT         = "wait"
    SCALE_IN     = "scale_in"
    PARTIAL_EXIT = "partial_exit"
    FULL_EXIT    = "full_exit"
    TIGHTEN_STOP = "tighten_stop"
    TRAIL_STOP   = "trail_stop"

class IVRegime(str, Enum):
    CRUSH  = "CRUSH"
    STABLE = "STABLE"
    SPIKE  = "SPIKE"

class ModelLifecycleStage(str, Enum):
    HYPOTHESIS  = "hypothesis"
    BACKTEST    = "backtest"
    CHALLENGER  = "challenger"
    SHADOW      = "shadow"
    APPROVED    = "approved"
    PRODUCTION  = "production"

class PromotionOutcome(str, Enum):
    PROMOTE              = "promote"
    REJECTED             = "rejected"
    BLOCKED              = "blocked"
    FINAL_OOS_CONTAMINATED = "final_oos_contaminated"

class GateResult(str, Enum):
    PASS                = "pass"
    FAIL                = "fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

class DriftSeverity(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"

class ImpactDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
```

### FeatureVector

```python
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime

class FeatureVector(BaseModel):
    model_config = ConfigDict(strict=True)

    symbol: str
    timestamp: datetime                    # UTC ISO-8601 — PIT reference point
    pit_validated: bool = False            # True once LeakageValidator passes

    # Market features (from data-service2.0)
    nifty_change_pct: float | None = None
    banknifty_change_pct: float | None = None
    india_vix: float | None = None
    nifty_atr_pct: float | None = None
    nifty_adx: float | None = None
    advance_decline_ratio: float | None = None
    market_breadth: float | None = None
    sector_strength: float | None = None
    volume_ratio: float | None = None
    gap_pct: float | None = None
    fii_net_cr: float | None = None
    put_call_ratio: float | None = None
    delivery_pct: float | None = None
    oi_buildup_score: float | None = None
    iv_rank: float | None = None

    # Qlib Alpha158 factors (158 dimensions, keys alpha158_0 .. alpha158_157)
    alpha158_factors: dict[str, float] = Field(default_factory=dict)

    # Qlib Alpha360 factors (360 dimensions, optional, config-gated)
    alpha360_factors: dict[str, float] = Field(default_factory=dict)

    # SentinelPulse news features
    news_impact_score: float = 0.0
    impact_direction: ImpactDirection = ImpactDirection.NEUTRAL
    impact_confidence: float = 0.0
    sentiment_overall: float = 0.0
    sentiment_market: float = 0.0
    sentiment_company: float = 0.0
    sentiment_macro: float = 0.0
    sentiment_risk: float = 0.0
    market_regime_nlp: str = "NEUTRAL"      # SentinelPulse market_regime field

    # Feature sourcing metadata
    news_available: bool = True
    missing_feature_families: list[str] = Field(default_factory=list)
    data_confidence_score: int = 100        # DataConfidenceScore from data-service2.0
    signal_engine_allowed: bool = True      # signalEngineAllowed flag
```

### FeatureQualityReport

```python
class FeatureQualityReport(BaseModel):
    model_config = ConfigDict(strict=True)

    batch_id: str
    timestamp: datetime
    total_features_requested: int
    missing_count: int
    imputed_count: int
    rejected_count: int
    unavailable_families: list[str]
    pit_violations_count: int
    discarded_backtest_records: int = 0     # populated in backtest mode
    processing_time_ms: float
```

### MetaOutput

```python
class NewsSignal(BaseModel):
    model_config = ConfigDict(strict=True)
    direction: int = Field(ge=-1, le=1)    # +1 / 0 / -1
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=200)  # <= 30 words from LLM

class ConfidenceDecomposition(BaseModel):
    model_config = ConfigDict(strict=True)
    base_confidence: float
    calibration_quality: float
    agreement_bonus: float
    data_quality_factor: float
    regime_confidence_factor: float

class MetaOutputExplainability(BaseModel):
    model_config = ConfigDict(strict=True)
    top_features: list[str]               # top-5 contributing features across all models
    conflicting_models: list[str]
    rationale: str                        # one-sentence natural-language rationale from LLM

class MetaOutput(BaseModel):
    model_config = ConfigDict(strict=True)

    action: Literal["BUY", "SELL", "WAIT", "NO_TRADE"]
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    agreement: float = Field(ge=0.0, le=1.0)
    agreement_ratio: float = Field(ge=0.0, le=1.0)
    ensemble_score: float
    reason_codes: list[str]
    contributing_models: list[str]
    abstention: bool
    provenance: PredictionProvenance
    news_sentiment_signal: NewsSignal
    decomposition: ConfidenceDecomposition
    explainability: MetaOutputExplainability
    request_id: str
    symbol: str
    timestamp: datetime
    latency_ms: float
```

### SignalEvent

```python
class SignalEvent(BaseModel):
    model_config = ConfigDict(strict=True)

    event_type: Literal["signal"] = "signal"
    symbol: str
    timestamp: datetime
    meta_output: MetaOutput
    feature_quality: FeatureQualityReport
```

### ModelArtifact

```python
class ModelArtifact(BaseModel):
    model_config = ConfigDict(strict=True)

    model_name: str
    version: str                           # e.g. "v1.2.0" or "v1.2.0-online-2025-07-15"
    lifecycle_stage: ModelLifecycleStage
    algorithm: str                         # "xgboost", "lightgbm", "catboost", "ppo", etc.
    training_date_range: tuple[date, date]
    artifact_path: str
    sha256_checksum: str
    mlflow_run_id: str
    ic_mean: float
    net_sharpe: float
    pbo: float
    brier_score: float | None = None
    max_drawdown: float
    is_champion: bool = False
    training_dataset_hash: str
    created_at: datetime
    consecutive_online_updates: int = 0
```

### PromotionDecision

```python
class GateEvaluation(BaseModel):
    model_config = ConfigDict(strict=True)
    gate_name: Literal["DATA", "PREDICTIVE", "CALIBRATION", "EXECUTION", "RISK", "STABILITY"]
    result: GateResult
    reason: str
    metrics: dict[str, float]

class PromotionDecision(BaseModel):
    model_config = ConfigDict(strict=True)

    challenger_id: str
    champion_id: str | None
    outcome: PromotionOutcome
    gate_evaluations: list[GateEvaluation]
    approval_policy: Literal["HUMAN_APPROVAL_REQUIRED", "AUTOMATIC"] | None
    approval_token: str | None = None
    reviewer_identity: str | None = None
    timestamp: datetime
    audit_log_entry_id: str
```

### DriftAlert

```python
class DriftAlert(BaseModel):
    model_config = ConfigDict(strict=True)

    alert_id: str
    alert_type: Literal["DRIFT_ALERT", "PERFORMANCE_DEGRADATION_ALERT"]
    severity: DriftSeverity
    model_name: str
    feature_name: str | None = None
    psi_value: float | None = None
    reference_mean: float | None = None
    reference_std: float | None = None
    current_mean: float | None = None
    current_std: float | None = None
    estimated_ic: float | None = None
    estimated_ece: float | None = None
    recommended_action: Literal["MONITOR", "RETRAIN", "ROLLBACK"] | None = None
    timestamp: datetime
    resolved: bool = False
```

### TrainingRun

```python
class TrainingRun(BaseModel):
    model_config = ConfigDict(strict=True)

    run_id: str
    model_name: str
    start_time: datetime
    end_time: datetime | None = None
    status: Literal["running", "completed", "failed", "aborted"]
    dataset_hash: str
    model_version: str | None = None
    ic_per_fold: list[float] = Field(default_factory=list)
    net_sharpe: float | None = None
    pbo: float | None = None
    max_drawdown: float | None = None
    gate_results: list[GateEvaluation] = Field(default_factory=list)
    approval_token: str | None = None
    rejection_reason: str | None = None
    mlflow_experiment_id: str | None = None
    hyperparameters: dict[str, float | int | str] = Field(default_factory=dict)
```

### Existing Schemas (preserved from alpha-forge ml-service, upgraded to Pydantic V2)

All existing request/response schemas are preserved and upgraded:

- `RegimePredictionRequest` / `RegimePredictionResponse` — market regime classification I/O
- `StockFeatures` / `RankingRequest` / `StockRank` / `RankingResponse` — stock ranking I/O
- `StrategyRequest` / `StrategyResponse` — strategy selection I/O
- `RiskRequest` / `RiskResponse` — per-trade risk estimation I/O
- `PortfolioAsset` / `PortfolioRequest` / `PortfolioAllocation` / `PortfolioResponse` — portfolio optimization I/O
- `ExecutionState` / `ExecutionDecision` — RL execution agent I/O
- `ExplainRequest` / `FeatureContribution` / `ExplainResponse` — SHAP explainability I/O

All schemas gain `model_config = ConfigDict(strict=True)` and `from __future__ import annotations`. `RegimePredictionResponse`, `RankingResponse`, and `RiskResponse` gain `provenance: PredictionProvenance` fields where missing.


---

## API Design

All v2 endpoints require `X-API-KEY` header authentication. All prediction responses include a `provenance: PredictionProvenance` field. All requests receive an `X-Request-ID` header for trace correlation.

### Prediction Endpoints

| Method | Path | Request | Response | p95 SLA |
|--------|------|---------|----------|---------|
| POST | `/v2/predict/regime` | `RegimePredictionRequest` | `RegimePredictionResponse` | 50ms |
| POST | `/v2/predict/rankings` | `RankingRequest` (up to 200 symbols) | `RankingResponse` | 200ms |
| POST | `/v2/predict/strategy` | `StrategyRequest` | `StrategyResponse` | 50ms |
| POST | `/v2/predict/risk` | `RiskRequest` | `RiskResponse` | 50ms |
| POST | `/v2/predict/portfolio` | `PortfolioRequest` | `PortfolioResponse` | 500ms |
| POST | `/v2/predict/portfolio-v2` | `PortfolioV2Request` | `PortfolioV2Response` | 500ms |
| POST | `/v2/predict/execution` | `ExecutionState` | `ExecutionDecision` | 50ms |
| POST | `/v2/predict/price-regime` | `PriceRegimeRequest` | `PriceRegimeResponse` | 50ms |
| POST | `/v2/predict/iv-regime` | `IVRegimeRequest` | `IVRegimeResponse` | 50ms |

### Meta Endpoints

| Method | Path | Request | Response | p95 SLA |
|--------|------|---------|----------|---------|
| POST | `/v2/meta/decide` | `MetaDecideRequest` | `MetaOutput` | 150ms |
| POST | `/v2/meta/fit` | `list[OOSRecord]` (min 30) | `MetaFitResponse` | — |

**MetaDecideRequest:**
```python
class MetaDecideRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    symbol: str
    regime: MarketRegime
    feature_vector: FeatureVector
    force_refresh_news: bool = False
```

**OOSRecord (for /v2/meta/fit):**
```python
class OOSRecord(BaseModel):
    model_config = ConfigDict(strict=True)
    model_id: str
    raw_score: float
    realized_outcome: float    # 0.0 or 1.0
    timestamp: datetime
```

### Analytics Endpoints

| Method | Path | Request | Response | p95 SLA |
|--------|------|---------|----------|---------|
| POST | `/v2/analytics/greeks` | `GreeksRequest` | `list[dict]` with Greeks | 100ms |
| GET/POST | `/v2/analytics/gex` | `GexRequest` | `GexResponse` | 100ms |
| GET/POST | `/v2/analytics/vpin` | `VpinRequest` | `VpinResponse` | 100ms |
| GET/POST | `/v2/analytics/vol-surface` | `VolSurfaceRequest` | `VolSurfaceResponse` | 100ms |

### Operational Endpoints

| Method | Path | Auth | Response |
|--------|------|------|----------|
| GET | `/health` | None | `{"status": "healthy", "timestamp": int, "version": str}` |
| GET | `/models/status` | API Key | Per-model loaded/version/has_trained_model |
| GET | `/models/registry` | API Key | `list[ModelArtifact]` |
| GET | `/features/quality` | API Key | `FeatureQualityReport` (latest batch), ≤2s p95 |
| GET | `/monitoring/drift` | API Key | Latest drift report for all models |
| GET | `/monitoring/performance` | API Key | Latest NannyML CBPE estimates + ECE |
| GET | `/monitoring/alerts` | API Key | `list[DriftAlert]` (active alerts) |
| POST | `/monitoring/performance/{model_name}/clear` | API Key | `{"cleared": true, "model": str}` |

### Training Endpoints

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | `/training/run` | `TrainingConfig` | `TrainingRun` (initial status) |
| GET | `/training/status/{run_id}` | — | `TrainingRunStatus` |

### WebSocket

| Protocol | Path | Description |
|----------|------|-------------|
| WS | `/v2/stream/signals` | Authenticate via `?api_key=...` query param. Server pushes `SignalEvent` JSON for each new MetaDecisionEngine output. Client disconnect closes stream gracefully. |

### Error Responses

| HTTP Status | Condition |
|-------------|-----------|
| 401 | Missing or invalid `X-API-KEY` |
| 422 | Pydantic V2 `ValidationError` — lists all field errors |
| 503 | Model unavailable (no artifact loaded, runtime exception) — enables retry logic |
| 500 | Unexpected internal error (should not occur in normal operation) |


---

## Storage Architecture

### Redis

All Redis access uses `redis.asyncio` with async/await throughout.

| Key Pattern | TTL | Content | Source |
|-------------|-----|---------|--------|
| `feature:{symbol}:{bucket}` | 60s (`FEATURE_CACHE_TTL`) | Serialized `FeatureVector` JSON | `FeaturePipeline` |
| `news:{symbol}` | 90s (`NEWS_CONTEXT_CACHE_TTL`) | SentinelPulse news context JSON | `SentinelPulseClient` |
| `signal:{symbol}:latest` | 300s | Serialized `MetaOutput` JSON | `MetaDecisionEngine` |
| `llm_news:{symbol}` | 60s | Serialized `NewsSignal` JSON | `LLMNewsReasoner` |
| `model:status` | — | Current model load status dict | `ModelRegistry` |
| `drift:latest` | 86400s | Latest drift report JSON | `DriftMonitor` |

### MLflow

- **Tracking URI**: configured via `MLFLOW_TRACKING_URI` (default: `http://localhost:5000`)
- **Experiment per model family**: `alphaforge-{model_name}-hpo`
- **Logged per run**: model name, version, training date range, validation date range, all hyperparameters, IC per fold, net Sharpe, max drawdown, PBO, SHA-256 hash of training dataset
- **Model registry**: MLflow Model Registry for champion/challenger versioning
- **Artifact storage**: model files, SHAP baseline values, calibrator objects

### File System

```
{MODEL_ARTIFACTS_PATH}/
├── market_regime/
│   ├── v1.0.0/
│   │   ├── model.json          # XGBoost booster
│   │   ├── metadata.json       # ModelArtifact (sans path)
│   │   └── model.json.sha256   # SHA-256 checksum file
│   └── v1.0.1-online-2025-07-15/
│       ├── model.json
│       ├── metadata.json
│       └── model.json.sha256
├── stock_ranker/
│   └── v1.0.0/
│       ├── model.txt           # LightGBM booster
│       ├── metadata.json
│       └── model.txt.sha256
├── strategy_selector/
├── risk_predictor/
│   └── v1.0.0/
│       ├── stop_model.json
│       ├── target_model.json
│       ├── drawdown_model.json
│       ├── metadata.json
│       └── *.sha256
├── rl_execution_agent/
│   └── v1.0.0/
│       ├── policy.zip          # FinRL/Stable-Baselines3 checkpoint
│       ├── metadata.json
│       └── policy.zip.sha256
└── portfolio_optimizer/        # algorithm-only, no artifact

{AUDIT_LOG_PATH}/
└── audit.jsonl                 # append-only JSON Lines
```

### In-Process State

| Object | Lifecycle | Description |
|--------|-----------|-------------|
| SHAP explainer instances | Lazy-loaded, LRU per model | `shap.TreeExplainer` / `shap.KernelExplainer` |
| Model instances | Loaded at startup via lifespan | All classifier/ranker/predictor objects |
| `news_context` LRU cache | 500 entries max, 90s TTL | Per-symbol SentinelPulse response |
| `PerformanceMonitor` ring buffers | Rolling 100 samples | IC tracking per model |
| `FeatureMonitor` sliding windows | Rolling 200 samples | Feature distribution tracking |

---

## Module Structure

```
ml-service2.0/
├── src/
│   ├── api/                        # FastAPI routers (one file per concern)
│   │   ├── __init__.py
│   │   ├── predictions.py          # /v2/predict/* endpoints
│   │   ├── meta.py                 # /v2/meta/* endpoints
│   │   ├── analytics.py            # /v2/analytics/* endpoints
│   │   ├── training.py             # /training/* endpoints
│   │   ├── monitoring.py           # /monitoring/* endpoints
│   │   ├── models.py               # /models/* endpoints
│   │   ├── features.py             # /features/* endpoints
│   │   └── health.py               # /health endpoint
│   ├── features/
│   │   ├── __init__.py
│   │   ├── pipeline.py             # FeaturePipeline — PIT-correct assembly
│   │   ├── qlib_engine.py          # QlibFeatureEngine — Alpha158/360 factors
│   │   ├── leakage_validator.py    # LeakageValidator — forward-look detection
│   │   └── volume.py               # VPIN, volume-derived features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── regime_classifier.py    # XGBoost regime model
│   │   ├── stock_ranker.py         # LightGBM ranking model
│   │   ├── strategy_selector.py    # CatBoost strategy model
│   │   ├── risk_predictor.py       # XGBoost ensemble risk model
│   │   ├── portfolio_optimizer.py  # Riskfolio-Lib HRP/CVaR
│   │   ├── rl_execution_agent.py   # FinRL PPO/SAC execution agent
│   │   ├── price_forecaster.py     # TFT price regime forecaster
│   │   └── iv_regime_classifier.py # PatchTST IV regime classifier
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pipeline.py             # TrainingPipeline — end-to-end training
│   │   ├── purged_kfold.py         # PurgedKFoldSplitter with embargo
│   │   ├── cpcv.py                 # CPCV path generation + PBO
│   │   ├── hpo.py                  # Optuna HPO wrapper
│   │   └── online_learner.py       # OnlineLearner — incremental updates
│   ├── meta/
│   │   ├── __init__.py
│   │   ├── engine.py               # MetaDecisionEngine
│   │   ├── calibration.py          # Platt / isotonic calibration
│   │   ├── ensemble.py             # Regime-aware ensemble weighter
│   │   ├── abstention.py           # AbstentionPolicy
│   │   └── llm_reasoner.py         # FinGPT/FinBERT via LangChain
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift_monitor.py        # Evidently AI + NannyML drift detection
│   │   ├── performance_monitor.py  # IC / ECE tracking
│   │   ├── feature_monitor.py      # Feature distribution monitoring
│   │   ├── alerts.py               # AlertSystem — structured alert emission
│   │   └── router.py               # /monitoring/* FastAPI router
│   ├── registry/
│   │   ├── __init__.py
│   │   ├── registry.py             # ModelRegistry — champion/challenger mgmt
│   │   └── promotion.py            # ModelPromotion — six-gate pipeline
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── explainer.py            # ModelExplainer — SHAP TreeExplainer/KernelExplainer
│   ├── streaming/
│   │   ├── __init__.py
│   │   └── streamer.py             # SignalStreamer — WebSocket manager
│   ├── audit/
│   │   ├── __init__.py
│   │   └── logger.py               # AuditLogger — append-only JSON Lines
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── data_service.py         # DataServiceClient — REST + gRPC
│   │   └── sentinel_pulse.py       # SentinelPulseClient — REST + LRU cache
│   ├── cache/
│   │   ├── __init__.py
│   │   └── redis_cache.py          # Async Redis cache layer
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── base.py                 # PredictionProvenance, DeploymentMode enums
│   │   ├── features.py             # FeatureVector, FeatureQualityReport
│   │   ├── predictions.py          # All prediction request/response schemas
│   │   ├── meta.py                 # MetaOutput, MetaDecideRequest, etc.
│   │   ├── registry.py             # ModelArtifact, PromotionDecision, TrainingRun
│   │   ├── monitoring.py           # DriftAlert
│   │   └── streaming.py            # SignalEvent
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── greeks.py               # Black-Scholes/Black-76 Greeks
│   │   ├── gex.py                  # Dealer Gamma Exposure
│   │   ├── vpin.py                 # VPIN computation
│   │   └── vol_surface.py          # SVI vol surface + term structure
│   ├── config.py                   # pydantic-settings Settings
│   └── main.py                     # FastAPI app creation + lifespan
├── tests/
│   ├── conftest.py                 # Shared fixtures (mock clients, etc.)
│   ├── test_feature_pipeline_pit_correctness.py
│   ├── test_meta_decision_engine.py
│   ├── test_qlib_feature_engineering.py
│   ├── test_finrl_execution_agent.py
│   ├── test_drift_monitoring.py
│   ├── test_signal_generation_latency.py
│   ├── test_model_training_purged_cv.py
│   ├── test_api_contracts.py
│   ├── test_portfolio_optimization.py
│   └── test_online_learning.py
├── protos/
│   └── market_data.proto           # gRPC protobuf for data-service2.0 streaming
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```


---

## Key Algorithms and Pseudocode

### LeakageValidator.validate

```python
def validate(self, feature_matrix: pd.DataFrame, label_vector: pd.Series) -> None:
    """
    Scan for look-ahead correlations across 1-22 trading day windows.
    Raises PITViolationError if any |correlation| > 0.05.
    """
    for window in range(1, 23):                           # 1 to 22 trading days
        shifted_labels = label_vector.shift(-window)      # future returns
        aligned = feature_matrix.align(shifted_labels, join="inner", axis=0)
        features_aligned, labels_aligned = aligned

        for feature_name in features_aligned.columns:
            feature_col = features_aligned[feature_name].dropna()
            label_col = labels_aligned.loc[feature_col.index].dropna()
            common_idx = feature_col.index.intersection(label_col.index)

            if len(common_idx) < 30:
                continue                                  # insufficient samples

            corr, _ = pearsonr(
                feature_col.loc[common_idx],
                label_col.loc[common_idx],
            )
            if abs(corr) > 0.05:
                raise PITViolationError(
                    feature=feature_name,
                    look_ahead_window=window,
                    correlation=corr,
                )
```

### PurgedKFoldSplitter.split

```python
def split(
    self,
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    n_splits: int = 5,
    embargo_days: int = 10,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Generate purged K-fold splits with embargo period.
    Guarantees zero temporal overlap between any training sample
    and its corresponding validation fold (including embargo buffer).
    """
    total = len(X)
    fold_size = total // n_splits

    for k in range(n_splits):
        val_start_idx = k * fold_size
        val_end_idx   = val_start_idx + fold_size if k < n_splits - 1 else total

        val_indices = np.arange(val_start_idx, val_end_idx)
        val_times   = timestamps.iloc[val_indices]

        val_start_time = val_times.min()
        val_end_time   = val_times.max()

        # Embargo window: exclude training samples within embargo_days of val boundary
        embargo_start = val_start_time - pd.Timedelta(days=embargo_days)
        embargo_end   = val_end_time   + pd.Timedelta(days=embargo_days)

        train_mask = (
            (timestamps < embargo_start) | (timestamps > embargo_end)
        )
        train_indices = np.where(train_mask.values)[0]

        # Purge: remove any training sample whose label window overlaps val feature window
        # Label observation window assumed to be `label_window_days` after feature timestamp
        label_window_days = getattr(self, "label_window_days", 5)
        purge_mask = np.array([
            not (
                timestamps.iloc[i] >= val_start_time - pd.Timedelta(days=label_window_days)
                and timestamps.iloc[i] <= val_end_time
            )
            for i in train_indices
        ])
        train_indices = train_indices[purge_mask]

        yield train_indices, val_indices
```

### MetaDecisionEngine.decide

```python
async def decide(
    self,
    model_outputs: dict[str, ModelOutput],
    news_context: NewsContext,
    regime: MarketRegime,
) -> MetaOutput:
    """
    Combine all base model outputs with LLM news reasoning into a MetaOutput.
    """
    # 1. Filter unavailable models
    available = {
        name: out for name, out in model_outputs.items()
        if out.provenance != PredictionProvenance.UNAVAILABLE
    }

    # 2. Absorbing identity: all UNAVAILABLE → NO_TRADE
    if len(available) < 3:
        return MetaOutput(
            action="NO_TRADE",
            provenance=PredictionProvenance.UNAVAILABLE,
            abstention=True,
            reason_codes=["INSUFFICIENT_AVAILABLE_MODELS"],
            ...
        )

    # 3. In VALIDATED_ML_ONLY mode: suppress HEURISTIC contributions
    if self.deployment_mode == DeploymentMode.VALIDATED_ML_ONLY:
        for name, out in available.items():
            if out.provenance == PredictionProvenance.HEURISTIC:
                available[name] = out.with_direction(0)

    # 4. Calibrate each model's raw score (Platt or isotonic, lower ECE wins)
    calibrated = {
        name: self.calibrators[name].calibrate(out.raw_score)
        for name, out in available.items()
    }

    # 5. Regime-aware ensemble weighting (IC-proportional, min=0.05, max=0.40, sum=1.0)
    weights = self.ensemble_weighter.compute_weights(available.keys(), regime)

    # 6. Compute ensemble score and agreement_ratio
    ensemble_score = sum(weights[n] * calibrated[n] for n in available)
    directions = [out.direction for out in available.values()]
    plurality_dir = max(set(directions), key=directions.count)
    agreement_ratio = directions.count(plurality_dir) / len(directions)

    # 7. LLM news reasoning (120ms timeout)
    news_signal = await self.llm_reasoner.reason(news_context, symbol=symbol)

    # 8. AbstentionPolicy
    data_quality = news_context.data_confidence_score / 100
    mean_confidence = sum(out.confidence for out in available.values()) / len(available)
    prob_stop = model_outputs.get("risk", {}).get("prob_stop_hit", 0)

    should_abstain = (
        agreement_ratio < 0.5
        or data_quality < 0.6
        or mean_confidence < 0.35
        or prob_stop > 0.65
    )

    # 9. News override
    if (
        news_context.news_impact_score > 0.7
        and news_signal.direction != 0
        and news_signal.direction != (1 if ensemble_score > 0.5 else -1)
    ):
        return MetaOutput(action="NO_TRADE", reason_codes=["NEWS_CONFLICT_OVERRIDE"], ...)

    if should_abstain:
        return MetaOutput(action="NO_TRADE", abstention=True, ...)

    # 10. Determine action from ensemble score
    action = "BUY" if ensemble_score > 0.65 else "SELL" if ensemble_score < 0.35 else "WAIT"

    # 11. Decompose confidence
    decomposition = ConfidenceDecomposition(
        base_confidence=ensemble_score,
        calibration_quality=1.0 - mean([self.calibrators[n].ece for n in available]),
        agreement_bonus=agreement_ratio * 0.1,
        data_quality_factor=data_quality,
        regime_confidence_factor=weights.get(regime.value, 1.0),
    )

    return MetaOutput(action=action, confidence=..., decomposition=decomposition, ...)
```

### DriftMonitor.compute_psi

```python
def compute_psi(
    self,
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)
    Uses 10 equal-frequency bins derived from the reference distribution.
    Returns 0.0 for identical distributions; > 0 for any shift.
    """
    # Build equal-frequency bin edges from the reference distribution
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, percentiles)
    bin_edges[0]  = -np.inf   # ensure all values are captured
    bin_edges[-1] =  np.inf

    # Compute expected (reference) and actual (current) bin percentages
    expected_counts = np.histogram(reference, bins=bin_edges)[0]
    actual_counts   = np.histogram(current,   bins=bin_edges)[0]

    expected_pct = (expected_counts + 1e-6) / len(reference)  # add epsilon to avoid log(0)
    actual_pct   = (actual_counts   + 1e-6) / len(current)

    # Normalize to sum to 1
    expected_pct /= expected_pct.sum()
    actual_pct   /= actual_pct.sum()

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return max(psi, 0.0)  # PSI is non-negative
```

### OnlineLearner.update

```python
def update(self, model_name: str, new_data: pd.DataFrame) -> str | None:
    """
    Apply one incremental update to a supported model.
    Returns the new artifact version string, or None if update was rejected.
    """
    artifact = self.registry.get_champion(model_name)
    if artifact is None or artifact.provenance != PredictionProvenance.TRAINED_MODEL:
        logger.warning("online_update_skipped_not_trained_model", model=model_name)
        return None

    if artifact.consecutive_online_updates >= self.max_consecutive_updates:
        self._trigger_full_retrain(model_name)
        return None

    # Incremental fit
    model = self._load_model(artifact)
    if model_name in ("market_regime", "risk_predictor"):
        model.update(new_data[self.feature_names[model_name]], new_data["label"])
    elif model_name == "stock_ranker":
        model = lgb.train(
            params=artifact.hyperparameters,
            train_set=lgb.Dataset(new_data[self.feature_names[model_name]], label=new_data["label"]),
            init_model=model,
            num_boost_round=10,
        )

    # Validate on most recent 10 trading days
    val_data = self._get_validation_window(model_name, trading_days=10)
    new_ic = self._compute_ic(model, val_data)
    old_ic = self._compute_ic(self._load_model(artifact), val_data)

    if new_ic < old_ic:
        logger.warning("online_update_rejected", model=model_name, new_ic=new_ic, old_ic=old_ic)
        self.audit.log_online_update_rejected(model_name, artifact.version, new_ic, old_ic)
        return None

    # Persist new versioned artifact
    new_version = f"{artifact.version}-online-{date.today().isoformat()}"
    new_path = self._save_artifact(model, model_name, new_version)
    checksum = sha256_file(new_path)

    new_artifact = artifact.model_copy(update={
        "version": new_version,
        "artifact_path": str(new_path),
        "sha256_checksum": checksum,
        "consecutive_online_updates": artifact.consecutive_online_updates + 1,
    })
    self.registry.register(new_artifact)
    self.audit.log_online_update(model_name, artifact.version, new_version, new_ic - old_ic)
    return new_version
```

### ModelPromotion.evaluate_gates

```python
def evaluate_gates(
    self,
    challenger: ModelArtifact,
    champion: ModelArtifact | None,
) -> PromotionDecision:
    """
    Evaluate all six promotion gates.
    All gates must PASS for outcome=PROMOTE.
    Any FAIL → outcome=REJECTED.
    Any INSUFFICIENT_EVIDENCE → outcome=BLOCKED.
    """
    gate_evals: list[GateEvaluation] = []

    # Gate 1: DATA
    data_result = self._evaluate_data_gate(challenger)
    gate_evals.append(data_result)

    # Gate 2: PREDICTIVE
    predictive_result = self._evaluate_predictive_gate(challenger, champion)
    gate_evals.append(predictive_result)

    # Gate 3: CALIBRATION
    calibration_result = self._evaluate_calibration_gate(challenger, champion)
    gate_evals.append(calibration_result)

    # Gate 4: EXECUTION
    execution_result = self._evaluate_execution_gate(challenger, champion)
    gate_evals.append(execution_result)

    # Gate 5: RISK
    risk_result = self._evaluate_risk_gate(challenger, champion)
    gate_evals.append(risk_result)

    # Gate 6: STABILITY
    stability_result = self._evaluate_stability_gate(challenger)
    gate_evals.append(stability_result)

    # Determine outcome
    results = [g.result for g in gate_evals]
    if GateResult.FAIL in results:
        outcome = PromotionOutcome.REJECTED
        policy  = None
    elif GateResult.INSUFFICIENT_EVIDENCE in results:
        outcome = PromotionOutcome.BLOCKED
        policy  = None
    else:
        outcome = PromotionOutcome.PROMOTE
        policy  = "HUMAN_APPROVAL_REQUIRED"

    decision = PromotionDecision(
        challenger_id=challenger.version,
        champion_id=champion.version if champion else None,
        outcome=outcome,
        gate_evaluations=gate_evals,
        approval_policy=policy,
        timestamp=datetime.utcnow(),
        audit_log_entry_id=self.audit.log_promotion_decision(...),
    )
    return decision

def _evaluate_data_gate(self, challenger: ModelArtifact) -> GateEvaluation:
    # FAIL if: final_oos_used_for_selection, integrity failure,
    #          OOS observations < 60, or evidence_level < LEVEL_B, LEVEL_C, LEVEL_D
    ...

def _evaluate_predictive_gate(self, challenger, champion) -> GateEvaluation:
    # With champion: challenger Rank IC > champion IC + margin (default 0.005)
    # Without champion: challenger Rank IC > 0
    margin = self.config.predictive_margin  # default 0.005
    if champion:
        passes = challenger.ic_mean > champion.ic_mean + margin
    else:
        passes = challenger.ic_mean > 0
    return GateEvaluation(gate_name="PREDICTIVE", result=GateResult.PASS if passes else GateResult.FAIL, ...)
```

### FeaturePipeline.build_vector

```python
async def build_vector(
    self,
    symbol: str,
    timestamp: datetime,
    mode: Literal["inference", "backtest"],
    pit_date: date | None = None,
) -> tuple[FeatureVector, FeatureQualityReport]:
    """
    Assemble a PIT-correct feature vector from data-service2.0 and SentinelPulse.
    """
    # 1. Check Redis cache
    cache_key = f"feature:{symbol}:{timestamp.strftime('%Y%m%d%H%M')}"
    if mode == "inference":
        cached = await self.cache.get(cache_key)
        if cached:
            return FeatureVector.model_validate_json(cached), FeatureQualityReport(...)

    missing_families: list[str] = []
    imputed_count = 0

    # 2. Fetch market features from data-service2.0
    try:
        market_data = await asyncio.wait_for(
            self.data_client.fetch_features(symbol, pit_date=pit_date),
            timeout=10.0,
        )
        signal_engine_allowed = market_data.signal_engine_allowed
        data_confidence_score = market_data.data_confidence_score

        # Gate checks
        if not signal_engine_allowed:
            logger.warning("signal_engine_not_allowed", symbol=symbol, ...)
            return _unavailable_vector(symbol, timestamp), _quality_report(...)

        if data_confidence_score < self.config.min_confidence_score:
            return _insufficient_evidence_vector(symbol, timestamp), _quality_report(...)

    except (asyncio.TimeoutError, httpx.RequestError):
        # data-service2.0 unreachable: return UNAVAILABLE
        return _unavailable_vector(symbol, timestamp), _quality_report(missing_families=["all_market"])

    # 3. Backtest mode: validate all data timestamps are before pit_date
    if mode == "backtest" and pit_date:
        market_data = self._filter_pit(market_data, pit_date)

    # 4. PIT timestamp validation: source_timestamp < feature vector timestamp
    for field_name, source_ts in market_data.source_timestamps.items():
        if source_ts >= timestamp:
            logger.error("pit_violation_detected", field=field_name, source_ts=source_ts, target_ts=timestamp)
            missing_families.append(field_name)
            imputed_count += 1

    # 5. Fetch news context from SentinelPulse
    news_features: NewsFeatures | None = None
    try:
        news_features = await asyncio.wait_for(
            self.sentinel_client.fetch_news_context(symbol),
            timeout=10.0,
        )
        if not news_features.has_required_fields():
            news_features = None  # treat as unreachable
    except (asyncio.TimeoutError, httpx.RequestError):
        news_features = None

    # 6. Apply degraded-mode substitution for missing news
    if news_features is None:
        news_kwargs = _degraded_news_defaults()
        missing_families.append("sentinel_pulse")
    else:
        news_kwargs = news_features.to_dict()

    # 7. Compute Qlib Alpha158 factors
    alpha158 = self.qlib_engine.compute_alpha158(market_data.ohlcv_df, symbol)

    # 8. Assemble FeatureVector
    vector = FeatureVector(
        symbol=symbol,
        timestamp=timestamp,
        pit_validated=True,
        **market_data.to_dict(),
        **news_kwargs,
        alpha158_factors=alpha158,
    )

    # 9. Cache in Redis
    if mode == "inference":
        await self.cache.set(cache_key, vector.model_dump_json(), ex=self.config.feature_cache_ttl)

    quality_report = FeatureQualityReport(
        batch_id=str(uuid4()),
        timestamp=datetime.utcnow(),
        total_features_requested=len(FeatureVector.model_fields),
        missing_count=len(missing_families),
        imputed_count=imputed_count,
        rejected_count=0,
        unavailable_families=missing_families,
        pit_violations_count=imputed_count,
        processing_time_ms=...,
    )
    return vector, quality_report
```

### RLExecutionAgent.act

```python
def act(self, observation: ExecutionState) -> ExecutionDecision:
    """
    FinRL inference with 50ms deadline and rule-based fallback.
    """
    artifact = self.registry.get_champion("rl_execution_agent")
    if artifact is None:
        return self._rule_based_fallback(observation, PredictionProvenance.HEURISTIC)

    start = time.perf_counter()
    try:
        obs_array = self._encode_observation(observation)
        action_idx, _ = self.policy.predict(obs_array, deterministic=True)
        action = ExecutionAction(self.ACTION_SPACE[action_idx])

        latency_ms = (time.perf_counter() - start) * 1000
        if latency_ms > 50:
            # Exceeded 50ms SLA → fall back to rule-based for this request
            logger.warning("rl_agent_sla_breach", latency_ms=latency_ms)
            return self._rule_based_fallback(observation, PredictionProvenance.HEURISTIC)

        return ExecutionDecision(
            action=action,
            confidence=float(np.max(self.policy.policy.get_distribution(obs_array).distribution.probs)),
            provenance=PredictionProvenance.TRAINED_MODEL,
            rationale=f"FinRL {artifact.algorithm} policy, version {artifact.version}",
        )
    except Exception as exc:
        logger.error("rl_agent_inference_failed", error=str(exc))
        return self._rule_based_fallback(observation, PredictionProvenance.HEURISTIC)
```


---

## Error Handling

### Degradation Hierarchy

Every model path follows a three-level degradation hierarchy:

```
TRAINED_MODEL  →  (artifact corrupted / IC below threshold)
HEURISTIC      →  (inference exception / all features NaN)
UNAVAILABLE
```

The `resolve_action()` function in `src/schemas/base.py` enforces that `VALIDATED_ML_ONLY` mode converts `HEURISTIC` to `INSUFFICIENT_EVIDENCE` (NO_TRADE).

### Circuit Breaker for data-service2.0

The `DataServiceClient` wraps all REST calls with a circuit breaker (using `tenacity` or equivalent):

```python
# 3 failures in 30s → circuit OPEN (fail fast for 60s)
# Half-open probe: 1 test request after 60s
# Success → circuit CLOSED

@circuit_breaker(failure_threshold=3, recovery_timeout=60)
async def fetch_features(self, symbol: str) -> MarketData:
    async with self.http_client.timeout(10.0):
        response = await self.http_client.get(
            f"{self.base_url}/v1/features/{symbol}",
            headers={"X-API-KEY": self.api_key},
        )
        response.raise_for_status()
        return MarketData.model_validate(response.json())
```

When the circuit is OPEN, `fetch_features` raises `DataServiceUnavailableError` immediately, and `FeaturePipeline` returns `PredictionProvenance.UNAVAILABLE` without waiting.

### gRPC Stream Recovery

```python
# Connection timeout: 5s (GRPC_CONNECTION_TIMEOUT)
# Per-call deadline: 2s (GRPC_PER_CALL_DEADLINE)
# Retry on stream interruption: up to 3 times with 1s delay
# After 3 failures: return UNAVAILABLE for all dependent predictions
```

### Retry with Exponential Backoff for SentinelPulse

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(httpx.RequestError),
)
async def fetch_news_context(self, symbol: str) -> NewsContext | None:
    ...
# On final failure: return None → FeaturePipeline applies degraded-mode substitution
```

### Artifact Integrity Validation at Startup

```python
def _load_artifact(self, path: Path, artifact: ModelArtifact) -> Any:
    expected = artifact.sha256_checksum
    actual = sha256_file(path)
    if actual != expected:
        logger.error(
            "artifact_integrity_failure",
            model=artifact.model_name,
            version=artifact.version,
            expected=expected,
            actual=actual,
        )
        raise ArtifactIntegrityFailure(artifact.model_name, artifact.version)
    return _deserialize_model(path)

# In lifespan startup: any ArtifactIntegrityFailure → fall back to heuristic for that model
# Service continues to start; affected model serves HEURISTIC predictions
```

### Audit Log Write Failure

```python
class AuditLogger:
    def log(self, entry: dict) -> None:
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except IOError as exc:
            # Critical: audit log write failure is a FATAL event
            # Log to stderr and raise - do not silently swallow
            logger.critical("audit_log_write_failure", error=str(exc))
            raise AuditLogWriteFailure(str(exc)) from exc

    def _check_append_only(self, path: Path) -> None:
        # Called at startup: verify log file mtime only increases
        # Any attempt to modify a prior entry raises AuditLogViolation
        ...
```

### 422 vs 503 Distinction

- **422**: Pydantic V2 `ValidationError` — returned by FastAPI automatically when request schema fails. Lists all field-level errors.
- **503**: Model unavailable (no artifact loaded, inference runtime exception). This enables callers to distinguish "I sent bad data" from "the service is degraded". Callers should implement retry logic on 503 but not on 422.


---

## Configuration Schema

All settings use `pydantic-settings` with `BaseSettings`. Values are read from environment variables and/or a `.env` file.

```python
from __future__ import annotations
from pydantic import Field, AnyHttpUrl
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Server ────────────────────────────────────────────────────────
    port: int = Field(default=8100, description="Port ml-service2.0 listens on")
    log_level: str = Field(default="INFO", description="structlog minimum level")
    deployment_mode: DeploymentMode = Field(
        default=DeploymentMode.RESEARCH,
        description="Controls heuristic fallback behavior: RESEARCH | PAPER | SHADOW | VALIDATED_ML_ONLY",
    )
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="CORS allowed origins (alpha-forge). Set to actual production URL in prod.",
    )

    # ── Authentication ────────────────────────────────────────────────
    ml_service_api_key: str = Field(
        description="API key required in X-API-KEY header for all prediction requests.",
    )

    # ── Upstream Services ─────────────────────────────────────────────
    data_service_2_url: AnyHttpUrl = Field(
        description="Base URL for data-service2.0, e.g. http://localhost:8200",
    )
    data_service_api_key: str = Field(
        description="API key or JWT for authenticating with data-service2.0 (production required)",
    )
    sentinel_pulse_url: AnyHttpUrl = Field(
        description="Base URL for SentinelPulse, e.g. http://localhost:3001",
    )
    sentinel_pulse_api_key: str = Field(
        default="",
        description="API key for SentinelPulse (required in production)",
    )

    # ── Storage ───────────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL for feature cache and signal cache",
    )
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI for experiment logging",
    )
    model_artifacts_path: Path = Field(
        default=Path("./artifacts"),
        description="Root path for model artifact storage (must be persistent volume in production)",
    )
    audit_log_path: Path = Field(
        default=Path("./audit.jsonl"),
        description="Path to append-only audit log file (JSON Lines format)",
    )

    # ── Feature Pipeline ─────────────────────────────────────────────
    feature_cache_ttl: int = Field(
        default=60,
        description="TTL in seconds for feature vectors cached in Redis",
        ge=10,
    )
    news_context_cache_ttl: int = Field(
        default=90,
        description="TTL in seconds for SentinelPulse news context in LRU cache",
        ge=10,
    )
    min_confidence_score: int = Field(
        default=70,
        description="Minimum DataConfidenceScore from data-service2.0 (0-100). Below threshold returns INSUFFICIENT_EVIDENCE.",
        ge=0,
        le=100,
    )
    alpha360_enabled: bool = Field(
        default=False,
        description="Enable Qlib Alpha360 (360-dimensional) factor computation (slower; disabled by default)",
    )

    # ── Training Pipeline ─────────────────────────────────────────────
    embargo_period_days: int = Field(
        default=10,
        description="Embargo period in trading days for purged K-fold CV (minimum 5)",
        ge=5,
    )
    shadow_trading_min_days: int = Field(
        default=20,
        description="Minimum trading days a model must spend in SHADOW stage before LIVE promotion eligibility",
        ge=1,
    )
    optuna_n_trials: int = Field(
        default=50,
        description="Number of Optuna HPO trials per model",
        ge=50,
    )
    cpcv_n_paths: int = Field(
        default=10,
        description="Number of overlapping test paths for CPCV",
        ge=10,
    )
    model_acceptance_min_ic: float = Field(
        default=0.02,
        description="Minimum mean IC for model acceptance gate",
    )
    model_acceptance_max_pbo: float = Field(
        default=0.5,
        description="Maximum backtest overfitting probability for acceptance gate",
    )

    # ── Online Learning ───────────────────────────────────────────────
    online_learning_enabled: bool = Field(
        default=False,
        description="Enable OnlineLearner incremental updates on IC degradation",
    )
    max_consecutive_online_updates: int = Field(
        default=5,
        description="Maximum consecutive incremental updates before full retrain is required",
        ge=1,
    )
    online_learning_ic_degradation_threshold: float = Field(
        default=0.20,
        description="IC degradation fraction from 90-day baseline that triggers online learning",
        ge=0.0,
        le=1.0,
    )

    # ── gRPC ──────────────────────────────────────────────────────────
    grpc_connection_timeout: float = Field(
        default=5.0,
        description="gRPC connection establishment timeout in seconds",
    )
    grpc_per_call_deadline: float = Field(
        default=2.0,
        description="gRPC per-call deadline in seconds",
    )
    grpc_max_reconnect_attempts: int = Field(
        default=3,
        description="Maximum reconnect attempts for interrupted gRPC stream before returning UNAVAILABLE",
    )

    # ── LLM / Meta-Decision Engine ────────────────────────────────────
    llm_inference_timeout_ms: int = Field(
        default=120,
        description="Maximum time (ms) for FinGPT/FinBERT LLM inference before falling back to cache",
        ge=50,
    )
    llm_model_name: str = Field(
        default="ProsusAI/finbert",
        description="HuggingFace model identifier for LLM news reasoning (FinBERT default, FinGPT for instruct)",
    )

    # ── Promotion Gates ───────────────────────────────────────────────
    predictive_gate_margin: float = Field(
        default=0.005,
        description="Minimum IC improvement over champion required for PREDICTIVE gate (range 0.001-0.05)",
        ge=0.001,
        le=0.05,
    )
    approval_token_expiry_hours: int = Field(
        default=24,
        description="Number of hours before an approvalToken expires for human-approved promotions",
    )

    # ── Performance ───────────────────────────────────────────────────
    uvicorn_workers: int = Field(
        default=4,
        description="Number of Uvicorn worker processes",
        ge=1,
    )

settings = Settings()
```

---

## Technology Decisions Matrix

| Library | Version | Module | Specific Use | Why Chosen | Alternative Considered |
|---------|---------|--------|-------------|-----------|----------------------|
| **Microsoft Qlib** | 0.9.x | `src/features/qlib_engine.py` | Alpha158/360 factor computation, purged K-fold CV, rolling IC | Purpose-built for quantitative finance; Alpha158/360 are peer-reviewed factors with proven IC on Asian equity markets; provides `RollingPurgedKFold` natively | Custom feature engineering — would require months to replicate 158 battle-tested factors |
| **FinRL-X** | 0.3.x | `src/models/rl_execution_agent.py` | PPO/SAC execution agent training with `StockTradingEnv` | Finance-specific gymnasium environment with NSE/position sizing extensions; integrates Stable-Baselines3 algorithms with financial reward shaping | Stable-Baselines3 alone — lacks the `StockTradingEnv` adapter and financial reward shaping primitives |
| **LangChain** | 0.3.x | `src/meta/llm_reasoner.py` | Orchestrating FinGPT/FinBERT for news_sentiment_signal with 120ms timeout | Standardized LLM chain with async support, caching, and timeout configuration; easy model swap (FinBERT ↔ FinGPT) without changing orchestration code | Raw HuggingFace `pipeline()` — lacks built-in async, caching, and chain composition |
| **FinBERT / FinGPT** | — | `src/meta/llm_reasoner.py` | Finance-tuned sentiment and direction extraction from SentinelPulse news context | FinBERT: default choice, proven on financial text, fast inference (~20ms on CPU); FinGPT: better for complex rationale generation (optional upgrade) | General-purpose GPT models — not fine-tuned on financial text; lower accuracy on market sentiment |
| **Riskfolio-Lib** | 6.x | `src/models/portfolio_optimizer.py` | HRP, CVaR-minimized MVO, ERC, Maximum Diversification | Provides all four required optimization methods under one API with Scipy/CVXPY solvers; HRP via Ward linkage directly callable | PyPortfolioOpt — already partially used but lacks CVaR and ERC; Riskfolio-Lib is a strict superset |
| **Evidently AI** | 0.4.x | `src/monitoring/drift_monitor.py` | `DataDriftPreset` for 7-day rolling vs training reference comparison | Provides DataDriftPreset and PSI metrics out of the box with Pandas integration; generates structured reports suitable for `/monitoring/drift` API | Custom PSI computation — implemented as `compute_psi()` fallback for unit tests but Evidently handles the full batch reporting |
| **NannyML** | 0.11.x | `src/monitoring/drift_monitor.py` | CBPE for estimated IC + ECE on unlabeled production data | Only library that provides Confidence-Based Performance Estimation without ground-truth labels; essential for post-deployment IC estimation | Evidently alone — requires ground truth for performance estimation, not available at prediction time |
| **gRPC** | `grpcio` 1.68.x | `src/clients/data_service.py` | Live tick feature streaming from data-service2.0 | 3-5× lower latency than REST for streaming; bidirectional streaming suits live tick subscription; reuses proto already used by data-service2.0 | REST long-polling — too high latency for 50ms inference SLA when live ticks are needed |
| **FastAPI + Uvicorn** | 0.115.x + 0.34.x | `src/main.py` | Async HTTP server | Native async/await, automatic Pydantic V2 validation, OpenAPI docs, WebSocket support; already proven in alpha-forge ml-service | Flask — sync-first, no native WebSocket; Django — too heavy for a microservice |
| **Redis (redis.asyncio)** | 5.x | `src/cache/redis_cache.py` | Feature cache (60s TTL), news context (90s), signal cache | Async-native client; sub-millisecond latency critical for 50ms SLA endpoints; TTL support prevents stale features from entering inference | In-memory dict — no TTL enforcement, no cross-process sharing, no persistence across restarts |
| **MLflow** | 2.x | `src/training/pipeline.py` | Experiment tracking, model registry, artifact storage | Self-hosted, no data leaves the cluster; first-class Python API; model registry maps to ModelRegistry in code; supports artifact URIs for SHA-256 checksums | Weights & Biases — cloud-hosted by default; sharing financial model training data with a third party is unacceptable in trading context |
| **SHAP** | 0.46.x | `src/explainability/explainer.py` | TreeExplainer (XGBoost/LightGBM/CatBoost), KernelExplainer (NN) | Industry standard for model explainability; TreeExplainer is exact and fast for tree models (~1ms per prediction) | LIME — approximation-based, less accurate; custom attribution — not industry-accepted |
| **Optuna** | 4.x | `src/training/hpo.py` | 50+ trial HPO with Spearman IC objective | Efficient TPE sampler with pruning; integrates with MLflow; already in pyproject.toml of original service | Ray Tune — heavier dependency; Hyperopt — less maintained |
| **Hypothesis** | 6.x | `tests/` | Property-based testing for all 10 test files | Best-in-class PBT library for Python; integrates with pytest; `st.floats`, `st.lists`, `st.builds` strategies cover all needed test cases | QuickCheck-style custom generators — not maintained; no shrinking support |
| **pytest-benchmark** | 5.x | `tests/test_signal_generation_latency.py` | p95 latency measurement for SLA validation | Built-in percentile computation; integrates with pytest fixtures | `time.perf_counter()` loops — usable but no built-in statistical reporting |

---

## CI/CD Pipeline Design

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  lint-type-check:
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install ruff mypy
      - run: ruff check src/ tests/
      - run: mypy --strict src/

  test:
    needs: lint-type-check
    steps:
      - run: pip install -e ".[dev]"
      - run: pytest tests/ --cov=src --cov-fail-under=90 --timeout=120
      - run: pytest tests/test_signal_generation_latency.py --benchmark-only

  integration-test:
    needs: test
    services:
      mock-data-service: { image: wiremock/wiremock:3.x }
      mock-sentinel-pulse: { image: wiremock/wiremock:3.x }
      redis: { image: redis:7-alpine }
      mlflow: { image: ghcr.io/mlflow/mlflow:v2.x }
    steps:
      - run: pytest tests/test_api_contracts.py -v

  docker-build:
    needs: integration-test
    steps:
      - run: docker build --target production -t ml-service2:${GITHUB_SHA} .
      - run: docker run --rm ml-service2:${GITHUB_SHA} python -c "from src.main import app"

  shadow-to-live-gate:
    # Only runs on main branch merges; requires PROMOTION_APPROVAL_TOKEN secret
    if: github.ref == 'refs/heads/main'
    needs: docker-build
    steps:
      - run: |
          curl -X POST http://${ML_SERVICE_URL}/training/promote \
            -H "X-API-KEY: ${ML_SERVICE_API_KEY}" \
            -H "Authorization: Bearer ${PROMOTION_APPROVAL_TOKEN}" \
            -d '{"challenger_id": "${CHALLENGER_VERSION}"}'
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        args: [--strict]
        additional_dependencies: [pydantic==2.10.3, fastapi==0.115.6]

  - repo: local
    hooks:
      - id: pytest-fast
        name: Run fast tests (no benchmarks)
        entry: pytest tests/ -m "not benchmark" --timeout=30
        language: system
        pass_filenames: false
```

### Docker Multi-Stage Build

```dockerfile
# Dockerfile
FROM python:3.11-slim AS base
WORKDIR /app
COPY pyproject.toml .

FROM base AS dev
RUN pip install -e ".[dev]"
COPY . .

FROM base AS builder
RUN pip install --no-cache-dir hatchling
RUN pip install --no-cache-dir .

FROM python:3.11-slim AS production
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ src/
COPY protos/ protos/
ENV PYTHONPATH=/app
EXPOSE 8100
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "4"]
```

### Integration Test docker-compose

```yaml
# docker-compose.test.yml
version: "3.9"
services:
  ml-service:
    build: { context: ., target: dev }
    ports: ["8100:8100"]
    environment:
      DATA_SERVICE_2_URL: http://mock-data-service:8080
      SENTINEL_PULSE_URL: http://mock-sentinel-pulse:8080
      REDIS_URL: redis://redis:6379
      ML_SERVICE_API_KEY: test-key
      DEPLOYMENT_MODE: research

  mock-data-service:
    image: wiremock/wiremock:3.x
    volumes: ["./tests/wiremock/data-service:/home/wiremock"]

  mock-sentinel-pulse:
    image: wiremock/wiremock:3.x
    volumes: ["./tests/wiremock/sentinel-pulse:/home/wiremock"]

  redis:
    image: redis:7-alpine

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.x
    ports: ["5000:5000"]
    command: mlflow server --host 0.0.0.0
```


---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: PIT Correctness

*For any* feature vector produced by the FeaturePipeline, every constituent feature's source timestamp must be strictly less than the feature vector's own target timestamp — no feature may contain information from the future relative to the prediction moment.

**Validates: Requirements 2.1, 2.2**

---

### Property 2: Feature Pipeline Idempotence

*For any* valid input data (OHLCV series, news context, option chain snapshot), computing the feature vector twice with the same inputs and the same timestamp produces identical results — the pipeline has no hidden state that changes between calls.

**Validates: Requirements 2.10**

---

### Property 3: Low DataConfidenceScore Returns INSUFFICIENT_EVIDENCE

*For any* data-service2.0 response where `DataConfidenceScore` is in the range [0, 69] (below the configured `min_confidence_score` of 70), the FeaturePipeline must return `PredictionProvenance.INSUFFICIENT_EVIDENCE` and must never proceed with model inference on that feature batch.

**Validates: Requirements 1.7**

---

### Property 4: signalEngineAllowed=false Returns UNAVAILABLE

*For any* data-service2.0 response where `signalEngineAllowed` is `false`, the FeaturePipeline must return `PredictionProvenance.UNAVAILABLE` for all prediction endpoints, regardless of other feature values.

**Validates: Requirements 1.5, 1.6**

---

### Property 5: Qlib Alpha158 Idempotence and Finiteness

*For any* valid OHLCV DataFrame with sufficient history, the Alpha158 factor computation is deterministic: running the computation twice on the same input produces the same 158-factor output. Furthermore, all 158 output factors are finite (no NaN or Inf) when the input data is complete.

**Validates: Requirements 2.3, 18.5**

---

### Property 6: LeakageValidator Detects Look-Ahead Correlations

*For any* feature matrix where at least one feature has a Pearson correlation exceeding 0.05 in absolute value with realized returns at any look-ahead window from 1 to 22 trading days, the LeakageValidator must raise a `PITViolationError` and must never allow that feature matrix to proceed to model training.

**Validates: Requirements 2.5**

---

### Property 7: Purged K-Fold Zero Temporal Overlap

*For any* time series of length ≥ 100 with a configured embargo period ≥ 5 trading days, the PurgedKFoldSplitter must produce splits where no training sample's timestamp is within `embargo_period` trading days of any validation sample's timestamp — the temporal gap between all training and validation boundaries is strictly greater than the embargo period.

**Validates: Requirements 3.1, 3.2, 18.9**

---

### Property 8: Risk Predictor Probability Invariant

*For any* valid `RiskRequest` with any combination of entry, stop-loss, target, and market condition features, the RiskPredictor output must satisfy `prob_stop_hit + prob_target_hit ≤ 1.0`. This is a fundamental probability constraint; outputs that would violate it must be clamped and a `CALIBRATION_VIOLATION` warning must be logged.

**Validates: Requirements 7.6**

---

### Property 9: Portfolio Optimizer Weight Invariants

*For any* valid returns matrix with 2 or more assets and 20 or more observations per asset, HRP optimization must produce weights that (a) sum to 1.0 within floating-point tolerance, (b) are all ≥ 0, and (c) produce a portfolio volatility ≤ the maximum individual asset volatility in the input (diversification invariant).

**Validates: Requirements 8.2, 8.7, 18.11**

---

### Property 10: Portfolio Optimizer Determinism

*For any* valid returns matrix and optimization method with the same random seed, running the optimizer twice produces identical weight vectors — the optimization is deterministic when the seed is fixed.

**Validates: Requirements 8.7**

---

### Property 11: Stock Ranker Idempotence

*For any* pair of identical stock feature vectors submitted to the StockRanker with the same model version loaded, the output rank and score must be identical — the ranker has no session state that varies between calls.

**Validates: Requirements 5.7**

---

### Property 12: Meta-Decision Engine Absorbing Identity

*For any* meta-decision request where all base model outputs carry `PredictionProvenance.UNAVAILABLE`, the MetaDecisionEngine must return `action: NO_TRADE` with `PredictionProvenance.UNAVAILABLE` — unavailability of all models absorbs to a definitive NO_TRADE rather than a spurious signal.

**Validates: Requirements 10.12, 18.4**

---

### Property 13: Meta-Decision Engine Idempotence

*For any* meta-decision input (set of model outputs + news context) with a fixed model state and calibration, calling `meta_decide(x)` twice must produce the same `MetaOutput` — the engine has no random state that varies between identical invocations.

**Validates: Requirements 17.3, 18.4**

---

### Property 14: Abstention Invariant

*For any* meta-decision input where the `agreement_ratio` is less than 0.5 (fewer than half of available models agree on the direction), the MetaDecisionEngine must not produce `action: BUY` or `action: SELL` — the output must be either `WAIT` or `NO_TRADE`.

**Validates: Requirements 10.4, 10.7, 18.4**

---

### Property 15: PSI Zero-Drift Identity

*For any* distribution D represented as a finite sample of real numbers, computing PSI(D, D) — comparing the distribution against itself using 10 equal-frequency bins — must produce a value ≤ 0.001 (approximately zero, accounting for floating-point precision).

**Validates: Requirements 12.8, 18.7**

---

### Property 16: PSI Monotonicity

*For any* reference distribution D and any shifted distribution D' where D' is obtained by adding a non-zero offset to D, PSI(D, D') must be strictly greater than PSI(D, D). Drift detection must be monotone: more shift produces a higher PSI value.

**Validates: Requirements 12.8, 18.7**

---

### Property 17: Online Learner Update Cap

*For any* model subjected to a sequence of incremental online learning updates, the online learner must never apply more than `max_consecutive_online_updates` (default 5) consecutive updates without triggering a full retraining cycle.

**Validates: Requirements 14.6, 18.12**

---

### Property 18: Online Learner Version Uniqueness

*For any* incremental update applied by the OnlineLearner, the new model version string must be strictly different from all previous version strings for that model — version strings are monotonically increasing and non-repeating.

**Validates: Requirements 14.4, 18.12**

---

### Property 19: Prior Artifact Immutability After Online Update

*For any* online learning update, the prior model artifact file must remain intact on disk — incremental updates create new files and never overwrite or delete the predecessor artifact.

**Validates: Requirements 14.4, 18.12**

---

### Property 20: Schema Round-Trip Serialization

*For any* valid instance of a Pydantic V2 schema (RegimePredictionResponse, MetaOutput, FeatureVector, etc.), serializing the object to JSON and deserializing it back must produce an object equal to the original — no data is silently corrupted at serialization boundaries.

**Validates: Requirements 19.1, 19.2, 19.4**

---

### Property 21: Idempotent Parse-Serialize Cycle

*For any* valid Pydantic request schema instance S, `parse(serialize(parse(S)))` must equal `parse(S)` — repeated serialize/parse cycles do not accumulate rounding errors or field mutations.

**Validates: Requirements 19.4**

---

### Property 22: Prediction Endpoint Idempotence

*For any* prediction request body and the same loaded model version, calling the same prediction endpoint twice must return identical responses — all prediction endpoints are pure functions of their inputs when model state is fixed.

**Validates: Requirements 17.3**

---

## Testing Strategy

### Dual Testing Approach

Every acceptance criterion is covered by at least one of:
- **Unit tests**: specific examples, edge cases, error conditions with concrete inputs
- **Property tests**: universal properties across randomized inputs (Hypothesis, min 100 iterations per test)
- **Integration tests**: API contracts, external service integration, latency SLAs

The two approaches are complementary: unit tests verify known-correct examples, property tests find unexpected edge cases across the full input space.

### Property-Based Testing Library

**Hypothesis 6.x** is the chosen PBT library for all property tests in this service.

**Configuration per property test:**
```python
from hypothesis import given, settings, HealthCheck

@settings(
    max_examples=100,               # minimum 100 iterations
    suppress_health_check=[HealthCheck.too_slow],
    deadline=500,                   # 500ms per test case
)
@given(...)
def test_property_N_name(self, ...):
    ...
    # Feature: ml-service2, Property N: <property_text>
```

### Test File Design

#### `tests/test_feature_pipeline_pit_correctness.py` — Properties 1, 2, 3, 4

**Hypothesis strategies:**
```python
# Random timestamps in trading hours
pit_timestamps = st.datetimes(
    min_value=datetime(2020, 1, 1, 9, 15),
    max_value=datetime(2025, 12, 31, 15, 30),
).filter(lambda dt: dt.weekday() < 5)  # weekdays only

# Feature vectors with arbitrary source timestamps
feature_vectors = st.builds(
    FeatureVector,
    symbol=st.from_regex(r"[A-Z]{3,6}", fullmatch=True),
    timestamp=pit_timestamps,
    data_confidence_score=st.integers(min_value=0, max_value=100),
    signal_engine_allowed=st.booleans(),
)
```

**Mock structures:**
```python
@pytest.fixture
def mock_data_client():
    """Returns a mock DataServiceClient that respects signalEngineAllowed and DataConfidenceScore."""
    client = AsyncMock(spec=DataServiceClient)
    client.fetch_features.return_value = MarketData(
        signal_engine_allowed=True,
        data_confidence_score=85,
        source_timestamps={},  # overridden per test
        ohlcv_df=pd.DataFrame(...),
    )
    return client

@pytest.fixture
def mock_sentinel_client():
    client = AsyncMock(spec=SentinelPulseClient)
    client.fetch_news_context.return_value = None  # degraded mode
    return client
```

**Property assertions:**
```python
@given(feature_vectors)
def test_pit_correctness_all_sources_before_target(self, vector):
    # Property 1: source_timestamp(f) < vector.timestamp for all features
    for field, source_ts in vector.source_timestamps.items():
        assert source_ts < vector.timestamp, f"Field {field} has future source timestamp"

@given(st.integers(min_value=0, max_value=69))
def test_low_confidence_returns_insufficient_evidence(self, score):
    # Property 3
    result = pipeline.build_vector(symbol="NIFTY", data_confidence_score=score)
    assert result.provenance == PredictionProvenance.INSUFFICIENT_EVIDENCE
```

---

#### `tests/test_meta_decision_engine.py` — Properties 12, 13, 14

**Hypothesis strategies:**
```python
# All-unavailable model outputs (Property 12)
unavailable_outputs = st.fixed_dictionaries({
    model: st.builds(ModelOutput, provenance=st.just(PredictionProvenance.UNAVAILABLE))
    for model in ["regime", "ranker", "strategy", "risk", "rl_agent"]
})

# Random model outputs with varying agreement (Properties 13, 14)
model_outputs_low_agreement = st.fixed_dictionaries({...}).filter(
    lambda outputs: _compute_agreement_ratio(outputs) < 0.5
)
```

**Property assertions:**
```python
@given(unavailable_outputs)
def test_all_unavailable_returns_no_trade(self, outputs):
    # Property 12: Absorbing identity
    result = engine.decide(outputs, news_context=mock_news_context())
    assert result.action == "NO_TRADE"
    assert result.provenance == PredictionProvenance.UNAVAILABLE

@given(meta_inputs)
def test_idempotence(self, inputs):
    # Property 13
    result1 = engine.decide(*inputs)
    result2 = engine.decide(*inputs)
    assert result1.action == result2.action
    assert abs(result1.confidence - result2.confidence) < 1e-9

@given(model_outputs_low_agreement)
def test_abstention_when_low_agreement(self, outputs):
    # Property 14
    result = engine.decide(outputs, news_context=mock_news_context())
    assert result.action in {"WAIT", "NO_TRADE"}
    assert result.abstention is True
```

---

#### `tests/test_qlib_feature_engineering.py` — Property 5

**Hypothesis strategies:**
```python
ohlcv_dataframes = st.builds(
    _build_ohlcv_df,
    n_bars=st.integers(min_value=252, max_value=500),
    base_price=st.floats(min_value=100.0, max_value=50000.0),
    volatility=st.floats(min_value=0.005, max_value=0.05),
)
```

**Property assertions:**
```python
@given(ohlcv_dataframes, st.from_regex(r"[A-Z]{3,6}", fullmatch=True))
def test_alpha158_idempotence(self, df, symbol):
    # Property 5a: idempotence
    result1 = engine.compute_alpha158(df, symbol)
    result2 = engine.compute_alpha158(df, symbol)
    assert result1 == result2

@given(ohlcv_dataframes, st.from_regex(r"[A-Z]{3,6}", fullmatch=True))
def test_alpha158_all_finite(self, df, symbol):
    # Property 5b: all outputs finite
    result = engine.compute_alpha158(df, symbol)
    for name, value in result.items():
        assert math.isfinite(value), f"Factor {name} is not finite"
```

---

#### `tests/test_finrl_execution_agent.py` — Action space + latency + fallback (integration)

```python
def test_action_space_is_exactly_7_elements():
    agent = RLExecutionAgent(registry=mock_registry_no_artifact())
    assert set(ExecutionAction) == {ExecutionAction.ENTER_NOW, ExecutionAction.WAIT,
                                     ExecutionAction.SCALE_IN, ExecutionAction.PARTIAL_EXIT,
                                     ExecutionAction.FULL_EXIT, ExecutionAction.TIGHTEN_STOP,
                                     ExecutionAction.TRAIL_STOP}

def test_agent_responds_within_50ms_p95():
    # Integration: 100 consecutive calls, measure p95
    agent = RLExecutionAgent(registry=mock_registry_with_artifact())
    states = [random_execution_state() for _ in range(100)]
    latencies = []
    for state in states:
        start = time.perf_counter()
        agent.act(state)
        latencies.append((time.perf_counter() - start) * 1000)
    latencies.sort()
    p95 = latencies[int(0.95 * len(latencies))]
    assert p95 <= 50.0, f"p95 latency {p95:.1f}ms exceeds 50ms SLA"

def test_fallback_when_no_artifact():
    agent = RLExecutionAgent(registry=mock_registry_no_artifact())
    decision = agent.act(random_execution_state())
    assert decision.provenance == PredictionProvenance.HEURISTIC
```

---

#### `tests/test_drift_monitoring.py` — Properties 15, 16

```python
@given(st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=100))
def test_psi_identity_zero_drift(self, distribution):
    # Property 15: PSI(D, D) ≈ 0
    dist = np.array(distribution)
    psi = monitor.compute_psi(dist, dist)
    assert psi < 0.001

@given(
    st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False), min_size=100),
    st.floats(min_value=1.0, max_value=5.0),  # non-zero shift
)
def test_psi_monotonicity(self, distribution, shift):
    # Property 16: shifted distribution has higher PSI
    ref = np.array(distribution)
    shifted = ref + shift
    psi_identity = monitor.compute_psi(ref, ref)
    psi_shifted  = monitor.compute_psi(ref, shifted)
    assert psi_shifted > psi_identity
```

---

#### `tests/test_signal_generation_latency.py` — Property 22 (timing)

```python
@pytest.mark.benchmark
@pytest.mark.parametrize("endpoint,payload_factory", [
    ("/v2/predict/regime",    make_regime_request),
    ("/v2/predict/strategy",  make_strategy_request),
    ("/v2/predict/risk",      make_risk_request),
    ("/v2/predict/execution", make_execution_request),
])
def test_single_symbol_endpoint_p95_latency(client, endpoint, payload_factory, benchmark):
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        response = client.post(endpoint, json=payload_factory())
        latencies.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200
    latencies.sort()
    p95 = latencies[int(0.95 * len(latencies))]
    assert p95 <= 50.0, f"{endpoint}: p95 {p95:.1f}ms > 50ms SLA"
```

---

#### `tests/test_model_training_purged_cv.py` — Property 7

```python
@given(
    ts_length=st.integers(min_value=100, max_value=1000),
    embargo_days=st.integers(min_value=5, max_value=20),
)
def test_zero_temporal_overlap_with_embargo(self, ts_length, embargo_days):
    # Property 7
    timestamps = pd.date_range("2020-01-01", periods=ts_length, freq="B")
    X = pd.DataFrame({"f1": np.random.randn(ts_length)})
    y = pd.Series(np.random.randn(ts_length))

    splitter = PurgedKFoldSplitter(embargo_days=embargo_days)
    for train_idx, val_idx in splitter.split(X, y, pd.Series(timestamps)):
        train_times = timestamps[train_idx]
        val_times   = timestamps[val_idx]
        val_start   = val_times.min()
        val_end     = val_times.max()
        embargo_td  = pd.Timedelta(days=embargo_days * 1.4)  # ~1.4 calendar days per trading day

        for t in train_times:
            assert not (val_start - embargo_td <= t <= val_end + embargo_td), \
                f"Training sample {t} is within embargo of validation fold [{val_start}, {val_end}]"
```

---

#### `tests/test_api_contracts.py` — Integration (Properties 20, 22)

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint,factory,response_model", [
    ("/v2/predict/regime", make_regime_payload, RegimePredictionResponse),
    ("/v2/predict/risk",   make_risk_payload,   RiskResponse),
    ("/v2/meta/decide",    make_meta_payload,   MetaOutput),
])
async def test_prediction_endpoint_has_provenance_field(async_client, endpoint, factory, response_model):
    response = await async_client.post(endpoint, json=factory())
    assert response.status_code == 200
    data = response.json()
    assert "provenance" in data
    assert data["provenance"] in [e.value for e in PredictionProvenance]

async def test_invalid_schema_returns_422(async_client):
    response = await async_client.post(
        "/v2/predict/regime",
        json={"nifty_change_pct": "not_a_float"},  # invalid type
        headers={"X-API-KEY": "test-key"},
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(e["loc"][-1] == "nifty_change_pct" for e in errors)
```

---

#### `tests/test_portfolio_optimization.py` — Properties 9, 10

```python
@given(
    n_assets=st.integers(min_value=2, max_value=30),
    n_obs=st.integers(min_value=20, max_value=252),
)
def test_hrp_weights_sum_to_one(self, n_assets, n_obs):
    # Property 9a
    returns = _random_returns_matrix(n_assets, n_obs)
    weights = optimizer.hrp_allocation(returns)["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9

@given(
    n_assets=st.integers(min_value=2, max_value=30),
    n_obs=st.integers(min_value=20, max_value=252),
)
def test_hrp_all_weights_non_negative(self, n_assets, n_obs):
    # Property 9b
    returns = _random_returns_matrix(n_assets, n_obs)
    weights = optimizer.hrp_allocation(returns)["weights"]
    assert all(w >= 0.0 for w in weights.values())

@given(
    n_assets=st.integers(min_value=2, max_value=30),
    n_obs=st.integers(min_value=20, max_value=252),
)
def test_hrp_diversification_invariant(self, n_assets, n_obs):
    # Property 9c: portfolio vol <= max individual asset vol
    returns = _random_returns_matrix(n_assets, n_obs)
    result = optimizer.hrp_allocation(returns)
    port_vol = result["risk_metrics"]["volatility"]
    max_asset_vol = max(returns[col].std() * np.sqrt(252) for col in returns.columns)
    assert port_vol <= max_asset_vol + 1e-6  # tolerance for floating point

@given(
    n_assets=st.integers(min_value=2, max_value=20),
    n_obs=st.integers(min_value=20, max_value=252),
)
def test_optimizer_determinism(self, n_assets, n_obs):
    # Property 10
    returns = _random_returns_matrix(n_assets, n_obs, seed=42)
    result1 = optimizer.hrp_allocation(returns)
    result2 = optimizer.hrp_allocation(returns)
    for symbol in result1["weights"]:
        assert abs(result1["weights"][symbol] - result2["weights"][symbol]) < 1e-12
```

---

#### `tests/test_online_learning.py` — Properties 17, 18, 19

```python
@given(
    n_updates=st.integers(min_value=6, max_value=20),
)
def test_update_cap_enforces_full_retrain_after_5_consecutive(self, n_updates):
    # Property 17
    learner = OnlineLearner(max_consecutive_updates=5, registry=mock_registry())
    full_retrain_triggered = False
    for i in range(n_updates):
        result = learner.update("market_regime", new_data=random_data())
        if result is None and i >= 5:
            full_retrain_triggered = True
            break
    assert full_retrain_triggered, "Full retrain must be triggered after 5 consecutive updates"

@given(
    n_updates=st.integers(min_value=1, max_value=5),
)
def test_version_strings_are_unique_after_updates(self, n_updates):
    # Property 18
    learner = OnlineLearner(max_consecutive_updates=5, registry=mock_registry())
    versions = set()
    for _ in range(n_updates):
        version = learner.update("market_regime", new_data=random_data())
        if version:
            assert version not in versions, "Version string must be unique"
            versions.add(version)

@given(st.integers(min_value=1, max_value=5))
def test_prior_artifact_intact_after_update(self, n_updates):
    # Property 19
    learner = OnlineLearner(max_consecutive_updates=5, registry=mock_registry())
    initial_artifact = learner.registry.get_champion("market_regime")
    for _ in range(n_updates):
        learner.update("market_regime", new_data=random_data())
    assert Path(initial_artifact.artifact_path).exists(), "Prior artifact must not be deleted"
```


---

## Open-Source Integration Blueprint

| Library | Version | Module in ml-service2.0 | Specific Use | Why Chosen Over Alternative | Integration Notes |
|---------|---------|------------------------|-------------|----------------------------|-------------------|
| **Microsoft Qlib** | 0.9.x | `src/features/qlib_engine.py`, `src/training/purged_kfold.py` | Alpha158/360 factor computation; `RollingPurgedKFold` for purged CV; double-ensembled training pipeline | Purpose-built for quantitative finance with peer-reviewed factor libraries. Alpha158/360 have documented IC > 0.05 on Asian equity markets out-of-the-box. `RollingPurgedKFold` implements López de Prado's purging natively. | Requires `qlib>=0.9.0`; must call `qlib.init(provider_uri=data_path)` once at startup; Alpha158 requires at least 200 bars of OHLCV history |
| **FinRL-X** | 0.3.x | `src/models/rl_execution_agent.py` | PPO/SAC training with `StockTradingEnv`; NSE session calendar extension | `StockTradingEnv` provides the gymnasium wrapper with financial reward shaping out of the box; PPO and SAC both supported with algorithm selector in config. Stable-Baselines3 alone lacks the financial environment adapter. | Depends on `stable-baselines3>=2.4.0` as backend; training exclusively on PIT-correct backtest data via `data-service2.0?pit_date=` parameter; policy saved as `.zip` checkpoint |
| **LangChain** | 0.3.x | `src/meta/llm_reasoner.py` | Orchestrating FinBERT/FinGPT inference with async, caching, 120ms timeout | Provides `AsyncLLMChain` with built-in timeout support and LRU caching of inference results. Swapping from FinBERT to FinGPT is a one-line config change. | `langchain-community` for HuggingFace integration; `langchain-huggingface` for local model loading; LLM instances loaded once at startup |
| **FinBERT** | (`ProsusAI/finbert`) | `src/meta/llm_reasoner.py` | Financial sentiment classification — direction (+1/0/-1), confidence, 30-word rationale | Fine-tuned on 10K financial news articles; ~20ms CPU inference vs ~150ms for general-purpose GPT-4o-mini; no API cost for on-prem deployment | Default `llm_model_name`; FinGPT (`THUDM/FinGPT-v3.3`) is configurable alternative for richer rationale generation; requires ~450MB model weight download on first run |
| **Riskfolio-Lib** | 6.x | `src/models/portfolio_optimizer.py` | HRP (Ward linkage), CVaR-minimized MVO, ERC, Maximum Diversification | Provides all four required optimization methods under a single Pandas-native API with CVXPY solvers. PyPortfolioOpt only covers MVO and max-Sharpe; lacks CVaR and ERC. Riskfolio-Lib is a strict superset. | `riskfolio-lib>=6.0.0`; depends on `cvxpy` for CVaR; CVXPY must have ECOS solver available; add `max_sector_weight` constraint via `riskfolio.constraints` module |
| **PyPortfolioOpt** | 1.5.5 | `src/models/portfolio_optimizer.py` (legacy `optimize()`) | Retained for backward-compatible `POST /v2/predict/portfolio` endpoint | Existing endpoint is in use by alpha-forge; migrating to Riskfolio-Lib for all new `/v2/predict/portfolio-v2` methods. PyPortfolioOpt retained only for the legacy single-method endpoint. | `pypfopt>=1.5.5`; EfficientFrontier + HRPOpt already in pyproject.toml |
| **Evidently AI** | 0.4.x | `src/monitoring/drift_monitor.py` | `DataDriftPreset` for feature distribution monitoring; PSI computation | Provides pre-built `DataDriftPreset` that computes PSI across all features in one call with Pandas integration. Structured report output feeds directly into `/monitoring/drift` API. | `evidently>=0.4.0`; reference dataset is the training data distribution stored at model registration time; `Report` object serialized to JSON for the API response |
| **NannyML** | 0.11.x | `src/monitoring/drift_monitor.py` | CBPE (Confidence-Based Performance Estimation) for label-free IC estimation | Only maintained library providing CBPE for post-deployment performance estimation without ground-truth labels. Custom drift detection cannot estimate IC without realized returns. | `nannyml>=0.11.0`; requires storing calibrated model confidence scores in production; CBPE chunker set to `PeriodBasedChunker(period='daily')` |
| **NautilusTrader** | — | Pattern reference only (not a dependency) | Architecture pattern for deterministic backtest environment | Event-driven, deterministic simulation pattern used as the template for `StockTradingEnv` extensions (session calendar, fill simulation). Not imported directly — pattern only. | No dependency; architectural influence only |
| **SHAP** | 0.46.x | `src/explainability/explainer.py` | `TreeExplainer` for XGBoost/LightGBM/CatBoost; `KernelExplainer` for NN models | Industry standard; `TreeExplainer` is exact (not an approximation) and runs in ~1ms per prediction for tree models. `KernelExplainer` is model-agnostic for the NN models. | `shap>=0.46.0`; explainer instances cached in-process per model; `shap.maskers.Independent` used for KernelExplainer background |
| **Optuna** | 4.x | `src/training/hpo.py` | 50+ trial HPO with TPE sampler; objective = mean Spearman IC | Efficient TPE sampler with MedianPruner for early stopping of poor trials. Native MLflow callback (`MLflowCallback`) logs every trial automatically. Already used in the original service. | `optuna>=4.1.0`; `MLflowCallback` from `optuna-integration`; studies persisted in `sqlite:///optuna.db` to survive restarts |
| **MLflow** | 2.x | `src/training/pipeline.py`, `src/registry/registry.py` | Experiment tracking, model registry, artifact storage with SHA-256 | Self-hosted (no data leaves the cluster); first-class Python API; `mlflow.register_model()` maps directly to ModelRegistry champion/challenger logic | `mlflow>=2.0.0`; tracking server runs as a separate Docker container; artifact store URI configurable (`MLFLOW_TRACKING_URI`) |
| **Hypothesis** | 6.x | All `tests/test_*.py` | Property-based testing for all 22 correctness properties | Best-in-class PBT for Python with automatic test case shrinking; integrates natively with pytest; `st.builds`, `st.floats`, `st.lists` cover all required strategies | No maintained alternative in the Python ecosystem with equivalent shrinking support |
| **pytest-benchmark** | 5.x | `tests/test_signal_generation_latency.py` | p95 latency measurement for all SLA endpoints | Built-in percentile computation and histogram reporting; integrates with pytest fixtures without extra setup | `time.perf_counter()` is used as a fallback for test environments where pytest-benchmark is unavailable |
| **stable-baselines3** | 2.4.0 | `src/models/rl_execution_agent.py` (via FinRL-X) | PPO and SAC algorithm implementations used by FinRL-X backend | FinRL-X delegates training to SB3 as its RL algorithm library; retained as a direct dependency for the rule-based `_rule_based_fallback()` execution policy | `stable-baselines3>=2.4.0`; the rule-based executor from alpha-forge is migrated into `_rule_based_fallback()` within `RLExecutionAgent` and is not an independent module |
| **grpcio + grpcio-tools** | 1.68.x | `src/clients/data_service.py`, `protos/` | gRPC streaming for live tick features from data-service2.0 | 3-5× lower latency than REST for streaming tick data; bidirectional streaming supports live subscriptions; reuses proto definition already used by data-service2.0 | `grpcio>=1.68.0`, `grpcio-tools` for stub generation from `market_data.proto`; stubs generated at build time via `python -m grpc_tools.protoc` |
| **structlog** | 24.4.x | All modules | Structured JSON logging with mandatory fields | Already used in original service; async-safe; renders JSON with all required fields: `timestamp`, `service`, `version`, `request_id`, `model_name`, `latency_ms`, `provenance`, `deployment_mode` | `structlog>=24.4.0`; configured with `JSONRenderer` in `src/config.py` lifespan |
| **httpx** | 0.28.x | `src/clients/data_service.py`, `src/clients/sentinel_pulse.py`, `tests/test_api_contracts.py` | Async HTTP client for data-service2.0 and SentinelPulse REST calls; `AsyncClient` for integration tests | Async-native; compatible with `asyncio`; timeout configuration per request; already in pyproject.toml | `httpx>=0.28.0` |

---

## Open Integration Points for Future Development

1. **price_forecaster.py (Temporal Fusion Transformer)** — The existing `PriceForecaster` with TFT is migrated as-is under `src/models/price_forecaster.py` and served at `POST /v2/predict/price-regime`. A future upgrade path to NautilusTrader's backtesting engine for TFT training data is architecturally pre-wired via the `BacktestMode` flag in `FeaturePipeline`.

2. **iv_regime_classifier.py (PatchTST)** — The existing `IVClassifier` with PatchTST is migrated under `src/models/iv_regime_classifier.py` and served at `POST /v2/predict/iv-regime`. The IV regime output is consumed as a mandatory feature by `StrategySelector` (Requirement 6.3).

3. **SentinelPulse PIT Training Endpoints** — The Feature Pipeline exclusively uses PIT-correct endpoints (`GET /api/v1/ml/features/asset/:assetId`, `/ml/training/samples`, `/ml/historical-reactions`) for all training data assembly. Any future addition of SentinelPulse training endpoint versions must be validated for `look_ahead_validated: true` (Requirement 1.13–1.16).

4. **gRPC Live Tick Streaming** — The `DataServiceClient` supports both REST (for inference with cached features) and gRPC bidirectional streaming (for live tick-level feature updates). The gRPC proto is consumed from data-service2.0's published `market_data.proto`; the stub is generated at build time. The streaming path is activated when `GRPC_ENABLED=true` in the environment.

