# Implementation Plan: ml-service2.0

## Overview

ml-service2.0 is a standalone, institutional-grade Python 3.11+ FastAPI microservice that replaces and vastly upgrades the legacy ML service embedded in alpha-forge. Tasks are ordered with TDD-first: every test task precedes its corresponding implementation task. Infrastructure and schemas come first; integration and end-to-end tests come last.

---

## Tasks

---

### Phase 0: Project Scaffold & Infrastructure

- [x] 1. Initialise project repository structure and pyproject.toml
  - Create the full directory layout defined in the design (`src/api/`, `src/features/`, `src/models/`, `src/training/`, `src/meta/`, `src/monitoring/`, `src/registry/`, `src/explainability/`, `src/streaming/`, `src/audit/`, `src/clients/`, `src/cache/`, `src/schemas/`, `src/analytics/`, `tests/`, `protos/`)
  - Write `pyproject.toml` targeting Python 3.11+, declaring all production dependencies (fastapi==0.115.x, uvicorn[standard], numpy, pandas, scikit-learn, xgboost, lightgbm, catboost, stable-baselines3, gymnasium, shap, riskfolio-lib, scipy, joblib, pydantic==2.x, pydantic-settings, httpx, python-dotenv, structlog, ta, optuna, qlib, finrl, langchain, langchain-community, langchain-huggingface, transformers, evidently, nannyml, redis[hiredis], grpcio, grpcio-tools, mlflow, tenacity) and dev dependencies (pytest, pytest-asyncio, pytest-cov, pytest-benchmark, hypothesis, ruff, mypy, wiremock-python or httpretty for mocks)
  - Add `Readme.md` with setup instructions
  - _Requirements: Req 15.1, Req 17.1_

- [x] 2. Implement all Pydantic V2 schemas (`src/schemas/`)
  - [x] 2.1 Implement core enums and base schemas (`src/schemas/base.py`)
    - Define `PredictionProvenance`, `DeploymentMode`, `MarketRegime`, `TradingStrategy`, `ExecutionAction`, `IVRegime`, `ModelLifecycleStage`, `PromotionOutcome`, `GateResult`, `DriftSeverity`, `ImpactDirection` — all as `str, Enum`
    - Add `is_live_eligible` property on `PredictionProvenance`
    - All enums must use `model_config = ConfigDict(strict=True)` where applicable
    - _Requirements: Req 4.1, Req 6.1, Req 9.2, Req 13.2, Req 17.2_

  - [x] 2.2 Implement feature schemas (`src/schemas/features.py`)
    - Define `FeatureVector` (all market fields, `alpha158_factors: dict[str, float]`, `alpha360_factors: dict[str, float]`, all SentinelPulse news fields including `impact_direction: ImpactDirection`, `pit_validated: bool`, `data_confidence_score: int`, `signal_engine_allowed: bool`)
    - Define `FeatureQualityReport` (batch_id, timestamp, total_features_requested, missing_count, imputed_count, rejected_count, unavailable_families, pit_violations_count, discarded_backtest_records, processing_time_ms)
    - All models use `model_config = ConfigDict(strict=True)` and `from __future__ import annotations`
    - _Requirements: Req 2.1, Req 2.9_

  - [x] 2.3 Implement prediction request/response schemas (`src/schemas/predictions.py`)
    - Migrate and upgrade from alpha-forge `src/schemas.py`: `RegimePredictionRequest`, `RegimePredictionResponse`, `StockFeatures`, `RankingRequest`, `StockRank`, `RankingResponse`, `StrategyRequest`, `StrategyResponse`, `RiskRequest`, `RiskResponse`, `PortfolioAsset`, `PortfolioRequest`, `PortfolioAllocation`, `PortfolioResponse`, `ExecutionState`, `ExecutionDecision`, `ExplainRequest`, `FeatureContribution`, `ExplainResponse`
    - Add `provenance: PredictionProvenance` to `RegimePredictionResponse`, `RankingResponse`, `RiskResponse` where missing
    - Add `PortfolioV2Request` and `PortfolioV2Response` for Riskfolio-Lib endpoint
    - Add `PriceRegimeRequest`, `PriceRegimeResponse`, `IVRegimeRequest`, `IVRegimeResponse`
    - All schemas use `ConfigDict(strict=True)` and `from __future__ import annotations`
    - _Requirements: Req 15.3, Req 17.2, Req 19.1_

  - [x] 2.4 Implement MetaOutput and meta schemas (`src/schemas/meta.py`)
    - Define `NewsSignal` (direction ∈ {-1,0,1}, confidence, rationale max 200 chars)
    - Define `ConfidenceDecomposition` (base_confidence, calibration_quality, agreement_bonus, data_quality_factor, regime_confidence_factor)
    - Define `MetaOutputExplainability` (top_features: list[str], conflicting_models: list[str], rationale: str)
    - Define `MetaOutput` (all fields from design: action, confidence, uncertainty, agreement, agreement_ratio, ensemble_score, reason_codes, contributing_models, abstention, provenance, news_sentiment_signal, decomposition, explainability, request_id, symbol, timestamp, latency_ms)
    - Define `MetaDecideRequest` (symbol, regime, feature_vector, force_refresh_news)
    - Define `OOSRecord` (model_id, raw_score, realized_outcome, timestamp)
    - Define `MetaFitResponse`
    - _Requirements: Req 10.1, Req 10.8, Req 19.2_

  - [x] 2.5 Implement registry, training, monitoring, and streaming schemas
    - `src/schemas/registry.py`: `ModelArtifact` (all fields from design), `GateEvaluation`, `PromotionDecision`, `TrainingRun`, `TrainingConfig`, `TrainingRunStatus`
    - `src/schemas/monitoring.py`: `DriftAlert`
    - `src/schemas/streaming.py`: `SignalEvent`
    - All use `ConfigDict(strict=True)` and `from __future__ import annotations`
    - _Requirements: Req 12.9, Req 13.1, Req 14.4_

- [x] 3. Implement pydantic-settings configuration (`src/config.py`)
  - Define `Settings(BaseSettings)` with `SettingsConfigDict(env_file=".env")` covering all 30+ environment variables from the design: port, log_level, deployment_mode, allowed_origins, ml_service_api_key, data_service_2_url, data_service_api_key, sentinel_pulse_url, sentinel_pulse_api_key, redis_url, mlflow_tracking_uri, model_artifacts_path, audit_log_path, feature_cache_ttl, news_context_cache_ttl, min_confidence_score (default 70), alpha360_enabled, embargo_period_days (default 10, min 5), shadow_trading_min_days, optuna_n_trials (default 50), cpcv_n_paths, model_acceptance_min_ic, model_acceptance_max_pbo, online_learning_enabled, max_consecutive_online_updates, online_learning_ic_degradation_threshold, grpc_connection_timeout, grpc_per_call_deadline, grpc_max_reconnect_attempts, llm_inference_timeout_ms, llm_model_name, predictive_gate_margin, approval_token_expiry_hours, uvicorn_workers
  - Create `.env.example` with all variable names and descriptions
  - Export `settings = Settings()` singleton
  - _Requirements: Req 1.3, Req 15.2, Req 16.7, Req 16.8_

- [x] 4. Implement structlog configuration and logging setup
  - Configure structlog with `JSONRenderer` outputting all mandatory fields: `timestamp`, `service`, `version`, `request_id`, `model_name`, `latency_ms`, `provenance`, `deployment_mode`
  - Add `get_logger()` helper and context-variable `request_id` binding
  - Integrate with FastAPI middleware so every request automatically binds `X-Request-ID` to the log context
  - _Requirements: Req 15.11, Req 17.8_

- [x] 5. Implement Redis async cache layer (`src/cache/redis_cache.py`)
  - Implement `RedisCache` using `redis.asyncio` with all key patterns from the design: `feature:{symbol}:{bucket}` (TTL=60s), `news:{symbol}` (TTL=90s), `signal:{symbol}:latest` (TTL=300s), `llm_news:{symbol}` (TTL=60s), `model:status` (no TTL), `drift:latest` (TTL=86400s)
  - Implement `get`, `set`, `delete`, `exists` with async/await
  - Add in-process LRU cache (max 500 entries) for SHAP explainer instances with 90s TTL for news context
  - _Requirements: Req 16.6, Req 16.7_

- [x] 6. Implement AuditLogger (`src/audit/logger.py`)
  - Implement `AuditLogger` writing JSON Lines to `audit.jsonl` (path from `Settings.audit_log_path`)
  - Implement `log_training_run()`, `log_promotion_decision()`, `log_online_update()`, `log_online_update_rejected()` methods, each with required fields: run_id, timestamp, event_type, model_name, model_version, gate_results, outcome, reviewer_identity (optional), dataset_hash
  - Implement `_check_append_only()` at startup: detect and raise `AuditLogViolation` if any past entry has been modified (verify via stored entry hashes)
  - Raise `AuditLogWriteFailure` on IOError and log to stderr (never swallow silently)
  - _Requirements: Req 3.9, Req 13.14, Req 17.5_

- [x] 7. Implement FastAPI application skeleton with middleware (`src/main.py`, `src/api/`)
  - Create FastAPI app with lifespan (startup: load all model artifacts, check AuditLog integrity, connect Redis, connect MLflow; shutdown: close connections)
  - Implement `X-API-KEY` authentication middleware (return 401 on missing/invalid key; exempt `/health`)
  - Implement `X-Request-ID` middleware (generate UUID if not present; bind to structlog context)
  - Implement CORS middleware with `allowed_origins` from settings
  - Implement global exception handlers: `RequestValidationError` → 422, model-unavailable exceptions → 503, unhandled → 500
  - Wire in all routers from `src/api/` (stubs initially)
  - Create `GET /health` endpoint returning `{"status": "healthy", "timestamp": int, "version": str}`
  - _Requirements: Req 15.2, Req 15.8, Req 15.9, Req 15.10, Req 15.11_

- [x] 8. Write test_api_contracts.py (TDD — write tests BEFORE Phase 1+ implementations)
  - [x] 8.1 Write schema round-trip serialization tests
    - For each Pydantic V2 schema (RegimePredictionResponse, MetaOutput, FeatureVector, RiskResponse, RankingResponse, PortfolioResponse, SignalEvent, DriftAlert, ModelArtifact), assert `parse(serialize(instance)) == instance` (Property 20)
    - Test idempotent parse-serialize cycle: `parse(serialize(parse(x))) == parse(x)` for all valid schemas (Property 21)
    - Test that unknown enum values in requests return HTTP 422 with field-level errors (Req 19.3)
    - _Requirements: Req 19.1, Req 19.2, Req 19.3, Req 19.4_

  - [x] 8.2 Write HTTP auth and status integration tests
    - Test all prediction endpoints return 401 when `X-API-KEY` is missing or invalid
    - Test `/health` returns 200 without API key
    - Test all prediction response bodies contain a `provenance` field with a valid `PredictionProvenance` string value
    - Test 422 is returned for malformed request bodies (missing required field, out-of-range, invalid enum)
    - Test 503 is returned when model is unavailable (mock unavailable model)
    - _Requirements: Req 15.2, Req 15.8, Req 15.9_

- [x] 9. Generate gRPC protobuf stubs
  - Add `protos/market_data.proto` defining the tick streaming service contract matching data-service2.0's gRPC API
  - Add `protos/generate_stubs.sh` that runs `python -m grpc_tools.protoc` to produce `src/clients/market_data_pb2.py` and `src/clients/market_data_pb2_grpc.py`
  - Add stub generation as a build step in `pyproject.toml` scripts section
  - _Requirements: Req 1.17, Req 1.18_

- [x] 10. Create Docker multi-stage Dockerfile and docker-compose
  - Write `Dockerfile` with `base` → `dev` → `builder` → `production` stages as per design
  - Write `docker-compose.yml` for development: ml-service2.0 (port 8100), Redis 7-alpine, MLflow 2.x, WireMock for mock-data-service, WireMock for mock-sentinel-pulse
  - Write `docker-compose.test.yml` for CI integration tests
  - Add WireMock stub files under `tests/wiremock/data-service/` and `tests/wiremock/sentinel-pulse/`
  - _Requirements: Req 17.1_

- [x] 11. Create CI/CD GitHub Actions workflow
  - Write `.github/workflows/ci.yml` with jobs: `lint-type-check` (ruff + mypy --strict), `test` (pytest --cov=src --cov-fail-under=90 --timeout=120 + benchmark run), `integration-test` (pytest test_api_contracts.py with docker-compose.test.yml services), `docker-build` (build + smoke test), `shadow-to-live-gate` (main branch only, requires PROMOTION_APPROVAL_TOKEN)
  - Write `.pre-commit-config.yaml` with ruff, ruff-format, mypy --strict hooks
  - _Requirements: Req 17.1, Req 18.1, Req 18.2_

---

### Phase 1: Data Client Layer

- [x] 12. Implement DataServiceClient (`src/clients/data_service.py`)
  - [x] 12.1 Implement REST client with auth and quality gates
    - `DataServiceClient` using `httpx.AsyncClient` with `X-API-KEY` header on every request
    - Implement `fetch_features(symbol, pit_date)` with 10s timeout
    - Gate 1: if response `signalEngineAllowed == false` → log structured warning (instrument, timestamp, DataConfidenceScore) and raise `SignalEngineNotAllowedError`
    - Gate 2: if response `DataConfidenceScore < settings.min_confidence_score` → raise `LowDataConfidenceError` with the score
    - Gate 3: if response has auth/authz error → log warning with credential type + instrument, raise `DataServiceAuthError` (no retry)
    - Return typed `MarketData` Pydantic model
    - _Requirements: Req 1.1, Req 1.3, Req 1.4, Req 1.5, Req 1.6, Req 1.7_

  - [x] 12.2 Implement circuit breaker and gRPC streaming
    - Wrap all REST calls with tenacity circuit breaker: 3 failures in 30s → circuit OPEN (fail fast for 60s)
    - Implement gRPC streaming client using generated stubs for live tick features with `grpc_connection_timeout=5s` and `grpc_per_call_deadline=2s`
    - Implement gRPC reconnect logic: up to 3 retries with 1s delay; after all fail → raise `DataServiceUnavailableError`
    - Validate `timeframe` parameter: only allow `1m, 5m, 10m, 15m, 30m, 1h, 1d, 1w, 1M`; never request `3m`
    - _Requirements: Req 1.8, Req 1.17, Req 1.18_

- [x] 13. Implement SentinelPulseClient (`src/clients/sentinel_pulse.py`)
  - Implement `SentinelPulseClient` using `httpx.AsyncClient` with Bearer auth header
  - Implement `fetch_news_context(instrument)` calling `GET /api/v1/alphaforge/news-context/:instrument` with 10s timeout
  - Validate required fields in response: `news_impact_score`, `impact_direction`, `impact_confidence`, and all `sentiment` sub-fields; treat partial responses as unreachable (return `None`)
  - Implement exponential backoff retry: up to 3 attempts, wait `[0.5s, 1.0s, 2.0s]`; on final failure return `None`
  - Implement LRU in-process cache: max 500 entries, TTL 90s (separate from Redis, as per design)
  - Implement PIT-correct training data methods: `fetch_pit_features(asset_id)`, `fetch_training_samples()`, `fetch_historical_reactions()` — calling `/api/v1/ml/features/asset/:assetId`, `/api/v1/ml/training/samples`, `/api/v1/ml/historical-reactions`
  - _Requirements: Req 1.2, Req 1.9, Req 1.10, Req 1.11, Req 1.13, Req 16.6_

---

### Phase 2: Feature Pipeline (TDD First)

- [x] 14. Write test_feature_pipeline_pit_correctness.py (TDD — write BEFORE FeaturePipeline implementation)
  - [x] 14.1 Write PIT correctness and idempotence property tests (Properties 1, 2)
    - Use Hypothesis `@given` with `pit_timestamps` strategy (weekday datetimes in trading hours 2020–2025)
    - Property 1: `∀ feature f ∈ vector: source_timestamp(f) < vector.timestamp` — for all generated feature vectors, assert no constituent feature has a future source timestamp
    - Property 2: calling `pipeline.build_vector(same_inputs_twice)` returns identical `FeatureVector` objects (idempotence)
    - Use `AsyncMock(spec=DataServiceClient)` and `AsyncMock(spec=SentinelPulseClient)` as mock fixtures
    - Verify these tests FAIL (red) before FeaturePipeline is implemented
    - _Requirements: Req 2.1, Req 2.10, Req 18.3_

  - [x] 14.2 Write DataConfidenceScore and signalEngineAllowed gate property tests (Properties 3, 4)
    - Property 3: `∀ score ∈ [0, 69]` → `pipeline.build_vector(...)` returns provenance `INSUFFICIENT_EVIDENCE`
    - Property 4: `signalEngineAllowed=false` → `build_vector()` returns provenance `UNAVAILABLE`
    - Property: `data-service2.0 unreachable` → `build_vector()` returns provenance `UNAVAILABLE` and does NOT fall back to any other source
    - Verify these tests FAIL (red) before FeaturePipeline is implemented
    - _Requirements: Req 1.5, Req 1.7, Req 1.12_

- [x] 15. Write test_qlib_feature_engineering.py (TDD — write BEFORE QlibFeatureEngine implementation)
  - Use Hypothesis `@given` with `ohlcv_dataframes` strategy (random OHLCV DataFrames with n_bars ∈ [252, 500])
  - Property 5a: `compute_alpha158(df, symbol) == compute_alpha158(df, symbol)` (idempotence) — same OHLCV produces identical 158-factor dict
  - Property 5b: all 158 output factor values are `math.isfinite(v)` — no NaN or Inf when input is complete
  - Property 5c: factor values are within documented ranges (assert no wildly out-of-range values using reasonable bounds)
  - Test that `compute_alpha360` produces exactly 360 keys when `alpha360_enabled=True`
  - Verify these tests FAIL (red) before QlibFeatureEngine is implemented
  - _Requirements: Req 2.3, Req 2.4, Req 18.5_

- [x] 16. Implement LeakageValidator (`src/features/leakage_validator.py`)
  - Implement `LeakageValidator.validate(feature_matrix: pd.DataFrame, label_vector: pd.Series) -> None`
  - Loop over look-ahead windows 1–22 trading days: shift labels by `-window`, compute Pearson correlation with each feature column, skip if fewer than 30 aligned samples
  - Raise `PITViolationError(feature, look_ahead_window, correlation)` if `abs(corr) > 0.05`
  - Log feature name, look-ahead window, and correlation value on every violation
  - _Requirements: Req 2.5, Req 18.3_

- [x] 17. Implement QlibFeatureEngine (`src/features/qlib_engine.py`)
  - Implement `QlibFeatureEngine` calling `qlib.init(provider_uri=...)` once at module import
  - Implement `compute_alpha158(ohlcv_df: pd.DataFrame, symbol: str) -> dict[str, float]` — requires ≥ 200 bars; returns dict with exactly 158 keys `alpha158_0..alpha158_157`
  - Implement `compute_alpha360(ohlcv_df: pd.DataFrame, symbol: str) -> dict[str, float]` — gated by `settings.alpha360_enabled`; returns 360 keys
  - Guarantee idempotence: no random state, no side effects between calls
  - Guarantee all outputs are finite for non-missing data
  - _Requirements: Req 2.3, Req 2.4, Req 5.3_

- [x] 18. Implement FeaturePipeline (`src/features/pipeline.py`)
  - [x] 18.1 Implement inference-mode vector assembly
    - Implement `FeaturePipeline.build_vector(symbol, timestamp, mode, pit_date)` following the pseudocode in the design exactly:
      1. Redis cache check (`feature:{symbol}:{bucket}`, TTL=60s)
      2. `DataServiceClient.fetch_features()` with 10s timeout; check `signalEngineAllowed` + `DataConfidenceScore` gates; on timeout/error return UNAVAILABLE
      3. PIT timestamp validation: log and skip any market data field whose `source_timestamp >= timestamp`
      4. `SentinelPulseClient.fetch_news_context()` with 10s timeout; on failure apply degraded-mode substitution (all numeric fields → 0.0, all directional fields → NEUTRAL); record in `FeatureQualityReport`
      5. `QlibFeatureEngine.compute_alpha158()` (and optionally alpha360)
      6. Assemble `FeatureVector` with `pit_validated=True`
      7. Cache result in Redis
      8. Assemble and return `FeatureQualityReport`
    - _Requirements: Req 1.11, Req 2.1, Req 2.6, Req 2.7, Req 2.8, Req 2.12_

  - [x] 18.2 Implement batch mode and backtest mode
    - Implement `build_batch(symbols, timestamp, mode)` — process all symbols concurrently with `asyncio.gather`; target 500 symbols in ≤ 60s under normal conditions
    - Implement backtest mode: include `?pit_date=YYYY-MM-DD` in all DataService calls; discard any data record with timestamp ≥ `pit_date`; record discarded count in `FeatureQualityReport`
    - Implement `GET /features/quality` endpoint returning the latest `FeatureQualityReport` with ≤ 2s p95 response time
    - _Requirements: Req 2.9, Req 2.11, Req 2.13, Req 15.6_

---

### Phase 3: Model Infrastructure (TDD First)

- [x] 19. Implement ModelRegistry (`src/registry/registry.py`)
  - Implement `ModelRegistry.register(artifact: ModelArtifact)` — write artifact metadata to `{MODEL_ARTIFACTS_PATH}/{model_name}/{version}/metadata.json`; compute SHA-256 of artifact file and write `{artifact_file}.sha256`; never overwrite an existing artifact path (raise `ArtifactAlreadyExistsError`)
  - Implement `get_champion(model_name: str) -> ModelArtifact | None`
  - Implement `list_registry() -> list[ModelArtifact]`
  - Implement `_load_artifact(path, artifact)`: compare computed SHA-256 against `artifact.sha256_checksum`; on mismatch log `ARTIFACT_INTEGRITY_FAILURE` event and raise `ArtifactIntegrityFailure`; calling code falls back to heuristic policy
  - Load all champion artifacts at FastAPI lifespan startup; set heuristic fallback for any model where artifact loading fails without crashing the service
  - _Requirements: Req 17.4, Req 13.11_

- [x] 20. Implement ModelExplainer (`src/explainability/explainer.py`)
  - Implement `ModelExplainer` with in-process LRU cache of `shap.TreeExplainer` / `shap.KernelExplainer` instances per model (lazy-loaded)
  - Use `shap.TreeExplainer` for XGBoost, LightGBM, CatBoost models
  - Use `shap.KernelExplainer` with `shap.maskers.Independent` for neural network models (PriceForecaster, IVRegimeClassifier)
  - For `HEURISTIC` predictions: return rule attribution listing which rule conditions fired and their contribution to the final score
  - On SHAP computation failure: log error with model name + feature dict; return `ExplainResponse` with all contributions = 0.0 and `base_value` = historical mean prediction (never return 500)
  - Implement `POST /v2/explain/{model_name}` endpoint accepting `ExplainRequest`, returning `ExplainResponse`
  - _Requirements: Req 11.1, Req 11.2, Req 11.3, Req 11.4, Req 11.6_

---

### Phase 4: Core Prediction Models (TDD First)

- [x] 21. Write test_signal_generation_latency.py (TDD — write BEFORE model implementations)
  - Use `pytest-benchmark` and `time.perf_counter()` to measure 100 consecutive calls per endpoint
  - Parametrize over all single-symbol endpoints: `/v2/predict/regime`, `/v2/predict/strategy`, `/v2/predict/risk`, `/v2/predict/execution`, `/v2/predict/price-regime`, `/v2/predict/iv-regime`
  - Assert p95 latency ≤ 50ms for all single-symbol endpoints
  - Assert p95 latency ≤ 200ms for `/v2/predict/rankings` (200 symbols)
  - Assert p95 latency ≤ 150ms for `/v2/meta/decide`
  - Assert p95 latency ≤ 500ms for `/v2/predict/portfolio-v2` (50 assets)
  - Assert p95 latency ≤ 100ms for all `/v2/analytics/*` endpoints
  - Verify these tests FAIL (red) before models are implemented (using stub 500ms sleep models)
  - _Requirements: Req 16.1, Req 16.2, Req 16.3, Req 16.4, Req 16.5, Req 18.8_

- [x] 22. Implement RegimeClassifier (`src/models/regime_classifier.py`) and endpoint
  - [x] 22.1 Implement RegimeClassifier core
    - Load trained XGBoost champion artifact via `ModelRegistry.get_champion("market_regime")`; if no artifact fall back to rule-based heuristic (`PredictionProvenance.HEURISTIC`)
    - Feature groups: NIFTY/BANKNIFTY price features (ATR, ADX, RSI, MACD), India VIX level + rate-of-change, market breadth (advance/decline ratio, % F&O stocks above 20 SMA), FII/DII net flow in crores, put-call ratio, SentinelPulse `market_regime`, Qlib Alpha158 cross-sectional breadth factors — all with max 5-minute data age check
    - Classify into exactly six regimes with confidence ∈ [0,1] and probabilities summing to 1.0
    - Confidence < 0.35 → set `PredictionProvenance.INSUFFICIENT_EVIDENCE` + reason code `LOW_REGIME_CONFIDENCE`
    - On consecutive different regime labels: emit structured `regime_transition_event` log entry (previous regime, new regime, confidence delta, top-3 contributing features)
    - _Requirements: Req 4.1, Req 4.2, Req 4.3, Req 4.6, Req 4.7_

  - [x] 22.2 Implement RegimeClassifier SHAP + online learning trigger
    - Attach top-10 SHAP features to every prediction response (direction, raw value, SHAP contribution)
    - Implement IC degradation detection hook: when `PerformanceMonitor` detects > 20% IC degradation from 90-day baseline over 30-day rolling window → trigger `OnlineLearner.initiate_update("market_regime", last_60_trading_days)`; continue serving from existing champion during update
    - _Requirements: Req 4.5, Req 4.8, Req 4.9, Req 11.1_

  - [x] 22.3 Implement `POST /v2/predict/regime` endpoint
    - Wire `RegimePredictionRequest` → `FeaturePipeline.build_vector()` → `RegimeClassifier.predict()` → `ModelExplainer.explain()` → `RegimePredictionResponse`
    - Respond within 50ms at p95
    - _Requirements: Req 4.4, Req 15.4_

- [x] 23. Implement StockRanker (`src/models/stock_ranker.py`) and endpoint
  - [x] 23.1 Implement StockRanker core
    - Load trained LightGBM champion artifact; if no artifact fall back to score-based heuristic (`PredictionProvenance.HEURISTIC`)
    - Feature groups: existing stock features from alpha-forge + Qlib Alpha360 factors + SentinelPulse `news_impact_score` + `impact_direction` + delivery % + OI build-up score + IV rank percentile + sector RS vs NIFTY
    - Produce outperformance score ∈ [0, 100] and rank for each symbol relative to the current regime
    - Enforce idempotence: same feature vector + same loaded model version → same rank + same score (Property 11)
    - When a symbol's SentinelPulse news context is unavailable: proceed with remaining features + record in per-symbol `FeatureQualityReport` field
    - _Requirements: Req 5.1, Req 5.2, Req 5.3, Req 5.6, Req 5.7_

  - [x] 23.2 Implement regime-conditioned weights and SHAP decomposition
    - Implement regime-conditioned ranking: load regime-specific weight tables stored in model artifact; select weights based on `regime` input field
    - Return SHAP-based top-5 factor decomposition per ranked symbol
    - _Requirements: Req 5.4, Req 5.5, Req 5.8_

  - [x] 23.3 Implement `POST /v2/predict/rankings` endpoint
    - Accept `RankingRequest` (up to 200 symbols); call `FeaturePipeline.build_batch()` then `StockRanker.rank_batch()`
    - Respond within 200ms at p95 for 200-symbol batch
    - _Requirements: Req 5.4, Req 15.4_

- [x] 24. Implement StrategySelector (`src/models/strategy_selector.py`) and endpoint
  - [x] 24.1 Implement StrategySelector core
    - Load trained CatBoost champion artifact; if no artifact fall back to rule-based heuristic (`PredictionProvenance.HEURISTIC`)
    - Accept `IV_regime` as mandatory input feature (`CRUSH` | `STABLE` | `SPIKE`); raise 422 if absent
    - Classify into exactly 8 strategies with calibrated probabilities (isotonic regression calibration trained on OOS fold predictions)
    - When top-strategy confidence < 0.40: include at least 2 alternative strategies in `alternatives` field
    - _Requirements: Req 6.1, Req 6.2, Req 6.3, Req 6.5, Req 6.6_

  - [x] 24.2 Implement `POST /v2/predict/strategy` endpoint
    - Wire `StrategyRequest` → feature pipeline → `StrategySelector.predict()` → `StrategyResponse`
    - Respond within 50ms at p95
    - _Requirements: Req 6.4, Req 15.4_

- [x] 25. Implement RiskPredictor (`src/models/risk_predictor.py`) and endpoint
  - [x] 25.1 Implement RiskPredictor core
    - Load XGBoost ensemble of three trained models (stop_model, target_model, drawdown_model); if no artifacts fall back to rule-based (`PredictionProvenance.HEURISTIC`)
    - Derive `stop_distance_atr`, `target_distance_atr`, `risk_reward_ratio` from trade geometry inputs before inference
    - Feature set: `regime_encoded`, `vix_regime`, `atr_pct`, `oi_buildup_score`, SentinelPulse `news_impact_score`, SentinelPulse `sentiment.risk`
    - Ensemble averaging across three models for all outputs
    - _Requirements: Req 7.2, Req 7.3, Req 7.4_

  - [x] 25.2 Implement probability invariant and HIGH_RISK_BLOCKED
    - Enforce `prob_stop_hit + prob_target_hit ≤ 1.0` (Property 8): log `CALIBRATION_VIOLATION` warning before clamping; never silently violate invariant
    - When `risk_score > 7.0` AND `DeploymentMode == VALIDATED_ML_ONLY`: append reason code `HIGH_RISK_BLOCKED` to response
    - Attach SHAP factor breakdown to every response
    - _Requirements: Req 7.6, Req 7.7, Req 11.1_

  - [x] 25.3 Implement `POST /v2/predict/risk` endpoint
    - Wire `RiskRequest` → `RiskPredictor.predict()` → `RiskResponse`
    - Respond within 50ms at p95
    - _Requirements: Req 7.5, Req 15.4_

- [x] 26. Implement PortfolioOptimizer (`src/models/portfolio_optimizer.py`) and endpoints
  - [x] 26.1 Implement Riskfolio-Lib optimization methods
    - Implement HRP using `riskfolio.HRPOpt` with Ward linkage clustering; return weights summing to 1.0 with all weights ≥ 0
    - Implement CVaR-minimized MVO using `riskfolio.RiskFolio` with configurable tail probability `alpha` (default 0.05)
    - Implement Equal Risk Contribution (ERC)
    - Implement Maximum Diversification
    - All methods: return `expected_return`, `portfolio_volatility`, `Sharpe_ratio`, `CVaR`, `max_drawdown_estimate`, `diversification_ratio`
    - Return `available: false` + reason `INSUFFICIENT_RETURN_HISTORY` when fewer than 20 observations per asset
    - Determinism: same inputs + same random seed → identical weights (Property 10); fix seeds in all solvers
    - _Requirements: Req 8.1, Req 8.2, Req 8.3, Req 8.5, Req 8.7, Req 8.8_

  - [x] 26.2 Implement sector weight constraint and legacy endpoint
    - Enforce configurable max sector weight constraint (default 40%): if violated, re-run optimization with binding sector constraint; raise if still violated after re-run
    - Implement `PredictionProvenance` assignment: Riskfolio-Lib available + sufficient data → `TRAINED_MODEL`; equal-weight fallback → `HEURISTIC`
    - Implement legacy `POST /v2/predict/portfolio` using PyPortfolioOpt (backward-compatible with alpha-forge)
    - Implement `POST /v2/predict/portfolio-v2` using Riskfolio-Lib (all four methods, method selector in request body)
    - Both endpoints respond within 500ms at p95 for ≤ 50 assets
    - _Requirements: Req 8.4, Req 8.6, Req 8.9, Req 15.4_

- [x] 27. Write test_portfolio_optimization.py (TDD — verify after implementation)
  - [x]* 27.1 Write Hypothesis property tests for HRP invariants (Properties 9, 10)
    - Property 9a: `sum(weights) == 1.0` (within 1e-9) for any valid returns matrix with 2+ assets, 20+ observations
    - Property 9b: `all(w >= 0 for w in weights)` for any valid returns matrix
    - Property 9c: `portfolio_volatility(HRP(R)) ≤ max_asset_volatility(R)` (diversification invariant)
    - Property 10: determinism — same inputs + same seed → identical weights twice (diff < 1e-12)
    - Hypothesis strategies: `n_assets = st.integers(2, 30)`, `n_obs = st.integers(20, 252)`, random returns via `st.randoms()`
    - _Requirements: Req 8.2, Req 8.7, Req 18.11_

- [x] 28. Implement PriceForecaster (`src/models/price_forecaster.py`) and endpoint
  - Migrate and upgrade TFT-based price forecaster from alpha-forge `src/price_forecaster.py`
  - Add `PredictionProvenance` field to response; no trained artifact → `HEURISTIC`
  - Integrate with `ModelRegistry` for champion artifact loading
  - Implement `POST /v2/predict/price-regime` endpoint; respond within 50ms at p95
  - _Requirements: Req 15.4, Req 17.4_

- [x] 29. Implement IVRegimeClassifier (`src/models/iv_regime_classifier.py`) and endpoint
  - Migrate and upgrade PatchTST-based IV regime classifier from alpha-forge `src/iv_regime_classifier.py`
  - Classify into `CRUSH | STABLE | SPIKE` with confidence; no trained artifact → `HEURISTIC`
  - Integrate with `ModelRegistry`
  - Implement `POST /v2/predict/iv-regime` endpoint; respond within 50ms at p95
  - This classifier's output is a mandatory input feature for `StrategySelector` (Req 6.3)
  - _Requirements: Req 6.3, Req 15.4, Req 17.4_

---

### Phase 5: Analytics Endpoints

- [x] 30. Implement Greeks computation and endpoint (`src/analytics/greeks.py`)
  - Implement Black-Scholes (equity options) and Black-76 (futures options) models for delta, gamma, theta, vega, rho
  - Migrate from alpha-forge `src/greeks.py` with upgrade to Pydantic V2 types
  - Implement `POST /v2/analytics/greeks` endpoint; respond within 100ms at p95
  - _Requirements: Req 15.5_

- [x] 31. Implement GEX computation and endpoint (`src/analytics/gex.py`)
  - Implement Dealer Gamma Exposure calculation: aggregate dealer gamma across all strikes from option chain snapshot
  - Migrate from alpha-forge `src/gex.py` with upgrade to Pydantic V2 types
  - Implement `POST /v2/analytics/gex` endpoint; respond within 100ms at p95
  - _Requirements: Req 15.5_

- [x] 32. Implement VPIN computation and endpoint (`src/analytics/vpin.py`)
  - Implement Volume-Synchronized Probability of Informed Trading using tick volume buckets
  - Migrate from alpha-forge tests/test_vpin.py reference implementation
  - Implement `POST /v2/analytics/vpin` endpoint; respond within 100ms at p95
  - _Requirements: Req 15.5_

- [x] 33. Implement Vol Surface computation and endpoint (`src/analytics/vol_surface.py`)
  - Implement SVI (Stochastic Volatility Inspired) parametric model for fitting implied volatility smiles
  - Implement term structure of implied volatility
  - Migrate from alpha-forge `src/vol_surface.py`
  - Implement `POST /v2/analytics/vol-surface` endpoint; respond within 100ms at p95
  - _Requirements: Req 15.5_

---

### Phase 6: FinRL Execution Agent (TDD First)

- [x] 34. Write test_finrl_execution_agent.py (TDD — write BEFORE RLExecutionAgent implementation)
  - Test action space is exactly the 7-element set `{ENTER_NOW, WAIT, SCALE_IN, PARTIAL_EXIT, FULL_EXIT, TIGHTEN_STOP, TRAIL_STOP}` and no other values are ever returned
  - Integration test: 100 consecutive calls using a loaded artifact, assert p95 latency ≤ 50ms using `time.perf_counter()`
  - Test heuristic fallback: when `ModelRegistry.get_champion("rl_execution_agent")` returns `None`, response must have `provenance == HEURISTIC`
  - Test SLA breach fallback: inject artificial 60ms delay in model inference; assert response has `provenance == HEURISTIC`
  - Verify these tests FAIL (red) before implementation
  - _Requirements: Req 9.2, Req 9.7, Req 9.8, Req 18.6_

- [x] 35. Implement RLExecutionAgent (`src/models/rl_execution_agent.py`)
  - [x] 35.1 Implement execution agent core
    - Load FinRL-X PPO or SAC trained artifact (algorithm configured via training pipeline config); no artifact → rule-based fallback (`HEURISTIC`)
    - Observation space: unrealized P&L %, time in trade (min), regime (encoded), volume ratio, price vs VWAP, ATR, momentum, IV regime (encoded), SentinelPulse `news_impact_score`, current risk score from `RiskPredictor`
    - Implement 7-action discrete action space exactly
    - Implement `act(state: ExecutionState) -> ExecutionDecision` following the pseudocode in the design: encode observation → `policy.predict(obs, deterministic=True)` → measure latency → if > 50ms fall back to rule-based
    - Implement OPE (Offline Policy Evaluation) with importance sampling on last 63 trading days of logged data; produce estimated Sharpe ratio
    - _Requirements: Req 9.1, Req 9.2, Req 9.3, Req 9.6, Req 9.7, Req 9.8, Req 9.9_

  - [x] 35.2 Implement rule-based policy fallback
    - Migrate rule-based execution policy from alpha-forge `src/rl/` as `_rule_based_fallback(state, provenance)` within `RLExecutionAgent`
    - Used when: no artifact, SLA breach, inference exception
    - Always sets `PredictionProvenance.HEURISTIC`
    - _Requirements: Req 9.7, Req 9.8_

- [x] 36. Implement `POST /v2/predict/execution` endpoint
  - Wire `ExecutionState` → `RLExecutionAgent.act()` → `ExecutionDecision`
  - Include `PredictionProvenance` in response
  - Respond within 50ms at p95
  - _Requirements: Req 9.8, Req 15.4_

---

### Phase 7: Training Pipeline (TDD First)

- [x] 37. Write test_model_training_purged_cv.py (TDD — write BEFORE PurgedKFoldSplitter implementation)
  - Use Hypothesis `@given(ts_length=st.integers(100, 1000), embargo_days=st.integers(5, 20))`
  - Property 7: for any split, assert that NO training sample's timestamp is within `embargo_days` trading days of any validation fold boundary — zero temporal overlap invariant
  - Property: `len(purged_train_fold) < len(full_train_fold)` when embargo is non-zero (samples are removed)
  - Use `pd.date_range(..., freq="B")` (business days) as the timestamp series
  - Verify these tests FAIL (red) before `PurgedKFoldSplitter` is implemented
  - _Requirements: Req 3.1, Req 3.2, Req 18.9_

- [x] 38. Implement PurgedKFoldSplitter (`src/training/purged_kfold.py`)
  - Implement `PurgedKFoldSplitter.split(X, y, timestamps, n_splits, embargo_days)` following the pseudocode in the design exactly
  - Embargo: exclude training samples within `embargo_days` calendar days of validation fold start/end boundaries
  - Purge: remove training samples whose label observation window (configurable `label_window_days`, default 5) overlaps any validation sample's feature observation window
  - Implement CPCV path generation (`src/training/cpcv.py`): generate at least `settings.cpcv_n_paths` overlapping test paths from the K folds; compute backtest overfitting probability (PBO) as fraction of paths with negative net return
  - _Requirements: Req 3.1, Req 3.2, Req 3.5_

- [x] 39. Implement TrainingPipeline core (`src/training/pipeline.py`)
  - [x] 39.1 Implement model training orchestration
    - Implement `TrainingPipeline.run(config: TrainingConfig) -> TrainingRun` as an async background task
    - Step 1: `FeaturePipeline.build_training_dataset(pit_date, symbols)` using PIT-correct mode
    - Step 2: `LeakageValidator.validate(feature_matrix, labels)` — abort with `TrainingAbortedError` on violation
    - Step 3: `PurgedKFoldSplitter.split()` with configured embargo
    - Step 4: per-fold Qlib Alpha158/360 computation
    - Step 5: Optuna HPO (50+ trials, objective = mean Spearman IC across held-out folds)
    - Step 6: CPCV (10+ paths, PBO computation)
    - Step 7: Model acceptance gating: IC ≥ 0.02 AND net Sharpe ≥ 0.0 AND PBO ≤ 0.5; on gate failure → archive with REJECTED status + rejection reason
    - _Requirements: Req 3.1, Req 3.2, Req 3.3, Req 3.4, Req 3.5, Req 3.6, Req 3.8_

  - [x] 39.2 Implement MLflow integration and audit log
    - Log every experiment run to MLflow: model name, version, training date range, validation date range, all hyperparameters, IC per fold, net Sharpe, max drawdown, PBO, SHA-256 hash of training dataset
    - Write every training run to the append-only JSON Lines audit log via `AuditLogger`
    - Enforce model SDLC: `HYPOTHESIS → BACKTEST → CHALLENGER → SHADOW → APPROVED → PRODUCTION` — no model advances without all required prior-stage gates returning PASS
    - Shadow trading minimum: model must spend ≥ `settings.shadow_trading_min_days` in SHADOW before APPROVED promotion
    - _Requirements: Req 3.7, Req 3.9, Req 3.10, Req 3.11_

- [x] 40. Implement Optuna HPO integration (`src/training/hpo.py`)
  - Implement `OptunaHPO.optimize(model_builder, X, y, timestamps, n_trials)` using TPE sampler + MedianPruner
  - Objective = mean Spearman IC across all held-out folds
  - Integrate `optuna_integration.MLflowCallback` to log every trial to MLflow automatically
  - Persist Optuna study to `sqlite:///optuna.db` to survive restarts
  - _Requirements: Req 3.6, Req 3.7_

- [x] 41. Implement training API endpoints
  - Implement `POST /training/run` accepting `TrainingConfig`; dispatch async background task; return initial `TrainingRun` status
  - Implement `GET /training/status/{run_id}` returning current `TrainingRunStatus`
  - _Requirements: Req 15.6_

---

### Phase 8: Model Promotion & Lifecycle (TDD First)

- [x] 42. Implement ModelPromotion six-gate pipeline (`src/registry/promotion.py`)
  - [x] 42.1 Implement DATA and PREDICTIVE gates
    - DATA gate: FAIL if `final_oos_used_for_selection == true` OR integrity verification fails OR OOS observation count < 60 (one ticker–date pair = one observation) OR evidence level < LEVEL_B OR evidence level is LEVEL_C or LEVEL_D
    - PREDICTIVE gate: with champion → PASS only if challenger Rank IC > champion IC + `settings.predictive_gate_margin`; without champion → PASS only if challenger Rank IC > 0
    - Mark challenger `FINAL_OOS_CONTAMINATED` permanently and block all promotions if final OOS data was read during any gate evaluation
    - _Requirements: Req 13.3, Req 13.4, Req 13.12, Req 13.13_

  - [x] 42.2 Implement CALIBRATION, EXECUTION, RISK, STABILITY gates
    - CALIBRATION gate: with champion → challenger Brier score ≤ champion Brier + 0.01; without → challenger Brier < 0.25
    - EXECUTION gate: challenger net return > 0 on held-out partition; with champion → challenger turnover increase ≤ 20%
    - RISK gate: with champion → challenger max drawdown ≤ champion max drawdown + 2pp; without → challenger max drawdown ≤ 20%
    - STABILITY gate: FAIL if IC decay status is FAILED/SIGNIFICANT_DECAY or feature drift severity is HIGH/CRITICAL
    - _Requirements: Req 13.5, Req 13.6, Req 13.7, Req 13.8_

  - [x] 42.3 Implement promotion decision assembly and lifecycle enforcement
    - Implement `evaluate_gates(challenger, champion) -> PromotionDecision` following the pseudocode in the design
    - All PASS → outcome=PROMOTE, policy=HUMAN_APPROVAL_REQUIRED
    - Any FAIL → outcome=REJECTED
    - Any INSUFFICIENT_EVIDENCE → outcome=BLOCKED (record specific gating gates in audit log)
    - Record every promotion decision in the append-only audit log with all required fields (timestamp, challenger_id, champion_id, gate results, outcome, approval_policy, reviewer_identity)
    - Enforce linear HYPOTHESIS → BACKTEST → CHALLENGER → SHADOW → APPROVED → PRODUCTION lifecycle; block any out-of-order advancement
    - _Requirements: Req 13.1, Req 13.2, Req 13.9, Req 13.10, Req 13.14_

- [x] 43. Implement approvalToken validation
  - Implement `ApprovalTokenValidator.validate(token: str) -> ReviewerIdentity`
  - Token must be issued by an identity in the configured authorized-reviewer registry (env-var list)
  - Token must not be expired (< 24 hours since issuance based on `settings.approval_token_expiry_hours`)
  - Invalid or expired token → raise `InvalidApprovalTokenError`; block `ModelRegistry.register_champion()` call
  - Record reviewer identity in promotion audit log entry when approval is provided
  - _Requirements: Req 13.11_

- [x] 44. Implement model status and registry endpoints
  - Implement `GET /models/status` returning per-model loaded status, version, and `has_trained_model` boolean
  - Implement `GET /models/registry` returning `list[ModelArtifact]` for all registered artifacts
  - Both endpoints require `X-API-KEY` authentication
  - _Requirements: Req 15.6_

---

### Phase 9: Online Learning (TDD First)

- [x] 45. Write test_online_learning.py (TDD — write BEFORE OnlineLearner implementation)
  - Use Hypothesis `@given(n_updates=st.integers(1, 5))`
  - Property 18: each incremental update produces a new version string strictly different from all previous versions for that model — assert `version_after != version_before` for all updates
  - Property 19: after any number of updates, the prior artifact file must still exist on disk — `Path(prior_artifact.artifact_path).exists() == True`
  - Property 17: mock a sequence of n=6+ updates; assert that after 5 consecutive updates the learner triggers a full retrain signal and does NOT apply a 6th incremental update
  - Verify these tests FAIL (red) before `OnlineLearner` is implemented
  - _Requirements: Req 14.4, Req 14.6, Req 18.12_

- [x] 46. Implement OnlineLearner (`src/training/online_learner.py`)
  - [x] 46.1 Implement incremental update logic
    - Implement `OnlineLearner.update(model_name, new_data) -> str | None` following the pseudocode in the design exactly
    - Guard 1: model must have `PredictionProvenance.TRAINED_MODEL`; heuristics are ineligible
    - Guard 2: `artifact.consecutive_online_updates < settings.max_consecutive_online_updates` (default 5); on limit → call `_trigger_full_retrain()` and return None
    - XGBoost: call `model.update(X, y)` on regime classifier and risk predictor
    - LightGBM: `lgb.train(params, new_dataset, init_model=model, num_boost_round=10)` for stock ranker
    - Create new versioned artifact: `{version}-online-{YYYY-MM-DD}` — never overwrite prior artifact
    - _Requirements: Req 14.1, Req 14.2, Req 14.3, Req 14.4, Req 14.6_

  - [x] 46.2 Implement validation window and audit logging
    - Validate updated model on last 10 trading days of held-out data before activation
    - If new IC < prior IC: discard update, log `ONLINE_UPDATE_REJECTED` event to AuditLogger, return None
    - If new IC ≥ prior IC: `ModelRegistry.register(new_artifact)`, increment `consecutive_online_updates`, log update to AuditLogger, return new version string
    - Continue serving from existing champion artifact during the update window
    - _Requirements: Req 14.5, Req 14.7_

---

### Phase 10: Meta-Decision Engine (TDD First)

- [x] 47. Write test_meta_decision_engine.py (TDD — write BEFORE MetaDecisionEngine implementation)
  - [x] 47.1 Write absorbing identity and idempotence property tests (Properties 12, 13)
    - Property 12 (absorbing identity): use Hypothesis `st.fixed_dictionaries` where all model outputs have `provenance=UNAVAILABLE`; assert `meta_decide(inputs).action == "NO_TRADE"` AND `meta_decide(inputs).provenance == UNAVAILABLE`
    - Property 13 (idempotence): for any `meta_inputs`, assert `meta_decide(x).action == meta_decide(x).action` and `abs(meta_decide(x).confidence - meta_decide(x).confidence) < 1e-9`
    - Verify these tests FAIL (red) before implementation
    - _Requirements: Req 10.12, Req 17.3, Req 18.4_

  - [x] 47.2 Write abstention invariant property test (Property 14)
    - Property 14: generate model outputs where `agreement_ratio < 0.5` (fewer than half of available models agree on direction); assert `meta_decide(inputs).action in {"WAIT", "NO_TRADE"}` AND `meta_decide(inputs).abstention == True`
    - Also test confidence + uncertainty = 1.0 for all valid inputs
    - Verify these tests FAIL (red) before implementation
    - _Requirements: Req 10.4, Req 10.7, Req 18.4_

- [x] 48. Implement CalibrationLayer (`src/meta/calibration.py`)
  - Implement `CalibrationLayer` supporting both Platt scaling (`sklearn.linear_model.LogisticRegression`) and isotonic regression (`sklearn.isotonic.IsotonicRegression`)
  - Select method with lower ECE (Expected Calibration Error) on held-out OOS fold predictions
  - If resulting ECE > 0.15: reject calibration update, retain previous calibrator, log rejection
  - Implement `POST /v2/meta/fit` endpoint: accept list of ≥ 30 `OOSRecord` objects; retrain meta-layer calibrators without touching base models; reject with error if < 30 records or missing required fields
  - _Requirements: Req 10.2, Req 10.11_

- [x] 49. Implement EnsembleWeighter (`src/meta/ensemble.py`)
  - Implement `EnsembleWeighter.compute_weights(model_names, regime) -> dict[str, float]`
  - Weights proportional to IC in current regime; loaded from per-regime IC tables stored in model artifacts
  - Hard constraints: no single model weight > 0.40, no model weight < 0.05, all weights sum to 1.0
  - Models with IC < 0.05 in current regime receive minimum weight 0.05
  - _Requirements: Req 10.3_

- [x] 50. Implement AbstentionPolicy (`src/meta/abstention.py`)
  - Implement `AbstentionPolicy.should_abstain(available_models, agreement_ratio, data_quality, mean_confidence, prob_stop_hit) -> bool`
  - Return True (→ NO_TRADE) when ANY of: `agreement_ratio < 0.5`, `data_quality < 0.6`, `mean_confidence < 0.35`, `prob_stop_hit > 0.65`, `len(available_models) < 3`
  - _Requirements: Req 10.7_

- [x] 51. Implement LLMNewsReasoner (`src/meta/llm_reasoner.py`)
  - Load FinBERT (`ProsusAI/finbert`) or FinGPT (configurable via `settings.llm_model_name`) via LangChain's `AsyncLLMChain`
  - Implement `reason(news_context, symbol) -> NewsSignal` with `settings.llm_inference_timeout_ms` (default 120ms) timeout
  - On timeout: check `llm_news:{symbol}` Redis key (60s TTL); if cached → return cached signal + add `NEWS_CACHE_FALLBACK` reason code
  - If no cached signal: return `NewsSignal(direction=0, confidence=0.0, rationale="")` + add `NEWS_TIMEOUT` reason code
  - Cache successful LLM result to `llm_news:{symbol}` Redis key
  - _Requirements: Req 10.5, Req 10.9_

- [x] 52. Implement ConfidenceDecomposer (`src/meta/calibration.py`)
  - Implement `ConfidenceDecomposer.decompose(available_models, calibrators, weights, regime, data_quality) -> ConfidenceDecomposition`
  - Five components: `base_confidence` = weighted ensemble score, `calibration_quality` = 1 - mean(ECE across calibrators), `agreement_bonus` = agreement_ratio × 0.1, `data_quality_factor` = data_confidence_score / 100, `regime_confidence_factor` = regime-specific weight value
  - _Requirements: Req 10.8_

- [x] 53. Implement MetaDecisionEngine (`src/meta/engine.py`)
  - [x] 53.1 Implement engine core
    - Implement `MetaDecisionEngine.decide(model_outputs, news_context, regime, symbol) -> MetaOutput` following the pseudocode in the design step-by-step (filter unavailable → absorbing identity check → VALIDATED_ML_ONLY suppression → calibrate → weight → agreement_ratio → LLM reasoning → AbstentionPolicy → NEWS_CONFLICT_OVERRIDE → action determination → decompose confidence → assemble MetaOutput)
    - NEWS_CONFLICT_OVERRIDE: when `news_impact_score > 0.7` AND `impact_direction` conflicts with ensemble direction → return `NO_TRADE` + `NEWS_CONFLICT_OVERRIDE` reason code
    - VALIDATED_ML_ONLY: convert all `HEURISTIC` model contributions to direction=0 before ensemble computation
    - Respond within 150ms at p95 with cached news context
    - _Requirements: Req 10.1, Req 10.3, Req 10.4, Req 10.5, Req 10.6, Req 10.7, Req 10.9, Req 10.10, Req 10.12_

  - [x] 53.2 Implement XAI explainability assembly
    - Populate `MetaOutput.explainability`: top-5 contributing features across all models (from SHAP), list of models with conflicting signals, one-sentence LLM-generated natural-language rationale
    - _Requirements: Req 11.5_

- [x] 54. Implement `POST /v2/meta/decide` endpoint
  - Wire `MetaDecideRequest` → `FeaturePipeline.build_vector()` → all base models → `MetaDecisionEngine.decide()` → `MetaOutput` response
  - `X-Request-ID` traced through entire call chain
  - _Requirements: Req 10.9, Req 15.4_

---

### Phase 11: Drift Monitoring (TDD First)

- [x] 55. Write test_drift_monitoring.py (TDD — write BEFORE DriftMonitor implementation)
  - Use Hypothesis `@given(st.lists(st.floats(-10, 10, allow_nan=False), min_size=100))`
  - Property 15 (PSI zero-drift identity): `compute_psi(D, D) < 0.001` for any distribution D — identical distributions have near-zero PSI
  - Use `@given(distribution, shift=st.floats(1.0, 5.0))`
  - Property 16 (PSI monotonicity): `compute_psi(ref, shifted) > compute_psi(ref, ref)` for any non-zero shift — drift detection is monotone
  - Verify these tests FAIL (red) before DriftMonitor is implemented
  - _Requirements: Req 12.8, Req 18.7_

- [x] 56. Implement DriftMonitor (`src/monitoring/drift_monitor.py`)
  - [x] 56.1 Implement PSI computation and Evidently AI integration
    - Implement `compute_psi(reference, current, n_bins=10) -> float` following the exact formula from the design: `PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)` using 10 equal-frequency bins from reference distribution; add 1e-6 epsilon to avoid log(0); return `max(psi, 0.0)`
    - Integrate Evidently AI `DataDriftPreset`: compare 7-day rolling production feature distribution against training reference distribution for every model
    - PSI > 0.2 and ≤ 0.25 → emit `DRIFT_ALERT` severity MEDIUM
    - PSI > 0.25 → emit `DRIFT_ALERT` severity HIGH + trigger `OnlineLearner` if `settings.online_learning_enabled`
    - _Requirements: Req 12.1, Req 12.2, Req 12.3, Req 12.8_

  - [x] 56.2 Implement NannyML CBPE integration and scheduling
    - Integrate NannyML `CBPE` with `PeriodBasedChunker(period='daily')` to estimate IC + ECE for each model on unlabeled production data
    - estimated IC < 0.015 → emit `PERFORMANCE_DEGRADATION_ALERT` + block model from MetaDecisionEngine contributions until cleared
    - Schedule `run_daily()` at 00:00 UTC using APScheduler or similar
    - _Requirements: Req 12.4, Req 12.5_

- [x] 57. Implement alert emission and monitoring endpoints
  - Implement `AlertSystem` emitting structured `DriftAlert` objects via structlog in a format consumable by monitoring routers
  - Include recommended action in alert payload: PSI ≤ 0.25 → `MONITOR`; PSI > 0.25, learning not triggered → `RETRAIN`; PSI > 0.25, learning triggered, IC still < 0.015 → `ROLLBACK`
  - Implement `GET /monitoring/drift` returning latest drift report for all models
  - Implement `GET /monitoring/performance` returning latest NannyML CBPE estimates + ECE scores
  - Implement `GET /monitoring/alerts` returning `list[DriftAlert]` (active alerts)
  - Implement `POST /monitoring/performance/{model_name}/clear`: remove `PERFORMANCE_DEGRADATION_ALERT` block, log clear event (operator identity + timestamp), re-enable model contribution to MetaDecisionEngine
  - _Requirements: Req 12.6, Req 12.7, Req 12.9, Req 12.10, Req 15.6_

---

### Phase 12: Signal Streaming

- [x] 58. Implement SignalStreamer and WebSocket endpoint (`src/streaming/streamer.py`)
  - Implement `SignalStreamer` managing WebSocket connections at `WS /v2/stream/signals`
  - Authenticate via `?api_key=...` query parameter on WebSocket upgrade
  - On new `MetaDecisionEngine` output: assemble `SignalEvent` (event_type="signal", symbol, timestamp, meta_output, feature_quality) and broadcast to all connected clients for that symbol
  - Handle client disconnection gracefully (close stream, no unhandled exceptions, clean up connection from registry)
  - Support multiple concurrent WebSocket clients without interference
  - _Requirements: Req 15.7_

---

### Phase 13: Integration & End-to-End Testing

- [x] 59. Implement full integration test environment
  - [x] 59.1 Set up docker-compose integration environment and conftest
    - Write `tests/conftest.py` with shared fixtures: async `httpx.AsyncClient` pointed at running ml-service2.0, mock DataServiceClient and SentinelPulseClient, mock `ModelRegistry` with and without champion artifacts, mock Redis, sample `FeatureVector` factory functions, sample model output factory functions
    - Verify docker-compose.test.yml starts all services cleanly
    - _Requirements: Req 18.10_

  - [x] 59.2 Complete test_api_contracts.py integration tests
    - Integration tests for ALL v2 endpoints using `httpx.AsyncClient`: verify correct HTTP status codes (200, 401, 422, 503), correct response schema shape, `provenance` field present and valid on every prediction response
    - Test `POST /v2/meta/fit` with < 30 records → 422, with missing required fields → 422, with valid 30+ records → 200
    - Test WebSocket `WS /v2/stream/signals`: connect, receive `SignalEvent`, disconnect gracefully
    - _Requirements: Req 15.1, Req 15.2, Req 15.3, Req 15.4, Req 15.6, Req 15.7, Req 15.8, Req 18.10_

- [x] 60. End-to-end inference flow integration test
  - Write `tests/test_e2e_inference.py` covering the full inference path: WireMock-stubbed DataService returns valid market features → SentinelPulse returns news context → FeaturePipeline assembles `FeatureVector` → all six base models run → SHAP explanations attached → MetaDecisionEngine produces `MetaOutput` → response includes `provenance`, `explainability`, `decomposition`, `news_sentiment_signal`
  - Test degraded mode: SentinelPulse WireMock returns 503 → inference completes with zeroed news features + `news_available=false` in feature vector
  - Test UNAVAILABLE mode: DataService WireMock returns 503 → all endpoints return `PredictionProvenance.UNAVAILABLE`
  - _Requirements: Req 1.11, Req 1.12, Req 17.3_

- [x] 61. End-to-end training flow integration test
  - Write `tests/test_e2e_training.py` covering: `POST /training/run` dispatches async training task → `GET /training/status/{run_id}` shows running → data ingestion from mock DataService → LeakageValidator runs → PurgedKFoldSplitter with embargo → all gate evaluations → `PromotionDecision` recorded in audit log → `GET /training/status/{run_id}` shows completed
  - Test TrainingAbortedError path: inject a PIT-violating feature → assert training aborted, audit log entry written with ABORTED status
  - _Requirements: Req 3.8, Req 3.9, Req 3.10_

- [x] 62. Performance and concurrency baseline test
  - Write `tests/test_performance.py` testing 100 concurrent prediction requests to each endpoint using `asyncio.gather`
  - Assert p95 latency does not degrade beyond 2× nominal single-request p95 under concurrent load
  - Use `httpx.AsyncClient` with all 100 requests launched simultaneously
  - _Requirements: Req 16.8_

- [x] 63. Security and boundary tests
  - Write `tests/test_security.py`
  - Test all prediction endpoints return 401 when X-API-KEY is absent
  - Test all prediction endpoints return 401 when X-API-KEY has wrong value
  - Test CORS: request from allowed origin returns CORS headers; request from unknown origin does not
  - Test `VALIDATED_ML_ONLY` mode: all HEURISTIC base models → MetaDecisionEngine returns `NO_TRADE` with `INSUFFICIENT_EVIDENCE` + `HEURISTIC_ONLY_BLOCKED` event logged
  - _Requirements: Req 15.2, Req 15.10, Req 17.7_

- [x] 64. Final checkpoint — ensure all tests pass, 90%+ coverage, zero mypy errors
  - Run `pytest tests/ --cov=src --cov-fail-under=90 --timeout=120`
  - Run `mypy --strict src/` — zero type errors
  - Run `ruff check src/ tests/` — zero lint errors
  - Verify all 10 mandatory test files exist and contain at least one property-based test using Hypothesis
  - Ensure all tasks pass, ask the user if questions arise.
  - _Requirements: Req 17.1, Req 18.1, Req 18.2_

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- The TDD mandate (Req 18.2) requires every test to be run and confirmed failing (red) before the corresponding implementation is written (green)
- Each task references specific requirements for full traceability
- Checkpoints (tasks 64) ensure incremental validation
- Property tests validate universal correctness properties (Properties 1–22 from design)
- Unit tests validate specific examples and edge cases
- Legacy alpha-forge `src/` code is reference-only — all new code is a clean rewrite in `ml-service2.0/src/`
- Port: 8100; all prediction endpoints under `/v2/` prefix

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1", "3"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4", "2.5", "4"] },
    { "id": 3, "tasks": ["5", "6", "9"] },
    { "id": 4, "tasks": ["7", "8.1", "8.2"] },
    { "id": 5, "tasks": ["10", "11"] },
    { "id": 6, "tasks": ["12.1", "13"] },
    { "id": 7, "tasks": ["12.2"] },
    { "id": 8, "tasks": ["14.1", "14.2", "15"] },
    { "id": 9, "tasks": ["16", "17"] },
    { "id": 10, "tasks": ["18.1"] },
    { "id": 11, "tasks": ["18.2", "19"] },
    { "id": 12, "tasks": ["20", "21"] },
    { "id": 13, "tasks": ["22.1", "23.1", "24.1", "25.1", "26.1", "28", "29"] },
    { "id": 14, "tasks": ["22.2", "23.2", "24.2", "25.2", "26.2"] },
    { "id": 15, "tasks": ["22.3", "23.3", "25.3", "27.1", "30", "31", "32", "33"] },
    { "id": 16, "tasks": ["34"] },
    { "id": 17, "tasks": ["35.1"] },
    { "id": 18, "tasks": ["35.2", "36", "37"] },
    { "id": 19, "tasks": ["38", "40"] },
    { "id": 20, "tasks": ["39.1"] },
    { "id": 21, "tasks": ["39.2", "41"] },
    { "id": 22, "tasks": ["42.1"] },
    { "id": 23, "tasks": ["42.2"] },
    { "id": 24, "tasks": ["42.3", "43"] },
    { "id": 25, "tasks": ["44", "45"] },
    { "id": 26, "tasks": ["46.1"] },
    { "id": 27, "tasks": ["46.2", "47.1", "47.2"] },
    { "id": 28, "tasks": ["48", "49", "50"] },
    { "id": 29, "tasks": ["51", "52"] },
    { "id": 30, "tasks": ["53.1"] },
    { "id": 31, "tasks": ["53.2", "54"] },
    { "id": 32, "tasks": ["55"] },
    { "id": 33, "tasks": ["56.1"] },
    { "id": 34, "tasks": ["56.2", "57"] },
    { "id": 35, "tasks": ["58"] },
    { "id": 36, "tasks": ["59.1"] },
    { "id": 37, "tasks": ["59.2", "60", "61", "62", "63"] },
    { "id": 38, "tasks": ["64"] }
  ]
}
```
