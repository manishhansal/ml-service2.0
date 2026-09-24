# Failure Modes
**ml-service2.0 — Safe Failure Behavior and Recovery Procedures**

*Date: 2026-09-24*

---

## 1. Failure Handling Principle

**The system must fail safely. When in doubt: NO_TRADE.**

A false positive (trading when shouldn't) costs money. A false negative (not trading when should) loses opportunity but preserves capital. Capital preservation beats opportunity maximization in failure conditions.

---

## 2. Failure Scenarios

### 2.1 data-service2.0 Unavailable

```
Scenario:
    DataServiceClient.get_historical_bars() raises DataServiceUnavailableError

Response:
    deployment_mode == validated_ml_only:
        → Return NO_TRADE
        → reason_codes = ["DATA_SERVICE_UNAVAILABLE"]
        → provenance = UNAVAILABLE
        → Log at ERROR level
        → Emit drift alert (circuit_breaker_open gauge = 1)
    
    deployment_mode == paper/research:
        → Return NO_TRADE (same, cannot fabricate market data)
    
    DO NOT:
        → Return heuristic prediction with fake confidence
        → Substitute cached data without flagging staleness
        → Silently proceed with zero-filled features
```

### 2.2 data-service2.0 Returns Degraded Data

```
Scenario:
    signalEngineAllowed=false OR DataConfidenceScore < threshold

Response:
    → Raise SignalEngineNotAllowedError or LowDataConfidenceError
    → Return NO_TRADE
    → reason_codes = ["LOW_DATA_QUALITY"]
    → provenance = INSUFFICIENT_EVIDENCE
    
    DO NOT:
        → Ignore the quality flag
        → Use the data with just a warning log
```

### 2.3 SentinelPulse Unavailable

```
Scenario:
    SentinelPulseClient.fetch_news_context() returns None

Response:
    → Proceed in market-only mode (no news features)
    → Reduce confidence by news_unavailable_penalty (configurable, default 0.10)
    → Add reason_code "NEWS_UNAVAILABLE" to signal
    → Log at WARN level
    
    DO NOT:
        → Fabricate news sentiment (e.g., "neutral" with confidence 0.5)
        → Block trade if market features are valid (news is additive, not required)
    
    AlphaForge behavior:
        → ML_MODE=fallback: proceed with market-only inference
        → ML_MODE=required: fail if news is explicitly required in configuration
```

### 2.4 Model Registry Unavailable / Corrupt Artifact

```
Scenario A: artifacts/ directory not readable
    → Log CRITICAL at startup
    → All models fall back to heuristic
    → Set provenance=HEURISTIC for all predictions
    → deployment_mode=validated_ml_only: refuse to start

Scenario B: SHA256 integrity check fails on artifact load
    → Log CRITICAL (ArtifactIntegrityFailure)
    → DO NOT load the corrupt artifact
    → Use previous champion or heuristic fallback
    → Alert operator

Scenario C: Model file missing despite registry entry
    → Log ERROR
    → Mark model as UNAVAILABLE in runtime
    → MetaDecisionEngine handles UNAVAILABLE model in quorum
```

### 2.5 Redis Unavailable

```
Scenario:
    RedisCache.connect() fails or times out

Response:
    → Fall back to in-process LRUCache (max_size=500, ttl=90s)
    → Continue serving requests
    → Log at WARN level
    → WebSocket stream: cannot deliver signals (connection will drop)
    
    DO NOT:
        → Block startup on Redis unavailability
        → Return errors to callers when LRU fallback is available
```

### 2.6 MLflow Unavailable

```
Scenario:
    MLflowTracker cannot connect to MLFLOW_TRACKING_URI

Response:
    → Log at WARN level
    → Continue training run
    → Write metrics to local JSON file (artifacts/training_fallback/)
    → DO NOT fail the training run
    → Warn in training result that MLflow logging was skipped
```

### 2.7 Drift Detected in Production

```
Scenario:
    DriftMonitor.get_severity(psi) == DriftSeverity.HIGH

Response (automated):
    → Set model status to DEGRADED in runtime
    → Reduce model's ensemble weight to MIN_WEIGHT (0.05)
    → Add "MODEL_DRIFT_DETECTED" to reason_codes
    → Create DriftAlert (persisted)
    → Recommend RETRAIN action
    
Response (manual trigger required):
    → Operator reviews DriftAlert
    → Initiates challenger training if drift is confirmed
    → After new champion promoted, clear drift status
    
    DO NOT:
        → Silently continue with drifted model at full weight
        → Automatically promote new model without gates
```

### 2.8 Malformed Feature Vector

```
Scenario:
    FeaturePipeline.build_vector() returns partial or malformed FeatureVector
    OR: Pydantic validation fails on feature schema

Response:
    → Log ERROR with exact validation error
    → Return NO_TRADE
    → reason_codes = ["FEATURE_VALIDATION_FAILURE"]
    → DO NOT proceed with partial features
    → DO NOT substitute NaN with 0
```

### 2.9 Model Inference Takes Too Long

```
Scenario:
    Model predict() call exceeds p99 latency threshold (e.g., 500ms)

Response:
    → LLMNewsReasoner: 120ms timeout enforced via asyncio.wait_for
      → Cache fallback → neutral signal
    → Other models: no timeout currently implemented (P3 gap)
    → Recommendation: add asyncio timeout wrappers for all models
```

### 2.10 Online Learning Update Fails

```
Scenario:
    OnlineLearner.update() fails due to:
    - consecutive_updates > max_consecutive_updates
    - IC below threshold
    - Feature schema mismatch

Response:
    → DO NOT update champion model
    → Log at WARN level
    → OnlineLearner.requires_full_retrain() returns True
    → Trigger full retraining pipeline
    → Keep existing champion running unchanged
    
    DO NOT:
        → Partially update the champion
        → Allow a corrupt update to persist
        → Continue online learning if checksum fails
```

### 2.11 PIT Violation Detected

```
Scenario:
    feature_as_of >= prediction_timestamp
    OR: SentinelNewsContext.as_of >= prediction_timestamp

Response:
    → Raise PITViolationError immediately
    → Log at CRITICAL level (this should never happen in production)
    → Return HTTP 500 (not 200 with NO_TRADE)
    → Alert operator immediately
    
    DO NOT:
        → Silently proceed with potentially contaminated features
        → Log and continue
        → Suppress the error
    
    This is a CODE BUG if it happens — investigate immediately.
```

---

## 3. Failure Recovery Checklist

### Data Service Recovery
1. Wait for circuit breaker to half-open (60s)
2. First probe request: if success, reset circuit
3. No manual intervention needed for transient failures

### SentinelPulse Recovery
- Automatic: retry with exponential backoff (3 attempts)
- LRU cache covers outages up to 90s TTL
- After 90s: market-only mode until service restores

### Model Rollback
1. Identify failing model via DriftAlert or performance metrics
2. Navigate to /v2/models/registry to identify champion and shadow
3. Invoke rollback API: `POST /registry/rollback/{model_id}`
4. Rollback loads previous artifact from registry
5. Confirm via monitoring that drift alert clears
6. Investigate root cause before re-promoting

### Full Service Recovery
1. Restart service: Redis reconnects, models reload from artifacts/
2. Audit log integrity check runs at startup
3. If integrity check fails: DO NOT start, investigate tamper
4. If models missing: start in HEURISTIC mode only, page operator

---

## 4. Chaos Test Scenarios

The following scenarios must be tested in CI or staging:

| Test | Expected Behavior |
|---|---|
| DataService returns 503 | NO_TRADE, circuit opens after 3 failures |
| DataService returns 429 (rate limit) | Retry with backoff, then NO_TRADE |
| DataService returns stale data (1h old) | NO_TRADE if staleness > threshold |
| SentinelPulse returns 503 | Market-only mode, degraded confidence |
| Redis outage | LRU fallback, continue serving |
| Missing candles in OHLCV | Feature pipeline handles gracefully |
| Duplicate candles in OHLCV | Deduplication, no IC inflation |
| Feature vector all NaN | NO_TRADE, FEATURE_VALIDATION_FAILURE |
| Model artifact corrupt | Load old champion, alert |
| PIT violation in feature | PITViolationError, HTTP 500, alert |
| Memory pressure | Evict LRU caches, continue |
| All models UNAVAILABLE | NO_TRADE, ALL_MODELS_UNAVAILABLE |
| MLflow down | Training continues, warn about missing tracking |

---

*End of Failure Modes*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Graceful degradation implemented and tested: DataService circuit breaker + UNAVAILABLE provenance; SentinelPulse unreachable -> news degrades to neutral; Redis outage -> LRU fallback; corrupt artifact -> ArtifactIntegrityFailure + heuristic fallback; all-models-unavailable -> NO_TRADE; PIT violation -> blocked in inference. A complete chaos suite (DataService stale-data, memory pressure, MLflow-down) remains future work. Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
