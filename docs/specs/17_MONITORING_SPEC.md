# Monitoring Specification
**ml-service2.0 — Observability, Drift Detection, and Performance Monitoring**

*Date: 2026-09-24*

---

## 1. Metrics to Track

### 1.1 Prediction Metrics (per inference)

| Metric | Type | Description |
|---|---|---|
| `inference_latency_ms` | Histogram (p50/p95/p99) | End-to-end latency from request to response |
| `feature_latency_ms` | Histogram | FeaturePipeline.build_vector() duration |
| `model_inference_latency_ms` | Histogram per model | Per-model predict() duration |
| `abstention_rate` | Counter | NO_TRADE decisions / total decisions |
| `signal_quality_tier` | Counter per tier | REJECTED/ABSTAIN/WATCH/VALID/HC counts |
| `agreement_ratio` | Histogram | Distribution of agreement_ratio values |
| `confidence` | Histogram | Distribution of confidence values |
| `provenance_distribution` | Counter per provenance | TRAINED_MODEL vs HEURISTIC vs UNAVAILABLE |

### 1.2 Data Quality Metrics

| Metric | Type | Description |
|---|---|---|
| `data_quality_score` | Histogram | Distribution of DataConfidenceScore values |
| `signal_engine_blocked` | Counter | Requests blocked by signalEngineAllowed=false |
| `low_confidence_blocked` | Counter | Requests blocked by DataConfidenceScore < threshold |
| `circuit_breaker_open` | Gauge | 1 when DataService circuit is open |
| `data_service_latency_ms` | Histogram | DataServiceClient request duration |
| `sentinel_latency_ms` | Histogram | SentinelPulseClient request duration |
| `news_cache_hit_rate` | Counter | LRU cache hits / total requests |

### 1.3 Model Performance Metrics

| Metric | Type | Description |
|---|---|---|
| `realized_ic` | Gauge | Rolling Spearman IC (populated by feedback loop) |
| `realized_sharpe` | Gauge | Rolling net Sharpe (populated by feedback loop) |
| `calibration_ece` | Gauge per model | Expected Calibration Error |
| `prediction_accuracy` | Gauge | Direction accuracy (populated by feedback loop) |
| `expected_vs_realized` | Histogram | Predicted probability vs. realized frequency |

### 1.4 Drift Metrics

| Metric | Type | Description |
|---|---|---|
| `feature_psi` | Gauge per feature | PSI value for each feature |
| `feature_drift_severity` | Gauge | 0=LOW, 1=MEDIUM, 2=HIGH |
| `model_drift_severity` | Gauge per model | Model-level drift severity |
| `drift_alerts_active` | Gauge | Count of active drift alerts |
| `retrain_recommended` | Gauge | 1 when PSI > HIGH_THRESHOLD |

---

## 2. Drift Detection

### 2.1 Feature Drift (PSI)

```python
# Implemented in DriftMonitor.compute_psi()
PSI_THRESHOLDS:
    LOW:    PSI <= 0.20
    MEDIUM: 0.20 < PSI <= 0.25
    HIGH:   PSI > 0.25

Reference distribution: training data distribution (must be stored at training time)
Current distribution: rolling 20-prediction window

Actions by severity:
    LOW:    MONITOR — log, no action
    MEDIUM: WARN — alert, increase monitoring frequency
    HIGH:   RETRAIN — trigger challenger training
```

### 2.2 Model Drift (NannyML CBPE)

When ground-truth labels are unavailable (typical between batch evaluation runs):

```python
# NannyML CBPE estimates model performance without labels
# Run on rolling window of 100 predictions
estimated_roc_auc = nannyml_estimator.estimate(current_predictions)
if estimated_roc_auc < baseline_roc_auc × 0.95:
    trigger_challenger_training()
```

### 2.3 Calibration Drift

```python
# Rolling ECE in 20-trade windows (requires realized outcomes)
# Only available after feedback loop is operational
rolling_ece = compute_ece(predicted_probs, realized_outcomes, window=20)
if rolling_ece > baseline_ece + 0.03:
    alert(f"Calibration drift detected: ECE={rolling_ece:.3f}")
```

---

## 3. Alerting Thresholds

| Alert | Threshold | Severity | Action |
|---|---|---|---|
| Feature PSI > 0.25 | PSI > 0.25 | CRITICAL | Trigger retraining |
| Abstention rate > 50% | >50% NO_TRADE | HIGH | Investigate data quality |
| Circuit breaker open > 5min | >300s open | HIGH | Page on-call |
| IC rolling < 0 | Mean IC over 50 trades < 0 | HIGH | Downgrade model |
| ECE > 0.10 | ECE > 0.10 | HIGH | Downgrade model calibration |
| p99 latency > 500ms | p99 > 500ms | MEDIUM | Profile and optimize |
| Data quality < 60% of requests OK | <60% pass | HIGH | Alert data team |

---

## 4. Monitoring Endpoints

| Endpoint | Description |
|---|---|
| `GET /monitoring/health` | Service health + model status |
| `GET /monitoring/alerts` | Active drift alerts |
| `GET /monitoring/metrics` | Prometheus metrics endpoint |
| `GET /monitoring/drift/{model_id}` | Per-model drift report |
| `GET /monitoring/performance` | Rolling model performance metrics |
| `GET /monitoring/calibration` | Calibration diagnostics |

---

## 5. Logging Standards

All logs use structlog JSON format. Required fields for inference logs:

```json
{
    "event": "inference_complete",
    "timestamp": "2026-09-24T09:30:00.123Z",
    "request_id": "uuid",
    "symbol": "RELIANCE",
    "action": "NO_TRADE",
    "reason_codes": ["LOW_AGREEMENT"],
    "confidence": 0.43,
    "agreement_ratio": 0.43,
    "abstention": true,
    "provenance": "heuristic",
    "data_quality": 0.85,
    "latency_ms": 45,
    "regime": "sideways",
    "n_available_models": 7
}
```

---

## 6. Implementation Status

| Component | Implemented | Working | Gaps |
|---|---|---|---|
| PSI drift detector | ✓ | ~ | No reference distribution stored |
| Evidently wrapper | ✓ | ~ | Not wired to inference |
| NannyML wrapper | ✓ | ~ | Not wired to inference |
| Drift alerts schema | ✓ | ✓ | |
| Prometheus metrics | ✗ | ✗ | Not added to ml-service2.0 |
| Feedback loop metrics | ✗ | ✗ | Feedback loop doesn't exist yet |
| Calibration monitoring | ✗ | ✗ | No fitted calibrators |
| Performance tracking | ✗ | ✗ | No outcome data |

---

*End of Monitoring Specification*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Implemented by src/monitoring/reference.py (ReferenceDistributionStore snapshots training feature + prediction distributions; DriftGate computes feature + prediction PSI vs reference and maps severity to action: LOW->MONITOR, MEDIUM->ALERT, HIGH->TRAIN_CHALLENGER, CRITICAL->BLOCK) and src/monitoring/performance_drift.py (PerformanceDriftTracker: rolling IC/Brier, >20% IC-decay detection). Existing DriftMonitor and DriftDetectorV3 retained. Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
