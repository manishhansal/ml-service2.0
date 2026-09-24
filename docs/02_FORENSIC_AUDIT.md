# Forensic Audit
**ml-service2.0 — Detailed Findings**

*Date: 2026-09-24*

This document contains the detailed findings from the forensic inspection. For the system-wide dependency map, see `ML_SYSTEM_FORENSIC_AUDIT.md`. For the component matrix, see `ML_COMPONENT_REALITY_MATRIX.md`.

---

## 1. Critical Findings (P0)

### Finding F-001: Empty Artifacts Directory
**File:** `artifacts/` (entire directory)  
**Finding:** The `artifacts/` directory is empty. No `.pkl`, `.json`, `.model`, or any artifact file exists. Every model class checks `model_path.exists()` at initialization; when false, it falls through to heuristic code paths. The service has never trained or loaded a real ML model.  
**Implication:** 100% of ML predictions are heuristic rule outputs.

### Finding F-002: MetaDecisionEngine data_quality Hardcoded
**File:** `src/meta/engine.py`, line constructing `AbstentionContext`  
**Code:**  
```python
abstention_ctx = AbstentionContext(
    agreement_ratio=agreement_ratio,
    data_quality=1.0,       # simplified: assumes good data quality
    mean_confidence=mean_confidence,
    ...
)
```
**Finding:** `data_quality=1.0` is a hardcoded constant with an inline comment "simplified: assumes good data quality." This means the `LOW_DATA_QUALITY` abstention condition can never trigger, even when DataService returns a confidence score of 10.  
**Implication:** Safety gate permanently bypassed.

### Finding F-003: BaseMLModel Not Used by Any Concrete Class
**File:** `src/models/base.py` vs. `src/models/regime_classifier.py` (and all other models)  
**Finding:** `BaseMLModel` defines an abstract interface with `predict()`, `fit()`, `score()`, `health_check()`. However, `RegimeClassifier`, `StockRanker`, `StrategySelector`, `RiskPredictor`, `PriceForecaster`, `IVRegimeClassifier`, and `RLExecutionAgent` do NOT inherit from `BaseMLModel`. The class hierarchy is:
```
BaseMLModel (abstract) ← never subclassed
RegimeClassifier       (standalone, own interface)
StockRanker           (standalone, own interface)
...
```
**Implication:** The abstract contract is not enforced. Models can diverge from the interface silently.

### Finding F-004: Two Competing MetaEngine Implementations
**File:** `src/meta_engine/engine.py`  
**Finding:** The file contains both:
- `MetaEngine` — Phase 2 stub, ALL methods raise `NotImplementedError`
- `MetaEngineV3` — Phase 3 implementation, wraps `MetaDecisionEngine`  

And separately, `src/meta/engine.py` contains:
- `MetaDecisionEngine` — the actual production implementation

The relationship: `MetaEngineV3` → wraps → `MetaDecisionEngine`. But `MetaEngine` stub remains in the same file for "TDD compliance."  
**Implication:** Confusing to maintainers; risk of accidentally using stub.

### Finding F-005: TrainingPipeline Has No Data Source
**File:** `src/training/pipeline.py`  
**Finding:** `TrainingPipeline.run_training(model_name, X, y, timestamps, model_factory, ...)` requires pre-built feature matrices. There is no code in the service that constructs these matrices from DataService historical data. The training pipeline is an isolated function with no data source.  
**Implication:** Training cannot be initiated from within the service.

### Finding F-006: No Label Construction Code Anywhere
**Search result:** `grep -r "triple_barrier\|triple barrier\|label_factory\|LabelFactory" src/` — zero results  
**Finding:** No label construction code exists in any file in ml-service2.0.  
**Implication:** Even if X and timestamps could be constructed, y (labels) cannot be constructed.

---

## 2. Significant Findings (P1)

### Finding F-007: CalibrationLayer Identity Transform
**File:** `src/meta/calibration.py`  
**Code:**  
```python
def calibrate(self, model_id: str, raw_score: float) -> float:
    if model_id not in self._calibrators:
        return float(raw_score)  # No calibrator fitted — return raw score
    ...
```
**Finding:** At startup, `_calibrators` is always empty. No persistence mechanism loads fitted calibrators. Every calibrate() call returns the raw score unchanged.  
**Implication:** Reported confidence scores are not calibrated probabilities.

### Finding F-008: EnsembleWeighter IC Registry Never Populated
**File:** `src/meta/ensemble.py`  
**Code:**  
```python
def __init__(self) -> None:
    self._ic_registry: dict[str, dict[str, float]] = {}
```
**Finding:** `_ic_registry` starts empty and has no persistence. `register_ic()` is never called by any production code path. All models receive MIN_WEIGHT (0.05) or equal weight.  
**Implication:** IC-proportional weighting never activates in production.

### Finding F-009: gRPC Client Dead Code
**File:** `src/clients/grpc_client.py`, `src/clients/market_data_pb2*.py`  
**Finding:** gRPC client and generated proto stubs exist but the client is never instantiated by `main.py` or any router. All DataService communication uses REST via `DataServiceClient`.  
**Implication:** Dead code adds maintenance burden and Docker image size.

### Finding F-010: MLflow Never Configured
**File:** `.env.example`, `src/training/mlflow_tracker.py`  
**Finding:** `MLFLOW_TRACKING_URI` is not in `.env.example`. MLflow tracker exists but MLflow connectivity is never verified at startup. Training results may be silently lost.

### Finding F-011: test_coverage_boost Files Inflate Coverage
**Files:** `tests/test_coverage_boost*.py` (7 files), `tests/test_*_mocked*.py` (8 files)  
**Finding:** 15 of 35 test files are explicitly coverage-boosting mocks. Example from `test_coverage_boost_mocked.py`:
```python
# Mock everything at module level
mock_pipeline = MagicMock()
mock_pipeline.build_vector.return_value = AsyncMock(return_value={})
# Then call the function — coverage is achieved without real assertion
```
**Implication:** 86.47% coverage figure is misleading. Real functional coverage is substantially lower.

---

## 3. Positive Findings

### Finding F-012: PurgedKFoldSplitter is Correctly Implemented
**File:** `src/training/purged_kfold.py`  
**Finding:** Implementation correctly:
- Enforces minimum 5-day embargo
- Produces walk-forward splits (validation never before training)
- Purges training samples whose label window overlaps validation
- Falls back to unpurged training with warning (not silent failure)
- Has `n_splits >= 2` validation

### Finding F-013: ModelPromotion Gate Logic is Sound
**File:** `src/registry/promotion.py`  
**Finding:** 6-gate promotion correctly:
- Blocks on `final_oos_used_for_selection=True` permanently
- Requires look_ahead_validated=True
- Requires OOS count >= 60
- Compares challenger vs. champion on predictive and calibration metrics
- Writes to immutable audit log

### Finding F-014: AuditLogger Tamper Detection is Functional
**File:** `src/audit/logger.py`  
**Evidence:** `audit.jsonl` contains real entries from test runs, and each entry carries an `entry_hash` computed from the content. The `_check_append_only()` method verifies the hash chain at startup. This is genuinely production-grade.

### Finding F-015: Pydantic V2 Strict Mode Correctly Applied
**File:** `src/schemas/base.py`  
**Finding:** `BaseSchema` uses `ConfigDict(strict=True, frozen=True)`. This prevents silent type coercion and makes schemas immutable. All response schemas inherit this.

### Finding F-016: AbstentionPolicy 5-Condition Check is Correct
**File:** `src/meta/abstention.py`  
**Finding:** All 5 conditions (agreement, data_quality, confidence, stop_prob, model_count) are checked independently and OR'd together. Any single failure triggers NO_TRADE. No condition can be silently skipped.

---

## 4. Audit Anomalies

### Anomaly A-001: audit.jsonl Contains Real Promotion Decisions
The `audit.jsonl` file in the workspace root contains 20+ real promotion decision entries and 1 training run entry. The training run (run_id `f1097534`) shows:
```
ic_mean: 0.0159 (below 0.02 threshold → REJECTED)
```
This was a test/dev run, not production data. The IC gate correctly rejected it.

### Anomaly A-002: Two Drift Detector Implementations
`src/monitoring/drift.py` contains both `DriftDetector` and `DriftDetectorV3`. There is no clear documentation on which is canonical. The `test_drift_monitoring.py` tests appear to cover the newer `DriftMonitor` class in `drift_monitor.py`, not `drift.py`.

---

*End of Forensic Audit Findings*
