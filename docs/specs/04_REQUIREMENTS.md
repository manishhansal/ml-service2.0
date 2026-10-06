# Requirements Specification
**ml-service2.0 — Functional Requirements**

*Date: 2026-09-24*
*Format: REQ-ID | Description | Priority | Acceptance Criteria | Implementation Status*

---

## Data and PIT Requirements

### ML-REQ-001
**The prediction engine SHALL reject inference when market-data freshness exceeds the configured horizon.**
- Priority: P0
- Acceptance: Stale input (DataServiceResponse.as_of > threshold) produces NO_TRADE; audit event generated; no signal persisted as executable
- Status: PARTIAL — circuit breaker exists; explicit staleness rejection not implemented

### ML-REQ-002
**Every FeatureVector SHALL carry a `feature_as_of` timestamp representing the latest data timestamp used.**
- Priority: P0
- Acceptance: `FeatureVector.feature_as_of` is present; assert feature_as_of < prediction_timestamp on every request
- Status: NOT IMPLEMENTED — `feature_as_of` absent from FeatureVector schema

### ML-REQ-003
**The prediction engine SHALL never use data with `published_at > prediction_timestamp` from SentinelPulse.**
- Priority: P0
- Acceptance: `SentinelNewsContext.as_of < prediction_timestamp` validated before news features are consumed; future news raises PITViolationError
- Status: PARTIAL — `as_of` is validated non-future in schema; not compared to prediction_timestamp at runtime

### ML-REQ-004
**Every prediction response SHALL include `prediction_timestamp`, `feature_as_of`, and `data_as_of`.**
- Priority: P0
- Acceptance: All prediction schemas include these three fields; they are populated with UTC datetimes; they cannot be null
- Status: NOT IMPLEMENTED

### ML-REQ-005
**Missing OI, IV, Greeks, and bid/ask SHALL NOT be zero-filled.**
- Priority: P0
- Acceptance: Null OI fields remain null; a model that receives null OI uses degraded-mode feature (not 0); test confirms 0 substitution does not occur
- Status: PARTIAL — OHLCVBar validates non-zero for O/H/L/C/V; OI/IV null handling not explicitly tested

### ML-REQ-006
**The ML service SHALL NOT call any external market data provider directly.**
- Priority: P0
- Acceptance: No import of Angel One, Upstox, Yahoo Finance, Binance, NSE client in ml-service2.0; all data comes through DataServiceClient
- Status: PASS — only DataServiceClient and SentinelPulseClient used

---

## Model Training Requirements

### ML-REQ-010
**The training pipeline SHALL construct training labels using a documented, PIT-correct labeling method.**
- Priority: P0
- Acceptance: LabelFactory exists; labels carry `feature_as_of`, `label_start`, `label_end`; leakage test passes
- Status: NOT IMPLEMENTED

### ML-REQ-011
**The training pipeline SHALL support triple-barrier labeling with ATR-calibrated barriers.**
- Priority: P1
- Acceptance: `LabelFactory.triple_barrier(ohlcv, atr_multiple=2.0)` returns labeled DataFrame with path-dependent outcomes
- Status: NOT IMPLEMENTED

### ML-REQ-012
**Every training run SHALL produce a dataset version record with hash, range, universe, feature schema, and label schema.**
- Priority: P1
- Acceptance: `DatasetVersion` object stored with model artifact; reproducible from stored metadata
- Status: PARTIAL — dataset_hash stored; range stored; schema version not tracked

### ML-REQ-013
**The training pipeline SHALL use PurgedKFold with minimum 5-day embargo and label_horizon_days matching the prediction horizon.**
- Priority: P0
- Acceptance: PurgedKFoldSplitter used in TrainingPipeline; embargo_days >= 5; label_horizon_days = target horizon
- Status: PASS — PurgedKFoldSplitter implemented with min 5-day embargo enforcement

### ML-REQ-014
**The training pipeline SHALL run walk-forward validation with at minimum 3 non-overlapping OOS windows.**
- Priority: P0
- Acceptance: WalkForwardValidator reports per-window metrics; worst-window Sharpe is documented; not just mean
- Status: NOT IMPLEMENTED

### ML-REQ-015
**The training pipeline SHALL compute acceptance gates: IC >= 0.02, net Sharpe >= 0.0, PBO <= 0.5.**
- Priority: P0
- Acceptance: Test confirms REJECTED status when IC < 0.02; REJECTED when Sharpe < 0; REJECTED when PBO > 0.5
- Status: PASS — implemented and verified via audit.jsonl

### ML-REQ-016
**Every ML model SHALL demonstrate it beats a naive baseline after transaction costs.**
- Priority: P1
- Acceptance: BaselineEvaluator computes buy-and-hold and naive momentum metrics for same OOS period; model IC/Sharpe must exceed baseline
- Status: NOT IMPLEMENTED

---

## Model Validation Requirements

### ML-REQ-020
**The CPCV validation SHALL estimate PBO via combinatorial path enumeration.**
- Priority: P1
- Acceptance: CPCVSplitter generates configurable number of backtest paths; PBO computed as fraction of paths with negative Sharpe; reported in training results
- Status: PARTIAL — `_compute_pbo` approximates as fraction of negative-Sharpe folds; true CPCV path enumeration not implemented

### ML-REQ-021
**Calibration SHALL be validated on OOS data with ECE < 0.05 before promotion.**
- Priority: P1
- Acceptance: CalibrationValidator computes ECE per decile; ECE gate added to ModelPromotion; model with ECE >= 0.05 cannot be promoted
- Status: NOT IMPLEMENTED — calibration gate exists but ECE threshold not enforced in promotion

### ML-REQ-022
**Every probability-producing model SHALL have a fitted calibrator (Platt or isotonic) from training holdout data.**
- Priority: P0
- Acceptance: CalibrationLayer.has_calibrator(model_id) returns True for all production models; uncalibrated model cannot be CHAMPION
- Status: NOT IMPLEMENTED — CalibrationLayer exists but no fitted calibrators

---

## Signal Requirements

### ML-REQ-030
**NO_TRADE SHALL be a first-class decision output, not a fallback.**
- Priority: P0
- Acceptance: NO_TRADE can be returned when evidence is insufficient, even when data is available and models are loaded
- Status: PASS — AbstentionPolicy triggers NO_TRADE on 5 conditions

### ML-REQ-031
**Every signal SHALL carry: signal_id, prediction_timestamp, feature_as_of, model_version, calibration_score, regime, abstention_reason.**
- Priority: P0
- Acceptance: Signal schema validated by Pydantic V2; none of these fields are nullable in production mode
- Status: PARTIAL — many fields present; `prediction_timestamp`, `feature_as_of`, `signal_id` missing

### ML-REQ-032
**Signal tiers SHALL be derived from empirically validated thresholds, not arbitrary confidence cutoffs.**
- Priority: P1
- Acceptance: Signal tier thresholds (REJECTED/ABSTAIN/WATCH/VALID/HIGH_CONVICTION) are learned from historical outcomes; documented with validation data
- Status: NOT IMPLEMENTED

### ML-REQ-033
**Signals from heuristic-only models SHALL be marked with `provenance=HEURISTIC` and SHALL NOT be eligible for live capital allocation.**
- Priority: P0
- Acceptance: `PredictionProvenance.is_live_eligible` returns False for HEURISTIC; AlphaForge respects this field
- Status: PASS in schema — but AlphaForge does not currently filter on is_live_eligible

---

## Expected Value Requirements

### ML-REQ-040
**The decision engine SHALL compute risk-adjusted expected value: EV = P(target) × payoff − P(stop) × loss − costs − uncertainty_penalty.**
- Priority: P1
- Acceptance: ExpectedValueEngine implemented; EV is the primary gate for NO_TRADE decision (EV < 0 → NO_TRADE regardless of confidence)
- Status: NOT IMPLEMENTED

### ML-REQ-041
**Estimated probabilities P(target_hit) and P(stop_hit) SHALL sum to <= 1.0.**
- Priority: P0
- Acceptance: Property test verifies P(target) + P(stop) + P(neither) = 1.0; test covers stop > target case
- Status: PARTIAL — RiskPredictor returns these separately but sum not validated

---

## Model Registry Requirements

### ML-REQ-050
**Every model SHALL progress through: HYPOTHESIS → BACKTEST → CHALLENGER → SHADOW → APPROVED → PRODUCTION.**
- Priority: P0
- Acceptance: Backward transitions raise ValueError; skipping stages raises ValueError
- Status: PASS — validate_lifecycle_transition() implemented

### ML-REQ-051
**A CHAMPION model SHALL never be mutated in place. Every update creates a new version.**
- Priority: P0
- Acceptance: `ArtifactAlreadyExistsError` raised on duplicate model_id; promotion creates new artifact
- Status: PASS — ModelRegistry raises ArtifactAlreadyExistsError

### ML-REQ-052
**Model promotion SHALL require human approval for PROMOTE outcome.**
- Priority: P0
- Acceptance: ApprovalToken is time-limited; promotion without valid token raises error; token is logged in audit
- Status: PASS — ApprovalToken implemented

### ML-REQ-053
**Automated model promotion SHALL be impossible without completing all 6 gates.**
- Priority: P0
- Acceptance: A model with any FAIL gate cannot reach PRODUCTION; this is enforced in code, not just policy
- Status: PASS — _determine_outcome() checks all gates

---

## Online Learning Requirements

### ML-REQ-060
**Online learning SHALL NOT update the champion model directly.**
- Priority: P0
- Acceptance: OnlineLearner updates a SHADOW copy; champion remains unchanged until explicit promotion
- Status: PARTIAL — OnlineLearner has architecture for this but is not connected to shadow/champion separation

### ML-REQ-061
**Online learning SHALL be blocked when: consecutive updates > limit, drift > threshold, performance below floor.**
- Priority: P0
- Acceptance: `requires_full_retrain()` returns True when limits exceeded; update blocked
- Status: PASS — OnlineLearner.requires_full_retrain() implemented

### ML-REQ-062
**Online learning SHALL require at minimum 30 new samples per update.**
- Priority: P0
- Acceptance: Update rejected with `InsufficientSamplesError` when n_samples < 30
- Status: PASS — min_samples_per_update enforced in OnlineLearner

---

## Monitoring Requirements

### ML-REQ-070
**Feature drift SHALL be computed continuously using PSI against the training reference distribution.**
- Priority: P1
- Acceptance: DriftMonitor.compute_psi() called for all production features at each inference batch; PSI > 0.25 triggers RETRAIN recommendation
- Status: PARTIAL — PSI implemented; reference distribution not stored; inference-time computation not wired

### ML-REQ-071
**A RETRAIN recommendation SHALL automatically trigger a new challenger training run.**
- Priority: P1
- Acceptance: RetrainTrigger exists; connects drift alert to TrainingPipeline; new challenger enters promotion pipeline
- Status: NOT IMPLEMENTED

### ML-REQ-072
**Calibration drift SHALL be measured: expected probability vs. realized frequency in rolling 20-trade windows.**
- Priority: P1
- Acceptance: CalibrationTracker computes rolling ECE; alert when ECE degrades > 0.03 from baseline
- Status: NOT IMPLEMENTED

---

## Failure Handling Requirements

### ML-REQ-080
**When data-service2.0 is unavailable, the ML service SHALL return NO_TRADE, not a heuristic prediction.**
- Priority: P0
- Acceptance: DataServiceUnavailableError → NO_TRADE with reason_code UNAVAILABLE_DATA; VALIDATED_ML_ONLY mode enforces this; fallback mode may allow heuristic with explicit provenance
- Status: PARTIAL — deployment_mode gate exists; not consistently enforced across all models

### ML-REQ-081
**When a model is unavailable, the system SHALL use the remaining available models for quorum, potentially reaching INSUFFICIENT_MODELS abstention.**
- Priority: P0
- Acceptance: ModelProvenace.UNAVAILABLE models excluded from quorum; < 3 available → NO_TRADE
- Status: PASS — MetaDecisionEngine correctly handles UNAVAILABLE models

### ML-REQ-082
**When drift is detected in a model's input features, that model's predictions SHALL be downgraded to HEURISTIC provenance.**
- Priority: P1
- Acceptance: DriftMonitor triggers provenance downgrade for affected model; MetaDecisionEngine uses degraded provenance in weakest-link calculation
- Status: NOT IMPLEMENTED

---

## Explainability Requirements

### ML-REQ-090
**Every production prediction SHALL carry: top-3 contributing features, SHAP values, regime, data quality, news contribution.**
- Priority: P1
- Acceptance: ExplainResponse returned with every MetaOutput; SHAP values present for TRAINED_MODEL provenance; reason codes present for all
- Status: PARTIAL — SHAP infrastructure exists; fires only with trained models; MetaOutput carries explainability block

### ML-REQ-091
**The decision reason SHALL be auditable: it MUST be derivable from logged model outputs, not just from the final decision.**
- Priority: P0
- Acceptance: AuditLogger records model_outputs, weights, calibrated_scores, and final action; decision can be reproduced from log
- Status: PARTIAL — AuditLogger records training runs and promotions; inference-time model outputs not logged per-prediction

---

## Reproducibility Requirements

### ML-REQ-100
**The same dataset + code + random seed SHALL produce materially identical training results.**
- Priority: P0
- Acceptance: Rerunning TrainingPipeline with identical inputs produces IC within 0.001; random seed documented in audit log
- Status: PARTIAL — dataset_hash tracked; random seed not explicitly tracked in audit entries

### ML-REQ-101
**Every training run SHALL record: code SHA, dataset version, feature schema version, random seed, dependency lock.**
- Priority: P0
- Acceptance: TrainingResult.provenance_record includes all five fields; stored in audit log
- Status: PARTIAL — dataset_hash, model_version present; code SHA, feature schema version, dependency lock absent

---

*End of Requirements Specification*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Requirements realized across src/data, src/features, src/models, src/training, src/backtest, src/meta, src/monitoring, src/registry, and src/explainability. All P0 blockers (P0-001..P0-008) are closed. See reports/ML_SERVICE_FINAL_CERTIFICATION.md and reports/ml_certification.json for per-requirement evidence and the live-data certification result (RESEARCH_READY / PAPER_READY; no cost-surviving edge verified on real data).
