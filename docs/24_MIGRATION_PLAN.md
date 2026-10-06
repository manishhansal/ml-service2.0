# Migration Plan
**ml-service2.0 — Phased Upgrade from Heuristic to Validated ML Engine**

*Date: 2026-09-24*

---

## 1. Migration Principles

1. **Preserve working components** — do not rewrite code that is structurally correct
2. **Incremental deployment** — each phase is independently testable and deployable  
3. **Backward compatibility** — AlphaForge API contracts must not break
4. **Evidence gates** — no phase progresses without measurable improvement
5. **Fail safe** — if any phase introduces regression, rollback immediately

---

## 2. Phase Plan

### PHASE 0 — Forensic Audit [COMPLETE]
- ✓ All 4 repositories inspected
- ✓ Component reality matrix produced
- ✓ Gap analysis documented
- ✓ Documentation suite produced

---

### PHASE 2 — Data Contracts and PIT Guarantees

**Goal:** Guarantee PIT correctness and data contract compliance before any model training.

**Tasks:**
1. Add `prediction_timestamp`, `feature_as_of`, `data_as_of`, `expires_at` to `MetaOutput`, `GoNoGoDecision`, and all prediction schemas
2. Add `signal_id` (UUID) to all prediction responses for audit trail
3. Thread `data_quality` from DataServiceClient through FeaturePipeline → MetaDecisionEngine → AbstentionContext (remove hardcoded `data_quality=1.0`)
4. Wire `LookAheadGuard.check()` into `FeaturePipeline.build_vector()` and all predict endpoints
5. Add `NEWS_FEATURE_ASOF` validation: assert `SentinelNewsContext.as_of < prediction_timestamp`
6. Add `feature_as_of` field to `FeatureVector` Pydantic schema
7. Add inference-time PIT property tests

**Breaking changes:** None (all additions are new fields, backward compatible)  
**Test gate:** All PIT property tests pass; all contract tests pass  
**Estimated effort:** 3-5 days

---

### PHASE 3 — Feature and Label Factory

**Goal:** Build the foundational infrastructure for real model training.

**Tasks:**
1. Build `src/training/label_factory.py`:
   - `fixed_horizon()` — net returns with cost adjustment
   - `triple_barrier()` — path-dependent labels
   - `volatility_adjusted()` — vol-normalized returns
   - `meta_label()` — secondary filter labels
2. Build `src/training/data_ingestion.py`:
   - `DataIngestionPipeline.fetch_training_dataset()` — pulls historical OHLCV from DataService
   - Handles pagination, rate limiting, gap detection
   - Assembles `(X, y, timestamps)` for TrainingPipeline
3. Extend `FeaturePipeline` with full Groups A-D features (price, volume, microstructure, derivatives)
4. Add `DatasetVersion` schema to record: data range, universe, feature schema version, label schema version
5. Add feature validation tests per specification

**Breaking changes:** None  
**Test gate:** LabelFactory unit tests pass; PIT validation tests pass on synthetic data  
**Estimated effort:** 2-3 weeks

---

### PHASE 4 — Baseline Models

**Goal:** Establish training infrastructure validity with simple models that must beat naive baselines.

**Tasks:**
1. Build `src/training/baseline_evaluator.py`:
   - `BuyAndHoldBaseline`, `NaiveMomentumBaseline`, `MeanReversionBaseline`
   - `AlphaForgeRuleBaseline` — replicate current NSE 8-factor score
2. Train Logistic Regression on real NIFTY data as "first model ever trained"
3. Run `TrainingPipeline.run_training()` for the first time on real historical data
4. Verify IC > 0 and beats naive momentum (if not, do NOT proceed)
5. Build `src/training/walk_forward_validator.py`
6. Run 5-window walk-forward and confirm consistent positive IC
7. Fit first `CalibrationLayer` calibrators from training holdout

**Critical decision point:** If logistic regression does not show positive OOS IC:
- Do NOT advance to complex models
- Investigate feature quality, label construction, data quality
- Document findings in `reports/` and update gap analysis

**Test gate:** Logistic regression shows mean OOS IC > 0.01 and beats naive momentum  
**Estimated effort:** 2-3 weeks

---

### PHASE 5 — Advanced Model Zoo

**Goal:** Train tree models and establish multi-model ensemble.

**Tasks:**
1. Train `RegimeClassifier` (XGBoost) on historical NIFTY data with regime labels
2. Train `RiskPredictor` (XGBoost) on historical trade setup data with triple-barrier labels
3. Train `StockRanker` (LightGBM LambdaRank) on F&O universe cross-sectional data
4. Train `StrategySelector` (CatBoost) with walk-forward strategy outcome labels
5. Populate `EnsembleWeighter` IC registry from training results
6. Evaluate IC-weighted ensemble vs. equal-weighted ensemble
7. Run `ModelPromotion.evaluate_all_gates()` on all trained challengers
8. Promote first CHAMPION for at least one model

**Critical decision point:** If tree models do not beat logistic regression baseline after costs:
- Investigate feature quality and label construction
- Do NOT add deep learning models to compensate
- Fix the fundamentals

**Test gate:** At least one model reaches CHAMPION status via all 6 promotion gates  
**Estimated effort:** 3-4 weeks

---

### PHASE 6 — Validation and Backtesting

**Goal:** Validate OOS edge with rigorous statistical testing.

**Tasks:**
1. Build cost-aware backtest engine with realistic execution assumptions
2. Run full walk-forward validation (12 windows) on all trained models
3. Run CPCV to estimate PBO for each model
4. Compute bootstrap confidence intervals on IC
5. Run ablation studies: no-news, no-derivatives, no-regime, no-volume
6. Validate SentinelPulse incremental contribution
7. Run regime robustness testing (6 regimes)
8. Compute deflated Sharpe Ratio
9. Statistical significance testing (permutation test, p < 0.05)

**Critical decision point:** If PBO > 0.5 or IC bootstrap CI includes 0:
- Do NOT deploy models to production
- Document findings honestly in `reports/NO_VERIFIED_EDGE.md`
- Return to Phase 3/4 with revised approach

**Test gate:** PBO < 0.5; IC 95% CI excludes 0; OOS Sharpe > 0 after costs  
**Estimated effort:** 2-3 weeks

---

### PHASE 7 — Calibration and Ensemble

**Goal:** Fit calibrators and validate ensemble weighting.

**Tasks:**
1. Fit Platt scaling calibrators for all production-eligible models
2. Validate ECE < 0.05 on OOS data for each model
3. Add ECE gate to `ModelPromotion`
4. Enable `ConfidenceDecomposer` with fitted components
5. Validate IC-weighted ensemble beats equal-weighted ensemble OOS
6. Implement `ExpectedValueEngine` and EV-based NO_TRADE gate

**Test gate:** ECE < 0.05 for all production models; calibration reliability diagrams available  
**Estimated effort:** 1-2 weeks

---

### PHASE 8 — Risk and Execution

**Goal:** Complete risk estimation and position sizing.

**Tasks:**
1. Validate `RiskPredictor` calibration (stop/target probability calibration)
2. Implement fractional Kelly position sizing
3. Add portfolio-level risk constraints
4. Evaluate `RLExecutionAgent` ablation (if positive: train; if not: replace with rules)
5. Validate slippage model against paper trade data

**Test gate:** Position sizing property tests pass; RiskPredictor ECE < 0.05  
**Estimated effort:** 1-2 weeks

---

### PHASE 9 — Decision Engine Hardening

**Goal:** Harden MetaDecisionEngine with all data quality gates and EV engine.

**Tasks:**
1. Implement Stage 7 (Expected Value gate) in decision hierarchy
2. Implement Stage 11 (Calibration check) in decision hierarchy
3. Add per-signal `prediction_timestamp` and `expires_at` enforcement
4. Implement signal tier system based on empirical thresholds
5. Full chaos test suite pass

**Test gate:** All 14 decision stages tested; chaos tests pass; 0 P0 gaps remain  
**Estimated effort:** 1-2 weeks

---

### PHASE 10 — Online Learning

**Goal:** Connect feedback loop from AlphaForge to OnlineLearner.

**Tasks:**
1. Build `POST /train/feedback` endpoint in ml-service2.0
2. Build AlphaForge worker job that sends trade outcomes to /train/feedback
3. Wire OnlineLearner to shadow model (not champion)
4. Implement champion/shadow promotion from online learning
5. Test rollback scenario

**Test gate:** Feedback loop functional; OnlineLearner safety limits validated  
**Estimated effort:** 1-2 weeks

---

### PHASE 11 — Monitoring

**Goal:** Full observability for production monitoring.

**Tasks:**
1. Add Prometheus metrics to ml-service2.0
2. Wire DriftMonitor to inference path (store reference distributions at training time)
3. Implement automated retraining trigger from drift alerts
4. Implement calibration drift monitoring
5. Build monitoring dashboard (Grafana or similar)

**Test gate:** All monitoring endpoints functional; drift detection verified with synthetic drift injection  
**Estimated effort:** 1-2 weeks

---

### PHASE 12 — Research Agent (Optional)

**Goal:** Build automated research agent for hypothesis generation and experiment planning.

**Tasks:**
1. Build `src/research/agent.py` using LangGraph
2. Implement experiment planning workflow
3. Implement post-trade analysis report generation
4. Implement model failure investigation workflow
5. STRICT: agent must NOT be in live signal path

**Test gate:** Agent produces reports but cannot trigger promotions or trading  
**Estimated effort:** 2-3 weeks (optional — can skip if resources constrained)

---

### PHASE 13 — Certification

**Goal:** Produce evidence that the system is production-ready.

**Tasks:**
1. Run full end-to-end validation with real data
2. Generate `reports/ml_certification.json`
3. Generate `reports/ML_SERVICE_FINAL_CERTIFICATION.md`
4. Certify all 20 acceptance criteria (A through T)
5. Document all known limitations and remaining risks

**Test gate:** All 20 acceptance criteria met  
**Estimated effort:** 1 week

---

### PHASE 14 — Shadow Deployment

**Goal:** Run ML signals in shadow mode alongside existing rule engine.

**Tasks:**
1. Deploy ml-service2.0 in shadow mode (`deployment_mode=shadow`)
2. Compare ML signals vs. AlphaForge rule engine outcomes for 20 trading days
3. Measure incremental return from ML enhancement
4. Confirm no data quality incidents
5. Confirm latency within SLA

**Decision gate:** If ML shadow performance ≤ rule engine after costs: DO NOT PROMOTE  
**Estimated duration:** 20 trading days (~4 weeks)

---

### PHASE 15 — Production Release

**Tasks:**
1. Change `deployment_mode=validated_ml_only` in production
2. Monitor first week closely
3. Keep rollback procedure documented and tested
4. Continue monitoring drift and calibration weekly

---

## 3. Rollback Procedure

If any phase introduces regression:

```bash
# 1. Switch to paper mode immediately
ML_MODE=fallback  # AlphaForge falls back to rule engine

# 2. Roll back model artifact
POST /registry/rollback/{model_id}

# 3. Verify health
GET /health
GET /v2/models/status

# 4. Investigate root cause before re-attempting promotion
```

---

*End of Migration Plan*

---

## IMPLEMENTATION STATUS ADDENDUM (2026-09-24, post-execution)

The migration/implementation phase is complete. All P0 blockers (P0-001..P0-008)
are closed and the full data→features→labels→training→validation→calibration→
backtest→risk→decision→registry→shadow→feedback→drift→self-learning→explainability
lifecycle is implemented, tested (159 new tests; 1,134 suite total passing), and
executed end-to-end against LIVE data-service2.0 real NSE data.

Current phase: **implementation COMPLETE; certification RESEARCH_READY / PAPER_READY.**
Not advanced to shadow/production migration because no trained model shows a
cost-surviving edge on real data (honest negative result). See
`reports/ml_certification.json` and `reports/ML_SERVICE_FINAL_CERTIFICATION.md`
for the authoritative status and per-criterion evidence.
