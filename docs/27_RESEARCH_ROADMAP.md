# Research Roadmap
**ml-service2.0 — Quantitative Research Agenda**

*Date: 2026-09-24*

---

## 1. Research Priorities

Research follows a strict evidence-first policy:

1. Simple first: build simple model → validate → then add complexity
2. Document hypotheses before testing them
3. Document negative results — what didn't work is as important as what did
4. Every research experiment must produce a machine-readable result
5. Research findings feed into the migration plan; research does not bypass gates

---

## 2. Research Pipeline

Every research hypothesis follows this pipeline:

```
HYPOTHESIS
    └── documented in research log with: rationale, expected IC, test plan
DATASET
    └── PIT-correct, immutable, versioned
FEATURE
    └── validated against leakage, stability tests
MODEL
    └── trained with PurgedKFold, minimum 50 HPO trials
BACKTEST
    └── cost-aware, realistic execution
WALK-FORWARD
    └── minimum 5 windows, report worst window
ROBUSTNESS
    └── parameter perturbation, symbol perturbation, regime breakdown
ABLATION
    └── confirm incremental contribution of each new component
STATISTICAL TEST
    └── bootstrap CI on IC, permutation test, deflated Sharpe
DECISION
    └── PROMOTE to challenger OR document as REJECTED (with reason)
```

---

## 3. Near-Term Research Questions (Phase 4-6 support)

### Q1: Which features are most predictive for NIFTY regime classification?
- Hypothesis: VIX level + market breadth + NIFTY trend are the 3 most important features
- Test: Train XGBoost on Groups F+I, compute SHAP importance
- Success criteria: Top-3 features have stable SHAP across 5 OOS windows

### Q2: Does triple-barrier outperform fixed-horizon labels for short-horizon models?
- Hypothesis: Triple-barrier reduces label noise and improves IC for 5m-15m horizons
- Test: Train same model with both label types, compare OOS IC
- Success criteria: Triple-barrier IC > fixed-horizon IC by >= 0.005

### Q3: Does SentinelPulse news intelligence add incremental predictive value?
- Hypothesis: News features add 0.01-0.03 IC improvement for event-driven symbols
- Test: Train with Group G features vs. without; measure incremental IC
- Success criteria: IC improvement > 0.005 on same OOS test set
- Null hypothesis to document if rejected: "News features add no incremental value to market data features"

### Q4: What is the optimal volatility regime for mean-reversion vs. trend-following?
- Hypothesis: Trend-following outperforms in LOW_VOL + TRENDING; mean-reversion outperforms in HIGH_VOL + SIDEWAYS
- Test: Compute IC separately for each RegimeClassifier output × StrategySelector selection
- Success criteria: IC positive for correct strategy × regime combinations

### Q5: What is the minimum viable feature set?
- Hypothesis: 15-20 features captures 90% of predictive value; adding more reduces IC via noise
- Test: Recursive feature elimination starting from full feature set
- Success criteria: Identify feature count where IC is maximized on OOS data

---

## 4. Medium-Term Research Questions (Phase 7-9)

### Q6: Does IV skew predict direction better than IV level for index options?
- Hypothesis: PCR + skew is more predictive than ATM IV alone for NIFTY
- Test: Ablation study comparing D-group derivatives subsets

### Q7: Is fractional Kelly the optimal position sizing approach?
- Hypothesis: Fractional Kelly (25%) produces higher risk-adjusted returns than fixed 2% sizing
- Test: Simulate different sizing approaches on paper trade history
- Note: requires outcome data from feedback loop

### Q8: Does regime-conditional calibration improve ECE vs. global calibration?
- Hypothesis: Calibrating separately per regime improves calibration quality
- Test: Compare ECE(global Platt) vs. ECE(per-regime Platt) on OOS data

### Q9: What is the IC half-life for NSE intraday signals?
- Hypothesis: IC for 5m predictions decays to near-zero within 30 minutes
- Test: Compute IC as a function of prediction age; fit exponential decay curve
- Implication: Sets signal expiry policy

### Q10: Does cross-sectional ranking add value over individual stock models?
- Hypothesis: Cross-sectional StockRanker IC > individual stock IC on same data
- Test: Compare ranking IC vs. individual prediction IC on F&O universe

---

## 5. Long-Term Research Questions (Phase 10+)

### Q11: Can online learning from NSE intraday data materially improve IC within a session?
- Hypothesis: Intraday regime shifts are predictable from volume/OI patterns; online updating captures this
- Requires: Feedback loop operational, 3+ months of outcome data

### Q12: Does the LLM Research Agent generate useful hypotheses?
- Hypothesis: Agent-generated feature hypotheses have IC > 0 at rate > 30%
- Test: Agent proposes 10 hypotheses; researcher blindly validates each
- Requires: Research agent implementation (Phase 12)

### Q13: What is the optimal ensemble architecture for NSE F&O?
- Hypothesis: 4-model ensemble (Regime + Risk + Ranker + Price) outperforms 7-model ensemble
- Test: Ablation over ensemble compositions; compare OOS IC and Sharpe

---

## 6. Research Log Format

Every experiment is logged in `experiments/` as:

```json
{
    "experiment_id": "EXP-001",
    "hypothesis": "Triple-barrier labels outperform fixed-horizon labels",
    "status": "COMPLETED",
    "started_at": "2026-10-01",
    "completed_at": "2026-10-14",
    "dataset_version": "v1.0.0-nsefno-20220101-20251231",
    "model": "LightGBM-v1",
    "label_type_a": "triple_barrier",
    "label_type_b": "fixed_horizon_5bar",
    "result": {
        "ic_triple_barrier": 0.034,
        "ic_fixed_horizon": 0.021,
        "ic_difference": 0.013,
        "statistical_significance": {"p_value": 0.032, "significant": true},
        "conclusion": "SUPPORTED",
        "decision": "PROMOTE triple_barrier for 5m-1h models"
    },
    "mlflow_run_id": "abc123",
    "code_sha": "f1a2b3c4"
}
```

---

*End of Research Roadmap*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Research harness delivered: scripts/run_certification.py runs the full lifecycle against LIVE data-service2.0 (real NSE data) or a clearly-labelled synthetic fallback. Open research items surfaced by the real-data run: no cost-surviving edge on daily bars (OOS IC=0, PBO=0.6, net Sharpe negative); NSE index instruments report zero traded volume which degrades volume-based features. Indicated next directions: liquid single-stock universe, intraday bars, richer/derivatives features, and an empirical SentinelPulse incremental-value study (news must not be force-included). Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
