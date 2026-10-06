# Decision Engine Specification
**ml-service2.0 — Meta-Decision Engine Design**

*Date: 2026-09-24*

---

## 1. Decision Hierarchy

Every inference request passes through a deterministic decision hierarchy. Stages are evaluated in strict order. Earlier stages can terminate the decision with NO_TRADE before later stages run.

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: DATA QUALITY GATE                                      │
│   Input: DataServiceResponse.quality                            │
│   Pass condition: signalEngineAllowed=true AND                  │
│                   DataConfidenceScore >= min_confidence (60)    │
│   On fail: NO_TRADE, reason=DATA_QUALITY_FAILURE                │
└─────────────────────────────────────────────────────────────────┘
                              │ PASS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: PIT VALIDATION                                         │
│   Input: feature_as_of, prediction_timestamp, news_as_of        │
│   Pass condition: feature_as_of < prediction_timestamp AND      │
│                   news_as_of < prediction_timestamp             │
│   On fail: NO_TRADE + raise PITViolationError                   │
└─────────────────────────────────────────────────────────────────┘
                              │ PASS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: MARKET / SESSION VALIDITY                              │
│   Input: current IST timestamp                                  │
│   Pass condition: NSE is open AND                               │
│                   not in last 15 minutes (15:00-15:15 IST)      │
│                   (unless EXIT signal)                          │
│   On fail: NO_TRADE, reason=MARKET_CLOSED                       │
└─────────────────────────────────────────────────────────────────┘
                              │ PASS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: MODEL ELIGIBILITY                                      │
│   Input: deployed models in ModelRegistry                       │
│   Pass condition: at least 3 models with lifecycle=PRODUCTION   │
│                   and provenance=TRAINED_MODEL                  │
│   Degraded mode: if deployment_mode=research/paper, allow       │
│                  HEURISTIC models (explicitly flagged)          │
│   On fail: NO_TRADE, reason=INSUFFICIENT_ELIGIBLE_MODELS        │
└─────────────────────────────────────────────────────────────────┘
                              │ PASS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 5: REGIME ASSESSMENT                                      │
│   Input: RegimeClassifier prediction                            │
│   Output: regime, regime_probability                            │
│   Note: regime informs EnsembleWeighter IC selection            │
│   On fail (model unavailable): use SIDEWAYS as default          │
└─────────────────────────────────────────────────────────────────┘
                              │ ALWAYS PASS (regime has default)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 6: DIRECTIONAL EDGE                                       │
│   Input: all model predictions (regime, ranker, strategy, risk, │
│           price, iv, execution)                                  │
│   Output: agreement_ratio, plurality_direction                  │
│   Note: WAIT votes reduce denominator but not numerator         │
│   On fail: agreement_ratio < 0.5 → triggers abstention          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 7: EXPECTED VALUE                                         │
│   Input: RiskPredictor probabilities + trade setup              │
│   Compute: EV = P(target) × payoff − P(stop) × loss − costs   │
│   On fail: EV < 0 → NO_TRADE, reason=NEGATIVE_EXPECTED_VALUE   │
│   Note: NOT YET IMPLEMENTED — needs ExpectedValueEngine         │
└─────────────────────────────────────────────────────────────────┘
                              │ EV >= 0
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 8: RISK GATE                                              │
│   Input: RiskPredictor.prob_stop_hit                            │
│   Pass condition: prob_stop_hit <= 0.65                         │
│   On fail: NO_TRADE, reason=HIGH_STOP_PROBABILITY               │
└─────────────────────────────────────────────────────────────────┘
                              │ PASS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 9: EXECUTION FEASIBILITY                                  │
│   Input: RLExecutionAgent decision                              │
│   Pass condition: action ∈ {ENTER_NOW, SCALE_IN}               │
│   On WAIT: defer to next bar                                    │
│   Note: RL agent is currently heuristic                         │
└─────────────────────────────────────────────────────────────────┘
                              │ PASS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 10: PORTFOLIO CONSTRAINTS                                 │
│   Input: AlphaForge current portfolio state (from request)      │
│   Pass condition: position fits within:                         │
│     - Max sector concentration                                  │
│     - Max single position size                                  │
│     - Daily loss limit not reached                              │
│   Note: AlphaForge enforces these, not ml-service2.0           │
└─────────────────────────────────────────────────────────────────┘
                              │ PASS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 11: CALIBRATION CHECK                                     │
│   Input: CalibrationLayer ECE for contributing models           │
│   Pass condition: ECE < 0.10 for champion models               │
│   On fail: downgrade quality_tier to WATCH                      │
│   Note: NOT YET ENFORCED — calibrators not yet fitted          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 12: ABSTENTION THRESHOLD (AbstentionPolicy)              │
│   Conditions:                                                   │
│     agreement_ratio < 0.5 → LOW_AGREEMENT                      │
│     data_quality < 0.6    → LOW_DATA_QUALITY                   │
│     mean_confidence < 0.35 → LOW_CONFIDENCE                    │
│     prob_stop_hit > 0.65  → HIGH_STOP_PROBABILITY              │
│     n_available_models < 3 → INSUFFICIENT_MODELS               │
│   On ANY condition: NO_TRADE + abstention=True                  │
└─────────────────────────────────────────────────────────────────┘
                              │ No abstention
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 13: NEWS CONFLICT CHECK (optional)                        │
│   Input: LLMNewsReasoner NewsSignal                             │
│   Pass condition: news not strongly contrary to ensemble        │
│   On conflict (confidence >= 0.7 AND direction opposite):       │
│     Override to WAIT, reason=NEWS_CONFLICT_OVERRIDE             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 14: FINAL SIGNAL                                          │
│   Output: BUY | SELL | WAIT | NO_TRADE                          │
│   With: confidence, uncertainty, agreement_ratio,               │
│         provenance, reason_codes, explainability                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Expected Value Engine (Target Implementation)

```python
class ExpectedValueEngine:
    """
    Computes risk-adjusted expected value for a proposed trade.
    
    EV formula:
        EV = P(target) × target_payoff_pct
           − P(stop) × stop_loss_pct
           − expected_cost_pct
           − expected_slippage_pct
           − uncertainty_penalty
    
    uncertainty_penalty = uncertainty × (target_payoff + stop_loss) / 2
    
    Minimum EV for trade to proceed: EV > 0 (strictly positive)
    HIGH_CONVICTION requires: EV > 2 × round_trip_cost_pct
    """
    
    def compute(
        self,
        entry: float,
        stop: float,
        target: float,
        prob_target: float,        # From RiskPredictor
        prob_stop: float,          # From RiskPredictor
        uncertainty: float,        # From MetaDecisionEngine
        cost_pct: float = 0.0010,  # Round-trip cost
        slippage_pct: float = 0.0002,  # Estimated slippage
    ) -> ExpectedValueResult:
        ...
```

---

## 3. Position Sizing

Position size is NOT simply confidence × max_position. It must account for:

```
Kelly-inspired position sizing:

kelly_f = (p_win × avg_win - p_loss × avg_loss) / avg_win

where:
    p_win = target_probability (calibrated)
    p_loss = stop_probability (calibrated)
    avg_win = (target - entry) / entry
    avg_loss = (entry - stop) / entry

Fractional Kelly (safety factor):
    position_size = kelly_f × kelly_fraction (default: 0.25)
    # Using full Kelly is reckless — fractional Kelly reduces drawdown

Hard limits:
    position_size = min(position_size, max_position_pct)
    where max_position_pct = 0.02 (2% of capital per position default)
    
    Daily loss limit: if realized_loss_today > daily_limit (e.g., 1%):
        position_size = 0 (no new positions)

Regime-conditional scaling:
    CRASH regime: position_size × 0.25
    VOLATILE regime: position_size × 0.50
    BEAR regime: short only, position_size × 0.75
```

---

## 4. Abstention Decision Tree

```
Is data available and of sufficient quality?
    NO → NO_TRADE (DATA_QUALITY_FAILURE)
    
Are timestamps PIT-correct?
    NO → NO_TRADE (PIT_VIOLATION) + alert

Is market open and within valid session?
    NO → NO_TRADE (MARKET_CLOSED)

Are >= 3 eligible models available?
    NO → NO_TRADE (INSUFFICIENT_MODELS)

Is agreement_ratio >= 0.5?
    NO → NO_TRADE (LOW_AGREEMENT)

Is mean_confidence >= 0.35?
    NO → NO_TRADE (LOW_CONFIDENCE)

Is expected value > 0 after costs?
    NO → NO_TRADE (NEGATIVE_EXPECTED_VALUE) [when EV engine implemented]

Is prob_stop_hit <= 0.65?
    NO → NO_TRADE (HIGH_STOP_PROBABILITY)

Is there a strong conflicting news signal?
    YES → WAIT (NEWS_CONFLICT_OVERRIDE)

All gates passed?
    → BUY or SELL with computed confidence, position size, stop, targets
```

---

## 5. Decision Output Guarantees

The MetaDecisionEngine provides the following mathematical guarantees:

1. `confidence ∈ [0.0, 1.0]`
2. `uncertainty ∈ [0.0, 1.0]`
3. `confidence + uncertainty ≤ 1.0`
4. `agreement_ratio ∈ [0.0, 1.0]`
5. `abstention == True` when `agreement_ratio < 0.5`
6. `abstention == True` → `action ∈ {NO_TRADE, WAIT}`
7. When all models UNAVAILABLE → `action == NO_TRADE, provenance == UNAVAILABLE`
8. When n_available < 3 → `action == NO_TRADE, reason_codes contains INSUFFICIENT_MODELS`
9. Same inputs produce same outputs (idempotent)
10. `prediction_timestamp` must be present (P0 gap — not yet implemented)

---

*End of Decision Engine Specification*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Implemented by src/meta/engine.py (MetaDecisionEngine) with: real data_quality threading (DATA_QUALITY gate, replacing the previous hardcoded 1.0), calibrated-probability consumption, abstention policy, and an explicit expected-net-edge gate (src/meta/expected_value.py) that returns INSUFFICIENT_EDGE / NO_TRADE when E[net] after costs is not positive. Confidence alone is never sufficient to trade. Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
