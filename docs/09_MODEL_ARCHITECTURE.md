# Model Architecture
**ml-service2.0 — Model Zoo Design and Specifications**

*Date: 2026-09-24*

---

## 1. Model Design Principles

1. **Simpler models that survive OOS validation are preferred over complex ones that don't**
2. **Every model complexity increase must demonstrate incremental OOS value**
3. **All model probabilities must be calibrated before production**
4. **No model operates in isolation** — all contribute to the MetaDecisionEngine ensemble
5. **Heuristic fallback is explicitly labelled** — `provenance=HEURISTIC` not `trained_model`
6. **Deep learning and RL are only justified after ablation confirms value**

---

## 2. Baseline Models (Must Beat Before Promoting Advanced Models)

These baselines establish the minimum bar. Any ML model must clearly beat all relevant baselines after costs.

| Baseline | Description | Implementation |
|---|---|---|
| `buy_and_hold` | Long only, full period | Fixed 100% position |
| `random_50_50` | Random 50/50 long/short | Monte Carlo distribution |
| `naive_momentum` | Long when ret_5 > 0, else short | Simple sign rule |
| `naive_mean_reversion` | Opposite of ret_5 direction | Simple contrarian |
| `vwap_rule` | Long when close > VWAP, else short | Common heuristic |
| `current_alphaforge_rule` | Existing NSE 8-factor score | Current production rule |

**Acceptance criteria:** ML model IC > baseline IC by margin ≥ 0.005 on OOS data (after costs).

---

## 3. Model Specifications

### 3.1 RegimeClassifier

**Purpose:** Classify current market regime from NIFTY/macro features  
**Algorithm (target):** XGBoost multi-class classifier  
**Current state:** Heuristic rule-based  
**Input features:** Groups F (regime) + I (market context) + D.vix  

```
Classes (6):
    STRONG_BULL: trending up, low vol, broad participation
    BULL: trending up, moderate vol
    SIDEWAYS: no clear trend, low vol
    VOLATILE: high vol, unclear direction
    BEAR: trending down, moderate vol
    CRASH: sharp down, very high vol, panic

Output:
    regime: str (predicted class)
    confidence: float [0, 1] (calibrated probability of predicted class)
    probabilities: dict[str, float] (all 6 class probabilities, sum=1)
    
Hyperparameter search space:
    max_depth: [3, 8]
    learning_rate: [0.001, 0.1] (log scale)
    n_estimators: [100, 500]
    subsample: [0.6, 1.0]
    colsample_bytree: [0.6, 1.0]
    
Label construction:
    - Manual labeling via regime tagging of historical NIFTY data
    - Alternatively: K-Means clustering on volatility + trend features
    - HMM alternative: Gaussian HMM with 6 states
    
Validation:
    - Cannot use IC directly (multi-class problem)
    - Metric: macro F1 score across all 6 regimes
    - Walk-forward: must correctly classify at least 60% in each OOS window
    - Regime continuity: predicted regime sequence should not oscillate rapidly
```

### 3.2 StockRanker

**Purpose:** Rank NSE F&O stocks by outperformance probability over 1-5 day horizon  
**Algorithm (target):** LightGBM LambdaRank (Learning to Rank)  
**Current state:** Heuristic factor scoring  
**Input features:** Groups A + B + D + E (cross-sectional)  

```
Target label:
    - Volatility-adjusted 5-bar return (Group A label_type=volatility_adjusted)
    - Cross-sectional rank within universe at each timestamp
    
Output:
    rankings: list[{symbol, score, rank, factors}]
    
LambdaRank objective: directly optimizes NDCG@K
Alternative: Pairwise ranking (LightGBM pairwise objective)

Hyperparameter search space:
    num_leaves: [31, 255]
    learning_rate: [0.001, 0.1] (log scale)
    n_estimators: [100, 1000]
    min_child_samples: [5, 100]
    subsample: [0.6, 1.0]
    
Validation metric: Spearman IC between predicted rank and realized rank
Acceptance: Mean IC >= 0.03 across 5 OOS folds
```

### 3.3 StrategySelector

**Purpose:** Select optimal trading strategy given market regime and technical context  
**Algorithm (target):** CatBoost multi-class classifier  
**Current state:** Heuristic rule-based  
**Input features:** Groups A (momentum/vol) + D (IV regime) + F (regime) + H (session)  

```
Classes (8):
    BREAKOUT, MOMENTUM, TREND_FOLLOWING, MEAN_REVERSION,
    VWAP_BOUNCE, RANGE_TRADING, SCALPING, VOLATILITY_BREAKOUT

Label construction:
    Challenge: which strategy "should have" been used at each historical bar?
    Approach 1: Label by regime (BULL → MOMENTUM, SIDEWAYS → MEAN_REVERSION, etc.)
    Approach 2: Walk-forward evaluation — at each bar, compute which strategy
                had the best risk-adjusted return over the next horizon
                (computationally expensive but more rigorous)
    Recommended: Approach 2 for training, Approach 1 as meta-label validation
    
Output:
    strategy: str (selected strategy)
    confidence: float [0, 1]
    alternatives: list[{strategy, probability}]
    
Validation: Strategy labels from historical backtesting outcomes
```

### 3.4 RiskPredictor

**Purpose:** Estimate stop-hit and target-hit probabilities for a given trade setup  
**Algorithm (target):** XGBoost binary classifiers (separate stop and target models)  
**Current state:** Heuristic rule-based  
**Input features:** Groups A (ATR/vol) + D (IV/PCR) + F (regime) + trade setup features  

```
TWO separate models:
    Model A: P(stop_hit)   — probability stop is reached before target or horizon
    Model B: P(target_hit) — probability target is reached before stop or horizon

Constraint: P(stop_hit) + P(target_hit) <= 1.0 (validated by property test)

Trade setup features (additional inputs):
    entry_price, stop_loss, target_price, atr, regime, vix, rsi, adx
    
Label construction:
    Use triple-barrier method — barrier_type is the label:
    LOWER → stop_hit=1, target_hit=0
    UPPER → stop_hit=0, target_hit=1
    VERTICAL → stop_hit=0, target_hit=0
    
Output:
    prob_stop_hit: float [0, 1]
    prob_target_hit: float [0, 1]
    expected_drawdown_pct: float
    suggested_position_size_pct: float  # Kelly-inspired but bounded
    risk_score: float [0, 10]           # 0=low risk, 10=max risk
    
Position sizing formula:
    kelly_fraction = (p_win × win_size - p_loss × loss_size) / win_size
    bounded_fraction = min(kelly_fraction, max_position_pct) × conviction_scalar
    # Never exceed 5% of risk budget on single position
```

### 3.5 PriceForecaster

**Purpose:** Classify 1-hour ahead price regime (bull/bear/flat) from OHLCV bars  
**Algorithm options:**
- Option A: LightGBM on 60-bar features (simple, fast)
- Option B: TFT via darts (deep learning, high complexity)
- **Recommendation: Start with Option A, only add Option B if ablation shows improvement**

```
Current state: Pure heuristic (rule-based regime from momentum/vol)
Target: LightGBM on engineered features from last 60 bars

Input: last_60_bars: list[list[float]] (shape [60, 9])
       Columns: [open, high, low, close, volume, vpin, atm_iv, pcr, oi_buildup]

Output:
    regime: "bull" | "bear" | "flat"
    probability: float [0, 1]
    q10: float   — 10th percentile expected move
    q90: float   — 90th percentile expected move
    
Label: Triple-barrier label over next 12 bars (1h at 5m resolution)
```

### 3.6 IVRegimeClassifier

**Purpose:** Classify current implied volatility regime  
**Algorithm (target):** LightGBM or simple threshold model on IV percentile  
**Current state:** Heuristic percentile-based  

```
Classes (3):
    CRUSH: IV collapsing, below 20th percentile
    STABLE: IV near historical mean, 20-80th percentile
    SPIKE: IV expanding, above 80th percentile

Input features:
    atm_iv, iv_rank, iv_change_1d, vix_change, realized_vol_20
    
This model may remain heuristic since IV regime classification is
well-defined by the above thresholds. Evaluate whether ML adds
material improvement over percentile-based classification.
```

### 3.7 RLExecutionAgent

**Purpose:** Determine optimal entry/exit timing within a bar  
**Algorithm (target):** PPO or SAC via stable-baselines3  
**Current state:** Heuristic deterministic policy  

**⚠️ WARNING:** The RL agent should NOT be prioritized until:
1. All other models are trained and validated
2. Ablation study confirms RL beats simple rule-based execution after costs
3. RL environment is validated for PIT correctness (no look-ahead in reward)

```
State space (10 features):
    unrealized_pnl_pct, time_in_trade_minutes, regime_encoded,
    volume_ratio, price_vs_vwap, atr_norm, momentum,
    iv_regime_encoded, news_impact_score, current_risk_score

Action space (7 discrete actions):
    ENTER_NOW, WAIT, SCALE_IN, PARTIAL_EXIT, FULL_EXIT,
    TIGHTEN_STOP, TRAIL_STOP

Reward function:
    r_t = risk_adjusted_pnl_t - slippage_cost - holding_cost
    Must NOT include look-ahead: reward computed from actual execution prices

Evaluation:
    Compare vs. always_enter_at_signal baseline after slippage
    Require improvement > 10bps annualized to justify complexity
```

---

## 4. Ensemble Architecture

```
MetaDecisionEngine
    │
    ├── Input: 7 model outputs (action, confidence, direction, provenance)
    │
    ├── Step 1: Partition (available vs UNAVAILABLE)
    ├── Step 2: Quorum check (< 3 → NO_TRADE)
    ├── Step 3: CalibrationLayer (Platt/Isotonic per model)
    ├── Step 4: EnsembleWeighter (IC-proportional per regime)
    │           │
    │           IC registry: {model_id: {regime: ic_score}}
    │           Population: from TrainingPipeline results
    │           Default: equal weight (fallback only)
    │
    ├── Step 5: Directional agreement_ratio
    │           (BUY votes / total available — WAIT reduces denominator)
    │
    ├── Step 6: Weighted ensemble score
    ├── Step 7: AbstentionPolicy (5 conditions)
    ├── Step 8: LLMNewsReasoner (FinBERT — optional)
    ├── Step 9: News conflict override (if strong contrary signal)
    ├── Step 10: Final action determination
    ├── Step 11: Confidence = mean_confidence × agreement_ratio
    ├── Step 12: Weakest-link provenance
    └── Step 13: XAI explainability + ConfidenceDecomposition
```

### 4.1 Ensemble Weighting Policy

| Condition | Weighting Approach |
|---|---|
| IC registry populated | IC-proportional, [0.05, 0.40] per model, sum=1 |
| IC registry empty | Equal weight: 1/n_available each |
| Model IC < 0.05 | Floor weight: 0.05 |
| Model UNAVAILABLE | Weight: 0.0 (excluded) |
| n=2 models | MAX_WEIGHT cap relaxed (cannot both be ≤0.40 and sum=1) |

### 4.2 Abstention Conditions

| Condition | Threshold | Abstention Code |
|---|---|---|
| agreement_ratio | < 0.5 | LOW_AGREEMENT |
| data_quality | < 0.6 | LOW_DATA_QUALITY |
| mean_confidence | < 0.35 | LOW_CONFIDENCE |
| prob_stop_hit | > 0.65 | HIGH_STOP_PROBABILITY |
| n_available_models | < 3 | INSUFFICIENT_MODELS |

---

## 5. Model Zoo Expansion Plan

### Phase 4 additions (after baseline models validated):
- **Volatility Model:** predict realized vol over horizon (quantile regression)
- **MeanReversionStrength:** predict R-squared of mean reversion tendency
- **ExpiryPressureModel:** options expiry flow prediction

### Phase 5 additions (after Phase 4 validated):
- **CrossSectionalRanker:** universe-level ranking with cross-sectional features
- **NewsImpactModel:** predict direction and magnitude of news-driven moves
- **RegimeTransitionModel:** predict probability of regime change in next N bars

### Principles for additions:
- Each new model must demonstrate incremental IC improvement in ablation
- No model is added just because it is technically interesting
- Heavy dependencies (PyTorch, RL) only if OOS improvement > 10bps

---

*End of Model Architecture*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Implemented by src/models/estimators.py — baselines (logistic, ridge, naive momentum, naive mean-reversion) and advanced models (LightGBM, XGBoost, CatBoost) behind a uniform interface. Champion selected by out-of-sample evidence with a parsimony tiebreak. On live real NSE data the logistic baseline matched the advanced models (complexity not earned; all showed OOS IC=0.0). Legacy per-family models load champions from the registry. RL/deep models remain research-only pending demonstrated incremental value. Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
