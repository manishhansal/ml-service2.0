# Validation Framework
**ml-service2.0 — Model Validation, Backtesting, and Statistical Rigor**

*Date: 2026-09-24*

---

## 1. Validation Philosophy

A model is NOT successful because:
- Training loss is low
- In-sample accuracy is high
- A single backtest Sharpe is high

A model is successful ONLY when its edge survives all of the following:

1. Point-in-time (PIT) validation
2. Purged/embargoed cross-validation
3. Walk-forward validation with multiple OOS windows
4. Cost-aware execution
5. Multiple market regimes
6. Parameter perturbation
7. Feature perturbation
8. Symbol perturbation
9. Statistical significance testing
10. Baseline comparison

---

## 2. Purged K-Fold Cross-Validation

**Implementation:** `PurgedKFoldSplitter` in `src/training/purged_kfold.py`

```
Algorithm:
    1. Sort by timestamp
    2. Divide into (n_splits + 1) windows
       Window 0: burn-in (never used for validation)
       Windows 1..n_splits: validation folds
    
    For each fold k=1..n_splits:
        Training candidates: all samples BEFORE validation window
        
        Embargo: exclude training samples within embargo_days before val_start
        Purge:   exclude training samples whose label window overlaps val_start
        
        Guarantee: no training sample's label overlaps any validation sample
```

**Parameters:**
- `n_splits = 5` (minimum), `n_splits = 10` (preferred for stable estimates)
- `embargo_days = 10` (minimum enforced), increase to `label_horizon_days × 2`
- `label_horizon_days` must match the actual prediction horizon

**Critical:** The `embargo_days` and `label_horizon_days` parameters must be set correctly for each model. A 1-day model needs at least `embargo_days=5` (1 week safety margin). A 1-hour model at 5m bars needs at least `embargo_days=5` (5 trading days).

---

## 3. Walk-Forward Validation

**Target implementation:** `WalkForwardValidator` (not yet implemented)

```
Algorithm: Expanding window walk-forward

For each window w=1..W:
    Train on: all data from inception to window_start(w)
    OOS test: data from window_start(w) to window_end(w)
    
    Report per window:
        - IC (Spearman Information Coefficient)
        - Net Sharpe (after costs)
        - Max drawdown
        - Win rate
        - Turnover
    
Aggregate metrics to report:
    - Mean IC across windows
    - Median IC across windows
    - Std of IC across windows
    - Worst-window IC
    - Percentage of windows with IC > 0
    - Percentage of windows with positive net Sharpe
    - Mean max drawdown
    
Minimum windows: 5 (current production standard)
Preferred windows: 12-24 (monthly OOS periods over 1-2 years)
```

**Rejection criteria:**
- Mean IC < 0.02
- Worst-window IC < -0.05
- < 60% of windows with positive IC
- Mean net Sharpe < 0.0

---

## 4. Combinatorial Purged Cross-Validation (CPCV)

**Target implementation:** `CPCVSplitter` (partial — needs full path enumeration)

CPCV generates multiple backtest paths by using different combinations of K training folds and M test folds. This enables PBO (Probability of Backtest Overfitting) estimation.

```
Algorithm:
    1. Split data into N groups (time-ordered)
    2. Generate C(N, k) combinations of k training groups
    3. For each combination:
       - Train on k groups
       - Test on remaining N-k groups
    4. Collect N_paths backtest Sharpe ratios
    5. PBO = fraction of paths with Sharpe < 0

Recommended: N=6, k=2 → C(6,2)=15 paths
```

**Current implementation status:** `_compute_pbo()` in `TrainingPipeline` approximates PBO as the fraction of K-fold Sharpes that are negative. This is an approximation. Full CPCV requires enumerating all path combinations.

---

## 5. Backtest Engine Requirements

The current backtest in `TrainingPipeline` only computes IC and Sharpe. A proper backtest must simulate realistic execution:

```
Requirements for production backtest:

1. Transaction costs (per side):
   - NSE brokerage: 0.01%-0.03% per trade
   - STT: 0.025% on sell (equity delivery), 0.01% (intraday)
   - Exchange charges: ~0.003%
   - GST: 18% on brokerage + charges
   - SEBI fees: negligible
   TOTAL ROUND-TRIP: ~10-15bps (use 10bps for F&O scalping, 15bps for equity intraday)

2. Slippage model:
   - 5m bar: assume 0.02% slippage (half-spread approximation)
   - More accurate: linear market impact = volume_fraction × impact_coefficient
   - For small positions: 1bps slippage is reasonable

3. Execution assumptions:
   - Entry at NEXT bar's open (not at signal bar's close)
   - Exit at stop price or next available bar if gapped
   - Partial fills NOT modeled initially (assume full fill)

4. Position constraints:
   - Max position per symbol: configurable
   - Session rules: no positions held overnight for intraday models
   - Expiry: close positions before weekly expiry close

5. Market hours:
   - No entries in last 15 minutes (15:00-15:15 IST)
   - No entries in pre-open (09:00-09:15 IST)
   - Settlement on EOD for carry-overnight models
```

---

## 6. Profitability Metrics

Every model must report ALL of the following:

```python
class ModelValidationMetrics:
    # Predictive quality
    ic_mean: float              # Spearman IC (0 = no skill, >0.05 = useful)
    ic_std: float               # IC volatility
    ic_sharpe: float            # IC × sqrt(252) / IC_std (IR)
    ic_positive_pct: float      # % of periods with positive IC
    
    # Profitability (AFTER costs)
    net_pnl_pct: float          # Total % PnL after costs
    cagr: float | None          # Only if > 1 year OOS
    sharpe_net: float           # Net Sharpe (annualized)
    sortino_net: float          # Sortino ratio
    calmar: float | None        # Only if > 6 months OOS
    max_drawdown: float         # Maximum peak-to-trough drawdown
    max_drawdown_duration_days: int
    profit_factor: float        # Gross profit / Gross loss
    expectancy: float           # Mean trade PnL
    win_rate: float             # Fraction of winning trades
    avg_win_pct: float
    avg_loss_pct: float
    
    # Risk
    var_95: float               # 95% VaR (negative number)
    cvar_95: float              # 95% CVaR (expected shortfall)
    tail_ratio: float           # 95th percentile return / 5th percentile return
    
    # Execution
    turnover_annual: float      # Annual turnover ratio
    avg_holding_bars: float     # Average holding period in bars
    trade_frequency_daily: float  # Average trades per day
    
    # Regime breakdown
    sharpe_by_regime: dict[str, float]
    ic_by_regime: dict[str, float]
    pnl_by_symbol: dict[str, float]
    pnl_by_session: dict[str, float]
    pnl_by_horizon: dict[str, float]
    pnl_by_confidence_bucket: dict[str, float]  # <30%, 30-50%, 50-70%, >70%
```

---

## 7. Statistical Significance

### 7.1 Bootstrap Confidence Intervals on IC

```python
# 10,000 bootstrap samples
ic_distribution = bootstrap(ic_per_trade, n_samples=10000)
ic_95_ci = (np.percentile(ic_distribution, 2.5),
            np.percentile(ic_distribution, 97.5))

# Reject if 95% CI includes 0
assert ic_95_ci[0] > 0, "IC not statistically significant at 95% confidence"
```

### 7.2 Permutation Test

```python
# Permute labels 1000 times, compute IC
permuted_ics = [spearmanr(predictions, shuffle(actuals)) for _ in range(1000)]
p_value = mean(permuted_ics >= observed_ic)
assert p_value < 0.05, f"IC not significant: p={p_value:.3f}"
```

### 7.3 Deflated Sharpe Ratio

```python
# Account for multiple hypothesis testing
# López de Prado (2016) formula
n_trials_tested = len(study.trials)
deflated_sharpe = sharpe_net * sqrt(1 - skewness_correction) * correction_factor(n_trials)

# Require deflated Sharpe > 0 at 95% confidence
```

### 7.4 Multiple Testing Correction

When testing N features or N models:
- Apply Bonferroni correction: `alpha_corrected = alpha / N`
- Or Benjamini-Hochberg for FDR control
- Document how many hypotheses were tested

---

## 8. Regime Robustness Testing

Every model must be tested separately in each regime:

```
Regimes to test (must have >= 200 labeled samples per regime):
    STRONG_BULL:  trending up, low vol
    BULL:         mild uptrend
    SIDEWAYS:     no trend, low vol
    VOLATILE:     high vol, unclear direction
    BEAR:         downtrend
    CRASH:        panic, very high vol

For each regime:
    - Report IC, Sharpe, win rate
    - Flag any regime where model has negative mean IC
    - Flag any regime where model significantly underperforms baseline

A model that works only in BULL regimes MUST NOT be used in BEAR or CRASH conditions.
The RegimeClassifier gates model deployment to appropriate regimes.
```

---

## 9. Validation Checklist

Before any model proceeds from CHALLENGER to SHADOW:

- [ ] PIT correctness validated (no look-ahead in features or labels)
- [ ] PurgedKFold with minimum 5-day embargo: pass
- [ ] Walk-forward validation (minimum 5 OOS windows): pass
- [ ] Mean IC >= 0.02 across OOS windows
- [ ] Net Sharpe >= 0.0 after costs across OOS windows
- [ ] PBO <= 0.50 (bootstrap or CPCV estimate)
- [ ] Beats naive momentum baseline on OOS IC
- [ ] Beats buy-and-hold baseline on net Sharpe
- [ ] Calibration: ECE < 0.05 on OOS data
- [ ] Statistical significance: IC bootstrap CI excludes 0 at 95%
- [ ] Regime breakdown: IC positive in at least 4/6 regimes
- [ ] Parameter perturbation: +/-20% hyperparameter change produces < 30% IC degradation
- [ ] Feature perturbation: removing any single feature produces < 20% IC degradation
- [ ] Symbol perturbation: OOS IC stable across different symbols
- [ ] Promotion decision documented in AuditLogger

---

*End of Validation Framework*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Implemented by src/training/walk_forward.py (WalkForwardValidator, >=5 anchored OOS windows, purge + embargo, worst-window reporting), src/training/cpcv.py (real CombinatorialPurgedCV with C(N,k) path enumeration + PBO distribution), and src/training/purged_kfold.py. Leakage validation runs per-symbol in src/data/dataset_builder.py. Honesty verified by test: noise yields IC~0; a learnable signal yields IC>0. Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
