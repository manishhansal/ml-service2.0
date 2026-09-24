# Risk Specification
**ml-service2.0 — Risk Estimation and Position Management**

*Date: 2026-09-24*

---

## 1. Risk Model Components

### Stop/Target Probability Estimation

The RiskPredictor must output:
- `prob_stop_hit ∈ [0, 1]` — probability stop is reached before target or horizon
- `prob_target_hit ∈ [0, 1]` — probability target is reached before stop or horizon
- `prob_neither = 1 - prob_stop - prob_target` — time exit probability

**Invariant:** `prob_stop_hit + prob_target_hit <= 1.0` (verified by property test)

### Expected Adverse/Favorable Excursion

- `expected_adverse_excursion` — estimated maximum drawdown from entry before exit
- `expected_favorable_excursion` — estimated maximum gain from entry before exit
- Source: historical distribution of MAE/MFE from triple-barrier training data

---

## 2. Position Sizing Formula

```python
# Fractional Kelly (conservative)
f_kelly = (p_win × r_win - p_loss × r_loss) / r_win
f_fractional = f_kelly × kelly_fraction  # default 0.25

# Bounded by hard limits
position_size = min(
    f_fractional,
    max_single_position_pct,   # e.g., 0.02 = 2% of capital
    volatility_budget_scaling, # reduce in high-vol regime
)

# Hard floor
position_size = max(position_size, 0.0)

# Regime scaling
if regime == CRASH:    position_size *= 0.25
elif regime == VOLATILE: position_size *= 0.50
elif regime == BEAR:   position_size *= 0.75
```

---

## 3. Portfolio-Level Risk

- Maximum 5 open positions simultaneously
- Maximum 30% of risk budget in any single sector
- Maximum 10% drawdown before all positions paused
- Correlation limit: no two positions with |ρ| > 0.7

---

## 4. Risk Gate in Decision Hierarchy

`prob_stop_hit > 0.65` → NO_TRADE regardless of direction confidence

This prevents entering trades with unfavorable risk dynamics even when directional models agree.

---

*End of Risk Specification*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Implemented by src/backtest/position_sizing.py — fractional-Kelly sizing bounded by hard caps (max position weight, gross exposure, sector concentration, vol target, drawdown throttle). Model confidence can NEVER bypass a hard cap (final size = min of the Kelly suggestion and every cap; proven by test). P(stop)+P(target) <= 1 is enforced in the expected-value calc and RiskPredictor. Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
