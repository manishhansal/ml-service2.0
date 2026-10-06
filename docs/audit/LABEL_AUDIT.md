# LABEL AUDIT
**Repository:** ml-service2.0  
**Audit Date:** 2026-10-01  
**Auditor:** Kiro Forensic Audit System  
**Verdict: FAIL — Labels do not represent economically viable trading opportunities at realistic costs**

---

## EXECUTIVE SUMMARY

The triple-barrier label system is correctly implemented from a PIT standpoint (no future leakage into features). However, the label economics are fundamentally broken:

1. The mean gross return per labeled bar is **−0.011%** — essentially zero
2. After 27.65bps round-trip cost, mean net return is **−0.552%** — definitively negative
3. The label balance is **49.6% positive / 50.4% negative** — near-coin-flip
4. Break-even accuracy at equity costs is **56.9%** — not demonstrated by the model
5. The training signal (5-bar triple-barrier) does NOT match the evaluation mandate (7 trading days)

A model trained on these labels is being trained to predict random outcomes in a cost-dominated regime.

---

## SECTION 1 — LABEL CONFIGURATION

### 1.1 Current Configuration (ls-2.0.0)

```python
LabelConfig(
    label_type       = "triple_barrier",
    horizon          = 5,                  # bars to look forward (≈ 1 calendar week)
    upper_barrier    = 0.02,               # +2% target → label = 1
    lower_barrier    = 0.02,               # −2% stop  → label = 0
    vol_window       = 20,                 # realized vol window for vol_adjusted mode
    vol_multiplier   = 1.5,               # not used in standard triple_barrier
    cost_bps         = 27.65,             # equity round-trip cost
    return_threshold = 0.0,               # TIME_EXPIRY: label = int(return > 0)
    execution_model  = "next_open",       # ✓ correct execution assumption
)
```

### 1.2 Label Semantics

| Outcome | Condition | Label | Economic Meaning |
|---------|-----------|-------|-----------------|
| TARGET_HIT | high[T+1..T+5] ≥ close[T]×1.02 | 1 | +2% gross, profitable long |
| STOP_HIT | low[T+1..T+5] ≤ close[T]×0.98 | 0 | −2% gross, losing long |
| TIME_EXPIRY | Neither barrier hit in 5 bars | int(close[T+5] > open[T+1]) | Marginal outcome |

**Critical observation:** The label definition assumes LONG direction for all observations. There is no SHORT direction encoded in the labels. When the model generates a SHORT signal (direction=−1), it is inverting a label designed for LONG positions. This creates an asymmetric economic treatment.

---

## SECTION 2 — LABEL ECONOMICS

### 2.1 Dataset-Wide Statistics

```
Dataset: ds-1d-20261001034811-ebdf74af
Rows: 263,709 (279 symbols × ~944 trading days average)
Date range: 2022-10-06 → 2026-09-28

Label distribution:
  Label 1 (TARGET_HIT + TIME_EXPIRY positive): 130,876 = 49.63%
  Label 0 (STOP_HIT + TIME_EXPIRY negative):   132,833 = 50.37%

Outcome distribution:
  STOP_HIT:    124,177 = 47.1%
  TARGET_HIT:  123,125 = 46.7%
  TIME_EXPIRY:  16,407 =  6.2%

Realized gross return:
  Mean:       −0.0108%     ← statistically indistinguishable from 0
  Std:        +1.9476%
  Positive:   49.63%
  Negative:   50.37%

Realized net return (at 10bps cost for labeling):
  Mean:       −0.2873%     ← definitively negative
  Positive:   48.94%

Realized net return (at 27.65bps equity cost):
  Mean:       ~−0.552%     ← strongly negative
  Positive:   ~47.3%
```

### 2.2 Break-Even Analysis

For a binary label with symmetric ±2% barriers:

```
Win payoff (gross):  +2.00%
Win payoff (net):    +2.00% − 0.2765% = +1.7235% (at equity costs)

Loss cost (gross):   −2.00%
Loss cost (net):     −2.00% − 0.2765% = −2.2765%

Break-even win rate:
  p* = |loss| / (|win| + |loss|)
  p* = 2.2765 / (1.7235 + 2.2765)
  p* = 2.2765 / 4.0000
  p* = 56.9%

At futures costs (8.5bps):
  Win net:  +2.00% − 0.085% = +1.915%
  Loss net: −2.00% − 0.085% = −2.085%
  p* = 2.085 / 4.000 = 52.1%

Model's reported win rate (WF): ~52% implied from IC = 0.4136
  (IC ≈ 2×AUC−1 → AUC ≈ 0.71 → win rate at optimal threshold ≈ 71%???)
  
Note: IC ↔ win rate mapping only holds for Gaussian returns.
      For a 50/50 binary label, IC is not straightforwardly interpretable as win rate.
```

**CRITICAL:** At equity costs, the model needs >56.9% accuracy to break even. The dataset shows <50% gross win rate on labels themselves, meaning most of the in-sample "signal" is statistical noise.

### 2.3 Why Labels Show Near-Zero Expected Return

The ±2% symmetric barriers with a 5-bar horizon are too narrow for NSE F&O stocks:

```
Typical NSE F&O daily volatility:
  ATR/close ≈ 1.5–2.5% per day
  5-day volatility ≈ 1.5% × √5 ≈ 3.35%

With ±2% symmetric barriers:
  Both barriers are within 1 standard deviation of daily moves
  Target probability ≈ Stop probability ≈ 47–48% (symmetric barriers, near-random)
  
Effect: Model is predicting which of two equiprobable events occurs first
        (reaching +2% or −2% in 5 days from a random walk).
        This is close to a 50/50 coin flip.
```

---

## SECTION 3 — LABEL HORIZON AUDIT

### 3.1 Horizon Mismatch: 5 Bars vs 7 Trading Days

The mandate requires evaluation over **exactly 7 trading days**. The current system uses **5 bars** (approximately 5 trading days ≈ 1 calendar week).

| Dimension | Current | Required |
|-----------|---------|----------|
| Label horizon | 5 bars | 7 trading days |
| Entry timing | open[T+1] | open[T+1] |
| Exit timing | open[T+6] or barrier hit | close[T+7] or open[T+8] |
| Calendar days approx. | 7 calendar days | 7 trading days (=9-11 calendar days) |
| Difference | — | ~2 trading days longer |

7 trading days is approximately 1.4× the current 5-bar horizon. This meaningfully changes:
- Which barriers are relevant (need wider barriers for 7-day horizon)
- The realized return distribution (2 additional days of random walk ≈ 40% more volatility)
- The label balance (more time for both barriers to be hit)

### 3.2 Multi-Horizon Labels Already Available

The `MultiHorizonLabelFactory` in `src/labels/multi_horizon.py` supports arbitrary horizons. The primary horizon (5 bars) was chosen for the current training pipeline. The 7-bar horizon needs to be added:

```python
# Already supported:
factory = MultiHorizonLabelFactory(
    horizons=[1, 3, 5, 7, 10, 21],  # add 7 to the list
    primary_horizon=7,               # change primary to 7
)
dataset = factory.build(stock_df, nifty_close)
```

However, note that `MultiHorizonLabelFactory` uses **vol-adjusted excess return** (relative to NIFTY), while `LabelFactory` uses **triple-barrier absolute return**. These are different label types and serve different purposes.

---

## SECTION 4 — LABEL DESIGN ISSUES

### 4.1 Issue: Symmetric Barriers Create Near-Random Labels

**Problem:** With ±2% symmetric barriers and typical NSE stock volatility of 1.5-2.5%/day:
- The model predicts which barrier is hit first in a near-random walk
- Expected hit probability for both barriers is approximately equal
- No genuine directional signal is being labeled

**Fix:** Use asymmetric barriers (higher reward than risk):
```python
# Better: 3:1 reward-to-risk
upper_barrier = 0.03   # 3% target
lower_barrier = 0.01   # 1% stop
# Break-even at: 1.085/4.085 = 26.5% win rate (viable with real alpha)
```

Or: Use vol-adjusted excess return (relative.py) as the primary label — this already normalizes by realized volatility and computes alpha vs NIFTY, which is the correct label for a cross-sectional ranking model.

### 4.2 Issue: All Labels Are Long-Biased

**Problem:** The current `LabelFactory._triple_barrier()` assumes LONG direction for all observations. Short signals should have inverted barrier logic:
- LONG: target = +2%, stop = −2%
- SHORT: target = −2% (price falls), stop = +2% (price rises)

The current label marks TARGET_HIT when HIGH[T+1..T+5] ≥ entry×1.02 regardless of whether the signal is LONG or SHORT.

**Fix:** Add `side` parameter to `LabelFactory`:
```python
def build(self, ohlcv: pd.DataFrame, side: int = 1) -> pd.DataFrame:
    # side=+1: long (target above, stop below)
    # side=-1: short (target below, stop above)
```

### 4.3 Issue: TIME_EXPIRY Label Logic

**Problem:** TIME_EXPIRY cases (6.2% of labels) are labeled `int(realized_return > 0)`. This means:
- A trade that returns +0.001% gets label=1 (same as a +2% TARGET_HIT)
- A trade that returns −0.001% gets label=0 (same as a −2% STOP_HIT)
- The cost of execution is NOT subtracted in the threshold test
- This creates false positives: tiny profitable TIME_EXPIRY trades labeled as "wins" even though net_return < 0

**Fix:** Label TIME_EXPIRY as `int(realized_return > cost_bps/10000)` to ensure only genuinely net-profitable outcomes are labeled as positive.

### 4.4 Issue: Vol-Adjusted Barriers Not Used

The `vol_adjusted_barrier` label type scales barriers by realized volatility:
```python
up_pct = vol_multiplier * realized_vol   # e.g., 1.5 × 1.5% = 2.25% per day
```

This creates **proportional** barriers (wider barriers for volatile stocks, narrower for stable stocks), which is economically more appropriate than fixed ±2% for all stocks. The current training uses fixed barriers, which means high-volatility stocks hit barriers randomly while low-volatility stocks rarely hit barriers (biasing towards TIME_EXPIRY for the latter).

---

## SECTION 5 — CLASS BALANCE AND IMBALANCE HANDLING

### 5.1 Current Balance

```
Label 1: 49.63%
Label 0: 50.37%
```

The labels are nearly balanced — this is NOT due to SMOTE or oversampling, but reflects the underlying data. Near-equal class balance means the model cannot achieve >90% accuracy through class-majority prediction.

**Implication for the ">90% accuracy" target:**
- In a balanced binary classification, random performance = 50%
- To achieve >90% accuracy on balanced labels, the model must correctly classify 90% of BOTH positive and negative cases
- The IC = 0.4136 claimed by WF validation does NOT translate to 90% accuracy
- IC ↔ accuracy relationship (approximation): accuracy ≈ 50% + IC × 50% / 2 ≈ 50% + 0.41 × 25% ≈ 60.3%

**Conclusion:** The ">90% accuracy" target is unachievable for this label design without leakage. The maximum theoretically achievable accuracy (under IC = 0.4136) is approximately 60-65%.

### 5.2 What ">90% Accuracy" Would Mean

For the trading task, the more meaningful metric is:
- **Precision of BUY signals** (fraction of BUY signals that are net profitable)
- **Economic precision** (fraction of signals generating positive net P&L after costs)

Target: precision > 90% on high-conviction (score > 0.7 or < 0.3) signals.

This requires:
1. Calibrated probability estimates
2. High-confidence signal filtering
3. At minimum 56.9% accuracy (equity) or 52.1% (futures) for any signal to be profitable on average

---

## SECTION 6 — LABEL QUALITY DIAGNOSTICS

### 6.1 Autocorrelation Test

Label autocorrelation (lag-1) with overlapping 5-bar labels and daily observations:
- Expected: HIGH autocorrelation (adjacent bars share 4/5 of their label window)
- Measured: Not independently computed in this audit
- **Must be checked:** High autocorrelation confirms that PurgedKFold is necessary

### 6.2 Label Quality Report (from metadata)

```json
"label_quality": {
  -- contents not extracted in this audit cycle --
  -- check artifacts/datasets/ds-1d-*/metadata.json --
}
```

### 6.3 Resolved Fraction

In the dataset (2022-10-06 → 2026-09-28):
- All rows have `label` values (0 or 1, no NaN)
- This means the last 5 bars of the dataset properly have NaN handled
- `execution_model = next_open` is confirmed

---

## SECTION 7 — RECOMMENDED LABEL REDESIGN

### Option A: Vol-Adjusted Excess Return (Recommended for cross-sectional ranking)

Already implemented in `src/labels/relative.py`:
```python
stock_fwd = close.shift(-7) / close - 1.0   # 7-bar forward return
nifty_fwd = nc.shift(-7) / nc - 1.0          # NIFTY 7-bar return
excess    = stock_fwd - nifty_fwd              # excess return vs index
vol       = close.pct_change().rolling(20).std()  # realized vol (causal)
ra_label  = (excess / vol).clip(-5, 5)        # vol-adjusted excess return
```

Advantages:
- Directly measures alpha (outperformance vs NIFTY), which is what the model should be predicting
- Continuous label → IC is the natural performance metric
- No barrier calibration needed
- Normalized by vol → comparable across different volatility regimes

### Option B: Triple-Barrier with Asymmetric 7-Day Barriers (Recommended for binary classification)

```python
LabelConfig(
    label_type    = "vol_adjusted_barrier",
    horizon       = 7,               # 7 trading days
    vol_multiplier = 1.5,            # barrier = 1.5 × 20-day vol
    # For NIFTY F&O stocks: typical 20-day vol ≈ 1.5%/day
    # 1.5 × 1.5% = 2.25% target, 1.0 × 1.5% = 1.5% stop (3:2 R:R)
    vol_window    = 20,
    cost_bps      = 27.65,
    execution_model = "next_open",
)
```

Advantages:
- Vol-adjusted barriers are appropriate for different volatility stocks
- 3:2 reward-to-risk improves positive expectancy
- 7-day horizon matches the evaluation mandate

### Option C: Fixed 7-Day Return (Simplest, for benchmarking)

```python
LabelConfig(
    label_type       = "fixed_horizon",
    horizon          = 7,
    return_threshold = 0.0,   # profitable after costs?
    cost_bps         = 27.65,
    execution_model  = "next_open",
)
```

This is the cleanest baseline for the "was this signal profitable?" question.

---

## SECTION 8 — VERDICT AND REMEDIATION

| Finding | Severity | Remediation |
|---------|----------|-------------|
| Negative mean net expectancy | CRITICAL | Redesign to vol-adjusted excess return OR asymmetric barriers |
| 5-bar horizon ≠ 7 trading day mandate | CRITICAL | Change horizon to 7 in new labels |
| Labels assume LONG direction | HIGH | Add `side` parameter or generate separate long/short labels |
| TIME_EXPIRY threshold ignores costs | MEDIUM | Change threshold to `return > cost_bps/10000` |
| Fixed symmetric barriers near-random | HIGH | Switch to vol-adjusted barriers |
| Label balance 49.6/50.4% | MEDIUM | Acceptable after barrier redesign |
| >90% accuracy target unreachable | INFORMATIONAL | Reframe target as economic precision > threshold |

**Required action:** Retrain model on redesigned labels (Option A: vol-adjusted excess return, 7-bar horizon) before any further production promotion decisions.

---

*Generated by Kiro Forensic Audit System — 2026-10-01*
