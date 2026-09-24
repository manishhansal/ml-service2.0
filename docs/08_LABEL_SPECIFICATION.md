# Label Specification
**ml-service2.0 — Target Construction and Label Engineering**

*Date: 2026-09-24*
*Status: TARGET specification — LabelFactory not yet implemented*

---

## 1. Label Engineering Principles

1. **Labels must be PIT-correct** — computed only from price/event data available after the horizon has elapsed
2. **Labels must account for transaction costs** — a trade with gross return < round-trip cost is a loss
3. **Labels must be horizon-specific** — 5m labels are not appropriate for 1d models
4. **Labels must carry provenance** — every label records: `feature_as_of`, `label_start`, `label_end`, `label_bar_timestamps`
5. **Naive labels are forbidden** — `future_close > current_close` is not a valid trading label (ignores costs, volatility, path)

---

## 2. Label Types

### 2.1 Fixed-Horizon Log Return (Baseline)

```
label_return_h = log(close_{t+h} / close_t) - transaction_cost_pct

Parameters:
    h: horizon in bars (e.g., 5=5 bars, 12=1h at 5m bars, 78=1d at 5m bars)
    transaction_cost_pct: round-trip cost (default: 0.10% = 10bps)
    use_vwap: bool — use VWAP as exit price instead of close

PIT fields:
    feature_as_of: close_t timestamp
    label_start: close_t timestamp  
    label_end: close_{t+h} timestamp
    label_bar_timestamps: [t, t+1, ..., t+h]

Binary variant:
    y = 1 if label_return_h > 0 else 0
    y = 1 if label_return_h > cost_threshold else (0 if label_return_h < -cost_threshold else NaN)
    # NaN = too close to zero to label — these samples MUST be dropped, not forced to 0 or 1
```

**Warning:** Fixed-horizon labels assume the position is held for exactly h bars regardless of path. This overestimates performance for strategies with stop losses.

### 2.2 Triple-Barrier Label (Primary)

The triple-barrier method (López de Prado, 2018) generates path-dependent labels that account for early exit:

```
barriers:
    upper_barrier = close_t × (1 + pt × sigma)   # take-profit
    lower_barrier = close_t × (1 - sl × sigma)   # stop-loss
    vertical_barrier = t + h bars                 # time horizon

label:
    +1 if upper_barrier hit before lower_barrier and before t+h
    -1 if lower_barrier hit before upper_barrier and before t+h
     0 if vertical_barrier hit first (neither target nor stop reached)

Parameters:
    pt: profit-taking multiplier (default: 2.0)
    sl: stop-loss multiplier (default: 1.0)  [asymmetric: allow asymmetric R:R]
    sigma: volatility estimate (ATR_14 / close_t is appropriate for intraday)
    h: maximum holding period in bars

Volatility calibration:
    sigma_fast = ATR_5 / close_t   (for 5m-15m horizons)
    sigma_slow = ATR_14 / close_t  (for 1h-1d horizons)
    sigma = regime-weighted blend when RegimeClassifier is trained

PIT fields:
    feature_as_of: close_t timestamp
    label_start: close_t timestamp
    label_end: timestamp of first barrier hit (not necessarily t+h)
    label_bar_timestamps: [t, ..., barrier_hit_t]
    barrier_type: "UPPER" | "LOWER" | "VERTICAL"
```

**Why triple-barrier matters:** A stock that rallies 2% and then drops 5% has a fixed-horizon return of -3%, but a trader with a proper stop loss would have exited at -1%. Triple-barrier captures the path.

### 2.3 Volatility-Adjusted Return

```
label = label_return_h / sigma_h

where sigma_h = sqrt(h) × realized_vol_daily  (using 20-bar rolling vol)

Effect: normalizes returns by contemporaneous volatility so the model
targets risk-adjusted rather than raw return. This prevents high-vol
regime samples from dominating.

Use case: Ranking models (StockRanker) where cross-sectional normalization
is needed.
```

### 2.4 Meta-Label (Signal Quality)

```
meta_label: binary label for WHETHER a primary model's signal is profitable

Procedure:
    1. Generate primary model predictions on training data
    2. For each prediction with confidence > threshold:
       meta_y = 1 if actual outcome matches predicted direction else 0
    3. Train secondary model to predict meta_y from same features
    
Use case: RiskPredictor can be trained as a meta-labeler on RegimeClassifier
predictions — it learns to filter out the regime classifier's false signals.
```

### 2.5 Event-Based Label

```
For news-driven models (SentinelPulse integration):

label_window:
    event_start: NewsEvent.eventTimestamp
    measurement_window: [+5m, +30m] | [+1h, +4h] | [+1d]
    
label: abnormal_return = 
    (close_{event+window} - close_{event}) / close_{event}
    - expected_return_from_market_model(window)  # Beta-adjusted

Use case: Measuring market reaction to news events for SentinelPulse model training.
SentinelPulse already stores this in NewsMarketReaction table.
```

---

## 3. Horizon Mapping

| Horizon | Label type | Bars | Transaction cost | Min history |
|---|---|---|---|---|
| 1m scalp | Fixed-horizon return | 1 bar | 8bps | 500 bars |
| 5m | Triple-barrier | 5 bars | 8bps | 1000 bars |
| 15m | Triple-barrier | 15 bars | 10bps | 1000 bars |
| 1h | Triple-barrier | 12 bars (5m) | 12bps | 2000 bars |
| 1d | Triple-barrier | 78 bars (5m) | 15bps | 500 days |

**Note:** Transaction cost increases with horizon because longer holds mean more potential adverse execution. 10bps is the current flat assumption — this must be replaced with empirically measured costs from paper trading.

---

## 4. Label Quality Requirements

### 4.1 Class Balance

```python
# Check for extreme imbalance
positive_rate = (y == 1).mean()
assert 0.3 < positive_rate < 0.7, (
    f"Label imbalance {positive_rate:.2%} — recalibrate barriers or reject"
)

# For three-class triple-barrier
# Acceptable: +1: 25-40%, -1: 25-40%, 0: 20-50%
# Reject: any class < 15% or > 60%
```

### 4.2 PIT Validation

```python
# Every label must be strictly future-dated from features
assert label_end > feature_as_of
assert label_start >= feature_as_of

# No bar in label_bar_timestamps is accessible before feature_as_of
for bar_ts in label_bar_timestamps:
    assert bar_ts > feature_as_of
```

### 4.3 Cost Adjustment

```python
# Net label must account for round-trip cost
gross_return = log(exit_price / entry_price)
net_return = gross_return - transaction_cost_pct
label_return = net_return  # use net, not gross
```

### 4.4 Handling Ambiguous Cases

```python
# Samples too close to zero should be dropped (not labeled)
dead_zone = 0.5 * transaction_cost_pct

if abs(net_return) < dead_zone:
    label = NaN  # DROP this sample — do not force to 0 or 1
```

---

## 5. Label Schema

Every labeled sample must carry:

```python
class LabeledSample(BaseSchema):
    # Identity
    sample_id: str                     # UUID v4
    symbol: str
    timeframe: str
    label_type: str                    # "triple_barrier" | "fixed_horizon" | "meta"
    
    # Feature data
    feature_as_of: datetime            # UTC — features extracted at this time
    
    # Label data
    label_start: datetime              # UTC — label observation starts
    label_end: datetime                # UTC — label observation ends
    label_value: float                 # +1, -1, or 0 (or NaN for drop)
    label_return_gross: float          # Raw return before costs
    label_return_net: float            # Return after transaction costs
    
    # Triple-barrier specific
    barrier_type: str | None           # "UPPER" | "LOWER" | "VERTICAL"
    barrier_hit_price: float | None
    barrier_pt: float | None           # Profit-taking multiplier used
    barrier_sl: float | None           # Stop-loss multiplier used
    sigma_at_entry: float | None       # Volatility estimate used
    
    # Costs
    transaction_cost_pct: float        # Assumed round-trip cost
    slippage_pct: float                # Assumed slippage
    
    # Dataset provenance
    dataset_version: str               # Version hash of the dataset
    feature_schema_version: str
    label_schema_version: str
    data_provider: str                 # AngelOne | Upstox | etc.
```

---

## 6. LabelFactory Interface (Target Implementation)

```python
class LabelFactory:
    """
    Constructs PIT-correct training labels from OHLCV data.
    All methods return LabeledSample records with full provenance.
    """
    
    def fixed_horizon(
        self,
        ohlcv: pd.DataFrame,  # [timestamp, open, high, low, close, volume]
        horizon: int,          # bars ahead
        cost_pct: float = 0.0010,
        dead_zone_multiple: float = 0.5,
    ) -> pd.Series:            # net returns, NaN for dead zone
        """Generate fixed-horizon net returns."""
        ...
    
    def triple_barrier(
        self,
        ohlcv: pd.DataFrame,
        pt: float = 2.0,       # profit-taking multiplier
        sl: float = 1.0,       # stop-loss multiplier
        max_bars: int = 20,    # maximum holding period
        sigma_window: int = 14,  # ATR window for barrier calibration
        cost_pct: float = 0.0010,
    ) -> pd.DataFrame:         # columns: label, barrier_type, label_end, return_net
        """Generate triple-barrier labels."""
        ...
    
    def meta_label(
        self,
        primary_model_predictions: pd.Series,  # +1/-1 predictions
        ohlcv: pd.DataFrame,
        horizon: int,
        cost_pct: float = 0.0010,
    ) -> pd.Series:            # binary: 1=correct direction, 0=wrong
        """Meta-labeling: label whether primary model's signal is profitable."""
        ...
    
    def volatility_adjusted(
        self,
        ohlcv: pd.DataFrame,
        horizon: int,
        vol_window: int = 20,
        cost_pct: float = 0.0010,
    ) -> pd.Series:            # vol-adjusted returns
        """Volatility-normalized label for ranking models."""
        ...
```

---

*End of Label Specification*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Implemented by src/data/labels.py — LabelFactory (fixed_horizon, triple_barrier, vol_adjusted_barrier, meta_label) with MAE/MFE, realized cost, and label_quality_report() (class balance, autocorrelation). PIT-safe: labels look forward and the final `horizon` rows are unresolved. Tested in tests/test_label_factory.py. Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
