# Test Strategy
**ml-service2.0 — Test Pyramid, TDD Policy, and Quality Gates**

*Date: 2026-09-24*

---

## 1. TDD Mandate

**Test-Driven Development is mandatory for all new code in ml-service2.0.**

Process for every change:
1. Write a failing test that specifies the desired behavior
2. Implement the minimum code to make the test pass
3. Run the test to confirm it passes
4. Refactor while keeping tests green
5. Run the complete relevant test suite

**NEVER:** implement first, test later. This is especially critical for numerical financial code where incorrect behavior can be silently profitable in one regime and silently catastrophic in another.

---

## 2. Test Pyramid

```
                    ┌────────────────────────────┐
                    │     E2E / Integration      │  (5%)
                    │  Real data, live services  │
                    └────────────────────────────┘
                  ┌──────────────────────────────────┐
                  │    Contract / Data Quality       │  (10%)
                  │  Schema validation, PIT tests   │
                  └──────────────────────────────────┘
                ┌────────────────────────────────────────┐
                │    Statistical / Backtest Tests        │  (15%)
                │  Walk-forward, calibration, leakage    │
                └────────────────────────────────────────┘
              ┌──────────────────────────────────────────────┐
              │         Integration Tests (mocked)           │  (20%)
              │   Full pipeline, mocked external services    │
              └──────────────────────────────────────────────┘
            ┌────────────────────────────────────────────────────┐
            │                Unit Tests                          │  (50%)
            │  Pure functions, isolated components, property tests│
            └────────────────────────────────────────────────────┘
```

---

## 3. Current Test Status

**WARNING: Current test suite is misleading.**

| File pattern | Count | Status | Issue |
|---|---|---|---|
| `test_coverage_boost*.py` | 7 | ⚠️ INVALID | Provides line coverage with no real assertions |
| `test_*_mocked*.py` | 8 | ⚠️ WEAK | Mocks all external dependencies |
| `test_meta_decision*.py` | 2 | ✓ Good | Tests real financial logic |
| `test_feature_pipeline_pit_correctness.py` | 1 | ✓ Good | Property-based PIT tests |
| `test_model_training_purged_cv.py` | 1 | ✓ Good | Real algorithm test |
| `test_data_contracts.py` | 1 | ✓ Good | Schema validation |
| `test_drift_monitoring.py` | 1 | ✓ Good | PSI math test |
| `test_risk_predictor.py` | 1 | ✓ Good | Risk logic test |
| `test_e2e_*.py` | 2 | ⚠️ Weak | E2E but fully mocked |
| `test_performance.py` | 1 | ✓ Good | Latency benchmarks |
| `test_security.py` | 1 | ✓ Good | Auth, CORS |

**Action required:** Delete or explicitly quarantine all `test_coverage_boost*.py` files. Real coverage is ~60-65%.

---

## 4. Required Test Categories

### 4.1 Unit Tests (50% of suite)

**Financial calculation tests** — must use known analytical examples:

```python
# Example: Greeks test with known BS formula result
def test_delta_call_atm():
    """At-the-money call delta ≈ 0.5 (Black-Scholes closed form)"""
    result = compute_greeks_bs(
        spot=100, strike=100, r=0.071, t=0.25, sigma=0.20,
        option_type="call"
    )
    assert abs(result.delta - 0.5379) < 0.001  # Known analytical value

# Example: PurgedKFold property test
@given(...)
def test_no_overlap_between_train_and_val(timestamps, ...):
    """Train and validation sets must be disjoint"""
    for train_idx, val_idx in splitter.split(X, y, timestamps):
        assert len(set(train_idx) & set(val_idx)) == 0

# Example: Position sizing invariant
@given(...)
def test_position_size_never_negative(prob_target, prob_stop, ...):
    """Position size must always be >= 0"""
    size = compute_position_size(prob_target, prob_stop, ...)
    assert size >= 0
```

**Target: all numerical functions have analytical verification tests.**

### 4.2 Property Tests (using Hypothesis)

Critical invariants to property-test:

```python
# Probability invariants
@given(st.floats(0, 1), st.floats(0, 1))
def test_confidence_uncertainty_sum(conf, uncertainty):
    """confidence + uncertainty must never exceed 1.0"""
    assert confidence + uncertainty <= 1.0 + 1e-12

# PIT invariants
@given(trading_datetimes(), trading_datetimes())
def test_feature_as_of_before_prediction(feature_ts, pred_ts):
    """feature_as_of must be strictly before prediction_timestamp"""
    assume(feature_ts < pred_ts)
    vector = FeatureVector(feature_as_of=feature_ts, ...)
    # No feature in vector may come from after feature_ts

# Stop/target relationship
@given(st.floats(100, 200), st.floats(0.01, 0.10), st.floats(0.01, 0.10))
def test_stop_not_equal_to_target(price, stop_dist, target_dist):
    """Stop and target must not be equal"""
    stop = price × (1 - stop_dist)
    target = price × (1 + target_dist)
    assert stop != target

# Triple-barrier label completeness
@given(...)
def test_triple_barrier_label_in_valid_set():
    """Triple-barrier labels must be -1, 0, or 1 (or NaN)"""
    label = triple_barrier(...)
    assert label in {-1, 0, 1, float('nan')}

# Weights sum to 1
@given(...)
def test_ensemble_weights_sum_to_one(model_ids, ic_scores, regime):
    weights = weighter.compute_weights(model_ids, ic_scores, regime)
    available_sum = sum(w for w in weights.values() if w > 0)
    assert abs(available_sum - 1.0) < 1e-10

# Missing values not zero-filled
@given(...)
def test_null_oi_never_becomes_zero(option_chain_with_nulls):
    result = compute_gex(option_chain_with_nulls, spot, lot_size)
    # Strikes with null OI were excluded, not zero-filled
```

### 4.3 Contract Tests

```python
# DataService contract
def test_ohlcv_bar_rejects_zero_close():
    with pytest.raises(ValidationError):
        OHLCVBar(..., close=0.0)

def test_data_service_3m_interval_banned():
    with pytest.raises(ValueError, match="3m interval"):
        client.get_historical_bars("NIFTY", interval="3m")

# SentinelPulse contract
def test_news_context_future_as_of_rejected():
    future_context = SentinelNewsContext(as_of=datetime.now() + timedelta(hours=1), ...)
    with pytest.raises(ValidationError):
        # as_of must not be in the future

# Signal contract
def test_signal_must_have_prediction_timestamp():
    # Once prediction_timestamp is added to schema, this must not be None
    signal = Signal(...)
    assert signal.prediction_timestamp is not None
```

### 4.4 Leakage Tests

```python
# Critical: these tests prevent data snooping
def test_no_future_close_in_features(feature_matrix, bar_timestamps):
    """Feature at time t must not correlate with close at t+1 above threshold"""
    validator = LeakageValidator(threshold=0.05)
    result = validator.validate(feature_matrix, future_returns)
    assert result is None  # No PITViolationError

def test_news_as_of_before_feature_as_of(feature_vector, news_context):
    """News data must not be from after the feature construction timestamp"""
    assert news_context.as_of < feature_vector.feature_as_of

def test_no_survivorship_bias_in_universe(historical_universe, current_universe):
    """Training universe must not include stocks added after training period"""
    for symbol in training_universe:
        assert symbol in historical_universe[training_date]

def test_adjusted_prices_are_pio_correct(adjusted_bars, raw_bars):
    """Corporate action adjustments must not backfill future splits"""
    # A split on 2024-01-15 should not adjust bars from 2024-01-14
```

### 4.5 Statistical Tests

```python
# Walk-forward test
def test_walk_forward_validation_5_windows():
    validator = WalkForwardValidator(n_windows=5, ...)
    results = validator.validate(model, X, y, timestamps)
    assert results.mean_ic >= 0.02
    assert results.pct_positive_windows >= 0.6
    assert results.worst_window_ic >= -0.05

# Calibration test
def test_calibrated_probabilities_match_realized_rates():
    """95% confidence predictions should realize ~95% of the time"""
    for bucket, (pred_prob, realized_rate) in calibration_buckets.items():
        assert abs(pred_prob - realized_rate) < 0.05  # ECE within 5%
```

### 4.6 Chaos Tests

```python
@pytest.mark.chaos
async def test_data_service_503_returns_no_trade(mock_data_service_503):
    result = await meta_engine.decide(symbol="NIFTY", ...)
    assert result.action == "NO_TRADE"
    assert "DATA_SERVICE_UNAVAILABLE" in result.reason_codes

@pytest.mark.chaos
async def test_all_models_unavailable_returns_no_trade():
    outputs = [{"provenance": "unavailable"} for _ in range(7)]
    result = engine.decide(model_outputs=outputs, symbol="NIFTY")
    assert result.action == "NO_TRADE"
    assert result.abstention is True
    assert "ALL_MODELS_UNAVAILABLE" in result.reason_codes
```

---

## 5. Coverage Policy

| Category | Minimum coverage |
|---|---|
| `src/meta/` | 95% |
| `src/training/` | 90% |
| `src/features/` | 90% |
| `src/analytics/` | 90% |
| `src/models/` | 85% |
| `src/registry/` | 90% |
| `src/clients/` | 85% |
| `src/api/` | 80% |
| `Overall` | 90% |

**Coverage must represent real assertions, not mock call counts.**

---

## 6. CI/CD Test Gates

All tests must pass before any code is merged to main:

```yaml
# Required CI gates
- ruff check src/ tests/          # Linting
- mypy src/                        # Type checking
- pytest tests/ -m "unit"         # Unit tests (< 30s)
- pytest tests/ -m "integration"  # Integration tests (< 5min, mocked)
- pytest tests/ -m "pit"          # PIT correctness tests
- pytest tests/ -m "statistical"  # Statistical tests (may be slow)
# Do NOT require:
# - pytest tests/ -m "chaos"       # Chaos tests: run manually or nightly
# - pytest tests/ -m "e2e_real"    # Real data tests: run in staging only
```

---

*End of Test Strategy*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

159 new behavioural tests added with real assertions across ingestion, labels, features, dataset build, estimators, walk-forward, CPCV, training orchestrator, calibration, backtest, position sizing, expected value, lifecycle, drift, feedback, self-learning, decision trace, signal contract, and an end-to-end certification test. Full suite: 1,134 passing. The test harness (tests/conftest.py) was hardened to force the test API key so the suite is hermetic regardless of the caller's environment. Key honesty properties are tested (noise -> IC~0; confidence cannot bypass risk caps; champion never auto-mutated). Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
