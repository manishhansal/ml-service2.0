# Data Contracts
**ml-service2.0 — Interface Specifications and Data Contracts**

*Date: 2026-09-24*

---

## 1. Contract Principles

1. **`null != 0`** — Missing OI, IV, Greeks, bid/ask must never be zero-filled. Null means unknown; zero means zero.
2. **All timestamps are UTC** — IST conversion is the consumer's responsibility.
3. **PIT correctness** — No data timestamped after `prediction_timestamp` may be consumed.
4. **Immutable schemas** — Frozen Pydantic models; modification raises ValidationError.
5. **Quality gates are blockers** — `signalEngineAllowed=false` or `DataConfidenceScore < threshold` blocks inference, does not degrade gracefully into guesses.

---

## 2. data-service2.0 → ml-service2.0

### 2.1 Live Quote / Historical Bars Request

```
GET /v1/india/quotes/{symbol}
GET /v1/india/historical/{symbol}/{interval}?from={ISO}&to={ISO}

Headers:
    X-API-KEY: <key>

Canonical intervals: 1m | 5m | 10m | 15m | 30m | 1h | 1d | 1w | 1M
FORBIDDEN interval: 3m (ML service enforces ban)
```

### 2.2 DataServiceResponse

```python
class OHLCVBar:
    symbol: str                    # uppercase, e.g. "RELIANCE"
    timestamp: datetime            # UTC, bar close time
    open: float                    # > 0
    high: float                    # >= low
    low: float                     # > 0
    close: float                   # between low and high
    volume: float                  # >= 0
    oi: float | None               # Open Interest — NULL if unavailable
    interval: str                  # canonical interval string
    provider: str                  # e.g. "ANGEL_ONE"
    data_confidence_score: int     # 0-100

class DataQualityMetadata:
    score: int                     # 0-100 (DataConfidenceScore)
    signal_engine_allowed: bool    # False → block inference
    is_stale: bool
    staleness_seconds: int | None
    missing_fields: list[str]      # e.g. ["oi", "iv"]
    provider: str

class DataServiceResponse:
    symbol: str                    # uppercase
    bars: list[OHLCVBar]
    quality: DataQualityMetadata
    as_of: datetime                # UTC — data freshness timestamp
    provider: str
    market_status: str             # e.g. "OPEN", "CLOSED", "PRE_OPEN"
```

### 2.3 Quality Gate Semantics

| Field | Value | ML Service Action |
|---|---|---|
| `signal_engine_allowed` | `false` | Raise `SignalEngineNotAllowedError` → NO_TRADE, provenance=UNAVAILABLE |
| `data_confidence_score` | < `min_confidence_score` (default 60) | Raise `LowDataConfidenceError` → NO_TRADE, provenance=INSUFFICIENT_EVIDENCE |
| `data_confidence_score` | 60-79 | WARN; use data; set provenance=INSUFFICIENT_EVIDENCE |
| `data_confidence_score` | >= 80 | OK; provenance may be TRAINED_MODEL |
| `is_stale` | `true` | WARN; downgrade confidence; check `staleness_seconds` vs. horizon |
| `missing_fields` | contains "oi" | Use null, never substitute 0 |
| `missing_fields` | contains "iv" | Use null, never substitute 0 |

### 2.4 Option Chain Contract

```python
# Consumed for Greeks, GEX, VPIN, RiskPredictor
class OptionChainEntry:
    strike: float
    expiry: date
    option_type: str               # "CE" or "PE"
    ltp: float | None
    bid: float | None              # NULL if market closed
    ask: float | None              # NULL if market closed
    iv: float | None               # Implied volatility — NULL if uncomputable
    oi: int | None                 # Open Interest in lots — NULL if unavailable
    oi_change: int | None          # OI change from previous session
    volume: int | None
    delta: float | None            # NULL when IV unavailable
    gamma: float | None
    theta: float | None
    vega: float | None
```

**Critical:** `iv=None` means IV could not be computed. `iv=0.0` means IV is literally zero — which is financially impossible for a live option. Treat `iv=0.0` as data quality failure.

---

## 3. sentinelpulse → ml-service2.0

### 3.1 News Context Request

```
GET /api/v1/alphaforge/news-context/{instrument}

Headers:
    Authorization: Bearer <api_key>
```

### 3.2 SentinelNewsContext

```python
class SentinelSentimentBreakdown:
    overall: float                 # [-1, 1] composite sentiment
    market: float                  # [-1, 1] market-level sentiment
    company: float                 # [-1, 1] company-specific sentiment
    macro: float                   # [-1, 1] macro sentiment
    risk: float                    # [-1, 1] risk appetite signal

class SentinelNewsContext:
    instrument: str                # uppercase instrument symbol
    as_of: datetime                # UTC — PIT boundary for news features
                                   # MUST be < prediction_timestamp
    news_impact_score: float       # [0, 1] — mandatory
    impact_direction: str          # BULLISH | BEARISH | NEUTRAL — mandatory
    impact_confidence: float       # [0, 1] — mandatory
    sentiment: SentinelSentimentBreakdown  # all 5 fields mandatory
    latest_event: dict | None      # Most recent structured event
    market_regime: str | None      # SentinelPulse's regime classification
    event_tags: list[str]          # Event type labels
    news_velocity: float | None    # Article publication rate vs. baseline
    novelty_score: float | None    # How novel is this news [0, 1]
    source_quality: float | None   # Weighted source tier score [0, 1]
```

### 3.3 Mandatory Fields

If any of these fields are missing or null, SentinelPulseClient treats the response as equivalent to SentinelPulse being unavailable:
- `news_impact_score`
- `impact_direction`
- `impact_confidence`
- `sentiment.overall`
- `sentiment.market`
- `sentiment.company`
- `sentiment.macro`
- `sentiment.risk`

### 3.4 PIT Validation Rule

```python
# MUST be enforced before consuming news features
assert context.as_of < prediction_timestamp, (
    f"PIT violation: news as_of {context.as_of} >= "
    f"prediction_timestamp {prediction_timestamp}"
)
```

### 3.5 ML Feature Vector from SentinelPulse

```
GET /api/v1/ml/features/asset/{assetId}

Returns: NewsFeature record with computedAt (PIT boundary)
Fields: featureVector (JSON blob), featureType, window, value, momentum, baseline
```

---

## 4. ml-service2.0 → alpha-forge

### 4.1 MetaDecision Response

```python
class MetaOutput:  # (actual Pydantic schema)
    action: str                    # BUY | SELL | WAIT | NO_TRADE
    confidence: float              # [0, 1] calibrated confidence
    uncertainty: float             # [0, 1] — confidence + uncertainty <= 1.0
    agreement: float               # [0, 1] mirrors confidence
    agreement_ratio: float         # [0, 1] fraction of models voting plurality
    ensemble_score: float | None   # signed [-1, 1]
    reason_codes: list[str]        # structured audit codes
    contributing_models: list[str]
    abstention: bool               # True when AbstentionPolicy triggered
    provenance: PredictionProvenance  # weakest-link across models
    news_sentiment_signal: NewsSignal | None
    decomposition: ConfidenceDecomposition | None
    explainability: MetaOutputExplainability | None
    symbol: str
    # MISSING (must add): prediction_timestamp, feature_as_of, expires_at
```

### 4.2 Invariants Guaranteed by MetaDecisionEngine

```
confidence + uncertainty <= 1.0
confidence ∈ [0, 1]
uncertainty ∈ [0, 1]
agreement_ratio ∈ [0, 1]
abstention == True ⟹ action ∈ {NO_TRADE, WAIT}
all_models_unavailable ⟹ action == NO_TRADE ∧ abstention == True
n_available_models < 3 ⟹ action == NO_TRADE ∧ abstention == True
agreement_ratio < 0.5 ⟹ abstention == True  (via AbstentionPolicy)
```

### 4.3 AlphaForge Integration Rules (from ml-service2-integration.ts)

```typescript
// Rule 1: Abstention is a hard gate
if (meta.abstention || meta.action === "NO_TRADE" || meta.agreement_ratio < 0.5) {
    opportunity.decision = "ABSTAIN"  // unconditional override
}

// Rule 2: ML contributes 15% to finalScore
finalScore = existingScore * 0.85 + mlScore * 100 * 0.15

// Rule 3: ML unavailable → heuristic score unchanged
if (meta === null) {
    opportunity.evidence.missingFactors.push("ML MetaDecisionEngine unavailable")
    // no score change
}

// Rule 4: provenance is_live_eligible check
// NOTE: AlphaForge currently does NOT enforce this — gap P1-003 in alpha-forge
```

---

## 5. Signal Schema (Target State)

The following is the target Signal schema that must be implemented to satisfy the audit requirements. It extends the current `MetaOutput`:

```python
class Signal(BaseSchema):
    # Identity
    signal_id: str                         # UUID v4
    timestamp: datetime                    # UTC signal generation time
    prediction_timestamp: datetime         # UTC (= timestamp for synchronous)
    feature_as_of: datetime                # UTC latest data timestamp used
    data_as_of: datetime                   # UTC DataService as_of
    news_as_of: datetime | None            # UTC SentinelPulse as_of
    expires_at: datetime                   # UTC signal expiry

    # Instrument
    symbol: str                            # NSE symbol uppercase
    market: str                            # NSE | NFO | CDS | BFO
    timeframe: str                         # 5m | 15m | 1h | 1d

    # Decision
    action: str                            # LONG | SHORT | EXIT | HOLD | NO_TRADE
    abstention: bool
    abstention_reason: str | None

    # Model Provenance
    model_version: str                     # e.g. "regime_classifier/v1.2.3"
    feature_version: str                   # Feature schema version hash
    data_version: str                      # Dataset version hash
    provenance: PredictionProvenance

    # Probabilities (all calibrated)
    direction_probability: float           # P(correct direction) ∈ [0, 1]
    target_probability: float              # P(target hit before stop) ∈ [0, 1]
    stop_probability: float               # P(stop hit before target) ∈ [0, 1]
    # Invariant: target_probability + stop_probability <= 1.0

    # Expected Value
    expected_return: float                 # Expected % return gross
    expected_value: float                  # EV net of costs
    uncertainty: float                     # Epistemic uncertainty ∈ [0, 1]

    # Trade Structure
    entry: float | None
    stop: float | None
    targets: list[float]
    risk_reward: float | None

    # Risk
    position_size: float                   # fraction of risk budget [0, 1]
    expected_drawdown_pct: float | None
    expected_adverse_excursion: float | None

    # Execution
    expected_slippage_pct: float | None
    expected_cost_pct: float | None

    # Regime
    regime: str                            # MarketRegime value
    regime_probability: float

    # Model Agreement
    confidence: float                      # [0, 1] calibrated
    agreement_ratio: float                 # [0, 1]
    calibration_score: float               # ECE [0, 1] lower is better
    data_confidence: float                 # DataConfidenceScore / 100

    # News
    news_impact: float | None
    news_confidence: float | None
    news_direction: int | None             # -1 | 0 | 1

    # Quality Tier
    quality_tier: str                      # REJECTED | ABSTAIN | WATCH | VALID | HIGH_CONVICTION

    # Explainability
    top_features: list[str]
    decision_reason: str                   # ≤ 200 chars

    # Provenance chain
    provenance_chain: dict                 # full model_id → contribution map
```

---

## 6. Error Response Contract

All error responses from ml-service2.0:

```python
{
    "detail": str,          # Human-readable error message
    "error_code": str,      # Machine-readable code (e.g. "DATA_QUALITY_GATE_FAILURE")
    "request_id": str,      # X-Request-ID trace
    "prediction_result": {  # Present on quality gate failures
        "action": "NO_TRADE",
        "reason_codes": ["LOW_DATA_QUALITY"],
        "provenance": "unavailable"
    }
}
```

HTTP status codes:
- `200` — successful prediction (including NO_TRADE)
- `400` — invalid request (schema validation)
- `401` — missing or invalid API key
- `422` — Pydantic validation error (field-level detail)
- `503` — service not ready (startup or circuit open)
- `500` — unexpected server error

---

## 7. Version Compatibility

Current API version: **v2** (`/v2/*` prefix)

Breaking change policy:
- New required fields → new API version (`/v3/*`)
- New optional fields → backward compatible, added to `/v2/`
- Enum value additions → backward compatible
- Enum value removals → breaking → new version

AlphaForge currently depends on: `v2/predict/*`, `v2/meta/*`, `v2/analytics/*`, `v2/explain/*`, `monitoring/*`

---

*End of Data Contracts*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Implemented by `src/data/contracts.py` (wire schemas) and `src/data/ingestion.py` (DataIngestionPipeline). Contracts verified live against data-service2.0 real NSE responses (fields: time epoch, open/high/low/close, volume incl. 0 for indices, oi, provider, sourceType).

Status: IMPLEMENTED and tested. Authoritative certification: `reports/ML_SERVICE_FINAL_CERTIFICATION.md`.
