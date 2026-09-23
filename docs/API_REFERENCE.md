# API Reference — ml-service2.0

**Base URL:** `http://localhost:8100`  
**API Version:** v2  
**OpenAPI UI:** `http://localhost:8100/docs` (available when the service is running)

---

## Table of Contents

1. [Authentication](#authentication)
2. [Response Conventions](#response-conventions)
3. [Error Codes](#error-codes)
4. [Prediction Endpoints](#prediction-endpoints)
   - [POST /v2/predict/regime](#post-v2predictregime)
   - [POST /v2/predict/rankings](#post-v2predictrankings)
   - [POST /v2/predict/strategy](#post-v2predictstrategy)
   - [POST /v2/predict/risk](#post-v2predictrisk)
   - [POST /v2/predict/portfolio](#post-v2predictportfolio)
   - [POST /v2/predict/portfolio-v2](#post-v2predictportfolio-v2)
   - [POST /v2/predict/execution](#post-v2predictexecution)
   - [POST /v2/predict/price-regime](#post-v2predictprice-regime)
   - [POST /v2/predict/iv-regime](#post-v2predictiv-regime)
5. [Meta Endpoints](#meta-endpoints)
   - [POST /v2/meta/decide](#post-v2metadecide)
   - [POST /v2/meta/fit](#post-v2metafit)
6. [Analytics Endpoints](#analytics-endpoints)
   - [POST /v2/analytics/greeks](#post-v2analyticsgreeks)
   - [POST /v2/analytics/gex](#post-v2analyticsgex)
   - [POST /v2/analytics/vpin](#post-v2analyticsvpin)
   - [POST /v2/analytics/vol-surface](#post-v2analyticsvol-surface)
7. [Explain Endpoints](#explain-endpoints)
   - [POST /v2/explain/{model_name}](#post-v2explainmodel_name)
8. [Training Endpoints](#training-endpoints)
   - [POST /training/run](#post-trainingrun)
   - [GET /training/status/{run_id}](#get-trainingstatusrun_id)
9. [Monitoring Endpoints](#monitoring-endpoints)
   - [GET /monitoring/drift](#get-monitoringdrift)
   - [GET /monitoring/performance](#get-monitoringperformance)
   - [GET /monitoring/alerts](#get-monitoringalerts)
   - [POST /monitoring/performance/{model_name}/clear](#post-monitoringperformancemodel_nameclear)
10. [Registry Endpoints](#registry-endpoints)
    - [GET /v2/models/status](#get-v2modelsstatus)
    - [GET /v2/models/registry](#get-v2modelsregistry)
11. [Features Endpoint](#features-endpoint)
    - [GET /v2/features/quality](#get-v2featuresquality)
12. [Health Endpoint](#health-endpoint)
    - [GET /health](#get-health)
13. [WebSocket](#websocket)
    - [WS /v2/stream/signals](#ws-v2streamsignals)
14. [Enumerations Reference](#enumerations-reference)

---

## Authentication

All endpoints **except `/health`** require API key authentication via the `X-API-KEY` header.

```
X-API-KEY: your-ml-service-api-key-here
```

The key must match the `ML_SERVICE_API_KEY` environment variable configured on the server.

For the WebSocket endpoint, pass the key as a query parameter:

```
WS /v2/stream/signals?api_key=your-ml-service-api-key-here
```

**Missing or invalid key:** HTTP `401 Unauthorized`

```json
{
  "detail": "Invalid or missing API key"
}
```

Every request is also stamped with an `X-Request-ID` header for distributed tracing. If you supply one, it is echoed back and included in all structured log events for that request. If you do not supply one, the service generates a UUID and attaches it to the response.

---

## Response Conventions

### Provenance field

Every prediction response includes a `provenance` field indicating the evidence quality of the prediction:

| Value | Meaning | Live-eligible |
|---|---|---|
| `trained_model` | A validated ML artifact was used for inference | Yes |
| `heuristic` | No trained artifact available; rule-based fallback was used | No |
| `insufficient_evidence` | Data quality or confidence below threshold | No |
| `unavailable` | Data source (data-service2.0) was unreachable | No |

When `DEPLOYMENT_MODE=validated_ml_only`, only `trained_model` responses are actionable. Clients should check this field before acting on any signal.

### SHAP explanations

All prediction responses from model endpoints include a `shap_explanation` block listing the **top 10 features** by absolute SHAP contribution:

```json
{
  "shap_explanation": {
    "top_features": [
      {"feature": "india_vix", "contribution": 0.142, "direction": "positive", "raw_value": 18.3},
      {"feature": "adx",       "contribution": 0.087, "direction": "negative", "raw_value": 24.1}
    ],
    "base_value": 0.52
  }
}
```

For heuristic predictions, `shap_explanation` is replaced with a rule attribution that lists which conditions fired.

### Pagination

Monitoring list endpoints (`/monitoring/alerts`) return at most 50 items. No pagination cursor is exposed in the current API version.

---

## Error Codes

| HTTP Status | Condition | Retry? |
|---|---|---|
| `200 OK` | Successful response | — |
| `401 Unauthorized` | Missing or invalid `X-API-KEY` | No — fix credentials |
| `422 Unprocessable Entity` | Request body fails Pydantic V2 schema validation | No — fix request |
| `503 Service Unavailable` | Model artifact not loaded or runtime inference exception | Yes — with backoff |
| `500 Internal Server Error` | Unexpected internal error (should not occur in normal operation) | No |

### 422 error body structure

```json
{
  "detail": [
    {
      "type":  "float_type",
      "loc":   ["body", "nifty_change_pct"],
      "msg":   "Input should be a valid number",
      "input": "not_a_float"
    }
  ]
}
```

Every field-level error lists `type`, `loc` (path into the request body), `msg`, and the `input` value that failed.

### 503 error body structure

```json
{
  "detail": "Model 'regime_classifier' is not available. No trained artifact loaded.",
  "model":  "regime_classifier",
  "fallback_available": true
}
```

Clients should implement exponential backoff on 503 responses. Unlike 422, a 503 is a transient condition that may resolve when a trained artifact is promoted.

---

## Prediction Endpoints

All prediction endpoints are under the `/v2/predict/` prefix, require `X-API-KEY` authentication, and accept/return `application/json`.

---

### POST /v2/predict/regime

Classifies the current market into one of six regimes.

**p95 SLA:** 50 ms

#### Request body — `RegimePredictionRequest`

```json
{
  "nifty_change_pct":      -0.42,
  "banknifty_change_pct":  -0.61,
  "india_vix":             18.3,
  "advance_decline_ratio":  0.72,
  "market_breadth":         0.44,
  "fii_net_cr":           -850.0,
  "put_call_ratio":         1.12,
  "nifty_atr_pct":          0.68,
  "nifty_adx":             27.5,
  "sector_strength":        0.38,
  "force_heuristic":        false
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `nifty_change_pct` | `float` | Yes | NIFTY intraday change percentage |
| `banknifty_change_pct` | `float` | Yes | BANKNIFTY intraday change percentage |
| `india_vix` | `float` | Yes | Current India VIX level |
| `advance_decline_ratio` | `float` | Yes | Advancing stocks / Declining stocks ratio |
| `market_breadth` | `float` | Yes | Fraction of F&O stocks above 20-day SMA |
| `fii_net_cr` | `float` | No | FII net flow in crores (positive = buying) |
| `put_call_ratio` | `float` | No | NIFTY put/call open interest ratio |
| `nifty_atr_pct` | `float` | No | ATR as percentage of spot |
| `nifty_adx` | `float` | No | Average Directional Index (ADX) |
| `sector_strength` | `float` | No | Composite sector relative strength vs NIFTY |
| `force_heuristic` | `bool` | No | If `true`, bypasses ML model and uses rule-based fallback |

#### Response — `RegimePredictionResponse`

```json
{
  "regime":      "bear",
  "confidence":  0.74,
  "provenance":  "trained_model",
  "probabilities": {
    "strong_bull": 0.02,
    "bull":        0.04,
    "sideways":    0.08,
    "volatile":    0.12,
    "bear":        0.74,
    "crash":       0.00
  },
  "shap_explanation": {
    "top_features": [
      {"feature": "india_vix",             "contribution": 0.21, "direction": "positive", "raw_value": 18.3},
      {"feature": "fii_net_cr",            "contribution": 0.18, "direction": "positive", "raw_value": -850.0},
      {"feature": "advance_decline_ratio", "contribution": 0.14, "direction": "positive", "raw_value": 0.72}
    ],
    "base_value": 0.52
  },
  "reason_codes":  [],
  "latency_ms":    12.4,
  "request_id":    "req-abc123"
}
```

| Field | Type | Description |
|---|---|---|
| `regime` | `MarketRegime` | Predicted regime label |
| `confidence` | `float [0,1]` | Model confidence in the predicted regime |
| `provenance` | `PredictionProvenance` | Evidence quality |
| `probabilities` | `dict` | Softmax probability over all six regimes (sums to 1.0) |
| `shap_explanation` | `object` | Top-10 SHAP features |
| `reason_codes` | `list[str]` | e.g. `["LOW_REGIME_CONFIDENCE"]` when confidence < 0.35 |
| `latency_ms` | `float` | Inference latency in milliseconds |
| `request_id` | `str` | Echo of `X-Request-ID` |

**Note:** When `confidence < 0.35`, the response includes `"LOW_REGIME_CONFIDENCE"` in `reason_codes` and `provenance` is set to `"insufficient_evidence"`.

---

### POST /v2/predict/rankings

Ranks up to 200 F&O symbols by outperformance score relative to the current regime.

**p95 SLA:** 200 ms for 200 symbols

#### Request body — `RankingRequest`

```json
{
  "stocks": [
    {
      "symbol":             "RELIANCE",
      "current_price":      2950.50,
      "volume_ratio":       1.42,
      "rsi":                58.3,
      "atr":                48.2,
      "momentum_5d":         0.032,
      "oi_change_pct":       4.5,
      "delivery_pct":       38.2,
      "iv_rank":            42.0,
      "sector":             "energy",
      "news_impact_score":   0.65,
      "impact_direction":   "bullish"
    }
  ],
  "regime": "bull",
  "top_n":  20
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `stocks` | `list[StockFeatures]` | Yes | List of 1–200 stock feature objects |
| `regime` | `MarketRegime` | Yes | Current market regime (used for regime-conditioned weights) |
| `top_n` | `int` | No | Return only the top N ranked symbols (default: all) |

**`StockFeatures` object fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `symbol` | `str` | Yes | NSE symbol (e.g. `"RELIANCE"`) |
| `current_price` | `float` | Yes | Last traded price |
| `volume_ratio` | `float` | Yes | Current volume / 20-day average volume |
| `rsi` | `float` | No | 14-period RSI |
| `atr` | `float` | No | 14-period ATR (absolute) |
| `momentum_5d` | `float` | No | 5-day price momentum |
| `oi_change_pct` | `float` | No | Open interest change percentage |
| `delivery_pct` | `float` | No | NSE delivery percentage |
| `iv_rank` | `float` | No | IV rank percentile [0, 100] |
| `sector` | `str` | No | Sector identifier |
| `news_impact_score` | `float` | No | SentinelPulse news impact score [0, 1] |
| `impact_direction` | `ImpactDirection` | No | `"bullish"` / `"bearish"` / `"neutral"` |

#### Response — `RankingResponse`

```json
{
  "rankings": [
    {
      "symbol":      "RELIANCE",
      "rank":         1,
      "score":       82.4,
      "confidence":   0.71,
      "provenance":  "trained_model",
      "shap_top5": [
        {"feature": "volume_ratio",    "contribution": 0.18},
        {"feature": "delivery_pct",    "contribution": 0.13},
        {"feature": "news_impact_score", "contribution": 0.11},
        {"feature": "oi_change_pct",   "contribution": 0.09},
        {"feature": "iv_rank",         "contribution": 0.07}
      ]
    }
  ],
  "regime":       "bull",
  "model_version": "v1.2.0",
  "latency_ms":   148.7,
  "request_id":   "req-def456"
}
```

| Field | Type | Description |
|---|---|---|
| `rankings` | `list[StockRank]` | Ranked list, descending by score |
| `score` | `float [0,100]` | Outperformance score |
| `rank` | `int` | Rank position (1 = top pick) |
| `shap_top5` | `list` | Top-5 SHAP contributors per symbol |

---

### POST /v2/predict/strategy

Recommends one of eight trading strategies with calibrated probabilities.

**p95 SLA:** 50 ms

#### Request body — `StrategyRequest`

```json
{
  "symbol":     "NIFTY",
  "regime":     "volatile",
  "rsi":        44.2,
  "adx":        31.7,
  "iv_regime":  "SPIKE",
  "atr_pct":     0.82,
  "volume_ratio": 1.35,
  "vix_level":  22.4
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `symbol` | `str` | Yes | Symbol or index name |
| `regime` | `MarketRegime` | Yes | Current market regime |
| `rsi` | `float` | Yes | 14-period RSI |
| `adx` | `float` | Yes | ADX value |
| `iv_regime` | `IVRegime` | **Yes** | IV regime — `"CRUSH"`, `"STABLE"`, or `"SPIKE"` (mandatory feature) |
| `atr_pct` | `float` | No | ATR as percentage of price |
| `volume_ratio` | `float` | No | Current / average volume |
| `vix_level` | `float` | No | India VIX current level |

#### Response — `StrategyResponse`

```json
{
  "strategy":   "volatility_breakout",
  "confidence":  0.68,
  "provenance": "trained_model",
  "probabilities": {
    "breakout":            0.08,
    "momentum":            0.06,
    "trend_following":     0.04,
    "mean_reversion":      0.05,
    "vwap_bounce":         0.03,
    "range_trading":       0.02,
    "scalping":            0.04,
    "volatility_breakout": 0.68
  },
  "alternatives": [
    {"strategy": "breakout",  "confidence": 0.08},
    {"strategy": "momentum",  "confidence": 0.06}
  ],
  "shap_explanation": { "top_features": [...], "base_value": 0.5 },
  "latency_ms":  9.1,
  "request_id": "req-ghi789"
}
```

**Note:** When the top-strategy `confidence < 0.40`, the response always includes at least two `alternatives` entries.

---

### POST /v2/predict/risk

Estimates per-trade risk: probability of stop hit, target hit, expected drawdown, and suggested position size.

**p95 SLA:** 50 ms

#### Request body — `RiskRequest`

```json
{
  "symbol":        "BANKNIFTY",
  "direction":     "long",
  "entry_price":   44200.0,
  "stop_loss":     43800.0,
  "target_price":  45000.0,
  "atr":           320.0,
  "regime":        "bull",
  "vix_level":     16.2,
  "news_impact_score": 0.35,
  "sentiment_risk":    0.22,
  "oi_buildup_score":  0.65
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `symbol` | `str` | Yes | NSE symbol |
| `direction` | `"long"` \| `"short"` | Yes | Trade direction |
| `entry_price` | `float` | Yes | Entry price |
| `stop_loss` | `float` | Yes | Stop-loss price |
| `target_price` | `float` | Yes | Target price |
| `atr` | `float` | Yes | 14-period ATR (absolute) |
| `regime` | `MarketRegime` | Yes | Current market regime |
| `vix_level` | `float` | Yes | India VIX |
| `news_impact_score` | `float` | No | SentinelPulse news impact [0, 1] |
| `sentiment_risk` | `float` | No | SentinelPulse `sentiment.risk` score |
| `oi_buildup_score` | `float` | No | OI build-up score [0, 1] |

#### Response — `RiskResponse`

```json
{
  "prob_stop_hit":              0.38,
  "prob_target_hit":            0.52,
  "expected_drawdown_pct":      1.4,
  "suggested_position_size_pct": 2.5,
  "risk_score":                 4.2,
  "stop_distance_atr":          1.25,
  "target_distance_atr":        2.50,
  "risk_reward_ratio":          2.0,
  "provenance":                "trained_model",
  "reason_codes":               [],
  "shap_explanation": {
    "top_features": [
      {"feature": "stop_distance_atr", "contribution": 0.22, "direction": "positive", "raw_value": 1.25}
    ],
    "base_value": 0.40
  },
  "latency_ms":  11.3,
  "request_id": "req-jkl012"
}
```

| Field | Type | Description |
|---|---|---|
| `prob_stop_hit` | `float [0,1]` | Estimated probability stop-loss is hit first |
| `prob_target_hit` | `float [0,1]` | Estimated probability target is hit first |
| `expected_drawdown_pct` | `float` | Expected maximum drawdown percentage |
| `suggested_position_size_pct` | `float` | Recommended position size as % of capital |
| `risk_score` | `float [0,10]` | Composite risk score (higher = riskier) |
| `reason_codes` | `list[str]` | Includes `"HIGH_RISK_BLOCKED"` when `risk_score > 7.0` and `DEPLOYMENT_MODE=validated_ml_only` |

**Invariant:** `prob_stop_hit + prob_target_hit ≤ 1.0` is always enforced. If the raw model output would violate this, the values are clamped and `"CALIBRATION_VIOLATION"` appears in logs (not in the response).

---

### POST /v2/predict/portfolio

**Legacy** portfolio optimization endpoint using PyPortfolioOpt HRP. For new integrations use `/portfolio-v2`.

**p95 SLA:** 500 ms for up to 50 assets

#### Request body — `PortfolioRequest`

```json
{
  "assets": [
    {
      "symbol":          "RELIANCE",
      "expected_return":  0.12,
      "volatility":       0.22,
      "weight_constraint": 0.30
    }
  ],
  "risk_tolerance":  0.15,
  "optimization_method": "hrp"
}
```

#### Response — `PortfolioResponse`

```json
{
  "allocations": [
    {"symbol": "RELIANCE", "weight": 0.28, "shares": 5, "capital": 14752.50}
  ],
  "portfolio_metrics": {
    "expected_return": 0.114,
    "volatility":      0.189,
    "sharpe_ratio":    0.603
  },
  "provenance": "trained_model",
  "latency_ms":  231.0
}
```

---

### POST /v2/predict/portfolio-v2

Institutional-grade portfolio optimization using Riskfolio-Lib. Supports HRP, CVaR-minimized MVO, Equal Risk Contribution, and Maximum Diversification.

**p95 SLA:** 500 ms for up to 50 assets

#### Request body — `PortfolioV2Request`

```json
{
  "symbols":  ["RELIANCE", "INFY", "HDFCBANK", "TCS", "WIPRO"],
  "returns":  null,
  "method":   "hrp",
  "alpha":    0.05,
  "max_sector_weight": 0.40,
  "random_seed": 42
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `symbols` | `list[str]` | Yes | List of 2–50 NSE symbols |
| `returns` | `list[list[float]]` \| `null` | No | Historical return matrix (T × N). If `null`, fetched from data-service2.0 |
| `method` | `str` | Yes | `"hrp"` / `"cvar"` / `"erc"` / `"max_div"` |
| `alpha` | `float` | No | CVaR tail probability (default 0.05). Only used when `method="cvar"` |
| `max_sector_weight` | `float` | No | Maximum weight for any single sector (default 0.40) |
| `random_seed` | `int` | No | Random seed for deterministic results |

#### Response

```json
{
  "weights": {
    "RELIANCE":  0.22,
    "INFY":      0.19,
    "HDFCBANK":  0.25,
    "TCS":       0.18,
    "WIPRO":     0.16
  },
  "risk_metrics": {
    "expected_return":  0.118,
    "volatility":       0.175,
    "sharpe_ratio":     0.674,
    "cvar_95":         -0.042,
    "max_drawdown_est": 0.083,
    "diversification_ratio": 1.34
  },
  "available":  true,
  "method":    "hrp",
  "provenance": "trained_model",
  "latency_ms": 312.8
}
```

When fewer than 20 observations per asset are available:

```json
{
  "available":          false,
  "reason":            "INSUFFICIENT_RETURN_HISTORY",
  "min_observations":   20,
  "observations_found": 14,
  "latency_ms":         8.2
}
```

---

### POST /v2/predict/execution

FinRL deep reinforcement learning execution agent. Returns the recommended trade action given the current execution state.

**p95 SLA:** 50 ms

#### Request body — `ExecutionState`

```json
{
  "symbol":              "NIFTY",
  "unrealized_pnl_pct":  1.2,
  "time_in_trade_min":   47,
  "regime":             "bull",
  "volume_ratio":        1.18,
  "price_vs_vwap":       0.003,
  "atr":                 95.0,
  "momentum":            0.018,
  "iv_regime":          "STABLE",
  "news_impact_score":   0.30,
  "risk_score":          3.5
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `symbol` | `str` | Yes | NSE symbol |
| `unrealized_pnl_pct` | `float` | Yes | Unrealized P&L as percentage |
| `time_in_trade_min` | `int` | Yes | Minutes elapsed since entry |
| `regime` | `MarketRegime` | Yes | Current market regime |
| `volume_ratio` | `float` | Yes | Current / average volume ratio |
| `price_vs_vwap` | `float` | Yes | (price − VWAP) / VWAP |
| `atr` | `float` | Yes | Current ATR |
| `momentum` | `float` | Yes | Short-term momentum signal |
| `iv_regime` | `IVRegime` | Yes | IV regime classification |
| `news_impact_score` | `float` | No | SentinelPulse news impact |
| `risk_score` | `float` | No | Current risk score from /predict/risk |

#### Response — `ExecutionDecision`

```json
{
  "action":      "TRAIL_STOP",
  "confidence":   0.63,
  "provenance":  "trained_model",
  "rationale":   "FinRL PPO policy v1.1.0: momentum exhausted, IV expanding",
  "latency_ms":  14.2,
  "request_id":  "req-mno345"
}
```

**Action space:**

| Value | Description |
|---|---|
| `ENTER_NOW` | Enter position immediately |
| `WAIT` | Hold — conditions not yet optimal |
| `SCALE_IN` | Add to existing position |
| `PARTIAL_EXIT` | Exit a portion of the position |
| `FULL_EXIT` | Close entire position |
| `TIGHTEN_STOP` | Move stop closer to current price |
| `TRAIL_STOP` | Switch to trailing stop mode |

---

### POST /v2/predict/price-regime

Short-horizon price regime forecast using the PatchTST temporal model.

**p95 SLA:** 50 ms

#### Request body — `PriceRegimeRequest`

```json
{
  "symbol":      "RELIANCE",
  "last_60_bars": [
    [2920.0, 2935.0, 2915.0, 2928.0, 1250000],
    "..."
  ],
  "bar_interval": "15m"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `symbol` | `str` | Yes | NSE symbol |
| `last_60_bars` | `list[list[float]]` | Yes | Shape `[60, 5]` — rows are `[open, high, low, close, volume]` in ascending time order |
| `bar_interval` | `str` | No | Bar timeframe — one of `1m, 5m, 15m, 30m, 1h, 1d` (default `"15m"`) |

**Note:** The `3m` timeframe is never valid for Indian market data and will return a `422` error.

#### Response — `PriceRegimeResponse`

```json
{
  "regime":      "trending_up",
  "probability":  0.71,
  "q10":         2905.0,
  "q90":         2975.0,
  "horizon_bars": 5,
  "provenance":  "trained_model",
  "latency_ms":   18.6,
  "request_id":  "req-pqr678"
}
```

---

### POST /v2/predict/iv-regime

Classifies the current implied volatility regime using the PatchTST model.

**p95 SLA:** 50 ms

#### Request body — `IVRegimeRequest`

```json
{
  "symbol": "NIFTY",
  "data": [
    [15.2, 14.8, 15.5, 16.1, 0.45],
    "..."
  ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `symbol` | `str` | Yes | Underlying symbol |
| `data` | `list[list[float]]` | Yes | Shape `[20, 5]` — each row is `[atm_iv, pcr, vix, realized_vol, skew]` in ascending time order |

#### Response — `IVRegimeResponse`

```json
{
  "iv_regime":   "SPIKE",
  "confidence":   0.81,
  "probabilities": {
    "CRUSH":  0.06,
    "STABLE": 0.13,
    "SPIKE":  0.81
  },
  "provenance":  "trained_model",
  "latency_ms":   9.4,
  "request_id":  "req-stu901"
}
```

---

## Meta Endpoints

### POST /v2/meta/decide

LLM-augmented meta-decision engine. Combines all base model signals, calibrates them, applies regime-aware ensemble weighting, checks abstention policy, and runs FinBERT/FinGPT news reasoning to produce a final Go/No-Go decision.

**p95 SLA:** 150 ms (with cached news context)

#### Request body — `MetaDecideRequest`

```json
{
  "symbol":             "BANKNIFTY",
  "regime":             "volatile",
  "feature_vector": {
    "symbol":           "BANKNIFTY",
    "timestamp":        "2025-07-15T09:45:00Z",
    "india_vix":        21.8,
    "put_call_ratio":   1.24,
    "news_impact_score": 0.78,
    "impact_direction": "bearish"
  },
  "force_refresh_news": false
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `symbol` | `str` | Yes | Symbol to decide on |
| `regime` | `MarketRegime` | Yes | Current market regime |
| `feature_vector` | `FeatureVector` | Yes | Full feature vector for ensemble input |
| `force_refresh_news` | `bool` | No | Bypass 90-second LRU news context cache |

#### Response — `MetaOutput`

```json
{
  "action":         "NO_TRADE",
  "confidence":      0.42,
  "uncertainty":     0.58,
  "agreement":       0.60,
  "agreement_ratio": 0.40,
  "ensemble_score":  0.38,
  "provenance":     "trained_model",
  "abstention":      true,
  "reason_codes":   ["NEWS_CONFLICT_OVERRIDE"],
  "contributing_models": ["regime_classifier", "stock_ranker", "risk_predictor"],
  "decomposition": {
    "base_confidence":          0.42,
    "calibration_quality":      0.88,
    "agreement_bonus":          0.04,
    "data_quality_factor":      0.85,
    "regime_confidence_factor": 0.76
  },
  "news_sentiment_signal": {
    "direction":   -1,
    "confidence":   0.83,
    "rationale":   "High-impact bearish news conflicts with bullish technical signals"
  },
  "explainability": {
    "top_features":        ["news_impact_score", "put_call_ratio", "india_vix"],
    "conflicting_models":  ["risk_predictor"],
    "rationale":           "News conflict override due to high-impact bearish sentiment exceeding threshold"
  },
  "symbol":     "BANKNIFTY",
  "timestamp":  "2025-07-15T09:45:02.411Z",
  "latency_ms":  88.3,
  "request_id":  "req-vwx234"
}
```

**Action values:**

| Value | Meaning |
|---|---|
| `BUY` | Ensemble recommends a long entry |
| `SELL` | Ensemble recommends a short entry |
| `WAIT` | Signals are positive but conditions not yet optimal |
| `NO_TRADE` | Abstention policy activated, news conflict, or all models unavailable |

**Common reason codes:**

| Code | Trigger condition |
|---|---|
| `NEWS_CONFLICT_OVERRIDE` | `news_impact_score > 0.7` AND news direction conflicts with ensemble |
| `LOW_AGREEMENT` | `agreement_ratio < 0.5` |
| `LOW_DATA_QUALITY` | `data_quality < 0.6` |
| `LOW_MEAN_CONFIDENCE` | `mean_confidence < 0.35` |
| `HIGH_STOP_PROB` | `prob_stop_hit > 0.65` |
| `INSUFFICIENT_AVAILABLE_MODELS` | Fewer than 3 models returned non-UNAVAILABLE results |
| `NEWS_CACHE_FALLBACK` | LLM inference timed out; used cached signal |
| `NEWS_TIMEOUT` | LLM timed out and no cached signal existed |
| `HEURISTIC_ONLY_BLOCKED` | `DEPLOYMENT_MODE=validated_ml_only` and all models are `HEURISTIC` |
| `HIGH_RISK_BLOCKED` | `risk_score > 7.0` in `VALIDATED_ML_ONLY` mode |

**Absorbing identity:** When ALL base models return `UNAVAILABLE`, the response is always `action: NO_TRADE` with `provenance: unavailable`.

---

### POST /v2/meta/fit

Retrains the meta-layer calibrators (Platt scaling / isotonic regression) on a new batch of OOS prediction records. Does not modify base models.

#### Request body

```json
[
  {
    "model_id":        "regime_classifier",
    "raw_score":        0.78,
    "realized_outcome": 1.0,
    "timestamp":       "2025-07-01T15:30:00Z"
  },
  "... (minimum 30 records required)"
]
```

| Field | Type | Required | Description |
|---|---|---|---|
| `model_id` | `str` | Yes | Model name matching a registered artifact |
| `raw_score` | `float` | Yes | Raw model output score before calibration |
| `realized_outcome` | `float` | Yes | Ground truth outcome: `0.0` (incorrect) or `1.0` (correct) |
| `timestamp` | `datetime` | Yes | Prediction timestamp (ISO-8601 UTC) |

**Minimum 30 records required.** If fewer are submitted, returns `422` with:

```json
{
  "detail": "Minimum 30 OOS records required; received 18"
}
```

#### Response — `MetaFitResponse`

```json
{
  "status":          "fitted",
  "records_used":     47,
  "ece_before":       0.12,
  "ece_after":        0.08,
  "calibrators_updated": ["regime_classifier", "risk_predictor"],
  "calibrators_unchanged": ["stock_ranker"],
  "timestamp":       "2025-07-15T10:00:00Z"
}
```

Calibrators that would result in ECE > 0.15 are not updated; their previous calibrator is retained and they appear in `calibrators_unchanged`.

---

## Analytics Endpoints

Compute derivatives analytics from raw option chain data. These endpoints are computationally independent of the ML model layer and do not require trained artifacts.

**p95 SLA:** 100 ms for all analytics endpoints

---

### POST /v2/analytics/greeks

Computes Black-Scholes/Black-76 Greeks for an option chain.

#### Request body

```json
{
  "chain": [
    {
      "strike":         44000,
      "expiry":         "2025-07-31",
      "option_type":    "CE",
      "market_iv":       0.182,
      "last_price":     312.0,
      "open_interest": 12500
    }
  ],
  "spot":       44200.0,
  "india_vix":  18.2,
  "expiry_dt":  "2025-07-31"
}
```

#### Response

```json
[
  {
    "strike":      44000,
    "option_type": "CE",
    "delta":        0.612,
    "gamma":        0.0014,
    "theta":       -18.4,
    "vega":         52.3,
    "rho":           8.1,
    "iv":           0.182,
    "charm":       -0.0008,
    "vanna":        0.024
  }
]
```

---

### POST /v2/analytics/gex

Computes dealer Gamma Exposure across the option chain and identifies the gamma-flip level.

#### Request body

```json
{
  "chain_snapshot": [...],
  "spot":   44200.0,
  "symbol": "NIFTY"
}
```

#### Response

```json
{
  "gamma_flip":             44000.0,
  "net_gex":                -284500000.0,
  "gex_by_strike": {
    "43500": -82000000,
    "44000": -145000000,
    "44500":  22000000
  },
  "gex_walls": {
    "support": [43800, 43500],
    "resistance": [44500, 45000]
  },
  "expected_move_pct":       0.84,
  "negative_gex_regime":     true
}
```

| Field | Description |
|---|---|
| `gamma_flip` | Strike level where dealer net gamma changes sign |
| `gex_walls` | Price levels with the largest absolute GEX (act as support/resistance) |
| `expected_move_pct` | GEX-implied 1-day expected move as a percentage |
| `negative_gex_regime` | `true` when aggregate dealer gamma is negative (amplified moves) |

---

### POST /v2/analytics/vpin

Computes Volume-synchronized Probability of Informed Trading (VPIN), a measure of adverse selection risk.

#### Request body

```json
{
  "symbol":      "RELIANCE",
  "bars": [
    {"open": 2920.0, "high": 2935.0, "low": 2915.0, "close": 2928.0, "volume": 1250000},
    "..."
  ],
  "bucket_size": 50000,
  "n_buckets":   50
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `symbol` | `str` | Yes | NSE symbol |
| `bars` | `list[dict]` | Yes | OHLCV bar list |
| `bucket_size` | `int` | No | Volume per bucket (default 50,000) |
| `n_buckets` | `int` | No | Number of buckets to compute VPIN over (default 50) |

#### Response

```json
{
  "vpin":           0.43,
  "classification": "elevated",
  "bucket_history": [0.38, 0.41, 0.43, 0.45, 0.43],
  "interpretation": "Elevated informed trading activity. Monitor for directional move."
}
```

**Classification thresholds:**

| Value | VPIN range |
|---|---|
| `benign` | < 0.35 |
| `elevated` | 0.35 – 0.65 |
| `toxic` | > 0.65 |

---

### POST /v2/analytics/vol-surface

Fits an SVI (Stochastic Volatility Inspired) implied volatility surface and term structure.

#### Request body

```json
{
  "symbol": "NIFTY",
  "snapshots_by_expiry": {
    "2025-07-31": [
      {"strike": 43500, "iv": 0.195},
      {"strike": 44000, "iv": 0.182},
      {"strike": 44500, "iv": 0.171}
    ],
    "2025-08-28": [
      {"strike": 43500, "iv": 0.211},
      "..."
    ]
  }
}
```

#### Response

```json
{
  "iv_by_expiry": {
    "2025-07-31": {
      "atm_iv":      0.182,
      "skew":       -0.018,
      "smile_width":  0.024
    }
  },
  "term_structure": {
    "dates":      ["2025-07-31", "2025-08-28"],
    "atm_ivs":    [0.182, 0.198],
    "contango":    true
  },
  "svi_params": {
    "2025-07-31": {"a": 0.032, "b": 0.12, "rho": -0.32, "m": 0.0, "sigma": 0.25}
  }
}
```

---

## Explain Endpoints

### POST /v2/explain/{model_name}

Returns a full SHAP explanation for a given feature input and model.

**Path parameter:** `model_name` — one of `regime_classifier`, `stock_ranker`, `strategy_selector`, `risk_predictor`, `portfolio_optimizer`, `rl_execution_agent`, `price_forecaster`, `iv_regime_classifier`.

#### Request body — `ExplainRequest`

```json
{
  "features": {
    "india_vix":             18.3,
    "advance_decline_ratio":  0.72,
    "put_call_ratio":         1.12,
    "fii_net_cr":           -850.0
  },
  "prediction": 0.74
}
```

#### Response — `ExplainResponse`

```json
{
  "model_name":     "regime_classifier",
  "model_version":  "v1.2.0",
  "contributions": {
    "india_vix":              0.142,
    "fii_net_cr":             0.118,
    "advance_decline_ratio":  0.089,
    "put_call_ratio":         0.074
  },
  "base_value":      0.52,
  "total_positive":  0.38,
  "total_negative": -0.07,
  "top_drivers": [
    {"feature": "india_vix", "contribution": 0.142, "direction": "positive"},
    {"feature": "fii_net_cr", "contribution": 0.118, "direction": "positive"}
  ]
}
```

When SHAP computation fails (numerical instability, missing feature), all `contributions` are set to `0.0` and `base_value` is set to the model's historical mean prediction. A `503` is never returned for explain failures.

---

## Training Endpoints

These endpoints do not use the `/v2/` prefix.

---

### POST /training/run

Submits an asynchronous training job.

#### Request body — `TrainingConfig`

```json
{
  "model_name":          "regime_classifier",
  "start_date":          "2022-01-01",
  "end_date":            "2024-12-31",
  "embargo_period_days":  10,
  "n_trials":             50,
  "alpha360_enabled":     false,
  "notes":               "Monthly refresh run"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `model_name` | `str` | Yes | Target model: `regime_classifier`, `stock_ranker`, `strategy_selector`, `risk_predictor`, `rl_execution_agent` |
| `start_date` | `date` | Yes | Training data start (ISO date) |
| `end_date` | `date` | Yes | Training data end (ISO date) |
| `embargo_period_days` | `int` | No | Purged K-fold embargo (min 5, default 10) |
| `n_trials` | `int` | No | Optuna HPO trials (min 50, default 50) |
| `alpha360_enabled` | `bool` | No | Enable Alpha360 factors (slower) |
| `notes` | `str` | No | Free-text notes logged to MLflow |

#### Response — `TrainingRunStatus`

```json
{
  "run_id":  "run-2025-07-15-abc123",
  "status":  "QUEUED",
  "model_name": "regime_classifier"
}
```

---

### GET /training/status/{run_id}

Polls training job status.

#### Response — `TrainingRunStatus`

```json
{
  "run_id":       "run-2025-07-15-abc123",
  "status":       "RUNNING",
  "model_name":   "regime_classifier",
  "start_time":   "2025-07-15T10:00:00Z",
  "end_time":      null,
  "ic_per_fold":  [0.031, 0.028, 0.034],
  "net_sharpe":    null,
  "pbo":           null,
  "rejection_reason": null,
  "mlflow_experiment_id": "exp-456"
}
```

**Status values:** `QUEUED` → `RUNNING` → `COMPLETED` / `FAILED` / `ABORTED`

---

## Monitoring Endpoints

---

### GET /monitoring/drift

Returns the latest Evidently AI drift report for all models.

#### Response

```json
{
  "report_timestamp":  "2025-07-15T00:00:01Z",
  "models": {
    "regime_classifier": {
      "features_tested":  47,
      "drifted_features":  3,
      "max_psi":           0.23,
      "top_drifted": [
        {
          "feature":    "india_vix",
          "psi":         0.23,
          "ref_mean":    15.2,
          "ref_std":      3.1,
          "curr_mean":   19.8,
          "curr_std":     4.4,
          "severity":   "MEDIUM",
          "recommendation": "MONITOR"
        }
      ]
    }
  }
}
```

---

### GET /monitoring/performance

Returns NannyML CBPE-estimated performance metrics for all models (no ground-truth labels required).

#### Response

```json
{
  "report_timestamp": "2025-07-15T00:00:05Z",
  "models": {
    "regime_classifier": {
      "estimated_ic":  0.028,
      "estimated_ece": 0.09,
      "blocked":       false,
      "90d_baseline_ic": 0.031
    },
    "stock_ranker": {
      "estimated_ic":  0.014,
      "estimated_ece": 0.11,
      "blocked":       true,
      "block_reason":  "ESTIMATED_IC_BELOW_THRESHOLD"
    }
  }
}
```

---

### GET /monitoring/alerts

Returns the last 50 active drift and performance alerts.

#### Response

```json
{
  "alerts": [
    {
      "alert_id":    "alert-abc789",
      "alert_type":  "DRIFT_ALERT",
      "severity":    "MEDIUM",
      "model_name":  "regime_classifier",
      "feature_name": "india_vix",
      "psi_value":    0.23,
      "reference_mean": 15.2,
      "reference_std":   3.1,
      "current_mean":   19.8,
      "current_std":     4.4,
      "recommended_action": "MONITOR",
      "timestamp":   "2025-07-15T00:00:01Z",
      "resolved":     false
    }
  ]
}
```

---

### POST /monitoring/performance/{model_name}/clear

Clears the performance degradation block on a model, re-enabling its contribution to the MetaDecisionEngine.

**Path parameter:** `model_name` — e.g. `stock_ranker`

#### Response

```json
{
  "cleared":    true,
  "model":     "stock_ranker",
  "cleared_by": "ops-user@firm.com",
  "timestamp":  "2025-07-15T11:00:00Z"
}
```

---

## Registry Endpoints

---

### GET /v2/models/status

Returns the current load status and provenance for every model.

#### Response

```json
{
  "models": {
    "regime_classifier": {
      "loaded":    true,
      "version":  "v1.2.0",
      "provenance": "trained_model",
      "artifact_path": "artifacts/market_regime/v1.2.0/model.json"
    },
    "stock_ranker": {
      "loaded":     true,
      "version":   "v1.0.3-online-2025-07-10",
      "provenance": "trained_model"
    },
    "rl_execution_agent": {
      "loaded":     false,
      "version":    null,
      "provenance": "heuristic"
    }
  }
}
```

---

### GET /v2/models/registry

Returns all registered model artifacts (champion and historical).

#### Response

```json
{
  "artifacts": [
    {
      "model_name":    "regime_classifier",
      "version":       "v1.2.0",
      "lifecycle_stage": "production",
      "algorithm":     "xgboost",
      "training_date_range": ["2022-01-01", "2024-12-31"],
      "ic_mean":        0.031,
      "net_sharpe":     1.24,
      "pbo":            0.12,
      "max_drawdown":   0.089,
      "is_champion":    true,
      "sha256_checksum": "a1b2c3...",
      "created_at":    "2025-06-01T00:00:00Z",
      "consecutive_online_updates": 0
    }
  ]
}
```

---

## Features Endpoint

### GET /v2/features/quality

Returns the `FeatureQualityReport` from the most recent feature batch.

**p95 SLA:** 2 seconds

#### Response

```json
{
  "batch_id":                  "batch-2025-07-15-09-45",
  "timestamp":                 "2025-07-15T09:45:02Z",
  "total_features_requested":   186,
  "missing_count":               3,
  "imputed_count":               3,
  "rejected_count":              0,
  "unavailable_families":       ["sentinel_pulse"],
  "pit_violations_count":        0,
  "discarded_backtest_records":  0,
  "processing_time_ms":         382.4
}
```

---

## Health Endpoint

### GET /health

Liveness probe. No authentication required.

#### Response

```json
{
  "status":    "healthy",
  "timestamp":  1752569100,
  "version":   "2.0.0"
}
```

Returns `200 OK` when the service is running. Does not check Redis, MLflow, or upstream services — use `/v2/models/status` for dependency health.

---

## WebSocket

### WS /v2/stream/signals

Real-time signal stream. The server pushes a `SignalEvent` JSON message for every new MetaDecisionEngine output.

#### Connection

```
WS ws://localhost:8100/v2/stream/signals?api_key=your-ml-service-api-key-here
```

Authentication is via the `api_key` query parameter. Missing or invalid key closes the connection with `code 4001`.

#### `SignalEvent` message schema

```json
{
  "event_type":  "signal",
  "symbol":      "NIFTY",
  "timestamp":   "2025-07-15T09:45:02Z",
  "meta_output": {
    "action":          "BUY",
    "confidence":       0.74,
    "uncertainty":      0.26,
    "agreement_ratio":  0.80,
    "provenance":      "trained_model",
    "abstention":       false,
    "reason_codes":    [],
    "ensemble_score":   0.71,
    "decomposition": { "..." : "..." },
    "news_sentiment_signal": {
      "direction":   1,
      "confidence":   0.66,
      "rationale":   "Positive macro sentiment supports bullish stance"
    },
    "explainability": {
      "top_features":       ["india_vix", "fii_net_cr", "pcr"],
      "conflicting_models": [],
      "rationale":          "Broad consensus across models with positive news alignment"
    }
  },
  "feature_quality": {
    "missing_count":          0,
    "unavailable_families":  [],
    "pit_violations_count":   0
  }
}
```

The stream remains open indefinitely. The server closes gracefully when the service shuts down. Clients should reconnect with exponential backoff on unexpected closure.

---

## Enumerations Reference

### `MarketRegime`

| Value | Description |
|---|---|
| `strong_bull` | Strong uptrend, high breadth |
| `bull` | Moderate uptrend |
| `sideways` | Ranging / consolidation |
| `volatile` | High volatility, no clear direction |
| `bear` | Moderate downtrend |
| `crash` | Rapid broad-market decline |

### `TradingStrategy`

`breakout` · `momentum` · `trend_following` · `mean_reversion` · `vwap_bounce` · `range_trading` · `scalping` · `volatility_breakout`

### `ExecutionAction`

`ENTER_NOW` · `WAIT` · `SCALE_IN` · `PARTIAL_EXIT` · `FULL_EXIT` · `TIGHTEN_STOP` · `TRAIL_STOP`

### `IVRegime`

| Value | Description |
|---|---|
| `CRUSH` | IV contracting (post-event, time decay dominant) |
| `STABLE` | Normal IV environment |
| `SPIKE` | IV expanding (fear / event risk) |

### `PredictionProvenance`

| Value | Live-eligible | Description |
|---|---|---|
| `trained_model` | Yes | ML artifact used |
| `heuristic` | No | Rule-based fallback |
| `insufficient_evidence` | No | Data quality below threshold |
| `unavailable` | No | Data source unreachable |

### `DeploymentMode`

| Value | Behavior |
|---|---|
| `research` | Full heuristic fallback allowed; signals are advisory |
| `paper` | Heuristic allowed; signals paper-traded |
| `shadow` | Heuristic allowed; signals logged for shadow evaluation |
| `validated_ml_only` | Only `trained_model` signals are acted on; heuristic → NO_TRADE |

### `ImpactDirection`

`bullish` · `bearish` · `neutral`
