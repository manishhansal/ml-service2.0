# ML System Forensic Audit
**AlphaForge Quantitative Decision Engine — System-Wide Dependency Map**

*Original audit date: 2026-09-24*
*Updated: 2026-09-25 — confirmation phase complete*

> **RESOLUTION SUMMARY (2026-09-25):** All findings from the 2026-09-24 forensic
> audit have been addressed or explicitly accepted as known limitations. See
> `docs/ML_GAP_ANALYSIS.md` for the full resolution table.
>
> Key facts:
> - All 8 P0 gaps closed (trained artifacts, labels, ingestion, PIT chain, etc.)
> - 217/220 F&O symbols ingested (98.6% universe coverage)
> - CONFIRMATION_BASELINE_V1 frozen: IC=0.486, Sharpe=5.47 at 10bps, PBO=0.00
> - Forward paper Session 1 started: 65 signals, resolve 2026-09-30
> - SentinelPulse: 919 live articles; historical pre-2026-09-18 unavailable
> - Certification state: PAPER_ELIGIBLE / SHADOW_BLOCKED (forward paper pending)
>
> Original audit preserved below for audit trail.

---

## 0. Audit Methodology

Every claim in this document was verified against actual source code, not README claims or documentation. For each architectural claim the following four states are distinguished:

| State | Meaning |
|---|---|
| **DOCUMENTED** | Described in README, docs, or comments |
| **IMPLEMENTED** | Code exists in source |
| **INTEGRATED** | Connected to other components and called at runtime |
| **PRODUCTION-VERIFIED** | Demonstrated on real data with measured outcomes |

---

## 1. System Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════╗
║                     AlphaForge Ecosystem                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  External Providers                                                  ║
║  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   ║
║  │ AngelOne │  │  Upstox  │  │  NSE WS  │  │ Crypto Exchanges │   ║
║  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   ║
║       │              │              │                  │              ║
║       └──────────────┴──────────────┴──────────────────┘            ║
║                              │                                       ║
║                              ▼                                       ║
║  ┌───────────────────────────────────────────────────────────────┐  ║
║  │          data-service2.0 (Python/FastAPI, port 8200)         │  ║
║  │  Providers: AngelOne, Upstox, NSE, Binance, Deribit          │  ║
║  │  Storage: PostgreSQL + Redis                                  │  ║
║  │  Quality: DataConfidenceScore, signalEngineAllowed            │  ║
║  │  Outputs: OHLCV, quotes, option chains, OI, Greeks, IV       │  ║
║  └───────────────────────────────┬───────────────────────────────┘  ║
║                                  │ REST /v1/india/* + gRPC stream   ║
║       ┌──────────────────────────┼──────────────────┐               ║
║       │                          │                  │               ║
║       ▼                          ▼                  ▼               ║
║  ┌─────────────┐   ┌─────────────────────────┐  ┌────────────────┐ ║
║  │SentinelPulse│   │    ml-service2.0         │  │  alpha-forge   │ ║
║  │(TS/Fastify  │   │  (Python/FastAPI,8100)   │  │ (Next.js 14,  │ ║
║  │ port 3001)  │   │                          │  │  port 3000)   │ ║
║  │             │   │  FeaturePipeline         │  │               │ ║
║  │ News NLP    │──►│  QlibFeatureEngine       │  │ Rule Engine   │ ║
║  │ Sentiment   │   │  MetaDecisionEngine      │◄─│ ML Client     │ ║
║  │ Events      │   │  TrainingPipeline        │  │ Paper Trading │ ║
║  │ ML API      │   │  ModelRegistry           │  │ Workers       │ ║
║  │             │   │  Monitoring              │  │               │ ║
║  └─────────────┘   └──────────────┬───────────┘  └───────┬────────┘ ║
║                                   │                       │          ║
║                                   └───────────────────────┘          ║
║                                    Signals / MetaDecision            ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 2. Detailed Dependency Map

### 2.1 Data Flow: Data Service → ML Service

```
data-service2.0
    │
    ├── GET /v1/india/quotes/{symbol}
    │       → DataServiceClient.get_live_quote()
    │       → FeaturePipeline._fetch_market_data()
    │       → OHLCVBar schema validation
    │       → DataConfidenceScore gate (blocks if < min_confidence)
    │       → signalEngineAllowed gate (blocks if False)
    │
    ├── GET /v1/india/historical/{symbol}/{interval}
    │       → DataServiceClient.get_historical_bars()
    │       → QlibFeatureEngine.compute_alpha158()
    │       → ~15 technical features
    │
    └── GET /v1/india/option-chain
            → DataServiceClient.get_option_chain()
            → RiskPredictor.predict() (PCR, OI inputs)
            → analytics/greeks.py (Black-Scholes)
            → analytics/gex.py (Dealer GEX)
```

**Contract fields consumed from DataService:**
- `data.close`, `data.open`, `data.high`, `data.low`, `data.volume`
- `metadata.quality.score` (DataConfidenceScore, 0-100)
- `metadata.quality.signalEngineAllowed` (bool)
- `metadata.dataAsOf` (ISO-8601 UTC timestamp)

**Data quality guards implemented:**
- Circuit breaker: 3 failures → 60s open
- 401/403 abort-and-log without retry ✓
- 3m interval ban for Indian market data ✓
- `signalEngineAllowed=false` → `UNAVAILABLE` provenance ✓
- `DataConfidenceScore < threshold` → `INSUFFICIENT_EVIDENCE` ✓

---

### 2.2 Data Flow: SentinelPulse → ML Service

```
SentinelPulse (port 3001)
    │
    ├── GET /api/v1/alphaforge/news-context/:instrument
    │       → SentinelPulseClient.fetch_news_context()
    │       → Mandatory field validation (news_impact_score, impact_direction,
    │         impact_confidence, sentiment.{overall,market,company,macro,risk})
    │       → LRU cache (500 entries, 90s TTL)
    │       → FeaturePipeline._fetch_news_context()
    │       → LLMNewsReasoner.reason() → NewsSignal
    │
    ├── GET /api/v1/news/regime
    │       → SentinelPulseClient.fetch_market_regime()
    │
    ├── GET /api/v1/alphaforge/context/market
    │       → SentinelPulseClient.fetch_market_context()
    │
    ├── GET /api/v1/ml/features/asset/:assetId
    │       → SentinelPulseClient.fetch_ml_features()
    │       → NewsFeature vectors (PIT-stamped by computedAt)
    │
    └── GET /api/v1/ml/training/samples
            → SentinelPulseClient.fetch_training_samples()
            → NewsTrainingSample with labels (for offline training)
```

**Contract fields consumed from SentinelPulse:**
- `news_impact_score` (float [0,1])
- `impact_direction` (BULLISH | BEARISH | NEUTRAL)
- `impact_confidence` (float [0,1])
- `sentiment.{overall, market, company, macro, risk}` (float [-1,1])
- `as_of` (ISO-8601 UTC — PIT boundary)

---

### 2.3 Data Flow: ML Service → AlphaForge

```
ml-service2.0
    │
    ├── POST /v2/predict/regime
    │       ← AlphaForge: predictRegime(features)
    │       → RegimeClassifier.predict() → RegimePredictionResponse
    │       → 6 regimes: strong_bull/bull/sideways/volatile/bear/crash
    │
    ├── POST /v2/predict/rankings
    │       ← AlphaForge: predictRankings(stocks, regime, topN)
    │       → StockRanker.rank() → RankingResponse
    │
    ├── POST /v2/predict/strategy
    │       ← AlphaForge: predictStrategy(params)
    │       → StrategySelector.select() → StrategyResponse
    │
    ├── POST /v2/predict/risk
    │       ← AlphaForge: predictRisk(params)
    │       → RiskPredictor.predict() → RiskResponse
    │
    ├── POST /v2/meta/decide
    │       ← AlphaForge: predictMetaDecision(model_outputs, news_context)
    │       → MetaDecisionEngine.decide() → MetaOutput
    │       → AbstentionPolicy check
    │       → EnsembleWeighter IC-proportional weights
    │       → CalibrationLayer Platt/Isotonic scaling
    │       → LLMNewsReasoner (optional FinBERT)
    │       → GoNoGoDecision: {action, confidence, uncertainty, abstention}
    │
    ├── POST /v2/predict/portfolio-v2
    │       ← AlphaForge: predictPortfolioV2(params)
    │       → Riskfolio-Lib optimization
    │
    ├── POST /v2/predict/execution
    │       ← AlphaForge: predictExecution(state)
    │       → RLExecutionAgent.act() → ExecutionDecision
    │
    ├── POST /v2/analytics/greeks
    │       ← AlphaForge: fetchOptionChainGreeks(chain, spot, vix, expiry)
    │       → greeks.compute_chain_greeks() → enriched chain
    │
    ├── GET  /v2/analytics/vol-surface
    │       ← AlphaForge: fetchVolSurface(symbol)
    │       → vol_surface.compute_vol_surface()
    │
    └── WS   /v2/stream/signals
            → SignalStreamer WebSocket push
```

**AlphaForge signal integration:**
- ML MetaDecision contributes 15% weight to opportunity `finalScore`
- `abstention=true` OR `action=NO_TRADE` → opportunity overridden to `ABSTAIN`
- `agreement_ratio < 0.5` → also triggers ABSTAIN
- `ML_MODE=fallback` (default): heuristic scoring if ML down
- `ML_MODE=required`: hard fail if ML unreachable

---

### 2.4 Data Flow: AlphaForge → SentinelPulse

```
alpha-forge
    │
    └── sentinel-client.ts
            → GET /api/v1/alphaforge/news-context/:instrument
            → GET /api/v1/news/regime
            → News sentiment injected into opportunity evidence
            → sentinelToNewsContext() adapter for ML integration
```

---

### 2.5 Missing Data Flows (CRITICAL GAPS)

```
AlphaForge paper-trade outcomes
    │
    ├── ✗ NO feedback to ML Service training pipeline
    ├── ✗ NO outcome resolution stored for model evaluation
    ├── ✗ NO online learning trigger from realized PnL
    └── ✗ NO champion/challenger comparison against live outcomes
```

---

## 3. Repository Inventory

### 3.1 ml-service2.0

| Directory | Contents | Status |
|---|---|---|
| `src/api/` | FastAPI routers: predict, meta, analytics, explain, training, monitoring | Implemented |
| `src/features/` | FeaturePipeline, QlibFeatureEngine, LeakageValidator | Implemented |
| `src/models/` | 7 model wrappers: all heuristic fallback, no trained artifacts | Heuristic only |
| `src/training/` | TrainingPipeline, PurgedKFoldSplitter, OnlineLearner, HPO, MLflowTracker | Implemented, not connected to live data |
| `src/meta/` | MetaDecisionEngine, CalibrationLayer, EnsembleWeighter, AbstentionPolicy, LLMNewsReasoner | Implemented |
| `src/registry/` | ModelRegistry (file-based, SHA256), ModelPromotion (6 gates) | Implemented |
| `src/monitoring/` | DriftDetector (PSI), DriftMonitor (Evidently/NannyML wrappers) | Implemented, optional dependencies |
| `src/analytics/` | Greeks (BS/Black-76), GEX, VPIN, VolSurface | Implemented |
| `src/clients/` | DataServiceClient (circuit breaker), SentinelPulseClient (LRU cache) | Implemented |
| `src/schemas/` | Pydantic V2 schemas: base, predictions, features, meta, monitoring, registry, streaming | Implemented |
| `src/explainability/` | ModelExplainer (SHAP TreeExplainer/KernelExplainer) | Implemented, fires only with trained models |
| `src/audit/` | AuditLogger (JSONL append-only, tamper detection) | Implemented |
| `src/cache/` | RedisCache (async), LRUCache (in-process) | Implemented |
| `src/streaming/` | SignalStreamer (WebSocket broadcast) | Implemented |
| `src/data/` | DataServiceResponse, OHLCVBar, SentinelNewsContext contracts | Implemented |
| `src/meta_engine/` | MetaEngine (Phase 2 stub), MetaEngineV3 (Phase 3 wrapper) | Both present |
| `configs/` | model_params.yaml, qlib_alpha158.yaml | Present |
| `protos/` | market_data.proto, generated stubs | Present |
| `artifacts/` | Model artifact storage directory | Empty — no trained models |
| `tests/` | 35 test files, ~86% line coverage | Mostly mocked |

### 3.2 alpha-forge

| Directory | Contents | Status |
|---|---|---|
| `src/app/api/` | 18 Next.js API routes | Implemented |
| `src/app/api/in/ml-predictions/` | ML predictions proxy route | Implemented |
| `src/lib/india/ml-client.ts` | Full ML service client (10 endpoints) | Implemented, used |
| `src/lib/india/ml-service2-integration.ts` | MetaDecision integration, applyMetaDecision | Implemented |
| `src/features/signals/engine.ts` | Crypto signal scoring (rule-based) | Implemented |
| `src/services/india/signals/score.ts` | India NSE 8-factor scoring | Implemented |
| `src/services/india/signals/snapshotter.ts` | 60s signal snapshots during market hours | Implemented |
| `worker/src/jobs/` | alerts.ts, auto-trader.ts (paper trading) | Implemented |
| `prisma/` | Database schema | Present |

### 3.3 data-service2.0

| Directory | Contents | Status |
|---|---|---|
| `src/api/india.py` | NSE quotes, option chain, market status | Implemented |
| `src/providers/` | AngelOne, Upstox, NSE adapters | Implemented |
| `src/engines/` | MarketEngine, MarketSessionEngine | Implemented |
| `src/db/` | Async SQLAlchemy + Alembic | Implemented |
| `src/cache/` | Redis async cache | Implemented |
| `src/observability/` | OpenTelemetry, Prometheus | Implemented |
| `src/stores/` | Candle stores, OI stores | Implemented |

### 3.4 SentinelPulse

| Directory | Contents | Status |
|---|---|---|
| `src/api/ml/` | ML feature, training, historical-reactions endpoints | Implemented |
| `src/api/alphaforge/` | news-context, regime endpoints | Implemented |
| `prisma/schema.prisma` | 22-table schema: articles, events, sentiment, features, training samples | Implemented |
| `src/engines/` | NLP pipeline, sentiment engine | Implemented |
| `src/workers/` | Article ingestion workers | Implemented |

---

## 4. Interface Contracts

### 4.1 DataService → ML Service Contract

```python
# OHLCVBar (data/contracts.py)
class OHLCVBar:
    symbol: str (uppercase)
    timestamp: datetime (UTC, non-future)
    open: float (> 0)
    high: float (>= low)
    low: float (> 0)
    close: float (between low and high)
    volume: float (>= 0)
    interval: str (canonical: 1m|5m|10m|15m|30m|1h|1d|1w|1M)
    provider: str
    data_confidence_score: int (0-100)
    signal_engine_allowed: bool

# DataServiceResponse
class DataServiceResponse:
    symbol: str (uppercase)
    bars: list[OHLCVBar]
    quality: DataQualityMetadata
    as_of: datetime (UTC)
    provider: str
```

**IMPORTANT:** `null != 0` — missing OI/IV/Greeks must NOT be zero-filled.

### 4.2 SentinelPulse → ML Service Contract

```python
# SentinelNewsContext (data/contracts.py)
class SentinelNewsContext:
    instrument: str (uppercase)
    as_of: datetime (UTC, non-future — PIT boundary)
    news_impact_score: float [0, 1]
    impact_direction: ImpactDirection (BULLISH|BEARISH|NEUTRAL)
    impact_confidence: float [0, 1]
    sentiment: SentinelSentimentBreakdown
    latest_event: dict | None
    market_regime: str | None

class SentinelSentimentBreakdown:
    overall: float [-1, 1]
    market: float [-1, 1]
    company: float [-1, 1]
    macro: float [-1, 1]
    risk: float [-1, 1]
```

**PIT rule:** `as_of` must be strictly less than `prediction_timestamp`. Future news is forbidden.

### 4.3 ML Service → AlphaForge Contract

```typescript
// MLMetaDecision (ml-client.ts)
interface MLMetaDecision {
    action: "BUY" | "SELL" | "WAIT" | "NO_TRADE"
    confidence: number          // [0, 1]
    uncertainty: number         // [0, 1], confidence + uncertainty <= 1.0
    agreement_ratio: number     // [0, 1], < 0.5 triggers ABSTAIN
    abstention: boolean
    provenance: string          // weakest-link across all models
    reason_codes: string[]      // structured audit codes
    rationale?: string          // ≤ 200 chars
    explainability?: {
        top_features: FeatureContribution[]
        dominant_model: string
        news_signal?: NewsSignal
    }
}
```

**Critical invariants:**
- `confidence + uncertainty <= 1.0` (guaranteed by MetaDecisionEngine)
- `abstention=true` → AlphaForge overrides to `ABSTAIN` (hard gate)
- `action=NO_TRADE` → AlphaForge overrides to `ABSTAIN`

---

## 5. Timestamp Semantics

| Field | Source | Semantics | PIT Safe? |
|---|---|---|---|
| `OHLCVBar.timestamp` | DataService | Bar close time, UTC | Yes |
| `DataServiceResponse.as_of` | DataService | Data freshness timestamp | Yes |
| `SentinelNewsContext.as_of` | SentinelPulse | News context computation time | Yes — validated non-future |
| `FeatureVector.feature_as_of` | ML Service | Feature construction time | **Needs explicit tracking** |
| `MetaOutput.symbol` | ML Service | At decision time | Partial — no explicit timestamp |
| `GoNoGoDecision` | ML Service | No explicit timestamp field | **MISSING** |
| `MLMetaDecision` | AlphaForge | Received at API call time | Implicit only |

**Critical finding:** The ML service `MetaOutput` and `GoNoGoDecision` schemas do not include a `prediction_timestamp` field. This is a P1 gap — every prediction must carry an explicit timestamp for audit and PIT validation.

---

## 6. Source of Truth Map

| Data Type | Source of Truth | Consumers | Notes |
|---|---|---|---|
| Market OHLCV | data-service2.0 | ML Service, AlphaForge | ML must NOT call providers directly |
| Option chain | data-service2.0 | ML Service, AlphaForge | OI/IV null ≠ zero |
| News events | SentinelPulse | ML Service, AlphaForge | Future news forbidden |
| Model artifacts | ml-service2.0 artifacts/ | MetaDecisionEngine | File-based, SHA256 integrity |
| Training experiments | MLflow (configured) | TrainingPipeline | Optional — MLflow may not be running |
| Signals | ml-service2.0 | AlphaForge | Via REST + WebSocket |
| Paper trade outcomes | alpha-forge DB (Prisma) | Worker jobs | NOT fed back to ML |

---

## 7. Known Technical Debt

### 7.1 Dual MetaEngine Implementations
Two parallel MetaEngine implementations exist:
- `src/meta_engine/engine.py::MetaEngine` — Phase 2 stub (raises NotImplementedError)
- `src/meta_engine/engine.py::MetaEngineV3` — Phase 3 wrapper around MetaDecisionEngine
- `src/meta/engine.py::MetaDecisionEngine` — actual implementation

This creates confusion and maintenance overhead. The stub should be deprecated or clearly marked as test-only.

### 7.2 Coverage Inflation via Mock Tests
The 86.47% coverage figure is inflated by 10+ `test_coverage_boost*.py` files that mock all external dependencies. Real functional coverage is significantly lower. True integration test coverage against live data is ~0%.

### 7.3 QlibFeatureEngine Native Fallback
The `QlibFeatureEngine` imports Qlib optionally and computes Alpha158/Alpha360 natively (via its own `_compute_factors_native()` method) when Qlib is unavailable. This means the service can run without Qlib, but the native implementation may diverge from Qlib's reference implementation over time.

### 7.4 Hardcoded data_quality=1.0 in MetaDecisionEngine
`AbstentionContext` is constructed with `data_quality=1.0` hardcoded in `MetaDecisionEngine.decide()`. The actual data quality from DataServiceClient is not threaded through to the abstention logic. This means low-quality data cannot trigger the `LOW_DATA_QUALITY` abstention condition.

### 7.5 BaseMLModel Abstract Methods
`BaseMLModel.predict()`, `.fit()`, and `.score()` all raise `NotImplementedError`. The concrete model classes (RegimeClassifier, StockRanker, etc.) do NOT inherit from BaseMLModel — they are standalone classes with their own interfaces. The base class is structurally orphaned.

### 7.6 Feedback Loop Gap
Paper trade outcomes from AlphaForge are stored in Prisma/PostgreSQL but are never routed back to ml-service2.0 for:
- OnlineLearner updates
- Model performance evaluation
- Champion/challenger comparison
- Feature importance recalibration

---

## 8. Verification Status Summary

| Claim | Documented | Implemented | Integrated | Production-Verified |
|---|---|---|---|---|
| 7 ML model families | ✓ | ✓ | ✓ | ✗ (all heuristic) |
| MetaDecisionEngine | ✓ | ✓ | ✓ | ✗ (no live data tested) |
| 6-gate promotion | ✓ | ✓ | Partial | ✗ |
| PurgedKFold CV | ✓ | ✓ | Partial | ✗ |
| CPCV validation | ✓ | Partial | ✗ | ✗ |
| Calibration (Platt/Isotonic) | ✓ | ✓ | Partial | ✗ (no fitted calibrators) |
| SHAP explainability | ✓ | ✓ | ✗ | ✗ (no trained models) |
| Drift monitoring (PSI) | ✓ | ✓ | Partial | ✗ |
| Evidently integration | ✓ | ✓ (wrapper) | ✗ | ✗ |
| NannyML integration | ✓ | ✓ (wrapper) | ✗ | ✗ |
| OnlineLearner | ✓ | ✓ | ✗ | ✗ |
| Feedback loop | ✓ | ✗ | ✗ | ✗ |
| Triple-barrier labels | ✗ | ✗ | ✗ | ✗ |
| Walk-forward validation | ✓ | Partial | ✗ | ✗ |
| Real model training | ✗ | ✗ | ✗ | ✗ |
| Transaction-cost backtest | ✓ | Partial (10bps flat) | ✗ | ✗ |
| AlphaForge→ML integration | ✓ | ✓ | ✓ | Partial (live route exists) |
| ML→DataService integration | ✓ | ✓ | ✓ | Partial |
| ML→SentinelPulse integration | ✓ | ✓ | ✓ | Partial |

---

*End of ML System Forensic Audit*
