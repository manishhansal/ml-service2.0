# Architecture Decision Record — ml-service2.0

This document records the key architectural decisions made during the design of ml-service2.0, the rationale behind each decision, and the alternatives that were considered and rejected. It is intended as a durable reference for anyone who works on this service or integrates with it.

---

## Table of Contents

1. [Why Extract from alpha-forge?](#why-extract-from-alpha-forge)
2. [Three-Repository Data Flow](#three-repository-data-flow)
3. [Why data-service2.0 is the Sole Market Data Authority](#why-data-service20-is-the-sole-market-data-authority)
4. [Why SentinelPulse is the Sole NLP Authority](#why-sentinelpulse-is-the-sole-nlp-authority)
5. [Feature Pipeline PIT Correctness](#feature-pipeline-pit-correctness)
6. [MetaDecisionEngine Design](#metadecisionengine-design)
7. [Why Riskfolio-Lib over PyPortfolioOpt](#why-riskfolio-lib-over-pypfopt)
8. [Why FinRL-X over Raw stable-baselines3](#why-finrl-x-over-raw-stable-baselines3)
9. [Why Evidently AI + NannyML Together](#why-evidently-ai--nannyml-together)
10. [Audit Log Design](#audit-log-design)
11. [gRPC for Live Tick Streaming](#grpc-for-live-tick-streaming)
12. [Qlib for Feature Engineering and CV](#qlib-for-feature-engineering-and-cv)
13. [LangChain for LLM Orchestration](#langchain-for-llm-orchestration)
14. [SHAP as the Standard for XAI](#shap-as-the-standard-for-xai)
15. [Pydantic V2 Strict Mode Throughout](#pydantic-v2-strict-mode-throughout)
16. [The Six-Gate Promotion Pipeline Rationale](#the-six-gate-promotion-pipeline-rationale)
17. [WebSocket for Real-Time Signal Streaming](#websocket-for-real-time-signal-streaming)
18. [Component Interaction Map](#component-interaction-map)

---

## Why Extract from alpha-forge?

**Decision:** ml-service2.0 is a standalone Python microservice, separate from the alpha-forge Next.js + Python monorepo.

**Context:** The original alpha-forge repository embedded all ML inference and training logic alongside the trading UI and order management code. This created several operational problems:

- **Deployment coupling.** Any ML model update required redeploying the entire alpha-forge stack, including the Next.js UI and the order routing layer. A training job that exceeded memory limits could destabilize the trading interface.
- **Scaling mismatch.** ML inference is CPU/memory-intensive (SHAP computation, LightGBM ranking over 200 symbols). The UI tier needs fast horizontal scaling for WebSocket connections. These have incompatible scaling characteristics.
- **Versioning fragility.** Model versioning was tangled with application versioning. A bug fix in the UI could inadvertently ship an untested model change.
- **Future consumers.** The AlphaForge platform will eventually serve multiple strategy products. A standalone ML service can be consumed by future clients beyond alpha-forge without duplicating the ML stack.

**Alternatives considered:**

- *Sidecar pattern within alpha-forge.* Runs the ML process as a sidecar container alongside the alpha-forge server. Simpler operationally, but still couples deployment cycles and prevents independent scaling.
- *In-process ML in alpha-forge (status quo).* The original approach. Ruled out because of the scaling and versioning problems described above.

**Decision outcome:** ml-service2.0 runs on its own port (8100), has its own versioning, its own CI/CD pipeline, and its own `pyproject.toml`. alpha-forge calls it over HTTP/WebSocket. Neither service has a code-level dependency on the other.

---

## Three-Repository Data Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  alpha-forge (port 3000)                                               │
│  Next.js + Python trading platform                                     │
│  Consumer: calls /v2/predict/*, /v2/meta/decide, WS /v2/stream/signals │
│                                                                        │
└────────────────────┬───────────────────────────────────────────────────┘
                     │ REST + WebSocket
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  ml-service2.0 (port 8100)                                             │
│  Inference · Training · Monitoring · Explainability                    │
│                                                                        │
└──────────┬─────────────────────────────────────┬───────────────────────┘
           │ REST + gRPC                          │ REST
           ▼                                      ▼
┌─────────────────────────┐          ┌──────────────────────────┐
│  data-service2.0         │          │  SentinelPulse            │
│  (port 8200)             │          │  (port 3001)              │
│  Market data sole        │          │  NLP/news sole            │
│  authority               │          │  authority                │
│  NSE · NFO · Crypto      │          │  Sentiment · Event impact │
│  OHLCV · Options · Ticks │          │  PIT-correct ML features  │
└─────────────────────────┘          └──────────────────────────┘
```

**Information flow rules:**

1. ml-service2.0 **never** connects directly to NSE, Upstox, Angel One, Binance, Deribit, or any market data provider. All market data enters through data-service2.0.
2. ml-service2.0 **never** connects directly to any news API (Bloomberg, Refinitiv, Twitter/X). All NLP features enter through SentinelPulse.
3. alpha-forge **never** calls data-service2.0 or SentinelPulse directly for ML feature purposes (though it may call them for UI-level data display).
4. The three services are version-independent. Breaking changes in one service's API must be negotiated through a versioned API contract, not through shared code.

---

## Why data-service2.0 is the Sole Market Data Authority

**Decision:** ml-service2.0 fetches all market features exclusively from `data-service2.0`. Direct connections to any exchange or data vendor are permanently forbidden.

**Rationale:**

1. **3m timeframe ban.** The Indian market structure means 3-minute bars are a common artefact of misaligned data pipelines. data-service2.0 enforces the canonical timeframe list (`1m, 5m, 10m, 15m, 30m, 1h, 1d, 1w, 1M`) and rejects 3m requests. If ml-service2.0 were to call the exchange directly, this guardrail would need to be duplicated — and inevitably drift.

2. **Quality gates.** data-service2.0 computes a `DataConfidenceScore` (0–100) and a `signalEngineAllowed` boolean for every response. These flags represent a consensus quality assessment across multiple data sources. Replicating this logic in ml-service2.0 would create two competing quality assessors.

3. **PIT backtest mode.** data-service2.0 supports a `?pit_date=YYYY-MM-DD` parameter that returns only data that would have been available at that date, with post-hoc revisions removed. This is essential for point-in-time correct training. Building this logic in ml-service2.0 would require maintaining a full historical tick database, which is data-service2.0's responsibility.

4. **Single source of truth.** When a prediction is audited, the feature vector must be reproducible. If ml-service2.0 pulled from multiple data sources, reproducing the exact feature vector for an audit investigation would require knowing which source was used for each field at a specific timestamp.

**Failure handling:**

| data-service2.0 response | ml-service2.0 behaviour |
|---|---|
| `signalEngineAllowed: false` | Return `UNAVAILABLE` for all prediction endpoints |
| `DataConfidenceScore` < 70 | Return `INSUFFICIENT_EVIDENCE` |
| Authentication error | Return `UNAVAILABLE`, log warning, no retry |
| Timeout (> 10s) or connection error | Return `UNAVAILABLE` via circuit breaker |
| gRPC stream interrupted | Retry up to 3 times; then `UNAVAILABLE` |

**The circuit breaker (3 failures / 30s → OPEN for 60s)** prevents the timeout cascade where a slow data-service2.0 causes all 100 concurrent inference requests to wait 10 seconds each, exhausting the Uvicorn thread pool. Once the circuit opens, requests fail fast with `UNAVAILABLE` immediately, and the half-open probe resumes after 60 seconds.

---

## Why SentinelPulse is the Sole NLP Authority

**Decision:** All news-derived features enter through SentinelPulse's PIT-correct endpoints. ml-service2.0 does not subscribe to news feeds directly.

**Rationale:**

1. **`look_ahead_validated` flag.** SentinelPulse's training data endpoints return a `look_ahead_validated: true` flag on every sample, certifying that the sentiment score was computed using only information available before the event. A raw news API would not provide this guarantee. Training on news data without PIT validation is one of the most common sources of look-ahead bias in systematic strategies.

2. **Pre-computed structured features.** SentinelPulse returns structured fields (`news_impact_score`, `impact_direction`, `impact_confidence`, `sentiment.overall`, `sentiment.market`, `sentiment.company`, `sentiment.macro`, `sentiment.risk`) that are directly usable as model features without any NLP processing in ml-service2.0. This keeps the ML service focused on model inference rather than text processing.

3. **Degraded-mode substitution.** When SentinelPulse is unreachable, the Feature Pipeline applies a deterministic substitution (all numeric → 0.0, directional → NEUTRAL) and continues inference using only market-data features. This design means SentinelPulse unavailability never causes a `503` for inference endpoints.

4. **LLM news reasoning.** The MetaDecisionEngine uses FinBERT/FinGPT to process the SentinelPulse news context package, not raw news text. This means the LLM receives a curated, structured summary rather than unfiltered article text, which improves inference latency and reduces hallucination risk.

**Degraded mode contract:** When SentinelPulse is unreachable for a symbol:
- All `news_*` fields in the `FeatureVector` are set to their zero/neutral defaults
- `news_available: false` is set on the feature vector
- The affected instrument, timestamp, and zeroed fields are recorded in the `FeatureQualityReport`
- The `MetaOutput.explainability` report notes that news context was unavailable

---

## Feature Pipeline PIT Correctness

**Decision:** The `FeaturePipeline` enforces point-in-time correctness through a combination of timestamp validation, training data cutoffs, backtest mode, and the `LeakageValidator`.

**Why this matters:** In a trading context, using future information during training (look-ahead bias) is equivalent to insider trading — the model appears to predict the future because it was trained on it. The consequences range from inflated backtest metrics to actual live trading losses when the model's "alpha" disappears.

### 09:15 IST training data cutoff

When building a training sample for date T, the Feature Pipeline uses **only data with observation timestamps strictly before 09:15 IST on date T** (the NSE regular session open). Any data record with a timestamp at or after 09:15 IST on date T is discarded and the record is logged.

**Why 09:15 IST specifically:** This is the NSE regular session open. Any market-derived feature (VIX, option chain snapshot, OI data) that contains information from the T-day session would constitute look-ahead — a live system making predictions before the open would not have that data. Using the previous business day's close is conservative but unambiguous.

### `LeakageValidator`

The `LeakageValidator` runs before every training job and computes the Pearson correlation between each feature and realized returns over look-ahead windows of **1 to 22 trading days** (approximately one calendar month). If any feature exhibits an absolute correlation exceeding 0.05 over any window, a `PITViolationError` is raised and training is aborted.

**Why 0.05 and 22 days:** 
- 0.05 is a conservative but practical threshold. A feature with 5% correlation to future returns is suspicious enough to investigate; it could reflect autocorrelation, data contamination, or a genuine (but survivorship-biased) signal.
- 22 trading days covers the label observation window used for monthly rebalancing strategies. Checking beyond 22 days would falsely flag features that correlate with macro trends rather than look-ahead contamination.

### Backtest mode (`?pit_date`)

When `mode="backtest"` and a `pit_date` is specified, the Feature Pipeline adds `?pit_date=YYYY-MM-DD` to every data-service2.0 request. Any response data with a timestamp after `pit_date` is discarded. The count of discarded records is included in the `FeatureQualityReport`.

This mode allows the training pipeline to reconstruct exactly the feature vectors that would have been available at any historical date, enabling PIT-correct backtesting.

### Idempotence

The Feature Pipeline is **deterministic given the same input data and timestamp**. There is no random state that varies between calls. This property is verified by the property-based test in `tests/test_feature_pipeline_pit_correctness.py` (Property 2).

---

## MetaDecisionEngine Design

**Decision:** A dedicated MetaDecisionEngine layer sits above all base models, applies calibration, regime-aware ensemble weighting, abstention policy, and LLM news override before producing the final trading decision.

**Why a meta-layer at all:** Individual models have complementary weaknesses. The regime classifier may lag regime transitions. The stock ranker may be overconfident on illiquid symbols. The risk predictor may underestimate tail risk in crash regimes. A meta-layer that weights models by their recent IC in the current regime is systematically better than any single model or a static ensemble.

### Design flow

```
1. Filter unavailable models
   └─ < 3 available → absorbing NO_TRADE immediately

2. VALIDATED_ML_ONLY guard
   └─ HEURISTIC contributions → direction = 0

3. Platt / isotonic calibration per model
   └─ Selects method with lower ECE on held-out OOS fold
   └─ Rejects calibration update if ECE > 0.15 (retains prior calibrator)

4. Regime-aware ensemble weighting
   └─ Weight ∝ model IC in current regime
   └─ min weight: 0.05, max weight: 0.40
   └─ Models with IC < 0.05 in current regime → receive min weight 0.05
   └─ Weights normalized to sum to 1.0

5. Agreement ratio computation
   └─ agreement_ratio = |models matching plurality direction| / |available models|

6. LLM news reasoning (120ms timeout)
   └─ FinBERT (default) or FinGPT processes SentinelPulse news context
   └─ Returns direction (+1/0/-1), confidence, 30-word rationale
   └─ Cached 60s in Redis; on timeout: use cache → NEWS_CACHE_FALLBACK
   └─ No cache: direction = 0 → NEWS_TIMEOUT

7. AbstentionPolicy check (any condition → NO_TRADE)
   └─ agreement_ratio < 0.5
   └─ data_quality < 0.6
   └─ mean_confidence < 0.35
   └─ prob_stop_hit > 0.65
   └─ available models < 3

8. News conflict override
   └─ news_impact_score > 0.7 AND news direction ≠ ensemble direction → NO_TRADE

9. Final action from ensemble_score
   └─ > 0.65 → BUY
   └─ < 0.35 → SELL
   └─ otherwise → WAIT

10. ConfidenceDecomposition
    └─ Five-component breakdown: base, calibration, agreement, data quality, regime
```

**Why the 0.05 minimum weight:** A model with IC below 0.05 in the current regime is arguably useless — but completely zeroing it out would allow a single failing model to silence potentially useful signals from the ensemble. The 0.05 floor means even a bad model can nudge the ensemble slightly while preventing any single model from dominating.

**Why the 0.40 maximum weight:** Preventing any single model from exceeding 40% weight guards against the meta-layer collapsing into a single-model decision when one model dominates in a regime. The 0.40 cap forces diversification across the ensemble.

**Why absorbing identity (all UNAVAILABLE → NO_TRADE):** A trading system that produces `BUY` signals when it has no evidence is more dangerous than one that abstains. `NO_TRADE` with `UNAVAILABLE` provenance is a safe, interpretable default. The caller (alpha-forge) can check the provenance field and avoid placing orders when the ML layer is completely dark.

---

## Why Riskfolio-Lib over PyPortfolioOpt

**Decision:** New portfolio optimization endpoints (`/predict/portfolio-v2`) use Riskfolio-Lib. The legacy endpoint (`/predict/portfolio`) retains PyPortfolioOpt for backward compatibility.

| Criterion | PyPortfolioOpt 1.5.5 | Riskfolio-Lib 6.x |
|---|---|---|
| HRP (Ward linkage) | Yes | Yes |
| CVaR-minimized MVO | No | **Yes** |
| Equal Risk Contribution (ERC) | No | **Yes** |
| Maximum Diversification | No | **Yes** |
| Sector weight constraints | Limited | **Full support** |
| Determinism (same seed → same result) | Yes | Yes |
| Active maintenance | Moderate | Active |
| Pandas-native API | Yes | Yes |
| CVXPY solver integration | Limited | **Full (ECOS, SCS, Clarabel)** |

**Key differentiator — CVaR minimization:** PyPortfolioOpt's MVO minimizes variance (Markowitz). CVaR minimization (Riskfolio-Lib) minimizes the expected loss in the tail of the return distribution. For a trading portfolio with heavy-tail returns (NSE equity), CVaR-based allocation produces meaningfully different and more robust portfolios than variance minimization.

**Key differentiator — Full constraint support:** The `max_sector_weight` constraint (default 40%) is a hard requirement for regulatory compliance. Riskfolio-Lib expresses this as a CVXPY constraint, which is solved exactly. PyPortfolioOpt's constraint API is less expressive and does not support binding sector constraints across all optimization methods.

**Why retain PyPortfolioOpt at all:** The legacy `/predict/portfolio` endpoint is in active use by alpha-forge. Migrating it to Riskfolio-Lib would change the weight calculations and require re-validation of all existing portfolio allocations. Maintaining the legacy endpoint with PyPortfolioOpt while introducing Riskfolio-Lib for all new functionality is the safer migration path.

---

## Why FinRL-X over Raw stable-baselines3

**Decision:** The RL execution agent is trained using FinRL-X (built on stable-baselines3) rather than a bare stable-baselines3 implementation.

**What FinRL-X adds:**

1. **`StockTradingEnv`.** A gymnasium-compatible environment pre-wired for financial trading, with position sizing, transaction cost simulation, and episode termination at session end. Replicating this from scratch with stable-baselines3 alone would require 500+ lines of environment code.

2. **NSE session calendar extension.** The environment is extended with the NSE session calendar (375 minutes per session, excluding Indian market holidays). The RL agent's episode always terminates at 15:30 IST, and the reward function penalizes holding open positions at session close — a constraint that a generic `StockTradingEnv` does not enforce by default.

3. **Offline Policy Evaluation (OPE).** FinRL-X includes importance sampling OPE, which estimates the Sharpe ratio of the trained policy on historical logged data without requiring live simulation. This is used as a pre-deployment gate: policies with OPE estimated Sharpe < 0.0 are rejected before they can be promoted to production.

4. **Algorithm configurability.** The algorithm (PPO vs SAC) is a training configuration parameter, not a code change. Switching from PPO to SAC is a one-line config update (`algorithm: sac` in `TrainingConfig`).

**Why not custom gymnasium environment + SB3 directly:** The financial environment logic (stop-loss handling, position sizing, reward shaping, NSE calendar) would need to be written, tested, and maintained in ml-service2.0. FinRL-X provides this as a vetted, battle-tested library component. The rule-based fallback policy (`_rule_based_fallback()`) is the only custom code needed.

---

## Why Evidently AI + NannyML Together

**Decision:** The `DriftMonitor` uses both Evidently AI (for feature drift) and NannyML (for performance estimation), treating them as complementary rather than substitutable.

| Capability | Evidently AI | NannyML |
|---|---|---|
| Feature distribution drift (PSI, KS test) | **Yes** | Limited |
| Concept drift detection (output drift) | **Yes** | Yes |
| `DataDriftPreset` (all features in one call) | **Yes** | No |
| Performance estimation without ground truth | No | **Yes (CBPE)** |
| Estimated IC on unlabeled production data | No | **Yes** |
| Expected Calibration Error estimation | No | **Yes** |
| Structured report for API consumption | **Yes** | Limited |

**The gap that NannyML fills:** Feature drift (PSI) tells us the input distribution has shifted. It does not tell us whether the model is still making good predictions on the new distribution. A feature might drift in a way that the model handles gracefully, or it might drift in a direction the model has never seen. NannyML's CBPE answers the question "is our model still good?" without waiting for realized returns — which may not be available for days or weeks.

**Example where one without the other would fail:**

- *Evidently alone:* PSI is low (features look stable), but the model's Brier score has been silently increasing because the feature interactions changed in a way that PSI doesn't capture. NannyML would flag this as estimated IC degradation.
- *NannyML alone:* Estimated IC drops. We know the model is underperforming but don't know which features are the culprit or whether it's the data or the model. Evidently's feature-level PSI report pinpoints the drifted features, guiding targeted retraining.

**Why not a custom PSI implementation only:** A custom PSI function is implemented in `DriftMonitor.compute_psi()` and is tested by property-based tests (Properties 15 and 16). It is used as a fallback and for unit testing. For the full production drift report, Evidently AI's `DataDriftPreset` is preferred because it handles edge cases (empty bins, zero variance features, categorical features) that a custom implementation would need to address individually.

---

## Audit Log Design

**Decision:** The audit log is an append-only JSON Lines file with HMAC-SHA256 tamper detection, fsync on every write, and a `AuditLogViolation` error if any modification or deletion is attempted.

**Why append-only and not a database:** A database can be truncated, rows can be deleted, and schema migrations can alter historical records. An append-only flat file with a checksum provides a simpler tamper-evidence guarantee: the only operation ever performed is `open(path, "a")` followed by `fsync()`. No delete, update, or truncate paths exist in the code.

**Why fsync on every write:** Without `fsync`, the OS may buffer the write and lose it on a crash. For a trading audit log, a missing record is a compliance failure. The performance cost of `fsync` (a few milliseconds per write) is acceptable because audit log writes are low-frequency compared to inference requests.

**Why JSON Lines and not a structured binary format:** JSON Lines is human-readable and trivially parseable with `jq`, Python, or any log aggregation pipeline. Binary formats require custom tooling. The audit log is not a performance-critical hot path — it records training runs and promotions, not per-prediction events.

**Tamper detection approach:**

Each audit log entry includes a chain hash: `SHA-256(content + previous_entry_hash)`. Any modification to a prior entry changes its hash, which invalidates all subsequent chain hashes. At startup, the service verifies the chain integrity and raises `AuditLogViolation` if any break is detected.

This provides tamper evidence, not tamper prevention. A determined actor with filesystem access can still modify both the entry and the chain hashes. Production deployment should back up the audit log to an append-only object storage bucket (e.g., S3 with Object Lock) to provide a cryptographically independent tamper-evidence record.

---

## gRPC for Live Tick Streaming

**Decision:** Live tick features from data-service2.0 are streamed via bidirectional gRPC, not via REST polling.

**Why gRPC and not REST long-polling or SSE:**

| Factor | REST polling | SSE | gRPC streaming |
|---|---|---|---|
| Latency per update | High (HTTP overhead per request) | Medium | **Low** |
| Bidirectional support | No | Server-only | **Yes** |
| Protocol overhead | High | Medium | **Low (HTTP/2 + protobuf)** |
| Connection multiplexing | No | Limited | **Yes (HTTP/2)** |
| Type-safe contracts | No | No | **Yes (protobuf schema)** |

For a 50ms p95 inference SLA on regime classification and risk prediction, the feature fetch latency budget is approximately 5–10ms. REST polling at this frequency would create excessive HTTP overhead and connection churn. gRPC's persistent connection and protobuf binary encoding reduce per-message overhead to microseconds.

**Failure handling:** If the gRPC stream from data-service2.0 is interrupted, the `DataServiceClient` retries up to `GRPC_MAX_RECONNECT_ATTEMPTS` (default 3) times with 1-second delays. After all retries are exhausted, any prediction request depending on live tick features returns `PredictionProvenance.UNAVAILABLE`. The circuit breaker is not applied to gRPC streams — reconnect attempts are handled by the gRPC client library's built-in retry policy.

---

## Qlib for Feature Engineering and CV

**Decision:** Microsoft Qlib provides the Alpha158/Alpha360 factor libraries and the `RollingPurgedKFold` cross-validation implementation.

**Why Qlib and not custom factor implementations:**

1. **Alpha158 factors are peer-reviewed.** The 158 factors in the Alpha158 library correspond to documented financial signals (momentum, value, quality, volatility, volume) with known IC ranges on Asian equity markets. Building and validating 158 equivalent factors from scratch would take months and would not benefit from Qlib's ongoing research updates.

2. **`RollingPurgedKFold` is non-trivial to implement correctly.** The purging and embargo logic must correctly handle overlapping label windows, variable-frequency data, and the interaction between fold boundaries and the embargo period. Qlib's implementation has been tested on production financial data. A custom implementation would require extensive property-based testing to achieve the same confidence.

3. **Consistency between training and backtesting.** When Qlib computes Alpha158 factors for the training pipeline, it uses the same computation graph as when computing factors for live inference. Any custom implementation risks subtle differences between training-time and inference-time factor computation.

**Dependency initialization:** Qlib must be initialized once at service startup with `qlib.init(provider_uri=data_path)`. The `provider_uri` points to a local cache directory where Qlib stores pre-computed factor data. In containerized deployments, this directory must be on a persistent volume.

---

## LangChain for LLM Orchestration

**Decision:** The `LLMNewsReasoner` component uses LangChain to orchestrate FinBERT/FinGPT inference, not the raw HuggingFace `pipeline()` API.

**What LangChain provides:**

1. **Async support.** `AsyncLLMChain` allows the LLM inference step to run non-blocking within FastAPI's async request handler. The raw HuggingFace `pipeline()` is synchronous and would block the event loop.

2. **Timeout configuration.** LangChain wraps inference in an async timeout context, enabling the 120ms deadline (`LLM_INFERENCE_TIMEOUT_MS`) to be enforced cleanly. Implementing this around the raw HuggingFace pipeline requires careful asyncio wrapping.

3. **Built-in caching.** LangChain supports in-memory and Redis-backed caching of LLM responses. The `llm_news:{symbol}` cache key (60s TTL) is managed by LangChain's cache layer, not by custom Redis code.

4. **Model swap.** Switching from FinBERT to FinGPT is a one-line config change (`LLM_MODEL_NAME=THUDM/FinGPT-v3.3`). LangChain's `HuggingFacePipeline` wrapper handles the model loading transparently.

**Why FinBERT as the default over FinGPT:**

- FinBERT (`ProsusAI/finbert`) is ~450MB and runs in ~20ms on CPU. FinGPT (`THUDM/FinGPT-v3.3`) is ~7GB and runs in ~300ms+ on CPU.
- For a 120ms inference budget within a 150ms p95 API SLA, FinBERT fits; FinGPT does not without GPU acceleration.
- FinBERT's three-class output (positive/negative/neutral) maps directly to the `direction (+1/0/-1)` field. FinGPT's generative output requires parsing, which adds latency.
- FinGPT is configurable as an alternative via `LLM_MODEL_NAME` for users with GPU infrastructure who want richer rationale generation.

---

## SHAP as the Standard for XAI

**Decision:** Every model prediction is accompanied by a SHAP explanation. SHAP `TreeExplainer` is used for tree-based models; `KernelExplainer` is used for neural network models.

**Why SHAP and not LIME or attention weights:**

| Criterion | LIME | LIME / Grad-CAM | SHAP TreeExplainer |
|---|---|---|---|
| Exact (not approximation) | No | No | **Yes (for tree models)** |
| Consistent (same feature always same attribution) | No | No | **Yes** |
| Locally accurate (attributions sum to prediction) | Approximate | No | **Yes** |
| Latency (tree models, 1 prediction) | ~50ms | N/A | **~1ms** |
| Industry acceptance | Limited | Limited | **Widely accepted** |

SHAP's **local accuracy** property means the sum of all feature contributions equals the difference between the prediction and the base value. This is not true for LIME approximations. For a trading context where attribution is used for regulatory audit, exact attributions are essential.

**Why TreeExplainer specifically:** `shap.TreeExplainer` uses the tree structure to compute exact Shapley values in polynomial time (rather than the exponential-time exact algorithm). For XGBoost and LightGBM models with hundreds of trees, `TreeExplainer` runs in approximately 1ms per prediction — fast enough to include in every inference response without impacting the 50ms p95 SLA.

**KernelExplainer for neural networks:** SHAP's `KernelExplainer` is model-agnostic and works for any differentiable or black-box model. It is slower (~100ms for TFT/PatchTST models) but is run asynchronously and cached per (model_version, feature_hash) pair. If SHAP computation fails (numerical instability, missing feature), the response returns all contributions set to 0.0 and `base_value` equal to the model's historical mean prediction — a `503` is never raised for explain failures.

---

## Pydantic V2 Strict Mode Throughout

**Decision:** All request and response schemas use `model_config = ConfigDict(strict=True)`. All Python source files use `from __future__ import annotations`.

**Why strict mode:** Pydantic V2's default mode performs coercions — a string `"1.5"` is silently converted to a float. In a trading context, silent coercions are dangerous: a request with `"risk_score": "none"` that silently becomes `0.0` would suppress a risk warning rather than returning a 422 error.

With `strict=True`, any type mismatch raises a `ValidationError` immediately. The caller receives a field-level 422 error identifying exactly which field has the wrong type, rather than a silent data corruption followed by an incorrect prediction.

**Why `from __future__ import annotations`:** Python 3.11+ evaluates type annotations lazily with this import, enabling forward references in type hints without workarounds. It is also required for `mypy --strict` to fully resolve generic types in Pydantic V2 models.

---

## The Six-Gate Promotion Pipeline Rationale

**Decision:** Every challenger model must pass six structured gates (DATA, PREDICTIVE, CALIBRATION, EXECUTION, RISK, STABILITY) before replacing the champion.

**Why six separate gates and not a single composite score:**

A composite score trades off between dimensions — a model could score highly on predictive power while having poor calibration, and the composite score might still be acceptable. In a live trading system, poor calibration means position sizes based on the model's probability estimates are systematically wrong. Each gate enforces a distinct non-negotiable requirement.

**Gate-by-gate rationale:**

| Gate | Why it exists |
|---|---|
| DATA | Prevents models trained on contaminated or insufficient OOS evidence from ever reaching production. OOS contamination is irreversible — once a model has seen its test set, its metrics cannot be trusted. |
| PREDICTIVE | Prevents replacing a champion with a challenger that is statistically indistinguishable or worse. The 0.005 minimum IC improvement margin filters out noise-level improvements that would not survive a regime shift. |
| CALIBRATION | Ensures probability estimates are reliable. An uncalibrated model that says "70% probability of target hit" when the actual rate is 50% leads to systematically oversized positions. |
| EXECUTION | Verifies that the model's signals translate into positive economic value after real-world transaction costs. A model with high IC but high turnover can have negative net returns after 10bp round-trip costs. |
| RISK | Ensures the challenger does not introduce catastrophic drawdown risk. A 2pp drawdown tolerance relative to the champion prevents ratcheting risk with each model refresh. |
| STABILITY | Guards against promoting a model that is already decaying or operating on drifted features. A model that passes all other gates but shows IC decay status `FAILED` would likely underperform immediately after promotion. |

**Why HUMAN_APPROVAL_REQUIRED:** All six gates passing is a necessary but not sufficient condition for live promotion. A human reviewer must confirm that the challenger was evaluated correctly, that no unusual market conditions affected the evaluation period, and that the operator is comfortable promoting the model. The approvalToken mechanism provides an auditable record of who approved each promotion and when.

---

## WebSocket for Real-Time Signal Streaming

**Decision:** Real-time signals are pushed to alpha-forge via WebSocket at `WS /v2/stream/signals`, not via REST polling or server-sent events (SSE).

**Why WebSocket and not REST polling:**

REST polling at the frequency required for live trading (sub-second) creates N concurrent HTTP requests per connected alpha-forge client. At 10 connected sessions each polling at 500ms intervals, that is 20 HTTP requests per second consuming TCP connections, HTTP handshakes, and JSON parsing overhead per request.

**Why WebSocket and not SSE:**

SSE (Server-Sent Events) is a one-way server-push protocol. The WebSocket protocol supports bidirectional messaging, which allows the client to send symbol subscription filters over the same connection in the future — without requiring a second HTTP channel. SSE would also require a separate HTTP connection for any client-to-server communication.

**Connection lifecycle:**

```
1. Client connects: WS /v2/stream/signals?api_key=<key>
2. Server validates API key → reject with code 4001 on failure
3. Server adds client to broadcast set
4. MetaDecisionEngine.decide() emits a MetaOutput
5. SignalStreamer broadcasts SignalEvent JSON to all connected clients
6. Client disconnects → server removes from broadcast set (graceful close)
7. Service shutdown → server sends close frame to all clients
```

**Backpressure:** If a client is consuming signals slower than the MetaDecisionEngine is producing them, the SignalStreamer drops stale signals rather than buffering them indefinitely. Each symbol's latest signal overwrites the previous undelivered signal in a per-client deque.

---

## Component Interaction Map

The following diagram shows which internal components communicate with which, and via what mechanism.

```
FastAPI Router
├── (HTTP request) ──────────────────────────────────────────────┐
│                                                                  ▼
│                                                       FeaturePipeline
│                                                       ├── DataServiceClient ──→ data-service2.0 (REST + gRPC)
│                                                       ├── SentinelPulseClient → SentinelPulse (REST)
│                                                       ├── QlibFeatureEngine (in-process)
│                                                       ├── LeakageValidator (in-process, training only)
│                                                       └── RedisCache ──────────→ Redis
│                                                                  │
│                                                          FeatureVector
│                                                                  ▼
│                                                         Model Layer
│                                                         ├── RegimeClassifier
│                                                         ├── StockRanker
│                                                         ├── StrategySelector
│                                                         ├── RiskPredictor
│                                                         ├── PortfolioOptimizer
│                                                         ├── RLExecutionAgent
│                                                         ├── PriceForecaster
│                                                         └── IVRegimeClassifier
│                                                                  │
│                                                         Model Outputs
│                                                                  ▼
│                                                    MetaDecisionEngine
│                                                    ├── CalibrationLayer (in-process)
│                                                    ├── EnsembleWeighter (in-process)
│                                                    ├── AbstentionPolicy (in-process)
│                                                    ├── LLMNewsReasoner ──────────→ FinBERT/FinGPT (HuggingFace, in-process)
│                                                    └── ConfidenceDecomposer (in-process)
│                                                                  │
│                                                           MetaOutput
│                                                                  │
│                              ┌───────────────────────────────────┤
│                              ▼                                   ▼
│                    ModelExplainer                       SignalStreamer
│                    (SHAP, in-process)                   (WebSocket push)
│                              │                                   │
└── (HTTP response) ◄──────────┤                                   │
                               │                                   ▼
                        AuditLogger ◄──────────────────── alpha-forge (WS client)
                        (append-only JSONL)
                               ▲
             ┌─────────────────┴───────────────┐
             │                                 │
      TrainingPipeline                  ModelPromotion
      ├── PurgedKFoldSplitter           ├── Six gate evaluators
      ├── CPCV                          └── ApprovalTokenManager
      ├── OptunaHPO ──────→ MLflow
      └── OnlineLearner
                               ▲
                               │
                        ModelRegistry ──────→ MODEL_ARTIFACTS_PATH (filesystem)
                               ▲
                               │
                        DriftMonitor
                        ├── Evidently AI (in-process)
                        └── NannyML (in-process)
```

Every arrow that crosses the process boundary (data-service2.0, SentinelPulse, Redis, MLflow, FinBERT/FinGPT, filesystem) has an explicit failure mode and a defined fallback behavior. Every arrow that stays within the process is synchronous and deterministic.
