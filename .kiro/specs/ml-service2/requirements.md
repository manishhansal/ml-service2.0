# Requirements Document

## Introduction

ml-service2.0 is a standalone, institutional-grade, self-learning ML microservice extracted from the alpha-forge repository and rebuilt as an independent service. It serves as the dedicated AI/ML inference, training, and monitoring layer for the AlphaForge trading platform and is designed to be consumed by future clients beyond alpha-forge.

The service operates within a three-repository architecture:

- **alpha-forge** (Consumer) — Next.js + Python trading platform that calls ml-service2.0 over REST/WebSocket for all ML predictions and signal generation.
- **data-service2.0** (Market Data Provider, port 8200) — The single authority for all market data (NSE/NFO, Crypto, OHLCV, live ticks, option chains, F&O universe). ml-service2.0 MUST fetch all market features exclusively from data-service2.0. Direct connections to NSE, Angel One, Upstox, Binance, or any market data provider are permanently forbidden.
- **SentinelPulse** (NLP/News Intelligence, port 3001) — PIT-correct news sentiment, event impact scores, market regime context, and ML training features. ml-service2.0 consumes SentinelPulse for news-derived feature enrichment and the LLM meta-decision engine.

The service upgrades all six existing model families (regime classifier, stock ranker, strategy selector, risk predictor, portfolio optimizer, RL execution agent), adds a new LLM Meta-Decision Engine, implements Qlib-based feature engineering with Alpha158/Alpha360 factor libraries, integrates FinRL deep reinforcement learning, deploys Evidently AI / NannyML drift monitoring, and supports a full SDLC lifecycle from hypothesis through shadow paper trading to live execution with human approval gates.

---

## Glossary

- **ML_Service**: ml-service2.0 — the standalone service described in this document.
- **Data_Service**: data-service2.0 running at port 8200 — the sole source of market data for the ML_Service.
- **SentinelPulse**: The NLP/news intelligence service at port 3001 providing PIT-correct sentiment and event impact features.
- **Alpha_Forge**: The Next.js + Python trading platform that is the primary consumer of ML_Service.
- **Feature_Pipeline**: The PIT-correct ingestion and engineering layer that transforms raw data from Data_Service and SentinelPulse into model-ready feature vectors.
- **Qlib**: Microsoft's AI-oriented quantitative investment platform providing the Alpha158/Alpha360 factor libraries and double-ensembled purged K-fold cross-validation.
- **Alpha158**: Qlib's 158-factor stock feature library for Chinese and Indian equity markets.
- **Alpha360**: Qlib's 360-dimensional higher-order cross-sectional factor library.
- **FinRL**: An open-source deep reinforcement learning framework for financial portfolio allocation and trade execution.
- **FinGPT**: Open-source large language model fine-tuned on financial text.
- **FinBERT**: BERT-based pre-trained transformer for financial sentiment classification.
- **Meta_Decision_Engine**: The LLM-augmented arbiter that combines all model outputs and SentinelPulse news context into a final Go/No-Go trading decision with XAI rationale.
- **Riskfolio_Lib**: Python library for hierarchical risk parity (HRP) and CVaR portfolio optimization.
- **Evidently_AI**: Open-source ML monitoring library for data drift, concept drift, and PSI detection.
- **NannyML**: Open-source ML monitoring library for post-deployment performance estimation without ground-truth labels.
- **NautilusTrader**: Event-driven, low-latency trading framework used as the architectural pattern for deterministic backtesting.
- **PIT_Correctness**: Point-in-time correctness — the guarantee that no feature vector used during training or inference contains information that would not have been available at the time the prediction was made.
- **Purged_KFold_CV**: Purged K-fold cross-validation that removes training samples temporally adjacent to validation samples to prevent label leakage.
- **Embargo_Period**: A time buffer between training and validation folds in purged K-fold CV, typically 5–20 trading days, to prevent leakage through autocorrelation.
- **PredictionProvenance**: Enumeration classifying the source and evidence quality of every model prediction: TRAINED_MODEL | HEURISTIC | INSUFFICIENT_EVIDENCE | UNAVAILABLE.
- **DeploymentMode**: Enumeration controlling heuristic fallback behaviour: RESEARCH | PAPER | SHADOW | VALIDATED_ML_ONLY.
- **Champion_Model**: The current production model registered in the Model_Registry for a given model family.
- **Challenger_Model**: A newly trained candidate model evaluated against the champion through six promotion gates.
- **Promotion_Gate**: One of six structured evaluation dimensions (DATA, PREDICTIVE, CALIBRATION, EXECUTION, RISK, STABILITY) that a challenger must pass to replace the champion.
- **Model_Acceptance_Gate**: Threshold criteria (IC > 0.02, net Sharpe > 0 after costs, CPCV fraction positive > 0.50) that a model must pass to be considered a valid challenger.
- **SHAP**: SHapley Additive exPlanations — the XAI method used to produce feature contribution scores for every prediction.
- **MLflow**: Experiment tracking and model registry platform.
- **gRPC**: High-performance RPC protocol used for low-latency feature fetching between ML_Service and Data_Service.
- **VPIN**: Volume-synchronized Probability of Informed Trading — a measure of adverse selection risk.
- **GEX**: Dealer Gamma Exposure — aggregate dealer gamma across all strikes.
- **SVI**: Stochastic Volatility Inspired parametric model for fitting implied volatility smiles.
- **HRP**: Hierarchical Risk Parity — a portfolio optimization method based on hierarchical clustering.
- **CVaR**: Conditional Value at Risk — expected loss in the tail of the return distribution.
- **PSI**: Population Stability Index — a metric measuring how much a feature distribution has shifted between two time periods.
- **IC**: Information Coefficient — rank correlation between predicted scores and realized returns.
- **DataConfidenceScore**: Integer in [0, 95] produced by Data_Service's quality gate; ML_Service requires a minimum threshold to allow signal generation.
- **signalEngineAllowed**: Boolean flag from Data_Service's quality gate; ML_Service MUST check this flag before consuming features for inference.
- **approvalToken**: A cryptographically signed token issued by the human approval workflow authorizing promotion of a challenger model to production.
- **CPCV**: Combinatorial Purged Cross-Validation — an advanced backtesting framework from Marcos López de Prado that generates multiple test paths to estimate strategy performance with backtest overfitting probability.
- **OOS**: Out-of-sample — data or predictions that come from time periods or folds the model was not trained on.
- **NSE**: National Stock Exchange of India.
- **NFO**: NSE Futures and Options segment.

---

## Requirements

### Requirement 1: Three-Repo Architecture and Data Ingestion Contracts

**User Story:** As a quantitative researcher, I want ml-service2.0 to enforce strict source-of-truth boundaries for market data and news intelligence so that no feature used in training or inference introduces look-ahead bias or bypasses the established data authority chain.

#### Acceptance Criteria

1. THE ML_Service SHALL fetch all market data features exclusively from Data_Service at `http://{DATA_SERVICE_2_URL}/v1/` and SHALL NOT establish direct connections to NSE, BSE, Angel One, Upstox, Binance, Deribit, or any market data provider.
2. THE ML_Service SHALL fetch all news context, sentiment scores, and NLP-derived features exclusively from SentinelPulse at `http://{SENTINEL_PULSE_URL}/api/v1/`.
3. WHEN the ML_Service requests features from Data_Service, THE ML_Service SHALL include an `X-API-KEY` header or a JWT Bearer token in every request.
4. IF Data_Service returns an authentication or authorization error in response to a feature request, THEN THE ML_Service SHALL abort that request, log a warning message indicating the credential type used and the instrument requested, and return `PredictionProvenance.UNAVAILABLE` to the caller without retrying.
5. WHEN Data_Service returns a response where `signalEngineAllowed` is `false`, THE ML_Service SHALL reject the feature batch and return `PredictionProvenance.UNAVAILABLE` to the caller.
6. IF Data_Service returns a response where `signalEngineAllowed` is `false`, THEN THE ML_Service SHALL log a structured warning containing at minimum the instrument identifier, the request timestamp, and the `DataConfidenceScore` value received.
7. WHEN Data_Service returns a `DataConfidenceScore` below the configured `min_confidence_score` threshold (default 70 on a 0–100 integer scale), THE ML_Service SHALL treat the batch as low-quality and return `PredictionProvenance.INSUFFICIENT_EVIDENCE` to the caller.
8. THE ML_Service SHALL use the canonical timeframes `1m, 5m, 10m, 15m, 30m, 1h, 1d, 1w, 1M` when requesting historical OHLCV data from Data_Service, and SHALL NEVER request the `3m` timeframe for Indian market data.
9. WHEN SentinelPulse returns news context for an instrument via `GET /api/v1/alphaforge/news-context/:instrument`, THE ML_Service SHALL extract `news_impact_score`, `impact_direction`, `impact_confidence`, and all `sentiment` sub-fields and pass them into the Feature_Pipeline as first-class features.
10. IF a SentinelPulse news context response is missing one or more of the mandatory fields `news_impact_score`, `impact_direction`, `impact_confidence`, or any `sentiment` sub-field, THEN THE ML_Service SHALL treat that response equivalently to SentinelPulse being unreachable for that instrument and apply the degraded-mode substitution defined in criterion 11.
11. WHEN SentinelPulse is unreachable, THE ML_Service SHALL proceed with inference using only market-data features, SHALL set all numeric news-derived feature fields to `0.0` and all directional news-derived feature fields (including `impact_direction`) to `NEUTRAL`, and SHALL record the affected instrument, the timestamp, and the list of zeroed feature fields in the prediction's explainability report.
12. WHEN Data_Service is unreachable, THE ML_Service SHALL return `PredictionProvenance.UNAVAILABLE` for all prediction endpoints and SHALL NOT fall back to any alternative market data source.
13. WHEN the ML_Service populates a training dataset, THE ML_Service SHALL source news feature vectors exclusively from the SentinelPulse PIT-correct endpoints `GET /api/v1/ml/features/asset/:assetId`, `GET /api/v1/ml/training/samples`, and `GET /api/v1/ml/historical-reactions`.
14. WHEN the ML_Service ingests a feature vector into the training pipeline, THE ML_Service SHALL validate that the vector carries the flag `look_ahead_validated: true` and SHALL reject any vector where that flag is `false` or absent.
15. IF a SentinelPulse training sample has `look_ahead_validated: false` or the flag is absent, THEN THE ML_Service SHALL discard that sample, log the instrument identifier and the sample timestamp, and continue processing the remaining batch.
16. IF the proportion of discarded samples in a single training batch exceeds 10% of total samples in that batch, THEN THE ML_Service SHALL emit a warning indicating the batch identifier, the total sample count, and the discard count, and SHALL NOT proceed with training on that batch without explicit operator acknowledgement.
17. WHEN the ML_Service establishes a gRPC streaming connection to Data_Service for live tick features, THE ML_Service SHALL apply a connection timeout of 5 seconds and a per-call deadline of 2 seconds.
18. IF a gRPC stream from Data_Service is interrupted after establishment, THEN THE ML_Service SHALL attempt to re-establish the stream up to 3 times with a 1-second delay between attempts, and IF all retry attempts fail, THEN THE ML_Service SHALL return `PredictionProvenance.UNAVAILABLE` for any prediction request depending on that stream.

---

### Requirement 2: PIT-Correct Feature Engineering Pipeline

**User Story:** As a quantitative researcher, I want all features used in training and inference to be point-in-time correct so that backtested performance faithfully represents what would have been achievable in live trading.

#### Acceptance Criteria

1. THE Feature_Pipeline SHALL enforce PIT correctness by timestamping every feature vector with a UTC ISO-8601 datetime representing the exact moment at which all constituent raw data would have been receivable by a downstream consumer in a live market scenario, with no future information included.
2. WHEN the Feature_Pipeline constructs a training sample for date T, THE Feature_Pipeline SHALL use only data with observation timestamps strictly before 09:15 IST on date T (NSE regular session open); IF any data record has an observation timestamp at or after 09:15 IST on date T, THEN THE Feature_Pipeline SHALL discard that record and log the record's instrument identifier and timestamp.
3. THE Feature_Pipeline SHALL integrate Microsoft Qlib's Alpha158 factor library to produce at minimum 158 cross-sectional equity factors for each symbol in the F&O universe.
4. WHERE the Alpha360 factor library is enabled via configuration, THE Feature_Pipeline SHALL produce an additional 360-dimensional higher-order factor vector for each symbol.
5. THE Feature_Pipeline SHALL implement a `LeakageValidator` that, given a feature matrix and a label vector, computes the Pearson correlation between each feature and realized returns over look-ahead windows of 1 to 22 trading days, and SHALL raise a `PITViolationError` containing the feature name, look-ahead window, and correlation value when any feature exhibits a look-ahead correlation exceeding 0.05 in absolute value.
6. WHEN processing option chain features (IV, Greeks, GEX, VPIN), THE Feature_Pipeline SHALL use only the chain snapshot with the most recent timestamp strictly before the feature vector's timestamp; IF no snapshot exists within the prior 60 minutes, THEN THE Feature_Pipeline SHALL treat those features as unavailable and substitute NaN.
7. THE Feature_Pipeline SHALL fetch the following feature families from Data_Service with a request timeout of 10 seconds per call; IF a Data_Service call times out or returns an error, THEN THE Feature_Pipeline SHALL mark that feature family as unavailable for the affected instruments and continue: OHLCV candles, intraday tick aggregates, option chain snapshots, Greeks (delta, gamma, theta, vega), GEX by strike, VPIN bucket history, IV surface by expiry, F&O OI and delivery percentage, market breadth (advance/decline ratio, % above SMAs), India VIX, and sector performance.
8. THE Feature_Pipeline SHALL fetch the following feature families from SentinelPulse with a request timeout of 10 seconds per call; IF a SentinelPulse call times out or returns an error, THEN THE Feature_Pipeline SHALL apply the degraded-mode substitution from Requirement 1, criterion 11: `news_impact_score`, `impact_direction`, `impact_confidence`, `sentiment.overall`, `sentiment.market`, `sentiment.company`, `sentiment.macro`, `sentiment.risk`, and `market_regime`.
9. THE Feature_Pipeline SHALL produce a `FeatureQualityReport` for every batch containing: total_features_requested, missing_count, imputed_count, rejected_count, unavailable_families (list), and pit_violations_count; THE Feature_Pipeline SHALL expose this report through `GET /features/quality` with a response time of ≤ 2 seconds at p95.
10. FOR ALL valid feature vectors v, computing the feature vector on the same input data twice SHALL produce an identical result; IF the Feature_Pipeline detects a non-idempotent output for any vector, THE Feature_Pipeline SHALL raise a `FeatureIdempotenceViolation` error and log the input data hash and both output vectors.
11. THE Feature_Pipeline SHALL support backtest mode in which all Data_Service calls include a `?pit_date=YYYY-MM-DD` query parameter; WHEN backtest mode is active, IF the response from Data_Service contains any data with timestamp after the specified `pit_date`, THE Feature_Pipeline SHALL discard that data and record the count of discarded records in the `FeatureQualityReport`.
12. WHEN a feature family is entirely unavailable (e.g. option chain for a non-F&O symbol), THE Feature_Pipeline SHALL substitute that family's features with NaN and record the substitution in the `FeatureQualityReport`, rather than raising an exception.
13. THE Feature_Pipeline SHALL process a batch of 500 symbols within 60 seconds when Data_Service and SentinelPulse respond within their normal latency envelopes.

---

### Requirement 3: Qlib-Based Model Training Pipeline

**User Story:** As a quantitative researcher, I want models trained using Qlib's double-ensembled purged K-fold cross-validation with embargo periods so that reported backtested performance is free of look-ahead bias and selection bias.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL implement double-ensembled purged K-fold cross-validation using Qlib's `RollingPurgedKFold` with a configurable embargo period (default 10 trading days, minimum 5 trading days) for all supervised models (regime classifier, stock ranker, strategy selector, risk predictor).
2. WHEN splitting data into training and validation folds, THE Training_Pipeline SHALL purge all training samples whose label observation window shares at least one trading day in common with any validation sample's feature observation window.
3. THE Training_Pipeline SHALL compute the Information Coefficient (IC) as the Spearman rank correlation between predicted scores and realized returns on each held-out fold; IF a model's mean IC across all folds is below 0.02, THEN THE Training_Pipeline SHALL archive the model with a `REJECTED` status and a rejection reason of `IC_BELOW_THRESHOLD`, and SHALL block that model from advancing in the lifecycle.
4. THE Training_Pipeline SHALL compute net Sharpe ratio after simulated transaction costs of 10 basis points per round-trip trade on each validation fold; IF a model's net Sharpe ratio on any fold is below 0.0, THEN THE Training_Pipeline SHALL archive the model with status `REJECTED` and rejection reason `NEGATIVE_NET_SHARPE`.
5. THE Training_Pipeline SHALL apply CPCV to generate at least 10 overlapping test paths and SHALL compute the backtest overfitting probability (PBO); IF PBO exceeds 0.5, THEN THE Training_Pipeline SHALL archive the model with status `REJECTED` and rejection reason `HIGH_PBO`.
6. THE Training_Pipeline SHALL use Optuna for hyperparameter optimization with a minimum of 50 trials per model, using the mean Spearman IC across all held-out validation folds as the objective metric.
7. THE Training_Pipeline SHALL track every experiment run in MLflow, logging: model name, version, training date range, validation date range, all hyperparameters, IC per fold, net Sharpe, max drawdown, PBO, and a SHA-256 hash of the training dataset.
8. WHEN the Training_Pipeline receives a PIT-violation report from the LeakageValidator, THE Training_Pipeline SHALL abort training, log the violation with affected feature names and correlation values, and raise a `TrainingAbortedError`.
9. THE Training_Pipeline SHALL produce an immutable, append-only training audit log in structured JSON Lines format, where each entry records the training run ID, start time, end time, dataset hash, model version, gate results, and the approvalToken if one was required.
10. THE Training_Pipeline SHALL implement the complete model SDLC lifecycle: `Idea → Hypothesis → Qlib_Backtest → Shadow_Paper_Trading → Live`, and SHALL enforce that no model advances to a later stage without all required prior-stage gates returning `PASS`.
11. WHEN a model completes the Qlib_Backtest stage and passes the Model_Acceptance_Gate (IC ≥ 0.02 AND net Sharpe ≥ 0.0 AND PBO ≤ 0.5), THE Training_Pipeline SHALL automatically enroll the model in Shadow_Paper_Trading for a configurable minimum duration (default 20 trading days) before eligibility for Live promotion.

---

### Requirement 4: Market Regime Classifier (Upgraded)

**User Story:** As a signal engineer, I want an upgraded market regime classifier that uses Qlib Alpha factors and SentinelPulse macro context so that regime detection is more robust and less susceptible to whipsaws during volatile transitions.

#### Acceptance Criteria

1. THE Regime_Classifier SHALL classify the current market into one of six regimes: `strong_bull`, `bull`, `sideways`, `volatile`, `bear`, or `crash`, with a confidence score in [0, 1] and a probability distribution across all six classes that sums to 1.0.
2. IF a trained XGBoost model artifact with a valid model version string and training timestamp is registered in the Model_Registry, THEN THE Regime_Classifier SHALL use that artifact for inference; otherwise THE Regime_Classifier SHALL use the rule-based heuristic fallback and set `PredictionProvenance.HEURISTIC`.
3. THE Regime_Classifier SHALL incorporate at minimum the following feature groups, each with a data freshness age not exceeding 5 minutes at inference time: NIFTY/BANKNIFTY price-derived features (ATR, ADX, RSI, MACD), India VIX level and rate-of-change, market breadth (advance/decline ratio, % F&O stocks above 20 SMA), FII/DII net flow in crores, put-call ratio, SentinelPulse `market_regime`, and Qlib Alpha158 cross-sectional breadth factors.
4. WHEN the Regime_Classifier receives a prediction request, THE Regime_Classifier SHALL respond within 50 milliseconds at the 95th percentile (p95) measured over a rolling 5-minute window.
5. THE Regime_Classifier SHALL attach a `SHAP` explanation to every prediction response, listing the top 10 features by absolute SHAP contribution along with their direction (positive/negative) and raw values.
6. WHEN the Regime_Classifier produces a prediction with confidence below 0.35, THE Regime_Classifier SHALL set `PredictionProvenance.INSUFFICIENT_EVIDENCE` and include a reason code `LOW_REGIME_CONFIDENCE` in the response.
7. WHEN consecutive predictions produce different regime labels for the same symbol, THE Regime_Classifier SHALL emit a structured `regime_transition_event` log entry containing the previous regime, the new regime, the confidence delta, and the top 3 contributing features.
8. WHEN the PerformanceMonitor detects IC degradation exceeding 20% from the 90-day trailing baseline over a rolling 30-day window, THE Regime_Classifier SHALL trigger an incremental model update using the most recent 60 trading days of data; WHEN the incremental update is in progress, THE Regime_Classifier SHALL continue serving predictions from the existing champion artifact without interruption.
9. WHEN the incremental update artifact is available, THE Regime_Classifier SHALL validate it on the most recent 10 trading days before activation; IF the updated artifact's IC on that window is lower than the current champion's IC on the same window, THEN THE Regime_Classifier SHALL discard the update and log an `ONLINE_UPDATE_REJECTED` event.

---

### Requirement 5: Stock Ranker (Upgraded)

**User Story:** As a signal engineer, I want an upgraded stock ranker that incorporates Qlib Alpha360 factors, SentinelPulse news impact scores, and regime-aware weighting so that the F&O universe ranking is more predictive across different market conditions.

#### Acceptance Criteria

1. THE Stock_Ranker SHALL produce an outperformance score in [0, 100] and a rank for each symbol in the input batch, relative to the current market regime.
2. WHEN a trained LightGBM model artifact is available and registered in the Model_Registry, THE Stock_Ranker SHALL use the trained model; otherwise THE Stock_Ranker SHALL use the score-based heuristic fallback and set `PredictionProvenance.HEURISTIC`.
3. THE Stock_Ranker SHALL incorporate the following feature groups in addition to existing features: Qlib Alpha360 factors for each symbol, SentinelPulse `news_impact_score` and `impact_direction` for each symbol (fetched via `GET /api/v1/alphaforge/news-context/:instrument`), delivery percentage, OI build-up score, IV rank percentile, and sector relative strength vs NIFTY.
4. THE Stock_Ranker SHALL respond to a batch ranking request of up to 200 symbols within 200 milliseconds at p95.
5. THE Stock_Ranker SHALL return a `SHAP`-based factor decomposition for each ranked symbol, listing the top 5 features contributing to the score.
6. WHEN the input batch contains symbols for which SentinelPulse news context is unavailable, THE Stock_Ranker SHALL proceed with the remaining features and record the missing news context in the per-symbol `FeatureQualityReport`.
7. FOR ALL inputs i1 and i2 where i1 and i2 are identical feature vectors for the same symbol and the same model version is loaded, `ranker.rank(i1) == ranker.rank(i2)` (idempotence property).
8. THE Stock_Ranker SHALL support regime-conditioned ranking: the feature weights applied SHALL differ by regime, using regime-specific weight tables stored in the model artifact.

---

### Requirement 6: Strategy Selector (Upgraded)

**User Story:** As a signal engineer, I want the strategy selector to use regime context and IV regime to choose from the full set of eight trading strategies with calibrated probabilities so that the recommended execution approach matches the current market microstructure.

#### Acceptance Criteria

1. THE Strategy_Selector SHALL recommend one of eight strategies: `breakout`, `momentum`, `trend_following`, `mean_reversion`, `vwap_bounce`, `range_trading`, `scalping`, or `volatility_breakout`, along with a confidence score in [0, 1] and a probability distribution across all eight strategies.
2. WHEN a trained CatBoost model artifact is available, THE Strategy_Selector SHALL use the trained model; otherwise THE Strategy_Selector SHALL apply the rule-based heuristic and set `PredictionProvenance.HEURISTIC`.
3. THE Strategy_Selector SHALL incorporate the IV regime classification (`CRUSH` | `STABLE` | `SPIKE`) as a mandatory input feature.
4. THE Strategy_Selector SHALL respond within 50 milliseconds at p95.
5. THE Strategy_Selector SHALL return calibrated probabilities for all eight strategies using isotonic regression calibration trained on OOS fold predictions.
6. WHEN the Strategy_Selector confidence is below 0.40 for the top-ranked strategy, THE Strategy_Selector SHALL include at least two alternative strategies in the response with their respective probabilities.

---

### Requirement 7: Risk Predictor (Upgraded)

**User Story:** As a risk manager, I want the risk predictor to estimate P(stop hit), P(target hit), expected drawdown, and suggested position size with calibrated confidence intervals so that position sizing is grounded in empirical evidence rather than fixed rules.

#### Acceptance Criteria

1. THE Risk_Predictor SHALL produce: `prob_stop_hit` ∈ [0,1], `prob_target_hit` ∈ [0,1], `expected_drawdown_pct`, `suggested_position_size_pct`, `risk_score` ∈ [0,10], and a SHAP-based factor breakdown.
2. WHEN a trained XGBoost ensemble of three models is available, THE Risk_Predictor SHALL use ensemble averaging for inference; otherwise THE Risk_Predictor SHALL use the rule-based fallback and set `PredictionProvenance.HEURISTIC`.
3. THE Risk_Predictor SHALL derive `stop_distance_atr`, `target_distance_atr`, and `risk_reward_ratio` from the trade geometry inputs before inference.
4. THE Risk_Predictor SHALL incorporate `regime_encoded`, `vix_regime`, `atr_pct`, `oi_buildup_score`, SentinelPulse `news_impact_score`, and SentinelPulse `sentiment.risk` as features.
5. THE Risk_Predictor SHALL respond within 50 milliseconds at p95.
6. THE Risk_Predictor SHALL produce `prob_stop_hit + prob_target_hit ≤ 1.0` for all inputs, and SHALL log a `CALIBRATION_VIOLATION` warning when this invariant would be breached before clamping.
7. WHEN `risk_score > 7.0` and `DeploymentMode == VALIDATED_ML_ONLY`, THE Risk_Predictor SHALL append the reason code `HIGH_RISK_BLOCKED` to the response, and the Meta_Decision_Engine SHALL override the final action to `NO_TRADE`.

---

### Requirement 8: Portfolio Optimizer (Upgraded)

**User Story:** As a portfolio manager, I want institutional-grade portfolio optimization using Riskfolio-Lib's HRP and CVaR methods with full constraint support so that capital is allocated efficiently across selected instruments.

#### Acceptance Criteria

1. THE Portfolio_Optimizer SHALL support at minimum the following optimization methods: HRP (Hierarchical Risk Parity), CVaR-minimized MVO, Equal Risk Contribution (ERC), and Maximum Diversification.
2. WHEN the HRP method is selected, THE Portfolio_Optimizer SHALL use Riskfolio-Lib's `HRPOpt` with Ward linkage clustering and returns a weight vector that sums to 1.0 with all weights ≥ 0.
3. WHEN the CVaR method is selected, THE Portfolio_Optimizer SHALL use Riskfolio-Lib's `RiskFolio` with a configurable tail probability alpha (default 0.05) and return weights that sum to 1.0 with all weights ≥ 0.
4. THE Portfolio_Optimizer SHALL enforce a configurable maximum sector weight constraint (default 40%) and SHALL reject any optimization result that violates the constraint, re-running with a binding constraint.
5. THE Portfolio_Optimizer SHALL compute and return: expected return, portfolio volatility, Sharpe ratio, CVaR, max drawdown estimate, and diversification ratio for the resulting allocation.
6. THE Portfolio_Optimizer SHALL respond within 500 milliseconds at p95 for portfolios of up to 50 assets.
7. FOR ALL valid return matrices R and optimization method M, running the optimizer twice with the same inputs and random seed SHALL produce identical weights (determinism property).
8. WHEN the returns matrix contains fewer than 20 observations per asset, THE Portfolio_Optimizer SHALL return `available: false` with reason `INSUFFICIENT_RETURN_HISTORY` rather than producing unreliable estimates.
9. THE Portfolio_Optimizer SHALL attach a `PredictionProvenance` field; when Riskfolio-Lib is available and sufficient data exists the provenance SHALL be `TRAINED_MODEL`; when falling back to equal-weight allocation the provenance SHALL be `HEURISTIC`.

---

### Requirement 9: FinRL Deep Reinforcement Learning Execution Agent

**User Story:** As a signal engineer, I want to replace the rule-based execution policy with a FinRL deep reinforcement learning agent so that execution timing decisions (hold, trail stop, partial exit, full exit, scale in) are learned from historical market behaviour rather than hardcoded heuristics.

#### Acceptance Criteria

1. IF a trained RL_Agent artifact with a valid model version, training timestamp, and OPE Sharpe score is registered in the Model_Registry, THEN THE ML_Service SHALL use the RL_Agent implemented with FinRL-X using a PPO or SAC algorithm as the execution policy, with the algorithm selection configurable via the training pipeline.
2. THE RL_Agent SHALL support the following discrete action space: `ENTER_NOW`, `WAIT`, `SCALE_IN`, `PARTIAL_EXIT`, `FULL_EXIT`, `TIGHTEN_STOP`, `TRAIL_STOP`.
3. THE RL_Agent's observation space SHALL include: unrealized P&L percentage, time in trade (minutes), current regime, volume ratio, price vs VWAP, ATR, momentum, IV regime, SentinelPulse `news_impact_score`, and current risk score from the Risk_Predictor.
4. THE RL_Agent's reward function SHALL be shaped to maximize risk-adjusted return (Sharpe ratio) and SHALL penalize: stop loss hits, holding any open position at session close, and turnover exceeding 10 round-trip trades per session.
5. THE RL_Agent SHALL be trained exclusively on historical simulation data using Data_Service's PIT-correct backtest mode and SHALL never be trained on live or shadow trading data.
6. THE RL_Agent SHALL use FinRL's `StockTradingEnv` as the base environment, extended with the NSE session calendar (375 minutes per session) and Indian market-specific position sizing rules as defined in the Risk_Predictor configuration.
7. WHEN the RL_Agent is unavailable (no trained artifact registered in the Model_Registry), THE ML_Service SHALL fall back to the rule-based policy and set `PredictionProvenance.HEURISTIC` in the response.
8. IF the RL_Agent fails to return an action within 50 milliseconds at p95 measured over a rolling 1-minute window during inference, THEN THE ML_Service SHALL fall back to the rule-based policy for that request and set `PredictionProvenance.HEURISTIC` in the response.
9. THE RL_Agent SHALL support offline policy evaluation (OPE) using importance sampling on the most recent 63 trading days of historical logged data, and SHALL produce an estimated Sharpe ratio as output before deployment.
10. IF the RL_Agent's OPE estimated Sharpe ratio is below 0.0 on the validation period, THEN THE Training_Pipeline SHALL reject the agent, retain the current champion policy, and record the rejection reason in the Model_Registry.

---

### Requirement 10: LLM Meta-Decision Engine

**User Story:** As a trading system architect, I want an LLM-powered meta-decision engine that reasons over all model outputs, SentinelPulse news context, and conflicting technical signals to produce a final Go/No-Go trading decision with a natural-language XAI rationale so that trading decisions are explainable to human reviewers.

#### Acceptance Criteria

1. THE Meta_Decision_Engine SHALL combine outputs from all base models (regime classifier, stock ranker, strategy selector, risk predictor, price forecaster, IV regime classifier, RL execution agent) and SentinelPulse news context into a single `MetaOutput` with fields: `action` (`BUY` | `SELL` | `WAIT` | `NO_TRADE`), `confidence` ∈ [0,1], `uncertainty`, `agreement`, `reason_codes`, `contributing_models`, `ensemble_score`, `decomposition`, `abstention`, and `explainability`.
2. THE Meta_Decision_Engine SHALL calibrate each base model's raw score using Platt scaling or isotonic regression, selecting the method with the lower ECE (Expected Calibration Error) on the held-out OOS fold, and SHALL reject a calibration update if the resulting ECE exceeds 0.15, retaining the previous calibrator in that case.
3. THE Meta_Decision_Engine SHALL apply regime-aware ensemble weighting, assigning each model a weight proportional to its IC in the current regime, with no single model's weight exceeding 0.4 and no model's weight falling below 0.05, such that all weights sum to 1.0; models with an IC below 0.05 in the current regime SHALL receive the minimum weight of 0.05.
4. WHEN model signals disagree directionally (one model predicts BUY, another predicts SELL), THE Meta_Decision_Engine SHALL compute an `agreement_ratio` as the fraction of available (non-UNAVAILABLE) models whose directional prediction matches the plurality direction, and SHALL activate the abstention policy if the ratio falls below 0.5.
5. THE Meta_Decision_Engine SHALL integrate FinGPT or FinBERT (via LangChain or LlamaIndex) to process the SentinelPulse news context package and produce a `news_sentiment_signal` with direction (+1/0/-1), confidence ∈ [0,1], and a rationale of no more than 30 words extracted from the news context.
6. WHEN the SentinelPulse `news_impact_score` exceeds 0.7 AND `impact_direction` conflicts with the ensemble direction, THE Meta_Decision_Engine SHALL override the action to `NO_TRADE` and include reason code `NEWS_CONFLICT_OVERRIDE`.
7. THE Meta_Decision_Engine SHALL implement an `AbstentionPolicy` that returns `NO_TRADE` when any of the following conditions hold: `agreement_ratio < 0.5`, `data_quality < 0.6`, `mean_confidence < 0.35`, `risk.prob_stop_hit > 0.65`, or the number of available (non-UNAVAILABLE) models is fewer than 3.
8. THE Meta_Decision_Engine SHALL produce a `ConfidenceDecomposition` breaking the final confidence score into five components: `base_confidence`, `calibration_quality`, `agreement_bonus`, `data_quality_factor`, and `regime_confidence_factor`.
9. WHEN the Meta_Decision_Engine receives a `POST /meta/decide` request, THE Meta_Decision_Engine SHALL return the `MetaOutput` within 150 milliseconds at p95 when the news context is already cached; IF the LLM inference step does not complete within 120 milliseconds, THEN THE Meta_Decision_Engine SHALL use the most recent cached `news_sentiment_signal` for that symbol (if available within the last 60 seconds) and include reason code `NEWS_CACHE_FALLBACK` in the response, or set `news_sentiment_signal.direction` to 0 and include reason code `NEWS_TIMEOUT` if no cached signal exists.
10. WHEN `DeploymentMode == VALIDATED_ML_ONLY`, THE Meta_Decision_Engine SHALL only accept `BUY` or `SELL` actions from base models with `PredictionProvenance.TRAINED_MODEL`, converting all `HEURISTIC` model contributions to neutral (direction = 0) before ensemble computation.
11. THE Meta_Decision_Engine SHALL expose a fit endpoint `POST /meta/fit` that accepts a list of at least 30 OOS prediction records and retrains the meta-layer calibrators without touching the base models; IF the submitted list contains fewer than 30 records or any record is missing required fields (`model_id`, `raw_score`, `realized_outcome`), THEN THE Meta_Decision_Engine SHALL reject the request with an error indicating the validation failure and leave existing calibrators unchanged.
12. FOR ALL inputs where all base models report `UNAVAILABLE`, THE Meta_Decision_Engine SHALL return `action: NO_TRADE` with `PredictionProvenance.UNAVAILABLE` (absorbing identity property).

---

### Requirement 11: Signal Generation and XAI

**User Story:** As a trader, I want every trading signal to carry a full SHAP-based explainability report and a natural-language rationale so that I can understand and audit the basis for every recommendation before acting on it.

#### Acceptance Criteria

1. THE ML_Service SHALL attach a SHAP explanation to every prediction response from every model endpoint, listing at minimum the top 10 features by absolute SHAP value, their direction (positive/negative), and their raw values.
2. THE ML_Service SHALL register each trained model with the `ModelExplainer` using SHAP TreeExplainer for tree-based models (XGBoost, LightGBM, CatBoost) and SHAP KernelExplainer for neural network models.
3. WHEN a prediction carries `PredictionProvenance.HEURISTIC`, THE ML_Service SHALL still return a SHAP-equivalent rule attribution listing which rule conditions were triggered and their contribution to the final score.
4. THE ML_Service SHALL expose `POST /explain/{model_name}` accepting a feature dict and a prediction, returning a full `ExplainResponse` with `contributions`, `base_value`, `total_positive`, `total_negative`, and `top_drivers`.
5. THE Meta_Decision_Engine SHALL include in the `MetaOutput.explainability` field: the top 5 contributing features across all models, a list of models with conflicting signals, and a one-sentence natural-language rationale generated by the LLM component.
6. WHEN a model is explainability-registered and a SHAP computation fails (numerical instability, missing feature), THE ML_Service SHALL log the failure with the model name and feature dict and SHALL return `ExplainResponse` with all contributions set to 0.0 and `base_value` set to the model's historical mean prediction, rather than returning a 500 error.

---

### Requirement 12: Evidently AI / NannyML Drift Monitoring

**User Story:** As an ML operations engineer, I want automated drift monitoring using Evidently AI and NannyML so that feature drift, concept drift, and PSI degradation are detected and alerted before they silently degrade model performance.

#### Acceptance Criteria

1. WHEN the scheduled drift monitoring job runs at 00:00 UTC each day, THE Drift_Monitor SHALL execute Evidently AI's `DataDriftPreset` on every model's feature distribution, comparing the current 7-day rolling window against the training reference distribution.
2. IF the Drift_Monitor detects a PSI value greater than 0.2 and less than or equal to 0.25 for any feature in a model's input space, THEN THE Drift_Monitor SHALL raise a `DRIFT_ALERT` with severity `MEDIUM` and log the feature name, PSI value, reference distribution mean and std, current distribution mean and std, and affected model names.
3. IF the Drift_Monitor detects a PSI value exceeding 0.25 for any feature, THEN THE Drift_Monitor SHALL raise a `DRIFT_ALERT` with severity `HIGH` and, IF online learning is enabled for the affected model, SHALL trigger the model's online learning update procedure.
4. WHEN the scheduled drift monitoring job runs at 00:00 UTC each day, THE Drift_Monitor SHALL use NannyML's `CBPE` (Confidence-Based Performance Estimation) to estimate model performance on unlabeled production data, producing a daily estimated IC and ECE (Expected Calibration Error) for each model.
5. IF NannyML's CBPE estimated IC falls below 0.015 for any model, THEN THE Drift_Monitor SHALL raise a `PERFORMANCE_DEGRADATION_ALERT` and SHALL block that model from contributing to the Meta_Decision_Engine until an operator explicitly clears the block via `POST /monitoring/performance/{model_name}/clear`.
6. THE Drift_Monitor SHALL expose a `GET /monitoring/drift` endpoint returning the latest drift report for all models, and a `GET /monitoring/performance` endpoint returning the latest NannyML performance estimates including the ECE score.
7. THE Drift_Monitor SHALL emit structured log events for every alert in `structlog` format, compatible with the `AlertSystem` and consumable by the monitoring router at `GET /monitoring/alerts`.
8. THE Drift_Monitor SHALL compute PSI using the formula `PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)` across 10 equal-frequency bins derived from the reference distribution.
9. WHEN a drift alert is raised, THE Drift_Monitor SHALL include in the alert payload: the model name, feature name, PSI value, current distribution mean and std, reference distribution mean and std, and a recommended action determined as follows: PSI ≤ 0.25 → `MONITOR`; PSI > 0.25 AND online learning not yet triggered → `RETRAIN`; PSI > 0.25 AND online learning triggered but estimated IC still below 0.015 → `ROLLBACK`.
10. WHEN an operator sends `POST /monitoring/performance/{model_name}/clear`, THE Drift_Monitor SHALL remove the `PERFORMANCE_DEGRADATION_ALERT` block for that model, log the clear event with operator identity and timestamp, and re-enable the model's contribution to the Meta_Decision_Engine.

---

### Requirement 13: Model Lifecycle and Promotion Gates

**User Story:** As an ML operations engineer, I want a structured six-gate promotion process for every model challenger so that no model reaches production without demonstrating sufficient predictive, calibration, execution, risk, and stability evidence.

#### Acceptance Criteria

1. THE Model_Lifecycle SHALL enforce a linear promotion pipeline: `HYPOTHESIS → BACKTEST → CHALLENGER → SHADOW → APPROVED → PRODUCTION`, and SHALL prevent any model from advancing past a stage unless all required gates for the prior stage have passed.
2. THE Promotion_Gate SHALL evaluate six dimensions: DATA, PREDICTIVE, CALIBRATION, EXECUTION, RISK, and STABILITY, each returning one of `PASS`, `FAIL`, or `INSUFFICIENT_EVIDENCE`.
3. IF `final_oos_used_for_selection` is `true`, OR evidence integrity verification fails, OR the OOS observation count (where one observation is one distinct ticker–date pair in the out-of-sample partition) is below 60, OR evidence level is below `LEVEL_B`, THEN THE DATA gate SHALL return `FAIL` and block promotion.
4. IF a champion model exists, THEN THE PREDICTIVE gate SHALL return `PASS` only when the challenger's Rank IC exceeds the champion's Rank IC by at least the configured minimum margin (default: 0.005, valid range: 0.001–0.05); IF no champion model exists, THEN THE PREDICTIVE gate SHALL return `PASS` only when the challenger's Rank IC is greater than 0.
5. IF a champion model exists, THEN THE CALIBRATION gate SHALL return `PASS` only when the challenger's Brier score is no more than 0.01 worse than the champion's Brier score; IF no champion model exists, THEN THE CALIBRATION gate SHALL return `PASS` only when the challenger's Brier score is less than 0.25.
6. IF the challenger's net return over the held-out validation partition (the same date range used for OOS evaluation, excluding any data used in training or hyperparameter selection) is not greater than 0, THEN THE EXECUTION gate SHALL return `FAIL`; IF a champion model exists and the challenger's turnover over that same partition exceeds the champion's turnover by more than 20% in relative terms, THEN THE EXECUTION gate SHALL return `FAIL`.
7. IF a champion model exists, THEN THE RISK gate SHALL return `PASS` only when the challenger's maximum drawdown over the held-out validation partition is no more than 2% worse (in absolute percentage points) than the champion's maximum drawdown; IF no champion model exists, THEN THE RISK gate SHALL return `PASS` only when the challenger's maximum drawdown over the held-out validation partition does not exceed 20%.
8. IF the challenger's IC decay status is `FAILED` or `SIGNIFICANT_DECAY`, OR the challenger's feature drift severity is `HIGH` or `CRITICAL`, THEN THE STABILITY gate SHALL return `FAIL`.
9. WHEN all six gates return `PASS`, THE Promotion_Gate SHALL produce a `PromotionDecision` with outcome `PROMOTE` and approval policy `HUMAN_APPROVAL_REQUIRED`.
10. IF any gate returns `INSUFFICIENT_EVIDENCE`, THEN THE Promotion_Gate SHALL produce a `PromotionDecision` with outcome `BLOCKED` and SHALL record the specific gate(s) returning `INSUFFICIENT_EVIDENCE` in the audit log entry; promotion SHALL NOT proceed until the identified gate(s) return `PASS`.
11. WHEN `HUMAN_APPROVAL_REQUIRED`, THE Model_Lifecycle SHALL require an `approvalToken` that was issued by an identity present in the authorized-reviewer registry and has not expired (expiry defined as more than 24 hours elapsed since issuance) before writing the new champion model to the Model_Registry.
12. IF a model's evidence level is `LEVEL_C` or `LEVEL_D`, THEN THE Model_Lifecycle SHALL return `FAIL` on the DATA gate and block all promotion attempts, regardless of outcomes from the remaining five gates.
13. WHEN the Model_Lifecycle detects that the final OOS partition data has been read during any gate evaluation, hyperparameter selection, or model selection step for the current challenger, THE Model_Lifecycle SHALL permanently mark the challenger as `FINAL_OOS_CONTAMINATED` and block all subsequent promotion attempts for that challenger.
14. THE Model_Lifecycle SHALL record every promotion decision (including rejections and `BLOCKED` outcomes) in the append-only audit log with: timestamp, challenger ID, champion ID, gate results for all six gates, outcome, approval policy, and the reviewer identity when a human approval was provided.

---

### Requirement 14: Online Learning and Adaptive Retraining

**User Story:** As an ML operations engineer, I want models to adapt to regime shifts through online learning without requiring a full retraining cycle so that prediction quality is maintained during rapidly evolving market conditions.

#### Acceptance Criteria

1. THE ML_Service SHALL implement an `OnlineLearner` component that accepts a stream of recent (label, feature) pairs and applies incremental updates to supported models (XGBoost's `model.update()` for regime classifier and risk predictor) without modifying the full model artifact.
2. WHEN the PerformanceMonitor detects IC degradation exceeding 20% from the 90-day trailing baseline over a rolling 30-day window for any model, THE OnlineLearner SHALL initiate an incremental update using the most recent 60 trading days of data.
3. THE OnlineLearner SHALL only apply incremental updates to models with `PredictionProvenance.TRAINED_MODEL`; heuristic fallback models are not eligible for online learning.
4. WHEN an online learning update is applied, THE ML_Service SHALL create a new versioned model artifact (e.g. `v1.0.3-online-2025-07-15`), SHALL NOT overwrite the prior artifact, and SHALL log the update event to the immutable audit log.
5. THE OnlineLearner SHALL validate the updated model on the most recent 10 trading days of held-out data before activating it; WHEN the updated model's IC on the validation window is lower than the prior model's IC on the same window, THE OnlineLearner SHALL discard the update and log a `ONLINE_UPDATE_REJECTED` event.
6. THE OnlineLearner SHALL apply a maximum of 5 consecutive incremental updates before requiring a full retraining cycle with purged K-fold CV.
7. WHEN online learning is triggered, THE ML_Service SHALL continue serving predictions with the current model artifact until the new artifact is validated and registered.

---

### Requirement 15: API Contracts, Authentication, and Versioning

**User Story:** As an API consumer (alpha-forge), I want ml-service2.0 to expose a stable, versioned, authenticated REST API and a WebSocket streaming interface so that client integration is predictable and decoupled from internal implementation changes.

#### Acceptance Criteria

1. THE ML_Service SHALL expose all prediction endpoints under the `/v2/` prefix to distinguish them from the legacy v1 endpoints.
2. THE ML_Service SHALL authenticate all incoming requests using `X-API-KEY` header authentication, with configurable keys stored in environment variables.
3. THE ML_Service SHALL maintain backward-compatible response schemas: WHEN a new field is added to a response, THE ML_Service SHALL add it with a default value that preserves the existing semantics; existing fields SHALL NOT be renamed or removed in a minor version.
4. THE ML_Service SHALL expose the following prediction endpoints: `POST /v2/predict/regime`, `POST /v2/predict/rankings`, `POST /v2/predict/strategy`, `POST /v2/predict/risk`, `POST /v2/predict/portfolio`, `POST /v2/predict/portfolio-v2`, `POST /v2/predict/execution`, `POST /v2/predict/price-regime`, `POST /v2/predict/iv-regime`, and `POST /v2/meta/decide`.
5. THE ML_Service SHALL expose the following analytics endpoints: `POST /v2/analytics/greeks`, `POST /v2/analytics/gex`, `POST /v2/analytics/vpin`, `POST /v2/analytics/vol-surface`.
6. THE ML_Service SHALL expose the following operational endpoints: `GET /health`, `GET /models/status`, `GET /monitoring/drift`, `GET /monitoring/performance`, `GET /monitoring/alerts`, `GET /features/quality`, `POST /meta/fit`, `GET /models/registry`.
7. THE ML_Service SHALL stream real-time signals via WebSocket at `WS /v2/stream/signals`, emitting a `MetaOutput` message for each symbol whenever the Meta_Decision_Engine produces a new decision, and SHALL close the stream gracefully when the client disconnects.
8. WHEN a prediction request is received with an invalid schema (missing required field, out-of-range value, unknown enum member), THE ML_Service SHALL return HTTP 422 with a Pydantic V2 `ValidationError` response body listing all field errors.
9. WHEN a model is unavailable (no artifact loaded, runtime exception during inference), THE ML_Service SHALL return HTTP 503 with a structured error body rather than HTTP 500, enabling callers to implement appropriate retry logic.
10. THE ML_Service SHALL implement CORS for the alpha-forge origin (`http://localhost:3000` in development) and SHALL accept origins from the `ALLOWED_ORIGINS` environment variable in production.
11. THE ML_Service SHALL implement request-level latency tracing using `X-Request-ID` headers, logging the header value alongside all structured log events so that full request traces can be reconstructed.

---

### Requirement 16: Non-Functional Requirements — Performance

**User Story:** As a system architect, I want all prediction endpoints to meet strict latency SLAs so that the ML service does not become the bottleneck in the trading signal pipeline.

#### Acceptance Criteria

1. THE ML_Service SHALL serve all single-symbol prediction endpoints (`/predict/regime`, `/predict/strategy`, `/predict/risk`, `/predict/execution`, `/predict/price-regime`, `/predict/iv-regime`) within 50 milliseconds at p95.
2. THE ML_Service SHALL serve the batch ranking endpoint (`/predict/rankings`) for up to 200 symbols within 200 milliseconds at p95.
3. THE ML_Service SHALL serve the meta-decision endpoint (`/meta/decide`) within 150 milliseconds at p95 including LLM-augmented news reasoning on cached context.
4. THE ML_Service SHALL serve the portfolio optimization endpoint (`/predict/portfolio-v2`) for up to 50 assets within 500 milliseconds at p95.
5. THE ML_Service SHALL serve the analytics endpoints (`/analytics/greeks`, `/analytics/gex`, `/analytics/vpin`, `/analytics/vol-surface`) within 100 milliseconds at p95.
6. THE ML_Service SHALL cache SentinelPulse news context responses in an in-process LRU cache with a TTL of 90 seconds and a maximum of 500 entries per symbol.
7. THE ML_Service SHALL cache feature vectors from Data_Service with a TTL matching the `FEATURE_CACHE_TTL` environment variable (default 60 seconds).
8. THE ML_Service SHALL support at minimum 100 concurrent prediction requests without degradation in p95 latency beyond 2×, using FastAPI's async endpoint handlers and a configurable number of Uvicorn worker processes.

---

### Requirement 17: Non-Functional Requirements — Correctness and Reliability

**User Story:** As a system architect, I want all prediction endpoints to be idempotent, fault-tolerant, and rigorously typed so that the service behaves predictably in production and integrates safely with alpha-forge.

#### Acceptance Criteria

1. THE ML_Service SHALL be implemented in Python 3.11+ with strict mypy typing (`mypy --strict`) and SHALL produce zero type errors in CI.
2. THE ML_Service SHALL use Pydantic V2 for all request and response schema validation with `model_config = ConfigDict(strict=True)` to prevent silent coercion.
3. FOR ALL prediction endpoints, given the same request body and the same loaded model version, the response SHALL be identical (idempotent prediction property).
4. WHEN a model artifact file is corrupted or has an invalid SHA-256 hash at startup, THE ML_Service SHALL refuse to load that artifact, log a `ARTIFACT_INTEGRITY_FAILURE` event, and fall back to the heuristic policy for that model rather than crashing.
5. THE ML_Service SHALL maintain an append-only, immutable training audit log stored in JSON Lines format; WHEN any process attempts to modify or delete a prior log entry, THE ML_Service SHALL raise an `AuditLogViolation` error.
6. THE ML_Service SHALL enforce that historical training data is never modified after ingestion; all training datasets SHALL be stored with content-addressed checksums (SHA-256) and any mismatch between stored and computed checksums SHALL trigger a `DATASET_INTEGRITY_VIOLATION` alert.
7. WHEN all base models are in `HEURISTIC` state and `DeploymentMode == VALIDATED_ML_ONLY`, THE ML_Service SHALL return `NO_TRADE` with `PredictionProvenance.INSUFFICIENT_EVIDENCE` for all signal requests and SHALL log the `HEURISTIC_ONLY_BLOCKED` event.
8. THE ML_Service SHALL implement structured JSON logging using `structlog` with mandatory fields: `timestamp`, `service`, `version`, `request_id`, `model_name`, `latency_ms`, `provenance`, and `deployment_mode`.

---

### Requirement 18: Non-Functional Requirements — Testing and TDD

**User Story:** As a software engineer, I want a comprehensive test suite with at least 90% coverage, property-based tests using Hypothesis, and a TDD mandate so that correctness is verified before code reaches production.

#### Acceptance Criteria

1. THE ML_Service test suite SHALL achieve at minimum 90% line coverage measured by `pytest-cov` with `--fail-under=90`.
2. ALL new modules SHALL have tests written BEFORE the implementation code is written (TDD mandate), verified by requiring the test to fail (red) before the implementation is written (green).
3. THE test file `tests/test_feature_pipeline_pit_correctness.py` SHALL contain property-based tests using `hypothesis` that verify: for all randomly generated feature timestamps T, no feature in the output vector has a source timestamp greater than T (PIT correctness property).
4. THE test file `tests/test_meta_decision_engine.py` SHALL contain property-based tests verifying: `meta_decide(all_unavailable_models) == NO_TRADE` (absorbing identity), `meta_decide(i) == meta_decide(i)` for all inputs (idempotence), and `meta_decide` never produces `BUY` when `agreement_ratio < 0.5` (abstention invariant).
5. THE test file `tests/test_qlib_feature_engineering.py` SHALL contain property-based tests verifying: for any valid OHLCV DataFrame, the Alpha158 feature computation is idempotent (running twice produces the same result), all 158 features are finite (no NaN or Inf in non-missing outputs), and feature values are bounded within their documented ranges.
6. THE test file `tests/test_finrl_execution_agent.py` SHALL contain integration tests verifying: the RL agent's action space is exactly the 7-element discrete set, the agent responds within 50ms p95 on 100 consecutive inference calls, and the fallback to rule-based policy activates correctly when no trained artifact is present.
7. THE test file `tests/test_drift_monitoring.py` SHALL contain property-based tests verifying: `PSI(ref, ref) ≈ 0.0` (identical distributions have zero drift), and `PSI(ref, shifted) > PSI(ref, ref)` for any distribution with non-zero shift (monotonicity property).
8. THE test file `tests/test_signal_generation_latency.py` SHALL contain timing tests verifying that 100 consecutive calls to each single-symbol prediction endpoint complete within the p95 latency SLA, using `time.perf_counter()` and `pytest-benchmark`.
9. THE test file `tests/test_model_training_purged_cv.py` SHALL contain property-based tests verifying: for any time series of length ≥ 100, the purged K-fold splitter produces zero temporal overlap between any training fold and its corresponding validation fold with the configured embargo period.
10. THE test file `tests/test_api_contracts.py` SHALL contain integration tests for all v2 endpoints using `httpx.AsyncClient`, verifying correct HTTP status codes, schema validation, and `PredictionProvenance` field presence on every prediction response.
11. THE test file `tests/test_portfolio_optimization.py` SHALL contain property-based tests verifying: for any valid returns matrix with 2+ assets and 20+ observations, HRP weights sum to 1.0, all weights are ≥ 0, and the portfolio volatility is ≤ the maximum individual asset volatility (HRP diversification invariant).
12. THE test file `tests/test_online_learning.py` SHALL contain property-based tests verifying: each incremental update produces a new, strictly different model version string; after an update, the prior model artifact remains intact on disk; and the online learner never applies more than 5 consecutive updates without triggering a full retrain.

---

### Requirement 19: Parsers, Serializers, and API Contract Correctness

**User Story:** As an API integration engineer, I want all Pydantic schemas to have round-trip serialization tests and all configuration parsers to be independently tested so that no data is silently corrupted at service boundaries.

#### Acceptance Criteria

1. WHEN a valid `RegimePredictionResponse` is serialized to JSON and deserialized back, THE Schema_Parser SHALL produce an object equal to the original (round-trip property).
2. WHEN a valid `MetaOutput` dict is serialized using `.to_dict()` and then parsed back through a `MetaOutput` Pydantic model, THE Schema_Parser SHALL produce an equivalent object (round-trip property).
3. WHEN an invalid enum value is provided in any request schema (e.g. an unknown `MarketRegime` or `TradingStrategy` string), THE ML_Service SHALL return HTTP 422 with a field-level validation error identifying the exact field and invalid value.
4. FOR ALL valid Pydantic request schemas S, parsing then serializing then parsing SHALL produce the same object: `parse(serialize(parse(S))) == parse(S)` (idempotent parse-serialize cycle).
5. THE `PredictionProvenance` enum SHALL be serialized as its string value in all JSON responses, and any consumer receiving an unknown provenance string SHALL be able to treat it as `UNAVAILABLE` without crashing.

---

## Phase 2: TDD Scaffolding Plan

The following test files SHALL be created before any implementation code is written. Each file SHALL start with a failing test (red phase) to confirm TDD compliance.

### Test File Inventory

| File | Primary Focus | Testing Strategy |
|---|---|---|
| `tests/test_feature_pipeline_pit_correctness.py` | PIT correctness invariant on all feature vectors | Property-based (Hypothesis) |
| `tests/test_meta_decision_engine.py` | Absorbing identity, idempotence, abstention invariant | Property-based (Hypothesis) |
| `tests/test_qlib_feature_engineering.py` | Alpha158/360 idempotence, finite outputs, bounded ranges | Property-based (Hypothesis) |
| `tests/test_finrl_execution_agent.py` | Action space, latency SLA, heuristic fallback | Integration + timing |
| `tests/test_drift_monitoring.py` | PSI identity (zero drift on identical dist.), monotonicity | Property-based (Hypothesis) |
| `tests/test_signal_generation_latency.py` | p95 latency SLA for all single-symbol endpoints | Timing (pytest-benchmark) |
| `tests/test_model_training_purged_cv.py` | Zero temporal overlap between folds with embargo | Property-based (Hypothesis) |
| `tests/test_api_contracts.py` | HTTP status, schema validation, provenance field | Integration (httpx.AsyncClient) |
| `tests/test_portfolio_optimization.py` | Weight sum = 1.0, all ≥ 0, HRP diversification invariant | Property-based (Hypothesis) |
| `tests/test_online_learning.py` | Version uniqueness, artifact immutability, 5-update cap | Property-based (Hypothesis) |

### Correctness Properties by Domain

**Feature Pipeline (PIT Correctness)**
- `∀ feature vector v, ∀ feature f ∈ v: source_timestamp(f) < target_timestamp(v)`
- `encode(decode(v)) == v` for all serialized feature vectors (round-trip)
- `pipeline(x) == pipeline(x)` for all inputs (idempotence)

**Meta Decision Engine**
- `meta_decide({all models unavailable}) == NO_TRADE` (absorbing identity)
- `meta_decide(x) == meta_decide(x)` for all x (idempotence)
- `agreement_ratio(x) < 0.5 ⟹ action(meta_decide(x)) ∈ {WAIT, NO_TRADE}` (abstention invariant)
- `confidence(meta_decide(x)) + uncertainty(meta_decide(x)) == 1.0` (probability invariant)

**Drift Monitoring**
- `PSI(D, D) ≈ 0.0` for all distributions D (zero drift identity)
- `PSI(D_ref, D_shifted) > PSI(D_ref, D_ref)` for any non-zero shift (monotonicity)

**Portfolio Optimization**
- `sum(weights(HRP(R))) == 1.0` for all valid return matrices R (weight normalization invariant)
- `all(w ≥ 0 for w in weights(HRP(R)))` for all valid R (non-negativity invariant)
- `portfolio_volatility(HRP(R)) ≤ max(asset_volatilities(R))` (diversification invariant)

**Purged K-Fold CV**
- `∀ fold k, ∀ (t_train, t_val): |t_val - t_train| > embargo_period` (no overlap property)
- `len(purged_train_fold) < len(full_train_fold)` when embargo is non-zero (samples removed property)

**Online Learning**
- `version(model_after_update) ≠ version(model_before_update)` (version uniqueness)
- `artifact_exists(prior_version) == True` after any number of updates (immutability)
- `count_consecutive_updates(model) ≤ 5` (update cap invariant)

**API Contracts (Round-Trip)**
- `parse(serialize(parse(x))) == parse(x)` for all valid request/response schemas (idempotent parse-serialize)
- `decode(encode(MetaOutput)) == MetaOutput` for all valid MetaOutput instances (round-trip)
