# Open Source Library Evaluation
**AlphaForge ml-service2.0 — Dependency Assessment and Recommendations**

*Evaluation Date: 2026-09-24*
*Context: Indian equity/derivatives market (NSE F&O), intraday to daily horizons*

---

## Evaluation Criteria

Each library is assessed on:
- **Purpose:** What problem it solves
- **License:** Commercial viability
- **Maturity:** Production readiness (1=experimental, 5=battle-tested)
- **Performance:** Computational efficiency for our use case
- **Financial relevance:** How directly applicable to NSE F&O
- **Integration difficulty:** Cost to integrate/maintain (1=trivial, 5=very high)
- **Current overlap:** Whether functionality already exists in the codebase
- **Recommendation:** Use / Optional / Reject

---

## 1. Core ML and Quantitative Libraries

| Library | Version in use | Purpose | License | Maturity | Performance | Financial Relevance | Integration Difficulty | Current Overlap | Recommendation | Reason |
|---|---|---|---|---|---|---|---|---|---|---|
| **scikit-learn** | 1.5.2 | Baselines, calibration, preprocessing | BSD-3 | 5 | High | High | 1 | Present | **USE** | Essential; Platt scaling, isotonic regression, logistic regression baselines, feature selection |
| **LightGBM** | 4.5.0 | Gradient boosting ranking/classification | MIT | 5 | Very High | Very High | 1 | Present | **USE** | Best choice for StockRanker (LTR); fast on tabular data; handles categorical natively |
| **XGBoost** | 2.1.3 | Gradient boosting classification | Apache-2.0 | 5 | Very High | Very High | 1 | Present | **USE** | Best for RegimeClassifier; GPU support; strong regularisation |
| **CatBoost** | 1.2.7 | Gradient boosting with categorical | Apache-2.0 | 5 | High | High | 2 | Present | **USE** | Best for StrategySelector; good OOB handling; categorical without encoding |
| **NumPy** | 1.26.4 | Numerical arrays | BSD | 5 | Very High | Essential | 1 | Present | **USE** | Non-negotiable |
| **pandas** | 2.2.3 | Tabular data | BSD | 5 | Medium | High | 1 | Present | **USE** | Training pipelines; keep Polars for DataService ingestion layer |
| **SciPy** | 1.14.1 | Statistical functions, optimization | BSD | 5 | High | High | 1 | Present | **USE** | Spearman IC, brentq IV solver, statistical tests |
| **SHAP** | 0.46.0 | Explainability (feature attribution) | MIT | 4 | Medium | High | 2 | Present | **USE** | TreeExplainer for all tree models; required for production explainability |

---

## 2. Quantitative / Financial Research

| Library | Version | Purpose | License | Maturity | Performance | Financial Relevance | Integration Difficulty | Current Overlap | Recommendation | Reason |
|---|---|---|---|---|---|---|---|---|---|---|
| **Microsoft Qlib** | Not installed (optional) | Alpha factor research, ML pipeline for quant | MIT | 4 | High | Very High | 4 | Partial (native fallback exists) | **OPTIONAL** | QlibFeatureEngine already implements Alpha158 natively; Qlib adds alpha research framework, backtesting, online learning; high integration cost justifies only if full research platform is needed |
| **mlfinlab** | Not present | ML for finance (triple-barrier, fractional differentiation, CPCV) | BSD-3-Clause | 3 | Medium | Very High | 3 | ✗ None | **OPTIONAL** | Triple-barrier labeling reference implementation; fractional differentiation; CPCV implementation exists here. Evaluate before writing from scratch. License: BSD allows commercial use. |
| **vectorbt** | Not present | Vectorized backtesting | AGPL-3.0 | 4 | Very High | High | 3 | ✗ None | **REJECT** | AGPL license incompatible with proprietary software. Performance is excellent but license risk is unacceptable. Alternative: write own vectorized backtest or use Backtrader. |
| **Backtrader** | Not present | Event-driven backtesting | GPL-3.0 | 4 | Medium | High | 3 | ✗ None | **REJECT** | GPL license creates risk for proprietary system. Consider porting critical features to own event-driven backtest engine. |
| **Zipline Reloaded** | Not present | Event-driven backtesting | Apache-2.0 | 3 | Medium | High | 4 | ✗ None | **OPTIONAL** | Apache-2.0 is clean. Indian market calendars require customization. High integration cost. Only use if existing backtest is insufficient. |
| **LEAN (QuantConnect)** | Not present | Full algorithmic trading platform | Apache-2.0 | 5 | High | Medium | 5 | ✗ None | **REJECT** | Overkill for this use case. Designed for strategy development, not an ML service component. Integration cost prohibitive. |
| **PyPortfolioOpt** | 1.5.5 | Portfolio optimization (MVO, CVaR, etc.) | MIT | 4 | Medium | High | 2 | Present | **USE** | Clean implementation; good for mean-variance and risk-parity optimization; keep alongside Riskfolio-Lib |
| **Riskfolio-Lib** | 6.3.0 | Advanced portfolio optimization | BSD-3 | 4 | Medium | High | 2 | Present | **USE** | More comprehensive than PyPortfolioOpt; supports risk parity, CVaR optimization, hierarchical risk parity |
| **statsmodels** | Not present | Statistical models (ARIMA, VAR, regime) | BSD | 5 | Medium | High | 2 | ✗ None | **OPTIONAL** | Kalman filter state-space models; HMM support via `hmmlearn`; ARIMA baselines; useful for regime detection |

---

## 3. Time Series Specific

| Library | Version | Purpose | License | Maturity | Performance | Financial Relevance | Integration Difficulty | Current Overlap | Recommendation | Reason |
|---|---|---|---|---|---|---|---|---|---|---|
| **sktime** | Not present | Unified time series API | BSD-3 | 4 | Medium | Medium | 3 | ✗ None | **OPTIONAL** | Useful for time series classification and forecasting experiments. Not needed for tree models. Consider if deep TS models are prioritized. |
| **PyTorch Forecasting** | Not present | TFT, NBeats, DeepAR | MIT | 4 | High | High | 4 | ✗ None | **OPTIONAL** | Best available TFT implementation; needed if PriceForecaster deep learning path is chosen. High integration cost and GPU requirement. Only recommend after ablation shows value. |
| **darts** | Not present (placeholder in PriceForecaster) | Time series forecasting library | Apache-2.0 | 4 | Medium | Medium | 3 | Placeholder | **OPTIONAL** | PriceForecaster has placeholder for darts TFT. Evaluate before committing: run ablation comparing TFT to LightGBM on same labels. TFT rarely dominates tree models on tabular financial data. |
| **hmmlearn** | Not present | Hidden Markov Models | BSD-3 | 4 | Medium | High | 2 | ✗ None | **OPTIONAL** | Gaussian HMM for regime detection; strong theoretical justification; lightweight; consider for RegimeClassifier if supervised approach shows instability |

---

## 4. Monitoring and Observability

| Library | Version | Purpose | License | Maturity | Performance | Financial Relevance | Integration Difficulty | Current Overlap | Recommendation | Reason |
|---|---|---|---|---|---|---|---|---|---|---|
| **Evidently** | 0.5.0 | Data/model drift, data quality reports | Apache-2.0 | 4 | Medium | Medium | 2 | Present (wrapper) | **USE** | DataDriftPreset is production-ready; integrates cleanly; keep as optional enhancement to PSI-based DriftMonitor |
| **NannyML** | 0.13.0 | Performance estimation without labels (CBPE) | Apache-2.0 | 3 | Medium | High | 3 | Present (wrapper) | **OPTIONAL** | CBPE is uniquely valuable for financial ML — estimates model performance without realized labels. Useful for monitoring between batch evaluations. Integration complexity is medium. |
| **MLflow** | 2.19.0 | Experiment tracking, model registry | Apache-2.0 | 5 | Medium | Medium | 2 | Present | **USE** | Non-negotiable for experiment reproducibility. Currently unconfigured. Priority: configure `MLFLOW_TRACKING_URI` and validate. |
| **Optuna** | 4.1.0 | Hyperparameter optimization | MIT | 5 | High | High | 1 | Present | **USE** | Best HPO library available. TPE sampler + MedianPruner is the right default choice. |
| **prometheus-client** | Not present in ML service | Metrics export | Apache-2.0 | 5 | Very High | Medium | 2 | ✗ None | **USE** | Add to ml-service2.0; export inference latency, drift alerts, model prediction distribution, abstention rate |

---

## 5. Deep Learning and RL

| Library | Version | Purpose | License | Maturity | Performance | Financial Relevance | Integration Difficulty | Current Overlap | Recommendation | Reason |
|---|---|---|---|---|---|---|---|---|---|---|
| **PyTorch** | Not present (transitive via SB3/transformers) | Deep learning | BSD-3 | 5 | Very High | High | 3 | Indirect | **OPTIONAL** | Required only if deep learning path is chosen. Do NOT add deep models before ablation confirms they beat tree models. |
| **transformers (HuggingFace)** | 4.47.1 | NLP/FinBERT | Apache-2.0 | 5 | Medium | High | 2 | Present | **USE (limited scope)** | Used only for LLMNewsReasoner (FinBERT text classification). Keep but isolate: news classification only, never in the numerical prediction path. |
| **langchain** | 0.3.14 | LLM orchestration | MIT | 3 | Low | Low | 3 | Present | **OPTIONAL (restricted)** | Must NOT be in live signal path. Use only for research agent, diagnostic workflows, experiment planning. Current version 0.3.14 is stable. |
| **langchain-community** | 0.3.14 | LangChain community integrations | MIT | 3 | Low | Low | 3 | Present | **OPTIONAL (restricted)** | Same policy as langchain |
| **stable-baselines3** | 2.4.0 | RL algorithms (PPO, SAC, A2C) | MIT | 4 | High | Low | 3 | Present | **CONDITIONAL** | Keep only if RLExecutionAgent is actually trained and ablation shows improvement. Currently provides zero value — all inference is heuristic. If ablation is negative, remove to reduce image size (~500MB). |
| **gymnasium** | 1.0.0 | RL environment API | MIT | 4 | High | Low | 2 | Present | **CONDITIONAL** | Same as SB3 — keep only if RL is validated. |
| **finrl** | 0.3.7 | RL for finance | MIT | 2 | Medium | Medium | 4 | Present | **CONDITIONAL** | Maturity score 2 — experimental. FinRL has known issues with data leakage in some environments. If RL path is chosen, carefully validate environments for PIT correctness. Consider writing custom Gymnasium env instead. |

---

## 6. Feature Engineering and Data

| Library | Version | Purpose | License | Maturity | Performance | Financial Relevance | Integration Difficulty | Current Overlap | Recommendation | Reason |
|---|---|---|---|---|---|---|---|---|---|---|
| **Polars** | Not in ML service (in DataService) | Fast DataFrame operations | MIT | 4 | Very High | High | 2 | DataService only | **USE (add to ML service)** | Add to ml-service2.0 for historical data ingestion and feature computation. Significantly faster than pandas for large OHLCV datasets. |
| **PyArrow** | Not in ML service | Columnar data | Apache-2.0 | 5 | Very High | Medium | 1 | DataService only | **OPTIONAL** | Add if using Parquet for dataset storage. Pairs with Polars. |
| **DuckDB** | Not present | In-process SQL analytics | MIT | 4 | Very High | High | 2 | ✗ None | **OPTIONAL** | Consider for historical feature dataset storage and querying. Avoids Pandas for large datasets. Supports Parquet natively. |
| **ta (technical analysis)** | 0.11.0 | 60+ technical indicators | MIT | 3 | Medium | High | 1 | Present | **USE** | Convenient; not critical path (can be replaced). Good for rapid feature development. |
| **joblib** | 1.4.2 | Parallelism, caching, serialization | BSD | 5 | High | Medium | 1 | Present | **USE** | Essential for model serialization alongside pickle |

---

## 7. Infrastructure

| Library | Version | Purpose | License | Maturity | Performance | Financial Relevance | Integration Difficulty | Current Overlap | Recommendation | Reason |
|---|---|---|---|---|---|---|---|---|---|---|
| **FastAPI** | 0.115.6 | Async HTTP framework | MIT | 5 | Very High | Medium | 1 | Present | **USE** | Correct choice; Pydantic V2 native; async; keep |
| **Pydantic** | 2.10.3 | Schema validation | MIT | 5 | Very High | High | 1 | Present | **USE** | V2 strict mode is correctly used; keep |
| **Redis** | 5.2.1 | Caching, pub/sub | MIT | 5 | Very High | Medium | 1 | Present | **USE** | LRU cache fallback; WebSocket pub/sub; keep |
| **httpx** | 0.28.1 | Async HTTP client | BSD | 5 | High | Medium | 1 | Present | **USE** | Correct choice for DataService/SentinelPulse calls; keep |
| **tenacity** | 9.0.0 | Retry logic | Apache-2.0 | 5 | High | Medium | 1 | Present | **USE** | Currently underused — manually implementing retry in clients; use tenacity decorators instead |
| **structlog** | 24.4.0 | Structured logging | MIT/Apache | 5 | High | Medium | 1 | Present | **USE** | Correct choice; keep |
| **grpcio** | 1.69.0 | gRPC | Apache-2.0 | 5 | Very High | Medium | 3 | Present (unused) | **CONDITIONAL** | Keep only if gRPC streaming client is actually wired. Currently dead code. |

---

## 8. Testing

| Library | Version | Purpose | License | Maturity | Recommendation | Reason |
|---|---|---|---|---|---|---|
| **pytest** | 8.3.4 | Test runner | MIT | 5 | **USE** | Keep |
| **Hypothesis** | 6.123.4 | Property-based testing | MPL-2.0 | 5 | **USE** | Critical for financial invariant testing; underused currently |
| **pytest-asyncio** | 0.24.0 | Async test support | Apache-2.0 | 4 | **USE** | Keep |
| **pytest-benchmark** | 5.1.0 | Performance benchmarks | BSD | 4 | **USE** | Use for p95 inference latency targets |
| **pytest-timeout** | 2.3.1 | Test timeouts | MIT | 4 | **USE** | Keep |

---

## 9. LangChain / LangGraph Policy

Per the master spec, LangChain/LangGraph must NOT be in the numerical prediction path.

**Permitted uses:**
- Research agent for experiment hypothesis generation
- Diagnostic workflows for model failure investigation
- Post-trade analysis report generation
- Automated research reports comparing challengers

**Forbidden uses:**
- Generating price predictions or probabilities
- Determining position sizes, stop losses, or targets
- Overriding model ensemble outputs
- Any component in the live signal path

**Implementation boundary:** LangChain components must sit entirely outside `src/api/predict.py`, `src/api/meta.py`, and `src/meta/engine.py`. They belong in a separate `src/research/` module that has no runtime dependency from the inference path.

---

## 10. Recommended Additions (Not Currently in pyproject.toml)

| Library | Rationale | Priority |
|---|---|---|
| `polars>=1.0.0` | Fast historical data ingestion for training | P1 |
| `prometheus-client>=0.20.0` | Inference latency, drift metric export | P1 |
| `hmmlearn>=0.3.0` | HMM regime detection alternative | P2 |
| `pyarrow>=15.0.0` | Parquet dataset storage | P2 |
| `mlfinlab` (if BSD-3 license confirmed) | Triple-barrier labels, CPCV reference | P2 |
| `duckdb>=0.10.0` | In-process SQL for feature dataset queries | P3 |

## 11. Recommended Removals (If Not Validated by Ablation)

| Library | Condition for removal | Current weight |
|---|---|---|
| `finrl==0.3.7` | If RL ablation shows no improvement | ~500MB+ Docker layer |
| `stable-baselines3==2.4.0` | Same as finrl | ~200MB |
| `gymnasium==1.0.0` | Same as finrl | ~50MB |
| `darts` (if added) | If TFT ablation shows no improvement over LightGBM | ~300MB |

---

*End of Open Source Evaluation*
