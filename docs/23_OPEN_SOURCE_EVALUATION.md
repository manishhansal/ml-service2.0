# Open Source Evaluation (Summary Reference)
**ml-service2.0**

*Date: 2026-09-24*

See full evaluation in: `docs/OPEN_SOURCE_EVALUATION.md`

---

## Quick Reference: Final Recommendations

| Library | Recommendation | Reason |
|---|---|---|
| scikit-learn, LightGBM, XGBoost, CatBoost | **USE** | Core ML — keep |
| NumPy, pandas, SciPy | **USE** | Essential |
| SHAP | **USE** | Required for explainability |
| PyPortfolioOpt, Riskfolio-Lib | **USE** | Portfolio optimization |
| MLflow, Optuna | **USE** | Experiment tracking, HPO |
| FastAPI, Pydantic V2, Redis, structlog | **USE** | Infrastructure |
| Evidently | **USE** | Drift monitoring |
| transformers/FinBERT | **USE (limited)** | News classification only |
| Microsoft Qlib | **OPTIONAL** | Already have native fallback |
| NannyML | **OPTIONAL** | CBPE is uniquely valuable |
| hmmlearn | **OPTIONAL** | HMM regime detection alternative |
| mlfinlab | **OPTIONAL** | Triple-barrier reference (verify BSD-3) |
| Polars | **ADD** | Historical data ingestion speed |
| prometheus-client | **ADD** | Metrics export |
| langchain/LangGraph | **OPTIONAL (restricted)** | Research agent only, NOT signal path |
| stable-baselines3, gymnasium, finrl | **CONDITIONAL** | Remove if RL ablation negative |
| vectorbt | **REJECT** | AGPL license |
| Backtrader | **REJECT** | GPL license |
| LEAN | **REJECT** | Overkill complexity |
| darts | **OPTIONAL** | Only if TFT ablation positive |

---

*See OPEN_SOURCE_EVALUATION.md for complete analysis*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

No new heavy dependencies were introduced during implementation. The delivered pipeline uses the already-approved stack (numpy, pandas, scikit-learn, xgboost, lightgbm, scipy, pydantic). CatBoost is used only if already installed (optional). No RL/deep-learning dependencies were added to the production path (RL remains research-only pending demonstrated value). Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
