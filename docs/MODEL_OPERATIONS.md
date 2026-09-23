# Model Operations Guide — ml-service2.0

This guide covers the full model lifecycle: from hypothesis through shadow trading to production promotion, including online learning, drift monitoring, and the provenance system that governs which predictions are eligible for live capital deployment.

---

## Table of Contents

1. [Overview: Model Lifecycle Stages](#overview-model-lifecycle-stages)
2. [Submitting a Training Run](#submitting-a-training-run)
3. [Training Pipeline Internals](#training-pipeline-internals)
   - [Purged K-Fold Cross-Validation](#purged-k-fold-cross-validation)
   - [CPCV and Backtest Overfitting Probability](#cpcv-and-backtest-overfitting-probability)
   - [Optuna HPO](#optuna-hpo)
   - [Model Acceptance Gate](#model-acceptance-gate)
4. [Six-Gate Promotion Pipeline](#six-gate-promotion-pipeline)
   - [DATA Gate](#data-gate)
   - [PREDICTIVE Gate](#predictive-gate)
   - [CALIBRATION Gate](#calibration-gate)
   - [EXECUTION Gate](#execution-gate)
   - [RISK Gate](#risk-gate)
   - [STABILITY Gate](#stability-gate)
   - [Promotion Outcomes](#promotion-outcomes)
5. [Human Approval Workflow](#human-approval-workflow)
6. [Shadow Trading Period](#shadow-trading-period)
7. [Online Learning](#online-learning)
   - [Trigger Conditions](#trigger-conditions)
   - [Update Procedure](#update-procedure)
   - [Update Cap and Full Retraining](#update-cap-and-full-retraining)
8. [Drift Monitoring](#drift-monitoring)
   - [Feature Drift Detection (Evidently AI)](#feature-drift-detection-evidently-ai)
   - [Performance Estimation (NannyML)](#performance-estimation-nannyml)
   - [PSI Thresholds and Recommended Actions](#psi-thresholds-and-recommended-actions)
   - [Clearing a Performance Block](#clearing-a-performance-block)
9. [Artifact Storage and Integrity](#artifact-storage-and-integrity)
10. [PredictionProvenance Enum](#predictionprovenance-enum)
11. [DeploymentMode Enum](#deploymentmode-enum)
12. [Audit Log](#audit-log)
13. [MLflow Integration](#mlflow-integration)
14. [Runbooks](#runbooks)
    - [Runbook: Promoting a Challenger to Production](#runbook-promoting-a-challenger-to-production)
    - [Runbook: Rolling Back a Champion](#runbook-rolling-back-a-champion)
    - [Runbook: Investigating a Drift Alert](#runbook-investigating-a-drift-alert)
    - [Runbook: Clearing a Performance Block](#runbook-clearing-a-performance-block)

---

## Overview: Model Lifecycle Stages

Every model in ml-service2.0 follows a mandatory linear lifecycle. No model may skip a stage or advance backward.

```
HYPOTHESIS
    │
    │   Researcher defines the signal hypothesis and feature set.
    │   Recorded in the audit log; no code changes required.
    ▼
BACKTEST
    │
    │   Training pipeline runs purged K-fold CV, CPCV, Optuna HPO.
    │   Model acceptance gate evaluated (IC, Sharpe, PBO).
    │   All six promotion gates evaluated.
    ▼
CHALLENGER
    │
    │   Model has passed the BACKTEST gate and is awaiting shadow evaluation.
    │   Predictions are computed but not acted on.
    ▼
SHADOW
    │
    │   Model runs in parallel with the current champion for a minimum of
    │   SHADOW_TRADING_MIN_DAYS trading days (default: 20).
    │   Shadow performance is tracked in the audit log.
    ▼
APPROVED
    │
    │   All six promotion gates have returned PASS.
    │   Awaiting human approvalToken (required before writing champion).
    ▼
PRODUCTION
    
    Model is the active champion. Predictions are live-eligible.
    (PredictionProvenance.TRAINED_MODEL)
```

Each stage transition is recorded in the append-only audit log with timestamp, model version, gate results, and reviewer identity (for APPROVED → PRODUCTION transitions).

---

## Submitting a Training Run

Training jobs are submitted asynchronously via the REST API.

```bash
curl -X POST http://localhost:8100/training/run \
  -H "X-API-KEY: $ML_SERVICE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name":          "regime_classifier",
    "start_date":          "2022-01-01",
    "end_date":            "2024-12-31",
    "embargo_period_days":  10,
    "n_trials":             50,
    "alpha360_enabled":     false,
    "notes":               "Quarterly refresh — July 2025"
  }'
```

Response:

```json
{
  "run_id":     "run-2025-07-15-abc123",
  "status":     "QUEUED",
  "model_name": "regime_classifier"
}
```

Poll for status:

```bash
curl http://localhost:8100/training/status/run-2025-07-15-abc123 \
  -H "X-API-KEY: $ML_SERVICE_API_KEY"
```

**Supported model names:**

| Model name | Algorithm | Description |
|---|---|---|
| `regime_classifier` | XGBoost | Six-regime market classification |
| `stock_ranker` | LightGBM | F&O universe outperformance ranking |
| `strategy_selector` | CatBoost | Eight-strategy selection |
| `risk_predictor` | XGBoost (ensemble of 3) | Per-trade risk estimation |
| `rl_execution_agent` | FinRL-X (PPO/SAC) | Execution timing RL agent |

---

## Training Pipeline Internals

### Purged K-Fold Cross-Validation

The training pipeline uses a modified Qlib `RollingPurgedKFold` that:

1. Splits the time series into K folds (default K=5).
2. For each fold, **purges** all training samples whose label observation window overlaps the validation fold's feature window.
3. Applies an **embargo period** of `EMBARGO_PERIOD_DAYS` (default 10, minimum 5) trading days around each fold boundary.

The embargo removes the temporal adjacency between training and validation that would allow autocorrelation to leak information. The effect is that `len(purged_train_fold) < len(full_train_fold)` whenever embargo > 0.

**Why this matters:** Without purging and embargo, a model trained on Tuesday data can trivially overfit to Wednesday autocorrelation, producing inflated backtest IC that disappears in live trading.

### CPCV and Backtest Overfitting Probability

Combinatorial Purged Cross-Validation (CPCV) generates `CPCV_N_PATHS` (default 10) overlapping test paths across the full data range. For each path, the training pipeline computes out-of-sample (OOS) performance, then estimates the **Backtest Overfitting Probability (PBO)** — the probability that the selected model was chosen by luck rather than genuine alpha.

**Acceptance threshold:** `PBO ≤ MODEL_ACCEPTANCE_MAX_PBO` (default 0.5). Models with PBO > 0.5 are rejected with reason `HIGH_PBO`.

### Optuna HPO

Hyperparameter optimization runs `OPTUNA_N_TRIALS` (default 50, minimum 50) trials using the Tree-structured Parzen Estimator (TPE) sampler.

**Objective:** Mean Spearman rank correlation (IC) across all held-out validation folds.

All trials are logged to the MLflow experiment for the model family. The study is persisted in `sqlite:///optuna.db` so it survives restarts and can be inspected post-training.

### Model Acceptance Gate

Before a model advances to the CHALLENGER stage, it must pass all three acceptance criteria:

| Criterion | Threshold | Rejection reason |
|---|---|---|
| Mean IC (Spearman) across folds | ≥ `MODEL_ACCEPTANCE_MIN_IC` (0.02) | `IC_BELOW_THRESHOLD` |
| Net Sharpe ratio (after 10bp costs) | ≥ 0.0 | `NEGATIVE_NET_SHARPE` |
| CPCV Backtest Overfitting Probability | ≤ `MODEL_ACCEPTANCE_MAX_PBO` (0.5) | `HIGH_PBO` |

Rejected models are archived with `REJECTED` status. They are not eligible for any further promotion attempts. A new training run must be submitted from scratch.

---

## Six-Gate Promotion Pipeline

When a CHALLENGER model is ready for live deployment, it goes through six structured evaluation gates against the current champion (or against absolute thresholds when no champion exists).

### DATA Gate

**Purpose:** Verify OOS evidence is clean, sufficient, and uncontaminated.

| Check | FAIL condition |
|---|---|
| `final_oos_used_for_selection` | `true` → OOS data was used during model selection |
| Evidence integrity | Failed SHA-256 verification on OOS partition |
| OOS observation count | < 60 distinct ticker–date pairs in the OOS partition |
| Evidence level | `LEVEL_C` or `LEVEL_D` |

If the DATA gate fails, all other gates are irrelevant. The challenger is blocked from promotion regardless of its predictive metrics.

**`FINAL_OOS_CONTAMINATED` flag:** If OOS data is ever read during gate evaluation, hyperparameter selection, or model selection for the current challenger, that challenger is **permanently** marked `FINAL_OOS_CONTAMINATED`. No future promotion attempt is possible for that artifact version.

### PREDICTIVE Gate

**Purpose:** Verify the challenger is meaningfully more predictive than the champion.

| Condition | Threshold |
|---|---|
| Champion exists | Challenger Rank IC > Champion Rank IC + `PREDICTIVE_GATE_MARGIN` (default 0.005) |
| No champion | Challenger Rank IC > 0 |

The `PREDICTIVE_GATE_MARGIN` is configurable via environment variable (range 0.001–0.05). A margin that is too tight promotes noisy improvements; a margin that is too wide blocks genuinely better models.

### CALIBRATION Gate

**Purpose:** Verify that the challenger's probability estimates are no worse than the champion's.

| Condition | Threshold |
|---|---|
| Champion exists | Challenger Brier score ≤ Champion Brier score + 0.01 |
| No champion | Challenger Brier score < 0.25 |

Brier score measures the mean squared error between predicted probabilities and actual outcomes. Lower is better.

### EXECUTION Gate

**Purpose:** Verify the challenger generates positive economic value on the held-out partition.

| Check | FAIL condition |
|---|---|
| Net return | Challenger net return on held-out partition ≤ 0 |
| Turnover (with champion) | Challenger turnover > Champion turnover × 1.20 |

The held-out partition is the **same date range used for OOS evaluation** — it excludes any data used in training or hyperparameter selection.

### RISK Gate

**Purpose:** Verify the challenger's worst-case loss is acceptable.

| Condition | Threshold |
|---|---|
| Champion exists | Challenger max drawdown ≤ Champion max drawdown + 2pp (absolute percentage points) |
| No champion | Challenger max drawdown ≤ 20% |

Maximum drawdown is measured over the held-out validation partition using the same date range as the EXECUTION gate.

### STABILITY Gate

**Purpose:** Verify the challenger's predictive power and feature stability are acceptable.

| Check | FAIL condition |
|---|---|
| IC decay status | `FAILED` or `SIGNIFICANT_DECAY` |
| Feature drift severity | `HIGH` or `CRITICAL` |

IC decay is assessed by comparing the challenger's IC on recent data against its IC during training. Feature drift is assessed by comparing the challenger's training feature distribution against the current production distribution.

### Promotion Outcomes

| Outcome | Meaning |
|---|---|
| `PROMOTE` | All six gates returned `PASS`. Awaiting human approvalToken |
| `REJECTED` | One or more gates returned `FAIL`. Training run archived |
| `BLOCKED` | One or more gates returned `INSUFFICIENT_EVIDENCE`. Promotion paused until gate re-evaluates |

Every promotion decision (including rejections) is written to the audit log with timestamp, challenger ID, champion ID, all six gate results, and the reviewer identity when a human approval was provided.

---

## Human Approval Workflow

When all six gates return `PASS`, the promotion outcome is `PROMOTE` with `approval_policy: HUMAN_APPROVAL_REQUIRED`. The new model does not become champion until an **approvalToken** is provided.

### Issuing an approvalToken

```python
from src.registry.promotion import ApprovalTokenManager

manager = ApprovalTokenManager(
    authorized_reviewers=["alice@firm.com", "bob@firm.com"],
    expiry_hours=24,  # controlled by APPROVAL_TOKEN_EXPIRY_HOURS
)

token = manager.issue_token(
    challenger_id="regime_classifier-v1.2.0",
    reviewer_identity="alice@firm.com",
)
print(token)
# eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Submitting the approvalToken

```bash
curl -X POST http://localhost:8100/training/promote \
  -H "X-API-KEY: $ML_SERVICE_API_KEY" \
  -H "Authorization: Bearer $APPROVAL_TOKEN" \
  -d '{
    "challenger_id": "regime_classifier-v1.2.0"
  }'
```

**Token constraints:**
- Must be issued by an identity in the `authorized_reviewers` registry
- Expires after `APPROVAL_TOKEN_EXPIRY_HOURS` (default 24 hours) from issuance
- Tied to a specific `challenger_id` — cannot be reused for a different artifact
- Token issuance and use are both recorded in the audit log

---

## Shadow Trading Period

After passing the BACKTEST stage, a model must spend at least `SHADOW_TRADING_MIN_DAYS` (default 20) trading days in the **SHADOW** stage before it becomes eligible for live promotion.

During shadow trading:
- The challenger model runs in parallel with the champion
- Its predictions are logged to the audit log under `event_type: shadow_prediction`
- Shadow predictions are **not** surfaced to alpha-forge in production responses
- IC degradation checks from the champion do not trigger online learning on the challenger

Shadow performance metrics accumulate in MLflow under the experiment run. Operators can inspect shadow vs. champion IC and Sharpe in the MLflow UI before approving live promotion.

---

## Online Learning

Online learning provides incremental model updates when IC degrades, without requiring a full retraining cycle with purged K-fold CV.

**Supported models:** `regime_classifier` (XGBoost `model.update()`), `risk_predictor` (XGBoost `model.update()`), `stock_ranker` (LightGBM incremental fit).

**Eligibility:** Only models with `PredictionProvenance.TRAINED_MODEL` are eligible. Heuristic fallback models are never updated via online learning.

### Trigger Conditions

Online learning is triggered automatically when **both** conditions are met:

1. `ONLINE_LEARNING_ENABLED=true`
2. `PerformanceMonitor` detects IC degradation exceeding `ONLINE_LEARNING_IC_DEGRADATION_THRESHOLD` (default 20%) compared to the **90-day trailing baseline**, measured over a **rolling 30-day window**

Example: If the regime classifier's 90-day baseline IC is `0.031` and the rolling 30-day IC drops below `0.031 × (1 − 0.20) = 0.0248`, online learning is triggered.

Online learning can also be triggered manually by the `DriftMonitor` when PSI > 0.25 on a feature used by the affected model (see [Drift Monitoring](#drift-monitoring)).

### Update Procedure

```
1. Guard checks:
   - model provenance must be TRAINED_MODEL
   - consecutive_online_updates < MAX_CONSECUTIVE_ONLINE_UPDATES (default 5)

2. Fetch the most recent 60 trading days of (feature, label) pairs from data-service2.0

3. Apply incremental fit:
   - XGBoost: model.update(X_new, y_new, xgb_model=current_booster)
   - LightGBM: lgb.train(params, train_set, init_model=current_booster, num_boost_round=10)

4. Validate on the most recent 10 trading days (held-out, not used in the update):
   - If new IC < prior IC on validation window → DISCARD update, log ONLINE_UPDATE_REJECTED
   - If new IC ≥ prior IC → proceed

5. Create a new versioned artifact:
   version format: {base_version}-online-{YYYY-MM-DD}
   e.g.: v1.2.0-online-2025-07-15

6. Write new artifact to MODEL_ARTIFACTS_PATH/{model_name}/{new_version}/
   Compute and store SHA-256 checksum

7. Register new artifact in ModelRegistry. Prior artifact is NOT overwritten.

8. Increment consecutive_online_updates counter on the new artifact.

9. Log to audit log: run_id, prior_version, new_version, ic_delta, timestamp.

10. Continue serving predictions from the existing champion until the new artifact
    is registered (zero-downtime update).
```

### Update Cap and Full Retraining

After `MAX_CONSECUTIVE_ONLINE_UPDATES` (default 5) consecutive incremental updates, the model **requires a full retraining cycle** before any further online learning can be applied.

```
v1.2.0
v1.2.0-online-2025-07-01   (update 1)
v1.2.0-online-2025-07-05   (update 2)
v1.2.0-online-2025-07-09   (update 3)
v1.2.0-online-2025-07-13   (update 4)
v1.2.0-online-2025-07-17   (update 5)  ← MAX reached
         ↓
FULL RETRAIN TRIGGERED → v1.3.0 (fresh purged K-fold CV)
```

The `consecutive_online_updates` counter resets to 0 when a full retraining run produces a new base version that is promoted to production.

---

## Drift Monitoring

The `DriftMonitor` runs automatically at 00:00 UTC every day and exposes results via the monitoring API endpoints.

### Feature Drift Detection (Evidently AI)

Each day, `DriftMonitor` compares the **7-day rolling production feature distribution** against the **training reference distribution** (the distribution at the time the model was trained) using Evidently AI's `DataDriftPreset`.

PSI (Population Stability Index) is computed per feature using 10 equal-frequency bins from the reference distribution:

```
PSI = Σ (actual_pct − expected_pct) × ln(actual_pct / expected_pct)
```

A small epsilon (`1e-6`) is added to each bin count before computing the logarithm to prevent log(0). The result is always ≥ 0; PSI is exactly 0 when both distributions are identical.

### Performance Estimation (NannyML)

NannyML's Confidence-Based Performance Estimation (CBPE) estimates model IC and ECE on **unlabeled production data** — no ground-truth realized returns are required. This gives a daily estimated IC for each model without waiting for return data to arrive.

The estimator is calibrated at model training time and uses production confidence scores to estimate what IC the model would achieve on the current data.

### PSI Thresholds and Recommended Actions

| PSI range | Severity | Action | Auto-trigger |
|---|---|---|---|
| 0.0 – 0.20 | None | — | — |
| 0.20 – 0.25 | `MEDIUM` | `MONITOR` | None |
| > 0.25 | `HIGH` | `RETRAIN` | Triggers OnlineLearner (if enabled) |
| > 0.25 + IC < 0.015 + learning triggered | `HIGH` | `ROLLBACK` | PERFORMANCE_DEGRADATION_ALERT |

**MEDIUM alert:** `DRIFT_ALERT` logged with feature name, PSI value, reference mean/std, current mean/std, and recommendation `MONITOR`. No automated action.

**HIGH alert:** `DRIFT_ALERT` logged with recommendation `RETRAIN`. If `ONLINE_LEARNING_ENABLED=true`, the `OnlineLearner` is automatically triggered for the affected model. If `ONLINE_LEARNING_ENABLED=false`, the alert recommendation is `RETRAIN` and operator manual action is required.

**ROLLBACK recommendation:** If PSI > 0.25 AND online learning was triggered AND the estimated IC is still below 0.015 after the update, the recommendation escalates to `ROLLBACK`. The model remains blocked until the operator manually clears it or a full retrain is promoted.

### NannyML Performance Degradation Alert

If the CBPE-estimated IC for any model falls below 0.015:

1. A `PERFORMANCE_DEGRADATION_ALERT` is raised
2. The model is **blocked from contributing to the MetaDecisionEngine** until cleared
3. Blocked models appear in `/monitoring/performance` with `"blocked": true`
4. The MetaDecisionEngine treats a blocked model as if its provenance were `UNAVAILABLE`

### Clearing a Performance Block

```bash
curl -X POST \
  http://localhost:8100/monitoring/performance/stock_ranker/clear \
  -H "X-API-KEY: $ML_SERVICE_API_KEY"
```

Response:

```json
{
  "cleared":    true,
  "model":     "stock_ranker",
  "cleared_by": "ops-user@firm.com",
  "timestamp":  "2025-07-15T11:00:00Z"
}
```

The clear event is written to the audit log with the operator identity and timestamp. The model's contribution to the MetaDecisionEngine is re-enabled immediately.

**When to clear a block:**
- After a successful online learning update has been validated
- After a full retraining run and a new champion has been promoted
- After manual review confirms the estimated IC drop was a false positive (e.g., data pipeline issue)

**Do not clear a block if** the PSI recommendation is still `ROLLBACK` and no corrective action has been taken. Prematurely clearing a block in `VALIDATED_ML_ONLY` mode may allow degraded model predictions to influence live trading.

---

## Artifact Storage and Integrity

Model artifacts are stored on the filesystem under `MODEL_ARTIFACTS_PATH` (default `./artifacts`). Each artifact version has its own directory with an immutable SHA-256 checksum file.

```
artifacts/
├── market_regime/
│   ├── v1.2.0/
│   │   ├── model.json           # XGBoost booster (binary)
│   │   ├── metadata.json        # ModelArtifact fields
│   │   └── model.json.sha256    # SHA-256 checksum
│   └── v1.2.0-online-2025-07-15/
│       ├── model.json
│       ├── metadata.json
│       └── model.json.sha256
├── stock_ranker/
│   └── v1.0.3/
│       ├── model.txt            # LightGBM text booster
│       ├── metadata.json
│       └── model.txt.sha256
├── strategy_selector/
│   └── v1.1.0/
│       ├── model.cbm            # CatBoost model
│       ├── metadata.json
│       └── model.cbm.sha256
├── risk_predictor/
│   └── v1.0.0/
│       ├── stop_model.json      # XGBoost — P(stop hit)
│       ├── target_model.json    # XGBoost — P(target hit)
│       ├── drawdown_model.json  # XGBoost — expected drawdown
│       ├── metadata.json
│       └── *.sha256
└── rl_execution_agent/
    └── v1.0.0/
        ├── policy.zip           # Stable-Baselines3 checkpoint
        ├── metadata.json
        └── policy.zip.sha256
```

**At startup**, the service computes the SHA-256 hash of every registered artifact file and compares it against the stored `.sha256` file. If they do not match:

1. `ARTIFACT_INTEGRITY_FAILURE` is logged with the model name, version, expected hash, and computed hash
2. The service **does not load** that artifact
3. The affected model falls back to the heuristic policy (`PredictionProvenance.HEURISTIC`)
4. The service continues to start — a single corrupt artifact does not crash the service

**Immutability:** Once written, artifact files are never overwritten or deleted by the service. Online learning updates create new directories (`v1.2.0-online-2025-07-15/`) alongside the prior version. All prior versions remain on disk.

**Backup recommendation:** Back up `MODEL_ARTIFACTS_PATH` and `AUDIT_LOG_PATH` to object storage (e.g. S3, GCS) on a daily schedule. The audit log should be backed up continuously if the underlying filesystem does not provide crash-consistent snapshots.

---

## PredictionProvenance Enum

The `provenance` field on every prediction response indicates the evidence quality and live-eligibility of the prediction.

| Value | Live-eligible | When it appears |
|---|---|---|
| `trained_model` | **Yes** | A validated ML artifact was used for inference |
| `heuristic` | No | No trained artifact available; rule-based fallback used |
| `insufficient_evidence` | No | `DataConfidenceScore` below threshold, or model confidence < 0.35 |
| `unavailable` | No | data-service2.0 unreachable, or `signalEngineAllowed=false` |

### Provenance transitions

```
TRAINED_MODEL
    │  artifact corrupted at startup
    │  or IC drops below threshold
    ▼
HEURISTIC
    │  inference exception
    │  or all features NaN
    ▼
UNAVAILABLE
```

**In `DEPLOYMENT_MODE=validated_ml_only`:**
- `HEURISTIC` predictions are converted to `INSUFFICIENT_EVIDENCE` before reaching the MetaDecisionEngine
- All endpoints return `NO_TRADE` for any symbol where the best available provenance is `HEURISTIC`
- `HEURISTIC_ONLY_BLOCKED` is appended to `reason_codes` and logged as a structured event

### Checking provenance in alpha-forge

```python
from ml_client import MLServiceClient

client = MLServiceClient(base_url="http://localhost:8100", api_key="...")
result = client.predict_regime(request)

if result.provenance != "trained_model":
    logger.warning("non_ml_prediction", provenance=result.provenance)
    # Do not size positions based on HEURISTIC / INSUFFICIENT_EVIDENCE
```

---

## DeploymentMode Enum

`DEPLOYMENT_MODE` controls how the service handles non-trained-model predictions across the entire request pipeline.

| Mode | Description | Typical use |
|---|---|---|
| `research` | All provenances allowed; heuristics surfaced as normal predictions | Development, backtesting |
| `paper` | All provenances allowed; signals paper-traded (no live capital) | Paper trading evaluation |
| `shadow` | All provenances allowed; signals logged for shadow evaluation | Pre-production shadow run |
| `validated_ml_only` | Only `trained_model` signals are actionable; all others → `NO_TRADE` | Live production |

In `validated_ml_only` mode, the MetaDecisionEngine enforces two additional checks:

1. Base model outputs with `PredictionProvenance.HEURISTIC` have their direction set to `0` before ensemble weighting (they do not influence the final action)
2. If **all** base models are `HEURISTIC`, the response is always `action: NO_TRADE` with `provenance: insufficient_evidence` and reason code `HEURISTIC_ONLY_BLOCKED`

---

## Audit Log

The audit log at `AUDIT_LOG_PATH` (`./audit.jsonl` by default) is an **append-only** JSON Lines file. Every significant event is written synchronously with `fsync` before the operation proceeds.

### Events recorded

| Event type | Trigger |
|---|---|
| `training_run_started` | Training job begins |
| `training_run_completed` | Training job finishes (pass or fail) |
| `model_acceptance_gate_result` | IC/Sharpe/PBO gate result for each trained model |
| `promotion_decision` | All six gate results + outcome |
| `approval_token_issued` | Human reviewer issues an approvalToken |
| `approval_token_used` | approvalToken submitted for promotion |
| `champion_registered` | New champion written to ModelRegistry |
| `shadow_prediction` | Challenger prediction logged during shadow phase |
| `online_update_started` | OnlineLearner begins incremental update |
| `online_update_validated` | Incremental update passes IC validation |
| `online_update_rejected` | Incremental update discarded (new IC < prior IC) |
| `online_update_cap_exceeded` | Full retrain triggered after 5 consecutive updates |
| `artifact_integrity_failure` | SHA-256 mismatch detected at startup |
| `performance_block_cleared` | Operator clears a PERFORMANCE_DEGRADATION_ALERT block |

### Sample entry

```json
{
  "event_type":     "promotion_decision",
  "run_id":         "run-2025-07-15-abc123",
  "timestamp":      "2025-07-15T18:30:00Z",
  "challenger_id":  "regime_classifier-v1.2.0",
  "champion_id":    "regime_classifier-v1.1.0",
  "gate_results": {
    "DATA":         "pass",
    "PREDICTIVE":   "pass",
    "CALIBRATION":  "pass",
    "EXECUTION":    "pass",
    "RISK":         "pass",
    "STABILITY":    "pass"
  },
  "outcome":         "promote",
  "approval_policy": "HUMAN_APPROVAL_REQUIRED",
  "reviewer_identity": null,
  "dataset_hash":   "a1b2c3d4..."
}
```

**Tamper protection:** The service raises `AuditLogViolation` if any process attempts to modify or delete a prior log entry. In production, the audit log directory should be mounted read-only for all processes except ml-service2.0, and write access should be revoked after each rotation.

---

## MLflow Integration

Each training run creates an MLflow run under the model family's experiment:

| Field logged | Description |
|---|---|
| `model_name` | Model family name |
| `version` | Version string |
| `training_start_date` | ISO date |
| `training_end_date` | ISO date |
| `validation_start_date` | ISO date |
| `validation_end_date` | ISO date |
| `ic_fold_{k}` | Spearman IC on each held-out fold |
| `ic_mean` | Mean IC across all folds |
| `net_sharpe` | Net Sharpe after 10bp transaction costs |
| `max_drawdown` | Maximum drawdown on validation partition |
| `pbo` | CPCV backtest overfitting probability |
| `brier_score` | Brier score on calibration partition |
| `training_dataset_hash` | SHA-256 of training dataset |
| All Optuna hyperparameters | Logged as MLflow params |

Access the MLflow UI at `http://localhost:5000` (default) to compare runs, filter by IC, and inspect hyperparameter importance.

---

## Runbooks

### Runbook: Promoting a Challenger to Production

**Precondition:** Training run has completed and all six gates returned `PASS`.

```bash
# 1. Verify gate results
curl http://localhost:8100/training/status/run-2025-07-15-abc123 \
  -H "X-API-KEY: $ML_SERVICE_API_KEY"

# 2. Issue an approvalToken (authorized reviewer only)
python -c "
from src.registry.promotion import ApprovalTokenManager
mgr = ApprovalTokenManager(['alice@firm.com'], expiry_hours=24)
print(mgr.issue_token('regime_classifier-v1.2.0', 'alice@firm.com'))
"

# 3. Submit the promotion with the token
curl -X POST http://localhost:8100/training/promote \
  -H "X-API-KEY: $ML_SERVICE_API_KEY" \
  -H "Authorization: Bearer $APPROVAL_TOKEN" \
  -d '{"challenger_id": "regime_classifier-v1.2.0"}'

# 4. Verify the new champion
curl http://localhost:8100/v2/models/status \
  -H "X-API-KEY: $ML_SERVICE_API_KEY"
```

### Runbook: Rolling Back a Champion

If a recently promoted champion is producing degraded predictions, roll back to the prior champion:

```bash
# 1. Find the prior champion version in the audit log
grep '"champion_registered"' audit.jsonl | tail -2

# 2. Re-register the prior version as champion
python -c "
from src.registry.registry import ModelRegistry
from src.config import settings
registry = ModelRegistry(settings)
registry.set_champion('regime_classifier', version='v1.1.0')
"

# 3. Restart the service to reload the artifact
# (or call the internal reload endpoint if available)

# 4. Verify
curl http://localhost:8100/v2/models/status \
  -H "X-API-KEY: $ML_SERVICE_API_KEY"
```

Rolling back does not delete the v1.2.0 artifact. The version remains on disk and in MLflow. It can be re-promoted later after investigation.

### Runbook: Investigating a Drift Alert

```bash
# 1. Check current alerts
curl http://localhost:8100/monitoring/alerts \
  -H "X-API-KEY: $ML_SERVICE_API_KEY"

# 2. Get the full drift report
curl http://localhost:8100/monitoring/drift \
  -H "X-API-KEY: $ML_SERVICE_API_KEY"

# 3. Check which model is affected and its estimated IC
curl http://localhost:8100/monitoring/performance \
  -H "X-API-KEY: $ML_SERVICE_API_KEY"
```

**If PSI is MEDIUM (0.20–0.25):**
- Check whether the drifted feature is meaningful to the model (inspect SHAP top-10)
- Monitor daily until drift resolves or escalates to HIGH
- No immediate action required

**If PSI is HIGH (> 0.25):**
- Check `ONLINE_LEARNING_ENABLED`. If `true`, an online learning update should already be in progress
- If online learning is disabled, submit a fresh training run
- Monitor estimated IC via `/monitoring/performance` daily

**If recommendation is ROLLBACK:**
- The online learning update did not recover IC — a full retrain is needed
- Submit a full training run: `POST /training/run`
- Do not promote until the new model passes all six gates

### Runbook: Clearing a Performance Block

```bash
# 1. Confirm the model is blocked and the reason
curl http://localhost:8100/monitoring/performance \
  -H "X-API-KEY: $ML_SERVICE_API_KEY"

# 2. Investigate the cause (check drift report, audit log entries)
grep '"performance_degradation"' audit.jsonl | tail -5

# 3. Once the cause is resolved (new champion promoted, drift resolved):
curl -X POST \
  http://localhost:8100/monitoring/performance/stock_ranker/clear \
  -H "X-API-KEY: $ML_SERVICE_API_KEY"

# 4. Confirm the block is cleared
curl http://localhost:8100/monitoring/performance \
  -H "X-API-KEY: $ML_SERVICE_API_KEY"
```

Do not clear a performance block in `DEPLOYMENT_MODE=validated_ml_only` unless you have verified either:
- A new champion has been promoted that resolves the degradation, **or**
- The estimated IC drop was a data pipeline artifact (e.g., a bad batch from data-service2.0) that has since been corrected
