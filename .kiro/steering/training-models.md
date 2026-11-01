---
inclusion: fileMatch
fileMatchPattern: "src/{training,models}/**"
---

# Training & Models — Karpathy Guidelines (Service-Specific)

> Applies on top of `karpathy-guidelines.md` when working in `src/training/` or `src/models/`.

## Think Before Coding

- The active model is **v2c LGBMRegressor** (65 features, fs-2.0.0, artifact `v2-20261001184632`). Before touching training code, confirm whether you're working on the live v2c or the in-progress v2d.
- Label construction is a **PIT-critical** operation. State your assumption about the label horizon and cross-sectional rank construction before touching `src/labels/` or the label column in any training script.
- WalkForwardValidator, CPCV, and SelfLearningLoop have specific fold/embargo mechanics. Do not adjust them without understanding the full walk-forward split — a wrong assumption here produces look-ahead bias that passes tests but corrupts live alpha.
- Any hyperparameter change must state the expected IC impact before implementation.

## Simplicity First

- HPO (Optuna) trials are already defined. Do not add a second HPO framework.
- Do not introduce new base estimator classes unless the task explicitly requires one. Prefer wrapping sklearn-compatible estimators.
- Training scripts in `scripts/` are procedural pipelines — keep them that way. Adding classes to a training script for single-use logic is over-engineering.
- `joblib` is already available for parallelism — do not add `multiprocessing` or `concurrent.futures` unless `joblib` demonstrably cannot do the job.

## Surgical Changes

- **Never modify `artifacts/v2_model/`** as a side effect of code changes. Artifact files are production outputs.
- `src/training/train_all.py` and `src/training/data_pipeline.py` are excluded from coverage metrics (they require live data). Do not add mocked tests just to inflate coverage on these files.
- The 65-feature schema (`fs-2.0.0`) is locked by the G_PARITY gate. Any addition or removal of features must go through `src/validation/` leakage checks and update the feature count documentation.
- `src/registry/` handles CHALLENGER→SHADOW→PRODUCTION promotion. Do not manually advance a model's stage in code — that's the registry's job.

## Goal-Driven Execution

- For model training changes, success = IC ≥ current baseline (+0.040 XS-Spearman OOS), not just "the script runs."
- Verify with a backtest run that produces a concrete IC metric before calling a training change done.
- For label or feature changes, run the leakage validator (`src/validation/`) and verify zero-INVALID result.
- Test marker: use `pytest.mark.slow` for backtest/training tests and `pytest.mark.pit` for PIT correctness tests.
