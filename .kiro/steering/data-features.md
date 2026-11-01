---
inclusion: fileMatch
fileMatchPattern: "src/{data,features,labels,validation}/**"
---

# Data, Features & Validation — Karpathy Guidelines (Service-Specific)

> Applies on top of `karpathy-guidelines.md` when working in `src/data/`, `src/features/`, `src/labels/`, or `src/validation/`.

## Think Before Coding

- **PIT (Point-In-Time) correctness is non-negotiable.** Before touching any feature calculation, state explicitly:
  - What timestamp is the "as-of" date?
  - Does the calculation use only data available at that timestamp?
  - Is any look-ahead possible (e.g., using `shift(0)` instead of `shift(1)` on a label)?
- The 65-feature schema (`fs-2.0.0`) is locked by the G_PARITY and G_LEAK gates. Any addition, removal, or renaming of a feature must be treated as a gate-affecting change — flag it before proceeding.
- `SignalDB` writes to PostgreSQL (TimescaleDB) at port 5445. Before adding a new table or column, check whether the schema migration is needed and whether it's backward-compatible.
- The label is a **7-day cross-sectional rank excess return**. Do not change the horizon or rank construction without flagging the G_LABEL gate impact.

## Simplicity First

- Feature families (`src/features/families/`) are organized by type (momentum, volatility, volume, cross_sectional, derivatives, market_structure). Add new features to the appropriate family module — do not create a new top-level feature file for a single feature.
- `ExpandedFeatureFactory` is the single entry point for all feature construction. Do not bypass it by calling family functions directly in training or inference code.
- Label construction in `MultiHorizonLabelFactory` already handles [1, 3, 5, 10, 21]-day horizons. Do not add a new horizon class — add the horizon to the existing factory.
- `FeedbackStore` already provides the outcome feedback loop. Do not add a second outcome-tracking mechanism.

## Surgical Changes

- Feature files listed in `[tool.coverage.run] omit` in `pyproject.toml` are excluded from coverage because they require live data. Do not add mocked tests to inflate their coverage — it creates false confidence in PIT correctness.
- `src/validation/protocol.py` and `freeze.py` are the PIT audit boundary. Do not change their interfaces as a side effect of feature work.
- Do not modify the existing 65 feature names/calculations — they are referenced by name in the serialized model artifact. Renaming a feature silently breaks inference.
- `src/data/dataset_builder.py` (DatasetBuilder) controls the training dataset assembly. Do not add inference-time transformations here.

## Goal-Driven Execution

- For feature additions: success = (a) leakage validator returns zero-INVALID, (b) `|r| < 0.15` vs forward returns, (c) feature appears in ExpandedFeatureFactory output with correct dtype.
- For label changes: success = the new label produces IC > 0 on OOS data AND passes the G_LABEL economic-validity check.
- For SignalDB schema changes: success = migration runs cleanly AND existing query patterns in `scripts/signal_ledger_daily.py` still return correct results.
- Test markers: `pytest.mark.pit` for PIT correctness tests, `pytest.mark.unit` for isolated feature math tests.
