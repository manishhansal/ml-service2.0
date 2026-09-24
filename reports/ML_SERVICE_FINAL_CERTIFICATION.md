# ML Service Final Certification Report
**AlphaForge ml-service2.0 — Post-Implementation Certification**

*Report Date: 2026-09-24 (updated after the F&O-universe-restoration & broad cross-sectional research phase)*
*Certification Status: **RESEARCH_READY** — infrastructure + evidence-chain integrity in place; NO cost-surviving edge verified*
*git SHA: 296c0033957f645f837aedea062f09ba437a2f70 (HEAD; this phase's changes are uncommitted working-tree changes)*
*Final decision level: **A — NO VERIFIED EDGE***

---

## F&O Universe Restoration & Broad Cross-Sectional Research Phase (2026-09-24)

This phase closed the biggest research blocker (the `fno-universe` 503) and used
the restored broad universe to test the central unanswered question (§108):
**does a broad, liquid, PIT-safe F&O equity universe reveal cross-sectional alpha
that survives turnover, realistic costs, and OOS validation?**

**1. `fno-universe` 503 — root-caused and FIXED (in data-service2.0).** The
endpoint was not a data problem: a valid ACTIVE snapshot (239 constituents, 220
real F&O equities) already existed in the database. The route read **only** a
process-local in-memory cache with **no DB fallback**, and that cache was only
warmed by the 08:45 IST scheduler job — so after any API restart the endpoint
returned 503 for up to a day despite valid persisted data. Fix (non-faking): add
a DB fallback to the route, warm the cache from the DB on startup, and add
explicit failure semantics (`DATABASE_UNAVAILABLE` / `NO_UNIVERSE` /
`PARTIAL_UNIVERSE` / `VALID_UNIVERSE`). Endpoint now returns **HTTP 200**. Verified
live and with 89 passing data-service unit tests.

**2. Broad universe confirmed real.** 220 symbols reported; **217 have VALID daily
OHLCV** (median ~2,481 bars ≈ 10 years). Research gate:
`CROSS_SECTIONAL_READY_PREFERRED`. Survivorship is `CURRENT_UNIVERSE_ONLY`
(single open `effective_from`), so all results are **SURVIVORSHIP_LIMITED** and
are **not** presented as unbiased historical evidence (§11, §54).

**3. Cross-sectional research (daily, 209 symbols, 439k rows, 32 experiments).**
Labels {raw, excess, residual, rank} × feature sets {BASE, BASE+XS} × models
{logistic, ridge, lightgbm, xgboost}. The **primary metric is per-timestamp
cross-sectional Spearman RANK IC** (Pearson IC is outlier-sensitive on fat-tailed
daily returns and is used only as a diagnostic — a methodology correction applied
uniformly, §23/§40/§81). Best close-to-close **rank IC ≈ 0.19**, leakage-clean
(null test collapses it to ≈ 0), and **independently reproduced from 365,153
persisted OOS predictions (abs_diff 0.0)**.

**4. The decisive economic test — and why it is NO EDGE.** The signal is a
1-day cross-sectional **reversal** (`corr(score, ret_1) ≈ -0.5`). A close-to-close
spread is **not** valid economic evidence (§32/§33). Under a **real-OHLCV
next-open backtest** (signal at close[T] → enter at open[T+1] → exit at
open[T+1+h]), the rank IC **decays to ≈ 0.02** and every configuration
(long-short / long-only, holding 1–5, cost 5–30 bps) has a **negative gross and
net Sharpe (-11 to -14)**. The apparent edge was a same-bar close-timing /
bid-ask-bounce artifact. **State: ECONOMICALLY_UNVIABLE.** 15m cross-sectional
corroborates (rank IC 0.02–0.035, also economically unviable).

This is the exact failure the mandate warns about: **statistical significance did
not convert to economic significance.** It extends the prior 5m
"statistically-interesting-but-economically-unviable" finding to a broad universe.

**5. Infrastructure hardening (built regardless of edge).**

- **Stale-data blocking (§62) — closed** (was PARTIAL). New
  `src/features/stale_guard.py` with timeframe-specific thresholds (5m→12 min,
  15m→40 min, 1h→150 min, 1d→session-aware), wired into the live inference path;
  raises `StaleDataError` (NO_TRADE). 8 tests.
- **LookAheadGuard (§63)** confirmed wired into live inference with
  `enforce_pit=True` blocking.
- **Forward-paper runner (§55–§61)** — `src/analytics/forward_paper.py`:
  immutable append-only signal store; `resolve_due()` gates on wall-clock elapsed
  time so historical replay cannot masquerade as forward-paper; full versioning;
  no auto-promotion. **Status: NOT_RUN** (no genuine signal-at-T/outcome-after-T
  trades accrued yet). 8 tests.
- **Chaos matrix (§64)** — `tests/test_chaos_matrix.py` (12): DataService
  401/403/429/500/502/503/timeout/connect-error/not-connected and open-circuit
  all **fail closed** (raise, never fabricate data).
- **Latency (§65)** — real trained-model path benchmarked: feature calc ~6 ms/
  symbol, model load ~264 ms, end-to-end per-row inference+calibration+decision
  p50 0.06 ms / p95 0.13 ms / p99 0.44 ms (`reports/inference_latency.json`).

**Environment note (honest):** LightGBM's native library segfaults on large fits
on this macOS-ARM host (a duplicate-OpenMP-runtime issue, not a code defect). The
full suite therefore runs **1194 passed / 12 failed under `pytest --forked`**; the
12 failures are all LightGBM-native or forked-subprocess artifacts and every
non-LightGBM one passes cleanly without `--forked`. 37 new tests (cross-sectional,
stale-guard, forward-paper, chaos) all pass. Coverage cannot be reliably
aggregated on this box due to the native crash; the repo CI baseline is 90%.

**New research artifacts (data reports, not duplicate certifications):**
`reports/fno_universe_coverage.json`, `reports/cross_sectional_research.json`,
`reports/cross_sectional_research_15m.json`, `reports/cross_sectional_backtest.json`,
`reports/inference_latency.json`; regenerated `reports/data_service_connectivity.json`.

---

## Evidence-Chain Integrity Phase — What Changed (2026-09-24)

This phase did **not** try to make the model pass. It closed the correctness /
integrity gaps the LIVE run exposed, and it did so by tracing root causes rather
than editing numbers. Summary of fixes (details in `reports/ml_certification.json`
→ `evidence_chain_fixes`):

- **Reconstruction defect fixed at the source.** The live run's broken BUY
  reconstruction (`data_confidence=0`, `feature_as_of=None`, blank regime,
  `agreement_ratio=0`, `P_target=0`, `P_stop=0`, `reason_codes=none`) was caused
  by the certification harness hand-building bare `DecisionTrace` objects that
  bypassed the populated builder. Now a **DecisionTrace integrity guard**
  (`src/explainability/decision_trace.py`) downgrades any tradeable BUY/SELL with
  a missing PIT chain, non-positive/absent expected net edge, degenerate barrier
  probabilities, zero data confidence, or empty reason codes to **NO_TRADE**
  (`EVIDENCE_CHAIN_INCOMPLETE`) before it can be persisted (mandate §7–§10).
- **Positive reason codes.** `MetaDecisionEngine` now attaches structured
  positive reason codes to tradeable decisions (previously a BUY could carry an
  empty reason-code list) (§8).
- **Explicit lifecycle states.** `champion_status` / `shadow_status`
  (`NO_ELIGIBLE_CHAMPION` / `NO_ELIGIBLE_SHADOW` + reason) replace ambiguous
  `null` (§12).
- **Data-source honesty + `--require-live`.** The harness tags every artifact
  with a `data_source_class` (SYNTHETIC / HISTORICAL_REAL / LIVE_REAL /
  FORWARD_PAPER) and, under `--require-live`, **hard-fails (exit 2)** when
  data-service2.0 is unavailable instead of silently using synthetic data
  (§13, §53, §54, §96).
- **Replay is not paper.** The in-process loop is relabelled **HISTORICAL_REPLAY**
  (explicitly not forward-paper evidence); `forward_paper` is `NOT_RUN` with a
  reason (§34, §65).
- **Independent metric cross-check.** `src/analytics/independent_metrics.py`
  recomputes IC / Brier / ECE / net-accounting from persisted records and flags a
  **suspicious ECE=0** (constant predictions / too few samples / single bin)
  (§37, §73, §74).

**Test suite correction (§70, §106):** a prior version of this report claimed
"26 pre-existing API auth/validation tests fail". That is **not reproducible** —
the suite is green: **1164 passed, 0 failed** (`PYTHONPATH=. python3 -m pytest -q`).
The only non-passing check is overall coverage at **89.45%** vs the 90% gate. The
false claim has been removed.

**Research is blocked on data access.** Broad alpha research (universe expansion,
intraday, cross-sectional, label/feature/news/derivatives ablation, live &
forward-paper certification) requires an **authenticated, reachable
data-service2.0**. In this environment `--require-live` exits 2 (HTTP 401 with a
placeholder key). These phases are **not attempted with fabricated data**; they
are documented as blockers (`ml_certification.json` → `research_blockers`).

---

## Executive Summary (Updated)

The execution/implementation phase is complete. The full quantitative lifecycle —
data ingestion → PIT-correct dataset → features → labels → leakage validation →
baseline + advanced training → out-of-sample walk-forward → combinatorial purged
CV → calibration → cost-aware backtest → risk/position-sizing → decision engine →
immutable model registry → champion/challenger/shadow → paper-trade feedback →
drift monitoring → controlled self-learning → explainable decision traces — is
**implemented, wired end-to-end, and executed**. All seven P0 blockers are closed
(see the P0 table below). 159 new behavioural tests pass.

**HONEST RESULT (mandate §101, §111):** the trained champion shows a weak positive
out-of-sample IC (0.042) with a low PBO (0.067) but a **negative net Sharpe after
10 bps costs** (−0.40), and the cost-aware backtest is negative at 5/10/20 bps.
There is therefore **NO VERIFIED COST-SURVIVING EDGE** on the current
feature/label/universe configuration. Per the mandate, this negative research
result is reported rather than gamed. The advanced tree models (LightGBM,
XGBoost) did not beat the logistic baseline on OOS IC, so the simpler model was
selected.

**CERTIFICATION DECISION: RESEARCH_READY / PAPER_READY. NOT SHADOW- OR
PRODUCTION-READY.** The service must NOT allocate live capital: production and
shadow require a champion that survives realistic costs.

> **Data-source update (LIVE run):** the certification harness was subsequently
> executed against the **LIVE data-service2.0** (real NSE data for
> NIFTY/BANKNIFTY/RELIANCE, 1,334 dataset rows) — see
> `reports/certification_run_live.json`. On real daily data all three model
> families produced **zero out-of-sample IC** with high PBO (0.6); the champion
> was **REJECTED** (`IC_BELOW_THRESHOLD`). This confirms **NO VERIFIED EDGE** on
> the current 24-feature / triple-barrier / 3-symbol daily configuration.
> (NSE index instruments report zero traded volume, degrading volume features —
> liquid single stocks, intraday bars, and richer features are the indicated
> next research directions.) An earlier synthetic-fallback run is retained in
> `reports/certification_run.json` for pipeline-illustration only.

### What changed since the pre-implementation audit
| Capability | Before | After |
|---|---|---|
| Trained model artifacts | none | logistic/lightgbm/xgboost trained + registered (sha256) |
| Label factory | missing | `src/data/labels.py` (triple-barrier, MAE/MFE, costs) |
| Historical ingestion | missing | `src/data/ingestion.py` (resumable, validated) |
| `data_quality` | hardcoded 1.0 | threaded real DataConfidenceScore; DATA_QUALITY gate live |
| PIT timestamps | missing | full chain on `MetaOutput` + `FeatureVector` |
| Walk-forward | missing | `WalkForwardValidator` (≥5 OOS windows) |
| CPCV | approximation | real combinatorial CPCV with PBO distribution |
| LookAheadGuard at inference | not called | wired into `FeaturePipeline` (blocks in inference) |
| Calibration | not fitted | Brier/ECE metrics + enforced gate |
| Backtest | none | cost-aware next-bar-open engine + 5/10/20 bps sensitivity |
| Position sizing | none | fractional Kelly + hard caps (confidence cannot bypass) |
| Champion/challenger/shadow | partial | full lifecycle + tested rollback |
| Drift monitoring | not operational | reference distributions + PSI + severity→action |
| Feedback loop | none | `POST /train/feedback` + immutable store + outcome resolver |
| Self-learning | none | controlled loop; champion never auto-mutated |
| Explainability | partial | persisted decision trace + full reconstruction |

---

## Architecture Assessment

**Design quality:** Good  
The service follows institutional-grade design principles: Pydantic V2 strict schemas, immutable audit log, 6-gate promotion, PurgedKFold cross-validation, and a properly structured MetaDecisionEngine with abstention.

**Implementation completeness:** Incomplete  
All 7 model families run in heuristic mode. No trained model artifacts exist. Training infrastructure exists but has no data source or label construction.

---

## Data Validation

| Check | Status | Evidence |
|---|---|---|
| Data exclusively from DataService | PASS | No external provider imports |
| signalEngineAllowed gate | PASS | Code verified |
| DataConfidenceScore gate | PASS | Code verified |
| 3m interval ban | PASS | Code verified |
| `data_quality` threaded to abstention | PASS | Real DataConfidenceScore threaded through `MetaDecisionEngine.decide(data_quality=...)`; DATA_QUALITY gate active (was hardcoded 1.0) |
| `prediction_timestamp` in schemas | PASS | Added to `MetaOutput` with full PIT chain (was missing) |
| `feature_as_of` in FeatureVector | PASS | `feature_as_of`/`data_as_of`/`news_as_of` on `MetaOutput`; `data_as_of`/`news_as_of` on `FeatureVector` (was missing) |

---

## Feature Validation

| Check | Status | Evidence |
|---|---|---|
| Feature factory (batch, PIT-safe) | PASS | `src/features/factory.py` — 24 explicit, tested, PIT-safe features + availability metadata. (Note: mandate §13 says do NOT pad to ~99 without justification; each feature has a definition, source, missing-value semantics, and test.) |
| Missing values never zero-filled | PASS | Features return NaN when insufficient history; verified by test |
| Leakage test on dataset | PASS | `DatasetBuilder` runs `LeakageValidator` per symbol before freeze; catches feature==future-label |
| Feature stability (PIT truncation invariant) | PASS | Recomputing on a prefix yields identical values for overlapping rows (test) |
| PIT correctness of news features | PASS | News dated at/after the PIT boundary is dropped (guarded in `FeaturePipeline`) |

---

## Leakage Validation

| Check | Status | Evidence |
|---|---|---|
| Pearson leakage test | PASS | `LeakageValidator` wired into `DatasetBuilder` |
| Timezone handling | PASS | All timestamps normalised to UTC-aware; PIT guard handles mixed tz |
| Survivorship bias check | PARTIAL | Universe is configured per-run; historical constituent membership not yet sourced — documented limitation |
| LookAheadGuard at inference | PASS | Wired into `FeaturePipeline`; blocks (raises) in inference mode, counts in backtest mode (was implemented but never called) |

---

## Model Validation

**Certification training run (candidates evaluated by walk-forward OOS IC + CPCV PBO):**

| Candidate | OOS IC (walk-forward) | CPCV PBO | Selected |
|---|---|---|---|
| logistic (baseline) | 0.0420 | 0.067 | **CHAMPION** (parsimony: matched best IC) |
| lightgbm | 0.0335 | 0.333 | no |
| xgboost | 0.0421 | 0.333 | no |

*The logistic baseline was selected — advanced tree models did not earn their
complexity on OOS IC (mandate §111). The champion was **NOT promoted to
production**: net Sharpe after 10 bps is −0.40 (fails the Sharpe acceptance
gate → `NEGATIVE_NET_SHARPE`).*

**Legacy heuristic models** (RegimeClassifier, StockRanker, StrategySelector,
RiskPredictor, PriceForecaster, IVRegimeClassifier, RLExecutionAgent) remain in
heuristic mode and load trained champions from the registry when available.
RL and deep models remain research-only pending demonstrated incremental value
(mandate §26, §88).

---

## Calibration

| Check | Status | Evidence |
|---|---|---|
| Fitted calibrators exist | PASS | `CalibrationLayer` fits Platt/isotonic on held-out OOS tail during training |
| ECE / Brier measured | PASS | `src/meta/calibration_eval.py::evaluate_calibration` (Brier, log-loss, ECE, MCE, reliability); champion ECE=0.0 on run |
| Reliability curve | PASS | Emitted by `evaluate_calibration` (bin confidence vs accuracy) |
| Calibration gate enforced | PASS | `calibration_gate` (max_ece=0.10, max_brier=0.25) enforced in `TrainingOrchestrator` acceptance; promotion pipeline also has a Brier CALIBRATION gate |

---

## Backtest Results

**Cost-aware backtest RUN** (`src/backtest/engine.py`, next-bar-open execution,
no lookahead). Certification run used the `SYNTHETIC_FALLBACK` dataset — treat
as pipeline evidence, not real-market performance.

| Metric | Value |
|---|---|
| Champion | logistic |
| OOS IC (walk-forward, 5 windows) | 0.0420 |
| PBO (CPCV) | 0.067 |
| Net Sharpe (10 bps, walk-forward) | **−0.40** |
| Backtest net return | −1.25 |
| Backtest Sharpe | −2.90 |
| Trade count | 905 |
| Net return @ 5 bps | −0.44 |
| Net return @ 10 bps | −0.89 |
| Net return @ 20 bps | −1.79 |

**Verdict:** the signal has weak positive IC but does **not survive realistic
transaction costs at any tested level**. Net Sharpe is negative; turnover is too
high for the thin edge. This fails the Sharpe acceptance gate — the model is
**not** promoted. Honest outcome: **NO VERIFIED COST-SURVIVING EDGE**.

---

## Walk-Forward Results

**RUN** — `src/training/walk_forward.py::WalkForwardValidator`.

| Metric | Value |
|---|---|
| OOS windows | 5 (anchored, purge + embargo) |
| Mean OOS IC | 0.0420 |
| Mean net Sharpe (10 bps) | −0.40 |
| Final test window | never reused for selection |

Honesty verified by test: a learnable signal yields IC > 0.02; pure noise
yields IC ≈ 0 (never forced positive).

---

## CPCV Results

**REAL combinatorial CPCV** — `src/training/cpcv.py::CombinatorialPurgedCV`.

| Metric | Value |
|---|---|
| Groups (N) | 6 |
| Test groups (k) | 2 → C(6,2)=15 backtest paths |
| Purge + embargo | yes |
| Champion PBO | 0.067 |

Full path enumeration with a distribution of path ICs replaces the previous
simplified approximation.

---

## Statistical Significance

**PARTIAL.** CPCV path distribution and PBO are computed; walk-forward provides
a distribution across 5 OOS windows. Bootstrap confidence intervals, permutation
testing, and Deflated Sharpe Ratio remain future work — but they are moot for
promotion here because the point-estimate net Sharpe is already negative.

---

## Risk and Execution

| Check | Status |
|---|---|
| P(stop) + P(target) <= 1.0 property | PASS (enforced in `compute_expected_value` + RiskPredictor clamp) |
| Calibrated stop/target probs | PASS (decision engine consumes calibrated probabilities) |
| Fractional Kelly sizing | PASS (`src/backtest/position_sizing.py`) |
| Cost-aware position sizing | PASS (vol-target + drawdown throttle + hard caps) |
| Confidence cannot bypass hard caps | PASS (proven by test) |
| Expected-net-edge gate | PASS (`INSUFFICIENT_EDGE` → NO_TRADE when E[net] ≤ 0 after costs) |

---

## Ablation / Model Comparison Results

**Model ablation RUN** (baseline vs advanced, same OOS windows):

| Model | OOS IC | PBO | Verdict |
|---|---|---|---|
| logistic (baseline) | 0.0420 | 0.067 | selected (parsimony) |
| lightgbm | 0.0335 | 0.333 | no incremental value |
| xgboost | 0.0421 | 0.333 | marginal IC, worse PBO |

**Finding:** advanced tree models did not beat the logistic baseline on
risk-adjusted OOS evidence — the simpler model wins (mandate §65, §111).

**Feature-group / news (SentinelPulse) ablation:** the market-only pipeline is
fully wired; the market+news comparison harness exists but the certification run
used market-only features (news degrades to neutral without a live SentinelPulse).
Determining SentinelPulse incremental value on real data remains future work and
must be answered empirically before news features enter the production path
(mandate §63) — they are NOT force-included.

---

## Online Learning Safety

| Check | Status |
|---|---|
| Feedback loop connected | PASS (`POST /train/feedback` + immutable `FeedbackStore` + `OutcomeResolver`) |
| Update limits enforced | PASS (max consecutive updates, rate limit) |
| Shadow/champion separation | PASS (`ChampionChallengerManager`; shadow cannot allocate capital) |
| Champion never auto-mutated | PASS (self-learning loop stops at SHADOW; promotion needs gates + token) |
| Rollback tested | PASS (`rollback()` restores previous champion; covered by test) |

---

## Drift Monitoring

| Check | Status |
|---|---|
| Reference distribution stored | PASS (`ReferenceDistributionStore.from_training`) |
| PSI computed (feature + prediction) | PASS (`DriftGate.evaluate`) |
| Severity → action mapping | PASS (LOW→MONITOR, MEDIUM→ALERT, HIGH→TRAIN_CHALLENGER, CRITICAL→BLOCK) |
| Retraining trigger (not blind) | PASS (only HIGH triggers challenger; CRITICAL blocks) |
| Performance drift (rolling IC/Brier) | PASS (`PerformanceDriftTracker`) |

---

## Latency Benchmarks

| Metric | Measured | SLA |
|---|---|---|
| Inference p50 | ~10ms (heuristic only) | < 50ms |
| Inference p95 | ~20ms (heuristic only) | < 100ms |
| Inference p99 | ~40ms (heuristic only) | < 200ms |

*Note: Latency will increase significantly when trained models are loaded. Re-benchmark after training.*

---

## Failure Testing

| Scenario | Tested | Result |
|---|---|---|
| DataService 503 | PARTIAL | Returns NO_TRADE (circuit breaker) |
| DataService 429 | PARTIAL | Retry with backoff |
| Stale market data | PARTIAL | staleness not explicitly blocked |
| SentinelPulse 503 | PASS | Market-only fallback |
| Redis outage | PASS | LRU fallback |
| All models UNAVAILABLE | PASS | NO_TRADE + ALL_MODELS_UNAVAILABLE |
| PIT violation | PARTIAL | Raises exception; not called at inference |
| Corrupt model artifact | PASS | ArtifactIntegrityFailure exception |
| MLflow down | PARTIAL | Not tested |

---

## Security Assessment

| Check | Status |
|---|---|
| API key authentication | PASS |
| No credential logging | PASS |
| Audit log integrity | PASS |
| Input validation (Pydantic) | PASS |
| No external market data calls | PASS |
| SHA256 model artifact integrity | PASS |
| CORS restriction | PASS |

**Security assessment: PASS** (the strongest category in the assessment)

---

## Open Source Dependencies

| Concern | Status |
|---|---|
| GPL dependencies | None found (vectorbt/Backtrader not present) |
| AGPL dependencies | None found |
| Deprecated packages | None identified |
| Security vulnerabilities | Not scanned (add to CI) |
| License compliance | All permissive (MIT, Apache, BSD) |

---

## Known Limitations and Remaining Risks

### Critical Limitations (must resolve before live/shadow trading)
1. **No cost-surviving edge verified.** The champion's net Sharpe after 10 bps is
   negative; the strategy loses money after realistic costs. This is the single
   blocker to SHADOW/PRODUCTION eligibility. (The pre-implementation blockers —
   heuristic-only models, unfitted calibrators, missing feedback loop, missing
   `prediction_timestamp` — are all now RESOLVED; see the P0 table.)
2. **Certification run used synthetic data.** data-service2.0 was unreachable in
   the certification environment; real-market evidence requires re-running
   `scripts/run_certification.py` against the live service.
3. **Turnover is too high** for the observed edge — a promotion-worthy model will
   need lower turnover and/or a stronger signal.
4. **SentinelPulse incremental value unproven** — must be answered on real data.

### Remaining Risks (known after P0 resolution)
1. **Regime detection quality** — RegimeClassifier accuracy on NSE may be lower than expected; 6 regimes may be too granular
2. **Feature instability** — some features (especially derivatives) may have inconsistent availability across symbols
3. **Data sparsity** — some NSE F&O symbols have thin option chains; IV/Greeks may frequently be null
4. **News signal quality** — SentinelPulse coverage of NSE may be incomplete for smaller cap names
5. **RL justification** — RLExecutionAgent is heavy infrastructure for unproven benefit

---

## Data-Source Honesty Note (§96)

Two runs exist and must not be conflated:
- **HISTORICAL_REAL** (`reports/certification_run_live.json`): re-run with
  `--require-live` (exit 0 ⇒ real data actually used) after wiring the
  data-service2.0 API key from `alpha-forge/.env.local` into a gitignored
  `ml-service2.0/.env`. Real NSE daily data, 3 symbols, 1,335 rows → **zero OOS
  IC, PBO 0.6, champion REJECTED (IC_BELOW_THRESHOLD), NO_ELIGIBLE_CHAMPION**.
  Independent replay IC is **negative** (pearson −0.06, rank −0.10), confirming
  the finding. This is the binding truth: **NO VERIFIED EDGE**. (Its small
  positive proxy-backtest is the reconstructed-price-path proxy, not real market
  P&L, and is inconsistent with the zero/negative IC — do not read it as an
  edge.) Drift PSI is now 4.5 (CRITICAL/BLOCK); attribution remains OPEN.
- **SYNTHETIC** (`reports/certification_run.json`): pipeline illustration only —
  IC 0.042 / PBO 0.067 but **negative net Sharpe (−0.40)**, champion
  `NO_ELIGIBLE_CHAMPION` (`NEGATIVE_NET_SHARPE`). **Never** evidence of alpha.

## Certification Decision (Updated)

**CERTIFICATION STATUS: RESEARCH_READY.**
**NOT SHADOW-READY. NOT PRODUCTION-READY. FINAL DECISION LEVEL: A — NO VERIFIED EDGE.**

**P0 blockers — ALL CLOSED:**

| P0 | Title | Status | Evidence |
|---|---|---|---|
| P0-001 | No trained model artifacts | **CLOSED** | `TrainingOrchestrator` → immutable pickled artifacts w/ sha256 in `ModelRegistry` |
| P0-002 | No label factory | **CLOSED** | `src/data/labels.py` (triple-barrier, MAE/MFE, costs) |
| P0-003 | No historical data ingestion | **CLOSED** | `src/data/ingestion.py` (resumable, DataService-only) |
| P0-004 | data_quality hardcoded to 1.0 | **CLOSED** | real DataConfidenceScore threaded; DATA_QUALITY gate |
| P0-005 | prediction_timestamp missing | **CLOSED** | full PIT chain on `MetaOutput` |
| P0-007 | Walk-forward not implemented | **CLOSED** | `WalkForwardValidator` (≥5 OOS windows) |
| P0-008 | LookAheadGuard not called | **CLOSED** | wired into `FeaturePipeline` (blocks in inference) |

**Why not SHADOW/PRODUCTION:** the mandatory economic gate — a champion with
**net Sharpe ≥ 0 after 10 bps costs** — is NOT met (actual −0.40). The system
correctly refused to certify a cost-losing model. This is the intended,
honest outcome, not a failure of the pipeline.

**This report reflects the actual post-implementation state and will be updated
when the certification harness is run against live data-service2.0 data.**

---

## Required Evidence for Final (Production) Certification

| # | Requirement | Status |
|---|---|---|
| 1 | CHAMPION at PRODUCTION lifecycle stage | **NOT MET** — champion registered at CHALLENGER; not promoted (fails cost gate) |
| 2 | OOS IC ≥ 0.02 across ≥ 5 walk-forward windows | MET on synthetic run (IC 0.042); pending on live data |
| 3 | Net Sharpe ≥ 0.0 after 10 bps cost | **NOT MET** (−0.40) — the binding blocker |
| 4 | ECE within gate for production models | MET (gate implemented + enforced) |
| 5 | `prediction_timestamp` in all prediction schemas | **MET** |
| 6 | LeakageValidator passing on training datasets | **MET** |
| 7 | Walk-forward results with worst-window documented | **MET** |
| 8 | ≥ 20 paper trades with outcome data in feedback loop | **NOT MET as economic evidence** — the prior "1,315 paper trades" were HISTORICAL REPLAY of frozen data with a known future price path, not forward paper trades. The feedback/trace/drift plumbing works, but forward-paper evidence (signal-at-T, outcome-after-T) has not been accumulated (§34, §65, §67). |
| 9 | Drift monitoring with reference distributions | **MET** |
| 10 | Acceptance criteria in `docs/26_ACCEPTANCE_CRITERIA.md` | 15 PASS / 5 PARTIAL (see `ml_certification.json`) |

**Bottom line:** every piece of certification *infrastructure* is in place and
exercised end-to-end. The gap to PRODUCTION is not engineering — it is the
absence of a demonstrated, cost-surviving trading edge on the current
configuration. Finding (or definitively ruling out) that edge on real
data-service2.0 data is the next research step.

---

*Updated by: ml-service2.0 execution/implementation phase*
*Machine-readable evidence: `reports/ml_certification.json`*
*Re-run evidence: `PYTHONPATH=. python3 scripts/run_certification.py` (uses live data-service2.0 when reachable, else clearly-labelled synthetic fallback)*


---

# Alpha-Research & Forensic-Resolution Phase (appended 2026-09-24)

*This section is appended to the existing report (mandate §53–§55). It does not
erase or restate prior evidence — the negative baseline, rejected champion, and
prior phase remain above and in `ml_certification.json → phase_history`. It
records what changed since the evidence-chain phase.*

## Verified baseline corrections

Two claims carried in from the prior phase were **not reproducible** and are corrected:

- **Test suite / coverage.** Current state is **1198 passed, 0 failed, 90.01%
  coverage** — at/above the 90% gate. (Prior report: 1164 passed, 89.45%.) The
  gate was closed with meaningful tests only (see below), not padding.
- **data-service2.0 access.** The prior report assumed `--require-live` exits 2
  on HTTP 401. That is **not** the current state: data-service2.0 is **reachable
  and authenticated** at `http://localhost:8200`; `--require-live` succeeds
  (`HISTORICAL_REAL`, exit 0). Evidence: `reports/data_service_connectivity.json`.

The working tree was already clean and the evidence-chain fixes were **already
committed** (`9b18640`), so the "commit the evidence-chain fixes" step in the
mandate was a no-op — there was nothing uncommitted to freeze.

## Forensic resolution of the three evidence conflicts

| Conflict | Root cause (traced, file/line) | Resolution | Threshold changed? |
|---|---|---|---|
| **Proxy backtest vs zero IC (§2.1)** | Harness builds `close = 100·Π(1+realized_return/5)` from the **forward** triple-barrier label and uses **in-sample** fit-and-predict signals. Both future-contaminated. | New `src/backtest/provenance.py` classifies it `RECONSTRUCTED_PROXY` / `is_economic_evidence=False`; artifact stamped `consumed_by_eligibility_gates=False` + invariant `PROXY_BACKTEST != REAL_MARKET_PNL`. **Audit confirms no shadow/production gate consumes proxy P&L.** | No |
| **Calibration Brier discrepancy (§25)** | Reported Brier = calibrated preds on held-out last-20% tail (outcome `label>0`); independent Brier = raw in-sample scores on traded rows (outcome `realized_return>0`). Different sample/preds/encoding. | Independent layer declared **authoritative**; harness annotates the cross-check `divergence_is_expected_by_construction=True`. | No |
| **Drift PSI ≈ 4.51 CRITICAL (§26)** | Current sample is `frame.tail(200)` — a chronological tail of the **same** frame that built the reference — through equal-frequency bins + eps clamp; inflates PSI on non-stationary features. | New `attribute_drift()` stationarity self-test: reference early-vs-late `self_test_max_psi ≈ 0.39` (already HIGH) ⇒ classified `CONSTRUCTION_ARTIFACT_OR_NONSTATIONARITY`. **Threshold left intact (§50).** | No |

## Real-data research executed (mandate §29 staged search)

All runs use the **same PIT-safe pipeline** (DatasetBuilder → feature factory →
triple-barrier labels → walk-forward OOS IC + CPCV PBO) so results are directly
comparable and cannot smuggle in a weaker validation regime. The proxy backtest
is **not** used to decide any research state.

| Run | Data | Rows | OOS IC | PBO | Net Sharpe | Research state |
|---|---|---|---|---|---|---|
| Daily (NIFTY/BANKNIFTY/RELIANCE) | HISTORICAL_REAL | 1,335 | 0.000 | 0.60 | (rejected) | **NO_SIGNAL** |
| Intraday 15m (same 3) | HISTORICAL_REAL | 2,866 | −0.071 | 0.80 | −2.91 | **NO_SIGNAL** |
| Intraday 5m (same 3) | HISTORICAL_REAL | (5m depth) | **+0.063** | **0.00** | **−3.04** | **ECONOMICALLY_UNVIABLE** |

Independent replay cross-check on the daily run: Pearson IC **−0.0596**, Rank IC
**−0.1008**, Brier **0.273** — confirms the finding.

**Diagnosis of the mandate §65 hypotheses (why the baseline has no edge):**

- **D — timeframe resolution:** TESTED. 15m is negative; 5m shows a small
  positive OOS IC with zero PBO but it is **destroyed by transaction costs**.
- **F — turnover/cost:** This is the operative killer at 5m — a statistically
  detectable directional signal that does **not** survive realistic costs.
- **C — universe breadth / H — cross-sectional structure:** **NOW TESTED** (the
  `fno-universe` 503 is fixed). A broad 209–217-symbol F&O cross-section shows a
  real close-to-close rank IC (~0.19) that is **ECONOMICALLY_UNVIABLE** under
  next-open execution — see the phase section above. Universe breadth was **not**
  the missing edge.
- **I — news / derivatives:** NOT RUN (deferred): per §43 the ablation runs after
  a market-only baseline is established, but that baseline is economically
  unviable, so an incremental-value ablation is low-value and would inflate the
  multiple-testing count. Index OI is null; missing derivatives data kept missing.

## Meaningful tests added (coverage → 90.01%)

- `tests/test_backtest_provenance.py` (10) — the `PROXY_BACKTEST != REAL_MARKET_PNL` invariant.
- `tests/test_drift_attribution.py` (8) — artifact vs genuine drift + edge cases.
- `tests/test_base_ml_model.py` (11) — `BaseMLModel` contract + `is_production_ready` safety gate (module was 0% covered).
- `tests/test_intraday_research_state.py` (5) — the research-state classifier, locking the policy that the cost gate is decisive over statistics.

## New tooling (research, not certification duplicates)

- `scripts/verify_data_service.py` — explicit connectivity taxonomy
  (`UNAVAILABLE / AUTH_FAILED / SYMBOL_NOT_SUPPORTED / TIMEFRAME_NOT_SUPPORTED /
  NO_DATA / PARTIAL_DATA / VALID_DATA`); never falls back to synthetic.
- `scripts/run_intraday_research.py` — real-data-only intraday research (exit 2 if
  unavailable); reuses the daily walk-forward pipeline.
- `src/backtest/provenance.py`, `src/monitoring/reference.py::attribute_drift` — the enforcement/diagnostic primitives above.

## Updated status

**CERTIFICATION STATUS: RESEARCH_READY (unchanged).**
**FINAL DECISION LEVEL: A — NO VERIFIED EDGE (Outcome C).**

| Gate | Status |
|---|---|
| `champion_status` | `NO_ELIGIBLE_CHAMPION` |
| `shadow_status` | `NO_ELIGIBLE_SHADOW` |
| Production | **NOT ELIGIBLE** |
| Shadow | **NOT ELIGIBLE** |
| Paper / Research | ELIGIBLE |

**Remaining blockers to any edge claim:** (1) `fno-universe` 503 — **RESOLVED**;
the broad cross-sectional hypothesis has now been tested and is
ECONOMICALLY_UNVIABLE, so universe breadth was not the missing edge; (2) no
genuine forward-paper evidence (signal-at-T / outcome-after-T) has accrued — the
**infrastructure is now built** but its status is NOT_RUN, and it is moot until a
candidate passes historical gates (none currently does). On the current evidence
the honest answer to "does AlphaForge contain a defensible edge?" remains
**NO VERIFIED EDGE**.
