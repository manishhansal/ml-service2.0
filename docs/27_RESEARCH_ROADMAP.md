# Research Roadmap
**ml-service2.0 — Quantitative Research Agenda**

*Original date: 2026-09-24*
*Updated: 2026-09-25 — post-confirmation-phase*

---

## Current Research State (2026-09-25)

```
CONFIRMATION_BASELINE_V1
  model:     lightgbm (1.0.0-20260925080931531542)
  universe:  65 symbols F&O daily (CURRENT_UNIVERSE_ONLY)
  IC:        0.486 (barrier-clamped; continuous unknown)
  Sharpe:    5.47 at 10bps (all 6 years positive)
  PBO:       0.000
  forward:   Session 1 live (65 signals, resolve 2026-09-30)
  state:     PAPER_ELIGIBLE / SHADOW_BLOCKED
```

## Immediate research priorities (ordered, do not skip ahead)

### Priority R-01: Resolve forward paper Session 1 (2026-09-30)
- Fetch open prices for 2026-09-30 from data-service2.0
- Record realized outcomes in feedback store
- Compute: forward-paper IC, hit rate, gross/net PnL per signal
- If outcomes match historical pattern: schedule Session 2, 3, ... (daily)
- **Gate:** this must happen before any other research change to the live model

### Priority R-02: Measure continuous-return IC
- The frozen parquet has only barrier-clamped ±2% returns
- To measure true continuous IC: re-evaluate model against fresh OHLCV bars
  using `(open[T+1+h] - open[T+1]) / open[T+1]` as the target
- Script: add `--continuous-ic` flag to `scripts/run_confirmation.py`
- Expected result: IC lower than 0.486 (possibly ~0.15–0.30 if momentum is real,
  possibly near zero if signal is purely a classification artifact)
- **Blocker B-002** — required before shadow gate

### Priority R-03: Market+news ablation (when SentinelPulse has ≥252 days)
- SentinelPulse live news started 2026-09-18; 252 trading days ≈ 2027-09-17
- Until then: periodic check via `make backfill-status`
- When ready: run 5 pre-registered ablation experiments (A–E from CONFIRMATION_PROTOCOL)
- Record all 5 in RESEARCH_TRIAL_LEDGER before observing results
- **Do not add news features to the live model until ablation proves incremental value**

### Priority R-04: Expand confirmation to 217-symbol universe
- `data/1d/1d/` now has 217 symbol parquet files (217/220, 98.6% coverage)
- Re-run `DatasetBuilder` on the expanded universe to build a new dataset
- Re-run confirmation protocol on the expanded dataset
- **Classification:** EXPLORATORY (not confirmatory) — different universe than CONFIRMATION_BASELINE_V1
- Record in RESEARCH_TRIAL_LEDGER before running

---

## Research principles (unchanged)

Research follows a strict evidence-first policy:

1. Simple first: build simple model → validate → then add complexity
2. Document hypotheses before testing them
3. Document negative results — what didn't work is as important as what did
4. Every research experiment must produce a machine-readable result
5. Research findings feed into the migration plan; research does not bypass gates
6. No tuning after seeing OOS results — any post-hoc change is EXPLORATORY

---

## What NOT to do now

- Do not retrain the model on the expanded 217-symbol dataset and claim it is the confirmed candidate
- Do not tune hyperparameters because the confirmation Sharpe is already positive
- Do not modify the execution convention (next_open is fixed)
- Do not add SentinelPulse features to the live model without ablation evidence
- Do not skip the forward paper gate to proceed to shadow

---

## Original research questions (from 2026-09-24 — updated with outcomes)

### Q1: Which features are most predictive?
**Answered (2026-09-25):** `ret_1` is dominant (865 gain importance, 5:1 over next feature).
Feature attribution confirms momentum. Not an artifact of label construction.

### Q2: Triple-barrier vs fixed-horizon?
**Status:** Triple-barrier with ±2% barriers used in CONFIRMATION_BASELINE_V1.
**Finding:** 88.8% of returns are clamped at ±2% — raises the IC artificially for classification.
Continuous-return evaluation (R-02) will determine whether fixed-horizon outperforms.

### Q3: Does SentinelPulse add incremental value?
**Status:** BLOCKED — 919 articles (7 days), insufficient for ablation.
**When:** when live news reaches ≥252 trading-day coverage (~2027-09-17).

### Q4: Optimal volatility regime?
**Partial answer (2026-09-25):** All 6 years (2021–2026) show positive IC and positive Sharpe.
No regime dependence detected. Regime breakdown is consistent.

### Q5: Minimum viable feature set?
**Partial answer:** `ret_1` drives most predictive value. Recursive feature elimination
would likely converge to ~5–10 features. Not explored yet — not a priority while
forward paper is unresolved.

### Q6: Does IV skew predict direction better than IV level for index options?
- Hypothesis: PCR + skew is more predictive than ATM IV alone for NIFTY
- Test: Ablation study comparing D-group derivatives subsets

### Q7: Is fractional Kelly the optimal position sizing approach?
- Hypothesis: Fractional Kelly (25%) produces higher risk-adjusted returns than fixed 2% sizing
- Test: Simulate different sizing approaches on paper trade history
- Note: requires outcome data from feedback loop

### Q8: Does regime-conditional calibration improve ECE vs. global calibration?
- Hypothesis: Calibrating separately per regime improves calibration quality
- Test: Compare ECE(global Platt) vs. ECE(per-regime Platt) on OOS data

### Q9: What is the IC half-life for NSE intraday signals?
- Hypothesis: IC for 5m predictions decays to near-zero within 30 minutes
- Test: Compute IC as a function of prediction age; fit exponential decay curve
- Implication: Sets signal expiry policy

### Q10: Does cross-sectional ranking add value over individual stock models?
- Hypothesis: Cross-sectional StockRanker IC > individual stock IC on same data
- Test: Compare ranking IC vs. individual prediction IC on F&O universe

---

## 5. Long-Term Research Questions (Phase 10+)

### Q11: Can online learning from NSE intraday data materially improve IC within a session?
- Hypothesis: Intraday regime shifts are predictable from volume/OI patterns; online updating captures this
- Requires: Feedback loop operational, 3+ months of outcome data

### Q12: Does the LLM Research Agent generate useful hypotheses?
- Hypothesis: Agent-generated feature hypotheses have IC > 0 at rate > 30%
- Test: Agent proposes 10 hypotheses; researcher blindly validates each
- Requires: Research agent implementation (Phase 12)

### Q13: What is the optimal ensemble architecture for NSE F&O?
- Hypothesis: 4-model ensemble (Regime + Risk + Ranker + Price) outperforms 7-model ensemble
- Test: Ablation over ensemble compositions; compare OOS IC and Sharpe

---

## 6. Research Log Format

Every experiment is logged in `experiments/` as:

```json
{
    "experiment_id": "EXP-001",
    "hypothesis": "Triple-barrier labels outperform fixed-horizon labels",
    "status": "COMPLETED",
    "started_at": "2026-10-01",
    "completed_at": "2026-10-14",
    "dataset_version": "v1.0.0-nsefno-20220101-20251231",
    "model": "LightGBM-v1",
    "label_type_a": "triple_barrier",
    "label_type_b": "fixed_horizon_5bar",
    "result": {
        "ic_triple_barrier": 0.034,
        "ic_fixed_horizon": 0.021,
        "ic_difference": 0.013,
        "statistical_significance": {"p_value": 0.032, "significant": true},
        "conclusion": "SUPPORTED",
        "decision": "PROMOTE triple_barrier for 5m-1h models"
    },
    "mlflow_run_id": "abc123",
    "code_sha": "f1a2b3c4"
}
```

---

*End of Research Roadmap*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Research harness delivered: scripts/run_certification.py runs the full lifecycle against LIVE data-service2.0 (real NSE data) or a clearly-labelled synthetic fallback. Open research items surfaced by the real-data run: no cost-surviving edge on daily bars (OOS IC=0, PBO=0.6, net Sharpe negative); NSE index instruments report zero traded volume which degrades volume-based features. Indicated next directions: liquid single-stock universe, intraday bars, richer/derivatives features, and an empirical SentinelPulse incremental-value study (news must not be force-included). Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
