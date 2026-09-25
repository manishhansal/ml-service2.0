# Executive Summary
**AlphaForge ml-service2.0 — Confirmation Phase Complete**

*Last updated: 2026-09-25*
*Status: **PAPER_ELIGIBLE** — forward paper Session 1 started; SHADOW_BLOCKED pending resolution*

---

## Changelog

| Date | Phase | Key outcome |
|---|---|---|
| 2026-09-24 | Forensic audit + implementation | All 8 P0 blockers closed; infrastructure built |
| 2026-09-24 | Evidence-chain hardening | Proxy backtest != real P&L invariant enforced |
| 2026-09-24 | F&O universe restoration | fno-universe 503 root-caused and fixed; 220 symbols confirmed |
| 2026-09-24 | Cross-sectional research | 209 symbols, rank IC 0.19 close-to-close → ECONOMICALLY_UNVIABLE at next-open |
| 2026-09-24 | Docker ML training | Real NSE data, 10 symbols → IC < 0.02, REJECTED |
| 2026-09-25 | Extended research (22 → 65 symbols) | IC 0.29–0.47 (barrier-clamped), PBO=0.00, Sharpe 2.0–3.8 → PAPER_ELIGIBLE |
| 2026-09-25 | Confirmation phase | Pre-registered protocol, 7/7 gates PASS, forward paper started |
| 2026-09-25 | Universe expansion | 217/220 symbols ingested (98.6%) |
| 2026-09-25 | SentinelPulse backfill | 919 articles processed; historical news unavailable pre-2026-09-18 |
| 2026-09-25 | Forward paper Session 1 | 65 signals persisted, resolve 2026-09-30 |

---

## Current State (2026-09-25)

### The honest summary

**The architecture is good. The signal exists statistically but has not been proven executable.**

All 8 P0 blockers are closed. The training pipeline is real, PIT-safe, leakage-validated, and run on real NSE data. A 65-symbol lightgbm model with 5-year history (127,122 rows) shows IC=0.49 against barrier-clamped returns, PBO=0.00, Sharpe 5.4 at 10 bps across all 6 years (2021–2026). All 4 null tests reject H0 at the 100th percentile. The DSR is 1.0 even after 57 trials.

**The critical caveat:** The IC is measured against barrier-clamped ±2% returns (88.8% of labels are exactly ±2%). This is a classification IC. The model has learned strong NSE daily momentum (ret_1 is the top feature by 5:1 margin over next feature, with 5x lag-0/lag-1 decay confirming PIT safety). Whether this translates to continuous-return alpha has not been proven — the frozen parquet does not contain unclamped prices.

Forward paper Session 1 is live. 65 signals were generated at 18:26 UTC on 2026-09-25. Outcomes resolve after 5 trading days (2026-09-30). Minimum 20 resolved TRADE signals required before any shadow eligibility decision.

---

## What changed since the pre-implementation audit

| Claim then | Reality now |
|---|---|
| "7 ML models — all heuristic" | lightgbm champion trained on 127k real NSE bars, SHA256-verified, IC=0.49 OOS |
| "Calibrated probabilities — none" | Isotonic calibrator fitted; ECE=0.0; Brier=0.208 |
| "Equal-weighted ensemble — IC registry empty" | Parsimony selection: lightgbm chosen over logistic/xgboost on OOS IC + PBO |
| "Model promotion gates — none promoted" | CHALLENGER stage; correctly gated from shadow pending forward paper |
| "PurgedKFold — not run on real data" | 5-window walk-forward + CPCV, all OOS windows positive, PBO=0.00 |
| "SHAP — zero contributions" | Feature importance documented; ret_1 dominant (5:1) |
| "Online learning — no feedback loop" | ForwardPaperRunner live; outcome resolution infrastructure ready |
| "Drift monitoring — no reference distribution" | Reference distributions established in training; PSI gating active |
| "No trained artifacts" | model.pkl SHA256=97e601197c...; dataset hash=ee508cb6...; deterministic |
| "No forward paper evidence" | Session 1: 65 signals, 54 TRADE, resolve 2026-09-30 |

---

## Certification State

```
RESEARCH_SIGNAL_CONFIRMED:  YES
PAPER_ELIGIBLE:              YES — statistical gates pass, Session 1 live
SHADOW_ELIGIBLE:             NO  — forward paper pending (0/20 resolved)
PRODUCTION_ELIGIBLE:         NO  — requires shadow period after paper
```

---

## Remaining blockers to shadow

| # | Blocker | Status |
|---|---|---|
| B-001 | Forward paper — no resolved trades | **IN_PROGRESS** (Session 1 live, resolve 2026-09-30) |
| B-002 | Continuous-return IC unknown | **OPEN** — parquet has barrier-clamped returns only |
| B-003 | Universe 65/220 | **RESOLVED** — 217/220 (98.6%) after ingestion run |
| B-004 | SentinelPulse 0 historical samples | **PARTIAL** — 919 live articles; pre-2026-09-18 unavailable from sources |
| B-005 | Survivorship (accepted limitation) | **ACCEPTED** — CURRENT_UNIVERSE_ONLY documented |

---

## What must happen next

1. **Monday 2026-09-29 market close:** run `make forward-paper` for Session 2
2. **After 2026-09-30 18:30 UTC:** Session 1 signals become resolvable — fetch open prices, record outcomes
3. **After 20 resolved TRADE signals:** compute forward-paper IC and Sharpe; re-evaluate shadow gate
4. **Re-evaluate continuous IC:** run model against fresh OHLCV data with unclamped returns from data-service2.0
5. **SentinelPulse:** wait for live news to accumulate; run market+news ablation when ≥252 trading-day coverage exists

---

*Machine-readable evidence: `reports/final_certification.json`*
*Canonical report: `reports/ML_SERVICE_FINAL_CERTIFICATION.md`*
*Frozen baseline: `artifacts/confirmation/CONFIRMATION_BASELINE_V1.json`*
*Forward paper store: `artifacts/forward_paper/signals.jsonl`*
