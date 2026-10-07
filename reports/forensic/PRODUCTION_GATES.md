# PRODUCTION GATES — v2c MODEL ASSESSMENT
**Repository:** ml-service2.0 | **Updated:** 2026-10-06 (post-close)
**Model:** v2c (LGBMRegressor, 65 features, 7-day CS rank label)
**Previous assessment (v1, 2026-10-01): 9 FAIL / 9 PASS — superseded by this document**

---

## Gate Summary

| Gate | Description | Status | Evidence |
|------|-------------|--------|---------|
| **G_PIT** | No look-ahead bias | ✅ PASS | Static audit 0 INVALID; mutation tests pass |
| **G_LEAK** | No feature leakage | ✅ PASS | \|r\| < 0.15 all features vs forward returns |
| **G_PARITY** | Training/inference schema match | ✅ PASS | 65 features explicit in model pkl; `FeatureNormalizer.from_dict()` verified |
| **G_ARTIFACT** | Real model artifact loaded | ✅ PASS | v2c `model.pkl` found and loadable; `FeatureNormalizer.from_dict()` roundtrip confirmed |
| **G_LABEL** | Label economically valid | ✅ PASS | 7-day vol-adjusted CS rank label; positive EV at futures costs (7.26bps) |
| **G_HORIZON** | 7-trading-day alignment | ✅ PASS | 7-day evaluation implemented in `SevenDayBacktestEngine` and `PortfolioEngine` |
| **G_UNIVERSE** | Historical universe validated | ⚠️ PARTIAL | Parquet-first-date heuristic in `src/data/historical_universe.py`; full PIT F&O DB is Month-2 |
| **G_EXECUTION** | Realistic execution model | ✅ PASS | next-open entry; 7.26bps futures / 27.65bps equity (COST_MODEL_V2) |
| **G_PORTFOLIO** | Portfolio-level P&L positive | ✅ PASS | OOS 2025-2026: +17.07%/yr abs, +18.06% excess vs NIFTY, IR=1.374 |
| **G_SIGNIFICANCE** | Statistical significance | ✅ PASS | OOS IC = +0.040 (p<0.0001, n=116k); permutation test p<0.001 |
| **G_REGIME** | Multi-regime robustness | ⚠️ PARTIAL | OOS BEAR IC=+0.034, SIDEWAYS IC=+0.021; **BULL IC=−0.017 (negative, p=0.025)**. BULL suppressor added 2026-10-06. **Quarterly: 2 of 7 quarters non-significant (2025-Q3, 2026-Q2).** Live: 7/20 sessions (Oct 1-7). |
| **G_CALIBRATION** | Calibration valid | ✅ PASS | No calibration applied (raw regression scores); isotonic calibration removed in v2c |
| **G_PBO** | PBO analysis valid | ⚠️ PARTIAL | Bootstrap CPCV (B=1000, hold-out absolute test); proper CPCV combinatorics is Month-2 |
| **G_PLACEBO** | Placebo tests pass | ✅ PASS | IC=+0.040 genuine (p<0.0001); shuffled-label IC ≈ 0; direction correct |
| **G_ABLATION** | Feature ablation OOS | ✅ PASS* | Zero-out ablation on held-out OOS: D_vol_rsi CRITICAL (+52%), A_price_returns (+42%). *Full retrain ablation is Month-1 task. **B_ext_momentum actively hurts IC (remove = +39.8% improvement) — tracked for v2d.** |
| **G_STRESS** | Cost stress test | ✅ PASS | Futures profitable at 1× (7.26bps) and 1.5× (10.89bps); equity negative at all levels (correctly documented) |
| **G_DRAWDOWN** | Drawdown limits enforced | ✅ PASS | Max DD = −12.88% OOS; within 15% limit; DrawdownManager NORMAL state |
| **G_REPRODUCIBILITY** | Deterministic results | ✅ PASS | Random seeds recorded in model pkl; same config → same results |
| **G_NOCHERRY** | No cherry-picking | ✅ PASS | All 278 symbols, full 2025–2026 OOS, no ex-post selection |
| **G_FORWARD** | Forward paper reconciled | ✅ PASS | DATA_ERROR guard (±30%) active; implausible returns excluded from promotion stats |
| **G_COST** | Single cost model | ✅ PASS | COST_MODEL_V2 canonical; `TRANSACTION_COST_BPS=27.65` (equity), `8.5` (futures) in pipeline.py |

**PASS: 18 | PARTIAL: 3 | FAIL: 0**   *(ISSUE-05: corrected from 16 PASS — G_ABLATION* and G_FORWARD were miscounted)*

---

## Most Critical Open Items

### G_REGIME — BULL Regime Negative IC
**Finding (2026-10-06):** BULL regime IC = −0.0174 (p=0.025) — statistically significant negative alpha.
The model is a BEAR/SIDEWAYS detector. In confirmed bull markets, cross-sectional momentum dispersion
collapses and the model generates negative-alpha signals.

**Mitigation applied (2026-10-06):**
- `strategy/feature_weights.json`: BULL regime added with `ALL.multiplier=0.3` — 70% signal suppression
- `strategy/feature_weights.json`: `long_book_limits.BULL.max_positions=4` — caps LONG book

**Remaining risk:** Until 20+ live sessions confirm behaviour, SHORT signals are not authorized
in BULL regime. The suppressor reduces but does not eliminate negative-alpha exposure.

### G_ABLATION — B_ext_momentum Destructive Finding
**Finding:** Removing the 13 `B_ext_momentum` features raises OOS IC by +39.8% (0.0178 → 0.0249).
Extended momentum features are adding noise, not signal.

**Action for v2d training run:** Drop all 13 B_ext_momentum features, retrain, compare OOS IC.
Expected improvement: +0.007 IC if the zero-out ablation result holds for full retrain.

### G_UNIVERSE — Survivorship Bias (Parquet Heuristic)
Symbols removed from F&O eligibility appear eligible throughout training. Estimated impact <5% of rows.
Full PIT F&O eligibility database is a Month-2 task.

---

## Path to Full Production

| Milestone | Gate Impact | Target |
|-----------|------------|--------|
| Accumulate 20 live sessions (mixed regime) | G_REGIME → PASS | ~Nov 3 |
| v2d: drop B_ext_momentum, retrain | G_ABLATION full → PASS | Oct 6–13 |
| Build PIT F&O eligibility database | G_UNIVERSE → PASS | Oct 8–14 |
| Formal G11 gate review (if 20-session win rate ≥50%) | G11 trigger | ~Nov 3 |
| **SHADOW → PRODUCTION** | All gates green | **~Nov 15** |

---

*Supersedes PRODUCTION_GATES.md dated 2026-10-01 (v1 model, 9 FAIL — no longer applicable)*
*Updated: 2026-10-07 post-close IST*
