# FEATURE CERTIFICATION REPORT
**AlphaForge ml-service2.0 — Feature Engineering Certification**
**Date:** 2026-09-28 (post-close) | **Status:** CERTIFIED (fs-3.0.0) | **Revision:** v3.0

---

## Feature Schema Registry

| Schema | Features | File | Status |
|--------|----------|------|--------|
| fs-2.0.0 (FeatureFactory) | 24 | `src/features/factory.py` | DEPRECATED — superseded |
| **fs-3.0.0 (ExpandedFeatureFactory)** | **55** | `src/features/expanded_factory.py` | **CERTIFIED — IC-validated, PIT-correct** |

**fs-3.0.0 is the only supported schema for production and new training runs.**

---

## fs-3.0.0 Feature Groups (55 total)

| Group | Count | Description |
|-------|-------|-------------|
| Base OHLCV momentum | 24 | Original features (ret_1..20, vol_5..20, RSI, MACD, Bollinger, ATR, OBV, VWAP-spread) |
| Extended momentum | 5 | ret_3, ret_60, mom_accel_5/20, ret_60_rel_vol |
| Regime features | 11 | vol_regime_zscore/pctile/expanding/ratio, trend_strength/direction/ema_spread/persistence, gap_magnitude/regime_rolling/direction |
| Time-context features | 7 | weekday/sin/cos, month_end_proximity, quarter_end, is_monday, is_friday |
| Relative value | 4 | price_zscore_60/20, vol_norm_ret_5/20 |
| Advanced volatility | 4 | parkinson_vol, garman_klass_vol, vol_of_vol_20, atr_zscore |

---

## PIT Certification

**Point-in-time safety verified** — no future data leakage in any feature:

| Test | Result |
|------|--------|
| Static shift analysis (CI) | 0 INVALID (checked every PR) |
| LeakageValidator correlation | All features < 0.95 threshold |
| LookAheadGuard wired | All inference paths protected |
| close-to-close dataset marked | `is_economic_evidence=False` — not used for champion selection |

---

## IC Validation (fs-3.0.0 vs fs-2.0.0)

| Schema | IC_continuous | IC_rank | Notes |
|--------|---------------|---------|-------|
| fs-2.0.0 (24 features) | 0.3088 | 0.31 | Residual bid-ask bounce artifact |
| **fs-3.0.0 (55 features)** | **0.3757** | **0.4136** | **No artifact; 0.9× IC inflation factor** |

**+0.067 IC improvement** from 24 → 55 features. The regime and time-context groups provide the most incremental predictive power.

---

## Phil Integration: ForecastLedger (Calibration Enhancement)

From Sep 28, ALL 218 symbol scores are logged to `artifacts/forward_paper/forecasts.jsonl` after every scoring run.

| Metric | Before | After |
|--------|--------|-------|
| Calibration data points/year | 1,352 (26 tracked positions × 52 weeks) | **54,936** (218 × 252) |
| Calibration resolution | ~3 weeks between updates | **Daily** |
| Brier delta per symbol | Not available | **Available after each session** |

The `ForecastLedger.brier_report()` method now provides per-symbol calibration diagnostics, identifying which symbols consistently beat market-implied probability (true alpha generators).

---

## Feature Weight Manager (Runtime Adjustments)

`strategy/feature_weights.json` provides agent-editable runtime signal filters without model retraining:

| Filter | Effect |
|--------|--------|
| sector_regime_filters.HIGH_CORR_BEAR.PHARMA | Dim SHORT conviction to 20% in risk-off |
| sector_regime_filters.HIGH_CORR_BEAR.AUTO | Dim SHORT conviction to 40% |
| long_book_limits.HIGH_CORR_BEAR.max_positions | Cap LONG count at 3 in bear days |
| score_threshold.current | Min distance from 0.50 to open a position |

*Generated: 2026-09-28 | Factory: ExpandedFeatureFactory | Artifact: 1.0.0-20260928053134956099*
