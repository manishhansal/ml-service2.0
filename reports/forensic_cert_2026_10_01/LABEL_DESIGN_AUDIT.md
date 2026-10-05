# LABEL DESIGN AUDIT
**Repository:** ml-service2.0 | **Date:** 2026-10-01

---

## Current Label (ls-2.0.0) — FAILS

```
Type:            triple_barrier
Horizon:         5 bars (≈ 5 trading days)
Upper barrier:   +2% (TARGET_HIT → label=1)
Lower barrier:   −2% (STOP_HIT → label=0)
TIME_EXPIRY:     label = int(realized_return > 0)  ← WRONG: ignores cost
Execution:       next_open (entry at open[T+1])     ✓
Cost in labels:  27.65 bps                          ✓
Direction:       LONG-only semantics (barriers symmetric)  ✗ WRONG for SHORT
```

**Why it fails:**
1. Mean net return = −0.31%/trade (negative EV at equity costs)
2. Break-even win rate = 56.9% (model achieves 47% — 10pp gap)
3. TIME_EXPIRY threshold ignores cost (small positive return labels as +1 even if net < 0)
4. 5-bar horizon ≠ 7-trading-day mandate
5. Long-only semantics: SHORT signals are labeled with LONG barrier outcomes (semantically wrong)

---

## Label A — 7-Day Net Return (implemented in `src/labels/seven_day.py`)

```python
from src.labels.seven_day import generate_7d_excess_return_label

label = generate_7d_excess_return_label(
    stock_df=df,
    nifty_close=nifty_close,
    horizon=7,
)
# Returns vol-adjusted excess return vs NIFTY, clipped to [-5, 5]
# PIT-safe: vol uses only data ≤ T
# Label value at T uses close[T+7] — genuinely future (correct for labels)
```

**Economics:** Continuous label; positive label = outperforms NIFTY = cross-sectional alpha. No barrier calibration needed.

---

## Label B — 7-Day Asymmetric Barrier (implemented in `src/labels/seven_day.py`)

```python
from src.labels.seven_day import generate_7d_asymmetric_barrier_label

label_df = generate_7d_asymmetric_barrier_label(
    stock_df=df,
    cost_bps=27.35,   # COST_MODEL_V2 equity
    horizon=7,
    target_multiplier=1.5,   # target = vol × 1.5 × √7
    stop_multiplier=0.75,    # stop   = vol × 0.75 × √7 → 2:1 R:R
)
```

**Economics:** 2:1 R:R, break-even = ~35% win rate (vs 56.9% old). TIME_EXPIRY uses cost-adjusted threshold.

---

## Label C — Side-Aware LONG/SHORT

The current implementation treats all observations as LONG. A SHORT signal should have:
- Target: price FALLS below entry × (1 − target_pct)
- Stop: price RISES above entry × (1 + stop_pct)

Implementation gap in `src/data/labels.py`: the `LabelFactory._triple_barrier()` uses the same barrier logic regardless of intended direction. Fix:

```python
def _triple_barrier(self, ohlcv: pd.DataFrame, side: int = 1) -> pd.DataFrame:
    # side = +1 (LONG): target = entry*(1+up), stop = entry*(1-dn)
    # side = -1 (SHORT): target = entry*(1-up), stop = entry*(1+dn)
    if side == 1:
        target_lvl = entry * (1 + up_pct)
        stop_lvl   = entry * (1 - dn_pct)
    else:
        target_lvl = entry * (1 - up_pct)
        stop_lvl   = entry * (1 + dn_pct)
```

This is NOT yet implemented and must be fixed before training a model that generates both LONG and SHORT signals.

---

## Label D — Separate LONG and SHORT Datasets

Until side-aware barriers are implemented, generate two separate datasets:
1. LONG dataset: barriers as currently implemented
2. SHORT dataset: inverted barriers (separately labeled, separately trained)

This avoids the semantic mismatch where a stock that fell (SHORT profitable) gets label=0 (STOP_HIT for LONG).
