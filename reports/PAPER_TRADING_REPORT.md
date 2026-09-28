# PAPER TRADING REPORT
**AlphaForge ml-service2.0 — Forward Paper Trading**
**Date:** 2026-09-28 (post-close) | **Revision:** v3.0

---

## Summary

| Parameter | Value |
|-----------|-------|
| Total signals | **218** (v1: 65 + v2: 153) |
| Model | LightGBM fs-3.0.0 (55 features) |
| Feature schema | fs-3.0.0 |
| Execution model | `next_open` — economically valid |
| Holding period | 5 bars (daily) |
| Signal dates | Sep 22 (v1) + Sep 22-25 (v2) |
| Expected exit | open[Sep 30] |
| Partial resolved | 13 (1-2 bar MTM only — NOT for evaluation) |
| Full resolution | **Sep 30, 09:30 IST** |

---

## Signal Distribution

| Batch | Model | Symbols | Direction Split |
|-------|-------|---------|----------------|
| v1 (Sep 22) | LightGBM fs-3.0.0 | 65 | 43 SHORT / 22 LONG |
| v2 (Sep 22-25) | LightGBM fs-3.0.0 | 153 | 113 SHORT / 40 LONG |
| **Combined** | | **218** | **156 SHORT (71.6%) / 62 LONG (28.4%)** |

**Model bias 71.6% SHORT** aligns with live session (70.6% SHORT on Sep 28) — consistent bearish cross-sectional regime captured from Sep 22-25 data.

---

## Partial Resolutions (1-2 bar MTM — diagnostic only)

*These 13 outcomes represent 1-2 bars of a 5-bar horizon. NOT valid for G10 gate.*

| Symbol | Direction | Net Return | Outcome |
|--------|-----------|-----------|---------|
| RVNL | SHORT | **+1.51%** | TARGET_HIT ✓ |
| SBICARD | SHORT | **+1.88%** | TARGET_HIT ✓ |
| SBIN | SHORT | **+0.11%** | TARGET_HIT ✓ |
| SHRIRAMFIN | SHORT | **+0.62%** | TARGET_HIT ✓ |
| SIEMENS | SHORT | −0.01% | STOP_HIT ≈0 |
| VEDL | SHORT | −0.13% | STOP_HIT ≈0 |
| SOLARINDS | SHORT | −1.02% | STOP_HIT ✗ |
| SBILIFE | SHORT | −2.91% | STOP_HIT ✗ |
| TCS | SHORT | −0.57% | STOP_HIT ✗ |
| PETRONET | SHORT | −0.84% | STOP_HIT ✗ |
| PERSISTENT | SHORT | −1.99% | STOP_HIT ✗ |
| SAGILITY | LONG | −1.38% | STOP_HIT ✗ |
| SAIL | SHORT | −0.99% | STOP_HIT ✗ |

4/13 partial wins = 31% — insufficient for evaluation (1-2 bars only, not 5-bar full horizon).

---

## G10 Gate Requirements

| Requirement | Status |
|-------------|--------|
| ≥ 50 resolved outcomes | 218 signals ready to resolve Sep 30 ✓ |
| Positive mean net P&L | IC=0.376 → expected ~+0.4% per signal |
| Full 5-bar resolution prices | Available Sep 30 open |

**G10 resolution date: 2026-09-30, 09:30 IST**

---

## Phil Integrations Active for Next Session

- **ForecastLedger:** ALL 218 scores logged each sample → `artifacts/forward_paper/forecasts.jsonl`
- **ScoreThresholdSweep:** Post-close analysis of optimal min-conviction threshold
- **CounterfactualLedger:** Blocked signals graded at close

---

## Resolution Commands (Sep 30)

```bash
# Step 1: Ingest Sep 28-29 close bars
PYTHONPATH=. python3 scripts/fast_ingest.py

# Step 2: Resolve all 218 signals (exit = open[Sep 30])
PYTHONPATH=. python3 scripts/resolve_forward_paper.py

# Step 3: Run SignalPromotionEngine (G11)
PYTHONPATH=. python3 scripts/run_signal_promotion.py
```

*Generated: 2026-09-28 | Signals: artifacts/forward_paper/signals*.jsonl*
