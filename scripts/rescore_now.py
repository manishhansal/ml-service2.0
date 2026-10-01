#!/usr/bin/env python3
"""
rescore_now.py — Fresh off-session scoring pass.

Loads the SHADOW LightGBM model and scores all 218 symbols using the
most recent bars in each parquet.  Writes a fresh latest_scores.json
so AlphaForge shows current signals (not the stale seeded values).

Usage:
    PYTHONPATH=. python3 scripts/rescore_now.py
    make rescore        # (added to Makefile)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)
os.environ["LOG_LEVEL"] = "ERROR"

import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
PARQUET_DIR  = BASE / "data" / "1d" / "1d"
SESSION_DIR  = BASE / "artifacts" / "live_session"
OUT_PATH     = SESSION_DIR / "latest_scores.json"

SESSION_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE))

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    for pattern in ["artifacts/expanded_lgbm/*/model.pkl",
                    "artifacts/registry/stage_a_1d/*/model.pkl"]:
        paths = sorted(BASE.glob(pattern))
        if paths:
            with open(paths[-1], "rb") as f:
                p = pickle.load(f)

            # Reconstruct FeatureNormalizer from its serialised state dict.
            # The orchestrator saves normalizer_state as a plain dict via
            # FeatureNormalizer.to_dict(); we must call from_dict() to restore
            # the fitted object before calling .transform().
            normalizer = None
            raw_state = p.get("normalizer_state")
            if isinstance(raw_state, dict) and raw_state:
                try:
                    from src.features.normalizer import FeatureNormalizer
                    normalizer = FeatureNormalizer.from_dict(raw_state)
                except Exception as e:
                    print(f"[warn] Could not reconstruct FeatureNormalizer: {e}", file=sys.stderr)
            elif raw_state is not None:
                # Already a live object (older pickle format)
                normalizer = raw_state

            return (p["estimator"], p["feature_names"],
                    normalizer, p.get("feature_schema_version", "?"))
    return None, [], None, "?"


# ── Score one symbol ──────────────────────────────────────────────────────────
def score_symbol(sym: str, estimator, feat_names: list, normalizer=None) -> dict | None:
    pf = PARQUET_DIR / f"{sym}.parquet"
    if not pf.exists():
        return None
    try:
        from src.features.expanded_factory import ExpandedFeatureFactory
        from src.features.factory import FeatureFactory

        df = pd.read_parquet(pf)
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        factory = ExpandedFeatureFactory() if len(feat_names) > 24 else FeatureFactory()
        features, _ = factory.build(df)
        if features.empty:
            return None

        last = features.iloc[-1].fillna(0)
        X = np.array([[last.get(c, 0.0) for c in feat_names]])

        if normalizer is not None:
            try:
                X_df = pd.DataFrame(X, columns=feat_names)
                X = normalizer.transform(X_df).to_numpy(dtype=float)
            except Exception:
                pass

        score = float(estimator.predict(X)[0])
        return {
            "symbol":    sym,
            "score":     round(score, 4),
            "direction": 1 if score >= 0.5 else -1,
            "data_date": str(df.index.max().date()),
        }
    except Exception:
        return None


# ── Conviction grade ──────────────────────────────────────────────────────────
def conviction(score: float) -> str:
    d = abs(score - 0.5)
    if d >= 0.40: return "S"
    if d >= 0.30: return "A"
    if d >= 0.20: return "B"
    if d >= 0.10: return "C"
    return "D"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  RESCORE NOW — fresh parquet data")
    print(f"  {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    estimator, feat_names, normalizer, schema = load_model()
    if estimator is None:
        print("ERROR: no model found in artifacts/")
        sys.exit(1)
    print(f"\n  Model loaded: schema={schema}  features={len(feat_names)}")

    symbols = sorted(pf.stem for pf in PARQUET_DIR.glob("*.parquet"))
    print(f"  Symbols to score: {len(symbols)}\n")

    scores: list[dict] = []
    failed = 0
    for i, sym in enumerate(symbols):
        r = score_symbol(sym, estimator, feat_names, normalizer)
        if r:
            scores.append(r)
        else:
            failed += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1:3d}/{len(symbols)}] scored={len(scores)}")

    print(f"\n  Scored: {len(scores)}  Failed/skipped: {failed}")

    # Rank + grade
    scores.sort(key=lambda s: abs(s["score"] - 0.5), reverse=True)
    for i, s in enumerate(scores):
        s["rank"] = i + 1
        s["conviction"] = conviction(s["score"])

    n_long  = sum(1 for s in scores if s["direction"] == 1)
    n_short = sum(1 for s in scores if s["direction"] == -1)

    # Show top signals
    print(f"\n  n_long={n_long}  n_short={n_short}")
    print("\n  Top 5 LONG:")
    for s in [x for x in scores if x["direction"] == 1][:5]:
        print(f"    #{s['rank']:3d} {s['symbol']:20s} score={s['score']:.4f} grade={s['conviction']} data={s['data_date']}")
    print("\n  Top 5 SHORT:")
    for s in [x for x in scores if x["direction"] == -1][:5]:
        print(f"    #{s['rank']:3d} {s['symbol']:20s} score={s['score']:.4f} grade={s['conviction']} data={s['data_date']}")

    # Data date distribution
    from collections import Counter
    dates = Counter(s["data_date"] for s in scores)
    print(f"\n  Data date distribution:")
    for d, n in sorted(dates.items(), reverse=True):
        print(f"    {d}: {n} symbols")

    # Write snapshot
    snapshot = {
        "session_date":  datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "generated_at":  datetime.now(tz=timezone.utc).isoformat(),
        "market_open":   False,
        "model_version": schema,
        "n_scored":      len(scores),
        "n_long":        n_long,
        "n_short":       n_short,
        "nifty_chg":     None,
        "nifty_ltp":     None,
        "session_pnl":   {"mean_net": None, "win_rate": None, "n_positions": 0},
        "signals":       scores,
        "stale":         False,
    }
    tmp = OUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(snapshot, default=str))
    tmp.replace(OUT_PATH)

    print(f"\n  Written → {OUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
