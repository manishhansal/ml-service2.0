"""
train_v2.py — Sprint 1B: Retrain with all forensic fixes applied.

What this fixes vs the existing shadow model (expanded_lgbm v1):
  FIX-01  Label: 5-bar ±2% triple-barrier → 7-day vol-adjusted excess return vs NIFTY
  FIX-02  Model: LGBMClassifier → LGBMRegressor (regression; optimizes IC directly)
  FIX-03  Calib: IsotonicWrapper → NONE (raw scores; no variance destruction)
  FIX-04  Split: Walk-forward within training → hard 2025+ holdout
  FIX-05  Norm:  No normalizer at inference → fit on train, save in artifact
  FIX-06  Dir:   score > 0.5 → cross-sectional rank (top/bottom 20%)
  FIX-07  Cost:  10 bps → 27.35 bps (COST_MODEL_V2)
  FIX-08  CV:    5-fold WF on all data → PurgedKFold on train only

Usage:
    PYTHONPATH=. python3 scripts/train_v2.py
    PYTHONPATH=. python3 scripts/train_v2.py --dataset artifacts/datasets/ds-1d-.../data.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.training.v2_pipeline import V2TrainingConfig, V2TrainingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v2 model with all forensic fixes")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to pre-built dataset parquet (uses most recent if not specified)"
    )
    parser.add_argument(
        "--parquet-dir", type=str, default=str(ROOT / "data" / "1d" / "1d"),
        help="Path to raw OHLCV parquets (used if --dataset not specified)"
    )
    parser.add_argument("--train-end",  default="2023-12-31")
    parser.add_argument("--val-end",    default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--horizon",    type=int, default=7)
    parser.add_argument("--estimators", type=int, default=400)
    args = parser.parse_args()

    print("="*65)
    print("  V2 TRAINING PIPELINE — ALL FORENSIC FIXES APPLIED")
    print("="*65)

    cfg = V2TrainingConfig(
        train_end    = args.train_end,
        val_end      = args.val_end,
        test_start   = args.test_start,
        label_horizon = args.horizon,
    )
    cfg.lgbm_params["n_estimators"] = args.estimators

    pipeline = V2TrainingPipeline(cfg)

    # Determine data source
    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        # Find most recent dataset
        datasets = sorted(ROOT.glob("artifacts/datasets/ds-1d-*/data.parquet"))
        if datasets:
            dataset_path = datasets[-1]
            print(f"  Using latest dataset: {dataset_path.parent.name}")
        else:
            dataset_path = None
            print(f"  No pre-built dataset found; loading from parquets: {args.parquet_dir}")

    try:
        if dataset_path and dataset_path.exists():
            result = pipeline.run(dataset_parquet=str(dataset_path))
        else:
            result = pipeline.run(parquet_dir=args.parquet_dir)
    except Exception as exc:
        print(f"\n✗ Training failed: {exc}")
        raise

    # Print final summary
    print("\n" + "="*65)
    print("  TRAINING RESULT SUMMARY")
    print("="*65)
    print(f"  Model version:  {result.model_version}")
    print(f"  CV IC:          {result.cv_ic_mean:.4f} ± {result.cv_ic_std:.4f}")
    print(f"  Val  IC:        {result.val_ic:.4f}")
    print(f"  Test IC (OOS):  {result.test_ic:.4f}  p={result.test_ic_pval:.4f}")
    print(f"  Test win rate:  {result.test_win_rate:.1%}")
    print(f"  Test EV/trade:  {result.test_ev*100:.3f}%")
    print(f"  Artifact:       {result.artifact_path}")
    print()
    profitable = result.test_ev > 0 and result.test_win_rate > 0.50
    if profitable:
        print("  ✓ PROFITABLE: Test EV > 0 and win rate > 50%")
    else:
        print("  ✗ NOT PROFITABLE — proceeding to Sprint 2 (feature engineering)")
        if result.test_ic < 0.005:
            print("    → Test IC < 0.005: model has insufficient signal")
            print("    → Sprint 2A: add cross-sectional and sector-relative features")
        elif result.test_win_rate <= 0.50:
            print(f"    → Win rate {result.test_win_rate:.1%} ≤ 50%: directional skill below random")
            print("    → Sprint 2B: regime-conditional signal filtering")
    print("="*65)


if __name__ == "__main__":
    main()
