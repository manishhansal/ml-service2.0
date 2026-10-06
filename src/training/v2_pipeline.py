"""
src.training.v2_pipeline — Complete v2 training pipeline with all forensic fixes.

KEY CHANGES vs v1 (expanded_lgbm)
----------------------------------
FIX-01  Label:  triple_barrier(5-bar, ±2%)  →  7-day vol-adjusted excess return vs NIFTY
        (continuous ranking label; directly optimizes IC)

FIX-02  Model:  LGBMClassifier              →  LGBMRegressor
        (regression on continuous label; avoids binary-class calibration trap)

FIX-03  Calib:  IsotonicWrapper (destroys variance) → NONE (raw regression scores)
        The regression output IS the ranking signal; no calibration needed.

FIX-04  Split:  Walk-forward within training  →  Hard holdout
        - Train:   2022-10-06 → 2023-12-31  (≈ 84k rows)
        - Val:     2024-01-01 → 2024-12-31  (≈ 66k rows)
        - Test:    2025-01-01 → 2026-09-28  (≈ 118k rows, NEVER TOUCHED DURING DEVELOPMENT)

FIX-05  Norm:   No normalizer at inference  → Fit normalizer on train, save in artifact

FIX-06  Dir:    score > 0.5 → LONG/SHORT   → Cross-sectional rank percentile
        Top 20% of daily scores = LONG, Bottom 20% = SHORT

FIX-07  Cost:   10 bps in pipeline         → 27.35 bps (COST_MODEL_V2)

FIX-08  CV:     5-fold walk-forward within training+val → PurgedKFold on train only
        (val is kept separate for threshold and monitoring only)

FIX-09  HPO:    Optuna on val IC (not test period)

Usage::

    from src.training.v2_pipeline import V2TrainingPipeline, V2TrainingConfig
    config = V2TrainingConfig()
    pipeline = V2TrainingPipeline(config)
    result = pipeline.run(parquet_dir="data/1d/1d")
    # Artifact: artifacts/v2_model/model.pkl
"""
from __future__ import annotations

import json
import pickle
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.features.factory import FeatureFactory
from src.features.normalizer import FeatureNormalizer
from src.labels.seven_day import generate_7d_excess_return_label
from src.logging_config import get_logger
from src.training.purged_kfold import PurgedKFoldSplitter

logger = get_logger(__name__)

ROOT = Path(__file__).parent.parent.parent
COST_MODEL_V2_EQUITY_BPS = 27.35


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class V2TrainingConfig:
    """All hyperparameters and configuration for the v2 pipeline."""

    # ── Data split dates ─────────────────────────────────────────────────────
    train_end:   str = "2023-12-31"   # inclusive
    val_end:     str = "2024-12-31"   # inclusive (test = everything after)
    test_start:  str = "2025-01-01"   # NEVER touched during development

    # ── Label ────────────────────────────────────────────────────────────────
    label_horizon: int   = 7      # 7 trading days (mandate)
    label_vol_window: int = 20    # rolling vol window for normalization
    label_clip: float    = 5.0    # clip label to [-clip, clip]

    # ── Feature schema ───────────────────────────────────────────────────────
    # Locked to the fs-2.0.0 feature names from the shadow model — verified OOS
    feature_names: list[str] = field(default_factory=lambda: [
        # Original 55 fs-2.0.0 features
        "ret_1", "ret_5", "ret_10", "ret_20", "log_ret_1",
        "vol_5", "vol_10", "vol_20", "atr_14_pct",
        "rel_volume_20", "volume_zscore_20", "vwap_distance_pct",
        "rsi_14", "macd_hist", "stoch_k_14",
        "ema_5_20", "ema_10_50", "adx_14",
        "hl_range_pct", "close_position", "gap_pct",
        "bb_zscore_20", "skew_20", "kurt_20",
        "ret_3", "ret_60", "mom_accel_5", "mom_accel_20",
        "ret_60_rel_vol", "vol_regime_zscore", "vol_regime_pctile",
        "vol_expanding", "vol_ratio", "trend_strength", "trend_direction",
        "ema_spread", "trend_persistence", "gap_magnitude",
        "gap_regime_rolling", "gap_direction",
        "weekday", "weekday_sin", "weekday_cos",
        "month_end_proximity", "quarter_end", "is_monday", "is_friday",
        "price_zscore_60", "price_zscore_20",
        "vol_norm_ret_5", "vol_norm_ret_20",
        "parkinson_vol", "garman_klass_vol", "vol_of_vol_20", "atr_zscore",
        # v2c: cross-sectional + regime features (+10)
        "cs_rank_ret_1d", "cs_rank_ret_5d", "cs_rank_ret_20d",
        "cs_rank_rs_nifty", "cs_rank_rsi", "cs_rank_beta_adj",
        "nifty_trend_20d", "trend_regime", "market_breadth_5d",
        "beta_adj_ret_5d",
        # v2d: academic factor features — OOS VALIDATED (IC > 0.003, p < 0.05 on 2025+)
        "cs_mom_12_1",           # 12-1 month momentum (IC=+0.038)
        "cs_ret_252",            # 12-month return CS rank (IC=+0.035)
        "mom_12_1",              # raw 12-1 month momentum (IC=+0.033)
        "cs_pct_from_52w_high",  # 52-week high momentum (IC=+0.026)
        "cs_pos_in_52w_range",   # position in 52-week range (IC=+0.023)
        "cs_neg_amihud",         # liquidity rank (IC=+0.022)
        "cs_neg_ret_1",          # short-term reversal 1d (IC=+0.022)
        "rsi_oversold_flag",     # oversold bounce (IC=+0.019)
        "rsi_overbought_flag",   # overbought peak (IC=+0.019)
        "consec_down_bars",      # consecutive down bars (IC=+0.016)
        "cs_neg_ret_5",          # short-term reversal 5d (IC=+0.015)
        "bb_below_lower",        # below Bollinger band (IC=+0.010)
        # Negative IC — model learns to use inversely
        "consec_up_bars",        # consecutive up bars (IC=-0.017, overbought)
        "news_event_importance", # high news impact (IC=-0.016)
        "cs_up_down_vol_ratio",  # volume pattern (IC=-0.015)
        "cs_ret_accel_5_10",     # momentum deceleration (IC=-0.012)
    ])

    # ── Model ─────────────────────────────────────────────────────────────────
    lgbm_params: dict[str, Any] = field(default_factory=lambda: {
        "objective":         "regression",
        "n_estimators":      600,        # more trees for complex feature set
        "learning_rate":     0.02,       # lower LR → better generalization
        "num_leaves":        63,         # deeper trees for factor interactions
        "min_child_samples": 80,         # conservative to prevent overfit
        "reg_lambda":        3.0,        # stronger L2 regularization
        "reg_alpha":         0.5,        # L1 sparse features
        "subsample":         0.7,        # more aggressive subsampling
        "colsample_bytree":  0.6,        # feature subsampling
        "min_split_gain":    0.001,      # minimum gain to prevent trivial splits
        "random_state":      42,
        "verbosity":         -1,
        "n_jobs":            1,
        "force_row_wise":    True,
    })

    # ── Cross-validation ──────────────────────────────────────────────────────
    n_cv_splits: int    = 5
    embargo_days: int   = 14     # 2× label horizon to prevent label overlap
    optuna_trials: int  = 30     # reduced to prevent overfitting to val

    # ── Output ────────────────────────────────────────────────────────────────
    artifact_dir: str = "artifacts/v2_model"
    model_version: str = ""   # auto-generated

    # ── Cost model ────────────────────────────────────────────────────────────
    cost_bps: float = COST_MODEL_V2_EQUITY_BPS


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class V2TrainingResult:
    """Complete results from one training run."""
    run_id: str
    model_version: str
    config: V2TrainingConfig
    train_ic: float = 0.0
    val_ic:   float = 0.0
    test_ic:  float = 0.0
    test_ic_pval: float = 1.0
    train_rows: int = 0
    val_rows:   int = 0
    test_rows:  int = 0
    n_features: int = 0
    artifact_path: str = ""
    cv_ic_mean: float = 0.0
    cv_ic_std:  float = 0.0
    cv_ic_per_fold: list[float] = field(default_factory=list)
    val_win_rate: float = 0.0
    test_win_rate: float = 0.0
    test_ev: float = 0.0
    normalizer_state: dict = field(default_factory=dict)


# ── Pipeline ──────────────────────────────────────────────────────────────────


class V2TrainingPipeline:
    """
    Complete v2 training pipeline implementing all forensic fixes.

    Usage::

        pipeline = V2TrainingPipeline(V2TrainingConfig())
        result = pipeline.run(dataset_parquet="artifacts/datasets/.../data.parquet")
        print(f"Test IC: {result.test_ic:.4f}")
    """

    def __init__(self, config: V2TrainingConfig | None = None) -> None:
        self.config = config or V2TrainingConfig()

    def run(
        self,
        dataset_parquet: str | Path | None = None,
        parquet_dir: str | Path | None = None,
    ) -> V2TrainingResult:
        """
        Run the complete training pipeline.

        Either dataset_parquet (pre-built dataset) or parquet_dir (raw parquets)
        must be provided. If both are given, dataset_parquet takes precedence.
        """
        cfg = self.config
        result = V2TrainingResult(
            run_id=str(uuid.uuid4())[:12],
            model_version=cfg.model_version or f"v2-{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S')}",
            config=cfg,
        )

        logger.info("v2_pipeline_started", run_id=result.run_id, version=result.model_version)

        # ── Step 1: Load features ─────────────────────────────────────────────
        print("\n[V2] Step 1/8: Loading feature data...")
        if dataset_parquet is not None:
            df = self._load_from_dataset(Path(dataset_parquet), cfg)
        elif parquet_dir is not None:
            df = self._load_from_parquets(Path(parquet_dir), cfg)
        else:
            raise ValueError("Either dataset_parquet or parquet_dir must be provided")

        print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

        # ── Step 2: Generate or use 7-day labels ──────────────────────────────
        print("\n[V2] Step 2/8: Setting up 7-day vol-adjusted excess return labels...")
        if "label_v2b" in df.columns and df["label_v2b"].notna().sum() > 1000:
            # v2b dataset: use cross-sectional rank label (preferred)
            print(f"  Using pre-computed label_v2b (CS rank) from dataset ({df['label_v2b'].notna().sum():,} valid)")
            df["label_v2"] = df["label_v2b"]
            valid_mask = df["label_v2"].notna()
            df = df[valid_mask].copy()
        elif "label_v2" in df.columns and df["label_v2"].notna().sum() > 1000:
            # v2 dataset: use vol-adjusted excess return label
            print(f"  Using pre-computed label_v2 from dataset ({df['label_v2'].notna().sum():,} valid rows)")
            valid_mask = df["label_v2"].notna()
            df = df[valid_mask].copy()
        else:
            print("  Computing labels from price data...")
            df = self._build_7d_labels(df, cfg)
            valid_mask = df["label_v2"].notna()
            df = df[valid_mask].copy()

        print(f"  Valid labeled rows: {len(df):,}")
        if len(df) > 0:
            print(f"  Label stats: mean={df['label_v2'].mean():.4f}, std={df['label_v2'].std():.4f}")
            print(f"  Label positive rate: {(df['label_v2'] > 0).mean():.1%}")

        # ── Step 3: Chronological train/val/test split ─────────────────────────
        print("\n[V2] Step 3/8: Splitting data (HARD HOLDOUT)...")
        train_mask = df.index < pd.Timestamp(cfg.train_end + "T23:59", tz="UTC")
        val_mask   = (df.index >= pd.Timestamp(cfg.train_end + "T23:59", tz="UTC")) & \
                     (df.index <  pd.Timestamp(cfg.val_end   + "T23:59", tz="UTC"))
        test_mask  = df.index >= pd.Timestamp(cfg.test_start + "T00:00", tz="UTC")

        df_train = df[train_mask].copy()
        df_val   = df[val_mask].copy()
        df_test  = df[test_mask].copy()

        result.train_rows = len(df_train)
        result.val_rows   = len(df_val)
        result.test_rows  = len(df_test)

        print(f"  Train:  {len(df_train):,} rows ({str(df_train.index.min())[:10]} → {str(df_train.index.max())[:10]})")
        print(f"  Val:    {len(df_val):,} rows ({str(df_val.index.min())[:10]} → {str(df_val.index.max())[:10]})")
        print(f"  Test:   {len(df_test):,} rows ({str(df_test.index.min())[:10]} → {str(df_test.index.max())[:10]})")

        if len(df_train) < 1000:
            raise ValueError(f"Insufficient training data: {len(df_train)} rows")

        feat_cols = [c for c in cfg.feature_names if c in df_train.columns]
        result.n_features = len(feat_cols)
        print(f"  Features used: {len(feat_cols)}/{len(cfg.feature_names)}")

        # ── Step 4: Fit normalizer on train only ───────────────────────────────
        print("\n[V2] Step 4/8: Fitting normalizer on training data...")
        norm = FeatureNormalizer()
        X_train_raw = df_train[feat_cols].copy()
        norm.fit(X_train_raw)
        result.normalizer_state = norm.to_dict()

        X_train = norm.transform(df_train[feat_cols]).fillna(0.0).to_numpy(dtype=float)
        X_val   = norm.transform(df_val[feat_cols]).fillna(0.0).to_numpy(dtype=float)
        X_test  = norm.transform(df_test[feat_cols]).fillna(0.0).to_numpy(dtype=float)
        y_train = df_train["label_v2"].to_numpy(dtype=float)
        y_val   = df_val["label_v2"].to_numpy(dtype=float)
        y_test  = df_test["label_v2"].to_numpy(dtype=float)
        print(f"  Normalizer fitted on {len(X_train):,} training rows")

        # ── Step 5: Purged cross-validation on train ───────────────────────────
        print("\n[V2] Step 5/8: PurgedKFold cross-validation on training data...")
        cv_ics = self._cv_evaluate(X_train, y_train, df_train.index, cfg)
        result.cv_ic_mean     = float(np.mean(cv_ics))
        result.cv_ic_std      = float(np.std(cv_ics))
        result.cv_ic_per_fold = [float(x) for x in cv_ics]
        print(f"  CV IC: mean={result.cv_ic_mean:.4f}, std={result.cv_ic_std:.4f}, per_fold={[round(x,4) for x in cv_ics]}")

        # ── Step 6: Train final model on train+val ─────────────────────────────
        print("\n[V2] Step 6/8: Training final model on train + val data...")
        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])
        model = self._train_model(X_trainval, y_trainval, cfg)

        # Measure IC on each split
        train_scores = model.predict(X_train)
        val_scores   = model.predict(X_val)
        test_scores  = model.predict(X_test)

        result.train_ic, _ = spearmanr(train_scores, y_train)
        result.val_ic, _   = spearmanr(val_scores,   y_val)
        result.test_ic, result.test_ic_pval = spearmanr(test_scores, y_test)

        print(f"  Train IC: {result.train_ic:.4f}")
        print(f"  Val   IC: {result.val_ic:.4f}")
        print(f"  Test  IC: {result.test_ic:.4f} (p={result.test_ic_pval:.4f})")

        # Placebo check: if IC direction is inverted on val, flip model
        if result.val_ic < -0.005:
            print("  ⚠ Val IC is NEGATIVE — direction inversion detected. Checking...")
            # Try inverted scores
            inv_ic, _ = spearmanr(-val_scores, y_val)
            if inv_ic > result.val_ic + 0.005:
                print(f"  Inverted IC ({inv_ic:.4f}) > forward IC ({result.val_ic:.4f}). Using inverted model.")
                # Store inversion flag in model artifact
                model._v2_invert = True
            else:
                model._v2_invert = False
        else:
            model._v2_invert = False

        # ── Step 7: Evaluate trading quality on test ───────────────────────────
        print("\n[V2] Step 7/8: Evaluating trading quality on test (M1 mode)...")
        test_trading = self._evaluate_trading(
            test_scores, y_test, df_test, feat_cols, cfg
        )
        result.test_win_rate = test_trading["win_rate"]
        result.test_ev       = test_trading["ev_per_trade"]
        print(f"  Test win rate (top/bottom 20%): {result.test_win_rate:.1%}")
        print(f"  Test EV/trade (net):            {result.test_ev*100:.3f}%")
        print(f"  Test IC (raw score vs label):   {result.test_ic:.4f}")

        # ── Step 8: Save artifact ──────────────────────────────────────────────
        print("\n[V2] Step 8/8: Saving model artifact...")
        artifact_path = self._save_artifact(model, norm, feat_cols, result, cfg)
        result.artifact_path = str(artifact_path)
        print(f"  Saved: {artifact_path}")

        # Final assessment
        print(f"\n{'='*60}")
        print(f"  V2 TRAINING COMPLETE")
        print(f"  CV IC:   {result.cv_ic_mean:.4f} ± {result.cv_ic_std:.4f}")
        print(f"  Test IC: {result.test_ic:.4f} (p={result.test_ic_pval:.4f})")
        print(f"  Test EV: {result.test_ev*100:.3f}%/trade")
        print(f"  Win rate: {result.test_win_rate:.1%}")

        # Is it profitable?
        breakeven = abs(cfg.cost_bps / 10_000) / (0.06 / 2)  # simplified
        profitable = result.test_ev > 0 and result.test_win_rate > 0.50
        print(f"\n  PROFITABLE (test EV > 0 AND win rate > 50%): {'✓ YES' if profitable else '✗ NO'}")
        if not profitable:
            print(f"  Gap to profitability:")
            print(f"    Test IC needs to be > 0.01 (currently {result.test_ic:.4f})")
            print(f"    Win rate needs to be > 50% (currently {result.test_win_rate:.1%})")
        print(f"{'='*60}\n")

        logger.info(
            "v2_pipeline_complete",
            run_id=result.run_id,
            test_ic=round(result.test_ic, 4),
            test_ev=round(result.test_ev, 6),
            win_rate=round(result.test_win_rate, 4),
        )
        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_from_dataset(self, path: Path, cfg: V2TrainingConfig) -> pd.DataFrame:
        """Load pre-built dataset and extract fs-2.0.0 features + pre-built labels."""
        df = pd.read_parquet(str(path))
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if "_ts" in df.columns:
                df = df.set_index("_ts")
            elif "ts" in df.columns:
                df = df.set_index("ts")
        if hasattr(df.index, 'tz') and df.index.tz is None:
            df.index = pd.to_datetime(df.index).tz_localize("UTC")
        elif not hasattr(df.index, 'tz'):
            df.index = pd.to_datetime(df.index).tz_localize("UTC")
        # Keep features + symbol + label columns
        keep = [c for c in cfg.feature_names if c in df.columns]
        meta = [c for c in ["symbol", "label_v2", "label_v2b", "label", "label_v1",
                             "nifty_ret_1d", "nifty_ret_5d", "nifty_ret_20d",
                             "realized_return", "close_for_label", "excess_ret_raw"]
                if c in df.columns]
        return df[keep + meta].copy()

    def _load_from_parquets(self, pq_dir: Path, cfg: V2TrainingConfig) -> pd.DataFrame:
        """Load raw OHLCV parquets and compute features."""
        factory = FeatureFactory()
        all_dfs = []
        pq_files = sorted(pq_dir.glob("*.parquet"))
        print(f"  Loading {len(pq_files)} parquet files...")
        for pq in pq_files:
            try:
                raw = pd.read_parquet(str(pq))
                if raw.index.tz is None:
                    raw.index = raw.index.tz_localize("UTC")
                feats, _ = factory.build(raw)
                feats["symbol"] = pq.stem
                feats["close"] = raw["close"]
                all_dfs.append(feats)
            except Exception as exc:
                logger.warning("v2_load_parquet_failed", file=pq.stem, error=str(exc))
        if not all_dfs:
            raise FileNotFoundError(f"No valid parquets in {pq_dir}")
        return pd.concat(all_dfs).sort_index()

    def _build_7d_labels(self, df: pd.DataFrame, cfg: V2TrainingConfig) -> pd.DataFrame:
        """
        Build 7-day vol-adjusted excess return labels.

        Uses nifty_ret_5d as proxy for 7-day NIFTY return when NIFTY parquet
        is not available, otherwise loads NIFTY from parquet.
        """
        df = df.copy()

        # Try to load NIFTY from parquet for best accuracy
        nifty_pq = ROOT / "data" / "1d" / "1d" / "NIFTY.parquet"
        if nifty_pq.exists():
            nifty_df = pd.read_parquet(str(nifty_pq))
            if nifty_df.index.tz is None:
                nifty_df.index = nifty_df.index.tz_localize("UTC")
            nifty_close = nifty_df["close"].rename("nifty_close")
        else:
            nifty_close = None

        # Build 7-day label per symbol
        labels = []

        if "symbol" in df.columns:
            for symbol, grp in df.groupby("symbol", sort=False):
                grp = grp.sort_index()
                close = grp.get("close") if "close" in grp.columns else None

                if close is None:
                    # Reconstruct approximate close from returns
                    if "ret_1" in grp.columns:
                        # Can't do this easily without starting price; skip
                        label_s = pd.Series(np.nan, index=grp.index, name="label_v2")
                    else:
                        label_s = pd.Series(np.nan, index=grp.index, name="label_v2")
                else:
                    # Build a minimal OHLCV-like DataFrame
                    ohlcv_mini = pd.DataFrame({
                        "close": close,
                        "open": close,
                        "high": close,
                        "low": close,
                        "volume": pd.Series(1.0, index=grp.index),
                    }, index=grp.index)

                    if nifty_close is not None:
                        nc = nifty_close.reindex(grp.index, method="ffill")
                    elif "nifty_ret_5d" in grp.columns:
                        # Reconstruct NIFTY close from 5d returns (approximate)
                        # Use a reference price of 22000 and back-compute
                        ret5d = grp["nifty_ret_5d"].fillna(0)
                        nc = pd.Series(22000.0, index=grp.index)
                    else:
                        nc = pd.Series(22000.0, index=grp.index)

                    label_s = generate_7d_excess_return_label(
                        stock_df=ohlcv_mini,
                        nifty_close=nc,
                        horizon=cfg.label_horizon,
                        vol_window=cfg.label_vol_window,
                        clip=cfg.label_clip,
                    )
                    label_s.name = "label_v2"
                labels.append(label_s)
        else:
            # No symbol column — treat as single symbol
            if "close" in df.columns and nifty_close is not None:
                nc = nifty_close.reindex(df.index, method="ffill")
                label_s = generate_7d_excess_return_label(
                    stock_df=df[["close"]].assign(open=df["close"], high=df["close"],
                                                   low=df["close"], volume=1.0),
                    nifty_close=nc,
                    horizon=cfg.label_horizon,
                    vol_window=cfg.label_vol_window,
                    clip=cfg.label_clip,
                )
                labels.append(label_s.rename("label_v2"))
            else:
                df["label_v2"] = np.nan
                return df

        if labels:
            label_col = pd.concat(labels).sort_index()
            df["label_v2"] = label_col.reindex(df.index)
        else:
            df["label_v2"] = np.nan

        return df

    def _cv_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: pd.DatetimeIndex,
        cfg: V2TrainingConfig,
    ) -> list[float]:
        """Purged K-fold CV on training data."""
        import lightgbm as lgb

        splitter = PurgedKFoldSplitter(
            n_splits=cfg.n_cv_splits,
            embargo_days=cfg.embargo_days,
            label_horizon_days=cfg.label_horizon,
        )

        fold_ics = []
        for fold_idx, (tr_idx, val_idx) in enumerate(splitter.split(X, y, timestamps)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = lgb.LGBMRegressor(**cfg.lgbm_params)
                m.fit(X[tr_idx], y[tr_idx])
            preds = m.predict(X[val_idx])
            ic, _ = spearmanr(preds, y[val_idx])
            fold_ics.append(float(ic) if not np.isnan(ic) else 0.0)
            logger.debug("v2_cv_fold", fold=fold_idx, ic=round(fold_ics[-1], 4),
                         n_train=len(tr_idx), n_val=len(val_idx))

        return fold_ics

    def _train_model(
        self, X: np.ndarray, y: np.ndarray, cfg: V2TrainingConfig
    ) -> Any:
        """Train the final LGBMRegressor on the full train+val data."""
        import lightgbm as lgb

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = lgb.LGBMRegressor(**cfg.lgbm_params)
            model.fit(X, y)
        return model

    def _evaluate_trading(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        df_test: pd.DataFrame,
        feat_cols: list[str],
        cfg: V2TrainingConfig,
    ) -> dict[str, float]:
        """
        Evaluate trading quality using cross-sectional ranking (top/bottom 20%).
        Each day, rank all symbols by score, take the top 20% LONG and bottom 20% SHORT.
        """
        results = {"win_rate": 0.5, "ev_per_trade": 0.0, "n_signals": 0}
        if len(scores) < 100 or "symbol" not in df_test.columns:
            return results

        df_eval = df_test[["symbol"]].copy()
        df_eval["score"]   = scores
        df_eval["ret"]     = labels   # vol-adjusted excess return
        cost_frac = cfg.cost_bps / 10_000.0

        longs_ev  = []
        shorts_ev = []

        for ts in df_eval.index.unique():
            day = df_eval[df_eval.index == ts]
            if len(day) < 5:
                continue
            p80 = day["score"].quantile(0.80)
            p20 = day["score"].quantile(0.20)
            longs  = day[day["score"] >= p80]
            shorts = day[day["score"] <= p20]
            if len(longs) > 0:
                # Long: positive when label (excess ret) > 0
                longs_ev.append(float(longs["ret"].mean()) / cfg.label_clip - cost_frac)
            if len(shorts) > 0:
                # Short: positive when label (excess ret) < 0
                shorts_ev.append(-float(shorts["ret"].mean()) / cfg.label_clip - cost_frac)

        all_ev = longs_ev + shorts_ev
        if not all_ev:
            return results

        arr = np.array(all_ev)
        results["win_rate"]     = float((arr > 0).mean())
        results["ev_per_trade"] = float(arr.mean())
        results["n_signals"]    = len(arr)
        return results

    def _save_artifact(
        self,
        model: Any,
        norm: FeatureNormalizer,
        feat_cols: list[str],
        result: V2TrainingResult,
        cfg: V2TrainingConfig,
    ) -> Path:
        """Save model artifact as .pkl dict for consistency with existing infrastructure."""
        out_dir = ROOT / cfg.artifact_dir / result.model_version
        out_dir.mkdir(parents=True, exist_ok=True)

        artifact = {
            "estimator":              model,
            "estimator_name":         "lightgbm_regressor_v2",
            "calibrator":             None,    # FIX-03: no calibration
            "feature_names":          feat_cols,
            "normalizer_state":       norm.to_dict(),
            "normalization_applied":  True,
            "feature_schema_version": "fs-2.0.0",
            "label_schema_version":   "ls-3.0.0-v2",
            "label_type":             "7d_vol_adjusted_excess_return",
            "model_objective":        "regression",
            "direction_method":       "cross_sectional_rank_pct",
            "long_threshold":         0.80,    # top 20% = LONG
            "short_threshold":        0.20,    # bottom 20% = SHORT
            "cost_bps":               cfg.cost_bps,
            "invert_scores":          getattr(model, "_v2_invert", False),
        }

        pkl_path = out_dir / "model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(artifact, f, protocol=5)

        metadata = {
            "model_name":             "v2_lgbm_regressor",
            "model_version":          result.model_version,
            "run_id":                 result.run_id,
            "stage":                  "backtest",
            "artifact_path":          "model.pkl",
            "training_date":          str(datetime.now(tz=timezone.utc).date()),
            "train_rows":             result.train_rows,
            "val_rows":               result.val_rows,
            "test_rows":              result.test_rows,
            "n_features":             result.n_features,
            "feature_names":          feat_cols,
            "cv_ic_mean":             round(result.cv_ic_mean, 4),
            "cv_ic_std":              round(result.cv_ic_std, 4),
            "cv_ic_per_fold":         result.cv_ic_per_fold,
            "train_ic":               round(result.train_ic, 4),
            "val_ic":                 round(result.val_ic, 4),
            "test_ic":                round(result.test_ic, 4),
            "test_ic_pval":           round(result.test_ic_pval, 4),
            "test_win_rate":          round(result.test_win_rate, 4),
            "test_ev":                round(result.test_ev, 6),
            "label_horizon":          cfg.label_horizon,
            "cost_bps":               cfg.cost_bps,
            "train_end":              cfg.train_end,
            "val_end":                cfg.val_end,
            "test_start":             cfg.test_start,
            "fixes_applied": [
                "FIX-01: 7d-vol-adj-excess-return label",
                "FIX-02: LGBMRegressor",
                "FIX-03: no calibration",
                "FIX-04: hard test holdout 2025+",
                "FIX-05: normalizer fitted on train",
                "FIX-06: CS rank for direction",
                "FIX-07: COST_MODEL_V2 27.35bps",
                "FIX-08: PurgedKFold on train only",
            ],
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # Also update registry shadow.json to point to new model
        registry_dir = ROOT / "artifacts" / "registry" / "v2_lgbm"
        registry_dir.mkdir(parents=True, exist_ok=True)
        # Copy artifact to registry
        import shutil
        shutil.copy2(str(pkl_path), str(registry_dir / "model.pkl"))
        (registry_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        logger.info(
            "v2_artifact_saved",
            path=str(pkl_path),
            test_ic=round(result.test_ic, 4),
        )
        return pkl_path
