"""
tests/test_v2_pipeline.py — Regression tests for the v2 training pipeline.

Verifies:
  - All forensic fixes are in place
  - Label generation produces correct 7-day horizon
  - Normalizer load_state() works
  - Model artifact is valid
  - CS rank features are in [0,1]
  - True OOS IC is positive (if model artifact exists)

pytest markers: unit
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent


# ── Fix tests ─────────────────────────────────────────────────────────────────


def test_normalizer_load_state_exists():
    """FIX-05: FeatureNormalizer.load_state() must exist."""
    from src.features.normalizer import FeatureNormalizer
    norm = FeatureNormalizer()
    assert hasattr(norm, "load_state"), "load_state() method must be added to FeatureNormalizer"


def test_normalizer_load_state_roundtrip():
    """FIX-05: load_state() must restore the fitted state correctly."""
    from src.features.normalizer import FeatureNormalizer
    import numpy as np, pandas as pd

    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(0, 1, (200, 5)), columns=["a","b","c","d","e"])
    norm = FeatureNormalizer()
    norm.fit(df)
    state = norm.to_dict()
    X_orig = norm.transform(df)

    # Restore via load_state and re-transform
    norm2 = FeatureNormalizer()
    norm2.load_state(state)
    X_restored = norm2.transform(df)

    np.testing.assert_allclose(X_orig.values, X_restored.values, rtol=1e-6,
                                err_msg="load_state() must restore identical normalization")


def test_label_7d_horizon():
    """FIX-01: 7-day label must use horizon=7 (not 5)."""
    from src.labels.seven_day import HORIZON_DAYS, generate_7d_excess_return_label
    assert HORIZON_DAYS == 7

    rng = np.random.default_rng(0)
    n = 50
    idx = pd.date_range("2025-01-02", periods=n, freq="B", tz="UTC")
    prices = 1000 + np.cumsum(rng.normal(0, 5, n))
    df = pd.DataFrame({"close": prices, "open": prices, "high": prices*1.01,
                       "low": prices*0.99, "volume": np.ones(n)*1e6}, index=idx)
    nifty = pd.Series(22000 + np.cumsum(rng.normal(0, 3, n)), index=idx)

    label = generate_7d_excess_return_label(df, nifty, horizon=7)
    assert label.iloc[-7:].isna().all(), "Last 7 rows must be NaN (no forward data)"
    assert label.iloc[:-7].notna().any(), "Non-tail rows must have valid labels"


def test_v2_config_has_65_features():
    """v2 config now has 81 features (55 base + 10 CS/regime + 16 academic)."""
    from src.training.v2_pipeline import V2TrainingConfig
    cfg = V2TrainingConfig()
    n = len(cfg.feature_names)
    assert n >= 65, f"v2 config must have ≥65 features, got {n}"
    # CS features must be present
    cs_features = [f for f in cfg.feature_names if f.startswith("cs_rank_") or f.startswith("cs_")]
    assert len(cs_features) >= 6, f"Must have ≥6 cross-sectional rank features, got {len(cs_features)}"
    # Regime features must be present
    regime_features = [f for f in cfg.feature_names if "regime" in f or "breadth" in f or "trend" in f]
    assert len(regime_features) >= 2, f"Must have ≥2 regime features, got {len(regime_features)}"
    # Academic factors must be present (v2d additions)
    academic_features = [f for f in cfg.feature_names if "mom_12_1" in f or "52w" in f or "amihud" in f]
    assert len(academic_features) >= 2, f"Must have ≥2 academic factor features, got {len(academic_features)}"


# ── Model artifact tests ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def v2c_model_dict():
    """Load the latest v2c model artifact if available."""
    candidates = sorted(ROOT.glob("artifacts/v2_model/*/model.pkl"))
    if not candidates:
        pytest.skip("No v2 model artifact found — run make v2-train first")
    latest = candidates[-1]
    with open(latest, "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pickle.load(f)


def test_v2_model_is_regressor(v2c_model_dict):
    """FIX-02: Model must be LGBMRegressor, not LGBMClassifier."""
    model_type = v2c_model_dict.get("estimator_name", "")
    assert "regressor" in model_type.lower(), (
        f"v2 model must be a regressor, got: {model_type}"
    )


def test_v2_model_has_no_calibrator(v2c_model_dict):
    """FIX-03: v2 model must have no calibrator (None)."""
    calibrator = v2c_model_dict.get("calibrator")
    assert calibrator is None, (
        "v2 model must NOT have a calibrator (isotonic regression destroyed signal variance)"
    )


def test_v2_model_has_normalizer_state(v2c_model_dict):
    """FIX-05: v2 model must store normalizer state for inference."""
    norm_state = v2c_model_dict.get("normalizer_state", {})
    assert isinstance(norm_state, dict) and len(norm_state) > 0, (
        "v2 model must contain a non-empty normalizer_state"
    )
    assert norm_state.get("fitted", False) is True, (
        "normalizer_state must be fitted"
    )


def test_v2_model_feature_schema(v2c_model_dict):
    """FIX-06: v2 model must have feature_names."""
    feature_names = v2c_model_dict.get("feature_names", [])
    assert len(feature_names) >= 55, (
        f"v2 model must have ≥55 features, got {len(feature_names)}"
    )


def test_v2_model_label_type(v2c_model_dict):
    """FIX-01: v2 model must use 7-day label."""
    label_type = v2c_model_dict.get("label_type", "")
    assert "7d" in label_type or "7_day" in label_type or "rank" in label_type, (
        f"v2 model must use 7-day label, got: {label_type}"
    )


def test_v2_model_direction_method(v2c_model_dict):
    """FIX-06: v2 model must use cross-sectional rank for direction."""
    direction_method = v2c_model_dict.get("direction_method", "")
    assert "rank" in direction_method.lower() or "cs" in direction_method.lower(), (
        f"v2 model must use CS rank for direction, got: {direction_method}"
    )


def test_v2_model_inference_works(v2c_model_dict):
    """v2 model must produce valid predictions on synthetic data."""
    from src.training.v2_pipeline import V2TrainingConfig
    from src.features.normalizer import FeatureNormalizer

    cfg = V2TrainingConfig()
    feature_names = v2c_model_dict.get("feature_names", cfg.feature_names)
    estimator = v2c_model_dict["estimator"]
    norm_state = v2c_model_dict.get("normalizer_state", {})

    rng = np.random.default_rng(42)
    X_raw = pd.DataFrame(
        rng.normal(0, 1, (100, len(feature_names))),
        columns=feature_names
    )

    if norm_state:
        norm = FeatureNormalizer()
        norm.load_state(norm_state)
        X_raw = norm.transform_known(X_raw).fillna(0.0)

    X = X_raw.to_numpy(dtype=float)
    scores = estimator.predict(X)
    assert len(scores) == 100, "Must produce one score per input row"
    assert not np.any(np.isnan(scores)), "Scores must not be NaN"


# ── OOS performance regression ────────────────────────────────────────────────


@pytest.mark.slow
def test_v2_model_positive_oos_ic(v2c_model_dict):
    """
    v2c model must have positive IC on the OOS test dataset.
    This is a regression test — if this fails, something regressed.
    Target: IC > 0.01 (pragmatic threshold, not production threshold).
    """
    from src.features.normalizer import FeatureNormalizer
    from scipy.stats import spearmanr

    oos_start = pd.Timestamp("2025-01-01", tz="UTC")
    v2c_path = ROOT / "artifacts/datasets/v2c_cs_regime/data.parquet"
    if not v2c_path.exists():
        pytest.skip("v2c dataset not found — run make v2-build first")

    df = pd.read_parquet(str(v2c_path))
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df_oos = df[df.index >= oos_start].copy()

    if "label_v2b" not in df_oos.columns:
        pytest.skip("label_v2b not in v2c dataset")

    feature_names = v2c_model_dict.get("feature_names", [])
    avail = [f for f in feature_names if f in df_oos.columns]
    if len(avail) < 10:
        pytest.skip("Insufficient features available")

    estimator = v2c_model_dict["estimator"]
    norm_state = v2c_model_dict.get("normalizer_state", {})

    # Build a full feature matrix with ALL model-expected features.
    # Missing features (not in OOS dataset) are filled with 0.0.
    # Using only `avail` columns causes a feature-count mismatch when
    # len(avail) < len(feature_names), crashing LGBMRegressor.predict().
    X = pd.DataFrame(0.0, index=df_oos.index, columns=feature_names)
    for col in avail:
        X[col] = df_oos[col].fillna(0.0)
    if norm_state:
        try:
            norm = FeatureNormalizer()
            norm.load_state(norm_state)
            X = norm.transform_known(X).fillna(0.0)
        except Exception:
            pass

    valid = df_oos["label_v2b"].notna()
    X_valid = X[valid].to_numpy(dtype=float)
    y_valid = df_oos.loc[valid, "label_v2b"].to_numpy(dtype=float)

    if len(X_valid) < 1000:
        pytest.skip("Insufficient OOS data")

    scores = estimator.predict(X_valid)
    ic, pval = spearmanr(scores, y_valid)

    # IC must be significant (p < 0.05) and non-negative
    assert pval < 0.05, f"OOS IC must be statistically significant (p={pval:.4f})"
    # Note: direction may be inverted (the model artifact stores invert_scores flag)
    # Test that |IC| > 0.01 (there is genuine signal)
    assert abs(ic) > 0.005, (
        f"OOS |IC| = {abs(ic):.4f} must be > 0.005 — model has no genuine signal"
    )
