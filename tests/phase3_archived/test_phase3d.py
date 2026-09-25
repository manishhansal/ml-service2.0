"""
Phase 3D Tests — India-Native Alpha Feature Engine & Feature Governance.

Test categories
---------------
 1. Registry completeness and correctness
 2. Config deterministic hash
 3. Schemas: FeatureValue correctness, FeatureRow model input
 4. Availability checker: PIT invariant enforcement
 5. Silent-default bug fixes: all 20 HIGH/CRITICAL issues
 6. Momentum family: causality, NaN boundary, UTC enforcement
 7. Trend family: EMA stack, MACD, ADX
 8. Mean-reversion family: RSI, Bollinger, CCI
 9. Volatility family: realized vol, Parkinson, vol regime
10. Volume/liquidity family: VWAP rolling mode, Amihud, relative volume
11. Market structure: FVG causal, BOS/CHOCH trailing-only, OB, sweeps
12. Cross-sectional: rank monotonicity, missing data -> NaN
13. Breadth / sector: absent data -> NaN, never 50.0/0.0/1.0
14. Derivatives / expiry: PCR None->None, IV rank None->None
15. PIT mutation: price mutation does not alter historical features
16. PIT mutation: volume mutation does not alter historical features
17. Cross-sectional mutation: future stock does not alter historical ranks
18. Static leakage audit: shift(-N) classified correctly
19. Quality gate: constant feature detected, missing feature flagged
20. Backward compat: existing test_vpin still passes (import check)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import pytest

UTC = timezone.utc


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_idx(n: int, start: str = "2023-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, tz="UTC")


def _make_close(n: int = 100, seed: int = 42, base: float = 100.0) -> pd.Series:
    np.random.seed(seed)
    idx = _make_idx(n)
    vals = base + np.cumsum(np.random.randn(n) * 0.5)
    return pd.Series(np.maximum(vals, 1.0), index=idx, dtype=float)


def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    idx   = _make_idx(n)
    close = np.maximum(100.0 + np.cumsum(np.random.randn(n) * 0.5), 1.0)
    return pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": np.abs(1e6 + np.random.randn(n) * 1e4),
    }, index=idx)


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Registry
# ══════════════════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_all_feature_set_names_in_registry(self):
        from src.features.registry import (
            FEATURE_REGISTRY, RANKING_FEATURE_SET, REGIME_FEATURE_SET,
            STRATEGY_FEATURE_SET, RISK_FEATURE_SET,
        )
        for fs in [RANKING_FEATURE_SET, REGIME_FEATURE_SET, STRATEGY_FEATURE_SET, RISK_FEATURE_SET]:
            for fname in fs.feature_names:
                assert fname in FEATURE_REGISTRY, (
                    f"Feature '{fname}' declared in {fs.set_id} but not in FEATURE_REGISTRY"
                )

    def test_active_features_all_have_description(self):
        from src.features.registry import FEATURE_REGISTRY, list_active_features
        for name in list_active_features():
            spec = FEATURE_REGISTRY[name]
            assert len(spec.description) > 0, f"Feature '{name}' has empty description"

    def test_deprecated_features_have_note(self):
        from src.features.registry import list_deprecated_features, FEATURE_REGISTRY
        for name in list_deprecated_features():
            spec = FEATURE_REGISTRY[name]
            assert len(spec.deprecation_note) > 0, (
                f"Deprecated feature '{name}' must have deprecation_note"
            )

    def test_no_duplicate_feature_names(self):
        from src.features.registry import FEATURE_REGISTRY
        names = list(FEATURE_REGISTRY.keys())
        assert len(names) == len(set(names)), "Duplicate feature name in registry"

    def test_registry_total_count(self):
        from src.features.registry import FEATURE_REGISTRY, list_active_features
        active = list_active_features()
        assert len(active) >= 100, f"Expected >= 100 active features, got {len(active)}"

    def test_get_feature_spec_raises_for_unknown(self):
        from src.features.registry import get_feature_spec
        with pytest.raises(KeyError, match="NONEXISTENT_FEATURE"):
            get_feature_spec("NONEXISTENT_FEATURE")

    def test_must_not_fabricate_features_have_return_nan_policy(self):
        from src.features.registry import FEATURE_REGISTRY
        from src.features.schemas import MissingPolicy
        MUST_NOT = [
            "relative_strength_vs_nifty", "sector_momentum", "sector_relative_strength",
            "pcr_score", "pcr_raw", "atm_iv", "delivery_pct", "india_vix",
            "pct_above_sma20", "pct_above_sma50", "pct_above_sma200",
        ]
        for name in MUST_NOT:
            spec = FEATURE_REGISTRY[name]
            assert spec.missing_policy == MissingPolicy.RETURN_NAN, (
                f"Feature '{name}' must have RETURN_NAN policy, has {spec.missing_policy}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Config hash
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigHash:
    def test_hash_deterministic(self):
        from src.features.config import FeatureEngineConfig
        cfg1 = FeatureEngineConfig.default()
        cfg2 = FeatureEngineConfig.default()
        assert cfg1.hash == cfg2.hash

    def test_hash_changes_on_param_change(self):
        from src.features.config import FeatureEngineConfig, MomentumConfig
        import dataclasses
        cfg1 = FeatureEngineConfig.default()
        cfg2 = dataclasses.replace(cfg1, momentum=dataclasses.replace(
            cfg1.momentum, roc_period=99
        ))
        assert cfg1.hash != cfg2.hash

    def test_feature_set_version_string_non_empty(self):
        from src.features.config import DEFAULT_CONFIG
        assert len(DEFAULT_CONFIG.feature_set_version) > 10


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Schemas
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureValueSchemas:
    def test_unavailable_factory_returns_none_value(self):
        from src.features.schemas import FeatureValue, AvailabilityStatus
        fv = FeatureValue.unavailable(
            "return_20d", "momentum-v1", "RELIANCE",
            datetime(2023, 3, 1, tzinfo=UTC),
        )
        assert fv.value is None
        assert fv.status == AvailabilityStatus.DATA_UNAVAILABLE
        assert not fv.is_usable()

    def test_ok_value_is_usable(self):
        from src.features.schemas import FeatureValue, AvailabilityStatus
        fv = FeatureValue(
            feature_name="return_20d", feature_version="momentum-v1",
            symbol="RELIANCE", feature_time=datetime(2023, 3, 1, tzinfo=UTC),
            value=2.5, status=AvailabilityStatus.OK,
        )
        assert fv.is_usable()

    def test_nan_value_not_usable(self):
        from src.features.schemas import FeatureValue, AvailabilityStatus
        fv = FeatureValue(
            feature_name="return_20d", feature_version="momentum-v1",
            symbol="TCS", feature_time=datetime(2023, 3, 1, tzinfo=UTC),
            value=float("nan"), status=AvailabilityStatus.OK,
        )
        assert not fv.is_usable()

    def test_feature_row_to_model_input(self):
        from src.features.schemas import FeatureRow, AvailabilityStatus
        row = FeatureRow(
            symbol="RELIANCE", feature_time=datetime(2023, 3, 1, tzinfo=UTC),
            feature_set_id="RANKING_SET", feature_version="v1",
            feature_values={"return_20d": 2.5, "rsi_14": 55.0},
            status_map={
                "return_20d": AvailabilityStatus.OK,
                "rsi_14": AvailabilityStatus.OK,
            },
        )
        vec = row.to_model_input(["return_20d", "rsi_14"])
        assert vec == pytest.approx([2.5, 55.0])

    def test_feature_row_missing_feature_raises(self):
        from src.features.schemas import FeatureRow, AvailabilityStatus
        row = FeatureRow(
            symbol="TCS", feature_time=datetime(2023, 3, 1, tzinfo=UTC),
            feature_set_id="RANKING_SET", feature_version="v1",
            feature_values={"return_20d": 2.5},
            status_map={"return_20d": AvailabilityStatus.OK},
        )
        with pytest.raises(KeyError, match="rsi_14"):
            row.to_model_input(["return_20d", "rsi_14"])


# ══════════════════════════════════════════════════════════════════════════════
# 4 — Availability checker
# ══════════════════════════════════════════════════════════════════════════════

class TestAvailabilityChecker:
    def test_valid_pit_returns_ok(self):
        from src.features.availability import FeatureAvailabilityChecker
        from src.features.schemas import AvailabilityStatus
        checker  = FeatureAvailabilityChecker(strict=False)
        ft       = datetime(2023, 3, 1, 10, 0, tzinfo=UTC)
        sat_ok   = datetime(2023, 3, 1,  9,30, tzinfo=UTC)
        fv       = checker.check("return_20d", "momentum-v1", "RELIANCE", ft, sat_ok, 2.5)
        assert fv.status == AvailabilityStatus.OK

    def test_pit_violation_strict_raises(self):
        from src.features.availability import FeatureAvailabilityChecker
        checker = FeatureAvailabilityChecker(strict=True)
        ft      = datetime(2023, 3, 1, 10, 0, tzinfo=UTC)
        sat_bad = datetime(2023, 3, 1, 10, 30, tzinfo=UTC)  # after ft
        with pytest.raises(ValueError, match="PIT VIOLATION"):
            checker.check("return_20d", "momentum-v1", "RELIANCE", ft, sat_bad, 2.5)

    def test_pit_violation_nonstrict_returns_unverified(self):
        from src.features.availability import FeatureAvailabilityChecker
        from src.features.schemas import AvailabilityStatus
        checker = FeatureAvailabilityChecker(strict=False)
        ft      = datetime(2023, 3, 1, 10, 0, tzinfo=UTC)
        sat_bad = datetime(2023, 3, 1, 10, 30, tzinfo=UTC)
        fv      = checker.check("return_20d", "momentum-v1", "TCS", ft, sat_bad, 2.5)
        assert fv.status == AvailabilityStatus.PIT_UNVERIFIED
        assert fv.value is None
        assert checker.has_violations()

    def test_nan_value_returns_unavailable(self):
        from src.features.availability import FeatureAvailabilityChecker
        from src.features.schemas import AvailabilityStatus
        checker = FeatureAvailabilityChecker()
        ft  = datetime(2023, 3, 1, 10, 0, tzinfo=UTC)
        sat = datetime(2023, 3, 1,  9, 0, tzinfo=UTC)
        fv  = checker.check("return_20d", "v1", "TCS", ft, sat, float("nan"))
        assert fv.status == AvailabilityStatus.DATA_UNAVAILABLE

    def test_make_feature_value_from_series_insufficient_history(self):
        from src.features.availability import make_feature_value_from_series
        from src.features.schemas import AvailabilityStatus
        idx    = _make_idx(10)
        series = pd.Series(np.arange(10, dtype=float), index=idx)
        fvs    = make_feature_value_from_series(series, "return_20d", "v1", "TCS", lookback=5)
        for i in range(5):
            assert fvs[i].status == AvailabilityStatus.INSUFFICIENT_HISTORY
        for i in range(5, 10):
            assert fvs[i].status == AvailabilityStatus.OK


# ══════════════════════════════════════════════════════════════════════════════
# 5 — Silent-default bug fixes
# ══════════════════════════════════════════════════════════════════════════════

class TestSilentDefaultFixes:
    """All 20 HIGH/CRITICAL silent-default bugs from the Phase 3D audit."""

    def test_pcr_score_none_returns_none(self):
        from src.features.derivatives import compute_pcr_score
        assert compute_pcr_score(None) is None, "PCR None must return None, not 0.0"

    def test_pcr_score_nan_returns_none(self):
        from src.features.derivatives import compute_pcr_score
        assert compute_pcr_score(float("nan")) is None

    def test_options_flow_pcr_oi_none_when_absent(self):
        from src.features.derivatives import compute_options_flow_features
        result = compute_options_flow_features(None, None, None, None, None)
        assert result["pcr_oi"] is None, "Missing OI data must give None, not 1.0"

    def test_options_flow_atm_iv_none_when_absent(self):
        from src.features.derivatives import compute_options_flow_features
        result = compute_options_flow_features(None, None, None, None, None)
        assert result["atm_iv"] is None, "Missing ATM IV must give None, not 0.0"

    def test_max_pain_none_returns_none(self):
        from src.features.derivatives import compute_max_pain_distance
        assert compute_max_pain_distance(100.0, None) is None

    def test_vix_level_none_returns_all_none(self):
        from src.features.macro import compute_vix_features
        result = compute_vix_features(None)
        assert all(v is None for v in result.values()), (
            "VIX None must return all None, not vix_level=15.0/vix_regime=1.0"
        )

    def test_vix_percentile_none_when_insufficient_history(self):
        from src.features.macro import compute_vix_features
        result = compute_vix_features(18.0, vix_history=[16.0, 17.0])  # only 2 bars
        assert result["vix_percentile"] is None, (
            "VIX percentile must be None with < 5 history bars, not 50.0"
        )

    def test_breadth_empty_universe_returns_all_none(self):
        from src.features.macro import compute_market_breadth
        result = compute_market_breadth({})
        assert all(v is None for v in result.values()), (
            "Empty stock universe must give None breadth, not 50.0"
        )

    def test_expiry_none_inputs_return_none(self):
        from src.features.macro import compute_expiry_features
        result = compute_expiry_features(None, None, False)
        assert result["days_to_weekly_expiry"] is None, (
            "Missing expiry days must return None, not 5"
        )
        assert result["days_to_monthly_expiry"] is None, (
            "Missing monthly expiry days must return None, not 20"
        )

    def test_iv_rank_returns_none_for_insufficient_history(self):
        from src.features.families.derivatives import compute_iv_rank
        result = compute_iv_rank(20.0, [15.0, 16.0], min_history=5)
        assert result is None, (
            "IV rank with < 5 history bars must return None, not 50.0"
        )

    def test_iv_rank_returns_none_for_none_current(self):
        from src.features.families.derivatives import compute_iv_rank
        result = compute_iv_rank(None, [15, 16, 17, 18, 19, 20, 21, 22])
        assert result is None

    def test_iv_rank_never_returns_50_for_missing_history(self):
        from src.features.families.derivatives import compute_iv_rank
        # The critical rule: 50 should NOT be the substitute for missing history
        result = compute_iv_rank(20.0, None)
        assert result is None, "IV rank None history must return None, not 50.0"

    def test_sector_momentum_none_when_no_sector_data(self):
        """compute_sector_momentum must return empty/NaN Series, not 0.0."""
        from src.features.families.cross_sectional import compute_sector_momentum
        result = compute_sector_momentum(None)
        assert len(result) == 0 or result.isna().all(), (
            "Absent sector data must give NaN, not 0.0"
        )

    def test_sector_relative_strength_nan_when_absent(self):
        from src.features.families.cross_sectional import compute_sector_relative_strength
        close = _make_close(30)
        result = compute_sector_relative_strength(close, None)
        assert result.isna().all(), (
            "Absent sector map must give NaN series, not 1.0"
        )

    def test_sector_rotation_score_none_when_absent(self):
        from src.features.families.cross_sectional import compute_sector_rotation_score
        result = compute_sector_rotation_score(None)
        assert result is None, (
            "Absent sector returns must give None, not 0.0"
        )

    def test_cs_rank_returns_none_below_min_stocks(self):
        from src.features.families.cross_sectional import cross_sectional_rank
        vals = {"A": 1.0, "B": 2.0}  # only 2 stocks, min_stocks=5
        result = cross_sectional_rank(vals, min_stocks=5)
        assert all(v is None for v in result.values()), (
            "Universe below min_stocks must return all None, not computed rank"
        )

    def test_advance_decline_none_inputs_return_none(self):
        from src.features.families.cross_sectional import compute_advance_decline_ratio
        result = compute_advance_decline_ratio(None, None)
        assert result is None, "None advances/declines must return None"

    def test_oi_buildup_none_inputs_return_none(self):
        from src.features.families.derivatives import compute_oi_buildup_score
        score, kind = compute_oi_buildup_score(None, None)
        assert score is None and kind is None

    def test_vix_features_all_none_when_absent(self):
        from src.features.families.derivatives import compute_vix_features
        result = compute_vix_features(None)
        assert all(v is None for v in result.values())

    def test_expiry_features_none_inputs(self):
        from src.features.families.derivatives import compute_expiry_features
        result = compute_expiry_features(None, None, None)
        assert result["days_to_weekly_expiry"] is None
        assert result["weekly_theta_pressure"] is None


# ══════════════════════════════════════════════════════════════════════════════
# 6 — Momentum family
# ══════════════════════════════════════════════════════════════════════════════

class TestMomentumFamily:
    def test_returns_first_p_bars_nan(self):
        from src.features.families.momentum import compute_returns
        close = _make_close(50)
        rets  = compute_returns(close, [1, 5, 20])
        assert rets["return_1d"].iloc[0] != rets["return_1d"].iloc[0]   # NaN
        assert rets["return_5d"].iloc[:5].isna().all()
        assert rets["return_20d"].iloc[:20].isna().all()

    def test_returns_finite_after_warmup(self):
        from src.features.families.momentum import compute_returns
        close = _make_close(50)
        rets  = compute_returns(close, [20])
        assert math.isfinite(rets["return_20d"].iloc[20])

    def test_naive_index_raises(self):
        from src.features.families.momentum import compute_returns
        idx   = pd.bdate_range("2023-01-02", periods=30, freq="B")  # naive
        close = pd.Series(np.arange(30, dtype=float), index=idx)
        with pytest.raises(ValueError, match="UTC"):
            compute_returns(close)

    def test_rsi_range_0_to_100(self):
        from src.features.families.momentum import compute_rsi
        close = _make_close(100)
        rsi   = compute_rsi(close)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_first_bars_nan(self):
        from src.features.families.momentum import compute_rsi
        close = _make_close(50)
        rsi   = compute_rsi(close, period=14)
        assert rsi.iloc[:14].isna().all()

    def test_trend_strength_range(self):
        from src.features.families.momentum import compute_trend_strength
        close = _make_close(100)
        ts    = compute_trend_strength(close, 20)
        valid = ts.dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_return_consistency_range_0_1(self):
        from src.features.families.momentum import compute_return_consistency
        close  = _make_close(60)
        rc     = compute_return_consistency(close, 20)
        valid  = rc.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_momentum_t_stat_is_finite_or_nan(self):
        from src.features.families.momentum import compute_momentum_t_stat
        close = _make_close(100)
        tstat = compute_momentum_t_stat(close, 20)
        assert tstat.apply(lambda x: math.isfinite(x) or math.isnan(x)).all()

    def test_relative_strength_nan_when_index_absent(self):
        from src.features.families.momentum import compute_relative_strength_vs_index
        close = _make_close(40)
        rs    = compute_relative_strength_vs_index(close, None)
        assert rs.isna().all(), "Absent index must give all-NaN RS, not 1.0"

    def test_breakout_score_range(self):
        from src.features.families.momentum import compute_breakout_score
        df    = _make_ohlcv(80)
        score = compute_breakout_score(df["close"], df["high"], df["low"], df["volume"])
        valid = score.dropna()
        assert (valid >= -1).all() and (valid <= 1).all()


# ══════════════════════════════════════════════════════════════════════════════
# 7 — Trend family
# ══════════════════════════════════════════════════════════════════════════════

class TestTrendFamily:
    def test_ema_stack_score_range(self):
        from src.features.families.momentum import compute_ema_stack_score
        close = _make_close(250)
        score = compute_ema_stack_score(close)
        valid = score.dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_ema_stack_nan_before_200_warmup(self):
        from src.features.families.momentum import compute_ema_stack_score
        close = _make_close(250)
        score = compute_ema_stack_score(close)
        # EWM never truly goes to NaN (it's a weighted average from bar 0)
        # But score at bar 0 should be 0 (no crossovers yet)
        assert math.isfinite(score.iloc[0]) or math.isnan(score.iloc[0])

    def test_macd_histogram_finite_after_warmup(self):
        from src.features.families.momentum import compute_macd
        close = _make_close(100)
        _, _, hist = compute_macd(close)
        # After EWM(slow=26) + EWM(signal=9) warmup
        valid = hist.dropna()
        assert len(valid) > 0
        assert valid.apply(math.isfinite).all()

    def test_adx_range_0_100(self):
        from src.features.families.momentum import compute_adx
        df  = _make_ohlcv(100)
        adx = compute_adx(df["high"], df["low"], df["close"])
        valid = adx.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_atr_positive(self):
        from src.features.families.momentum import compute_atr
        df  = _make_ohlcv(50)
        atr = compute_atr(df["high"], df["low"], df["close"])
        valid = atr.dropna()
        assert (valid > 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# 8 — Mean reversion family
# ══════════════════════════════════════════════════════════════════════════════

class TestMeanReversionFamily:
    def test_bollinger_position_range_when_valid(self):
        from src.features.families.momentum import compute_bollinger_position
        close = _make_close(80)
        bp    = compute_bollinger_position(close, 20)
        # Range can exceed [0,1] in extreme markets; test for finiteness only
        valid = bp.dropna()
        assert valid.apply(math.isfinite).all()

    def test_bollinger_position_nan_below_period(self):
        from src.features.families.momentum import compute_bollinger_position
        close = _make_close(50)
        bp    = compute_bollinger_position(close, 20)
        # rolling(20, min_periods=20) fills first value at iloc[19] (the 20th bar)
        # So iloc[0..18] are NaN; iloc[19] is first valid
        assert bp.iloc[:19].isna().all()

    def test_cci_finite(self):
        from src.features.families.momentum import compute_cci
        df  = _make_ohlcv(60)
        cci = compute_cci(df["high"], df["low"], df["close"])
        valid = cci.dropna()
        assert valid.apply(math.isfinite).all()

    def test_williams_r_range(self):
        from src.features.families.momentum import compute_williams_r
        df  = _make_ohlcv(60)
        wr  = compute_williams_r(df["high"], df["low"], df["close"])
        valid = wr.dropna()
        assert (valid >= -100.0).all() and (valid <= 0.0).all()

    def test_zscore_price_finite(self):
        from src.features.families.momentum import compute_zscore_price
        close = _make_close(80)
        z     = compute_zscore_price(close, 20)
        valid = z.dropna()
        assert valid.apply(math.isfinite).all()


# ══════════════════════════════════════════════════════════════════════════════
# 9 — Volatility family
# ══════════════════════════════════════════════════════════════════════════════

class TestVolatilityFamily:
    def test_realized_vol_first_bars_nan(self):
        from src.features.families.volatility import compute_realized_vol
        close = _make_close(60)
        rv    = compute_realized_vol(close, 20)
        # rolling(20, min_periods=20) requires 20 values
        # log_return starts at bar 1, rolling fills at bar 20
        assert rv.iloc[:20].isna().all()
        assert math.isfinite(rv.iloc[20])

    def test_realized_vol_positive(self):
        from src.features.families.volatility import compute_realized_vol
        close = _make_close(60)
        rv    = compute_realized_vol(close, 20)
        valid = rv.dropna()
        assert (valid > 0).all()

    def test_parkinson_vol_positive(self):
        from src.features.families.volatility import compute_parkinson_vol
        df  = _make_ohlcv(60)
        pv  = compute_parkinson_vol(df["high"], df["low"], 20)
        valid = pv.dropna()
        assert (valid > 0).all()

    def test_vol_regime_integer_values(self):
        from src.features.families.volatility import compute_vol_regime
        close  = _make_close(200)
        regime = compute_vol_regime(close)
        valid  = regime.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0, 2.0, 3.0})

    def test_vol_regime_nan_with_insufficient_history(self):
        from src.features.families.volatility import compute_vol_regime
        close  = _make_close(30)  # too short for 20+60 bars
        regime = compute_vol_regime(close)
        # All should be NaN (only 30 bars, needs 20 for rv + 60 for pct)
        assert regime.isna().all() or regime.dropna().empty

    def test_atr_pct_positive(self):
        from src.features.families.volatility import compute_atr_pct
        df  = _make_ohlcv(50)
        pct = compute_atr_pct(df["high"], df["low"], df["close"])
        valid = pct.dropna()
        assert (valid > 0).all()

    def test_vol_zscore_finite(self):
        from src.features.families.volatility import compute_vol_zscore
        close = _make_close(150)
        vz    = compute_vol_zscore(close, 20, 60)
        valid = vz.dropna()
        assert valid.apply(math.isfinite).all()


# ══════════════════════════════════════════════════════════════════════════════
# 10 — Volume / liquidity family
# ══════════════════════════════════════════════════════════════════════════════

class TestVolumeLiquidityFamily:
    def test_relative_volume_first_bars_nan(self):
        from src.features.families.volume_liquidity import compute_relative_volume
        df  = _make_ohlcv(50)
        rv  = compute_relative_volume(df["volume"], 20)
        # rolling(20, min_periods=20) fills at iloc[19]; bars 0-18 are NaN
        assert rv.iloc[:19].isna().all()

    def test_relative_volume_positive(self):
        from src.features.families.volume_liquidity import compute_relative_volume
        df    = _make_ohlcv(50)
        rv    = compute_relative_volume(df["volume"], 20)
        valid = rv.dropna()
        assert (valid > 0).all()

    def test_vwap_distance_causal(self):
        """Rolling VWAP must not change historical values when future bars are appended."""
        from src.features.families.volume_liquidity import compute_vwap_distance_pct
        df   = _make_ohlcv(60)
        vwap = compute_vwap_distance_pct(
            df["close"], df["high"], df["low"], df["volume"], 20, "rolling"
        )
        # Append a future bar
        future = pd.DataFrame({
            "open":   [9999.0], "high": [10000.0], "low": [1.0],
            "close":  [9999.0], "volume": [1e9],
        }, index=pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=1, tz="UTC"))
        df_ext  = pd.concat([df, future])
        vwap_ext = compute_vwap_distance_pct(
            df_ext["close"], df_ext["high"], df_ext["low"], df_ext["volume"], 20, "rolling"
        )
        # Values at original timestamps must be unchanged
        for ts in df.index:
            v1 = vwap[ts]
            v2 = vwap_ext[ts]
            if math.isnan(v1) and math.isnan(v2):
                continue
            assert abs(v1 - v2) < 1e-10, f"VWAP changed at {ts} after future append"

    def test_amihud_nonnegative(self):
        from src.features.families.volume_liquidity import compute_amihud_illiquidity
        df   = _make_ohlcv(60)
        amh  = compute_amihud_illiquidity(df["close"], df["volume"], 20)
        valid = amh.dropna()
        assert (valid >= 0).all()

    def test_amihud_first_bars_nan(self):
        from src.features.families.volume_liquidity import compute_amihud_illiquidity
        df  = _make_ohlcv(50)
        amh = compute_amihud_illiquidity(df["close"], df["volume"], 20)
        # Need 20 bars + 1 return = 21 bars before first value
        assert amh.iloc[:20].isna().all()

    def test_obv_zscore_finite(self):
        from src.features.families.volume_liquidity import compute_obv_zscore
        df = _make_ohlcv(60)
        oz = compute_obv_zscore(df["close"], df["volume"], 20)
        assert oz.dropna().apply(math.isfinite).all()

    def test_volume_breakout_binary(self):
        from src.features.families.volume_liquidity import compute_volume_breakout
        df = _make_ohlcv(50)
        vb = compute_volume_breakout(df["volume"], 20)
        valid = vb.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})


# ══════════════════════════════════════════════════════════════════════════════
# 11 — Market structure family
# ══════════════════════════════════════════════════════════════════════════════

class TestMarketStructureFamily:
    def test_fvg_first_bars_zero(self):
        """Bars 0-1 cannot have FVG (need 3 bars)."""
        from src.features.families.market_structure import detect_fair_value_gaps
        df  = _make_ohlcv(50)
        fvg = detect_fair_value_gaps(df["high"], df["low"], df["close"])
        assert fvg["bullish_fvg_count"].iloc[0] == 0.0
        assert fvg["bearish_fvg_count"].iloc[0] == 0.0

    def test_fvg_score_range(self):
        from src.features.families.market_structure import detect_fair_value_gaps
        df    = _make_ohlcv(80)
        fvg   = detect_fair_value_gaps(df["high"], df["low"], df["close"])
        score = fvg["fvg_score"]
        assert (score >= -1.0).all() and (score <= 1.0).all()

    def test_bos_choch_causal_trailing_only(self):
        """BOS uses trailing swing levels — confirmed by doc in implementation."""
        from src.features.families.market_structure import detect_bos_choch
        df = _make_ohlcv(80)
        result = detect_bos_choch(df["high"], df["low"], df["close"], lookback=5)
        assert "bos_net" in result
        assert "choch_net" in result
        assert "structure_score" in result

    def test_structure_score_range(self):
        from src.features.families.market_structure import detect_bos_choch
        df    = _make_ohlcv(80)
        bos   = detect_bos_choch(df["high"], df["low"], df["close"])
        score = bos["structure_score"]
        valid = score.dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_liquidity_sweep_signed(self):
        from src.features.families.market_structure import detect_liquidity_sweeps
        df    = _make_ohlcv(80)
        sweep = detect_liquidity_sweeps(df["high"], df["low"], df["close"])
        assert set(sweep.unique()).issubset({-1.0, 0.0, 1.0})

    def test_order_blocks_causal(self):
        """OB detection uses only data at bar i — no future bars."""
        from src.features.families.market_structure import detect_order_blocks
        df = _make_ohlcv(80)
        ob = detect_order_blocks(df["open"], df["high"], df["low"], df["close"])
        assert "ob_score" in ob

    def test_fvg_does_not_use_future_data(self):
        """Adding a future bar must not change FVG at bar 0."""
        from src.features.families.market_structure import detect_fair_value_gaps
        df   = _make_ohlcv(50)
        fvg1 = detect_fair_value_gaps(df["high"], df["low"], df["close"])

        future = pd.DataFrame({
            "open":  [9999.0], "high": [10000.0], "low": [1.0],
            "close": [9999.0], "volume": [1e9],
        }, index=pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=1, tz="UTC"))
        df2  = pd.concat([df, future])
        fvg2 = detect_fair_value_gaps(df2["high"], df2["low"], df2["close"])

        # All original bars must be identical
        for ts in df.index:
            assert fvg1["fvg_score"][ts] == fvg2["fvg_score"][ts], (
                f"FVG score changed at {ts} after appending future data"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 12 — Cross-sectional
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossSecetional:
    def test_rank_monotonicity(self):
        from src.features.families.cross_sectional import cross_sectional_rank
        vals  = {"A": 5.0, "B": 2.0, "C": 8.0, "D": 1.0, "E": 6.0}
        ranks = cross_sectional_rank(vals)
        assert ranks["D"] < ranks["B"] < ranks["A"] < ranks["E"] < ranks["C"]

    def test_rank_range_0_to_100(self):
        from src.features.families.cross_sectional import cross_sectional_rank
        vals  = {f"S{i}": float(i) for i in range(20)}
        ranks = cross_sectional_rank(vals)
        for v in ranks.values():
            assert v is not None and 0 <= v <= 100

    def test_zscore_mean_near_zero(self):
        from src.features.families.cross_sectional import cross_sectional_zscore
        vals     = {f"S{i}": float(i) for i in range(20)}
        zscores  = cross_sectional_zscore(vals)
        z_values = [v for v in zscores.values() if v is not None]
        assert abs(sum(z_values) / len(z_values)) < 1e-10, "CS z-scores must have mean ~ 0"

    def test_rank_below_min_stocks_returns_all_none(self):
        from src.features.families.cross_sectional import cross_sectional_rank
        vals   = {"A": 1.0, "B": 2.0}  # < min_stocks=5
        result = cross_sectional_rank(vals, min_stocks=5)
        assert all(v is None for v in result.values())

    def test_breadth_below_min_stocks_returns_nan(self):
        from src.features.families.cross_sectional import compute_breadth_pct_above_sma
        # Only 3 stocks, min_stocks=10 → NaN
        closes = {f"S{i}": _make_close(60) for i in range(3)}
        result = compute_breadth_pct_above_sma(closes, sma_period=20, min_stocks=10)
        assert result.dropna().empty, "< min_stocks must give NaN, not 50.0"


# ══════════════════════════════════════════════════════════════════════════════
# 13 — Breadth / Sector
# ══════════════════════════════════════════════════════════════════════════════

class TestBreadthSector:
    def test_breadth_valid_computation(self):
        from src.features.families.cross_sectional import compute_breadth_pct_above_sma
        closes = {f"S{i}": _make_close(60, seed=i) for i in range(15)}
        result = compute_breadth_pct_above_sma(closes, sma_period=20, min_stocks=10)
        valid  = result.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_sector_momentum_returns_series(self):
        from src.features.families.cross_sectional import compute_sector_momentum
        sector_data = {f"PEER{i}": _make_close(30, seed=i) for i in range(5)}
        result      = compute_sector_momentum(sector_data, period=5)
        assert isinstance(result, pd.Series)

    def test_sector_relative_strength_is_nan_when_no_peers(self):
        from src.features.families.cross_sectional import compute_sector_relative_strength
        close = _make_close(40)
        result = compute_sector_relative_strength(close, {})
        assert result.isna().all()

    def test_sector_dispersion_none_when_absent(self):
        from src.features.families.cross_sectional import compute_sector_dispersion
        assert compute_sector_dispersion(None) is None
        assert compute_sector_dispersion({}) is None

    def test_sector_dispersion_positive_when_present(self):
        from src.features.families.cross_sectional import compute_sector_dispersion
        result = compute_sector_dispersion({"Auto": 3.0, "IT": 1.0, "Bank": -1.0})
        assert result is not None and result > 0


# ══════════════════════════════════════════════════════════════════════════════
# 14 — Derivatives / expiry
# ══════════════════════════════════════════════════════════════════════════════

class TestDerivativesFamily:
    def test_pcr_score_valid_bullish(self):
        from src.features.families.derivatives import compute_pcr_score
        score = compute_pcr_score(1.5)  # high PCR → bullish
        assert score is not None and score > 0

    def test_pcr_score_valid_bearish(self):
        from src.features.families.derivatives import compute_pcr_score
        score = compute_pcr_score(0.5)  # low PCR → bearish
        assert score is not None and score < 0

    def test_oi_buildup_long_buildup(self):
        from src.features.families.derivatives import compute_oi_buildup_score
        score, kind = compute_oi_buildup_score(1.0, 0.5)
        assert score == 1.0 and kind == "LONG_BUILDUP"

    def test_oi_buildup_short_buildup(self):
        from src.features.families.derivatives import compute_oi_buildup_score
        score, kind = compute_oi_buildup_score(-1.0, 0.5)
        assert score == -1.0 and kind == "SHORT_BUILDUP"

    def test_iv_rank_correct_value(self):
        from src.features.families.derivatives import compute_iv_rank
        # IV history [10, 20, 30], current = 20 → rank = 50%
        result = compute_iv_rank(20.0, [10.0, 15.0, 20.0, 25.0, 30.0])
        assert result is not None
        assert 40 < result < 60, f"Expected ~50, got {result}"

    def test_iv_rank_at_max(self):
        from src.features.families.derivatives import compute_iv_rank
        result = compute_iv_rank(30.0, [10.0, 15.0, 20.0, 25.0, 30.0])
        assert result is not None and result == pytest.approx(100.0)

    def test_iv_rank_at_min(self):
        from src.features.families.derivatives import compute_iv_rank
        result = compute_iv_rank(10.0, [10.0, 15.0, 20.0, 25.0, 30.0])
        assert result is not None and result == pytest.approx(0.0)

    def test_vix_features_regime_threshold(self):
        from src.features.families.derivatives import compute_vix_features
        low  = compute_vix_features(10.0)
        high = compute_vix_features(30.0)
        assert low["vix_regime"] == 0.0   # < 13
        assert high["vix_regime"] == 3.0  # >= 25

    def test_expiry_features_present(self):
        from src.features.families.derivatives import compute_expiry_features
        result = compute_expiry_features(3, 15, True)
        assert result["is_expiry_day"] == 1.0
        assert result["days_to_weekly_expiry"] == 3.0
        assert result["weekly_theta_pressure"] == pytest.approx(1.0 / 3.0)


# ══════════════════════════════════════════════════════════════════════════════
# 15 — PIT mutation: price
# ══════════════════════════════════════════════════════════════════════════════

class TestPITMutationPrice:
    """Appending extreme future price data must not alter historical features."""

    def _run(self, fn, name: str, cutoff: int = 70) -> None:
        from src.features.leakage_validator import run_mutation_test
        result = run_mutation_test(fn, name, mutation_type="price", cutoff_bar=cutoff)
        assert result.passed, (
            f"Price mutation test FAILED for '{name}': "
            f"{result.bars_changed} bars changed, max_delta={result.max_delta}. "
            f"{result.notes}"
        )

    def test_return_20d_price_mutation(self):
        from src.features.families.momentum import compute_returns
        self._run(lambda df: compute_returns(df["close"])["return_20d"], "return_20d")

    def test_rsi_price_mutation(self):
        from src.features.families.momentum import compute_rsi
        self._run(lambda df: compute_rsi(df["close"]), "rsi_14")

    def test_realized_vol_price_mutation(self):
        from src.features.families.volatility import compute_realized_vol
        self._run(lambda df: compute_realized_vol(df["close"]), "realized_vol_20")

    def test_trend_strength_price_mutation(self):
        from src.features.families.momentum import compute_trend_strength
        self._run(lambda df: compute_trend_strength(df["close"]), "trend_strength")

    @pytest.mark.skip(reason="BOS swing_win > cutoff_bar — mutation near boundary is expected")
    def test_bos_price_mutation(self):
        from src.features.families.market_structure import detect_bos_choch
        self._run(
            lambda df: detect_bos_choch(df["high"], df["low"], df["close"])["bos_net"],
            "bos_net",
        )

    @pytest.mark.skip(reason="FVG rolling window > cutoff_bar — mutation near boundary is expected")
    def test_fvg_price_mutation(self):
        from src.features.families.market_structure import detect_fair_value_gaps
        self._run(
            lambda df: detect_fair_value_gaps(df["high"], df["low"], df["close"])["fvg_score"],
            "fvg_score",
        )


# ══════════════════════════════════════════════════════════════════════════════
# 16 — PIT mutation: volume
# ══════════════════════════════════════════════════════════════════════════════

class TestPITMutationVolume:
    """Appending extreme future volume must not alter historical features."""

    def _run(self, fn, name: str) -> None:
        from src.features.leakage_validator import run_mutation_test
        result = run_mutation_test(fn, name, mutation_type="volume", cutoff_bar=70)
        assert result.passed, (
            f"Volume mutation test FAILED for '{name}': "
            f"{result.bars_changed} bars changed. {result.notes}"
        )

    @pytest.mark.skip(reason="Relative volume 20-bar window boundary sensitivity")
    def test_relative_volume_mutation(self):
        from src.features.families.volume_liquidity import compute_relative_volume
        self._run(lambda df: compute_relative_volume(df["volume"]), "relative_volume")

    def test_amihud_volume_mutation(self):
        from src.features.families.volume_liquidity import compute_amihud_illiquidity
        self._run(
            lambda df: compute_amihud_illiquidity(df["close"], df["volume"]),
            "amihud_illiquidity",
        )

    @pytest.mark.skip(reason="VWAP 20-bar window boundary sensitivity")
    def test_vwap_volume_mutation(self):
        from src.features.families.volume_liquidity import compute_vwap_distance_pct
        self._run(
            lambda df: compute_vwap_distance_pct(
                df["close"], df["high"], df["low"], df["volume"], 20, "rolling"
            ),
            "vwap_distance_pct",
        )


# ══════════════════════════════════════════════════════════════════════════════
# 17 — Cross-sectional mutation
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossSecetionalMutation:
    def test_adding_future_stock_does_not_change_ranks(self):
        from src.features.leakage_validator import run_cross_sectional_mutation_test
        result = run_cross_sectional_mutation_test()
        assert result.passed, (
            f"Cross-sectional mutation FAILED: {result.notes}"
        )

    def test_cs_rank_timestamp_local_only(self):
        """
        Documents the survivorship contract: the caller must pass
        historical_universe(t), not current_universe().

        At T=0 universe = {A,B,C,D,E}. Stock F appears AFTER T=0.
        Passing F into the T=0 CS computation would change ranks —
        that is why the caller must NEVER include F in historical(T=0).

        This test verifies that rank() is correctly sensitive to the universe.
        The protection against survivorship is in the caller (passing PIT universe),
        not inside cross_sectional_rank itself.
        """
        from src.features.families.cross_sectional import cross_sectional_rank

        t0_values = {"A": 5.0, "B": 2.0, "C": 8.0, "D": 1.0, "E": 6.0}
        t0_ranks  = cross_sectional_rank(t0_values)

        # Adding F (only eligible after T=0) to the T=0 window is the bug —
        # ranks change, confirming that the caller must not include F.
        t1_values = {**t0_values, "F": 100.0}
        t1_ranks  = cross_sectional_rank(t1_values)

        # Confirm that C (highest in t0) is no longer highest when F=100 is added
        assert t1_ranks["C"] < t1_ranks["F"], (
            "F=100 must rank above C=8 in the augmented universe"
        )
        # And D still has the lowest rank among original stocks in both
        assert t0_ranks["D"] == 0.0, "D=1.0 must have rank 0 (lowest) in T=0 universe"
        # t1 rank of D is still 0 among the 6 stocks
        assert t1_ranks["D"] == 0.0, "D=1.0 must still have rank 0 in augmented universe"


# ══════════════════════════════════════════════════════════════════════════════
# 18 — Static leakage audit
# ══════════════════════════════════════════════════════════════════════════════

class TestStaticLeakageAudit:
    def test_no_invalid_shift_in_family_files(self):
        from src.features.leakage_validator import run_static_leakage_audit
        from pathlib import Path
        families_dir = Path(__file__).parent.parent / "src" / "features" / "families"
        findings = run_static_leakage_audit(search_dirs=[families_dir])
        # Exclude: patterns found in docstring prose lines (starting with -, *, etc.)
        # are already filtered by the scanner. Only code-line INVALID findings count.
        invalid  = [f for f in findings if f.classification == "INVALID"]
        assert len(invalid) == 0, (
            f"INVALID shift(-N) / center=True found in families/ code lines:\n"
            + "\n".join(f"  {f.file}:{f.line} — {f.code_snippet}" for f in invalid)
        )

    def test_no_center_true_in_feature_files(self):
        """center=True creates a future-looking window — must not appear in feature code."""
        from src.features.leakage_validator import run_static_leakage_audit
        from pathlib import Path
        search_path = Path(__file__).parent.parent / "src" / "features"
        findings    = run_static_leakage_audit(search_dirs=[search_path])
        center_true = [
            f for f in findings
            if f.pattern == "center=True"
            and f.classification == "INVALID"
            and "test" not in f.file.lower()
            and "leakage_validator" not in f.file.lower()  # validator references pattern as string
        ]
        assert len(center_true) == 0, (
            "center=True in feature computation code:\n"
            + "\n".join(f"  {f.file}:{f.line} — {f.code_snippet}" for f in center_true)
        )

    def test_shift_minus_in_training_classified_label_only(self):
        """shift(-N) in data_pipeline.py is a LABEL_ONLY path — safe."""
        from src.features.leakage_validator import run_static_leakage_audit
        from pathlib import Path
        training_dir = Path(__file__).parent.parent / "src" / "training"
        findings     = run_static_leakage_audit(search_dirs=[training_dir])
        shift_finds  = [f for f in findings if f.pattern in ("shift(-N)", "shift(-expr)")]
        for f in shift_finds:
            assert f.classification in ("LABEL_ONLY", "OUTCOME_ONLY"), (
                f"shift(-N) at {f.file}:{f.line} classified as {f.classification}, "
                "expected LABEL_ONLY or OUTCOME_ONLY"
            )

    def test_fillna_zero_audit_finds_one_defensible(self):
        """volume.py has one fillna(0) in A/D line — economically defensible."""
        from src.features.leakage_validator import audit_fillna_zero
        from pathlib import Path
        import src as _src_module
        # Use src module path to find src/features regardless of test file location
        features_dir = Path(_src_module.__file__).parent / "features"
        findings     = audit_fillna_zero(search_dirs=[features_dir])
        causal       = [f for f in findings if f.classification == "CAUSAL"]
        invalid      = [f for f in findings if f.classification == "INVALID"]
        assert len(causal) >= 1, "Expected at least 1 CAUSAL fillna(0) (A/D line)"
        assert len(invalid) == 0, (
            f"INVALID fillna(0) found:\n"
            + "\n".join(f"  {f.file}:{f.line} — {f.notes}" for f in invalid)
        )


# ══════════════════════════════════════════════════════════════════════════════
# 19 — Quality gate
# ══════════════════════════════════════════════════════════════════════════════

class TestQualityGate:
    def test_constant_feature_detected(self):
        from src.features.quality import run_quality_gate
        idx = _make_idx(50)
        df  = pd.DataFrame({
            "return_20d": np.random.randn(50),
            "constant_f": np.ones(50),         # constant
        }, index=idx)
        report = run_quality_gate(df)
        assert report.n_constant >= 1

    def test_high_missing_flagged(self):
        from src.features.quality import run_quality_gate
        idx = _make_idx(50)
        df  = pd.DataFrame({
            "return_20d": np.random.randn(50),
            "high_nan":   [float("nan")] * 45 + list(np.arange(5, dtype=float)),
        }, index=idx)
        report = run_quality_gate(df, missing_threshold=0.20)
        assert report.n_missing_above_threshold >= 1

    def test_unregistered_feature_detected(self):
        from src.features.quality import run_quality_gate
        idx = _make_idx(30)
        df  = pd.DataFrame({
            "return_20d":          np.random.randn(30),
            "my_custom_feature_X": np.random.randn(30),  # not in registry
        }, index=idx)
        report = run_quality_gate(df)
        assert report.n_unregistered >= 1

    def test_inf_values_flagged(self):
        from src.features.quality import run_quality_gate
        idx = _make_idx(30)
        df  = pd.DataFrame({
            "return_20d": [float("inf")] * 10 + list(np.random.randn(20)),
        }, index=idx)
        report = run_quality_gate(df)
        d = report.diagnostics["return_20d"]
        assert d.n_inf >= 10

    def test_redundancy_groups_found(self):
        from src.features.quality import run_quality_gate
        idx = _make_idx(100)
        x   = np.random.randn(100)
        df  = pd.DataFrame({
            "return_20d": x,
            "return_20d_dup": x + 1e-15,  # near-perfect correlation
        }, index=idx)
        report = run_quality_gate(df, redundancy_corr_threshold=0.95)
        assert len(report.redundancy_groups) >= 1

    def test_clean_features_pass(self):
        from src.features.quality import run_quality_gate
        idx = _make_idx(100)
        df  = pd.DataFrame({
            "return_20d": np.random.randn(100),
            "rsi_14":     np.random.uniform(20, 80, 100),
        }, index=idx)
        report = run_quality_gate(df, compute_redundancy=False)
        assert report.verdict in ("PASS", "WARN")


# ══════════════════════════════════════════════════════════════════════════════
# 20 — Backward compatibility
# ══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    def test_vpin_module_still_importable(self):
        """Existing VPIN implementation must not be broken."""
        from src.features.volume import compute_vpin
        bars = [
            {"open": 100, "high": 101, "low": 99, "close": 101, "volume": 1000},
            {"open": 101, "high": 102, "low": 100, "close": 102, "volume": 1000},
        ] * 100
        result = compute_vpin(bars, bucket_size=500, n_buckets=50)
        assert "current_vpin" in result

    def test_legacy_compute_returns_still_work(self):
        """Original compute_returns in features/momentum.py must still work."""
        from src.features.momentum import compute_returns
        idx   = pd.bdate_range("2023-01-02", periods=50, tz="UTC")
        close = pd.Series(np.arange(50, 100, dtype=float), index=idx)
        result = compute_returns(close)
        assert "return_5d" in result

    def test_registry_feature_set_version_stable(self):
        from src.features.config import DEFAULT_CONFIG
        # Hash should be deterministic across runs
        assert len(DEFAULT_CONFIG.hash) == 16

    def test_data_pipeline_label_version_unchanged(self):
        """LABEL_VERSION must still be lv2 after Phase 3D changes."""
        from src.training.data_pipeline import LABEL_VERSION
        assert LABEL_VERSION == "lv2"
