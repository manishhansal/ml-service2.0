"""
src.features.factory — batch FeatureFactory for training/backtest (Phase C).

Whereas ``FeaturePipeline`` builds a single PIT-correct feature vector for live
inference, ``FeatureFactory`` computes a full time-series feature MATRIX over an
entire OHLCV history — one row per bar — for dataset construction and training.

Every feature here has:
  - a mathematical definition (documented inline)
  - a defined source (OHLCV column(s))
  - PIT semantics (uses ONLY past/current bars; shift where needed)
  - missing-value semantics (NaN when insufficient history — NEVER zero-filled)

Feature groups (07_FEATURE_SPECIFICATION):
  price, returns, volatility, volume, VWAP, momentum, trend, range,
  candle-structure. Cross-sectional / derivatives / OI / IV / news features
  are added by the dataset builder when those data families are available;
  when unavailable a feature is left as NaN and flagged in the availability
  metadata rather than fabricated.

Requirements: Phase C, 07_FEATURE_SPECIFICATION, Req 2.x.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

FEATURE_SCHEMA_VERSION = "fs-2.0.0"


@dataclass
class FeatureAvailability:
    """Per-feature availability metadata (Phase 14)."""

    total_rows: int = 0
    per_feature_missing: dict[str, int] = field(default_factory=dict)
    unavailable_families: list[str] = field(default_factory=list)

    def missing_fraction(self, feature: str) -> float:
        if self.total_rows == 0:
            return 1.0
        return self.per_feature_missing.get(feature, 0) / self.total_rows


class FeatureFactory:
    """
    Computes a PIT-safe feature matrix from a clean OHLCV DataFrame.

    Usage::

        factory = FeatureFactory()
        features, availability = factory.build(ohlcv_df)
        # features: DataFrame aligned to ohlcv_df.index, one column per feature
    """

    # Ordered, explicit feature list (schema contract).
    FEATURE_NAMES: list[str] = [
        # price / returns
        "ret_1", "ret_5", "ret_10", "ret_20",
        "log_ret_1",
        # volatility
        "vol_5", "vol_10", "vol_20", "atr_14_pct",
        # volume
        "rel_volume_20", "volume_zscore_20",
        # vwap
        "vwap_distance_pct",
        # momentum / oscillators
        "rsi_14", "macd_hist", "stoch_k_14",
        # trend
        "ema_5_20", "ema_10_50", "adx_14",
        # range / candle structure
        "hl_range_pct", "close_position", "gap_pct",
        # bollinger
        "bb_zscore_20",
        # rolling stats
        "skew_20", "kurt_20",
    ]

    def build(
        self, ohlcv: pd.DataFrame
    ) -> tuple[pd.DataFrame, FeatureAvailability]:
        """
        Build the full feature matrix.

        Returns a tuple (features_df, availability). Rows with insufficient
        lookback contain NaN for the affected features — the caller decides
        whether to drop or impute (imputation is explicit, never silent-zero).
        """
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(ohlcv.columns)
        if missing:
            raise ValueError(f"FeatureFactory requires OHLCV columns; missing {missing}")

        df = ohlcv.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)

        out = pd.DataFrame(index=df.index)
        rets = close.pct_change()

        # ── price / returns ────────────────────────────────────────────────
        out["ret_1"] = rets
        out["ret_5"] = close.pct_change(5)
        out["ret_10"] = close.pct_change(10)
        out["ret_20"] = close.pct_change(20)
        out["log_ret_1"] = np.log(close / close.shift(1))

        # ── volatility ─────────────────────────────────────────────────────
        out["vol_5"] = rets.rolling(5).std()
        out["vol_10"] = rets.rolling(10).std()
        out["vol_20"] = rets.rolling(20).std()
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        out["atr_14_pct"] = tr.rolling(14).mean() / close

        # ── volume ─────────────────────────────────────────────────────────
        vol_ma20 = volume.rolling(20).mean()
        out["rel_volume_20"] = volume / vol_ma20
        vol_std20 = volume.rolling(20).std()
        out["volume_zscore_20"] = (volume - vol_ma20) / vol_std20

        # ── vwap (rolling typical-price VWAP) ──────────────────────────────
        typical = (high + low + close) / 3.0
        rolling_pv = (typical * volume).rolling(20).sum()
        rolling_v = volume.rolling(20).sum()
        vwap = rolling_pv / rolling_v
        out["vwap_distance_pct"] = (close - vwap) / vwap

        # ── momentum / oscillators ──────────────────────────────────────────
        out["rsi_14"] = self._rsi(close, 14)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        out["macd_hist"] = (macd_line - macd_signal) / close
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        out["stoch_k_14"] = (close - low14) / (high14 - low14)

        # ── trend ──────────────────────────────────────────────────────────
        ema5 = close.ewm(span=5, adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema10 = close.ewm(span=10, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        out["ema_5_20"] = ema5 / ema20 - 1.0
        out["ema_10_50"] = ema10 / ema50 - 1.0
        out["adx_14"] = self._adx(high, low, close, 14)

        # ── range / candle structure ───────────────────────────────────────
        out["hl_range_pct"] = (high - low) / close
        out["close_position"] = (close - low) / (high - low)
        out["gap_pct"] = (open_ - close.shift(1)) / close.shift(1)

        # ── bollinger ──────────────────────────────────────────────────────
        ma20 = close.rolling(20).mean()
        sd20 = close.rolling(20).std()
        out["bb_zscore_20"] = (close - ma20) / sd20

        # ── rolling higher moments ──────────────────────────────────────────
        out["skew_20"] = rets.rolling(20).skew()
        out["kurt_20"] = rets.rolling(20).kurt()

        # Replace inf with NaN (division edge cases) — NEVER zero.
        out = out.replace([np.inf, -np.inf], np.nan)

        # Enforce the declared schema order / membership.
        out = out[self.FEATURE_NAMES]

        availability = self._availability(out)
        return out, availability

    # ── Availability metadata ───────────────────────────────────────────────

    def _availability(self, features: pd.DataFrame) -> FeatureAvailability:
        avail = FeatureAvailability(total_rows=len(features))
        for col in features.columns:
            avail.per_feature_missing[col] = int(features[col].isna().sum())
        return avail

    # ── Indicators ────────────────────────────────────────────────────────

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta).clip(lower=0).rolling(window).mean()
        rs = gain / loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window).mean()
        plus_di = 100.0 * plus_dm.rolling(window).mean() / atr
        minus_di = 100.0 * minus_dm.rolling(window).mean() / atr
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        return dx.rolling(window).mean()
