"""
FVGScanner — top-level orchestrator for the Fair Value Gap Scanner.

Pipeline per symbol:
  detect_fvg_zones → update_fvg_lifecycle (for existing zones) → rank_fvg_zones

Supported timeframes: "5m", "15m", "1h", "1d"
Per-timeframe expiry bar overrides are applied before calling the zone manager.
"""
from __future__ import annotations
import time
from datetime import datetime, timezone
import pandas as pd
import structlog
from src.scanners.fair_value_gap.config import FVGConfig
from src.scanners.fair_value_gap.detector import detect_fvg_zones
from src.scanners.fair_value_gap.zone_manager import update_fvg_lifecycle
from src.scanners.fair_value_gap.scorer import rank_fvg_zones
from src.scanners.fair_value_gap.schemas import FVGLifecycle, FVGScanResult, FVGZone

_logger = structlog.get_logger(__name__)
SUPPORTED_TIMEFRAMES = frozenset({"5m", "15m", "1h", "1d"})


class FVGScanner:
    def __init__(self, config: FVGConfig | None = None) -> None:
        self.config = config or FVGConfig()

    def scan(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        timeframe: str,
        existing_zones: dict[str, list[FVGZone]] | None = None,
    ) -> FVGScanResult:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Valid values: 5m, 15m, 1h, 1d")

        # Per-timeframe expiry override
        tf_expiry = {
            "5m":  self.config.tf_5m_expiry_bars,
            "15m": self.config.tf_15m_expiry_bars,
            "1h":  self.config.tf_1h_expiry_bars,
            "1d":  self.config.tf_1d_expiry_bars,
        }
        effective_config = self.config.model_copy(update={"fvg_expiry_bars": tf_expiry[timeframe]})

        start_ts = time.time()
        all_zones: list[FVGZone] = []

        for symbol, df in ohlcv_by_symbol.items():
            try:
                new_zones = detect_fvg_zones(df, symbol, timeframe, effective_config)

                # Update lifecycle of existing zones
                if existing_zones and existing_zones.get(symbol) and len(df) > 0:
                    last = df.iloc[-1]
                    bar_idx = len(df) - 1
                    updated_existing = [
                        update_fvg_lifecycle(
                            z, float(last["low"]), float(last["high"]),
                            float(last["close"]), bar_idx, z.candle3_bar_index,
                            effective_config
                        )
                        for z in existing_zones[symbol]
                    ]
                    active_existing = [
                        z for z in updated_existing
                        if z.lifecycle_state not in (FVGLifecycle.FILLED, FVGLifecycle.EXPIRED)
                    ]
                    all_zones.extend(active_existing)

                all_zones.extend(new_zones)
            except Exception as exc:
                _logger.error("fvg_scanner_symbol_error", symbol=symbol, error=str(exc))

        # Score and rank
        all_zones = rank_fvg_zones(all_zones, ohlcv_by_symbol, effective_config)

        # Compute intraday trading levels
        from src.scanners.intraday import compute_zone_intraday
        from src.scanners.fair_value_gap.schemas import FVGDirection

        intraday_zones: list[FVGZone] = []
        for z in all_zones:
            df = ohlcv_by_symbol.get(z.symbol)
            if df is not None and len(df) > 0:
                current_close = float(df["close"].iloc[-1])
                prev_close = df["close"].shift(1)
                tr = pd.concat(
                    [
                        df["high"] - df["low"],
                        (df["high"] - prev_close).abs(),
                        (df["low"] - prev_close).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                atr = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
                is_bullish = z.direction == FVGDirection.BULLISH
                levels = compute_zone_intraday(is_bullish, z.fvg_high, z.fvg_low, current_close, atr)
                intraday_zones.append(z.model_copy(update=levels))
            else:
                intraday_zones.append(z)
        all_zones = intraday_zones

        # ── Scanner-specific pre-filter (combined_score) ──────────────────────
        all_zones = [z for z in all_zones if z.combined_score >= self.config.min_combined_score]

        # ── Quality gate (trend + liquidity + R:R) ────────────────────────────
        from src.scanners.quality import (
            compute_market_direction,
            compute_symbol_trend,
            passes_liquidity_filter,
            passes_quality_filters,
        )
        _market_dir = compute_market_direction(ohlcv_by_symbol)
        _qf: list[FVGZone] = []
        for _z in all_zones:
            _df = ohlcv_by_symbol.get(_z.symbol)
            if _df is None:
                continue
            if not passes_liquidity_filter(_df, self.config.min_daily_value_crore):
                continue
            if self.config.require_trend_alignment:
                _trend = compute_symbol_trend(_df)
                if not passes_quality_filters(_z, _trend, _market_dir, self.config.min_rr_ratio):
                    continue
            elif (_z.intraday_rr_ratio or 0.0) < self.config.min_rr_ratio:
                continue
            _qf.append(_z)
        all_zones = _qf

        _logger.info(
            "fvg_scan_complete",
            symbols=len(ohlcv_by_symbol),
            n_zones=len(all_zones),
            elapsed_ms=int((time.time() - start_ts) * 1000),
        )

        return FVGScanResult(
            zones=all_zones,
            scanned_at=datetime.now(tz=timezone.utc),
            n_symbols_scanned=len(ohlcv_by_symbol),
        )
