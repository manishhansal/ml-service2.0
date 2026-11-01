"""
SDScanner — top-level orchestrator for the Supply & Demand Zone Scanner.

Pipeline per symbol:
  detect_sd_zones → update_sd_lifecycle (for existing zones) → rank_sd_zones

Supported timeframes: "5m", "15m", "1h", "1d"
"""
from __future__ import annotations
import time
from datetime import datetime, timezone
import pandas as pd
import structlog
from src.scanners.supply_demand.config import SDConfig
from src.scanners.supply_demand.detector import detect_sd_zones
from src.scanners.supply_demand.zone_manager import update_sd_lifecycle
from src.scanners.supply_demand.scorer import rank_sd_zones
from src.scanners.supply_demand.schemas import SDLifecycle, SDScanResult, SDZone

_logger = structlog.get_logger(__name__)
SUPPORTED_TIMEFRAMES = frozenset({"5m", "15m", "1h", "1d"})


class SDScanner:
    def __init__(self, config: SDConfig | None = None) -> None:
        self.config = config or SDConfig()

    def scan(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        timeframe: str,
        existing_zones: dict[str, list[SDZone]] | None = None,
    ) -> SDScanResult:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Valid values: 5m, 15m, 1h, 1d")

        start_ts = time.time()
        all_zones: list[SDZone] = []

        for symbol, df in ohlcv_by_symbol.items():
            try:
                new_zones = detect_sd_zones(df, symbol, timeframe, self.config)

                if existing_zones and existing_zones.get(symbol) and len(df) > 0:
                    last = df.iloc[-1]
                    bar_idx = len(df) - 1
                    updated_existing = [
                        update_sd_lifecycle(
                            z, float(last["low"]), float(last["high"]),
                            float(last["close"]), bar_idx, self.config
                        )
                        for z in existing_zones[symbol]
                    ]
                    active_existing = [
                        z for z in updated_existing
                        if z.lifecycle_state != SDLifecycle.BROKEN
                    ]
                    all_zones.extend(active_existing)

                all_zones.extend(new_zones)
            except Exception as exc:
                _logger.error("sd_scanner_symbol_error", symbol=symbol, error=str(exc))

        all_zones = rank_sd_zones(all_zones, ohlcv_by_symbol, self.config)

        # Compute intraday trading levels
        from src.scanners.intraday import compute_zone_intraday
        from src.scanners.supply_demand.schemas import SDDirection

        intraday_zones: list[SDZone] = []
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
                is_bullish = z.direction == SDDirection.DEMAND
                levels = compute_zone_intraday(is_bullish, z.zone_high, z.zone_low, current_close, atr)
                intraday_zones.append(z.model_copy(update=levels))
            else:
                intraday_zones.append(z)
        all_zones = intraday_zones

        # ── Scanner-specific pre-filter (fresh_zones_only) ────────────────────
        if self.config.fresh_zones_only:
            all_zones = [z for z in all_zones if z.lifecycle_state == SDLifecycle.FRESH]

        # ── Quality gate (trend + liquidity + R:R) ────────────────────────────
        from src.scanners.quality import (
            compute_market_direction,
            compute_symbol_trend,
            passes_liquidity_filter,
            passes_quality_filters,
        )
        _market_dir = compute_market_direction(ohlcv_by_symbol)
        _qf: list[SDZone] = []
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
            "sd_scan_complete",
            symbols=len(ohlcv_by_symbol),
            n_zones=len(all_zones),
            elapsed_ms=int((time.time() - start_ts) * 1000),
        )

        return SDScanResult(
            zones=all_zones,
            scanned_at=datetime.now(tz=timezone.utc),
            n_symbols_scanned=len(ohlcv_by_symbol),
        )
