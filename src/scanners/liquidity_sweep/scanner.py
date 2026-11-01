"""
SweepScanner — top-level orchestrator for the Liquidity Sweep Scanner.

Pipeline per symbol:
  detect_sweeps → update_sweep_lifecycle (for existing signals) → rank_sweeps

Supported timeframes: "5m", "15m", "1h", "1d"
"""
from __future__ import annotations
import time
from datetime import datetime, timezone
import pandas as pd
import structlog
from src.scanners.liquidity_sweep.config import SweepConfig
from src.scanners.liquidity_sweep.detector import detect_sweeps
from src.scanners.liquidity_sweep.signal_manager import update_sweep_lifecycle
from src.scanners.liquidity_sweep.scorer import rank_sweeps
from src.scanners.liquidity_sweep.schemas import SweepLifecycle, SweepScanResult, SweepSignal

_logger = structlog.get_logger(__name__)
SUPPORTED_TIMEFRAMES = frozenset({"5m", "15m", "1h", "1d"})


class SweepScanner:
    def __init__(self, config: SweepConfig | None = None) -> None:
        self.config = config or SweepConfig()

    def scan(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        timeframe: str,
        existing_signals: dict[str, list[SweepSignal]] | None = None,
    ) -> SweepScanResult:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Valid values: 5m, 15m, 1h, 1d")

        start_ts = time.time()
        all_signals: list[SweepSignal] = []

        for symbol, df in ohlcv_by_symbol.items():
            try:
                new_signals = detect_sweeps(df, symbol, timeframe, self.config)

                if existing_signals and existing_signals.get(symbol) and len(df) > 0:
                    last_close = float(df["close"].iloc[-1])
                    bar_idx = len(df) - 1
                    updated_existing = [
                        update_sweep_lifecycle(s, last_close, bar_idx, self.config)
                        for s in existing_signals[symbol]
                    ]
                    active_existing = [
                        s for s in updated_existing
                        if s.lifecycle_state != SweepLifecycle.INVALIDATED
                    ]
                    all_signals.extend(active_existing)

                all_signals.extend(new_signals)
            except Exception as exc:
                _logger.error("sweep_scanner_symbol_error", symbol=symbol, error=str(exc))

        all_signals = rank_sweeps(all_signals, self.config, ohlcv_by_symbol)

        # Compute intraday trading levels
        from src.scanners.intraday import compute_sweep_intraday
        from src.scanners.liquidity_sweep.schemas import SweepDirection

        intraday_signals: list[SweepSignal] = []
        for s in all_signals:
            df = ohlcv_by_symbol.get(s.symbol)
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
                is_bullish = s.direction == SweepDirection.BULLISH
                levels = compute_sweep_intraday(
                    is_bullish,
                    s.swept_level,
                    s.sweep_zone_high,
                    s.sweep_zone_low,
                    current_close,
                    atr,
                )
                intraday_signals.append(s.model_copy(update=levels))
            else:
                intraday_signals.append(s)
        all_signals = intraday_signals

        # ── Scanner-specific pre-filter (equal_levels_count) ──────────────────
        all_signals = [s for s in all_signals if s.equal_levels_count >= self.config.min_equal_levels_count]

        # ── Quality gate (trend + liquidity + R:R) ────────────────────────────
        from src.scanners.quality import (
            compute_market_direction,
            compute_symbol_trend,
            passes_liquidity_filter,
            passes_quality_filters,
        )
        _market_dir = compute_market_direction(ohlcv_by_symbol)
        _qf: list[SweepSignal] = []
        for _s in all_signals:
            _df = ohlcv_by_symbol.get(_s.symbol)
            if _df is None:
                continue
            if not passes_liquidity_filter(_df, self.config.min_daily_value_crore):
                continue
            if self.config.require_trend_alignment:
                _trend = compute_symbol_trend(_df)
                if not passes_quality_filters(_s, _trend, _market_dir, self.config.min_rr_ratio):
                    continue
            elif (_s.intraday_rr_ratio or 0.0) < self.config.min_rr_ratio:
                continue
            _qf.append(_s)
        all_signals = _qf

        _logger.info(
            "sweep_scan_complete",
            symbols=len(ohlcv_by_symbol),
            n_signals=len(all_signals),
            elapsed_ms=int((time.time() - start_ts) * 1000),
        )

        return SweepScanResult(
            signals=all_signals,
            scanned_at=datetime.now(tz=timezone.utc),
            n_symbols_scanned=len(ohlcv_by_symbol),
        )
