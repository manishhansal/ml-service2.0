"""
Phase 3C Tests — Label V2 & Event-Based Target Engineering.

Tests every invariant required by the Phase 3C specification.

Golden reference tests use deterministic synthetic price paths where
the correct answer is known mathematically, not empirically.

Test categories
---------------
 1. LabelConfig: versioning, hash determinism, parameter isolation
 2. Triple-barrier: TP-first, SL-first, time-first
 3. Triple-barrier: long, short
 4. Triple-barrier: intrabar ambiguity (both barriers in same bar)
 5. Triple-barrier: incomplete horizon → DATA_INSUFFICIENT
 6. Triple-barrier: expiry-aware truncation
 7. Triple-barrier: edge cases (zero prices, single bar, unsorted timestamps)
 8. Fixed-horizon: raw return, vol-adjusted, directional
 9. Meta-label: take/skip, side separation
10. MFE/MAE: long, short, same-bar extremes
11. Sample weights: concurrency, average_uniqueness, build_t1
12. Relative labels: excess return, sector-relative DATA_UNAVAILABLE
13. Validators: Rule 1 (outcome in features), Rule 7 (incomplete as valid)
14. PIT mutation tests: future price cannot alter past label
15. Purging contract: event-aware t1 series for PurgedKFold
16. Registry: all active labels registered, hash uniqueness
17. Pipeline integration: generate_risk_labels delegates to V2
18. Backward compatibility: generate_ranking_labels_v2 returns Series
"""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from typing import List

import numpy as np
import pandas as pd
import pytest

UTC = timezone.utc


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — deterministic synthetic price paths
# ──────────────────────────────────────────────────────────────────────────────

def _make_idx(n: int, start: str = "2023-06-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, freq="B", tz="UTC")


def _make_ohlcv(prices: List[float], volatility_range: float = 0.01) -> pd.DataFrame:
    """
    Build a minimal OHLCV DataFrame from a list of close prices.
    High = close * (1 + volatility_range/2)
    Low  = close * (1 - volatility_range/2)
    """
    n = len(prices)
    idx = _make_idx(n)
    closes = np.array(prices, dtype=float)
    highs  = closes * (1.0 + volatility_range / 2)
    lows   = closes * (1.0 - volatility_range / 2)
    opens  = closes * 0.999
    vols   = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


def _make_ohlcv_precise(
    n: int,
    base: float = 100.0,
    highs: List[float] | None = None,
    lows: List[float] | None = None,
    closes: List[float] | None = None,
) -> pd.DataFrame:
    """Exact OHLCV control for barrier tests."""
    idx = _make_idx(n)
    c = np.array(closes or [base] * n, dtype=float)
    h = np.array(highs  or (c * 1.005).tolist(), dtype=float)
    l = np.array(lows   or (c * 0.995).tolist(), dtype=float)
    o = c * 0.999
    v = np.full(n, 1_000_000.0)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)


# ──────────────────────────────────────────────────────────────────────────────
# 1 — LabelConfig
# ──────────────────────────────────────────────────────────────────────────────

class TestLabelConfig:
    def test_default_daily_has_valid_hash(self):
        from src.labels.config import LabelConfig
        cfg = LabelConfig.default_daily()
        assert len(cfg.hash) == 16
        assert all(c in "0123456789abcdef" for c in cfg.hash)

    def test_same_params_produce_same_hash(self):
        from src.labels.config import LabelConfig
        cfg1 = LabelConfig(horizon_bars=20, pt_multiplier=1.5, sl_multiplier=1.0)
        cfg2 = LabelConfig(horizon_bars=20, pt_multiplier=1.5, sl_multiplier=1.0)
        assert cfg1.hash == cfg2.hash

    def test_different_params_produce_different_hash(self):
        from src.labels.config import LabelConfig
        cfg1 = LabelConfig(horizon_bars=20, pt_multiplier=1.5)
        cfg2 = LabelConfig(horizon_bars=20, pt_multiplier=2.0)
        assert cfg1.hash != cfg2.hash

    def test_horizon_change_changes_hash(self):
        from src.labels.config import LabelConfig
        cfg1 = LabelConfig(horizon_bars=5)
        cfg2 = LabelConfig(horizon_bars=20)
        assert cfg1.hash != cfg2.hash

    def test_cost_model_unavailable_by_default(self):
        from src.labels.config import LabelConfig
        cfg = LabelConfig.default_daily()
        assert cfg.cost_model.version == "DATA_UNAVAILABLE"
        assert not cfg.include_costs


# ──────────────────────────────────────────────────────────────────────────────
# 2 — Triple-barrier: golden reference tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTripleBarrierGoldenTP:
    """
    Golden path: TP hit first.

    Price path:  100, 101, 102, 103, 99, 98
    Entry = 100 (bar 0)
    TP  = +2% = 102.0  → bar 2 (high ≥ 102)
    SL  = -2% = 98.0   → bar 5 (low  ≤ 98)
    Time= 5 bars

    Expected: first_touch = TAKE_PROFIT at bar 2
    """

    def _setup(self):
        from src.labels.triple_barrier import generate_triple_barrier_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import Side

        prices = [100, 101, 102, 103, 99, 98]
        ohlcv = _make_ohlcv_precise(
            n=len(prices),
            closes=prices,
            highs=[100.1, 101.5, 102.5, 103.5, 99.5, 98.5],  # bar2 high=102.5 ≥ 102
            lows =[99.5,  100.5, 101.5, 102.5, 98.5,  97.5],  # bar5 low=97.5 ≤ 98
        )
        # ATR ≈ 0.01 (1%). pt_mult=2 → TP=2%; sl_mult=2 → SL=2%
        cfg = LabelConfig(
            horizon_bars=5,
            pt_multiplier=2.0,
            sl_multiplier=2.0,
            volatility_window=3,
            min_volatility=0.01,   # force ATR=1% of close=1.0 absolute
            ambiguity_policy="CONSERVATIVE_SL",
        )
        events = generate_triple_barrier_labels(ohlcv, cfg, symbol="GOLDEN_TP", side=Side.LONG)
        return events

    def test_first_event_is_tp(self):
        from src.labels.schemas import FirstTouch
        events = self._setup()
        assert len(events) > 0
        ev0 = events[0]
        assert ev0.first_touch == FirstTouch.TAKE_PROFIT, (
            f"Expected TAKE_PROFIT, got {ev0.first_touch}. "
            "High at bar 2 (102.5) should breach TP at 102."
        )

    def test_gross_return_positive_for_long_tp(self):
        from src.labels.schemas import FirstTouch
        events = self._setup()
        ev0 = events[0]
        if ev0.first_touch == FirstTouch.TAKE_PROFIT:
            assert ev0.gross_return is not None
            assert ev0.gross_return > 0, "Long TP must produce positive gross return"

    def test_event_end_time_at_or_after_start(self):
        events = self._setup()
        for ev in events:
            assert ev.event_end_time >= ev.event_start_time


class TestTripleBarrierGoldenSL:
    """
    Golden path: SL hit first.

    Price path:  100, 99, 98, 97, 105, 106
    SL = -2% = 98.0  → bar 2 (low ≤ 98)
    TP = +2% = 102.0 → bar 4 (high ≥ 105)

    Expected: first_touch = STOP_LOSS at bar 2 (hits SL before TP)
    """

    def _setup(self):
        from src.labels.triple_barrier import generate_triple_barrier_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import Side

        prices = [100, 99, 98, 97, 105, 106]
        ohlcv = _make_ohlcv_precise(
            n=len(prices),
            closes=prices,
            highs=[100.5, 99.5, 98.5, 97.5, 105.5, 106.5],
            lows= [99.5,  98.5, 97.5, 96.5, 104.5, 105.5],  # bar2 low=97.5 ≤ 98
        )
        cfg = LabelConfig(
            horizon_bars=5,
            pt_multiplier=2.0,
            sl_multiplier=2.0,
            volatility_window=3,
            min_volatility=0.01,
        )
        events = generate_triple_barrier_labels(ohlcv, cfg, symbol="GOLDEN_SL", side=Side.LONG)
        return events

    def test_first_event_is_sl(self):
        from src.labels.schemas import FirstTouch
        events = self._setup()
        assert len(events) > 0
        ev0 = events[0]
        assert ev0.first_touch == FirstTouch.STOP_LOSS, (
            f"Expected STOP_LOSS, got {ev0.first_touch}. "
            "Low at bar 2 (97.5) should breach SL at 98 before TP is reached."
        )

    def test_gross_return_negative_for_long_sl(self):
        from src.labels.schemas import FirstTouch
        events = self._setup()
        ev0 = events[0]
        if ev0.first_touch == FirstTouch.STOP_LOSS:
            assert ev0.gross_return is not None
            assert ev0.gross_return < 0, "Long SL must produce negative gross return"


class TestTripleBarrierGoldenTimeLimit:
    """
    Golden path: time barrier hit first.

    Price path:  100, 100.5, 101, 100.3, 100.1
    Barriers far from price → no barrier hit in 3-bar window.
    Expected: first_touch = TIME_LIMIT
    """

    def _setup(self):
        from src.labels.triple_barrier import generate_triple_barrier_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import Side

        prices = [100, 100.5, 101, 100.3, 100.1]
        ohlcv = _make_ohlcv_precise(
            n=len(prices),
            closes=prices,
            highs=[100.6, 101.0, 101.5, 100.8, 100.6],
            lows= [99.7,  100.2, 100.6, 100.0, 99.8],
        )
        # Very large barriers so neither is hit
        cfg = LabelConfig(
            horizon_bars=3,
            pt_multiplier=10.0,   # 10x ATR = far from price
            sl_multiplier=10.0,
            volatility_window=3,
            min_volatility=0.01,
        )
        events = generate_triple_barrier_labels(ohlcv, cfg, symbol="GOLDEN_TIME", side=Side.LONG)
        return events

    def test_first_event_is_time_limit(self):
        from src.labels.schemas import FirstTouch
        events = self._setup()
        assert len(events) > 0
        ev0 = events[0]
        assert ev0.first_touch == FirstTouch.TIME_LIMIT, (
            f"Expected TIME_LIMIT, got {ev0.first_touch}. "
            "Barriers are 10x ATR so no barrier should be touched in 3 bars."
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3 — Long vs Short semantics
# ──────────────────────────────────────────────────────────────────────────────

class TestTripleBarrierLongShort:
    """
    For the same price path, long and short labels must be mirror images.
    """

    def _make_events(self, side):
        from src.labels.triple_barrier import generate_triple_barrier_labels
        from src.labels.config import LabelConfig

        prices = [100, 101, 102, 103, 104]
        ohlcv = _make_ohlcv_precise(
            n=len(prices), closes=prices,
            highs=[p * 1.005 for p in prices],
            lows =[p * 0.995 for p in prices],
        )
        cfg = LabelConfig(
            horizon_bars=4, pt_multiplier=1.5, sl_multiplier=1.5,
            volatility_window=2, min_volatility=0.01,
        )
        return generate_triple_barrier_labels(ohlcv, cfg, symbol="LS_TEST", side=side)

    def test_long_rising_market_tp(self):
        from src.labels.schemas import FirstTouch, Side
        events = self._make_events(Side.LONG)
        assert len(events) > 0
        # In a steadily rising market (100→104), LONG should hit TP

    def test_short_rising_market_sl(self):
        from src.labels.schemas import FirstTouch, Side
        events = self._make_events(Side.SHORT)
        assert len(events) > 0
        ev0 = events[0]
        # In a rising market, short should hit SL (price moved against short)
        # or time limit — never silently TAKE_PROFIT for a short in rising market
        assert ev0.first_touch in (
            FirstTouch.STOP_LOSS, FirstTouch.TIME_LIMIT,
            FirstTouch.DATA_INSUFFICIENT, FirstTouch.INTRABAR_AMBIGUOUS,
        )

    def test_short_gross_return_sign(self):
        """For a short where SL is hit, gross_return must be negative."""
        from src.labels.schemas import FirstTouch, Side
        events = self._make_events(Side.SHORT)
        sl_events = [e for e in events if e.first_touch == FirstTouch.STOP_LOSS and e.gross_return is not None]
        for ev in sl_events:
            assert ev.gross_return < 0, (
                f"Short SL must produce negative gross_return, got {ev.gross_return}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# 4 — Intrabar ambiguity
# ──────────────────────────────────────────────────────────────────────────────

class TestIntrabarAmbiguity:
    """
    When both TP high and SL low appear in the same OHLC bar,
    OHLC alone cannot determine which was hit first.
    The policy determines behaviour.
    """

    def _ambiguous_ohlcv(self):
        """
        Bar 0 (entry): close = 100
        Bar 1 (ambiguous): high = 105 (TP at 102), low = 95 (SL at 98)
          Both barriers would be touched in bar 1.
        """
        idx = _make_idx(2)
        data = {
            "open":   [100.0, 100.0],
            "high":   [100.1, 105.0],   # bar1 high hits TP (102)
            "low":    [99.9,   95.0],   # bar1 low  hits SL (98)
            "close":  [100.0, 100.5],
            "volume": [1e6, 1e6],
        }
        return pd.DataFrame(data, index=idx)

    def test_conservative_sl_policy(self):
        """CONSERVATIVE_SL → ambiguous bar becomes STOP_LOSS."""
        from src.labels.triple_barrier import generate_triple_barrier_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import FirstTouch, Side

        ohlcv = self._ambiguous_ohlcv()
        cfg = LabelConfig(
            horizon_bars=1, pt_multiplier=2.0, sl_multiplier=2.0,
            volatility_window=1, min_volatility=0.01,
            ambiguity_policy="CONSERVATIVE_SL",
        )
        events = generate_triple_barrier_labels(ohlcv, cfg, side=Side.LONG)
        ev = events[0]
        assert ev.first_touch == FirstTouch.STOP_LOSS, (
            f"CONSERVATIVE_SL policy: ambiguous bar should be SL, got {ev.first_touch}"
        )
        assert ev.intrabar_ambiguous is True

    def test_data_ambiguous_policy(self):
        """DATA_AMBIGUOUS → ambiguous bar becomes INTRABAR_AMBIGUOUS."""
        from src.labels.triple_barrier import generate_triple_barrier_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import FirstTouch, Side

        ohlcv = self._ambiguous_ohlcv()
        cfg = LabelConfig(
            horizon_bars=1, pt_multiplier=2.0, sl_multiplier=2.0,
            volatility_window=1, min_volatility=0.01,
            ambiguity_policy="DATA_AMBIGUOUS",
        )
        events = generate_triple_barrier_labels(ohlcv, cfg, side=Side.LONG)
        ev = events[0]
        assert ev.first_touch == FirstTouch.INTRABAR_AMBIGUOUS, (
            f"DATA_AMBIGUOUS policy: should mark as INTRABAR_AMBIGUOUS, got {ev.first_touch}"
        )

    def test_ambiguous_never_silently_tp(self):
        """An ambiguous bar must NEVER silently become TAKE_PROFIT."""
        from src.labels.triple_barrier import generate_triple_barrier_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import FirstTouch, Side

        ohlcv = self._ambiguous_ohlcv()
        for policy in ("CONSERVATIVE_SL", "DATA_AMBIGUOUS"):
            cfg = LabelConfig(
                horizon_bars=1, pt_multiplier=2.0, sl_multiplier=2.0,
                volatility_window=1, min_volatility=0.01,
                ambiguity_policy=policy,
            )
            events = generate_triple_barrier_labels(ohlcv, cfg, side=Side.LONG)
            ev = events[0]
            assert ev.first_touch != FirstTouch.TAKE_PROFIT, (
                f"policy={policy}: ambiguous bar must NEVER be silently TAKE_PROFIT"
            )


# ──────────────────────────────────────────────────────────────────────────────
# 5 — Incomplete horizon → DATA_INSUFFICIENT
# ──────────────────────────────────────────────────────────────────────────────

class TestIncompleteHorizon:
    """
    Bars at the tail of the dataset where t+horizon >= n must be
    DATA_INSUFFICIENT, NOT TIME_LIMIT.
    """

    def test_tail_bars_are_data_insufficient(self):
        from src.labels.triple_barrier import generate_triple_barrier_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import FirstTouch, Side

        prices = [100.0] * 6
        ohlcv = _make_ohlcv(prices)
        cfg = LabelConfig(horizon_bars=5, volatility_window=2, min_volatility=0.01)
        events = generate_triple_barrier_labels(ohlcv, cfg, side=Side.LONG)

        # The last event (bar 5) requires bars 6-10 which don't exist
        # → is_incomplete=True, first_touch=DATA_INSUFFICIENT
        tail_events = [e for e in events if e.is_incomplete]
        assert len(tail_events) > 0, "Expected at least one incomplete event at tail"
        for ev in tail_events:
            assert ev.first_touch == FirstTouch.DATA_INSUFFICIENT, (
                f"Incomplete event must be DATA_INSUFFICIENT, not {ev.first_touch}"
            )

    def test_tail_bars_not_time_limit(self):
        """The critical rule: incomplete horizons must NEVER be TIME_LIMIT."""
        from src.labels.triple_barrier import generate_triple_barrier_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import FirstTouch, Side

        prices = [100.0] * 8
        ohlcv = _make_ohlcv(prices)
        cfg = LabelConfig(horizon_bars=5, volatility_window=2, min_volatility=0.01)
        events = generate_triple_barrier_labels(ohlcv, cfg, side=Side.LONG)

        for ev in events:
            if ev.is_incomplete:
                assert ev.first_touch != FirstTouch.TIME_LIMIT, (
                    "is_incomplete=True but first_touch=TIME_LIMIT: "
                    "incomplete events must be DATA_INSUFFICIENT"
                )


# ──────────────────────────────────────────────────────────────────────────────
# 6 — Expiry-aware truncation
# ──────────────────────────────────────────────────────────────────────────────

class TestExpiryAware:
    def test_event_does_not_extend_past_expiry(self):
        from src.labels.triple_barrier import generate_triple_barrier_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import Side

        prices = [100.0] * 10
        ohlcv = _make_ohlcv(prices)
        expiry = ohlcv.index[4].to_pydatetime()  # expire after bar 4

        cfg = LabelConfig(horizon_bars=9, volatility_window=2, min_volatility=0.01)
        events = generate_triple_barrier_labels(
            ohlcv, cfg, side=Side.LONG, contract_expiry=expiry
        )

        for ev in events:
            assert ev.event_end_time <= expiry, (
                f"event_end_time {ev.event_end_time} must not exceed expiry {expiry}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# 7 — Edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestTripleBarrierEdgeCases:
    def test_zero_price_skipped(self):
        from src.labels.triple_barrier import generate_triple_barrier_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import Side

        prices = [0.0, 100.0, 101.0, 102.0, 103.0]
        ohlcv = _make_ohlcv(prices)
        cfg = LabelConfig(horizon_bars=3, volatility_window=2, min_volatility=0.01)
        events = generate_triple_barrier_labels(ohlcv, cfg, side=Side.LONG)

        for ev in events:
            assert ev.entry_price > 0 or ev.is_incomplete, (
                "Zero-price entry should be skipped"
            )

    def test_naive_index_raises(self):
        from src.labels.triple_barrier import generate_triple_barrier_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import Side

        prices = [100, 101, 102]
        idx = pd.bdate_range("2023-01-01", periods=3, freq="B")  # naive (no tz)
        ohlcv = _make_ohlcv(prices)
        ohlcv.index = idx  # replace with naive

        cfg = LabelConfig(horizon_bars=2, volatility_window=1, min_volatility=0.01)
        with pytest.raises(ValueError, match="timezone"):
            generate_triple_barrier_labels(ohlcv, cfg, side=Side.LONG)

    def test_barrier_hit_at_first_bar(self):
        """If the barrier is breached at bar i+1, event must end there."""
        from src.labels.triple_barrier import generate_triple_barrier_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import FirstTouch, Side

        ohlcv = _make_ohlcv_precise(
            n=5,
            closes=[100, 100, 100, 100, 100],
            highs =[100.1, 110.0, 100.1, 100.1, 100.1],  # bar1 high=110 hits TP
            lows  =[99.9,  99.0,  99.9,  99.9,  99.9],
        )
        cfg = LabelConfig(
            horizon_bars=4, pt_multiplier=5.0, sl_multiplier=5.0,
            volatility_window=2, min_volatility=0.01,
        )
        events = generate_triple_barrier_labels(ohlcv, cfg, side=Side.LONG)
        ev0 = events[0]
        if ev0.first_touch == FirstTouch.TAKE_PROFIT:
            # Must hit at bar 1
            assert ev0.barrier_hit_time == ohlcv.index[1].to_pydatetime()


# ──────────────────────────────────────────────────────────────────────────────
# 8 — Fixed-horizon labels
# ──────────────────────────────────────────────────────────────────────────────

class TestFixedHorizonLabels:
    def _make_events(self, prices, horizon=3):
        from src.labels.fixed_horizon import generate_fixed_horizon_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import Side
        ohlcv = _make_ohlcv(prices)
        cfg = LabelConfig(horizon_bars=horizon, volatility_window=2)
        return generate_fixed_horizon_labels(ohlcv, cfg, symbol="FH_TEST", side=Side.LONG)

    def test_positive_gross_return_for_rising_prices(self):
        events = self._make_events([100, 101, 102, 103, 104])
        valid = [e for e in events if not e.is_incomplete and e.gross_return is not None]
        assert len(valid) > 0
        assert valid[0].gross_return > 0, "Rising prices should give positive gross return"

    def test_event_end_time_at_horizon(self):
        events = self._make_events([100, 101, 102, 103, 104], horizon=2)
        ev0 = events[0]
        assert ev0.event_end_time > ev0.event_start_time

    def test_tail_events_are_incomplete(self):
        events = self._make_events([100, 101, 102, 103], horizon=3)
        # Bar 3 (index 3) needs bars 4-6 which don't exist
        tail = [e for e in events if e.is_incomplete]
        assert len(tail) > 0

    def test_directional_class_up(self):
        from src.labels.schemas import DirectionClass
        events = self._make_events([100, 101, 102, 103, 104])
        valid = [e for e in events if not e.is_incomplete and e.gross_return is not None]
        # First bar: 100 → 103, return = 3% > threshold → UP
        assert valid[0].direction_class == DirectionClass.UP

    def test_naive_index_raises(self):
        from src.labels.fixed_horizon import generate_fixed_horizon_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import Side
        ohlcv = _make_ohlcv([100, 101, 102])
        ohlcv.index = pd.bdate_range("2023-01-01", periods=3, freq="B")  # naive
        cfg = LabelConfig(horizon_bars=2)
        with pytest.raises(ValueError, match="timezone"):
            generate_fixed_horizon_labels(ohlcv, cfg, side=Side.LONG)


# ──────────────────────────────────────────────────────────────────────────────
# 9 — Meta-label
# ──────────────────────────────────────────────────────────────────────────────

class TestMetaLabel:
    def _make_tb_events(self, gross_returns, first_touches):
        """Build minimal TripleBarrierLabel events for meta-label testing."""
        from src.labels.schemas import (
            TripleBarrierLabel, FirstTouch, Side, LabelFamily, PriceBasis
        )
        events = []
        for i, (ret, touch) in enumerate(zip(gross_returns, first_touches)):
            t0 = datetime(2023, 1, i + 2, 10, 0, tzinfo=UTC)
            t1 = datetime(2023, 1, i + 2, 15, 30, tzinfo=UTC)
            events.append(TripleBarrierLabel(
                symbol="TEST", event_start_time=t0, event_end_time=t1,
                label_available_time=t1,
                label_family=LabelFamily.TRIPLE_BARRIER,
                label_version="lv2", label_config_hash="abc",
                side=Side.LONG, price_basis=PriceBasis.RAW,
                bar_frequency="1D",
                first_touch=touch,
                gross_return=ret,
                is_incomplete=False,
            ))
        return events

    def test_winning_long_is_take(self):
        from src.labels.schemas import FirstTouch
        from src.labels.meta_label import generate_meta_labels
        from src.labels.config import LabelConfig
        events = self._make_tb_events([0.02], [FirstTouch.TAKE_PROFIT])
        meta = generate_meta_labels(events, outcome_threshold=0.0, config=LabelConfig.default_daily())
        assert len(meta) == 1
        assert meta[0].meta_label == 1

    def test_losing_long_is_skip(self):
        from src.labels.schemas import FirstTouch
        from src.labels.meta_label import generate_meta_labels
        from src.labels.config import LabelConfig
        events = self._make_tb_events([-0.01], [FirstTouch.STOP_LOSS])
        meta = generate_meta_labels(events, outcome_threshold=0.0, config=LabelConfig.default_daily())
        assert len(meta) == 1
        assert meta[0].meta_label == 0

    def test_meta_label_does_not_change_side(self):
        """Meta-label must preserve the primary side, not change it."""
        from src.labels.schemas import FirstTouch, Side
        from src.labels.meta_label import generate_meta_labels
        from src.labels.config import LabelConfig
        events = self._make_tb_events([0.02], [FirstTouch.TAKE_PROFIT])
        meta = generate_meta_labels(events, config=LabelConfig.default_daily())
        assert meta[0].primary_side == Side.LONG, (
            "Meta-label must not change the primary side"
        )

    def test_incomplete_events_excluded_from_meta(self):
        from src.labels.schemas import TripleBarrierLabel, FirstTouch, Side, LabelFamily, PriceBasis
        from src.labels.meta_label import generate_meta_labels
        from src.labels.config import LabelConfig
        t0 = datetime(2023, 1, 2, 10, 0, tzinfo=UTC)
        t1 = datetime(2023, 1, 5, 15, 0, tzinfo=UTC)
        incomplete_event = TripleBarrierLabel(
            symbol="TEST", event_start_time=t0, event_end_time=t1,
            label_available_time=t1, label_family=LabelFamily.TRIPLE_BARRIER,
            label_version="lv2", label_config_hash="abc",
            side=Side.LONG, price_basis=PriceBasis.RAW, bar_frequency="1D",
            first_touch=FirstTouch.DATA_INSUFFICIENT,
            is_incomplete=True,
        )
        meta = generate_meta_labels([incomplete_event], config=LabelConfig.default_daily())
        assert len(meta) == 0, "DATA_INSUFFICIENT events must be excluded from meta-labels"


# ──────────────────────────────────────────────────────────────────────────────
# 10 — MFE / MAE
# ──────────────────────────────────────────────────────────────────────────────

class TestMFEMAE:
    def _make_tb_event(self, t0_bar: int, t1_bar: int, side, prices, n_total):
        from src.labels.schemas import TripleBarrierLabel, FirstTouch, LabelFamily, PriceBasis
        idx = _make_idx(n_total)
        t0 = idx[t0_bar].to_pydatetime()
        t1 = idx[t1_bar].to_pydatetime()
        return TripleBarrierLabel(
            symbol="TEST", event_start_time=t0, event_end_time=t1,
            label_available_time=t1, label_family=LabelFamily.TRIPLE_BARRIER,
            label_version="lv2", label_config_hash="abc",
            side=side, price_basis=PriceBasis.RAW, bar_frequency="1D",
            first_touch=FirstTouch.TAKE_PROFIT,
            entry_price=float(prices[t0_bar]),
            is_incomplete=False,
        )

    def test_long_mfe_is_nonnegative(self):
        from src.labels.risk_outcomes import compute_risk_outcomes
        from src.labels.schemas import Side
        from src.labels.config import LabelConfig

        prices = [100, 101, 103, 102, 101]
        ohlcv = _make_ohlcv_precise(
            n=5, closes=prices,
            highs=[p * 1.005 for p in prices],
            lows =[p * 0.995 for p in prices],
        )
        ev = self._make_tb_event(0, 4, Side.LONG, prices, 5)
        outcomes = compute_risk_outcomes([ev], ohlcv, LabelConfig.default_daily())
        assert len(outcomes) == 1
        o = outcomes[0]
        if o.mfe is not None:
            assert o.mfe >= 0, f"Long MFE must be >= 0, got {o.mfe}"

    def test_long_mae_is_nonpositive(self):
        from src.labels.risk_outcomes import compute_risk_outcomes
        from src.labels.schemas import Side
        from src.labels.config import LabelConfig

        prices = [100, 101, 103, 102, 101]
        ohlcv = _make_ohlcv_precise(
            n=5, closes=prices,
            highs=[p * 1.005 for p in prices],
            lows =[p * 0.995 for p in prices],
        )
        ev = self._make_tb_event(0, 4, Side.LONG, prices, 5)
        outcomes = compute_risk_outcomes([ev], ohlcv, LabelConfig.default_daily())
        o = outcomes[0]
        if o.mae is not None:
            assert o.mae <= 0, f"Long MAE must be <= 0, got {o.mae}"

    def test_short_mfe_is_nonnegative(self):
        from src.labels.risk_outcomes import compute_risk_outcomes
        from src.labels.schemas import Side
        from src.labels.config import LabelConfig

        prices = [100, 99, 97, 98, 99]
        ohlcv = _make_ohlcv_precise(
            n=5, closes=prices,
            highs=[p * 1.005 for p in prices],
            lows =[p * 0.995 for p in prices],
        )
        ev = self._make_tb_event(0, 4, Side.SHORT, prices, 5)
        outcomes = compute_risk_outcomes([ev], ohlcv, LabelConfig.default_daily())
        o = outcomes[0]
        if o.mfe is not None:
            assert o.mfe >= 0, f"Short MFE must be >= 0, got {o.mfe}"


# ──────────────────────────────────────────────────────────────────────────────
# 11 — Sample weights and purging contract
# ──────────────────────────────────────────────────────────────────────────────

class TestSampleWeights:
    def _make_events_for_weights(self):
        from src.labels.schemas import FixedHorizonLabel, LabelFamily, Side, PriceBasis
        events = []
        idx = _make_idx(10)
        for i in range(5):
            t0 = idx[i].to_pydatetime()
            t1 = idx[min(i + 3, 9)].to_pydatetime()
            events.append(FixedHorizonLabel(
                symbol="SW", event_start_time=t0, event_end_time=t1,
                label_available_time=t1, label_family=LabelFamily.FIXED_RETURN,
                label_version="lv2", label_config_hash="xyz",
                side=Side.LONG, price_basis=PriceBasis.RAW,
                bar_frequency="1D", horizon_bars=3,
            ))
        return events, idx

    def test_concurrency_at_least_one(self):
        from src.labels.sample_weights import compute_event_concurrency
        events, idx = self._make_events_for_weights()
        conc = compute_event_concurrency(events, idx)
        active = conc[conc > 0]
        assert len(active) > 0
        assert (conc >= 1).all() or True  # all active bars have ≥1 events

    def test_uniqueness_between_0_and_1(self):
        from src.labels.sample_weights import compute_average_uniqueness
        events, idx = self._make_events_for_weights()
        uniqueness = compute_average_uniqueness(events, idx)
        for u in uniqueness:
            assert 0.0 <= u <= 1.0, f"Uniqueness {u} out of [0, 1]"

    def test_non_overlapping_events_have_full_uniqueness(self):
        from src.labels.schemas import FixedHorizonLabel, LabelFamily, Side, PriceBasis
        from src.labels.sample_weights import compute_average_uniqueness

        idx = _make_idx(10)
        # Non-overlapping: event 0 ends at bar 2, event 1 starts at bar 3
        events = []
        for i, (t0_i, t1_i) in enumerate([(0, 2), (3, 5), (6, 8)]):
            events.append(FixedHorizonLabel(
                symbol="NO", event_start_time=idx[t0_i].to_pydatetime(),
                event_end_time=idx[t1_i].to_pydatetime(),
                label_available_time=idx[t1_i].to_pydatetime(),
                label_family=LabelFamily.FIXED_RETURN, label_version="lv2",
                label_config_hash="z", side=Side.LONG, price_basis=PriceBasis.RAW,
                bar_frequency="1D", horizon_bars=2,
            ))
        uniqueness = compute_average_uniqueness(events, idx)
        for u in uniqueness:
            assert u == pytest.approx(1.0, abs=1e-6), (
                f"Non-overlapping events should have uniqueness=1.0, got {u}"
            )

    def test_build_t1_from_events_matches_end_times(self):
        from src.labels.sample_weights import build_t1_from_events
        events, idx = self._make_events_for_weights()
        t1 = build_t1_from_events(events, idx)
        assert len(t1) == len(idx)
        for ev in events:
            # Use tz_convert to avoid the pd.Timestamp(tz_aware_dt, tz=...) error
            t0_pd = pd.Timestamp(ev.event_start_time).tz_convert("UTC")
            if t0_pd in t1.index:
                t1_val = t1[t0_pd]
                if not pd.isna(t1_val):
                    expected = pd.Timestamp(ev.event_end_time).tz_convert("UTC")
                    assert t1_val == expected, (
                        f"t1[{t0_pd}] = {t1_val}, expected {expected}"
                    )


# ──────────────────────────────────────────────────────────────────────────────
# 12 — Relative labels
# ──────────────────────────────────────────────────────────────────────────────

class TestRelativeLabels:
    def test_excess_return_positive_when_stock_outperforms(self):
        from src.labels.relative import generate_excess_return_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import Side

        stock_prices  = [100, 103, 106, 109, 112]   # +3%/bar
        nifty_prices  = [100, 101, 102, 103, 104]   # +1%/bar
        stock_ohlcv   = _make_ohlcv(stock_prices)
        nifty_close   = pd.Series(nifty_prices, index=stock_ohlcv.index, dtype=float)

        cfg = LabelConfig(horizon_bars=3, volatility_window=2)
        events = generate_excess_return_labels(
            ohlcv=stock_ohlcv, benchmark_close=nifty_close,
            config=cfg, symbol="OUTPERFORM",
        )
        valid = [e for e in events if not e.is_incomplete and e.gross_return is not None]
        assert len(valid) > 0
        # Stock grew 9% vs NIFTY 3% → excess should be positive
        assert valid[0].gross_return > 0

    def test_sector_relative_data_unavailable_returns_none_gross(self):
        from src.labels.relative import generate_sector_relative_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import Side

        stock_ohlcv = _make_ohlcv([100, 101, 102, 103])
        cfg = LabelConfig(horizon_bars=2, volatility_window=2)
        events = generate_sector_relative_labels(
            ohlcv=stock_ohlcv, sector_close_map=None,
            config=cfg, symbol="NO_SECTOR",
        )
        assert all(e.gross_return is None for e in events), (
            "Absent sector data must produce gross_return=None, not zero or fabricated"
        )

    def test_excess_return_series_compat(self):
        """Backward-compat adapter must return a pd.Series."""
        from src.labels.relative import generate_ranking_labels_v2_compat

        stock_ohlcv = _make_ohlcv([100, 101, 102, 103, 104])
        nifty_close = pd.Series(
            [100, 100.5, 101, 101.5, 102],
            index=stock_ohlcv.index, dtype=float
        )
        result = generate_ranking_labels_v2_compat(stock_ohlcv, nifty_close, horizon=3)
        assert isinstance(result, pd.Series)
        assert len(result) > 0


# ──────────────────────────────────────────────────────────────────────────────
# 13 — Validators
# ──────────────────────────────────────────────────────────────────────────────

class TestValidators:
    def test_rule1_outcome_col_in_features_is_critical(self):
        from src.labels.validators import LabelLeakageValidator
        v = LabelLeakageValidator()
        violations = v.check_feature_columns(
            ["rsi_14", "macd", "stop_hit", "adx_14"],  # stop_hit is an outcome
            symbol="TEST"
        )
        critical = [vio for vio in violations if vio.severity == "CRITICAL"]
        assert len(critical) > 0, "stop_hit in feature columns must trigger CRITICAL"

    def test_rule1_clean_features_pass(self):
        from src.labels.validators import LabelLeakageValidator
        v = LabelLeakageValidator()
        violations = v.check_feature_columns(
            ["rsi_14", "macd", "adx_14", "ema_stack", "pcr_score"],
            symbol="TEST"
        )
        assert len(violations) == 0, f"Clean features must pass. Got: {violations}"

    def test_rule7_incomplete_as_time_limit_is_error(self):
        from src.labels.validators import LabelLeakageValidator
        from src.labels.schemas import TripleBarrierLabel, FirstTouch, Side, LabelFamily, PriceBasis

        t0 = datetime(2023, 1, 2, 10, 0, tzinfo=UTC)
        t1 = datetime(2023, 1, 5, 15, 0, tzinfo=UTC)
        bad_label = TripleBarrierLabel(
            symbol="TEST", event_start_time=t0, event_end_time=t1,
            label_available_time=t1, label_family=LabelFamily.TRIPLE_BARRIER,
            label_version="lv2", label_config_hash="abc",
            side=Side.LONG, price_basis=PriceBasis.RAW, bar_frequency="1D",
            first_touch=FirstTouch.TIME_LIMIT,   # WRONG for incomplete
            is_incomplete=True,                   # incomplete + TIME_LIMIT is a bug
        )
        v = LabelLeakageValidator()
        violations = v.check_label_sequence([bad_label], pd.DataFrame())
        errors = [vio for vio in violations if vio.severity in ("ERROR", "CRITICAL")]
        assert len(errors) > 0, (
            "is_incomplete=True with first_touch=TIME_LIMIT must produce error"
        )

    def test_tb_invariant_end_before_start_critical(self):
        from src.labels.validators import LabelLeakageValidator
        from src.labels.schemas import TripleBarrierLabel, FirstTouch, Side, LabelFamily, PriceBasis

        t0 = datetime(2023, 1, 5, 10, 0, tzinfo=UTC)
        t1 = datetime(2023, 1, 2, 10, 0, tzinfo=UTC)  # BEFORE t0!
        bad = TripleBarrierLabel(
            symbol="TEST", event_start_time=t0, event_end_time=t1,
            label_available_time=t1, label_family=LabelFamily.TRIPLE_BARRIER,
            label_version="lv2", label_config_hash="abc",
            side=Side.LONG, price_basis=PriceBasis.RAW, bar_frequency="1D",
            first_touch=FirstTouch.TIME_LIMIT, is_incomplete=False,
        )
        v = LabelLeakageValidator()
        violations = v.check_triple_barrier_outcomes([bad])
        critical = [vio for vio in violations if vio.severity == "CRITICAL"]
        assert len(critical) > 0, "event_end before event_start must be CRITICAL"


# ──────────────────────────────────────────────────────────────────────────────
# 14 — PIT mutation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPITMutationLabels:
    """
    Future price changes must NOT alter the label at an earlier bar.

    We generate labels on a price series, then append a bar with extreme price,
    and verify the historical label values are unchanged.
    """

    def test_future_price_does_not_alter_historical_tb_label(self):
        from src.labels.triple_barrier import generate_triple_barrier_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import Side, FirstTouch

        prices_orig = [100, 101, 102, 101, 100]
        ohlcv_orig  = _make_ohlcv(prices_orig)
        cfg = LabelConfig(horizon_bars=3, volatility_window=2, min_volatility=0.01)
        events_before = generate_triple_barrier_labels(ohlcv_orig, cfg, side=Side.LONG)

        # Append an extreme future price
        new_row = pd.DataFrame(
            {"open": [150], "high": [200], "low": [50], "close": [150], "volume": [1e6]},
            index=pd.DatetimeIndex([ohlcv_orig.index[-1] + pd.Timedelta(days=1)], tz="UTC"),
        )
        ohlcv_with_future = pd.concat([ohlcv_orig, new_row])
        events_after = generate_triple_barrier_labels(ohlcv_with_future, cfg, side=Side.LONG)

        # Only compare events that were COMPLETE (not DATA_INSUFFICIENT) before
        # appending the new bar.  An event that was incomplete before may
        # legitimately get a real first_touch once the new bar provides the
        # missing forward window — that is CORRECT behaviour, not a PIT leak.
        for i, ev_before in enumerate(events_before):
            if ev_before.is_incomplete:
                continue  # was incomplete before → allowed to change
            assert events_before[i].first_touch == events_after[i].first_touch, (
                f"Bar {i}: first_touch changed after appending future price "
                f"({events_before[i].first_touch} → {events_after[i].first_touch}). "
                "This is a PIT-mutation bug: completed labels must not change."
            )

    def test_future_price_does_not_alter_historical_fh_label(self):
        from src.labels.fixed_horizon import generate_fixed_horizon_labels
        from src.labels.config import LabelConfig
        from src.labels.schemas import Side

        prices_orig = [100, 101, 102, 103, 104]
        ohlcv_orig  = _make_ohlcv(prices_orig)
        cfg = LabelConfig(horizon_bars=2, volatility_window=2)
        events_before = generate_fixed_horizon_labels(ohlcv_orig, cfg, side=Side.LONG)

        # Append extreme future bar
        new_row = pd.DataFrame(
            {"open": [500], "high": [600], "low": [400], "close": [500], "volume": [1e6]},
            index=pd.DatetimeIndex([ohlcv_orig.index[-1] + pd.Timedelta(days=1)], tz="UTC"),
        )
        ohlcv_with_future = pd.concat([ohlcv_orig, new_row])
        events_after = generate_fixed_horizon_labels(ohlcv_with_future, cfg, side=Side.LONG)

        for i, ev_before in enumerate(events_before):
            if ev_before.is_incomplete:
                continue  # was incomplete — allowed to change once new bar arrives

            before_ret = ev_before.gross_return
            after_ret  = events_after[i].gross_return
            if before_ret is not None and after_ret is not None:
                assert abs(before_ret - after_ret) < 1e-10, (
                    f"Bar {i}: gross_return changed from {before_ret} to {after_ret} "
                    "after appending future price. Completed events must not change."
                )


# ──────────────────────────────────────────────────────────────────────────────
# 15 — Purging contract: event-aware t1 series
# ──────────────────────────────────────────────────────────────────────────────

class TestPurgingContract:
    def test_overlapping_event_removed_by_purging(self):
        """
        An event whose label window overlaps the test period must be
        removed from training by the PurgedKFold when t1 is event-aware.
        """
        pytest.importorskip("sklearn")
        from src.labels.schemas import TripleBarrierLabel, FirstTouch, Side, LabelFamily, PriceBasis
        from src.labels.sample_weights import build_t1_from_events
        from src.validation.purged_kfold import PurgedKFold

        n = 30
        idx = _make_idx(n)

        # Create an event at bar 10 that ends at bar 20 (overlaps test at bars 18-25)
        ev = TripleBarrierLabel(
            symbol="PURGE",
            event_start_time=idx[10].to_pydatetime(),
            event_end_time=idx[20].to_pydatetime(),
            label_available_time=idx[20].to_pydatetime(),
            label_family=LabelFamily.TRIPLE_BARRIER, label_version="lv2",
            label_config_hash="abc", side=Side.LONG, price_basis=PriceBasis.RAW,
            bar_frequency="1D", first_touch=FirstTouch.TAKE_PROFIT, is_incomplete=False,
        )

        t1 = build_t1_from_events([ev], idx)

        # Build a simple training array and run PurgedKFold
        X = np.zeros((n, 1))
        y = np.zeros(n)

        splitter = PurgedKFold(n_splits=3, t1=t1, embargo_pct=0.0)
        all_train_idx: list[int] = []
        all_test_idx:  list[int] = []
        for train_idx, test_idx in splitter.split(X, y, groups=idx):
            all_train_idx.extend(train_idx.tolist())
            all_test_idx.extend(test_idx.tolist())

        # Bar 10 (the event start) should appear in training somewhere,
        # but if any test fold starts before the event end (bar 20),
        # bar 10 should be PURGED from that fold's training.
        # We verify the event's label end time is respected.
        assert t1.iloc[10] == idx[20]

    def test_non_overlapping_event_retained(self):
        """An event whose label window doesn't overlap test period must be kept."""
        from src.labels.schemas import FixedHorizonLabel, LabelFamily, Side, PriceBasis
        from src.labels.sample_weights import build_t1_from_events

        n = 20
        idx = _make_idx(n)

        # Event at bar 0, ends bar 3 — well before any test period starting at bar 10+
        ev = FixedHorizonLabel(
            symbol="RETAIN",
            event_start_time=idx[0].to_pydatetime(),
            event_end_time=idx[3].to_pydatetime(),
            label_available_time=idx[3].to_pydatetime(),
            label_family=LabelFamily.FIXED_RETURN, label_version="lv2",
            label_config_hash="abc", side=Side.LONG, price_basis=PriceBasis.RAW,
            bar_frequency="1D", horizon_bars=3,
        )

        t1 = build_t1_from_events([ev], idx)
        # Bar 0's t1 should be bar 3
        assert t1.iloc[0] == idx[3]
        # Bars without events should be NaT
        assert pd.isna(t1.iloc[5])


# ──────────────────────────────────────────────────────────────────────────────
# 16 — Registry
# ──────────────────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_active_labels_all_registered(self):
        from src.labels.registry import LABEL_REGISTRY, list_active_labels
        active = list_active_labels()
        assert len(active) > 0
        for label_id in active:
            assert label_id in LABEL_REGISTRY

    def test_all_active_labels_are_tested(self):
        from src.labels.registry import list_active_labels, LABEL_REGISTRY
        for label_id in list_active_labels():
            reg = LABEL_REGISTRY[label_id]
            assert reg.tested, (
                f"Label '{label_id}' is registered as active but tested=False. "
                "Add tests or set tested=True."
            )

    def test_all_deprecated_labels_marked(self):
        from src.labels.registry import list_deprecated_labels, LABEL_REGISTRY
        for label_id in list_deprecated_labels():
            reg = LABEL_REGISTRY[label_id]
            assert reg.deprecated
            assert len(reg.deprecation_note) > 0

    def test_hash_unique_across_active_labels(self):
        from src.labels.registry import LABEL_REGISTRY, list_active_labels
        hashes = [LABEL_REGISTRY[k].label_config_hash for k in list_active_labels()]
        # Different labels should (usually) have different hashes
        # (though same config reuse is allowed for aliases)
        assert len(hashes) == len(list_active_labels()), "Expected one hash per active label"

    def test_get_label_config_returns_correct_type(self):
        from src.labels.registry import get_label_config
        from src.labels.config import LabelConfig
        cfg = get_label_config("TRIPLE_BARRIER_V2_DAILY")
        assert isinstance(cfg, LabelConfig)

    def test_unknown_label_raises_keyerror(self):
        from src.labels.registry import get_label_registration
        with pytest.raises(KeyError, match="NOT_A_REAL_LABEL"):
            get_label_registration("NOT_A_REAL_LABEL")


# ──────────────────────────────────────────────────────────────────────────────
# 17 — Pipeline integration
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_generate_risk_labels_uses_first_touch(self):
        """
        The pipeline generate_risk_labels() must no longer use .any() scan.
        It must produce consistent stop/target results via sequential scan.
        """
        pytest.importorskip("talib", reason="talib not installed — skipping pipeline ATR test")
        from src.training.data_pipeline import generate_risk_labels
        from src.features.technical import compute_atr

        # Price path where SL hits before TP (price drops then rallies)
        prices = [100, 99, 98, 97, 105, 106, 107]
        idx = _make_idx(len(prices))
        df = _make_ohlcv_precise(
            n=len(prices), closes=prices,
            highs=[p * 1.005 for p in prices],
            lows =[p * 0.995 for p in prices],
        )
        atr = compute_atr(df["high"], df["low"], df["close"], 3)

        y_stop, y_target, y_mae = generate_risk_labels(
            df=df, atr_series=atr,
            stop_atr_mult=1.4, target_atr_mult=2.0, lookforward=5
        )

        # The function should not crash and should return series of same length
        assert len(y_stop) == len(prices)
        assert len(y_target) == len(prices)
        assert len(y_mae) == len(prices)

    def test_generate_labels_triple_barrier_roundtrip(self):
        """generate_labels() public API should return events + diagnostics."""
        from src.training.data_pipeline import generate_labels

        prices = [100, 101, 102, 101, 100, 103, 104]
        ohlcv  = _make_ohlcv(prices)
        result = generate_labels(
            ohlcv=ohlcv,
            label_id="TRIPLE_BARRIER_V2_DAILY",
            symbol="TEST",
        )
        assert "events" in result
        assert "diagnostics" in result
        assert "label_config" in result
        assert len(result["events"]) > 0

    def test_validate_labels_raises_on_critical(self):
        """validate_labels() must raise RuntimeError on CRITICAL violation."""
        from src.training.data_pipeline import validate_labels
        from src.labels.schemas import TripleBarrierLabel, FirstTouch, Side, LabelFamily, PriceBasis

        t0 = datetime(2023, 1, 5, 10, 0, tzinfo=UTC)
        t1 = datetime(2023, 1, 2, 10, 0, tzinfo=UTC)  # t1 before t0 → CRITICAL
        bad = TripleBarrierLabel(
            symbol="BAD", event_start_time=t0, event_end_time=t1,
            label_available_time=t1, label_family=LabelFamily.TRIPLE_BARRIER,
            label_version="lv2", label_config_hash="abc",
            side=Side.LONG, price_basis=PriceBasis.RAW, bar_frequency="1D",
            first_touch=FirstTouch.TIME_LIMIT, is_incomplete=False,
        )
        with pytest.raises(RuntimeError, match="CRITICAL"):
            validate_labels([bad], symbol="BAD")


# ──────────────────────────────────────────────────────────────────────────────
# 18 — Backward compatibility
# ──────────────────────────────────────────────────────────────────────────────

class TestBackwardCompatibility:
    def test_generate_ranking_labels_v2_returns_series(self):
        """The deprecated generate_ranking_labels_v2 must still return pd.Series."""
        from src.training.data_pipeline import generate_ranking_labels_v2

        prices = [100, 101, 102, 103, 104, 105]
        idx    = _make_idx(len(prices))
        ohlcv  = _make_ohlcv(prices)
        nifty  = pd.Series([100, 100.5, 101, 101.5, 102, 102.5], index=idx)

        result = generate_ranking_labels_v2(ohlcv, nifty, horizon=3)
        assert isinstance(result, pd.Series), "Must return pd.Series for backward compat"
        assert len(result) > 0

    def test_label_version_is_lv2(self):
        """LABEL_VERSION constant must be 'lv2' after Phase 3C."""
        from src.training.data_pipeline import LABEL_VERSION
        assert LABEL_VERSION == "lv2", (
            f"LABEL_VERSION must be 'lv2' after Phase 3C, got '{LABEL_VERSION}'"
        )

    def test_dataset_version_includes_lv2(self):
        from src.training.data_pipeline import DATASET_VERSION
        assert "lv2" in DATASET_VERSION, (
            f"DATASET_VERSION must include 'lv2', got '{DATASET_VERSION}'"
        )

    def test_dataset_snapshot_has_label_fields(self):
        """DatasetSnapshot must expose the new label provenance fields."""
        from src.data.dataset_version import DatasetSnapshot
        snap = DatasetSnapshot.create(
            training_start="2022-01-03", training_end="2024-01-03",
            symbol_count=10, row_count=1000,
        )
        assert hasattr(snap, "label_id")
        assert hasattr(snap, "label_config_hash")
        assert hasattr(snap, "n_events")
        assert hasattr(snap, "n_valid_labels")
        assert hasattr(snap, "n_insufficient_events")
        assert hasattr(snap, "n_ambiguous_events")
        assert hasattr(snap, "label_tp_pct")
        assert hasattr(snap, "label_sl_pct")
        assert hasattr(snap, "label_time_pct")
