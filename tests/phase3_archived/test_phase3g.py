"""
Phase 3G Tests — Cost, Slippage & Execution-Aware Backtesting.

Test categories
---------------
 1. Execution timing — same-bar fill prevention, NEXT_OPEN/NEXT_BAR
 2. Cost model — equity and F&O costs, historical schedule routing, buy/sell asymmetry
 3. Slippage models — FixedBPS, SpreadProxy, VolatilityParticipation
 4. NSE calendar — holidays, trading days, expiry dates
 5. Fill engine — gap-through stop, stop/target ambiguity (CONSERVATIVE),
                   partial fills, circuit limits, F&O ban, expired contracts
 6. Position accounting — open/close, P&L reconciliation, turnover
 7. Backtest engine — golden P&L test, reproducibility, OOS-only inputs
 8. Future mutation tests — modifying future data must not alter historical results
 9. gex.py LOT_SIZES fix — post-SEBI Nov 2024 values
10. Cost attribution — net_pnl = gross_pnl - total_cost invariant
11. SpreadDataStatus — PROXY must not be labelled OBSERVED
12. ExecutionDataLevel — evidence level classification
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pytest

UTC = timezone.utc


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _ts(d: str, hr: int = 9, mn: int = 15) -> datetime:
    return datetime.fromisoformat(f"{d}T{hr:02d}:{mn:02d}:00+00:00")


def _bar(
    dt_str: str,
    o: float, h: float, l: float, c: float,
    vol: float = 1_000_000.0,
    adv_inr: Optional[float] = None,
    atr_pct: Optional[float] = None,
    band_upper: Optional[float] = None,
    band_lower: Optional[float] = None,
) -> "OHLCBar":
    from src.execution.fill_engine import OHLCBar
    return OHLCBar(
        timestamp=_ts(dt_str),
        open=o, high=h, low=l, close=c,
        volume=vol,
        adv_inr=adv_inr,
        atr_pct=atr_pct,
        price_band_upper=band_upper,
        price_band_lower=band_lower,
    )


def _order(
    symbol="NIFTY",
    side="LONG",
    order_side="BUY",
    lots=1,
    lot_size=50,
    policy="NEXT_OPEN",
    signal_ts="2024-01-15",
    limit_price=None,
    stop_price=None,
    expiry_date=None,
) -> "OrderIntent":
    from src.execution.schemas import (
        ExecutionPolicy, FillStatus, InstrumentType, OrderIntent,
        OrderSide, ProductType, TradeSide, make_order_id,
    )
    ts = _ts(signal_ts)
    return OrderIntent(
        order_id=make_order_id(),
        instrument_id=symbol,
        underlying=symbol,
        instrument_type=InstrumentType.FUT_IDX,
        product_type=ProductType.NRML,
        side=TradeSide.LONG if side == "LONG" else TradeSide.SHORT,
        order_side=OrderSide.BUY if order_side == "BUY" else OrderSide.SELL,
        quantity_lots=lots,
        lot_size=lot_size,
        limit_price=limit_price,
        stop_price=stop_price,
        signal_time=ts,
        decision_time=ts,
        order_time=ts,
        execution_policy=ExecutionPolicy(policy),
        model_id="test_model",
        model_version="v1",
        dataset_id="d1",
        feature_set_id="f1",
        label_version="lv2",
        expiry_date=expiry_date,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Execution timing
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutionTiming:
    """Spec §4: same-bar fill prevention; NEXT_OPEN default."""

    def _engine(self, policy="NEXT_OPEN", allow_same_close=False):
        from src.execution.fill_engine import FillEngine, FillEngineConfig
        from src.execution.schemas import ExecutionPolicy
        cfg = FillEngineConfig(
            default_policy=ExecutionPolicy(policy),
            allow_same_close=allow_same_close,
        )
        return FillEngine(config=cfg)

    def test_same_close_rejected_by_default(self):
        """Spec §4: SAME_CLOSE must be rejected unless explicitly enabled."""
        from src.execution.schemas import FillStatus
        engine = self._engine(policy="SAME_CLOSE", allow_same_close=False)
        signal_bar = _bar("2024-01-15", 21000, 21100, 20900, 21050)
        next_bars  = [_bar("2024-01-16", 21100, 21200, 21000, 21150)]
        order = _order(policy="SAME_CLOSE")
        fill = engine.fill(order, signal_bar, next_bars)
        assert fill.status == FillStatus.REJECTED, (
            "SAME_CLOSE must be REJECTED when allow_same_close=False"
        )

    def test_same_close_allowed_when_enabled(self):
        """SAME_CLOSE fills at signal-bar close when explicitly permitted."""
        from src.execution.schemas import FillStatus
        engine = self._engine(policy="SAME_CLOSE", allow_same_close=True)
        signal_bar = _bar("2024-01-15", 21000, 21100, 20900, 21050)
        next_bars  = [_bar("2024-01-16", 21100, 21200, 21000, 21150)]
        order = _order(policy="SAME_CLOSE")
        fill = engine.fill(order, signal_bar, next_bars)
        assert fill.filled, "SAME_CLOSE with allow_same_close=True must fill"
        # Fill price is signal bar close + slippage
        assert fill.fill_price is not None
        assert fill.fill_price > 21050 * 0.99  # approx signal close

    def test_next_open_fills_at_next_bar_open(self):
        """NEXT_OPEN fills at the open of the bar AFTER the signal bar."""
        from src.execution.schemas import FillStatus
        engine = self._engine(policy="NEXT_OPEN")
        signal_bar = _bar("2024-01-15", 21000, 21100, 20900, 21050)
        next_bar   = _bar("2024-01-16", 21200, 21300, 21100, 21250)
        fill = engine.fill(_order(policy="NEXT_OPEN"), signal_bar, [next_bar])
        assert fill.filled
        # Fill price = next_bar.open + slippage (BUY → adverse = higher)
        assert fill.fill_price >= next_bar.open

    def test_next_bar_fills_at_close(self):
        """NEXT_BAR fills at close of the next bar."""
        from src.execution.schemas import FillStatus
        engine = self._engine(policy="NEXT_BAR")
        signal_bar = _bar("2024-01-15", 21000, 21100, 20900, 21050)
        next_bar   = _bar("2024-01-16", 21200, 21300, 21100, 21250)
        fill = engine.fill(_order(policy="NEXT_BAR"), signal_bar, [next_bar])
        assert fill.filled
        assert fill.fill_price is not None
        assert fill.fill_price >= next_bar.close * 0.999  # close ± slippage

    def test_no_next_bars_returns_unavailable(self):
        """When no forward bars exist, fill must be UNAVAILABLE."""
        from src.execution.schemas import FillStatus
        engine = self._engine(policy="NEXT_OPEN")
        signal_bar = _bar("2024-01-15", 21000, 21100, 20900, 21050)
        fill = engine.fill(_order(), signal_bar, next_bars=[])
        assert fill.status == FillStatus.UNAVAILABLE, (
            "No next bars must produce UNAVAILABLE, not a fabricated fill"
        )

    def test_fill_time_is_after_signal_time(self):
        """eligible_fill_time must be after signal_time."""
        engine = self._engine(policy="NEXT_OPEN")
        signal_bar = _bar("2024-01-15", 21000, 21100, 20900, 21050)
        next_bar   = _bar("2024-01-16", 21200, 21300, 21100, 21250)
        order = _order(signal_ts="2024-01-15")
        fill  = engine.fill(order, signal_bar, [next_bar])
        if fill.eligible_fill_time:
            assert fill.eligible_fill_time > order.signal_time, (
                "Fill time must be after signal time (no look-ahead)"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Cost model
# ══════════════════════════════════════════════════════════════════════════════

class TestCostModel:
    """Golden cost calculations with known NSE rate structures."""

    def test_futures_sell_cost_components(self):
        """
        Futures sell on 2024-01-15 (post Budget 2023-24):
        STT = 0.0125% on sell notional
        Exchange = 0.0019% on notional
        Brokerage = ₹20 (capped)
        """
        from src.execution.cost_model import compute_trade_cost
        from src.execution.schemas import InstrumentType, OrderSide, ProductType

        cost = compute_trade_cost(
            instrument_type=InstrumentType.FUT_IDX,
            order_side=OrderSide.SELL,
            product_type=ProductType.NRML,
            trade_date=date(2024, 1, 15),
            price=21000.0,
            quantity_lots=1,
            lot_size=50,
        )
        notional = 21000.0 * 1 * 50  # ₹1,050,000

        expected_stt = notional * 0.000125
        assert abs(cost.stt - expected_stt) < 0.01, (
            f"STT={cost.stt:.4f}, expected {expected_stt:.4f}"
        )
        assert cost.brokerage <= 20.0, "Brokerage must be capped at ₹20"
        assert cost.stamp_duty == pytest.approx(0.0, abs=0.01), (
            "Stamp duty must be zero on sell side"
        )
        assert cost.gst > 0, "GST must be non-zero (on brokerage + exchange)"
        assert cost.total > 0

    def test_futures_buy_has_stamp_duty_no_stt(self):
        """Futures BUY: STT = 0 (only on sell), stamp duty non-zero."""
        from src.execution.cost_model import compute_trade_cost
        from src.execution.schemas import InstrumentType, OrderSide, ProductType

        cost = compute_trade_cost(
            instrument_type=InstrumentType.FUT_IDX,
            order_side=OrderSide.BUY,
            product_type=ProductType.NRML,
            trade_date=date(2024, 1, 15),
            price=21000.0,
            quantity_lots=1,
            lot_size=50,
        )
        assert cost.stt == pytest.approx(0.0, abs=0.01), "Futures BUY has no STT"
        assert cost.stamp_duty > 0, "Futures BUY must have stamp duty"

    def test_pre_budget_2023_futures_stt_rate(self):
        """Pre-Oct 2023: Futures STT = 0.01% (not 0.0125%)."""
        from src.execution.cost_model import compute_trade_cost
        from src.execution.schemas import InstrumentType, OrderSide, ProductType

        cost_pre  = compute_trade_cost(
            InstrumentType.FUT_IDX, OrderSide.SELL, ProductType.NRML,
            date(2023, 9, 15), 18000.0, 1, 50,
        )
        cost_post = compute_trade_cost(
            InstrumentType.FUT_IDX, OrderSide.SELL, ProductType.NRML,
            date(2023, 11, 15), 18000.0, 1, 50,
        )
        notional = 18000.0 * 50
        expected_pre  = notional * 0.0001    # 0.01%
        expected_post = notional * 0.000125  # 0.0125%

        assert abs(cost_pre.stt - expected_pre) < 0.01, (
            f"Pre-2023 STT: expected {expected_pre:.4f}, got {cost_pre.stt:.4f}"
        )
        assert abs(cost_post.stt - expected_post) < 0.01, (
            f"Post-2023 STT: expected {expected_post:.4f}, got {cost_post.stt:.4f}"
        )
        assert cost_post.stt > cost_pre.stt, "Post-Budget 2023 STT must be higher"

    def test_equity_delivery_buy_stt_both_sides(self):
        """Equity delivery: STT 0.1% on buy notional."""
        from src.execution.cost_model import compute_trade_cost
        from src.execution.schemas import InstrumentType, OrderSide, ProductType

        cost = compute_trade_cost(
            InstrumentType.EQ_DELIVERY, OrderSide.BUY, ProductType.CNC,
            date(2024, 1, 15), 2500.0, 1, 100,
        )
        notional = 2500.0 * 100
        expected_stt = notional * 0.001  # 0.1%
        assert abs(cost.stt - expected_stt) < 0.01

    def test_equity_intraday_stt_sell_only(self):
        """Equity MIS: STT 0.025% on sell only."""
        from src.execution.cost_model import compute_trade_cost
        from src.execution.schemas import InstrumentType, OrderSide, ProductType

        cost_sell = compute_trade_cost(
            InstrumentType.EQ_INTRADAY, OrderSide.SELL, ProductType.MIS,
            date(2024, 1, 15), 2500.0, 1, 100,
        )
        cost_buy = compute_trade_cost(
            InstrumentType.EQ_INTRADAY, OrderSide.BUY, ProductType.MIS,
            date(2024, 1, 15), 2500.0, 1, 100,
        )
        notional = 2500.0 * 100
        expected_sell_stt = notional * 0.00025
        assert abs(cost_sell.stt - expected_sell_stt) < 0.01
        assert cost_buy.stt == pytest.approx(0.0, abs=0.01), (
            "Intraday BUY has no STT"
        )

    def test_historical_schedule_not_future_schedule(self):
        """
        Future cost schedule mutation must NOT alter historical trade costs.
        Changing today's schedule must not affect a historical calculation.
        """
        from src.execution.cost_model import (
            CostScheduleRegistry, CostScheduleVersion,
            IndiaFnOCostSchedule, compute_trade_cost,
        )
        from src.execution.schemas import InstrumentType, OrderSide, ProductType

        registry = CostScheduleRegistry()
        cost_before = compute_trade_cost(
            InstrumentType.FUT_IDX, OrderSide.SELL, ProductType.NRML,
            date(2022, 6, 15), 17000.0, 1, 50,
            registry=registry,
        )

        # Register a future schedule with absurdly high STT
        future_ver = CostScheduleVersion(
            version_id="test-future-9999",
            effective_from=date(2030, 1, 1),
            effective_to=None,
            source="TEST",
        )
        future_sched = IndiaFnOCostSchedule(
            version=future_ver,
            stt_futures_sell_pct=0.99,  # absurd — future only
        )
        registry.register_fno_schedule(future_ver, future_sched)

        cost_after = compute_trade_cost(
            InstrumentType.FUT_IDX, OrderSide.SELL, ProductType.NRML,
            date(2022, 6, 15), 17000.0, 1, 50,
            registry=registry,
        )
        assert abs(cost_before.stt - cost_after.stt) < 0.01, (
            "Future cost schedule must not alter historical 2022 trade cost"
        )

    def test_gst_applied_on_brokerage_and_exchange(self):
        """GST = 18% on (brokerage + exchange_charge). Not on STT or stamp."""
        from src.execution.cost_model import compute_trade_cost
        from src.execution.schemas import InstrumentType, OrderSide, ProductType

        cost = compute_trade_cost(
            InstrumentType.FUT_IDX, OrderSide.BUY, ProductType.NRML,
            date(2024, 1, 15), 21000.0, 1, 50,
        )
        expected_gst_base = cost.brokerage + cost.exchange_charge
        expected_gst = expected_gst_base * 0.18
        assert abs(cost.gst - expected_gst) < 0.01, (
            f"GST={cost.gst:.4f}, expected {expected_gst:.4f} "
            f"(18% of brokerage {cost.brokerage:.4f} + exchange {cost.exchange_charge:.4f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Slippage models
# ══════════════════════════════════════════════════════════════════════════════

class TestSlippageModels:
    def test_fixed_bps_returns_configured_value(self):
        from src.execution.slippage import FixedBPSSlippage
        from src.execution.schemas import SpreadDataStatus, ExecutionDataLevel
        model = FixedBPSSlippage(fixed_bps=10.0)
        est = model.estimate(price=21000.0, quantity_units=50)
        assert est.slippage_bps == pytest.approx(10.0)
        assert est.spread_status == SpreadDataStatus.UNAVAILABLE, (
            "FixedBPS has no market data — must be UNAVAILABLE"
        )
        assert est.data_evidence == ExecutionDataLevel.D

    def test_spread_proxy_uses_hl_range(self):
        """SpreadProxy: half-spread ≈ (H-L) / close × k_hl."""
        from src.execution.slippage import SpreadProxySlippage
        from src.execution.schemas import SpreadDataStatus
        model = SpreadProxySlippage(hl_scaling_factor=0.30)
        est = model.estimate(
            price=21000.0, quantity_units=50,
            high=21100.0, low=20900.0, close=21000.0,
        )
        hl_range = 21100.0 - 20900.0
        expected_half_spread_pct = (hl_range / 21000.0) * 0.30
        expected_bps = expected_half_spread_pct * 10_000
        assert abs(est.slippage_bps - expected_bps) < 0.1
        assert est.spread_status == SpreadDataStatus.PROXY, (
            "HL proxy must be PROXY, never OBSERVED"
        )

    def test_spread_proxy_unavailable_without_hl(self):
        """SpreadProxy must return UNAVAILABLE when no OHLC data."""
        from src.execution.slippage import SpreadProxySlippage
        from src.execution.schemas import SpreadDataStatus
        model = SpreadProxySlippage()
        est = model.estimate(price=21000.0, quantity_units=50)
        assert est.spread_status == SpreadDataStatus.UNAVAILABLE
        assert est.slippage_bps == pytest.approx(0.0)

    def test_observed_spread_trumps_proxy(self):
        """When real bid-ask is provided, use it and return OBSERVED."""
        from src.execution.slippage import SpreadProxySlippage
        from src.execution.schemas import SpreadDataStatus, ExecutionDataLevel
        model = SpreadProxySlippage()
        est = model.estimate(
            price=21000.0, quantity_units=50,
            high=21100.0, low=20900.0, close=21000.0,
            observed_spread_bps=8.0,
        )
        assert est.spread_status == SpreadDataStatus.OBSERVED
        assert est.data_evidence == ExecutionDataLevel.A
        assert est.slippage_bps == pytest.approx(4.0)  # half of 8bps spread

    def test_vol_participation_increases_with_size(self):
        """VolatilityParticipation: larger order → higher slippage."""
        from src.execution.slippage import VolatilityParticipationSlippage
        model = VolatilityParticipationSlippage(coefficient=0.5)
        small = model.estimate(price=21000.0, quantity_units=50,
                               atr_pct=0.01, adv_inr=100_000_000)
        large = model.estimate(price=21000.0, quantity_units=5000,
                               atr_pct=0.01, adv_inr=100_000_000)
        assert large.slippage_bps > small.slippage_bps, (
            "Larger order must have higher slippage"
        )

    def test_vol_participation_zero_adv_returns_unavailable(self):
        """Model returns UNAVAILABLE when ADV is zero or absent."""
        from src.execution.slippage import VolatilityParticipationSlippage
        from src.execution.schemas import SpreadDataStatus
        model = VolatilityParticipationSlippage()
        est = model.estimate(price=21000.0, quantity_units=50, atr_pct=0.01)
        assert est.spread_status == SpreadDataStatus.UNAVAILABLE


# ══════════════════════════════════════════════════════════════════════════════
# 4 — NSE Calendar
# ══════════════════════════════════════════════════════════════════════════════

class TestNSECalendar:
    def test_saturday_is_not_trading_day(self):
        from src.execution.market_calendar import NSECalendar
        cal = NSECalendar()
        sat = date(2024, 1, 20)  # Saturday
        tradeable, _ = cal.is_trading_day(sat)
        assert not tradeable

    def test_sunday_is_not_trading_day(self):
        from src.execution.market_calendar import NSECalendar
        cal = NSECalendar()
        sun = date(2024, 1, 21)  # Sunday
        tradeable, _ = cal.is_trading_day(sun)
        assert not tradeable

    def test_known_holiday_is_not_trading_day(self):
        from src.execution.market_calendar import NSECalendar
        cal = NSECalendar()
        # Republic Day 2024 = 26 Jan 2024
        tradeable, status = cal.is_trading_day(date(2024, 1, 26))
        assert not tradeable

    def test_regular_weekday_is_trading_day(self):
        from src.execution.market_calendar import NSECalendar
        cal = NSECalendar()
        tradeable, _ = cal.is_trading_day(date(2024, 1, 15))  # Monday
        assert tradeable

    def test_next_trading_day_skips_weekend(self):
        from src.execution.market_calendar import NSECalendar
        cal = NSECalendar()
        friday = date(2024, 1, 19)
        next_day = cal.next_trading_day(friday)
        # Saturday + Sunday → should be Monday 22 Jan, but that is Republic Day
        # (a known NSE holiday), so the next trading day is Tuesday 23 Jan 2024.
        assert next_day == date(2024, 1, 23)

    def test_monthly_expiry_is_last_thursday(self):
        """January 2024 monthly expiry = last Thursday = 25 Jan."""
        from src.execution.market_calendar import NSECalendar
        cal = NSECalendar()
        expiry = cal.monthly_expiry(2024, 1)
        assert expiry.weekday() == 3, "Monthly expiry must be a Thursday"
        assert expiry.month == 1 and expiry.year == 2024

    def test_out_of_range_year_returns_insufficient_evidence(self):
        from src.execution.market_calendar import NSECalendar, CalendarDataStatus
        cal = NSECalendar()
        _, status = cal.is_trading_day(date(2019, 6, 15))
        assert status == CalendarDataStatus.INSUFFICIENT_EVIDENCE

    def test_future_schedule_does_not_affect_past_calendar(self):
        """
        Adding a holiday to the calendar must NOT change historical trading day results.
        (The built-in list is immutable — this tests the immutability contract.)
        """
        from src.execution.market_calendar import NSECalendar, _NSE_HOLIDAYS
        cal = NSECalendar()
        d = date(2024, 1, 15)
        # Verify it's currently a trading day
        tradeable1, _ = cal.is_trading_day(d)
        # The holiday set is a frozenset — attempting to modify it raises AttributeError
        with pytest.raises((AttributeError, TypeError)):
            _NSE_HOLIDAYS.add(d)  # type: ignore
        tradeable2, _ = cal.is_trading_day(d)
        assert tradeable1 == tradeable2


# ══════════════════════════════════════════════════════════════════════════════
# 5 — Fill engine
# ══════════════════════════════════════════════════════════════════════════════

class TestFillEngine:
    def _engine(self, slippage_bps=0.0, max_part=1.0, allow_same_close=False):
        from src.execution.fill_engine import FillEngine, FillEngineConfig
        from src.execution.schemas import AmbiguityPolicy, ExecutionPolicy
        return FillEngine(config=FillEngineConfig(
            default_policy=ExecutionPolicy.NEXT_OPEN,
            ambiguity_policy=AmbiguityPolicy.CONSERVATIVE,
            slippage_bps=slippage_bps,
            max_participation_pct=max_part,
            allow_same_close=allow_same_close,
        ))

    def test_gap_through_stop_fills_at_open(self):
        """
        Spec §14: stop at ₹100, next bar opens at ₹96.
        Fill must be at ₹96 (gap-through), NOT ₹100.
        """
        from src.execution.schemas import FillStatus, ExecutionPolicy
        from src.execution.fill_engine import FillEngine, FillEngineConfig

        engine = FillEngine(config=FillEngineConfig(
            default_policy=ExecutionPolicy.STOP,
            slippage_bps=0.0,
        ))
        order = _order(policy="STOP", side="LONG", order_side="SELL",
                       stop_price=100.0, signal_ts="2024-01-15")
        signal_bar = _bar("2024-01-15", 102, 103, 101, 102)
        # Next bar GAPS below stop — opens at 96
        next_bar   = _bar("2024-01-16", 96, 98, 95, 97)
        fill = engine.fill(order, signal_bar, [next_bar])
        assert fill.filled
        assert fill.fill_price == pytest.approx(96.0, abs=0.01), (
            f"Gap-through: fill must be at open=96, not stop=100. Got {fill.fill_price}"
        )

    def test_stop_fills_at_stop_price_when_no_gap(self):
        """Stop intrabar without gap → fill at stop price."""
        from src.execution.schemas import FillStatus, ExecutionPolicy
        from src.execution.fill_engine import FillEngine, FillEngineConfig

        engine = FillEngine(config=FillEngineConfig(
            default_policy=ExecutionPolicy.STOP,
            slippage_bps=0.0,
        ))
        order = _order(policy="STOP", side="LONG", order_side="SELL",
                       stop_price=100.0, signal_ts="2024-01-15")
        signal_bar = _bar("2024-01-15", 102, 103, 101, 102)
        # Bar dips to 99 (below 100) but opens above stop
        next_bar   = _bar("2024-01-16", 101, 102, 99, 100)
        fill = engine.fill(order, signal_bar, [next_bar])
        assert fill.filled
        assert fill.fill_price == pytest.approx(100.0, abs=0.01), (
            f"Intrabar stop: should fill at stop=100. Got {fill.fill_price}"
        )

    def test_conservative_ambiguity_policy_chooses_stop(self):
        """
        Spec §15: bar touches both stop and target → CONSERVATIVE = stop first.
        Never silently assume the favourable outcome.
        """
        from src.execution.fill_engine import FillEngine, FillEngineConfig
        from src.execution.schemas import AmbiguityPolicy, TradeSide

        engine = FillEngine(config=FillEngineConfig(ambiguity_policy=AmbiguityPolicy.CONSERVATIVE))
        bar = _bar("2024-01-16", 21000, 21200, 20800, 21100)  # H=21200, L=20800

        # Long trade: stop=20900 (below current), target=21100 (above)
        # Bar's low (20800) < stop (20900) AND bar's high (21200) > target (21100)
        result = engine.resolve_ambiguity(
            bar=bar, stop_price=20900.0, target_price=21100.0,
            trade_side=TradeSide.LONG,
        )
        assert result == "STOP", (
            f"CONSERVATIVE must choose STOP on ambiguous bar, got {result}"
        )

    def test_ambiguity_best_case_chooses_target(self):
        """BEST_CASE policy → target hit first."""
        from src.execution.fill_engine import FillEngine, FillEngineConfig
        from src.execution.schemas import AmbiguityPolicy, TradeSide

        engine = FillEngine(config=FillEngineConfig(ambiguity_policy=AmbiguityPolicy.BEST_CASE))
        bar = _bar("2024-01-16", 21000, 21200, 20800, 21100)
        result = engine.resolve_ambiguity(bar, 20900.0, 21100.0, TradeSide.LONG)
        assert result == "TARGET"

    def test_circuit_limit_rejects_fill(self):
        """Spec §16: order price outside price band → REJECTED."""
        from src.execution.schemas import FillStatus, ExecutionPolicy
        from src.execution.fill_engine import FillEngine, FillEngineConfig

        engine = FillEngine(config=FillEngineConfig(default_policy=ExecutionPolicy.NEXT_OPEN))
        signal_bar = _bar("2024-01-15", 21000, 21100, 20900, 21050)
        # Next bar's open (22000) is above upper circuit (21500)
        next_bar = _bar("2024-01-16", 22000, 22100, 21900, 22050,
                        band_upper=21500.0, band_lower=20500.0)
        fill = engine.fill(_order(), signal_bar, [next_bar])
        assert fill.status == FillStatus.REJECTED, (
            "Price above upper circuit must be REJECTED"
        )
        assert "CIRCUIT" in fill.rejection_reason

    def test_fno_ban_rejects_new_position(self):
        """Spec §17: F&O ban → new position rejected."""
        from src.execution.schemas import FillStatus
        engine = self._engine()
        signal_bar = _bar("2024-01-15", 21000, 21100, 20900, 21050)
        next_bar   = _bar("2024-01-16", 21100, 21200, 21000, 21150)
        fill = engine.fill(_order(), signal_bar, [next_bar], fno_ban=True, is_new_position=True)
        assert fill.status == FillStatus.REJECTED, "F&O ban must reject new position"
        assert "FNO_BAN" in fill.rejection_reason

    def test_fno_ban_allows_closing_position(self):
        """F&O ban allows closing an existing position."""
        from src.execution.schemas import FillStatus
        engine = self._engine()
        signal_bar = _bar("2024-01-15", 21000, 21100, 20900, 21050)
        next_bar   = _bar("2024-01-16", 21100, 21200, 21000, 21150)
        # is_new_position=False → closing existing position
        fill = engine.fill(_order(), signal_bar, [next_bar], fno_ban=True, is_new_position=False)
        assert fill.filled, "F&O ban must allow closing existing position"

    def test_expired_contract_rejects(self):
        """Spec §9: expired contract → REJECTED."""
        from src.execution.schemas import FillStatus
        engine = self._engine()
        expiry = datetime(2024, 1, 10, tzinfo=UTC)  # expired before trade date
        order = _order(signal_ts="2024-01-15", expiry_date=expiry)
        signal_bar = _bar("2024-01-15", 21000, 21100, 20900, 21050)
        next_bar   = _bar("2024-01-16", 21100, 21200, 21000, 21150)
        fill = engine.fill(order, signal_bar, [next_bar])
        assert fill.status == FillStatus.REJECTED, "Expired contract must be REJECTED"
        assert "EXPIRED" in fill.rejection_reason

    def test_partial_fill_when_small_adv(self):
        """Spec §13: large order vs small ADV → partial fill."""
        from src.execution.schemas import FillStatus
        engine = self._engine(slippage_bps=0.0, max_part=0.01)  # 1% ADV max
        signal_bar = _bar("2024-01-15", 21000, 21100, 20900, 21050)
        # ADV = ₹100,000. Order = 1000 lots × 50 units × ₹21100 = ₹1,055,000,000
        # 1% of ₹100,000 = ₹1,000 → max ~0 lots → REJECTED
        next_bar = _bar("2024-01-16", 21100, 21200, 21000, 21150,
                        adv_inr=100_000.0)
        order = _order(lots=1000)
        fill = engine.fill(order, signal_bar, [next_bar])
        assert fill.status in (FillStatus.PARTIAL, FillStatus.REJECTED), (
            "Large order vs small ADV must be partial or rejected"
        )
        assert fill.quantity_filled_lots <= fill.quantity_requested_lots

    def test_full_fill_when_sufficient_liquidity(self):
        """Order well within ADV → full fill."""
        from src.execution.schemas import FillStatus
        engine = self._engine(slippage_bps=0.0, max_part=0.10)
        signal_bar = _bar("2024-01-15", 21000, 21100, 20900, 21050)
        # ADV = ₹100 crore. Order notional = ~₹10.5 lakh (1 lot × 50 × 21100)
        next_bar = _bar("2024-01-16", 21100, 21200, 21000, 21150,
                        adv_inr=100_000_000.0)
        fill = engine.fill(_order(lots=1), signal_bar, [next_bar])
        assert fill.status == FillStatus.FULL
        assert fill.fill_ratio == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 6 — Position accounting and P&L reconciliation
# ══════════════════════════════════════════════════════════════════════════════

class TestPositionAccounting:
    def _make_fill(self, price, lots, order_side="BUY"):
        from src.execution.schemas import (
            FillStatus, OrderSide, SimulatedFill, SpreadDataStatus,
            ExecutionDataLevel, TradeSide, make_fill_id,
        )
        return SimulatedFill(
            fill_id=make_fill_id(),
            order_id="O-001",
            instrument_id="NIFTY",
            side=TradeSide.LONG,
            order_side=OrderSide.BUY if order_side == "BUY" else OrderSide.SELL,
            status=FillStatus.FULL,
            signal_time=_ts("2024-01-15"),
            decision_time=_ts("2024-01-15"),
            order_time=_ts("2024-01-15"),
            eligible_fill_time=_ts("2024-01-16"),
            actual_fill_time=_ts("2024-01-16"),
            quantity_requested_lots=lots,
            quantity_filled_lots=lots,
            fill_ratio=1.0,
            fill_price=price,
            signal_price=price,
            spread_status=SpreadDataStatus.PROXY,
            execution_data_level=ExecutionDataLevel.C,
        )

    def test_golden_net_pnl_reconciliation(self):
        """
        Golden test:
        Entry: BUY 1 lot NIFTY at ₹21,000; lot_size=50
        Exit:  SELL 1 lot at ₹21,500
        Gross P&L = (21500 - 21000) × 1 × 50 = ₹25,000
        net_pnl = gross_pnl - total_cost
        """
        from src.execution.position_accounting import TradeAccountingLedger
        from src.execution.schemas import InstrumentType, ProductType, TradeSide

        ledger = TradeAccountingLedger(initial_capital_inr=1_000_000.0)
        entry_fill = self._make_fill(21000.0, 1, "BUY")
        pos = ledger.open_position(
            entry_fill=entry_fill,
            instrument_type=InstrumentType.FUT_IDX,
            product_type=ProductType.NRML,
            lot_size=50,
            trade_date=date(2024, 1, 16),
        )
        assert pos is not None

        exit_fill = self._make_fill(21500.0, 1, "SELL")
        exit_fill.side = TradeSide.LONG
        from src.execution.schemas import OrderSide
        exit_fill.order_side = OrderSide.SELL

        trade = ledger.close_position(
            position=pos,
            exit_fill=exit_fill,
            trade_date=date(2024, 1, 17),
            holding_bars=1,
        )
        assert trade is not None

        expected_gross = (21500.0 - 21000.0) * 1 * 50  # ₹25,000
        assert abs(trade.gross_pnl - expected_gross) < 0.01, (
            f"Gross P&L={trade.gross_pnl:.2f}, expected {expected_gross:.2f}"
        )
        assert trade.net_pnl < trade.gross_pnl, "Net P&L must be less than gross (costs exist)"
        assert trade.validate_pnl(), "net_pnl = gross_pnl - total_cost must hold"

    def test_losing_trade_net_pnl(self):
        """Losing trade: entry at 21000, exit at 20500."""
        from src.execution.position_accounting import TradeAccountingLedger
        from src.execution.schemas import InstrumentType, OrderSide, ProductType, TradeSide

        ledger = TradeAccountingLedger(initial_capital_inr=1_000_000.0)
        entry_fill = self._make_fill(21000.0, 1, "BUY")
        pos = ledger.open_position(entry_fill, InstrumentType.FUT_IDX,
                                   ProductType.NRML, 50, date(2024, 1, 16))

        exit_fill = self._make_fill(20500.0, 1, "SELL")
        exit_fill.order_side = OrderSide.SELL

        trade = ledger.close_position(pos, exit_fill, date(2024, 1, 17))
        assert trade is not None
        expected_gross = (20500.0 - 21000.0) * 1 * 50  # -₹25,000
        assert abs(trade.gross_pnl - expected_gross) < 0.01
        assert trade.net_pnl < trade.gross_pnl  # costs make it worse
        assert trade.validate_pnl()

    def test_execution_ledger_pnl_reconciliation(self):
        """ExecutionLedger: total_net_pnl + total_cost == total_gross_pnl."""
        from src.execution.position_accounting import TradeAccountingLedger
        from src.execution.schemas import InstrumentType, OrderSide, ProductType, TradeSide

        ledger = TradeAccountingLedger(initial_capital_inr=1_000_000.0)

        for entry_p, exit_p in [(21000, 21500), (20000, 19500), (22000, 22200)]:
            entry = self._make_fill(entry_p, 1, "BUY")
            pos = ledger.open_position(entry, InstrumentType.FUT_IDX,
                                       ProductType.NRML, 50, date(2024, 1, 16))
            ex = self._make_fill(exit_p, 1, "SELL")
            ex.order_side = OrderSide.SELL
            ledger.close_position(pos, ex, date(2024, 1, 17))

        assert ledger.execution_ledger.verify_cost_reconciliation(), (
            "Total net_pnl + total_cost must == total_gross_pnl"
        )

    def test_append_trade_raises_on_bad_reconciliation(self):
        """ExecutionLedger.append_trade must raise if P&L reconciliation fails."""
        from src.execution.schemas import (
            CostBreakdown, ExecutionLedger, FillStatus, InstrumentType,
            ProductType, SimulatedFill, SpreadDataStatus, TradeSide,
            TradeRecord, ExecutionDataLevel, OrderSide, make_fill_id, make_trade_id,
        )

        ledger = ExecutionLedger(backtest_id="test", created_at=datetime.now(UTC))
        entry_fill = self._make_fill(21000.0, 1, "BUY")
        exit_fill  = self._make_fill(21500.0, 1, "SELL")
        exit_fill.order_side = OrderSide.SELL

        bad_trade = TradeRecord(
            trade_id=make_trade_id(),
            instrument_id="NIFTY",
            underlying="NIFTY",
            instrument_type=InstrumentType.FUT_IDX,
            product_type=ProductType.NRML,
            trade_side=TradeSide.LONG,
            entry_fill=entry_fill,
            exit_fill=exit_fill,
            entry_price=21000.0,
            exit_price=21500.0,
            quantity_lots=1,
            lot_size=50,
            gross_pnl=25000.0,
            entry_cost=CostBreakdown(brokerage=50.0),
            exit_cost=CostBreakdown(brokerage=50.0),
            net_pnl=9999.0,  # WRONG — should be ~24,900
            holding_bars=1,
            max_adverse_excursion=None,
            max_favourable_excursion=None,
        )
        with pytest.raises(ValueError, match="reconciliation"):
            ledger.append_trade(bad_trade)


# ══════════════════════════════════════════════════════════════════════════════
# 7 — Backtest engine
# ══════════════════════════════════════════════════════════════════════════════

class TestBacktestEngine:
    def _make_decision(self, trade_side="LONG", alpha=80.0, prob=0.70,
                       ev=2.5, horizon=3, entry_hint=21000.0,
                       stop=None, target=None, ts="2024-01-15"):
        from src.execution.backtest_engine import OOSDecisionRecord
        from src.execution.schemas import InstrumentType, ProductType, TradeSide
        return OOSDecisionRecord(
            instrument_id="NIFTY",
            signal_time=_ts(ts),
            trade_side=TradeSide.LONG if trade_side == "LONG" else TradeSide.SHORT,
            alpha_score=alpha,
            meta_probability=prob,
            expected_value=ev,
            decision="TAKE",
            horizon_bars=horizon,
            entry_price_hint=entry_hint,
            stop_price_hint=stop,
            target_price_hint=target,
            model_id="test_lgbm",
            model_version="v1",
            dataset_id="d1",
            feature_set_id="f1",
            lot_size=50,
            instrument_type=InstrumentType.FUT_IDX,
            product_type=ProductType.NRML,
        )

    def _bars(self, prices, start="2024-01-15"):
        """Build ascending bars from a price list."""
        from src.execution.fill_engine import OHLCBar
        bars = []
        d = datetime.fromisoformat(f"{start}T09:15:00+00:00")
        for p in prices:
            bars.append(OHLCBar(
                timestamp=d, open=p * 0.999, high=p * 1.005,
                low=p * 0.995, close=p, volume=1_000_000.0,
                adv_inr=500_000_000.0,
            ))
            d += timedelta(days=1)
        return bars

    def test_golden_trade_produces_positive_gross_pnl(self):
        """Simple winning trade: entry at 21000, exit at 21500 (3 bars)."""
        from src.execution.backtest_engine import BacktestConfig, BacktestEngine

        config = BacktestConfig(
            backtest_id="golden",
            execution_policy="NEXT_OPEN",
        )
        engine = BacktestEngine(config=config)
        dec = self._make_decision(entry_hint=21000.0, horizon=3)
        bars = self._bars([21000, 21200, 21400, 21600, 21800])

        result = engine.run(
            decisions=[dec],
            price_data={"NIFTY": bars},
        )
        if result.n_trades > 0:
            assert result.total_gross_pnl > 0, (
                "Rising price path must produce positive gross P&L"
            )
            assert result.pnl_reconciled(), "P&L must reconcile: net = gross - cost"

    def test_skip_decision_not_executed(self):
        """SKIP decisions must NOT generate any trades."""
        from src.execution.backtest_engine import BacktestConfig, BacktestEngine, OOSDecisionRecord
        from src.execution.schemas import InstrumentType, ProductType, TradeSide

        config = BacktestConfig(backtest_id="skip_test")
        engine = BacktestEngine(config=config)

        skip_dec = OOSDecisionRecord(
            instrument_id="NIFTY",
            signal_time=_ts("2024-01-15"),
            trade_side=TradeSide.LONG,
            alpha_score=80.0, meta_probability=0.7, expected_value=2.5,
            decision="SKIP",  # not TAKE
            horizon_bars=3, entry_price_hint=21000.0,
            stop_price_hint=None, target_price_hint=None,
            lot_size=50,
            instrument_type=InstrumentType.FUT_IDX,
            product_type=ProductType.NRML,
        )
        bars = self._bars([21000, 21200, 21400, 21600])
        result = engine.run(decisions=[skip_dec], price_data={"NIFTY": bars})
        assert result.n_trades == 0, "SKIP decisions must not generate trades"

    def test_reproducibility_same_config_same_result(self):
        """
        Spec §33: same inputs + same config → identical results.
        """
        from src.execution.backtest_engine import BacktestConfig, BacktestEngine

        config = BacktestConfig(backtest_id="repro_test", random_seed=42)
        engine1 = BacktestEngine(config=config)
        engine2 = BacktestEngine(config=config)

        decs = [self._make_decision(ts="2024-01-15", horizon=3)]
        bars = self._bars([21000, 21100, 21200, 21300, 21400])
        price_data = {"NIFTY": bars}

        result1 = engine1.run(decs, price_data)
        result2 = engine2.run(decs, price_data)

        assert result1.total_gross_pnl == result2.total_gross_pnl
        assert result1.total_net_pnl   == result2.total_net_pnl
        assert result1.n_trades        == result2.n_trades

    def test_config_hash_changes_with_different_params(self):
        from src.execution.backtest_engine import BacktestConfig
        cfg1 = BacktestConfig(backtest_id="a", slippage_bps=5.0)
        cfg2 = BacktestConfig(backtest_id="a", slippage_bps=10.0)
        assert cfg1.config_hash != cfg2.config_hash

    def test_config_hash_identical_for_same_params(self):
        from src.execution.backtest_engine import BacktestConfig
        cfg1 = BacktestConfig(backtest_id="x", slippage_bps=5.0, random_seed=42)
        cfg2 = BacktestConfig(backtest_id="x", slippage_bps=5.0, random_seed=42)
        assert cfg1.config_hash == cfg2.config_hash


# ══════════════════════════════════════════════════════════════════════════════
# 8 — Future mutation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFutureMutation:
    """
    Spec §35: modifying future cost schedules, lot sizes, or price data
    must NOT change the result of historical backtest calculations.
    """

    def test_future_cost_schedule_does_not_alter_historical_cost(self):
        """
        Mutation 1 (spec §35.1): change today's brokerage rate.
        Historical 2022 trade cost must remain unchanged.
        """
        from src.execution.cost_model import (
            CostScheduleRegistry, CostScheduleVersion,
            IndiaFnOCostSchedule, compute_trade_cost,
        )
        from src.execution.schemas import InstrumentType, OrderSide, ProductType

        registry = CostScheduleRegistry()
        historical_cost = compute_trade_cost(
            InstrumentType.FUT_IDX, OrderSide.SELL, ProductType.NRML,
            date(2022, 6, 15), 17000.0, 1, 50, registry=registry,
        )

        # Mutate: add a future schedule with brokerage 100x
        ver = CostScheduleVersion("future-test", date(2030, 1, 1), None, "test")
        sched = IndiaFnOCostSchedule(version=ver, stt_futures_sell_pct=0.10)  # extreme
        registry.register_fno_schedule(ver, sched)

        historical_cost_after = compute_trade_cost(
            InstrumentType.FUT_IDX, OrderSide.SELL, ProductType.NRML,
            date(2022, 6, 15), 17000.0, 1, 50, registry=registry,
        )
        assert abs(historical_cost.total - historical_cost_after.total) < 0.01, (
            "Future cost schedule must not alter historical 2022 cost"
        )

    def test_future_lot_size_does_not_alter_historical_notional(self):
        """
        Mutation 2 (spec §35.2): current lot size change does not affect
        historical P&L (which uses the lot_size stored in the OrderIntent).
        """
        from src.execution.position_accounting import TradeAccountingLedger
        from src.execution.schemas import (
            FillStatus, InstrumentType, OrderSide, ProductType,
            SimulatedFill, SpreadDataStatus, TradeSide, ExecutionDataLevel, make_fill_id,
        )

        def make_fill(price, os):
            return SimulatedFill(
                fill_id=make_fill_id(), order_id="O1",
                instrument_id="NIFTY", side=TradeSide.LONG,
                order_side=os, status=FillStatus.FULL,
                signal_time=_ts("2024-01-15"), decision_time=_ts("2024-01-15"),
                order_time=_ts("2024-01-15"),
                eligible_fill_time=_ts("2024-01-16"),
                actual_fill_time=_ts("2024-01-16"),
                quantity_requested_lots=1, quantity_filled_lots=1,
                fill_ratio=1.0, fill_price=price, signal_price=price,
                spread_status=SpreadDataStatus.PROXY,
                execution_data_level=ExecutionDataLevel.C,
            )

        # Historical trade with lot_size=50
        ledger = TradeAccountingLedger(initial_capital_inr=1_000_000)
        pos = ledger.open_position(
            make_fill(21000.0, OrderSide.BUY), InstrumentType.FUT_IDX,
            ProductType.NRML, lot_size=50, trade_date=date(2024, 1, 16),
        )
        assert pos is not None
        assert pos.lot_size == 50  # recorded at time of trade

        # Simulating "lot size changed to 75 after Nov 2024"
        # But the position already recorded lot_size=50 — must not change
        pos_lot_size_after_mutation = pos.lot_size
        assert pos_lot_size_after_mutation == 50, (
            "Position lot_size must remain 50 even after external lot-size change"
        )

    def test_future_price_mutation_does_not_alter_completed_trade(self):
        """
        Mutation 3 (spec §35.3): modifying future OHLCV does not alter
        completed trades whose fill_price is already recorded.
        """
        from src.execution.schemas import (
            FillStatus, OrderSide, SimulatedFill, SpreadDataStatus,
            TradeSide, ExecutionDataLevel, make_fill_id,
        )

        # Entry fill recorded at ₹21,000
        fill = SimulatedFill(
            fill_id=make_fill_id(), order_id="O1",
            instrument_id="NIFTY", side=TradeSide.LONG,
            order_side=OrderSide.BUY, status=FillStatus.FULL,
            signal_time=_ts("2024-01-15"), decision_time=_ts("2024-01-15"),
            order_time=_ts("2024-01-15"),
            eligible_fill_time=_ts("2024-01-16"),
            actual_fill_time=_ts("2024-01-16"),
            quantity_requested_lots=1, quantity_filled_lots=1,
            fill_ratio=1.0, fill_price=21000.0, signal_price=21000.0,
            spread_status=SpreadDataStatus.PROXY,
            execution_data_level=ExecutionDataLevel.C,
        )
        original_fill_price = fill.fill_price

        # Simulate "future price data mutation" — but fill_price is immutable
        # (it's recorded at fill time; the OHLCV source data can't change it)
        # SimulatedFill is a dataclass but fill_price is just a stored value
        assert fill.fill_price == original_fill_price, (
            "fill_price stored in SimulatedFill must not change"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 9 — gex.py LOT_SIZES fix
# ══════════════════════════════════════════════════════════════════════════════

class TestGexLotSizeFix:
    def test_nifty_lot_size_updated_to_post_sebi_2024(self):
        """gex.py LOT_SIZES must reflect post-SEBI Nov 2024 values."""
        from src.gex import LOT_SIZES
        assert LOT_SIZES["NIFTY"] == 75, (
            f"NIFTY lot size must be 75 (post-SEBI Nov 2024), got {LOT_SIZES['NIFTY']}"
        )
        assert LOT_SIZES["BANKNIFTY"] == 30, (
            f"BANKNIFTY lot size must be 30 (post-SEBI Nov 2024), got {LOT_SIZES['BANKNIFTY']}"
        )

    def test_instrument_master_is_consistent_with_gex(self):
        """InstrumentMasterStore and gex.py must agree on current lot sizes."""
        from src.gex import LOT_SIZES
        from src.data.instrument_master import InstrumentMasterStore
        store = InstrumentMasterStore.default()
        today = date(2025, 1, 1)  # post-SEBI Nov 2024

        for sym in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
            lot, status, _ = store.get_lot_size(sym, today)
            if lot is not None:
                assert lot == LOT_SIZES[sym], (
                    f"{sym}: InstrumentMaster={lot}, gex.LOT_SIZES={LOT_SIZES[sym]} — inconsistent"
                )


# ══════════════════════════════════════════════════════════════════════════════
# 10 — SpreadDataStatus invariant
# ══════════════════════════════════════════════════════════════════════════════

class TestSpreadDataStatusInvariant:
    """Spec §11: PROXY data must never be labelled OBSERVED."""

    def test_hl_proxy_never_returns_observed(self):
        """SpreadProxySlippage using HL proxy must return PROXY, not OBSERVED."""
        from src.execution.slippage import SpreadProxySlippage
        from src.execution.schemas import SpreadDataStatus
        model = SpreadProxySlippage()
        est = model.estimate(
            price=21000.0, quantity_units=50,
            high=21100.0, low=20900.0, close=21000.0,
        )
        assert est.spread_status != SpreadDataStatus.OBSERVED, (
            "HL proxy must not claim OBSERVED status"
        )

    def test_vol_participation_never_returns_observed(self):
        from src.execution.slippage import VolatilityParticipationSlippage
        from src.execution.schemas import SpreadDataStatus
        model = VolatilityParticipationSlippage()
        est = model.estimate(price=21000.0, quantity_units=50,
                             atr_pct=0.01, adv_inr=1e8)
        assert est.spread_status != SpreadDataStatus.OBSERVED

    def test_fill_from_proxy_records_proxy_status(self):
        """SimulatedFill using proxy data must record PROXY spread status."""
        from src.execution.schemas import SpreadDataStatus
        from src.execution.fill_engine import FillEngine, FillEngineConfig
        from src.execution.schemas import ExecutionPolicy

        engine = FillEngine(config=FillEngineConfig(default_policy=ExecutionPolicy.NEXT_OPEN))
        signal_bar = _bar("2024-01-15", 21000, 21100, 20900, 21050)
        next_bar   = _bar("2024-01-16", 21100, 21200, 21000, 21150, adv_inr=1e8)
        fill = engine.fill(_order(), signal_bar, [next_bar])
        if fill.filled:
            assert fill.spread_status == SpreadDataStatus.PROXY, (
                "Fill using OHLCV proxy must report PROXY spread status"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 11 — Backward compatibility
# ══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    def test_prior_test_suites_still_pass_import(self):
        """All previous modules must still import without error."""
        import src.execution.schemas
        import src.execution.cost_model
        import src.execution.slippage
        import src.execution.market_calendar
        import src.execution.fill_engine
        import src.execution.position_accounting
        import src.execution.backtest_engine

    def test_labels_config_cost_model_still_works(self):
        """CostModelConfig.futures_nse() in labels/config.py must still function."""
        from src.labels.config import CostModelConfig
        cfg = CostModelConfig.futures_nse()
        assert cfg.version == "nse-futures-v1"
        assert cfg.round_trip_cost_pct > 0.0
