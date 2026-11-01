# Implementation Plan: Smart Money Scanners

## Overview

TDD-first Python implementation of four Smart Money Concept (SMC) scanner modules that
extend `src/scanners/` and expose four new routes on the existing `src/api/scanners.py`
router. All 29 requirements (7 per scanner + API + PBT groups) and all 20 correctness
properties from the design are covered.

**Modules**: `fair_value_gap` (Scanner 2), `supply_demand` (Scanner 3),
`liquidity_sweep` (Scanner 4), `market_structure` (Scanner 5).  
**Endpoints**: `GET /v2/scanners/fair-value-gaps`, `/supply-demand`,
`/liquidity-sweeps`, `/market-structure`.

**Discipline**: Every TDD red-phase test in Task 2 is written before any production code
in Tasks 3–6. All 20 `hypothesis` property-based tests enforce the formal correctness
properties. Each implementation task makes a specific slice of red-phase tests go green.
Tasks marked `*` are optional and can be skipped for a faster MVP.

---

## Tasks

- [x] 1. Scaffold all four scanner packages — configuration and schemas
  - [x] 1.1 Create `src/scanners/fair_value_gap/` package with config and schemas
    - Create `src/scanners/fair_value_gap/__init__.py` with docstring:
      `"""Fair Value Gap Scanner — 3-candle imbalance zone detection for the F&O universe."""`
    - Implement `FVGConfig(BaseSettings)` in `src/scanners/fair_value_gap/config.py`:
      - `SettingsConfigDict(env_prefix="FVG_", extra="ignore")`
      - Fields with defaults: `atr_window: int = 14`, `impulse_atr_multiplier: float = 1.0`,
        `min_gap_atr_multiplier: float = 0.5`, `fvg_expiry_bars: int = 100`,
        `tf_5m_expiry_bars: int = 288`, `tf_15m_expiry_bars: int = 96`,
        `tf_1h_expiry_bars: int = 24`, `tf_1d_expiry_bars: int = 20`,
        `confluence_price_pct: float = 0.01`, `default_limit: int = 50`, `max_limit: int = 200`
      - All fields use `pydantic.Field(description=...)` for self-documentation
    - Implement enums and Pydantic models in `src/scanners/fair_value_gap/schemas.py`:
      - `FVGDirection(str, Enum)`: `BULLISH = "bullish"`, `BEARISH = "bearish"`
      - `FVGLifecycle(str, Enum)`: `UNFILLED`, `PARTIALLY_FILLED`, `FILLED`, `EXPIRED`
      - `FVGZone(BaseSchema)` with all fields: `symbol`, `direction`, `timeframe`,
        `fvg_high`, `fvg_low`, `fvg_midpoint`, `fvg_timestamp`, `candle1_bar_index`,
        `candle3_bar_index`, `impulse_atr`, `lifecycle_state=FVGLifecycle.UNFILLED`,
        `fill_timestamp=None`, `gap_size_score=0.0`, `volume_score=0.0`, `combined_score=0.0`
      - `FVGScanResult(BaseSchema)`: `zones: list[FVGZone]`, `scanned_at: datetime`,
        `n_symbols_scanned: int`
      - `FVGScanResponse(BaseSchema)`: `scanned_at`, `n_symbols_scanned`, `n_zones`,
        `zones: list[FVGZone]`
      - All models inherit from `src.schemas.base.BaseSchema` (`strict=True, frozen=True`)
    - _Requirements: 1.1–1.7, 2.1–2.6, 3.1–3.7, 4.1–4.7, 5.1–5.6, 6.1–6.9, 7.1–7.8_

  - [x] 1.2 Create `src/scanners/supply_demand/` package with config and schemas
    - Create `src/scanners/supply_demand/__init__.py` with docstring:
      `"""Supply & Demand Zone Scanner — consolidation-departure zone detection for the F&O universe."""`
    - Implement `SDConfig(BaseSettings)` in `src/scanners/supply_demand/config.py`:
      - `SettingsConfigDict(env_prefix="SD_", extra="ignore")`
      - Fields: `atr_window: int = 14`, `base_body_threshold: float = 0.5`,
        `departure_body_threshold: float = 1.5`, `min_zone_width_multiplier: float = 0.3`,
        `min_base_candles: int = 2`, `max_base_candles: int = 5`,
        `confluence_price_pct: float = 0.01`, `default_limit: int = 50`, `max_limit: int = 200`
    - Implement enums and Pydantic models in `src/scanners/supply_demand/schemas.py`:
      - `SDZoneType(str, Enum)`: `RBR`, `DBD`, `RBD`, `DBR`
      - `SDDirection(str, Enum)`: `SUPPLY = "SUPPLY"`, `DEMAND = "DEMAND"`
      - `SDLifecycle(str, Enum)`: `FRESH`, `TESTED`, `USED`, `BROKEN`
      - `SDZone(BaseSchema)` with all fields: `symbol`, `direction`, `zone_type`,
        `timeframe`, `zone_high`, `zone_low`, `entry_leg_timestamp`, `departure_timestamp`,
        `base_start_bar_index`, `base_end_bar_index`, `departure_bar_index`,
        `lifecycle_state=SDLifecycle.FRESH`, `zone_strength=1.0`, `broken_timestamp=None`,
        `multi_tf_confluence=False`, `departure_strength=0.0`, `combined_score=0.0`
      - `SDScanResult(BaseSchema)`: `zones`, `scanned_at`, `n_symbols_scanned`
      - `SDScanResponse(BaseSchema)`: `scanned_at`, `n_symbols_scanned`, `n_zones`, `zones`
    - _Requirements: 8.1–8.7, 9.1–9.6, 10.1–10.7, 11.1–11.7, 12.1–12.5, 13.1–13.9, 14.1–14.8_

  - [x] 1.3 Create `src/scanners/liquidity_sweep/` package with config and schemas
    - Create `src/scanners/liquidity_sweep/__init__.py` with docstring:
      `"""Liquidity Sweep Scanner — stop-hunt reversal event detection for the F&O universe."""`
    - Implement `SweepConfig(BaseSettings)` in `src/scanners/liquidity_sweep/config.py`:
      - `SettingsConfigDict(env_prefix="SWEEP_", extra="ignore")`
      - Fields: `atr_window: int = 14`, `swing_lookback: int = 10`,
        `volume_confirmation_multiplier: float = 1.5`, `avg_volume_window: int = 20`,
        `require_reversal_confirmation: bool = True`,
        `equal_level_tolerance_pct: float = 0.001`,
        `confluence_atr_multiplier: float = 0.5`,
        `default_limit: int = 50`, `max_limit: int = 200`
    - Implement enums and Pydantic models in `src/scanners/liquidity_sweep/schemas.py`:
      - `SweepDirection(str, Enum)`: `BULLISH = "bullish"`, `BEARISH = "bearish"`
      - `SweepLifecycle(str, Enum)`: `ACTIVE`, `CONFIRMED`, `INVALIDATED`
      - `SweepSignal(BaseSchema)` with all fields: `symbol`, `direction`, `timeframe`,
        `swept_level`, `sweep_zone_high`, `sweep_zone_low`, `sweep_timestamp`,
        `sweep_bar_index`, `lifecycle_state=SweepLifecycle.ACTIVE`,
        `equal_levels_count=1`, `sweep_magnitude=0.0`, `volume_ratio=0.0`,
        `conviction_score=0.0`, `confirmed_timestamp=None`, `multi_tf_confluence=False`
      - `SweepScanResult(BaseSchema)`: `signals`, `scanned_at`, `n_symbols_scanned`
      - `SweepScanResponse(BaseSchema)`: `scanned_at`, `n_symbols_scanned`,
        `n_signals`, `signals`
    - _Requirements: 15.1–15.6, 16.1–16.6, 17.1–17.6, 18.1–18.7, 19.1–19.5, 20.1–20.8, 21.1–21.8_

  - [x] 1.4 Create `src/scanners/market_structure/` package with config and schemas
    - Create `src/scanners/market_structure/__init__.py` with docstring:
      `"""Market Structure Scanner — BoS/ChoCh signal detection for the F&O universe."""`
    - Implement `MSConfig(BaseSettings)` in `src/scanners/market_structure/config.py`:
      - `SettingsConfigDict(env_prefix="MS_", extra="ignore")`
      - Fields: `atr_window: int = 14`, `swing_lookback: int = 10`,
        `tf_5m_swing_lookback: int = 5`, `tf_15m_swing_lookback: int = 7`,
        `tf_1h_swing_lookback: int = 10`, `tf_1d_swing_lookback: int = 5`,
        `confluence_bars_window: int = 3`,
        `default_limit: int = 50`, `max_limit: int = 200`
    - Implement enums and Pydantic models in `src/scanners/market_structure/schemas.py`:
      - `MSSignalType(str, Enum)`: `BOS_BULLISH`, `BOS_BEARISH`,
        `CHOCH_BULLISH`, `CHOCH_BEARISH`
      - `TrendState(str, Enum)`: `UNDEFINED`, `BULLISH`, `BEARISH`
      - `MSSignal(BaseSchema)` with all fields: `symbol`, `signal_type`, `timeframe`,
        `broken_level`, `close_at_break`, `break_timestamp`, `break_bar_index`,
        `trend_state` (state AFTER signal), `is_choch`, `swing_count`,
        `break_magnitude=0.0`, `conviction_score=0.0`, `invalidated=False`,
        `multi_tf_confluence=False`
      - `MSScanResult(BaseSchema)`: `signals`, `scanned_at`, `n_symbols_scanned`
      - `MSScanResponse(BaseSchema)`: `scanned_at`, `n_symbols_scanned`,
        `n_signals`, `signals`
    - _Requirements: 22.1–22.7, 23.1–23.6, 24.1–24.7, 25.1–25.7, 26.1–26.7, 27.1–27.5, 28.1–28.9, 29.1–29.9_

- [x] 2. Write full TDD red-phase test suite in `tests/test_smart_money_scanners.py`
  (all tests must fail initially — no production code exists yet)

  - [x] 2.1 Write Hypothesis strategies and test file scaffolding
    - Import all schema types, configs, and pytest markers; stub out empty `pass` bodies
      for every named test function so the file is valid Python before any impl exists
    - Implement `ohlcv_strategy(min_rows, max_rows)` `@st.composite` strategy: generates
      a `pd.DataFrame` with UTC `DatetimeIndex`, columns `open/high/low/close/volume`,
      enforcing `high ≥ max(open,close) ≥ min(open,close) ≥ low > 0` and
      `volume ≥ 0`; use
      `price_st = st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False)`
    - Implement `fvg_zone_strategy()` `@st.composite`: draws `fvg_low` then
      `fvg_high > fvg_low`, computes `fvg_midpoint = (fvg_high + fvg_low) / 2`, draws
      all enum fields from `FVGDirection` / `FVGLifecycle`, ensures
      `candle3_bar_index >= candle1_bar_index + 2`, all timestamps in UTC
    - Implement `sd_zone_strategy()` `@st.composite`: draws `zone_low`, `zone_high`,
      `zone_strength` from `{1.0, 0.7, 0.4, 0.0}`, all enum fields from
      `SDZoneType` / `SDLifecycle` with consistent `direction` mapping
      (RBR/DBR→DEMAND, RBD/DBD→SUPPLY), draws three bar indices ensuring
      `base_start <= base_end < departure`
    - Implement `sweep_signal_strategy()` `@st.composite`: draws `swept_level`,
      `sweep_zone_high = swept_level + delta` (delta > 0),
      `sweep_zone_low = swept_level - delta`, all enum fields from
      `SweepDirection` / `SweepLifecycle`, non-negative score fields
    - Implement `ms_signal_strategy()` `@st.composite`: draws `broken_level`,
      `close_at_break` with consistent direction (`close_at_break > broken_level` for
      BULLISH BoS/ChoCh, `< broken_level` for BEARISH), all enum fields from
      `MSSignalType` / `TrendState`, `is_choch` consistent with `signal_type`
    - _Requirements: 7.3, 7.7, 14.3, 14.7, 21.7, 29.8_

  - [x] 2.2 Write TDD red-phase tests for FVG scanner (Req 1–7)
    - `test_fvg_bullish_detection_req_1_1` — hand-craft a 3-candle OHLCV where
      `low[2] > high[0]` AND `close[2] > high[0]`; call `detect_fvg_zones`; assert
      `len == 1`, `z.direction == BULLISH`, `z.fvg_low == df.high.iloc[0]`,
      `z.fvg_high == df.low.iloc[2]` (Req 1.1)
    - `test_fvg_bearish_detection_req_1_2` — symmetric: `high[2] < low[0]` AND
      `close[2] < low[0]`; assert `z.direction == BEARISH`,
      `z.fvg_high == df.low.iloc[0]`, `z.fvg_low == df.high.iloc[2]` (Req 1.2)
    - `test_fvg_midpoint_computed_req_1_3` — detect any bullish FVG; assert
      `z.fvg_midpoint == pytest.approx((z.fvg_high + z.fvg_low) / 2)` (Req 1.3)
    - `test_fvg_overlapping_windows_req_1_4` — craft 4-candle OHLCV with two valid
      overlapping FVG windows (c0-c1-c2 and c1-c2-c3); assert `len(zones) == 2` (Req 1.4)
    - `test_fvg_fewer_than_3_bars_req_1_5` — call with 2-row DataFrame; assert `== []`,
      no exception (Req 1.5)
    - `test_fvg_nan_skipped_req_1_6` — insert NaN in `close[1]`; assert that window
      is skipped but adjacent valid windows are still detected (Req 1.6)
    - `test_fvg_timestamp_is_candle3_req_1_7` — assert `z.fvg_timestamp == df.index[2]`
      for a single-window bullish FVG (Req 1.7)
    - `test_fvg_impulse_filter_rejects_small_body_req_2_1` — set `impulse_atr_multiplier=2.0`;
      craft a window where body of c1 is exactly `1.9 × ATR14[1]`; assert `== []` (Req 2.1)
    - `test_fvg_gap_size_filter_rejects_micro_gap_req_2_2` — craft a bullish FVG where
      `fvg_high - fvg_low = 0.4 × ATR14[2]`; default `min_gap_atr_multiplier=0.5`;
      assert `== []` (Req 2.2)
    - `test_fvg_atr_trailing_only_req_2_4` — compute `detect_fvg_zones` on a 20-bar
      series; inspect returned `impulse_atr` values; verify they equal the trailing
      rolling mean at bar `i+1`, not a centered value (Req 2.4)
    - `test_fvg_atr_min_periods_req_2_5` — series of 5 bars; no exception; ATR
      computed with available bars only (Req 2.5)
    - `test_fvg_lifecycle_unfilled_initial_req_3_1` — newly created zone via
      `update_fvg_lifecycle`; assert `lifecycle_state == UNFILLED` (Req 3.1)
    - `test_fvg_lifecycle_partial_fill_req_3_2` — pass candle entering zone range
      (`low ≤ fvg_high AND high ≥ fvg_low`) to an UNFILLED zone; assert
      `PARTIALLY_FILLED` (Req 3.2)
    - `test_fvg_lifecycle_filled_req_3_3` — pass candle crossing midpoint for a
      PARTIALLY_FILLED bullish zone (`close < fvg_midpoint`); assert `FILLED` (Req 3.3)
    - `test_fvg_lifecycle_terminal_filled_req_3_4` — call `update_fvg_lifecycle` on a
      FILLED zone with any candle; assert state unchanged (Req 3.4)
    - `test_fvg_lifecycle_expiry_req_3_5` — UNFILLED zone with
      `bar_age > fvg_expiry_bars`; assert `EXPIRED` (Req 3.5)
    - `test_fvg_scorer_gap_size_score_req_4_1` — compute scorer on known zone; assert
      `gap_size_score == pytest.approx((fvg_high - fvg_low) / atr_at_c3)` (Req 4.1)
    - `test_fvg_scorer_volume_score_req_4_2` — assert
      `volume_score == pytest.approx(vol_c2 / avg_vol_20_c2)` (Req 4.2)
    - `test_fvg_scorer_combined_score_req_4_3` — assert
      `combined_score == pytest.approx(gap_size_score * volume_score)` (Req 4.3)
    - `test_fvg_scorer_sorted_descending_req_4_4` — rank 3 zones with known scores;
      verify output order is descending by `combined_score` (Req 4.4)
    - `test_fvg_scorer_zero_avg_volume_req_4_6` — avg_vol_20 = 0; assert
      `volume_score == 0.0`, zone placed at end (Req 4.6)
    - `test_fvg_timeframe_5m_expiry_req_5_2` — `timeframe="5m"` in config; assert
      effective expiry = 288 bars (Req 5.2)
    - `test_fvg_timeframe_1d_expiry_req_5_3` — `timeframe="1d"`, expiry = 20 (Req 5.3)
    - `test_fvg_invalid_timeframe_raises_req_5_4` — pass `timeframe="1w"`; assert
      `ValueError` with message listing `5m, 15m, 1h, 1d` (Req 5.4)
    - `test_fvg_api_returns_200` — `GET /v2/scanners/fair-value-gaps` with valid
      API key → HTTP 200, JSON body has `zones` list (Req 6.1)
    - `test_fvg_api_limit_param_req_6_2` — `?limit=3` with 5 zones → 3 returned (Req 6.2)
    - `test_fvg_api_direction_filter_req_6_3` — `?direction=bullish` → all zones are
      `BULLISH` (Req 6.3)
    - `test_fvg_api_timeframe_filter_req_6_4` — `?timeframe=1h` → all zones are `1h`
      (Req 6.4)
    - `test_fvg_api_empty_returns_200_req_6_7` — no active zones → 200 with empty list
      (Req 6.7)
    - `test_fvg_api_limit_out_of_range_req_6_9` — `?limit=0` and `?limit=201` → 422
      (Req 6.9)
    - Mark all `@pytest.mark.tdd` + `@pytest.mark.unit`; API tests also
      `@pytest.mark.integration`
    - _Requirements: 1.1–1.7, 2.1–2.6, 3.1–3.7, 4.1–4.7, 5.1–5.6, 6.1–6.9, 7.1–7.8_

  - [x] 2.3 Write TDD red-phase tests for S&D scanner (Req 8–14)
    - `test_sd_base_detection_req_8_1` — craft OHLCV with 3 small-body candles
      (`|close-open| < 0.5 × ATR14`) followed by an impulsive departure; assert
      `len(zones) == 1` and `base_start_bar_index` / `base_end_bar_index` correct (Req 8.1)
    - `test_sd_zone_bounds_req_8_2` — assert `zone_high == max(high[base_indices])` and
      `zone_low == min(low[base_indices])` (Req 8.2)
    - `test_sd_departure_filter_req_8_3` — departure body `< 1.5 × ATR14`; assert `== []`
      (Req 8.3)
    - `test_sd_min_zone_width_filter_req_8_4` — `zone_high - zone_low < 0.3 × ATR14`;
      assert `== []` (Req 8.4)
    - `test_sd_fewer_than_5_bars_req_8_5` — 4-row DataFrame; assert `== []` (Req 8.5)
    - `test_sd_nan_skipped_req_8_6` — NaN in base candle; assert window skipped (Req 8.6)
    - `test_sd_rbr_classification_req_9_1` — entry_leg closes UP, departure closes UP;
      assert `zone_type == RBR`, `direction == DEMAND` (Req 9.1)
    - `test_sd_dbd_classification_req_9_2` — entry DOWN, departure DOWN; assert `DBD`,
      `SUPPLY` (Req 9.2)
    - `test_sd_rbd_classification_req_9_3` — entry UP, departure DOWN; assert `RBD`,
      `SUPPLY` (Req 9.3)
    - `test_sd_dbr_classification_req_9_4` — entry DOWN, departure UP; assert `DBR`,
      `DEMAND` (Req 9.4)
    - `test_sd_no_entry_leg_skipped_req_9_5` — base starts at index 0; assert zone
      skipped (no entry_leg) (Req 9.5)
    - `test_sd_lifecycle_fresh_initial_req_10_1` — new SDZone; assert
      `lifecycle_state == FRESH`, `zone_strength == 1.0` (Req 10.1)
    - `test_sd_lifecycle_fresh_to_tested_req_10_2` — candle enters DEMAND zone without
      closing through (`close > zone_low`); assert `TESTED`, `zone_strength == 0.7`
      (Req 10.2)
    - `test_sd_lifecycle_tested_to_used_req_10_3` — second entry on TESTED zone without
      closing through; assert `USED`, `zone_strength == 0.4` (Req 10.3)
    - `test_sd_lifecycle_broken_req_10_4` — `close < zone_low` for DEMAND zone at any
      state; assert `BROKEN`, `zone_strength == 0.0` (Req 10.4)
    - `test_sd_lifecycle_terminal_broken_req_10_5` — call `update_sd_lifecycle` on BROKEN
      zone; assert unchanged (Req 10.5)
    - `test_sd_scorer_departure_strength_req_11_1` — assert
      `departure_strength == abs(close[dep] - open[dep]) / atr[dep]` (Req 11.1)
    - `test_sd_scorer_combined_score_req_11_3` — assert
      `combined_score == departure_strength * zone_strength` (Req 11.3)
    - `test_sd_scorer_sorted_descending_req_11_4` — rank 3 zones; verify descending
      `combined_score` order (Req 11.4)
    - `test_sd_scorer_zero_atr_req_11_6` — ATR=0; assert `departure_strength == 0.0`
      (Req 11.6)
    - `test_sd_multi_tf_confluence_req_12_5` — DEMAND zone on `1h` and `1d` with
      midpoints within 1%; assert both have `multi_tf_confluence == True` (Req 12.5)
    - `test_sd_invalid_timeframe_raises_req_12_3` — `ValueError` for `"1w"` (Req 12.3)
    - `test_sd_api_returns_200_req_13_1` — `GET /v2/scanners/supply-demand` → 200 (Req 13.1)
    - `test_sd_api_limit_req_13_2` — `?limit=2` with 5 zones → 2 returned (Req 13.2)
    - `test_sd_api_direction_filter_req_13_3` — `?direction=supply` → all `SUPPLY` (Req 13.3)
    - `test_sd_api_zone_type_filter_req_13_4` — `?zone_type=RBR` → all `RBR` (Req 13.4)
    - `test_sd_api_empty_returns_200_req_13_8` — no zones → 200, empty list (Req 13.8)
    - `test_sd_api_limit_out_of_range_req_13_9` — `?limit=0` → 422 (Req 13.9)
    - Mark all `@pytest.mark.tdd` + `@pytest.mark.unit`; API tests also
      `@pytest.mark.integration`
    - _Requirements: 8.1–8.7, 9.1–9.6, 10.1–10.7, 11.1–11.7, 12.1–12.5, 13.1–13.9, 14.1–14.8_

  - [x] 2.4 Write TDD red-phase tests for Liquidity Sweep scanner (Req 15–21)
    - `test_sweep_swing_high_trailing_req_15_1` — compute `detect_sweeps` on known series;
      verify `sw_high[k] == max(high[k-9:k+1])` (no look-ahead) (Req 15.1)
    - `test_sweep_swing_low_trailing_req_15_2` — symmetric for swing low (Req 15.2)
    - `test_sweep_liquidity_pool_highs_req_15_3` — two consecutive swing highs within
      tolerance; assert `equal_levels_count >= 2` (Req 15.3)
    - `test_sweep_too_few_bars_req_15_5` — `len(df) < swing_lookback + 1`; assert `== []`
      (Req 15.5)
    - `test_sweep_bullish_detection_req_16_1` — candle `k` with `low[k] < sw_low[k-1]`
      AND `close[k] > sw_low[k-1]` AND volume ≥ threshold; assert `SweepSignal` emitted
      with `direction == BULLISH`, `swept_level == sw_low[k-1]` (Req 16.1)
    - `test_sweep_bearish_detection_req_16_2` — symmetric bearish sweep (Req 16.2)
    - `test_sweep_volume_filter_req_16_3` — volume `< 1.5 × avg_vol_20`; assert `== []`
      (Req 16.3)
    - `test_sweep_zone_bounds_req_16_4` — assert `sweep_zone_high = swept_level + 0.5 × atr`,
      `sweep_zone_low = swept_level - 0.5 × atr` (Req 16.4)
    - `test_sweep_reversal_confirmation_req_16_5` — `require_reversal_confirmation=True`;
      `close[k+1] <= close[k]` (no reversal); assert `== []` (Req 16.5)
    - `test_sweep_lifecycle_active_initial_req_17_1` — new SweepSignal has
      `lifecycle_state == ACTIVE` (Req 17.1)
    - `test_sweep_lifecycle_confirmed_req_17_2` — next close continues reversal direction;
      assert `CONFIRMED` (Req 17.2)
    - `test_sweep_lifecycle_invalidated_req_17_3` — close crosses back through swept_level
      for bullish sweep (`close < swept_level`); assert `INVALIDATED` (Req 17.3)
    - `test_sweep_lifecycle_terminal_invalidated_req_17_4` — any further update on
      INVALIDATED; assert unchanged (Req 17.4)
    - `test_sweep_scorer_magnitude_req_18_1` — assert
      `sweep_magnitude == abs(low[k] - swept_level) / atr[k]` for bullish (Req 18.1)
    - `test_sweep_scorer_volume_ratio_req_18_2` — assert
      `volume_ratio == vol[k] / avg_vol_20[k]` (Req 18.2)
    - `test_sweep_scorer_conviction_score_req_18_3` — assert
      `conviction_score == sweep_magnitude * volume_ratio * equal_levels_count` (Req 18.3)
    - `test_sweep_scorer_sorted_descending_req_18_4` — rank 3 signals; descending
      `conviction_score` (Req 18.4)
    - `test_sweep_scorer_zero_avg_vol_req_18_6` — avg_vol=0; `volume_ratio == 0.0` (Req 18.6)
    - `test_sweep_invalid_timeframe_raises_req_19_3` — `ValueError` for `"1w"` (Req 19.3)
    - `test_sweep_multi_tf_confluence_req_19_5` — bullish sweep on `1h` and `1d` with
      swept levels within `0.5 × ATR14`; assert both have `multi_tf_confluence == True`
      (Req 19.5)
    - `test_sweep_api_returns_200_req_20_1` — `GET /v2/scanners/liquidity-sweeps` → 200
      (Req 20.1)
    - `test_sweep_api_limit_req_20_2` — `?limit=2` → 2 signals (Req 20.2)
    - `test_sweep_api_direction_filter_req_20_3` — `?direction=bearish` → all BEARISH
      (Req 20.3)
    - `test_sweep_api_empty_returns_200_req_20_7` — no signals → 200, empty list (Req 20.7)
    - `test_sweep_api_limit_out_of_range_req_20_8` — `?limit=0` → 422 (Req 20.8)
    - Mark all `@pytest.mark.tdd` + `@pytest.mark.unit`; API tests also
      `@pytest.mark.integration`
    - _Requirements: 15.1–15.6, 16.1–16.6, 17.1–17.6, 18.1–18.7, 19.1–19.5, 20.1–20.8, 21.1–21.8_

  - [x] 2.5 Write TDD red-phase tests for Market Structure scanner (Req 22–29)
    - `test_ms_swing_high_hh_req_22_1` — new swing high exceeds prior; assert `HH`
      classification (Req 22.1)
    - `test_ms_swing_high_lh_req_22_2` — new swing high below prior; assert `LH` (Req 22.2)
    - `test_ms_swing_low_ll_req_22_3` — new swing low below prior; assert `LL` (Req 22.3)
    - `test_ms_swing_low_hl_req_22_4` — new swing low above prior; assert `HL` (Req 22.4)
    - `test_ms_swing_trailing_only_req_22_6` — verify `sw_h[k] == high.rolling(lookback, min_periods=1).max().iloc[k]`; no `center=True` (Req 22.6)
    - `test_ms_bos_bullish_req_23_1` — `Trend_State=BULLISH`; `close[k] > most_recent_sh`;
      assert `BOS_BULLISH` signal emitted (Req 23.1)
    - `test_ms_bos_bearish_req_23_2` — `Trend_State=BEARISH`; `close[k] < most_recent_sl`;
      assert `BOS_BEARISH` (Req 23.2)
    - `test_ms_bos_wick_only_no_signal_req_23_3` — wick crosses swing level; close does
      NOT cross; assert no BoS signal emitted (Req 23.3)
    - `test_ms_bos_records_evidence_req_23_4` — assert `close_at_break`, `broken_level`,
      `break_timestamp` all set correctly (Req 23.4)
    - `test_ms_bos_level_consumed_once_req_23_5` — same level cannot trigger BoS twice;
      second crossing → no duplicate signal (Req 23.5)
    - `test_ms_bos_undefined_no_signal_req_23_6` — `Trend_State=UNDEFINED`; assert no
      BoS signals (Req 23.6)
    - `test_ms_choch_bullish_req_24_1` — `Trend_State=BEARISH`; `close[k] > sh`; assert
      `CHOCH_BULLISH`, `is_choch=True` (Req 24.1)
    - `test_ms_choch_bearish_req_24_2` — `Trend_State=BULLISH`; `close[k] < sl`; assert
      `CHOCH_BEARISH` (Req 24.2)
    - `test_ms_choch_transitions_state_bullish_req_24_4` — after `CHOCH_BULLISH`;
      `Trend_State == BULLISH` (Req 24.4)
    - `test_ms_choch_transitions_state_bearish_req_24_5` — after `CHOCH_BEARISH`;
      `Trend_State == BEARISH` (Req 24.5)
    - `test_ms_choch_invalidated_req_24_6` — close below broken_level of `CHOCH_BULLISH`;
      `invalidated == True`, Trend_State unchanged (Req 24.6)
    - `test_ms_state_undefined_initial_req_25_1` — fresh TrendStateTracker;
      `trend_state == UNDEFINED` (Req 25.1)
    - `test_ms_state_undefined_to_bullish_hh_hl_req_25_2` — HH-HL sequence;
      `Trend_State` transitions to `BULLISH` (Req 25.2)
    - `test_ms_state_undefined_to_bearish_ll_lh_req_25_3` — LL-LH sequence;
      `Trend_State` transitions to `BEARISH` (Req 25.3)
    - `test_ms_bos_does_not_change_state_req_25_6` — BoS signal; `Trend_State` unchanged
      (Req 25.6)
    - `test_ms_scorer_break_magnitude_req_26_1` — assert
      `break_magnitude == abs(close_at_break - broken_level) / atr[k]` (Req 26.1)
    - `test_ms_scorer_conviction_choch_req_26_3` — ChoCh signal; assert
      `conviction_score == break_magnitude * swing_count * 1.5` (Req 26.3)
    - `test_ms_scorer_sorted_descending_req_26_4` — rank 3 signals; descending order
      (Req 26.4)
    - `test_ms_scorer_zero_atr_req_26_6` — ATR=0; `break_magnitude == 0.0` (Req 26.6)
    - `test_ms_invalid_timeframe_raises_req_27_3` — `ValueError` for `"1w"` (Req 27.3)
    - `test_ms_separate_trend_states_per_timeframe_req_27_4` — two timeframes maintain
      independent Trend_States (Req 27.4)
    - `test_ms_multi_tf_confluence_req_27_5` — `CHOCH_BULLISH` on `1h` and `1d` within
      3 bars; both `multi_tf_confluence == True` (Req 27.5)
    - `test_ms_api_returns_200_req_28_1` — `GET /v2/scanners/market-structure` → 200
      (Req 28.1)
    - `test_ms_api_limit_req_28_2` — `?limit=3` → 3 signals (Req 28.2)
    - `test_ms_api_signal_type_filter_req_28_3` — `?signal_type=BOS_BULLISH` → all
      `BOS_BULLISH` (Req 28.3)
    - `test_ms_api_empty_returns_200_req_28_7` — no signals → 200, empty list (Req 28.7)
    - `test_ms_api_limit_out_of_range_req_28_9` — `?limit=0` → 422 (Req 28.9)
    - Mark all `@pytest.mark.tdd` + `@pytest.mark.unit`; API tests also
      `@pytest.mark.integration`
    - _Requirements: 22.1–22.7, 23.1–23.6, 24.1–24.7, 25.1–25.7, 26.1–26.7, 27.1–27.5, 28.1–28.9, 29.1–29.9_

  - [x] 2.6 Write property-based tests for correctness Properties 1–20
    - `test_fvg_geometric_invariant_prop_1` — `@given(ohlcv_strategy(min_rows=3))`; for any
      BULLISH zone in `detect_fvg_zones` output: `fvg_low == high[c1]`,
      `fvg_high == low[c3]`, `fvg_high > fvg_low`,
      `midpoint == (fvg_high + fvg_low) / 2`; `@settings(max_examples=200)`;
      annotate `# Feature: smart-money-scanners, Property FVG-1: Bullish Detection Geometric Invariant`
      (Properties 1 / Req 1.1, 1.3, 2.1, 2.2)
    - `test_fvg_no_lookahead_prop_2` — `@given(ohlcv_strategy(), st.integers(1,5))`; append
      N extra rows to series S to get S′; zones from S must equal zones from S′ filtered
      to `candle3_bar_index < len(S)`; `max_examples=200`;
      annotate `Property FVG-2: No-Lookahead Invariant` (Req 7.2, 2.4)
    - `test_fvg_lifecycle_terminal_prop_3` — `@given(fvg_zone_strategy(), ...)`;
      build a FILLED/EXPIRED zone; call `update_fvg_lifecycle` with any candle values;
      assert `lifecycle_state` unchanged; `max_examples=200`;
      annotate `Property FVG-3: Lifecycle Terminal State` (Req 3.4, 3.5)
    - `test_fvg_scoring_monotonicity_prop_4` — `@given(st.lists(fvg_zone_strategy(), min_size=1, max_size=50))`;
      assert `ranked[k].combined_score >= ranked[k+1].combined_score` for all k after
      `rank_fvg_zones`; `max_examples=200`;
      annotate `Property FVG-4: Scoring Monotonicity` (Req 4.3, 4.4, 4.5)
    - `test_fvg_round_trip_prop_5` — `@given(fvg_zone_strategy())`; assert
      `FVGZone(**z.model_dump(mode="json")) == z`; `max_examples=100`;
      annotate `Property FVG-5: Round-Trip Serialization` (Req 7.4)
    - `test_sd_base_departure_invariant_prop_6` — `@given(ohlcv_strategy(min_rows=7))`; for
      each SDZone in `detect_sd_zones` output: every base candle satisfies body < threshold×ATR,
      departure candle satisfies body > threshold×ATR; `max_examples=200`;
      annotate `Property SD-1: Base Candle Body Invariant` (Req 8.1, 8.3, 14.1)
    - `test_sd_zone_type_direction_invariant_prop_7` — `@given(sd_zone_strategy())`; assert
      `direction == DEMAND iff zone_type in (RBR, DBR)`,
      `direction == SUPPLY iff zone_type in (RBD, DBD)`; `max_examples=200`;
      annotate `Property SD-2: Zone Type Direction Invariant` (Req 9.1–9.4)
    - `test_sd_strength_monotonicity_prop_8` — `@given(sd_zone_strategy(), st.lists(...))`; apply
      sequence of `update_sd_lifecycle` calls; assert `zone_strength` never increases across
      calls; `max_examples=200`;
      annotate `Property SD-3: Zone Strength Monotonicity` (Req 10.1–10.4, 14.6)
    - `test_sd_scoring_monotonicity_prop_9` — `@given(st.lists(sd_zone_strategy(), min_size=1, max_size=50))`;
      after `rank_sd_zones`: `zones[k].combined_score >= zones[k+1].combined_score`;
      `max_examples=200`;
      annotate `Property SD-4: Scoring Monotonicity` (Req 11.3–11.5, 14.5)
    - `test_sd_round_trip_prop_10` — `@given(sd_zone_strategy())`; assert
      `SDZone(**z.model_dump(mode="json")) == z`; `max_examples=100`;
      annotate `Property SD-5: Round-Trip Serialization` (Req 14.4)
    - `test_sweep_geometric_invariant_prop_11` — `@given(ohlcv_strategy(min_rows=12))`; for each
      BULLISH SweepSignal: `low[k] < swept_level`, `close[k] > swept_level`,
      `volume[k] >= multiplier * avg_vol[k]`; `max_examples=200`;
      annotate `Property SWEEP-1: Sweep Geometric Invariant` (Req 16.1–16.3)
    - `test_sweep_no_lookahead_prop_12` — `@given(ohlcv_strategy(), st.integers(1,5))`; append N
      rows; set of SweepSignals from original == set from extended filtered by
      `sweep_bar_index < len(original)`; `max_examples=200`;
      annotate `Property SWEEP-2: No-Lookahead Invariant` (Req 21.2, 21.3)
    - `test_sweep_lifecycle_terminal_prop_13` — `@given(sweep_signal_strategy(), st.floats())`; build
      INVALIDATED signal; call `update_sweep_lifecycle`; assert still `INVALIDATED`;
      `max_examples=200`;
      annotate `Property SWEEP-3: Lifecycle Terminal State` (Req 17.4)
    - `test_sweep_scoring_monotonicity_prop_14` — `@given(st.lists(sweep_signal_strategy(), min_size=1, max_size=50))`;
      after `rank_sweeps`: `signals[k].conviction_score >= signals[k+1].conviction_score`;
      `max_examples=200`;
      annotate `Property SWEEP-4: Scoring Monotonicity` (Req 18.3–18.5)
    - `test_sweep_round_trip_prop_15` — `@given(sweep_signal_strategy())`; assert
      `SweepSignal(**s.model_dump(mode="json")) == s`; `max_examples=100`;
      annotate `Property SWEEP-5: Round-Trip Serialization` (Req 21.5)
    - `test_ms_bos_close_confirm_prop_16` — `@given(ohlcv_strategy(min_rows=15))`; for every
      BoS signal (`is_choch=False`): `BOS_BULLISH → close_at_break > broken_level`,
      `BOS_BEARISH → close_at_break < broken_level`; `max_examples=200`;
      annotate `Property MS-1: BoS Close-Confirmation Invariant` (Req 23.1–23.3)
    - `test_ms_state_only_changes_on_choch_prop_17` — `@given(ohlcv_strategy(min_rows=15))`; in
      the emitted MSSignal list, `trend_state` must change value only at signals where
      `is_choch=True`; `max_examples=200`;
      annotate `Property MS-2: Trend State Only Changes on ChoCh` (Req 24.4, 24.5, 25.4–25.6, 29.7)
    - `test_ms_scoring_monotonicity_prop_18` — `@given(st.lists(ms_signal_strategy(), min_size=1, max_size=50))`;
      after `rank_ms_signals`: descending `conviction_score`; `max_examples=200`;
      annotate `Property MS-3: Scoring Monotonicity` (Req 26.3–26.5)
    - `test_ms_no_lookahead_prop_19` — `@given(ohlcv_strategy(), st.integers(1,5))`; append N rows;
      MSSignal set from original == set from extended filtered by
      `break_bar_index < len(original)`; `max_examples=200`;
      annotate `Property MS-4: No-Lookahead Invariant` (Req 29.1–29.3)
    - `test_ms_round_trip_prop_20` — `@given(ms_signal_strategy())`; assert
      `MSSignal(**m.model_dump(mode="json")) == m`; `max_examples=100`;
      annotate `Property MS-5: Round-Trip Serialization` (Req 29.5)
    - Decorate all with `@pytest.mark.unit`, `@pytest.mark.slow`,
      `@settings(max_examples=N, derandomize=True)` as specified per property
    - _Requirements: 7.3–7.6, 14.3–14.6, 21.4–21.6, 29.4–29.7_

  - [x] 2.7 Write PIT (Point-In-Time) tests for all four scanners
    - `test_fvg_pit_timestamp` — for all `FVGZone` objects produced by `detect_fvg_zones`
      on a known series: assert `z.fvg_timestamp == df.index[z.candle3_bar_index]`; no
      zone carries a timestamp later than its candle3 bar; mark `@pytest.mark.pit`
      (Req 7.8)
    - `test_sd_pit_departure_timestamp` — for all `SDZone` objects: assert
      `z.departure_timestamp == df.index[z.departure_bar_index]`; mark
      `@pytest.mark.pit` (Req 14.8)
    - `test_sweep_pit_timestamp` — for all `SweepSignal` objects: assert
      `s.sweep_timestamp == df.index[s.sweep_bar_index]`; mark `@pytest.mark.pit`
      (Req 21.8)
    - `test_ms_pit_break_timestamp` — for all `MSSignal` objects: assert
      `m.break_timestamp == df.index[m.break_bar_index]`; mark `@pytest.mark.pit`
      (Req 29.9)
    - `test_fvg_no_future_data_in_zone` — modify a candle at `k > candle3_bar_index` in
      the series; re-run detector; verify zone from bars `[i, i+1, i+2]` is unchanged;
      mark `@pytest.mark.pit` (Req 7.2)
    - `test_sd_no_future_data_in_zone` — modify candle at `k > departure_bar_index`;
      verify SDZone unchanged; mark `@pytest.mark.pit` (Req 14.2)
    - `test_sweep_no_future_data_in_signal` — modify candle at `k > sweep_bar_index`;
      verify SweepSignal unchanged; mark `@pytest.mark.pit` (Req 21.3)
    - `test_ms_no_future_data_in_signal` — modify candle at `k > break_bar_index`;
      verify MSSignal unchanged; mark `@pytest.mark.pit` (Req 29.3)
    - _Requirements: 7.8, 14.8, 21.8, 29.9_

- [x] 3. Implement Fair Value Gap scanner
  - [x] 3.1 Implement `detect_fvg_zones()` in `src/scanners/fair_value_gap/detector.py`
    - Guard: `if len(df) < 3: return []`; return `[]` if all OHLCV columns are NaN
    - Compute true range: `prev_close = df["close"].shift(1)`;
      `tr = pd.concat([df["high"]-df["low"], (df["high"]-prev_close).abs(), (df["low"]-prev_close).abs()], axis=1).max(axis=1)`
    - Compute ATR14: `atr = tr.rolling(window=config.atr_window, min_periods=1).mean()`
      — **never** `center=True`
    - Compute `avg_vol = df["volume"].rolling(window=20, min_periods=1).mean()`
    - Iterate windows `for i in range(len(df) - 2)`: extract rows `c1=i, c2=i+1, c3=i+2`
    - Per-window NaN guard: skip if any of `open/high/low/close/volume` is NaN at c1, c2, c3
      (use `pd.notna(df.iloc[j][["open","high","low","close","volume"]]).all()`)
    - Impulse filter: `body2 = abs(df["close"].iloc[c2] - df["open"].iloc[c2])`; skip if
      `body2 < config.impulse_atr_multiplier * atr.iloc[c2]`; log skipped at DEBUG with
      `structlog.get_logger(__name__).debug("fvg_impulse_rejected", bar_index=c2, body2=body2, atr=atr.iloc[c2])`
    - Bullish FVG: `if low[c3] > high[c1] and close[c3] > high[c1]` →
      `fvg_high=low[c3]`, `fvg_low=high[c1]`, `direction=BULLISH`
    - Bearish FVG: `elif high[c3] < low[c1] and close[c3] < low[c1]` →
      `fvg_high=low[c1]`, `fvg_low=high[c3]`, `direction=BEARISH`
    - Gap-size filter: skip if `(fvg_high - fvg_low) < config.min_gap_atr_multiplier * atr.iloc[c3]`
    - Emit `FVGZone(symbol=symbol, direction=direction, timeframe=timeframe,
      fvg_high=fvg_high, fvg_low=fvg_low, fvg_midpoint=(fvg_high+fvg_low)/2,
      fvg_timestamp=df.index[c3], candle1_bar_index=i, candle3_bar_index=i+2,
      impulse_atr=atr.iloc[c2], lifecycle_state=FVGLifecycle.UNFILLED)`
    - _Requirements: 1.1–1.7, 2.1–2.6, 7.1, 7.2_

  - [x] 3.2 Implement `update_fvg_lifecycle()` in `src/scanners/fair_value_gap/zone_manager.py`
    - Signature: `update_fvg_lifecycle(zone: FVGZone, candle_low: float, candle_high: float, candle_close: float, bar_idx: int, config: FVGConfig) -> FVGZone`
    - **Step 1 — terminal guard**: if `zone.lifecycle_state in (FILLED, EXPIRED)` → return
      zone unchanged immediately
    - **Step 2 — fill check**: if candle enters zone (`candle_low <= zone.fvg_high AND
      candle_high >= zone.fvg_low`) AND midpoint is crossed:
      - Bullish: `candle_close < zone.fvg_midpoint` → FILLED
      - Bearish: `candle_close > zone.fvg_midpoint` → FILLED
      - Return `.model_copy(update={"lifecycle_state": FILLED, "fill_timestamp": utc_now})`
    - **Step 3 — partial fill**: candle enters zone but does NOT cross midpoint → PARTIALLY_FILLED;
      `.model_copy(update={"lifecycle_state": PARTIALLY_FILLED})`
    - **Step 4 — expiry**: if `zone.lifecycle_state == UNFILLED` AND
      `bar_idx - zone.candle3_bar_index > config.fvg_expiry_bars` → EXPIRED;
      `.model_copy(update={"lifecycle_state": EXPIRED})`
    - All mutations via `.model_copy(update={...})` — never mutate frozen `FVGZone` in place
    - _Requirements: 3.1–3.7_

  - [x] 3.3 Implement `rank_fvg_zones()` in `src/scanners/fair_value_gap/scorer.py`
    - Signature: `rank_fvg_zones(zones: list[FVGZone], df_by_symbol: dict[str, pd.DataFrame], config: FVGConfig) -> list[FVGZone]`
    - For each zone, look up `df = df_by_symbol.get(zone.symbol)`; if not found or
      `zone.candle3_bar_index` out of range, set `gap_size_score=0.0`, `volume_score=0.0`
    - Compute `atr_c3` at `candle3_bar_index` (same ATR formula as detector);
      `gap_size_score = (zone.fvg_high - zone.fvg_low) / atr_c3` (0.0 if ATR is 0 or NaN)
    - Compute `vol_c2 = df["volume"].iloc[c2]`; `avg_vol_c2 = df["volume"].rolling(20, min_periods=1).mean().iloc[c2]`
    - `volume_score = vol_c2 / avg_vol_c2` if `avg_vol_c2 > 0` else `0.0`
    - `combined_score = gap_size_score * volume_score`
    - Update each zone via `.model_copy(update={"gap_size_score": gs, "volume_score": vs, "combined_score": cs})`
    - Sort: descending `combined_score`; tie-break descending `fvg_timestamp`
    - Return updated sorted list; all non-score fields preserved byte-for-byte
    - _Requirements: 4.1–4.7_

  - [x] 3.4 Implement `FVGScanner.scan()` in `src/scanners/fair_value_gap/scanner.py`
    - Implement `FVGScanner` class: `__init__(self, config: FVGConfig | None = None)`;
      store config, fall back to `FVGConfig()`
    - `scan(ohlcv_by_symbol, timeframe, existing_zones=None) -> FVGScanResult`
    - **Validate timeframe**: raise `ValueError(f"Unsupported timeframe '{tf}'. Valid values: 5m, 15m, 1h, 1d")` for unsupported values
    - **Per-timeframe expiry**: read `config.tf_5m_expiry_bars`, `tf_15m_expiry_bars`,
      `tf_1h_expiry_bars`, `tf_1d_expiry_bars` from config; create effective config copy
      via `config.model_copy(update={"fvg_expiry_bars": tf_expiry})`
    - **Per-symbol pipeline** (wrapped in `try/except Exception` — one bad symbol never aborts):
      - `new_zones = detect_fvg_zones(df, symbol, timeframe, eff_config)`
      - If `existing_zones[symbol]` provided: for each existing zone, call
        `update_fvg_lifecycle(zone, ...)` using the latest candle
      - Merge: active existing zones + new zones
    - **Cross-symbol confluence**: for same symbol, compare zones across timeframes (if multi-TF data
      provided); set `multi_tf_confluence=True` if `|z1.fvg_midpoint - z2.fvg_midpoint| / z1.fvg_midpoint < config.confluence_price_pct`
    - **Rank**: call `rank_fvg_zones(all_zones, ohlcv_by_symbol, config)` to sort globally
    - **Log**: `structlog.get_logger(__name__).info("fvg_scan_complete", n_symbols=..., n_zones=..., timeframe=...)`
    - Return `FVGScanResult(zones=ranked, scanned_at=datetime.now(tz=timezone.utc), n_symbols_scanned=len(ohlcv_by_symbol))`
    - _Requirements: 5.1–5.6, 7.1–7.2_

  - [x]* 3.5 Write additional edge-case unit tests for FVG scanner
    - `test_fvg_detect_exact_boundary_gap_size` — `gap_size == min_gap_atr_multiplier * atr` exactly
      (boundary inclusive); assert zone IS detected
    - `test_fvg_detect_both_directions_same_series` — series with one bullish and one bearish
      FVG; assert both are returned
    - `test_fvg_zone_manager_no_mutation` — verify `update_fvg_lifecycle` returns NEW object; input
      zone `id` != output zone `id`
    - `test_fvg_scanner_empty_dict` — `scan({}, "1h")` → `FVGScanResult` with `zones=[]`,
      no exception
    - `test_fvg_scanner_per_symbol_error_isolated` — one symbol raises in detect; other symbol
      still returns zones (Req 5.1)
    - Add to `tests/test_smart_money_scanners.py`; mark `@pytest.mark.unit`
    - _Requirements: 2.2, 3.4, 5.1_

- [x] 4. Implement Supply & Demand Zone scanner
  - [x] 4.1 Implement `detect_sd_zones()` in `src/scanners/supply_demand/detector.py`
    - Guard: `if len(df) < 5: return []`
    - Compute ATR14 and `body = abs(df["close"] - df["open"])` vectorially
    - Outer loop: `b = 1` (need entry_leg at `b-1`); skip if NaN in OHLCV at `b-1`
    - Inner scan: advance `base_end` while `body[base_end] < config.base_body_threshold * atr[base_end]`
      AND `base_end < b + config.max_base_candles`; accumulate base candles
    - Require `base_len = base_end - b >= config.min_base_candles`; else `b += 1; continue`
    - `dep = base_end`; guard: `dep >= len(df)` → break; skip if NaN in any base or dep candle
    - Departure filter: `body[dep] < config.departure_body_threshold * atr[dep]` → `b += 1; continue`
    - Zone width filter: `zone_high - zone_low < config.min_zone_width_multiplier * atr[dep]` →
      `b += 1; continue`
    - Classification: `entry_bullish = close[b-1] > open[b-1]`; `dep_bullish = close[dep] > open[dep]`;
      `zone_type = {(T,T): RBR, (F,F): DBD, (T,F): RBD, (F,T): DBR}[(entry_bullish, dep_bullish)]`
    - Emit `SDZone(symbol, direction=DEMAND if zone_type in (RBR,DBR) else SUPPLY, zone_type,
      timeframe, zone_high=max(high[b:base_end]), zone_low=min(low[b:base_end]),
      entry_leg_timestamp=df.index[b-1], departure_timestamp=df.index[dep],
      base_start_bar_index=b, base_end_bar_index=base_end-1,
      departure_bar_index=dep, lifecycle_state=FRESH, zone_strength=1.0)`
    - Advance: `b = dep + 1` (allow overlapping patterns)
    - _Requirements: 8.1–8.7, 9.1–9.6, 14.1, 14.2_

  - [x] 4.2 Implement `update_sd_lifecycle()` in `src/scanners/supply_demand/zone_manager.py`
    - Signature: `update_sd_lifecycle(zone: SDZone, candle_low: float, candle_high: float, candle_close: float, bar_idx: int, config: SDConfig) -> SDZone`
    - **Step 1 — terminal guard**: `BROKEN` → return unchanged
    - **Step 2 — broken check** (any state): 
      - DEMAND: `candle_close < zone.zone_low` → BROKEN, `zone_strength=0.0`,
        `broken_timestamp=utc_now`
      - SUPPLY: `candle_close > zone.zone_high` → BROKEN
      - Return `.model_copy(update={...})`
    - **Entered zone** predicate: `candle_low <= zone.zone_high AND candle_high >= zone.zone_low`
    - **Step 3 — FRESH → TESTED**: entered zone AND close does NOT cross far side → TESTED,
      `zone_strength=0.7`; `.model_copy(update={...})`
    - **Step 4 — TESTED → USED**: entered zone again AND close does NOT cross far side → USED,
      `zone_strength=0.4`; `.model_copy(update={...})`
    - Record `broken_timestamp` on all BROKEN transitions
    - _Requirements: 10.1–10.7_

  - [x] 4.3 Implement `rank_sd_zones()` in `src/scanners/supply_demand/scorer.py`
    - For each zone: look up `df = df_by_symbol[zone.symbol]`; extract departure candle at
      `zone.departure_bar_index`
    - `atr_dep = atr_series.iloc[dep]`; if `atr_dep == 0 or NaN`: `departure_strength = 0.0`
    - Else `departure_strength = abs(close[dep] - open[dep]) / atr_dep`
    - `zone_freshness = zone.zone_strength` (1.0 / 0.7 / 0.4 / 0.0 per lifecycle state)
    - `combined_score = departure_strength * zone_freshness`
    - Update via `.model_copy(update={"departure_strength": ds, "combined_score": cs})`
    - Sort: descending `combined_score`; tie-break descending `departure_timestamp`
    - All other fields preserved unchanged
    - _Requirements: 11.1–11.7_

  - [x] 4.4 Implement `SDScanner.scan()` in `src/scanners/supply_demand/scanner.py`
    - Same orchestration pattern as `FVGScanner.scan()`:
    - Validate timeframe; raise `ValueError` for unsupported values
    - Per-symbol: `detect_sd_zones` → lifecycle-update existing → merge → score
    - Multi-TF confluence: for same symbol, compare DEMAND zones across `1h` and `1d`;
      if zone midpoints within `config.confluence_price_pct` → set `multi_tf_confluence=True`
      on both via `.model_copy`
    - `rank_sd_zones(all_zones, ohlcv_by_symbol, config)` for global cross-symbol sort
    - Log completion via structlog; per-symbol errors isolated via `try/except`
    - Return `SDScanResult(zones=ranked, scanned_at=..., n_symbols_scanned=...)`
    - _Requirements: 12.1–12.5_

  - [x]* 4.5 Write additional edge-case unit tests for S&D scanner
    - `test_sd_base_exactly_2_candles` — minimum 2-candle base with valid departure;
      assert zone detected
    - `test_sd_base_exactly_5_candles` — maximum 5-candle base; assert detected
    - `test_sd_base_6_candles_not_detected` — 6 small-body candles; only first 5 form base;
      verify `base_end_bar_index - base_start_bar_index <= 4`
    - `test_sd_lifecycle_broken_from_all_states` — transition to BROKEN from FRESH, TESTED,
      and USED; assert `zone_strength == 0.0` in each case
    - `test_sd_scanner_empty_dict` — `scan({}, "1h")` → no exception, empty zones
    - Add to `tests/test_smart_money_scanners.py`; mark `@pytest.mark.unit`
    - _Requirements: 8.1, 10.4, 12.1_

- [x] 5. Implement Liquidity Sweep scanner
  - [x] 5.1 Implement `detect_sweeps()` in `src/scanners/liquidity_sweep/detector.py`
    - Guard: `if len(df) < config.swing_lookback + 1: return []`
    - Compute ATR14, `avg_vol = df["volume"].rolling(20, min_periods=1).mean()`;
      `sw_high = df["high"].rolling(config.swing_lookback, min_periods=1).max()`
      `sw_low = df["low"].rolling(config.swing_lookback, min_periods=1).min()`
      — all trailing, no `center=True`
    - If `config.require_reversal_confirmation`: `upper_k = len(df) - 2` else `upper_k = len(df) - 1`
    - Iterate `for k in range(config.swing_lookback, upper_k + 1)`:
      - Skip if NaN in `high/low/close/volume` at `k`
      - Volume check: `volume[k] >= config.volume_confirmation_multiplier * avg_vol[k]`; skip if not
      - Bullish sweep: `low[k] < sw_low[k-1] AND close[k] > sw_low[k-1]`
      - Bearish sweep: `high[k] > sw_high[k-1] AND close[k] < sw_high[k-1]`
      - If neither pattern matched, continue
      - Reversal confirmation (if enabled): skip if NaN at `k+1`; bullish: `close[k+1] <= close[k]` → skip;
        bearish: `close[k+1] >= close[k]` → skip
      - `direction = BULLISH if bullish_sweep else BEARISH`
      - `swept = sw_low[k-1] if bullish_sweep else sw_high[k-1]`
      - Count equal consecutive swing levels: `tolerance = max(0.001 * close[k], 0.1 * atr[k])`;
        count backward how many consecutive sw_low (or sw_high) values are within tolerance
      - Emit `SweepSignal(symbol, direction, timeframe, swept_level=swept,
        sweep_zone_high=swept + 0.5*atr[k], sweep_zone_low=swept - 0.5*atr[k],
        sweep_timestamp=df.index[k], sweep_bar_index=k,
        equal_levels_count=eq_count, lifecycle_state=ACTIVE)`
    - _Requirements: 15.1–15.6, 16.1–16.6, 21.1, 21.2_

  - [x] 5.2 Implement `update_sweep_lifecycle()` in `src/scanners/liquidity_sweep/signal_manager.py`
    - Signature: `update_sweep_lifecycle(signal: SweepSignal, next_close: float, bar_idx: int, config: SweepConfig) -> SweepSignal`
    - **Step 1 — terminal guard**: `INVALIDATED` → return unchanged
    - **Step 2 — invalidation** (any state): 
      - Bullish: `next_close < signal.swept_level` → INVALIDATED
      - Bearish: `next_close > signal.swept_level` → INVALIDATED
      - `.model_copy(update={"lifecycle_state": INVALIDATED})`
    - **Step 3 — confirmation** (ACTIVE only):
      - Bullish: `next_close > signal.sweep_zone_low` (close continues upward) → CONFIRMED
      - Bearish: `next_close < signal.sweep_zone_high` (close continues downward) → CONFIRMED
      - `.model_copy(update={"lifecycle_state": CONFIRMED, "confirmed_timestamp": utc_now})`
    - _Requirements: 17.1–17.6_

  - [x] 5.3 Implement `rank_sweeps()` in `src/scanners/liquidity_sweep/scorer.py`
    - For each signal: look up `df = df_by_symbol[signal.symbol]` and `k = signal.sweep_bar_index`
    - Compute ATR at k (or reuse from df); if ATR=0 or NaN: `sweep_magnitude = 0.0`
    - Bullish: `wick_extreme = df["low"].iloc[k]`; bearish: `wick_extreme = df["high"].iloc[k]`
    - `sweep_magnitude = abs(wick_extreme - signal.swept_level) / atr[k]`
    - `avg_vol = df["volume"].rolling(20, min_periods=1).mean().iloc[k]`
    - `volume_ratio = volume[k] / avg_vol` if `avg_vol > 0` else `0.0`
    - `conviction_score = sweep_magnitude * volume_ratio * signal.equal_levels_count`
    - Update via `.model_copy(update={"sweep_magnitude": sm, "volume_ratio": vr, "conviction_score": cs})`
    - Sort: descending `conviction_score`; tie-break descending `sweep_timestamp`
    - _Requirements: 18.1–18.7_

  - [x] 5.4 Implement `SweepScanner.scan()` in `src/scanners/liquidity_sweep/scanner.py`
    - Same orchestration pattern: validate timeframe; per-symbol error isolation
    - `detect_sweeps(df, symbol, timeframe, config)` → lifecycle-update existing → merge
    - Multi-TF confluence: for same symbol, compare swept levels across `1h` and `1d`;
      if `abs(s1.swept_level - s2.swept_level) <= config.confluence_atr_multiplier * atr` → both
      get `multi_tf_confluence=True`
    - `rank_sweeps(all_signals, ohlcv_by_symbol, config)` for global sort
    - Return `SweepScanResult(signals=ranked, scanned_at=..., n_symbols_scanned=...)`
    - _Requirements: 19.1–19.5_

  - [x]* 5.5 Write additional edge-case unit tests for Liquidity Sweep scanner
    - `test_sweep_equal_levels_count_3` — 3 consecutive swing highs within tolerance;
      assert `equal_levels_count == 3`
    - `test_sweep_wick_no_close_reversal_not_detected` — wick below sw_low, close also
      below sw_low (no reversal); assert no BULLISH sweep
    - `test_sweep_confirmation_disabled` — `require_reversal_confirmation=False`; sweep
      emitted without requiring `k+1` candle
    - `test_sweep_scanner_empty_dict` — `scan({}, "1h")` → empty signals, no exception
    - Add to `tests/test_smart_money_scanners.py`; mark `@pytest.mark.unit`
    - _Requirements: 15.3, 16.1, 16.5, 19.1_

- [x] 6. Implement Market Structure scanner
  - [x] 6.1 Implement `detect_market_structure()` in `src/scanners/market_structure/detector.py`
    - Guard: `if len(df) < config.swing_lookback + 1: return []`
    - Compute ATR14; `sw_h = df["high"].rolling(config.swing_lookback, min_periods=1).max()`;
      `sw_l = df["low"].rolling(config.swing_lookback, min_periods=1).min()` — trailing only
    - Initialise: `trend_state=UNDEFINED`, `swing_highs=[]`, `swing_lows=[]`,
      `used_levels=set()`, `swing_count=0`, `signals=[]`
    - Main loop `for k in range(config.swing_lookback, len(df))`:
      - Skip if NaN in `high/low/close` at k
      - Detect swing high: `is_swing_high = (df["high"].iloc[k] == sw_h.iloc[k])`
      - Detect swing low: `is_swing_low = (df["low"].iloc[k] == sw_l.iloc[k])`
      - Classify and record: if `is_swing_high` and prior swing high exists: HH if `high[k] > prior_sh`, else LH; append `(k, high[k])` to `swing_highs`
      - Classify and record: if `is_swing_low` and prior swing low exists: LL if `low[k] < prior_sl`, else HL; append `(k, low[k])` to `swing_lows`
      - If `trend_state == UNDEFINED`: infer initial trend from accumulated swing_highs/swing_lows
        (HH+HL sequence → BULLISH; LL+LH sequence → BEARISH); continue without emitting
      - If `trend_state == BULLISH`:
        - BoS: `close[k] > last_sh_price` AND `(last_sh_price, BOS_BULLISH) not in used_levels` →
          emit `MSSignal(signal_type=BOS_BULLISH, broken_level=last_sh_price, close_at_break=close[k],
          break_timestamp=df.index[k], break_bar_index=k, trend_state=BULLISH, is_choch=False,
          swing_count=swing_count)`; add to `used_levels`; `swing_count += 1`
        - ChoCh: `close[k] < last_sl_price` AND `(last_sl_price, CHOCH_BEARISH) not in used_levels` →
          emit `MSSignal(signal_type=CHOCH_BEARISH, ..., is_choch=True, trend_state=BEARISH)`;
          `trend_state = BEARISH`; `swing_count = 0`
      - If `trend_state == BEARISH`: symmetric — BOS_BEARISH on `close < last_sl`, CHOCH_BULLISH
        on `close > last_sh`
    - Return `signals`
    - _Requirements: 22.1–22.7, 23.1–23.6, 24.1–24.7, 29.1–29.3_

  - [x] 6.2 Implement `TrendStateTracker` in `src/scanners/market_structure/state_manager.py`
    - Class `TrendStateTracker`:
      - `__init__`: `self.trend_state = TrendState.UNDEFINED`, `self.swing_highs: list[tuple[int, float]] = []`,
        `self.swing_lows: list[tuple[int, float]] = []`, `self.swing_count: int = 0`
      - `on_choch_bullish(self)`: `self.trend_state = TrendState.BULLISH; self.swing_count = 0`
      - `on_choch_bearish(self)`: `self.trend_state = TrendState.BEARISH; self.swing_count = 0`
      - `on_bos(self)`: `self.swing_count += 1` — BoS does NOT change `trend_state` (Req 25.6)
      - `add_swing_high(self, bar_idx, price)`: append to `swing_highs`; update `trend_state` if
        UNDEFINED and enough data exists
      - `add_swing_low(self, bar_idx, price)`: symmetric
      - `reset(self)`: restore initial state for testing
    - Detector uses `TrendStateTracker` internally; state is reconstructable from OHLCV alone (Req 25.7)
    - _Requirements: 25.1–25.7_

  - [x] 6.3 Implement `rank_ms_signals()` in `src/scanners/market_structure/scorer.py`
    - For each signal: look up `k = signal.break_bar_index`; compute ATR at k
    - If ATR=0 or NaN: `break_magnitude = 0.0`
    - Else `break_magnitude = abs(signal.close_at_break - signal.broken_level) / atr[k]`
    - `choch_multiplier = 1.5 if signal.is_choch else 1.0`
    - `conviction_score = break_magnitude * signal.swing_count * choch_multiplier`
    - Update via `.model_copy(update={"break_magnitude": bm, "conviction_score": cs})`
    - Sort: descending `conviction_score`; tie-break descending `break_timestamp`
    - _Requirements: 26.1–26.7_

  - [x] 6.4 Implement `MSScanner.scan()` in `src/scanners/market_structure/scanner.py`
    - Same orchestration pattern: validate timeframe; per-symbol `TrendStateTracker` instances;
      per-symbol error isolation
    - Apply per-timeframe `swing_lookback` override from config
      (`tf_5m=5`, `tf_15m=7`, `tf_1h=10`, `tf_1d=5`)
    - `detect_market_structure(df, symbol, timeframe, eff_config)` per symbol
    - Multi-TF confluence: for same symbol, match `CHOCH_BULLISH` signals across `1h` and `1d`
      within `config.confluence_bars_window` bars of each other; set `multi_tf_confluence=True`
      on both via `.model_copy`
    - `rank_ms_signals(all_signals, ohlcv_by_symbol, config)` for global sort
    - Return `MSScanResult(signals=ranked, scanned_at=..., n_symbols_scanned=...)`
    - _Requirements: 27.1–27.5_

  - [x]* 6.5 Write additional edge-case unit tests for Market Structure scanner
    - `test_ms_detect_bos_after_choch_direction_change` — `CHOCH_BEARISH` → `Trend_State=BEARISH`;
      subsequent `BOS_BEARISH` emitted, `Trend_State` unchanged after BoS
    - `test_ms_same_level_not_reused` — craft series where two candles would both trigger
      BoS on same swing level; assert only one `BOS_BULLISH` signal
    - `test_ms_tracker_choch_resets_swing_count` — `swing_count > 0`; emit ChoCh; assert
      `swing_count == 0`
    - `test_ms_scanner_empty_dict` — `scan({}, "1h")` → empty signals, no exception
    - `test_ms_separate_timeframe_states` — same symbol scanned with `1h` and `1d`; each
      returns independent Trend_State (Req 27.4)
    - Add to `tests/test_smart_money_scanners.py`; mark `@pytest.mark.unit`
    - _Requirements: 24.4, 23.5, 25.6, 27.4_

- [x] 7. Checkpoint — run unit tests for all four scanners
  - Run `pytest tests/test_smart_money_scanners.py -m "unit and not slow"` and ensure all TDD
    red-phase tests for the four scanner components (Tasks 3–6) are now green; ask the user
    if questions arise.

- [x] 8. Extend `src/api/scanners.py` with four new scanner routes
  - [x] 8.1 Add `GET /v2/scanners/fair-value-gaps` to the existing router
    - Add module-level instances (after existing `_scanner`):
      `_fvg_scanner = FVGScanner()` and import `FVGScanner`, `FVGConfig`, `FVGZone`,
      `FVGScanResponse`, `FVGLifecycle` from their respective modules
    - Define `VALID_FVG_DIRECTIONS = frozenset({"bullish", "bearish"})`
    - Implement `async def get_fair_value_gaps(limit: int = Query(default=50, ge=1, le=200), direction: str | None = Query(default=None), timeframe: str | None = Query(default=None), include_filled: bool = Query(default=False)) -> JSONResponse`
    - Validate `direction` ∈ `VALID_FVG_DIRECTIONS | {None}` → HTTP 422 for invalid values
    - Validate `timeframe` ∈ `VALID_TIMEFRAMES | {None}` → HTTP 422 for invalid values
    - Call `_fetch_ohlcv(timeframe or "1h")`; call `_fvg_scanner.scan(ohlcv, timeframe or "1h")`
    - Filter: if not `include_filled`, exclude `FILLED` and `EXPIRED` zones; apply direction and timeframe filters; apply `[:limit]` slice
    - Build `FVGScanResponse(scanned_at=..., n_symbols_scanned=..., n_zones=len(zones), zones=zones)`;
      return `JSONResponse(content=response.model_dump(mode="json"))`
    - Empty zones → HTTP 200 (never 404)
    - _Requirements: 6.1–6.9_

  - [x] 8.2 Add `GET /v2/scanners/supply-demand` to the existing router
    - Add module-level: `_sd_scanner = SDScanner()` and imports from `supply_demand`
    - Define `VALID_SD_DIRECTIONS = frozenset({"supply", "demand"})`;
      `VALID_ZONE_TYPES = frozenset({"RBR", "DBD", "RBD", "DBR"})`
    - Implement `async def get_supply_demand(limit, direction, timeframe, include_broken, zone_type) -> JSONResponse`
    - Validate `direction` ∈ `VALID_SD_DIRECTIONS | {None}` → 422; `zone_type` ∈ `VALID_ZONE_TYPES | {None}` → 422
    - Call `_sd_scanner.scan(ohlcv, timeframe or "1h")`
    - Filter: if not `include_broken`, exclude `BROKEN` zones; apply `direction`, `zone_type`,
      `timeframe` filters; apply `[:limit]`
    - Build `SDScanResponse`; return as JSONResponse; empty → 200
    - _Requirements: 13.1–13.9_

  - [x] 8.3 Add `GET /v2/scanners/liquidity-sweeps` to the existing router
    - Add module-level: `_sweep_scanner = SweepScanner()` and imports from `liquidity_sweep`
    - Implement `async def get_liquidity_sweeps(limit, direction, timeframe, min_conviction) -> JSONResponse`
    - Validate `direction` ∈ `{"bullish", "bearish"} | {None}` → 422
    - Validate `timeframe` ∈ `VALID_TIMEFRAMES | {None}` → 422
    - Call `_sweep_scanner.scan(ohlcv, timeframe or "1h")`
    - Filter: exclude `INVALIDATED` by default; apply direction, timeframe filters;
      if `min_conviction` provided: `[s for s in signals if s.conviction_score >= min_conviction]`;
      apply `[:limit]`
    - Build `SweepScanResponse`; return as JSONResponse; empty → 200
    - _Requirements: 20.1–20.8_

  - [x] 8.4 Add `GET /v2/scanners/market-structure` to the existing router
    - Add module-level: `_ms_scanner = MSScanner()` and imports from `market_structure`
    - Define `VALID_SIGNAL_TYPES = frozenset({"BOS_BULLISH", "BOS_BEARISH", "CHOCH_BULLISH", "CHOCH_BEARISH"})`
    - Implement `async def get_market_structure(limit, timeframe, signal_type) -> JSONResponse`
    - Validate `timeframe` ∈ `VALID_TIMEFRAMES | {None}` → 422
    - Validate `signal_type` ∈ `VALID_SIGNAL_TYPES | {None}` → 422
    - Call `_ms_scanner.scan(ohlcv, timeframe or "1h")`
    - Filter: apply `signal_type` and `timeframe` filters; apply `[:limit]`
    - Build `MSScanResponse`; return as JSONResponse; empty → 200
    - _Requirements: 28.1–28.9_

- [x] 9. Final checkpoint — full test suite green
  - Run `pytest tests/test_smart_money_scanners.py --hypothesis-seed=0` and ensure all TDD
    red-phase tests (now green), PBT property tests, PIT tests, and API integration tests
    pass; ask the user if questions arise.

---

## Notes

- Tasks marked `*` are optional and can be skipped for a faster MVP
- All TDD tests in Task 2 are mandatory — they form the red-phase specification; they must all
  FAIL before Tasks 3–6 are implemented
- All zone/signal models are `frozen=True` Pydantic schemas; **never** mutate in place —
  always use `.model_copy(update={...})`
- `src/api/scanners.py` is **extended** — the file already exports `router = APIRouter(...)`;
  Task 8 adds four new route functions to the same router instance without replacing anything
- ATR14 is computed identically across all four detectors: trailing `.rolling(window=14, min_periods=1).mean()` on true range — **never** `center=True`
- Run property tests: `pytest -m "slow" --hypothesis-seed=0` (reproducible in CI)
- Run fast unit tests: `pytest tests/test_smart_money_scanners.py -m "unit and not slow"`
- `X-API-KEY` auth is global via `api_key_middleware` in `main.py` — scanner routes add zero auth code
- All four scanner instances are created at module level in `src/api/scanners.py` alongside the existing `_scanner` instance
- `_fetch_ohlcv` stub in the API remains the same function across all four routes

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0,  "tasks": ["1.1", "1.2", "1.3", "1.4"] },
    { "id": 1,  "tasks": ["2.1"] },
    { "id": 2,  "tasks": ["2.2"] },
    { "id": 3,  "tasks": ["2.3"] },
    { "id": 4,  "tasks": ["2.4"] },
    { "id": 5,  "tasks": ["2.5"] },
    { "id": 6,  "tasks": ["2.6"] },
    { "id": 7,  "tasks": ["2.7"] },
    { "id": 8,  "tasks": ["3.1", "4.1", "5.1", "6.1"] },
    { "id": 9,  "tasks": ["3.2", "4.2", "5.2", "6.2"] },
    { "id": 10, "tasks": ["3.3", "4.3", "5.3", "6.3"] },
    { "id": 11, "tasks": ["3.4", "4.4", "5.4", "6.4"] },
    { "id": 12, "tasks": ["3.5", "4.5", "5.5", "6.5", "7"] },
    { "id": 13, "tasks": ["8.1", "8.2", "8.3", "8.4"] },
    { "id": 14, "tasks": ["9"] }
  ]
}
```
