# Implementation Plan: Order Block Scanner

## Overview

TDD-first Python implementation of the Order Block Scanner — a composable pipeline that
identifies institutional order block zones across the F&O universe (~200+ NSE stocks) and
exposes them via `GET /v2/scanners/order-blocks`. All 11 correctness properties from the
design and all 10 requirement groups are covered.

**Discipline**: Every test (TDD red phase) is written in Task 2 before any production
code exists in Tasks 3–11. `hypothesis` property-based tests enforce all 11 formal
properties. Each implementation task makes a specific slice of the red-phase tests go
green. Tasks marked `*` are optional and can be skipped for a faster MVP.

---

## Tasks

- [x] 1. Scaffold module structure, configuration, and data models
  - [x] 1.1 Create package directories and `__init__.py` files
    - Create `src/scanners/__init__.py` with docstring: `"""Order Block Scanner module — institutional zone detection for the F&O universe."""`
    - Create `src/scanners/order_block/__init__.py` with brief package docstring
    - _Requirements: 1–9 (foundation for all components)_

  - [x] 1.2 Implement `OrderBlockConfig` in `src/scanners/order_block/config.py`
    - Inherit from `pydantic_settings.BaseSettings`; set `SettingsConfigDict(env_prefix="OB_", extra="ignore")`
    - Define all fields with defaults: `atr_window=14`, `impulse_multiplier=1.5`, `lookback_bars=5`, `swing_lookback=10`, `max_confirmation_bars=20`, `tf_5m_confirmation_bars=30`, `tf_1d_confirmation_bars=10`, `strength_multiplier=1.5`, `atr_strength_window=50`, `avg_volume_window=20`, `zone_expiry_bars=200`, `confluence_price_pct=0.02`, `default_limit=50`, `max_limit=200`
    - Each field uses `pydantic.Field(description=...)` for self-documentation
    - _Requirements: 1.1, 2.1, 3.1, 4.1–4.4, 6.8, 7.2, 7.3, 8.2_

  - [x] 1.3 Implement enums and Pydantic models in `src/scanners/order_block/schemas.py`
    - Implement three `(str, Enum)` enums: `OBZoneDirection` (BULLISH/BEARISH), `OBZoneLifecycle` (ACTIVE/TESTED/RETESTED/INVALIDATED/EXPIRED), `OBMarketRegime` (BULLISH_SWEEP/BEARISH_SWEEP/MIXED)
    - Implement `OBCandidate(BaseSchema)` with all 12 fields: `symbol`, `ob_bar_index`, `ob_timestamp`, `direction`, `ob_high`, `ob_low`, `ob_open`, `ob_close`, `ob_volume`, `impulse_bar_index`, `impulse_atr`, `timeframe`
    - Implement `OBCandidateConfirmed(OBCandidate)` adding: `bos_bar_index`, `bos_timestamp`, `bos_level`
    - Implement `OBZone(BaseSchema)` with all 14 fields from design §Data Models including `retest_count=0`, `invalidation_timestamp=None`, `multi_tf_confluence=False`
    - Implement `BreadthResult(BaseSchema)` with `bullish_breadth`, `bearish_breadth`, `market_regime`, `alignment_scores: dict[str, float]`
    - Implement `OrderBlockScanResult(BaseSchema)` with `zones: list[OBZone]`, `breadth: BreadthResult`, `scanned_at: datetime`, `n_symbols_scanned: int`
    - Implement `OrderBlockScanResponse(BaseSchema)` with all 7 top-level fields + `zones: list[OBZone]` (maps to Req 8 AC5+AC6)
    - Inherit all models from `src.schemas.base.BaseSchema` (`strict=True, frozen=True`)
    - _Requirements: 8.5, 8.6, 10.2_

- [x] 2. Write full TDD red-phase test suite (all tests must fail initially — no production code exists yet)
  - [x] 2.1 Write Hypothesis strategies and test scaffolding in `tests/test_order_block_scanner.py`
    - Import all schema types, config, and pytest markers; stub out empty `pass` bodies for every named test function listed in design §Testing Strategy so the file is valid Python
    - Implement `ohlcv_series_strategy(min_bars, max_bars)` `@st.composite` strategy: generates a `pd.DataFrame` with UTC `DatetimeIndex`, columns `open/high/low/close/volume`, enforcing `high ≥ max(open,close) ≥ min(open,close) ≥ low > 0`
    - Implement `ob_zone_strategy()` `@st.composite` strategy: generates a random valid `OBZone` with `ob_low ≤ ob_high`, all enum fields drawn from their enum members, timestamps in UTC, scores as non-negative floats
    - Implement `candle_strategy()` generating a single-row OHLCV `pd.Series`
    - Use `price_st = st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False)` and `volume_st = st.floats(min_value=0.0, max_value=1e10, allow_nan=False)` per Req 10.7
    - _Requirements: 10.7, 10.8_

  - [x] 2.2 Write TDD red-phase tests for `OrderBlockDetector` (Req 1, all 7 ACs)
    - `test_bullish_ob_candidate_detected` — hand-crafted OHLCV with bearish candle at index `i` followed by bullish impulse at `j` (body > 1.5×ATR14) → `detect_candidates` returns candidate with `ob_bar_index=i` and `direction=BULLISH` (Req 1.1)
    - `test_bearish_ob_candidate_detected` — symmetric: bullish candle at `i`, bearish impulse at `j` → `direction=BEARISH` (Req 1.2)
    - `test_no_lookahead_atr_computation` — verify ATR series has no `NaN` from a centered window; confirm ATR at bar 14 uses only bars 0–14 (Req 1.3, 9.1, 9.2)
    - `test_fewer_than_15_bars_returns_empty` — `len(df) < 15` → returns `[]`, no exception raised (Req 1.4)
    - `test_nan_bars_are_skipped` — insert `NaN` in `close` at bar `i`; detector skips that bar and returns candidates from clean bars only (Req 1.5)
    - `test_last_bullish_candle_selected_as_bearish_ob` — three consecutive bullish candles before bearish impulse → `ob_bar_index` is the index of the **last** bullish candle (Req 1.6, 1.7)
    - `test_deduplication_keeps_highest_conviction` — two impulse bars both referencing the same OB candle → only one candidate returned (design §detector.py step 6)
    - Mark all `@pytest.mark.tdd` and `@pytest.mark.unit`
    - _Requirements: 1.1–1.7, 9.1, 9.2_

  - [x] 2.3 Write TDD red-phase tests for `StructureValidator` (Req 2, all 6 ACs)
    - `test_bullish_bos_confirmed_on_close_above_swing_high` — `close[j] > swing_high_at_i` → returns `OBCandidateConfirmed` (Req 2.1)
    - `test_bearish_bos_confirmed_on_close_below_swing_low` — `close[j] < swing_low_at_i` → confirmed (Req 2.2)
    - `test_wick_above_swing_high_does_not_confirm` — `high[j] > swing_high_at_i` but `close[j] ≤ swing_high_at_i` → candidate remains unconfirmed (Req 2.3)
    - `test_swing_reference_is_trailing_only` — swing levels computed with backward rolling only; no future bars influence bar `i` reference (Req 2.4, 9.4)
    - `test_bos_confirmation_records_evidence` — confirmed candidate stores `bos_bar_index`, `bos_level`, `bos_timestamp` matching the confirming bar (Req 2.5)
    - `test_candidate_expires_when_no_bos_within_window` — no close-based BoS in first `max_confirmation_bars` bars → candidate excluded from output (Req 2.6)
    - `test_confirmation_at_exact_boundary_included` — BoS at bar `i + max_confirmation_bars` (inclusive) → confirmed, not expired
    - Mark all `@pytest.mark.tdd` and `@pytest.mark.unit`
    - _Requirements: 2.1–2.6, 9.4_

  - [x] 2.4 Write TDD red-phase tests for `StrengthFilter` and `VolumeRanker` (Req 3, 4)
    - `test_move_strength_ratio_formula` — `ratio = abs(close[j] - open[j]) / ATR50[j]`; verify exact arithmetic on a known series (Req 4.1)
    - `test_strength_filter_passes_ratio_at_threshold` — ratio exactly `1.5` → candidate passes (boundary-inclusive) (Req 4.2)
    - `test_strength_filter_rejects_below_threshold` — ratio `1.49` → candidate excluded (Req 4.2)
    - `test_filtered_zone_logged_at_debug` — mock structlog; verify excluded candidate emits DEBUG log with `symbol`, `bar_index`, computed `ratio` (Req 4.3)
    - `test_short_series_uses_min_periods_1` — `len(df) = 10` → ATR50 computed with `min_periods=1`, no exception (Req 4.4)
    - `test_strength_filter_is_per_symbol` — two symbols with different ATR profiles; each evaluated against its own ATR (Req 4.5)
    - `test_volume_score_formula` — `volume_score = ob_volume / avg_volume_20`; exact computation on known series (Req 3.1)
    - `test_single_symbol_zones_sorted_descending` — three zones with different volume_scores → output order is descending (Req 3.2)
    - `test_cross_symbol_zones_sorted_descending` — zones from two symbols interleaved; output globally descending by volume_score (Req 3.3)
    - `test_zero_avg_volume_sets_score_to_zero` — `avg_volume_20 == 0` → `volume_score = 0.0`, zone placed at end (Req 3.4)
    - `test_volume_ranker_preserves_all_metadata` — sort does not alter `symbol`, `direction`, `ob_high`, `ob_low`, `ob_timestamp`, `bos_timestamp`, `lifecycle_state`, `timeframe` (Req 3.5)
    - Mark all `@pytest.mark.tdd` and `@pytest.mark.unit`
    - _Requirements: 3.1–3.5, 4.1–4.5_

  - [x] 2.5 Write TDD red-phase tests for `BreadthTracker` and `ZoneManager` (Req 5, 6)
    - `test_bullish_breadth_counts_active_bullish_zones` — 3 of 5 symbols have ACTIVE bullish zones → `bullish_breadth = 0.6` (Req 5.1)
    - `test_bearish_breadth_counts_active_bearish_zones` — symmetric for bearish (Req 5.2)
    - `test_breadth_alignment_score_matches_direction` — bullish zone gets `bullish_breadth` value; bearish zone gets `bearish_breadth` (Req 5.3)
    - `test_regime_bullish_sweep_above_60pct` — `bullish_breadth = 0.61` → `market_regime = BULLISH_SWEEP` (Req 5.4)
    - `test_regime_mixed_at_exactly_60pct` — `bullish_breadth = 0.60` → `market_regime = MIXED` (threshold is strictly `> 0.6`) (Req 5.4, 5.6)
    - `test_regime_bearish_sweep_above_60pct` — `bearish_breadth = 0.61` → `market_regime = BEARISH_SWEEP` (Req 5.5)
    - `test_regime_mixed_when_neither_exceeds_threshold` — both below `0.6` → MIXED (Req 5.6)
    - `test_empty_universe_returns_zero_breadth` — `total_symbols = 0` → `bullish_breadth = bearish_breadth = 0.0`, MIXED, no exception (Req 5.7)
    - `test_new_confirmed_zone_has_active_state` — freshly created `OBZone` → `lifecycle_state = ACTIVE`, `retest_count = 0` (Req 6.1)
    - `test_bullish_zone_transitions_to_tested` — `candle_low ∈ [ob_low, ob_high]` and `close ≥ ob_low` → state becomes TESTED (Req 6.2)
    - `test_tested_zone_transitions_to_retested` — second entry condition on TESTED zone → RETESTED, `retest_count = 2` (Req 6.3)
    - `test_bullish_zone_invalidated_when_close_below_ob_low` — `close < ob_low` from any state → INVALIDATED with `invalidation_timestamp` set (Req 6.4)
    - `test_bearish_zone_invalidated_when_close_above_ob_high` — `close > ob_high` → INVALIDATED (Req 6.5)
    - `test_invalidated_zone_stays_invalidated` — any further candle on INVALIDATED zone → still INVALIDATED (Req 6.6)
    - `test_active_zone_expires_after_timeout` — `candle_bar_index - creation_bar_index > zone_expiry_bars` with no test event → EXPIRED (Req 6.8)
    - Mark all `@pytest.mark.tdd` and `@pytest.mark.unit`
    - _Requirements: 5.1–5.7, 6.1–6.8_

  - [x] 2.6 Write TDD red-phase tests for multi-timeframe support and ScannerAPI (Req 7, 8, 9)
    - `test_valid_timeframes_accepted` — ObScanner.scan with each of `"5m"`, `"15m"`, `"1h"`, `"1d"` → no ValueError (Req 7.1)
    - `test_5m_max_confirmation_bars_override` — `timeframe="5m"` → effective `max_confirmation_bars = 30` (Req 7.2)
    - `test_1d_max_confirmation_bars_override` — `timeframe="1d"` → effective `max_confirmation_bars = 10` (Req 7.3)
    - `test_invalid_timeframe_raises_value_error` — `timeframe="1w"` → `ValueError` with message listing `5m, 15m, 1h, 1d` (Req 7.4)
    - `test_zones_tagged_with_source_timeframe` — all zones in scan result carry `timeframe` matching the scan parameter; no cross-timeframe mixing (Req 7.5)
    - `test_multi_tf_confluence_flag_set` — same symbol with bullish zones on `1h` and `1d` within 2% price proximity → both zones have `multi_tf_confluence = True` (Req 7.6)
    - `test_endpoint_returns_200_with_zones` — `GET /v2/scanners/order-blocks` with valid API key → HTTP 200, JSON with `zones` field (Req 8.1)
    - `test_limit_parameter_caps_results` — `?limit=3` with 10 zones available → response contains 3 zones (Req 8.2)
    - `test_default_limit_is_50` — no `limit` param → at most 50 zones (Req 8.2)
    - `test_direction_filter_bullish` — `?direction=bullish` → all returned zones have `direction = "bullish"` (Req 8.3)
    - `test_direction_filter_bearish` — `?direction=bearish` → all returned zones have `direction = "bearish"` (Req 8.3)
    - `test_direction_omitted_returns_both_directions` — no direction param → both bullish and bearish zones returned (Req 8.3)
    - `test_timeframe_filter_applied` — `?timeframe=1h` → all returned zones have `timeframe = "1h"` (Req 8.4)
    - `test_each_zone_has_all_12_required_fields` — zone JSON contains all of: `symbol`, `direction`, `timeframe`, `ob_high`, `ob_low`, `ob_timestamp`, `bos_timestamp`, `volume_score`, `move_strength_ratio`, `breadth_alignment_score`, `lifecycle_state`, `multi_tf_confluence` (Req 8.5)
    - `test_response_has_all_6_top_level_fields` — response JSON has: `scanned_at`, `n_symbols_scanned`, `market_regime`, `bullish_breadth`, `bearish_breadth`, `n_zones` (Req 8.6)
    - `test_empty_zones_returns_200_not_404` — no active zones after filters → HTTP 200, `zones=[]`, `n_zones=0` (Req 8.7)
    - `test_missing_api_key_returns_401` — request with no `X-API-KEY` header → HTTP 401 (Req 8.8)
    - `test_limit_zero_returns_422` — `?limit=0` → HTTP 422 (Req 8.9)
    - `test_limit_201_returns_422` — `?limit=201` → HTTP 422 (Req 8.9)
    - `test_no_lookahead_rolling_windows_used` — assert ATR and swing computations use only backward-looking `.rolling()` with no `center=True` (verified by inspecting computed series) (Req 9.1, 9.2)
    - `test_appending_candle_does_not_change_prior_zones` — scan on `df[0:n]`, append one row to get `df[0:n+1]`, scan again; zone set for bars `[0, n-1]` is identical (Req 9.3)
    - Mark API tests with `@pytest.mark.tdd` and `@pytest.mark.integration`; unit tests with `@pytest.mark.tdd` and `@pytest.mark.unit`
    - _Requirements: 7.1–7.6, 8.1–8.9, 9.1–9.4_

  - [x] 2.7 Write property-based tests for correctness Properties 1–11
    - `test_candidate_detection_definition` — `@given(ohlcv_series_strategy())`: for any generated series where `detect_candidates` returns a bullish candidate at `i`, candle `i` must be bearish AND a bar `j > i` exists with bullish body `> 1.5×ATR14[j]`; annotate `# Feature: order-block-scanner, Property 1` (Req 1.1, 1.2)
    - `test_last_opposing_candle_selection` — `@given(...)`: for any OHLCV with ≥2 consecutive opposing candles before an impulse, OB index equals index of the most recent opposing candle; annotate Property 2 (Req 1.6, 1.7)
    - `test_bos_confirmation_invariant` — `@given(...)`: for any confirmed `OBZone`, there exists bar `j > i` where close crosses the correct swing level; annotate Property 3 (Req 2.1, 2.2, 10.1)
    - `test_no_lookahead_invariant` — `@given(ohlcv_series_strategy())`: appending any generated candle to a series must not alter zones detected in prior bars; compare zone sets by `(ob_timestamp, direction)`; annotate Property 4 (Req 1.3, 9.1–9.4, 10.4)
    - `test_volume_monotonicity_and_metadata` — `@given(st.lists(ob_zone_strategy(), min_size=2, max_size=20))`: after `rank_by_volume`, `volume_score[k] ≥ volume_score[k+1]` for all k; all metadata fields unchanged; annotate Property 5 (Req 3.2, 3.3, 3.5, 10.3)
    - `test_strength_filter_correctness` — `@given(...)`: every zone in filter output has `move_strength_ratio ≥ strength_multiplier`; no zone below threshold appears; annotate Property 6 (Req 4.1, 4.2)
    - `test_breadth_bounds_and_alignment` — `@given(...)`: `bullish_breadth ∈ [0.0, 1.0]`, `bearish_breadth ∈ [0.0, 1.0]`; each bullish zone's `breadth_alignment_score == bullish_breadth`; annotate Property 7 (Req 5.1–5.3)
    - `test_zone_terminal_invalidation` — `@given(ob_zone_strategy(), st.lists(candle_strategy(), min_size=1, max_size=50))`: once a zone reaches INVALIDATED after any invalidating candle, all subsequent `update_zone_lifecycle` calls leave it INVALIDATED; annotate Property 8 (Req 6.4–6.6)
    - `test_ob_zone_round_trip_serialization` — `@given(ob_zone_strategy())`: `OBZone.model_validate(z.model_dump(mode="json")) == z` for any generated OBZone; annotate Property 9 (Req 10.2)
    - `test_timeframe_tagging_consistency` — `@given(...)`: for any single-timeframe scan, every zone in the result has `zone.timeframe == scan_timeframe`; annotate Property 10 (Req 7.5)
    - `test_api_response_schema_completeness` — `@given(st.lists(ob_zone_strategy(), min_size=1, max_size=10))`: each zone's `model_dump(mode="json")` dict contains exactly all 12 required fields from Req 8 AC5; annotate Property 11 (Req 8.5)
    - Decorate all with `@pytest.mark.unit`, `@pytest.mark.slow`, and `@settings(max_examples=100, derandomize=True)`
    - _Requirements: 10.1–10.4, 10.7, 10.8_

  - [x] 2.8 Write PIT (Point-In-Time) tests and remaining coverage
    - `test_no_zone_uses_future_timestamp` — for all zones in a scan result, `ob_timestamp ≤ bos_timestamp ≤ scanned_at`; mark `@pytest.mark.pit` and `@pytest.mark.unit` (Req 10.6)
    - `test_swing_reference_level_reflects_lookback_shift` — reference level at bar `i` equals `swing_high.shift(swing_lookback).iloc[i]`; verify with a series where the shift makes a material difference (Req 9.4)
    - `test_modifying_future_candle_does_not_alter_prior_zone_set` — change `close` at bar `k > i`; re-run detector; zone at `i` is unchanged (Req 9.3)
    - Mark PIT tests `@pytest.mark.pit` and unit tests `@pytest.mark.unit`
    - _Requirements: 9.3, 9.4, 10.6_

- [x] 3. Implement `OrderBlockDetector` in `src/scanners/order_block/detector.py`
  - [x] 3.1 Implement `detect_candidates()` function
    - Guard: return `[]` if `len(df) < 15` or DataFrame has no non-NaN rows
    - Compute true range: `tr = max(high-low, |high-prev_close|, |low-prev_close|)` using `.shift(1)` for prev_close
    - Compute ATR14: `tr.rolling(window=config.atr_window, min_periods=1).mean()` — never `center=True`
    - Build `body_size = abs(close - open)`; `impulse_bull` mask = `(body_size > config.impulse_multiplier * atr) & (close > open)`; `impulse_bear` mask = symmetric
    - For each impulse bar `j`: scan backward from `j-1` to `max(0, j-config.lookback_bars)` for the **last** opposing candle; skip bars where any of `open/high/low/close/volume` is NaN
    - Deduplicate by OB candle index: if multiple impulse bars reference same `i`, retain the entry with the largest `body_size[j]`
    - Return `list[OBCandidate]` with all fields populated; `timeframe` passed through from argument
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 9.1, 9.2_

  - [x]* 3.2 Write example-based unit tests for `OrderBlockDetector` edge cases
    - Concrete test: minimal 15-bar series with exactly one known bullish OB → `len(result) == 1` and fields match expected values
    - Concrete test: exactly 14 bars → `result == []`
    - Concrete test: all NaN values in `close` column → `result == []`, no exception
    - Add to `tests/test_order_block_scanner.py`; mark `@pytest.mark.unit`
    - _Requirements: 1.4, 1.5_

- [x] 4. Implement `StructureValidator` in `src/scanners/order_block/structure.py`
  - [x] 4.1 Implement `validate_structure()` function
    - Compute `swing_high = df['high'].rolling(config.swing_lookback * 2 + 1, min_periods=config.swing_lookback + 1).max()` once for the whole DataFrame
    - Compute `swing_low` symmetrically using `df['low'].rolling(...).min()`
    - Shift both series: `ref_high = swing_high.shift(config.swing_lookback)` and `ref_low = swing_low.shift(config.swing_lookback)` to satisfy Req 9.4
    - For each candidate at OB index `i`: extract `ref_high.iloc[i]` or `ref_low.iloc[i]` as the reference level
    - Apply timeframe-specific `max_confirmation_bars` when provided via config
    - Scan confirmation window `[i+1, min(i+max_confirmation_bars, len(df)-1)]`: for bullish, find first `j` where `df['close'].iloc[j] > ref_high`; for bearish, `< ref_low`; wicks do NOT confirm
    - Confirmed → return `OBCandidateConfirmed` with `bos_bar_index=j`, `bos_timestamp=df.index[j]`, `bos_level=ref_high/ref_low`
    - No confirmation → mark EXPIRED, exclude from output
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 9.4_

  - [x]* 4.2 Write example-based unit tests for `StructureValidator` edge cases
    - Concrete test: wick only (high > swing_high, close ≤ swing_high) → not confirmed
    - Concrete test: BoS exactly at bar `i + max_confirmation_bars` (inclusive) → confirmed
    - Concrete test: BoS at `i + max_confirmation_bars + 1` → expired, excluded
    - Add to `tests/test_order_block_scanner.py`; mark `@pytest.mark.unit`
    - _Requirements: 2.3, 2.6_

- [x] 5. Implement `StrengthFilter` in `src/scanners/order_block/strength.py`
  - [x] 5.1 Implement `filter_by_strength()` function
    - Compute ATR50 from true range: `tr.rolling(config.atr_strength_window, min_periods=1).mean()` — never `center=True`
    - Per candidate: `move_strength_ratio = abs(df['close'].iloc[j] - df['open'].iloc[j]) / atr50.iloc[j]`
    - If `ratio >= config.strength_multiplier`: include in output
    - Else: emit structlog DEBUG event with keys `symbol`, `bar_index`, `move_strength_ratio`; exclude candidate
    - Accept a `structlog.BoundLogger` parameter; use `get_logger(__name__)` as default in module scope
    - Single-symbol `df` accepted; `ObScanner` calls this once per symbol (no cross-symbol ATR can occur)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x]* 5.2 Write example-based unit tests for `StrengthFilter` edge cases
    - Concrete test: `move_strength_ratio` exactly at `strength_multiplier` → candidate passes (boundary inclusive)
    - Concrete test: 8-bar series (< 50 bars) → ATR computed with `min_periods=1`, no exception, correct ratio
    - Add to `tests/test_order_block_scanner.py`; mark `@pytest.mark.unit`
    - _Requirements: 4.2, 4.4_

- [x] 6. Implement `VolumeRanker` in `src/scanners/order_block/volume.py`
  - [x] 6.1 Implement `rank_by_volume()` function
    - For each zone: look up its `df = df_by_symbol[zone.symbol]`, compute `avg_vol_20 = df['volume'].rolling(config.avg_volume_window, min_periods=1).mean()`, then index by `zone.ob_timestamp`
    - If `avg_vol_20 == 0.0` or NaN: set `volume_score = 0.0`; else `volume_score = zone.ob_volume / avg_vol_20`
    - Apply score via `.model_copy(update={'volume_score': score})` — never mutate the frozen `OBZone`
    - Sort: primary key `−volume_score` descending, tie-break `−ob_timestamp` (most recent first)
    - Return sorted list; all fields except `volume_score` must be byte-for-byte identical to input
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x]* 6.2 Write example-based unit tests for `VolumeRanker` edge cases
    - Concrete test: single zone → returned unchanged, `volume_score` set correctly
    - Concrete test: all zones have `avg_vol_20 = 0` → all `volume_score = 0.0`, sorted by `ob_timestamp` desc
    - Add to `tests/test_order_block_scanner.py`; mark `@pytest.mark.unit`
    - _Requirements: 3.4_

- [x] 7. Implement `BreadthTracker` in `src/scanners/order_block/breadth.py`
  - [x] 7.1 Implement `compute_breadth()` function
    - Count `bull_symbols` = number of symbols with ≥1 zone where `direction == BULLISH` and `lifecycle_state == ACTIVE`
    - Count `bear_symbols` symmetrically
    - Guard: if `total_symbols == 0` → return `BreadthResult` with both breadths `0.0` and `market_regime = MIXED`
    - `bullish_breadth = bull_symbols / total_symbols`, `bearish_breadth = bear_symbols / total_symbols`
    - Determine regime: `bullish_breadth > 0.6` → BULLISH_SWEEP; `bearish_breadth > 0.6` → BEARISH_SWEEP; else MIXED (threshold is **strictly** greater than 0.6)
    - Build `alignment_scores` dict: BULLISH zones → `bullish_breadth`, BEARISH zones → `bearish_breadth`
    - Return `BreadthResult` with all fields populated
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [x]* 7.2 Write example-based unit tests for `BreadthTracker` edge cases
    - Concrete test: `bullish_breadth = 0.600000` (exactly 60%) → regime is MIXED (not BULLISH_SWEEP)
    - Concrete test: `bullish_breadth = 0.600001` → regime is BULLISH_SWEEP
    - Add to `tests/test_order_block_scanner.py`; mark `@pytest.mark.unit`
    - _Requirements: 5.4, 5.6_

- [x] 8. Implement `ZoneManager` in `src/scanners/order_block/zone_manager.py`
  - [x] 8.1 Implement `update_zone_lifecycle()` state-machine function
    - Signature: `update_zone_lifecycle(zone, candle_low, candle_high, candle_close, candle_bar_index, creation_bar_index, config) -> OBZone`
    - **Step 1 — terminal guard**: if `zone.lifecycle_state == INVALIDATED` → return zone unchanged immediately
    - **Step 2 — invalidation** (any non-INVALIDATED state): bullish: `candle_close < zone.ob_low` → `.model_copy(update={'lifecycle_state': INVALIDATED, 'invalidation_timestamp': utc_now})`; bearish: `candle_close > zone.ob_high` → same
    - **Step 3 — expiry** (only ACTIVE): `(candle_bar_index - creation_bar_index) > config.zone_expiry_bars` → `.model_copy(update={'lifecycle_state': EXPIRED})`
    - **Step 4 — test transition**: ACTIVE bullish: `candle_low ≥ zone.ob_low AND candle_low ≤ zone.ob_high AND candle_close ≥ zone.ob_low` → `.model_copy(update={'lifecycle_state': TESTED, 'retest_count': 1})`
    - **Step 5 — retest transition**: TESTED bullish: same entry condition → `.model_copy(update={'lifecycle_state': RETESTED, 'retest_count': zone.retest_count + 1})`
    - Bearish transitions are symmetric (use `candle_high` for entry into bearish zone range)
    - All updates via `.model_copy(update={...})` — frozen model, no mutation in place
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [x]* 8.2 Write example-based unit tests for `ZoneManager` edge cases
    - Concrete test: RETESTED zone hit by invalidating candle → final state is INVALIDATED
    - Concrete test: EXPIRED zone hit by invalidating candle → final state is INVALIDATED (invalidation overrides EXPIRED)
    - Concrete test: zone age exactly equal to `zone_expiry_bars` → still ACTIVE; age `+ 1` → EXPIRED
    - Add to `tests/test_order_block_scanner.py`; mark `@pytest.mark.unit`
    - _Requirements: 6.4, 6.6, 6.8_

- [x] 9. Checkpoint — verify all component tests pass
  - Ensure all tests for tasks 3–8 pass (detector, structure, strength, volume, breadth, zone_manager); run `pytest tests/test_order_block_scanner.py -m "unit and not slow"`; ask the user if questions arise.

- [x] 10. Implement `ObScanner` orchestrator in `src/scanners/order_block/scanner.py`
  - [x] 10.1 Implement `ObScanner` class with `scan()` method
    - Constructor: `__init__(self, config: OrderBlockConfig | None = None)` — store config, fall back to `OrderBlockConfig()`
    - `scan(ohlcv_by_symbol, timeframe, existing_zones=None) -> OrderBlockScanResult`
    - **Validate timeframe**: raise `ValueError("Unsupported timeframe '{tf}'. Valid values: 5m, 15m, 1h, 1d")` for unsupported values
    - **Apply per-timeframe config overrides**: `5m` → `max_confirmation_bars = config.tf_5m_confirmation_bars`; `1d` → `config.tf_1d_confirmation_bars`; create modified config copy via `config.model_copy()`
    - **Per-symbol pipeline** (independent per symbol): `detect_candidates` → `validate_structure` → `filter_by_strength` → convert confirmed to `OBZone(lifecycle_state=ACTIVE, ...)`
    - **Lifecycle update**: if `existing_zones[symbol]` provided, run `update_zone_lifecycle` for each existing zone against the latest candle; merge non-INVALIDATED existing + new zones
    - **Global ranking**: pass all zones and `ohlcv_by_symbol` to `rank_by_volume`
    - **Confluence**: call `_compute_multi_tf_confluence(zones, config)` after ranking
    - **Breadth**: call `compute_breadth(zones_by_symbol, len(ohlcv_by_symbol))` and assign `breadth_alignment_score` to each zone
    - **Logging**: structlog INFO on completion (`symbols_scanned`, `n_zones`, `market_regime`, `elapsed_ms`); structlog ERROR on per-symbol exception (isolate, continue)
    - Return `OrderBlockScanResult(zones=final_zones, breadth=breadth, scanned_at=datetime.now(tz=timezone.utc), n_symbols_scanned=len(ohlcv_by_symbol))`
    - _Requirements: 1.1–1.7, 2.1–2.6, 3.1–3.5, 4.1–4.5, 5.1–5.7, 6.1–6.8, 7.1–7.5, 9.1–9.4_

  - [x] 10.2 Implement `_compute_multi_tf_confluence()` helper in `scanner.py`
    - Accept `zones: list[OBZone]` and `config: OrderBlockConfig`; return updated `list[OBZone]`
    - For the same `symbol`, pair each bullish `1h` zone with each bullish `1d` zone
    - Compute proximity: `abs(z1.ob_high - z2.ob_high) / z1.ob_high < config.confluence_price_pct`
    - On match: update both zones via `.model_copy(update={'multi_tf_confluence': True})`
    - Return full zone list with confluence flags applied
    - _Requirements: 7.6_

  - [x]* 10.3 Write example-based unit tests for `ObScanner` orchestration
    - Concrete test: `timeframe="1w"` → `ValueError` raised with correct message
    - Concrete test: `ohlcv_by_symbol = {}` → result has `zones=[]`, `bullish_breadth=0.0`, `market_regime=MIXED`, no exception
    - Concrete test: single symbol with a hand-crafted series containing one known bullish OB → `scan()` returns exactly one zone with all fields populated and `lifecycle_state = ACTIVE`
    - Add to `tests/test_order_block_scanner.py`; mark `@pytest.mark.unit`
    - _Requirements: 7.4, 7.5_

- [x] 11. Implement `ScannerAPI` and register the FastAPI router
  - [x] 11.1 Implement `GET /v2/scanners/order-blocks` in `src/api/scanners.py`
    - Create `router = APIRouter(tags=["scanners"])` and module-level `_config = OrderBlockConfig()`, `_scanner = ObScanner(config=_config)`
    - Define `VALID_TIMEFRAMES = frozenset({"5m", "15m", "1h", "1d"})`
    - Implement `async def get_order_blocks(limit, direction, timeframe, include_invalidated)` with FastAPI `Query` annotations: `limit: int = Query(default=50, ge=1, le=200)` (FastAPI validates automatically → HTTP 422), `direction: str | None = Query(default=None)`, `timeframe: str | None = Query(default=None)`, `include_invalidated: bool = Query(default=False)`
    - Validate `direction` ∈ `{"bullish", "bearish", None}` → return `JSONResponse(status_code=422, ...)` for invalid values
    - Validate `timeframe` ∈ `VALID_TIMEFRAMES | {None}` → return `JSONResponse(status_code=422, ...)` for invalid values
    - Call `_fetch_ohlcv(timeframe or "1h")` (stub that returns empty dict; real implementation injects market data)
    - Call `_scanner.scan(ohlcv_by_symbol, timeframe=timeframe or "1h")`
    - Filter: exclude INVALIDATED (unless `include_invalidated=True`), apply `direction` and `timeframe` filters, apply `[:limit]` slice
    - Build `OrderBlockScanResponse` and return `JSONResponse(content=response.model_dump(mode="json"))`
    - Empty `zones=[]` → HTTP 200 (never 404); HTTP 401 handled globally by `api_key_middleware`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9_

  - [x] 11.2 Register scanner router in `src/main.py`
    - In the `# ── Router registration ──` section, after the existing router imports, add:
      `from src.api import scanners as scanners_api`
    - After the existing `app.include_router(...)` calls, add:
      `app.include_router(scanners_api.router, prefix="/v2")`
    - No other changes to `main.py` are needed; `api_key_middleware` already covers `/v2/scanners/order-blocks`
    - _Requirements: 8.1, 8.8_

  - [x]* 11.3 Write API integration tests using FastAPI `TestClient`
    - Create a `TestClient(app)` fixture; mock `_fetch_ohlcv` or `ObScanner.scan` to return a deterministic set of 5 zones across 2 symbols and 2 timeframes
    - Implement integration tests for all `test_endpoint_*`, `test_limit_*`, `test_direction_*`, `test_timeframe_filter_*`, `test_missing_api_key_*`, and `test_limit_*_returns_422` from task 2.6 (replacing stub bodies with real assertions)
    - Mark all with `@pytest.mark.integration`
    - Add to `tests/test_order_block_scanner.py`
    - _Requirements: 8.1–8.9_

- [x] 12. Final checkpoint — full test suite green
  - Run `pytest tests/test_order_block_scanner.py --hypothesis-seed=0` and ensure all TDD red-phase tests (now green), PBT tests, PIT tests, and API integration tests pass; ask the user if questions arise.

---

## Notes

- Tasks marked `*` are optional and can be skipped for a faster MVP
- All TDD tests in task 2 are mandatory — they are the red-phase specification
- `OBZone` is `frozen=True`; **never** mutate in place — always use `.model_copy(update={...})`
- Run property tests: `pytest -m "slow" --hypothesis-seed=0` (reproducible in CI per Req 10.8)
- Run fast unit tests: `pytest tests/test_order_block_scanner.py -m "unit and not slow"`
- `X-API-KEY` auth is global via `api_key_middleware` in `main.py` — scanner router adds zero auth code
- `OrderBlockConfig` is passed as a single injected object to all sub-components — no global state, no monkey-patching needed in tests
- `_fetch_ohlcv` in the API is a stub; real market data sourcing is an integration concern outside scanner scope

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["2.1"] },
    { "id": 3, "tasks": ["2.2"] },
    { "id": 4, "tasks": ["2.3"] },
    { "id": 5, "tasks": ["2.4"] },
    { "id": 6, "tasks": ["2.5"] },
    { "id": 7, "tasks": ["2.6"] },
    { "id": 8, "tasks": ["2.7"] },
    { "id": 9, "tasks": ["2.8"] },
    { "id": 10, "tasks": ["3.1"] },
    { "id": 11, "tasks": ["3.2", "4.1"] },
    { "id": 12, "tasks": ["4.2", "5.1"] },
    { "id": 13, "tasks": ["5.2", "6.1"] },
    { "id": 14, "tasks": ["6.2", "7.1", "8.1"] },
    { "id": 15, "tasks": ["7.2", "10.1"] },
    { "id": 16, "tasks": ["8.2", "10.2"] },
    { "id": 17, "tasks": ["10.3", "11.1"] },
    { "id": 18, "tasks": ["11.2"] },
    { "id": 19, "tasks": ["11.3"] }
  ]
}
```
