# Requirements Document

## Introduction

The Order Block Scanner is a new `src/scanners/` module for the AlphaForge ML service.
It identifies institutional order block zones across the F&O (Futures & Options) universe
(~200+ NSE stocks), ranks them by volume and liquidity, and exposes a REST endpoint so
the AlphaForge UI can surface the highest-conviction zones in real time.

An **order block** is the last opposing candle immediately before a strong impulsive move
that breaks market structure. These zones mark institutional order clusters; price
frequently returns to them before continuing the prevailing trend.

The scanner is implemented TDD-first: every acceptance criterion below is written to drive
a pytest test case before the production code exists. Property-based tests use
`hypothesis`; deterministic tests use `pytest` with the markers defined in `pyproject.toml`
(`unit`, `integration`, `slow`, `pit`, `tdd`).

---

## Glossary

- **OrderBlockDetector**: The core detection engine that identifies bullish and bearish
  order block candles from OHLCV series data.
- **StructureValidator**: The component that verifies a valid Break of Structure (BoS)
  exists after a candidate order block candle, using close prices only (no wicks).
- **VolumeRanker**: The component that scores and sorts confirmed order block zones by the
  volume traded at the zone candle, enabling liquidity-first ordering.
- **StrengthFilter**: The component that computes each move's strength relative to the
  stock's own trailing 50-session ATR baseline and filters out weak moves.
- **BreadthTracker**: The component that counts how many F&O symbols have aligned
  (same-direction) confirmed order blocks at a given point in time.
- **ZoneManager**: The component that tracks the lifecycle (active → tested → retested →
  invalidated) of each confirmed order block zone.
- **ObScanner**: The top-level orchestrator that runs all sub-components against the full
  F&O universe and returns a ranked list of signals.
- **ScannerAPI**: The FastAPI router that exposes the scanner results as a REST endpoint.
- **OHLCV_Series**: A `pandas.DataFrame` with columns `open`, `high`, `low`, `close`,
  `volume` indexed by UTC timestamp, one row per candle.
- **Swing_High**: The highest `high` over a configurable trailing window of bars, computed
  without look-ahead (trailing only, no centered window).
- **Swing_Low**: The lowest `low` over a configurable trailing window of bars, computed
  without look-ahead.
- **BoS**: Break of Structure — a candle whose *close* exceeds the most recent Swing_High
  (bullish BoS) or falls below the most recent Swing_Low (bearish BoS).
- **OB_Zone**: A confirmed order block zone defined by the high and low of the order block
  candle together with its timestamp, direction, volume, and lifecycle state.
- **Timeframe**: The candle aggregation interval — one of `5m`, `15m`, `1h`, `1d`.
- **Invalidation**: A bullish OB_Zone is invalidated when a subsequent candle closes below
  the zone's low; a bearish OB_Zone is invalidated when a subsequent candle closes above
  the zone's high.
- **F&O_Universe**: The list of NSE stocks eligible for Futures & Options trading
  (approximately 200+ symbols), as maintained in the service's symbol registry.

---

## Requirements

### Requirement 1: Order Block Candle Detection

**User Story:** As a quantitative analyst, I want the OrderBlockDetector to identify the
last opposing candle before a strong impulsive move, so that I can locate institutional
entry zones accurately.

#### Acceptance Criteria

1. WHEN an OHLCV_Series is provided, THE OrderBlockDetector SHALL identify every candle
   at index `i` as a **bullish order block candidate** when the candle at index `i` is
   bearish (`close[i] < open[i]`) AND a subsequent candle at index `j > i` is bullish
   (`close[j] > open[j]`) with a body size exceeding 1.5 × the 14-bar ATR at bar `j`.

2. WHEN an OHLCV_Series is provided, THE OrderBlockDetector SHALL identify every candle
   at index `i` as a **bearish order block candidate** when the candle at index `i` is
   bullish (`close[i] > open[i]`) AND a subsequent candle at index `j > i` is bearish
   (`close[j] < open[j]`) with a body size exceeding 1.5 × the 14-bar ATR at bar `j`.

3. THE OrderBlockDetector SHALL use only data at indices ≤ `i` when computing ATR and
   swing levels for the candle at index `i` (no look-ahead / no future data leakage).

4. WHEN an OHLCV_Series contains fewer than 15 bars, THE OrderBlockDetector SHALL return
   an empty list of candidates without raising an exception.

5. IF the OHLCV_Series contains `NaN` values in any of `open`, `high`, `low`, `close`, or
   `volume`, THEN THE OrderBlockDetector SHALL skip affected bars and continue processing
   the remaining bars.

6. THE OrderBlockDetector SHALL identify the **last** bearish candle before the impulsive
   bull move as the bullish order block (not the first or an intermediate bearish candle).

7. THE OrderBlockDetector SHALL identify the **last** bullish candle before the impulsive
   bear move as the bearish order block (not the first or an intermediate bullish candle).

---

### Requirement 2: Break of Structure Validation

**User Story:** As a trader, I want BoS confirmation based on candle closes (not wicks),
so that I can avoid false signals from temporary price spikes.

#### Acceptance Criteria

1. WHEN a bullish order block candidate exists at index `i`, THE StructureValidator SHALL
   confirm the candidate only when a candle at index `j > i` has `close[j]` strictly
   greater than the Swing_High computed over bars `[i - lookback, i]` (trailing window,
   default `lookback = 10`).

2. WHEN a bearish order block candidate exists at index `i`, THE StructureValidator SHALL
   confirm the candidate only when a candle at index `j > i` has `close[j]` strictly
   less than the Swing_Low computed over bars `[i - lookback, i]` (trailing window,
   default `lookback = 10`).

3. IF a candle's `high` or `low` wicks beyond the Swing_High or Swing_Low but the `close`
   does NOT cross the level, THEN THE StructureValidator SHALL NOT count that candle as a
   valid BoS and SHALL leave the candidate unconfirmed.

4. THE StructureValidator SHALL compute the Swing_High and Swing_Low using only trailing
   (backward-looking) rolling windows so that feature values at bar `i` cannot change when
   any bar at index `> i` is modified.

5. WHEN a BoS is confirmed, THE StructureValidator SHALL record the confirming bar index
   `j` and the exact Swing_High or Swing_Low level that was broken, so consumers can
   verify the evidence.

6. IF no BoS-confirming candle exists within `max_confirmation_bars` bars after the
   candidate (default `max_confirmation_bars = 20`), THEN THE StructureValidator SHALL
   mark the candidate as **expired** and SHALL NOT include it in confirmed OB_Zones.

---

### Requirement 3: Volume and Liquidity Ranking

**User Story:** As a portfolio manager, I want order blocks ranked by the volume traded at
the zone, so that I surface institutional-grade zones with the deepest liquidity first.

#### Acceptance Criteria

1. THE VolumeRanker SHALL assign each confirmed OB_Zone a `volume_score` equal to the
   volume at the order block candle divided by the 20-bar rolling average volume at that
   candle's timestamp (`ob_volume / avg_volume_20`), producing a relative volume multiple.

2. WHEN the ObScanner returns a list of confirmed OB_Zones for a single symbol, THE
   VolumeRanker SHALL sort the list in descending order of `volume_score` so that the
   zone with the highest relative volume appears first.

3. WHEN the ObScanner returns a cross-symbol list of confirmed OB_Zones, THE VolumeRanker
   SHALL sort the combined list in descending order of `volume_score` so that
   highest-liquidity zones from any symbol bubble to the top regardless of symbol.

4. IF `avg_volume_20` is zero or unavailable for a given bar, THEN THE VolumeRanker SHALL
   set `volume_score` to `0.0` for that zone and SHALL place it at the end of the sorted
   list.

5. THE VolumeRanker SHALL preserve the original zone metadata (`symbol`, `direction`,
   `ob_high`, `ob_low`, `ob_timestamp`, `bos_timestamp`, `lifecycle_state`) unchanged
   after sorting; only the sort order changes.

---

### Requirement 4: Relative Strength Filter

**User Story:** As a signal engineer, I want weak moves filtered out using each stock's
own baseline volatility, so that the scanner ignores noise and focuses on genuinely
impulsive moves.

#### Acceptance Criteria

1. THE StrengthFilter SHALL compute a per-symbol `move_strength_ratio` as the absolute
   price change of the impulsive candle (`|close[j] - open[j]|`) divided by the trailing
   50-session ATR measured at bar `j` using only data available at bar `j`.

2. THE StrengthFilter SHALL pass only order block candidates whose confirming impulsive
   candle has `move_strength_ratio ≥ 1.5` (default threshold, configurable via
   `strength_multiplier` parameter).

3. WHEN a candidate's `move_strength_ratio` is below the configured threshold, THE
   StrengthFilter SHALL exclude the candidate from the confirmed OB_Zone list and SHALL
   log the symbol, bar index, and computed ratio at `DEBUG` level.

4. IF the OHLCV_Series contains fewer than 50 bars when the impulsive candle is
   encountered, THEN THE StrengthFilter SHALL compute ATR using all available bars
   (`min_periods = 1`) rather than requiring exactly 50 bars.

5. THE StrengthFilter SHALL NOT compare move sizes across different symbols; each symbol
   is evaluated against its own trailing ATR baseline exclusively.

---

### Requirement 5: Market Breadth Signal

**User Story:** As a trader, I want to know how many F&O stocks share the same order block
direction at a given time, so that I can weight directional zones higher when the broader
market agrees.

#### Acceptance Criteria

1. WHEN the ObScanner completes a scan of the F&O_Universe, THE BreadthTracker SHALL
   compute `bullish_breadth` as the count of symbols that have at least one active
   bullish OB_Zone, divided by the total number of symbols scanned.

2. WHEN the ObScanner completes a scan of the F&O_Universe, THE BreadthTracker SHALL
   compute `bearish_breadth` as the count of symbols that have at least one active
   bearish OB_Zone, divided by the total number of symbols scanned.

3. THE BreadthTracker SHALL assign each confirmed OB_Zone a `breadth_alignment_score`
   equal to `bullish_breadth` for bullish zones and `bearish_breadth` for bearish zones,
   so the final ranking can use breadth as a secondary sort key.

4. WHEN `bullish_breadth` is greater than `0.6` (i.e., more than 60% of scanned symbols
   have bullish zones), THE BreadthTracker SHALL set `market_regime = "BULLISH_SWEEP"`.

5. WHEN `bearish_breadth` is greater than `0.6`, THE BreadthTracker SHALL set
   `market_regime = "BEARISH_SWEEP"`.

6. WHEN neither `bullish_breadth` nor `bearish_breadth` exceeds `0.6`, THE BreadthTracker
   SHALL set `market_regime = "MIXED"`.

7. IF the F&O_Universe contains zero symbols (empty scan), THEN THE BreadthTracker SHALL
   set both breadth values to `0.0` and `market_regime = "MIXED"` without raising an
   exception.

---

### Requirement 6: Zone Lifecycle Management

**User Story:** As a risk manager, I want order block zones to track whether they have
been tested, retested, or invalidated, so that I can distinguish fresh zones from used-up
or broken ones.

#### Acceptance Criteria

1. THE ZoneManager SHALL create each new confirmed OB_Zone with `lifecycle_state =
   "ACTIVE"`.

2. WHEN a new candle's `low` enters the range `[ob_low, ob_high]` for a bullish OB_Zone
   in `"ACTIVE"` state without the candle closing below `ob_low`, THE ZoneManager SHALL
   transition the zone to `lifecycle_state = "TESTED"`.

3. WHEN a `"TESTED"` bullish OB_Zone is entered by a subsequent candle's `low` for a
   second time without closing below `ob_low`, THE ZoneManager SHALL transition the zone
   to `lifecycle_state = "RETESTED"` and SHALL record the count of retest events.

4. WHEN a candle closes below `ob_low` for a bullish OB_Zone (regardless of its current
   lifecycle state), THE ZoneManager SHALL transition the zone to `lifecycle_state =
   "INVALIDATED"` and SHALL record the invalidating bar's timestamp.

5. WHEN a candle closes above `ob_high` for a bearish OB_Zone (regardless of its current
   lifecycle state), THE ZoneManager SHALL transition the zone to `lifecycle_state =
   "INVALIDATED"` and SHALL record the invalidating bar's timestamp.

6. THE ZoneManager SHALL never transition a zone out of `"INVALIDATED"` state; once
   invalidated, the zone remains permanently invalidated.

7. WHEN the ScannerAPI response is generated, THE ZoneManager SHALL exclude all zones
   with `lifecycle_state = "INVALIDATED"` from the active signal list unless the caller
   explicitly sets `include_invalidated = true`.

8. IF a zone has been in `"ACTIVE"` state for more than `zone_expiry_bars` bars (default
   `200`) without any test event, THEN THE ZoneManager SHALL transition the zone to
   `lifecycle_state = "EXPIRED"` to prevent stale zones from polluting results.

---

### Requirement 7: Multi-Timeframe Support

**User Story:** As a multi-timeframe analyst, I want to detect order blocks on 5m, 15m,
1h, and 1d candle data, so that I can identify zones across different holding horizons.

#### Acceptance Criteria

1. THE OrderBlockDetector SHALL accept a `timeframe` parameter with valid values `"5m"`,
   `"15m"`, `"1h"`, and `"1d"`, and SHALL process the OHLCV_Series at that timeframe
   without modification to the detection logic.

2. WHEN `timeframe = "5m"`, THE StructureValidator SHALL use `lookback = 10` bars and
   `max_confirmation_bars = 30` as defaults.

3. WHEN `timeframe = "1d"`, THE StructureValidator SHALL use `lookback = 10` bars and
   `max_confirmation_bars = 10` as defaults.

4. IF an unsupported timeframe value is provided, THEN THE OrderBlockDetector SHALL raise
   a `ValueError` with a message listing the supported timeframe values.

5. WHEN the ObScanner scans a symbol across multiple timeframes, THE ObScanner SHALL tag
   each OB_Zone with its source `timeframe` and SHALL NOT merge zones from different
   timeframes into a single zone.

6. WHEN a bullish OB_Zone is confirmed on both `1h` and `1d` timeframes within 2% price
   proximity, THE ObScanner SHALL compute a `multi_tf_confluence` boolean flag `true` for
   both zones to indicate higher conviction.

---

### Requirement 8: Scanner REST API

**User Story:** As a front-end developer, I want a REST endpoint that returns the top-N
order block signals sorted by volume and liquidity, so that the AlphaForge UI can display
the strongest institutional zones without custom filtering.

#### Acceptance Criteria

1. THE ScannerAPI SHALL expose a `GET /v2/scanners/order-blocks` endpoint that returns a
   JSON response containing a ranked list of confirmed, non-invalidated OB_Zones.

2. WHEN the request includes query parameter `limit` (integer, 1–200), THE ScannerAPI
   SHALL return at most `limit` zones; when `limit` is omitted, THE ScannerAPI SHALL
   default to `50`.

3. WHEN the request includes query parameter `direction` with value `"bullish"` or
   `"bearish"`, THE ScannerAPI SHALL filter results to only zones matching that direction;
   when `direction` is omitted, THE ScannerAPI SHALL return both directions.

4. WHEN the request includes query parameter `timeframe` (one of `"5m"`, `"15m"`, `"1h"`,
   `"1d"`), THE ScannerAPI SHALL return only zones detected on that timeframe; when
   `timeframe` is omitted, THE ScannerAPI SHALL return zones from all timeframes.

5. THE ScannerAPI response SHALL include the following fields for each OB_Zone:
   `symbol` (string), `direction` (string `"bullish"` or `"bearish"`), `timeframe`
   (string), `ob_high` (float), `ob_low` (float), `ob_timestamp` (ISO-8601 UTC string),
   `bos_timestamp` (ISO-8601 UTC string), `volume_score` (float), `move_strength_ratio`
   (float), `breadth_alignment_score` (float), `lifecycle_state` (string),
   `multi_tf_confluence` (boolean).

6. THE ScannerAPI response SHALL include top-level fields: `scanned_at` (ISO-8601 UTC
   timestamp of scan completion), `n_symbols_scanned` (integer), `market_regime` (string),
   `bullish_breadth` (float), `bearish_breadth` (float), `n_zones` (integer count of
   zones in the response).

7. IF no confirmed OB_Zones exist after filtering, THE ScannerAPI SHALL return HTTP 200
   with an empty `zones` list and SHALL NOT return HTTP 404.

8. IF the authentication header `X-API-KEY` is missing or invalid, THEN THE ScannerAPI
   SHALL return HTTP 401 without a zone payload, consistent with all other authenticated
   endpoints in the service.

9. WHEN the `limit` query parameter is less than 1 or greater than 200, THE ScannerAPI
   SHALL return HTTP 422 with a descriptive validation error.

---

### Requirement 9: No-Lookahead Data Integrity

**User Story:** As a backtesting engineer, I want every feature value at bar `i` to depend
only on data at bars ≤ `i`, so that simulated signals cannot incorporate future information
that would not have been available in real time.

#### Acceptance Criteria

1. THE OrderBlockDetector SHALL compute all swing levels using trailing (backward-looking)
   rolling windows exclusively; centered rolling windows are prohibited.

2. THE OrderBlockDetector SHALL compute ATR using `.rolling(window=N, min_periods=1)`
   without any `center=True` argument, so that the ATR at bar `i` uses only bars
   `[i-N+1, i]`.

3. FOR ALL detected OB_Zones, modifying any candle at index `k > i` in the OHLCV_Series
   SHALL NOT change the zone record that was detected at index `i`.

4. THE StructureValidator SHALL shift the Swing_High and Swing_Low series by `lookback`
   bars when computing the reference level at bar `i`, so the reference level reflects the
   structure known `lookback` bars prior to the current bar, not the current bar's own
   swing.

---

### Requirement 10: Property-Based and TDD Test Coverage

**User Story:** As a test engineer, I want property-based tests using `hypothesis` and TDD
red-phase tests using `pytest`, so that the scanner is verified against a wide range of
generated inputs before production code is written.

#### Acceptance Criteria

1. THE test suite SHALL include a property-based test (marked `@pytest.mark.unit` and
   `@hypothesis.given(...)`) that verifies: for any synthetically generated OHLCV_Series
   that contains a confirmed bullish OB_Zone at index `i`, there ALWAYS exists at least
   one bar at index `j > i` where `close[j]` exceeds the Swing_High at bar `i`.

2. THE test suite SHALL include a property-based test that verifies the round-trip
   invariant: serialising an OB_Zone to a dict and deserialising back to an OB_Zone
   produces an object equal to the original for any generated OB_Zone.

3. THE test suite SHALL include a property-based test that verifies the volume ranking
   invariant: for any list of generated OB_Zones, after VolumeRanker sorts them, the
   `volume_score` of zone `k` is always greater than or equal to the `volume_score` of
   zone `k+1` (descending order invariant).

4. THE test suite SHALL include a property-based test that verifies the no-lookahead
   invariant: appending an arbitrary candle to any OHLCV_Series SHALL NOT alter the set
   of OB_Zone records previously detected in that series.

5. THE test suite SHALL include a TDD red-phase test (marked `@pytest.mark.tdd`) for each
   acceptance criterion in Requirements 1–9 before the production implementation exists,
   so that all tests initially fail.

6. THE test suite SHALL include PIT (Point-In-Time) tests (marked `@pytest.mark.pit`)
   that verify no zone uses a source timestamp later than the computation timestamp.

7. WHEN the hypothesis `@given` strategy generates OHLCV data, THE test suite SHALL use
   `st.floats(min_value=1.0, max_value=100000.0, allow_nan=False, allow_infinity=False)`
   for price columns and `st.floats(min_value=0.0, max_value=1e10, allow_nan=False)`
   for volume to ensure realistic, non-degenerate inputs.

8. THE test suite SHALL use `@pytest.mark.slow` for any test that runs hypothesis with
   more than 50 examples, and SHALL use `--hypothesis-seed=0` in CI to ensure
   reproducible failures.
