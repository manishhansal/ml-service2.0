# Requirements Document

## Introduction

This document specifies requirements for four new Smart Money Concept (SMC) scanner
modules that extend the `src/scanners/` package of the AlphaForge ML service. Each
scanner detects a distinct institutional price pattern across the NSE F&O universe
(~200+ symbols) and exposes a dedicated `GET /v2/scanners/` REST endpoint.

The four modules are:

- **Scanner 2 — Fair Value Gap** (`src/scanners/fair_value_gap/`,
  `GET /v2/scanners/fair-value-gaps`): identifies 3-candle imbalance zones where price
  moved so fast that an unfilled gap persists between candle 1 and candle 3.
- **Scanner 3 — Supply & Demand Zones** (`src/scanners/supply_demand/`,
  `GET /v2/scanners/supply-demand`): identifies institutional consolidation-departure
  patterns (RBR, DBD, RBD, DBR) that mark supply and demand zones.
- **Scanner 4 — Liquidity Sweep** (`src/scanners/liquidity_sweep/`,
  `GET /v2/scanners/liquidity-sweeps`): detects stop-hunt events where price briefly
  breaks a swing high or low with a wick then closes back on the reversal side.
- **Scanner 5 — Market Structure** (`src/scanners/market_structure/`,
  `GET /v2/scanners/market-structure`): tracks the higher-highs / lower-lows framework
  and emits Break of Structure (BoS) and Change of Character (ChoCh) signals.

All four modules follow the same conventions as `src/scanners/order_block/`:
- `BaseSchema` (`strict=True, frozen=True`) for all Pydantic models.
- `(str, Enum)` pattern for all enumerations.
- `structlog` for all structured logging.
- ATR computed via `.rolling(window=N, min_periods=1)` — never with `center=True`.
- `.model_copy(update={})` for all immutable model mutations.
- `pytest` markers: `unit`, `integration`, `slow`, `pit`, `tdd`.
- `hypothesis` for property-based tests (PBT).
- Authentication handled globally by `api_key_middleware` in `main.py`; zero auth code
  in routers.

Each scanner is implemented TDD-first: every acceptance criterion below is expressed as
a `pytest` test that FAILS before the production implementation exists.

---

## Glossary

### Common Terms

- **OHLCV_Series**: A `pandas.DataFrame` with columns `open`, `high`, `low`, `close`,
  `volume` indexed by a timezone-aware UTC `DatetimeIndex`, one row per candle.
- **ATR14**: The 14-bar Average True Range computed as a trailing rolling mean of true
  range using `.rolling(window=14, min_periods=1)` — never with `center=True`.
- **avg_volume_20**: The 20-bar trailing rolling average of the `volume` column, computed
  using `.rolling(window=20, min_periods=1)`.
- **Timeframe**: The candle aggregation interval — one of `"5m"`, `"15m"`, `"1h"`,
  `"1d"`.
- **F&O_Universe**: The list of NSE stocks eligible for Futures & Options trading
  (~200+ symbols) maintained in the service's symbol registry.
- **BaseSchema**: The project-wide Pydantic base model configured with
  `strict=True, frozen=True`.
- **Swing_High**: The maximum `high` over a configurable trailing window of bars,
  computed without look-ahead (trailing only, no centered window).
- **Swing_Low**: The minimum `low` over a configurable trailing window of bars, computed
  without look-ahead.

### Fair Value Gap (FVG) Terms

- **FVGDetector**: The core detection engine that identifies 3-candle fair value gaps
  from an OHLCV_Series.
- **FVGZoneManager**: The component that tracks the lifecycle state of each FVG_Zone as
  new candles arrive.
- **FVGScorer**: The component that computes `gap_size_score`, `volume_score`, and
  `combined_score` and produces a ranked list of FVG_Zones.
- **FVGScanner**: The top-level orchestrator that runs FVGDetector → FVGZoneManager →
  FVGScorer across the F&O_Universe.
- **FVGScannerAPI**: The FastAPI router exposing `GET /v2/scanners/fair-value-gaps`.
- **FVG_Zone**: A confirmed fair value gap zone carrying `fvg_high`, `fvg_low`,
  `fvg_midpoint`, `direction`, `fvg_timestamp`, `lifecycle_state`, `gap_size_score`,
  `volume_score`, and `combined_score`.
- **Candle_1**: The candle at index `i` in a 3-candle FVG pattern — the pre-impulse
  candle.
- **Candle_2**: The candle at index `i+1` — the impulse candle whose body must exceed
  `impulse_atr_multiplier × ATR14`.
- **Candle_3**: The candle at index `i+2` — the post-impulse candle whose close confirms
  the gap.
- **FVG_High**: Upper boundary of the gap — `low[Candle_3]` for a bullish FVG;
  `low[Candle_1]` for a bearish FVG.
- **FVG_Low**: Lower boundary of the gap — `high[Candle_1]` for a bullish FVG;
  `high[Candle_3]` for a bearish FVG.
- **FVG_Midpoint**: The arithmetic mean of `FVG_High` and `FVG_Low`, used as the
  fill-threshold for lifecycle transitions.

### Supply and Demand Zone Terms

- **SDDetector**: The core detection engine that identifies supply and demand zones from
  OHLCV consolidation-departure pattern sequences.
- **SDZoneManager**: The component that tracks the lifecycle state of each SD_Zone.
- **SDScorer**: The component that computes zone scores and produces a ranked list.
- **SDScanner**: The top-level orchestrator for the supply and demand pipeline.
- **SDScannerAPI**: The FastAPI router exposing `GET /v2/scanners/supply-demand`.
- **SD_Zone**: A confirmed supply or demand zone defined by `zone_high`, `zone_low`,
  `zone_type`, `direction`, `lifecycle_state`, `zone_strength`, `departure_strength`,
  and `combined_score`.
- **Base_Candles**: A contiguous run of 2 to 5 candles where every candle body
  (`|close − open|`) is strictly less than `base_body_threshold × ATR14` at that bar
  (default `base_body_threshold = 0.5`) — the consolidation phase.
- **Departure_Candle**: The candle immediately following the last Base_Candle, whose body
  exceeds `departure_body_threshold × ATR14` (default `departure_body_threshold = 1.5`)
  — the impulsive departure.
- **Entry_Leg_Candle**: The candle immediately preceding the first Base_Candle.
- **Zone_Type**: One of `RBR` (Rally-Base-Rally), `DBD` (Drop-Base-Drop),
  `RBD` (Rally-Base-Drop), `DBR` (Drop-Base-Rally).
- **Zone_Direction**: `SUPPLY` for zone types `RBD` and `DBD`; `DEMAND` for zone types
  `RBR` and `DBR`.

### Liquidity Sweep Terms

- **SweepDetector**: The core detection engine that identifies liquidity sweep candles
  from an OHLCV_Series.
- **SweepZoneManager**: The component that tracks the lifecycle state of each
  Sweep_Signal.
- **SweepScorer**: The component that computes sweep scores and produces a ranked list.
- **SweepScanner**: The top-level orchestrator for the liquidity sweep pipeline.
- **SweepScannerAPI**: The FastAPI router exposing `GET /v2/scanners/liquidity-sweeps`.
- **Sweep_Signal**: A confirmed liquidity sweep event carrying `direction`,
  `swept_level`, `sweep_zone_high`, `sweep_zone_low`, `sweep_timestamp`,
  `lifecycle_state`, `sweep_magnitude`, `volume_ratio`, `equal_levels_count`, and
  `conviction_score`.
- **Liquidity_Pool**: Two or more consecutive Swing_Highs within
  `max(0.001 × close[k], 0.1 × ATR14[k])` of each other — or the equivalent cluster of
  Swing_Lows — marking a price level with clustered stop orders.
- **Swept_Level**: The specific Swing_High or Swing_Low level that the sweep candle's
  wick breached before closing on the reversal side.
- **Sweep_Zone**: The price region `[Swept_Level − 0.5 × ATR14, Swept_Level +
  0.5 × ATR14]` marking the sweep area.

### Market Structure Terms

- **MSDetector**: The core detection engine that classifies swing points and generates
  BoS/ChoCh signals from an OHLCV_Series.
- **MSStateManager**: The component that maintains the Trend_State machine for each
  symbol-timeframe pair.
- **MSScorer**: The component that computes signal conviction scores and produces a
  ranked list of MS_Signals.
- **MSScanner**: The top-level orchestrator for the market structure pipeline.
- **MSScannerAPI**: The FastAPI router exposing `GET /v2/scanners/market-structure`.
- **MS_Signal**: A confirmed BoS or ChoCh event carrying `signal_type`, `broken_level`,
  `close_at_break`, `break_timestamp`, `trend_state`, `is_choch`, `break_magnitude`,
  `swing_count`, `conviction_score`, `invalidated`, and `multi_tf_confluence`.
- **Trend_State**: The current assessed market direction — one of `"UNDEFINED"`,
  `"BULLISH"`, or `"BEARISH"`.
- **Higher_High (HH)**: A confirmed swing high whose value strictly exceeds the
  immediately preceding confirmed swing high.
- **Higher_Low (HL)**: A confirmed swing low whose value strictly exceeds the immediately
  preceding confirmed swing low.
- **Lower_High (LH)**: A confirmed swing high whose value is strictly below the
  immediately preceding confirmed swing high.
- **Lower_Low (LL)**: A confirmed swing low whose value is strictly below the immediately
  preceding confirmed swing low.
- **BoS**: Break of Structure — a close-based breach of the most recent opposite-side
  swing level in the direction of the prevailing Trend_State, indicating continuation.
- **ChoCh**: Change of Character — a close-based breach of the most recent opposite-side
  swing level against the prevailing Trend_State, indicating a potential trend reversal.

---

## Requirements

### Requirement 1: FVG 3-Candle Imbalance Detection

**User Story:** As a quantitative analyst, I want the FVGDetector to identify all valid
3-candle fair value gaps in an OHLCV_Series, so that I can locate price imbalance zones
where institutional orders remain unfilled.

#### Acceptance Criteria

1. WHEN an OHLCV_Series is provided, THE FVGDetector SHALL identify a **bullish FVG**
   anchored at Candle_1 index `i` when `low[i+2] > high[i]` AND `close[i+2] > high[i]`,
   and SHALL set `FVG_High = low[i+2]` and `FVG_Low = high[i]`.

2. WHEN an OHLCV_Series is provided, THE FVGDetector SHALL identify a **bearish FVG**
   anchored at Candle_1 index `i` when `high[i+2] < low[i]` AND `close[i+2] < low[i]`,
   and SHALL set `FVG_High = low[i]` and `FVG_Low = high[i+2]`.

3. THE FVGDetector SHALL compute `FVG_Midpoint = (FVG_High + FVG_Low) / 2` for each
   detected FVG_Zone and store it on the zone object.

4. THE FVGDetector SHALL evaluate every valid consecutive 3-candle window in the series;
   when two qualifying windows share a candle, THE FVGDetector SHALL emit a separate
   FVG_Zone for each qualifying window.

5. WHEN an OHLCV_Series contains fewer than 3 bars, THE FVGDetector SHALL return an
   empty list without raising an exception.

6. IF the OHLCV_Series contains `NaN` in any of `open`, `high`, `low`, `close`, or
   `volume` at candle indices `i`, `i+1`, or `i+2`, THEN THE FVGDetector SHALL skip that
   3-candle window and continue processing the remaining bars.

7. THE FVGDetector SHALL tag each emitted FVG_Zone with the bar timestamp of Candle_3
   as `fvg_timestamp` — the bar at which the zone became observable.

---

### Requirement 2: Impulse Candle Qualification and Gap Size Filter

**User Story:** As a signal engineer, I want only statistically significant gaps confirmed
by a strong impulse candle to reach the active zone list, so that micro-gaps and noise
are excluded from trading signals.

#### Acceptance Criteria

1. THE FVGDetector SHALL reject any 3-candle FVG candidate where
   `|close[i+1] - open[i+1]| < impulse_atr_multiplier × ATR14[i+1]` (default
   `impulse_atr_multiplier = 1.0`), ensuring Candle_2 is a genuine impulse move.

2. THE FVGDetector SHALL reject any 3-candle FVG candidate where
   `(FVG_High - FVG_Low) < min_gap_atr_multiplier × ATR14[i+2]` (default
   `min_gap_atr_multiplier = 0.5`), filtering micro-gaps smaller than half the recent
   average volatility.

3. WHEN a candidate fails the impulse size filter, THE FVGDetector SHALL log the bar
   index, computed body size, ATR14 value, and rejection reason at `DEBUG` level using
   structlog.

4. THE FVGDetector SHALL compute ATR14 at each bar using `.rolling(window=14,
   min_periods=1).mean()` on the true range series without `center=True`, so the ATR at
   index `k` uses only data from bars at indices ≤ `k`.

5. IF the OHLCV_Series contains fewer than 14 bars when a 3-candle window is evaluated,
   THE FVGDetector SHALL compute ATR14 using all available bars (`min_periods=1`) rather
   than requiring exactly 14 bars.

6. THE FVGDetector SHALL accept per-call overrides for `impulse_atr_multiplier` and
   `min_gap_atr_multiplier` via the config object, allowing stricter or looser thresholds
   per timeframe.

---

### Requirement 3: FVG Lifecycle Management

**User Story:** As a risk manager, I want FVG zones to transition through
`UNFILLED → PARTIALLY_FILLED → FILLED` states as price interacts with them, so that I
can distinguish fresh imbalances from already-mitigated gaps.

#### Acceptance Criteria

1. THE FVGZoneManager SHALL create each new confirmed FVG_Zone with
   `lifecycle_state = "UNFILLED"`.

2. WHEN a candle's price range intersects the FVG_Zone (`low ≤ FVG_High` AND
   `high ≥ FVG_Low`) for a zone in `"UNFILLED"` state, THE FVGZoneManager SHALL
   transition the zone to `lifecycle_state = "PARTIALLY_FILLED"`.

3. WHEN a candle closes on the far side of the FVG_Midpoint for a zone in
   `"PARTIALLY_FILLED"` state — specifically `close < FVG_Midpoint` for a bullish zone
   or `close > FVG_Midpoint` for a bearish zone — THE FVGZoneManager SHALL transition
   the zone to `lifecycle_state = "FILLED"`.

4. THE FVGZoneManager SHALL NEVER transition a zone out of `"FILLED"` state; once
   filled, the zone remains permanently filled.

5. IF a zone has been in `"UNFILLED"` state for more than `fvg_expiry_bars` bars
   (default `100`) without any price interaction, THEN THE FVGZoneManager SHALL
   transition the zone to `lifecycle_state = "EXPIRED"` to prevent stale zones from
   polluting results.

6. WHEN the FVGScannerAPI response is generated, THE FVGZoneManager SHALL exclude all
   zones with `lifecycle_state = "FILLED"` or `"EXPIRED"` from the default active signal
   list unless the caller explicitly sets `include_filled = true`.

7. THE FVGZoneManager SHALL record the bar timestamp of each lifecycle transition on the
   FVG_Zone so consumers can audit the fill history.

---

### Requirement 4: FVG Scoring and Ranking

**User Story:** As a portfolio manager, I want FVG zones ranked by a composite score
combining relative gap size and impulse volume, so that the highest-conviction imbalances
appear first in the API response.

#### Acceptance Criteria

1. THE FVGScorer SHALL compute `gap_size_score = (FVG_High - FVG_Low) / ATR14[i+2]` for
   each FVG_Zone, expressing gap width as a multiple of recent volatility.

2. THE FVGScorer SHALL compute `volume_score = volume[i+1] / avg_volume_20[i+1]` for
   each FVG_Zone, measuring the relative volume of Candle_2.

3. THE FVGScorer SHALL compute `combined_score = gap_size_score × volume_score` and use
   it as the primary sort key when ranking FVG_Zones.

4. WHEN the FVGScanner returns a list of active FVG_Zones for a single symbol, THE
   FVGScorer SHALL sort the list in descending order of `combined_score`.

5. WHEN the FVGScanner returns a cross-symbol list of FVG_Zones, THE FVGScorer SHALL
   sort the combined list in descending order of `combined_score` so that the
   highest-conviction zones from any symbol appear first.

6. IF `avg_volume_20` is zero or unavailable for a given bar, THEN THE FVGScorer SHALL
   set `volume_score = 0.0` for that zone and SHALL place it at the end of the sorted
   list.

7. THE FVGScorer SHALL preserve all zone metadata (`symbol`, `direction`, `timeframe`,
   `fvg_high`, `fvg_low`, `fvg_midpoint`, `fvg_timestamp`, `lifecycle_state`) unchanged
   after scoring; only scores and sort order are added or updated.

---

### Requirement 5: FVG Multi-Timeframe Support

**User Story:** As a multi-timeframe analyst, I want to detect FVG zones on 5m, 15m, 1h,
and 1d candle data, so that I can identify imbalances across different holding horizons.

#### Acceptance Criteria

1. THE FVGDetector SHALL accept a `timeframe` parameter with valid values `"5m"`, `"15m"`,
   `"1h"`, and `"1d"` and SHALL process the OHLCV_Series at that timeframe without
   modification to the gap detection logic.

2. WHEN `timeframe = "5m"`, THE FVGZoneManager SHALL use `fvg_expiry_bars = 288` as the
   default expiry (approximately 24 hours at 5-minute resolution).

3. WHEN `timeframe = "1d"`, THE FVGZoneManager SHALL use `fvg_expiry_bars = 20` as the
   default expiry (20 trading days).

4. IF an unsupported timeframe value is provided, THEN THE FVGDetector SHALL raise a
   `ValueError` with a message listing the four supported timeframe values.

5. WHEN the FVGScanner scans a symbol across multiple timeframes, THE FVGScanner SHALL
   tag each FVG_Zone with its source `timeframe` and SHALL NOT merge zones from different
   timeframes.

6. THE FVGDetector SHALL accept per-timeframe config overrides for
   `impulse_atr_multiplier`, `min_gap_atr_multiplier`, and `fvg_expiry_bars` via the
   config object so that filters can be tuned independently per timeframe.

---

### Requirement 6: FVG REST API

**User Story:** As a front-end developer, I want a REST endpoint that returns ranked
active FVG zones, so that the AlphaForge UI can display the strongest imbalance zones
without performing custom filtering.

#### Acceptance Criteria

1. THE FVGScannerAPI SHALL expose a `GET /v2/scanners/fair-value-gaps` endpoint that
   returns a JSON response containing a ranked list of active FVG_Zones.

2. WHEN the request includes query parameter `limit` (integer, 1–200), THE FVGScannerAPI
   SHALL return at most `limit` zones; when `limit` is omitted, THE FVGScannerAPI SHALL
   default to `50`.

3. WHEN the request includes query parameter `direction` with value `"bullish"` or
   `"bearish"`, THE FVGScannerAPI SHALL filter the response to zones matching that
   direction; when `direction` is omitted, THE FVGScannerAPI SHALL return both
   directions.

4. WHEN the request includes query parameter `timeframe` (one of `"5m"`, `"15m"`,
   `"1h"`, `"1d"`), THE FVGScannerAPI SHALL return only zones detected on that
   timeframe; when `timeframe` is omitted, THE FVGScannerAPI SHALL return zones from all
   timeframes.

5. THE FVGScannerAPI response SHALL include the following fields for each FVG_Zone:
   `symbol` (string), `direction` (string `"bullish"` or `"bearish"`), `timeframe`
   (string), `fvg_high` (float), `fvg_low` (float), `fvg_midpoint` (float),
   `fvg_timestamp` (ISO-8601 UTC string), `lifecycle_state` (string), `gap_size_score`
   (float), `volume_score` (float), `combined_score` (float).

6. THE FVGScannerAPI response SHALL include the following top-level fields: `scanned_at`
   (ISO-8601 UTC timestamp of scan completion), `n_symbols_scanned` (integer), `n_zones`
   (integer count of zones returned).

7. IF no active FVG_Zones exist after filtering, THE FVGScannerAPI SHALL return HTTP 200
   with an empty `zones` list and SHALL NOT return HTTP 404.

8. IF the authentication header `X-API-KEY` is missing or invalid, THEN THE
   FVGScannerAPI SHALL return HTTP 401 without a zone payload, consistent with all other
   authenticated endpoints in the service.

9. WHEN the `limit` query parameter is less than 1 or greater than 200, THE
   FVGScannerAPI SHALL return HTTP 422 with a descriptive validation error.

---

### Requirement 7: FVG No-Lookahead Data Integrity and Property-Based Testing

**User Story:** As a backtesting engineer, I want every FVG feature value to depend only
on data at bars ≤ `i+2`, and I want property-based tests using `hypothesis` that verify
correctness across generated inputs before production code exists.

#### Acceptance Criteria

1. THE FVGDetector SHALL compute ATR14 using `.rolling(window=14, min_periods=1)` without
   `center=True`, ensuring the ATR value at bar `k` uses only data from bars at indices
   ≤ `k`.

2. FOR ALL detected FVG_Zones, modifying any candle at index `k > i+2` in the
   OHLCV_Series SHALL NOT change the FVG_Zone record that was detected from bars
   `[i, i+1, i+2]`.

3. THE test suite SHALL include a property-based test (marked `@pytest.mark.unit` and
   `@hypothesis.given`) verifying: for any synthetically generated OHLCV_Series
   containing a valid bullish FVG at index `i`, `low[i+2] > high[i]` always holds AND
   `close[i+2] > high[i]` always holds.

4. THE test suite SHALL include a property-based test verifying the round-trip invariant:
   serialising an FVG_Zone to a dict and deserialising back to an FVG_Zone produces an
   object equal to the original for any generated FVG_Zone.

5. THE test suite SHALL include a property-based test verifying the scoring invariant:
   for any list of generated FVG_Zones, after FVGScorer sorts them,
   `combined_score[k] >= combined_score[k+1]` holds for all consecutive zone pairs.

6. THE test suite SHALL include a property-based test verifying the no-lookahead
   invariant: appending an arbitrary candle to any OHLCV_Series SHALL NOT alter the set
   of FVG_Zone records previously detected in that series.

7. THE test suite SHALL include a TDD red-phase test (marked `@pytest.mark.tdd`) for
   each acceptance criterion in Requirements 1–6 before the production implementation
   exists, so that all tests initially fail.

8. THE test suite SHALL include PIT tests (marked `@pytest.mark.pit`) verifying that no
   FVG_Zone carries a `fvg_timestamp` later than the bar at index `i+2`.

---

### Requirement 8: Supply and Demand Zone Base Detection

**User Story:** As a quantitative analyst, I want the SDDetector to identify the
consolidation base (2–5 small-body candles) within an OHLCV_Series, so that I can mark
the price levels where institutional orders were placed before a departure move.

#### Acceptance Criteria

1. WHEN an OHLCV_Series is provided, THE SDDetector SHALL identify a **Base** as a
   contiguous run of 2 to 5 candles where every candle body size
   (`|close[k] - open[k]|`) is strictly less than
   `base_body_threshold × ATR14[k]` (default `base_body_threshold = 0.5`).

2. WHEN a Base is identified, THE SDDetector SHALL compute
   `zone_high = max(high[k] for k in base_indices)` and
   `zone_low = min(low[k] for k in base_indices)` to define the SD_Zone price bounds.

3. THE SDDetector SHALL require that the Departure_Candle immediately following the last
   Base_Candle has a body size strictly greater than
   `departure_body_threshold × ATR14[departure_index]`
   (default `departure_body_threshold = 1.5`).

4. THE SDDetector SHALL reject any Base where
   `zone_high - zone_low < min_zone_width_multiplier × ATR14[departure_index]`
   (default `min_zone_width_multiplier = 0.3`), filtering zones too narrow to be
   actionable.

5. WHEN an OHLCV_Series contains fewer than 5 bars, THE SDDetector SHALL return an empty
   list without raising an exception.

6. IF the OHLCV_Series contains `NaN` in any OHLC column within the Base_Candles or
   Departure_Candle window, THEN THE SDDetector SHALL skip that pattern window and
   continue processing the remaining series.

7. THE SDDetector SHALL record the bar indices of the Entry_Leg_Candle, all Base_Candles,
   and the Departure_Candle on the SD_Zone so that downstream components and tests can
   verify every input that produced the zone.

---

### Requirement 9: Supply and Demand Zone Pattern Classification

**User Story:** As a trader, I want each detected zone classified as RBR, DBD, RBD, or
DBR, so that I can distinguish demand zones (RBR, DBR) from supply zones (RBD, DBD) and
filter by institutional intent.

#### Acceptance Criteria

1. WHEN the Entry_Leg_Candle (at `base_start_index − 1`) closes higher than it opens AND
   the Departure_Candle closes higher than it opens, THE SDDetector SHALL classify the
   zone as `zone_type = "RBR"` with `direction = "DEMAND"`.

2. WHEN the Entry_Leg_Candle closes lower than it opens AND the Departure_Candle closes
   lower than it opens, THE SDDetector SHALL classify the zone as `zone_type = "DBD"`
   with `direction = "SUPPLY"`.

3. WHEN the Entry_Leg_Candle closes higher than it opens AND the Departure_Candle closes
   lower than it opens, THE SDDetector SHALL classify the zone as `zone_type = "RBD"`
   with `direction = "SUPPLY"`.

4. WHEN the Entry_Leg_Candle closes lower than it opens AND the Departure_Candle closes
   higher than it opens, THE SDDetector SHALL classify the zone as `zone_type = "DBR"`
   with `direction = "DEMAND"`.

5. IF the OHLCV_Series does not contain a bar at index `base_start_index − 1` (i.e., the
   Base begins at index 0), THEN THE SDDetector SHALL skip that Base; a valid
   Entry_Leg_Candle at index `base_start_index − 1` is required for classification.

6. THE SDDetector SHALL record the Departure_Candle's bar timestamp as
   `departure_timestamp` and the Entry_Leg_Candle's bar timestamp as
   `entry_leg_timestamp` on the SD_Zone.

---

### Requirement 10: Supply and Demand Zone Lifecycle Management

**User Story:** As a risk manager, I want S&D zones to transition through
`FRESH → TESTED → USED → BROKEN` states with a degrading strength score, so that I can
prioritise untested institutional zones over weakened ones.

#### Acceptance Criteria

1. THE SDZoneManager SHALL create each new confirmed SD_Zone with
   `lifecycle_state = "FRESH"` and `zone_strength = 1.0`.

2. WHEN a candle's price range enters the SD_Zone (`low ≤ zone_high` AND
   `high ≥ zone_low`) for a `"FRESH"` zone and the candle does NOT close on the far side
   of the zone (`close > zone_low` for a DEMAND zone, `close < zone_high` for a SUPPLY
   zone), THE SDZoneManager SHALL transition the zone to `lifecycle_state = "TESTED"` and
   set `zone_strength = 0.7`.

3. WHEN a `"TESTED"` SD_Zone is entered by a subsequent candle's price range without the
   candle closing on the far side of the zone, THE SDZoneManager SHALL transition the
   zone to `lifecycle_state = "USED"` and set `zone_strength = 0.4`.

4. WHEN a candle closes beyond the far side of the SD_Zone at any lifecycle state —
   specifically `close < zone_low` for a DEMAND zone or `close > zone_high` for a SUPPLY
   zone — THE SDZoneManager SHALL transition the zone to `lifecycle_state = "BROKEN"` and
   set `zone_strength = 0.0`.

5. THE SDZoneManager SHALL NEVER transition a zone out of `"BROKEN"` state; once broken,
   the zone remains permanently broken.

6. WHEN the SDScannerAPI response is generated, THE SDZoneManager SHALL exclude zones
   with `lifecycle_state = "BROKEN"` from the default active signal list unless the
   caller explicitly sets `include_broken = true`.

7. THE SDZoneManager SHALL record the bar timestamp of each lifecycle transition on the
   SD_Zone so consumers can audit the full interaction history.

---

### Requirement 11: Supply and Demand Zone Scoring and Ranking

**User Story:** As a portfolio manager, I want S&D zones ranked by departure strength and
zone freshness, so that zones with the highest institutional conviction and least
mitigation appear first.

#### Acceptance Criteria

1. THE SDScorer SHALL compute
   `departure_strength = |close[departure_index] - open[departure_index]| / ATR14[departure_index]`
   for each SD_Zone, expressing the departure move as a multiple of recent volatility.

2. THE SDScorer SHALL compute `zone_freshness` as the numeric strength value of the
   zone's current `lifecycle_state`: `1.0` for `"FRESH"`, `0.7` for `"TESTED"`, `0.4`
   for `"USED"`, and `0.0` for `"BROKEN"`.

3. THE SDScorer SHALL compute `combined_score = departure_strength × zone_freshness` and
   use it as the primary sort key when ranking SD_Zones.

4. WHEN the SDScanner returns a list of active SD_Zones for a single symbol, THE
   SDScorer SHALL sort the list in descending order of `combined_score`.

5. WHEN the SDScanner returns a cross-symbol list of SD_Zones, THE SDScorer SHALL sort
   the combined list in descending order of `combined_score` so that the
   highest-conviction zones from any symbol appear first.

6. IF `ATR14` is zero or unavailable at the departure index, THEN THE SDScorer SHALL set
   `departure_strength = 0.0` for that zone and SHALL place it at the end of the sorted
   list.

7. THE SDScorer SHALL preserve all zone metadata (`symbol`, `direction`, `zone_type`,
   `timeframe`, `zone_high`, `zone_low`, `entry_leg_timestamp`, `departure_timestamp`,
   `lifecycle_state`) unchanged after scoring; only scores and sort order are added or
   updated.

---

### Requirement 12: Supply and Demand Multi-Timeframe Support

**User Story:** As a multi-timeframe analyst, I want to detect supply and demand zones on
5m, 15m, 1h, and 1d candle data, so that I can identify institutional zones at different
temporal scales.

#### Acceptance Criteria

1. THE SDDetector SHALL accept a `timeframe` parameter with valid values `"5m"`, `"15m"`,
   `"1h"`, and `"1d"` and SHALL process the OHLCV_Series at that timeframe without
   modification to the zone detection logic.

2. THE SDDetector SHALL accept per-timeframe config overrides for
   `base_body_threshold`, `departure_body_threshold`, and `min_zone_width_multiplier` via
   the config object, allowing independent tuning per timeframe.

3. IF an unsupported timeframe value is provided, THEN THE SDDetector SHALL raise a
   `ValueError` with a message listing the four supported timeframe values.

4. WHEN the SDScanner scans a symbol across multiple timeframes, THE SDScanner SHALL tag
   each SD_Zone with its source `timeframe` and SHALL NOT merge zones from different
   timeframes.

5. WHEN a DEMAND SD_Zone is confirmed on both `1h` and `1d` timeframes with zone
   midpoints within `1%` of each other in price, THE SDScanner SHALL set
   `multi_tf_confluence = True` on both zones to indicate higher conviction.

---

### Requirement 13: Supply and Demand REST API

**User Story:** As a front-end developer, I want a REST endpoint returning ranked S&D
zones, so that the AlphaForge UI can display institutional supply and demand zones without
custom filtering logic.

#### Acceptance Criteria

1. THE SDScannerAPI SHALL expose a `GET /v2/scanners/supply-demand` endpoint that returns
   a JSON response containing a ranked list of active SD_Zones.

2. WHEN the request includes query parameter `limit` (integer, 1–200), THE SDScannerAPI
   SHALL return at most `limit` zones; when `limit` is omitted, THE SDScannerAPI SHALL
   default to `50`.

3. WHEN the request includes query parameter `direction` with value `"supply"` or
   `"demand"`, THE SDScannerAPI SHALL filter results to zones matching that direction.

4. WHEN the request includes query parameter `zone_type` with a value of `"RBR"`,
   `"DBD"`, `"RBD"`, or `"DBR"`, THE SDScannerAPI SHALL filter results to zones of that
   specific zone type.

5. WHEN the request includes query parameter `timeframe` (one of `"5m"`, `"15m"`,
   `"1h"`, `"1d"`), THE SDScannerAPI SHALL return only zones detected on that timeframe.

6. THE SDScannerAPI response SHALL include the following fields for each SD_Zone:
   `symbol` (string), `direction` (string `"supply"` or `"demand"`), `zone_type`
   (string), `timeframe` (string), `zone_high` (float), `zone_low` (float),
   `entry_leg_timestamp` (ISO-8601 UTC), `departure_timestamp` (ISO-8601 UTC),
   `lifecycle_state` (string), `zone_strength` (float), `departure_strength` (float),
   `combined_score` (float), `multi_tf_confluence` (boolean).

7. THE SDScannerAPI response SHALL include the following top-level fields: `scanned_at`
   (ISO-8601 UTC), `n_symbols_scanned` (integer), `n_zones` (integer count of zones
   returned).

8. IF no active SD_Zones exist after filtering, THE SDScannerAPI SHALL return HTTP 200
   with an empty `zones` list and SHALL NOT return HTTP 404.

9. WHEN the `limit` query parameter is less than 1 or greater than 200, THE SDScannerAPI
   SHALL return HTTP 422 with a descriptive validation error.

---

### Requirement 14: S&D No-Lookahead Data Integrity and Property-Based Testing

**User Story:** As a backtesting engineer, I want all S&D zone features to use only data
available at or before the departure bar, and I want property-based tests using
`hypothesis` that verify correctness across generated inputs before production code
exists.

#### Acceptance Criteria

1. THE SDDetector SHALL compute ATR14 using `.rolling(window=14, min_periods=1)` without
   `center=True`, ensuring the ATR at bar `k` uses only data from bars at indices ≤ `k`.

2. FOR ALL detected SD_Zones, modifying any candle at index `k > departure_index` in the
   OHLCV_Series SHALL NOT change any SD_Zone detected using only data up to and including
   `departure_index`.

3. THE test suite SHALL include a property-based test (marked `@pytest.mark.unit` and
   `@hypothesis.given`) verifying: for any synthetically generated OHLCV_Series
   containing a valid Base at indices `[b, b+1, ..., b+n]`, every candle body in the
   Base satisfies `|close[k] - open[k]| < 0.5 × ATR14[k]` for all `k` in the Base.

4. THE test suite SHALL include a property-based test verifying the round-trip invariant:
   serialising an SD_Zone to a dict and deserialising back to an SD_Zone produces an
   object equal to the original for any generated SD_Zone.

5. THE test suite SHALL include a property-based test verifying the scoring invariant:
   for any list of generated SD_Zones, after SDScorer sorts them,
   `combined_score[k] >= combined_score[k+1]` holds for all consecutive zone pairs.

6. THE test suite SHALL include a property-based test verifying the lifecycle
   monotonicity invariant: the `zone_strength` value of any SD_Zone can only decrease or
   remain equal across lifecycle transitions; `zone_strength` SHALL NEVER increase.

7. THE test suite SHALL include a TDD red-phase test (marked `@pytest.mark.tdd`) for
   each acceptance criterion in Requirements 8–13 before the production implementation
   exists, so that all tests initially fail.

8. THE test suite SHALL include PIT tests (marked `@pytest.mark.pit`) verifying that no
   SD_Zone carries a `departure_timestamp` later than the Departure_Candle's actual bar
   timestamp.

---

### Requirement 15: Swing Level Identification and Liquidity Pool Detection

**User Story:** As a quantitative analyst, I want the SweepDetector to identify swing
highs, swing lows, and clusters of equal levels (liquidity pools), so that I can locate
where stop losses are likely to accumulate before a sweep event.

#### Acceptance Criteria

1. WHEN an OHLCV_Series is provided, THE SweepDetector SHALL compute a trailing
   Swing_High at bar `k` as `max(high[k - swing_lookback + 1 : k + 1])` using only data
   at indices ≤ `k` (default `swing_lookback = 10`).

2. WHEN an OHLCV_Series is provided, THE SweepDetector SHALL compute a trailing
   Swing_Low at bar `k` as `min(low[k - swing_lookback + 1 : k + 1])` using only data
   at indices ≤ `k`.

3. THE SweepDetector SHALL identify a Liquidity_Pool of **highs** when two or more
   consecutive Swing_High values are within `max(0.001 × close[k], 0.1 × ATR14[k])` of
   each other and SHALL record the count of clustered levels as `equal_levels_count`.

4. THE SweepDetector SHALL identify a Liquidity_Pool of **lows** when two or more
   consecutive Swing_Low values are within `max(0.001 × close[k], 0.1 × ATR14[k])` of
   each other and SHALL record the count of clustered levels as `equal_levels_count`.

5. WHEN an OHLCV_Series contains fewer than `swing_lookback + 1` bars, THE SweepDetector
   SHALL return an empty list without raising an exception.

6. IF the OHLCV_Series contains `NaN` in `high`, `low`, or `close` at bar `k`, THEN THE
   SweepDetector SHALL skip bar `k` when computing Swing_High and Swing_Low and continue
   with the remaining bars.

---

### Requirement 16: Liquidity Sweep Candle Detection

**User Story:** As a trader, I want the SweepDetector to identify candles that break a
swing level with a wick and immediately close on the reversal side, so that I can catch
institutional stop-hunt reversals before price resumes its true direction.

#### Acceptance Criteria

1. WHEN a candle at bar `k` has `low[k] < Swing_Low[k-1]` (wick breaks below the prior
   Swing_Low) AND `close[k] > Swing_Low[k-1]` (candle closes above the Swing_Low), THE
   SweepDetector SHALL emit a **bullish Sweep_Signal** with
   `swept_level = Swing_Low[k-1]`.

2. WHEN a candle at bar `k` has `high[k] > Swing_High[k-1]` (wick breaks above the
   prior Swing_High) AND `close[k] < Swing_High[k-1]` (candle closes below the
   Swing_High), THE SweepDetector SHALL emit a **bearish Sweep_Signal** with
   `swept_level = Swing_High[k-1]`.

3. THE SweepDetector SHALL require the sweep candle's
   `volume[k] >= volume_confirmation_multiplier × avg_volume_20[k]`
   (default `volume_confirmation_multiplier = 1.5`); candles below this threshold SHALL
   NOT generate a Sweep_Signal.

4. THE SweepDetector SHALL compute `sweep_zone_high = swept_level + 0.5 × ATR14[k]` and
   `sweep_zone_low = swept_level - 0.5 × ATR14[k]` for each confirmed Sweep_Signal.

5. WHEN `require_reversal_confirmation = True` (configurable, default `True`), THE
   SweepDetector SHALL require candle `k+1` to close further in the reversal direction
   — `close[k+1] > close[k]` for a bullish sweep or `close[k+1] < close[k]` for a
   bearish sweep — before emitting the Sweep_Signal.

6. IF the OHLCV_Series contains `NaN` in `high`, `low`, `close`, or `volume` at bar `k`,
   THEN THE SweepDetector SHALL skip bar `k` and continue processing the remaining bars.

---

### Requirement 17: Sweep Lifecycle Management

**User Story:** As a risk manager, I want each Sweep_Signal to transition through
`ACTIVE → CONFIRMED → INVALIDATED` states, so that unconfirmed sweeps are clearly
distinguished from validated reversals and from failed stop-hunt attempts.

#### Acceptance Criteria

1. THE SweepZoneManager SHALL create each new Sweep_Signal with
   `lifecycle_state = "ACTIVE"` when `require_reversal_confirmation = False`, or
   immediately upon detection before the confirmation candle is evaluated when
   `require_reversal_confirmation = True`.

2. WHEN the candle at index `k+1` closes further in the reversal direction (as defined in
   Requirement 16, AC5), THE SweepZoneManager SHALL transition the Sweep_Signal to
   `lifecycle_state = "CONFIRMED"`.

3. WHEN a subsequent candle closes through the Swept_Level in the direction of the
   original sweep — `close < swept_level` for a bullish sweep or
   `close > swept_level` for a bearish sweep — THE SweepZoneManager SHALL transition the
   Sweep_Signal to `lifecycle_state = "INVALIDATED"`.

4. THE SweepZoneManager SHALL NEVER transition a Sweep_Signal out of `"INVALIDATED"`
   state; once invalidated, the signal remains permanently invalidated.

5. WHEN the SweepScannerAPI response is generated, THE SweepZoneManager SHALL include
   only Sweep_Signals with `lifecycle_state = "CONFIRMED"` or `"ACTIVE"` in the default
   response; `"INVALIDATED"` signals are excluded unless the caller sets
   `include_invalidated = true`.

6. THE SweepZoneManager SHALL record the bar timestamp of each lifecycle transition on
   the Sweep_Signal so consumers can audit the full history of any signal.

---

### Requirement 18: Sweep Scoring and Ranking

**User Story:** As a portfolio manager, I want sweep signals ranked by sweep magnitude,
volume ratio, and the count of equal levels swept, so that the highest-conviction
reversal opportunities appear first.

#### Acceptance Criteria

1. THE SweepScorer SHALL compute `sweep_magnitude = abs(wick_extreme - swept_level) /
   ATR14[k]`, where `wick_extreme = low[k]` for a bullish sweep and
   `wick_extreme = high[k]` for a bearish sweep.

2. THE SweepScorer SHALL compute `volume_ratio = volume[k] / avg_volume_20[k]` for the
   sweep candle at bar `k`.

3. THE SweepScorer SHALL compute
   `conviction_score = sweep_magnitude × volume_ratio × equal_levels_count` and use it
   as the primary sort key when ranking Sweep_Signals.

4. WHEN the SweepScanner returns a list of Sweep_Signals for a single symbol, THE
   SweepScorer SHALL sort the list in descending order of `conviction_score`.

5. WHEN the SweepScanner returns a cross-symbol list of Sweep_Signals, THE SweepScorer
   SHALL sort the combined list in descending order of `conviction_score` so that the
   highest-conviction signals from any symbol appear first.

6. IF `avg_volume_20` is zero or unavailable for a given bar, THEN THE SweepScorer SHALL
   set `volume_ratio = 0.0` for that signal and SHALL place it at the end of the sorted
   list.

7. THE SweepScorer SHALL preserve all signal metadata (`symbol`, `direction`, `timeframe`,
   `swept_level`, `sweep_zone_high`, `sweep_zone_low`, `sweep_timestamp`,
   `lifecycle_state`, `equal_levels_count`) unchanged after scoring.

---

### Requirement 19: Liquidity Sweep Multi-Timeframe Support

**User Story:** As a multi-timeframe analyst, I want liquidity sweeps detected on 5m,
15m, 1h, and 1d candle data, so that I can identify stop-hunt reversals at different
trading scales.

#### Acceptance Criteria

1. THE SweepDetector SHALL accept a `timeframe` parameter with valid values `"5m"`,
   `"15m"`, `"1h"`, and `"1d"` and SHALL process the OHLCV_Series at that timeframe
   without modification to the sweep detection logic.

2. THE SweepDetector SHALL accept per-timeframe config overrides for `swing_lookback` and
   `volume_confirmation_multiplier` via the config object.

3. IF an unsupported timeframe value is provided, THEN THE SweepDetector SHALL raise a
   `ValueError` with a message listing the four supported timeframe values.

4. WHEN the SweepScanner scans a symbol across multiple timeframes, THE SweepScanner
   SHALL tag each Sweep_Signal with its source `timeframe` and SHALL NOT merge signals
   from different timeframes.

5. WHEN a bullish Sweep_Signal is confirmed on both `1h` and `1d` timeframes for the
   same symbol with swept levels within `0.5 × ATR14` of each other, THE SweepScanner
   SHALL set `multi_tf_confluence = True` on both signals.

---

### Requirement 20: Liquidity Sweep REST API

**User Story:** As a front-end developer, I want a REST endpoint returning ranked
liquidity sweep signals, so that the AlphaForge UI can surface the strongest stop-hunt
reversal opportunities without custom filtering.

#### Acceptance Criteria

1. THE SweepScannerAPI SHALL expose a `GET /v2/scanners/liquidity-sweeps` endpoint that
   returns a JSON response containing a ranked list of Sweep_Signals.

2. WHEN the request includes query parameter `limit` (integer, 1–200), THE
   SweepScannerAPI SHALL return at most `limit` signals; when `limit` is omitted, THE
   SweepScannerAPI SHALL default to `50`.

3. WHEN the request includes query parameter `direction` with value `"bullish"` or
   `"bearish"`, THE SweepScannerAPI SHALL filter results to signals of that direction.

4. WHEN the request includes query parameter `timeframe` (one of `"5m"`, `"15m"`, `"1h"`,
   `"1d"`), THE SweepScannerAPI SHALL return only signals detected on that timeframe.

5. THE SweepScannerAPI response SHALL include the following fields for each Sweep_Signal:
   `symbol` (string), `direction` (string `"bullish"` or `"bearish"`), `timeframe`
   (string), `swept_level` (float), `sweep_zone_high` (float), `sweep_zone_low` (float),
   `sweep_timestamp` (ISO-8601 UTC), `lifecycle_state` (string), `sweep_magnitude`
   (float), `volume_ratio` (float), `equal_levels_count` (integer), `conviction_score`
   (float), `multi_tf_confluence` (boolean).

6. THE SweepScannerAPI response SHALL include the following top-level fields:
   `scanned_at` (ISO-8601 UTC), `n_symbols_scanned` (integer), `n_signals` (integer
   count of signals returned).

7. IF no Sweep_Signals exist after filtering, THE SweepScannerAPI SHALL return HTTP 200
   with an empty `signals` list and SHALL NOT return HTTP 404.

8. WHEN the `limit` query parameter is less than 1 or greater than 200, THE
   SweepScannerAPI SHALL return HTTP 422 with a descriptive validation error.

---

### Requirement 21: Sweep No-Lookahead Data Integrity and Property-Based Testing

**User Story:** As a backtesting engineer, I want all sweep detection computations to use
only backward-looking data, and I want property-based tests using `hypothesis` that
verify correctness across generated inputs before production code exists.

#### Acceptance Criteria

1. THE SweepDetector SHALL compute Swing_High and Swing_Low using trailing rolling
   windows exclusively; centered rolling windows are prohibited.

2. THE SweepDetector SHALL compute ATR14 and avg_volume_20 using
   `.rolling(window=N, min_periods=1)` without `center=True`, ensuring all values at bar
   `k` use only data from bars at indices ≤ `k`.

3. FOR ALL detected Sweep_Signals, modifying any candle at index `k > sweep_bar_index`
   in the OHLCV_Series SHALL NOT change the Sweep_Signal record detected at
   `sweep_bar_index`.

4. THE test suite SHALL include a property-based test (marked `@pytest.mark.unit` and
   `@hypothesis.given`) verifying: for any OHLCV_Series containing a confirmed bullish
   Sweep_Signal at bar `k`, `low[k] < Swing_Low[k-1]` AND `close[k] > Swing_Low[k-1]`
   always holds.

5. THE test suite SHALL include a property-based test verifying the round-trip invariant:
   serialising a Sweep_Signal to a dict and deserialising back to a Sweep_Signal
   produces an object equal to the original for any generated Sweep_Signal.

6. THE test suite SHALL include a property-based test verifying the scoring invariant:
   for any list of generated Sweep_Signals, after SweepScorer sorts them,
   `conviction_score[k] >= conviction_score[k+1]` holds for all consecutive pairs.

7. THE test suite SHALL include a TDD red-phase test (marked `@pytest.mark.tdd`) for
   each acceptance criterion in Requirements 15–20 before the production implementation
   exists, so that all tests initially fail.

8. THE test suite SHALL include PIT tests (marked `@pytest.mark.pit`) verifying that no
   Sweep_Signal carries a `sweep_timestamp` later than the sweep candle's actual bar
   timestamp at index `k`.

---

### Requirement 22: Swing Point Classification

**User Story:** As a quantitative analyst, I want the MSDetector to classify each new
confirmed swing point as HH, HL, LH, or LL relative to the previous confirmed swing, so
that I can track the market's structural progression bar-by-bar.

#### Acceptance Criteria

1. WHEN the MSDetector identifies a new swing high at bar `k` (where
   `high[k] == max(high[k - swing_lookback + 1 : k + 1])`) whose value strictly exceeds
   the most recently confirmed swing high, THE MSDetector SHALL classify it as `HH`.

2. WHEN the MSDetector identifies a new swing high at bar `k` whose value is strictly
   less than the most recently confirmed swing high, THE MSDetector SHALL classify it as
   `LH`.

3. WHEN the MSDetector identifies a new swing low at bar `k` (where
   `low[k] == min(low[k - swing_lookback + 1 : k + 1])`) whose value is strictly less
   than the most recently confirmed swing low, THE MSDetector SHALL classify it as `LL`.

4. WHEN the MSDetector identifies a new swing low at bar `k` whose value strictly exceeds
   the most recently confirmed swing low, THE MSDetector SHALL classify it as `HL`.

5. IF no prior confirmed swing high exists in the series (i.e., the first detected swing
   high), THE MSDetector SHALL record it as the reference swing high without assigning
   an HH or LH classification.

6. THE MSDetector SHALL compute Swing_High and Swing_Low using trailing rolling windows
   exclusively; no centered window may be used.

7. THE MSDetector SHALL record the bar index and price level of every confirmed swing
   high and swing low, preserving the full sequence for BoS and ChoCh detection in
   Requirements 23 and 24.

---

### Requirement 23: Break of Structure Detection

**User Story:** As a trend trader, I want BoS signals generated when price closes beyond
the most recent swing level in the direction of the prevailing Trend_State, so that I
receive close-confirmed trend continuation signals without false triggers from wick
spikes.

#### Acceptance Criteria

1. WHEN `Trend_State = "BULLISH"` and a candle at bar `k` has
   `close[k] > most_recent_swing_high`, THE MSDetector SHALL emit a `BOS_BULLISH`
   MS_Signal with `broken_level = most_recent_swing_high` and `is_choch = False`.

2. WHEN `Trend_State = "BEARISH"` and a candle at bar `k` has
   `close[k] < most_recent_swing_low`, THE MSDetector SHALL emit a `BOS_BEARISH`
   MS_Signal with `broken_level = most_recent_swing_low` and `is_choch = False`.

3. IF a candle's `high` or `low` wick crosses the swing level but `close[k]` does NOT
   cross the level, THEN THE MSDetector SHALL NOT emit a BoS signal for that bar.

4. WHEN a BoS MS_Signal is emitted, THE MSDetector SHALL record `close_at_break =
   close[k]`, `broken_level`, and `break_timestamp` (the bar's UTC timestamp) on the
   MS_Signal.

5. THE MSDetector SHALL consume each swing level at most once as a BoS trigger; once a
   level has generated a BoS signal, THE MSDetector SHALL NOT emit another BoS for the
   same level on a subsequent bar.

6. WHEN `Trend_State = "UNDEFINED"`, THE MSDetector SHALL NOT emit any BoS signals;
   BoS detection requires an established Trend_State of `"BULLISH"` or `"BEARISH"`.

---

### Requirement 24: Change of Character Detection

**User Story:** As a reversal trader, I want ChoCh signals generated when price closes
beyond the most recent swing level against the prevailing Trend_State, so that I receive
early close-confirmed reversal signals that score 1.5× higher than BoS signals.

#### Acceptance Criteria

1. WHEN `Trend_State = "BEARISH"` and a candle at bar `k` has
   `close[k] > most_recent_swing_high`, THE MSDetector SHALL emit a `CHOCH_BULLISH`
   MS_Signal with `broken_level = most_recent_swing_high` and `is_choch = True`.

2. WHEN `Trend_State = "BULLISH"` and a candle at bar `k` has
   `close[k] < most_recent_swing_low`, THE MSDetector SHALL emit a `CHOCH_BEARISH`
   MS_Signal with `broken_level = most_recent_swing_low` and `is_choch = True`.

3. IF a candle's wick crosses the opposite-trend swing level without `close[k]` crossing
   it, THEN THE MSDetector SHALL NOT emit a ChoCh signal for that bar.

4. WHEN a `CHOCH_BULLISH` MS_Signal is emitted, THE MSStateManager SHALL transition
   `Trend_State` from `"BEARISH"` to `"BULLISH"` immediately.

5. WHEN a `CHOCH_BEARISH` MS_Signal is emitted, THE MSStateManager SHALL transition
   `Trend_State` from `"BULLISH"` to `"BEARISH"` immediately.

6. IF a subsequent candle closes below the `broken_level` of a `CHOCH_BULLISH` signal,
   THEN THE MSStateManager SHALL set `invalidated = True` on that MS_Signal without
   reverting the current `Trend_State`.

7. WHEN a ChoCh MS_Signal is emitted, THE MSDetector SHALL record `close_at_break =
   close[k]`, `broken_level`, and `break_timestamp` on the MS_Signal.

---

### Requirement 25: Trend State Machine

**User Story:** As a systems engineer, I want a deterministic state machine that tracks
the market Trend_State for each symbol-timeframe pair, so that BoS and ChoCh signals are
always generated from the correct structural context.

#### Acceptance Criteria

1. THE MSStateManager SHALL initialise `Trend_State = "UNDEFINED"` for each
   symbol-timeframe pair when the OHLCV_Series contains no confirmed swing points yet.

2. WHEN the first HH-HL sequence is confirmed (a swing high that is an HH followed by a
   swing low that is an HL), THE MSStateManager SHALL transition `Trend_State` from
   `"UNDEFINED"` to `"BULLISH"`.

3. WHEN the first LL-LH sequence is confirmed (a swing low that is a LL followed by a
   swing high that is a LH), THE MSStateManager SHALL transition `Trend_State` from
   `"UNDEFINED"` to `"BEARISH"`.

4. WHEN a `CHOCH_BULLISH` signal is confirmed, THE MSStateManager SHALL set
   `Trend_State = "BULLISH"` regardless of its prior value.

5. WHEN a `CHOCH_BEARISH` signal is confirmed, THE MSStateManager SHALL set
   `Trend_State = "BEARISH"` regardless of its prior value.

6. WHEN a `BOS_BULLISH` or `BOS_BEARISH` signal is emitted, THE MSStateManager SHALL
   leave `Trend_State` unchanged; BoS confirms but does not alter the trend direction.

7. THE MSStateManager SHALL store the complete sequence of confirmed swing points per
   symbol-timeframe pair so that the Trend_State is fully reproducible from the
   OHLCV_Series alone without external state.

---

### Requirement 26: Market Structure Scoring and Ranking

**User Story:** As a portfolio manager, I want MS signals ranked by break magnitude,
swing count, and a 1.5× multiplier for ChoCh signals, so that the highest-conviction
structural events appear first in the API response.

#### Acceptance Criteria

1. THE MSScorer SHALL compute
   `break_magnitude = abs(close_at_break - broken_level) / ATR14[k]` for each
   MS_Signal, expressing the conviction of the break as a multiple of recent volatility.

2. THE MSScorer SHALL compute `swing_count` as the number of confirmed alternating swing
   points (HH-HL or LL-LH) in the current trend sequence preceding the signal.

3. THE MSScorer SHALL compute
   `conviction_score = break_magnitude × swing_count × (1.5 if is_choch else 1.0)` and
   use it as the primary sort key when ranking MS_Signals.

4. WHEN the MSScanner returns a list of MS_Signals for a single symbol, THE MSScorer
   SHALL sort the list in descending order of `conviction_score`.

5. WHEN the MSScanner returns a cross-symbol list of MS_Signals, THE MSScorer SHALL sort
   the combined list in descending order of `conviction_score` so that the
   highest-conviction signals from any symbol appear first.

6. IF `ATR14` is zero or unavailable at bar `k`, THEN THE MSScorer SHALL set
   `break_magnitude = 0.0` for that signal and SHALL place it at the end of the sorted
   list.

7. THE MSScorer SHALL preserve all signal metadata (`symbol`, `signal_type`, `timeframe`,
   `broken_level`, `close_at_break`, `break_timestamp`, `trend_state`, `is_choch`,
   `invalidated`) unchanged after scoring; only scores and sort order are added or
   updated.

---

### Requirement 27: Market Structure Multi-Timeframe Support

**User Story:** As a multi-timeframe analyst, I want to detect BoS and ChoCh signals on
5m, 15m, 1h, and 1d candle data, so that I can confirm structural breaks across multiple
time horizons before committing to a position.

#### Acceptance Criteria

1. THE MSDetector SHALL accept a `timeframe` parameter with valid values `"5m"`, `"15m"`,
   `"1h"`, and `"1d"` and SHALL process the OHLCV_Series at that timeframe without
   modification to the signal detection logic.

2. THE MSDetector SHALL accept a per-timeframe config override for `swing_lookback` via
   the config object.

3. IF an unsupported timeframe value is provided, THEN THE MSDetector SHALL raise a
   `ValueError` with a message listing the four supported timeframe values.

4. WHEN the MSScanner scans a symbol across multiple timeframes, THE MSScanner SHALL tag
   each MS_Signal with its source `timeframe` and SHALL maintain separate Trend_States
   per timeframe; states from different timeframes SHALL NOT influence each other.

5. WHEN a `CHOCH_BULLISH` MS_Signal is confirmed on both `1h` and `1d` timeframes for
   the same symbol within 3 bars of each other (measured on their respective timeframes),
   THE MSScanner SHALL set `multi_tf_confluence = True` on both signals.

---

### Requirement 28: Market Structure REST API

**User Story:** As a front-end developer, I want a REST endpoint returning ranked BoS and
ChoCh signals, so that the AlphaForge UI can display the most significant structural
market events without custom filtering logic.

#### Acceptance Criteria

1. THE MSScannerAPI SHALL expose a `GET /v2/scanners/market-structure` endpoint that
   returns a JSON response containing a ranked list of MS_Signals.

2. WHEN the request includes query parameter `limit` (integer, 1–200), THE MSScannerAPI
   SHALL return at most `limit` signals; when `limit` is omitted, THE MSScannerAPI SHALL
   default to `50`.

3. WHEN the request includes query parameter `signal_type` with a value of
   `"BOS_BULLISH"`, `"BOS_BEARISH"`, `"CHOCH_BULLISH"`, or `"CHOCH_BEARISH"`, THE
   MSScannerAPI SHALL filter results to signals of that type.

4. WHEN the request includes query parameter `timeframe` (one of `"5m"`, `"15m"`, `"1h"`,
   `"1d"`), THE MSScannerAPI SHALL return only signals detected on that timeframe.

5. THE MSScannerAPI response SHALL include the following fields for each MS_Signal:
   `symbol` (string), `signal_type` (string), `timeframe` (string), `broken_level`
   (float), `close_at_break` (float), `break_timestamp` (ISO-8601 UTC), `trend_state`
   (string), `is_choch` (boolean), `break_magnitude` (float), `swing_count` (integer),
   `conviction_score` (float), `invalidated` (boolean), `multi_tf_confluence` (boolean).

6. THE MSScannerAPI response SHALL include the following top-level fields: `scanned_at`
   (ISO-8601 UTC), `n_symbols_scanned` (integer), `n_signals` (integer count of signals
   returned).

7. IF no MS_Signals exist after filtering, THE MSScannerAPI SHALL return HTTP 200 with an
   empty `signals` list and SHALL NOT return HTTP 404.

8. IF the authentication header `X-API-KEY` is missing or invalid, THEN THE MSScannerAPI
   SHALL return HTTP 401 without a signal payload, consistent with all other
   authenticated endpoints in the service.

9. WHEN the `limit` query parameter is less than 1 or greater than 200, THE MSScannerAPI
   SHALL return HTTP 422 with a descriptive validation error.

---

### Requirement 29: Market Structure No-Lookahead Data Integrity and Property-Based Testing

**User Story:** As a backtesting engineer, I want all market structure computations to
use only backward-looking data, and I want property-based tests using `hypothesis` that
verify correctness across generated inputs before production code exists.

#### Acceptance Criteria

1. THE MSDetector SHALL compute Swing_High and Swing_Low using trailing rolling windows
   exclusively; centered rolling windows are prohibited.

2. THE MSDetector SHALL compute ATR14 using `.rolling(window=14, min_periods=1)` without
   `center=True`, ensuring values at bar `k` use only data from bars at indices ≤ `k`.

3. FOR ALL detected MS_Signals, modifying any candle at index `k > signal_bar_index` in
   the OHLCV_Series SHALL NOT change the MS_Signal record detected at `signal_bar_index`.

4. THE test suite SHALL include a property-based test (marked `@pytest.mark.unit` and
   `@hypothesis.given`) verifying: for any OHLCV_Series containing a confirmed
   `BOS_BULLISH` at bar `k`, `close[k] > most_recent_swing_high` always holds and no
   data beyond bar `k` was used to produce the signal.

5. THE test suite SHALL include a property-based test verifying the round-trip invariant:
   serialising an MS_Signal to a dict and deserialising back to an MS_Signal produces an
   object equal to the original for any generated MS_Signal.

6. THE test suite SHALL include a property-based test verifying the scoring invariant:
   for any list of generated MS_Signals, after MSScorer sorts them,
   `conviction_score[k] >= conviction_score[k+1]` holds for all consecutive pairs.

7. THE test suite SHALL include a property-based test verifying the Trend_State machine
   invariant: for any OHLCV_Series, `Trend_State` transitions ONLY on ChoCh events;
   BoS events SHALL NEVER change the `Trend_State`.

8. THE test suite SHALL include a TDD red-phase test (marked `@pytest.mark.tdd`) for
   each acceptance criterion in Requirements 22–28 before the production implementation
   exists, so that all tests initially fail.

9. THE test suite SHALL include PIT tests (marked `@pytest.mark.pit`) verifying that no
   MS_Signal carries a `break_timestamp` later than the bar at index `signal_bar_index`'s
   actual UTC timestamp.
