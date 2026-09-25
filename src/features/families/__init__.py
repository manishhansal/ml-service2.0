"""
Feature Families — Phase 3D.

Each sub-module implements one economic feature family.
All implementations are talib-free (pure numpy/pandas) so they run
in test environments without TA-Lib installed.

talib-backed variants remain in features/technical.py and are imported
by features/engineer.py for production use.

PIT invariant enforced in every family:
    - Only trailing rolling windows (never center=True)
    - No shift(-N) in any feature computation path
    - Insufficient lookback → NaN with INSUFFICIENT_HISTORY status
    - Missing source data → NaN with DATA_UNAVAILABLE status
"""
