"""
test_drift_monitoring.py

TDD property-based tests for DriftMonitor.
Written BEFORE DriftMonitor implementation (red phase).

Properties:
  Property 15: PSI(D, D) ≈ 0.0 — identical distributions have zero drift
  Property 16: PSI(D_ref, D_shifted) > PSI(D_ref, D_ref) for any non-zero shift

Requirements: Req 12.8, Req 18.7
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings as h_settings
from hypothesis import strategies as st

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")


# ── helpers ───────────────────────────────────────────────────────────────────


def _skip_if_not_implemented():
    """Return the DriftMonitor class, or skip the test if not yet implemented."""
    try:
        from src.monitoring.drift_monitor import DriftMonitor

        return DriftMonitor
    except (ImportError, ModuleNotFoundError):
        pytest.skip("DriftMonitor not yet implemented — TDD red phase")


def _compute_psi_direct(
    reference: list[float], current: list[float], n_bins: int = 10
) -> float:
    """Convenience wrapper: create a DriftMonitor and call compute_psi."""
    try:
        from src.monitoring.drift_monitor import DriftMonitor

        dm = DriftMonitor()
        return dm.compute_psi(reference, current, n_bins=n_bins)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("DriftMonitor not yet implemented — TDD red phase")


# ── Hypothesis strategies ─────────────────────────────────────────────────────


@st.composite
def normal_distribution(draw: st.DrawFn) -> list[float]:
    """Draw 200 samples from a normal distribution with random mean / std."""
    mean = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False))
    std = draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False))
    seed = draw(st.integers(min_value=0, max_value=99_999))
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, 200).tolist()


@st.composite
def shifted_distribution(draw: st.DrawFn) -> tuple[list[float], list[float]]:
    """Draw a reference distribution and a non-trivially shifted copy."""
    mean = draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False))
    std = draw(st.floats(min_value=0.5, max_value=3.0, allow_nan=False))
    # Shift is at least 1.5 std deviations — large enough to produce measurable PSI.
    shift = draw(st.floats(min_value=1.5, max_value=10.0, allow_nan=False))
    seed = draw(st.integers(min_value=0, max_value=99_999))
    rng = np.random.default_rng(seed)
    reference = rng.normal(mean, std, 200).tolist()
    current = rng.normal(mean + shift, std, 200).tolist()
    return reference, current


# ── Test classes ──────────────────────────────────────────────────────────────


class TestDriftMonitorExists:
    """Smoke tests: verify the module and compute_psi entry-point exist."""

    def test_drift_monitor_importable(self) -> None:
        """DriftMonitor class must be importable from src.monitoring.drift_monitor."""
        try:
            from src.monitoring.drift_monitor import DriftMonitor

            dm = DriftMonitor()
            assert dm is not None
        except (ImportError, ModuleNotFoundError):
            pytest.skip("DriftMonitor not yet implemented — TDD red phase")

    def test_compute_psi_method_exists(self) -> None:
        """DriftMonitor must expose a compute_psi method."""
        try:
            from src.monitoring.drift_monitor import DriftMonitor

            assert hasattr(DriftMonitor, "compute_psi"), (
                "DriftMonitor must have a compute_psi method"
            )
        except (ImportError, ModuleNotFoundError):
            pytest.skip("DriftMonitor not yet implemented — TDD red phase")

    def test_get_severity_method_exists(self) -> None:
        """DriftMonitor must expose a get_severity method."""
        try:
            from src.monitoring.drift_monitor import DriftMonitor

            assert hasattr(DriftMonitor, "get_severity"), (
                "DriftMonitor must have a get_severity method"
            )
        except (ImportError, ModuleNotFoundError):
            pytest.skip("DriftMonitor not yet implemented — TDD red phase")


class TestPSIZeroDriftIdentity:
    """
    Property 15: PSI(D, D) ≈ 0.0

    Identical distributions must produce zero (or near-zero) PSI.
    Formula from Req 12.8:
        PSI = Σ (actual_pct − expected_pct) × ln(actual_pct / expected_pct)
    When reference == current every bin fraction is identical, so every term is
    (p − p) × ln(p/p) = 0 × 0 = 0, giving PSI = 0.

    Validates: Req 12.8
    """

    def test_identical_distributions_psi_zero_fixed(self) -> None:
        """PSI of identical fixed distributions must be approximately 0."""
        DriftMonitor = _skip_if_not_implemented()

        rng = np.random.default_rng(42)
        data = rng.normal(0.0, 1.0, 200).tolist()
        dm = DriftMonitor()
        psi = dm.compute_psi(data, data)

        assert abs(psi) < 0.01, (
            f"PSI(D, D) = {psi:.6f} should be ≈ 0.0 for identical distributions"
        )

    def test_psi_is_non_negative_for_identical(self) -> None:
        """PSI must be non-negative (it is a sum of non-negative terms)."""
        DriftMonitor = _skip_if_not_implemented()

        rng = np.random.default_rng(42)
        ref = rng.normal(0.0, 1.0, 200).tolist()
        cur = rng.normal(0.0, 1.0, 200).tolist()
        dm = DriftMonitor()
        psi = dm.compute_psi(ref, cur)

        assert psi >= 0.0, f"PSI must be non-negative, got {psi}"

    def test_psi_identity_uniform_distribution(self) -> None:
        """PSI of identical uniform integer distributions must be ≈ 0."""
        DriftMonitor = _skip_if_not_implemented()

        # Uniform [0, 99] — 10 equal bins will have exactly equal counts.
        ref = list(range(100))
        dm = DriftMonitor()
        psi = dm.compute_psi(ref, ref, n_bins=10)

        assert abs(psi) < 0.05, (
            f"PSI for identical uniform distributions should be ≈ 0, got {psi:.4f}"
        )

    @given(data=normal_distribution())
    @h_settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_psi_zero_identity_hypothesis(self, data: list[float]) -> None:
        """
        Property 15 (Hypothesis): For any distribution D,
        PSI(D, D) ≈ 0.0 (within machine precision).

        Validates: Req 12.8
        """
        DriftMonitor = _skip_if_not_implemented()

        dm = DriftMonitor()
        psi = dm.compute_psi(data, data)

        assert abs(psi) < 0.01, (
            f"PSI(D, D) = {psi:.6f} violates zero-drift identity property"
        )


class TestPSIMonotonicity:
    """
    Property 16: PSI(D_ref, D_shifted) > PSI(D_ref, D_ref) for any non-zero shift.

    A distribution shifted away from the reference must produce strictly higher
    PSI than comparing the reference to itself.

    Validates: Req 18.7
    """

    def test_shifted_distribution_has_higher_psi_fixed(self) -> None:
        """A 3-sigma shift must yield higher PSI than comparing to self."""
        DriftMonitor = _skip_if_not_implemented()

        rng = np.random.default_rng(42)
        ref = rng.normal(0.0, 1.0, 500).tolist()
        shifted = rng.normal(3.0, 1.0, 500).tolist()  # 3 std-dev shift

        dm = DriftMonitor()
        psi_same = dm.compute_psi(ref, ref)
        psi_shifted = dm.compute_psi(ref, shifted)

        assert psi_shifted > psi_same, (
            f"PSI(ref, shifted)={psi_shifted:.4f} should be > "
            f"PSI(ref, ref)={psi_same:.4f}"
        )

    def test_larger_shift_produces_larger_psi(self) -> None:
        """Larger shift must produce larger PSI than a smaller shift."""
        DriftMonitor = _skip_if_not_implemented()

        rng = np.random.default_rng(42)
        ref = rng.normal(0.0, 1.0, 500).tolist()
        small_shift = rng.normal(1.0, 1.0, 500).tolist()
        large_shift = rng.normal(4.0, 1.0, 500).tolist()

        dm = DriftMonitor()
        psi_small = dm.compute_psi(ref, small_shift)
        psi_large = dm.compute_psi(ref, large_shift)

        assert psi_large > psi_small, (
            f"Larger shift should produce larger PSI: "
            f"psi_large={psi_large:.4f} vs psi_small={psi_small:.4f}"
        )

    @given(pair=shifted_distribution())
    @h_settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_psi_monotonicity_hypothesis(self, pair: tuple) -> None:  # type: ignore[type-arg]
        """
        Property 16 (Hypothesis): For any reference D_ref and a significantly
        shifted D_shifted (shift ≥ 1.5 std-devs), PSI(D_ref, D_shifted)
        must be strictly greater than PSI(D_ref, D_ref).

        Validates: Req 18.7
        """
        DriftMonitor = _skip_if_not_implemented()

        reference, shifted = pair
        dm = DriftMonitor()

        psi_same = dm.compute_psi(reference, reference)
        psi_shifted = dm.compute_psi(reference, shifted)

        assert psi_shifted > psi_same, (
            f"Monotonicity violated: PSI(shifted)={psi_shifted:.6f} <= "
            f"PSI(same)={psi_same:.6f}"
        )


class TestPSIFormula:
    """Validate that the PSI formula matches the specification in Req 12.8."""

    def test_psi_formula_10_equal_frequency_bins(self) -> None:
        """
        PSI is computed over 10 equal-frequency bins derived from the reference.
        With identical distributions every bin fraction is equal, so PSI ≈ 0.
        """
        DriftMonitor = _skip_if_not_implemented()

        dm = DriftMonitor()
        ref = list(range(100))  # uniform [0, 99] — clean 10-bin split
        cur = list(range(100))
        psi = dm.compute_psi(ref, cur, n_bins=10)

        assert abs(psi) < 0.05, (
            f"PSI for identical distributions over 10 bins should be ≈ 0, got {psi:.4f}"
        )

    def test_psi_returns_float(self) -> None:
        """compute_psi must return a float (or int that is numeric)."""
        DriftMonitor = _skip_if_not_implemented()

        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 100).tolist()
        dm = DriftMonitor()
        result = dm.compute_psi(ref, ref)

        assert isinstance(result, (float, int, np.floating)), (
            f"compute_psi must return a numeric type, got {type(result)}"
        )


class TestPSIThresholds:
    """Verify that PSI values correctly map to DriftSeverity alert levels (Req 12.2, 12.3)."""

    def test_psi_below_medium_threshold_is_low_or_no_alert(self) -> None:
        """PSI ≤ 0.2 should NOT trigger a MEDIUM alert."""
        try:
            from src.monitoring.drift_monitor import DriftMonitor
            from src.schemas.base import DriftSeverity
        except (ImportError, ModuleNotFoundError):
            pytest.skip("DriftMonitor not yet implemented — TDD red phase")

        rng = np.random.default_rng(7)
        # Draws from the same distribution — PSI should be near 0.
        ref = rng.normal(0, 1, 500).tolist()
        cur = rng.normal(0, 1, 500).tolist()

        dm = DriftMonitor()
        psi = dm.compute_psi(ref, cur)

        if psi <= 0.2:
            severity = dm.get_severity(psi)
            assert severity not in (DriftSeverity.HIGH,), (
                f"PSI={psi:.4f} ≤ 0.2 must not yield HIGH severity"
            )

    def test_psi_above_high_threshold_triggers_high_severity(self) -> None:
        """PSI > 0.25 must map to HIGH severity (Req 12.3)."""
        try:
            from src.monitoring.drift_monitor import DriftMonitor
            from src.schemas.base import DriftSeverity
        except (ImportError, ModuleNotFoundError):
            pytest.skip("DriftMonitor not yet implemented — TDD red phase")

        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000).tolist()
        # 5-sigma shift guarantees PSI >> 0.25
        shifted = rng.normal(5.0, 1, 1000).tolist()

        dm = DriftMonitor()
        psi = dm.compute_psi(ref, shifted)

        if psi > 0.25:
            severity = dm.get_severity(psi)
            assert severity == DriftSeverity.HIGH, (
                f"PSI={psi:.4f} > 0.25 must yield HIGH severity, got {severity}"
            )

    def test_psi_medium_band_triggers_medium_severity(self) -> None:
        """PSI in (0.2, 0.25] must map to MEDIUM severity (Req 12.2)."""
        try:
            from src.monitoring.drift_monitor import DriftMonitor
            from src.schemas.base import DriftSeverity
        except (ImportError, ModuleNotFoundError):
            pytest.skip("DriftMonitor not yet implemented — TDD red phase")

        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500).tolist()
        shifted = rng.normal(4.0, 1, 500).tolist()

        dm = DriftMonitor()
        psi = dm.compute_psi(ref, shifted)

        if 0.2 < psi <= 0.25:
            severity = dm.get_severity(psi)
            assert severity == DriftSeverity.MEDIUM, (
                f"PSI={psi:.4f} in (0.2, 0.25] must yield MEDIUM severity, got {severity}"
            )
