"""Tests for drift attribution (mandate §26): artifact vs genuine drift."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.reference import (
    ReferenceDistribution,
    attribute_drift,
)


def _ref(feature_samples: dict[str, list[float]]) -> ReferenceDistribution:
    return ReferenceDistribution(
        model_name="m", version="v",
        feature_samples=feature_samples, prediction_samples=[0.5] * 10,
    )


class TestAttributeDrift:
    def test_no_drift_when_current_matches_reference(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(0, 1, 500).tolist()
        ref = _ref({"f": vals})
        cur = pd.DataFrame({"f": rng.normal(0, 1, 200)})
        att = attribute_drift(ref, cur)
        assert att.classification == "NO_SIGNIFICANT_DRIFT"
        assert att.likely_construction_artifact is False

    def test_trending_feature_flagged_as_construction_artifact(self):
        # A strongly trending (non-stationary) feature: reference is a full ramp,
        # current is the top tail of the same ramp. Both the live PSI and the
        # reference's own early-vs-late self-test blow up -> artifact.
        ramp = np.linspace(0, 100, 600)
        ref = _ref({"f": ramp.tolist()})
        cur = pd.DataFrame({"f": ramp[-200:]})  # chronological tail
        att = attribute_drift(ref, cur)
        assert att.max_psi > 0.25
        assert att.self_test_max_psi > 0.25
        assert att.likely_construction_artifact is True
        assert att.classification == "CONSTRUCTION_ARTIFACT_OR_NONSTATIONARITY"

    def test_stable_reference_but_shifted_live_is_possible_genuine_drift(self):
        rng = np.random.default_rng(1)
        # Stationary reference (low self-test PSI), but the live window is a
        # genuinely different population (mean-shifted) -> possible genuine drift.
        ref_vals = rng.normal(0, 1, 800).tolist()
        ref = _ref({"f": ref_vals})
        cur = pd.DataFrame({"f": rng.normal(5, 1, 200)})  # shifted far away
        att = attribute_drift(ref, cur)
        assert att.max_psi > 0.25
        assert att.self_test_max_psi <= 0.25
        assert att.classification == "POSSIBLE_GENUINE_DRIFT"
        assert att.likely_construction_artifact is False

    def test_missing_feature_in_current_is_skipped(self):
        ref = _ref({"f": list(range(100)), "g": list(range(100))})
        cur = pd.DataFrame({"f": list(range(100))})  # no 'g'
        att = attribute_drift(ref, cur)
        assert "g" not in att.per_feature
        assert "f" in att.per_feature

    def test_feature_with_too_few_reference_values_is_skipped(self):
        # < 4 reference values -> feature is skipped entirely (not classifiable).
        ref = _ref({"tiny": [1.0, 2.0, 3.0]})
        cur = pd.DataFrame({"tiny": [1.0, 2.0, 3.0, 4.0]})
        att = attribute_drift(ref, cur)
        assert "tiny" not in att.per_feature
        assert att.classification == "NO_SIGNIFICANT_DRIFT"

    def test_empty_current_column_is_skipped(self):
        ref = _ref({"f": list(range(100))})
        cur = pd.DataFrame({"f": [float("nan")] * 5})  # all NaN -> empty after dropna
        att = attribute_drift(ref, cur)
        assert "f" not in att.per_feature

    def test_medium_severity_band(self):
        # Deterministically find a shift whose PSI lands in the MEDIUM band
        # (0.20, 0.25], exercising the MEDIUM branch of the severity classifier.
        rng = np.random.default_rng(7)
        ref_vals = rng.normal(0, 1, 6000).tolist()
        ref = _ref({"f": ref_vals})
        cur_pool = rng.normal(0, 1, 3000)
        found_medium = False
        # Fine grid over a shift range that brackets the MEDIUM band.
        for shift in np.linspace(0.15, 0.55, 41):
            cur = pd.DataFrame({"f": cur_pool + shift})
            sev = attribute_drift(ref, cur).per_feature["f"]["live_severity"]
            if sev == "MEDIUM":
                found_medium = True
                break
        assert found_medium, "expected at least one MEDIUM-band severity on the grid"

    def test_to_dict_shape(self):
        ref = _ref({"f": list(range(100))})
        cur = pd.DataFrame({"f": list(range(80, 100))})
        d = attribute_drift(ref, cur).to_dict()
        for key in (
            "max_psi", "severity", "self_test_max_psi", "self_test_severity",
            "likely_construction_artifact", "classification", "note", "per_feature",
        ):
            assert key in d
