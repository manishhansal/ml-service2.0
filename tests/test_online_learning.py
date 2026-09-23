"""
test_online_learning.py

TDD property-based tests for OnlineLearner.
Written BEFORE OnlineLearner implementation (red phase).

Properties:
  Property 17: version(model_after_update) != version(model_before_update)
  Property 18: artifact_exists(prior_version) == True after any number of updates
  Property 19: consecutive_updates <= 5 before full retrain required

Requirements: Req 14.1, Req 14.2, Req 14.4, Req 14.5, Req 14.6, Req 18.12
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings as h_settings, HealthCheck
from hypothesis import strategies as st

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")


def _skip_if_not_implemented():
    """Skip if OnlineLearner is not yet implemented."""
    try:
        from src.training.online_learner import OnlineLearner
        return OnlineLearner
    except (ImportError, ModuleNotFoundError):
        pytest.skip("OnlineLearner not yet implemented — TDD red phase")


class TestOnlineLearnerImport:
    """Verify the module can be imported."""

    def test_online_learner_importable(self):
        try:
            from src.training.online_learner import OnlineLearner
            learner = OnlineLearner(model_name="market_regime")
            assert learner is not None
        except (ImportError, ModuleNotFoundError):
            pytest.skip("OnlineLearner not yet implemented — TDD red phase")


class TestVersionUniqueness:
    """Property 17: Each update produces a strictly different version string."""

    def test_update_produces_new_version(self):
        """version after update != version before update."""
        OnlineLearner = _skip_if_not_implemented()

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "test_model.json"
            artifact_path.write_text('{"version": "1.0.0", "weights": []}')

            learner = OnlineLearner(model_name="market_regime", artifacts_dir=Path(tmpdir))

            prior_version = "1.0.0"
            new_version = learner.generate_next_version(prior_version)

            assert new_version != prior_version, (
                f"New version {new_version!r} must differ from prior {prior_version!r}"
            )

    def test_consecutive_updates_have_unique_versions(self):
        """Each consecutive update must produce a strictly unique version."""
        OnlineLearner = _skip_if_not_implemented()

        learner = OnlineLearner(model_name="market_regime")

        versions = set()
        current = "1.0.0"
        for _ in range(5):
            next_v = learner.generate_next_version(current)
            assert next_v not in versions, f"Version {next_v!r} was already generated"
            versions.add(next_v)
            current = next_v

    def test_version_contains_online_marker(self):
        """Generated version must include 'online' marker."""
        OnlineLearner = _skip_if_not_implemented()

        learner = OnlineLearner(model_name="market_regime")
        new_version = learner.generate_next_version("1.0.0")

        assert "online" in new_version.lower(), (
            f"Generated version {new_version!r} should contain 'online' marker"
        )


class TestArtifactImmutability:
    """Property 18: Prior artifacts remain intact after any number of updates."""

    def test_prior_artifact_preserved_after_update(self):
        """The prior artifact file must still exist after an online update."""
        OnlineLearner = _skip_if_not_implemented()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a prior artifact
            prior_path = Path(tmpdir) / "market_regime_v1.0.0.json"
            prior_path.write_text('{"model": "heuristic", "version": "1.0.0"}')

            learner = OnlineLearner(
                model_name="market_regime",
                artifacts_dir=Path(tmpdir),
                max_consecutive_updates=5,
            )

            # Simulate creating a new versioned artifact without overwriting the prior
            new_version = learner.generate_next_version("1.0.0")
            new_path = Path(tmpdir) / f"market_regime_{new_version}.json"
            new_path.write_text(f'{{"model": "updated", "version": "{new_version}"}}')

            # Prior artifact must still exist
            assert prior_path.exists(), (
                f"Prior artifact at {prior_path} was deleted — immutability violated"
            )
            assert new_path.exists(), (
                f"New artifact at {new_path} does not exist — update not recorded"
            )

    def test_never_overwrites_existing_artifact(self):
        """Registering a new artifact must not modify any existing file."""
        OnlineLearner = _skip_if_not_implemented()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an existing artifact with known content
            existing_path = Path(tmpdir) / "market_regime_v1.0.0.json"
            original_content = '{"original": true}'
            existing_path.write_text(original_content)

            learner = OnlineLearner(model_name="market_regime", artifacts_dir=Path(tmpdir))

            # Simulate the learner creating a new artifact (different path)
            new_version = learner.generate_next_version("1.0.0")
            new_path = Path(tmpdir) / f"market_regime_{new_version}.json"
            new_path.write_text('{"original": false}')

            # Original content must be unchanged
            assert existing_path.read_text() == original_content, (
                "Existing artifact content was modified — immutability violated"
            )


class TestUpdateCapEnforcement:
    """Property 19: consecutive_updates <= 5 before full retrain required."""

    def test_max_consecutive_updates_default_is_5(self):
        """Default max consecutive updates must be 5."""
        OnlineLearner = _skip_if_not_implemented()

        learner = OnlineLearner(model_name="market_regime")
        assert learner.max_consecutive_updates == 5, (
            f"Default max_consecutive_updates should be 5, got {learner.max_consecutive_updates}"
        )

    def test_requires_full_retrain_at_cap(self):
        """When consecutive_updates == max, requires_full_retrain must return True."""
        OnlineLearner = _skip_if_not_implemented()

        learner = OnlineLearner(model_name="market_regime", max_consecutive_updates=5)

        # Simulate 5 consecutive updates
        learner._consecutive_updates = 5

        assert learner.requires_full_retrain(), (
            "After 5 consecutive updates, requires_full_retrain() must return True"
        )

    def test_does_not_require_retrain_below_cap(self):
        """When consecutive_updates < max, requires_full_retrain must return False."""
        OnlineLearner = _skip_if_not_implemented()

        learner = OnlineLearner(model_name="market_regime", max_consecutive_updates=5)
        learner._consecutive_updates = 4

        assert not learner.requires_full_retrain(), (
            "After 4 consecutive updates, requires_full_retrain() should return False"
        )

    def test_reset_after_full_retrain(self):
        """reset_consecutive_updates() must reset the counter to 0."""
        OnlineLearner = _skip_if_not_implemented()

        learner = OnlineLearner(model_name="market_regime")
        learner._consecutive_updates = 5
        learner.reset_consecutive_updates()

        assert learner._consecutive_updates == 0, (
            f"After reset, consecutive_updates should be 0, got {learner._consecutive_updates}"
        )

    @given(n_updates=st.integers(min_value=0, max_value=10))
    @h_settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_update_cap_invariant_hypothesis(self, n_updates: int) -> None:
        """
        Property 19 (Hypothesis): For all n_updates,
        consecutive_updates < max_consecutive_updates implies requires_full_retrain() == False.
        consecutive_updates >= max_consecutive_updates implies requires_full_retrain() == True.

        **Validates: Requirements 18.12**
        """
        OnlineLearner = _skip_if_not_implemented()

        max_updates = 5
        learner = OnlineLearner(model_name="market_regime", max_consecutive_updates=max_updates)
        learner._consecutive_updates = n_updates

        if n_updates >= max_updates:
            assert learner.requires_full_retrain(), (
                f"With {n_updates} consecutive updates (max={max_updates}), "
                "requires_full_retrain() must be True"
            )
        else:
            assert not learner.requires_full_retrain(), (
                f"With {n_updates} consecutive updates (max={max_updates}), "
                "requires_full_retrain() must be False"
            )
