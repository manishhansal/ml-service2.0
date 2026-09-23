"""
ModelRegistry — champion/challenger artifact management with SHA-256 integrity.

Design guarantees:
- Artifacts are stored with SHA-256 checksums computed at registration time.
- At load time, the checksum is recomputed and compared; on mismatch, raises
  ArtifactIntegrityFailure and falls back to heuristic policy.
- Once an artifact is registered (written), it cannot be overwritten; any
  attempt raises ArtifactAlreadyExistsError.
- Champion artifacts are loaded at FastAPI lifespan startup.
- Thread-safe via internal threading.Lock.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from src.config import settings
from src.logging_config import get_logger
from src.schemas.base import ModelLifecycleStage, PredictionProvenance
from src.schemas.registry import ModelArtifact

logger = get_logger(__name__)


class ArtifactIntegrityFailure(Exception):
    """Raised when a model artifact's SHA-256 checksum does not match."""


class ArtifactAlreadyExistsError(Exception):
    """Raised when attempting to register an artifact that already exists."""


class ModelRegistry:
    """
    Thread-safe registry for model artifacts.

    Directory layout under settings.model_artifacts_path:

        {model_artifacts_path}/
          {model_name}/
            {version}/
              metadata.json      ← ModelArtifact JSON (excluding artifact file)
              {artifact_file}    ← the actual model file (joblib, .json, .txt, .pt, etc.)
              {artifact_file}.sha256  ← hex digest of the artifact file
            champion.json        ← points to the current champion version

    Usage::

        registry = ModelRegistry()
        registry.load_all_champions()  # call at startup

        artifact = registry.get_champion("market_regime")
        model_path = registry.get_artifact_path(artifact)
    """

    def __init__(self, artifacts_path: Path | None = None) -> None:
        self._root = artifacts_path or settings.model_artifacts_path
        self._root.mkdir(parents=True, exist_ok=True)
        self._champions: dict[str, ModelArtifact] = {}
        self._lock = Lock()

    # ── SHA-256 helpers ───────────────────────────────────────────────────────

    @staticmethod
    def compute_file_sha256(file_path: Path) -> str:
        """Compute the SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with file_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def compute_dict_sha256(data: dict[str, Any]) -> str:
        """Compute SHA-256 hex digest of a JSON-serialized dict (sorted keys)."""
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # ── Public API ────────────────────────────────────────────────────────────

    def register(
        self,
        artifact: ModelArtifact,
        artifact_file_path: Path | None = None,
    ) -> None:
        """
        Register a new model artifact in the registry.

        Steps:
        1. Create directory: {root}/{model_name}/{version}/
        2. Copy artifact file to the version directory (if artifact_file_path is provided)
        3. Compute and write SHA-256 checksum of the artifact file
        4. Write metadata.json with ModelArtifact schema
        5. Update champion.json if stage == PRODUCTION

        Args:
            artifact:           ModelArtifact schema with all required fields.
            artifact_file_path: Source path of the artifact file to copy.
                                If None, the artifact.artifact_path is used as-is.

        Raises:
            ArtifactAlreadyExistsError: If the version directory already exists.
        """
        with self._lock:
            version_dir = self._root / artifact.model_name / artifact.version

            if version_dir.exists():
                raise ArtifactAlreadyExistsError(
                    f"Artifact already exists: {version_dir}. "
                    "Use a new version string to register an updated model."
                )

            version_dir.mkdir(parents=True, exist_ok=True)

            # Copy artifact file if source path is provided
            final_artifact_path = Path(artifact.artifact_path)
            if artifact_file_path is not None and artifact_file_path.exists():
                dest = version_dir / artifact_file_path.name
                shutil.copy2(artifact_file_path, dest)
                final_artifact_path = dest

            # Compute and write SHA-256 checksum
            sha256: str
            if final_artifact_path.exists():
                sha256 = self.compute_file_sha256(final_artifact_path)
                checksum_path = version_dir / f"{final_artifact_path.name}.sha256"
                checksum_path.write_text(sha256, encoding="utf-8")
            else:
                # No artifact file — compute hash of metadata instead
                sha256 = self.compute_dict_sha256(artifact.model_dump())

            # Write metadata.json — use model_copy to update the checksum field
            metadata = artifact.model_copy(update={"sha256_checksum": sha256})
            metadata_path = version_dir / "metadata.json"
            metadata_path.write_text(
                metadata.model_dump_json(indent=2), encoding="utf-8"
            )

            # Update champion.json if this is a PRODUCTION stage artifact
            if artifact.stage == ModelLifecycleStage.PRODUCTION:
                champion_path = self._root / artifact.model_name / "champion.json"
                champion_path.write_text(
                    json.dumps({"version": artifact.version}, indent=2),
                    encoding="utf-8",
                )
                # Also update in-memory champion
                self._champions[artifact.model_name] = metadata

            logger.info(
                "artifact_registered",
                model_name=artifact.model_name,
                version=artifact.version,
                stage=artifact.stage.value,
                sha256=sha256[:16] + "...",
            )

    def get_champion(self, model_name: str) -> ModelArtifact | None:
        """
        Return the current champion artifact for model_name.

        Returns None if no champion is registered.
        """
        with self._lock:
            # Check in-memory first
            if model_name in self._champions:
                return self._champions[model_name]

            # Try to load from disk
            champion_path = self._root / model_name / "champion.json"
            if not champion_path.exists():
                return None

            try:
                data = json.loads(champion_path.read_text(encoding="utf-8"))
                version = data["version"]
                metadata_path = self._root / model_name / version / "metadata.json"
                if metadata_path.exists():
                    artifact = ModelArtifact.model_validate_json(
                        metadata_path.read_text(encoding="utf-8")
                    )
                    self._champions[model_name] = artifact
                    return artifact
            except Exception as exc:
                logger.warning(
                    "champion_load_failed",
                    model_name=model_name,
                    error=str(exc),
                )
            return None

    def load_artifact(self, artifact: ModelArtifact) -> Path:
        """
        Validate the artifact's SHA-256 checksum and return its filesystem path.

        Raises:
            ArtifactIntegrityFailure: If the checksum does not match.
            FileNotFoundError:        If the artifact file does not exist.
        """
        artifact_path = Path(artifact.artifact_path)

        if not artifact_path.exists():
            # Try relative to root/model_name/version/
            alt_path = (
                self._root
                / artifact.model_name
                / artifact.version
                / artifact_path.name
            )
            if alt_path.exists():
                artifact_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Artifact file not found: {artifact.artifact_path}"
                )

        # Verify checksum
        actual_sha256 = self.compute_file_sha256(artifact_path)
        expected_sha256 = artifact.sha256_checksum

        if actual_sha256 != expected_sha256:
            logger.critical(
                "ARTIFACT_INTEGRITY_FAILURE",
                model_name=artifact.model_name,
                version=artifact.version,
                expected_sha256=expected_sha256[:16] + "...",
                actual_sha256=actual_sha256[:16] + "...",
            )
            raise ArtifactIntegrityFailure(
                f"SHA-256 mismatch for {artifact.model_name} v{artifact.version}: "
                f"expected={expected_sha256[:16]}..., actual={actual_sha256[:16]}..."
            )

        return artifact_path

    def load_all_champions(self) -> dict[str, ModelArtifact | None]:
        """
        Load all champion artifacts at startup. Returns mapping of model_name → artifact.

        On ArtifactIntegrityFailure for any model: logs CRITICAL, does NOT crash
        the service, removes that model from the in-memory registry so it falls
        back to heuristic.
        On FileNotFoundError: model has no registered champion, returns None for
        that model.
        """
        known_models = [
            "market_regime",
            "stock_ranker",
            "strategy_selector",
            "risk_predictor",
            "portfolio_optimizer",
            "rl_execution_agent",
            "price_forecaster",
            "iv_regime_classifier",
        ]

        results: dict[str, ModelArtifact | None] = {}

        for model_name in known_models:
            artifact = self.get_champion(model_name)
            if artifact is None:
                results[model_name] = None
                continue

            try:
                self.load_artifact(artifact)
                results[model_name] = artifact
                logger.info(
                    "champion_loaded",
                    model_name=model_name,
                    version=artifact.version,
                )
            except ArtifactIntegrityFailure:
                # Remove from in-memory cache — will fall back to heuristic
                with self._lock:
                    self._champions.pop(model_name, None)
                results[model_name] = None
            except FileNotFoundError:
                logger.warning(
                    "champion_artifact_file_not_found",
                    model_name=model_name,
                    artifact_path=artifact.artifact_path,
                )
                results[model_name] = None

        return results

    def list_registry(self) -> list[ModelArtifact]:
        """List all registered model artifacts across all models and versions."""
        artifacts: list[ModelArtifact] = []

        if not self._root.exists():
            return artifacts

        for model_dir in self._root.iterdir():
            if not model_dir.is_dir():
                continue
            for version_dir in model_dir.iterdir():
                # Skip champion.json (not a directory) and non-directories
                if not version_dir.is_dir():
                    continue
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        artifact = ModelArtifact.model_validate_json(
                            metadata_path.read_text(encoding="utf-8")
                        )
                        artifacts.append(artifact)
                    except Exception as exc:
                        logger.warning(
                            "artifact_metadata_parse_error",
                            path=str(metadata_path),
                            error=str(exc),
                        )

        return artifacts

    def get_artifact_path(self, artifact: ModelArtifact) -> Path:
        """Return the filesystem path of the artifact file within the registry."""
        version_dir = self._root / artifact.model_name / artifact.version
        artifact_file = Path(artifact.artifact_path).name
        return version_dir / artifact_file
