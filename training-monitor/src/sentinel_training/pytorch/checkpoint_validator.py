"""Checkpoint validation via deterministic random projections and SHA-256.

After each checkpoint save, computes a compact fingerprint of the model
parameters using a deterministic random projection. Sudden jumps in the
fingerprint between checkpoints indicate possible silent data corruption.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from sentinel_training.common.anomaly_detector import AnomalyScore, AnomalyType
from sentinel_training.common.config import CheckpointValidationConfig

logger = structlog.get_logger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@dataclass
class CheckpointFingerprint:
    """Fingerprint of a model checkpoint."""

    step: int
    projection_hash: str
    sha256: str
    projection_vector: np.ndarray
    timestamp: float


class CheckpointValidator:
    """Validates checkpoint integrity using random projections and SHA-256.

    Computes a deterministic random projection of all model parameters to
    produce a compact fingerprint. Comparing fingerprints across time detects
    sudden parameter jumps that may indicate SDC-corrupted checkpoints.

    Args:
        config: Checkpoint validation configuration.
    """

    def __init__(self, config: CheckpointValidationConfig) -> None:
        self._config = config
        self._fingerprints: list[CheckpointFingerprint] = []
        self._rng = np.random.RandomState(config.projection_seed)
        self._projection_matrices: dict[int, np.ndarray] = {}

    def _get_projection_matrix(self, param_size: int) -> np.ndarray:
        """Get or create a deterministic random projection matrix.

        Uses the configured seed for reproducibility. The projection reduces
        high-dimensional parameter vectors to a low-dimensional fingerprint.

        Args:
            param_size: Total number of parameters.

        Returns:
            Projection matrix of shape (projection_dim, param_size).
        """
        if param_size not in self._projection_matrices:
            rng = np.random.RandomState(self._config.projection_seed)
            # Use sparse random projection for efficiency
            proj = rng.randn(self._config.projection_dim, param_size).astype(np.float32)
            proj /= np.sqrt(self._config.projection_dim)
            self._projection_matrices[param_size] = proj
        return self._projection_matrices[param_size]

    def compute_fingerprint(
        self,
        state_dict: dict[str, Any],
        step: int,
    ) -> CheckpointFingerprint:
        """Compute a fingerprint for a model state dict.

        Args:
            state_dict: The model's state dictionary (parameter name -> tensor/array).
            step: The training step at which this checkpoint was created.

        Returns:
            CheckpointFingerprint containing projection hash and SHA-256.
        """
        # Flatten all parameters into a single vector
        flat_params = self._flatten_state_dict(state_dict)
        param_size = len(flat_params)

        # Random projection
        proj_matrix = self._get_projection_matrix(param_size)
        projection = proj_matrix @ flat_params

        # Hash the projection
        proj_bytes = projection.tobytes()
        projection_hash = hashlib.sha256(proj_bytes).hexdigest()

        # Full SHA-256 of the state dict for integrity
        sha256 = self._compute_sha256(state_dict) if self._config.sha256_verify else ""

        fp = CheckpointFingerprint(
            step=step,
            projection_hash=projection_hash,
            sha256=sha256,
            projection_vector=projection,
            timestamp=time.time(),
        )
        self._fingerprints.append(fp)

        logger.info(
            "checkpoint_fingerprint",
            step=step,
            projection_hash=projection_hash[:16] + "...",
            sha256=sha256[:16] + "..." if sha256 else "disabled",
        )

        return fp

    def validate(
        self,
        state_dict: dict[str, Any],
        step: int,
    ) -> AnomalyScore | None:
        """Validate a checkpoint against previous fingerprints.

        Computes the fingerprint and compares it against the most recent
        previous fingerprint. A large jump in the projection space indicates
        possible SDC.

        Args:
            state_dict: The model state dictionary to validate.
            step: The current training step.

        Returns:
            AnomalyScore if divergence is detected, else None.
        """
        fp = self.compute_fingerprint(state_dict, step)

        if len(self._fingerprints) < 2:
            return None

        prev_fp = self._fingerprints[-2]

        # Compute distance between projection vectors
        diff = fp.projection_vector - prev_fp.projection_vector
        distance = float(np.linalg.norm(diff))
        prev_norm = float(np.linalg.norm(prev_fp.projection_vector))

        if prev_norm > 1e-12:
            relative_distance = distance / prev_norm
        else:
            relative_distance = distance

        if relative_distance > self._config.max_param_divergence:
            anomaly = AnomalyScore(
                anomaly_type=AnomalyType.CHECKPOINT_DIVERGENCE,
                score=relative_distance,
                threshold=self._config.max_param_divergence,
                observed_value=relative_distance,
                expected_value=0.0,
                step=step,
                metadata={
                    "projection_hash": fp.projection_hash,
                    "prev_projection_hash": prev_fp.projection_hash,
                    "absolute_distance": distance,
                    "prev_step": prev_fp.step,
                    "steps_between": step - prev_fp.step,
                },
            )
            logger.warning(
                "checkpoint_divergence",
                step=step,
                relative_distance=relative_distance,
                threshold=self._config.max_param_divergence,
                prev_step=prev_fp.step,
            )
            return anomaly

        logger.debug(
            "checkpoint_ok",
            step=step,
            relative_distance=relative_distance,
        )
        return None

    def verify_sha256(self, state_dict: dict[str, Any], expected_sha256: str) -> bool:
        """Verify the SHA-256 hash of a state dict.

        Args:
            state_dict: The model state dictionary.
            expected_sha256: The expected SHA-256 hex digest.

        Returns:
            True if the hash matches.
        """
        actual = self._compute_sha256(state_dict)
        matches = actual == expected_sha256
        if not matches:
            logger.error(
                "checkpoint_sha256_mismatch",
                expected=expected_sha256[:16] + "...",
                actual=actual[:16] + "...",
            )
        return matches

    @staticmethod
    def _flatten_state_dict(state_dict: dict[str, Any]) -> np.ndarray:
        """Flatten a state dict into a single float32 numpy array.

        Args:
            state_dict: Dictionary mapping parameter names to tensors/arrays.

        Returns:
            1-D numpy array of all parameter values concatenated.
        """
        parts: list[np.ndarray] = []
        for key in sorted(state_dict.keys()):
            value = state_dict[key]
            if torch is not None and isinstance(value, torch.Tensor):
                arr = value.detach().cpu().float().numpy().ravel()
            elif isinstance(value, np.ndarray):
                arr = value.astype(np.float32).ravel()
            else:
                try:
                    arr = np.asarray(value, dtype=np.float32).ravel()
                except (TypeError, ValueError):
                    continue
            parts.append(arr)

        if not parts:
            return np.zeros(1, dtype=np.float32)
        return np.concatenate(parts)

    @staticmethod
    def _compute_sha256(state_dict: dict[str, Any]) -> str:
        """Compute SHA-256 hash of a state dict.

        Uses a deterministic serialization order (sorted keys) and
        consistent float32 representation.

        Args:
            state_dict: Dictionary mapping parameter names to tensors/arrays.

        Returns:
            Hex digest of the SHA-256 hash.
        """
        hasher = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            hasher.update(key.encode("utf-8"))
            value = state_dict[key]
            if torch is not None and isinstance(value, torch.Tensor):
                data = value.detach().cpu().float().numpy().tobytes()
            elif isinstance(value, np.ndarray):
                data = value.astype(np.float32).tobytes()
            else:
                try:
                    data = np.asarray(value, dtype=np.float32).tobytes()
                except (TypeError, ValueError):
                    continue
            hasher.update(data)
        return hasher.hexdigest()

    @property
    def fingerprints(self) -> list[CheckpointFingerprint]:
        """All computed fingerprints."""
        return list(self._fingerprints)

    def reset(self) -> None:
        """Reset all stored fingerprints."""
        self._fingerprints.clear()
