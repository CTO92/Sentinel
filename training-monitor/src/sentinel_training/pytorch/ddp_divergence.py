"""Cross-rank divergence detection for DDP/FSDP/DeepSpeed training.

In distributed data-parallel training, all ranks should produce identical
gradients before allreduce. By periodically hashing pre-allreduce gradients
and comparing across ranks, we can identify a specific GPU producing
corrupted results.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import structlog

from sentinel_training.common.anomaly_detector import AnomalyScore, AnomalyType
from sentinel_training.common.config import CrossRankDivergenceConfig

logger = structlog.get_logger(__name__)

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class DDPDivergenceDetector:
    """Detects gradient divergence across ranks in distributed training.

    Periodically computes a hash of pre-allreduce gradients on each rank
    and uses allgather to compare. If any rank's hash differs, that rank
    is flagged as a suspect for silent data corruption.

    Args:
        config: Cross-rank divergence configuration.
        rank: The current process rank.
        world_size: Total number of ranks.
    """

    def __init__(
        self,
        config: CrossRankDivergenceConfig,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self._config = config
        self._rank = rank
        self._world_size = world_size
        self._step_count: int = 0
        self._anomalies: list[AnomalyScore] = []
        self._rng = np.random.RandomState(config.hash_seed)
        self._projection_vector: np.ndarray | None = None

    def compute_gradient_hash(
        self,
        model: "nn.Module",
    ) -> str:
        """Compute a deterministic hash of the model's current gradients.

        Uses a random projection to reduce the gradient to a compact
        representation, then hashes the result.

        Args:
            model: The PyTorch model with populated gradients.

        Returns:
            Hex digest of the gradient hash.
        """
        # Collect all gradient values into a flat vector
        grad_parts: list[np.ndarray] = []
        for name, param in sorted(model.named_parameters()):
            if param.grad is not None:
                grad_np = param.grad.detach().cpu().float().numpy().ravel()
                grad_parts.append(grad_np)

        if not grad_parts:
            return hashlib.sha256(b"empty_gradients").hexdigest()

        flat_grads = np.concatenate(grad_parts)
        grad_size = len(flat_grads)

        # Get or create projection vector
        if self._projection_vector is None or len(self._projection_vector) != grad_size:
            rng = np.random.RandomState(self._config.hash_seed)
            self._projection_vector = rng.randn(
                self._config.projection_dim, grad_size
            ).astype(np.float32)

        # Project and hash
        projected = self._projection_vector @ flat_grads.astype(np.float32)
        # Quantize to reduce floating-point noise within tolerance
        quantized = np.round(projected / self._config.tolerance) * self._config.tolerance
        return hashlib.sha256(quantized.tobytes()).hexdigest()

    def compute_gradient_hash_from_tensors(
        self,
        gradients: list["torch.Tensor"],
    ) -> str:
        """Compute gradient hash from a list of gradient tensors.

        Args:
            gradients: List of gradient tensors.

        Returns:
            Hex digest of the gradient hash.
        """
        grad_parts: list[np.ndarray] = []
        for grad in gradients:
            if grad is not None:
                grad_parts.append(grad.detach().cpu().float().numpy().ravel())

        if not grad_parts:
            return hashlib.sha256(b"empty_gradients").hexdigest()

        flat_grads = np.concatenate(grad_parts)
        grad_size = len(flat_grads)

        if self._projection_vector is None or self._projection_vector.shape[1] != grad_size:
            rng = np.random.RandomState(self._config.hash_seed)
            self._projection_vector = rng.randn(
                self._config.projection_dim, grad_size
            ).astype(np.float32)

        projected = self._projection_vector @ flat_grads.astype(np.float32)
        quantized = np.round(projected / self._config.tolerance) * self._config.tolerance
        return hashlib.sha256(quantized.tobytes()).hexdigest()

    def check_divergence(
        self,
        model: "nn.Module",
        process_group: Any = None,
    ) -> AnomalyScore | None:
        """Check for gradient divergence across ranks.

        Should be called every `check_interval` steps, before allreduce.
        Uses allgather to collect hashes from all ranks and compares them.

        Args:
            model: The model with populated pre-allreduce gradients.
            process_group: Optional torch.distributed process group.

        Returns:
            AnomalyScore if divergence is detected, else None.
        """
        self._step_count += 1

        if self._step_count % self._config.check_interval != 0:
            return None

        if self._world_size <= 1:
            return None

        local_hash = self.compute_gradient_hash(model)

        # Gather hashes from all ranks
        all_hashes = self._allgather_hashes(local_hash, process_group)

        if all_hashes is None:
            return None

        return self._analyze_hashes(all_hashes, local_hash)

    def check_divergence_from_hashes(
        self,
        local_hash: str,
        all_hashes: list[str],
    ) -> AnomalyScore | None:
        """Check for divergence given pre-computed hashes (for testing).

        Args:
            local_hash: This rank's gradient hash.
            all_hashes: All ranks' gradient hashes.

        Returns:
            AnomalyScore if divergence is detected, else None.
        """
        self._step_count += 1
        return self._analyze_hashes(all_hashes, local_hash)

    def _analyze_hashes(
        self,
        all_hashes: list[str],
        local_hash: str,
    ) -> AnomalyScore | None:
        """Analyze collected hashes for divergence.

        Identifies the minority hash(es) as suspect ranks.

        Args:
            all_hashes: Hashes from all ranks.
            local_hash: This rank's hash.

        Returns:
            AnomalyScore if divergence is detected.
        """
        unique_hashes = set(all_hashes)
        if len(unique_hashes) == 1:
            logger.debug(
                "cross_rank_consistent",
                step=self._step_count,
                world_size=self._world_size,
            )
            return None

        # Find the majority hash
        hash_counts: dict[str, int] = {}
        for h in all_hashes:
            hash_counts[h] = hash_counts.get(h, 0) + 1

        majority_hash = max(hash_counts, key=lambda k: hash_counts[k])

        # Identify suspect ranks (those not matching majority)
        suspect_ranks = [
            rank for rank, h in enumerate(all_hashes) if h != majority_hash
        ]

        # Determine if this rank is suspect
        is_local_suspect = local_hash != majority_hash

        anomaly = AnomalyScore(
            anomaly_type=AnomalyType.CROSS_RANK_DIVERGENCE,
            score=float(len(suspect_ranks)),
            threshold=0.0,  # Any divergence is anomalous
            observed_value=float(len(unique_hashes)),
            expected_value=1.0,
            step=self._step_count,
            rank=self._rank,
            metadata={
                "suspect_ranks": suspect_ranks,
                "unique_hash_count": len(unique_hashes),
                "majority_count": hash_counts[majority_hash],
                "is_local_suspect": is_local_suspect,
                "hash_distribution": {h: c for h, c in hash_counts.items()},
            },
        )
        self._anomalies.append(anomaly)
        logger.critical(
            "cross_rank_divergence",
            step=self._step_count,
            suspect_ranks=suspect_ranks,
            unique_hashes=len(unique_hashes),
            rank=self._rank,
        )
        return anomaly

    def _allgather_hashes(
        self,
        local_hash: str,
        process_group: Any = None,
    ) -> list[str] | None:
        """Gather gradient hashes from all ranks via allgather.

        Args:
            local_hash: This rank's gradient hash.
            process_group: Optional process group.

        Returns:
            List of hashes from all ranks, or None if distributed is unavailable.
        """
        if dist is None or not dist.is_initialized():
            logger.warning("distributed_not_available")
            return None

        # Encode hash as a tensor for allgather
        hash_bytes = local_hash.encode("utf-8")[:64]  # SHA-256 hex is 64 chars
        hash_tensor = torch.zeros(64, dtype=torch.uint8)
        for i, b in enumerate(hash_bytes):
            hash_tensor[i] = b

        gather_list = [torch.zeros(64, dtype=torch.uint8) for _ in range(self._world_size)]

        try:
            dist.all_gather(gather_list, hash_tensor, group=process_group)
        except Exception as exc:
            logger.error("allgather_failed", error=str(exc))
            return None

        all_hashes: list[str] = []
        for t in gather_list:
            hash_str = bytes(t.numpy().tolist()).decode("utf-8").rstrip("\x00")
            all_hashes.append(hash_str)

        return all_hashes

    @property
    def step_count(self) -> int:
        """Number of steps processed."""
        return self._step_count

    @property
    def anomalies(self) -> list[AnomalyScore]:
        """All recorded anomalies."""
        return list(self._anomalies)

    def reset(self) -> None:
        """Reset all state."""
        self._step_count = 0
        self._anomalies.clear()
        self._projection_vector = None
