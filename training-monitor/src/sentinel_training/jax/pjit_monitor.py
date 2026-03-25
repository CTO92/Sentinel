"""Sharding-aware monitoring for JAX pjit/sharded computations.

Inspects JAX sharding information to track per-device computation results
and detect divergence between shards that may indicate SDC on a specific
device.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import structlog

from sentinel_training.common.anomaly_detector import (
    AnomalyScore,
    AnomalyType,
    EWMATracker,
)
from sentinel_training.common.config import CrossRankDivergenceConfig

logger = structlog.get_logger(__name__)

try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec, Mesh
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]


class PjitMonitor:
    """Monitors sharded JAX computations for per-device divergence.

    In pjit/sharded training, each device holds a shard of the parameters
    and gradients. This monitor detects when a specific device's shard
    diverges from expected patterns, indicating possible SDC.

    Args:
        config: Cross-rank divergence configuration.
    """

    def __init__(self, config: CrossRankDivergenceConfig) -> None:
        self._config = config
        self._step_count: int = 0
        self._anomalies: list[AnomalyScore] = []
        self._per_device_trackers: dict[str, dict[str, EWMATracker]] = {}

    def _get_tracker(self, device_id: str, param_name: str) -> EWMATracker:
        """Get or create an EWMA tracker for a device-parameter pair."""
        if device_id not in self._per_device_trackers:
            self._per_device_trackers[device_id] = {}
        device_trackers = self._per_device_trackers[device_id]
        if param_name not in device_trackers:
            device_trackers[param_name] = EWMATracker(
                alpha=0.05,
                warmup_steps=self._config.check_interval,
            )
        return device_trackers[param_name]

    def check_sharded_array(
        self,
        arr: Any,
        name: str = "param",
    ) -> list[AnomalyScore]:
        """Check a sharded array for per-device divergence.

        Inspects the sharding of the array, extracts per-device shards,
        and compares their norms to detect outliers.

        Args:
            arr: A JAX array, potentially sharded across devices.
            name: Name for this parameter (used in logging).

        Returns:
            List of anomalies detected.
        """
        if jax is None:
            return []

        self._step_count += 1
        anomalies: list[AnomalyScore] = []

        # Get sharding info
        sharding = getattr(arr, "sharding", None)
        if sharding is None:
            # Not sharded, single-device array
            return []

        # Get the per-device arrays
        device_arrays = self._get_per_device_arrays(arr)
        if len(device_arrays) <= 1:
            return []

        # Compute norms per device
        device_norms: dict[str, float] = {}
        for device_id, shard in device_arrays.items():
            norm_val = float(jnp.linalg.norm(jnp.ravel(shard)))
            device_norms[device_id] = norm_val
            tracker = self._get_tracker(device_id, name)
            tracker.update(norm_val)

        # Check for outliers via median absolute deviation
        norms_list = list(device_norms.values())
        if len(norms_list) < 2:
            return []

        median_norm = float(np.median(norms_list))
        mad = float(np.median([abs(n - median_norm) for n in norms_list]))
        if mad < 1e-12:
            # All norms are nearly identical
            return []

        threshold = self._config.tolerance
        for device_id, norm_val in device_norms.items():
            deviation = abs(norm_val - median_norm) / mad
            if deviation > 3.0:  # 3 MAD threshold
                anomaly = AnomalyScore(
                    anomaly_type=AnomalyType.CROSS_RANK_DIVERGENCE,
                    score=deviation,
                    threshold=3.0,
                    observed_value=norm_val,
                    expected_value=median_norm,
                    step=self._step_count,
                    layer_name=name,
                    metadata={
                        "device_id": device_id,
                        "mad": mad,
                        "all_norms": device_norms,
                    },
                )
                self._anomalies.append(anomaly)
                anomalies.append(anomaly)
                logger.warning(
                    "pjit_device_divergence",
                    device=device_id,
                    param=name,
                    deviation=deviation,
                    norm=norm_val,
                    median=median_norm,
                    step=self._step_count,
                )

        return anomalies

    def check_sharded_gradients(
        self,
        grad_tree: Any,
        prefix: str = "",
    ) -> list[AnomalyScore]:
        """Check all sharded gradients in a pytree for device divergence.

        Args:
            grad_tree: A pytree of potentially sharded gradient arrays.
            prefix: Path prefix for naming.

        Returns:
            List of all anomalies detected.
        """
        if jax is None:
            return []

        all_anomalies: list[AnomalyScore] = []
        leaves_with_paths = self._flatten_with_paths(grad_tree, prefix)

        for path, leaf in leaves_with_paths:
            if leaf is not None:
                anomalies = self.check_sharded_array(leaf, name=path)
                all_anomalies.extend(anomalies)

        return all_anomalies

    def compute_device_hashes(
        self,
        arr: Any,
    ) -> dict[str, str]:
        """Compute a hash of each device's shard for comparison.

        Args:
            arr: A sharded JAX array.

        Returns:
            Dictionary mapping device IDs to hex hash strings.
        """
        if jax is None:
            return {}

        device_arrays = self._get_per_device_arrays(arr)
        hashes: dict[str, str] = {}

        for device_id, shard in device_arrays.items():
            shard_np = np.asarray(shard, dtype=np.float32)
            quantized = np.round(shard_np / self._config.tolerance) * self._config.tolerance
            hashes[device_id] = hashlib.sha256(quantized.tobytes()).hexdigest()

        return hashes

    @staticmethod
    def _get_per_device_arrays(arr: Any) -> dict[str, Any]:
        """Extract per-device shard arrays from a sharded JAX array.

        Args:
            arr: A sharded JAX array.

        Returns:
            Dictionary mapping device ID strings to shard arrays.
        """
        result: dict[str, Any] = {}

        if not hasattr(arr, "addressable_shards"):
            return result

        for shard in arr.addressable_shards:
            device_id = str(shard.device)
            result[device_id] = shard.data

        return result

    @staticmethod
    def _flatten_with_paths(
        pytree: Any, prefix: str = ""
    ) -> list[tuple[str, Any]]:
        """Flatten a pytree with string paths."""
        result: list[tuple[str, Any]] = []

        def _traverse(obj: Any, path: str) -> None:
            if isinstance(obj, dict):
                for key in sorted(obj.keys()):
                    child = f"{path}.{key}" if path else str(key)
                    _traverse(obj[key], child)
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    _traverse(item, f"{path}[{i}]")
            else:
                result.append((path if path else "param", obj))

        _traverse(pytree, prefix)
        return result

    @property
    def step_count(self) -> int:
        """Number of steps processed."""
        return self._step_count

    @property
    def anomalies(self) -> list[AnomalyScore]:
        """All detected anomalies."""
        return list(self._anomalies)

    def reset(self) -> None:
        """Reset all state."""
        self._step_count = 0
        self._anomalies.clear()
        self._per_device_trackers.clear()
