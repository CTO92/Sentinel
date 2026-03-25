"""JAX-idiomatic gradient norm monitoring.

Provides the same statistical anomaly detection as the PyTorch version
but operates on JAX pytrees and sharded arrays.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import structlog

from sentinel_training.common.anomaly_detector import (
    AnomalyScore,
    AnomalyType,
    EWMATracker,
)
from sentinel_training.common.config import GradientNormConfig
from sentinel_training.metrics import MetricsReporter

logger = structlog.get_logger(__name__)

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]


class JAXGradientMonitor:
    """Monitors gradient norms across JAX pytrees.

    Tracks per-leaf gradient L2 norms using EWMA baselines and detects
    anomalies that may indicate SDC.

    Args:
        config: Gradient norm monitoring configuration.
        metrics: Optional Prometheus metrics reporter.
    """

    def __init__(
        self,
        config: GradientNormConfig,
        metrics: MetricsReporter | None = None,
    ) -> None:
        self._config = config
        self._metrics = metrics
        self._trackers: dict[str, EWMATracker] = {}
        self._step_count: int = 0
        self._anomalies: list[AnomalyScore] = []

    def _get_tracker(self, name: str) -> EWMATracker:
        """Get or create an EWMA tracker for a leaf parameter."""
        if name not in self._trackers:
            self._trackers[name] = EWMATracker(
                alpha=self._config.ewma_lambda,
                warmup_steps=self._config.warmup_steps,
            )
        return self._trackers[name]

    def check_gradients(
        self,
        grads: Any,
        prefix: str = "",
    ) -> list[AnomalyScore]:
        """Check gradient norms for all leaves in a pytree.

        Flattens the pytree, computes L2 norms for each leaf, and checks
        against EWMA baselines.

        Args:
            grads: A JAX pytree of gradient arrays.
            prefix: Optional prefix for leaf naming.

        Returns:
            List of detected anomalies.
        """
        if jax is None:
            return []

        self._step_count += 1
        anomalies: list[AnomalyScore] = []

        leaves_with_paths = self._flatten_with_paths(grads, prefix)

        for path, leaf in leaves_with_paths:
            if leaf is None:
                continue

            norm = float(jnp.linalg.norm(jnp.ravel(leaf)))

            if self._metrics is not None:
                self._metrics.record_gradient_norm(path, norm)

            tracker = self._get_tracker(path)
            z = tracker.z_score(norm)
            tracker.update(norm)

            if not tracker.is_warmed_up:
                continue

            if z > self._config.sigma_multiplier:
                anomaly_type = (
                    AnomalyType.GRADIENT_SPIKE
                    if norm > tracker.mean
                    else AnomalyType.GRADIENT_COLLAPSE
                )
                anomaly = AnomalyScore(
                    anomaly_type=anomaly_type,
                    score=z,
                    threshold=self._config.sigma_multiplier,
                    observed_value=norm,
                    expected_value=tracker.mean,
                    step=self._step_count,
                    layer_name=path,
                    metadata={"ewma_std": tracker.std},
                )
                self._anomalies.append(anomaly)
                anomalies.append(anomaly)
                logger.warning(
                    "jax_gradient_anomaly",
                    type=anomaly_type.value,
                    path=path,
                    z_score=z,
                    norm=norm,
                    step=self._step_count,
                )

        return anomalies

    @staticmethod
    def _flatten_with_paths(
        pytree: Any, prefix: str = ""
    ) -> list[tuple[str, Any]]:
        """Flatten a pytree and generate string paths for each leaf.

        Args:
            pytree: A JAX pytree.
            prefix: Path prefix.

        Returns:
            List of (path, leaf) tuples.
        """
        result: list[tuple[str, Any]] = []

        def _traverse(obj: Any, path: str) -> None:
            if isinstance(obj, dict):
                for key in sorted(obj.keys()):
                    child_path = f"{path}.{key}" if path else str(key)
                    _traverse(obj[key], child_path)
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    child_path = f"{path}[{i}]"
                    _traverse(item, child_path)
            else:
                result.append((path if path else "param", obj))

        _traverse(pytree, prefix)
        return result

    def get_norms(self) -> dict[str, float]:
        """Get current EWMA mean norms for all tracked leaves.

        Returns:
            Dictionary mapping leaf paths to EWMA means.
        """
        return {name: t.mean for name, t in self._trackers.items()}

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
        self._trackers.clear()
        self._step_count = 0
        self._anomalies.clear()
