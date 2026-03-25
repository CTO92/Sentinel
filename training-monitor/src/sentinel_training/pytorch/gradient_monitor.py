"""Per-layer gradient norm monitoring for PyTorch models.

Tracks gradient L2 norms using EWMA baselines to detect sudden spikes or
collapses that may indicate silent data corruption in the backward pass.
"""

from __future__ import annotations

import math

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
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class GradientMonitor:
    """Monitors per-layer gradient norms to detect SDC in backward passes.

    Maintains EWMA trackers for each parameter's gradient L2 norm and flags
    anomalies when norms deviate significantly from baseline.

    Args:
        config: Gradient norm monitoring configuration.
        metrics: Optional metrics reporter for Prometheus integration.
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
        self._gradient_accumulation_steps: int = 1
        self._accumulated_norms: dict[str, list[float]] = {}

    def set_gradient_accumulation(self, steps: int) -> None:
        """Configure gradient accumulation awareness.

        Args:
            steps: Number of micro-steps per optimizer step.
        """
        self._gradient_accumulation_steps = max(1, steps)

    def _get_tracker(self, name: str) -> EWMATracker:
        """Get or create an EWMA tracker for a named parameter.

        Args:
            name: Parameter/layer name.

        Returns:
            The EWMA tracker for this parameter.
        """
        if name not in self._trackers:
            self._trackers[name] = EWMATracker(
                alpha=self._config.ewma_lambda,
                warmup_steps=self._config.warmup_steps,
            )
        return self._trackers[name]

    def on_gradient(self, name: str, grad: "torch.Tensor") -> AnomalyScore | None:
        """Process a gradient tensor for a named parameter.

        Computes the L2 norm, updates the EWMA tracker, and checks for anomalies.

        Args:
            name: The parameter name (e.g., 'layer1.weight').
            grad: The gradient tensor.

        Returns:
            An AnomalyScore if an anomaly is detected, else None.
        """
        if grad is None:
            return None

        # Handle mixed precision: convert to float32 for norm computation
        if self._config.mixed_precision_aware and grad.dtype != torch.float32:
            grad_for_norm = grad.float()
        else:
            grad_for_norm = grad

        norm = float(torch.linalg.vector_norm(grad_for_norm).item())

        # Handle gradient accumulation: collect norms and average
        if self._gradient_accumulation_steps > 1:
            if name not in self._accumulated_norms:
                self._accumulated_norms[name] = []
            self._accumulated_norms[name].append(norm)
            if len(self._accumulated_norms[name]) < self._gradient_accumulation_steps:
                return None
            norm = sum(self._accumulated_norms[name]) / len(self._accumulated_norms[name])
            self._accumulated_norms[name] = []

        # Report metric
        if self._metrics is not None:
            self._metrics.record_gradient_norm(name, norm)

        tracker = self._get_tracker(name)

        # Compute z-score before updating
        z = tracker.z_score(norm)
        tracker.update(norm)

        if not tracker.is_warmed_up:
            return None

        # Check for anomaly
        if z > self._config.sigma_multiplier:
            # Determine if this is a spike or collapse
            if norm > tracker.mean:
                anomaly_type = AnomalyType.GRADIENT_SPIKE
            else:
                anomaly_type = AnomalyType.GRADIENT_COLLAPSE

            score = AnomalyScore(
                anomaly_type=anomaly_type,
                score=z,
                threshold=self._config.sigma_multiplier,
                observed_value=norm,
                expected_value=tracker.mean,
                step=self._step_count,
                layer_name=name,
                metadata={
                    "ewma_std": tracker.std,
                    "ewma_count": tracker.count,
                },
            )
            self._anomalies.append(score)
            logger.warning(
                "gradient_anomaly",
                type=anomaly_type.value,
                layer=name,
                z_score=z,
                norm=norm,
                ewma_mean=tracker.mean,
                step=self._step_count,
            )
            return score

        return None

    def on_step(self) -> list[AnomalyScore]:
        """Called at the end of each training step.

        Returns:
            List of anomalies detected during this step.
        """
        self._step_count += 1
        step_anomalies = [
            a for a in self._anomalies if a.step == self._step_count - 1
        ]
        return step_anomalies

    def check_all_gradients(
        self, model: "nn.Module"
    ) -> list[AnomalyScore]:
        """Check gradient norms for all parameters of a model.

        Convenience method to process all parameter gradients at once.

        Args:
            model: The PyTorch model to inspect.

        Returns:
            List of anomalies detected.
        """
        anomalies: list[AnomalyScore] = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                result = self.on_gradient(name, param.grad)
                if result is not None:
                    anomalies.append(result)
        return anomalies

    def get_norms(self) -> dict[str, float]:
        """Get current EWMA means for all tracked parameters.

        Returns:
            Dictionary mapping parameter names to their EWMA mean gradient norms.
        """
        return {name: tracker.mean for name, tracker in self._trackers.items()}

    def reset(self) -> None:
        """Reset all trackers and state."""
        self._trackers.clear()
        self._anomalies.clear()
        self._step_count = 0
        self._accumulated_norms.clear()

    @property
    def step_count(self) -> int:
        """Number of steps processed."""
        return self._step_count

    @property
    def recent_anomalies(self) -> list[AnomalyScore]:
        """All recorded anomalies."""
        return list(self._anomalies)
