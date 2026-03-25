"""Prometheus metrics for Sentinel Training Monitor.

Exposes training-loop-level observability metrics for monitoring overhead,
anomaly rates, gradient norms, and loss values.
"""

from __future__ import annotations

from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram, Summary

logger = structlog.get_logger(__name__)

# -- Counters --

steps_monitored_total = Counter(
    "sentinel_training_steps_monitored_total",
    "Total number of training steps monitored by Sentinel",
)

anomalies_total = Counter(
    "sentinel_training_anomalies_total",
    "Total anomalies detected, labeled by type",
    ["type"],
)

recomputation_total = Counter(
    "sentinel_training_recomputation_total",
    "Total number of SDC recomputation verifications triggered",
)

# -- Gauges --

gradient_norm = Gauge(
    "sentinel_training_gradient_norm",
    "Current gradient L2 norm per layer",
    ["layer"],
)

loss_value = Gauge(
    "sentinel_training_loss_value",
    "Current training loss value",
)

composite_anomaly_score = Gauge(
    "sentinel_training_composite_anomaly_score",
    "Current composite anomaly score from the anomaly detector",
)

active_ranks = Gauge(
    "sentinel_training_active_ranks",
    "Number of active training ranks being monitored",
)

# -- Histograms / Summaries --

overhead_seconds = Histogram(
    "sentinel_training_overhead_seconds",
    "Per-step monitoring overhead in seconds",
    buckets=(0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1),
)

gradient_norm_distribution = Summary(
    "sentinel_training_gradient_norm_distribution",
    "Distribution of gradient norms across all layers",
)


class MetricsReporter:
    """Facade for reporting Sentinel metrics to Prometheus.

    Wraps the raw Prometheus metrics to provide a cleaner interface and
    ensure consistent labeling.

    Args:
        prefix: Metric name prefix (unused, kept for config compat).
        enabled: Whether metrics reporting is active.
    """

    def __init__(self, enabled: bool = True, prefix: str = "sentinel_training") -> None:
        self._enabled = enabled
        self._prefix = prefix

    def record_step(self) -> None:
        """Increment the step counter."""
        if self._enabled:
            steps_monitored_total.inc()

    def record_gradient_norm(self, layer_name: str, norm_value: float) -> None:
        """Record a gradient norm observation.

        Args:
            layer_name: Name of the model layer.
            norm_value: The L2 gradient norm value.
        """
        if self._enabled:
            gradient_norm.labels(layer=layer_name).set(norm_value)
            gradient_norm_distribution.observe(norm_value)

    def record_loss(self, value: float) -> None:
        """Record the current loss value.

        Args:
            value: The training loss.
        """
        if self._enabled:
            loss_value.set(value)

    def record_anomaly(self, anomaly_type: str) -> None:
        """Record an anomaly detection event.

        Args:
            anomaly_type: The type of anomaly detected.
        """
        if self._enabled:
            anomalies_total.labels(type=anomaly_type).inc()

    def record_recomputation(self) -> None:
        """Record a recomputation verification trigger."""
        if self._enabled:
            recomputation_total.inc()

    def record_overhead(self, seconds: float) -> None:
        """Record the per-step monitoring overhead.

        Args:
            seconds: Time spent in monitoring for this step.
        """
        if self._enabled:
            overhead_seconds.observe(seconds)

    def record_composite_score(self, score: float) -> None:
        """Record the current composite anomaly score.

        Args:
            score: The composite anomaly score.
        """
        if self._enabled:
            composite_anomaly_score.set(score)

    def set_active_ranks(self, count: int) -> None:
        """Set the number of active ranks being monitored.

        Args:
            count: Number of active training ranks.
        """
        if self._enabled:
            active_ranks.set(count)
