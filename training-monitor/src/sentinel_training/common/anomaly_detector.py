"""Shared anomaly detection primitives: EWMA tracking, anomaly scoring, and alerting."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class AnomalyType(str, Enum):
    """Categories of training anomalies."""

    GRADIENT_SPIKE = "gradient_spike"
    GRADIENT_COLLAPSE = "gradient_collapse"
    LOSS_SPIKE = "loss_spike"
    LOSS_PLATEAU = "loss_plateau"
    LOSS_NAN = "loss_nan"
    CROSS_RANK_DIVERGENCE = "cross_rank_divergence"
    CHECKPOINT_DIVERGENCE = "checkpoint_divergence"


@dataclass
class AnomalyScore:
    """A scored anomaly event with metadata."""

    anomaly_type: AnomalyType
    score: float
    threshold: float
    observed_value: float
    expected_value: float
    timestamp: float = field(default_factory=time.time)
    step: int = 0
    rank: int | None = None
    layer_name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_anomalous(self) -> bool:
        """Whether the score exceeds the threshold."""
        return self.score > self.threshold

    def to_dict(self) -> dict[str, object]:
        """Serialize to a dictionary for gRPC transmission."""
        return {
            "anomaly_type": self.anomaly_type.value,
            "score": self.score,
            "threshold": self.threshold,
            "observed_value": self.observed_value,
            "expected_value": self.expected_value,
            "timestamp": self.timestamp,
            "step": self.step,
            "rank": self.rank,
            "layer_name": self.layer_name,
            "metadata": self.metadata,
        }


class EWMATracker:
    """Exponentially Weighted Moving Average tracker with variance estimation.

    Maintains a running mean and variance using the EWMA update rule. The
    variance is used to compute z-scores for anomaly detection.

    Args:
        alpha: Smoothing factor in (0, 1]. Lower values track longer history.
        warmup_steps: Number of observations before anomaly detection activates.
    """

    def __init__(self, alpha: float = 0.05, warmup_steps: int = 50) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._warmup_steps = warmup_steps
        self._mean: float = 0.0
        self._var: float = 0.0
        self._count: int = 0
        self._initialized: bool = False

    @property
    def mean(self) -> float:
        """Current EWMA mean estimate."""
        return self._mean

    @property
    def std(self) -> float:
        """Current EWMA standard deviation estimate."""
        return math.sqrt(max(self._var, 1e-12))

    @property
    def count(self) -> int:
        """Number of observations seen."""
        return self._count

    @property
    def is_warmed_up(self) -> bool:
        """Whether enough observations have been collected."""
        return self._count >= self._warmup_steps

    def update(self, value: float) -> None:
        """Update the EWMA mean and variance with a new observation.

        Args:
            value: The new observation to incorporate.
        """
        if not math.isfinite(value):
            logger.warning("ewma_non_finite_value", value=value, count=self._count)
            return

        self._count += 1
        if not self._initialized:
            self._mean = value
            self._var = 0.0
            self._initialized = True
            return

        delta = value - self._mean
        self._mean = self._mean + self._alpha * delta
        # Welford-like EWMA variance update
        self._var = (1.0 - self._alpha) * (self._var + self._alpha * delta * delta)

    def z_score(self, value: float) -> float:
        """Compute the z-score of a value relative to the current distribution.

        Args:
            value: The observation to score.

        Returns:
            The absolute z-score. Returns 0.0 if not yet warmed up or std is near zero.
        """
        if not self.is_warmed_up:
            return 0.0
        std = self.std
        if std < 1e-12:
            return 0.0
        return abs(value - self._mean) / std

    def reset(self) -> None:
        """Reset the tracker to its initial state."""
        self._mean = 0.0
        self._var = 0.0
        self._count = 0
        self._initialized = False


class AnomalyDetector:
    """Composite anomaly detector that aggregates signals from multiple sources.

    Combines individual anomaly scores using configurable weights and maintains
    a time-windowed history for temporal aggregation.

    Args:
        window_seconds: Duration of the sliding window for anomaly aggregation.
        composite_threshold: Threshold on the weighted composite score.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        composite_threshold: float = 1.0,
    ) -> None:
        self._window_seconds = window_seconds
        self._composite_threshold = composite_threshold
        self._history: deque[AnomalyScore] = deque()
        self._type_weights: dict[AnomalyType, float] = {
            AnomalyType.GRADIENT_SPIKE: 1.0,
            AnomalyType.GRADIENT_COLLAPSE: 1.5,
            AnomalyType.LOSS_SPIKE: 0.8,
            AnomalyType.LOSS_PLATEAU: 0.5,
            AnomalyType.LOSS_NAN: 2.0,
            AnomalyType.CROSS_RANK_DIVERGENCE: 2.0,
            AnomalyType.CHECKPOINT_DIVERGENCE: 1.5,
        }

    def record(self, score: AnomalyScore) -> None:
        """Record an anomaly score and prune old entries.

        Args:
            score: The anomaly score to record.
        """
        self._history.append(score)
        self._prune()

    def _prune(self) -> None:
        """Remove entries outside the time window."""
        cutoff = time.time() - self._window_seconds
        while self._history and self._history[0].timestamp < cutoff:
            self._history.popleft()

    def composite_score(self) -> float:
        """Compute a weighted composite anomaly score over the current window.

        Returns:
            The weighted composite score. Higher values indicate more anomalous behavior.
        """
        self._prune()
        if not self._history:
            return 0.0

        total = 0.0
        for entry in self._history:
            if entry.is_anomalous:
                weight = self._type_weights.get(entry.anomaly_type, 1.0)
                # Scale by how far above threshold
                excess = (entry.score - entry.threshold) / max(entry.threshold, 1e-12)
                total += weight * excess
        return total

    def is_sdc_suspected(self) -> bool:
        """Whether the composite score suggests Silent Data Corruption.

        Returns:
            True if composite score exceeds the configured threshold.
        """
        return self.composite_score() > self._composite_threshold

    def recent_anomalies(
        self, anomaly_type: AnomalyType | None = None
    ) -> list[AnomalyScore]:
        """Return recent anomalies, optionally filtered by type.

        Args:
            anomaly_type: If provided, only return anomalies of this type.

        Returns:
            List of recent anomaly scores within the time window.
        """
        self._prune()
        if anomaly_type is None:
            return list(self._history)
        return [s for s in self._history if s.anomaly_type == anomaly_type]

    def summarize(self) -> dict[str, object]:
        """Produce a summary of the current anomaly window.

        Returns:
            Dictionary with composite score, anomaly counts, and suspect status.
        """
        self._prune()
        counts: dict[str, int] = {}
        for entry in self._history:
            if entry.is_anomalous:
                key = entry.anomaly_type.value
                counts[key] = counts.get(key, 0) + 1
        return {
            "composite_score": self.composite_score(),
            "sdc_suspected": self.is_sdc_suspected(),
            "anomaly_counts": counts,
            "window_size": len(self._history),
        }


class ARPredictor:
    """Autoregressive predictor for time series (used by loss monitor).

    Fits an AR(p) model using least-squares on a sliding window of observations.

    Args:
        order: The AR model order (number of lagged terms).
        min_samples: Minimum observations before prediction is available.
    """

    def __init__(self, order: int = 5, min_samples: int | None = None) -> None:
        self._order = order
        self._min_samples = min_samples if min_samples is not None else order + 10
        self._history: list[float] = []
        self._coefficients: np.ndarray | None = None
        self._residual_std: float = 0.0

    @property
    def order(self) -> int:
        """The AR model order."""
        return self._order

    @property
    def residual_std(self) -> float:
        """Standard deviation of the residuals from the last fit."""
        return self._residual_std

    @property
    def is_ready(self) -> bool:
        """Whether enough data has been collected for prediction."""
        return len(self._history) >= self._min_samples

    def update(self, value: float) -> None:
        """Add a new observation and re-fit the model.

        Args:
            value: The new time series value.
        """
        if not math.isfinite(value):
            return
        self._history.append(value)
        if len(self._history) >= self._min_samples:
            self._fit()

    def _fit(self) -> None:
        """Fit the AR coefficients using ordinary least squares."""
        data = np.array(self._history, dtype=np.float64)
        n = len(data)
        p = self._order

        # Build the design matrix
        X = np.zeros((n - p, p), dtype=np.float64)
        y = data[p:]
        for i in range(p):
            X[:, i] = data[p - i - 1 : n - i - 1]

        # Solve via least squares with regularization
        try:
            reg = 1e-8 * np.eye(p)
            self._coefficients = np.linalg.solve(X.T @ X + reg, X.T @ y)
            residuals = y - X @ self._coefficients
            self._residual_std = float(np.std(residuals)) if len(residuals) > 0 else 0.0
        except np.linalg.LinAlgError:
            logger.warning("ar_fit_failed", history_len=n)
            self._coefficients = None
            self._residual_std = 0.0

    def predict(self) -> float | None:
        """Predict the next value based on the AR model.

        Returns:
            The predicted next value, or None if the model is not ready.
        """
        if self._coefficients is None or len(self._history) < self._order:
            return None
        recent = np.array(self._history[-self._order :], dtype=np.float64)[::-1]
        return float(np.dot(self._coefficients, recent))

    def residual_z_score(self, observed: float) -> float:
        """Compute the z-score of an observation relative to the AR prediction.

        Args:
            observed: The actual observed value.

        Returns:
            The absolute z-score. Returns 0.0 if not ready.
        """
        predicted = self.predict()
        if predicted is None or self._residual_std < 1e-12:
            return 0.0
        return abs(observed - predicted) / self._residual_std

    def reset(self) -> None:
        """Reset the predictor to its initial state."""
        self._history.clear()
        self._coefficients = None
        self._residual_std = 0.0
