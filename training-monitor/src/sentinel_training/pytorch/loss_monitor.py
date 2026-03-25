"""Loss trajectory monitoring for PyTorch training.

Uses an autoregressive model to predict expected loss values and detect
anomalies such as sudden spikes, implausible plateaus, and NaN propagation
that may indicate silent data corruption.
"""

from __future__ import annotations

import math
import time
from typing import Any

import structlog

from sentinel_training.common.anomaly_detector import (
    ARPredictor,
    AnomalyScore,
    AnomalyType,
)
from sentinel_training.common.config import LossTrackingConfig
from sentinel_training.metrics import MetricsReporter

logger = structlog.get_logger(__name__)


class LossMonitor:
    """Monitors training loss trajectory for anomalies indicating SDC.

    Maintains an AR(p) predictor of the loss curve and flags deviations,
    NaN values, and implausible plateaus.

    Args:
        config: Loss tracking configuration.
        metrics: Optional Prometheus metrics reporter.
    """

    def __init__(
        self,
        config: LossTrackingConfig,
        metrics: MetricsReporter | None = None,
    ) -> None:
        self._config = config
        self._metrics = metrics
        self._predictor = ARPredictor(
            order=config.ar_order,
            min_samples=config.warmup_steps,
        )
        self._task_predictors: dict[str, ARPredictor] = {}
        self._step_count: int = 0
        self._anomalies: list[AnomalyScore] = []
        self._loss_history: list[float] = []
        self._nan_count: int = 0

    def on_loss(
        self,
        loss_value: float,
        task_name: str | None = None,
    ) -> list[AnomalyScore]:
        """Process a loss value and check for anomalies.

        Args:
            loss_value: The computed training loss.
            task_name: Optional task name for multi-task training.

        Returns:
            List of anomalies detected for this loss value.
        """
        self._step_count += 1
        anomalies: list[AnomalyScore] = []

        # Record metric
        if self._metrics is not None:
            if math.isfinite(loss_value):
                self._metrics.record_loss(loss_value)

        # NaN detection
        if self._config.nan_detection and not math.isfinite(loss_value):
            self._nan_count += 1
            anomaly = AnomalyScore(
                anomaly_type=AnomalyType.LOSS_NAN,
                score=float("inf"),
                threshold=0.0,
                observed_value=loss_value if not math.isnan(loss_value) else 0.0,
                expected_value=self._predictor.predict() or 0.0,
                step=self._step_count,
                metadata={"nan_count": self._nan_count},
            )
            self._anomalies.append(anomaly)
            anomalies.append(anomaly)
            logger.error(
                "loss_nan_detected",
                step=self._step_count,
                nan_count=self._nan_count,
            )
            return anomalies

        self._loss_history.append(loss_value)

        # Choose predictor (multi-task or primary)
        if self._config.multi_task and task_name is not None:
            predictor = self._get_task_predictor(task_name)
        else:
            predictor = self._predictor

        # AR prediction anomaly check
        z_score = predictor.residual_z_score(loss_value)
        predictor.update(loss_value)

        if predictor.is_ready and z_score > self._config.sigma_multiplier:
            predicted = predictor.predict()
            anomaly = AnomalyScore(
                anomaly_type=AnomalyType.LOSS_SPIKE,
                score=z_score,
                threshold=self._config.sigma_multiplier,
                observed_value=loss_value,
                expected_value=predicted if predicted is not None else 0.0,
                step=self._step_count,
                metadata={
                    "residual_std": predictor.residual_std,
                    "task_name": task_name,
                },
            )
            self._anomalies.append(anomaly)
            anomalies.append(anomaly)
            logger.warning(
                "loss_spike_detected",
                step=self._step_count,
                z_score=z_score,
                observed=loss_value,
                predicted=predicted,
                task=task_name,
            )

        # Plateau detection
        plateau_anomaly = self._check_plateau()
        if plateau_anomaly is not None:
            self._anomalies.append(plateau_anomaly)
            anomalies.append(plateau_anomaly)

        return anomalies

    def _get_task_predictor(self, task_name: str) -> ARPredictor:
        """Get or create an AR predictor for a specific task.

        Args:
            task_name: The task identifier.

        Returns:
            ARPredictor for the given task.
        """
        if task_name not in self._task_predictors:
            self._task_predictors[task_name] = ARPredictor(
                order=self._config.ar_order,
                min_samples=self._config.warmup_steps,
            )
        return self._task_predictors[task_name]

    def _check_plateau(self) -> AnomalyScore | None:
        """Check for implausible loss plateaus.

        A plateau is detected when the loss variance over a window is
        suspiciously low, which may indicate the model is computing
        corrupted but constant outputs.

        Returns:
            AnomalyScore if a plateau is detected, else None.
        """
        window = self._config.plateau_window
        if len(self._loss_history) < window:
            return None

        recent = self._loss_history[-window:]
        mean_val = sum(recent) / len(recent)
        variance = sum((x - mean_val) ** 2 for x in recent) / len(recent)

        if variance < self._config.plateau_tolerance:
            # Only flag if we're past warmup and loss is not near zero
            # (near-zero loss with no variance is expected at convergence for some tasks)
            if self._step_count > self._config.warmup_steps and abs(mean_val) > 1e-3:
                anomaly = AnomalyScore(
                    anomaly_type=AnomalyType.LOSS_PLATEAU,
                    score=1.0 / max(variance, 1e-15),
                    threshold=1.0 / self._config.plateau_tolerance,
                    observed_value=mean_val,
                    expected_value=mean_val,
                    step=self._step_count,
                    metadata={
                        "variance": variance,
                        "window_size": window,
                    },
                )
                logger.warning(
                    "loss_plateau_detected",
                    step=self._step_count,
                    variance=variance,
                    mean_loss=mean_val,
                )
                return anomaly

        return None

    @property
    def step_count(self) -> int:
        """Number of steps processed."""
        return self._step_count

    @property
    def loss_history(self) -> list[float]:
        """Complete loss history."""
        return list(self._loss_history)

    @property
    def recent_anomalies(self) -> list[AnomalyScore]:
        """All recorded anomalies."""
        return list(self._anomalies)

    @property
    def nan_count(self) -> int:
        """Total NaN losses observed."""
        return self._nan_count

    def reset(self) -> None:
        """Reset all state."""
        self._predictor.reset()
        self._task_predictors.clear()
        self._step_count = 0
        self._anomalies.clear()
        self._loss_history.clear()
        self._nan_count = 0
