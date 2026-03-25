"""Tests for loss trajectory monitoring: AR prediction, anomaly detection, NaN handling."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel_training.common.anomaly_detector import AnomalyType, ARPredictor
from sentinel_training.common.config import LossTrackingConfig
from sentinel_training.pytorch.loss_monitor import LossMonitor


class TestARPredictor:
    """Tests for the autoregressive loss predictor."""

    def test_initialization(self) -> None:
        predictor = ARPredictor(order=5)
        assert predictor.order == 5
        assert not predictor.is_ready
        assert predictor.predict() is None

    def test_becomes_ready_after_min_samples(self) -> None:
        predictor = ARPredictor(order=3, min_samples=10)
        for i in range(9):
            predictor.update(float(i))
            assert not predictor.is_ready
        predictor.update(9.0)
        assert predictor.is_ready

    def test_predicts_linear_trend(self) -> None:
        predictor = ARPredictor(order=3, min_samples=20)
        # Feed a linear sequence: loss = 10 - 0.1 * step
        for step in range(50):
            value = 10.0 - 0.1 * step
            predictor.update(value)

        predicted = predictor.predict()
        assert predicted is not None
        expected = 10.0 - 0.1 * 50
        # Should be reasonably close to the linear trend
        assert abs(predicted - expected) < 1.0

    def test_residual_z_score_for_normal(self) -> None:
        predictor = ARPredictor(order=5, min_samples=30)
        rng = np.random.RandomState(42)
        # Feed slowly decreasing loss with noise
        for step in range(100):
            value = 5.0 - 0.01 * step + rng.normal(0, 0.05)
            predictor.update(value)

        # A value close to the trend should have low z-score
        z = predictor.residual_z_score(5.0 - 0.01 * 100)
        assert z < 3.0

    def test_residual_z_score_for_outlier(self) -> None:
        predictor = ARPredictor(order=5, min_samples=30)
        rng = np.random.RandomState(42)
        for step in range(100):
            value = 5.0 - 0.01 * step + rng.normal(0, 0.05)
            predictor.update(value)

        # A value far from the trend should have high z-score
        z = predictor.residual_z_score(100.0)
        assert z > 5.0

    def test_non_finite_values_skipped(self) -> None:
        predictor = ARPredictor(order=3, min_samples=5)
        for i in range(10):
            predictor.update(float(i))
        count_before = len(predictor._history)
        predictor.update(float("nan"))
        assert len(predictor._history) == count_before

    def test_reset(self) -> None:
        predictor = ARPredictor(order=3, min_samples=5)
        for i in range(10):
            predictor.update(float(i))
        predictor.reset()
        assert not predictor.is_ready
        assert predictor.predict() is None


class TestLossMonitor:
    """Tests for the loss trajectory monitor."""

    def _make_config(self, **kwargs: object) -> LossTrackingConfig:
        defaults = {
            "ar_order": 5,
            "sigma_multiplier": 3.0,
            "warmup_steps": 20,
            "nan_detection": True,
            "plateau_window": 10,
            "plateau_tolerance": 1e-7,
        }
        defaults.update(kwargs)
        return LossTrackingConfig(**defaults)  # type: ignore[arg-type]

    def test_normal_loss_trajectory(self) -> None:
        config = self._make_config()
        monitor = LossMonitor(config)

        rng = np.random.RandomState(42)
        for step in range(100):
            loss = 5.0 - 0.01 * step + rng.normal(0, 0.05)
            anomalies = monitor.on_loss(loss)
            # During normal training, should detect no anomalies (after warmup stabilizes)

        # Most steps should be anomaly-free
        total_anomalies = len(monitor.recent_anomalies)
        assert total_anomalies < 5  # Allow a few false positives from randomness

    def test_nan_detection(self) -> None:
        config = self._make_config()
        monitor = LossMonitor(config)

        # Normal values first
        for step in range(30):
            monitor.on_loss(1.0 - 0.01 * step)

        # NaN loss
        anomalies = monitor.on_loss(float("nan"))
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.LOSS_NAN
        assert monitor.nan_count == 1

    def test_inf_detection(self) -> None:
        config = self._make_config()
        monitor = LossMonitor(config)

        for step in range(30):
            monitor.on_loss(1.0 - 0.01 * step)

        anomalies = monitor.on_loss(float("inf"))
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.LOSS_NAN

    def test_loss_spike_detection(self) -> None:
        config = self._make_config(warmup_steps=15, sigma_multiplier=3.0)
        monitor = LossMonitor(config)

        rng = np.random.RandomState(42)
        # Build a stable baseline
        for step in range(50):
            loss = 2.0 - 0.005 * step + rng.normal(0, 0.02)
            monitor.on_loss(loss)

        # Inject a massive spike
        anomalies = monitor.on_loss(100.0)
        assert len(anomalies) > 0
        spike_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.LOSS_SPIKE]
        assert len(spike_anomalies) > 0

    def test_plateau_detection(self) -> None:
        config = self._make_config(
            warmup_steps=5,
            plateau_window=10,
            plateau_tolerance=1e-7,
        )
        monitor = LossMonitor(config)

        # Normal decrease first
        for step in range(10):
            monitor.on_loss(5.0 - 0.1 * step)

        # Now flat plateau at a non-zero value
        for step in range(20):
            anomalies = monitor.on_loss(3.5)

        # Should eventually detect plateau
        plateau_anomalies = [
            a for a in monitor.recent_anomalies
            if a.anomaly_type == AnomalyType.LOSS_PLATEAU
        ]
        assert len(plateau_anomalies) > 0

    def test_multi_task_tracking(self) -> None:
        config = self._make_config(multi_task=True, warmup_steps=10)
        monitor = LossMonitor(config)

        rng = np.random.RandomState(42)
        for step in range(50):
            loss_a = 3.0 - 0.01 * step + rng.normal(0, 0.02)
            loss_b = 5.0 - 0.02 * step + rng.normal(0, 0.03)
            monitor.on_loss(loss_a, task_name="task_a")
            monitor.on_loss(loss_b, task_name="task_b")

        # Both tasks should have been tracked
        assert monitor.step_count == 100  # 50 steps * 2 tasks

    def test_step_counting(self) -> None:
        config = self._make_config()
        monitor = LossMonitor(config)

        assert monitor.step_count == 0
        monitor.on_loss(1.0)
        assert monitor.step_count == 1
        monitor.on_loss(0.9)
        assert monitor.step_count == 2

    def test_loss_history(self) -> None:
        config = self._make_config()
        monitor = LossMonitor(config)

        values = [3.0, 2.5, 2.0, 1.5, 1.0]
        for v in values:
            monitor.on_loss(v)

        assert monitor.loss_history == values

    def test_reset(self) -> None:
        config = self._make_config()
        monitor = LossMonitor(config)

        for step in range(20):
            monitor.on_loss(float(step))

        monitor.reset()
        assert monitor.step_count == 0
        assert len(monitor.loss_history) == 0
        assert len(monitor.recent_anomalies) == 0
        assert monitor.nan_count == 0

    def test_multiple_nan_counting(self) -> None:
        config = self._make_config()
        monitor = LossMonitor(config)

        monitor.on_loss(float("nan"))
        monitor.on_loss(float("nan"))
        monitor.on_loss(float("nan"))
        assert monitor.nan_count == 3
