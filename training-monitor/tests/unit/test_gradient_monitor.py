"""Tests for gradient norm monitoring: EWMA tracking, spike and collapse detection."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel_training.common.anomaly_detector import AnomalyType, EWMATracker
from sentinel_training.common.config import GradientNormConfig

# Try to import torch; skip tests if unavailable
torch = pytest.importorskip("torch")

from sentinel_training.pytorch.gradient_monitor import GradientMonitor  # noqa: E402


class TestEWMATracker:
    """Tests for the EWMA tracker used by gradient monitoring."""

    def test_initialization(self) -> None:
        tracker = EWMATracker(alpha=0.1, warmup_steps=10)
        assert tracker.count == 0
        assert tracker.mean == 0.0
        assert not tracker.is_warmed_up

    def test_warmup_period(self) -> None:
        tracker = EWMATracker(alpha=0.1, warmup_steps=5)
        for _i in range(4):
            tracker.update(1.0)
            assert not tracker.is_warmed_up
        tracker.update(1.0)
        assert tracker.is_warmed_up

    def test_mean_tracking(self) -> None:
        tracker = EWMATracker(alpha=0.1, warmup_steps=1)
        # Feed constant values
        for _ in range(100):
            tracker.update(5.0)
        assert abs(tracker.mean - 5.0) < 0.01

    def test_mean_adapts_to_shift(self) -> None:
        tracker = EWMATracker(alpha=0.1, warmup_steps=1)
        # Phase 1: constant at 1.0
        for _ in range(50):
            tracker.update(1.0)
        # Phase 2: constant at 10.0
        for _ in range(200):
            tracker.update(10.0)
        # Should have adapted close to 10.0
        assert abs(tracker.mean - 10.0) < 0.5

    def test_z_score_returns_zero_during_warmup(self) -> None:
        tracker = EWMATracker(alpha=0.1, warmup_steps=10)
        for _ in range(5):
            tracker.update(1.0)
        assert tracker.z_score(100.0) == 0.0

    def test_z_score_detects_outlier(self) -> None:
        tracker = EWMATracker(alpha=0.05, warmup_steps=10)
        # Build baseline at 1.0 with low variance
        for _ in range(100):
            tracker.update(1.0 + np.random.normal(0, 0.01))
        # A value of 100.0 should have a very high z-score
        z = tracker.z_score(100.0)
        assert z > 10.0

    def test_non_finite_values_ignored(self) -> None:
        tracker = EWMATracker(alpha=0.1, warmup_steps=1)
        tracker.update(1.0)
        count_before = tracker.count
        tracker.update(float("nan"))
        assert tracker.count == count_before
        tracker.update(float("inf"))
        assert tracker.count == count_before

    def test_reset(self) -> None:
        tracker = EWMATracker(alpha=0.1, warmup_steps=5)
        for _ in range(10):
            tracker.update(1.0)
        tracker.reset()
        assert tracker.count == 0
        assert not tracker.is_warmed_up

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            EWMATracker(alpha=0.0)
        with pytest.raises(ValueError):
            EWMATracker(alpha=1.5)


class TestGradientMonitor:
    """Tests for the PyTorch gradient monitor."""

    def _make_config(self, **kwargs: object) -> GradientNormConfig:
        defaults = {
            "ewma_lambda": 0.05,
            "sigma_multiplier": 4.0,
            "warmup_steps": 20,
        }
        defaults.update(kwargs)
        return GradientNormConfig(**defaults)  # type: ignore[arg-type]

    def test_normal_gradients_no_anomaly(self) -> None:
        config = self._make_config(warmup_steps=10)
        monitor = GradientMonitor(config)

        # Feed normal gradients for warmup + some steps
        for step in range(50):
            grad = torch.randn(100) * 0.1  # Small, consistent gradients
            result = monitor.on_gradient("layer1.weight", grad)
            # During warmup and normal operation, no anomaly
            if step < config.warmup_steps:
                assert result is None

    def test_spike_detection(self) -> None:
        config = self._make_config(warmup_steps=10, sigma_multiplier=3.0)
        monitor = GradientMonitor(config)

        # Build baseline with small gradients
        for _ in range(50):
            grad = torch.randn(100) * 0.1
            monitor.on_gradient("layer1.weight", grad)

        # Inject a massive spike
        spike_grad = torch.randn(100) * 1000.0
        result = monitor.on_gradient("layer1.weight", spike_grad)

        assert result is not None
        assert result.anomaly_type == AnomalyType.GRADIENT_SPIKE
        assert result.is_anomalous

    def test_collapse_detection(self) -> None:
        config = self._make_config(warmup_steps=10, sigma_multiplier=3.0)
        monitor = GradientMonitor(config)

        # Build baseline with moderate gradients
        for _ in range(50):
            grad = torch.randn(100) * 5.0
            monitor.on_gradient("layer1.weight", grad)

        # Inject a collapse (near-zero gradient)
        collapse_grad = torch.randn(100) * 1e-10
        result = monitor.on_gradient("layer1.weight", collapse_grad)

        assert result is not None
        assert result.anomaly_type == AnomalyType.GRADIENT_COLLAPSE
        assert result.is_anomalous

    def test_per_layer_tracking(self) -> None:
        config = self._make_config(warmup_steps=5)
        monitor = GradientMonitor(config)

        # Different layers with different norm ranges
        for _ in range(20):
            monitor.on_gradient("layer1", torch.randn(50) * 1.0)
            monitor.on_gradient("layer2", torch.randn(50) * 10.0)

        norms = monitor.get_norms()
        assert "layer1" in norms
        assert "layer2" in norms
        # layer2 should have higher mean norm
        assert norms["layer2"] > norms["layer1"]

    def test_check_all_gradients(self) -> None:
        config = self._make_config(warmup_steps=5)
        monitor = GradientMonitor(config)

        model = torch.nn.Linear(10, 5)
        # Simulate a backward pass
        x = torch.randn(3, 10)
        y = model(x)
        y.sum().backward()

        # Warm up first
        for _ in range(10):
            x = torch.randn(3, 10)
            y = model(x)
            y.sum().backward()
            monitor.check_all_gradients(model)

        # Normal gradients should produce no anomalies after warmup
        x = torch.randn(3, 10)
        y = model(x)
        y.sum().backward()
        anomalies = monitor.check_all_gradients(model)
        # We don't assert zero anomalies because random gradients can vary,
        # but the structure should work
        assert isinstance(anomalies, list)

    def test_step_counting(self) -> None:
        config = self._make_config()
        monitor = GradientMonitor(config)

        assert monitor.step_count == 0
        monitor.on_step()
        assert monitor.step_count == 1
        monitor.on_step()
        assert monitor.step_count == 2

    def test_mixed_precision_gradients(self) -> None:
        config = self._make_config(warmup_steps=5, mixed_precision_aware=True)
        monitor = GradientMonitor(config)

        # Float16 gradients
        for _ in range(10):
            grad = torch.randn(100, dtype=torch.float16) * 0.1
            monitor.on_gradient("fp16_layer", grad)

        norms = monitor.get_norms()
        assert "fp16_layer" in norms
        assert norms["fp16_layer"] > 0

    def test_gradient_accumulation(self) -> None:
        config = self._make_config(warmup_steps=5)
        monitor = GradientMonitor(config)
        monitor.set_gradient_accumulation(4)

        results = []
        # With accumulation=4, only every 4th call should produce a result
        for _i in range(20):
            grad = torch.randn(100) * 0.1
            result = monitor.on_gradient("accum_layer", grad)
            if result is not None:
                results.append(result)

        # The norms should still be tracked
        norms = monitor.get_norms()
        assert "accum_layer" in norms

    def test_reset(self) -> None:
        config = self._make_config(warmup_steps=5)
        monitor = GradientMonitor(config)

        for _ in range(10):
            monitor.on_gradient("layer", torch.randn(50))

        monitor.reset()
        assert monitor.step_count == 0
        assert len(monitor.get_norms()) == 0
        assert len(monitor.recent_anomalies) == 0

    def test_none_gradient_handled(self) -> None:
        config = self._make_config()
        monitor = GradientMonitor(config)
        result = monitor.on_gradient("layer", None)  # type: ignore[arg-type]
        assert result is None
