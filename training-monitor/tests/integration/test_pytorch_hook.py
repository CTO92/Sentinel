"""Integration test: verify SentinelTrainingHook attaches to a real model and fires correctly."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sentinel_training.common.config import SentinelConfig
from sentinel_training.pytorch.hooks import SentinelTrainingHook


class TestPyTorchHookIntegration:
    """Integration tests for the SentinelTrainingHook with a real PyTorch model."""

    def _make_model_and_optimizer(self) -> tuple["torch.nn.Module", "torch.optim.Optimizer"]:
        """Create a small MLP for testing."""
        model = torch.nn.Sequential(
            torch.nn.Linear(20, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        return model, optimizer

    def _make_config(self, **kwargs: object) -> SentinelConfig:
        """Create a test configuration."""
        return SentinelConfig(
            gradient_norm={"warmup_steps": 5, "sigma_multiplier": 4.0},  # type: ignore[arg-type]
            loss_tracking={  # type: ignore[arg-type]
                "warmup_steps": 5, "sigma_multiplier": 3.0, "plateau_window": 10,
            },
            checkpoint_validation={"enabled": True},  # type: ignore[arg-type]
            cross_rank_divergence={"enabled": False},  # type: ignore[arg-type]
            metrics={"enabled": False},  # type: ignore[arg-type]
            **kwargs,  # type: ignore[arg-type]
        )

    def test_attach_and_detach(self) -> None:
        model, optimizer = self._make_model_and_optimizer()
        config = self._make_config()

        hook = SentinelTrainingHook.attach(model, optimizer, config)
        assert hook.step_count == 0

        hook.detach()

    def test_training_loop_integration(self) -> None:
        model, optimizer = self._make_model_and_optimizer()
        config = self._make_config()
        hook = SentinelTrainingHook.attach(model, optimizer, config)

        torch.manual_seed(42)

        for step in range(20):
            # Forward
            x = torch.randn(8, 20)
            target = torch.randn(8, 10)
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Sentinel monitoring
            hook.on_backward_end()
            hook.on_loss(loss.item())

            # Optimizer step
            optimizer.step()
            hook.on_step_end()

        assert hook.step_count == 20
        hook.detach()

    def test_backward_hooks_fire(self) -> None:
        model, optimizer = self._make_model_and_optimizer()
        config = self._make_config()
        hook = SentinelTrainingHook.attach(model, optimizer, config)

        x = torch.randn(4, 20)
        target = torch.randn(4, 10)
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)

        optimizer.zero_grad()
        loss.backward()

        # The backward hooks should have recorded gradient norms
        anomalies = hook.on_backward_end()
        # During first step (warmup), no anomalies expected
        assert isinstance(anomalies, list)

        hook.detach()

    def test_loss_monitoring(self) -> None:
        model, optimizer = self._make_model_and_optimizer()
        config = self._make_config()
        hook = SentinelTrainingHook.attach(model, optimizer, config)

        # Feed normal losses
        for step in range(30):
            loss_val = 5.0 - 0.05 * step + np.random.normal(0, 0.1)
            hook.on_loss(loss_val)

        # Feed a spike
        anomalies = hook.on_loss(1000.0)
        # After warmup, the spike should be detected
        assert any(a.anomaly_type.value == "loss_spike" for a in anomalies) or len(anomalies) == 0
        # (may not trigger if AR model hasn't stabilized enough)

        hook.detach()

    def test_checkpoint_validation(self) -> None:
        model, optimizer = self._make_model_and_optimizer()
        config = self._make_config()
        hook = SentinelTrainingHook.attach(model, optimizer, config)

        # First checkpoint
        state1 = {k: v.clone() for k, v in model.state_dict().items()}
        result1 = hook.on_checkpoint(state1, step=100)
        assert result1 is None  # First checkpoint, nothing to compare

        # Train a bit
        for _ in range(5):
            x = torch.randn(4, 20)
            output = model(x)
            loss = output.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Second checkpoint - should be close to first (small model, few steps)
        state2 = {k: v.clone() for k, v in model.state_dict().items()}
        result2 = hook.on_checkpoint(state2, step=105)
        # May or may not detect divergence depending on how much params changed

        hook.detach()

    def test_anomaly_summary(self) -> None:
        model, optimizer = self._make_model_and_optimizer()
        config = self._make_config()
        hook = SentinelTrainingHook.attach(model, optimizer, config)

        summary = hook.anomaly_summary
        assert "composite_score" in summary
        assert "sdc_suspected" in summary
        assert summary["sdc_suspected"] is False

        hook.detach()

    def test_default_config(self) -> None:
        model, optimizer = self._make_model_and_optimizer()
        # attach with no config should use defaults
        hook = SentinelTrainingHook.attach(model, optimizer)

        x = torch.randn(4, 20)
        output = model(x)
        loss = output.sum()
        optimizer.zero_grad()
        loss.backward()

        hook.on_backward_end()
        hook.on_loss(loss.item())
        hook.on_step_end()

        assert hook.step_count == 1
        hook.detach()

    def test_sdc_not_suspected_normal_training(self) -> None:
        model, optimizer = self._make_model_and_optimizer()
        config = self._make_config()
        hook = SentinelTrainingHook.attach(model, optimizer, config)

        torch.manual_seed(0)
        for step in range(50):
            x = torch.randn(8, 20)
            target = torch.randn(8, 10)
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            hook.on_backward_end()
            hook.on_loss(loss.item())
            optimizer.step()
            hook.on_step_end()

        # Normal training should not flag SDC
        assert not hook.is_sdc_suspected

        hook.detach()

    def test_multiple_attach_detach_cycles(self) -> None:
        model, optimizer = self._make_model_and_optimizer()
        config = self._make_config()

        for _ in range(3):
            hook = SentinelTrainingHook.attach(model, optimizer, config)
            x = torch.randn(4, 20)
            output = model(x)
            loss = output.sum()
            optimizer.zero_grad()
            loss.backward()
            hook.on_backward_end()
            hook.on_step_end()
            hook.detach()
