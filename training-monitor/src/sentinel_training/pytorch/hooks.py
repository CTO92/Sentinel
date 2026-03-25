"""Main entry point for PyTorch integration: SentinelTrainingHook.

Attaches to a PyTorch model and optimizer to monitor gradient norms, loss
trajectories, checkpoint integrity, and cross-rank divergence. Designed for
minimal overhead (~0.59ms per step).
"""

from __future__ import annotations

import time
from typing import Any, Callable

import structlog

from sentinel_training.common.anomaly_detector import AnomalyDetector, AnomalyScore
from sentinel_training.common.config import SentinelConfig
from sentinel_training.common.grpc_client import GrpcAnomalyClient
from sentinel_training.metrics import MetricsReporter
from sentinel_training.pytorch.checkpoint_validator import CheckpointValidator
from sentinel_training.pytorch.ddp_divergence import DDPDivergenceDetector
from sentinel_training.pytorch.gradient_monitor import GradientMonitor
from sentinel_training.pytorch.loss_monitor import LossMonitor

logger = structlog.get_logger(__name__)

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class SentinelTrainingHook:
    """Sentinel SDC detection hook for PyTorch training.

    Integrates into the training loop via callback methods and PyTorch hooks
    to monitor gradient norms, loss trajectories, checkpoint integrity, and
    cross-rank gradient divergence.

    Usage::

        hook = SentinelTrainingHook.attach(model, optimizer, config)
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            hook.on_backward_end()
            hook.on_loss(loss.item())
            optimizer.step()
            hook.on_step_end()
            if should_checkpoint:
                save_checkpoint(model)
                hook.on_checkpoint(model.state_dict(), step)

    Args:
        model: The PyTorch model to monitor.
        optimizer: The optimizer (used for param group tracking).
        config: Sentinel configuration.
    """

    def __init__(
        self,
        model: "nn.Module",
        optimizer: "torch.optim.Optimizer",
        config: SentinelConfig,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._config = config
        self._step_count: int = 0

        # Sub-monitors
        self._metrics = MetricsReporter(
            enabled=config.metrics.enabled,
            prefix=config.metrics.prefix,
        )
        self._gradient_monitor = GradientMonitor(
            config=config.gradient_norm,
            metrics=self._metrics,
        )
        self._loss_monitor = LossMonitor(
            config=config.loss_tracking,
            metrics=self._metrics,
        )
        self._checkpoint_validator = CheckpointValidator(
            config=config.checkpoint_validation,
        )

        # DDP divergence detector
        rank, world_size = self._detect_distributed()
        self._ddp_detector = DDPDivergenceDetector(
            config=config.cross_rank_divergence,
            rank=rank,
            world_size=world_size,
        )
        self._metrics.set_active_ranks(world_size)

        # Anomaly aggregation
        self._anomaly_detector = AnomalyDetector()

        # gRPC client (lazy start)
        self._grpc_client: GrpcAnomalyClient | None = None
        if config.grpc.endpoint:
            self._grpc_client = GrpcAnomalyClient(
                config=config.grpc,
                node_id=config.node_id,
                cluster_id=config.cluster_id,
            )

        # Hook handles for cleanup
        self._hook_handles: list[Any] = []
        self._registered = False

    @classmethod
    def attach(
        cls,
        model: "nn.Module",
        optimizer: "torch.optim.Optimizer",
        config: SentinelConfig | None = None,
    ) -> "SentinelTrainingHook":
        """Attach Sentinel monitoring to a model and optimizer.

        This is the main entry point. It creates the hook, registers
        PyTorch backward/forward hooks, and optionally starts the gRPC client.

        Args:
            model: The PyTorch model to monitor.
            optimizer: The optimizer.
            config: Optional configuration. Uses defaults if not provided.

        Returns:
            The attached SentinelTrainingHook instance.
        """
        if config is None:
            config = SentinelConfig()

        hook = cls(model, optimizer, config)
        hook._register_hooks()

        logger.info(
            "sentinel_attached",
            model_params=sum(p.numel() for p in model.parameters()),
            gradient_monitoring=config.gradient_norm.enabled,
            loss_monitoring=config.loss_tracking.enabled,
            checkpoint_validation=config.checkpoint_validation.enabled,
            cross_rank_divergence=config.cross_rank_divergence.enabled,
        )

        return hook

    def _register_hooks(self) -> None:
        """Register PyTorch forward and backward hooks on the model."""
        if self._registered:
            return

        if self._config.gradient_norm.enabled:
            # Register backward hooks on all modules to capture gradients
            for name, module in self._model.named_modules():
                handle = module.register_full_backward_hook(
                    self._make_backward_hook(name)
                )
                self._hook_handles.append(handle)

        self._registered = True
        logger.debug("hooks_registered", hook_count=len(self._hook_handles))

    def _make_backward_hook(
        self, module_name: str
    ) -> Callable[..., None]:
        """Create a backward hook closure for a specific module.

        Args:
            module_name: Name of the module this hook is attached to.

        Returns:
            Hook function compatible with register_full_backward_hook.
        """
        gradient_monitor = self._gradient_monitor

        def hook(
            module: "nn.Module",
            grad_input: tuple[Any, ...],
            grad_output: tuple[Any, ...],
        ) -> None:
            for i, g in enumerate(grad_output):
                if g is not None and isinstance(g, torch.Tensor):
                    layer_name = f"{module_name}.grad_output.{i}" if module_name else f"grad_output.{i}"
                    gradient_monitor.on_gradient(layer_name, g)

        return hook

    def on_backward_end(self) -> list[AnomalyScore]:
        """Called after loss.backward() completes.

        Checks all parameter gradients and DDP divergence.

        Returns:
            List of anomalies detected.
        """
        anomalies: list[AnomalyScore] = []

        # Check per-parameter gradients
        if self._config.gradient_norm.enabled:
            grad_anomalies = self._gradient_monitor.check_all_gradients(self._model)
            anomalies.extend(grad_anomalies)

        # Check cross-rank divergence
        if self._config.cross_rank_divergence.enabled:
            ddp_anomaly = self._ddp_detector.check_divergence(self._model)
            if ddp_anomaly is not None:
                anomalies.append(ddp_anomaly)

        for a in anomalies:
            self._anomaly_detector.record(a)

        return anomalies

    def on_loss(self, loss_value: float, task_name: str | None = None) -> list[AnomalyScore]:
        """Called with the loss value at each step.

        Args:
            loss_value: The computed loss.
            task_name: Optional task name for multi-task training.

        Returns:
            List of anomalies detected.
        """
        anomalies = self._loss_monitor.on_loss(loss_value, task_name)
        for a in anomalies:
            self._anomaly_detector.record(a)
        return anomalies

    def on_step_end(self) -> None:
        """Called at the end of each training step.

        Increments counters, reports metrics, and sends any anomalies.
        """
        start = time.perf_counter()

        self._step_count += 1
        self._gradient_monitor.on_step()
        self._metrics.record_step()

        # Report composite anomaly score
        composite = self._anomaly_detector.composite_score()
        self._metrics.record_composite_score(composite)

        # Send anomalies via gRPC if any
        if self._grpc_client is not None:
            recent = self._anomaly_detector.recent_anomalies()
            for anomaly in recent:
                if anomaly.is_anomalous and anomaly.step == self._step_count:
                    self._grpc_client.send(anomaly)
                    self._metrics.record_anomaly(anomaly.anomaly_type.value)

        overhead = time.perf_counter() - start
        self._metrics.record_overhead(overhead)

    def on_checkpoint(
        self, state_dict: dict[str, Any], step: int
    ) -> AnomalyScore | None:
        """Called after a checkpoint is saved.

        Args:
            state_dict: The model state dictionary.
            step: The current training step.

        Returns:
            AnomalyScore if checkpoint divergence is detected.
        """
        if not self._config.checkpoint_validation.enabled:
            return None

        anomaly = self._checkpoint_validator.validate(state_dict, step)
        if anomaly is not None:
            self._anomaly_detector.record(anomaly)
            self._metrics.record_anomaly(anomaly.anomaly_type.value)
        return anomaly

    def start_grpc(self) -> None:
        """Start the gRPC client for streaming anomalies."""
        if self._grpc_client is not None:
            self._grpc_client.start()

    def stop_grpc(self) -> None:
        """Stop the gRPC client."""
        if self._grpc_client is not None:
            self._grpc_client.stop()

    def detach(self) -> None:
        """Remove all hooks and clean up resources."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._registered = False
        self.stop_grpc()
        logger.info("sentinel_detached", steps_monitored=self._step_count)

    @property
    def is_sdc_suspected(self) -> bool:
        """Whether the anomaly detector suspects SDC."""
        return self._anomaly_detector.is_sdc_suspected()

    @property
    def step_count(self) -> int:
        """Number of training steps monitored."""
        return self._step_count

    @property
    def anomaly_summary(self) -> dict[str, object]:
        """Summary of current anomaly state."""
        return self._anomaly_detector.summarize()

    @staticmethod
    def _detect_distributed() -> tuple[int, int]:
        """Detect the current distributed training rank and world size.

        Returns:
            Tuple of (rank, world_size). Returns (0, 1) if not distributed.
        """
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
        except Exception:
            pass
        return 0, 1
