"""End-to-end integration test for the Inference Monitor pipeline.

Uses a mock interceptor that yields synthetic tensors and a mock gRPC
client to verify that the full pipeline (capture -> analyze -> report)
works correctly.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence

import numpy as np
import pytest

from sentinel_inference.analyzers.logit_analyzer import AnomalyEvent
from sentinel_inference.config import (
    EWMAConfig,
    GRPCConfig,
    HealthCheckConfig,
    LogitAnalyzerConfig,
    MetricsConfig,
    MonitorConfig,
    SamplingConfig,
)
from sentinel_inference.interceptors.base import BaseInterceptor, TensorCapture, compute_input_hash
from sentinel_inference.monitor import InferenceMonitor


class MockInterceptor(BaseInterceptor):
    """Interceptor that yields tensors from a pre-loaded sequence."""

    def __init__(
        self,
        tensors: list[np.ndarray],
        *,
        gpu_id: int = 0,
        sample_rate: float = 1.0,
    ) -> None:
        super().__init__(gpu_id=gpu_id, sample_rate=sample_rate)
        self._tensors = tensors
        self._index = 0
        self._exhausted = asyncio.Event()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    @property
    def exhausted(self) -> asyncio.Event:
        return self._exhausted

    async def capture_output(self) -> TensorCapture | None:
        if self._index >= len(self._tensors):
            self._exhausted.set()
            await asyncio.sleep(0.5)
            return None

        tensor = self._tensors[self._index]
        self._index += 1
        return self._make_capture(
            tensor=tensor,
            request_id=f"mock_req_{self._index}",
            input_hash=compute_input_hash(tensor),
            model_name="mock_model",
        )


class MockGRPCClient:
    """Mock gRPC client that collects submitted anomaly events."""

    def __init__(self) -> None:
        self.events: list[AnomalyEvent] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def submit(self, events: Sequence[AnomalyEvent]) -> None:
        self.events.extend(events)


@pytest.fixture
def normal_tensors() -> list[np.ndarray]:
    """Generate a sequence of normal logit tensors."""
    rng = np.random.default_rng(42)
    return [rng.normal(0, 1, size=500).astype(np.float32) for _ in range(80)]


@pytest.fixture
def corrupted_tensors() -> list[np.ndarray]:
    """Generate logit tensors that transition from normal to corrupted."""
    rng = np.random.default_rng(42)
    normal = [rng.normal(0, 1, size=500).astype(np.float32) for _ in range(60)]
    # Corrupted: massive mean shift
    corrupted = [rng.normal(50, 1, size=500).astype(np.float32) for _ in range(40)]
    return normal + corrupted


def _make_config(health_port: int, metrics_port: int) -> MonitorConfig:
    """Build a config with short burn-in for testing."""
    return MonitorConfig(
        node_id="test_node",
        gpu_id=0,
        replica_id="test_replica",
        sampling=SamplingConfig(rate=1.0, max_queue_size=200, num_workers=1),
        logit_analyzer=LogitAnalyzerConfig(
            ewma=EWMAConfig(**{"lambda": 0.2, "L": 3.0, "burn_in": 30})
        ),
        grpc=GRPCConfig(endpoint="localhost:50099", batch_size=10, flush_interval_ms=100),
        health_check=HealthCheckConfig(port=health_port),
        metrics=MetricsConfig(port=metrics_port),
    )


@pytest.mark.asyncio
async def test_normal_data_no_anomalies(normal_tensors: list[np.ndarray]) -> None:
    """Normal data should produce zero or very few anomalies."""
    config = _make_config(health_port=18081, metrics_port=19091)
    monitor = InferenceMonitor(config)

    mock_interceptor = MockInterceptor(normal_tensors, sample_rate=1.0)
    mock_grpc = MockGRPCClient()
    monitor._interceptor = mock_interceptor  # type: ignore[assignment]
    monitor._grpc_client = mock_grpc  # type: ignore[assignment]

    await monitor.start()

    # Wait for all tensors to be processed
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(mock_interceptor.exhausted.wait(), timeout=10.0)

    # Give workers time to drain the queue
    await asyncio.sleep(1.0)
    await monitor.stop()

    # With L=3.0 and 80 normal samples, we expect very few false positives
    logit_anomalies = [e for e in mock_grpc.events if e.analyzer == "logit_analyzer"]
    assert len(logit_anomalies) < 5, f"Too many false positives: {len(logit_anomalies)}"


@pytest.mark.asyncio
async def test_corrupted_data_triggers_anomalies(
    corrupted_tensors: list[np.ndarray],
) -> None:
    """Shifted data after burn-in should trigger anomalies."""
    config = _make_config(health_port=18082, metrics_port=19092)
    monitor = InferenceMonitor(config)

    mock_interceptor = MockInterceptor(corrupted_tensors, sample_rate=1.0)
    mock_grpc = MockGRPCClient()
    monitor._interceptor = mock_interceptor  # type: ignore[assignment]
    monitor._grpc_client = mock_grpc  # type: ignore[assignment]

    await monitor.start()

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(mock_interceptor.exhausted.wait(), timeout=10.0)

    await asyncio.sleep(1.0)
    await monitor.stop()

    # Should have detected anomalies from the corrupted portion
    assert len(mock_grpc.events) > 0, "Expected anomalies from corrupted data"


@pytest.mark.asyncio
async def test_pipeline_processes_all_samples() -> None:
    """All enqueued samples should be processed."""
    rng = np.random.default_rng(99)
    tensors = [rng.normal(0, 1, size=100).astype(np.float32) for _ in range(20)]

    config = _make_config(health_port=18083, metrics_port=19093)
    monitor = InferenceMonitor(config)

    mock_interceptor = MockInterceptor(tensors, sample_rate=1.0)
    mock_grpc = MockGRPCClient()
    monitor._interceptor = mock_interceptor  # type: ignore[assignment]
    monitor._grpc_client = mock_grpc  # type: ignore[assignment]

    await monitor.start()

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(mock_interceptor.exhausted.wait(), timeout=10.0)

    await asyncio.sleep(1.0)
    await monitor.stop()

    # Queue should be empty after processing
    assert monitor._queue.empty()
