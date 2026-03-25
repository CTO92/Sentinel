"""Async gRPC client for streaming anomaly events to the Correlation Engine.

Batches events and sends them with exponential backoff on failure.
Supports mTLS for production deployments.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import structlog

from sentinel_inference.analyzers.logit_analyzer import AnomalyEvent
from sentinel_inference.config import GRPCConfig
from sentinel_inference.metrics import GRPC_ERRORS_TOTAL

logger = structlog.get_logger(__name__)


@dataclass
class AnomalyReport:
    """Wire-format anomaly report for gRPC transmission."""

    node_id: str
    gpu_id: int
    replica_id: str
    timestamp: float
    events: list[dict[str, Any]] = field(default_factory=list)


class CorrelationEngineClient:
    """Async gRPC client that batches and streams anomaly events.

    Events are accumulated in an internal buffer and flushed either when
    ``batch_size`` events are collected or after ``flush_interval_ms``
    milliseconds, whichever comes first.

    Parameters
    ----------
    config : GRPCConfig
        Connection and batching parameters.
    node_id : str
        Node identifier for reports.
    gpu_id : int
        GPU ordinal.
    replica_id : str
        Replica identifier.
    """

    def __init__(
        self,
        config: GRPCConfig,
        node_id: str = "unknown",
        gpu_id: int = 0,
        replica_id: str = "unknown",
    ) -> None:
        self._config = config
        self._node_id = node_id
        self._gpu_id = gpu_id
        self._replica_id = replica_id
        self._buffer: list[AnomalyEvent] = []
        self._lock = asyncio.Lock()
        self._channel: Any = None
        self._stub: Any = None
        self._running = False
        self._flush_task: asyncio.Task[None] | None = None
        self._consecutive_failures: int = 0

    async def start(self) -> None:
        """Open the gRPC channel and start the periodic flush task."""
        try:
            import grpc.aio  # type: ignore[import-untyped]

            if self._config.enable_mtls and self._config.ca_cert_path:
                with open(self._config.ca_cert_path, "rb") as f:
                    ca_cert = f.read()
                client_cert = None
                client_key = None
                if self._config.client_cert_path:
                    with open(self._config.client_cert_path, "rb") as f:
                        client_cert = f.read()
                if self._config.client_key_path:
                    with open(self._config.client_key_path, "rb") as f:
                        client_key = f.read()
                creds = grpc.ssl_channel_credentials(
                    root_certificates=ca_cert,
                    private_key=client_key,
                    certificate_chain=client_cert,
                )
                self._channel = grpc.aio.secure_channel(self._config.endpoint, creds)
            else:
                self._channel = grpc.aio.insecure_channel(self._config.endpoint)

            logger.info("grpc_client_started", endpoint=self._config.endpoint)
        except ImportError:
            logger.warning("grpc_not_available", msg="Running without gRPC reporting.")
            self._channel = None

        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())

    async def stop(self) -> None:
        """Flush remaining events and close the channel."""
        self._running = False
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final flush
        await self._flush()
        if self._channel is not None:
            await self._channel.close()
        logger.info("grpc_client_stopped")

    async def submit(self, events: Sequence[AnomalyEvent]) -> None:
        """Add anomaly events to the send buffer."""
        if not events:
            return
        async with self._lock:
            self._buffer.extend(events)
            if len(self._buffer) >= self._config.batch_size:
                await self._flush_locked()

    async def _periodic_flush(self) -> None:
        """Background task that flushes the buffer periodically."""
        interval = self._config.flush_interval_ms / 1000.0
        while self._running:
            await asyncio.sleep(interval)
            await self._flush()

    async def _flush(self) -> None:
        """Flush the buffer under lock."""
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        """Flush the buffer (caller holds the lock)."""
        if not self._buffer:
            return

        batch = self._buffer[:]
        self._buffer.clear()

        report = AnomalyReport(
            node_id=self._node_id,
            gpu_id=self._gpu_id,
            replica_id=self._replica_id,
            timestamp=time.time(),
            events=[
                {
                    "analyzer": e.analyzer,
                    "stat_name": e.stat_name,
                    "observed_value": e.observed_value,
                    "ewma_value": e.ewma_value,
                    "ucl": e.ucl,
                    "lcl": e.lcl,
                    "sample_count": e.sample_count,
                    "severity": e.severity,
                    "details": e.details,
                }
                for e in batch
            ],
        )

        await self._send_with_backoff(report)

    async def _send_with_backoff(self, report: AnomalyReport) -> None:
        """Send a report with exponential backoff on failure."""
        if self._channel is None:
            # No gRPC channel -- log only
            logger.info(
                "anomaly_report_local",
                event_count=len(report.events),
                node_id=report.node_id,
            )
            return

        backoff_ms = self._config.initial_backoff_ms
        for attempt in range(self._config.max_retries + 1):
            try:
                # In production this would call the generated stub.
                # Here we use a generic unary call pattern.
                await self._do_send(report)
                self._consecutive_failures = 0
                logger.debug(
                    "anomaly_report_sent",
                    event_count=len(report.events),
                    attempt=attempt,
                )
                return
            except Exception as exc:
                self._consecutive_failures += 1
                GRPC_ERRORS_TOTAL.labels(
                    node_id=self._node_id,
                    gpu_id=str(self._gpu_id),
                    error_type=type(exc).__name__,
                ).inc()
                logger.warning(
                    "grpc_send_failed",
                    attempt=attempt,
                    error=str(exc),
                    backoff_ms=backoff_ms,
                )
                if attempt < self._config.max_retries:
                    await asyncio.sleep(backoff_ms / 1000.0)
                    backoff_ms = min(backoff_ms * 2, self._config.max_backoff_ms)

        logger.error(
            "grpc_send_exhausted_retries",
            event_count=len(report.events),
            max_retries=self._config.max_retries,
        )

    async def _do_send(self, report: AnomalyReport) -> None:
        """Perform the actual gRPC call.

        This is a placeholder that serializes the report as JSON over a
        generic byte-stream call.  In production, this would use a
        protobuf-generated stub matching the Correlation Engine's service
        definition.
        """
        import json

        payload = json.dumps(asdict(report)).encode("utf-8")

        # Attempt a generic unary-unary call.  The actual RPC method name
        # would be defined by the Correlation Engine proto.
        try:
            import grpc  # type: ignore[import-untyped]

            call = self._channel.unary_unary(
                "/sentinel.CorrelationEngine/ReportAnomalies",
                request_serializer=lambda x: x,
                response_deserializer=lambda x: x,
            )
            await call(payload, timeout=5.0)
        except Exception:
            raise
