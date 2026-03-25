"""Async gRPC client for streaming anomaly events to the Correlation Engine.

Batches events and sends them with exponential backoff on failure.
Uses the protobuf-generated AnomalyService stub to stream AnomalyBatch
messages matching the sentinel.v1 wire format. Supports mTLS for
production deployments.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import structlog

from sentinel_inference.analyzers.logit_analyzer import AnomalyEvent
from sentinel_inference.config import GRPCConfig
from sentinel_inference.metrics import GRPC_ERRORS_TOTAL

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Protobuf message builders
# ---------------------------------------------------------------------------
# These functions construct protobuf-compatible dicts that map directly to the
# sentinel.v1.AnomalyBatch / AnomalyEvent wire format defined in anomaly.proto.
# When the generated _pb2 stubs are available they are used directly; otherwise
# we fall back to the raw dict -> bytes serialisation for forward-compatibility.
# ---------------------------------------------------------------------------

# Anomaly type mapping from analyzer names to proto enum values.
_ANOMALY_TYPE_MAP: dict[str, int] = {
    "logit_analyzer": 1,        # ANOMALY_TYPE_LOGIT_DRIFT
    "entropy_analyzer": 2,      # ANOMALY_TYPE_ENTROPY_ANOMALY
    "kl_divergence": 3,         # ANOMALY_TYPE_KL_DIVERGENCE
    "gradient_monitor": 4,      # ANOMALY_TYPE_GRADIENT_NORM_SPIKE
    "loss_monitor": 5,          # ANOMALY_TYPE_LOSS_SPIKE
    "ddp_divergence": 6,        # ANOMALY_TYPE_CROSS_RANK_DIVERGENCE
    "checkpoint_validator": 7,  # ANOMALY_TYPE_CHECKPOINT_DIVERGENCE
    "invariant_checker": 8,     # ANOMALY_TYPE_INVARIANT_VIOLATION
}

# Severity mapping from string labels to proto enum values.
_SEVERITY_MAP: dict[str, int] = {
    "info": 1,
    "warning": 2,
    "high": 3,
    "critical": 4,
}


@dataclass
class AnomalyReport:
    """Structured anomaly report for gRPC transmission.

    Fields map directly to the sentinel.v1.AnomalyBatch protobuf message.
    """

    source_hostname: str
    gpu_uuid: str
    gpu_hostname: str
    gpu_device_index: int
    replica_id: str
    timestamp: float
    sequence_number: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)


class CorrelationEngineClient:
    """Async gRPC client that batches and streams anomaly events.

    Events are accumulated in an internal buffer and flushed either when
    ``batch_size`` events are collected or after ``flush_interval_ms``
    milliseconds, whichever comes first.

    The client uses the sentinel.v1.AnomalyService/StreamAnomalyEvents
    bidirectional streaming RPC. Each AnomalyBatch is serialised as a
    protobuf message and sent on the stream; AnomalyAck responses are
    consumed asynchronously.

    Parameters
    ----------
    config : GRPCConfig
        Connection and batching parameters.
    node_id : str
        Node identifier for reports.
    gpu_id : int
        GPU ordinal.
    gpu_uuid : str
        NVIDIA UUID of the GPU being monitored.
    replica_id : str
        Replica identifier.
    """

    def __init__(
        self,
        config: GRPCConfig,
        node_id: str = "unknown",
        gpu_id: int = 0,
        gpu_uuid: str = "",
        replica_id: str = "unknown",
    ) -> None:
        self._config = config
        self._node_id = node_id
        self._gpu_id = gpu_id
        self._gpu_uuid = gpu_uuid
        self._replica_id = replica_id
        self._buffer: list[AnomalyEvent] = []
        self._lock = asyncio.Lock()
        self._channel: Any = None
        self._stub: Any = None
        self._stream: Any = None
        self._running = False
        self._flush_task: asyncio.Task[None] | None = None
        self._ack_task: asyncio.Task[None] | None = None
        self._consecutive_failures: int = 0
        self._sequence_number: int = 0
        self._pending_acks: dict[int, float] = {}  # seq -> send_time
        self._proto_available = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
                self._channel = grpc.aio.secure_channel(
                    self._config.endpoint,
                    creds,
                    options=[
                        ("grpc.keepalive_time_ms", 30_000),
                        ("grpc.keepalive_timeout_ms", 10_000),
                        ("grpc.http2.min_ping_interval_without_data_ms", 30_000),
                        ("grpc.max_send_message_length", 16 * 1024 * 1024),
                        ("grpc.max_receive_message_length", 16 * 1024 * 1024),
                    ],
                )
            else:
                self._channel = grpc.aio.insecure_channel(
                    self._config.endpoint,
                    options=[
                        ("grpc.keepalive_time_ms", 30_000),
                        ("grpc.keepalive_timeout_ms", 10_000),
                        ("grpc.max_send_message_length", 16 * 1024 * 1024),
                        ("grpc.max_receive_message_length", 16 * 1024 * 1024),
                    ],
                )

            # Try to import the generated protobuf stubs.
            self._stub = self._create_stub()
            logger.info("grpc_client_started", endpoint=self._config.endpoint,
                        proto_stubs=self._proto_available)
        except ImportError:
            logger.warning("grpc_not_available", msg="Running without gRPC reporting.")
            self._channel = None

        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())

    def _create_stub(self) -> Any:
        """Create the AnomalyService stub, trying generated stubs first."""
        try:
            from sentinel.v1 import anomaly_pb2_grpc  # type: ignore[import-untyped]
            stub = anomaly_pb2_grpc.AnomalyServiceStub(self._channel)
            self._proto_available = True
            logger.debug("using_generated_proto_stubs")
            return stub
        except ImportError:
            logger.debug("generated_proto_stubs_not_found",
                         msg="Falling back to manual serialization")
            self._proto_available = False
            return None

    async def stop(self) -> None:
        """Flush remaining events and close the channel."""
        self._running = False
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        if self._ack_task is not None:
            self._ack_task.cancel()
            try:
                await self._ack_task
            except asyncio.CancelledError:
                pass
        # Final flush.
        await self._flush()
        # Close the bidi stream if open.
        if self._stream is not None:
            try:
                await self._stream.done_writing()
            except Exception:
                pass
            self._stream = None
        if self._channel is not None:
            await self._channel.close()
        logger.info("grpc_client_stopped",
                     pending_acks=len(self._pending_acks))

    # ------------------------------------------------------------------
    # Event submission
    # ------------------------------------------------------------------

    async def submit(self, events: Sequence[AnomalyEvent]) -> None:
        """Add anomaly events to the send buffer."""
        if not events:
            return
        async with self._lock:
            self._buffer.extend(events)
            if len(self._buffer) >= self._config.batch_size:
                await self._flush_locked()

    # ------------------------------------------------------------------
    # Periodic flush
    # ------------------------------------------------------------------

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

        self._sequence_number += 1
        report = AnomalyReport(
            source_hostname=self._node_id,
            gpu_uuid=self._gpu_uuid,
            gpu_hostname=self._node_id,
            gpu_device_index=self._gpu_id,
            replica_id=self._replica_id,
            timestamp=time.time(),
            sequence_number=self._sequence_number,
            events=[self._event_to_proto_dict(e) for e in batch],
        )

        await self._send_with_backoff(report)

    # ------------------------------------------------------------------
    # Event serialization
    # ------------------------------------------------------------------

    def _event_to_proto_dict(self, event: AnomalyEvent) -> dict[str, Any]:
        """Convert an internal AnomalyEvent to a protobuf-compatible dict."""
        severity_str = getattr(event, "severity", "warning")
        if isinstance(severity_str, (int, float)):
            severity_val = int(severity_str)
        else:
            severity_val = _SEVERITY_MAP.get(str(severity_str).lower(), 2)

        anomaly_type_val = _ANOMALY_TYPE_MAP.get(
            getattr(event, "analyzer", ""), 0
        )

        # Generate a deterministic event ID from content hash.
        content = f"{event.analyzer}:{event.stat_name}:{event.sample_count}:{event.observed_value}"
        event_id = str(uuid.uuid5(uuid.NAMESPACE_OID, content))

        return {
            "event_id": event_id,
            "anomaly_type": anomaly_type_val,
            "source": 1,  # ANOMALY_SOURCE_INFERENCE_MONITOR
            "gpu": {
                "uuid": self._gpu_uuid,
                "hostname": self._node_id,
                "device_index": self._gpu_id,
            },
            "severity": severity_val,
            "score": float(event.observed_value),
            "threshold": float(event.ucl) if event.ucl else 0.0,
            "details": (
                f"{event.analyzer}/{event.stat_name}: observed={event.observed_value:.6f}, "
                f"ewma={event.ewma_value:.6f}, ucl={event.ucl}, lcl={event.lcl}"
            ),
            "tensor_fingerprint": hashlib.sha256(
                f"{event.observed_value}".encode()
            ).digest()[:16],
            "timestamp": time.time(),
            "metadata": event.details if isinstance(event.details, dict) else {},
            "step_number": event.sample_count,
        }

    # ------------------------------------------------------------------
    # Sending with backoff
    # ------------------------------------------------------------------

    async def _send_with_backoff(self, report: AnomalyReport) -> None:
        """Send a report with exponential backoff on failure."""
        if self._channel is None:
            # No gRPC channel — log locally only.
            logger.info(
                "anomaly_report_local",
                event_count=len(report.events),
                source_hostname=report.source_hostname,
            )
            return

        backoff_ms = self._config.initial_backoff_ms
        for attempt in range(self._config.max_retries + 1):
            try:
                await self._do_send(report)
                self._consecutive_failures = 0
                logger.debug(
                    "anomaly_report_sent",
                    event_count=len(report.events),
                    sequence_number=report.sequence_number,
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
                # Reset the stream on failure so we reconnect.
                self._stream = None
                if attempt < self._config.max_retries:
                    await asyncio.sleep(backoff_ms / 1000.0)
                    backoff_ms = min(backoff_ms * 2, self._config.max_backoff_ms)

        logger.error(
            "grpc_send_exhausted_retries",
            event_count=len(report.events),
            max_retries=self._config.max_retries,
        )

    async def _do_send(self, report: AnomalyReport) -> None:
        """Send an AnomalyBatch via the bidirectional streaming RPC.

        Uses the generated protobuf stubs when available. Falls back to
        manual protobuf construction via the well-known field wire format.
        """
        if self._proto_available and self._stub is not None:
            await self._do_send_proto(report)
        else:
            await self._do_send_manual(report)

    async def _do_send_proto(self, report: AnomalyReport) -> None:
        """Send using generated protobuf stubs."""
        from google.protobuf.timestamp_pb2 import Timestamp  # type: ignore[import-untyped]
        from sentinel.v1 import anomaly_pb2  # type: ignore[import-untyped]
        from sentinel.v1 import common_pb2  # type: ignore[import-untyped]

        # Build the AnomalyBatch protobuf message.
        batch = anomaly_pb2.AnomalyBatch()
        batch.source_hostname = report.source_hostname
        batch.sequence_number = report.sequence_number

        batch_ts = Timestamp()
        batch_ts.FromSeconds(int(report.timestamp))
        batch.batch_timestamp.CopyFrom(batch_ts)

        for evt_dict in report.events:
            evt = anomaly_pb2.AnomalyEvent()
            evt.event_id = evt_dict["event_id"]
            evt.anomaly_type = evt_dict["anomaly_type"]
            evt.source = evt_dict["source"]

            gpu = common_pb2.GpuIdentifier()
            gpu.uuid = evt_dict["gpu"]["uuid"]
            gpu.hostname = evt_dict["gpu"]["hostname"]
            gpu.device_index = evt_dict["gpu"]["device_index"]
            evt.gpu.CopyFrom(gpu)

            evt.severity = evt_dict["severity"]
            evt.score = evt_dict["score"]
            evt.threshold = evt_dict["threshold"]
            evt.details = evt_dict["details"]
            evt.tensor_fingerprint = evt_dict["tensor_fingerprint"]

            ts = Timestamp()
            ts.FromSeconds(int(evt_dict["timestamp"]))
            evt.timestamp.CopyFrom(ts)

            for k, v in evt_dict.get("metadata", {}).items():
                evt.metadata[k] = str(v)

            evt.step_number = evt_dict.get("step_number", 0)

            batch.events.append(evt)

        # Open or reuse the bidi stream.
        if self._stream is None:
            self._stream = self._stub.StreamAnomalyEvents()
            # Start background ack reader.
            self._ack_task = asyncio.create_task(
                self._read_acks_proto()
            )

        self._pending_acks[report.sequence_number] = time.time()
        await self._stream.write(batch)

    async def _read_acks_proto(self) -> None:
        """Background task: read AnomalyAck messages from the stream."""
        try:
            async for ack in self._stream:
                seq = ack.sequence_number
                send_time = self._pending_acks.pop(seq, None)
                if send_time is not None:
                    latency_ms = (time.time() - send_time) * 1000
                    logger.debug(
                        "anomaly_ack_received",
                        sequence_number=seq,
                        accepted=ack.accepted,
                        latency_ms=f"{latency_ms:.1f}",
                    )
                    if not ack.accepted:
                        logger.warning(
                            "anomaly_batch_rejected",
                            sequence_number=seq,
                            reason=ack.rejection_reason,
                        )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("ack_reader_error", error=str(exc))

    async def _do_send_manual(self, report: AnomalyReport) -> None:
        """Send using manual gRPC unary call when proto stubs are unavailable.

        Constructs the request as a JSON-encoded byte payload over a
        generic unary-unary call matching the AnomalyService method path.
        This allows the client to function without generated code, at the
        cost of requiring the server to accept JSON-encoded anomaly batches.
        """
        import json

        import grpc  # type: ignore[import-untyped]

        payload = json.dumps({
            "source_hostname": report.source_hostname,
            "sequence_number": report.sequence_number,
            "batch_timestamp": {"seconds": int(report.timestamp)},
            "events": report.events,
        }, default=str).encode("utf-8")

        call = self._channel.unary_unary(
            "/sentinel.v1.AnomalyService/StreamAnomalyEvents",
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )
        response = await call(payload, timeout=10.0)

        # Parse a minimal ack from the response.
        if response:
            try:
                ack = json.loads(response)
                if not ack.get("accepted", True):
                    logger.warning(
                        "anomaly_batch_rejected_manual",
                        reason=ack.get("rejection_reason", "unknown"),
                    )
            except (json.JSONDecodeError, TypeError):
                pass  # Server may not return JSON in fallback mode.
