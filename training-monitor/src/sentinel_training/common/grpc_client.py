"""Async gRPC client for streaming anomaly events to the Correlation Engine.

Provides batched, thread-safe event streaming with exponential backoff
reconnection and mTLS support.
"""

from __future__ import annotations

import atexit
import threading
import time
from collections import deque
from typing import Any

import grpc
import structlog

from sentinel_training.common.anomaly_detector import AnomalyScore
from sentinel_training.common.config import GrpcConfig

logger = structlog.get_logger(__name__)


class AnomalyEventBatch:
    """A batch of anomaly events to be sent to the Correlation Engine."""

    def __init__(self, events: list[dict[str, object]], timestamp: float) -> None:
        self.events = events
        self.timestamp = timestamp
        self.node_id: str | None = None
        self.cluster_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the batch for transmission."""
        return {
            "events": self.events,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "cluster_id": self.cluster_id,
            "event_count": len(self.events),
        }


class GrpcAnomalyClient:
    """Thread-safe gRPC client for streaming anomaly events.

    Events are buffered and sent in batches. The client manages its own
    background flush thread and handles reconnection with exponential backoff.

    Args:
        config: gRPC client configuration.
        node_id: Identifier for the current node.
        cluster_id: Identifier for the cluster.
    """

    def __init__(
        self,
        config: GrpcConfig,
        node_id: str | None = None,
        cluster_id: str | None = None,
    ) -> None:
        self._config = config
        self._node_id = node_id
        self._cluster_id = cluster_id

        self._buffer: deque[dict[str, object]] = deque()
        self._lock = threading.Lock()
        self._channel: grpc.Channel | None = None
        self._connected = False
        self._backoff_seconds = 1.0
        self._max_backoff = config.max_backoff_seconds

        self._flush_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._total_sent = 0
        self._total_failed = 0
        self._total_batches = 0

    def start(self) -> None:
        """Start the background flush thread and establish the gRPC connection."""
        self._stop_event.clear()
        self._connect()
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="sentinel-grpc-flush",
            daemon=True,
        )
        self._flush_thread.start()
        atexit.register(self.stop)
        logger.info(
            "grpc_client_started",
            endpoint=self._config.endpoint,
            batch_size=self._config.batch_size,
        )

    def stop(self) -> None:
        """Stop the flush thread and close the gRPC connection."""
        self._stop_event.set()
        if self._flush_thread is not None and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5.0)
        # Flush remaining events
        self._flush()
        self._disconnect()
        logger.info(
            "grpc_client_stopped",
            total_sent=self._total_sent,
            total_failed=self._total_failed,
        )

    def send(self, anomaly: AnomalyScore) -> None:
        """Queue an anomaly event for batched transmission.

        Thread-safe. Events are buffered and flushed periodically or when
        the batch size is reached.

        Args:
            anomaly: The anomaly score to send.
        """
        event = anomaly.to_dict()
        event["node_id"] = self._node_id
        event["cluster_id"] = self._cluster_id

        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self._config.batch_size:
                self._flush_locked()

    def _connect(self) -> None:
        """Establish gRPC channel with optional mTLS."""
        try:
            if self._config.mtls_enabled:
                if not all([
                    self._config.cert_path,
                    self._config.key_path,
                    self._config.ca_path,
                ]):
                    raise ValueError(
                        "mTLS enabled but cert_path, key_path, or ca_path not set"
                    )
                with open(self._config.cert_path, "rb") as f:  # type: ignore[arg-type]
                    cert = f.read()
                with open(self._config.key_path, "rb") as f:  # type: ignore[arg-type]
                    key = f.read()
                with open(self._config.ca_path, "rb") as f:  # type: ignore[arg-type]
                    ca = f.read()
                credentials = grpc.ssl_channel_credentials(
                    root_certificates=ca,
                    private_key=key,
                    certificate_chain=cert,
                )
                self._channel = grpc.secure_channel(
                    self._config.endpoint, credentials
                )
            else:
                self._channel = grpc.insecure_channel(self._config.endpoint)

            self._connected = True
            self._backoff_seconds = 1.0
            logger.info("grpc_connected", endpoint=self._config.endpoint)
        except Exception as exc:
            logger.error("grpc_connect_failed", error=str(exc))
            self._connected = False

    def _disconnect(self) -> None:
        """Close the gRPC channel."""
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:
                pass
            self._channel = None
            self._connected = False

    def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        self._disconnect()
        time.sleep(self._backoff_seconds)
        self._backoff_seconds = min(
            self._backoff_seconds * 2, self._max_backoff
        )
        self._connect()

    def _flush_loop(self) -> None:
        """Background thread that periodically flushes the event buffer."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._config.flush_interval_seconds)
            with self._lock:
                self._flush_locked()

    def _flush(self) -> None:
        """Flush buffered events (acquires lock)."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Flush buffered events (caller must hold lock)."""
        if not self._buffer:
            return

        events = list(self._buffer)
        self._buffer.clear()

        batch = AnomalyEventBatch(
            events=events,
            timestamp=time.time(),
        )
        batch.node_id = self._node_id
        batch.cluster_id = self._cluster_id

        if not self._connected:
            self._reconnect()

        if self._connected and self._channel is not None:
            try:
                self._send_batch(batch)
                self._total_sent += len(events)
                self._total_batches += 1
                self._backoff_seconds = 1.0
                logger.debug(
                    "grpc_batch_sent",
                    event_count=len(events),
                    total_sent=self._total_sent,
                )
            except grpc.RpcError as exc:
                logger.warning(
                    "grpc_send_failed",
                    error=str(exc),
                    event_count=len(events),
                )
                self._total_failed += len(events)
                # Re-queue events for retry (with limit to avoid memory growth)
                if len(self._buffer) < self._config.batch_size * 10:
                    self._buffer.extend(events)
                self._connected = False
        else:
            self._total_failed += len(events)
            logger.warning(
                "grpc_not_connected",
                dropped_events=len(events),
            )

    def _send_batch(self, batch: AnomalyEventBatch) -> None:
        """Send a batch via the gRPC channel.

        This is a simple unary call. In a full implementation, this would use
        a generated protobuf stub. For now, we serialize to JSON and send
        via a generic call, or simply log the batch if no stub is available.
        """
        if self._channel is None:
            raise grpc.RpcError("Channel is not connected")

        # In production, this would call a generated stub:
        # stub = anomaly_pb2_grpc.AnomalyServiceStub(self._channel)
        # stub.ReportAnomalies(request)
        #
        # For the framework, we use the channel's unary_unary method with
        # a serialized payload. Since the proto definition lives in the
        # correlation-engine component, we encode as JSON bytes and use
        # a well-known method path.
        import json

        payload = json.dumps(batch.to_dict()).encode("utf-8")

        call = self._channel.unary_unary(
            "/sentinel.correlation.AnomalyService/ReportAnomalies",
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )
        call(
            payload,
            timeout=self._config.timeout_seconds,
        )

    @property
    def stats(self) -> dict[str, int]:
        """Client statistics."""
        return {
            "total_sent": self._total_sent,
            "total_failed": self._total_failed,
            "total_batches": self._total_batches,
            "buffer_size": len(self._buffer),
        }
