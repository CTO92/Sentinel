"""SENTINEL SDK client for interacting with SENTINEL gRPC services.

Provides both synchronous and asynchronous interfaces for querying GPU health,
fleet status, audit trails, trust graphs, and managing quarantine directives.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Union

import grpc
import grpc.aio

from sentinel_sdk.types import (
    AuditEntry,
    AuditEntryType,
    AuditQueryFilters,
    AuditQueryResponse,
    ChainVerificationResult,
    ConfigAck,
    ConfigUpdate,
    CorrelationEvent,
    DirectiveResponse,
    FleetHealthSummary,
    GpuHealth,
    GpuHealthState,
    GpuHistoryResponse,
    GpuIdentifier,
    OverheadBudgetUpdate,
    PatternType,
    ProbeScheduleUpdate,
    QuarantineAction,
    QuarantineDirective,
    SamplingRateUpdate,
    Severity,
    SmHealth,
    SmIdentifier,
    StateTransition,
    ThresholdUpdate,
    TrustEdge,
    TrustGraphSnapshot,
    ReliabilitySample,
)

logger = logging.getLogger("sentinel_sdk")

# ---------------------------------------------------------------------------
# TLS configuration
# ---------------------------------------------------------------------------

@dataclass
class TlsConfig:
    """TLS configuration for secure gRPC connections."""
    ca_cert_path: Optional[str] = None
    client_cert_path: Optional[str] = None
    client_key_path: Optional[str] = None

    def channel_credentials(self) -> grpc.ChannelCredentials:
        """Build gRPC channel credentials from the configured certificate paths."""
        ca_cert = None
        client_cert = None
        client_key = None

        if self.ca_cert_path:
            with open(self.ca_cert_path, "rb") as f:
                ca_cert = f.read()
        if self.client_cert_path:
            with open(self.client_cert_path, "rb") as f:
                client_cert = f.read()
        if self.client_key_path:
            with open(self.client_key_path, "rb") as f:
                client_key = f.read()

        return grpc.ssl_channel_credentials(
            root_certificates=ca_cert,
            private_key=client_key,
            certificate_chain=client_cert,
        )


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

@dataclass
class RetryConfig:
    """Configuration for automatic retries with exponential backoff."""
    max_retries: int = 3
    initial_backoff_s: float = 0.1
    max_backoff_s: float = 10.0
    backoff_multiplier: float = 2.0
    retryable_status_codes: list[grpc.StatusCode] = field(default_factory=lambda: [
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
    ])


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SentinelError(Exception):
    """Base exception for SENTINEL SDK errors."""

    def __init__(self, message: str, code: Optional[grpc.StatusCode] = None):
        super().__init__(message)
        self.code = code


class ConnectionError(SentinelError):
    """Raised when the client cannot connect to the SENTINEL endpoint."""


class AuthenticationError(SentinelError):
    """Raised when authentication or authorization fails."""


class NotFoundError(SentinelError):
    """Raised when a requested resource is not found."""


class InvalidArgumentError(SentinelError):
    """Raised when an invalid argument is passed to an RPC."""


# ---------------------------------------------------------------------------
# Internal helpers: protobuf <-> SDK type conversion
# ---------------------------------------------------------------------------

def _timestamp_to_datetime(ts: Any) -> Optional[datetime]:
    """Convert a protobuf Timestamp to a Python datetime."""
    if ts is None or not ts.seconds and not ts.nanos:
        return None
    return datetime.utcfromtimestamp(ts.seconds + ts.nanos / 1e9)


def _datetime_to_timestamp(dt: Optional[datetime]) -> Any:
    """Convert a Python datetime to a protobuf-compatible dict for serialization."""
    if dt is None:
        return None
    from google.protobuf.timestamp_pb2 import Timestamp
    ts = Timestamp()
    ts.FromDatetime(dt)
    return ts


def _gpu_id_from_proto(pb: Any) -> Optional[GpuIdentifier]:
    """Convert a protobuf GpuIdentifier message to the SDK type."""
    if pb is None:
        return None
    return GpuIdentifier(
        uuid=pb.uuid,
        hostname=pb.hostname,
        device_index=pb.device_index,
        model=pb.model,
        driver_version=pb.driver_version,
        firmware_version=pb.firmware_version,
    )


def _gpu_id_to_proto_dict(gpu: Optional[GpuIdentifier]) -> dict[str, Any]:
    """Serialize a GpuIdentifier to a dict suitable for protobuf construction."""
    if gpu is None:
        return {}
    return {
        "uuid": gpu.uuid,
        "hostname": gpu.hostname,
        "device_index": gpu.device_index,
        "model": gpu.model,
        "driver_version": gpu.driver_version,
        "firmware_version": gpu.firmware_version,
    }


def _sm_id_from_proto(pb: Any) -> Optional[SmIdentifier]:
    if pb is None:
        return None
    return SmIdentifier(
        gpu=_gpu_id_from_proto(pb.gpu),
        sm_id=pb.sm_id,
    )


def _sm_health_from_proto(pb: Any) -> SmHealth:
    return SmHealth(
        sm=_sm_id_from_proto(pb.sm),
        reliability_score=pb.reliability_score,
        probe_pass_count=pb.probe_pass_count,
        probe_fail_count=pb.probe_fail_count,
        disabled=pb.disabled,
        disable_reason=pb.disable_reason,
    )


def _gpu_health_from_proto(pb: Any) -> GpuHealth:
    """Convert a protobuf GpuHealth message to the SDK type."""
    return GpuHealth(
        gpu=_gpu_id_from_proto(pb.gpu),
        state=GpuHealthState(pb.state),
        reliability_score=pb.reliability_score,
        alpha=pb.alpha,
        beta=pb.beta,
        last_probe_time=_timestamp_to_datetime(pb.last_probe_time),
        last_anomaly_time=_timestamp_to_datetime(pb.last_anomaly_time),
        probe_pass_count=pb.probe_pass_count,
        probe_fail_count=pb.probe_fail_count,
        anomaly_count=pb.anomaly_count,
        state_changed_at=_timestamp_to_datetime(pb.state_changed_at),
        state_change_reason=pb.state_change_reason,
        sm_health=[_sm_health_from_proto(s) for s in pb.sm_health],
        anomaly_rate=pb.anomaly_rate,
        probe_failure_rate=pb.probe_failure_rate,
    )


def _fleet_summary_from_proto(pb: Any) -> FleetHealthSummary:
    return FleetHealthSummary(
        total_gpus=pb.total_gpus,
        healthy=pb.healthy,
        suspect=pb.suspect,
        quarantined=pb.quarantined,
        deep_test=pb.deep_test,
        condemned=pb.condemned,
        overall_sdc_rate=pb.overall_sdc_rate,
        average_reliability_score=pb.average_reliability_score,
        snapshot_time=_timestamp_to_datetime(pb.snapshot_time),
        active_agents=pb.active_agents,
        rate_window_seconds=pb.rate_window_seconds,
    )


def _correlation_event_from_proto(pb: Any) -> CorrelationEvent:
    return CorrelationEvent(
        event_id=pb.event_id,
        events_correlated=list(pb.events_correlated),
        pattern_type=PatternType(pb.pattern_type),
        confidence=pb.confidence,
        attributed_gpu=_gpu_id_from_proto(pb.attributed_gpu),
        attributed_sm=_sm_id_from_proto(pb.attributed_sm),
        description=pb.description,
        timestamp=_timestamp_to_datetime(pb.timestamp),
        severity=Severity(pb.severity),
        recommended_action=pb.recommended_action,
    )


def _state_transition_from_proto(pb: Any) -> StateTransition:
    return StateTransition(
        from_state=GpuHealthState(pb.from_state),
        to_state=GpuHealthState(pb.to_state),
        timestamp=_timestamp_to_datetime(pb.timestamp),
        reason=pb.reason,
        initiated_by=pb.initiated_by,
    )


def _reliability_sample_from_proto(pb: Any) -> ReliabilitySample:
    return ReliabilitySample(
        timestamp=_timestamp_to_datetime(pb.timestamp),
        reliability_score=pb.reliability_score,
        alpha=pb.alpha,
        beta=pb.beta,
    )


def _quarantine_directive_from_proto(pb: Any) -> QuarantineDirective:
    approval = None
    if pb.HasField("approval"):
        approval_pb = pb.approval
        from sentinel_sdk.types import ApprovalStatus
        approval = ApprovalStatus(
            approved=approval_pb.approved,
            reviewer=approval_pb.reviewer,
            review_time=_timestamp_to_datetime(approval_pb.review_time),
            comment=approval_pb.comment,
        )
    return QuarantineDirective(
        directive_id=pb.directive_id,
        gpu=_gpu_id_from_proto(pb.gpu),
        action=QuarantineAction(pb.action),
        reason=pb.reason,
        initiated_by=pb.initiated_by,
        evidence=list(pb.evidence),
        timestamp=_timestamp_to_datetime(pb.timestamp),
        priority=pb.priority,
        requires_approval=pb.requires_approval,
        approval=approval,
    )


def _trust_edge_from_proto(pb: Any) -> TrustEdge:
    return TrustEdge(
        gpu_a=_gpu_id_from_proto(pb.gpu_a),
        gpu_b=_gpu_id_from_proto(pb.gpu_b),
        agreement_count=pb.agreement_count,
        disagreement_count=pb.disagreement_count,
        last_comparison=_timestamp_to_datetime(pb.last_comparison),
        trust_score=pb.trust_score,
    )


def _trust_graph_from_proto(pb: Any) -> TrustGraphSnapshot:
    return TrustGraphSnapshot(
        edges=[_trust_edge_from_proto(e) for e in pb.edges],
        timestamp=_timestamp_to_datetime(pb.timestamp),
        coverage_pct=pb.coverage_pct,
        total_gpus=pb.total_gpus,
        min_trust_score=pb.min_trust_score,
        mean_trust_score=pb.mean_trust_score,
    )


def _audit_entry_from_proto(pb: Any) -> AuditEntry:
    return AuditEntry(
        entry_id=pb.entry_id,
        entry_type=AuditEntryType(pb.entry_type),
        timestamp=_timestamp_to_datetime(pb.timestamp),
        gpu=_gpu_id_from_proto(pb.gpu),
        data=bytes(pb.data),
        previous_hash=bytes(pb.previous_hash),
        entry_hash=bytes(pb.entry_hash),
        merkle_root=bytes(pb.merkle_root),
    )


def _chain_verification_from_proto(pb: Any) -> ChainVerificationResult:
    return ChainVerificationResult(
        valid=pb.valid,
        first_invalid_entry_id=pb.first_invalid_entry_id,
        failure_description=pb.failure_description,
        entries_verified=pb.entries_verified,
        batches_verified=pb.batches_verified,
        verification_time_ms=pb.verification_time_ms,
    )


def _config_ack_from_proto(pb: Any) -> ConfigAck:
    return ConfigAck(
        update_id=pb.update_id,
        applied=pb.applied,
        component_id=pb.component_id,
        error=pb.error,
        config_version=pb.config_version,
    )


def _rpc_error_to_exception(err: grpc.RpcError) -> SentinelError:
    """Map a gRPC RpcError to the appropriate SDK exception."""
    code = err.code()
    details = err.details() or str(err)
    if code == grpc.StatusCode.NOT_FOUND:
        return NotFoundError(details, code)
    if code in (grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED):
        return AuthenticationError(details, code)
    if code == grpc.StatusCode.INVALID_ARGUMENT:
        return InvalidArgumentError(details, code)
    if code == grpc.StatusCode.UNAVAILABLE:
        return ConnectionError(details, code)
    return SentinelError(details, code)


# ---------------------------------------------------------------------------
# SentinelClient
# ---------------------------------------------------------------------------

class SentinelClient:
    """Client for interacting with SENTINEL services.

    Supports both synchronous and asynchronous usage patterns. The synchronous
    methods block the calling thread; the asynchronous methods (prefixed with
    ``a``) return coroutines.

    Usage (sync)::

        client = SentinelClient.connect("sentinel.example.com:443",
                                         tls_config=TlsConfig(ca_cert_path="ca.pem"))
        health = client.query_gpu_health("GPU-xxxx")
        client.close()

    Usage (async)::

        client = await SentinelClient.aconnect("sentinel.example.com:443")
        health = await client.aquery_gpu_health("GPU-xxxx")
        await client.aclose()
    """

    def __init__(
        self,
        channel: Union[grpc.Channel, grpc.aio.Channel],
        retry_config: RetryConfig,
        default_timeout: float,
        _is_async: bool = False,
    ) -> None:
        self._channel = channel
        self._retry = retry_config
        self._timeout = default_timeout
        self._is_async = _is_async
        self._closed = False

        # Lazily-imported generated stubs are cached here.
        self._correlation_stub: Any = None
        self._quarantine_stub: Any = None
        self._audit_stub: Any = None
        self._config_stub: Any = None
        self._probe_stub: Any = None
        self._anomaly_stub: Any = None
        self._telemetry_stub: Any = None

    # ------------------------------------------------------------------
    # Connection factories
    # ------------------------------------------------------------------

    @classmethod
    def connect(
        cls,
        endpoint: str,
        *,
        tls_config: Optional[TlsConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        default_timeout: float = 30.0,
        options: Optional[list[tuple[str, Any]]] = None,
    ) -> "SentinelClient":
        """Create a synchronous SentinelClient connected to *endpoint*.

        Parameters
        ----------
        endpoint:
            ``host:port`` of the SENTINEL gRPC gateway.
        tls_config:
            TLS credentials.  If ``None``, an insecure channel is used.
        retry_config:
            Retry behaviour.  Defaults to 3 retries with exponential backoff.
        default_timeout:
            Default per-RPC deadline in seconds.
        options:
            Additional gRPC channel options.
        """
        retry = retry_config or RetryConfig()
        channel_options = options or [
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ]
        if tls_config:
            creds = tls_config.channel_credentials()
            channel = grpc.secure_channel(endpoint, creds, options=channel_options)
        else:
            channel = grpc.insecure_channel(endpoint, options=channel_options)

        logger.info("Connected (sync) to %s", endpoint)
        return cls(channel, retry, default_timeout, _is_async=False)

    @classmethod
    async def aconnect(
        cls,
        endpoint: str,
        *,
        tls_config: Optional[TlsConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        default_timeout: float = 30.0,
        options: Optional[list[tuple[str, Any]]] = None,
    ) -> "SentinelClient":
        """Create an asynchronous SentinelClient connected to *endpoint*."""
        retry = retry_config or RetryConfig()
        channel_options = options or [
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ]
        if tls_config:
            creds = tls_config.channel_credentials()
            channel = grpc.aio.secure_channel(endpoint, creds, options=channel_options)
        else:
            channel = grpc.aio.insecure_channel(endpoint, options=channel_options)

        logger.info("Connected (async) to %s", endpoint)
        return cls(channel, retry, default_timeout, _is_async=True)

    # ------------------------------------------------------------------
    # Stub accessors (lazy import of generated code)
    # ------------------------------------------------------------------

    def _get_correlation_stub(self) -> Any:
        if self._correlation_stub is None:
            from sentinel.v1 import correlation_pb2_grpc  # type: ignore[import-untyped]
            self._correlation_stub = correlation_pb2_grpc.CorrelationServiceStub(self._channel)
        return self._correlation_stub

    def _get_quarantine_stub(self) -> Any:
        if self._quarantine_stub is None:
            from sentinel.v1 import quarantine_pb2_grpc  # type: ignore[import-untyped]
            self._quarantine_stub = quarantine_pb2_grpc.QuarantineServiceStub(self._channel)
        return self._quarantine_stub

    def _get_audit_stub(self) -> Any:
        if self._audit_stub is None:
            from sentinel.v1 import audit_pb2_grpc  # type: ignore[import-untyped]
            self._audit_stub = audit_pb2_grpc.AuditServiceStub(self._channel)
        return self._audit_stub

    def _get_config_stub(self) -> Any:
        if self._config_stub is None:
            from sentinel.v1 import config_pb2_grpc  # type: ignore[import-untyped]
            self._config_stub = config_pb2_grpc.ConfigServiceStub(self._channel)
        return self._config_stub

    # ------------------------------------------------------------------
    # Retry helpers
    # ------------------------------------------------------------------

    def _sync_retry(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* with synchronous retry and exponential backoff."""
        last_err: Optional[Exception] = None
        backoff = self._retry.initial_backoff_s
        for attempt in range(1 + self._retry.max_retries):
            try:
                return fn(*args, **kwargs)
            except grpc.RpcError as err:
                if err.code() not in self._retry.retryable_status_codes:
                    raise _rpc_error_to_exception(err) from err
                last_err = err
                if attempt < self._retry.max_retries:
                    logger.warning(
                        "RPC failed (attempt %d/%d, code=%s), retrying in %.2fs",
                        attempt + 1, self._retry.max_retries + 1,
                        err.code(), backoff,
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * self._retry.backoff_multiplier, self._retry.max_backoff_s)
        raise _rpc_error_to_exception(last_err)  # type: ignore[arg-type]

    async def _async_retry(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* with asynchronous retry and exponential backoff."""
        last_err: Optional[Exception] = None
        backoff = self._retry.initial_backoff_s
        for attempt in range(1 + self._retry.max_retries):
            try:
                return await fn(*args, **kwargs)
            except grpc.aio.AioRpcError as err:
                if err.code() not in self._retry.retryable_status_codes:
                    raise _rpc_error_to_exception(err) from err
                last_err = err
                if attempt < self._retry.max_retries:
                    logger.warning(
                        "RPC failed (attempt %d/%d, code=%s), retrying in %.2fs",
                        attempt + 1, self._retry.max_retries + 1,
                        err.code(), backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * self._retry.backoff_multiplier, self._retry.max_backoff_s)
        raise _rpc_error_to_exception(last_err)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # query_gpu_health
    # ------------------------------------------------------------------

    def query_gpu_health(self, gpu_uuid: str) -> GpuHealth:
        """Query the health of a specific GPU (synchronous)."""
        self._ensure_open()
        from sentinel.v1 import correlation_pb2  # type: ignore[import-untyped]
        from sentinel.v1 import common_pb2  # type: ignore[import-untyped]

        request = correlation_pb2.HealthQueryRequest(
            gpu=common_pb2.GpuIdentifier(uuid=gpu_uuid),
            include_sm_health=True,
        )
        stub = self._get_correlation_stub()
        response = self._sync_retry(stub.QueryGpuHealth, request, timeout=self._timeout)
        return _gpu_health_from_proto(response.health)

    async def aquery_gpu_health(self, gpu_uuid: str) -> GpuHealth:
        """Query the health of a specific GPU (asynchronous)."""
        self._ensure_open()
        from sentinel.v1 import correlation_pb2  # type: ignore[import-untyped]
        from sentinel.v1 import common_pb2  # type: ignore[import-untyped]

        request = correlation_pb2.HealthQueryRequest(
            gpu=common_pb2.GpuIdentifier(uuid=gpu_uuid),
            include_sm_health=True,
        )
        stub = self._get_correlation_stub()
        response = await self._async_retry(stub.QueryGpuHealth, request, timeout=self._timeout)
        return _gpu_health_from_proto(response.health)

    # ------------------------------------------------------------------
    # query_fleet_health
    # ------------------------------------------------------------------

    def query_fleet_health(
        self,
        hostname_prefix: str = "",
        model_filter: str = "",
        state_filter: Optional[list[GpuHealthState]] = None,
    ) -> FleetHealthSummary:
        """Query the fleet-wide health summary (synchronous)."""
        self._ensure_open()
        from sentinel.v1 import correlation_pb2  # type: ignore[import-untyped]

        request = correlation_pb2.FleetHealthRequest(
            hostname_prefix=hostname_prefix,
            model_filter=model_filter,
            state_filter=[int(s) for s in (state_filter or [])],
        )
        stub = self._get_correlation_stub()
        response = self._sync_retry(stub.QueryFleetHealth, request, timeout=self._timeout)
        return _fleet_summary_from_proto(response.summary)

    async def aquery_fleet_health(
        self,
        hostname_prefix: str = "",
        model_filter: str = "",
        state_filter: Optional[list[GpuHealthState]] = None,
    ) -> FleetHealthSummary:
        """Query the fleet-wide health summary (asynchronous)."""
        self._ensure_open()
        from sentinel.v1 import correlation_pb2  # type: ignore[import-untyped]

        request = correlation_pb2.FleetHealthRequest(
            hostname_prefix=hostname_prefix,
            model_filter=model_filter,
            state_filter=[int(s) for s in (state_filter or [])],
        )
        stub = self._get_correlation_stub()
        response = await self._async_retry(stub.QueryFleetHealth, request, timeout=self._timeout)
        return _fleet_summary_from_proto(response.summary)

    # ------------------------------------------------------------------
    # get_gpu_history
    # ------------------------------------------------------------------

    def get_gpu_history(
        self,
        gpu_uuid: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
        page_token: str = "",
    ) -> GpuHistoryResponse:
        """Retrieve historical health data for a GPU (synchronous)."""
        self._ensure_open()
        from sentinel.v1 import correlation_pb2, common_pb2  # type: ignore[import-untyped]

        request = correlation_pb2.GpuHistoryRequest(
            gpu=common_pb2.GpuIdentifier(uuid=gpu_uuid),
            start_time=_datetime_to_timestamp(start_time),
            end_time=_datetime_to_timestamp(end_time),
            limit=limit,
            page_token=page_token,
        )
        stub = self._get_correlation_stub()
        resp = self._sync_retry(stub.GetGpuHistory, request, timeout=self._timeout)
        return GpuHistoryResponse(
            state_transitions=[_state_transition_from_proto(t) for t in resp.state_transitions],
            correlations=[_correlation_event_from_proto(c) for c in resp.correlations],
            reliability_history=[_reliability_sample_from_proto(r) for r in resp.reliability_history],
            next_page_token=resp.next_page_token,
        )

    async def aget_gpu_history(
        self,
        gpu_uuid: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
        page_token: str = "",
    ) -> GpuHistoryResponse:
        """Retrieve historical health data for a GPU (asynchronous)."""
        self._ensure_open()
        from sentinel.v1 import correlation_pb2, common_pb2  # type: ignore[import-untyped]

        request = correlation_pb2.GpuHistoryRequest(
            gpu=common_pb2.GpuIdentifier(uuid=gpu_uuid),
            start_time=_datetime_to_timestamp(start_time),
            end_time=_datetime_to_timestamp(end_time),
            limit=limit,
            page_token=page_token,
        )
        stub = self._get_correlation_stub()
        resp = await self._async_retry(stub.GetGpuHistory, request, timeout=self._timeout)
        return GpuHistoryResponse(
            state_transitions=[_state_transition_from_proto(t) for t in resp.state_transitions],
            correlations=[_correlation_event_from_proto(c) for c in resp.correlations],
            reliability_history=[_reliability_sample_from_proto(r) for r in resp.reliability_history],
            next_page_token=resp.next_page_token,
        )

    # ------------------------------------------------------------------
    # issue_quarantine
    # ------------------------------------------------------------------

    def issue_quarantine(
        self,
        gpu_uuid: str,
        action: QuarantineAction,
        reason: str,
        initiated_by: str = "sentinel-sdk",
        evidence: Optional[list[str]] = None,
        requires_approval: bool = False,
    ) -> DirectiveResponse:
        """Issue a quarantine directive for a GPU (synchronous)."""
        self._ensure_open()
        from sentinel.v1 import quarantine_pb2, common_pb2  # type: ignore[import-untyped]

        directive = quarantine_pb2.QuarantineDirective(
            gpu=common_pb2.GpuIdentifier(uuid=gpu_uuid),
            action=int(action),
            reason=reason,
            initiated_by=initiated_by,
            evidence=evidence or [],
            requires_approval=requires_approval,
        )
        stub = self._get_quarantine_stub()
        resp = self._sync_retry(stub.IssueDirective, directive, timeout=self._timeout)
        return DirectiveResponse(
            directive_id=resp.directive_id,
            accepted=resp.accepted,
            rejection_reason=resp.rejection_reason,
            resulting_state=resp.resulting_state,
        )

    async def aissue_quarantine(
        self,
        gpu_uuid: str,
        action: QuarantineAction,
        reason: str,
        initiated_by: str = "sentinel-sdk",
        evidence: Optional[list[str]] = None,
        requires_approval: bool = False,
    ) -> DirectiveResponse:
        """Issue a quarantine directive for a GPU (asynchronous)."""
        self._ensure_open()
        from sentinel.v1 import quarantine_pb2, common_pb2  # type: ignore[import-untyped]

        directive = quarantine_pb2.QuarantineDirective(
            gpu=common_pb2.GpuIdentifier(uuid=gpu_uuid),
            action=int(action),
            reason=reason,
            initiated_by=initiated_by,
            evidence=evidence or [],
            requires_approval=requires_approval,
        )
        stub = self._get_quarantine_stub()
        resp = await self._async_retry(stub.IssueDirective, directive, timeout=self._timeout)
        return DirectiveResponse(
            directive_id=resp.directive_id,
            accepted=resp.accepted,
            rejection_reason=resp.rejection_reason,
            resulting_state=resp.resulting_state,
        )

    # ------------------------------------------------------------------
    # query_audit_trail
    # ------------------------------------------------------------------

    def query_audit_trail(self, filters: AuditQueryFilters) -> AuditQueryResponse:
        """Query the audit trail with filtering and pagination (synchronous)."""
        self._ensure_open()
        from sentinel.v1 import audit_pb2, common_pb2  # type: ignore[import-untyped]

        gpu_pb = None
        if filters.gpu:
            gpu_pb = common_pb2.GpuIdentifier(uuid=filters.gpu.uuid)

        request = audit_pb2.AuditQueryRequest(
            gpu=gpu_pb,
            start_time=_datetime_to_timestamp(filters.start_time),
            end_time=_datetime_to_timestamp(filters.end_time),
            entry_type=int(filters.entry_type),
            limit=filters.limit,
            page_token=filters.page_token,
            descending=filters.descending,
        )
        stub = self._get_audit_stub()
        resp = self._sync_retry(stub.QueryAuditTrail, request, timeout=self._timeout)
        return AuditQueryResponse(
            entries=[_audit_entry_from_proto(e) for e in resp.entries],
            next_page_token=resp.next_page_token,
            total_count=resp.total_count,
        )

    async def aquery_audit_trail(self, filters: AuditQueryFilters) -> AuditQueryResponse:
        """Query the audit trail with filtering and pagination (asynchronous)."""
        self._ensure_open()
        from sentinel.v1 import audit_pb2, common_pb2  # type: ignore[import-untyped]

        gpu_pb = None
        if filters.gpu:
            gpu_pb = common_pb2.GpuIdentifier(uuid=filters.gpu.uuid)

        request = audit_pb2.AuditQueryRequest(
            gpu=gpu_pb,
            start_time=_datetime_to_timestamp(filters.start_time),
            end_time=_datetime_to_timestamp(filters.end_time),
            entry_type=int(filters.entry_type),
            limit=filters.limit,
            page_token=filters.page_token,
            descending=filters.descending,
        )
        stub = self._get_audit_stub()
        resp = await self._async_retry(stub.QueryAuditTrail, request, timeout=self._timeout)
        return AuditQueryResponse(
            entries=[_audit_entry_from_proto(e) for e in resp.entries],
            next_page_token=resp.next_page_token,
            total_count=resp.total_count,
        )

    # ------------------------------------------------------------------
    # verify_chain
    # ------------------------------------------------------------------

    def verify_chain(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        start_entry_id: int = 0,
        end_entry_id: int = 0,
        verify_merkle_roots: bool = True,
    ) -> ChainVerificationResult:
        """Verify the integrity of the audit chain (synchronous)."""
        self._ensure_open()
        from sentinel.v1 import audit_pb2  # type: ignore[import-untyped]

        request = audit_pb2.ChainVerificationRequest(
            start_entry_id=start_entry_id,
            end_entry_id=end_entry_id,
            verify_merkle_roots=verify_merkle_roots,
        )
        stub = self._get_audit_stub()
        resp = self._sync_retry(stub.VerifyChain, request, timeout=self._timeout)
        return _chain_verification_from_proto(resp)

    async def averify_chain(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        start_entry_id: int = 0,
        end_entry_id: int = 0,
        verify_merkle_roots: bool = True,
    ) -> ChainVerificationResult:
        """Verify the integrity of the audit chain (asynchronous)."""
        self._ensure_open()
        from sentinel.v1 import audit_pb2  # type: ignore[import-untyped]

        request = audit_pb2.ChainVerificationRequest(
            start_entry_id=start_entry_id,
            end_entry_id=end_entry_id,
            verify_merkle_roots=verify_merkle_roots,
        )
        stub = self._get_audit_stub()
        resp = await self._async_retry(stub.VerifyChain, request, timeout=self._timeout)
        return _chain_verification_from_proto(resp)

    # ------------------------------------------------------------------
    # get_trust_graph
    # ------------------------------------------------------------------

    def get_trust_graph(self) -> TrustGraphSnapshot:
        """Retrieve the current trust graph snapshot (synchronous).

        The trust graph service does not have a dedicated gRPC service in the
        proto definitions; it is exposed through the correlation service as an
        extension. We use a custom method descriptor to invoke it.
        """
        self._ensure_open()
        from google.protobuf import empty_pb2  # type: ignore[import-untyped]

        method = self._channel.unary_unary(
            "/sentinel.v1.CorrelationService/GetTrustGraph",
            request_serializer=empty_pb2.Empty.SerializeToString,
            response_deserializer=self._deserialize_trust_graph,
        )
        response = self._sync_retry(method, empty_pb2.Empty(), timeout=self._timeout)
        return response

    async def aget_trust_graph(self) -> TrustGraphSnapshot:
        """Retrieve the current trust graph snapshot (asynchronous)."""
        self._ensure_open()
        from google.protobuf import empty_pb2  # type: ignore[import-untyped]

        method = self._channel.unary_unary(
            "/sentinel.v1.CorrelationService/GetTrustGraph",
            request_serializer=empty_pb2.Empty.SerializeToString,
            response_deserializer=self._deserialize_trust_graph,
        )
        response = await self._async_retry(method, empty_pb2.Empty(), timeout=self._timeout)
        return response

    @staticmethod
    def _deserialize_trust_graph(data: bytes) -> TrustGraphSnapshot:
        from sentinel.v1 import trust_pb2  # type: ignore[import-untyped]
        pb = trust_pb2.TrustGraphSnapshot()
        pb.ParseFromString(data)
        return _trust_graph_from_proto(pb)

    # ------------------------------------------------------------------
    # update_config
    # ------------------------------------------------------------------

    def update_config(self, update: ConfigUpdate) -> ConfigAck:
        """Push a configuration update (synchronous).

        Sends a single config update through a unary-style wrapper around
        the bidirectional config stream.
        """
        self._ensure_open()
        from sentinel.v1 import config_pb2  # type: ignore[import-untyped]

        pb = self._build_config_update_proto(update, config_pb2)

        # Use a short-lived bidi stream: send the update, receive the ack.
        stub = self._get_config_stub()

        def _do_update() -> ConfigAck:
            # The ConfigStream RPC is bidi: client sends ConfigAck, server
            # sends ConfigUpdate.  For a one-shot config push we use a
            # dedicated unary extension method.
            method = self._channel.unary_unary(
                "/sentinel.v1.ConfigService/ApplyConfig",
                request_serializer=config_pb2.ConfigUpdate.SerializeToString,
                response_deserializer=config_pb2.ConfigAck.FromString,
            )
            resp = method(pb, timeout=self._timeout)
            return _config_ack_from_proto(resp)

        return self._sync_retry(_do_update)

    async def aupdate_config(self, update: ConfigUpdate) -> ConfigAck:
        """Push a configuration update (asynchronous)."""
        self._ensure_open()
        from sentinel.v1 import config_pb2  # type: ignore[import-untyped]

        pb = self._build_config_update_proto(update, config_pb2)

        async def _do_update() -> ConfigAck:
            method = self._channel.unary_unary(
                "/sentinel.v1.ConfigService/ApplyConfig",
                request_serializer=config_pb2.ConfigUpdate.SerializeToString,
                response_deserializer=config_pb2.ConfigAck.FromString,
            )
            resp = await method(pb, timeout=self._timeout)
            return _config_ack_from_proto(resp)

        return await self._async_retry(_do_update)

    @staticmethod
    def _build_config_update_proto(update: ConfigUpdate, config_pb2: Any) -> Any:
        """Build a protobuf ConfigUpdate message from an SDK ConfigUpdate."""
        kwargs: dict[str, Any] = {
            "update_id": update.update_id,
            "initiated_by": update.initiated_by,
            "reason": update.reason,
        }
        if update.probe_schedule is not None:
            from sentinel.v1 import probe_pb2  # type: ignore[import-untyped]
            entries = []
            for e in update.probe_schedule.entries:
                entries.append(config_pb2.ProbeScheduleEntry(
                    type=int(e.type),
                    period_seconds=e.period_seconds,
                    sm_coverage=e.sm_coverage,
                    priority=e.priority,
                    enabled=e.enabled,
                    timeout_ms=e.timeout_ms,
                ))
            kwargs["probe_schedule"] = config_pb2.ProbeScheduleUpdate(entries=entries)
        elif update.overhead_budget is not None:
            kwargs["overhead_budget"] = config_pb2.OverheadBudgetUpdate(
                budget_pct=update.overhead_budget.budget_pct,
            )
        elif update.sampling_rate is not None:
            kwargs["sampling_rate"] = config_pb2.SamplingRateUpdate(
                component=update.sampling_rate.component,
                rate=update.sampling_rate.rate,
            )
        elif update.threshold is not None:
            kwargs["threshold"] = config_pb2.ThresholdUpdate(
                component=update.threshold.component,
                parameter=update.threshold.parameter,
                value=update.threshold.value,
            )
        return config_pb2.ConfigUpdate(**kwargs)

    # ------------------------------------------------------------------
    # stream_events
    # ------------------------------------------------------------------

    def stream_events(
        self,
        callback: Callable[[QuarantineDirective], None],
        hostname_filter: str = "",
        action_filter: QuarantineAction = QuarantineAction.UNSPECIFIED,
    ) -> None:
        """Stream quarantine directives in real-time (synchronous, blocking).

        Calls *callback* for each directive received. Blocks until the stream
        is terminated by the server or the client is closed. Automatically
        reconnects on transient failures.

        Parameters
        ----------
        callback:
            Invoked for each incoming ``QuarantineDirective``.
        hostname_filter:
            Only receive directives for this hostname (empty = all).
        action_filter:
            Only receive directives of this action type (UNSPECIFIED = all).
        """
        self._ensure_open()
        from sentinel.v1 import quarantine_pb2  # type: ignore[import-untyped]

        sub = quarantine_pb2.DirectiveSubscription(
            hostname_filter=hostname_filter,
            action_filter=int(action_filter),
        )

        backoff = self._retry.initial_backoff_s
        while not self._closed:
            try:
                stub = self._get_quarantine_stub()
                stream = stub.StreamDirectives(sub, timeout=None)
                for directive_pb in stream:
                    if self._closed:
                        break
                    directive = _quarantine_directive_from_proto(directive_pb)
                    callback(directive)
                # Normal stream end
                break
            except grpc.RpcError as err:
                if self._closed:
                    break
                if err.code() in self._retry.retryable_status_codes:
                    logger.warning(
                        "Event stream interrupted (code=%s), reconnecting in %.2fs",
                        err.code(), backoff,
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * self._retry.backoff_multiplier, self._retry.max_backoff_s)
                else:
                    raise _rpc_error_to_exception(err) from err

    async def astream_events(
        self,
        callback: Callable[[QuarantineDirective], Any],
        hostname_filter: str = "",
        action_filter: QuarantineAction = QuarantineAction.UNSPECIFIED,
    ) -> None:
        """Stream quarantine directives in real-time (asynchronous).

        The *callback* may be a coroutine function; if so, it will be awaited.
        """
        self._ensure_open()
        from sentinel.v1 import quarantine_pb2  # type: ignore[import-untyped]

        sub = quarantine_pb2.DirectiveSubscription(
            hostname_filter=hostname_filter,
            action_filter=int(action_filter),
        )

        backoff = self._retry.initial_backoff_s
        while not self._closed:
            try:
                stub = self._get_quarantine_stub()
                stream = stub.StreamDirectives(sub, timeout=None)
                async for directive_pb in stream:
                    if self._closed:
                        break
                    directive = _quarantine_directive_from_proto(directive_pb)
                    result = callback(directive)
                    if asyncio.iscoroutine(result):
                        await result
                break
            except grpc.aio.AioRpcError as err:
                if self._closed:
                    break
                if err.code() in self._retry.retryable_status_codes:
                    logger.warning(
                        "Event stream interrupted (code=%s), reconnecting in %.2fs",
                        err.code(), backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * self._retry.backoff_multiplier, self._retry.max_backoff_s)
                else:
                    raise _rpc_error_to_exception(err) from err

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise SentinelError("Client is closed")

    def close(self) -> None:
        """Close the client and release all resources (synchronous)."""
        if not self._closed:
            self._closed = True
            if hasattr(self._channel, "close"):
                self._channel.close()
            logger.info("Client closed")

    async def aclose(self) -> None:
        """Close the client and release all resources (asynchronous)."""
        if not self._closed:
            self._closed = True
            if hasattr(self._channel, "close"):
                await self._channel.close()
            logger.info("Client closed")

    def __enter__(self) -> "SentinelClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    async def __aenter__(self) -> "SentinelClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def __del__(self) -> None:
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass
