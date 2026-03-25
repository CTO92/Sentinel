"""SENTINEL SDK type definitions.

Pydantic models mirroring the protobuf types defined in proto/sentinel/v1/.
"""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# common.proto
# ---------------------------------------------------------------------------

class Severity(IntEnum):
    """Severity levels for events, anomalies, and alerts."""
    UNSPECIFIED = 0
    INFO = 1
    WARNING = 2
    HIGH = 3
    CRITICAL = 4


class GpuIdentifier(BaseModel):
    """Uniquely identifies a GPU within the fleet."""
    uuid: str = ""
    hostname: str = ""
    device_index: int = 0
    model: str = ""
    driver_version: str = ""
    firmware_version: str = ""


class SmIdentifier(BaseModel):
    """Identifies a specific Streaming Multiprocessor on a GPU."""
    gpu: Optional[GpuIdentifier] = None
    sm_id: int = 0


# ---------------------------------------------------------------------------
# probe.proto
# ---------------------------------------------------------------------------

class ProbeType(IntEnum):
    """Types of hardware integrity probes."""
    UNSPECIFIED = 0
    FMA = 1
    TENSOR_CORE = 2
    TRANSCENDENTAL = 3
    AES = 4
    MEMORY = 5
    REGISTER_FILE = 6
    SHARED_MEMORY = 7


class ProbeResult(IntEnum):
    """Result of a single probe execution."""
    UNSPECIFIED = 0
    PASS = 1
    FAIL = 2
    ERROR = 3
    TIMEOUT = 4


class MismatchDetail(BaseModel):
    """Detailed information about a bit-level mismatch when a probe fails."""
    byte_offset: int = 0
    expected_value: bytes = b""
    actual_value: bytes = b""
    differing_bits: list[int] = Field(default_factory=list)


class ProbeExecution(BaseModel):
    """A single probe execution with its result and environmental context."""
    execution_id: str = ""
    probe_type: ProbeType = ProbeType.UNSPECIFIED
    sm: Optional[SmIdentifier] = None
    result: ProbeResult = ProbeResult.UNSPECIFIED
    expected_hash: bytes = b""
    actual_hash: bytes = b""
    mismatch_detail: Optional[MismatchDetail] = None
    execution_time_ns: int = 0
    gpu_clock_mhz: int = 0
    gpu_temperature_c: float = 0.0
    gpu_power_w: float = 0.0
    timestamp: Optional[datetime] = None
    hmac_signature: bytes = b""


class ProbeScheduleOverride(BaseModel):
    """An override directive for a specific probe type's schedule."""
    probe_type: ProbeType = ProbeType.UNSPECIFIED
    period_seconds: int = 0
    duration_seconds: int = 0


class ProbeResultBatch(BaseModel):
    """A batch of probe results from a single agent."""
    agent_hostname: str = ""
    executions: list[ProbeExecution] = Field(default_factory=list)
    sequence_number: int = 0
    batch_timestamp: Optional[datetime] = None


class ProbeAck(BaseModel):
    """Acknowledgement from the correlation engine for received probe batches."""
    sequence_number: int = 0
    accepted: bool = False
    rejection_reason: str = ""
    schedule_overrides: list[ProbeScheduleOverride] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# anomaly.proto
# ---------------------------------------------------------------------------

class AnomalyType(IntEnum):
    """Classification of anomaly types detected by the monitoring subsystems."""
    UNSPECIFIED = 0
    LOGIT_DRIFT = 1
    ENTROPY_ANOMALY = 2
    KL_DIVERGENCE = 3
    GRADIENT_NORM_SPIKE = 4
    LOSS_SPIKE = 5
    CROSS_RANK_DIVERGENCE = 6
    CHECKPOINT_DIVERGENCE = 7
    INVARIANT_VIOLATION = 8


class AnomalySource(IntEnum):
    """Subsystem that detected the anomaly."""
    UNSPECIFIED = 0
    INFERENCE_MONITOR = 1
    TRAINING_MONITOR = 2
    INVARIANT_CHECKER = 3


class AnomalyEvent(BaseModel):
    """A single detected anomaly event."""
    event_id: str = ""
    anomaly_type: AnomalyType = AnomalyType.UNSPECIFIED
    source: AnomalySource = AnomalySource.UNSPECIFIED
    gpu: Optional[GpuIdentifier] = None
    severity: Severity = Severity.UNSPECIFIED
    score: float = 0.0
    threshold: float = 0.0
    details: str = ""
    tensor_fingerprint: bytes = b""
    timestamp: Optional[datetime] = None
    metadata: dict[str, str] = Field(default_factory=dict)
    layer_name: str = ""
    model_id: str = ""
    step_number: int = 0


# ---------------------------------------------------------------------------
# health.proto
# ---------------------------------------------------------------------------

class GpuHealthState(IntEnum):
    """Lifecycle states for a GPU in the fleet."""
    UNSPECIFIED = 0
    HEALTHY = 1
    SUSPECT = 2
    QUARANTINED = 3
    DEEP_TEST = 4
    CONDEMNED = 5


class SmHealth(BaseModel):
    """Health status for a single Streaming Multiprocessor."""
    sm: Optional[SmIdentifier] = None
    reliability_score: float = 0.0
    probe_pass_count: int = 0
    probe_fail_count: int = 0
    disabled: bool = False
    disable_reason: str = ""


class GpuHealth(BaseModel):
    """Health status and Bayesian reliability model for a single GPU."""
    gpu: Optional[GpuIdentifier] = None
    state: GpuHealthState = GpuHealthState.UNSPECIFIED
    reliability_score: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    last_probe_time: Optional[datetime] = None
    last_anomaly_time: Optional[datetime] = None
    probe_pass_count: int = 0
    probe_fail_count: int = 0
    anomaly_count: int = 0
    state_changed_at: Optional[datetime] = None
    state_change_reason: str = ""
    sm_health: list[SmHealth] = Field(default_factory=list)
    anomaly_rate: float = 0.0
    probe_failure_rate: float = 0.0


class FleetHealthSummary(BaseModel):
    """Aggregated health summary for the entire GPU fleet."""
    total_gpus: int = 0
    healthy: int = 0
    suspect: int = 0
    quarantined: int = 0
    deep_test: int = 0
    condemned: int = 0
    overall_sdc_rate: float = 0.0
    average_reliability_score: float = 0.0
    snapshot_time: Optional[datetime] = None
    active_agents: int = 0
    rate_window_seconds: int = 0


# ---------------------------------------------------------------------------
# trust.proto
# ---------------------------------------------------------------------------

class TrustEdge(BaseModel):
    """An edge in the GPU trust graph representing pairwise comparison history."""
    gpu_a: Optional[GpuIdentifier] = None
    gpu_b: Optional[GpuIdentifier] = None
    agreement_count: int = 0
    disagreement_count: int = 0
    last_comparison: Optional[datetime] = None
    trust_score: float = 0.0


class TrustGraphSnapshot(BaseModel):
    """Point-in-time snapshot of the entire trust graph."""
    edges: list[TrustEdge] = Field(default_factory=list)
    timestamp: Optional[datetime] = None
    coverage_pct: float = 0.0
    total_gpus: int = 0
    min_trust_score: float = 0.0
    mean_trust_score: float = 0.0


class TmrGpuResult(BaseModel):
    """Result of a single GPU's contribution to a TMR canary."""
    gpu: Optional[GpuIdentifier] = None
    output_fingerprint: bytes = b""
    execution_time_ns: int = 0
    success: bool = False
    error_message: str = ""


class TmrResult(BaseModel):
    """Aggregated result of a TMR canary run."""
    canary_id: str = ""
    results: list[TmrGpuResult] = Field(default_factory=list)
    consensus_hash: bytes = b""
    dissenting_gpu: Optional[GpuIdentifier] = None
    unanimous: bool = False
    timestamp: Optional[datetime] = None


# ---------------------------------------------------------------------------
# quarantine.proto
# ---------------------------------------------------------------------------

class QuarantineAction(IntEnum):
    """Actions that can be taken on a GPU's lifecycle state."""
    UNSPECIFIED = 0
    QUARANTINE = 1
    REINSTATE = 2
    CONDEMN = 3
    SCHEDULE_DEEP_TEST = 4


class ApprovalStatus(BaseModel):
    """Approval tracking for directives that require human sign-off."""
    approved: bool = False
    reviewer: str = ""
    review_time: Optional[datetime] = None
    comment: str = ""


class QuarantineDirective(BaseModel):
    """A directive to change a GPU's lifecycle state."""
    directive_id: str = ""
    gpu: Optional[GpuIdentifier] = None
    action: QuarantineAction = QuarantineAction.UNSPECIFIED
    reason: str = ""
    initiated_by: str = ""
    evidence: list[str] = Field(default_factory=list)
    timestamp: Optional[datetime] = None
    priority: int = 0
    requires_approval: bool = False
    approval: Optional[ApprovalStatus] = None


class DirectiveResponse(BaseModel):
    """Response to a directive issuance."""
    directive_id: str = ""
    accepted: bool = False
    rejection_reason: str = ""
    resulting_state: str = ""


# ---------------------------------------------------------------------------
# correlation.proto
# ---------------------------------------------------------------------------

class PatternType(IntEnum):
    """Types of correlation patterns detected across events."""
    UNSPECIFIED = 0
    MULTI_SIGNAL = 1
    SM_LOCALIZED = 2
    ENVIRONMENTAL = 3
    NODE_CORRELATED = 4
    FIRMWARE_CORRELATED = 5
    TMR_CONFIRMED = 6


class CorrelationEvent(BaseModel):
    """A correlation event linking multiple raw events into a higher-level finding."""
    event_id: str = ""
    events_correlated: list[str] = Field(default_factory=list)
    pattern_type: PatternType = PatternType.UNSPECIFIED
    confidence: float = 0.0
    attributed_gpu: Optional[GpuIdentifier] = None
    attributed_sm: Optional[SmIdentifier] = None
    description: str = ""
    timestamp: Optional[datetime] = None
    severity: Severity = Severity.UNSPECIFIED
    recommended_action: str = ""


class StateTransition(BaseModel):
    """A recorded GPU state transition."""
    from_state: GpuHealthState = GpuHealthState.UNSPECIFIED
    to_state: GpuHealthState = GpuHealthState.UNSPECIFIED
    timestamp: Optional[datetime] = None
    reason: str = ""
    initiated_by: str = ""


class ReliabilitySample(BaseModel):
    """A point-in-time reliability score sample."""
    timestamp: Optional[datetime] = None
    reliability_score: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0


class GpuHistoryResponse(BaseModel):
    """Response with historical health data."""
    state_transitions: list[StateTransition] = Field(default_factory=list)
    correlations: list[CorrelationEvent] = Field(default_factory=list)
    reliability_history: list[ReliabilitySample] = Field(default_factory=list)
    next_page_token: str = ""


# ---------------------------------------------------------------------------
# audit.proto
# ---------------------------------------------------------------------------

class AuditEntryType(IntEnum):
    """Types of entries that can appear in the audit ledger."""
    UNSPECIFIED = 0
    PROBE_RESULT = 1
    ANOMALY_EVENT = 2
    QUARANTINE_ACTION = 3
    CONFIG_CHANGE = 4
    TMR_RESULT = 5
    SYSTEM_EVENT = 6


class AuditEntry(BaseModel):
    """A single entry in the tamper-evident audit ledger."""
    entry_id: int = 0
    entry_type: AuditEntryType = AuditEntryType.UNSPECIFIED
    timestamp: Optional[datetime] = None
    gpu: Optional[GpuIdentifier] = None
    data: bytes = b""
    previous_hash: bytes = b""
    entry_hash: bytes = b""
    merkle_root: bytes = b""


class ChainVerificationResult(BaseModel):
    """Response from chain verification."""
    valid: bool = False
    first_invalid_entry_id: int = 0
    failure_description: str = ""
    entries_verified: int = 0
    batches_verified: int = 0
    verification_time_ms: int = 0


class AuditQueryFilters(BaseModel):
    """Filters for querying the audit trail."""
    gpu: Optional[GpuIdentifier] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    entry_type: AuditEntryType = AuditEntryType.UNSPECIFIED
    limit: int = 100
    page_token: str = ""
    descending: bool = False


class AuditQueryResponse(BaseModel):
    """Response from an audit trail query."""
    entries: list[AuditEntry] = Field(default_factory=list)
    next_page_token: str = ""
    total_count: int = 0


# ---------------------------------------------------------------------------
# config.proto
# ---------------------------------------------------------------------------

class ProbeScheduleEntry(BaseModel):
    """A single entry in the probe schedule."""
    type: ProbeType = ProbeType.UNSPECIFIED
    period_seconds: int = 0
    sm_coverage: float = 0.0
    priority: int = 0
    enabled: bool = False
    timeout_ms: int = 0


class ProbeScheduleUpdate(BaseModel):
    """Update to the probe execution schedule."""
    entries: list[ProbeScheduleEntry] = Field(default_factory=list)


class OverheadBudgetUpdate(BaseModel):
    """Update to the overall overhead budget."""
    budget_pct: float = 0.0


class SamplingRateUpdate(BaseModel):
    """Update to a sampling rate for a specific component."""
    component: str = ""
    rate: float = 0.0


class ThresholdUpdate(BaseModel):
    """Update to a threshold value for a specific component and parameter."""
    component: str = ""
    parameter: str = ""
    value: float = 0.0


class ConfigUpdate(BaseModel):
    """A dynamic configuration update pushed to agents or subsystems."""
    update_id: str = ""
    initiated_by: str = ""
    reason: str = ""
    probe_schedule: Optional[ProbeScheduleUpdate] = None
    overhead_budget: Optional[OverheadBudgetUpdate] = None
    sampling_rate: Optional[SamplingRateUpdate] = None
    threshold: Optional[ThresholdUpdate] = None


class ConfigAck(BaseModel):
    """Acknowledgement from a config update recipient."""
    update_id: str = ""
    applied: bool = False
    component_id: str = ""
    error: str = ""
    config_version: int = 0


# ---------------------------------------------------------------------------
# telemetry.proto
# ---------------------------------------------------------------------------

class ThermalReading(BaseModel):
    """Thermal sensor readings from a GPU."""
    gpu: Optional[GpuIdentifier] = None
    temperature_c: float = 0.0
    fan_speed_pct: float = 0.0
    throttle_active: bool = False
    memory_temperature_c: float = 0.0
    timestamp: Optional[datetime] = None


class PowerReading(BaseModel):
    """Electrical power readings from a GPU."""
    gpu: Optional[GpuIdentifier] = None
    power_w: float = 0.0
    voltage_mv: int = 0
    current_ma: int = 0
    power_limit_w: float = 0.0
    power_throttle_active: bool = False
    timestamp: Optional[datetime] = None


class EccCounters(BaseModel):
    """ECC memory error counters from a GPU."""
    gpu: Optional[GpuIdentifier] = None
    sram_corrected: int = 0
    sram_uncorrected: int = 0
    dram_corrected: int = 0
    dram_uncorrected: int = 0
    retired_pages: int = 0
    pending_retired_pages: int = 0
    reset_required: bool = False
    timestamp: Optional[datetime] = None


class NvLinkStatus(BaseModel):
    """Status of a single NVLink connection."""
    link_index: int = 0
    active: bool = False
    crc_errors: int = 0
    replay_errors: int = 0


class GpuTelemetryReport(BaseModel):
    """Aggregated telemetry report for a single GPU."""
    thermal: Optional[ThermalReading] = None
    power: Optional[PowerReading] = None
    ecc: Optional[EccCounters] = None
    gpu_utilization_pct: float = 0.0
    memory_utilization_pct: float = 0.0
    gpu_clock_mhz: int = 0
    mem_clock_mhz: int = 0
    pcie_gen: int = 0
    pcie_width: int = 0
    nvlink_status: list[NvLinkStatus] = Field(default_factory=list)
