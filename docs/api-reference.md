# SENTINEL API Reference

> **Status:** Pre-release Alpha
> **API Version:** `sentinel.v1`
> **Transport:** gRPC (primary), REST/HTTP (dashboard)

This document covers the complete SENTINEL API surface: the gRPC services used
by agents and SDKs, and the REST API exposed by the Correlation Engine for
dashboards and lightweight integrations.

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [gRPC Services](#grpc-services)
   - [ProbeService](#probeservice)
   - [TelemetryService](#telemetryservice)
   - [AnomalyService](#anomalyservice)
   - [CorrelationService](#correlationservice)
   - [QuarantineService](#quarantineservice)
   - [AuditService](#auditservice)
   - [ConfigService](#configservice)
4. [REST API](#rest-api)
   - [GET /api/v1/health](#get-apiv1health)
   - [GET /api/v1/fleet](#get-apiv1fleet)
   - [GET /api/v1/gpu/:uuid](#get-apiv1gpuuuid)
   - [GET /api/v1/gpu/:uuid/history](#get-apiv1gpuuuidhistory)
   - [POST /api/v1/quarantine](#post-apiv1quarantine)
   - [GET /api/v1/trust](#get-apiv1trust)
   - [GET /api/v1/audit](#get-apiv1audit)
5. [Common Types](#common-types)
6. [Error Codes](#error-codes)

---

## Overview

SENTINEL exposes two API layers:

| Layer | Transport | Port (default) | Use case |
|---|---|---|---|
| **gRPC** | HTTP/2 + protobuf | 50051 | Agent communication, SDK integration, streaming |
| **REST** | HTTP/1.1 + JSON | 8080 | Dashboards, browser-based tools, simple scripts |

The gRPC API is the primary interface. All streaming operations (probe
ingestion, telemetry, anomaly events, config distribution, quarantine
directives) are gRPC-only. The REST API is a read-heavy convenience layer
exposed by the Correlation Engine's embedded axum server.

### Proto package

All gRPC services are defined in the `sentinel.v1` package. Proto files live
under `proto/sentinel/v1/`.

---

## Authentication

### gRPC: Mutual TLS (mTLS)

All production gRPC connections must use mTLS. The server validates the client
certificate against a configured CA. Insecure connections are accepted only when
the server is started with `--insecure` (development mode).

Certificate paths are configured via server environment variables:

| Variable | Description |
|---|---|
| `SENTINEL_TLS_CA_CERT` | Path to CA certificate (PEM). |
| `SENTINEL_TLS_SERVER_CERT` | Path to server certificate (PEM). |
| `SENTINEL_TLS_SERVER_KEY` | Path to server private key (PEM). |

### REST: Bearer Token or mTLS

The REST API supports two authentication methods:

1. **Bearer token** -- pass a token in the `Authorization` header:
   ```
   Authorization: Bearer <token>
   ```

2. **mTLS** -- the same client certificates used for gRPC. The REST server
   shares the TLS configuration with the gRPC server.

For development, both authentication methods can be disabled.

---

## gRPC Services

### ProbeService

**File:** `proto/sentinel/v1/probe.proto`
**Full name:** `sentinel.v1.ProbeService`

The ProbeService handles bidirectional streaming of probe results between
agents and the Correlation Engine.

#### StreamProbeResults

```protobuf
rpc StreamProbeResults(stream ProbeResultBatch) returns (stream ProbeAck);
```

**Description:** Agents stream probe result batches to the engine. The engine
streams back acknowledgements and optional dynamic schedule adjustments. This
is a long-lived bidirectional stream -- agents typically maintain a single
stream for the lifetime of the process.

**Client sends:** `ProbeResultBatch`

| Field | Type | Description |
|---|---|---|
| `agent_hostname` | `string` | Agent's hostname for correlation. |
| `executions` | `repeated ProbeExecution` | Ordered list of probe executions in this batch. |
| `sequence_number` | `uint64` | Monotonically increasing sequence number for gap detection. |
| `batch_timestamp` | `Timestamp` | When the batch was assembled. |

**ProbeExecution fields:**

| Field | Type | Description |
|---|---|---|
| `execution_id` | `string` | Unique identifier (UUID v7). |
| `probe_type` | `ProbeType` | Type of probe (FMA, TENSOR_CORE, etc.). |
| `sm` | `SmIdentifier` | The SM on which this probe executed. |
| `result` | `ProbeResult` | PASS, FAIL, ERROR, or TIMEOUT. |
| `expected_hash` | `bytes` | SHA-256 of the expected (golden) output. |
| `actual_hash` | `bytes` | SHA-256 of the actual output. |
| `mismatch_detail` | `MismatchDetail` | Bit-level mismatch info (populated only on FAIL). |
| `execution_time_ns` | `uint64` | Wall-clock kernel execution time in nanoseconds. |
| `gpu_clock_mhz` | `uint32` | GPU core clock at time of execution. |
| `gpu_temperature_c` | `float` | GPU temperature at time of execution (Celsius). |
| `gpu_power_w` | `float` | GPU power draw at time of execution (Watts). |
| `timestamp` | `Timestamp` | When the probe completed. |
| `hmac_signature` | `bytes` | HMAC-SHA256 over fields 1-11, keyed with the agent secret. |

**Server responds:** `ProbeAck`

| Field | Type | Description |
|---|---|---|
| `sequence_number` | `uint64` | Sequence number being acknowledged. |
| `accepted` | `bool` | Whether the batch was accepted and persisted. |
| `rejection_reason` | `string` | Reason if not accepted (e.g., HMAC verification failure). |
| `schedule_overrides` | `repeated ProbeScheduleOverride` | Optional updated schedule to apply immediately. |

**Error codes:**

| Code | Condition |
|---|---|
| `UNAUTHENTICATED` | Client certificate validation failed. |
| `INVALID_ARGUMENT` | Malformed batch (missing hostname, empty executions). |
| `RESOURCE_EXHAUSTED` | Server backpressure -- client should slow down. |

**grpcurl example:**

```bash
# Note: StreamProbeResults is bidirectional streaming; grpcurl supports
# unary and server-streaming more naturally. Use grpcurl for testing other RPCs.
# For streaming, use the SDK or a custom gRPC client.
```

---

### TelemetryService

**File:** `proto/sentinel/v1/telemetry.proto`
**Full name:** `sentinel.v1.TelemetryService`

The TelemetryService handles bidirectional streaming of GPU environmental
telemetry (thermal, power, ECC, utilization).

#### StreamTelemetry

```protobuf
rpc StreamTelemetry(stream TelemetryBatch) returns (stream TelemetryAck);
```

**Description:** Agents stream telemetry reports for all GPUs on a host. The
engine acknowledges receipt. Telemetry data feeds the environmental correlation
detector.

**Client sends:** `TelemetryBatch`

| Field | Type | Description |
|---|---|---|
| `agent_hostname` | `string` | Hostname of the reporting agent. |
| `reports` | `repeated GpuTelemetryReport` | One report per GPU on this host. |
| `sequence_number` | `uint64` | Monotonically increasing sequence number. |
| `batch_timestamp` | `Timestamp` | When this batch was assembled. |

**GpuTelemetryReport fields:**

| Field | Type | Description |
|---|---|---|
| `thermal` | `ThermalReading` | Temperature, fan speed, throttle status. |
| `power` | `PowerReading` | Power draw, voltage, current, power limit. |
| `ecc` | `EccCounters` | SRAM/DRAM corrected and uncorrected error counts. |
| `gpu_utilization_pct` | `float` | GPU compute utilization (0-100). |
| `memory_utilization_pct` | `float` | Memory bandwidth utilization (0-100). |
| `gpu_clock_mhz` | `uint32` | Current GPU core clock. |
| `mem_clock_mhz` | `uint32` | Current memory clock. |
| `pcie_gen` | `uint32` | PCIe link generation. |
| `pcie_width` | `uint32` | PCIe link width. |
| `nvlink_status` | `repeated NvLinkStatus` | Per-link NVLink error counts. |

**Server responds:** `TelemetryAck`

| Field | Type | Description |
|---|---|---|
| `sequence_number` | `uint64` | Sequence number being acknowledged. |
| `accepted` | `bool` | Whether the batch was accepted. |

**Error codes:**

| Code | Condition |
|---|---|
| `UNAUTHENTICATED` | Client certificate validation failed. |
| `RESOURCE_EXHAUSTED` | Server backpressure. |

---

### AnomalyService

**File:** `proto/sentinel/v1/anomaly.proto`
**Full name:** `sentinel.v1.AnomalyService`

The AnomalyService handles bidirectional streaming of anomaly events from
inference and training monitors.

#### StreamAnomalyEvents

```protobuf
rpc StreamAnomalyEvents(stream AnomalyBatch) returns (stream AnomalyAck);
```

**Description:** Monitoring sidecars stream detected anomaly events. The engine
acknowledges receipt and may respond with urgency escalation signals.

**Client sends:** `AnomalyBatch`

| Field | Type | Description |
|---|---|---|
| `source_hostname` | `string` | Source hostname. |
| `events` | `repeated AnomalyEvent` | Anomaly events in this batch. |
| `sequence_number` | `uint64` | Monotonically increasing sequence number. |
| `batch_timestamp` | `Timestamp` | When this batch was assembled. |

**AnomalyEvent fields:**

| Field | Type | Description |
|---|---|---|
| `event_id` | `string` | Unique identifier (UUID v7). |
| `anomaly_type` | `AnomalyType` | Classification (LOGIT_DRIFT, KL_DIVERGENCE, etc.). |
| `source` | `AnomalySource` | Detecting subsystem (INFERENCE_MONITOR, TRAINING_MONITOR, INVARIANT_CHECKER). |
| `gpu` | `GpuIdentifier` | GPU associated with this anomaly. |
| `severity` | `Severity` | Severity assessment (INFO, WARNING, HIGH, CRITICAL). |
| `score` | `float` | Anomaly score (magnitude of deviation). |
| `threshold` | `float` | Threshold that was exceeded. |
| `details` | `string` | Human-readable description. |
| `tensor_fingerprint` | `bytes` | Hash of the tensor/output that triggered detection. |
| `timestamp` | `Timestamp` | When the anomaly was detected. |
| `metadata` | `map<string, string>` | Arbitrary key-value metadata. |
| `layer_name` | `string` | Layer name or operation (if applicable). |
| `model_id` | `string` | Model identifier (if known). |
| `step_number` | `uint64` | Training step or batch index (if applicable). |

**Server responds:** `AnomalyAck`

| Field | Type | Description |
|---|---|---|
| `sequence_number` | `uint64` | Sequence number being acknowledged. |
| `accepted` | `bool` | Whether the batch was accepted. |
| `rejection_reason` | `string` | Reason if rejected. |

**Error codes:**

| Code | Condition |
|---|---|
| `UNAUTHENTICATED` | Client certificate validation failed. |
| `INVALID_ARGUMENT` | Malformed event (missing GPU identifier, invalid anomaly type). |
| `RESOURCE_EXHAUSTED` | Server backpressure. |

---

### CorrelationService

**File:** `proto/sentinel/v1/correlation.proto`
**Full name:** `sentinel.v1.CorrelationService`

The CorrelationService provides unary RPCs for querying correlated health data.
This is the primary query interface used by SDKs and dashboards.

#### QueryGpuHealth

```protobuf
rpc QueryGpuHealth(HealthQueryRequest) returns (HealthQueryResponse);
```

**Description:** Query the health of a specific GPU, including its lifecycle
state, Bayesian reliability model, and optionally per-SM health breakdown.

**Request:** `HealthQueryRequest`

| Field | Type | Description |
|---|---|---|
| `gpu` | `GpuIdentifier` | GPU to query. Only `uuid` is required. |
| `include_sm_health` | `bool` | Whether to include per-SM breakdown. |

**Response:** `HealthQueryResponse`

| Field | Type | Description |
|---|---|---|
| `health` | `GpuHealth` | Current health record (see [GpuHealth](#gpuhealth-message)). |
| `recent_correlations` | `repeated CorrelationEvent` | Recent correlation events involving this GPU. |

**Error codes:**

| Code | Condition |
|---|---|
| `NOT_FOUND` | GPU UUID not known to the system. |
| `INVALID_ARGUMENT` | Missing or malformed UUID. |

**grpcurl example:**

```bash
grpcurl -plaintext \
  -d '{"gpu": {"uuid": "GPU-abcd1234-5678-9abc-def0-123456789abc"}, "include_sm_health": true}' \
  localhost:50051 sentinel.v1.CorrelationService/QueryGpuHealth
```

#### QueryFleetHealth

```protobuf
rpc QueryFleetHealth(FleetHealthRequest) returns (FleetHealthResponse);
```

**Description:** Query the fleet-wide health summary with optional filters.

**Request:** `FleetHealthRequest`

| Field | Type | Description |
|---|---|---|
| `hostname_prefix` | `string` | Filter by hostname prefix (optional, empty = all). |
| `model_filter` | `string` | Filter by GPU model (optional, empty = all). |
| `state_filter` | `repeated GpuHealthState` | Only include GPUs in these states (optional, empty = all). |

**Response:** `FleetHealthResponse`

| Field | Type | Description |
|---|---|---|
| `summary` | `FleetHealthSummary` | Aggregated fleet summary. |
| `gpu_health` | `repeated GpuHealth` | Per-GPU health (top N worst if fleet is large). |
| `truncated` | `bool` | Whether the response was truncated. |
| `total_matching` | `uint32` | Total matching GPUs (may exceed `gpu_health` length). |

**grpcurl example:**

```bash
grpcurl -plaintext \
  -d '{}' \
  localhost:50051 sentinel.v1.CorrelationService/QueryFleetHealth
```

```bash
# With filters:
grpcurl -plaintext \
  -d '{"hostname_prefix": "gpu-rack-01", "state_filter": [2, 3]}' \
  localhost:50051 sentinel.v1.CorrelationService/QueryFleetHealth
```

#### GetGpuHistory

```protobuf
rpc GetGpuHistory(GpuHistoryRequest) returns (GpuHistoryResponse);
```

**Description:** Query historical health data for a GPU within a time range.
Returns state transitions, correlation events, and reliability score time
series. Supports pagination.

**Request:** `GpuHistoryRequest`

| Field | Type | Description |
|---|---|---|
| `gpu` | `GpuIdentifier` | GPU to query. |
| `start_time` | `Timestamp` | Start of the time range. |
| `end_time` | `Timestamp` | End of the time range. |
| `limit` | `uint32` | Maximum number of events to return. |
| `page_token` | `string` | Pagination token for subsequent requests. |

**Response:** `GpuHistoryResponse`

| Field | Type | Description |
|---|---|---|
| `state_transitions` | `repeated StateTransition` | Historical state transitions. |
| `correlations` | `repeated CorrelationEvent` | Historical correlation events. |
| `reliability_history` | `repeated ReliabilitySample` | Reliability score time series. |
| `next_page_token` | `string` | Pagination token (empty if no more). |

**grpcurl example:**

```bash
grpcurl -plaintext \
  -d '{
    "gpu": {"uuid": "GPU-abcd1234-5678-9abc-def0-123456789abc"},
    "start_time": "2026-03-17T00:00:00Z",
    "end_time": "2026-03-24T00:00:00Z",
    "limit": 100
  }' \
  localhost:50051 sentinel.v1.CorrelationService/GetGpuHistory
```

---

### QuarantineService

**File:** `proto/sentinel/v1/quarantine.proto`
**Full name:** `sentinel.v1.QuarantineService`

The QuarantineService manages GPU lifecycle state transitions.

#### IssueDirective

```protobuf
rpc IssueDirective(QuarantineDirective) returns (DirectiveResponse);
```

**Description:** Issue a quarantine directive to change a GPU's lifecycle state.
The directive may be executed immediately or queued for human approval.

**Request:** `QuarantineDirective`

| Field | Type | Description |
|---|---|---|
| `directive_id` | `string` | Unique identifier (auto-generated if empty). |
| `gpu` | `GpuIdentifier` | The GPU to act upon. |
| `action` | `QuarantineAction` | QUARANTINE, REINSTATE, CONDEMN, or SCHEDULE_DEEP_TEST. |
| `reason` | `string` | Human-readable reason. |
| `initiated_by` | `string` | Who/what initiated this (e.g., `"correlation-engine"`, `"operator:jane"`). |
| `evidence` | `repeated string` | References to supporting evidence. |
| `timestamp` | `Timestamp` | When this directive was issued (auto-set if empty). |
| `priority` | `uint32` | Priority (lower = higher priority). |
| `requires_approval` | `bool` | Whether human approval is required. |
| `approval` | `ApprovalStatus` | Approval status (for pre-approved directives). |

**Response:** `DirectiveResponse`

| Field | Type | Description |
|---|---|---|
| `directive_id` | `string` | The directive ID that was processed. |
| `accepted` | `bool` | Whether the directive was accepted. |
| `rejection_reason` | `string` | Reason if not accepted. |
| `resulting_state` | `string` | Resulting GPU state after applying the directive. |

**Error codes:**

| Code | Condition |
|---|---|
| `NOT_FOUND` | GPU UUID not known. |
| `INVALID_ARGUMENT` | Invalid action or missing GPU identifier. |
| `FAILED_PRECONDITION` | Invalid state transition (e.g., reinstating a condemned GPU). |
| `PERMISSION_DENIED` | Insufficient permissions for this action. |

**grpcurl example:**

```bash
grpcurl -plaintext \
  -d '{
    "gpu": {"uuid": "GPU-abcd1234-5678-9abc-def0-123456789abc"},
    "action": 1,
    "reason": "Repeated FMA probe failures on SM 42",
    "initiated_by": "operator:jane",
    "evidence": ["probe-exec-001", "probe-exec-002"]
  }' \
  localhost:50051 sentinel.v1.QuarantineService/IssueDirective
```

#### StreamDirectives

```protobuf
rpc StreamDirectives(DirectiveSubscription) returns (stream QuarantineDirective);
```

**Description:** Subscribe to a stream of quarantine directives. Used by
orchestrators and agents to receive real-time notifications about GPU lifecycle
changes. The stream is server-streaming (client sends one request, server
streams responses).

**Request:** `DirectiveSubscription`

| Field | Type | Description |
|---|---|---|
| `hostname_filter` | `string` | Filter by hostname (empty = all hosts). |
| `action_filter` | `QuarantineAction` | Filter by action type (UNSPECIFIED = all). |

**Response:** `stream QuarantineDirective` (see IssueDirective request fields above)

**grpcurl example:**

```bash
# Stream all directives (blocks until cancelled):
grpcurl -plaintext \
  -d '{}' \
  localhost:50051 sentinel.v1.QuarantineService/StreamDirectives

# Filter to quarantine actions on a specific host:
grpcurl -plaintext \
  -d '{"hostname_filter": "gpu-node-01", "action_filter": 1}' \
  localhost:50051 sentinel.v1.QuarantineService/StreamDirectives
```

---

### AuditService

**File:** `proto/sentinel/v1/audit.proto`
**Full name:** `sentinel.v1.AuditService`

The AuditService manages the tamper-evident audit ledger. Entries form a hash
chain where each entry's hash includes the previous entry's hash, creating an
append-only, verifiable log.

#### IngestEvents

```protobuf
rpc IngestEvents(AuditIngestRequest) returns (AuditIngestResponse);
```

**Description:** Append events to the audit ledger. The engine assigns entry
IDs, computes hashes, and builds Merkle trees. This is typically called by
internal components, not external clients.

**Request:** `AuditIngestRequest`

| Field | Type | Description |
|---|---|---|
| `entries` | `repeated AuditEntry` | Events to append (entry_id and hashes are computed server-side). |

**Response:** `AuditIngestResponse`

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether all events were ingested. |
| `entries_ingested` | `uint32` | Number of entries ingested. |
| `last_entry_id` | `uint64` | Entry ID of the last ingested entry. |
| `error` | `string` | Error description if not all events were ingested. |

**Error codes:**

| Code | Condition |
|---|---|
| `PERMISSION_DENIED` | Client not authorized for audit ingestion. |
| `INVALID_ARGUMENT` | Malformed entries. |

**grpcurl example:**

```bash
grpcurl -plaintext \
  -d '{
    "entries": [
      {
        "entry_type": 6,
        "gpu": {"uuid": "GPU-abcd1234-5678-9abc-def0-123456789abc"},
        "data": "c3lzdGVtIGV2ZW50IGRhdGE="
      }
    ]
  }' \
  localhost:50051 sentinel.v1.AuditService/IngestEvents
```

#### VerifyChain

```protobuf
rpc VerifyChain(ChainVerificationRequest) returns (ChainVerificationResponse);
```

**Description:** Verify the integrity of the hash chain and optionally the
Merkle tree. Returns whether the chain is valid and, if not, the first point
of breakage.

**Request:** `ChainVerificationRequest`

| Field | Type | Description |
|---|---|---|
| `start_entry_id` | `uint64` | Starting entry ID (0 = from genesis). |
| `end_entry_id` | `uint64` | Ending entry ID (0 = through latest). |
| `verify_merkle_roots` | `bool` | Whether to also verify batch Merkle roots. |

**Response:** `ChainVerificationResponse`

| Field | Type | Description |
|---|---|---|
| `valid` | `bool` | Whether the chain is valid. |
| `first_invalid_entry_id` | `uint64` | Entry ID of the first break (if invalid). |
| `failure_description` | `string` | Description of the failure. |
| `entries_verified` | `uint64` | Total entries verified. |
| `batches_verified` | `uint64` | Total batches verified (if Merkle verification requested). |
| `verification_time_ms` | `uint64` | Time taken in milliseconds. |

**grpcurl example:**

```bash
grpcurl -plaintext \
  -d '{"verify_merkle_roots": true}' \
  localhost:50051 sentinel.v1.AuditService/VerifyChain
```

#### QueryAuditTrail

```protobuf
rpc QueryAuditTrail(AuditQueryRequest) returns (AuditQueryResponse);
```

**Description:** Query the audit trail with filtering and pagination.

**Request:** `AuditQueryRequest`

| Field | Type | Description |
|---|---|---|
| `gpu` | `GpuIdentifier` | Filter by GPU (optional). |
| `start_time` | `Timestamp` | Start of time range (optional). |
| `end_time` | `Timestamp` | End of time range (optional). |
| `entry_type` | `AuditEntryType` | Filter by type (UNSPECIFIED = all). |
| `limit` | `uint32` | Maximum entries to return. |
| `page_token` | `string` | Pagination token. |
| `descending` | `bool` | Sort order (true = newest first). |

**Response:** `AuditQueryResponse`

| Field | Type | Description |
|---|---|---|
| `entries` | `repeated AuditEntry` | Matching entries. |
| `next_page_token` | `string` | Pagination token (empty if done). |
| `total_count` | `uint64` | Total matching entries (may be approximate). |

**AuditEntry fields:**

| Field | Type | Description |
|---|---|---|
| `entry_id` | `uint64` | Monotonically increasing sequence number. |
| `entry_type` | `AuditEntryType` | PROBE_RESULT, ANOMALY_EVENT, QUARANTINE_ACTION, CONFIG_CHANGE, TMR_RESULT, or SYSTEM_EVENT. |
| `timestamp` | `Timestamp` | When this entry was recorded. |
| `gpu` | `GpuIdentifier` | GPU associated with this entry (may be absent for system events). |
| `data` | `bytes` | Serialized protobuf of the underlying event. |
| `previous_hash` | `bytes` | Hash of the preceding entry (32 zero bytes for genesis). |
| `entry_hash` | `bytes` | SHA-256 hash of this entry (computed over fields 1-6). |
| `merkle_root` | `bytes` | Merkle root of the containing batch (last entry only). |

**grpcurl example:**

```bash
grpcurl -plaintext \
  -d '{
    "gpu": {"uuid": "GPU-abcd1234-5678-9abc-def0-123456789abc"},
    "entry_type": 3,
    "limit": 20,
    "descending": true
  }' \
  localhost:50051 sentinel.v1.AuditService/QueryAuditTrail
```

---

### ConfigService

**File:** `proto/sentinel/v1/config.proto`
**Full name:** `sentinel.v1.ConfigService`

The ConfigService manages dynamic configuration distribution via bidirectional
streaming.

#### ConfigStream

```protobuf
rpc ConfigStream(stream ConfigAck) returns (stream ConfigUpdate);
```

**Description:** Bidirectional streaming for dynamic configuration management.
The engine pushes `ConfigUpdate` messages to agents/components; they respond
with `ConfigAck` to report whether the update was applied.

**Engine sends:** `ConfigUpdate`

| Field | Type | Description |
|---|---|---|
| `update_id` | `string` | Unique identifier for this config change. |
| `initiated_by` | `string` | Who initiated (e.g., `"operator:jane"`, `"auto-tuner"`). |
| `reason` | `string` | Human-readable reason. |
| `probe_schedule` | `ProbeScheduleUpdate` | New probe schedule (oneof). |
| `overhead_budget` | `OverheadBudgetUpdate` | New overhead budget (oneof). |
| `sampling_rate` | `SamplingRateUpdate` | New sampling rate (oneof). |
| `threshold` | `ThresholdUpdate` | New threshold (oneof). |

Only one of `probe_schedule`, `overhead_budget`, `sampling_rate`, or
`threshold` is set per update.

**ProbeScheduleUpdate:**

| Field | Type | Description |
|---|---|---|
| `entries` | `repeated ProbeScheduleEntry` | New schedule (replaces entire schedule). |

**ProbeScheduleEntry:**

| Field | Type | Description |
|---|---|---|
| `type` | `ProbeType` | Probe type. |
| `period_seconds` | `uint32` | Execution period in seconds. |
| `sm_coverage` | `double` | Fraction of SMs to cover per period [0.0, 1.0]. |
| `priority` | `uint32` | Scheduling priority (lower = higher). |
| `enabled` | `bool` | Whether enabled. |
| `timeout_ms` | `uint32` | Maximum execution time in ms. |

**OverheadBudgetUpdate:**

| Field | Type | Description |
|---|---|---|
| `budget_pct` | `double` | Maximum GPU time percentage [0.0, 100.0]. |

**SamplingRateUpdate:**

| Field | Type | Description |
|---|---|---|
| `component` | `string` | Component name (e.g., `"inference_monitor"`). |
| `rate` | `double` | New sampling rate [0.0, 1.0]. |

**ThresholdUpdate:**

| Field | Type | Description |
|---|---|---|
| `component` | `string` | Component name. |
| `parameter` | `string` | Parameter name (e.g., `"kl_divergence_threshold"`). |
| `value` | `double` | New threshold value. |

**Agent responds:** `ConfigAck`

| Field | Type | Description |
|---|---|---|
| `update_id` | `string` | The update_id being acknowledged. |
| `applied` | `bool` | Whether the update was applied. |
| `component_id` | `string` | Hostname of the acknowledging agent. |
| `error` | `string` | Error message if not applied. |
| `config_version` | `uint64` | Effective config version after this update. |

> **Note:** The SDKs also expose a unary `ApplyConfig` RPC (extension method)
> for one-shot configuration updates without maintaining a bidirectional stream.

---

## REST API

The REST API is served by the Correlation Engine's embedded axum HTTP server.
All endpoints return JSON. All timestamps are in RFC 3339 format.

Base URL: `http://<host>:8080/api/v1`

### GET /api/v1/health

Server health check endpoint.

**Response:**

```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 86412,
  "grpc_connected": true,
  "redis_connected": true,
  "active_agents": 48
}
```

**Status codes:**

| Code | Description |
|---|---|
| `200 OK` | Server is healthy. |
| `503 Service Unavailable` | Server is starting up or shutting down. |

**curl example:**

```bash
curl -s http://localhost:8080/api/v1/health | jq .
```

---

### GET /api/v1/fleet

Query fleet-wide health summary.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hostname_prefix` | `string` | (none) | Filter by hostname prefix. |
| `model` | `string` | (none) | Filter by GPU model. |
| `state` | `string` | (none) | Comma-separated state filter (e.g., `"suspect,quarantined"`). |

**Response:**

```json
{
  "summary": {
    "total_gpus": 1024,
    "healthy": 1018,
    "suspect": 4,
    "quarantined": 1,
    "deep_test": 1,
    "condemned": 0,
    "overall_sdc_rate": 0.000023,
    "average_reliability_score": 0.999847,
    "snapshot_time": "2026-03-24T12:00:00Z",
    "active_agents": 128,
    "rate_window_seconds": 3600
  },
  "gpu_health": [
    {
      "gpu": {
        "uuid": "GPU-abcd1234-5678-9abc-def0-123456789abc",
        "hostname": "gpu-node-07",
        "device_index": 2,
        "model": "NVIDIA H100 80GB HBM3",
        "driver_version": "535.129.03",
        "firmware_version": "96.00.89.00.01"
      },
      "state": "SUSPECT",
      "reliability_score": 0.9934,
      "probe_pass_count": 14892,
      "probe_fail_count": 3,
      "anomaly_count": 7,
      "anomaly_rate": 0.23,
      "probe_failure_rate": 0.01
    }
  ],
  "truncated": false,
  "total_matching": 1024
}
```

**curl example:**

```bash
# Full fleet.
curl -s http://localhost:8080/api/v1/fleet | jq .

# Filtered.
curl -s "http://localhost:8080/api/v1/fleet?hostname_prefix=gpu-rack-01&state=suspect,quarantined" | jq .
```

---

### GET /api/v1/gpu/:uuid

Query health of a specific GPU.

**Path parameters:**

| Parameter | Description |
|---|---|
| `uuid` | NVIDIA GPU UUID. |

**Response:**

```json
{
  "health": {
    "gpu": {
      "uuid": "GPU-abcd1234-5678-9abc-def0-123456789abc",
      "hostname": "gpu-node-07",
      "device_index": 2,
      "model": "NVIDIA H100 80GB HBM3",
      "driver_version": "535.129.03",
      "firmware_version": "96.00.89.00.01"
    },
    "state": "HEALTHY",
    "reliability_score": 0.999923,
    "alpha": 12984.0,
    "beta": 1.0,
    "last_probe_time": "2026-03-24T11:59:42Z",
    "last_anomaly_time": null,
    "probe_pass_count": 12984,
    "probe_fail_count": 0,
    "anomaly_count": 0,
    "state_changed_at": "2026-03-01T00:00:00Z",
    "state_change_reason": "Initial registration",
    "sm_health": [
      {
        "sm": {"sm_id": 0},
        "reliability_score": 0.999988,
        "probe_pass_count": 1620,
        "probe_fail_count": 0,
        "disabled": false
      }
    ],
    "anomaly_rate": 0.0,
    "probe_failure_rate": 0.0
  },
  "recent_correlations": []
}
```

**Status codes:**

| Code | Description |
|---|---|
| `200 OK` | GPU found. |
| `404 Not Found` | GPU UUID not known. |

**curl example:**

```bash
curl -s http://localhost:8080/api/v1/gpu/GPU-abcd1234-5678-9abc-def0-123456789abc | jq .
```

---

### GET /api/v1/gpu/:uuid/history

Query historical health data for a GPU.

**Path parameters:**

| Parameter | Description |
|---|---|
| `uuid` | NVIDIA GPU UUID. |

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start` | `string` (RFC 3339) | (required) | Start of time range. |
| `end` | `string` (RFC 3339) | (required) | End of time range. |
| `limit` | `integer` | `100` | Maximum events to return. |
| `page_token` | `string` | (none) | Pagination token. |

**Response:**

```json
{
  "state_transitions": [
    {
      "from_state": "HEALTHY",
      "to_state": "SUSPECT",
      "timestamp": "2026-03-20T14:23:00Z",
      "reason": "FMA probe failure rate exceeded threshold",
      "initiated_by": "correlation-engine"
    }
  ],
  "correlations": [
    {
      "event_id": "corr-abcd-1234",
      "events_correlated": ["probe-001", "anomaly-002"],
      "pattern_type": "SM_LOCALIZED",
      "confidence": 0.87,
      "description": "Probe failures and KL divergence anomaly co-occurring on SM 42",
      "severity": "HIGH",
      "recommended_action": "Consider quarantine"
    }
  ],
  "reliability_history": [
    {"timestamp": "2026-03-20T00:00:00Z", "reliability_score": 0.999999, "alpha": 12000, "beta": 1},
    {"timestamp": "2026-03-20T14:23:00Z", "reliability_score": 0.999800, "alpha": 12000, "beta": 3}
  ],
  "next_page_token": ""
}
```

**curl example:**

```bash
curl -s "http://localhost:8080/api/v1/gpu/GPU-abcd1234-5678-9abc-def0-123456789abc/history?start=2026-03-17T00:00:00Z&end=2026-03-24T00:00:00Z&limit=50" | jq .
```

---

### POST /api/v1/quarantine

Issue a quarantine directive.

**Request body:**

```json
{
  "gpu_uuid": "GPU-abcd1234-5678-9abc-def0-123456789abc",
  "action": "QUARANTINE",
  "reason": "Repeated FMA probe failures on SM 42",
  "initiated_by": "dashboard:operator-jane",
  "evidence": ["probe-exec-001", "probe-exec-002"],
  "requires_approval": true
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `gpu_uuid` | `string` | yes | GPU UUID to act upon. |
| `action` | `string` | yes | One of: `QUARANTINE`, `REINSTATE`, `CONDEMN`, `SCHEDULE_DEEP_TEST`. |
| `reason` | `string` | yes | Human-readable reason. |
| `initiated_by` | `string` | no | Who initiated this (defaults to `"rest-api"`). |
| `evidence` | `string[]` | no | Supporting evidence references. |
| `requires_approval` | `boolean` | no | Whether human approval is required (default `false`). |

**Response:**

```json
{
  "directive_id": "dir-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "accepted": true,
  "rejection_reason": "",
  "resulting_state": "QUARANTINED"
}
```

**Status codes:**

| Code | Description |
|---|---|
| `200 OK` | Directive processed (check `accepted` field). |
| `400 Bad Request` | Invalid request body. |
| `404 Not Found` | GPU UUID not known. |
| `403 Forbidden` | Insufficient permissions. |

**curl example:**

```bash
curl -s -X POST http://localhost:8080/api/v1/quarantine \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "gpu_uuid": "GPU-abcd1234-5678-9abc-def0-123456789abc",
    "action": "QUARANTINE",
    "reason": "Repeated FMA probe failures on SM 42",
    "initiated_by": "operator:jane"
  }' | jq .
```

---

### GET /api/v1/trust

Retrieve the current trust graph snapshot.

**Response:**

```json
{
  "edges": [
    {
      "gpu_a": {"uuid": "GPU-aaaa...", "hostname": "node-01"},
      "gpu_b": {"uuid": "GPU-bbbb...", "hostname": "node-01"},
      "agreement_count": 142,
      "disagreement_count": 0,
      "last_comparison": "2026-03-24T11:55:00Z",
      "trust_score": 1.0
    }
  ],
  "timestamp": "2026-03-24T12:00:00Z",
  "coverage_pct": 67.3,
  "total_gpus": 1024,
  "min_trust_score": 0.9856,
  "mean_trust_score": 0.9998
}
```

**curl example:**

```bash
curl -s http://localhost:8080/api/v1/trust | jq .
```

---

### GET /api/v1/audit

Query the audit trail.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gpu_uuid` | `string` | (none) | Filter by GPU UUID. |
| `start` | `string` (RFC 3339) | (none) | Start of time range. |
| `end` | `string` (RFC 3339) | (none) | End of time range. |
| `type` | `string` | (none) | Filter by entry type (e.g., `"QUARANTINE_ACTION"`). |
| `limit` | `integer` | `100` | Maximum entries to return. |
| `page_token` | `string` | (none) | Pagination token. |
| `order` | `string` | `"asc"` | Sort order (`"asc"` or `"desc"`). |

**Response:**

```json
{
  "entries": [
    {
      "entry_id": 42847,
      "entry_type": "QUARANTINE_ACTION",
      "timestamp": "2026-03-24T10:30:00Z",
      "gpu": {
        "uuid": "GPU-abcd1234-5678-9abc-def0-123456789abc",
        "hostname": "gpu-node-07"
      },
      "entry_hash": "a1b2c3d4e5f6...",
      "previous_hash": "f6e5d4c3b2a1..."
    }
  ],
  "next_page_token": "",
  "total_count": 1
}
```

**curl example:**

```bash
curl -s "http://localhost:8080/api/v1/audit?gpu_uuid=GPU-abcd1234-5678-9abc-def0-123456789abc&type=QUARANTINE_ACTION&limit=20&order=desc" | jq .
```

---

## Common Types

### GpuIdentifier

Used across all services to identify a GPU.

```protobuf
message GpuIdentifier {
  string uuid = 1;             // NVIDIA UUID.
  string hostname = 2;         // Host machine.
  uint32 device_index = 3;     // PCI device index (0-based).
  string model = 4;            // GPU model name.
  string driver_version = 5;   // Driver version.
  string firmware_version = 6; // Firmware/VBIOS version.
}
```

For query requests, only `uuid` is required. Other fields are populated in
responses.

### SmIdentifier

```protobuf
message SmIdentifier {
  GpuIdentifier gpu = 1;  // Containing GPU.
  uint32 sm_id = 2;        // SM index (0-based).
}
```

### Severity

```protobuf
enum Severity {
  SEVERITY_UNSPECIFIED = 0;
  INFO = 1;
  WARNING = 2;
  HIGH = 3;
  CRITICAL = 4;
}
```

### GpuHealth message

See [CorrelationService.QueryGpuHealth](#querygpuhealth) response for the
complete field listing.

---

## Error Codes

Standard gRPC error codes used by SENTINEL:

| Code | Number | Description |
|---|---|---|
| `OK` | 0 | Success. |
| `INVALID_ARGUMENT` | 3 | Missing required field, malformed UUID, etc. |
| `NOT_FOUND` | 5 | GPU UUID not known to the system. |
| `ALREADY_EXISTS` | 6 | Duplicate directive ID. |
| `PERMISSION_DENIED` | 7 | Client lacks permission for this operation. |
| `FAILED_PRECONDITION` | 9 | Invalid state transition. |
| `RESOURCE_EXHAUSTED` | 8 | Server backpressure or rate limit. |
| `UNAUTHENTICATED` | 16 | Client authentication failed. |
| `UNAVAILABLE` | 14 | Server temporarily unreachable. |
| `DEADLINE_EXCEEDED` | 4 | RPC timed out. |
| `INTERNAL` | 13 | Unexpected server error. |

For streaming RPCs, the error code may be returned either as a trailing status
or as a mid-stream error (for bidirectional streams). SDKs handle both cases
transparently.
