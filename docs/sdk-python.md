# SENTINEL Python SDK Guide

> **Package:** `sentinel-sdk`
> **Status:** Pre-release Alpha
> **Minimum Python:** 3.10+
> **License:** Apache-2.0

The SENTINEL Python SDK provides both synchronous and asynchronous interfaces for
querying GPU health, fleet status, audit trails, trust graphs, and managing
quarantine directives in a SENTINEL deployment.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Authentication](#authentication)
4. [Client API Reference](#client-api-reference)
   - [Connection](#connection)
   - [GPU Health](#gpu-health)
   - [Fleet Health](#fleet-health)
   - [GPU History](#gpu-history)
   - [Quarantine Directives](#quarantine-directives)
   - [Audit Trail](#audit-trail)
   - [Chain Verification](#chain-verification)
   - [Trust Graph](#trust-graph)
   - [Configuration Updates](#configuration-updates)
   - [Event Streaming](#event-streaming)
   - [Connection Management](#connection-management)
5. [Type Reference](#type-reference)
6. [Sync vs Async Usage Patterns](#sync-vs-async-usage-patterns)
7. [Error Handling](#error-handling)
8. [Configuration via Environment Variables](#configuration-via-environment-variables)
9. [Integration Examples](#integration-examples)
10. [Troubleshooting](#troubleshooting)
11. [Version Compatibility](#version-compatibility)

---

## Installation

```bash
pip install sentinel-sdk
```

The SDK depends on:
- `grpcio >= 1.60.0`
- `grpcio-tools >= 1.60.0` (for protobuf stubs)
- `pydantic >= 2.0`
- `protobuf >= 4.25.0`

For async support, no additional dependencies are required -- `grpcio` ships
with `grpc.aio` built in.

### Installing from source

```bash
git clone https://github.com/sentinel-sdc/sentinel.git
cd sentinel/sdk/python
pip install -e ".[dev]"
```

---

## Quick Start

### Synchronous

```python
from sentinel_sdk import SentinelClient, TlsConfig

# Connect to the SENTINEL correlation engine.
client = SentinelClient.connect(
    "sentinel.example.com:443",
    tls_config=TlsConfig(ca_cert_path="/etc/sentinel/ca.pem"),
)

# Check a single GPU.
health = client.query_gpu_health("GPU-abcd1234-5678-9abc-def0-123456789abc")
print(f"GPU state: {health.state.name}, reliability: {health.reliability_score:.4f}")

# Check the entire fleet.
fleet = client.query_fleet_health()
print(f"Fleet: {fleet.healthy}/{fleet.total_gpus} healthy, SDC rate: {fleet.overall_sdc_rate:.6f}")

client.close()
```

### Asynchronous

```python
import asyncio
from sentinel_sdk import SentinelClient, TlsConfig

async def main():
    client = await SentinelClient.aconnect(
        "sentinel.example.com:443",
        tls_config=TlsConfig(ca_cert_path="/etc/sentinel/ca.pem"),
    )

    health = await client.aquery_gpu_health("GPU-abcd1234-5678-9abc-def0-123456789abc")
    print(f"GPU state: {health.state.name}")

    await client.aclose()

asyncio.run(main())
```

### Context manager

```python
with SentinelClient.connect("localhost:50051") as client:
    fleet = client.query_fleet_health()
    print(f"{fleet.total_gpus} GPUs tracked")
```

```python
async with await SentinelClient.aconnect("localhost:50051") as client:
    fleet = await client.aquery_fleet_health()
    print(f"{fleet.total_gpus} GPUs tracked")
```

---

## Authentication

SENTINEL supports three connection modes:

### Insecure (development only)

```python
client = SentinelClient.connect("localhost:50051")
```

No TLS. Suitable only for local development. Never use in production.

### Server-side TLS

```python
client = SentinelClient.connect(
    "sentinel.example.com:443",
    tls_config=TlsConfig(
        ca_cert_path="/etc/sentinel/ca.pem",
    ),
)
```

The client verifies the server certificate against the provided CA. The server
does not authenticate the client.

### Mutual TLS (mTLS)

```python
client = SentinelClient.connect(
    "sentinel.example.com:443",
    tls_config=TlsConfig(
        ca_cert_path="/etc/sentinel/ca.pem",
        client_cert_path="/etc/sentinel/client.pem",
        client_key_path="/etc/sentinel/client-key.pem",
    ),
)
```

Both client and server authenticate each other. This is the recommended
production configuration.

### TlsConfig Reference

```python
@dataclass
class TlsConfig:
    ca_cert_path: Optional[str] = None        # Path to CA certificate (PEM).
    client_cert_path: Optional[str] = None     # Path to client certificate (PEM).
    client_key_path: Optional[str] = None      # Path to client private key (PEM).
```

| Field | Description |
|---|---|
| `ca_cert_path` | Path to the PEM-encoded CA certificate used to verify the server. If `None`, the system default trust store is used. |
| `client_cert_path` | Path to the PEM-encoded client certificate for mTLS. Must be provided together with `client_key_path`. |
| `client_key_path` | Path to the PEM-encoded client private key for mTLS. Must be provided together with `client_cert_path`. |

### RetryConfig Reference

```python
@dataclass
class RetryConfig:
    max_retries: int = 3                     # Maximum retry attempts.
    initial_backoff_s: float = 0.1           # Initial backoff in seconds.
    max_backoff_s: float = 10.0              # Maximum backoff cap in seconds.
    backoff_multiplier: float = 2.0          # Exponential backoff multiplier.
    retryable_status_codes: list[grpc.StatusCode] = [
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
    ]
```

| Field | Description |
|---|---|
| `max_retries` | Number of times to retry a failed RPC before raising. Set to `0` to disable retries. |
| `initial_backoff_s` | How long to wait before the first retry, in seconds. |
| `max_backoff_s` | Upper bound on backoff duration, in seconds. |
| `backoff_multiplier` | Each successive backoff is multiplied by this factor. |
| `retryable_status_codes` | gRPC status codes that trigger a retry. Non-matching codes raise immediately. |

---

## Client API Reference

### Connection

#### `SentinelClient.connect(endpoint, *, tls_config=None, retry_config=None, default_timeout=30.0, options=None)`

Create a synchronous client connected to the SENTINEL gRPC gateway.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `endpoint` | `str` | (required) | `host:port` of the SENTINEL gRPC gateway. |
| `tls_config` | `Optional[TlsConfig]` | `None` | TLS credentials. If `None`, an insecure channel is used. |
| `retry_config` | `Optional[RetryConfig]` | `None` | Retry behaviour. Defaults to 3 retries with exponential backoff. |
| `default_timeout` | `float` | `30.0` | Default per-RPC deadline in seconds. |
| `options` | `Optional[list[tuple[str, Any]]]` | `None` | Additional gRPC channel options (e.g., keepalive settings). |

**Returns:** `SentinelClient`

**Example:**

```python
client = SentinelClient.connect(
    "sentinel.prod.internal:443",
    tls_config=TlsConfig(ca_cert_path="ca.pem"),
    retry_config=RetryConfig(max_retries=5),
    default_timeout=15.0,
)
```

#### `SentinelClient.aconnect(endpoint, *, tls_config=None, retry_config=None, default_timeout=30.0, options=None)`

Create an asynchronous client connected to the SENTINEL gRPC gateway.
Same parameters as `connect`. Must be called with `await`.

**Returns:** `SentinelClient`

**Example:**

```python
client = await SentinelClient.aconnect(
    "sentinel.prod.internal:443",
    tls_config=TlsConfig(ca_cert_path="ca.pem"),
)
```

---

### GPU Health

#### `query_gpu_health(gpu_uuid: str) -> GpuHealth`

Query the health of a specific GPU by its UUID. Returns the GPU's current
lifecycle state, Bayesian reliability score, probe statistics, anomaly counts,
and per-SM health breakdown.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `gpu_uuid` | `str` | NVIDIA GPU UUID (e.g., `"GPU-abcd1234-..."`) |

**Returns:** [`GpuHealth`](#gpuhealth)

**Raises:** `NotFoundError` if the GPU UUID is unknown to the system.

**Example:**

```python
health = client.query_gpu_health("GPU-abcd1234-5678-9abc-def0-123456789abc")

print(f"State:       {health.state.name}")
print(f"Reliability: {health.reliability_score:.6f}")
print(f"Probes:      {health.probe_pass_count} pass / {health.probe_fail_count} fail")
print(f"Anomalies:   {health.anomaly_count}")

# Inspect per-SM health.
for sm in health.sm_health:
    if sm.probe_fail_count > 0:
        print(f"  SM {sm.sm.sm_id}: {sm.probe_fail_count} failures, "
              f"reliability={sm.reliability_score:.4f}")
```

#### `aquery_gpu_health(gpu_uuid: str) -> GpuHealth`

Async variant. Same parameters and return type.

```python
health = await client.aquery_gpu_health("GPU-abcd1234-5678-9abc-def0-123456789abc")
```

---

### Fleet Health

#### `query_fleet_health(hostname_prefix="", model_filter="", state_filter=None) -> FleetHealthSummary`

Query the fleet-wide health summary. Optionally filter by hostname prefix, GPU
model, or lifecycle state.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hostname_prefix` | `str` | `""` | Only include GPUs on hosts matching this prefix (e.g., `"gpu-rack-01"`). Empty string matches all. |
| `model_filter` | `str` | `""` | Only include GPUs of this model (e.g., `"H100"`). Empty string matches all. |
| `state_filter` | `Optional[list[GpuHealthState]]` | `None` | Only include GPUs in these lifecycle states. `None` includes all states. |

**Returns:** [`FleetHealthSummary`](#fleethealthsummary)

**Example:**

```python
# Get full fleet summary.
fleet = client.query_fleet_health()
print(f"Total: {fleet.total_gpus} | Healthy: {fleet.healthy} | "
      f"Suspect: {fleet.suspect} | Quarantined: {fleet.quarantined}")
print(f"SDC rate: {fleet.overall_sdc_rate:.6f} events/GPU-hour")
print(f"Active agents: {fleet.active_agents}")

# Filter to just H100s on rack 3.
from sentinel_sdk.types import GpuHealthState
fleet = client.query_fleet_health(
    hostname_prefix="gpu-rack-03",
    model_filter="H100",
    state_filter=[GpuHealthState.SUSPECT, GpuHealthState.QUARANTINED],
)
```

#### `aquery_fleet_health(...) -> FleetHealthSummary`

Async variant. Same parameters and return type.

---

### GPU History

#### `get_gpu_history(gpu_uuid, start_time, end_time, limit=1000, page_token="") -> GpuHistoryResponse`

Retrieve historical health data for a GPU within a time range, including state
transitions, correlation events, and reliability score time series.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gpu_uuid` | `str` | (required) | GPU UUID to query. |
| `start_time` | `datetime` | (required) | Start of the time range (UTC). |
| `end_time` | `datetime` | (required) | End of the time range (UTC). |
| `limit` | `int` | `1000` | Maximum number of events to return per page. |
| `page_token` | `str` | `""` | Pagination token from a previous response. |

**Returns:** [`GpuHistoryResponse`](#gpuhistoryresponse)

**Example:**

```python
from datetime import datetime, timedelta

end = datetime.utcnow()
start = end - timedelta(days=7)

history = client.get_gpu_history(
    "GPU-abcd1234-5678-9abc-def0-123456789abc",
    start_time=start,
    end_time=end,
)

print(f"State transitions: {len(history.state_transitions)}")
for t in history.state_transitions:
    print(f"  {t.timestamp}: {t.from_state.name} -> {t.to_state.name} ({t.reason})")

print(f"Correlations: {len(history.correlations)}")
print(f"Reliability samples: {len(history.reliability_history)}")

# Pagination.
while history.next_page_token:
    history = client.get_gpu_history(
        "GPU-abcd1234-5678-9abc-def0-123456789abc",
        start_time=start,
        end_time=end,
        page_token=history.next_page_token,
    )
    # Process next page...
```

#### `aget_gpu_history(...) -> GpuHistoryResponse`

Async variant. Same parameters and return type.

---

### Quarantine Directives

#### `issue_quarantine(gpu_uuid, action, reason, initiated_by="sentinel-sdk", evidence=None, requires_approval=False) -> DirectiveResponse`

Issue a quarantine directive to change a GPU's lifecycle state.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gpu_uuid` | `str` | (required) | GPU UUID to act upon. |
| `action` | `QuarantineAction` | (required) | The action to take. |
| `reason` | `str` | (required) | Human-readable reason for this action. |
| `initiated_by` | `str` | `"sentinel-sdk"` | Identifier for who/what initiated this directive. |
| `evidence` | `Optional[list[str]]` | `None` | List of supporting evidence references (event IDs, probe execution IDs, etc.). |
| `requires_approval` | `bool` | `False` | Whether this directive requires human approval before execution. |

**Returns:** [`DirectiveResponse`](#directiveresponse)

**QuarantineAction values:**

| Value | Description |
|---|---|
| `QuarantineAction.QUARANTINE` | Remove GPU from production workloads and place under investigation. |
| `QuarantineAction.REINSTATE` | Return a previously quarantined GPU to production. |
| `QuarantineAction.CONDEMN` | Permanently mark GPU as unreliable; schedule for hardware replacement. |
| `QuarantineAction.SCHEDULE_DEEP_TEST` | Initiate a comprehensive deep-diagnostic test suite on the GPU. |

**Example:**

```python
from sentinel_sdk.types import QuarantineAction

# Quarantine a suspect GPU.
resp = client.issue_quarantine(
    gpu_uuid="GPU-abcd1234-5678-9abc-def0-123456789abc",
    action=QuarantineAction.QUARANTINE,
    reason="Repeated FMA probe failures on SM 42",
    evidence=["probe-exec-001", "probe-exec-002", "anomaly-evt-007"],
    requires_approval=True,
)

if resp.accepted:
    print(f"Directive {resp.directive_id} accepted, GPU now {resp.resulting_state}")
else:
    print(f"Directive rejected: {resp.rejection_reason}")
```

#### `aissue_quarantine(...) -> DirectiveResponse`

Async variant. Same parameters and return type.

---

### Audit Trail

#### `query_audit_trail(filters: AuditQueryFilters) -> AuditQueryResponse`

Query the tamper-evident audit trail with filtering and pagination.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `filters` | `AuditQueryFilters` | Query filters (see below). |

**AuditQueryFilters fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `gpu` | `Optional[GpuIdentifier]` | `None` | Filter by GPU. |
| `start_time` | `Optional[datetime]` | `None` | Start of time range. |
| `end_time` | `Optional[datetime]` | `None` | End of time range. |
| `entry_type` | `AuditEntryType` | `UNSPECIFIED` | Filter by entry type. `UNSPECIFIED` returns all types. |
| `limit` | `int` | `100` | Maximum entries to return. |
| `page_token` | `str` | `""` | Pagination token. |
| `descending` | `bool` | `False` | If `True`, return newest entries first. |

**Returns:** [`AuditQueryResponse`](#auditqueryresponse)

**AuditEntryType values:**

| Value | Description |
|---|---|
| `PROBE_RESULT` | A probe execution result. |
| `ANOMALY_EVENT` | An anomaly detection event. |
| `QUARANTINE_ACTION` | A quarantine lifecycle action. |
| `CONFIG_CHANGE` | A configuration change. |
| `TMR_RESULT` | A TMR canary result. |
| `SYSTEM_EVENT` | A system-level event (startup, shutdown, error). |

**Example:**

```python
from datetime import datetime, timedelta
from sentinel_sdk.types import AuditQueryFilters, AuditEntryType, GpuIdentifier

filters = AuditQueryFilters(
    gpu=GpuIdentifier(uuid="GPU-abcd1234-5678-9abc-def0-123456789abc"),
    start_time=datetime.utcnow() - timedelta(hours=24),
    entry_type=AuditEntryType.QUARANTINE_ACTION,
    limit=50,
    descending=True,
)

result = client.query_audit_trail(filters)
print(f"Found {result.total_count} entries (showing {len(result.entries)})")

for entry in result.entries:
    print(f"  [{entry.entry_id}] {entry.entry_type.name} at {entry.timestamp}")
```

#### `aquery_audit_trail(filters: AuditQueryFilters) -> AuditQueryResponse`

Async variant. Same parameters and return type.

---

### Chain Verification

#### `verify_chain(start_time=None, end_time=None, start_entry_id=0, end_entry_id=0, verify_merkle_roots=True) -> ChainVerificationResult`

Verify the integrity of the audit chain. This checks that the hash chain is
unbroken and optionally verifies Merkle roots of batches.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start_time` | `Optional[datetime]` | `None` | Start of verification range (reserved for future use). |
| `end_time` | `Optional[datetime]` | `None` | End of verification range (reserved for future use). |
| `start_entry_id` | `int` | `0` | Starting entry ID (0 = from genesis). |
| `end_entry_id` | `int` | `0` | Ending entry ID (0 = through latest). |
| `verify_merkle_roots` | `bool` | `True` | Whether to also verify Merkle roots of batches. |

**Returns:** [`ChainVerificationResult`](#chainverificationresult)

**Example:**

```python
result = client.verify_chain()

if result.valid:
    print(f"Chain integrity verified: {result.entries_verified} entries, "
          f"{result.batches_verified} batches in {result.verification_time_ms}ms")
else:
    print(f"CHAIN BROKEN at entry {result.first_invalid_entry_id}: "
          f"{result.failure_description}")
```

#### `averify_chain(...) -> ChainVerificationResult`

Async variant. Same parameters and return type.

---

### Trust Graph

#### `get_trust_graph() -> TrustGraphSnapshot`

Retrieve a point-in-time snapshot of the GPU trust graph. The trust graph
records pairwise comparison history from TMR canary runs.

**Returns:** [`TrustGraphSnapshot`](#trustgraphsnapshot)

**Example:**

```python
graph = client.get_trust_graph()

print(f"Trust graph: {graph.total_gpus} GPUs, {len(graph.edges)} edges")
print(f"Coverage: {graph.coverage_pct:.1f}%")
print(f"Trust scores: min={graph.min_trust_score:.4f}, mean={graph.mean_trust_score:.4f}")

# Find low-trust edges.
for edge in graph.edges:
    if edge.trust_score < 0.95:
        print(f"  Low trust: {edge.gpu_a.uuid[:16]}... <-> {edge.gpu_b.uuid[:16]}... "
              f"score={edge.trust_score:.4f} "
              f"({edge.agreement_count} agree / {edge.disagreement_count} disagree)")
```

#### `aget_trust_graph() -> TrustGraphSnapshot`

Async variant. Same return type.

---

### Configuration Updates

#### `update_config(update: ConfigUpdate) -> ConfigAck`

Push a dynamic configuration update to agents or subsystems.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `update` | `ConfigUpdate` | The configuration update to apply. |

A `ConfigUpdate` carries exactly one of the following update payloads:

| Field | Type | Description |
|---|---|---|
| `probe_schedule` | `ProbeScheduleUpdate` | Replace the probe execution schedule. |
| `overhead_budget` | `OverheadBudgetUpdate` | Change the maximum GPU overhead budget. |
| `sampling_rate` | `SamplingRateUpdate` | Update a component's sampling rate. |
| `threshold` | `ThresholdUpdate` | Update a component's detection threshold. |

**Returns:** [`ConfigAck`](#configack)

**Example:**

```python
from sentinel_sdk.types import (
    ConfigUpdate, ProbeScheduleUpdate, ProbeScheduleEntry, ProbeType,
    OverheadBudgetUpdate, ThresholdUpdate,
)

# Update probe schedule.
ack = client.update_config(ConfigUpdate(
    update_id="cfg-001",
    initiated_by="operator:jane",
    reason="Increase FMA probe frequency during investigation",
    probe_schedule=ProbeScheduleUpdate(entries=[
        ProbeScheduleEntry(
            type=ProbeType.FMA,
            period_seconds=30,       # Every 30 seconds (was 60).
            sm_coverage=1.0,         # All SMs.
            priority=1,
            enabled=True,
            timeout_ms=5000,
        ),
        ProbeScheduleEntry(
            type=ProbeType.TENSOR_CORE,
            period_seconds=120,
            sm_coverage=0.25,
            priority=2,
            enabled=True,
            timeout_ms=10000,
        ),
    ]),
))

if ack.applied:
    print(f"Config {ack.update_id} applied by {ack.component_id} (v{ack.config_version})")
else:
    print(f"Config rejected: {ack.error}")

# Adjust overhead budget.
ack = client.update_config(ConfigUpdate(
    update_id="cfg-002",
    initiated_by="auto-tuner",
    reason="Reduce probe overhead during peak training",
    overhead_budget=OverheadBudgetUpdate(budget_pct=1.0),
))

# Adjust detection threshold.
ack = client.update_config(ConfigUpdate(
    update_id="cfg-003",
    initiated_by="operator:jane",
    reason="Lower KL divergence sensitivity",
    threshold=ThresholdUpdate(
        component="inference_monitor",
        parameter="kl_divergence_threshold",
        value=0.05,
    ),
))
```

#### `aupdate_config(update: ConfigUpdate) -> ConfigAck`

Async variant. Same parameters and return type.

---

### Event Streaming

#### `stream_events(callback, hostname_filter="", action_filter=QuarantineAction.UNSPECIFIED) -> None`

Stream quarantine directives in real-time. This method **blocks** until the
client is closed, the server terminates the stream, or a non-retryable error
occurs. The SDK automatically reconnects on transient failures.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `callback` | `Callable[[QuarantineDirective], None]` | (required) | Invoked for each incoming directive. |
| `hostname_filter` | `str` | `""` | Only receive directives for this hostname. Empty = all. |
| `action_filter` | `QuarantineAction` | `UNSPECIFIED` | Only receive directives of this type. `UNSPECIFIED` = all. |

**Example:**

```python
from sentinel_sdk.types import QuarantineDirective

def on_directive(d: QuarantineDirective):
    print(f"[{d.timestamp}] {d.action.name} -> {d.gpu.uuid} ({d.reason})")
    if d.requires_approval:
        print(f"  Awaiting approval from {d.initiated_by}")

# This blocks forever (until client.close() is called).
client.stream_events(on_directive)
```

#### `astream_events(callback, hostname_filter="", action_filter=QuarantineAction.UNSPECIFIED) -> None`

Async variant. The callback may be a regular function or a coroutine function
(`async def`). If it is a coroutine, it will be awaited.

```python
async def on_directive(d: QuarantineDirective):
    print(f"Directive: {d.action.name} -> {d.gpu.uuid}")
    # Can await other async operations here.

await client.astream_events(on_directive)
```

---

### Connection Management

#### `close() -> None`

Close the client and release all resources (synchronous).

#### `aclose() -> None`

Close the client and release all resources (asynchronous). Must be awaited.

The client supports context managers:

```python
# Sync
with SentinelClient.connect("localhost:50051") as client:
    ...

# Async
async with await SentinelClient.aconnect("localhost:50051") as client:
    ...
```

---

## Type Reference

All types are Pydantic models defined in `sentinel_sdk.types`. They mirror the
protobuf messages defined in `proto/sentinel/v1/`.

### Enumerations

#### Severity

| Value | Int | Description |
|---|---|---|
| `UNSPECIFIED` | 0 | Not set. |
| `INFO` | 1 | Informational. |
| `WARNING` | 2 | Warning level. |
| `HIGH` | 3 | High severity. |
| `CRITICAL` | 4 | Critical severity. |

#### ProbeType

| Value | Int | Description |
|---|---|---|
| `UNSPECIFIED` | 0 | Not set. |
| `FMA` | 1 | Fused multiply-add determinism check. |
| `TENSOR_CORE` | 2 | Tensor Core matrix-multiply reproducibility check. |
| `TRANSCENDENTAL` | 3 | Transcendental function (sin, cos, exp, log) accuracy check. |
| `AES` | 4 | AES-based combinational logic exhaustive-path check. |
| `MEMORY` | 5 | GPU global memory integrity check (walking-ones / MATS+). |
| `REGISTER_FILE` | 6 | Register file integrity check via known-pattern writes. |
| `SHARED_MEMORY` | 7 | Shared memory integrity check. |

#### ProbeResult

| Value | Int | Description |
|---|---|---|
| `UNSPECIFIED` | 0 | Not set. |
| `PASS` | 1 | Output matched expected golden value. |
| `FAIL` | 2 | Output did NOT match expected golden value. |
| `ERROR` | 3 | Execution error (kernel launch failure, etc.). |
| `TIMEOUT` | 4 | Probe did not complete within the allowed time window. |

#### GpuHealthState

| Value | Int | Description |
|---|---|---|
| `UNSPECIFIED` | 0 | Not set. |
| `HEALTHY` | 1 | Operating normally; no evidence of SDC. |
| `SUSPECT` | 2 | Anomalous signals detected; under increased monitoring. |
| `QUARANTINED` | 3 | Removed from production workloads pending investigation. |
| `DEEP_TEST` | 4 | Undergoing deep diagnostic testing. |
| `CONDEMNED` | 5 | Permanently marked as unreliable; must be replaced. |

#### QuarantineAction

| Value | Int | Description |
|---|---|---|
| `UNSPECIFIED` | 0 | Not set. |
| `QUARANTINE` | 1 | Remove from production; place under investigation. |
| `REINSTATE` | 2 | Return to production. |
| `CONDEMN` | 3 | Permanently mark as unreliable. |
| `SCHEDULE_DEEP_TEST` | 4 | Initiate deep diagnostic testing. |

#### AnomalyType

| Value | Int | Description |
|---|---|---|
| `UNSPECIFIED` | 0 | Not set. |
| `LOGIT_DRIFT` | 1 | EWMA-smoothed logit distribution drift beyond threshold. |
| `ENTROPY_ANOMALY` | 2 | Output entropy abnormally high or low. |
| `KL_DIVERGENCE` | 3 | KL divergence between reference and observed exceeds limit. |
| `GRADIENT_NORM_SPIKE` | 4 | Gradient norm spike during training. |
| `LOSS_SPIKE` | 5 | Training loss spike not explained by learning-rate schedule. |
| `CROSS_RANK_DIVERGENCE` | 6 | Divergence between ranks in data-parallel training. |
| `CHECKPOINT_DIVERGENCE` | 7 | Checkpointed model differs from expected. |
| `INVARIANT_VIOLATION` | 8 | Mathematical invariant violated (e.g., softmax sums to 1). |

#### AnomalySource

| Value | Int | Description |
|---|---|---|
| `UNSPECIFIED` | 0 | Not set. |
| `INFERENCE_MONITOR` | 1 | Inference monitoring sidecar. |
| `TRAINING_MONITOR` | 2 | Training monitoring hooks. |
| `INVARIANT_CHECKER` | 3 | Mathematical invariant checker. |

#### PatternType

| Value | Int | Description |
|---|---|---|
| `UNSPECIFIED` | 0 | Not set. |
| `MULTI_SIGNAL` | 1 | Multiple anomaly types on the same GPU in a time window. |
| `SM_LOCALIZED` | 2 | Probe failures and anomalies co-occurring on the same SM. |
| `ENVIRONMENTAL` | 3 | Thermal/power anomalies correlating with computation errors. |
| `NODE_CORRELATED` | 4 | Multiple GPUs on the same node showing correlated failures. |
| `FIRMWARE_CORRELATED` | 5 | Failures correlated with specific firmware or driver versions. |
| `TMR_CONFIRMED` | 6 | TMR dissent correlating with other signals. |

#### AuditEntryType

| Value | Int | Description |
|---|---|---|
| `UNSPECIFIED` | 0 | Not set. |
| `PROBE_RESULT` | 1 | A probe execution result. |
| `ANOMALY_EVENT` | 2 | An anomaly detection event. |
| `QUARANTINE_ACTION` | 3 | A quarantine lifecycle action. |
| `CONFIG_CHANGE` | 4 | A configuration change. |
| `TMR_RESULT` | 5 | A TMR canary result. |
| `SYSTEM_EVENT` | 6 | A system-level event. |

### Models

#### GpuIdentifier

Uniquely identifies a GPU within the fleet.

| Field | Type | Description |
|---|---|---|
| `uuid` | `str` | NVIDIA UUID (e.g., `"GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"`). |
| `hostname` | `str` | Hostname of the machine containing this GPU. |
| `device_index` | `int` | PCI device index on the host (0-based). |
| `model` | `str` | GPU model name (e.g., `"NVIDIA H100 80GB HBM3"`). |
| `driver_version` | `str` | Driver version string (e.g., `"535.129.03"`). |
| `firmware_version` | `str` | GPU firmware/VBIOS version. |

#### SmIdentifier

Identifies a specific Streaming Multiprocessor on a GPU.

| Field | Type | Description |
|---|---|---|
| `gpu` | `Optional[GpuIdentifier]` | The GPU containing this SM. |
| `sm_id` | `int` | SM index within the GPU (0-based). |

#### GpuHealth

Health status and Bayesian reliability model for a single GPU.

| Field | Type | Description |
|---|---|---|
| `gpu` | `Optional[GpuIdentifier]` | The GPU this health record describes. |
| `state` | `GpuHealthState` | Current lifecycle state. |
| `reliability_score` | `float` | Bayesian reliability score: `alpha / (alpha + beta)`. Range [0.0, 1.0]. |
| `alpha` | `float` | Beta distribution alpha parameter (successes + prior). |
| `beta` | `float` | Beta distribution beta parameter (failures + prior). |
| `last_probe_time` | `Optional[datetime]` | Timestamp of the most recent probe execution. |
| `last_anomaly_time` | `Optional[datetime]` | Timestamp of the most recent attributed anomaly. |
| `probe_pass_count` | `int` | Cumulative count of passed probes (lifetime). |
| `probe_fail_count` | `int` | Cumulative count of failed probes (lifetime). |
| `anomaly_count` | `int` | Cumulative count of attributed anomalies (lifetime). |
| `state_changed_at` | `Optional[datetime]` | When the GPU last transitioned to its current state. |
| `state_change_reason` | `str` | Reason for the most recent state change. |
| `sm_health` | `list[SmHealth]` | Per-SM health breakdown. |
| `anomaly_rate` | `float` | Current anomaly rate (anomalies per hour, rolling window). |
| `probe_failure_rate` | `float` | Current probe failure rate (failures per hour, rolling window). |

#### SmHealth

Health status for a single Streaming Multiprocessor.

| Field | Type | Description |
|---|---|---|
| `sm` | `Optional[SmIdentifier]` | The SM this record describes. |
| `reliability_score` | `float` | Bayesian reliability score for this SM. |
| `probe_pass_count` | `int` | Count of passed probes on this SM. |
| `probe_fail_count` | `int` | Count of failed probes on this SM. |
| `disabled` | `bool` | Whether this SM is currently disabled/masked. |
| `disable_reason` | `str` | Reason for disabling (if applicable). |

#### FleetHealthSummary

Aggregated health summary for the entire GPU fleet.

| Field | Type | Description |
|---|---|---|
| `total_gpus` | `int` | Total number of GPUs tracked. |
| `healthy` | `int` | Number of GPUs in HEALTHY state. |
| `suspect` | `int` | Number of GPUs in SUSPECT state. |
| `quarantined` | `int` | Number of GPUs in QUARANTINED state. |
| `deep_test` | `int` | Number of GPUs in DEEP_TEST state. |
| `condemned` | `int` | Number of GPUs in CONDEMNED state. |
| `overall_sdc_rate` | `float` | Fleet-wide estimated SDC rate (events per GPU-hour). |
| `average_reliability_score` | `float` | Fleet-wide average reliability score. |
| `snapshot_time` | `Optional[datetime]` | Timestamp of this summary snapshot. |
| `active_agents` | `int` | Number of active probe agents reporting. |
| `rate_window_seconds` | `int` | Time window over which rates are computed (seconds). |

#### GpuHistoryResponse

Historical health data for a GPU.

| Field | Type | Description |
|---|---|---|
| `state_transitions` | `list[StateTransition]` | Historical state transitions. |
| `correlations` | `list[CorrelationEvent]` | Historical correlation events. |
| `reliability_history` | `list[ReliabilitySample]` | Reliability score time series (sampled). |
| `next_page_token` | `str` | Pagination token for the next page (empty if no more results). |

#### StateTransition

A recorded GPU state transition.

| Field | Type | Description |
|---|---|---|
| `from_state` | `GpuHealthState` | Previous state. |
| `to_state` | `GpuHealthState` | New state. |
| `timestamp` | `Optional[datetime]` | When the transition occurred. |
| `reason` | `str` | Reason for the transition. |
| `initiated_by` | `str` | Who/what initiated the transition. |

#### ReliabilitySample

A point-in-time reliability score sample.

| Field | Type | Description |
|---|---|---|
| `timestamp` | `Optional[datetime]` | When this sample was recorded. |
| `reliability_score` | `float` | Reliability score at this point in time. |
| `alpha` | `float` | Beta distribution alpha parameter. |
| `beta` | `float` | Beta distribution beta parameter. |

#### CorrelationEvent

A correlation event linking multiple raw events into a higher-level finding.

| Field | Type | Description |
|---|---|---|
| `event_id` | `str` | Unique identifier (UUID v7). |
| `events_correlated` | `list[str]` | IDs of the raw events that were correlated. |
| `pattern_type` | `PatternType` | The type of pattern detected. |
| `confidence` | `float` | Confidence score [0.0, 1.0]. |
| `attributed_gpu` | `Optional[GpuIdentifier]` | GPU attributed as root cause (if determined). |
| `attributed_sm` | `Optional[SmIdentifier]` | SM attributed as root cause (if localized). |
| `description` | `str` | Human-readable description. |
| `timestamp` | `Optional[datetime]` | When this correlation was computed. |
| `severity` | `Severity` | Severity assessment. |
| `recommended_action` | `str` | Recommended action based on this correlation. |

#### QuarantineDirective

A directive to change a GPU's lifecycle state.

| Field | Type | Description |
|---|---|---|
| `directive_id` | `str` | Unique identifier (UUID v7). |
| `gpu` | `Optional[GpuIdentifier]` | The GPU to act upon. |
| `action` | `QuarantineAction` | The action to take. |
| `reason` | `str` | Human-readable reason. |
| `initiated_by` | `str` | Who/what initiated this directive. |
| `evidence` | `list[str]` | References to supporting evidence. |
| `timestamp` | `Optional[datetime]` | When this directive was issued. |
| `priority` | `int` | Priority (lower = higher priority). |
| `requires_approval` | `bool` | Whether human approval is required. |
| `approval` | `Optional[ApprovalStatus]` | Approval status (populated after review). |

#### ApprovalStatus

Approval tracking for directives requiring human sign-off.

| Field | Type | Description |
|---|---|---|
| `approved` | `bool` | Whether the directive has been approved. |
| `reviewer` | `str` | Who approved or rejected it. |
| `review_time` | `Optional[datetime]` | When the review decision was made. |
| `comment` | `str` | Optional comment from the reviewer. |

#### DirectiveResponse

Response to a directive issuance.

| Field | Type | Description |
|---|---|---|
| `directive_id` | `str` | The directive ID that was processed. |
| `accepted` | `bool` | Whether the directive was accepted. |
| `rejection_reason` | `str` | Reason if not accepted. |
| `resulting_state` | `str` | The resulting GPU state after applying the directive. |

#### AuditEntry

A single entry in the tamper-evident audit ledger.

| Field | Type | Description |
|---|---|---|
| `entry_id` | `int` | Monotonically increasing sequence number. |
| `entry_type` | `AuditEntryType` | Classification of this entry. |
| `timestamp` | `Optional[datetime]` | When this entry was recorded. |
| `gpu` | `Optional[GpuIdentifier]` | GPU associated with this entry. |
| `data` | `bytes` | Serialized protobuf of the underlying event. |
| `previous_hash` | `bytes` | Hash of the preceding entry in the chain. |
| `entry_hash` | `bytes` | SHA-256 hash of this entry. |
| `merkle_root` | `bytes` | Merkle root of the batch (last entry in batch only). |

#### AuditQueryFilters

Filters for querying the audit trail. See [Audit Trail](#audit-trail).

#### AuditQueryResponse

Response from an audit trail query.

| Field | Type | Description |
|---|---|---|
| `entries` | `list[AuditEntry]` | Matching audit entries. |
| `next_page_token` | `str` | Pagination token for the next page. |
| `total_count` | `int` | Total number of matching entries. |

#### ChainVerificationResult

Response from chain verification.

| Field | Type | Description |
|---|---|---|
| `valid` | `bool` | Whether the chain is valid over the requested range. |
| `first_invalid_entry_id` | `int` | Entry ID where the first break was detected (if invalid). |
| `failure_description` | `str` | Description of the verification failure (if any). |
| `entries_verified` | `int` | Total entries verified. |
| `batches_verified` | `int` | Total batches verified. |
| `verification_time_ms` | `int` | Time taken in milliseconds. |

#### TrustEdge

An edge in the GPU trust graph.

| Field | Type | Description |
|---|---|---|
| `gpu_a` | `Optional[GpuIdentifier]` | First GPU in the pair. |
| `gpu_b` | `Optional[GpuIdentifier]` | Second GPU in the pair. |
| `agreement_count` | `int` | Number of matching outputs. |
| `disagreement_count` | `int` | Number of differing outputs. |
| `last_comparison` | `Optional[datetime]` | Most recent comparison timestamp. |
| `trust_score` | `float` | Trust score: `agreement / (agreement + disagreement)`. Range [0.0, 1.0]. |

#### TrustGraphSnapshot

Point-in-time snapshot of the entire trust graph.

| Field | Type | Description |
|---|---|---|
| `edges` | `list[TrustEdge]` | All edges in the trust graph. |
| `timestamp` | `Optional[datetime]` | When this snapshot was taken. |
| `coverage_pct` | `float` | Percentage of all GPU pairs compared at least once. |
| `total_gpus` | `int` | Total GPUs in the graph. |
| `min_trust_score` | `float` | Minimum trust score across all edges. |
| `mean_trust_score` | `float` | Mean trust score across all edges. |

#### ConfigUpdate

A dynamic configuration update.

| Field | Type | Description |
|---|---|---|
| `update_id` | `str` | Unique identifier for this config change. |
| `initiated_by` | `str` | Who initiated this change. |
| `reason` | `str` | Human-readable reason. |
| `probe_schedule` | `Optional[ProbeScheduleUpdate]` | New probe schedule (replaces entire schedule). |
| `overhead_budget` | `Optional[OverheadBudgetUpdate]` | New overhead budget. |
| `sampling_rate` | `Optional[SamplingRateUpdate]` | New sampling rate for a component. |
| `threshold` | `Optional[ThresholdUpdate]` | New threshold for a component parameter. |

#### ConfigAck

Acknowledgement from a config update recipient.

| Field | Type | Description |
|---|---|---|
| `update_id` | `str` | The update_id being acknowledged. |
| `applied` | `bool` | Whether the update was applied successfully. |
| `component_id` | `str` | Hostname of the agent/component that processed the update. |
| `error` | `str` | Error message if not applied. |
| `config_version` | `int` | Effective configuration version after this update. |

#### ProbeScheduleEntry

A single entry in the probe schedule.

| Field | Type | Description |
|---|---|---|
| `type` | `ProbeType` | Probe type. |
| `period_seconds` | `int` | Execution period in seconds. |
| `sm_coverage` | `float` | Fraction of SMs to cover per period [0.0, 1.0]. |
| `priority` | `int` | Scheduling priority (lower = higher priority). |
| `enabled` | `bool` | Whether this probe type is enabled. |
| `timeout_ms` | `int` | Maximum allowed execution time in milliseconds. |

---

## Sync vs Async Usage Patterns

The SDK provides both synchronous and asynchronous methods for every operation.
Synchronous methods are plain function calls; asynchronous methods have an `a`
prefix and return coroutines.

| Synchronous | Asynchronous |
|---|---|
| `SentinelClient.connect()` | `await SentinelClient.aconnect()` |
| `client.query_gpu_health()` | `await client.aquery_gpu_health()` |
| `client.query_fleet_health()` | `await client.aquery_fleet_health()` |
| `client.get_gpu_history()` | `await client.aget_gpu_history()` |
| `client.issue_quarantine()` | `await client.aissue_quarantine()` |
| `client.query_audit_trail()` | `await client.aquery_audit_trail()` |
| `client.verify_chain()` | `await client.averify_chain()` |
| `client.get_trust_graph()` | `await client.aget_trust_graph()` |
| `client.update_config()` | `await client.aupdate_config()` |
| `client.stream_events()` | `await client.astream_events()` |
| `client.close()` | `await client.aclose()` |

### When to use sync vs async

**Use synchronous** when:
- Writing scripts, CLI tools, or notebooks
- Your application does not use asyncio
- You want the simplest possible code

**Use asynchronous** when:
- Your application already uses asyncio (e.g., FastAPI, aiohttp)
- You need to monitor multiple GPUs concurrently
- You are building a real-time dashboard
- You need to stream events without blocking the event loop

### Mixing sync and async

Do not call synchronous methods from within an async event loop -- they will
block the loop. Use the `a`-prefixed methods instead. If you must call sync
methods from async code, use `asyncio.to_thread()`:

```python
# Acceptable but not recommended:
result = await asyncio.to_thread(client.query_fleet_health)
```

---

## Error Handling

All SDK exceptions inherit from `SentinelError`:

```
SentinelError
  +-- ConnectionError        # Cannot connect to endpoint.
  +-- AuthenticationError    # Authentication/authorization failure.
  +-- NotFoundError          # Requested resource not found.
  +-- InvalidArgumentError   # Invalid argument passed to RPC.
```

Each exception has a `code` attribute containing the gRPC status code
(`grpc.StatusCode`).

### Error mapping

| gRPC Status Code | SDK Exception |
|---|---|
| `NOT_FOUND` | `NotFoundError` |
| `UNAUTHENTICATED` | `AuthenticationError` |
| `PERMISSION_DENIED` | `AuthenticationError` |
| `INVALID_ARGUMENT` | `InvalidArgumentError` |
| `UNAVAILABLE` | `ConnectionError` |
| All others | `SentinelError` |

### Retryable errors

By default, these status codes trigger automatic retries with exponential
backoff:
- `UNAVAILABLE` -- server temporarily unreachable
- `DEADLINE_EXCEEDED` -- RPC timed out
- `RESOURCE_EXHAUSTED` -- rate limited

All other errors raise immediately without retry.

### Example error handling

```python
from sentinel_sdk import SentinelClient, SentinelError, NotFoundError, ConnectionError

try:
    client = SentinelClient.connect("sentinel.example.com:443")
    health = client.query_gpu_health("GPU-nonexistent-uuid")
except NotFoundError as e:
    print(f"GPU not found: {e}")
except ConnectionError as e:
    print(f"Cannot reach SENTINEL: {e}")
except SentinelError as e:
    print(f"SENTINEL error (code={e.code}): {e}")
```

---

## Configuration via Environment Variables

The SDK reads the following environment variables as defaults:

| Variable | Description | Default |
|---|---|---|
| `SENTINEL_ENDPOINT` | `host:port` of the gRPC gateway | (none -- must be provided) |
| `SENTINEL_CA_CERT` | Path to CA certificate (PEM) | (none) |
| `SENTINEL_CLIENT_CERT` | Path to client certificate (PEM) | (none) |
| `SENTINEL_CLIENT_KEY` | Path to client private key (PEM) | (none) |
| `SENTINEL_TIMEOUT` | Default RPC timeout in seconds | `30` |
| `SENTINEL_MAX_RETRIES` | Maximum retry attempts | `3` |
| `SENTINEL_LOG_LEVEL` | SDK log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `WARNING` |

```python
import os
os.environ["SENTINEL_ENDPOINT"] = "sentinel.prod.internal:443"
os.environ["SENTINEL_CA_CERT"] = "/etc/sentinel/ca.pem"
```

---

## Integration Examples

### Monitor a Training Run

```python
"""Check GPU health before and during a training run, with automated
quarantine if failures are detected."""

import time
from datetime import datetime
from sentinel_sdk import SentinelClient, TlsConfig
from sentinel_sdk.types import GpuHealthState, QuarantineAction

GPUS = [
    "GPU-aaaa1111-2222-3333-4444-555566667777",
    "GPU-bbbb1111-2222-3333-4444-555566667777",
    "GPU-cccc1111-2222-3333-4444-555566667777",
    "GPU-dddd1111-2222-3333-4444-555566667777",
]

client = SentinelClient.connect(
    "sentinel.prod.internal:443",
    tls_config=TlsConfig(ca_cert_path="/etc/sentinel/ca.pem"),
)

# Pre-flight check: ensure all GPUs are healthy.
print("Pre-flight health check...")
for gpu_uuid in GPUS:
    health = client.query_gpu_health(gpu_uuid)
    if health.state != GpuHealthState.HEALTHY:
        print(f"  ABORT: {gpu_uuid} is {health.state.name}")
        exit(1)
    print(f"  {gpu_uuid}: OK (reliability={health.reliability_score:.6f})")

print("All GPUs healthy. Starting training...")
# launch_training(GPUS)  # Your training code here.

# Periodic monitoring during training.
for step in range(1000):
    # ... your training step ...
    if step % 100 == 0:
        for gpu_uuid in GPUS:
            health = client.query_gpu_health(gpu_uuid)
            if health.state == GpuHealthState.SUSPECT:
                print(f"WARNING: {gpu_uuid} is SUSPECT at step {step}")
            elif health.state in (GpuHealthState.QUARANTINED, GpuHealthState.CONDEMNED):
                print(f"CRITICAL: {gpu_uuid} is {health.state.name} at step {step}")
                print("Saving checkpoint and stopping training...")
                # save_checkpoint(step)
                break

client.close()
```

### Check GPU Health Before Launching Inference

```python
"""Pre-flight check for an inference service startup."""

from sentinel_sdk import SentinelClient
from sentinel_sdk.types import GpuHealthState

def preflight_check(gpu_uuid: str, min_reliability: float = 0.999) -> bool:
    """Returns True if the GPU is safe to use for inference."""
    client = SentinelClient.connect("sentinel.internal:443")

    try:
        health = client.query_gpu_health(gpu_uuid)

        if health.state != GpuHealthState.HEALTHY:
            print(f"GPU {gpu_uuid} is {health.state.name}, not HEALTHY")
            return False

        if health.reliability_score < min_reliability:
            print(f"GPU {gpu_uuid} reliability {health.reliability_score:.6f} "
                  f"below threshold {min_reliability}")
            return False

        if health.probe_failure_rate > 0:
            print(f"GPU {gpu_uuid} has active probe failures "
                  f"({health.probe_failure_rate:.2f}/hour)")
            return False

        return True
    finally:
        client.close()

# Usage:
if preflight_check("GPU-aaaa1111-2222-3333-4444-555566667777"):
    print("GPU is safe for inference")
    # start_inference_server()
else:
    print("GPU failed pre-flight check, selecting alternate GPU")
```

### Build a Custom Dashboard

```python
"""Async dashboard backend that periodically polls fleet health."""

import asyncio
import json
from sentinel_sdk import SentinelClient, TlsConfig

async def dashboard_loop():
    client = await SentinelClient.aconnect(
        "sentinel.internal:443",
        tls_config=TlsConfig(ca_cert_path="ca.pem"),
    )

    try:
        while True:
            fleet = await client.aquery_fleet_health()
            graph = await client.aget_trust_graph()

            dashboard_data = {
                "timestamp": fleet.snapshot_time.isoformat() if fleet.snapshot_time else None,
                "total_gpus": fleet.total_gpus,
                "healthy": fleet.healthy,
                "suspect": fleet.suspect,
                "quarantined": fleet.quarantined,
                "condemned": fleet.condemned,
                "sdc_rate": fleet.overall_sdc_rate,
                "avg_reliability": fleet.average_reliability_score,
                "active_agents": fleet.active_agents,
                "trust_coverage": graph.coverage_pct,
                "min_trust": graph.min_trust_score,
                "mean_trust": graph.mean_trust_score,
            }

            print(json.dumps(dashboard_data, indent=2))
            # In production: push to WebSocket clients, write to DB, etc.

            await asyncio.sleep(10)
    finally:
        await client.aclose()

asyncio.run(dashboard_loop())
```

### Automated Quarantine Workflow

```python
"""Automated quarantine workflow: listen for suspect GPUs and quarantine them."""

import asyncio
from sentinel_sdk import SentinelClient, TlsConfig
from sentinel_sdk.types import (
    QuarantineDirective, QuarantineAction, GpuHealthState,
)

async def auto_quarantine():
    client = await SentinelClient.aconnect(
        "sentinel.internal:443",
        tls_config=TlsConfig(
            ca_cert_path="/etc/sentinel/ca.pem",
            client_cert_path="/etc/sentinel/client.pem",
            client_key_path="/etc/sentinel/client-key.pem",
        ),
    )

    RELIABILITY_THRESHOLD = 0.990
    CHECK_INTERVAL = 30  # seconds

    try:
        while True:
            fleet = await client.aquery_fleet_health(
                state_filter=[GpuHealthState.SUSPECT],
            )

            if fleet.suspect > 0:
                # Query individual suspect GPUs.
                # (In production, use the fleet response's per-GPU data.)
                print(f"Found {fleet.suspect} suspect GPUs, investigating...")

            await asyncio.sleep(CHECK_INTERVAL)
    finally:
        await client.aclose()

asyncio.run(auto_quarantine())
```

### Audit Trail Export Script

```python
"""Export audit trail to JSONL for compliance archival."""

import json
from datetime import datetime, timedelta
from sentinel_sdk import SentinelClient
from sentinel_sdk.types import AuditQueryFilters

def export_audit_trail(output_path: str, days: int = 30):
    client = SentinelClient.connect("sentinel.internal:443")

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)

    filters = AuditQueryFilters(
        start_time=start_time,
        end_time=end_time,
        limit=500,
        descending=False,
    )

    total_exported = 0

    with open(output_path, "w") as f:
        while True:
            result = client.query_audit_trail(filters)

            for entry in result.entries:
                record = {
                    "entry_id": entry.entry_id,
                    "type": entry.entry_type.name,
                    "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                    "gpu_uuid": entry.gpu.uuid if entry.gpu else None,
                    "entry_hash": entry.entry_hash.hex(),
                    "previous_hash": entry.previous_hash.hex(),
                }
                f.write(json.dumps(record) + "\n")
                total_exported += 1

            if not result.next_page_token:
                break

            filters.page_token = result.next_page_token

    # Verify the chain covers the exported range.
    verification = client.verify_chain()
    print(f"Exported {total_exported} entries to {output_path}")
    print(f"Chain integrity: {'VALID' if verification.valid else 'INVALID'}")
    print(f"  Entries verified: {verification.entries_verified}")
    print(f"  Batches verified: {verification.batches_verified}")

    client.close()

export_audit_trail("audit_export.jsonl", days=90)
```

---

## Troubleshooting

### Connection refused

```
sentinel_sdk.ConnectionError: failed to connect to all addresses
```

**Cause:** The gRPC endpoint is unreachable.

**Solutions:**
1. Verify the endpoint address and port.
2. Check firewall rules allow traffic on the gRPC port (default: 50051 for
   insecure, 443 for TLS).
3. Ensure the SENTINEL correlation engine is running.

### TLS handshake failure

```
sentinel_sdk.ConnectionError: Ssl handshake failed
```

**Cause:** Certificate mismatch or expired certificates.

**Solutions:**
1. Verify the CA certificate matches the server's certificate chain.
2. Ensure certificates are not expired: `openssl x509 -in ca.pem -noout -dates`.
3. For mTLS, verify the client certificate is signed by the server's trusted CA.

### Deadline exceeded

```
sentinel_sdk.SentinelError: Deadline Exceeded
```

**Cause:** The RPC did not complete within the timeout.

**Solutions:**
1. Increase `default_timeout` when connecting.
2. For `verify_chain` on large ledgers, use a longer timeout.
3. Check network latency to the SENTINEL endpoint.

### GPU not found

```
sentinel_sdk.NotFoundError: GPU GPU-xxxx not found
```

**Cause:** The GPU UUID is not known to the SENTINEL system.

**Solutions:**
1. Verify the UUID format: `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.
2. Ensure a probe agent is running on the host with this GPU.
3. The GPU may not yet have been seen if the agent just started.

### Import errors for generated protobuf code

```
ModuleNotFoundError: No module named 'sentinel.v1'
```

**Cause:** The generated protobuf Python code is not installed.

**Solutions:**
1. Ensure you installed `sentinel-sdk` with protobuf stubs:
   `pip install sentinel-sdk[stubs]`.
2. Or generate the stubs from proto files:
   ```bash
   python -m grpc_tools.protoc \
     --proto_path=proto \
     --python_out=sdk/python/src \
     --grpc_python_out=sdk/python/src \
     proto/sentinel/v1/*.proto
   ```

### Logging

Enable SDK debug logging to diagnose issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("sentinel_sdk").setLevel(logging.DEBUG)
```

---

## Version Compatibility

| SDK Version | SENTINEL Server | Python | gRPC | Protobuf |
|---|---|---|---|---|
| 0.1.x (alpha) | 0.1.x | >= 3.10 | >= 1.60.0 | >= 4.25.0 |
| 0.2.x (planned) | 0.2.x | >= 3.10 | >= 1.62.0 | >= 4.25.0 |

The SDK follows the same versioning as the SENTINEL server. Minor version
mismatches are tolerated (e.g., SDK 0.1.3 with server 0.1.5), but major or
minor version mismatches may result in incompatible protobuf schemas.

The gRPC API uses `sentinel.v1` package versioning. Breaking changes will only
occur with a major package version bump (e.g., `sentinel.v2`).
