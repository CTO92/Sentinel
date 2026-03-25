# SENTINEL Go SDK Guide

> **Module:** `github.com/sentinel-sdc/sentinel-sdk-go`
> **Status:** Pre-release Alpha
> **Minimum Go:** 1.21+
> **License:** Apache-2.0

The SENTINEL Go SDK provides a client for querying GPU health, fleet status,
audit trails, trust graphs, and managing quarantine directives from Go
applications. It is designed for use in Kubernetes operators, monitoring daemons,
and infrastructure automation.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Client Creation and Options](#client-creation-and-options)
4. [API Reference](#api-reference)
   - [QueryGpuHealth](#querygpuhealth)
   - [QueryFleetHealth](#queryfleethealth)
   - [GetGpuHistory](#getgpuhistory)
   - [IssueQuarantine](#issuequarantine)
   - [QueryAuditTrail](#queryaudittrail)
   - [VerifyChain](#verifychain)
   - [GetTrustGraph](#gettrustgraph)
   - [StreamEvents](#streamevents)
   - [UpdateConfig](#updateconfig)
   - [Close](#close)
5. [Type Reference](#type-reference)
6. [Context Usage Patterns](#context-usage-patterns)
7. [Error Handling](#error-handling)
8. [Integration Examples](#integration-examples)
9. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
go get github.com/sentinel-sdc/sentinel-sdk-go
```

The module depends on:
- `google.golang.org/grpc`
- `google.golang.org/protobuf`

---

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    sentinel "github.com/sentinel-sdc/sentinel-sdk-go/sentinel"
)

func main() {
    // Create a client with TLS.
    client, err := sentinel.NewClient(
        "sentinel.example.com:443",
        sentinel.WithTLSCACert("/etc/sentinel/ca.pem"),
    )
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    defer client.Close()

    ctx := context.Background()

    // Query a single GPU.
    health, err := client.QueryGpuHealth(ctx, "GPU-abcd1234-5678-9abc-def0-123456789abc")
    if err != nil {
        log.Fatalf("QueryGpuHealth: %v", err)
    }
    fmt.Printf("GPU state: %s, reliability: %.6f\n", health.State, health.ReliabilityScore)

    // Query the fleet.
    fleet, err := client.QueryFleetHealth(ctx)
    if err != nil {
        log.Fatalf("QueryFleetHealth: %v", err)
    }
    fmt.Printf("Fleet: %d/%d healthy, SDC rate: %.6f\n",
        fleet.Healthy, fleet.TotalGPUs, fleet.OverallSDCRate)
}
```

---

## Client Creation and Options

### NewClient

```go
func NewClient(endpoint string, opts ...Option) (*Client, error)
```

Creates a new SENTINEL client connected to the specified endpoint. The client
uses gRPC for all communication.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `endpoint` | `string` | `host:port` of the SENTINEL gRPC gateway. |
| `opts` | `...Option` | Functional options to configure the client. |

**Returns:** `(*Client, error)`

### Options

Options follow the functional options pattern. All options are optional; the
client ships with sensible defaults.

#### WithTLS

```go
func WithTLS(caCertPath, clientCertPath, clientKeyPath string) Option
```

Configure mutual TLS (mTLS) using the provided certificate paths. Both the
client and server authenticate each other.

```go
client, err := sentinel.NewClient("sentinel.example.com:443",
    sentinel.WithTLS(
        "/etc/sentinel/ca.pem",
        "/etc/sentinel/client.pem",
        "/etc/sentinel/client-key.pem",
    ),
)
```

#### WithTLSCACert

```go
func WithTLSCACert(caCertPath string) Option
```

Configure server-side TLS only. The client verifies the server certificate
but the server does not authenticate the client.

```go
client, err := sentinel.NewClient("sentinel.example.com:443",
    sentinel.WithTLSCACert("/etc/sentinel/ca.pem"),
)
```

#### WithInsecureSkipVerify

```go
func WithInsecureSkipVerify() Option
```

Disable TLS certificate verification. For testing only -- never use in production.

```go
client, err := sentinel.NewClient("localhost:50051",
    sentinel.WithInsecureSkipVerify(),
)
```

#### WithTimeout

```go
func WithTimeout(d time.Duration) Option
```

Set the default per-RPC deadline. If a context already has a deadline, that
deadline takes precedence.

| Default | Description |
|---|---|
| `30 * time.Second` | Default per-RPC timeout. |

```go
client, err := sentinel.NewClient("sentinel.example.com:443",
    sentinel.WithTLSCACert("ca.pem"),
    sentinel.WithTimeout(15 * time.Second),
)
```

#### WithRetry

```go
func WithRetry(maxRetries int, initialBackoff, maxBackoff time.Duration, multiplier float64) Option
```

Configure retry behaviour with exponential backoff and jitter.

| Parameter | Default | Description |
|---|---|---|
| `maxRetries` | `3` | Maximum retry attempts. |
| `initialBackoff` | `100ms` | Initial backoff duration. |
| `maxBackoff` | `10s` | Maximum backoff cap. |
| `multiplier` | `2.0` | Exponential backoff multiplier. |

By default, these gRPC status codes trigger retries:
- `codes.Unavailable`
- `codes.DeadlineExceeded`
- `codes.ResourceExhausted`

```go
client, err := sentinel.NewClient("sentinel.example.com:443",
    sentinel.WithTLSCACert("ca.pem"),
    sentinel.WithRetry(5, 200*time.Millisecond, 30*time.Second, 2.0),
)
```

#### WithMaxRecvMsgSize

```go
func WithMaxRecvMsgSize(size int) Option
```

Set the maximum inbound message size in bytes.

| Default | Description |
|---|---|
| `64 * 1024 * 1024` (64 MiB) | Maximum inbound gRPC message size. |

```go
client, err := sentinel.NewClient("sentinel.example.com:443",
    sentinel.WithMaxRecvMsgSize(128 * 1024 * 1024), // 128 MiB
)
```

#### WithKeepalive

```go
func WithKeepalive(interval, timeout time.Duration) Option
```

Configure gRPC keepalive parameters.

| Parameter | Default | Description |
|---|---|---|
| `interval` | `30s` | How often to send keepalive pings. |
| `timeout` | `10s` | How long to wait for a ping response before considering the connection dead. |

```go
client, err := sentinel.NewClient("sentinel.example.com:443",
    sentinel.WithKeepalive(20*time.Second, 5*time.Second),
)
```

#### WithDialOptions

```go
func WithDialOptions(opts ...grpc.DialOption) Option
```

Append additional gRPC dial options for advanced use cases.

```go
client, err := sentinel.NewClient("sentinel.example.com:443",
    sentinel.WithDialOptions(
        grpc.WithUnaryInterceptor(myInterceptor),
    ),
)
```

### Default Configuration

| Setting | Default |
|---|---|
| Default timeout | 30 seconds |
| Max retries | 3 |
| Initial backoff | 100 milliseconds |
| Max backoff | 10 seconds |
| Backoff multiplier | 2.0 |
| Max receive message size | 64 MiB |
| Keepalive interval | 30 seconds |
| Keepalive timeout | 10 seconds |
| Retryable codes | `Unavailable`, `DeadlineExceeded`, `ResourceExhausted` |

---

## API Reference

### QueryGpuHealth

```go
func (c *Client) QueryGpuHealth(ctx context.Context, gpuUUID string) (*GpuHealth, error)
```

Query the health of a specific GPU by its UUID. Returns the GPU's current
lifecycle state, Bayesian reliability score, probe statistics, anomaly counts,
and per-SM health breakdown.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `ctx` | `context.Context` | Request context (for deadlines and cancellation). |
| `gpuUUID` | `string` | NVIDIA GPU UUID. |

**Returns:** `(*GpuHealth, error)`

**Example:**

```go
health, err := client.QueryGpuHealth(ctx, "GPU-abcd1234-5678-9abc-def0-123456789abc")
if err != nil {
    log.Fatalf("QueryGpuHealth: %v", err)
}

fmt.Printf("State: %s\n", health.State)
fmt.Printf("Reliability: %.6f\n", health.ReliabilityScore)
fmt.Printf("Probes: %d pass / %d fail\n", health.ProbePassCount, health.ProbeFailCount)
fmt.Printf("Anomalies: %d\n", health.AnomalyCount)

for _, sm := range health.SmHealth {
    if sm.ProbeFailCount > 0 {
        fmt.Printf("  SM %d: %d failures, reliability=%.4f\n",
            sm.SM.SmID, sm.ProbeFailCount, sm.ReliabilityScore)
    }
}
```

---

### QueryFleetHealth

```go
func (c *Client) QueryFleetHealth(ctx context.Context) (*FleetHealthSummary, error)
```

Query the fleet-wide health summary.

**Returns:** `(*FleetHealthSummary, error)`

**Example:**

```go
fleet, err := client.QueryFleetHealth(ctx)
if err != nil {
    log.Fatalf("QueryFleetHealth: %v", err)
}

fmt.Printf("Total: %d | Healthy: %d | Suspect: %d | Quarantined: %d\n",
    fleet.TotalGPUs, fleet.Healthy, fleet.Suspect, fleet.Quarantined)
fmt.Printf("SDC rate: %.6f events/GPU-hour\n", fleet.OverallSDCRate)
fmt.Printf("Active agents: %d\n", fleet.ActiveAgents)
```

---

### GetGpuHistory

```go
func (c *Client) GetGpuHistory(ctx context.Context, gpuUUID string, start, end time.Time) (*GpuHistoryResponse, error)
```

Retrieve historical health data for a GPU within a time range, including state
transitions, correlation events, and reliability score time series.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `ctx` | `context.Context` | Request context. |
| `gpuUUID` | `string` | GPU UUID to query. |
| `start` | `time.Time` | Start of the time range. |
| `end` | `time.Time` | End of the time range. |

**Returns:** `(*GpuHistoryResponse, error)`

**Example:**

```go
end := time.Now()
start := end.Add(-7 * 24 * time.Hour)

history, err := client.GetGpuHistory(ctx,
    "GPU-abcd1234-5678-9abc-def0-123456789abc", start, end)
if err != nil {
    log.Fatalf("GetGpuHistory: %v", err)
}

fmt.Printf("State transitions: %d\n", len(history.StateTransitions))
for _, t := range history.StateTransitions {
    fmt.Printf("  %v: %s -> %s (%s)\n",
        t.Timestamp, t.FromState, t.ToState, t.Reason)
}

fmt.Printf("Correlations: %d\n", len(history.Correlations))
fmt.Printf("Reliability samples: %d\n", len(history.ReliabilityHistory))
```

---

### IssueQuarantine

```go
func (c *Client) IssueQuarantine(ctx context.Context, gpuUUID string, action QuarantineAction, reason string) (*DirectiveResponse, error)
```

Issue a quarantine directive to change a GPU's lifecycle state.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `ctx` | `context.Context` | Request context. |
| `gpuUUID` | `string` | GPU UUID to act upon. |
| `action` | `QuarantineAction` | The action to take. |
| `reason` | `string` | Human-readable reason for this action. |

**QuarantineAction values:**

| Constant | Value | Description |
|---|---|---|
| `QuarantineActionQuarantine` | 1 | Remove from production; investigate. |
| `QuarantineActionReinstate` | 2 | Return to production. |
| `QuarantineActionCondemn` | 3 | Permanently mark as unreliable. |
| `QuarantineActionScheduleDeepTest` | 4 | Initiate deep diagnostic testing. |

**Returns:** `(*DirectiveResponse, error)`

**Example:**

```go
resp, err := client.IssueQuarantine(ctx,
    "GPU-abcd1234-5678-9abc-def0-123456789abc",
    sentinel.QuarantineActionQuarantine,
    "Repeated FMA probe failures on SM 42",
)
if err != nil {
    log.Fatalf("IssueQuarantine: %v", err)
}

if resp.Accepted {
    fmt.Printf("Directive %s accepted, GPU now %s\n",
        resp.DirectiveID, resp.ResultingState)
} else {
    fmt.Printf("Directive rejected: %s\n", resp.RejectionReason)
}
```

---

### QueryAuditTrail

```go
func (c *Client) QueryAuditTrail(ctx context.Context, filters *AuditQueryFilters) (*AuditQueryResponse, error)
```

Query the tamper-evident audit trail with filtering and pagination.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `ctx` | `context.Context` | Request context. |
| `filters` | `*AuditQueryFilters` | Query filters. May be `nil` for defaults (limit=100). |

**AuditQueryFilters fields:**

| Field | Type | Description |
|---|---|---|
| `GPU` | `*GpuIdentifier` | Filter by GPU (optional). |
| `StartTime` | `*time.Time` | Start of time range (optional). |
| `EndTime` | `*time.Time` | End of time range (optional). |
| `EntryType` | `AuditEntryType` | Filter by entry type. `0` = all types. |
| `Limit` | `uint32` | Maximum entries to return. |
| `PageToken` | `string` | Pagination token. |
| `Descending` | `bool` | If `true`, return newest entries first. |

**Returns:** `(*AuditQueryResponse, error)`

**Example:**

```go
start := time.Now().Add(-24 * time.Hour)

result, err := client.QueryAuditTrail(ctx, &sentinel.AuditQueryFilters{
    GPU:       &sentinel.GpuIdentifier{UUID: "GPU-abcd1234-5678-9abc-def0-123456789abc"},
    StartTime: &start,
    EntryType: sentinel.AuditEntryTypeQuarantineAction,
    Limit:     50,
    Descending: true,
})
if err != nil {
    log.Fatalf("QueryAuditTrail: %v", err)
}

fmt.Printf("Found %d entries (showing %d)\n", result.TotalCount, len(result.Entries))
for _, entry := range result.Entries {
    fmt.Printf("  [%d] %s at %v\n",
        entry.EntryID, entry.EntryType, entry.Timestamp)
}
```

---

### VerifyChain

```go
func (c *Client) VerifyChain(ctx context.Context, start, end time.Time) (*ChainVerificationResult, error)
```

Verify the integrity of the audit hash chain and Merkle tree.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `ctx` | `context.Context` | Request context. |
| `start` | `time.Time` | Start of verification range (reserved for future use). |
| `end` | `time.Time` | End of verification range (reserved for future use). |

**Returns:** `(*ChainVerificationResult, error)`

**Example:**

```go
result, err := client.VerifyChain(ctx, time.Time{}, time.Time{})
if err != nil {
    log.Fatalf("VerifyChain: %v", err)
}

if result.Valid {
    fmt.Printf("Chain verified: %d entries, %d batches in %dms\n",
        result.EntriesVerified, result.BatchesVerified, result.VerificationTimeMs)
} else {
    fmt.Printf("CHAIN BROKEN at entry %d: %s\n",
        result.FirstInvalidEntryID, result.FailureDescription)
}
```

---

### GetTrustGraph

```go
func (c *Client) GetTrustGraph(ctx context.Context) (*TrustGraphSnapshot, error)
```

Retrieve a point-in-time snapshot of the GPU trust graph. The trust graph
records pairwise comparison history from TMR canary runs.

**Returns:** `(*TrustGraphSnapshot, error)`

**Example:**

```go
graph, err := client.GetTrustGraph(ctx)
if err != nil {
    log.Fatalf("GetTrustGraph: %v", err)
}

fmt.Printf("Trust graph: %d GPUs, %d edges\n", graph.TotalGPUs, len(graph.Edges))
fmt.Printf("Coverage: %.1f%%\n", graph.CoveragePct)
fmt.Printf("Trust scores: min=%.4f, mean=%.4f\n",
    graph.MinTrustScore, graph.MeanTrustScore)

for _, edge := range graph.Edges {
    if edge.TrustScore < 0.95 {
        fmt.Printf("  Low trust: %s <-> %s (score=%.4f)\n",
            edge.GpuA.UUID[:16], edge.GpuB.UUID[:16], edge.TrustScore)
    }
}
```

---

### StreamEvents

```go
func (c *Client) StreamEvents(ctx context.Context, callback DirectiveCallback) error
```

Stream quarantine directives in real-time. Blocks until the context is
cancelled, the server closes the stream, or the callback returns a non-nil
error. Automatically reconnects on transient failures with exponential backoff.

**Types:**

```go
type DirectiveCallback func(*QuarantineDirective) error
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `ctx` | `context.Context` | Controls stream lifetime. Cancel to stop. |
| `callback` | `DirectiveCallback` | Invoked for each directive. Return non-nil error to stop. |

**Example:**

```go
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

err := client.StreamEvents(ctx, func(d *sentinel.QuarantineDirective) error {
    fmt.Printf("[%v] %s -> %s (%s)\n",
        d.Timestamp, d.Action, d.GPU.UUID, d.Reason)

    if d.RequiresApproval {
        fmt.Printf("  Awaiting approval from %s\n", d.InitiatedBy)
    }

    return nil // Return non-nil to stop streaming.
})
if err != nil {
    log.Printf("Stream ended: %v", err)
}
```

---

### UpdateConfig

```go
func (c *Client) UpdateConfig(ctx context.Context, update *ConfigUpdate) (*ConfigAck, error)
```

Push a dynamic configuration update to agents or subsystems.

**Returns:** `(*ConfigAck, error)`

**Example:**

```go
ack, err := client.UpdateConfig(ctx, &sentinel.ConfigUpdate{
    UpdateID:    "cfg-001",
    InitiatedBy: "operator:jane",
    Reason:      "Increase FMA probe frequency",
    ProbeSchedule: &sentinel.ProbeScheduleUpdate{
        Entries: []*sentinel.ProbeScheduleEntry{
            {
                Type:          sentinel.ProbeTypeFMA,
                PeriodSeconds: 30,
                SmCoverage:    1.0,
                Priority:      1,
                Enabled:       true,
                TimeoutMs:     5000,
            },
        },
    },
})
if err != nil {
    log.Fatalf("UpdateConfig: %v", err)
}

if ack.Applied {
    fmt.Printf("Config %s applied by %s (v%d)\n",
        ack.UpdateID, ack.ComponentID, ack.ConfigVersion)
} else {
    fmt.Printf("Config rejected: %s\n", ack.Error)
}
```

---

### Close

```go
func (c *Client) Close() error
```

Release all resources held by the client. Safe to call multiple times.

```go
defer client.Close()
```

---

## Type Reference

All types are defined in the `sentinel` package. They mirror the protobuf
messages in `proto/sentinel/v1/`.

### Enumerations

#### Severity

| Constant | Value | String |
|---|---|---|
| `SeverityUnspecified` | 0 | `"SEVERITY_UNSPECIFIED"` |
| `SeverityInfo` | 1 | `"INFO"` |
| `SeverityWarning` | 2 | `"WARNING"` |
| `SeverityHigh` | 3 | `"HIGH"` |
| `SeverityCritical` | 4 | `"CRITICAL"` |

All enum types implement `fmt.Stringer`.

#### ProbeType

| Constant | Value | Description |
|---|---|---|
| `ProbeTypeUnspecified` | 0 | Not set. |
| `ProbeTypeFMA` | 1 | Fused multiply-add determinism check. |
| `ProbeTypeTensorCore` | 2 | Tensor Core reproducibility check. |
| `ProbeTypeTranscendental` | 3 | Transcendental function accuracy check. |
| `ProbeTypeAES` | 4 | AES combinational logic check. |
| `ProbeTypeMemory` | 5 | GPU memory integrity check. |
| `ProbeTypeRegisterFile` | 6 | Register file integrity check. |
| `ProbeTypeSharedMemory` | 7 | Shared memory integrity check. |

#### GpuHealthState

| Constant | Value | Description |
|---|---|---|
| `GpuHealthStateUnspecified` | 0 | Not set. |
| `GpuHealthStateHealthy` | 1 | Operating normally. |
| `GpuHealthStateSuspect` | 2 | Under increased monitoring. |
| `GpuHealthStateQuarantined` | 3 | Removed from production. |
| `GpuHealthStateDeepTest` | 4 | Undergoing deep diagnostics. |
| `GpuHealthStateCondemned` | 5 | Permanently marked unreliable. |

#### QuarantineAction

| Constant | Value | Description |
|---|---|---|
| `QuarantineActionUnspecified` | 0 | Not set. |
| `QuarantineActionQuarantine` | 1 | Remove from production. |
| `QuarantineActionReinstate` | 2 | Return to production. |
| `QuarantineActionCondemn` | 3 | Mark as permanently unreliable. |
| `QuarantineActionScheduleDeepTest` | 4 | Initiate deep diagnostics. |

#### AuditEntryType

| Constant | Value | Description |
|---|---|---|
| `AuditEntryTypeUnspecified` | 0 | Not set. |
| `AuditEntryTypeProbeResult` | 1 | Probe execution result. |
| `AuditEntryTypeAnomalyEvent` | 2 | Anomaly detection event. |
| `AuditEntryTypeQuarantineAction` | 3 | Quarantine lifecycle action. |
| `AuditEntryTypeConfigChange` | 4 | Configuration change. |
| `AuditEntryTypeTMRResult` | 5 | TMR canary result. |
| `AuditEntryTypeSystemEvent` | 6 | System-level event. |

#### AnomalyType, AnomalySource, PatternType

See the Python SDK guide for the full enum value tables; the Go constants
follow the same naming pattern: `AnomalyTypeLogitDrift`,
`AnomalySourceInferenceMonitor`, `PatternTypeMultiSignal`, etc.

### Structs

#### GpuIdentifier

```go
type GpuIdentifier struct {
    UUID            string `json:"uuid"`
    Hostname        string `json:"hostname"`
    DeviceIndex     uint32 `json:"device_index"`
    Model           string `json:"model"`
    DriverVersion   string `json:"driver_version"`
    FirmwareVersion string `json:"firmware_version"`
}
```

| Field | Description |
|---|---|
| `UUID` | NVIDIA UUID (e.g., `"GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"`). |
| `Hostname` | Hostname of the machine containing this GPU. |
| `DeviceIndex` | PCI device index on the host (0-based). |
| `Model` | GPU model name (e.g., `"NVIDIA H100 80GB HBM3"`). |
| `DriverVersion` | Driver version string. |
| `FirmwareVersion` | GPU firmware/VBIOS version. |

#### GpuHealth

```go
type GpuHealth struct {
    GPU               *GpuIdentifier
    State             GpuHealthState
    ReliabilityScore  float64
    Alpha             float64
    Beta              float64
    LastProbeTime     *time.Time
    LastAnomalyTime   *time.Time
    ProbePassCount    uint64
    ProbeFailCount    uint64
    AnomalyCount      uint64
    StateChangedAt    *time.Time
    StateChangeReason string
    SmHealth          []*SmHealth
    AnomalyRate       float64
    ProbeFailureRate  float64
}
```

| Field | Description |
|---|---|
| `GPU` | The GPU this health record describes. |
| `State` | Current lifecycle state. |
| `ReliabilityScore` | Bayesian reliability score `alpha / (alpha + beta)` in [0.0, 1.0]. |
| `Alpha` | Beta distribution alpha (successes + prior). |
| `Beta` | Beta distribution beta (failures + prior). |
| `LastProbeTime` | Most recent probe execution timestamp. |
| `LastAnomalyTime` | Most recent attributed anomaly timestamp. |
| `ProbePassCount` | Lifetime passed probe count. |
| `ProbeFailCount` | Lifetime failed probe count. |
| `AnomalyCount` | Lifetime anomaly count. |
| `StateChangedAt` | When the GPU last transitioned to its current state. |
| `StateChangeReason` | Reason for the most recent state change. |
| `SmHealth` | Per-SM health breakdown. |
| `AnomalyRate` | Current anomaly rate (per hour, rolling window). |
| `ProbeFailureRate` | Current probe failure rate (per hour, rolling window). |

#### FleetHealthSummary

```go
type FleetHealthSummary struct {
    TotalGPUs               uint32
    Healthy                 uint32
    Suspect                 uint32
    Quarantined             uint32
    DeepTest                uint32
    Condemned               uint32
    OverallSDCRate          float64
    AverageReliabilityScore float64
    SnapshotTime            *time.Time
    ActiveAgents            uint32
    RateWindowSeconds       uint32
}
```

| Field | Description |
|---|---|
| `TotalGPUs` | Total GPUs tracked by the system. |
| `Healthy` | GPUs in HEALTHY state. |
| `Suspect` | GPUs in SUSPECT state. |
| `Quarantined` | GPUs in QUARANTINED state. |
| `DeepTest` | GPUs in DEEP_TEST state. |
| `Condemned` | GPUs in CONDEMNED state. |
| `OverallSDCRate` | Fleet-wide SDC rate (events per GPU-hour). |
| `AverageReliabilityScore` | Fleet-wide average reliability score. |
| `SnapshotTime` | Timestamp of this snapshot. |
| `ActiveAgents` | Number of active probe agents. |
| `RateWindowSeconds` | Time window for rate computation (seconds). |

#### GpuHistoryResponse

```go
type GpuHistoryResponse struct {
    StateTransitions   []*StateTransition
    Correlations       []*CorrelationEvent
    ReliabilityHistory []*ReliabilitySample
    NextPageToken      string
}
```

#### DirectiveResponse

```go
type DirectiveResponse struct {
    DirectiveID     string
    Accepted        bool
    RejectionReason string
    ResultingState  string
}
```

#### QuarantineDirective

```go
type QuarantineDirective struct {
    DirectiveID      string
    GPU              *GpuIdentifier
    Action           QuarantineAction
    Reason           string
    InitiatedBy      string
    Evidence         []string
    Timestamp        *time.Time
    Priority         uint32
    RequiresApproval bool
    Approval         *ApprovalStatus
}
```

#### TrustGraphSnapshot

```go
type TrustGraphSnapshot struct {
    Edges          []*TrustEdge
    Timestamp      *time.Time
    CoveragePct    float64
    TotalGPUs      uint32
    MinTrustScore  float64
    MeanTrustScore float64
}
```

#### ChainVerificationResult

```go
type ChainVerificationResult struct {
    Valid               bool
    FirstInvalidEntryID uint64
    FailureDescription  string
    EntriesVerified     uint64
    BatchesVerified     uint64
    VerificationTimeMs  uint64
}
```

#### ConfigUpdate, ConfigAck, AuditEntry, AuditQueryResponse, and others

All remaining types follow the same structure as their Python SDK counterparts.
See `sdk/go/sentinel/types.go` for the complete definitions.

---

## Context Usage Patterns

All Go SDK methods accept a `context.Context` as their first parameter. This is
idiomatic Go and gives you control over deadlines, cancellation, and metadata.

### Deadlines

If the context has a deadline, it takes precedence over the client's default
timeout:

```go
// This RPC will timeout in 5 seconds regardless of the client default.
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

health, err := client.QueryGpuHealth(ctx, gpuUUID)
```

If the context does not have a deadline, the client's default timeout
(configurable via `WithTimeout`) is applied automatically.

### Cancellation

Cancel a long-running operation:

```go
ctx, cancel := context.WithCancel(context.Background())

go func() {
    // Cancel after receiving a signal, user input, etc.
    <-sigChan
    cancel()
}()

// StreamEvents blocks until ctx is cancelled.
err := client.StreamEvents(ctx, callback)
```

### Metadata propagation

Pass gRPC metadata via context:

```go
import "google.golang.org/grpc/metadata"

md := metadata.New(map[string]string{
    "x-request-id": "req-12345",
})
ctx := metadata.NewOutgoingContext(context.Background(), md)

health, err := client.QueryGpuHealth(ctx, gpuUUID)
```

---

## Error Handling

All errors returned by the SDK are either `*SentinelError` or standard Go
errors (e.g., `context.Canceled`, `context.DeadlineExceeded`).

### SentinelError

```go
type SentinelError struct {
    Message string
    Code    codes.Code
}

func (e *SentinelError) Error() string
```

The `Code` field contains the gRPC status code. Use `errors.As` to inspect:

```go
health, err := client.QueryGpuHealth(ctx, gpuUUID)
if err != nil {
    var sentErr *sentinel.SentinelError
    if errors.As(err, &sentErr) {
        switch sentErr.Code {
        case codes.NotFound:
            log.Printf("GPU not found: %s", gpuUUID)
        case codes.Unauthenticated:
            log.Fatal("Authentication failed")
        case codes.PermissionDenied:
            log.Fatal("Permission denied")
        default:
            log.Printf("SENTINEL error (code=%s): %s", sentErr.Code, sentErr.Message)
        }
    } else {
        log.Printf("Unexpected error: %v", err)
    }
}
```

### Retryable errors

These gRPC codes trigger automatic retries (configurable via `WithRetry`):

| Code | Description |
|---|---|
| `codes.Unavailable` | Server temporarily unreachable. |
| `codes.DeadlineExceeded` | RPC timed out. |
| `codes.ResourceExhausted` | Rate limited. |

Non-retryable errors raise immediately without retry.

---

## Integration Examples

### Health Check in a Kubernetes Operator

```go
package controllers

import (
    "context"
    "fmt"

    sentinel "github.com/sentinel-sdc/sentinel-sdk-go/sentinel"
    "sigs.k8s.io/controller-runtime/pkg/log"
)

// GPUNodeReconciler manages GPU node lifecycle in a Kubernetes cluster.
type GPUNodeReconciler struct {
    sentinelClient *sentinel.Client
}

func NewGPUNodeReconciler(sentinelEndpoint string) (*GPUNodeReconciler, error) {
    client, err := sentinel.NewClient(sentinelEndpoint,
        sentinel.WithTLSCACert("/etc/sentinel/ca.pem"),
        sentinel.WithTimeout(10*time.Second),
    )
    if err != nil {
        return nil, fmt.Errorf("sentinel client: %w", err)
    }
    return &GPUNodeReconciler{sentinelClient: client}, nil
}

func (r *GPUNodeReconciler) IsGPUHealthy(ctx context.Context, gpuUUID string) (bool, error) {
    logger := log.FromContext(ctx)

    health, err := r.sentinelClient.QueryGpuHealth(ctx, gpuUUID)
    if err != nil {
        return false, fmt.Errorf("query GPU health: %w", err)
    }

    logger.Info("GPU health check",
        "gpu", gpuUUID,
        "state", health.State.String(),
        "reliability", health.ReliabilityScore,
    )

    switch health.State {
    case sentinel.GpuHealthStateHealthy:
        return true, nil
    case sentinel.GpuHealthStateSuspect:
        // Suspect GPUs can still serve, but log a warning.
        logger.Info("GPU is suspect, allowing with warning", "gpu", gpuUUID)
        return health.ReliabilityScore > 0.99, nil
    default:
        return false, nil
    }
}
```

### Fleet Monitoring Daemon

```go
package main

import (
    "context"
    "encoding/json"
    "log"
    "os"
    "os/signal"
    "time"

    sentinel "github.com/sentinel-sdc/sentinel-sdk-go/sentinel"
)

func main() {
    client, err := sentinel.NewClient("sentinel.internal:443",
        sentinel.WithTLSCACert("/etc/sentinel/ca.pem"),
        sentinel.WithRetry(5, 200*time.Millisecond, 30*time.Second, 2.0),
    )
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    defer client.Close()

    ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
    defer cancel()

    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()

    log.Println("Fleet monitoring daemon started")

    for {
        select {
        case <-ctx.Done():
            log.Println("Shutting down")
            return
        case <-ticker.C:
            fleet, err := client.QueryFleetHealth(ctx)
            if err != nil {
                log.Printf("Fleet health query failed: %v", err)
                continue
            }

            data, _ := json.Marshal(map[string]interface{}{
                "timestamp":   time.Now().UTC().Format(time.RFC3339),
                "total_gpus":  fleet.TotalGPUs,
                "healthy":     fleet.Healthy,
                "suspect":     fleet.Suspect,
                "quarantined": fleet.Quarantined,
                "condemned":   fleet.Condemned,
                "sdc_rate":    fleet.OverallSDCRate,
                "agents":      fleet.ActiveAgents,
            })
            log.Printf("Fleet status: %s", data)

            if fleet.Quarantined > 0 || fleet.Condemned > 0 {
                log.Printf("ALERT: %d quarantined, %d condemned GPUs",
                    fleet.Quarantined, fleet.Condemned)
            }
        }
    }
}
```

### Custom Alerting Integration

```go
package main

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "os"
    "os/signal"

    sentinel "github.com/sentinel-sdc/sentinel-sdk-go/sentinel"
)

func main() {
    client, err := sentinel.NewClient("sentinel.internal:443",
        sentinel.WithTLSCACert("/etc/sentinel/ca.pem"),
    )
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    defer client.Close()

    ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
    defer cancel()

    webhookURL := os.Getenv("ALERT_WEBHOOK_URL")

    log.Println("Streaming quarantine events...")
    err = client.StreamEvents(ctx, func(d *sentinel.QuarantineDirective) error {
        log.Printf("Directive: %s %s -> %s",
            d.Action, d.GPU.UUID, d.Reason)

        // Send webhook alert.
        payload, _ := json.Marshal(map[string]interface{}{
            "event":     "quarantine_directive",
            "gpu_uuid":  d.GPU.UUID,
            "hostname":  d.GPU.Hostname,
            "action":    d.Action.String(),
            "reason":    d.Reason,
            "initiated": d.InitiatedBy,
            "priority":  d.Priority,
        })

        resp, err := http.Post(webhookURL, "application/json",
            bytes.NewReader(payload))
        if err != nil {
            log.Printf("Webhook failed: %v", err)
            return nil // Continue streaming despite webhook failure.
        }
        resp.Body.Close()

        if resp.StatusCode >= 400 {
            log.Printf("Webhook returned %d", resp.StatusCode)
        }

        return nil
    })

    if err != nil {
        log.Printf("Stream ended: %v", err)
    }
}
```

---

## Troubleshooting

### Connection refused

```
sentinel: dial failed: ...connection refused
```

Verify the endpoint address and port. Ensure the SENTINEL correlation engine is
running and firewall rules allow traffic.

### TLS handshake failure

```
sentinel: TLS setup failed: ...
```

1. Verify the CA certificate matches the server's chain.
2. Check certificate expiration: `openssl x509 -in ca.pem -noout -dates`.
3. For mTLS, ensure the client cert is signed by the server's trusted CA.

### Context deadline exceeded

```
sentinel: Deadline Exceeded (code=DeadlineExceeded)
```

The RPC did not complete in time. Increase the timeout via `WithTimeout` or by
setting a longer deadline on the context.

### Client is closed

```
sentinel: client is closed (code=FailedPrecondition)
```

You called a method on a client after `Close()`. Create a new client or
restructure your code to avoid using a closed client.

### Large responses truncated

If fleet health or audit queries return truncated results, increase the max
receive message size:

```go
client, err := sentinel.NewClient(endpoint,
    sentinel.WithMaxRecvMsgSize(128 * 1024 * 1024),
)
```
