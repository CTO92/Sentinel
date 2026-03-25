# SENTINEL SOC 2 Control Mapping

> **Version:** 0.1.0-alpha | **Last Updated:** 2026-03-24

This document maps SENTINEL's capabilities to the SOC 2 Trust Services Criteria (TSC). It is intended for compliance auditors and security teams assessing how SENTINEL contributes to an organization's SOC 2 compliance posture.

SENTINEL is not a general-purpose SOC 2 compliance tool. It specifically addresses controls related to hardware integrity monitoring, anomaly detection, incident management, and audit trail integrity for GPU compute infrastructure.

## Table of Contents

1. [CC6.1 -- Logical Access Security](#cc61--logical-access-security)
2. [CC7.2 -- System Monitoring](#cc72--system-monitoring)
3. [CC8.1 -- Change Management](#cc81--change-management)
4. [A1.2 -- System Availability](#a12--system-availability)
5. [Additional Relevant Criteria](#additional-relevant-criteria)
6. [Evidence Generation](#evidence-generation)

---

## CC6.1 -- Logical Access Security

**Trust Services Criteria:** *The entity implements logical access security software, infrastructure, and architectures over protected information assets to protect them from security events to meet the entity's objectives.*

### What SENTINEL Provides

SENTINEL enforces layered access controls across all system components:

#### Mutual TLS (mTLS) Authentication

All inter-component communication uses mTLS:
- Probe agents authenticate to the correlation engine using client certificates.
- The correlation engine authenticates to the audit ledger using client certificates.
- Certificates are issued by a dedicated SENTINEL CA (or integrated with the organization's PKI via cert-manager).
- TLS 1.3 is required; earlier versions are rejected.

**Evidence source:** TLS certificate inventory and rotation logs in the audit trail. Certificate expiry warnings are generated automatically.

#### HMAC-Signed Probe Results

Every probe result is signed with HMAC-SHA256 before transmission:
- The correlation engine rejects any unsigned or incorrectly signed result.
- This prevents injection of false probe results that could mask a compromised GPU or trigger spurious quarantines.

**Evidence source:** HMAC verification success/failure counters in Prometheus metrics. Rejected messages are logged in the audit trail.

#### Role-Based Access Control (RBAC)

The SENTINEL dashboard and SDK enforce four predefined roles:

| Role | Create/Modify Config | Quarantine/Reinstate | View Fleet Health | View Audit Trail | Generate Reports |
|------|---------------------|---------------------|-------------------|-----------------|-----------------|
| `viewer` | No | No | Yes | No | No |
| `operator` | No | Yes | Yes | Yes | No |
| `admin` | Yes | Yes | Yes | Yes | Yes |
| `auditor` | No | No | No | Yes | Yes |

**Evidence source:** RBAC configuration is stored in the audit trail. Access attempts (granted and denied) are logged with user identity, action, and timestamp.

#### Network Segmentation

Kubernetes network policies restrict inter-component communication to the minimum necessary. Probe agents cannot reach the audit ledger directly. The dashboard cannot reach data stores directly. Only the correlation engine can reach alerting external endpoints.

**Evidence source:** Network policy definitions in `deploy/kubernetes/network-policies.yaml`, viewable in the audit trail as configuration artifacts.

### Audit Report Content for CC6.1

The SENTINEL SOC 2 report generator produces the following CC6.1 evidence:

- List of all active TLS certificates, their issuers, validity periods, and last rotation dates.
- Count of mTLS authentication failures (should be zero under normal operation).
- Count of HMAC verification failures (should be zero under normal operation).
- RBAC role assignments and all changes during the reporting period.
- Access log summary: number of API calls per role, any denied access attempts.
- Network policy configuration in effect during the reporting period.

---

## CC7.2 -- System Monitoring

**Trust Services Criteria:** *The entity monitors system components and the operation of those components for anomalies that are indicative of malicious acts, natural disasters, and errors affecting the entity's ability to meet its objectives; anomalies are analyzed to determine whether they represent security events.*

### What SENTINEL Provides

SENTINEL's primary purpose is anomaly detection and analysis. It provides comprehensive, continuous, automated monitoring of GPU compute infrastructure.

#### Continuous Hardware Monitoring

- **Probe agent:** Runs 7 types of deterministic probes (FMA, Tensor Core, Transcendental, AES, Memory, Register File, Shared Memory) on every GPU, testing every SM (Streaming Multiprocessor) at configurable intervals (default 60-600 seconds).
- **Telemetry collection:** Continuous monitoring of GPU temperature, voltage, power draw, ECC error counts, NVLink/PCIe error counts, and retired page counts.
- **Coverage:** 24/7 monitoring of every GPU in the fleet. No GPU operates without SENTINEL oversight.

**Evidence source:** Probe result records in ScyllaDB and the audit ledger. Metrics in Prometheus with configurable retention (default 30 days for raw metrics, indefinite for the audit trail).

#### Anomaly Detection and Analysis

- **Statistical output monitoring:** EWMA control charts, KL divergence, entropy monitoring, and KS tests continuously analyze inference and training outputs for anomalous patterns.
- **Cross-rank divergence detection:** During distributed training, gradient norms are compared across ranks to detect individual GPUs computing differently from peers.
- **Correlation engine:** Fuses signals from multiple detection layers and applies Bayesian analysis to determine whether anomalies indicate genuine hardware faults.

**Evidence source:** Anomaly event records in the audit trail, including the anomaly type, severity, contributing evidence, and the correlation engine's attribution decision.

#### Automated Response

- **Quarantine state machine:** Automatically removes GPUs from production workloads when the Bayesian reliability score drops below configurable thresholds.
- **Alert dispatch:** Anomaly and quarantine events are routed to configured notification channels (Slack, PagerDuty, email) with severity-based routing.
- **Deep test scheduling:** Quarantined GPUs can be automatically scheduled for comprehensive diagnostic testing.

**Evidence source:** Quarantine state transitions with timestamps, triggering evidence, and associated alerts -- all recorded in the audit trail.

#### Monitoring Timeliness

| Detection Type | Typical Detection Latency |
|---------------|--------------------------|
| Probe failure (consistent fault) | < 60 seconds |
| Probe failure (intermittent fault) | < 24 hours (depends on frequency) |
| Inference output anomaly | < 30 minutes (after EWMA warmup) |
| Training gradient divergence | < 10 training steps |
| TMR dissent | < TMR interval (default 10 minutes) |

### Audit Report Content for CC7.2

- Summary of monitoring coverage: number of GPUs monitored, probe execution count, sampling rates.
- Anomaly event summary: total events by type and severity, response times, false positive rate.
- Quarantine summary: GPUs quarantined during reporting period, time-to-quarantine, time-to-resolution.
- Alert summary: total alerts dispatched by channel and severity, acknowledgment times.
- Probe coverage gaps (if any): periods where monitoring was interrupted and the cause.

---

## CC8.1 -- Change Management

**Trust Services Criteria:** *The entity authorizes, designs, develops, configures, documents, tests, approves, and implements changes to infrastructure and software to meet its objectives.*

### What SENTINEL Provides

SENTINEL tracks all configuration changes to the detection system itself.

#### Configuration Change Audit

Every change to SENTINEL's configuration is recorded in the audit trail:

- Probe schedule changes (which probes are enabled, their periods, SM coverage).
- Threshold changes (Bayesian priors, quarantine thresholds, EWMA parameters).
- Alert rule changes (rules added, modified, or disabled).
- Role assignment changes (RBAC modifications).
- TLS certificate rotations.
- Dynamic configuration updates pushed via gRPC.

Each audit entry includes:
- Timestamp of the change.
- Identity of the operator who made the change (from mTLS client certificate or RBAC session).
- The previous configuration value.
- The new configuration value.
- The method of change (dashboard, SDK, gRPC, config file reload).

**Evidence source:** Configuration change records in the audit trail, queryable by time range, operator, and configuration section.

#### Approval Gates

When `require_approval` is enabled in the correlation engine configuration, critical state changes (quarantine, reinstatement, condemn) require approval from a second operator before executing. This provides a change management control for high-impact decisions.

**Evidence source:** Approval request records in the audit trail, including requestor, approver, and timestamps.

### Audit Report Content for CC8.1

- List of all configuration changes during the reporting period, with operator identity and justification.
- Approval workflow records (if enabled): requests, approvals/denials, and time-to-approve.
- Version history: SENTINEL component versions deployed during the reporting period and upgrade events.

---

## A1.2 -- System Availability

**Trust Services Criteria:** *The entity authorizes, designs, develops or acquires, implements, operates, approves, maintains, and monitors environmental protections, software, data backup and recovery infrastructure, and recovery plan procedures to meet its objectives.*

### What SENTINEL Provides

SENTINEL directly contributes to compute infrastructure availability by identifying and removing unreliable GPUs before they cause production failures.

#### Proactive Fault Identification

- GPUs are identified as SUSPECT before they cause visible production issues. This enables proactive replacement during maintenance windows rather than reactive replacement during incidents.
- The quarantine state machine automatically removes faulty GPUs from production scheduling, preventing them from corrupting training runs or inference outputs.

**Evidence source:** GPU lifecycle records in the audit trail: HEALTHY -> SUSPECT -> QUARANTINED -> DEEP_TEST -> CONDEMNED (or reinstated). Time-in-state metrics.

#### Hardware Asset Tracking

SENTINEL maintains a complete inventory of monitored GPU assets:

- GPU UUID, serial number, model, and firmware version.
- Host node, PCIe slot, and NVLink topology.
- Reliability score history.
- State history with transitions and reasons.
- Replacement records (when a CONDEMNED GPU is replaced).

**Evidence source:** GPU inventory data in PostgreSQL, queryable via the SDK and dashboard. Full lifecycle history in the audit trail.

#### Availability Metrics

SENTINEL tracks GPU availability metrics:

| Metric | Description |
|--------|-------------|
| GPU uptime | Percentage of time each GPU was in HEALTHY state and serving production workloads. |
| Mean time to quarantine | Average time from first SUSPECT signal to QUARANTINED state. |
| Mean time to resolution | Average time from QUARANTINED to either reinstated or replaced. |
| Fleet availability | Percentage of fleet capacity available at any point in time (total GPUs minus quarantined/condemned GPUs). |

**Evidence source:** Time-series metrics in Prometheus, long-term aggregates in the compliance report.

### Audit Report Content for A1.2

- Fleet availability over the reporting period (percentage, hourly granularity).
- GPU lifecycle events: how many GPUs were quarantined, condemned, reinstated, or replaced.
- Mean time to detection, quarantine, and resolution.
- Proactive vs. reactive fault identification rate (faults caught by SENTINEL before production impact vs. faults that caused production impact before SENTINEL flagged them).

---

## Additional Relevant Criteria

### CC3.4 -- Risk Assessment

SENTINEL provides data for ongoing risk assessment of GPU compute infrastructure:
- Per-GPU and fleet-wide reliability scores.
- Historical SDC detection rates by GPU model, age, and operating conditions.
- Trend analysis: is the fleet's SDC rate increasing over time (aging)?

### CC7.3 -- Evaluating Security Events

The correlation engine evaluates anomaly events using Bayesian attribution:
- Each event is assessed for its likelihood of representing genuine SDC vs. normal operational variation.
- Multiple events are correlated temporally and spatially to increase attribution confidence.
- Fleet-wide patterns are analyzed to distinguish hardware faults from software bugs.

### CC7.4 -- Responding to Security Events

SENTINEL implements automated and manual response procedures:
- Automated: quarantine state machine removes faulty GPUs from production.
- Automated: alert dispatch notifies operators via configured channels.
- Manual: operators can quarantine/reinstate GPUs, schedule deep tests, and generate replacement reports.
- All response actions are recorded in the tamper-evident audit trail.

---

## Evidence Generation

### Automated Report Generation

SOC 2 compliance reports can be generated on demand or on a schedule:

```python
from sentinel_sdk import SentinelClient

client = SentinelClient("audit-ledger:50052")

# Generate SOC 2 report for Q1 2026
report = client.generate_soc2_report(
    start_date="2026-01-01",
    end_date="2026-03-31",
    controls=["CC6.1", "CC7.2", "CC8.1", "A1.2"],
    output_format="pdf",
    include_raw_evidence=False  # Set True for detailed evidence export
)

report.save("sentinel_soc2_q1_2026.pdf")
```

### Evidence Integrity

All evidence is sourced from SENTINEL's tamper-evident audit trail:
- Every audit entry is cryptographically chained (SHA-256 hash chain).
- Batches of entries are committed with Merkle tree roots.
- Automatic integrity verification runs every 6 hours.
- Chain integrity status is included in the compliance report header.

An auditor can independently verify the integrity of the evidence chain:

```bash
sentinel-audit-ledger verify --full --attest \
  --output attestation.json \
  --config /etc/sentinel/config/sentinel.yaml
```

The attestation file contains:
- The chain root hash.
- The number of entries verified.
- The time range covered.
- The verification timestamp.
- The signature of the verifying instance.
