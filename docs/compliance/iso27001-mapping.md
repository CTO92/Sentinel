# SENTINEL ISO 27001 Control Mapping

> **Version:** 0.1.0-alpha | **Last Updated:** 2026-03-24

This document maps SENTINEL's capabilities to the ISO/IEC 27001:2022 Annex A controls. It is intended for information security teams and auditors assessing how SENTINEL supports an organization's Information Security Management System (ISMS) for GPU compute infrastructure.

SENTINEL contributes to a specific subset of ISO 27001 controls related to hardware asset management, operations security, and incident management. It does not address the full scope of ISO 27001 (e.g., human resources security, physical security, supplier management).

## Table of Contents

1. [A.8 -- Asset Management](#a8--asset-management)
2. [A.12 -- Operations Security](#a12--operations-security)
3. [A.16 -- Information Security Incident Management](#a16--information-security-incident-management)
4. [Additional Relevant Controls](#additional-relevant-controls)
5. [Evidence Generation](#evidence-generation)

---

## A.8 -- Asset Management

### A.8.1 -- Responsibility for Assets

**Control objective:** Assets associated with information and information processing facilities shall be identified and an inventory of these assets shall be drawn up and maintained.

#### A.8.1.1 -- Inventory of Assets

**Control description:** Information, other assets associated with information and information processing facilities shall be identified and an inventory of these assets shall be drawn up and maintained.

**SENTINEL implementation:**

SENTINEL automatically discovers and inventories all GPU assets in the monitored fleet:

- **Auto-discovery:** On startup, each probe agent queries NVML to discover all NVIDIA GPUs on the node. Each GPU is identified by its UUID (globally unique, immutable hardware identifier).
- **Asset attributes collected:**

  | Attribute | Source | Example |
  |-----------|--------|---------|
  | GPU UUID | NVML | `GPU-a1b2c3d4-e5f6-7890-abcd-ef1234567890` |
  | Serial number | NVML | `1234567890123` |
  | GPU model | NVML | `NVIDIA H100 80GB HBM3` |
  | GPU architecture | NVML | `Hopper` |
  | VBIOS version | NVML | `96.00.89.00.01` |
  | Driver version | NVML | `545.23.08` |
  | CUDA compute capability | CUDA Runtime | `9.0` |
  | Node hostname | OS | `gpu-node-042` |
  | PCIe bus ID | NVML | `0000:3b:00.0` |
  | NVLink topology | NVML | Connected to GPUs 1,2,3 via NVLink 4.0 |
  | Memory capacity | NVML | `81920 MiB` |
  | SM count | NVML | `132` |
  | ECC mode | NVML | `Enabled` |

- **Inventory updates:** The inventory is updated at probe agent startup and periodically (every telemetry collection interval, default 10 seconds) to detect hardware changes (hot-swap events, firmware updates).
- **Inventory storage:** GPU inventory is stored in PostgreSQL and replicated to the audit trail.

**Evidence:** Complete GPU asset inventory exportable via the SDK or dashboard. Historical inventory showing when GPUs were added, removed, or had attributes change.

#### A.8.1.3 -- Acceptable Use of Assets

**Control description:** Rules for the acceptable use of information and of assets associated with information and information processing facilities shall be identified, documented and implemented.

**SENTINEL implementation:**

SENTINEL enforces acceptable use of GPU assets through the quarantine state machine:

- GPUs in HEALTHY state are cleared for production workloads.
- GPUs in SUSPECT state are under increased monitoring but still available (operator may choose to restrict workloads).
- GPUs in QUARANTINED state are removed from production workload scheduling.
- GPUs in CONDEMNED state are permanently flagged for replacement and cannot be used.

The acceptable-use policy is codified in the quarantine threshold configuration and enforced automatically. Operators cannot override the CONDEMNED state.

**Evidence:** GPU state history showing which GPUs were cleared for production use at any point in time. Policy configuration (quarantine thresholds) stored in the audit trail.

### A.8.3 -- Information Transfer

**Control description:** Policies, procedures and controls shall protect the transfer of information through the use of all types of communication facilities.

**SENTINEL implementation:**

All data transfer between SENTINEL components is protected:

- **In transit:** mTLS encryption on all gRPC connections (TLS 1.3 required).
- **Integrity:** HMAC-SHA256 signing of probe results. Hash chain integrity for the audit ledger.
- **Authentication:** Mutual certificate authentication for all inter-component communication.

**Evidence:** TLS configuration, certificate inventory, HMAC verification metrics.

---

## A.12 -- Operations Security

### A.12.1 -- Operational Procedures and Responsibilities

**Control description:** Operating procedures shall be documented and made available to all users who need them.

**SENTINEL implementation:**

SENTINEL provides comprehensive operational documentation:

- [Operator Runbook](../operator-runbook.md): Day-to-day operations, alert triage, quarantine management.
- [Calibration Guide](../calibration-guide.md): Threshold tuning procedures.
- [Deployment Guide](../deployment-guide.md): Deployment and configuration procedures.

All operational actions taken within SENTINEL are recorded in the audit trail with the operator's identity and the action taken.

**Evidence:** Documentation artifacts. Audit trail records of operational actions.

### A.12.4 -- Logging and Monitoring

**Control description:** Event logs recording user activities, exceptions, faults and information security events shall be produced, kept and regularly reviewed.**

#### A.12.4.1 -- Event Logging

**SENTINEL implementation:**

SENTINEL produces comprehensive event logs at multiple levels:

| Log Level | Content | Storage | Retention |
|-----------|---------|---------|-----------|
| **Audit trail** | All probe results, anomaly events, state transitions, operator actions, configuration changes | Audit ledger (ScyllaDB + Merkle hash chain) | Configurable (default 365 days for details, forever for Merkle roots) |
| **Metrics** | Prometheus time-series metrics for all SENTINEL components | Prometheus TSDB | Configurable (default 30 days) |
| **Application logs** | Structured JSON logs from all SENTINEL components | stdout (forwarded to cluster logging) | Per organization's log retention policy |

Event logs include:
- Timestamp (UTC, microsecond precision).
- Source component (probe agent, inference monitor, training monitor, correlation engine, audit ledger).
- Event type and severity.
- GPU UUID and node hostname (where applicable).
- Operator identity (for manual actions).
- Cryptographic hash linking to the previous audit entry (for audit trail entries).

**Evidence:** Audit trail records. Prometheus metrics. Structured application logs.

#### A.12.4.2 -- Protection of Log Information

**SENTINEL implementation:**

SENTINEL's audit trail is tamper-evident:

- **Hash chain:** Each audit entry's hash includes the previous entry's hash: `Entry_N.hash = SHA-256(Entry_N.data || Entry_{N-1}.hash)`. Modifying any historical entry breaks the chain for all subsequent entries.
- **Merkle trees:** Batches of entries are committed with Merkle tree roots, enabling efficient integrity proofs.
- **Automatic verification:** Chain integrity is automatically verified every 6 hours (configurable). Any integrity violation triggers a CRITICAL alert.
- **Access control:** The audit ledger enforces write-through-API-only semantics. Direct database modifications bypass the hash chain and will be detected by the next integrity verification.
- **Separation of duties:** The audit ledger runs as a separate service from the correlation engine. Compromising one service does not grant write access to the other.

**Evidence:** Chain verification reports with root hashes and verification timestamps. Alert records for any integrity violations.

#### A.12.4.3 -- Administrator and Operator Logs

**SENTINEL implementation:**

All administrator and operator actions are recorded in the audit trail:

- Configuration changes (who changed what, when, from what value to what value).
- Manual quarantine and reinstatement decisions (who, which GPU, stated reason).
- Approval workflow records (requestor, approver, decision).
- Report generation events.
- RBAC changes (role assignments, modifications).

**Evidence:** Operator action records in the audit trail, filterable by operator identity and action type.

#### A.12.4.4 -- Clock Synchronization

**SENTINEL implementation:**

SENTINEL relies on the underlying infrastructure's clock synchronization (NTP or PTP). All timestamps are recorded in UTC.

The correlation engine's temporal windowing depends on clock synchronization between probe agents and the correlation engine. Clock skew greater than the correlation window (default 300 seconds) could cause events to be mis-correlated. Operators should ensure NTP is configured on all nodes.

**Evidence:** Timestamp consistency checks in the audit trail (sequential entries should have monotonically non-decreasing timestamps).

### A.12.6 -- Technical Vulnerability Management

**Control description:** Information about technical vulnerabilities of information systems being used shall be obtained in a timely fashion, the organization's exposure to such vulnerabilities evaluated, and appropriate measures taken to address the associated risk.

**SENTINEL implementation:**

SENTINEL directly addresses a class of technical vulnerability that traditional vulnerability management does not cover: hardware computational defects.

- **Vulnerability identification:** SENTINEL continuously tests for SDC-inducing hardware defects via computational probes and output monitoring.
- **Exposure evaluation:** The Bayesian attribution model quantifies the reliability risk for each GPU as a probability score.
- **Risk remediation:** The quarantine state machine automatically removes at-risk GPUs from production, eliminating exposure to the identified vulnerability.

**Evidence:** GPU reliability score history. Quarantine actions and their triggering evidence. Fleet-wide SDC rate trends.

---

## A.16 -- Information Security Incident Management

### A.16.1 -- Management of Information Security Incidents and Improvements

#### A.16.1.1 -- Responsibilities and Procedures

**Control description:** Management responsibilities and procedures shall be established to ensure a quick, effective and orderly response to information security incidents.

**SENTINEL implementation:**

SENTINEL codifies SDC incident response in its quarantine state machine and alert routing:

1. **Detection:** Probes, monitors, and TMR detect anomalous behavior.
2. **Classification:** The correlation engine classifies the severity (INFO, WARNING, HIGH, CRITICAL) based on Bayesian analysis.
3. **Notification:** Alerts are routed to the appropriate channels based on severity:
   - WARNING: Slack notification to the ops channel.
   - HIGH: Slack + PagerDuty on-call engineer.
   - CRITICAL: Slack + PagerDuty + email to management.
4. **Containment:** Automatic quarantine removes the GPU from production.
5. **Investigation:** Operators use the dashboard and SDK to review evidence, run deep tests, and determine root cause.
6. **Resolution:** GPU is either reinstated (transient fault) or condemned (permanent fault) and replaced.
7. **Post-incident review:** Full incident timeline is available in the audit trail for review.

**Evidence:** Incident records in the audit trail. Alert dispatch records. Quarantine timeline with state transitions and operator actions.

#### A.16.1.2 -- Reporting Information Security Events

**Control description:** Information security events shall be reported through appropriate management channels as quickly as possible.

**SENTINEL implementation:**

SENTINEL's alerting system provides automated event reporting:

| Event Type | Channels | Latency |
|-----------|----------|---------|
| Single probe failure | Slack | < 1 minute |
| Consecutive probe failures | Slack + PagerDuty | < 1 minute |
| Multi-SM probe failure | Slack + PagerDuty + Email | < 1 minute |
| GPU quarantined | Slack + PagerDuty + Email | Immediate |
| GPU condemned | Slack + PagerDuty + Email | Immediate |
| Audit chain integrity failure | Slack + PagerDuty + Email | Immediate |
| Fleet-wide SDC rate elevated | Slack + PagerDuty + Email | < 30 minutes |

Alert routing is configurable via `config/alerting/alert_rules.yaml`. Each alert includes:
- Severity level.
- Affected GPU UUID and hostname.
- Summary of the triggering event(s).
- Link to the SENTINEL dashboard for investigation.

**Evidence:** Alert dispatch records with timestamps, channels, and acknowledgment status.

#### A.16.1.4 -- Assessment of and Decision on Information Security Events

**Control description:** Information security events shall be assessed and it shall be decided if they are to be classified as information security incidents.

**SENTINEL implementation:**

The correlation engine performs automated event assessment:

- **Bayesian analysis:** Each event is assessed in the context of the GPU's full history using the Bayesian attribution model. A single probe failure on a GPU with 10,000 successful probes is assessed differently from a single probe failure on a GPU with 50 previous failures.
- **Temporal correlation:** Events occurring within the correlation window (default 300 seconds) are analyzed together. Convergent evidence from multiple detection layers increases confidence.
- **Pattern matching:** Fleet-wide patterns are analyzed to distinguish hardware faults from software bugs or infrastructure events.
- **Confidence scoring:** Each correlation event includes a confidence score. Only events above the minimum confidence threshold (default 0.6) generate alerts.

**Evidence:** Correlation event records with confidence scores, contributing evidence, and classification decisions.

#### A.16.1.5 -- Response to Information Security Incidents

**Control description:** Information security incidents shall be responded to in accordance with the documented procedures.

**SENTINEL implementation:**

Response procedures are codified in the quarantine state machine:

| GPU State | Automated Response | Required Operator Action |
|-----------|-------------------|------------------------|
| HEALTHY -> SUSPECT | Increase probe frequency. Begin TMR validation. | Monitor; no action required. |
| SUSPECT -> QUARANTINED | Remove from production scheduling. Alert operators. | Review evidence. Schedule deep test. |
| QUARANTINED -> DEEP_TEST | Run comprehensive diagnostic suite. | Review deep test results. |
| DEEP_TEST -> CONDEMNED | Flag for permanent removal. Alert management. | Initiate hardware replacement. |
| DEEP_TEST -> HEALTHY | Reinstate to production with reset prior. | Verify reinstatement. |

All responses are recorded in the audit trail with timestamps and outcomes.

**Evidence:** State transition records. Deep test results. Operator action logs. Time-to-response metrics.

#### A.16.1.6 -- Learning from Information Security Incidents

**Control description:** Knowledge gained from analyzing and resolving information security incidents shall be used to reduce the likelihood or impact of future incidents.

**SENTINEL implementation:**

SENTINEL supports post-incident learning through:

- **Complete incident timelines** in the audit trail, queryable by GPU, node, time range, and event type.
- **Trend analysis:** Historical SDC rates by GPU model, age, temperature range, and workload type, enabling identification of systemic issues.
- **Threshold refinement:** The [Calibration Guide](../calibration-guide.md) provides a methodology for tuning detection sensitivity based on operational experience.
- **Pattern library:** The correlation engine's pattern matcher can be updated with new patterns discovered during incident analysis.

**Evidence:** Incident timeline exports. Trend analysis reports. Configuration change records showing threshold adjustments made in response to incidents.

#### A.16.1.7 -- Collection of Evidence

**Control description:** The organization shall define and apply procedures for the identification, collection, acquisition and preservation of information, which can serve as evidence.

**SENTINEL implementation:**

Evidence collection and preservation is a core function of the audit ledger:

- **Identification:** Every event is assigned a unique ID, timestamp, and cryptographic hash.
- **Collection:** Events are collected in real time from probe agents, monitors, and the correlation engine. Collection is automated and continuous.
- **Preservation:** Events are preserved in the tamper-evident hash chain. The chain's cryptographic properties ensure that stored evidence cannot be modified without detection.
- **Integrity:** Automatic chain verification every 6 hours confirms evidence has not been tampered with.
- **Retention:** Configurable retention policies (default 365 days for details, forever for Merkle roots) ensure evidence is available for the required period.
- **Export:** Evidence can be exported in machine-readable formats (JSON, CSV) or human-readable reports (PDF) for external use.

**Evidence:** Hash chain verification reports. Retention policy configuration. Data export records.

---

## Additional Relevant Controls

### A.5.37 -- Documented Operating Procedures

SENTINEL contributes operational documentation:
- Deployment Guide for installation and configuration.
- Operator Runbook for day-to-day operations.
- Calibration Guide for sensitivity tuning.
- This compliance mapping document.

### A.8.8 -- Management of Technical Vulnerabilities

SENTINEL manages a specific class of technical vulnerability (hardware SDC) through continuous testing, risk scoring, and automated remediation (quarantine).

### A.8.15 -- Logging

SENTINEL provides comprehensive, tamper-evident logging of hardware integrity events, operational decisions, and configuration changes. See [A.12.4 -- Logging and Monitoring](#a124--logging-and-monitoring) above.

### A.8.16 -- Monitoring Activities

SENTINEL provides continuous monitoring of GPU compute integrity. See [A.12 -- Operations Security](#a12--operations-security) above.

---

## Evidence Generation

### ISO 27001 Compliance Reports

Generate ISO 27001 compliance reports on demand:

```python
from sentinel_sdk import SentinelClient

client = SentinelClient("audit-ledger:50052")

# Generate ISO 27001 report
report = client.generate_iso27001_report(
    start_date="2026-01-01",
    end_date="2026-03-31",
    controls=["A.8", "A.12", "A.16"],  # Or "all" for all mapped controls
    output_format="pdf",
    include_raw_evidence=False
)

report.save("sentinel_iso27001_q1_2026.pdf")
```

### Report Contents

Each generated report includes:

1. **Cover page:** Report period, generation timestamp, SENTINEL version, chain verification status.
2. **Executive summary:** Fleet size, GPUs monitored, SDC events detected, quarantine actions, false positive rate.
3. **Per-control evidence sections:**
   - Control description (from ISO 27001 Annex A).
   - SENTINEL implementation summary.
   - Quantitative evidence (metrics, counts, rates).
   - Qualitative evidence (configuration excerpts, procedure references).
   - Gaps or limitations (if any).
4. **Appendix A: Raw data export** (optional): Machine-readable evidence in JSON format.
5. **Appendix B: Chain integrity attestation:** Hash chain root, verification result, and number of entries verified.

### Evidence Integrity Guarantee

All evidence in SENTINEL reports is sourced from the tamper-evident audit trail. The report header includes:

- The audit chain root hash at the time of report generation.
- The number of audit entries covering the reporting period.
- The result of the chain integrity verification performed during report generation.

An auditor can independently verify the chain:

```bash
sentinel-audit-ledger verify --full --attest \
  --output attestation.json
```

And compare the root hash in the attestation with the root hash in the report to confirm that the report was generated from an intact, unmodified evidence chain.
