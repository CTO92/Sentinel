# SENTINEL Operator Runbook

> **Version:** 0.1.0-alpha | **Status:** Pre-release | **Last Updated:** 2026-03-24

This runbook provides day-to-day operational guidance for teams running SENTINEL in production. It covers monitoring, alert triage, quarantine management, audit trail operations, common scenarios, and troubleshooting.

## Table of Contents

1. [Monitoring SENTINEL](#monitoring-sentinel)
2. [Alert Triage](#alert-triage)
3. [Quarantine Management](#quarantine-management)
4. [Audit Trail Operations](#audit-trail-operations)
5. [Common Scenarios](#common-scenarios)
6. [Troubleshooting](#troubleshooting)

---

## Monitoring SENTINEL

### Key Prometheus Metrics

The following metrics should be actively monitored. All metrics are exported by the correlation engine (port 9090) and audit ledger (port 9091).

#### Probe Agent Health

| Metric | Type | Description | Alert Threshold |
|--------|------|-------------|----------------|
| `sentinel_probe_results_total{result="PASS"}` | Counter | Total successful probes | N/A (baseline) |
| `sentinel_probe_results_total{result="FAIL"}` | Counter | Total failed probes | Rate > 0 sustained |
| `sentinel_probe_results_total{result="TIMEOUT"}` | Counter | Total timed-out probes | Rate > 0.01/s |
| `sentinel_probe_duration_seconds` | Histogram | Probe execution time | p99 > 2x baseline |
| `sentinel_probe_overhead_pct` | Gauge | Current probe overhead (%) | > configured budget |
| `sentinel_agent_connected_gpus` | Gauge | Number of GPUs monitored per agent | Drops to 0 |
| `sentinel_agent_grpc_stream_active` | Gauge | Is the gRPC stream to correlation engine active | 0 = disconnected |

#### Correlation Engine Health

| Metric | Type | Description | Alert Threshold |
|--------|------|-------------|----------------|
| `sentinel_correlation_events_processed_total` | Counter | Total events processed | Sudden drop |
| `sentinel_correlation_buffer_size` | Gauge | Events in correlation buffer | > 80% of max |
| `sentinel_correlation_processing_latency_seconds` | Histogram | Event-to-decision latency | p99 > 5s |
| `sentinel_gpu_reliability_score` | Gauge (per GPU) | Current Bayesian reliability score | < 0.95 |
| `sentinel_gpu_state` | Gauge (per GPU) | Current quarantine state | Any non-HEALTHY |
| `sentinel_quarantine_total` | Counter | Total quarantine actions | Any increment |
| `sentinel_tmr_dissent_total` | Counter | TMR voting dissents | Any increment |
| `sentinel_grpc_active_streams` | Gauge | Active gRPC client streams | Sudden drop |

#### Audit Ledger Health

| Metric | Type | Description | Alert Threshold |
|--------|------|-------------|----------------|
| `sentinel_audit_entries_total` | Counter | Total audit entries written | Sustained zero rate |
| `sentinel_audit_batch_duration_seconds` | Histogram | Batch commit time | p99 > 10s |
| `sentinel_audit_chain_verified` | Gauge | Last verification result (1=ok, 0=fail) | 0 |
| `sentinel_audit_chain_length` | Gauge | Total entries in chain | N/A (monitoring) |
| `sentinel_audit_pending_entries` | Gauge | Entries awaiting batch commit | > 10,000 |

### Grafana Dashboard Setup

SENTINEL ships with pre-built Grafana dashboard JSON files. Import them via:

1. Navigate to Grafana -> Dashboards -> Import.
2. Upload JSON files from `deploy/grafana/provisioning/dashboards/`.
3. Select the Prometheus data source configured for SENTINEL metrics.

Recommended dashboards:

- **Fleet Health Overview**: Top-level view of all GPUs, color-coded by state (green=HEALTHY, yellow=SUSPECT, red=QUARANTINED, black=CONDEMNED). Shows fleet-wide SDC rate, quarantine count over time, and probe pass rate.
- **GPU Deep Dive**: Select a GPU UUID to see its reliability score history, probe results timeline, anomaly events, temperature/voltage correlation, and quarantine history.
- **Correlation Engine Internals**: Event processing rate, buffer utilization, Bayesian model update rate, TMR scheduling, alert dispatch rate.
- **Audit Ledger Status**: Chain integrity, batch processing rate, storage utilization, retention policy status.

### Alerting Rules for SENTINEL Itself

In addition to the SDC detection alerts (defined in `config/alerting/alert_rules.yaml`), configure these alerts for SENTINEL operational health:

```yaml
# Prometheus alerting rules for SENTINEL operational health
groups:
  - name: sentinel-operational
    rules:
      - alert: SentinelAgentDisconnected
        expr: sentinel_agent_grpc_stream_active == 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Probe agent on {{ $labels.hostname }} disconnected"
          description: "The probe agent has been disconnected for 5 minutes. GPUs on this node are not being monitored."

      - alert: SentinelCorrelationBufferHigh
        expr: sentinel_correlation_buffer_size / sentinel_correlation_buffer_max > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Correlation engine buffer at {{ $value | humanizePercentage }}"
          description: "Events may be dropped if the buffer fills up."

      - alert: SentinelCorrelationLatencyHigh
        expr: histogram_quantile(0.99, sentinel_correlation_processing_latency_seconds_bucket) > 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Correlation engine p99 latency is {{ $value }}s"

      - alert: SentinelAuditChainBroken
        expr: sentinel_audit_chain_verified == 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Audit ledger hash chain integrity check FAILED"
          description: "Immediate investigation required. This may indicate data tampering or storage corruption."

      - alert: SentinelAuditWriteStalled
        expr: rate(sentinel_audit_entries_total[5m]) == 0 and sentinel_audit_pending_entries > 0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Audit ledger has stopped writing despite pending entries"

      - alert: SentinelProbeOverheadExceeded
        expr: sentinel_probe_overhead_pct > 3.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Probe overhead on {{ $labels.hostname }} is {{ $value }}%"
          description: "Exceeds configured budget. Probes may be interfering with production workloads."
```

---

## Alert Triage

### When a GPU Enters SUSPECT State

**What happened:** The GPU's Bayesian reliability score dropped below 0.95 (configurable via `healthy_to_suspect_threshold`), or multiple anomaly events were received within the correlation window.

**Immediate actions:**

1. **Check the event history** for this GPU in the dashboard (GPU Deep Dive view). Look at what triggered the state change:
   - Was it a probe failure? Which probe type? Which SM?
   - Was it an inference/training anomaly? What kind?
   - Were there correlated telemetry events (thermal throttling, ECC errors)?

2. **Check for fleet-wide patterns.** Are other GPUs also entering SUSPECT? If multiple GPUs on the same node or in the same rack are suspect simultaneously, the root cause may be infrastructure (power, cooling, network) rather than individual GPU defects.

3. **No immediate action required.** SUSPECT is a monitoring-intensified state, not a production impact. The GPU continues serving production workloads but receives increased probe frequency automatically. If the failures were transient, the reliability score will recover, and the GPU will return to HEALTHY after meeting the reinstatement criteria (score >= 0.98 AND 1000 consecutive probe passes).

4. **Monitor over the next few hours.** If the GPU's reliability score continues to decline, it will be automatically quarantined when it crosses the 0.80 threshold.

**Escalation:** Escalate to hardware engineering if:
- The GPU has been SUSPECT for more than 24 hours without recovery.
- The probe failures are concentrated on specific SMs (may indicate a localized defect).
- Telemetry shows abnormal thermal or voltage behavior.

### When a GPU Is QUARANTINED

**What happened:** The GPU's reliability score dropped below 0.80, or it accumulated 3 consecutive probe failures. It has been automatically removed from production workload scheduling.

**Immediate actions:**

1. **Verify the quarantine is justified.** In the dashboard, review the evidence trail:
   - Click on the GPU in the Fleet Health view.
   - Review the "Quarantine Evidence" section showing all contributing events.
   - Check if the failures are diverse (multiple probe types, multiple SMs) or narrow (single probe type on one SM).

2. **Check workload impact.** Determine whether any production workloads were affected by the quarantine:
   - Was the GPU part of a DDP training group? If so, training may have been interrupted.
   - Was it serving inference traffic? Traffic should have been automatically rerouted.

3. **Decide on deep test vs. manual review:**
   - If the evidence is strong (multiple probe types failing, TMR dissent), schedule a deep test.
   - If the evidence is weak or the failures look like they could be false positives, perform a manual review (see [False Positive Investigation Checklist](#false-positive-investigation-checklist)).

4. **Log the triage in the audit trail** via the dashboard or SDK.

### Interpreting Probe Failures

#### Single Probe Failure

A single probe failure on a single SM is **not necessarily** SDC. It could be:
- A transient fault (cosmic ray, power glitch) -- genuine but not persistent.
- A race condition in probe scheduling (should not happen but is possible in alpha).
- A software bug in the probe implementation.

**Action:** No immediate action. The Bayesian model will incorporate this as weak evidence and adjust the reliability score slightly. Monitor for repetition.

#### Repeated Failures on the Same SM

Multiple failures on the same SM across different time windows is strong evidence of a localized hardware defect. This SM likely has a stuck-at fault, small-delay defect, or degraded component.

**Action:** The correlation engine will automatically escalate to SUSPECT and eventually QUARANTINED. Document the SM ID for hardware engineering. If the GPU is under warranty, this evidence supports an RMA.

#### GPU-Wide Failures (Multiple SMs, Multiple Probe Types)

Failures across many SMs and multiple probe types (e.g., FMA and Tensor Core probes both failing) indicate systemic GPU corruption.

**Action:** The GPU should be quarantined rapidly. This pattern strongly suggests hardware replacement. Schedule a deep test to confirm.

#### Single Probe Type Failing Fleet-Wide

If the same probe type (e.g., FMA probe) is failing across many GPUs simultaneously, this is more likely a **software issue** (probe bug, golden answer error, driver issue) than real SDC.

**Action:** Do NOT quarantine individual GPUs. Instead:
1. Check if a recent SENTINEL update changed the probe implementation.
2. Verify the golden answers are correct for the current GPU architecture.
3. Check for recent driver updates that may have changed behavior.
4. Pause the problematic probe type if needed while investigating.

### Interpreting Inference/Training Anomalies

Inference and training anomalies are **weaker signals** than probe failures because they can have many non-SDC causes:

- Model serving bugs.
- Data distribution shifts.
- Learning rate schedule changes.
- Batch size changes.
- Normal training instability.

The correlation engine weights anomaly events at 0.3x compared to probe failures at 1.0x for this reason.

**Anomaly is SDC if:**
- It is correlated with probe failures on the same GPU in the same time window.
- It is localized to a single GPU while all peers are normal (cross-rank divergence).
- It persists even after ruling out data/config changes.

**Anomaly is NOT SDC if:**
- It occurs fleet-wide simultaneously (data shift, software change).
- It correlates with a model deployment or configuration change.
- It occurs on a GPU with a perfect probe record.

### False Positive Investigation Checklist

When you suspect a quarantine decision was a false positive:

- [ ] Review all contributing events in the audit trail. Count distinct event types and SMs.
- [ ] Check if a software update (SENTINEL, driver, firmware) was deployed around the same time.
- [ ] Check if the probe golden answers match the GPU architecture (e.g., H100 vs. A100 golden answers).
- [ ] Check if the probe tolerance settings are appropriate for this GPU generation.
- [ ] Check the GPU's temperature and voltage history around the time of failure. Was the GPU operating outside normal ranges?
- [ ] Run the probe manually in standalone mode and verify the result.
- [ ] Check if other GPUs in the same node/rack/cluster experienced similar issues.
- [ ] If the evidence suggests a false positive, reinstate the GPU and adjust thresholds if needed (see [Calibration Guide](calibration-guide.md)).

---

## Quarantine Management

### Reviewing Quarantine Decisions

Via the dashboard:

1. Navigate to Fleet Health -> Quarantine Queue.
2. Each quarantined GPU shows:
   - UUID, hostname, slot, GPU model.
   - State (QUARANTINED, DEEP_TEST, CONDEMNED).
   - Time in current state.
   - Reliability score at quarantine time and current.
   - Number and types of contributing events.
   - Link to full evidence timeline.

Via the SDK:

```python
from sentinel_sdk import SentinelClient

client = SentinelClient("correlation-engine:50051")

# List all quarantined GPUs
quarantined = client.list_gpus(state="QUARANTINED")
for gpu in quarantined:
    print(f"{gpu.uuid} on {gpu.hostname}: score={gpu.reliability_score:.4f}, "
          f"quarantined_at={gpu.quarantined_at}, reason={gpu.quarantine_reason}")

# Get detailed evidence for a specific GPU
evidence = client.get_quarantine_evidence(gpu_uuid="GPU-abcd-1234")
for event in evidence:
    print(f"  {event.timestamp}: {event.event_type} - {event.description}")
```

### Manual Quarantine and Reinstatement

**Manual quarantine** (remove a GPU from production even if SENTINEL has not flagged it):

```python
client.quarantine_gpu(
    gpu_uuid="GPU-abcd-1234",
    reason="Operator-initiated: suspected hardware issue based on vendor advisory",
    operator="jane.doe@company.com"
)
```

Via the dashboard: Fleet Health -> select GPU -> Actions -> Quarantine.

**Manual reinstatement** (return a quarantined GPU to production):

```python
client.reinstate_gpu(
    gpu_uuid="GPU-abcd-1234",
    reason="Deep test passed; transient fault confirmed by vendor",
    operator="jane.doe@company.com",
    reset_reliability=True  # Reset Bayesian prior to default
)
```

Via the dashboard: Quarantine Queue -> select GPU -> Actions -> Reinstate.

**Warning:** Reinstating a GPU resets its reliability score to the prior (default `Beta(100, 1)`). If the GPU has a genuine hardware defect, it will be re-quarantined once probes detect the fault again.

If `require_approval` is enabled in the correlation engine configuration, quarantine and reinstatement actions require a second operator to approve before the state transition completes.

### Deep Test Workflow

When a GPU is quarantined, a deep test can be scheduled to run comprehensive diagnostics:

1. **Schedule the deep test:**
   ```python
   client.schedule_deep_test(gpu_uuid="GPU-abcd-1234")
   ```
   This transitions the GPU from QUARANTINED to DEEP_TEST state.

2. **Deep test execution:** The probe agent runs an exhaustive test suite on the GPU:
   - All probe types at maximum SM coverage (100%).
   - Increased iteration counts (10x normal) to catch intermittent faults.
   - BIST (Built-In Self-Test) via NVML if available.
   - Thermal stress test: run maximum-power kernels while monitoring for failures that appear only at high temperatures.
   - Memory exhaustive test: walking-ones/zeros across all addressable memory.

3. **Deep test results:**
   - **PASS:** The GPU is returned to HEALTHY with a reset reliability prior. This suggests the original failure was transient (e.g., cosmic ray, power glitch).
   - **FAIL:** The GPU is moved to CONDEMNED. Document the failure mode for hardware replacement.

4. **Typical deep test duration:** 15-60 minutes depending on GPU memory capacity and probe configuration.

### Hardware Replacement Workflow

When a GPU is CONDEMNED:

1. The GPU cannot serve production workloads and will not be reinstated.
2. Generate a hardware replacement report:
   ```python
   report = client.generate_replacement_report(gpu_uuid="GPU-abcd-1234")
   # Returns: serial number, failure timeline, probe evidence, thermal/voltage history
   ```
3. Open a hardware replacement ticket with the GPU vendor, attaching the replacement report as evidence.
4. After physical replacement, the new GPU will be auto-discovered by the probe agent on the node. It starts in HEALTHY state with a fresh Bayesian prior.
5. The old GPU's entry in the audit trail is retained indefinitely for compliance purposes.

---

## Audit Trail Operations

### Querying the Audit Trail

The audit trail records every significant event: probe results, anomaly detections, state transitions, operator actions, configuration changes, and system events.

**Via the SDK:**

```python
from sentinel_sdk import SentinelClient
from datetime import datetime, timedelta

client = SentinelClient("audit-ledger:50052")

# Query events by time range
events = client.query_audit(
    start_time=datetime.utcnow() - timedelta(hours=24),
    end_time=datetime.utcnow(),
    event_types=["quarantine", "probe_result"],
    gpu_uuid="GPU-abcd-1234",  # optional filter
    limit=1000
)

for entry in events:
    print(f"[{entry.timestamp}] {entry.event_type}: {entry.summary}")
    print(f"  Hash: {entry.hash}")
    print(f"  Prev: {entry.previous_hash}")
```

**Via the dashboard:** Audit Trail tab provides a searchable, filterable timeline view.

### Chain Verification

Verify the integrity of the audit hash chain:

```bash
# Verify the last 10,000 entries
sentinel-audit-ledger verify --depth 10000 --config /etc/sentinel/config/sentinel.yaml

# Verify the entire chain (may take a long time for large chains)
sentinel-audit-ledger verify --full --config /etc/sentinel/config/sentinel.yaml

# Verify and output a signed attestation
sentinel-audit-ledger verify --depth 10000 --attest --output attestation.json
```

**Via the SDK:**

```python
result = client.verify_chain(depth=10000)
print(f"Verified: {result.valid}")
print(f"Entries checked: {result.entries_checked}")
print(f"Chain root: {result.root_hash}")
if not result.valid:
    print(f"First broken entry: {result.first_broken_entry_id}")
```

Automatic verification runs every 6 hours (configurable via `auto_verify_interval_hours`). Failed verifications trigger a CRITICAL alert.

### Compliance Report Generation

#### SOC 2 Reports

```python
report = client.generate_soc2_report(
    start_date=datetime(2026, 1, 1),
    end_date=datetime(2026, 3, 31),
    output_format="pdf"  # or "json", "csv"
)
report.save("soc2_q1_2026.pdf")
```

The SOC 2 report includes:
- CC6.1 (Logical Access): RBAC configuration, access logs, mTLS certificate status.
- CC7.2 (System Monitoring): Probe execution summary, anomaly detection statistics, response times.
- CC8.1 (Change Management): Configuration change history from audit trail.
- A1.2 (System Availability): GPU uptime, quarantine durations, SENTINEL service availability.

#### ISO 27001 Reports

```python
report = client.generate_iso27001_report(
    start_date=datetime(2026, 1, 1),
    end_date=datetime(2026, 3, 31),
    controls=["A.8", "A.12", "A.16"],  # or "all"
    output_format="pdf"
)
report.save("iso27001_q1_2026.pdf")
```

See [compliance/soc2-controls.md](compliance/soc2-controls.md) and [compliance/iso27001-mapping.md](compliance/iso27001-mapping.md) for detailed control mappings.

### Data Retention and Pruning

Retention is configured in `sentinel.yaml` under `audit_ledger.retention`:

| Setting | Default | Description |
|---------|---------|-------------|
| `detail_retention_days` | 365 | How long to keep detailed audit entries (event data, evidence). |
| `summary_retention_days` | 0 (forever) | How long to keep Merkle roots and batch summaries. |
| `cleanup_interval_hours` | 24 | How often the retention cleanup job runs. |

**Important:** Setting `summary_retention_days` to a non-zero value means that cryptographic proof chains older than that period cannot be verified. For compliance purposes, keep summaries forever (0).

Manual pruning:

```bash
# Preview what would be pruned (dry run)
sentinel-audit-ledger prune --dry-run --before 2025-01-01

# Execute pruning
sentinel-audit-ledger prune --before 2025-01-01 --confirm
```

---

## Common Scenarios

### "A GPU Was Quarantined -- Now What?"

1. **Do not panic.** Quarantine is SENTINEL working as intended. The GPU has been safely removed from production workloads.

2. **Review the evidence** in the dashboard (Quarantine Queue -> select GPU -> Evidence Timeline).

3. **Decide on next steps:**

   | Evidence Pattern | Recommended Action |
   |-----------------|-------------------|
   | Single probe type, single SM, few failures | Likely transient. Schedule deep test. If it passes, reinstate. |
   | Multiple probe types, single SM | Localized defect. Schedule deep test. Likely will fail -> CONDEMNED. |
   | Multiple probe types, multiple SMs | Systemic GPU issue. Schedule deep test -> likely CONDEMNED -> hardware replacement. |
   | Anomaly events only, no probe failures | Possibly false positive. Review anomaly details. Consider reinstating with monitoring. |
   | TMR dissent + probe failures | Strong evidence. Schedule deep test -> likely CONDEMNED. |

4. **Schedule a deep test** if warranted (see [Deep Test Workflow](#deep-test-workflow)).

5. **If the GPU is condemned**, initiate hardware replacement (see [Hardware Replacement Workflow](#hardware-replacement-workflow)).

### "Training Loss Spiked -- Is It SDC?"

1. **Check the training monitor alerts.** Look for `LOSS_SPIKE` or `GRADIENT_NORM_SPIKE` anomaly events.

2. **Check for cross-rank divergence.** If the loss spike is accompanied by a `CROSS_RANK_DIVERGENCE` event, it strongly suggests one GPU is computing differently from peers. The divergent rank/GPU is identified in the event details.

3. **Rule out non-SDC causes:**
   - Did the learning rate schedule change at this step?
   - Did the data loader encounter an anomalous batch?
   - Was there a gradient accumulation boundary?
   - Did the optimizer state get corrupted (check checkpoint integrity)?

4. **Correlate with probe data.** In the dashboard, check if the GPU associated with the divergent rank has any probe failures around the same time.

5. **If SDC is confirmed:**
   - The affected GPU should already be flagged as SUSPECT or QUARANTINED by the correlation engine.
   - Roll back the training checkpoint to the last verified-good checkpoint (the training monitor tracks checkpoint hashes).
   - Resume training with the faulty GPU excluded.

### "Probe Failures Spiked Fleet-Wide -- Is It a Software Bug or Real Hardware?"

This is a critical triage question. A fleet-wide spike is almost never real SDC affecting all GPUs simultaneously.

1. **Check the pattern:**
   - Is it a single probe type or multiple types?
   - Is it all SMs or specific SMs?
   - Is it all GPU models or specific models?

2. **Common non-SDC causes for fleet-wide spikes:**

   | Pattern | Likely Cause | Action |
   |---------|-------------|--------|
   | Single probe type, all GPUs | Golden answer error or probe bug | Verify golden answers; check for recent SENTINEL update |
   | All probe types, all GPUs | Driver update changed behavior | Check for recent driver update; verify probe compatibility |
   | All probe types, specific GPU model | Architecture-specific issue | Check golden answer generation for that architecture |
   | Sporadic across all types | Infrastructure event (power, cooling) | Check datacenter environmental monitoring |

3. **Immediate action:** If you suspect a software issue, pause the affected probe type:
   ```python
   client.update_config({
       "probe_agent.schedule": {
           "FMA": {"enabled": False}  # Disable the suspect probe type
       }
   })
   ```

4. **Investigate and resolve** the root cause before re-enabling.

5. **If it was a false positive cascade**, reinstate all affected GPUs:
   ```python
   for gpu in client.list_gpus(state="QUARANTINED"):
       client.reinstate_gpu(
           gpu_uuid=gpu.uuid,
           reason="Fleet-wide false positive due to [root cause]",
           operator="ops-team@company.com",
           reset_reliability=True
       )
   ```

### "How Do I Add New GPU Nodes to the Monitored Fleet?"

1. **Deploy the probe agent** on the new node (via DaemonSet in Kubernetes, or systemd for bare metal).

2. **GPU auto-discovery** is enabled by default (`probe_agent.discovery.auto_detect: true`). The probe agent will detect all NVIDIA GPUs on the node via NVML and begin monitoring them automatically.

3. **Verify in the dashboard** that the new GPUs appear in the Fleet Health view with HEALTHY status and green indicators.

4. **New GPUs start with the default Bayesian prior** (`Beta(100, 1)`) -- they are assumed healthy until evidence suggests otherwise.

No correlation engine or audit ledger configuration changes are needed. The system scales dynamically.

### "How Do I Exclude Specific GPUs from Monitoring?"

**Option A: Exclude via configuration** (probe agent will not monitor these GPUs):

In `sentinel.yaml`:
```yaml
probe_agent:
  discovery:
    auto_detect: true
    exclude_gpu_uuids:
      - "GPU-xxxx-yyyy-zzzz"
      - "GPU-aaaa-bbbb-cccc"
```

**Option B: Explicit GPU list** (monitor only these GPUs):

```yaml
probe_agent:
  discovery:
    auto_detect: false
    gpu_uuids:
      - "GPU-1111-2222-3333"
      - "GPU-4444-5555-6666"
```

**Option C: Via the SDK** (dynamic exclusion without config change):

```python
client.set_gpu_monitoring(gpu_uuid="GPU-xxxx-yyyy-zzzz", enabled=False)
```

---

## Troubleshooting

### Probe Agent Not Connecting to Correlation Engine

**Symptoms:** `sentinel_agent_grpc_stream_active` is 0. Probe agent logs show connection errors.

**Diagnosis:**

1. **Check network connectivity:**
   ```bash
   # From the probe agent node
   curl -v telnet://correlation-engine:50051
   # Or in Kubernetes
   kubectl exec -n sentinel <probe-agent-pod> -- \
     /usr/local/bin/grpc_health_probe -addr=correlation-engine:50051
   ```

2. **Check TLS certificates:**
   - Verify that the probe agent's certificate is signed by the same CA that the correlation engine trusts.
   - Check certificate expiry: `openssl x509 -in /etc/sentinel/certs/agent.crt -noout -dates`.
   - Verify the CA certificate is present: `ls -la /etc/sentinel/certs/ca.crt`.

3. **Check network policies:** If running in Kubernetes, verify that the network policy allows probe-agent -> correlation-engine traffic on port 50051.

4. **Check correlation engine health:**
   ```bash
   kubectl logs -n sentinel deployment/correlation-engine --tail=100
   ```

5. **Check HMAC key:** The probe agent and correlation engine must share the same HMAC key. Verify that the `hmac_key` (or `SENTINEL_PROBE_AGENT_HMAC_KEY` env var) matches on both sides.

### High False Positive Rate

**Symptoms:** GPUs are being quarantined but deep tests pass. Operators are frequently reinstating GPUs.

**Diagnosis and fixes:**

1. **Check probe tolerances.** If transcendental probe tolerances are too tight for the GPU architecture, false positives will occur:
   ```yaml
   # In config/thresholds/probe_tolerances.yaml
   TRANSCENDENTAL:
     max_ulp: 2  # Increase from 1 to 2 if false positives on specific arch
   ```

2. **Check EWMA parameters.** If the sigma multiplier is too low, normal output variance will trigger anomalies:
   ```yaml
   inference_monitor:
     ewma:
       alpha: 0.005  # Decrease for smoother tracking (less reactive)
   ```

3. **Check quarantine thresholds.** If the suspect and quarantine thresholds are too aggressive:
   ```yaml
   correlation_engine:
     state_machine:
       healthy_to_suspect_threshold: 0.90  # Lower from 0.95
       suspect_to_quarantine_threshold: 0.70  # Lower from 0.80
   ```

4. **Increase the Bayesian prior strength.** A stronger prior requires more evidence before the reliability score drops:
   ```yaml
   correlation_engine:
     bayesian_model:
       prior_alpha: 200.0  # Increase from 100.0
   ```

5. **See the [Calibration Guide](calibration-guide.md)** for systematic threshold tuning.

### Missing Telemetry Data

**Symptoms:** Gaps in Grafana dashboards. Some GPUs show no recent probe results.

**Diagnosis:**

1. **Check probe agent status:**
   ```bash
   systemctl status sentinel-probe-agent  # bare metal
   kubectl logs -n sentinel daemonset/sentinel-probe-agent --tail=50  # Kubernetes
   ```

2. **Check if probes are timing out:** Look for `TIMEOUT` results in the metrics. If probes are timing out, the GPU may be overloaded or the timeout is too short.

3. **Check if the overhead budget is too tight:** If `overhead_budget_pct` is set very low (e.g., 0.5%), probes may be deferred so aggressively that they rarely run:
   ```bash
   curl -s http://<agent-metrics>:9092/metrics | grep sentinel_probe_overhead_pct
   ```

4. **Check gRPC batching:** If `batch_flush_interval_ms` is set very high, telemetry may be delayed. Check `sentinel_agent_grpc_stream_active` to confirm the stream is active.

5. **Check ScyllaDB health:** If ScyllaDB is unhealthy or overloaded, writes may be failing:
   ```bash
   kubectl exec -n sentinel sentinel-scylla-0 -- nodetool status
   ```

### Audit Chain Verification Failure

**Symptoms:** `sentinel_audit_chain_verified` drops to 0. CRITICAL alert fired.

**This is a serious event.** Possible causes:

1. **Storage corruption:** A disk failure or filesystem error corrupted stored audit entries.
2. **Software bug:** A bug in the hash computation or batch processing produced an incorrect hash.
3. **Tampering:** Someone modified audit entries in the database.
4. **Incomplete write:** A power failure or crash during batch commit left an incomplete entry.

**Investigation steps:**

1. **Identify the broken entry:**
   ```bash
   sentinel-audit-ledger verify --full --verbose
   # Output will show the first entry where hash verification fails
   ```

2. **Check storage health:**
   ```bash
   # ScyllaDB
   kubectl exec -n sentinel sentinel-scylla-0 -- nodetool scrub sentinel_audit
   # PostgreSQL
   psql -h sentinel-postgres -U sentinel -c "SELECT * FROM pg_stat_database WHERE datname='sentinel';"
   ```

3. **Check for concurrent writers:** Verify that only ONE audit ledger writer instance is running. Two writers would produce conflicting hash chains.

4. **If the break is at a recent entry** and correlates with a crash/restart, the entry can be repaired by re-processing the pending events for that batch.

5. **If the break is at an older entry** and cannot be explained by operational events, treat it as a potential security incident and investigate further.

### Performance Overhead Exceeding Budget

**Symptoms:** `sentinel_probe_overhead_pct` exceeds the configured budget. Production workloads may be affected.

**Immediate mitigation:**

1. **Switch to the low-overhead probe schedule:**
   ```python
   client.update_config({
       "probe_agent.schedule_file": "probe_schedules/low_overhead.yaml"
   })
   ```

2. **Reduce probe frequency** for high-cost probes:
   ```yaml
   # Increase periods for expensive probes
   schedule:
     - type: MEMORY
       period_seconds: 1800  # Increase from 600
     - type: TENSOR_CORE
       period_seconds: 120   # Increase from 60
   ```

3. **Reduce SM coverage:**
   ```yaml
   schedule:
     - type: FMA
       sm_coverage: 0.5  # Test half the SMs each cycle
   ```

4. **Disable optional features:**
   ```yaml
   inference_monitor:
     spectral:
       enabled: false
   probe_agent:
     kernel:
       use_cuda_graphs: true  # Ensure CUDA graphs are enabled for lower launch overhead
   ```

5. **See the [Calibration Guide](calibration-guide.md)** for systematic overhead management.
