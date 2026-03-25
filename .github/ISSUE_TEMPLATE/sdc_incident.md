---
name: SDC Incident Report
about: Report a Silent Data Corruption incident detected by SENTINEL or observed in production
title: "[SDC] "
labels: sdc-incident, critical
assignees: ''
---

## Incident Summary

Brief description of the SDC incident.

## Detection

- **Detected by**: (Probe Agent / Inference Monitor / Training Monitor / Correlation Engine / Manual)
- **Detection timestamp**: (ISO 8601)
- **Detection latency**: (time between corruption and detection, if known)
- **Confidence score**: (0.0 - 1.0)
- **Alert ID / Event ID**:

## Affected GPU(s)

| Field | Value |
|-------|-------|
| GPU Model | (e.g., A100 80GB SXM) |
| GPU UUID | |
| Node ID | |
| SM ID (if known) | |
| GPU Serial Number | |
| Driver Version | |
| CUDA Version | |
| VBIOS Version | |
| ECC Mode | (Enabled / Disabled) |
| Temperature at detection | |
| Power draw at detection | |
| Uptime since last reset | |

## Corruption Details

- **Corruption type**: (bit-flip / stuck-at / noise / tensor core / memory / unknown)
- **Memory region**: (weights / activations / gradients / other)
- **Magnitude of deviation**:
- **Affected tensor(s)**: (name, shape, dtype)
- **Probe type that detected**: (FMA / tensor core / memory / transcendental)

## Evidence

### Probe Results

```json
Paste probe result JSON here.
```

### Telemetry at Time of Incident

```json
Paste relevant telemetry (temperature, power, ECC counters) here.
```

### Correlation Engine Analysis

```json
Paste correlation analysis output here.
```

## Impact Assessment

- **Workload affected**: (model name, job ID)
- **Duration of corruption**: (how long was the GPU corrupt before detection?)
- **Data integrity**: (was any output served to users before detection?)
- **Recovery action taken**: (GPU quarantined / job restarted / node drained)

## Root Cause (if known)

Describe the suspected or confirmed root cause.

## Timeline

| Time | Event |
|------|-------|
| | First corruption (estimated) |
| | SENTINEL detection |
| | Alert generated |
| | Response action taken |
| | GPU quarantined / replaced |

## Lessons Learned

What could be improved in SENTINEL's detection for this class of fault?

## Audit Trail

Link to the audit ledger entry for this incident:
