# SENTINEL Calibration Guide

> **Version:** 0.1.0-alpha | **Status:** Pre-release | **Last Updated:** 2026-03-24

This guide covers how to tune SENTINEL's detection sensitivity to minimize false positives while maintaining high detection power. Calibration is essential for production deployments -- default settings are conservative starting points but should be adjusted based on your fleet's baseline behavior.

## Table of Contents

1. [Calibration Philosophy](#calibration-philosophy)
2. [Probe Tolerances](#probe-tolerances)
3. [EWMA Parameters](#ewma-parameters)
4. [KL Divergence Thresholds](#kl-divergence-thresholds)
5. [Bayesian Prior Selection](#bayesian-prior-selection)
6. [Quarantine Thresholds](#quarantine-thresholds)
7. [TMR Scheduling Frequency](#tmr-scheduling-frequency)
8. [Overhead Budget Management](#overhead-budget-management)
9. [Calibration Methodology](#calibration-methodology)
10. [Worked Example: 1000-GPU H100 Cluster Running LLM Inference](#worked-example)

---

## Calibration Philosophy

SENTINEL's detection pipeline has many tunable parameters. The right settings depend on:

- **Fleet size:** Larger fleets produce more events, requiring tighter false positive control.
- **GPU architecture:** Different architectures have different floating-point behaviors.
- **Workload type:** Training vs. inference, LLM vs. vision, latency-sensitive vs. throughput.
- **Risk tolerance:** How much overhead is acceptable vs. how quickly must SDC be detected.

The fundamental trade-off is:

```
More aggressive settings -> Faster detection, Higher false positive rate, Higher overhead
More conservative settings -> Slower detection, Lower false positive rate, Lower overhead
```

**Start conservative and tune based on observed data.** It is better to miss a rare SDC event initially than to flood operators with false alarms that erode trust in the system.

---

## Probe Tolerances

Probe tolerances define how probe results are compared against golden answers. They are configured in `config/thresholds/probe_tolerances.yaml`.

### Exact vs. ULP-Based Comparison

| Probe Type | Default Mode | Rationale |
|-----------|-------------|-----------|
| FMA | Exact (0 ULP) | FMA inputs are chosen to produce exactly representable results. Any deviation is corruption. |
| Tensor Core | Exact (0 ULP) | Same-architecture Tensor Core results are bit-reproducible for the same inputs. |
| AES | Exact (0 ULP) | Integer/bitwise operations must be bit-exact. |
| Memory | Exact | Known pattern write-read must match. |
| Register File | Exact | Known pattern write-read must match. |
| Shared Memory | Exact | Known pattern write-read must match. |
| Transcendental | ULP-based (1 ULP FP32, 2 ULP FP16/BF16) | Transcendental functions may have legitimate 1-ULP variation on some architectures. |

### Per-Architecture Calibration

Golden answers and tolerances must match the GPU architecture in use. The `tools/golden-answer-generator/` generates architecture-specific golden answer files.

If you observe false positives from a specific probe type:

1. **Verify golden answers match the architecture:**
   ```bash
   python tools/golden-answer-generator/verify.py \
     --golden-dir golden/ \
     --gpu-uuid GPU-xxxx-yyyy-zzzz
   ```

2. **Check for driver-version dependencies:** Some transcendental function implementations change across driver versions. If you update the NVIDIA driver, re-verify golden answers.

3. **Adjust ULP tolerance if needed:**
   ```yaml
   # In config/thresholds/probe_tolerances.yaml
   TRANSCENDENTAL:
     tolerance_mode: "ulp"
     max_ulp: 2      # Increase from 1 if seeing 1-ULP variations on your architecture
     fp16_max_ulp: 3  # Increase from 2 if needed
     bf16_max_ulp: 3
   ```

**Warning:** Increasing tolerances reduces detection sensitivity. A multi-bit SDC error that produces a 2-ULP deviation would be missed if tolerance is set to 2 ULP. Only increase tolerances after confirming that the observed variations are legitimate (reproducible on multiple known-good GPUs).

### How to Validate Tolerances

Run the following on a known-good GPU to establish the baseline variation:

```bash
# Run each probe type 10,000 times and record the max ULP deviation
./sentinel-probe-agent --mode benchmark --iterations 10000 --output baseline.json
```

The output shows the maximum observed ULP deviation for each probe type. Set your tolerance to `max_observed + 1 ULP` as a safety margin.

---

## EWMA Parameters

The EWMA (Exponentially Weighted Moving Average) control charts in the inference monitor are configured in `sentinel.yaml` under `inference_monitor.ewma`.

### Smoothing Factor (alpha)

```
EWMA_t = alpha * x_t + (1 - alpha) * EWMA_{t-1}
```

| alpha Value | Behavior | Best For |
|------------|----------|----------|
| 0.001 | Very smooth. Slow to react to changes. Low false positive rate. | Large, stable deployments with constant model and data. |
| 0.01 (default) | Moderate smoothing. Reacts to sustained shifts within ~100 samples. | General purpose. Good starting point. |
| 0.05 | Reactive. Detects shifts quickly but more susceptible to noise. | Small deployments or when rapid detection is critical. |
| 0.1 | Very reactive. High false positive risk. | Debugging/investigation mode only. |

**Tuning guidance:** If your inference workload has naturally high variance in output distributions (e.g., open-ended text generation where outputs vary widely), use a smaller alpha (0.005 or lower) to prevent normal variance from triggering anomalies. If your workload has low natural variance (e.g., classification with stable output distributions), a higher alpha (0.02-0.05) provides faster detection.

### Sigma Multiplier (k)

An anomaly is flagged when `|x_t - EWMA_t| > k * sigma_EWMA`.

| k Value | False Positive Rate (Gaussian) | Behavior |
|---------|-------------------------------|----------|
| 2.0 | ~4.6% | Very sensitive. Many false positives. |
| 3.0 | ~0.27% | Moderately sensitive. |
| 3.5 (default) | ~0.047% | Conservative. Good for production. |
| 4.0 | ~0.006% | Very conservative. May miss small corruptions. |
| 5.0 | ~0.00006% | Extremely conservative. Only detects large corruptions. |

**Tuning guidance:** At 1% sampling rate with 10,000 requests/second, you sample ~100 requests/second, or ~360,000 per hour. At k=3.0, you would expect ~972 false anomaly events per hour from Gaussian noise alone. At k=3.5, that drops to ~169 per hour. At k=4.0, it drops to ~22 per hour.

However, model outputs are often not Gaussian, so these rates are approximate. Calibrate empirically:

1. Deploy with k=4.0 (conservative) initially.
2. Monitor `sentinel_inference_anomaly_total` for one week.
3. If the anomaly rate is near zero on healthy GPUs, gradually reduce k to 3.5 for better sensitivity.
4. If you see a nonzero anomaly rate on known-good hardware, increase k until the false positive rate is acceptable.

### Warmup Period

The `warmup_samples` parameter (default 1000) defines how many samples must be collected before anomaly detection activates. During warmup, the EWMA converges to the true mean and variance.

- **Too short:** The EWMA has not converged, and early deviations trigger false alarms.
- **Too long:** Real SDC that occurs during warmup is not detected.

**Tuning guidance:** At your sampling rate, calculate how long warmup takes in wall-clock time:

```
warmup_duration = warmup_samples / (requests_per_second * sampling_rate)
```

For 10,000 req/s at 1% sampling: `1000 / (10000 * 0.01) = 10 seconds`. This is usually fast enough. For low-traffic services, you may need to reduce `warmup_samples` to 100-500.

---

## KL Divergence Thresholds

KL divergence monitors are configured in `sentinel.yaml` under `inference_monitor.kl_divergence`.

### Reference Distribution

The reference distribution is built from `reference_sample_count` samples (default 10,000) and updated every `reference_update_interval_s` seconds (default 3600).

**Tuning considerations:**

- **reference_sample_count:** More samples produce a more accurate reference distribution. At 1% sampling of 10,000 req/s, 10,000 samples take ~100 seconds to collect. This is a reasonable default.
- **reference_update_interval_s:** The reference must be updated frequently enough to track legitimate distribution shifts (model updates, seasonal data patterns) but not so frequently that the reference includes corrupted data. 3600 seconds (1 hour) is a conservative default.
- **binning_strategy:** `histogram` (default, 256 bins) works well for most workloads. `kde` is better when sample counts are small or the distribution is multi-modal, but has higher computational cost.

### KL Divergence Alert Threshold

KL divergence does not have a single universal threshold. The expected KL divergence between two samples from the same distribution depends on the distribution's entropy and the sample sizes.

**Calibration procedure:**

1. Collect KL divergence measurements from known-good hardware over several hours:
   ```python
   # Query historical KL divergence measurements
   metrics = client.query_metrics(
       metric_name="sentinel_inference_kl_divergence",
       start_time=datetime.utcnow() - timedelta(hours=24),
       step="1m"
   )
   ```

2. Compute the mean and standard deviation of baseline KL divergence.

3. Set the alert threshold to `mean + 5*stddev` for conservative detection, or `mean + 3*stddev` for more sensitive detection.

4. Configure in `config/thresholds/inference_thresholds.yaml`:
   ```yaml
   kl_divergence:
     warning_threshold: 0.05   # Example: mean + 3*stddev from your baseline
     critical_threshold: 0.15  # Example: mean + 5*stddev from your baseline
   ```

---

## Bayesian Prior Selection

The Bayesian attribution model in the correlation engine uses a Beta distribution prior to represent the initial belief about each GPU's reliability. Configured in `sentinel.yaml` under `correlation_engine.bayesian_model`.

### Prior Alpha and Beta

The prior `Beta(alpha, beta)` encodes:
- **alpha:** Effective number of prior successes (how many successful probes the GPU is "credited with" before any real data arrives).
- **beta:** Effective number of prior failures.

| Prior | Mean | Interpretation | Effect |
|-------|------|---------------|--------|
| Beta(10, 1) | 0.909 | Weak prior. GPU reaches SUSPECT after ~5 failures. | Fast detection but more false quarantines. |
| Beta(50, 1) | 0.980 | Moderate prior. GPU reaches SUSPECT after ~25 failures. | Balanced. |
| Beta(100, 1) (default) | 0.990 | Strong prior. GPU reaches SUSPECT after ~50 failures. | Conservative. Good for large fleets. |
| Beta(500, 1) | 0.998 | Very strong prior. GPU reaches SUSPECT after ~250 failures. | Very conservative. Slow detection. |

**How to think about it:** With `Beta(100, 1)`, a GPU starts with a reliability score of `100/101 = 0.990`. Each probe failure adds 1 to beta (and each success adds 1 to alpha). After `N` failures (and `M` successes), the score is `(100+M) / (101+M+N)`.

For the score to drop below 0.95 (SUSPECT threshold):
```
(100 + M) / (101 + M + N) < 0.95
N > 0.05 * (101 + M) / 0.95 - 0 ~ 5.3 + 0.053*M
```

With a 60-second probe cycle, after 1 hour (60 probes), `M = 60` if all pass. If 9 probes fail and 51 pass: score = `151/161 = 0.938 < 0.95` -> SUSPECT.

**Tuning guidance:**
- For fleets with < 100 GPUs, use a weaker prior (alpha=50) so that real SDC is detected quickly.
- For fleets with > 1000 GPUs, use a stronger prior (alpha=100-200) to avoid the base rate of transient faults causing quarantine churn.

### Weight Factors

| Event Type | Default Weight | Rationale |
|-----------|---------------|-----------|
| Probe success | 1.0 (alpha) | Direct, reliable evidence of correct computation. |
| Probe failure | 1.0 (beta) | Direct, reliable evidence of incorrect computation. |
| Anomaly event (inference/training) | 0.3 (beta) | Indirect evidence; could have non-SDC explanations. |
| TMR dissent | 2.0 (beta) | Strong evidence; GPU disagreed with two independent peers. |

Adjust weights if:
- Your anomaly detector has a high false positive rate: reduce anomaly weight to 0.1.
- Your anomaly detector is highly reliable (e.g., cross-rank divergence): increase anomaly weight to 0.5-1.0.
- TMR results are noisy (rare, should not happen): reduce TMR weight to 1.0.

### Time Decay Half-Life

The `decay_half_life_hours` setting (default 168 hours / 7 days) controls how quickly old observations lose influence on the reliability score. Old observations are scaled by `2^(-age / half_life)`.

| Half-Life | Effect |
|----------|--------|
| 24 hours | Aggressive decay. GPU recovers quickly from transient faults. But persistent intermittent faults may never accumulate enough evidence. |
| 168 hours (default) | Moderate decay. A week's evidence matters. Good balance for most fleets. |
| 720 hours (30 days) | Slow decay. Evidence accumulates over a month. Better for catching rare intermittent faults but slow to recover from transient issues. |

**Tuning guidance:** If your fleet has GPUs that are frequently cleared and re-quarantined (yo-yo effect), increase the half-life to retain more historical evidence. If you see GPUs stuck in SUSPECT for weeks after a single transient event, decrease the half-life.

---

## Quarantine Thresholds

Quarantine thresholds are configured in `sentinel.yaml` under `correlation_engine.state_machine`.

### Threshold Summary

| Threshold | Default | Description |
|----------|---------|-------------|
| `healthy_to_suspect_threshold` | 0.95 | Reliability score below which a GPU becomes SUSPECT. |
| `suspect_to_quarantine_threshold` | 0.80 | Reliability score below which a GPU is QUARANTINED. |
| `immediate_quarantine_failures` | 3 | Number of consecutive probe failures that triggers immediate quarantine (bypassing SUSPECT). |
| `quarantine_to_healthy_threshold` | 0.98 | Reliability score required for reinstatement. |
| `reinstatement_pass_count` | 1000 | Consecutive probe passes required for reinstatement. |
| `max_quarantine_hours` | 720 | Hours before quarantine converts to CONDEMNED. |

### Tuning Rationale

**SUSPECT threshold (0.95):** This threshold determines how early SENTINEL begins increased monitoring. A lower value (e.g., 0.90) means GPUs are given more benefit of the doubt, while a higher value (e.g., 0.98) means GPUs are flagged earlier.

**Quarantine threshold (0.80):** This is the production-impact threshold. A GPU below this score is removed from workloads. Setting this too high (e.g., 0.90) causes GPUs to be pulled too aggressively. Setting it too low (e.g., 0.60) means GPUs continue serving workloads despite substantial evidence of unreliability.

**Immediate quarantine (3 consecutive failures):** This bypasses the gradual Bayesian decay for obvious failure patterns. Three consecutive failures of the same probe type is extremely unlikely to be coincidence (probability of a false positive: roughly `p_fp^3`, where `p_fp` is the per-probe false positive rate -- vanishingly small if probes are well-calibrated).

**Reinstatement threshold (0.98):** This is deliberately higher than the suspect threshold (0.95) to create a hysteresis gap. A GPU that was quarantined at 0.80 must improve to 0.98 before reinstatement, preventing oscillation.

**Reinstatement pass count (1000):** At 60-second probe cycles, 1000 consecutive passes takes about 16.7 hours. This ensures the GPU has been stable for a meaningful period before returning to production.

---

## TMR Scheduling Frequency

TMR (Triple Modular Redundancy) canary configuration is in `sentinel.yaml` under `correlation_engine.tmr`.

| Setting | Default | Description |
|---------|---------|-------------|
| `interval_seconds` | 600 | How often TMR canaries run. |
| `timeout_ms` | 30000 | Per-GPU timeout for TMR computation. |
| `selection_strategy` | `suspect_priority` | How GPU triples are selected. |

### Frequency Trade-offs

| Interval | TMR Overhead (per GPU, approximate) | Fleet Coverage Time |
|----------|-------------------------------------|-------------------|
| 60s | ~0.5% | All GPUs tested in minutes |
| 300s | ~0.1% | All GPUs tested in ~1 hour |
| 600s (default) | ~0.05% | All GPUs tested in ~2 hours |
| 3600s | ~0.008% | All GPUs tested in ~12 hours |

**Tuning guidance:** TMR is the most expensive detection mechanism (it consumes GPU time on three GPUs per canary). In overhead-constrained environments, increase the interval. For SUSPECT GPUs, the `suspect_priority` strategy automatically increases their TMR frequency regardless of the base interval.

---

## Overhead Budget Management

The probe agent's overhead budget is configured via `probe_agent.overhead_budget_pct` (default 2.0).

### Measuring Actual Overhead

```bash
# Check current overhead via Prometheus
curl -s http://<agent-metrics>:9092/metrics | grep sentinel_probe_overhead_pct

# Run a dedicated overhead measurement
python benchmarks/overhead_measurement/probe_overhead.py \
  --schedule config/probe_schedules/default.yaml \
  --duration 600
```

### Adjusting the Budget

| Budget | Impact | Suitable For |
|--------|--------|-------------|
| 0.5% | Minimal impact. Probes run infrequently. | Ultra-latency-sensitive inference (< 10ms P99 SLO). |
| 1.0% | Low impact. Standard probes run but with longer periods. | Latency-sensitive inference. |
| 2.0% (default) | Moderate. All standard probes run at default frequency. | Most production workloads. |
| 5.0% | Higher impact. Aggressive probe schedule with full SM coverage. | Validation periods, new GPU qualification. |

### What Happens When Budget Is Exceeded

If probes consume more than the budget:

1. The scheduler defers lower-priority probes (Memory and Shared Memory probes are deferred first).
2. If the budget is still exceeded, higher-priority probes (FMA, Tensor Core) are given longer periods.
3. The `sentinel_probe_overhead_pct` metric exceeds the configured budget, and the `SentinelProbeOverheadExceeded` alert fires.
4. Probes are never fully stopped -- a minimum detection cadence is maintained even under budget pressure.

---

## Calibration Methodology

### Step 1: Establish Fleet Baseline

Before tuning thresholds, collect baseline data from your fleet with all GPUs known to be healthy (or at least not known to be faulty):

```bash
# Deploy SENTINEL with default settings
# Run for at least 72 hours to collect sufficient baseline data

# Export baseline metrics
sentinel-tools export-baseline \
  --start "2026-03-01T00:00:00Z" \
  --end "2026-03-04T00:00:00Z" \
  --output baseline_report.json
```

The baseline report includes:
- Per-probe-type pass rate (should be 100% on healthy hardware).
- Per-probe-type execution time distribution.
- EWMA baseline mean and standard deviation.
- KL divergence baseline distribution.
- GPU reliability score distribution.

### Step 2: Verify Zero False Positive Rate on Probes

Probe false positives (probe reports failure on healthy hardware) should be **exactly zero** for exact-tolerance probes (FMA, Tensor Core, AES, Memory, Register File, Shared Memory). If you observe any failures:

1. Verify golden answers are correct for your GPU architecture.
2. Verify the probe implementation is compatible with your driver version.
3. Investigate whether the "false positive" is actually a real fault.

For ULP-based probes (Transcendental), verify that the observed maximum ULP deviation is within your configured tolerance.

### Step 3: Tune EWMA and KL Thresholds

Using baseline data from known-good hardware:

1. Compute the empirical distribution of EWMA deviations and KL divergence values.
2. Set thresholds such that the expected false positive rate per GPU per day is < 0.01% (approximately 1 false positive per 10,000 GPU-days, or 1 per day in a 10,000-GPU fleet).
3. Validate by running with the new thresholds for 1 week and confirming the actual false positive rate.

### Step 4: Tune Bayesian Parameters

Based on your fleet size and risk tolerance:

1. Choose a prior strength (see table in [Bayesian Prior Selection](#bayesian-prior-selection)).
2. Set quarantine thresholds.
3. Simulate the Bayesian model's response to various failure scenarios:

   ```python
   from sentinel_sdk.tools import simulate_bayesian

   # Simulate: what happens if a GPU fails 1 probe per hour?
   sim = simulate_bayesian(
       prior_alpha=100,
       prior_beta=1,
       probe_period_s=60,
       failure_rate=1/60,  # 1 failure per 60 probes
       duration_hours=24,
       decay_half_life_hours=168
   )
   print(f"Time to SUSPECT: {sim.time_to_suspect_hours:.1f} hours")
   print(f"Time to QUARANTINE: {sim.time_to_quarantine_hours:.1f} hours")
   ```

4. Ensure that the time-to-quarantine is acceptable for your risk tolerance.

### Step 5: Iterate

Calibration is not a one-time activity. Review and adjust settings:
- After adding new GPU models to the fleet.
- After NVIDIA driver updates.
- After SENTINEL version upgrades.
- After changes to inference models or training workloads.
- Quarterly, as part of operational review.

---

## Worked Example

### Scenario: 1000-GPU H100 Cluster Running LLM Inference

**Cluster details:**
- 1,000 NVIDIA H100 GPUs across 125 nodes (8 GPUs per node).
- Running a 70B-parameter LLM for inference.
- Average throughput: 50,000 requests/second across the cluster (50 req/s per GPU).
- Latency SLO: P99 < 200ms.
- Overhead budget: 1.5% of GPU time.

### Probe Configuration

Given the 1.5% overhead budget, we use a slightly reduced schedule:

```yaml
# config/probe_schedules/h100_inference.yaml
schedule:
  - type: FMA
    period_seconds: 60
    sm_coverage: 1.0
    priority: 1
    enabled: true
    timeout_ms: 500

  - type: TENSOR_CORE
    period_seconds: 120       # Increased from 60 to reduce overhead
    sm_coverage: 1.0
    priority: 1
    enabled: true
    timeout_ms: 1000

  - type: TRANSCENDENTAL
    period_seconds: 180       # Increased from 120
    sm_coverage: 0.5
    priority: 2
    enabled: true
    timeout_ms: 500

  - type: AES
    period_seconds: 600       # Increased from 300
    sm_coverage: 0.25
    priority: 3
    enabled: true
    timeout_ms: 2000

  - type: MEMORY
    period_seconds: 1200      # Increased from 600
    sm_coverage: 0.5          # Reduced from 1.0
    priority: 4
    enabled: true
    timeout_ms: 5000

  - type: REGISTER_FILE
    period_seconds: 180       # Increased from 120
    sm_coverage: 1.0
    priority: 2
    enabled: true
    timeout_ms: 500

  - type: SHARED_MEMORY
    period_seconds: 600       # Increased from 300
    sm_coverage: 0.25         # Reduced from 0.5
    priority: 3
    enabled: true
    timeout_ms: 1000
```

This schedule hits approximately 1.2% overhead, leaving headroom for measurement error.

### Inference Monitor Configuration

```yaml
inference_monitor:
  sampling_rate: 0.005   # 0.5% sampling (reduced from 1% due to high throughput)
                         # = 250 samples/second fleet-wide, 0.25/GPU/second

  ewma:
    alpha: 0.005          # Slow-tracking EWMA for stable LLM inference
    warmup_samples: 2000  # Takes ~13 minutes at 0.25 samples/GPU/s

  kl_divergence:
    reference_update_interval_s: 7200  # Update every 2 hours (model is stable)
    reference_sample_count: 20000      # More samples for accurate reference
    binning_strategy: "histogram"
    num_bins: 256

  entropy:
    window_size: 2000

  spectral:
    enabled: false  # Disabled to save overhead

  fingerprinting:
    algorithm: "xxhash"   # Fast fingerprint for TMR
    intermediate_layers: false
```

### Correlation Engine Configuration

```yaml
correlation_engine:
  bayesian_model:
    prior_alpha: 150.0     # Stronger prior for 1000-GPU fleet
    prior_beta: 1.0
    decay_half_life_hours: 168.0

  state_machine:
    healthy_to_suspect_threshold: 0.95
    suspect_to_quarantine_threshold: 0.80
    immediate_quarantine_failures: 3
    quarantine_to_healthy_threshold: 0.98
    reinstatement_pass_count: 1500    # ~25 hours of clean probes
    max_quarantine_hours: 720
    require_approval: false

  tmr:
    enabled: true
    interval_seconds: 900       # Every 15 minutes (reduced from 10 min for overhead)
    selection_strategy: "suspect_priority"

  correlation:
    window_seconds: 300
    min_confidence: 0.65
```

### Expected Detection Performance

With these settings:

| Scenario | Expected Time to Detection |
|----------|--------------------------|
| Single SM producing wrong FMA results (every probe) | ~5 minutes (suspect), ~30 minutes (quarantine) |
| Single SM producing intermittent wrong results (10% of probes) | ~1 hour (suspect), ~4 hours (quarantine) |
| Inference output drift detectable by EWMA | ~30 minutes (after warmup) |
| Cross-GPU divergence in TMR canary | Immediate (next TMR round, ~15 minutes worst case) |
| Rare intermittent fault (1 failure per day) | ~3-5 days (suspect), ~7-14 days (quarantine) |

### Expected False Positive Rate

| Source | Expected FP Rate |
|--------|-----------------|
| Probe failures on healthy hardware | ~0 (exact-tolerance probes should never FP) |
| EWMA anomalies (k=3.5, alpha=0.005) | ~0.05% per GPU per day |
| KL divergence anomalies (5-sigma threshold) | ~0.01% per GPU per day |
| Combined (correlation engine requires convergent evidence) | < 0.001% per GPU per day |

At 1,000 GPUs: expect approximately 0.01 false quarantine events per day, or about 1 false quarantine per 100 days. This is manageable with the deep test workflow (false quarantines pass deep test and are automatically reinstated).

### Resource Sizing

For this 1,000-GPU cluster:

| Component | Instances | CPU | Memory |
|-----------|----------|-----|--------|
| Probe Agent | 125 (1 per node) | 0.5 core per node | 256 MB per node |
| Correlation Engine | 2 replicas | 4 cores each | 4 GB each |
| Audit Ledger (writer) | 1 | 2 cores | 2 GB |
| Audit Ledger (readers) | 2 | 1 core each | 1 GB each |
| PostgreSQL | 1 | 4 cores | 8 GB |
| ScyllaDB | 3 nodes | 4 cores each | 8 GB each |
| Redis | 1 | 1 core | 2 GB |

Total SENTINEL infrastructure overhead: ~30 CPU cores, ~50 GB RAM. This is approximately 0.3% of a 1,000-GPU cluster's total compute capacity -- negligible relative to the value of detecting SDC.
