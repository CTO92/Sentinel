# SENTINEL Architecture Overview

> **Version:** 0.1.0-alpha | **Status:** Pre-release | **Last Updated:** 2026-03-24

## Table of Contents

1. [System Overview](#system-overview)
2. [Detection Layers](#detection-layers)
   - [Layer 1: Deterministic Computational Probes](#layer-1-deterministic-computational-probes)
   - [Layer 2: Statistical Output Monitoring](#layer-2-statistical-output-monitoring)
   - [Layer 3: Selective Triple Modular Redundancy (TMR)](#layer-3-selective-triple-modular-redundancy-tmr)
   - [Layer 4: Self-Test Firmware Integration](#layer-4-self-test-firmware-integration)
   - [Layer 5: End-to-End Invariant Checking](#layer-5-end-to-end-invariant-checking)
3. [System Components](#system-components)
   - [Probe Agent (C++/CUDA)](#probe-agent-ccuda)
   - [Inference Monitor (Python)](#inference-monitor-python)
   - [Training Monitor (Python)](#training-monitor-python)
   - [Correlation Engine (Rust)](#correlation-engine-rust)
   - [Audit Ledger (Rust)](#audit-ledger-rust)
4. [Data Flow](#data-flow)
5. [Scalability](#scalability)
6. [Security Model](#security-model)

---

## System Overview

### The Problem

Silent Data Corruption (SDC) occurs when hardware faults produce incorrect computation results without raising any detectable error signal. In GPU clusters running AI workloads, SDC can corrupt training runs spanning weeks of compute time, produce subtly wrong inference outputs, and propagate through model weights and checkpoints in ways that are extremely difficult to diagnose after the fact.

Traditional error-detection mechanisms -- ECC memory, parity checks, watchdog timers -- catch only a subset of hardware faults. "Mercurial cores" (Google, 2021) and "small-delay defects" can produce arithmetically wrong results that pass all hardware error checks. At fleet scale (thousands of GPUs running continuously under thermal stress), the probability of at least one GPU exhibiting SDC at any moment approaches certainty.

### What SENTINEL Does

SENTINEL is a multi-layered detection framework that continuously validates the computational integrity of every GPU in a cluster. It combines:

- **Active probing** -- deterministic micro-benchmarks that test every arithmetic unit, memory subsystem, and register file against known-good answers.
- **Passive monitoring** -- statistical analysis of production inference and training outputs to catch SDC that manifests as anomalous model behavior.
- **Cross-GPU validation** -- selective Triple Modular Redundancy (TMR) where the same computation is run on three GPUs and results are compared by majority vote.
- **Bayesian attribution** -- a fleet-wide correlation engine that fuses signals from all sources to compute a per-GPU reliability score and automatically quarantine suspect hardware.
- **Tamper-evident audit trail** -- a cryptographically chained ledger of every detection event, quarantine decision, and operator action for compliance and forensics.

### High-Level Architecture

```
+=========================================================================+
|                            GPU CLUSTER                                   |
|                                                                          |
|  +------------------+  +------------------+  +------------------+        |
|  |   GPU Node 1     |  |   GPU Node 2     |  |   GPU Node N     |        |
|  | +------+ +-----+ |  | +------+ +-----+ |  | +------+ +-----+ |        |
|  | |Probe | |Inf. | |  | |Probe | |Train| |  | |Probe | |Inf. | |        |
|  | |Agent | |Mon. | |  | |Agent | |Mon. | |  | |Agent | |Mon. | |        |
|  | +--+---+ +--+--+ |  | +--+---+ +--+--+ |  | +--+---+ +--+--+ |        |
|  +----|---------|----|  +----|---------|----|  +----|---------|----|        |
|       |         |            |         |            |         |            |
+-------|---------|------------|---------|------------|---------|------+     |
        |         |            |         |            |         |           |
   gRPC |  gRPC   |       gRPC |  gRPC   |       gRPC |  gRPC   |           |
   Stream  Stream          Stream  Stream          Stream  Stream            |
        |         |            |         |            |         |           |
    +---v---------v------------v---------v------------v---------v---+      |
    |                  CORRELATION ENGINE (Rust)                     |      |
    |                                                                |      |
    |  +------------------+  +------------------+  +--------------+  |      |
    |  | Bayesian         |  | Temporal         |  | Trust        |  |      |
    |  | Attribution      |  | Correlation      |  | Graph &      |  |      |
    |  | Model            |  | Windows          |  | TMR          |  |      |
    |  +------------------+  +------------------+  +--------------+  |      |
    |  +------------------+  +------------------+  +--------------+  |      |
    |  | Quarantine       |  | Pattern          |  | Alert        |  |      |
    |  | State Machine    |  | Matcher          |  | Manager      |  |      |
    |  +--------+---------+  +------------------+  +--------------+  |      |
    +-----------|----------------------------------------------------+      |
                |                                                           |
    +-----------v--------------------------------------------------------+  |
    |                     AUDIT LEDGER (Rust)                              | |
    |  +------------------+  +------------------+  +--------------+       | |
    |  | Merkle Hash      |  | Batch            |  | Compliance   |       | |
    |  | Chain             |  | Processor        |  | Reports      |       | |
    |  +------------------+  +------------------+  +--------------+       | |
    +---------------------------------------------------------------------+ |
                                                                            |
    +---------------------------------------------------------------------+ |
    |                     DATA STORES                                      | |
    |  +-----------+   +-----------+   +-----------+                       | |
    |  |PostgreSQL |   | ScyllaDB  |   |  Redis    |                       | |
    |  |State &    |   |Time-series|   |Cache &    |                       | |
    |  |Metadata   |   |& Audit   |   |Pub/Sub    |                       | |
    |  +-----------+   +-----------+   +-----------+                       | |
    +---------------------------------------------------------------------+ |
                                                                            |
    +---------------------------------------------------------------------+ |
    |                     DASHBOARD (React/TypeScript)                      | |
    |  Real-time fleet health | SDC event timeline | GPU drill-down        | |
    +---------------------------------------------------------------------+ |
+=========================================================================+
```

---

## Detection Layers

SENTINEL employs a defense-in-depth strategy with five complementary detection layers. Each layer targets a different manifestation of SDC and operates on a different timescale, providing overlapping coverage so that a fault missed by one layer is likely caught by another.

### Layer 1: Deterministic Computational Probes

**Purpose:** Directly test GPU arithmetic units, memory, and register files by running computations with known inputs and comparing outputs against pre-computed golden answers. This is the primary, most direct SDC detection mechanism.

**Principle:** If a GPU is producing correct arithmetic results for known inputs, it is overwhelmingly likely to be producing correct results for production workloads. Conversely, if a probe fails, the specific SM (Streaming Multiprocessor) that produced the wrong answer is identified, enabling fine-grained fault isolation.

#### Probe Types

| Probe | What It Tests | Tolerance | Default Period |
|-------|--------------|-----------|----------------|
| **FMA** | Fused multiply-add units on every SM | Exact (0 ULP) | 60s |
| **Tensor Core** | HMMA (half-precision matrix multiply-accumulate) units | Exact (0 ULP) | 60s |
| **Transcendental** | sin, cos, exp, log SFUs (special function units) | 1 ULP (FP32), 2 ULP (FP16/BF16) | 120s |
| **AES** | Integer ALU and bitwise logic via AES-128 encrypt/decrypt | Exact (0 ULP) | 300s |
| **Memory** | Global memory via walking-ones, walking-zeros, MATS+ patterns | Exact | 600s |
| **Register File** | Register file via known-pattern write/read/verify | Exact | 120s |
| **Shared Memory** | Shared memory banks via address-as-data patterns | Exact | 300s |

#### SM Pinning

Each probe kernel is launched with explicit SM affinity so that SENTINEL knows exactly which SM produced each result. This is implemented via the `sm_affinity.cu` module:

1. On startup the probe agent queries the GPU's SM count via `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)`.
2. For each probe execution, the scheduler selects the set of target SMs based on the `sm_coverage` parameter (1.0 = all SMs, 0.5 = half, rotating).
3. The probe kernel uses cooperative groups and thread block affinity hints to pin each thread block to a specific SM. On Ampere and later architectures, `cudaLaunchAttributePreferredClusterSize` and MIG partition awareness are used for stricter isolation.
4. Results are indexed by SM ID, so a failure report contains the exact SM that returned the wrong answer.

SM pinning enables SENTINEL to distinguish between a single faulty SM (localized defect) and GPU-wide corruption (systemic issue), which is critical for the correlation engine's attribution model.

#### Golden Answer Methodology

Golden answers are the reference outputs against which probe results are compared. They must be:

- **Computed at higher precision than the GPU** to ensure the reference is correct. The golden answer generator (`tools/golden-answer-generator/`) uses Python's `mpmath` library for arbitrary-precision arithmetic.
- **Architecture-aware:** Different GPU architectures (Ampere, Hopper, Blackwell) may use different internal rounding modes for certain operations. Golden answers are generated per-architecture family.
- **Verified:** After generation, golden answers are independently verified by running the same computation on multiple known-good GPUs and confirming bit-exact agreement.
- **Input selection:** Probe inputs are carefully chosen to avoid rounding ambiguity. For FMA probes, inputs are selected such that `fma(a, b, c)` produces an exactly representable result in the target precision, eliminating any possibility of a legitimate rounding difference being misinterpreted as corruption.

For probes with non-zero ULP tolerance (transcendentals), the golden answer file includes both the expected value and the acceptable ULP range. The comparison logic computes the ULP distance between the GPU result and the golden answer and flags only results outside the tolerance window.

#### Probe Scheduling

Probes are scheduled by a priority-based scheduler (`probe-agent/src/agent/scheduler.cpp`) that respects the configurable overhead budget (default: 2% of GPU time). The scheduler:

1. Maintains a priority queue ordered by each probe's next-due timestamp and priority level.
2. Before launching a probe, estimates its execution time based on historical measurements.
3. If launching the probe would exceed the overhead budget in the current measurement window, the probe is deferred to the next window.
4. CUDA streams are used with low priority (`stream_priority: -1`) so probe kernels yield to production workloads.
5. Optionally, CUDA graphs are pre-compiled for probe kernels to minimize launch overhead (CPU-side cost of kernel dispatch).

The overhead budget is enforced using a sliding-window counter of probe kernel execution time divided by wall-clock time. If the budget is exceeded, the scheduler backs off exponentially until the ratio falls below budget.

### Layer 2: Statistical Output Monitoring

**Purpose:** Detect SDC that manifests as anomalous inference or training outputs, even when probes pass. This catches faults that are workload-dependent -- exercised only by specific input patterns or data-dependent control flow that probes do not cover.

#### EWMA Control Charts

The Exponentially Weighted Moving Average (EWMA) tracks key output statistics (logit means, variances, top-k token probabilities) over time. For each tracked statistic:

```
EWMA_t = alpha * x_t + (1 - alpha) * EWMA_{t-1}
```

Where `alpha` (default 0.01) controls the smoothing factor. A smaller alpha produces a smoother estimate that is slower to react to genuine distribution shifts, while a larger alpha is more responsive but more prone to false positives.

An anomaly is signaled when the current observation exceeds:

```
|x_t - EWMA_t| > k * sigma_EWMA
```

Where `k` (the sigma multiplier, default 3.5) controls sensitivity and `sigma_EWMA` is the EWMA-tracked standard deviation.

A warmup period (default 1000 samples) is enforced before anomaly detection activates, allowing the EWMA to converge to a stable baseline.

#### KL Divergence

The Kullback-Leibler divergence measures how the current output distribution has drifted from a reference distribution:

```
D_KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))
```

Where P is the current distribution and Q is the reference. The reference distribution is estimated from a configurable sample count (default 10,000 samples) and updated periodically (default every 3600 seconds).

A sustained elevation in KL divergence that is not explained by a known model update or data distribution shift may indicate SDC-induced drift.

The inference monitor supports two binning strategies for continuous distributions:
- **Histogram** (default): Fixed-width bins (default 256 bins) for logit or probability distributions.
- **KDE**: Kernel Density Estimation for smoother distribution estimates when sample counts are lower.

#### Entropy Monitoring

Output entropy (Shannon entropy of the softmax distribution) is tracked over a rolling window (default 1000 samples). SDC in certain compute paths can cause:
- **Entropy collapse**: Outputs become more confident (peaked distributions) due to corruption in normalization layers.
- **Entropy explosion**: Outputs become uniform due to corruption producing garbage logits.

Both directions are monitored. A sudden entropy shift that is not correlated with a model change is flagged as anomalous.

#### Spectral Analysis

When enabled (`spectral.enabled: true`), the inference monitor applies FFT-based spectral analysis to sequences of output statistics. This detects periodic or quasi-periodic corruption patterns that may indicate a clock-domain fault or a defect that activates only under specific thermal/voltage conditions.

An FFT window of 1024 samples (configurable, must be power of 2) is used. Anomalous spectral peaks -- frequencies that appear in the output statistics but have no corresponding frequency in the input distribution -- are flagged.

This detector has higher overhead than the others and is disabled by default.

#### Kolmogorov-Smirnov Tests

Two-sample KS tests compare the distribution of outputs from a specific GPU against the fleet-wide distribution. A statistically significant difference (after Bonferroni correction for multiple comparisons across the fleet) suggests that one GPU is computing differently from its peers.

#### Sampling Strategy

Monitoring every inference request or training step would impose unacceptable overhead. The inference monitor uses configurable sampling:

- **Inference:** Default 1% sampling rate (`sampling_rate: 0.01`). For a server processing 10,000 requests/second, this means ~100 samples/second -- sufficient for statistical power while adding minimal latency.
- **Training:** Default 10% sampling rate (`sampling_rate: 0.1`). Training steps are less frequent but more expensive, so a higher sampling rate is feasible.

The sampling decision is made at the interceptor level before any analysis, so unsampled requests incur zero monitoring overhead.

### Layer 3: Selective Triple Modular Redundancy (TMR)

**Purpose:** Provide definitive, ground-truth validation by running the same computation on three GPUs and comparing results via majority vote. TMR is the gold standard for fault detection but is expensive (3x compute cost), so SENTINEL uses it selectively.

#### Canary Batches

Rather than running TMR on all production traffic, SENTINEL periodically selects "canary batches" -- representative inference requests or training micro-steps -- and replicates them across three GPUs.

The TMR scheduler (`correlation-engine/src/trust/tmr_scheduler.rs`) selects GPU triples using one of three strategies:

- **`round_robin`**: Systematic rotation through all GPUs ensures every GPU is tested over time.
- **`random`**: Random GPU triple selection provides unbiased coverage.
- **`suspect_priority`** (default): GPUs with lower reliability scores are tested more frequently. A GPU in SUSPECT state might be included in every TMR round, while a HEALTHY GPU with a perfect record is tested only occasionally.

TMR canaries run at a configurable interval (default every 600 seconds) with a per-GPU timeout of 30 seconds.

#### Voting

When a TMR canary completes:

1. Results from all three GPUs are collected.
2. If all three agree (bit-exact after accounting for tolerances), the canary passes.
3. If two agree and one dissents, the dissenting GPU is flagged. The dissent event is a strong signal and directly updates the Bayesian model with a weighted failure.
4. If all three disagree, the canary is inconclusive and logged for manual review.

Voting operates on cryptographic fingerprints (xxHash or SHA-256) of the output tensors, not on the raw tensor data, to minimize network bandwidth.

#### Rotating Trust Graph

The trust graph (`correlation-engine/src/trust/trust_graph.rs`) tracks pairwise agreement history between GPUs. Each edge in the graph represents the number of times two GPUs agreed (or disagreed) in TMR rounds.

Over time, a GPU with a hardware defect will accumulate disagreements with many partners, while healthy GPUs will consistently agree with each other. The trust graph provides a secondary signal to the Bayesian model: a GPU that disagrees with many different partners is more likely to be faulty than one that disagreed with only one specific partner (which might indicate a fault in the other GPU).

Trust graph snapshots are persisted at configurable intervals (default every 3600 seconds) for historical analysis and forensics.

### Layer 4: Self-Test Firmware Integration

**Purpose:** Leverage GPU-internal diagnostic capabilities that are not accessible from normal CUDA kernels. This layer operates below the driver level.

#### Built-In Self-Test (BIST)

SENTINEL's probe agent can trigger NVIDIA's GPU self-test utilities (where available through the NVML API) to run hardware-level diagnostics. These tests exercise internal GPU components (memory controllers, NVLink transceivers, PCIe interfaces) that are not reachable from user-mode CUDA kernels.

BIST is typically run only during deep-test phases (when a GPU is already quarantined) because it is destructive to running workloads.

#### Thermal Stress Monitoring

The telemetry subsystem (`probe-agent/src/telemetry/thermal_monitor.cpp`) continuously monitors GPU junction temperature, hotspot temperature, and thermal throttling events via NVML.

Thermal data is correlated with probe failures to detect thermally-activated defects -- faults that manifest only when the GPU exceeds a certain temperature. This is a common failure mode for small-delay defects: timing margins shrink at higher temperatures, and a path that meets timing at 60C may fail at 85C.

#### Voltage Margin Monitoring

Voltage monitoring (`probe-agent/src/telemetry/voltage_monitor.cpp`) tracks GPU core voltage, memory voltage, and any undervoltage/overvoltage events. Like thermal monitoring, voltage data is correlated with probe failures.

Some SDC-inducing defects are voltage-marginal: the path works at nominal voltage but fails when voltage droops under heavy load (di/dt events). By correlating probe failures with voltage telemetry, SENTINEL can identify voltage-sensitive faults.

### Layer 5: End-to-End Invariant Checking

**Purpose:** Detect data corruption that occurs anywhere in the pipeline, not just within the GPU arithmetic units. This includes corruption during data movement (PCIe, NVLink, host memory), storage I/O, and checkpoint serialization.

#### Conservation Laws

For workloads with known mathematical invariants, SENTINEL can verify those invariants at runtime:

- **Softmax normalization:** Output probabilities must sum to 1.0 (within floating-point tolerance). A softmax output that sums to 0.97 or 1.15 indicates corruption in the normalization path.
- **Attention weight conservation:** In transformer architectures, attention weights for each query should sum to 1.0 across all keys.
- **Gradient norm consistency:** In data-parallel training, gradient norms across ranks should be approximately equal after all-reduce. A rank whose gradient norm diverges significantly from peers is suspect.
- **Loss function monotonicity:** While training loss is not strictly monotonic, sustained anomalous loss trajectories (spikes that do not correlate with learning rate schedules or data distribution changes) may indicate corruption.

Invariant violations are reported as high-severity anomaly events to the correlation engine.

#### Cryptographic Checksums

End-to-end data integrity is verified by computing cryptographic checksums at critical pipeline stages:

- **Checkpoint integrity:** Training checkpoints are hashed (BLAKE3 or SHA-256) at write time and verified at read time. The training monitor supports both `verify_on_save` and `verify_on_load` modes.
- **Tensor fingerprinting:** Output tensors can be fingerprinted using xxHash (fast, non-cryptographic) or SHA-256 (slower, cryptographic) for TMR comparison and historical anomaly tracking.
- **Weight integrity:** Model weights loaded for inference can be verified against a known-good hash to detect bit-flips during model loading or GPU-to-GPU transfer.

#### Storage Integrity

The audit ledger provides storage-level integrity via its Merkle hash chain (see [Audit Ledger](#audit-ledger-rust) below). Additionally, the training monitor validates checkpoint integrity across the entire storage path: GPU memory -> host memory -> filesystem/object storage -> load -> GPU memory.

---

## System Components

### Probe Agent (C++/CUDA)

**Location:** `probe-agent/`

The probe agent is a C++/CUDA binary that runs on every GPU node. It is responsible for executing probe kernels, collecting GPU telemetry, and streaming results to the correlation engine.

#### Threading Model

```
+---------------------------------------------------------------+
|  Probe Agent Process                                          |
|                                                               |
|  +------------------+     +------------------+                |
|  |  Main Thread     |     |  gRPC Client     |                |
|  |  - Config load   |     |  Thread          |                |
|  |  - Scheduler     |     |  - Batching      |                |
|  |  - Signal        |     |  - Streaming     |                |
|  |    handling      |     |  - Reconnect     |                |
|  +--------+---------+     +--------+---------+                |
|           |                        ^                          |
|           |  schedule              | results                  |
|           v                        |                          |
|  +--------+---------+     +--------+---------+                |
|  |  Probe Worker    |     |  Telemetry       |                |
|  |  Thread(s)       |     |  Collector       |                |
|  |  - CUDA stream   |     |  Thread          |                |
|  |    per GPU       |     |  - NVML polling  |                |
|  |  - Kernel launch |     |  - NVLink stats  |                |
|  |  - Result verify |     |  - PCIe stats    |                |
|  +------------------+     +------------------+                |
+---------------------------------------------------------------+
```

- **Main thread:** Loads configuration, initializes the scheduler, handles POSIX signals (SIGTERM for graceful shutdown, SIGHUP for config reload).
- **Probe worker threads:** One per GPU on multi-GPU nodes. Each thread owns a low-priority CUDA stream and executes probe kernels sequentially according to the schedule. CUDA graphs are optionally used to reduce launch overhead.
- **Telemetry collector thread:** Polls NVML at a configurable interval (default 10 seconds) for temperature, power, ECC counters, NVLink/PCIe error counters, and retired page counts.
- **gRPC client thread:** Batches probe results (default batch size: 64, max flush interval: 1000ms) and streams them to the correlation engine over a persistent gRPC connection. Handles reconnection with exponential backoff if the connection drops.

#### Probe Scheduling

The scheduler maintains a min-heap of probes ordered by their next-due timestamp. At each tick:

1. Pop the next due probe.
2. Check the overhead budget. If launching this probe would exceed the budget in the current 1-second measurement window, defer it by one period.
3. Launch the probe kernel on the appropriate CUDA stream with SM affinity.
4. Wait for completion (or timeout).
5. Compare results against golden answers using the configured tolerance mode.
6. Package the result (pass/fail, per-SM results, timing, telemetry snapshot) and enqueue it for the gRPC client thread.
7. Re-insert the probe into the heap with `next_due = now + period`.

#### Telemetry Collection

In addition to probe results, the agent collects GPU telemetry that is correlated with probe outcomes by the correlation engine:

- **Thermal:** Junction temperature, hotspot delta, throttle state, fan speed.
- **Power:** Instantaneous power draw, voltage, current.
- **Memory:** ECC corrected/uncorrected error counts, retired page counts.
- **Interconnect:** NVLink CRC error counts, PCIe replay counts.
- **Utilization:** SM utilization, memory bandwidth utilization (to contextualize overhead measurements).

### Inference Monitor (Python)

**Location:** `inference-monitor/`

The inference monitor is a Python library that integrates with inference serving frameworks to sample and analyze model outputs for SDC anomalies.

#### Interceptor Architecture

The inference monitor uses a pluggable interceptor pattern to support multiple inference frameworks:

```
+------------------------------------------------------------------+
|  Inference Server Process                                        |
|                                                                  |
|  +--------------------+                                          |
|  | Framework          |       +----------------------------+     |
|  | (vLLM, TRT-LLM,   | ----> | Interceptor                |     |
|  |  Triton, Generic)  |       | - Sample decision          |     |
|  +--------------------+       | - Tensor capture           |     |
|                               +-------------+--------------+     |
|                                             |                    |
|                               +-------------v--------------+     |
|                               | Analyzer Pipeline          |     |
|                               | +------------------------+ |     |
|                               | | Logit Analyzer (EWMA)  | |     |
|                               | +------------------------+ |     |
|                               | | KL Divergence          | |     |
|                               | +------------------------+ |     |
|                               | | Entropy Analyzer       | |     |
|                               | +------------------------+ |     |
|                               | | Spectral Analyzer      | |     |
|                               | +------------------------+ |     |
|                               | | Statistical Tests (KS) | |     |
|                               | +------------------------+ |     |
|                               +-------------+--------------+     |
|                                             |                    |
|                               +-------------v--------------+     |
|                               | gRPC Client (batched)      |     |
|                               +----------------------------+     |
+------------------------------------------------------------------+
```

Available interceptors:

| Interceptor | Framework | Integration Method |
|-------------|-----------|-------------------|
| `vllm_interceptor.py` | vLLM | Monkey-patches the sampler output path |
| `trtllm_interceptor.py` | TensorRT-LLM | Hooks into the TRT-LLM Python runtime |
| `triton_interceptor.py` | Triton Inference Server | gRPC/HTTP model wrapper |
| `generic_interceptor.py` | Any framework | Wraps model `__call__` or `forward` |

Each interceptor:

1. Decides whether to sample the current request (based on `sampling_rate`).
2. If sampling, captures the output tensors (logits, probabilities, or generated tokens).
3. Computes a fingerprint (xxHash or SHA-256) for TMR comparison.
4. Passes the captured data to the analyzer pipeline.

#### Analyzer Pipeline

Analyzers run sequentially on each sampled output. Each analyzer maintains its own state (running statistics, reference distributions) and independently decides whether the current sample is anomalous.

Anomaly events from any analyzer are batched (default batch size: 32, flush interval: 500ms) and streamed to the correlation engine. Each event includes:
- The anomaly type and severity.
- The GPU UUID that produced the output.
- The analyzer-specific evidence (e.g., EWMA deviation magnitude, KL divergence value).
- A timestamp and request identifier.

### Training Monitor (Python)

**Location:** `training-monitor/`

The training monitor is a Python library that hooks into training frameworks (PyTorch and JAX) to monitor training health and detect SDC-induced anomalies.

#### PyTorch Integration

```python
import sentinel_training

# Wrap your model
monitor = sentinel_training.pytorch.TrainingMonitor(config_path="sentinel.yaml")
monitor.attach(model, optimizer)

# Training loop proceeds normally -- hooks fire automatically
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

The PyTorch integration uses the following hooks:

| Hook | What It Monitors |
|------|-----------------|
| `hooks.py` | Registers `register_full_backward_hook` on all parameters to intercept gradient tensors |
| `gradient_monitor.py` | Tracks per-parameter and global gradient L2 norms; flags spikes > k*sigma above rolling mean |
| `loss_monitor.py` | Tracks loss trajectory; detects spikes accounting for learning rate schedule |
| `ddp_divergence.py` | Compares gradient norms across DDP ranks via NCCL/Gloo all-reduce; flags divergence > threshold |
| `checkpoint_validator.py` | Hashes checkpoints at save/load time using BLAKE3 or SHA-256 |

#### JAX Integration

| Module | What It Monitors |
|--------|-----------------|
| `gradient_monitor.py` | Monitors gradient norms via JAX's `jax.grad` wrapper |
| `pjit_monitor.py` | Hooks into `pjit`-compiled functions to monitor sharded computation |
| `transforms.py` | Custom JAX transforms for output fingerprinting |

#### DDP Divergence Detection

In data-parallel training, all ranks should compute approximately the same gradient norms (they process different data but the model is synchronized via all-reduce). The cross-rank monitor:

1. At configurable intervals (default every 10 steps), each rank broadcasts its local gradient L2 norm.
2. The monitor computes the max relative divergence: `max(|norm_i - median|) / median`.
3. If divergence exceeds the threshold (default 5%), an anomaly event is raised identifying the outlier rank and its associated GPU.

This is one of the most powerful SDC detection signals for training workloads: a single corrupted GPU in a DDP group will produce different gradients from all peers, and this divergence is measurable even when the corruption is small.

### Correlation Engine (Rust)

**Location:** `correlation-engine/`

The correlation engine is the central intelligence of SENTINEL. It ingests events from all probe agents and monitors, applies statistical and causal analysis, maintains per-GPU reliability scores, and makes quarantine decisions.

#### Bayesian Attribution Model

Each GPU maintains a Bayesian belief state modeled as a Beta distribution:

```
P(GPU is reliable | evidence) ~ Beta(alpha, beta)
```

Where:
- `alpha` = prior_alpha + number of successful probes (weighted)
- `beta` = prior_beta + number of failures and anomalies (weighted)

The **reliability score** is the mean of the Beta distribution:

```
reliability = alpha / (alpha + beta)
```

The **confidence interval** is computed from the Beta CDF. The 95% lower bound of the reliability score is used for quarantine decisions, ensuring that GPUs are not quarantined due to statistical noise when few observations are available.

**Prior selection:** The default prior is `Beta(100, 1)`, which encodes a strong prior belief that GPUs are reliable. This requires substantial evidence of failure before the reliability score drops significantly. The prior is configurable to match fleet-specific base rates.

**Evidence weighting:**
- A probe failure contributes weight **1.0** to beta (strong, direct evidence).
- An anomaly event from the inference or training monitor contributes weight **0.3** to beta (weaker, indirect evidence that could have non-SDC explanations).
- A successful probe contributes weight **1.0** to alpha.
- A TMR dissent contributes weight **2.0** to beta (very strong evidence -- the GPU disagreed with two independent peers).

**Time decay:** Old observations decay with a configurable half-life (default 168 hours / 7 days). This allows a GPU that experienced a transient issue to recover its reliability score over time, while persistent defects continue to accumulate evidence. Decay is implemented by scaling historical alpha/beta contributions by `2^(-t/half_life)` where `t` is the age of the observation.

**Per-SM Granularity:** The model optionally tracks per-SM beliefs within each GPU, enabling identification of individual faulty SMs. This is used in the probe scheduling feedback loop: an SM with a lower reliability score is probed more frequently.

#### Quarantine State Machine

```
                    +------------------+
                    |                  |
                    |     HEALTHY      |<---------+
                    |  score >= 0.95   |          |
                    +--------+---------+          |
                             |                    |
                score < 0.95 |                    | score >= 0.98
                     or      |                    | AND 1000
                  anomalies  |                    | consecutive
                             |                    | passes
                    +--------v---------+          |
                    |                  |          |
                    |     SUSPECT      +----------+
                    |  0.80 <= score   |  (cleared)
                    +--------+---------+
                             |
                 score < 0.80|
                     or      |
            3 consecutive    |
               failures      |
                             |
                    +--------v---------+
                    |                  |
                    |   QUARANTINED    +----------+
                    |  removed from    |          |
                    |  production      |          | passed
                    +--------+---------+          | deep test
                             |                    |
              auto or        |          +---------+---------+
              manual         |          |                   |
              trigger        +--------->+    DEEP_TEST      |
                             |          |   BIST + stress   |
                             |          +---------+---------+
                             |                    |
                     max     |                    | failed
                   quarantine|                    | deep test
                     time    |                    |
                   exceeded  |          +---------v---------+
                             +--------->+                   |
                                        |    CONDEMNED      |
                                        |  permanent        |
                                        |  removal          |
                                        +-------------------+
```

**State transitions:**

| From | To | Trigger | Reversible |
|------|----|---------|-----------|
| HEALTHY | SUSPECT | Reliability score drops below 0.95, or multiple anomaly events in correlation window | Yes |
| SUSPECT | HEALTHY | Reliability score recovers above 0.98 AND 1000 consecutive probe passes | Yes |
| SUSPECT | QUARANTINED | Reliability score drops below 0.80, or 3 consecutive probe failures | Yes |
| QUARANTINED | DEEP_TEST | Automatic (scheduled) or manual operator trigger | Yes |
| QUARANTINED | CONDEMNED | Maximum quarantine time exceeded (default 720 hours / 30 days) | No |
| DEEP_TEST | HEALTHY | Deep test suite passes; reliability score reset to prior | Yes |
| DEEP_TEST | CONDEMNED | Deep test suite fails | No |
| CONDEMNED | (none) | Terminal state; requires hardware replacement | No |

All transitions are logged to the audit ledger with the triggering evidence, operator identity (if manual), and timestamp.

**Approval gates:** The `require_approval` setting (default `false`) can require human approval for quarantine and reinstatement actions. When enabled, the state machine enters a pending state and emits an approval-request alert. The operator must approve via the dashboard or SDK before the transition completes.

#### Temporal Windowing

The correlation engine uses configurable time windows (default 300 seconds) to correlate events from different sources. When multiple events arrive for the same GPU within a window, they are analyzed together:

- A probe failure AND an inference anomaly within the same window is a much stronger signal than either alone.
- Multiple probe failures across different probe types (e.g., FMA and Tensor Core) within a window suggest a systemic GPU issue rather than a single-unit defect.
- Probe failures across multiple GPUs on the same node within a window may indicate a node-level issue (power supply, cooling, PCIe bus).

The correlation buffer holds up to 100,000 events (configurable). Events older than the correlation window are flushed to storage and removed from the active buffer.

#### Pattern Matching

The pattern matcher (`correlation-engine/src/correlation/pattern_matcher.rs`) identifies fleet-wide failure patterns:

- **Node-correlated failures:** Multiple GPUs on the same node failing simultaneously, suggesting a shared-infrastructure issue.
- **Topology-correlated failures:** GPUs connected via the same NVLink switch or PCIe root complex failing together.
- **Temporal clustering:** A burst of failures across multiple GPUs fleet-wide, suggesting a software bug (driver, firmware) rather than hardware SDC.
- **Probe-specific patterns:** A single probe type failing across many GPUs, suggesting a golden answer error or probe bug rather than real SDC.

Pattern detection is critical for avoiding false-positive cascades where a software issue triggers fleet-wide quarantines.

### Audit Ledger (Rust)

**Location:** `audit-ledger/`

The audit ledger is a tamper-evident, append-only log of every significant event in the SENTINEL system. It provides the compliance and forensics foundation.

#### Merkle Hash Chain

Every audit entry is cryptographically chained to its predecessor:

```
Entry_N.hash = SHA-256( Entry_N.data || Entry_{N-1}.hash )
```

This means that modifying any historical entry would change its hash, which would break the chain for all subsequent entries. Verification requires only walking the chain forward and recomputing hashes.

The first entry in the chain uses a zero hash (`0x0000...0000`) as its predecessor.

#### Batch Processing

For efficiency, entries are grouped into batches (default batch size: 1024 entries, flush interval: 60 seconds):

1. Pending entries are sorted by timestamp for deterministic ordering.
2. Each entry is chained to its predecessor (sequential hashing within the batch).
3. A Merkle tree is computed over all entry hashes in the batch.
4. The Merkle root is stored alongside the batch metadata.

The Merkle tree enables efficient integrity proofs: to prove that a specific entry exists and is unmodified, only `O(log N)` hashes are needed rather than the entire chain.

#### Storage

The audit ledger supports two storage backends:

- **ScyllaDB** (default): Used for the time-series entry data. Provides horizontal scalability and configurable replication (default RF=3) with tunable consistency (QUORUM writes, QUORUM reads for audit queries).
- **PostgreSQL**: Used for metadata, batch summaries, and Merkle roots. Supports table partitioning for efficient retention management.

#### PostgreSQL Partitioning

The audit metadata tables are partitioned by time range (`audit-ledger/src/storage/migrations/003_partitioning.sql`):

- Active partition: current month.
- Historical partitions: one per month.
- Partition pruning is automatic when retention limits are reached.

The `detail_retention_days` setting (default 365 days) controls how long detailed entry data is kept. The `summary_retention_days` setting (default 0 = forever) controls Merkle root and batch summary retention. This separation allows operators to prune detailed data for storage efficiency while retaining the cryptographic proof chain indefinitely.

#### Compliance Reporting

The audit ledger includes built-in compliance report generators:

- **SOC 2 reports** (`audit-ledger/src/compliance/soc2_report.rs`): Generates evidence reports mapped to SOC 2 Trust Services Criteria.
- **ISO 27001 reports** (`audit-ledger/src/compliance/iso27001_report.rs`): Generates evidence reports mapped to ISO 27001 Annex A controls.

Reports can be generated on-demand via the gRPC query service or scheduled for periodic generation.

#### Automatic Verification

The ledger runs automatic chain integrity checks at configurable intervals (default every 6 hours). Each check verifies the most recent N entries (default 10,000) by recomputing hashes and comparing against stored values. Any discrepancy triggers a CRITICAL alert.

---

## Data Flow

### Event Ingestion

```
GPU Node                    Correlation Engine              Audit Ledger
--------                    ------------------              ------------

Probe Agent  -- gRPC stream (ProbeResultBatch) -->  ProbeService
                                                       |
                                                       v
                                                    EventStore (ScyllaDB)
                                                       |
                                                       v
                                                    BayesianAttribution
                                                       |
                                                       v
                                                    QuarantineStateMachine
                                                       |
                                                       v
                                                    AlertManager
                                                       |
                                                       +-- gRPC --> IngestService
                                                                         |
                                                                         v
                                                                    ChainBuilder
                                                                         |
                                                                         v
                                                                    ScyllaDB/PG

Inf. Monitor -- gRPC stream (AnomalyEventBatch) -->  AnomalyService
                                                       |
                                                       v
                                                    (same pipeline as above)

Train. Mon.  -- gRPC stream (AnomalyEventBatch) -->  AnomalyService
                                                       |
                                                       v
                                                    (same pipeline as above)
```

### gRPC Streaming

All agent-to-engine communication uses **bidirectional gRPC streaming**:

- **Agent -> Engine:** Probe results and anomaly events are streamed in batches. Batching reduces per-message overhead and network round-trips. Default batch sizes are 64 (probe agent) and 32 (monitors), with maximum flush intervals to ensure timely delivery even under low event rates.
- **Engine -> Agent:** Configuration updates are streamed via the `ConfigService`. When an operator changes a probe schedule or threshold via the dashboard, the update is pushed to all connected agents without requiring a restart.

The gRPC connections use persistent streams with keepalive pings (every 30 seconds) and automatic reconnection with exponential backoff (1s, 2s, 4s, 8s, max 60s).

### Event Batching

Events are batched at multiple levels:

1. **Source batching:** The probe agent and monitors batch events locally before sending (reduces gRPC call frequency).
2. **Correlation batching:** The correlation engine buffers events in the temporal correlation window before making attribution decisions (ensures events from multiple sources for the same GPU are analyzed together).
3. **Audit batching:** The audit ledger batches entries for efficient Merkle tree computation and storage writes (reduces per-entry overhead).

---

## Scalability

SENTINEL is designed to scale from small development clusters (a handful of GPUs) to large production fleets (10,000+ GPUs).

### Probe Agent Scaling

Probe agents scale trivially because they run independently on each node. There is no inter-agent communication. Each agent only communicates with the correlation engine.

- **Per-node overhead:** Probe agents are designed for < 2% GPU time overhead (configurable). CPU overhead is minimal (< 0.5 core per node).
- **Multi-GPU nodes:** A single probe agent process manages all GPUs on a node using one worker thread per GPU.

### Correlation Engine Scaling

The correlation engine is the central bottleneck. Scaling strategies:

- **Horizontal scaling:** Multiple correlation engine replicas behind a load balancer, with consistent hashing by GPU UUID to ensure all events for a given GPU land on the same replica (necessary for stateful Bayesian tracking).
- **HPA:** In Kubernetes, the correlation engine Deployment uses Horizontal Pod Autoscaler based on CPU utilization and gRPC request rate.
- **Partitioning:** For very large fleets, GPUs can be partitioned across multiple correlation engine clusters by region, rack, or logical group. Each partition is independently managed.

**Sizing estimates:**

| Fleet Size | Correlation Engine Replicas | CPU Cores | Memory |
|------------|---------------------------|-----------|--------|
| 64 GPUs | 1 | 2 | 2 GB |
| 256 GPUs | 1 | 4 | 4 GB |
| 1,000 GPUs | 2-3 | 4 each | 4 GB each |
| 4,000 GPUs | 4-6 | 4 each | 8 GB each |
| 10,000 GPUs | 8-12 | 8 each | 16 GB each |

### Data Store Scaling

- **ScyllaDB:** Scales horizontally by adding nodes to the cluster. Time-series data (probe results, anomaly events) is the highest-volume data and benefits from ScyllaDB's write-optimized architecture.
- **PostgreSQL:** Handles state and metadata queries. For fleets > 5,000 GPUs, consider read replicas for dashboard queries.
- **Redis:** Used for caching (current GPU states, active correlation windows) and pub/sub (real-time dashboard updates). A single Redis instance handles fleets up to ~5,000 GPUs; Redis Cluster for larger deployments.

### Audit Ledger Scaling

The audit ledger is designed as a **single-writer** system to maintain sequential hash chain integrity. Scaling strategies:

- **Read replicas:** Multiple read-only replicas serve compliance queries and dashboard requests.
- **Batch throughput:** The batch processing architecture (1024 entries per batch, 60-second flush interval) can sustain ~17 entries/second continuously, or burst to much higher rates with larger batches. At 10,000 GPUs with 7 probe types and 60-second probe intervals, the sustained event rate is approximately 1,167 events/second -- well within batch processing capacity with appropriately sized batches.
- **Storage partitioning:** Time-based partitioning in PostgreSQL and TTL-based expiry in ScyllaDB keep storage costs manageable.

### Network Bandwidth

Each probe result message is approximately 200 bytes. At fleet scale:

| Fleet Size | Probe Events/sec | Bandwidth (probe results only) |
|------------|------------------|-------------------------------|
| 64 GPUs | ~7 | ~1.5 KB/s |
| 1,000 GPUs | ~117 | ~23 KB/s |
| 10,000 GPUs | ~1,167 | ~230 KB/s |

Network bandwidth for SENTINEL telemetry is negligible relative to production GPU cluster traffic.

---

## Security Model

### Mutual TLS (mTLS)

All gRPC connections between SENTINEL components use mutual TLS by default:

- Each component has its own TLS certificate and private key.
- The CA certificate is shared across all components.
- Clients verify the server certificate; servers verify the client certificate.
- TLS 1.3 is required; earlier versions are rejected.

Certificate paths are configured in `sentinel.yaml` under each component's `tls` section.

### HMAC Signing

Probe results are HMAC-signed by the probe agent before transmission:

```
signature = HMAC-SHA256(probe_result_bytes, shared_secret)
```

The correlation engine verifies the HMAC before processing any probe result. This prevents an attacker from injecting false probe results to mask a compromised GPU or trigger false quarantines.

The HMAC key should be injected via a secrets manager (e.g., Kubernetes Secrets, HashiCorp Vault) rather than stored in configuration files. The `hmac_key` field in `sentinel.yaml` is intentionally left empty as a reminder.

### Role-Based Access Control (RBAC)

The SENTINEL dashboard and SDK enforce RBAC:

| Role | Permissions |
|------|------------|
| `viewer` | Read-only access to fleet health, event history, audit trail |
| `operator` | Viewer permissions + manual quarantine/reinstatement, threshold adjustment |
| `admin` | Operator permissions + configuration changes, RBAC management, compliance report generation |
| `auditor` | Read-only access to audit trail and compliance reports; cannot view or modify operational settings |

RBAC is enforced at the gRPC service level in the correlation engine and audit ledger.

### Kubernetes Network Policies

The SENTINEL Kubernetes deployment includes network policies (`deploy/kubernetes/network-policies.yaml`) that restrict inter-component communication:

- Probe agents can only communicate with the correlation engine (port 50051).
- The correlation engine can communicate with the audit ledger (port 50052), data stores (PostgreSQL 5432, ScyllaDB 9042, Redis 6379), and the metrics endpoint.
- The audit ledger can communicate with data stores only.
- The dashboard can communicate with the correlation engine and audit ledger query services only.
- No component can communicate with external networks except for configured alerting channels (Slack webhook, PagerDuty API, SMTP).

### Audit Trail Integrity

The Merkle hash chain in the audit ledger provides tamper evidence. Even if an attacker gains access to the database, modifying historical entries without detection requires recomputing the entire hash chain from the modified entry forward, and any external verifier holding a previous Merkle root would detect the tampering.

Periodic chain verification (default every 6 hours) provides automated detection of any integrity violations.
