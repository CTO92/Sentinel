# Silent Data Corruption: A Primer

> **Version:** 0.1.0-alpha | **Last Updated:** 2026-03-24

This document provides background on Silent Data Corruption (SDC) -- what it is, why it happens, why GPUs are especially vulnerable, how it impacts AI workloads, and how SENTINEL addresses each challenge.

## Table of Contents

1. [What Is Silent Data Corruption?](#what-is-silent-data-corruption)
2. [Root Causes](#root-causes)
3. [Published Research](#published-research)
4. [Why GPUs Are Especially Vulnerable](#why-gpus-are-especially-vulnerable)
5. [Impact on AI Workloads](#impact-on-ai-workloads)
6. [Detection Challenges](#detection-challenges)
7. [How SENTINEL Addresses Each Challenge](#how-sentinel-addresses-each-challenge)

---

## What Is Silent Data Corruption?

Silent Data Corruption (SDC) occurs when a hardware fault causes incorrect computation results without triggering any error signal. The hardware does not report an error. The software does not crash. The result is simply wrong.

Consider a simple example: a floating-point multiply-add unit on a GPU is asked to compute `1.0 * 1.0 + 0.0`. The correct answer is `1.0`. On a healthy unit, the answer is always `1.0`. On a unit with SDC, the answer might be `1.0000001` -- or `0.99999` -- or `1.5` -- or even `NaN`. The key point is that the unit completes the operation, returns a result, and reports success. No exception is raised. No error flag is set. No interrupt fires.

This is fundamentally different from detected errors:

| Error Type | Detected? | System Response | Impact |
|-----------|-----------|-----------------|--------|
| ECC-corrected memory error | Yes | Hardware corrects automatically | None (transparent) |
| ECC-uncorrectable memory error | Yes | Hardware raises error; process killed | Visible crash |
| GPU hang/timeout | Yes | Driver resets GPU; process killed | Visible crash |
| Silent Data Corruption | **No** | **Nothing** | **Wrong results accepted as correct** |

SDC is the most dangerous class of hardware fault precisely because the system operates normally from an external perspective. The wrong results propagate downstream without any indication that something went wrong.

---

## Root Causes

SDC is not caused by a single mechanism. Multiple physical phenomena can produce incorrect computation results without triggering error-detection circuitry.

### Small-Delay Defects

Modern semiconductor manufacturing is imperfect. Some transistors in a chip may switch slightly slower than designed due to lithographic imprecision, doping variations, or contamination. These "small-delay defects" do not cause the transistor to fail outright -- it still switches -- but it switches too slowly to meet the clock timing requirements.

If a logic path contains a small-delay defect, the output of that path may not settle to the correct value before the next clock edge samples it. The result is a wrong value captured into a register. Critically, the standard manufacturing test suite may not catch this defect because:

1. The path may only be slow under specific input patterns (data-dependent delay).
2. The path may only be slow at specific temperatures (temperature-dependent delay).
3. The path may only be slow at specific voltages (voltage-dependent delay).

A chip that passes all factory tests can still exhibit SDC in the field when operating conditions (temperature, voltage, workload pattern) expose the defect.

### Aging and Degradation

Semiconductor devices degrade over time through several physical mechanisms:

- **Bias Temperature Instability (BTI):** Prolonged voltage stress shifts transistor threshold voltages, making some transistors slower. This is accelerated by high temperature and continuous operation -- exactly the conditions in a GPU datacenter.
- **Hot Carrier Injection (HCI):** High-energy charge carriers damage the gate oxide, degrading transistor performance over thousands of hours of operation.
- **Electromigration:** Current flowing through metal interconnects gradually moves atoms, thinning wires and increasing resistance. Eventually, a wire may become slow enough to cause timing violations.

A GPU that was perfectly functional on day one may develop SDC after months or years of continuous operation as aging pushes transistor speeds below the timing margin.

### Intermittent Activation

Some defects are genuinely intermittent: they manifest only under specific conditions and may not be reproducible on demand. Causes include:

- **Marginal solder joints:** A microscopic crack in a solder connection may cause intermittent contact depending on mechanical stress and thermal expansion.
- **Particle contamination:** A conductive particle trapped near a circuit trace may cause intermittent shorts depending on vibration and orientation.
- **Power delivery noise:** Rapid current changes (di/dt events) during heavy computation cause voltage droops that can push marginal paths below their timing threshold.

Intermittent SDC is especially insidious because the defect may not manifest during diagnostic testing, leading operators to reinstate GPUs that then fail again in production.

### Fan-Out Amplification

When a single gate or register has a defect, the impact may extend far beyond that single value. If the corrupted value feeds into many downstream computations (high fan-out), a single bit error can corrupt an entire computation tree. In GPU arithmetic:

- A corrupted bit in the exponent of a floating-point number changes the magnitude of the result by a factor of 2 (or more).
- A corrupted accumulator register in a matrix multiply affects every subsequent accumulation in that row/column.
- A corrupted shared memory location may be read by all threads in a threadblock.

Fan-out amplification means that even a very small hardware defect can produce arbitrarily large errors in the final computation result.

---

## Published Research

SDC in large-scale compute infrastructure has been documented by several major organizations.

### Google: Mercurial Cores (2021)

In "Silent Data Corruptions at Scale" (Google, 2021), Hochschild et al. described **mercurial cores** -- CPU cores that intermittently produce incorrect computation results without raising any machine-check exception. Key findings:

- SDC was observed at a rate of approximately 1 in several thousand CPUs in Google's fleet.
- The corruptions were intermittent and workload-dependent, making them difficult to detect with standard hardware diagnostics.
- Some mercurial cores produced incorrect results only for specific instruction sequences under specific data patterns.
- Google developed a detection framework using deterministic computations with known outputs -- the same fundamental approach SENTINEL uses for GPU probes.

### Meta: Silent Data Corruption in LLM Training (2024)

Meta reported that during the training of Llama 3, silent hardware errors corrupted training runs multiple times. Specifics from the Llama 3 report:

- Over 54 days of pre-training using 16,384 H100 GPUs, Meta experienced **419 unexpected job interruptions**, of which a significant fraction were traced to hardware faults.
- Some faults were silent -- they did not crash the job but instead corrupted gradient computations, causing training loss to spike or diverge.
- Meta developed custom monitoring to detect these issues, including cross-rank gradient norm comparison (similar to SENTINEL's DDP divergence detection) and periodic checkpoint validation.
- The scale of the training run (16,384 GPUs operating continuously) meant that statistically, at least one GPU exhibiting some form of fault at any given time was near-certain.

### NVIDIA/OCP: GPU SDC Detection Whitepaper (2024)

NVIDIA, in collaboration with the Open Compute Project (OCP), published guidance on GPU SDC detection. Key points:

- NVIDIA acknowledges that SDC can occur in GPU compute units (SMs), memory subsystems, and interconnects (NVLink, PCIe).
- ECC protects memory contents at rest but does not protect data during computation (arithmetic logic, register file operations).
- The recommended detection approach combines periodic diagnostic probes (testing arithmetic units with known-good inputs) and application-level output validation.
- GPU self-test (BIST) capabilities can detect some SDC-inducing defects but are destructive to running workloads and cannot be run continuously.

### Stanford: Test Escapes in Advanced Nodes (2023)

Research at Stanford and in the semiconductor testing community has documented the phenomenon of "test escapes" -- defects that pass all manufacturing tests but cause failures in the field.

- At advanced process nodes (7nm, 5nm, and below), the proportion of test escapes is increasing due to the growing complexity of defect mechanisms and the limitations of test coverage.
- Small-delay defects are particularly prone to test escape because they are exercised only by specific path activation conditions that may not be covered by standard test patterns.
- The industry trend toward lower voltage margins (for power efficiency) further narrows the gap between functional and defective, increasing the SDC risk for chips that passed factory tests.

---

## Why GPUs Are Especially Vulnerable

While SDC can affect any processor, GPUs have several characteristics that make them particularly vulnerable compared to CPUs.

### Arithmetic Density

A modern GPU contains thousands of arithmetic units packed into a single die:

| Unit Type | Approximate Count (H100) |
|-----------|-------------------------|
| CUDA Cores (FP32) | 16,896 |
| Tensor Cores | 528 |
| SMs (Streaming Multiprocessors) | 132 |
| Register file entries | ~20 million |
| Shared memory cells | ~18 MB total |

Each of these units is a potential source of SDC. With ~17,000 arithmetic units, the probability that at least one unit has a latent defect is much higher than for a CPU with ~50 ALUs.

### Sustained Thermal Stress

GPUs in AI datacenters operate at or near their thermal limits for extended periods:

- Typical GPU junction temperature during AI workloads: 75-85C, often reaching thermal throttling thresholds.
- GPUs run at high utilization 24/7 for training runs lasting days to months.
- Continuous high-temperature operation accelerates aging mechanisms (BTI, HCI, electromigration), increasing the probability of SDC over the GPU's lifetime.

CPUs in the same datacenters typically operate at lower temperatures and more variable utilization, experiencing less thermal aging.

### Reduced Redundancy

CPU designs include significant redundancy and error-checking in their arithmetic pipelines:

- CPUs commonly use dual-rail or checker logic for critical ALU operations.
- CPUs include parity on register files and internal buses.
- CPUs implement microarchitectural replay mechanisms that can re-execute operations if intermediate errors are detected.

GPU architectures prioritize throughput over redundancy:

- GPU arithmetic units generally do not have checker logic (the area cost would be prohibitive given the unit count).
- GPU register files may have ECC on some architectures but not all, and not all register paths are covered.
- GPU shared memory may not have ECC on all architectures.

This design trade-off is rational -- the throughput gain from thousands of simple, unchecked units outweighs the reliability gain from fewer, checked units -- but it means SDC is more likely to occur and less likely to be detected internally.

### Fault Masking in AI Workloads

AI workloads have a paradoxical relationship with SDC: they are simultaneously **tolerant** of small errors (which makes them robust) and **vulnerable** to SDC propagation (which makes detection harder).

**Tolerance:** Neural networks are inherently noise-tolerant. A small perturbation to a single intermediate value typically has no visible effect on the model's output because:
- Subsequent layers perform averaging and normalization.
- Softmax/activation functions saturate, damping out-of-range values.
- The model has learned to be robust to input noise during training.

**Vulnerability:** This tolerance becomes a liability for SDC detection because:
- Small corruptions are absorbed into the computation without producing obviously wrong outputs.
- The corruption accumulates: during training, a corrupted gradient update modifies the model weights, and those corrupted weights persist in the checkpoint.
- Over many training steps, accumulated small corruptions can steer the model in subtly wrong directions without triggering loss spikes or other obvious signals.
- For inference, a corrupted output may be indistinguishable from a correct output for any individual request -- only statistical analysis over many requests reveals the drift.

---

## Impact on AI Workloads

### Training Corruption

SDC during training can corrupt model weights in ways that are extremely costly:

1. **Gradient corruption:** A faulty GPU computes an incorrect gradient that is averaged with correct gradients from other GPUs during all-reduce. The incorrect gradient shifts the model weights in the wrong direction. Because modern training uses momentum and adaptive learning rates, the corrupted gradient is amplified over subsequent steps.

2. **Checkpoint contamination:** Once corrupted weights are saved to a checkpoint, the corruption is permanent. Future training runs starting from this checkpoint inherit the corruption.

3. **Silent quality degradation:** The model may still produce plausible outputs (it does not crash or produce NaN), but its quality is subtly degraded. For example, a corrupted language model may have slightly lower accuracy, slightly higher hallucination rate, or subtle biases. These effects may not be detected by standard evaluation benchmarks.

4. **Cost of recovery:** If training corruption is detected (e.g., through validation loss divergence), the training run must be rolled back to the last known-good checkpoint. For a training run consuming thousands of GPUs, even a few hours of rollback costs tens of thousands of dollars in compute time.

### Inference Errors

SDC during inference produces incorrect model outputs:

1. **Subtle errors:** A language model that should output "The capital of France is Paris" instead outputs "The capital of France is London." The output is syntactically valid and within the model's distribution, so no automated check flags it. Only a human (or a downstream system that validates the answer) would detect the error.

2. **Probabilistic manifestation:** SDC may not corrupt every inference request. If the fault is in a specific SM and only some requests land on that SM, corruption occurs only for a fraction of requests. This makes it hard to detect through spot checks.

3. **Safety-critical impact:** In applications where model outputs inform medical, legal, financial, or safety decisions, even a small rate of SDC-induced errors can have significant consequences.

### Data Integrity

SDC can also corrupt data that is not directly model-related:

1. **Tokenizer corruption:** If the tokenizer lookup table is stored in GPU memory and a cell flips, tokens are mapped to the wrong embeddings. Every subsequent computation is wrong.

2. **KV cache corruption:** In autoregressive inference, the key-value cache stores previously computed attention values. A corrupted cache entry affects every subsequent generated token.

3. **Communication corruption:** Data transmitted over NVLink or PCIe between GPUs may be corrupted in transit. While these interconnects have CRC protection, CRC has a non-zero undetected error rate, and multi-bit errors in specific patterns can evade CRC.

---

## Detection Challenges

Detecting SDC is fundamentally harder than detecting other types of hardware faults because the fault is, by definition, silent.

### No Error Signal

The defining characteristic of SDC is the absence of any error indication:

- No exception is raised.
- No error register is set.
- No interrupt is generated.
- No log entry is written.
- No metric changes.

The only way to detect SDC is to independently verify the correctness of computation results. This requires either: (a) knowing the correct answer in advance and comparing, or (b) detecting statistical anomalies in the output distribution.

### Intermittency

Many SDC-inducing defects are intermittent:

- The fault manifests only under specific input patterns that activate the defective logic path.
- The fault manifests only at specific temperatures (thermal-activated defect).
- The fault manifests only under specific voltage conditions (voltage-marginal path).
- The fault may appear for minutes, disappear for hours, and reappear.

This means that a GPU can pass a diagnostic test (e.g., a manufacturing self-test or a maintenance test) and still produce SDC during production workloads when conditions change. Detection systems must monitor continuously, not just periodically.

### Floating-Point Non-Determinism

Detecting SDC requires comparing a GPU's output against a known-good reference. For floating-point computations, this is complicated by legitimate sources of non-determinism:

- **Rounding mode differences:** Different GPU architectures may use different rounding modes for the same operation. For example, the `__fmaf_rn()` intrinsic is deterministic (round-to-nearest-even), but higher-level operations that decompose into multiple instructions may accumulate different rounding errors on different architectures.
- **Instruction scheduling:** The order of floating-point additions affects the result due to non-associativity. Different compiler versions, driver versions, or even different kernel launches may produce different addition orders.
- **Transcendental function implementations:** Functions like `sinf()` and `expf()` are allowed to differ by up to a few ULP (Units in the Last Place) across GPU architectures. A 1-ULP difference is legitimate; a 1000-ULP difference is SDC.

Any SDC detection system must distinguish between legitimate floating-point variation and true corruption. If the tolerance is too tight, false positives overwhelm operators. If the tolerance is too loose, real SDC is missed.

### Scale and Statistics

At fleet scale, even very low SDC rates produce a non-trivial number of events:

- If the per-GPU SDC probability is 0.01% per day, in a 10,000-GPU fleet, the expected number of SDC events per day is 1.
- But the false positive rate from noisy detectors also scales with fleet size.
- At 10,000 GPUs, even a 0.1% daily false positive rate per GPU means 10 false alarms per day.

A detection system must have a very low false positive rate (<< 0.01% per GPU per day) to avoid overwhelming operators at fleet scale, while maintaining a high true positive rate to catch real SDC.

---

## How SENTINEL Addresses Each Challenge

| Challenge | SENTINEL's Approach |
|-----------|-------------------|
| **No error signal** | Active probing: SENTINEL generates computations with known answers and independently verifies correctness, creating its own error signal where hardware provides none. |
| **Intermittency** | Continuous monitoring: probes run every 60-600 seconds, 24/7. Statistical monitors analyze every sampled inference/training output. The Bayesian model accumulates evidence over time, so even rare intermittent faults eventually lower the reliability score. |
| **Floating-point non-determinism** | Carefully chosen probe inputs that produce exactly representable results (zero ULP tolerance for FMA/Tensor Core). ULP-based tolerances for transcendental functions, calibrated per architecture. Golden answers generated at arbitrary precision using `mpmath`. |
| **Scale and statistics** | Bayesian attribution with a strong prior, requiring substantial evidence before quarantine. Fleet-wide pattern matching to detect software bugs vs. real SDC. Time-decaying evidence to allow recovery from transient faults. |
| **Workload-dependent activation** | Multi-layered detection: probes test hardware units directly, while inference/training monitors catch SDC that manifests only under production workloads. TMR canary batches provide ground-truth comparison for real workload data. |
| **Localization** | SM-pinned probes identify the exact SM that returned the wrong answer. Per-GPU Bayesian scoring isolates faulty GPUs. Cross-rank divergence identifies the specific DDP rank (and GPU) that diverged. |
| **Thermal/voltage sensitivity** | Telemetry correlation: probe failures are correlated with thermal and voltage data to detect thermally or voltage-activated defects. |
| **False positive management** | Multi-source correlation: a quarantine decision requires convergent evidence from multiple detection layers. Fleet-wide pattern matching suppresses false positives caused by software bugs. Configurable thresholds and approval gates allow operators to tune sensitivity. |
| **Compliance and forensics** | Tamper-evident audit ledger with Merkle hash chain: every event, decision, and operator action is permanently recorded. SOC 2 and ISO 27001 compliance reports are generated automatically. |
| **Production overhead** | Overhead budget enforcement: probes are scheduled within a configurable GPU time budget (default 2%). CUDA stream priorities and CUDA graphs minimize latency impact. Inference/training monitor sampling rates are configurable. |

### Defense in Depth

No single detection mechanism is sufficient. SDC is too diverse in its manifestation for any one approach to catch all cases. SENTINEL's five detection layers provide overlapping coverage:

```
Layer 1: Computational Probes     - Catches defects in arithmetic units,
                                    register files, and memory.

Layer 2: Statistical Monitoring   - Catches workload-dependent SDC that
                                    probes do not exercise.

Layer 3: Selective TMR            - Provides ground-truth validation
                                    through independent cross-GPU comparison.

Layer 4: Firmware Integration     - Accesses hardware-internal diagnostics
                                    unavailable to software.

Layer 5: Invariant Checking       - Catches end-to-end corruption across
                                    the entire pipeline, not just GPU compute.
```

A defect missed by one layer is likely caught by another. The correlation engine fuses signals from all layers to make high-confidence attribution decisions.
