# Writing Custom Probe Kernels

> **Status:** Pre-release Alpha
> **Language:** C++20 / CUDA
> **Location:** `probe-agent/src/probes/`

This guide walks through the process of designing, implementing, and integrating
a new probe type into the SENTINEL probe agent. Probes are the foundation of
SENTINEL's hardware integrity detection -- each one exercises a specific GPU
functional unit and compares the output against a known-good (golden) answer.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Probe Interface Reference](#probe-interface-reference)
3. [Step-by-Step: Creating a New Probe](#step-by-step-creating-a-new-probe)
   - [Step 1: Define the Probe Class](#step-1-define-the-probe-class)
   - [Step 2: Implement the CUDA Kernel with SM Pinning](#step-2-implement-the-cuda-kernel-with-sm-pinning)
   - [Step 3: Generate Golden Answers](#step-3-generate-golden-answers)
   - [Step 4: Register with the Scheduler](#step-4-register-with-the-scheduler)
   - [Step 5: Add Protobuf Type](#step-5-add-protobuf-type)
   - [Step 6: Write Tests](#step-6-write-tests)
4. [SM Affinity Deep Dive](#sm-affinity-deep-dive)
5. [Golden Answer Generation with MPFR](#golden-answer-generation-with-mpfr)
6. [Tolerance Selection](#tolerance-selection)
7. [Performance Budgeting](#performance-budgeting)
8. [Example: Implementing a Custom Reduction Probe](#example-implementing-a-custom-reduction-probe)
9. [Testing Probes with the SDC Injector](#testing-probes-with-the-sdc-injector)

---

## Architecture Overview

The SENTINEL probe agent runs on every GPU node in the cluster. Its job is to
continuously execute lightweight "probe kernels" on each GPU, compare their
outputs against precomputed golden answers, and report results to the
Correlation Engine.

```
+-----------------------------------------------------------+
|  Probe Agent (per node)                                    |
|                                                            |
|  +-------------+    +-----------+    +------------------+  |
|  |  Scheduler   |--->|  Executor  |--->|  Result Reporter |  |
|  | (priority    |    | (SM pin,  |    | (batch, HMAC,   |  |
|  |  queue)      |    |  stream)  |    |  gRPC stream)   |  |
|  +-------------+    +-----------+    +------------------+  |
|         |                 |                                 |
|         v                 v                                 |
|  +-------------+    +-----------+                          |
|  | Probe        |    | SM        |                          |
|  | Registry     |    | Affinity  |                          |
|  | (all probe   |    | Engine    |                          |
|  |  types)      |    |           |                          |
|  +-------------+    +-----------+                          |
+-----------------------------------------------------------+
```

### Lifecycle of a probe execution

1. **Scheduler** selects the next probe to run based on priority, period, and
   SM coverage configuration.
2. **SM Affinity Engine** determines which SM(s) to target. It computes
   over-launched grid dimensions so that at least one thread block lands on
   each target SM.
3. **Executor** launches the probe kernel on a dedicated CUDA stream. Blocks
   that land on non-target SMs exit immediately (via `%smid` PTX readback).
4. The probe kernel computes its test and writes results to a device buffer.
5. **Executor** copies results to host, computes SHA-256 of the output buffer,
   and compares against the precomputed golden hash.
6. **Result Reporter** batches probe results, computes HMAC-SHA256 signatures,
   and streams them to the Correlation Engine via gRPC.

### Existing probe types

| Probe | Functional Unit | Approach |
|---|---|---|
| **FMA** | FP32 FMA ALUs | 1024 test vectors, 16 repetitions, bit-exact comparison. |
| **Tensor Core** | Matrix units (HMMA) | FP16 GEMM on 16x16 tiles, bit-exact hash comparison. |
| **Transcendental** | SFU (sin, cos, exp, log) | 4096 inputs, ULP-tolerant comparison via MPFR golden answers. |
| **AES** | Integer ALU / combinational logic | AES-128 encryption of known plaintext, exact comparison. |
| **Memory** | Global memory (HBM) | Walking-ones and MATS+ pattern tests. |
| **Register File** | Register file | Write/readback of known patterns (all-zeros, all-ones, alternating). |
| **Shared Memory** | Shared memory (SMEM) | Pattern write/readback within each SM's shared memory. |

---

## Probe Interface Reference

Every probe implements the `ProbeInterface` abstract base class defined in
`probe-agent/src/probes/probe_interface.h`.

```cpp
class ProbeInterface {
public:
    virtual ~ProbeInterface() = default;

    // Identity
    virtual ProbeType type() const noexcept = 0;
    virtual std::string_view name() const noexcept = 0;

    // Lifecycle
    virtual bool initialize(int device_index) = 0;
    virtual ProbeResult execute(uint32_t sm_id, cudaStream_t stream) = 0;
    virtual void teardown() = 0;

    // Scheduling hints
    virtual std::size_t memory_footprint() const noexcept = 0;
    virtual uint32_t default_period_seconds() const noexcept = 0;
    virtual Priority default_priority() const noexcept = 0;
    virtual SmSelection default_sm_selection() const noexcept = 0;

protected:
    static uint64_t next_probe_id();
};
```

### Method descriptions

| Method | Description |
|---|---|
| `type()` | Return the `ProbeType` enum value for this probe. Must match the protobuf enum. |
| `name()` | Return a short lowercase name (e.g., `"fma"`, `"tensor_core"`). Used in logging and metrics. |
| `initialize(device_index)` | One-time setup: allocate device/host buffers, upload golden data, precompute test vectors. Called once per GPU at agent startup. Return `true` on success. |
| `execute(sm_id, stream)` | Run the probe on a specific SM using the given CUDA stream. Return a `ProbeResult` with pass/fail, timing data, and optional mismatch details. |
| `teardown()` | Release all device and host resources. Called at agent shutdown. |
| `memory_footprint()` | Return estimated device memory usage in bytes. Used by the scheduler to ensure probes fit within the memory budget. |
| `default_period_seconds()` | Recommended execution period. Operators can override via configuration. |
| `default_priority()` | Scheduling priority: `kHigh` runs before `kMedium` and `kLow`. |
| `default_sm_selection()` | Default SM coverage: `kAll` (every SM each period), `kSample25Pct` (random 25%), or `kSample10Pct` (random 10%). |
| `next_probe_id()` | Thread-safe monotonically increasing ID generator (inherited helper). |

### ProbeResult struct

```cpp
struct ProbeResult {
    uint64_t probe_id;
    ProbeType probe_type;
    std::string gpu_uuid;
    uint32_t sm_id;
    util::Sha256Digest expected_hash;
    util::Sha256Digest actual_hash;
    bool match;
    std::vector<MismatchDetail> mismatch_details;
    uint64_t execution_time_ns;
    uint32_t gpu_clock_mhz;
    float gpu_temp_c;
    float gpu_power_w;
    std::chrono::system_clock::time_point timestamp;
    util::Sha256Digest hmac;
};
```

The executor converts `match == false` to `PROBE_RESULT_FAIL` and `match == true`
to `PROBE_RESULT_PASS`. If `execute()` throws or the kernel fails to launch,
the result is `PROBE_RESULT_ERROR`. If execution exceeds the timeout, the
result is `PROBE_RESULT_TIMEOUT`.

---

## Step-by-Step: Creating a New Probe

This walkthrough creates a hypothetical **Dot Product Probe** that tests FP32
dot product operations.

### Step 1: Define the Probe Class

Create the header file `probe-agent/src/probes/dot_product_probe.h`:

```cpp
/// @file dot_product_probe.h
/// @brief Dot product determinism probe.
///
/// Tests FP32 dot product computation on individual SMs by computing
/// dot(a, b) for 512 carefully chosen vector pairs and comparing
/// results bit-exact against precomputed golden answers.

#pragma once

#include "probes/probe_interface.h"
#include <cstdint>
#include <memory>

namespace sentinel::probes {

class DotProductProbe : public ProbeInterface {
public:
    DotProductProbe();
    ~DotProductProbe() override;

    // Identity.
    [[nodiscard]] ProbeType type() const noexcept override {
        return ProbeType::kDotProduct;  // Add to enum first!
    }
    [[nodiscard]] std::string_view name() const noexcept override {
        return "dot_product";
    }

    // Lifecycle.
    bool initialize(int device_index) override;
    [[nodiscard]] ProbeResult execute(uint32_t sm_id,
                                       cudaStream_t stream) override;
    void teardown() override;

    // Scheduling hints.
    [[nodiscard]] std::size_t memory_footprint() const noexcept override;
    [[nodiscard]] uint32_t default_period_seconds() const noexcept override {
        return 120;  // Every 2 minutes.
    }
    [[nodiscard]] Priority default_priority() const noexcept override {
        return Priority::kMedium;
    }
    [[nodiscard]] SmSelection default_sm_selection() const noexcept override {
        return SmSelection::kSample25Pct;
    }

    // Configuration.
    static constexpr uint32_t kNumVectorPairs = 512;
    static constexpr uint32_t kVectorLength = 256;
    static constexpr uint32_t kRepetitions = 8;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::probes
```

Key design decisions:
- **Pimpl pattern**: The `Impl` struct holds all CUDA-specific state (device
  pointers, host buffers). This keeps CUDA headers out of the public interface.
- **Constants**: `kNumVectorPairs`, `kVectorLength`, and `kRepetitions` define
  the probe's workload size. More vectors = more coverage but higher overhead.
- **Scheduling**: `kMedium` priority and 25% SM sampling because dot products
  are less critical than FMA units.

### Step 2: Implement the CUDA Kernel with SM Pinning

Create `probe-agent/src/probes/dot_product_probe.cu`:

```cuda
/// @file dot_product_probe.cu
/// @brief Dot product probe CUDA kernel and host-side implementation.

#include "probes/dot_product_probe.h"
#include "probes/sm_affinity.h"
#include "util/crypto.h"
#include "util/logging.h"

#include <chrono>
#include <cstring>
#include <vector>

namespace sentinel::probes {

// -- Test vector structure --

struct DotProductVectors {
    float a[DotProductProbe::kVectorLength];
    float b[DotProductProbe::kVectorLength];
};

// -- Device kernel --

/// Dot product probe kernel. Each block checks if it is on the target SM.
/// If yes, computes dot products for all vector pairs with repetitions.
__global__ void dot_product_kernel(
    const DotProductVectors* __restrict__ vectors,
    float* __restrict__ results,
    uint32_t num_pairs,
    uint32_t vec_length,
    uint32_t repetitions,
    uint32_t target_sm_id,
    int* __restrict__ executed_flag)
{
    // SM affinity check via inline PTX.
    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    if (sm_id != target_sm_id) {
        return;
    }

    // Signal that at least one block ran on the target SM.
    if (threadIdx.x == 0) {
        atomicExch(executed_flag, 1);
    }
    __syncthreads();

    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;

    // Each thread handles a subset of vector pairs.
    for (uint32_t pair = tid; pair < num_pairs; pair += stride) {
        float sum = 0.0f;
        for (uint32_t rep = 0; rep < repetitions; rep++) {
            float dot = 0.0f;
            for (uint32_t i = 0; i < vec_length; i++) {
                dot = __fmaf_rn(vectors[pair].a[i], vectors[pair].b[i], dot);
            }
            sum += dot;
        }
        results[pair] = sum;
    }
}

// -- Host implementation --

struct DotProductProbe::Impl {
    int device_index = -1;
    int sm_count = 0;

    // Host-side golden data.
    std::vector<DotProductVectors> h_vectors;
    std::vector<float> h_golden_results;
    util::Sha256Digest golden_hash;

    // Device buffers.
    DotProductVectors* d_vectors = nullptr;
    float* d_results = nullptr;
    int* d_executed_flag = nullptr;

    // Host output buffer.
    std::vector<float> h_results;
};

DotProductProbe::DotProductProbe() : impl_(std::make_unique<Impl>()) {}
DotProductProbe::~DotProductProbe() { teardown(); }

bool DotProductProbe::initialize(int device_index) {
    impl_->device_index = device_index;
    cudaSetDevice(device_index);

    // Query SM count.
    impl_->sm_count = get_sm_count(device_index);
    if (impl_->sm_count == 0) {
        SENTINEL_LOG_ERROR("Failed to query SM count for device {}", device_index);
        return false;
    }

    // Generate test vectors (deterministic, seeded).
    impl_->h_vectors.resize(kNumVectorPairs);
    generate_test_vectors(impl_->h_vectors);  // See golden answer section.

    // Compute golden answers on host using double precision.
    impl_->h_golden_results.resize(kNumVectorPairs);
    compute_golden_results(impl_->h_vectors, impl_->h_golden_results);

    // Compute golden hash.
    impl_->golden_hash = util::sha256(
        impl_->h_golden_results.data(),
        impl_->h_golden_results.size() * sizeof(float));

    // Allocate device buffers.
    cudaMalloc(&impl_->d_vectors,
               kNumVectorPairs * sizeof(DotProductVectors));
    cudaMalloc(&impl_->d_results, kNumVectorPairs * sizeof(float));
    cudaMalloc(&impl_->d_executed_flag, sizeof(int));

    // Upload test vectors.
    cudaMemcpy(impl_->d_vectors, impl_->h_vectors.data(),
               kNumVectorPairs * sizeof(DotProductVectors),
               cudaMemcpyHostToDevice);

    // Allocate host output buffer.
    impl_->h_results.resize(kNumVectorPairs);

    SENTINEL_LOG_INFO("DotProductProbe initialized on device {} ({} SMs)",
                      device_index, impl_->sm_count);
    return true;
}

ProbeResult DotProductProbe::execute(uint32_t sm_id, cudaStream_t stream) {
    ProbeResult result;
    result.probe_id = next_probe_id();
    result.probe_type = ProbeType::kDotProduct;
    result.sm_id = sm_id;
    result.expected_hash = impl_->golden_hash;

    auto start = std::chrono::high_resolution_clock::now();

    // Clear executed flag.
    int zero = 0;
    cudaMemcpyAsync(impl_->d_executed_flag, &zero, sizeof(int),
                     cudaMemcpyHostToDevice, stream);

    // Clear results.
    cudaMemsetAsync(impl_->d_results, 0,
                     kNumVectorPairs * sizeof(float), stream);

    // Compute launch parameters for SM pinning.
    dim3 grid, block;
    compute_sm_pinned_launch_params(impl_->sm_count, 256, grid, block);

    // Launch kernel.
    dot_product_kernel<<<grid, block, 0, stream>>>(
        impl_->d_vectors, impl_->d_results,
        kNumVectorPairs, kVectorLength, kRepetitions,
        sm_id, impl_->d_executed_flag);

    // Copy results to host.
    cudaMemcpyAsync(impl_->h_results.data(), impl_->d_results,
                     kNumVectorPairs * sizeof(float),
                     cudaMemcpyDeviceToHost, stream);

    // Copy executed flag.
    int executed = 0;
    cudaMemcpyAsync(&executed, impl_->d_executed_flag, sizeof(int),
                     cudaMemcpyDeviceToHost, stream);

    // Synchronize.
    cudaStreamSynchronize(stream);

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    result.timestamp = std::chrono::system_clock::now();

    if (!executed) {
        // No block landed on the target SM. This is rare but possible.
        // Mark as ERROR so the scheduler retries.
        result.match = false;
        return result;
    }

    // Compute SHA-256 of actual results.
    result.actual_hash = util::sha256(
        impl_->h_results.data(),
        impl_->h_results.size() * sizeof(float));

    // Compare.
    result.match = (result.actual_hash == result.expected_hash);

    if (!result.match) {
        // Populate mismatch details.
        for (uint32_t i = 0; i < kNumVectorPairs; i++) {
            if (impl_->h_results[i] != impl_->h_golden_results[i]) {
                MismatchDetail detail;
                detail.byte_offset = i * sizeof(float);
                detail.expected_value.assign(
                    reinterpret_cast<uint8_t*>(&impl_->h_golden_results[i]),
                    reinterpret_cast<uint8_t*>(&impl_->h_golden_results[i]) + sizeof(float));
                detail.actual_value.assign(
                    reinterpret_cast<uint8_t*>(&impl_->h_results[i]),
                    reinterpret_cast<uint8_t*>(&impl_->h_results[i]) + sizeof(float));
                // Compute differing bits.
                uint32_t expected_bits, actual_bits;
                std::memcpy(&expected_bits, &impl_->h_golden_results[i], 4);
                std::memcpy(&actual_bits, &impl_->h_results[i], 4);
                uint32_t xor_val = expected_bits ^ actual_bits;
                for (int b = 0; b < 32; b++) {
                    if (xor_val & (1u << b)) {
                        detail.differing_bits.push_back(b);
                    }
                }
                result.mismatch_details.push_back(detail);
            }
        }
    }

    return result;
}

void DotProductProbe::teardown() {
    if (impl_->d_vectors) { cudaFree(impl_->d_vectors); impl_->d_vectors = nullptr; }
    if (impl_->d_results) { cudaFree(impl_->d_results); impl_->d_results = nullptr; }
    if (impl_->d_executed_flag) { cudaFree(impl_->d_executed_flag); impl_->d_executed_flag = nullptr; }
}

std::size_t DotProductProbe::memory_footprint() const noexcept {
    return kNumVectorPairs * sizeof(DotProductVectors)  // Input vectors.
         + kNumVectorPairs * sizeof(float)              // Results.
         + sizeof(int);                                 // Executed flag.
}

}  // namespace sentinel::probes
```

### Step 3: Generate Golden Answers

Golden answers must be generated with higher precision than the GPU will use.
For FP32 probes, compute golden answers in FP64 or with MPFR at 128-bit
precision.

```cpp
// In dot_product_probe.cu (or a separate golden generation tool)

#include <cmath>
#include <random>

void generate_test_vectors(std::vector<DotProductVectors>& vectors) {
    // Use a fixed seed for deterministic test vectors.
    std::mt19937 rng(0xDEADBEEF);

    // Distribution categories:
    std::uniform_real_distribution<float> normal(-1.0f, 1.0f);
    std::uniform_real_distribution<float> large(1e30f, 1e38f);
    std::uniform_real_distribution<float> small(1e-38f, 1e-30f);

    for (size_t pair = 0; pair < vectors.size(); pair++) {
        auto& v = vectors[pair];
        for (uint32_t i = 0; i < DotProductProbe::kVectorLength; i++) {
            if (pair < vectors.size() / 4) {
                // Normal range.
                v.a[i] = normal(rng);
                v.b[i] = normal(rng);
            } else if (pair < vectors.size() / 2) {
                // Large values (near overflow).
                v.a[i] = large(rng);
                v.b[i] = small(rng);  // Keep product in normal range.
            } else if (pair < 3 * vectors.size() / 4) {
                // Denormals.
                v.a[i] = std::ldexp(normal(rng), -126);
                v.b[i] = std::ldexp(normal(rng), -126);
            } else {
                // Special mantissa patterns.
                uint32_t bits = rng();
                std::memcpy(&v.a[i], &bits, sizeof(float));
                bits = rng();
                std::memcpy(&v.b[i], &bits, sizeof(float));
                // Ensure we do not get NaN/Inf.
                if (!std::isfinite(v.a[i])) v.a[i] = 1.0f;
                if (!std::isfinite(v.b[i])) v.b[i] = 1.0f;
            }
        }
    }
}

void compute_golden_results(const std::vector<DotProductVectors>& vectors,
                             std::vector<float>& results) {
    // Compute in double precision, then cast to float.
    // This matches IEEE 754 round-to-nearest-even for the final result.
    for (size_t pair = 0; pair < vectors.size(); pair++) {
        float sum = 0.0f;
        for (uint32_t rep = 0; rep < DotProductProbe::kRepetitions; rep++) {
            float dot = 0.0f;
            for (uint32_t i = 0; i < DotProductProbe::kVectorLength; i++) {
                // Use __fmaf_rn equivalent: standard FMA with round-to-nearest.
                dot = std::fma(vectors[pair].a[i], vectors[pair].b[i], dot);
            }
            sum += dot;
        }
        results[pair] = sum;
    }
}
```

**Important:** The golden computation must exactly replicate the GPU kernel's
arithmetic sequence, including accumulation order. FMA operations are
deterministic within a single SM on NVIDIA GPUs when using `__fmaf_rn`
(round-to-nearest-even). The golden host computation must use the same
rounding mode.

### Step 4: Register with the Scheduler

Edit `probe-agent/src/probes/probe_interface.h` to add the new probe type:

```cpp
enum class ProbeType : uint8_t {
    kFma           = 1,
    kTensorCore    = 2,
    kTranscendental = 3,
    kAes           = 4,
    kMemory        = 5,
    kRegisterFile  = 6,
    kSharedMemory  = 7,
    kDotProduct    = 8,   // <-- Add this.
};
```

Update `probe_type_name()`:

```cpp
case ProbeType::kDotProduct: return "dot_product";
```

Add the probe to the factory function in `probe_interface.cpp`:

```cpp
std::vector<std::unique_ptr<ProbeInterface>> create_all_probes() {
    std::vector<std::unique_ptr<ProbeInterface>> probes;
    probes.push_back(std::make_unique<FmaProbe>());
    probes.push_back(std::make_unique<TensorCoreProbe>());
    probes.push_back(std::make_unique<TranscendentalProbe>());
    probes.push_back(std::make_unique<AesProbe>());
    probes.push_back(std::make_unique<MemoryProbe>());
    probes.push_back(std::make_unique<RegisterFileProbe>());
    probes.push_back(std::make_unique<SharedMemoryProbe>());
    probes.push_back(std::make_unique<DotProductProbe>());  // <-- Add this.
    return probes;
}
```

### Step 5: Add Protobuf Type

Edit `proto/sentinel/v1/probe.proto` to add the new enum value:

```protobuf
enum ProbeType {
  PROBE_TYPE_UNSPECIFIED = 0;
  PROBE_TYPE_FMA = 1;
  PROBE_TYPE_TENSOR_CORE = 2;
  PROBE_TYPE_TRANSCENDENTAL = 3;
  PROBE_TYPE_AES = 4;
  PROBE_TYPE_MEMORY = 5;
  PROBE_TYPE_REGISTER_FILE = 6;
  PROBE_TYPE_SHARED_MEMORY = 7;
  PROBE_TYPE_DOT_PRODUCT = 8;   // <-- Add this.
}
```

Update the corresponding SDK types:

- **Python** (`sdk/python/src/sentinel_sdk/types.py`):
  ```python
  class ProbeType(IntEnum):
      DOT_PRODUCT = 8
  ```

- **Go** (`sdk/go/sentinel/types.go`):
  ```go
  ProbeTypeDotProduct ProbeType = 8
  ```

Regenerate protobuf stubs:

```bash
# From the repository root:
buf generate
```

### Step 6: Write Tests

Create `probe-agent/tests/dot_product_probe_test.cu`:

```cpp
#include "probes/dot_product_probe.h"
#include <gtest/gtest.h>

namespace sentinel::probes {

class DotProductProbeTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
        probe_ = std::make_unique<DotProductProbe>();
        ASSERT_TRUE(probe_->initialize(0));
        cudaStreamCreate(&stream_);
    }

    void TearDown() override {
        if (probe_) probe_->teardown();
        if (stream_) cudaStreamDestroy(stream_);
    }

    std::unique_ptr<DotProductProbe> probe_;
    cudaStream_t stream_ = nullptr;
};

// Verify that the probe passes on a healthy GPU.
TEST_F(DotProductProbeTest, PassesOnHealthyGpu) {
    int sm_count = get_sm_count(0);
    ASSERT_GT(sm_count, 0);

    // Test on SM 0.
    ProbeResult result = probe_->execute(0, stream_);
    EXPECT_TRUE(result.match) << "Probe should pass on a healthy GPU";
    EXPECT_EQ(result.expected_hash, result.actual_hash);
    EXPECT_GT(result.execution_time_ns, 0);
}

// Verify determinism: same SM produces the same hash.
TEST_F(DotProductProbeTest, Deterministic) {
    ProbeResult r1 = probe_->execute(0, stream_);
    ProbeResult r2 = probe_->execute(0, stream_);
    EXPECT_EQ(r1.actual_hash, r2.actual_hash)
        << "Same probe on same SM should produce identical results";
}

// Verify the probe runs on all SMs without error.
TEST_F(DotProductProbeTest, AllSMs) {
    int sm_count = get_sm_count(0);
    for (int sm = 0; sm < sm_count; sm++) {
        ProbeResult result = probe_->execute(sm, stream_);
        EXPECT_TRUE(result.match)
            << "Probe failed on SM " << sm;
    }
}

// Verify memory footprint is reasonable.
TEST_F(DotProductProbeTest, MemoryFootprint) {
    size_t footprint = probe_->memory_footprint();
    EXPECT_GT(footprint, 0);
    EXPECT_LT(footprint, 256 * 1024 * 1024)
        << "Probe should use less than 256 MiB of device memory";
}

// Verify scheduling defaults.
TEST_F(DotProductProbeTest, SchedulingDefaults) {
    EXPECT_EQ(probe_->type(), ProbeType::kDotProduct);
    EXPECT_EQ(probe_->name(), "dot_product");
    EXPECT_EQ(probe_->default_period_seconds(), 120);
    EXPECT_EQ(probe_->default_priority(), Priority::kMedium);
    EXPECT_EQ(probe_->default_sm_selection(), SmSelection::kSample25Pct);
}

}  // namespace sentinel::probes
```

---

## SM Affinity Deep Dive

SM pinning is essential for attributing probe failures to specific SMs.
SENTINEL uses an "over-launch with filter" strategy.

### How it works

1. **Over-launch:** The kernel is launched with enough thread blocks to cover
   every SM on the GPU (typically `sm_count * 2` blocks). CUDA's hardware
   scheduler distributes blocks across SMs round-robin.

2. **SM ID readback:** Each block reads its SM ID using inline PTX:
   ```
   asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
   ```

3. **Filter:** Blocks on non-target SMs return immediately. Only the block(s)
   on the target SM perform the actual probe computation.

4. **Execution flag:** An atomic flag in device memory confirms that at least
   one block executed on the target SM. If not (rare, due to scheduling
   anomalies), the probe is retried.

### Computing launch parameters

```cpp
void compute_sm_pinned_launch_params(int sm_count,
                                      int threads_per_block,
                                      dim3& grid_dim,
                                      dim3& block_dim) {
    // Launch 2x SM count blocks to ensure coverage.
    grid_dim = dim3(sm_count * 2);
    block_dim = dim3(threads_per_block);
}
```

### Why not CU_STREAM_PER_SM or CUDA Green Contexts?

- `CU_STREAM_PER_SM` is not publicly available in all CUDA toolkit versions.
- CUDA Green Contexts (Compute Capability 9.0+) offer SM partitioning but
  require CUDA 12+ and are not universally supported.
- The over-launch approach works on all GPU architectures from Volta (CC 7.0)
  onwards and has minimal overhead (early-exit blocks consume only a few
  nanoseconds).

### Performance impact

The overhead of non-target blocks is negligible:

| SM Count | Blocks launched | Target blocks | Non-target blocks | Overhead |
|---|---|---|---|---|
| 132 (H100) | 264 | 1-2 | 262-263 | < 1 microsecond |
| 108 (A100) | 216 | 1-2 | 214-215 | < 1 microsecond |

---

## Golden Answer Generation with MPFR

For probes that test floating-point operations, golden answers must be computed
at higher precision than the GPU uses. SENTINEL uses the
[GNU MPFR](https://www.mpfr.org/) library for this purpose.

### Why MPFR?

- IEEE 754 compliance with configurable precision (up to millions of bits)
- Correctly rounded results for all operations
- Support for all rounding modes (RN, RZ, RU, RD)
- Deterministic across platforms

### Workflow

1. **Generate test vectors** using a deterministic PRNG (fixed seed).
2. **Compute golden answers** using MPFR at 128-bit precision.
3. **Round to target precision** (FP32 or FP16) using `mpfr_get_flt()` with
   round-to-nearest-even (`MPFR_RNDN`).
4. **Hash the golden output** using SHA-256.
5. **Embed the hash** in the probe (compiled into the agent binary or loaded
   from a golden data file).

### Example: FMA golden answer

```cpp
#include <mpfr.h>

float compute_fma_golden(float a, float b, float c) {
    mpfr_t ma, mb, mc, result;
    mpfr_init2(ma, 128);
    mpfr_init2(mb, 128);
    mpfr_init2(mc, 128);
    mpfr_init2(result, 128);

    mpfr_set_flt(ma, a, MPFR_RNDN);
    mpfr_set_flt(mb, b, MPFR_RNDN);
    mpfr_set_flt(mc, c, MPFR_RNDN);

    // FMA: result = a * b + c with a single rounding.
    mpfr_fma(result, ma, mb, mc, MPFR_RNDN);

    float golden = mpfr_get_flt(result, MPFR_RNDN);

    mpfr_clear(ma);
    mpfr_clear(mb);
    mpfr_clear(mc);
    mpfr_clear(result);

    return golden;
}
```

### When MPFR is not needed

Some probes do not require high-precision golden answers:

- **AES probe**: AES is integer-based. The golden answer is computed using a
  reference AES implementation (e.g., OpenSSL).
- **Memory probe**: Walking-ones and MATS+ patterns are exact. Golden answers
  are the patterns themselves.
- **Register file probe**: Known bit patterns. Golden answers are trivial.

---

## Tolerance Selection

### Exact comparison (hash-based)

Most SENTINEL probes use **bit-exact comparison**: the SHA-256 of the GPU's
output must match the SHA-256 of the golden output. This is the strongest
possible test for SDC.

This works because:
- IEEE 754 requires deterministic results for +, -, *, /, sqrt, and FMA with a
  given rounding mode.
- NVIDIA GPUs guarantee bit-exact reproducibility for FP32 operations within a
  single SM when using the same instruction sequence.
- Probes are designed to use only deterministic operations.

### ULP-based comparison

Some operations are not guaranteed bit-exact:

- **Transcendental functions** (sin, cos, exp, log): NVIDIA specifies accuracy
  in ULPs (Units in the Last Place), not bit-exactness. For example,
  `__sinf()` is accurate to 2 ULP.
- **Cross-SM operations**: If a probe spans multiple SMs (avoid this),
  floating-point accumulation order may vary.

For these cases, SENTINEL uses ULP-based comparison:

```cpp
int32_t ulp_difference(float expected, float actual) {
    // Reinterpret as integers for ULP computation.
    int32_t expected_bits, actual_bits;
    std::memcpy(&expected_bits, &expected, sizeof(float));
    std::memcpy(&actual_bits, &actual, sizeof(float));

    // Handle sign bit (IEEE 754 integer ordering works for same-sign values).
    if (expected_bits < 0) expected_bits = 0x80000000 - expected_bits;
    if (actual_bits < 0) actual_bits = 0x80000000 - actual_bits;

    return std::abs(expected_bits - actual_bits);
}
```

The transcendental probe allows up to 2 ULP deviation for results within the
normal range. Any deviation beyond the documented accuracy bound indicates
potential SDC.

### Selecting the right tolerance

| Probe type | Tolerance | Rationale |
|---|---|---|
| FMA | Exact (0 ULP) | IEEE 754 mandates exact rounding for FMA. |
| Tensor Core | Exact (hash) | HMMA is deterministic on a single SM. |
| AES | Exact (hash) | Integer operations are always bit-exact. |
| Memory | Exact (pattern) | Memory reads must return exactly what was written. |
| Register file | Exact (pattern) | Same as memory. |
| Shared memory | Exact (pattern) | Same as memory. |
| Transcendental | 2 ULP | NVIDIA documents 2 ULP accuracy for SFU. |

---

## Performance Budgeting

SENTINEL probes must fit within a strict overhead budget. The default is 2% of
GPU time, configurable via `OverheadBudgetUpdate`.

### Measuring probe overhead

Each probe reports `execution_time_ns` in its result. The scheduler tracks the
aggregate GPU time consumed by all probes and throttles execution if the budget
is exceeded.

### Budget calculation

```
overhead_pct = (sum of probe execution times per second) / (1 second) * 100
```

For a budget of 2% on an H100:

```
Available probe time per second = 0.02 * 1,000,000,000 ns = 20,000,000 ns = 20 ms
```

### Probe timing guidelines

| Probe | Typical execution time | SM coverage | Period | Overhead per GPU |
|---|---|---|---|---|
| FMA | 200 us | 100% (132 SMs) | 60s | 132 * 200us / 60s = 0.44 ms/s (0.044%) |
| Tensor Core | 500 us | 25% (33 SMs) | 120s | 33 * 500us / 120s = 0.14 ms/s (0.014%) |
| Transcendental | 150 us | 100% | 60s | 132 * 150us / 60s = 0.33 ms/s (0.033%) |
| AES | 100 us | 10% (13 SMs) | 120s | 13 * 100us / 120s = 0.01 ms/s (0.001%) |
| Memory | 1 ms | 100% | 300s | 132 * 1ms / 300s = 0.44 ms/s (0.044%) |
| Register file | 50 us | 100% | 120s | 132 * 50us / 120s = 0.06 ms/s (0.006%) |
| Shared memory | 80 us | 100% | 120s | 132 * 80us / 120s = 0.09 ms/s (0.009%) |
| **Total** | | | | **~1.0 ms/s (0.1%)** |

This leaves significant headroom within the 2% budget for custom probes and
increased frequency during investigations.

### Design rules for keeping probes fast

1. **Minimize data transfers**: Upload test vectors once at `initialize()`,
   not per execution.
2. **Use small output buffers**: Hash the output on the GPU if possible to
   reduce DtoH transfer size.
3. **Avoid synchronization within kernels**: SM-pinned probes run on a single
   SM; no inter-SM sync needed.
4. **Keep repetition counts reasonable**: More repetitions increase sensitivity
   to intermittent faults but cost proportionally more time.
5. **Use CUDA streams**: The executor uses a dedicated stream so probes do not
   block production kernels.

---

## Example: Implementing a Custom Reduction Probe

This complete example implements a tree-reduction probe that tests the parallel
reduction pattern commonly used in softmax, layer norm, and loss computation.

### Design

- **What it tests**: FP32 addition in a tree-reduction pattern (warp shuffles
  and shared memory reduces).
- **Why it matters**: SDC in reduction operations can cause subtle errors in
  softmax outputs, loss values, and gradient norms.
- **Test vectors**: 64 fixed input arrays of 1024 elements each.
- **Golden answer**: Sum computed with Kahan compensated summation in FP64.
- **Tolerance**: Exact (the reduction uses `__shfl_xor_sync` and shared memory
  adds, which are deterministic on a single SM).

### Header: `reduction_probe.h`

```cpp
#pragma once
#include "probes/probe_interface.h"
#include <memory>

namespace sentinel::probes {

class ReductionProbe : public ProbeInterface {
public:
    ReductionProbe();
    ~ReductionProbe() override;

    [[nodiscard]] ProbeType type() const noexcept override { return ProbeType::kReduction; }
    [[nodiscard]] std::string_view name() const noexcept override { return "reduction"; }

    bool initialize(int device_index) override;
    [[nodiscard]] ProbeResult execute(uint32_t sm_id, cudaStream_t stream) override;
    void teardown() override;

    [[nodiscard]] std::size_t memory_footprint() const noexcept override;
    [[nodiscard]] uint32_t default_period_seconds() const noexcept override { return 90; }
    [[nodiscard]] Priority default_priority() const noexcept override { return Priority::kMedium; }
    [[nodiscard]] SmSelection default_sm_selection() const noexcept override { return SmSelection::kSample25Pct; }

    static constexpr uint32_t kNumArrays = 64;
    static constexpr uint32_t kArrayLength = 1024;
    static constexpr uint32_t kRepetitions = 4;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
```

### Kernel: `reduction_probe.cu`

```cuda
#include "probes/reduction_probe.h"
#include "probes/sm_affinity.h"
#include "util/crypto.h"

namespace sentinel::probes {

// Warp-level reduction using shuffle.
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory.
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];  // One slot per warp.

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces the per-warp sums.
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

__global__ void reduction_probe_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    uint32_t array_length,
    uint32_t num_arrays,
    uint32_t repetitions,
    uint32_t target_sm_id,
    int* __restrict__ executed_flag)
{
    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    if (sm_id != target_sm_id) return;

    if (threadIdx.x == 0) atomicExch(executed_flag, 1);
    __syncthreads();

    // Each block processes one array.
    for (uint32_t arr = blockIdx.x; arr < num_arrays; arr += gridDim.x) {
        float accumulated = 0.0f;
        for (uint32_t rep = 0; rep < repetitions; rep++) {
            // Load and reduce.
            float thread_sum = 0.0f;
            for (uint32_t i = threadIdx.x; i < array_length; i += blockDim.x) {
                thread_sum += input[arr * array_length + i];
            }
            float block_sum = block_reduce_sum(thread_sum);
            if (threadIdx.x == 0) {
                accumulated += block_sum;
            }
        }
        if (threadIdx.x == 0) {
            output[arr] = accumulated;
        }
    }
}

// ... (Impl struct, initialize, execute, teardown follow the same pattern
//      as the DotProductProbe example above.)

}
```

This probe tests the exact GPU functional units used by common DNN operations.
A single-bit error in a warp shuffle or shared memory write would cause a hash
mismatch and trigger investigation.

---

## Testing Probes with the SDC Injector

SENTINEL includes an SDC fault injector for testing probes in controlled
conditions. It uses CUDA Compute Sanitizer's fault injection API (where
available) or a software-based bit-flip injection.

### Software bit-flip injection

The injector modifies the GPU's output buffer after kernel execution but before
hash computation. This simulates an SDC event and verifies the probe correctly
detects it.

```bash
# Run the probe agent with fault injection enabled:
sentinel-probe-agent --inject-sdc \
    --inject-probe-type fma \
    --inject-sm 42 \
    --inject-bit-position 17 \
    --inject-probability 0.01
```

### Verification checklist

When testing a new probe with the injector:

1. **Single-bit flip**: Inject a 1-bit error. Verify the probe reports
   `PROBE_RESULT_FAIL` with the correct SM ID and mismatch details.

2. **Multi-bit flip**: Inject errors in multiple bit positions. Verify all
   differing bits are reported in `mismatch_details.differing_bits`.

3. **Different SMs**: Inject faults on different SMs. Verify the correct SM
   is attributed in the probe result.

4. **No injection**: Run without injection for 1000+ executions. Verify zero
   false positives.

5. **Performance**: Verify execution time stays within budget. Measure with
   `nsys profile` or the probe's reported `execution_time_ns`.

### Integration test

```bash
# Run the full test suite with SDC injection:
cd probe-agent
cmake --build build --target test
./build/tests/probe_integration_test --gtest_filter="*SDCInjection*"
```

Expected output:

```
[  PASSED  ] SDCInjection.DetectsFlippedBit (42 ms)
[  PASSED  ] SDCInjection.ReportsCorrectSM (38 ms)
[  PASSED  ] SDCInjection.ReportsMismatchDetails (41 ms)
[  PASSED  ] SDCInjection.NoFalsePositives (1204 ms)
[  PASSED  ] SDCInjection.WithinPerformanceBudget (52 ms)
```
