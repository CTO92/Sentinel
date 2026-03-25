/// @file fma_probe.cu
/// @brief FMA probe CUDA kernel and host-side implementation.
///
/// Generates 1024 FMA test vectors at compile/init time, executes them
/// on a pinned SM with 16 repetitions, and compares the output hash
/// against the precomputed golden SHA-256.

#include "probes/fma_probe.h"
#include "probes/sm_affinity.h"
#include "platform/cuda_runtime.h"
#include "util/crypto.h"
#include "util/logging.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace sentinel::probes {

// ── Test vector structure ────────────────────────────────────────────

struct FmaVector {
    float a;
    float b;
    float c;
};

// ── Device kernel ────────────────────────────────────────────────────

/// FMA probe kernel: each block checks if it landed on the target SM.
/// If yes, it computes FMA for all vectors kRepetitions times and writes
/// the results. If no, it exits immediately.
__global__ void fma_probe_kernel(const FmaVector* __restrict__ vectors,
                                  float* __restrict__ results,
                                  uint32_t num_vectors,
                                  uint32_t repetitions,
                                  uint32_t target_sm_id,
                                  int* __restrict__ executed_flag) {
    // SM affinity check.
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

    // Each thread processes a subset of vectors, all repetitions.
    for (uint32_t i = tid; i < num_vectors; i += stride) {
        float a = vectors[i].a;
        float b = vectors[i].b;
        float c = vectors[i].c;

        // Compute FMA repeatedly. The compiler must not optimize away
        // the loop because each iteration uses the previous result
        // as a dependency (through volatile).
        float result = __fmaf_rn(a, b, c);

        for (uint32_t rep = 1; rep < repetitions; ++rep) {
            // Each repetition should yield the same bit-exact result
            // for deterministic hardware.
            float check = __fmaf_rn(a, b, c);
            // XOR the bits: any difference will propagate to output.
            uint32_t r_bits, c_bits;
            std::memcpy(&r_bits, &result, sizeof(float));
            std::memcpy(&c_bits, &check, sizeof(float));
            r_bits |= (r_bits ^ c_bits);
            std::memcpy(&result, &r_bits, sizeof(float));
        }

        results[i] = result;
    }
}

// ── Host-side test vector generation ────────────────────────────────

namespace {

/// Generate the 1024 FMA test vectors covering critical IEEE 754 regions.
std::vector<FmaVector> generate_test_vectors() {
    std::vector<FmaVector> vectors;
    vectors.reserve(FmaProbe::kNumVectors);

    auto push = [&](float a, float b, float c) {
        vectors.push_back({a, b, c});
    };

    auto as_float = [](uint32_t bits) -> float {
        float f;
        std::memcpy(&f, &bits, sizeof(float));
        return f;
    };

    // --- Category 1: Normal range (256 vectors) ---
    // Small exponents.
    for (int i = 0; i < 64; ++i) {
        float scale = std::ldexp(1.0f, -60 + i);
        push(scale, 1.0f + static_cast<float>(i) * 0.001f, scale * 0.5f);
    }
    // Medium exponents.
    for (int i = 0; i < 64; ++i) {
        float val = 1.0f + static_cast<float>(i) * 0.1f;
        push(val, val + 1.0f, -val * val);
    }
    // Large exponents.
    for (int i = 0; i < 64; ++i) {
        float scale = std::ldexp(1.0f, 60 + i / 2);
        push(scale, 1.0f + static_cast<float>(i) * 1e-6f, -scale);
    }
    // Mixed signs.
    for (int i = 0; i < 64; ++i) {
        float a = (i & 1) ? -1.0f : 1.0f;
        float b = static_cast<float>(i + 1) * 0.7f;
        float c = (i & 2) ? -0.5f : 0.5f;
        push(a * b, b, c);
    }

    // --- Category 2: Denormals (256 vectors) ---
    for (int i = 0; i < 128; ++i) {
        // Smallest denormals.
        uint32_t bits_a = static_cast<uint32_t>(i + 1);
        uint32_t bits_b = static_cast<uint32_t>((i + 1) * 3);
        push(as_float(bits_a), as_float(bits_b), 0.0f);
    }
    for (int i = 0; i < 128; ++i) {
        // Denormal + normal.
        uint32_t bits_a = static_cast<uint32_t>(i + 1);
        push(as_float(bits_a), 1.0f, as_float(bits_a));
    }

    // --- Category 3: Exponent boundaries (256 vectors) ---
    // Near max float.
    for (int i = 0; i < 64; ++i) {
        float large = std::ldexp(1.0f, 126 - i);
        push(large, 1.0f + static_cast<float>(i) * 1e-7f, 0.0f);
    }
    // Near min normal.
    for (int i = 0; i < 64; ++i) {
        float small_val = std::ldexp(1.0f, -125 + i);
        push(small_val, 1.0f, -small_val * 0.5f);
    }
    // Overflow boundary.
    for (int i = 0; i < 64; ++i) {
        float a = std::ldexp(1.0f, 63 + i / 4);
        float b = std::ldexp(1.0f, 63);
        push(a, b, 0.0f);
    }
    // Underflow boundary.
    for (int i = 0; i < 64; ++i) {
        float a = std::ldexp(1.0f, -63 - i / 4);
        float b = std::ldexp(1.0f, -63);
        push(a, b, 0.0f);
    }

    // --- Category 4: Mantissa patterns (256 vectors) ---
    // All-ones mantissa.
    for (int i = 0; i < 64; ++i) {
        uint32_t bits = 0x3F7FFFFFu - static_cast<uint32_t>(i);
        push(as_float(bits), as_float(bits), 0.0f);
    }
    // Alternating bits.
    for (int i = 0; i < 64; ++i) {
        uint32_t bits_a = 0x3F555555u ^ (static_cast<uint32_t>(i) << 8);
        uint32_t bits_b = 0x3FAAAAABu ^ (static_cast<uint32_t>(i) << 4);
        push(as_float(bits_a), as_float(bits_b), 1.0f);
    }
    // Single-bit mantissa.
    for (int i = 0; i < 64; ++i) {
        uint32_t bit_pos = static_cast<uint32_t>(i % 23);
        uint32_t bits = 0x3F800000u | (1u << bit_pos);
        push(as_float(bits), as_float(bits), 0.0f);
    }
    // Powers of two (exact representable).
    for (int i = 0; i < 64; ++i) {
        float val = std::ldexp(1.0f, i - 32);
        push(val, val, val);
    }

    // Pad to exactly kNumVectors if needed.
    while (vectors.size() < FmaProbe::kNumVectors) {
        auto idx = vectors.size();
        push(static_cast<float>(idx) * 0.001f,
             static_cast<float>(idx) * 0.002f + 1.0f,
             static_cast<float>(idx) * 0.0001f);
    }
    vectors.resize(FmaProbe::kNumVectors);

    return vectors;
}

/// Compute golden FMA results on the host using strict IEEE 754 arithmetic.
std::vector<float> compute_golden_results(const std::vector<FmaVector>& vectors,
                                           uint32_t repetitions) {
    std::vector<float> results(vectors.size());
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        float a = vectors[i].a;
        float b = vectors[i].b;
        float c = vectors[i].c;
        float result = std::fma(a, b, c);

        for (uint32_t rep = 1; rep < repetitions; ++rep) {
            float check = std::fma(a, b, c);
            uint32_t r_bits, c_bits;
            std::memcpy(&r_bits, &result, sizeof(float));
            std::memcpy(&c_bits, &check, sizeof(float));
            r_bits |= (r_bits ^ c_bits);
            std::memcpy(&result, &r_bits, sizeof(float));
        }

        results[i] = result;
    }
    return results;
}

}  // namespace

// ── Impl struct ──────────────────────────────────────────────────────

struct FmaProbe::Impl {
    int device_index = -1;
    int sm_count = 0;

    // Test vectors (host).
    std::vector<FmaVector> host_vectors;

    // Golden results and hash.
    std::vector<float> golden_results;
    util::Sha256Digest golden_hash{};

    // Device buffers.
    platform::CudaDeviceBuffer d_vectors;
    platform::CudaDeviceBuffer d_results;
    platform::CudaDeviceBuffer d_flag;

    // Pinned host buffers for readback.
    platform::CudaPinnedBuffer h_results;
    platform::CudaPinnedBuffer h_flag;

    // Timing events.
    platform::CudaEvent start_event{cudaEventDefault};
    platform::CudaEvent stop_event{cudaEventDefault};
};

// ── ProbeInterface implementation ────────────────────────────────────

FmaProbe::FmaProbe() : impl_(std::make_unique<Impl>()) {}
FmaProbe::~FmaProbe() = default;

bool FmaProbe::initialize(int device_index) {
    try {
        platform::set_device(device_index);
        impl_->device_index = device_index;
        impl_->sm_count = get_sm_count(device_index);

        // Generate test vectors and golden answers.
        impl_->host_vectors = generate_test_vectors();
        impl_->golden_results = compute_golden_results(impl_->host_vectors, kRepetitions);

        // Compute golden hash.
        impl_->golden_hash = util::sha256(
            reinterpret_cast<const uint8_t*>(impl_->golden_results.data()),
            impl_->golden_results.size() * sizeof(float));

        // Allocate device memory.
        std::size_t vec_bytes = kNumVectors * sizeof(FmaVector);
        std::size_t res_bytes = kNumVectors * sizeof(float);

        impl_->d_vectors = platform::CudaDeviceBuffer(vec_bytes);
        impl_->d_results = platform::CudaDeviceBuffer(res_bytes);
        impl_->d_flag = platform::CudaDeviceBuffer(sizeof(int));

        // Allocate pinned host memory for readback.
        impl_->h_results = platform::CudaPinnedBuffer(res_bytes);
        impl_->h_flag = platform::CudaPinnedBuffer(sizeof(int));

        // Upload test vectors to device.
        impl_->d_vectors.copy_from_host(impl_->host_vectors.data(), vec_bytes);
        CUDA_CHECK(cudaDeviceSynchronize());

        SENTINEL_LOG_INFO("FMA probe initialized on device {} ({} SMs, {} vectors)",
                          device_index, impl_->sm_count, kNumVectors);
        return true;

    } catch (const platform::CudaError& e) {
        SENTINEL_LOG_ERROR("FMA probe initialization failed: {}", e.what());
        return false;
    }
}

ProbeResult FmaProbe::execute(uint32_t sm_id, cudaStream_t stream) {
    ProbeResult result;
    result.probe_id = next_probe_id();
    result.probe_type = ProbeType::kFma;
    result.sm_id = sm_id;
    result.expected_hash = impl_->golden_hash;

    try {
        platform::set_device(impl_->device_index);

        // Clear the execution flag.
        int zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(impl_->d_flag.get(), &zero, sizeof(int),
                                    cudaMemcpyHostToDevice, stream));

        // Clear results buffer.
        CUDA_CHECK(cudaMemsetAsync(impl_->d_results.get(), 0,
                                    kNumVectors * sizeof(float), stream));

        // Compute launch parameters for SM pinning.
        dim3 grid, block;
        compute_sm_pinned_launch_params(impl_->sm_count, 256, grid, block);

        // Record start event.
        impl_->start_event.record(stream);

        // Launch kernel.
        fma_probe_kernel<<<grid, block, 0, stream>>>(
            impl_->d_vectors.as<FmaVector>(),
            impl_->d_results.as<float>(),
            kNumVectors,
            kRepetitions,
            sm_id,
            impl_->d_flag.as<int>());

        // Record stop event.
        impl_->stop_event.record(stream);

        // Copy results back to pinned host memory.
        impl_->d_results.copy_to_host(impl_->h_results.get(),
                                       kNumVectors * sizeof(float), stream);
        CUDA_CHECK(cudaMemcpyAsync(impl_->h_flag.get(), impl_->d_flag.get(),
                                    sizeof(int), cudaMemcpyDeviceToHost, stream));

        // Synchronize.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Check if the kernel actually ran on the target SM.
        int flag = *impl_->h_flag.as<int>();
        if (flag == 0) {
            SENTINEL_LOG_WARN("FMA probe: no block landed on SM {}, retrying "
                              "with larger grid", sm_id);
            // Retry with a 4x larger grid.
            grid.x *= 4;
            zero = 0;
            CUDA_CHECK(cudaMemcpyAsync(impl_->d_flag.get(), &zero, sizeof(int),
                                        cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(impl_->d_results.get(), 0,
                                        kNumVectors * sizeof(float), stream));
            impl_->start_event.record(stream);
            fma_probe_kernel<<<grid, block, 0, stream>>>(
                impl_->d_vectors.as<FmaVector>(),
                impl_->d_results.as<float>(),
                kNumVectors, kRepetitions, sm_id,
                impl_->d_flag.as<int>());
            impl_->stop_event.record(stream);
            impl_->d_results.copy_to_host(impl_->h_results.get(),
                                           kNumVectors * sizeof(float), stream);
            CUDA_CHECK(cudaMemcpyAsync(impl_->h_flag.get(), impl_->d_flag.get(),
                                        sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // Compute execution time.
        float elapsed_ms = platform::CudaEvent::elapsed_ms(
            impl_->start_event, impl_->stop_event);
        result.execution_time_ns = static_cast<uint64_t>(elapsed_ms * 1e6);

        // Hash the actual results.
        result.actual_hash = util::sha256(
            reinterpret_cast<const uint8_t*>(impl_->h_results.get()),
            kNumVectors * sizeof(float));

        // Compare hashes.
        result.match = util::digest_equal(result.expected_hash, result.actual_hash);

        // If mismatch, find the differing elements.
        if (!result.match) {
            const float* actual = impl_->h_results.as<float>();
            for (uint32_t i = 0; i < kNumVectors; ++i) {
                uint32_t expected_bits, actual_bits;
                std::memcpy(&expected_bits, &impl_->golden_results[i], sizeof(float));
                std::memcpy(&actual_bits, &actual[i], sizeof(float));

                if (expected_bits != actual_bits) {
                    MismatchDetail detail;
                    detail.byte_offset = static_cast<uint64_t>(i) * sizeof(float);
                    detail.expected_value.resize(sizeof(float));
                    detail.actual_value.resize(sizeof(float));
                    std::memcpy(detail.expected_value.data(), &expected_bits, sizeof(float));
                    std::memcpy(detail.actual_value.data(), &actual_bits, sizeof(float));

                    uint32_t diff = expected_bits ^ actual_bits;
                    for (uint32_t bit = 0; bit < 32; ++bit) {
                        if (diff & (1u << bit)) {
                            detail.differing_bits.push_back(bit);
                        }
                    }
                    result.mismatch_details.push_back(std::move(detail));

                    // Limit detail collection to first 64 mismatches.
                    if (result.mismatch_details.size() >= 64) break;
                }
            }
        }

        result.timestamp = std::chrono::system_clock::now();

    } catch (const platform::CudaError& e) {
        SENTINEL_LOG_ERROR("FMA probe execution failed on SM {}: {}", sm_id, e.what());
        result.match = false;
        result.timestamp = std::chrono::system_clock::now();
    }

    return result;
}

void FmaProbe::teardown() {
    impl_->d_vectors = {};
    impl_->d_results = {};
    impl_->d_flag = {};
    impl_->h_results = {};
    impl_->h_flag = {};
    SENTINEL_LOG_INFO("FMA probe torn down on device {}", impl_->device_index);
}

std::size_t FmaProbe::memory_footprint() const noexcept {
    return kNumVectors * sizeof(FmaVector)    // d_vectors
         + kNumVectors * sizeof(float)        // d_results
         + sizeof(int)                        // d_flag
         + kNumVectors * sizeof(float)        // h_results (pinned)
         + sizeof(int);                       // h_flag (pinned)
}

}  // namespace sentinel::probes
