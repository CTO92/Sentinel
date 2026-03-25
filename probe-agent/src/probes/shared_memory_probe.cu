/// @file shared_memory_probe.cu
/// @brief Shared memory probe CUDA implementation.
///
/// Each kernel invocation tests the shared memory of the target SM by:
/// 1. Writing a known pattern to all shared memory words.
/// 2. Reading back and verifying.
/// 3. Repeating with multiple patterns (all-zeros, all-ones, walking-one,
///    alternating, address-as-data, inverse-address).

#include "probes/shared_memory_probe.h"
#include "probes/sm_affinity.h"
#include "platform/cuda_runtime.h"
#include "util/crypto.h"
#include "util/logging.h"

#include <cstring>
#include <vector>

namespace sentinel::probes {

// ── Device kernel ────────────────────────────────────────────────────

/// Shared memory probe kernel. Uses dynamic shared memory sized at launch.
/// Tests multiple patterns and reports error counts.
extern __shared__ uint32_t smem[];

__global__ void shared_memory_probe_kernel(
        uint32_t* __restrict__ error_counts,      // kNumPatterns error counts
        uint32_t smem_words,
        uint32_t target_sm_id,
        int* __restrict__ executed_flag) {
    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    if (sm_id != target_sm_id) return;

    if (threadIdx.x == 0) atomicExch(executed_flag, 1);
    __syncthreads();

    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;

    // Patterns to test.
    const uint32_t patterns[] = {
        0x00000000u,  // All zeros.
        0xFFFFFFFFu,  // All ones.
        0x55555555u,  // Alternating 01.
        0xAAAAAAAAu,  // Alternating 10.
        0x01010101u,  // Walking ones (byte-level).
        0xFEFEFEFEu,  // Walking zeros (byte-level).
    };
    constexpr uint32_t num_patterns = 6;

    for (uint32_t p = 0; p < num_patterns; ++p) {
        uint32_t pattern = patterns[p];
        uint32_t local_errors = 0;

        // Write pattern to all shared memory words.
        for (uint32_t i = tid; i < smem_words; i += stride) {
            smem[i] = pattern;
        }
        __syncthreads();

        // Read back and verify.
        for (uint32_t i = tid; i < smem_words; i += stride) {
            if (smem[i] != pattern) {
                ++local_errors;
            }
        }
        __syncthreads();

        // Aggregate errors.
        if (local_errors > 0) {
            atomicAdd(&error_counts[p], local_errors);
        }
        __syncthreads();

        // Also test address-as-data pattern (each word = its index ^ pattern).
        for (uint32_t i = tid; i < smem_words; i += stride) {
            smem[i] = i ^ pattern;
        }
        __syncthreads();

        for (uint32_t i = tid; i < smem_words; i += stride) {
            if (smem[i] != (i ^ pattern)) {
                atomicAdd(&error_counts[p], 1u);
            }
        }
        __syncthreads();
    }
}

// ── Impl ─────────────────────────────────────────────────────────────

struct SharedMemoryProbe::Impl {
    int device_index = -1;
    int sm_count = 0;
    std::size_t max_smem_per_block = 0;

    platform::CudaDeviceBuffer d_error_counts;
    platform::CudaDeviceBuffer d_flag;

    platform::CudaPinnedBuffer h_error_counts;
    platform::CudaPinnedBuffer h_flag;

    platform::CudaEvent start_event{cudaEventDefault};
    platform::CudaEvent stop_event{cudaEventDefault};
};

SharedMemoryProbe::SharedMemoryProbe() : impl_(std::make_unique<Impl>()) {}
SharedMemoryProbe::~SharedMemoryProbe() = default;

bool SharedMemoryProbe::initialize(int device_index) {
    try {
        platform::set_device(device_index);
        impl_->device_index = device_index;
        impl_->sm_count = get_sm_count(device_index);

        // Query maximum shared memory per block.
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_index));
        impl_->max_smem_per_block = prop.sharedMemPerBlock;

        impl_->d_error_counts = platform::CudaDeviceBuffer(kNumPatterns * sizeof(uint32_t));
        impl_->d_flag = platform::CudaDeviceBuffer(sizeof(int));

        impl_->h_error_counts = platform::CudaPinnedBuffer(kNumPatterns * sizeof(uint32_t));
        impl_->h_flag = platform::CudaPinnedBuffer(sizeof(int));

        SENTINEL_LOG_INFO("Shared memory probe initialized on device {} "
                          "(max smem per block: {} KB)",
                          device_index, impl_->max_smem_per_block / 1024);
        return true;

    } catch (const platform::CudaError& e) {
        SENTINEL_LOG_ERROR("Shared memory probe init failed: {}", e.what());
        return false;
    }
}

ProbeResult SharedMemoryProbe::execute(uint32_t sm_id, cudaStream_t stream) {
    ProbeResult result;
    result.probe_id = next_probe_id();
    result.probe_type = ProbeType::kSharedMemory;
    result.sm_id = sm_id;
    result.expected_hash = {};

    try {
        platform::set_device(impl_->device_index);

        // Use actual shared memory size, clamped to our test size.
        std::size_t smem_bytes = std::min(kSmemTestBytes, impl_->max_smem_per_block);
        uint32_t smem_words = static_cast<uint32_t>(smem_bytes / sizeof(uint32_t));

        int zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(impl_->d_flag.get(), &zero, sizeof(int),
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(impl_->d_error_counts.get(), 0,
                                    kNumPatterns * sizeof(uint32_t), stream));

        dim3 grid, block;
        compute_sm_pinned_launch_params(impl_->sm_count, 256, grid, block);

        // Set dynamic shared memory size for the kernel.
        cudaFuncSetAttribute(shared_memory_probe_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));

        impl_->start_event.record(stream);

        shared_memory_probe_kernel<<<grid, block, smem_bytes, stream>>>(
            impl_->d_error_counts.as<uint32_t>(),
            smem_words,
            sm_id,
            impl_->d_flag.as<int>());

        impl_->stop_event.record(stream);

        impl_->d_error_counts.copy_to_host(impl_->h_error_counts.get(),
                                            kNumPatterns * sizeof(uint32_t), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float elapsed_ms = platform::CudaEvent::elapsed_ms(
            impl_->start_event, impl_->stop_event);
        result.execution_time_ns = static_cast<uint64_t>(elapsed_ms * 1e6);

        // Check error counts.
        const uint32_t* errors = impl_->h_error_counts.as<uint32_t>();
        uint32_t total_errors = 0;
        for (uint32_t p = 0; p < kNumPatterns; ++p) {
            total_errors += errors[p];
        }

        result.match = (total_errors == 0);
        result.actual_hash = result.expected_hash;

        if (!result.match) {
            for (uint32_t p = 0; p < kNumPatterns; ++p) {
                if (errors[p] > 0) {
                    MismatchDetail d;
                    d.byte_offset = p;
                    uint32_t err = errors[p];
                    d.expected_value = {0, 0, 0, 0};
                    d.actual_value.resize(4);
                    std::memcpy(d.actual_value.data(), &err, 4);
                    result.mismatch_details.push_back(std::move(d));
                }
            }
            SENTINEL_LOG_WARN("Shared memory probe detected {} errors on SM {}",
                              total_errors, sm_id);
        }

        result.timestamp = std::chrono::system_clock::now();

    } catch (const platform::CudaError& e) {
        SENTINEL_LOG_ERROR("Shared memory probe failed on SM {}: {}", sm_id, e.what());
        result.match = false;
        result.timestamp = std::chrono::system_clock::now();
    }

    return result;
}

void SharedMemoryProbe::teardown() {
    impl_->d_error_counts = {};
    impl_->d_flag = {};
    impl_->h_error_counts = {};
    impl_->h_flag = {};
}

std::size_t SharedMemoryProbe::memory_footprint() const noexcept {
    return kNumPatterns * sizeof(uint32_t) + sizeof(int)     // device
         + kNumPatterns * sizeof(uint32_t) + sizeof(int);    // pinned host
    // Note: shared memory is not counted here as it is on-chip.
}

}  // namespace sentinel::probes
