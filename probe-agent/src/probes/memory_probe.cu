/// @file memory_probe.cu
/// @brief March C- memory test implementation for GPU global memory.
///
/// March C- consists of 6 march elements:
///   M0: up(w0)       — Write 0 to all cells ascending
///   M1: up(r0,w1)    — Read 0, write 1 ascending
///   M2: up(r1,w0)    — Read 1, write 0 ascending
///   M3: down(r0,w1)  — Read 0, write 1 descending
///   M4: down(r1,w0)  — Read 1, write 0 descending
///   M5: up(r0)       — Read 0 ascending (final verify)

#include "probes/memory_probe.h"
#include "probes/sm_affinity.h"
#include "platform/cuda_runtime.h"
#include "util/crypto.h"
#include "util/logging.h"

#include <cstring>
#include <vector>

namespace sentinel::probes {

// ── March C- kernels ────────────────────────────────────────────────

/// Write a pattern to all words in the test region.
__global__ void march_write_kernel(
        uint32_t* __restrict__ mem,
        uint32_t pattern,
        uint32_t num_words,
        uint32_t target_sm_id,
        int* __restrict__ executed_flag) {
    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    if (sm_id != target_sm_id) return;

    if (threadIdx.x == 0) atomicExch(executed_flag, 1);
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < num_words; i += blockDim.x) {
        mem[i] = pattern;
    }
}

/// Read and verify a pattern, then write a new pattern (ascending order).
/// Returns the number of errors detected in the error_count output.
__global__ void march_read_write_up_kernel(
        uint32_t* __restrict__ mem,
        uint32_t expected_pattern,
        uint32_t write_pattern,
        uint32_t num_words,
        uint32_t target_sm_id,
        int* __restrict__ executed_flag,
        uint32_t* __restrict__ error_count,
        uint32_t* __restrict__ first_error_addr) {
    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    if (sm_id != target_sm_id) return;

    if (threadIdx.x == 0) atomicExch(executed_flag, 1);
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < num_words; i += blockDim.x) {
        uint32_t val = mem[i];
        if (val != expected_pattern) {
            uint32_t err_idx = atomicAdd(error_count, 1u);
            if (err_idx == 0) {
                *first_error_addr = i;
            }
        }
        mem[i] = write_pattern;
    }
}

/// Read and verify a pattern, then write a new pattern (descending order).
__global__ void march_read_write_down_kernel(
        uint32_t* __restrict__ mem,
        uint32_t expected_pattern,
        uint32_t write_pattern,
        uint32_t num_words,
        uint32_t target_sm_id,
        int* __restrict__ executed_flag,
        uint32_t* __restrict__ error_count,
        uint32_t* __restrict__ first_error_addr) {
    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    if (sm_id != target_sm_id) return;

    if (threadIdx.x == 0) atomicExch(executed_flag, 1);
    __syncthreads();

    // Descending access: start from the end.
    for (uint32_t idx = threadIdx.x; idx < num_words; idx += blockDim.x) {
        uint32_t i = num_words - 1 - idx;
        uint32_t val = mem[i];
        if (val != expected_pattern) {
            uint32_t err_idx = atomicAdd(error_count, 1u);
            if (err_idx == 0) {
                *first_error_addr = i;
            }
        }
        mem[i] = write_pattern;
    }
}

/// Final read-only verification.
__global__ void march_verify_kernel(
        const uint32_t* __restrict__ mem,
        uint32_t expected_pattern,
        uint32_t num_words,
        uint32_t target_sm_id,
        int* __restrict__ executed_flag,
        uint32_t* __restrict__ error_count,
        uint32_t* __restrict__ first_error_addr) {
    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    if (sm_id != target_sm_id) return;

    if (threadIdx.x == 0) atomicExch(executed_flag, 1);
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < num_words; i += blockDim.x) {
        if (mem[i] != expected_pattern) {
            uint32_t err_idx = atomicAdd(error_count, 1u);
            if (err_idx == 0) {
                *first_error_addr = i;
            }
        }
    }
}

// ── Impl ─────────────────────────────────────────────────────────────

struct MemoryProbe::Impl {
    int device_index = -1;
    int sm_count = 0;

    platform::CudaDeviceBuffer d_test_region;
    platform::CudaDeviceBuffer d_flag;
    platform::CudaDeviceBuffer d_error_count;
    platform::CudaDeviceBuffer d_first_error_addr;

    platform::CudaPinnedBuffer h_flag;
    platform::CudaPinnedBuffer h_error_count;
    platform::CudaPinnedBuffer h_first_error_addr;

    platform::CudaEvent start_event{cudaEventDefault};
    platform::CudaEvent stop_event{cudaEventDefault};
};

MemoryProbe::MemoryProbe() : impl_(std::make_unique<Impl>()) {}
MemoryProbe::~MemoryProbe() = default;

bool MemoryProbe::initialize(int device_index) {
    try {
        platform::set_device(device_index);
        impl_->device_index = device_index;
        impl_->sm_count = get_sm_count(device_index);

        impl_->d_test_region = platform::CudaDeviceBuffer(kTestRegionSize);
        impl_->d_flag = platform::CudaDeviceBuffer(sizeof(int));
        impl_->d_error_count = platform::CudaDeviceBuffer(sizeof(uint32_t));
        impl_->d_first_error_addr = platform::CudaDeviceBuffer(sizeof(uint32_t));

        impl_->h_flag = platform::CudaPinnedBuffer(sizeof(int));
        impl_->h_error_count = platform::CudaPinnedBuffer(sizeof(uint32_t));
        impl_->h_first_error_addr = platform::CudaPinnedBuffer(sizeof(uint32_t));

        SENTINEL_LOG_INFO("Memory probe initialized on device {} ({} MB test region)",
                          device_index, kTestRegionSize / (1024 * 1024));
        return true;

    } catch (const platform::CudaError& e) {
        SENTINEL_LOG_ERROR("Memory probe init failed: {}", e.what());
        return false;
    }
}

ProbeResult MemoryProbe::execute(uint32_t sm_id, cudaStream_t stream) {
    ProbeResult result;
    result.probe_id = next_probe_id();
    result.probe_type = ProbeType::kMemory;
    result.sm_id = sm_id;
    result.expected_hash = {};  // Memory probe uses error counting, not hashing.

    try {
        platform::set_device(impl_->device_index);

        dim3 grid, block;
        compute_sm_pinned_launch_params(impl_->sm_count, 256, grid, block);

        uint32_t total_errors = 0;
        const uint32_t num_words = static_cast<uint32_t>(kTestRegionWords);

        auto reset_counters = [&]() {
            int zero_i = 0;
            uint32_t zero_u = 0;
            CUDA_CHECK(cudaMemcpyAsync(impl_->d_flag.get(), &zero_i, sizeof(int),
                                        cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(impl_->d_error_count.get(), &zero_u, sizeof(uint32_t),
                                        cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(impl_->d_first_error_addr.get(), &zero_u, sizeof(uint32_t),
                                        cudaMemcpyHostToDevice, stream));
        };

        auto read_errors = [&]() -> uint32_t {
            CUDA_CHECK(cudaMemcpyAsync(impl_->h_error_count.get(),
                                        impl_->d_error_count.get(),
                                        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return *impl_->h_error_count.as<uint32_t>();
        };

        impl_->start_event.record(stream);

        // M0: up(w0) — Write all zeros ascending.
        reset_counters();
        march_write_kernel<<<grid, block, 0, stream>>>(
            impl_->d_test_region.as<uint32_t>(), 0x00000000u, num_words,
            sm_id, impl_->d_flag.as<int>());

        // M1: up(r0,w1) — Read 0, write all-ones ascending.
        reset_counters();
        march_read_write_up_kernel<<<grid, block, 0, stream>>>(
            impl_->d_test_region.as<uint32_t>(), 0x00000000u, 0xFFFFFFFFu,
            num_words, sm_id, impl_->d_flag.as<int>(),
            impl_->d_error_count.as<uint32_t>(),
            impl_->d_first_error_addr.as<uint32_t>());
        total_errors += read_errors();

        // M2: up(r1,w0) — Read all-ones, write zeros ascending.
        reset_counters();
        march_read_write_up_kernel<<<grid, block, 0, stream>>>(
            impl_->d_test_region.as<uint32_t>(), 0xFFFFFFFFu, 0x00000000u,
            num_words, sm_id, impl_->d_flag.as<int>(),
            impl_->d_error_count.as<uint32_t>(),
            impl_->d_first_error_addr.as<uint32_t>());
        total_errors += read_errors();

        // M3: down(r0,w1) — Read 0, write all-ones descending.
        reset_counters();
        march_read_write_down_kernel<<<grid, block, 0, stream>>>(
            impl_->d_test_region.as<uint32_t>(), 0x00000000u, 0xFFFFFFFFu,
            num_words, sm_id, impl_->d_flag.as<int>(),
            impl_->d_error_count.as<uint32_t>(),
            impl_->d_first_error_addr.as<uint32_t>());
        total_errors += read_errors();

        // M4: down(r1,w0) — Read all-ones, write zeros descending.
        reset_counters();
        march_read_write_down_kernel<<<grid, block, 0, stream>>>(
            impl_->d_test_region.as<uint32_t>(), 0xFFFFFFFFu, 0x00000000u,
            num_words, sm_id, impl_->d_flag.as<int>(),
            impl_->d_error_count.as<uint32_t>(),
            impl_->d_first_error_addr.as<uint32_t>());
        total_errors += read_errors();

        // M5: up(r0) — Final verification read.
        reset_counters();
        march_verify_kernel<<<grid, block, 0, stream>>>(
            impl_->d_test_region.as<uint32_t>(), 0x00000000u,
            num_words, sm_id, impl_->d_flag.as<int>(),
            impl_->d_error_count.as<uint32_t>(),
            impl_->d_first_error_addr.as<uint32_t>());
        total_errors += read_errors();

        impl_->stop_event.record(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float elapsed_ms = platform::CudaEvent::elapsed_ms(
            impl_->start_event, impl_->stop_event);
        result.execution_time_ns = static_cast<uint64_t>(elapsed_ms * 1e6);

        result.match = (total_errors == 0);
        result.actual_hash = result.expected_hash;  // No hash for memory test.

        if (!result.match) {
            MismatchDetail detail;
            detail.byte_offset = 0;
            // Encode error count in expected/actual.
            uint32_t err_count = total_errors;
            detail.expected_value = {0, 0, 0, 0};
            detail.actual_value.resize(4);
            std::memcpy(detail.actual_value.data(), &err_count, 4);
            result.mismatch_details.push_back(std::move(detail));

            SENTINEL_LOG_WARN("Memory probe detected {} errors on SM {}",
                              total_errors, sm_id);
        }

        result.timestamp = std::chrono::system_clock::now();

    } catch (const platform::CudaError& e) {
        SENTINEL_LOG_ERROR("Memory probe failed on SM {}: {}", sm_id, e.what());
        result.match = false;
        result.timestamp = std::chrono::system_clock::now();
    }

    return result;
}

void MemoryProbe::teardown() {
    impl_->d_test_region = {};
    impl_->d_flag = {};
    impl_->d_error_count = {};
    impl_->d_first_error_addr = {};
    impl_->h_flag = {};
    impl_->h_error_count = {};
    impl_->h_first_error_addr = {};
}

std::size_t MemoryProbe::memory_footprint() const noexcept {
    return kTestRegionSize + sizeof(int) + sizeof(uint32_t) * 2  // device
         + sizeof(int) + sizeof(uint32_t) * 2;                  // pinned host
}

}  // namespace sentinel::probes
