/// @file register_file_probe.cu
/// @brief Register file probe CUDA implementation.
///
/// Uses inline PTX to write patterns directly into GPU registers and
/// read them back, bypassing compiler register allocation to test the
/// physical register file of each SM.

#include "probes/register_file_probe.h"
#include "probes/sm_affinity.h"
#include "platform/cuda_runtime.h"
#include "util/crypto.h"
#include "util/logging.h"

#include <cstring>
#include <vector>

namespace sentinel::probes {

// ── Device kernel ────────────────────────────────────────────────────

/// Register file probe kernel. Writes and reads patterns via inline PTX.
/// Each thread writes all patterns into registers and reads them back.
/// Results are written to output memory for host-side verification.
__global__ void register_file_probe_kernel(
        const uint32_t* __restrict__ patterns,   // kNumPatterns patterns
        uint32_t* __restrict__ results,           // kNumPatterns * blockDim.x results
        uint32_t num_patterns,
        uint32_t target_sm_id,
        int* __restrict__ executed_flag) {
    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    if (sm_id != target_sm_id) return;

    if (threadIdx.x == 0) atomicExch(executed_flag, 1);
    __syncthreads();

    const uint32_t tid = threadIdx.x;
    const uint32_t out_base = tid * num_patterns;

    // Process each pattern: write to a register via PTX, read back, store.
    for (uint32_t p = 0; p < num_patterns; ++p) {
        uint32_t pattern = patterns[p];
        uint32_t readback;

        // Write pattern to register and read it back using inline PTX.
        // The mov instruction tests the register write and read paths.
        asm volatile(
            "mov.u32 %0, %1;\n\t"  // Write pattern to register.
            : "=r"(readback)
            : "r"(pattern)
        );

        // Additional register-to-register moves to stress more of the
        // register file.
        uint32_t r1, r2, r3, r4;
        asm volatile(
            "mov.u32 %0, %4;\n\t"
            "mov.u32 %1, %0;\n\t"
            "mov.u32 %2, %1;\n\t"
            "mov.u32 %3, %2;\n\t"
            : "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4)
            : "r"(pattern)
        );

        // XOR chain to verify all intermediate values match.
        uint32_t check = readback ^ r1 ^ r2 ^ r3 ^ r4;
        // If all are equal to pattern: readback == pattern, r1..r4 == pattern.
        // check = pattern ^ pattern ^ pattern ^ pattern ^ pattern = pattern (odd count).
        // So check should equal pattern if all passed.
        results[out_base + p] = check;
    }
}

// ── Host-side implementation ────────────────────────────────────────

namespace {

/// Generate register test patterns.
std::vector<uint32_t> generate_register_patterns() {
    std::vector<uint32_t> patterns;
    patterns.reserve(RegisterFileProbe::kNumPatterns);

    // All zeros and all ones.
    patterns.push_back(0x00000000u);
    patterns.push_back(0xFFFFFFFFu);

    // Walking ones (32 patterns).
    for (int i = 0; i < 32; ++i) {
        patterns.push_back(1u << i);
    }

    // Walking zeros (32 patterns).
    for (int i = 0; i < 32; ++i) {
        patterns.push_back(~(1u << i));
    }

    // Alternating bit patterns.
    patterns.push_back(0x55555555u);
    patterns.push_back(0xAAAAAAAAu);
    patterns.push_back(0x33333333u);
    patterns.push_back(0xCCCCCCCCu);
    patterns.push_back(0x0F0F0F0Fu);
    patterns.push_back(0xF0F0F0F0u);
    patterns.push_back(0x00FF00FFu);
    patterns.push_back(0xFF00FF00u);
    patterns.push_back(0x0000FFFFu);
    patterns.push_back(0xFFFF0000u);

    // Byte-boundary patterns.
    patterns.push_back(0x01010101u);
    patterns.push_back(0x80808080u);
    patterns.push_back(0x7F7F7F7Fu);
    patterns.push_back(0xFEFEFEFEu);

    // Word-boundary patterns.
    patterns.push_back(0x00010001u);
    patterns.push_back(0x80008000u);
    patterns.push_back(0x7FFF7FFFu);
    patterns.push_back(0xFFFEFFFEu);

    // Checkerboard variants.
    patterns.push_back(0x12345678u);
    patterns.push_back(0x87654321u);
    patterns.push_back(0xDEADBEEFu);
    patterns.push_back(0xCAFEBABEu);

    // Pad to kNumPatterns.
    while (patterns.size() < RegisterFileProbe::kNumPatterns) {
        uint32_t idx = static_cast<uint32_t>(patterns.size());
        patterns.push_back(idx * 0x9E3779B9u);  // Golden ratio hash.
    }
    patterns.resize(RegisterFileProbe::kNumPatterns);

    return patterns;
}

/// Compute expected results for register probe verification.
/// For the XOR chain: readback ^ r1 ^ r2 ^ r3 ^ r4.
/// If all 5 registers hold pattern: pattern ^ pattern ^ pattern ^ pattern ^ pattern = pattern.
std::vector<uint32_t> compute_golden_register_results(
        const std::vector<uint32_t>& patterns, int threads_per_block) {
    std::vector<uint32_t> results(patterns.size() * threads_per_block);
    for (int t = 0; t < threads_per_block; ++t) {
        for (std::size_t p = 0; p < patterns.size(); ++p) {
            results[t * patterns.size() + p] = patterns[p];
        }
    }
    return results;
}

}  // namespace

struct RegisterFileProbe::Impl {
    int device_index = -1;
    int sm_count = 0;
    static constexpr int kThreadsPerBlock = 128;

    std::vector<uint32_t> host_patterns;
    std::vector<uint32_t> golden_results;
    util::Sha256Digest golden_hash{};

    platform::CudaDeviceBuffer d_patterns;
    platform::CudaDeviceBuffer d_results;
    platform::CudaDeviceBuffer d_flag;

    platform::CudaPinnedBuffer h_results;
    platform::CudaPinnedBuffer h_flag;

    platform::CudaEvent start_event{cudaEventDefault};
    platform::CudaEvent stop_event{cudaEventDefault};
};

RegisterFileProbe::RegisterFileProbe() : impl_(std::make_unique<Impl>()) {}
RegisterFileProbe::~RegisterFileProbe() = default;

bool RegisterFileProbe::initialize(int device_index) {
    try {
        platform::set_device(device_index);
        impl_->device_index = device_index;
        impl_->sm_count = get_sm_count(device_index);

        impl_->host_patterns = generate_register_patterns();
        impl_->golden_results = compute_golden_register_results(
            impl_->host_patterns, Impl::kThreadsPerBlock);

        impl_->golden_hash = util::sha256(
            reinterpret_cast<const uint8_t*>(impl_->golden_results.data()),
            impl_->golden_results.size() * sizeof(uint32_t));

        std::size_t pat_bytes = kNumPatterns * sizeof(uint32_t);
        std::size_t res_bytes = kNumPatterns * Impl::kThreadsPerBlock * sizeof(uint32_t);

        impl_->d_patterns = platform::CudaDeviceBuffer(pat_bytes);
        impl_->d_results = platform::CudaDeviceBuffer(res_bytes);
        impl_->d_flag = platform::CudaDeviceBuffer(sizeof(int));

        impl_->h_results = platform::CudaPinnedBuffer(res_bytes);
        impl_->h_flag = platform::CudaPinnedBuffer(sizeof(int));

        impl_->d_patterns.copy_from_host(impl_->host_patterns.data(), pat_bytes);
        CUDA_CHECK(cudaDeviceSynchronize());

        SENTINEL_LOG_INFO("Register file probe initialized on device {} ({} patterns)",
                          device_index, kNumPatterns);
        return true;

    } catch (const platform::CudaError& e) {
        SENTINEL_LOG_ERROR("Register file probe init failed: {}", e.what());
        return false;
    }
}

ProbeResult RegisterFileProbe::execute(uint32_t sm_id, cudaStream_t stream) {
    ProbeResult result;
    result.probe_id = next_probe_id();
    result.probe_type = ProbeType::kRegisterFile;
    result.sm_id = sm_id;
    result.expected_hash = impl_->golden_hash;

    try {
        platform::set_device(impl_->device_index);

        std::size_t res_bytes = kNumPatterns * Impl::kThreadsPerBlock * sizeof(uint32_t);

        int zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(impl_->d_flag.get(), &zero, sizeof(int),
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(impl_->d_results.get(), 0, res_bytes, stream));

        dim3 grid, block;
        compute_sm_pinned_launch_params(impl_->sm_count, Impl::kThreadsPerBlock, grid, block);

        impl_->start_event.record(stream);

        register_file_probe_kernel<<<grid, block, 0, stream>>>(
            impl_->d_patterns.as<uint32_t>(),
            impl_->d_results.as<uint32_t>(),
            kNumPatterns, sm_id,
            impl_->d_flag.as<int>());

        impl_->stop_event.record(stream);

        impl_->d_results.copy_to_host(impl_->h_results.get(), res_bytes, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float elapsed_ms = platform::CudaEvent::elapsed_ms(
            impl_->start_event, impl_->stop_event);
        result.execution_time_ns = static_cast<uint64_t>(elapsed_ms * 1e6);

        result.actual_hash = util::sha256(
            reinterpret_cast<const uint8_t*>(impl_->h_results.get()), res_bytes);

        result.match = util::digest_equal(result.expected_hash, result.actual_hash);

        if (!result.match) {
            const uint32_t* actual = impl_->h_results.as<uint32_t>();
            for (std::size_t i = 0; i < impl_->golden_results.size() && result.mismatch_details.size() < 64; ++i) {
                if (actual[i] != impl_->golden_results[i]) {
                    MismatchDetail d;
                    d.byte_offset = i * sizeof(uint32_t);
                    d.expected_value.resize(4);
                    d.actual_value.resize(4);
                    std::memcpy(d.expected_value.data(), &impl_->golden_results[i], 4);
                    std::memcpy(d.actual_value.data(), &actual[i], 4);
                    uint32_t diff = actual[i] ^ impl_->golden_results[i];
                    for (uint32_t bit = 0; bit < 32; ++bit) {
                        if (diff & (1u << bit)) d.differing_bits.push_back(bit);
                    }
                    result.mismatch_details.push_back(std::move(d));
                }
            }
        }

        result.timestamp = std::chrono::system_clock::now();

    } catch (const platform::CudaError& e) {
        SENTINEL_LOG_ERROR("Register file probe failed on SM {}: {}", sm_id, e.what());
        result.match = false;
        result.timestamp = std::chrono::system_clock::now();
    }

    return result;
}

void RegisterFileProbe::teardown() {
    impl_->d_patterns = {};
    impl_->d_results = {};
    impl_->d_flag = {};
    impl_->h_results = {};
    impl_->h_flag = {};
}

std::size_t RegisterFileProbe::memory_footprint() const noexcept {
    std::size_t res_count = kNumPatterns * Impl::kThreadsPerBlock;
    return kNumPatterns * sizeof(uint32_t)         // d_patterns
         + res_count * sizeof(uint32_t)            // d_results
         + sizeof(int)                             // d_flag
         + res_count * sizeof(uint32_t)            // h_results
         + sizeof(int);                            // h_flag
}

}  // namespace sentinel::probes
