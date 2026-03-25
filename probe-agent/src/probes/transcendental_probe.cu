/// @file transcendental_probe.cu
/// @brief Transcendental function probe CUDA implementation.

#include "probes/transcendental_probe.h"
#include "probes/sm_affinity.h"
#include "platform/cuda_runtime.h"
#include "util/crypto.h"
#include "util/logging.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace sentinel::probes {

// ── Device kernel ────────────────────────────────────────────────────

/// Transcendental probe kernel. Each thread computes transcendental functions
/// for a subset of input values. Only executes on the target SM.
__global__ void transcendental_probe_kernel(
        const float* __restrict__ inputs,    // kValuesPerFunction inputs
        float* __restrict__ results,         // kTotalOutputs results
        uint32_t values_per_func,
        uint32_t target_sm_id,
        int* __restrict__ executed_flag) {
    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    if (sm_id != target_sm_id) return;

    if (threadIdx.x == 0) atomicExch(executed_flag, 1);
    __syncthreads();

    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;

    for (uint32_t i = tid; i < values_per_func; i += stride) {
        float x = inputs[i];

        // Function 0: sinf
        results[0 * values_per_func + i] = __sinf(x);

        // Function 1: cosf
        results[1 * values_per_func + i] = __cosf(x);

        // Function 2: expf
        results[2 * values_per_func + i] = __expf(x);

        // Function 3: logf (use abs to avoid NaN on negative inputs)
        float abs_x = fabsf(x);
        results[3 * values_per_func + i] = (abs_x > 0.0f) ? __logf(abs_x) : -INFINITY;

        // Function 4: rsqrtf
        results[4 * values_per_func + i] = (abs_x > 0.0f) ? __frsqrt_rn(abs_x) : INFINITY;
    }
}

// ── Host-side implementation ────────────────────────────────────────

namespace {

/// Generate 256 input values covering key ranges for transcendental tests.
std::vector<float> generate_transcendental_inputs() {
    std::vector<float> inputs;
    inputs.reserve(TranscendentalProbe::kValuesPerFunction);

    // Uniformly spaced in [-pi, pi] for sin/cos (64 values).
    for (int i = 0; i < 64; ++i) {
        float t = -3.14159265358979f + (6.28318530717959f * static_cast<float>(i) / 63.0f);
        inputs.push_back(t);
    }

    // Small values near zero (32 values).
    for (int i = 0; i < 32; ++i) {
        float val = std::ldexp(1.0f, -20 + i);
        inputs.push_back(val);
    }

    // Moderate positive values for exp (32 values).
    for (int i = 0; i < 32; ++i) {
        inputs.push_back(static_cast<float>(i) * 0.25f);
    }

    // Negative values for exp (32 values).
    for (int i = 0; i < 32; ++i) {
        inputs.push_back(-static_cast<float>(i) * 0.25f);
    }

    // Values near exp overflow/underflow (16 values).
    for (int i = 0; i < 8; ++i) {
        inputs.push_back(80.0f + static_cast<float>(i));
        inputs.push_back(-80.0f - static_cast<float>(i));
    }

    // Large multiples of pi for sin/cos (16 values).
    for (int i = 0; i < 16; ++i) {
        inputs.push_back(3.14159265f * static_cast<float>(100 + i));
    }

    // Values for log/rsqrt: positive range (32 values).
    for (int i = 0; i < 32; ++i) {
        inputs.push_back(0.01f + static_cast<float>(i) * 10.0f);
    }

    // Pad to exactly kValuesPerFunction.
    while (inputs.size() < TranscendentalProbe::kValuesPerFunction) {
        inputs.push_back(static_cast<float>(inputs.size()) * 0.01f + 0.1f);
    }
    inputs.resize(TranscendentalProbe::kValuesPerFunction);

    return inputs;
}

/// Compute golden transcendental results using double precision.
/// Returns float results but computed via double to get the "most correct"
/// single-precision answer.
std::vector<float> compute_golden_transcendental(const std::vector<float>& inputs) {
    const uint32_t n = TranscendentalProbe::kValuesPerFunction;
    std::vector<float> results(TranscendentalProbe::kTotalOutputs);

    for (uint32_t i = 0; i < n; ++i) {
        double x = static_cast<double>(inputs[i]);

        results[0 * n + i] = static_cast<float>(std::sin(x));
        results[1 * n + i] = static_cast<float>(std::cos(x));
        results[2 * n + i] = static_cast<float>(std::exp(x));

        double abs_x = std::abs(x);
        results[3 * n + i] = (abs_x > 0.0) ? static_cast<float>(std::log(abs_x))
                                             : -std::numeric_limits<float>::infinity();
        results[4 * n + i] = (abs_x > 0.0) ? static_cast<float>(1.0 / std::sqrt(abs_x))
                                             : std::numeric_limits<float>::infinity();
    }
    return results;
}

/// Check if two floats are within 1 ULP of each other.
bool within_1_ulp(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
    if (a == b) return true;

    uint32_t a_bits, b_bits;
    std::memcpy(&a_bits, &a, sizeof(float));
    std::memcpy(&b_bits, &b, sizeof(float));

    // Handle sign difference.
    if ((a_bits >> 31) != (b_bits >> 31)) {
        // Both must be zero (positive and negative zero).
        return (a_bits & 0x7FFFFFFFu) == 0 && (b_bits & 0x7FFFFFFFu) == 0;
    }

    int64_t diff = static_cast<int64_t>(a_bits) - static_cast<int64_t>(b_bits);
    return std::abs(diff) <= 1;
}

}  // namespace

struct TranscendentalProbe::Impl {
    int device_index = -1;
    int sm_count = 0;

    std::vector<float> host_inputs;
    std::vector<float> golden_results;
    util::Sha256Digest golden_hash{};

    platform::CudaDeviceBuffer d_inputs;
    platform::CudaDeviceBuffer d_results;
    platform::CudaDeviceBuffer d_flag;

    platform::CudaPinnedBuffer h_results;
    platform::CudaPinnedBuffer h_flag;

    platform::CudaEvent start_event{cudaEventDefault};
    platform::CudaEvent stop_event{cudaEventDefault};
};

TranscendentalProbe::TranscendentalProbe() : impl_(std::make_unique<Impl>()) {}
TranscendentalProbe::~TranscendentalProbe() = default;

bool TranscendentalProbe::initialize(int device_index) {
    try {
        platform::set_device(device_index);
        impl_->device_index = device_index;
        impl_->sm_count = get_sm_count(device_index);

        impl_->host_inputs = generate_transcendental_inputs();
        impl_->golden_results = compute_golden_transcendental(impl_->host_inputs);

        // For transcendental probes we use the golden results hash but allow
        // 1 ULP tolerance in the comparison logic rather than strict hash match.
        impl_->golden_hash = util::sha256(
            reinterpret_cast<const uint8_t*>(impl_->golden_results.data()),
            impl_->golden_results.size() * sizeof(float));

        std::size_t input_bytes = kValuesPerFunction * sizeof(float);
        std::size_t result_bytes = kTotalOutputs * sizeof(float);

        impl_->d_inputs = platform::CudaDeviceBuffer(input_bytes);
        impl_->d_results = platform::CudaDeviceBuffer(result_bytes);
        impl_->d_flag = platform::CudaDeviceBuffer(sizeof(int));

        impl_->h_results = platform::CudaPinnedBuffer(result_bytes);
        impl_->h_flag = platform::CudaPinnedBuffer(sizeof(int));

        impl_->d_inputs.copy_from_host(impl_->host_inputs.data(), input_bytes);
        CUDA_CHECK(cudaDeviceSynchronize());

        SENTINEL_LOG_INFO("Transcendental probe initialized on device {} ({} SMs)",
                          device_index, impl_->sm_count);
        return true;

    } catch (const platform::CudaError& e) {
        SENTINEL_LOG_ERROR("Transcendental probe init failed: {}", e.what());
        return false;
    }
}

ProbeResult TranscendentalProbe::execute(uint32_t sm_id, cudaStream_t stream) {
    ProbeResult result;
    result.probe_id = next_probe_id();
    result.probe_type = ProbeType::kTranscendental;
    result.sm_id = sm_id;
    result.expected_hash = impl_->golden_hash;

    try {
        platform::set_device(impl_->device_index);

        int zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(impl_->d_flag.get(), &zero, sizeof(int),
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(impl_->d_results.get(), 0,
                                    kTotalOutputs * sizeof(float), stream));

        dim3 grid, block;
        compute_sm_pinned_launch_params(impl_->sm_count, 256, grid, block);

        impl_->start_event.record(stream);

        transcendental_probe_kernel<<<grid, block, 0, stream>>>(
            impl_->d_inputs.as<float>(),
            impl_->d_results.as<float>(),
            kValuesPerFunction,
            sm_id,
            impl_->d_flag.as<int>());

        impl_->stop_event.record(stream);

        impl_->d_results.copy_to_host(impl_->h_results.get(),
                                       kTotalOutputs * sizeof(float), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float elapsed_ms = platform::CudaEvent::elapsed_ms(
            impl_->start_event, impl_->stop_event);
        result.execution_time_ns = static_cast<uint64_t>(elapsed_ms * 1e6);

        // Hash actual results for the record (even though we use ULP comparison).
        result.actual_hash = util::sha256(
            reinterpret_cast<const uint8_t*>(impl_->h_results.get()),
            kTotalOutputs * sizeof(float));

        // Compare with 1 ULP tolerance.
        const float* actual = impl_->h_results.as<float>();
        result.match = true;

        for (uint32_t i = 0; i < kTotalOutputs; ++i) {
            if (!within_1_ulp(impl_->golden_results[i], actual[i])) {
                result.match = false;

                MismatchDetail detail;
                detail.byte_offset = static_cast<uint64_t>(i) * sizeof(float);
                detail.expected_value.resize(sizeof(float));
                detail.actual_value.resize(sizeof(float));
                uint32_t exp_bits, act_bits;
                std::memcpy(&exp_bits, &impl_->golden_results[i], sizeof(float));
                std::memcpy(&act_bits, &actual[i], sizeof(float));
                std::memcpy(detail.expected_value.data(), &exp_bits, sizeof(float));
                std::memcpy(detail.actual_value.data(), &act_bits, sizeof(float));

                uint32_t diff = exp_bits ^ act_bits;
                for (uint32_t bit = 0; bit < 32; ++bit) {
                    if (diff & (1u << bit)) detail.differing_bits.push_back(bit);
                }
                result.mismatch_details.push_back(std::move(detail));

                if (result.mismatch_details.size() >= 64) break;
            }
        }

        result.timestamp = std::chrono::system_clock::now();

    } catch (const platform::CudaError& e) {
        SENTINEL_LOG_ERROR("Transcendental probe failed on SM {}: {}", sm_id, e.what());
        result.match = false;
        result.timestamp = std::chrono::system_clock::now();
    }

    return result;
}

void TranscendentalProbe::teardown() {
    impl_->d_inputs = {};
    impl_->d_results = {};
    impl_->d_flag = {};
    impl_->h_results = {};
    impl_->h_flag = {};
}

std::size_t TranscendentalProbe::memory_footprint() const noexcept {
    return kValuesPerFunction * sizeof(float)   // d_inputs
         + kTotalOutputs * sizeof(float)        // d_results
         + sizeof(int)                          // d_flag
         + kTotalOutputs * sizeof(float)        // h_results
         + sizeof(int);                         // h_flag
}

}  // namespace sentinel::probes
