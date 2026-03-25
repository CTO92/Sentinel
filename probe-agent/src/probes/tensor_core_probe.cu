/// @file tensor_core_probe.cu
/// @brief Tensor Core probe CUDA kernel and host-side implementation.
///
/// Uses WMMA (Warp Matrix Multiply Accumulate) intrinsics to test
/// tensor core hardware with deterministic 16x16 matrix operations.

#include "probes/tensor_core_probe.h"
#include "probes/sm_affinity.h"
#include "platform/cuda_runtime.h"
#include "util/crypto.h"
#include "util/logging.h"

#include <cstring>
#include <vector>

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

namespace sentinel::probes {

// ── Device kernels ──────────────────────────────────────────────────

/// Tensor Core WMMA probe kernel. Each warp performs 16x16 matrix multiply
/// using WMMA, but only on the target SM. Other SMs exit immediately.
__global__ void tensor_core_probe_kernel(
        const half* __restrict__ matrices_a,     // kNumTestCases * 16 * 16
        const half* __restrict__ matrices_b,     // kNumTestCases * 16 * 16
        float* __restrict__ results_c,           // kNumTestCases * 16 * 16
        int num_test_cases,
        uint32_t target_sm_id,
        int* __restrict__ executed_flag) {
    // SM affinity check.
    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    if (sm_id != target_sm_id) {
        return;
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicExch(executed_flag, 1);
    }
    __syncthreads();

    // Each warp (32 threads) processes one test case.
    const int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / 32;

    if (warp_id >= num_test_cases) {
        return;
    }

    const int mat_size = 16 * 16;
    const half* a_ptr = matrices_a + warp_id * mat_size;
    const half* b_ptr = matrices_b + warp_id * mat_size;
    float* c_ptr = results_c + warp_id * mat_size;

    // Declare WMMA fragments.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Load input matrices.
    wmma::load_matrix_sync(a_frag, a_ptr, 16);
    wmma::load_matrix_sync(b_frag, b_ptr, 16);

    // Initialize accumulator to zero.
    wmma::fill_fragment(c_frag, 0.0f);

    // Perform matrix multiply-accumulate: C = A * B + 0.
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the result.
    wmma::store_matrix_sync(c_ptr, c_frag, 16, wmma::mem_row_major);
}

/// INT8 Tensor Core probe kernel using dp4a-style operations.
/// Tests INT8 matrix multiply on Turing+ architectures.
__global__ void tensor_core_int8_probe_kernel(
        const int8_t* __restrict__ matrices_a,   // kNumTestCases * 16 * 16
        const int8_t* __restrict__ matrices_b,   // kNumTestCases * 16 * 16
        int32_t* __restrict__ results_c,         // kNumTestCases * 16 * 16
        int num_test_cases,
        uint32_t target_sm_id,
        int* __restrict__ executed_flag) {
    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    if (sm_id != target_sm_id) {
        return;
    }

    if (threadIdx.x == 0) {
        atomicExch(executed_flag, 1);
    }
    __syncthreads();

    // Simple dp4a-based matrix multiply for INT8 verification.
    const int tid = threadIdx.x;
    const int mat_size = 16 * 16;

    for (int tc = 0; tc < num_test_cases; ++tc) {
        const int8_t* a = matrices_a + tc * mat_size;
        const int8_t* b = matrices_b + tc * mat_size;
        int32_t* c = results_c + tc * mat_size;

        // Each thread computes one or more elements of C.
        for (int elem = tid; elem < mat_size; elem += blockDim.x) {
            int row = elem / 16;
            int col = elem % 16;

            int32_t sum = 0;
            // Dot product of row from A and column from B.
            for (int k = 0; k < 16; k += 4) {
                // Pack 4 int8 values into int32 for dp4a.
                int32_t a_pack = 0;
                int32_t b_pack = 0;
                for (int j = 0; j < 4 && (k + j) < 16; ++j) {
                    a_pack |= (static_cast<uint32_t>(static_cast<uint8_t>(a[row * 16 + k + j])) << (j * 8));
                    b_pack |= (static_cast<uint32_t>(static_cast<uint8_t>(b[(k + j) * 16 + col])) << (j * 8));
                }
                // Use __dp4a intrinsic for int8 dot product.
                sum = __dp4a(a_pack, b_pack, sum);
            }
            c[elem] = sum;
        }
    }
}

// ── Host-side implementation ────────────────────────────────────────

namespace {

/// Generate FP16 identity matrix (16x16).
void make_identity_fp16(half* mat) {
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            mat[i * 16 + j] = (i == j) ? __float2half(1.0f) : __float2half(0.0f);
        }
    }
}

/// Generate FP16 permutation matrix (cyclic shift).
void make_permutation_fp16(half* mat) {
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            mat[i * 16 + j] = (j == ((i + 1) % 16)) ? __float2half(1.0f) : __float2half(0.0f);
        }
    }
}

/// Generate FP16 Hadamard-like matrix (scaled so entries are +/-1).
void make_hadamard_fp16(half* mat) {
    // Simple 16x16 Hadamard-like construction using Walsh matrix approach.
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            // Walsh-Hadamard: count number of 1-bits in i & j
            int bits = __builtin_popcount(i & j);
            float val = (bits % 2 == 0) ? 1.0f : -1.0f;
            mat[i * 16 + j] = __float2half(val);
        }
    }
}

/// Generate INT8 identity matrix.
void make_identity_int8(int8_t* mat) {
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            mat[i * 16 + j] = (i == j) ? 1 : 0;
        }
    }
}

/// Generate INT8 permutation matrix.
void make_permutation_int8(int8_t* mat) {
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            mat[i * 16 + j] = (j == ((i + 1) % 16)) ? 1 : 0;
        }
    }
}

/// Generate INT8 matrix with small values for testing.
void make_small_values_int8(int8_t* mat) {
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            mat[i * 16 + j] = static_cast<int8_t>((i * 16 + j) % 7 - 3);
        }
    }
}

/// Host-side FP16 matrix multiply reference for golden values.
void matmul_fp16_ref(const half* a, const half* b, float* c) {
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 16; ++k) {
                sum += __half2float(a[i * 16 + k]) * __half2float(b[k * 16 + j]);
            }
            c[i * 16 + j] = sum;
        }
    }
}

/// Host-side INT8 matrix multiply reference.
void matmul_int8_ref(const int8_t* a, const int8_t* b, int32_t* c) {
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            int32_t sum = 0;
            for (int k = 0; k < 16; ++k) {
                sum += static_cast<int32_t>(a[i * 16 + k]) * static_cast<int32_t>(b[k * 16 + j]);
            }
            c[i * 16 + j] = sum;
        }
    }
}

}  // namespace

struct TensorCoreProbe::Impl {
    int device_index = -1;
    int sm_count = 0;

    // FP16 test data.
    static constexpr int kFp16TestCases = 3;  // identity, permutation, hadamard
    std::vector<half> h_matrices_a_fp16;   // kFp16TestCases * 256
    std::vector<half> h_matrices_b_fp16;
    std::vector<float> golden_results_fp16;

    // INT8 test data.
    static constexpr int kInt8TestCases = 3;
    std::vector<int8_t> h_matrices_a_int8;
    std::vector<int8_t> h_matrices_b_int8;
    std::vector<int32_t> golden_results_int8;

    util::Sha256Digest golden_hash{};

    // Device buffers.
    platform::CudaDeviceBuffer d_a_fp16, d_b_fp16, d_c_fp16;
    platform::CudaDeviceBuffer d_a_int8, d_b_int8, d_c_int8;
    platform::CudaDeviceBuffer d_flag;

    // Pinned host.
    platform::CudaPinnedBuffer h_c_fp16, h_c_int8, h_flag;

    platform::CudaEvent start_event{cudaEventDefault};
    platform::CudaEvent stop_event{cudaEventDefault};
};

TensorCoreProbe::TensorCoreProbe() : impl_(std::make_unique<Impl>()) {}
TensorCoreProbe::~TensorCoreProbe() = default;

bool TensorCoreProbe::initialize(int device_index) {
    try {
        platform::set_device(device_index);
        impl_->device_index = device_index;
        impl_->sm_count = get_sm_count(device_index);

        const int mat_elems = kMatDim * kMatDim;

        // --- FP16 test matrices ---
        impl_->h_matrices_a_fp16.resize(Impl::kFp16TestCases * mat_elems);
        impl_->h_matrices_b_fp16.resize(Impl::kFp16TestCases * mat_elems);
        impl_->golden_results_fp16.resize(Impl::kFp16TestCases * mat_elems);

        // Test case 0: Identity * Identity = Identity
        make_identity_fp16(impl_->h_matrices_a_fp16.data());
        make_identity_fp16(impl_->h_matrices_b_fp16.data());

        // Test case 1: Permutation * Identity = Permutation
        make_permutation_fp16(impl_->h_matrices_a_fp16.data() + mat_elems);
        make_identity_fp16(impl_->h_matrices_b_fp16.data() + mat_elems);

        // Test case 2: Hadamard * Hadamard = 16 * Identity
        make_hadamard_fp16(impl_->h_matrices_a_fp16.data() + 2 * mat_elems);
        make_hadamard_fp16(impl_->h_matrices_b_fp16.data() + 2 * mat_elems);

        // Compute golden FP16 results.
        for (int tc = 0; tc < Impl::kFp16TestCases; ++tc) {
            matmul_fp16_ref(
                impl_->h_matrices_a_fp16.data() + tc * mat_elems,
                impl_->h_matrices_b_fp16.data() + tc * mat_elems,
                impl_->golden_results_fp16.data() + tc * mat_elems);
        }

        // --- INT8 test matrices ---
        impl_->h_matrices_a_int8.resize(Impl::kInt8TestCases * mat_elems);
        impl_->h_matrices_b_int8.resize(Impl::kInt8TestCases * mat_elems);
        impl_->golden_results_int8.resize(Impl::kInt8TestCases * mat_elems);

        make_identity_int8(impl_->h_matrices_a_int8.data());
        make_identity_int8(impl_->h_matrices_b_int8.data());

        make_permutation_int8(impl_->h_matrices_a_int8.data() + mat_elems);
        make_identity_int8(impl_->h_matrices_b_int8.data() + mat_elems);

        make_small_values_int8(impl_->h_matrices_a_int8.data() + 2 * mat_elems);
        make_identity_int8(impl_->h_matrices_b_int8.data() + 2 * mat_elems);

        for (int tc = 0; tc < Impl::kInt8TestCases; ++tc) {
            matmul_int8_ref(
                impl_->h_matrices_a_int8.data() + tc * mat_elems,
                impl_->h_matrices_b_int8.data() + tc * mat_elems,
                impl_->golden_results_int8.data() + tc * mat_elems);
        }

        // Compute combined golden hash over FP16 and INT8 results.
        util::Sha256Hasher hasher;
        hasher.update(reinterpret_cast<const uint8_t*>(impl_->golden_results_fp16.data()),
                      impl_->golden_results_fp16.size() * sizeof(float));
        hasher.update(reinterpret_cast<const uint8_t*>(impl_->golden_results_int8.data()),
                      impl_->golden_results_int8.size() * sizeof(int32_t));
        impl_->golden_hash = hasher.finalize();

        // --- Allocate device memory ---
        std::size_t fp16_mat_bytes = Impl::kFp16TestCases * mat_elems * sizeof(half);
        std::size_t fp16_res_bytes = Impl::kFp16TestCases * mat_elems * sizeof(float);
        std::size_t int8_mat_bytes = Impl::kInt8TestCases * mat_elems * sizeof(int8_t);
        std::size_t int8_res_bytes = Impl::kInt8TestCases * mat_elems * sizeof(int32_t);

        impl_->d_a_fp16 = platform::CudaDeviceBuffer(fp16_mat_bytes);
        impl_->d_b_fp16 = platform::CudaDeviceBuffer(fp16_mat_bytes);
        impl_->d_c_fp16 = platform::CudaDeviceBuffer(fp16_res_bytes);
        impl_->d_a_int8 = platform::CudaDeviceBuffer(int8_mat_bytes);
        impl_->d_b_int8 = platform::CudaDeviceBuffer(int8_mat_bytes);
        impl_->d_c_int8 = platform::CudaDeviceBuffer(int8_res_bytes);
        impl_->d_flag = platform::CudaDeviceBuffer(sizeof(int));

        impl_->h_c_fp16 = platform::CudaPinnedBuffer(fp16_res_bytes);
        impl_->h_c_int8 = platform::CudaPinnedBuffer(int8_res_bytes);
        impl_->h_flag = platform::CudaPinnedBuffer(sizeof(int));

        // Upload inputs.
        impl_->d_a_fp16.copy_from_host(impl_->h_matrices_a_fp16.data(), fp16_mat_bytes);
        impl_->d_b_fp16.copy_from_host(impl_->h_matrices_b_fp16.data(), fp16_mat_bytes);
        impl_->d_a_int8.copy_from_host(impl_->h_matrices_a_int8.data(), int8_mat_bytes);
        impl_->d_b_int8.copy_from_host(impl_->h_matrices_b_int8.data(), int8_mat_bytes);
        CUDA_CHECK(cudaDeviceSynchronize());

        SENTINEL_LOG_INFO("Tensor Core probe initialized on device {} ({} SMs)",
                          device_index, impl_->sm_count);
        return true;

    } catch (const platform::CudaError& e) {
        SENTINEL_LOG_ERROR("Tensor Core probe init failed: {}", e.what());
        return false;
    }
}

ProbeResult TensorCoreProbe::execute(uint32_t sm_id, cudaStream_t stream) {
    ProbeResult result;
    result.probe_id = next_probe_id();
    result.probe_type = ProbeType::kTensorCore;
    result.sm_id = sm_id;
    result.expected_hash = impl_->golden_hash;

    try {
        platform::set_device(impl_->device_index);
        const int mat_elems = kMatDim * kMatDim;

        // Clear flag.
        int zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(impl_->d_flag.get(), &zero, sizeof(int),
                                    cudaMemcpyHostToDevice, stream));

        // Launch FP16 WMMA kernel.
        // WMMA requires at least one full warp (32 threads). We need one
        // warp per test case.
        dim3 grid, block;
        compute_sm_pinned_launch_params(impl_->sm_count, 32 * Impl::kFp16TestCases, grid, block);
        // Reshape block as 2D: warps along y.
        block = dim3(32, Impl::kFp16TestCases, 1);

        impl_->start_event.record(stream);

        tensor_core_probe_kernel<<<grid, block, 0, stream>>>(
            impl_->d_a_fp16.as<half>(),
            impl_->d_b_fp16.as<half>(),
            impl_->d_c_fp16.as<float>(),
            Impl::kFp16TestCases,
            sm_id,
            impl_->d_flag.as<int>());

        // Launch INT8 kernel.
        dim3 grid2, block2;
        compute_sm_pinned_launch_params(impl_->sm_count, 256, grid2, block2);

        // Reset flag for INT8 kernel.
        CUDA_CHECK(cudaMemcpyAsync(impl_->d_flag.get(), &zero, sizeof(int),
                                    cudaMemcpyHostToDevice, stream));

        tensor_core_int8_probe_kernel<<<grid2, block2, 0, stream>>>(
            impl_->d_a_int8.as<int8_t>(),
            impl_->d_b_int8.as<int8_t>(),
            impl_->d_c_int8.as<int32_t>(),
            Impl::kInt8TestCases,
            sm_id,
            impl_->d_flag.as<int>());

        impl_->stop_event.record(stream);

        // Readback.
        std::size_t fp16_res_bytes = Impl::kFp16TestCases * mat_elems * sizeof(float);
        std::size_t int8_res_bytes = Impl::kInt8TestCases * mat_elems * sizeof(int32_t);
        impl_->d_c_fp16.copy_to_host(impl_->h_c_fp16.get(), fp16_res_bytes, stream);
        impl_->d_c_int8.copy_to_host(impl_->h_c_int8.get(), int8_res_bytes, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float elapsed_ms = platform::CudaEvent::elapsed_ms(
            impl_->start_event, impl_->stop_event);
        result.execution_time_ns = static_cast<uint64_t>(elapsed_ms * 1e6);

        // Hash actual results (combined FP16 + INT8).
        util::Sha256Hasher hasher;
        hasher.update(reinterpret_cast<const uint8_t*>(impl_->h_c_fp16.get()), fp16_res_bytes);
        hasher.update(reinterpret_cast<const uint8_t*>(impl_->h_c_int8.get()), int8_res_bytes);
        result.actual_hash = hasher.finalize();

        result.match = util::digest_equal(result.expected_hash, result.actual_hash);

        if (!result.match) {
            // Collect FP16 mismatches.
            const float* actual_fp16 = impl_->h_c_fp16.as<float>();
            for (int i = 0; i < Impl::kFp16TestCases * mat_elems && result.mismatch_details.size() < 64; ++i) {
                uint32_t exp_bits, act_bits;
                std::memcpy(&exp_bits, &impl_->golden_results_fp16[i], sizeof(float));
                std::memcpy(&act_bits, &actual_fp16[i], sizeof(float));
                if (exp_bits != act_bits) {
                    MismatchDetail d;
                    d.byte_offset = static_cast<uint64_t>(i) * sizeof(float);
                    d.expected_value.resize(4);
                    d.actual_value.resize(4);
                    std::memcpy(d.expected_value.data(), &exp_bits, 4);
                    std::memcpy(d.actual_value.data(), &act_bits, 4);
                    uint32_t diff = exp_bits ^ act_bits;
                    for (uint32_t bit = 0; bit < 32; ++bit) {
                        if (diff & (1u << bit)) d.differing_bits.push_back(bit);
                    }
                    result.mismatch_details.push_back(std::move(d));
                }
            }
        }

        result.timestamp = std::chrono::system_clock::now();

    } catch (const platform::CudaError& e) {
        SENTINEL_LOG_ERROR("Tensor Core probe failed on SM {}: {}", sm_id, e.what());
        result.match = false;
        result.timestamp = std::chrono::system_clock::now();
    }

    return result;
}

void TensorCoreProbe::teardown() {
    impl_->d_a_fp16 = {};
    impl_->d_b_fp16 = {};
    impl_->d_c_fp16 = {};
    impl_->d_a_int8 = {};
    impl_->d_b_int8 = {};
    impl_->d_c_int8 = {};
    impl_->d_flag = {};
    impl_->h_c_fp16 = {};
    impl_->h_c_int8 = {};
    impl_->h_flag = {};
}

std::size_t TensorCoreProbe::memory_footprint() const noexcept {
    const int mat_elems = kMatDim * kMatDim;
    return (Impl::kFp16TestCases * mat_elems * sizeof(half) * 2)    // A, B fp16
         + (Impl::kFp16TestCases * mat_elems * sizeof(float))       // C fp16
         + (Impl::kInt8TestCases * mat_elems * sizeof(int8_t) * 2)  // A, B int8
         + (Impl::kInt8TestCases * mat_elems * sizeof(int32_t))     // C int8
         + sizeof(int);                                              // flag
}

}  // namespace sentinel::probes
