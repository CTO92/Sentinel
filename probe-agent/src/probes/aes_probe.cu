/// @file aes_probe.cu
/// @brief AES-128-ECB probe CUDA implementation.
///
/// Implements AES-128 encryption entirely in GPU registers using the
/// standard Rijndael S-box and key schedule. The S-box is stored in
/// constant memory for fast access.

#include "probes/aes_probe.h"
#include "probes/sm_affinity.h"
#include "platform/cuda_runtime.h"
#include "util/crypto.h"
#include "util/logging.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace sentinel::probes {

// ── AES S-box in constant memory ────────────────────────────────────

__constant__ uint8_t d_sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

// AES round constants.
__constant__ uint8_t d_rcon[10] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

// ── Device-side AES-128 implementation ──────────────────────────────

__device__ void aes_sub_bytes(uint8_t state[16]) {
    for (int i = 0; i < 16; ++i) {
        state[i] = d_sbox[state[i]];
    }
}

__device__ void aes_shift_rows(uint8_t state[16]) {
    uint8_t tmp;
    // Row 1: shift left by 1.
    tmp = state[1];
    state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = tmp;
    // Row 2: shift left by 2.
    tmp = state[2]; state[2] = state[10]; state[10] = tmp;
    tmp = state[6]; state[6] = state[14]; state[14] = tmp;
    // Row 3: shift left by 3.
    tmp = state[15];
    state[15] = state[11]; state[11] = state[7]; state[7] = state[3]; state[3] = tmp;
}

__device__ uint8_t xtime(uint8_t a) {
    return static_cast<uint8_t>((a << 1) ^ (((a >> 7) & 1) * 0x1b));
}

__device__ void aes_mix_columns(uint8_t state[16]) {
    for (int i = 0; i < 4; ++i) {
        int c = i * 4;
        uint8_t a0 = state[c], a1 = state[c+1], a2 = state[c+2], a3 = state[c+3];
        uint8_t h = a0 ^ a1 ^ a2 ^ a3;
        state[c]   = a0 ^ h ^ xtime(a0 ^ a1);
        state[c+1] = a1 ^ h ^ xtime(a1 ^ a2);
        state[c+2] = a2 ^ h ^ xtime(a2 ^ a3);
        state[c+3] = a3 ^ h ^ xtime(a3 ^ a0);
    }
}

__device__ void aes_add_round_key(uint8_t state[16], const uint8_t* round_key) {
    for (int i = 0; i < 16; ++i) {
        state[i] ^= round_key[i];
    }
}

__device__ void aes_key_expansion(const uint8_t key[16], uint8_t expanded[176]) {
    // Copy original key.
    for (int i = 0; i < 16; ++i) expanded[i] = key[i];

    for (int i = 4; i < 44; ++i) {
        uint8_t temp[4];
        int k = (i - 1) * 4;
        for (int j = 0; j < 4; ++j) temp[j] = expanded[k + j];

        if (i % 4 == 0) {
            // RotWord + SubWord + Rcon
            uint8_t t = temp[0];
            temp[0] = d_sbox[temp[1]] ^ d_rcon[i / 4 - 1];
            temp[1] = d_sbox[temp[2]];
            temp[2] = d_sbox[temp[3]];
            temp[3] = d_sbox[t];
        }

        int base = (i - 4) * 4;
        for (int j = 0; j < 4; ++j) {
            expanded[i * 4 + j] = expanded[base + j] ^ temp[j];
        }
    }
}

__device__ void aes_encrypt_block(const uint8_t plaintext[16],
                                    uint8_t ciphertext[16],
                                    const uint8_t expanded_key[176]) {
    uint8_t state[16];
    for (int i = 0; i < 16; ++i) state[i] = plaintext[i];

    aes_add_round_key(state, expanded_key);

    for (int round = 1; round < 10; ++round) {
        aes_sub_bytes(state);
        aes_shift_rows(state);
        aes_mix_columns(state);
        aes_add_round_key(state, expanded_key + round * 16);
    }

    aes_sub_bytes(state);
    aes_shift_rows(state);
    aes_add_round_key(state, expanded_key + 160);

    for (int i = 0; i < 16; ++i) ciphertext[i] = state[i];
}

// ── AES probe kernel ────────────────────────────────────────────────

__global__ void aes_probe_kernel(
        const uint8_t* __restrict__ plaintext,   // 4096 bytes
        uint8_t* __restrict__ ciphertext,         // 4096 bytes
        const uint8_t* __restrict__ key,          // 16 bytes
        uint32_t num_blocks,
        uint32_t target_sm_id,
        int* __restrict__ executed_flag) {
    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    if (sm_id != target_sm_id) return;

    if (threadIdx.x == 0) atomicExch(executed_flag, 1);
    __syncthreads();

    // Expand key in registers (each thread does this — small cost).
    uint8_t expanded_key[176];
    aes_key_expansion(key, expanded_key);

    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;

    // Each thread encrypts one or more 16-byte blocks.
    for (uint32_t blk = tid; blk < num_blocks; blk += stride) {
        aes_encrypt_block(
            plaintext + blk * 16,
            ciphertext + blk * 16,
            expanded_key);
    }
}

// ── Host-side AES reference ─────────────────────────────────────────

namespace {

static const uint8_t h_sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

static const uint8_t h_rcon[10] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

void host_aes_key_expansion(const uint8_t key[16], uint8_t expanded[176]) {
    std::memcpy(expanded, key, 16);
    for (int i = 4; i < 44; ++i) {
        uint8_t temp[4];
        std::memcpy(temp, expanded + (i - 1) * 4, 4);
        if (i % 4 == 0) {
            uint8_t t = temp[0];
            temp[0] = h_sbox[temp[1]] ^ h_rcon[i / 4 - 1];
            temp[1] = h_sbox[temp[2]];
            temp[2] = h_sbox[temp[3]];
            temp[3] = h_sbox[t];
        }
        for (int j = 0; j < 4; ++j) {
            expanded[i * 4 + j] = expanded[(i - 4) * 4 + j] ^ temp[j];
        }
    }
}

uint8_t host_xtime(uint8_t a) {
    return static_cast<uint8_t>((a << 1) ^ (((a >> 7) & 1) * 0x1b));
}

void host_aes_encrypt_block(const uint8_t in[16], uint8_t out[16],
                             const uint8_t expanded_key[176]) {
    uint8_t state[16];
    std::memcpy(state, in, 16);

    for (int i = 0; i < 16; ++i) state[i] ^= expanded_key[i];

    for (int round = 1; round < 10; ++round) {
        for (int i = 0; i < 16; ++i) state[i] = h_sbox[state[i]];

        uint8_t tmp;
        tmp = state[1]; state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = tmp;
        tmp = state[2]; state[2] = state[10]; state[10] = tmp;
        tmp = state[6]; state[6] = state[14]; state[14] = tmp;
        tmp = state[15]; state[15] = state[11]; state[11] = state[7]; state[7] = state[3]; state[3] = tmp;

        for (int i = 0; i < 4; ++i) {
            int c = i * 4;
            uint8_t a0 = state[c], a1 = state[c+1], a2 = state[c+2], a3 = state[c+3];
            uint8_t h = a0 ^ a1 ^ a2 ^ a3;
            state[c]   = a0 ^ h ^ host_xtime(a0 ^ a1);
            state[c+1] = a1 ^ h ^ host_xtime(a1 ^ a2);
            state[c+2] = a2 ^ h ^ host_xtime(a2 ^ a3);
            state[c+3] = a3 ^ h ^ host_xtime(a3 ^ a0);
        }

        for (int i = 0; i < 16; ++i) state[i] ^= expanded_key[round * 16 + i];
    }

    for (int i = 0; i < 16; ++i) state[i] = h_sbox[state[i]];
    uint8_t tmp;
    tmp = state[1]; state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = tmp;
    tmp = state[2]; state[2] = state[10]; state[10] = tmp;
    tmp = state[6]; state[6] = state[14]; state[14] = tmp;
    tmp = state[15]; state[15] = state[11]; state[11] = state[7]; state[7] = state[3]; state[3] = tmp;
    for (int i = 0; i < 16; ++i) state[i] ^= expanded_key[160 + i];

    std::memcpy(out, state, 16);
}

/// Fixed AES-128 key for the probe (deterministic, not secret).
constexpr uint8_t kProbeKey[16] = {
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c
};

/// Generate deterministic 4KB plaintext.
std::vector<uint8_t> generate_plaintext() {
    std::vector<uint8_t> pt(AesProbe::kDataSize);
    for (uint32_t i = 0; i < AesProbe::kDataSize; ++i) {
        // Deterministic pattern using a simple PRNG-like sequence.
        pt[i] = static_cast<uint8_t>((i * 0x9E3779B9u + 0x6A09E667u) >> 24);
    }
    return pt;
}

/// Compute golden AES-128-ECB ciphertext on host.
std::vector<uint8_t> compute_golden_ciphertext(const std::vector<uint8_t>& plaintext) {
    std::vector<uint8_t> ct(plaintext.size());
    uint8_t expanded_key[176];
    host_aes_key_expansion(kProbeKey, expanded_key);

    for (uint32_t blk = 0; blk < AesProbe::kNumBlocks; ++blk) {
        host_aes_encrypt_block(
            plaintext.data() + blk * 16,
            ct.data() + blk * 16,
            expanded_key);
    }
    return ct;
}

}  // namespace

struct AesProbe::Impl {
    int device_index = -1;
    int sm_count = 0;

    std::vector<uint8_t> host_plaintext;
    std::vector<uint8_t> golden_ciphertext;
    util::Sha256Digest golden_hash{};

    platform::CudaDeviceBuffer d_plaintext;
    platform::CudaDeviceBuffer d_ciphertext;
    platform::CudaDeviceBuffer d_key;
    platform::CudaDeviceBuffer d_flag;

    platform::CudaPinnedBuffer h_ciphertext;
    platform::CudaPinnedBuffer h_flag;

    platform::CudaEvent start_event{cudaEventDefault};
    platform::CudaEvent stop_event{cudaEventDefault};
};

AesProbe::AesProbe() : impl_(std::make_unique<Impl>()) {}
AesProbe::~AesProbe() = default;

bool AesProbe::initialize(int device_index) {
    try {
        platform::set_device(device_index);
        impl_->device_index = device_index;
        impl_->sm_count = get_sm_count(device_index);

        impl_->host_plaintext = generate_plaintext();
        impl_->golden_ciphertext = compute_golden_ciphertext(impl_->host_plaintext);
        impl_->golden_hash = util::sha256(impl_->golden_ciphertext.data(),
                                           impl_->golden_ciphertext.size());

        impl_->d_plaintext = platform::CudaDeviceBuffer(kDataSize);
        impl_->d_ciphertext = platform::CudaDeviceBuffer(kDataSize);
        impl_->d_key = platform::CudaDeviceBuffer(16);
        impl_->d_flag = platform::CudaDeviceBuffer(sizeof(int));

        impl_->h_ciphertext = platform::CudaPinnedBuffer(kDataSize);
        impl_->h_flag = platform::CudaPinnedBuffer(sizeof(int));

        impl_->d_plaintext.copy_from_host(impl_->host_plaintext.data(), kDataSize);
        impl_->d_key.copy_from_host(kProbeKey, 16);
        CUDA_CHECK(cudaDeviceSynchronize());

        SENTINEL_LOG_INFO("AES probe initialized on device {} ({} SMs)", device_index, impl_->sm_count);
        return true;

    } catch (const platform::CudaError& e) {
        SENTINEL_LOG_ERROR("AES probe init failed: {}", e.what());
        return false;
    }
}

ProbeResult AesProbe::execute(uint32_t sm_id, cudaStream_t stream) {
    ProbeResult result;
    result.probe_id = next_probe_id();
    result.probe_type = ProbeType::kAes;
    result.sm_id = sm_id;
    result.expected_hash = impl_->golden_hash;

    try {
        platform::set_device(impl_->device_index);

        int zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(impl_->d_flag.get(), &zero, sizeof(int),
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(impl_->d_ciphertext.get(), 0, kDataSize, stream));

        dim3 grid, block;
        compute_sm_pinned_launch_params(impl_->sm_count, 256, grid, block);

        impl_->start_event.record(stream);

        aes_probe_kernel<<<grid, block, 0, stream>>>(
            impl_->d_plaintext.as<uint8_t>(),
            impl_->d_ciphertext.as<uint8_t>(),
            impl_->d_key.as<uint8_t>(),
            kNumBlocks, sm_id,
            impl_->d_flag.as<int>());

        impl_->stop_event.record(stream);

        impl_->d_ciphertext.copy_to_host(impl_->h_ciphertext.get(), kDataSize, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float elapsed_ms = platform::CudaEvent::elapsed_ms(
            impl_->start_event, impl_->stop_event);
        result.execution_time_ns = static_cast<uint64_t>(elapsed_ms * 1e6);

        result.actual_hash = util::sha256(
            reinterpret_cast<const uint8_t*>(impl_->h_ciphertext.get()), kDataSize);

        result.match = util::digest_equal(result.expected_hash, result.actual_hash);

        if (!result.match) {
            const uint8_t* actual = impl_->h_ciphertext.as<uint8_t>();
            for (uint32_t i = 0; i < kDataSize && result.mismatch_details.size() < 64; ++i) {
                if (actual[i] != impl_->golden_ciphertext[i]) {
                    MismatchDetail d;
                    d.byte_offset = i;
                    d.expected_value = {impl_->golden_ciphertext[i]};
                    d.actual_value = {actual[i]};
                    uint8_t diff = actual[i] ^ impl_->golden_ciphertext[i];
                    for (uint32_t bit = 0; bit < 8; ++bit) {
                        if (diff & (1u << bit)) d.differing_bits.push_back(bit);
                    }
                    result.mismatch_details.push_back(std::move(d));
                }
            }
        }

        result.timestamp = std::chrono::system_clock::now();

    } catch (const platform::CudaError& e) {
        SENTINEL_LOG_ERROR("AES probe failed on SM {}: {}", sm_id, e.what());
        result.match = false;
        result.timestamp = std::chrono::system_clock::now();
    }

    return result;
}

void AesProbe::teardown() {
    impl_->d_plaintext = {};
    impl_->d_ciphertext = {};
    impl_->d_key = {};
    impl_->d_flag = {};
    impl_->h_ciphertext = {};
    impl_->h_flag = {};
}

std::size_t AesProbe::memory_footprint() const noexcept {
    return kDataSize * 2 + 16 + sizeof(int)    // device
         + kDataSize + sizeof(int);            // pinned host
}

}  // namespace sentinel::probes
